"""Server GPU preprocessing: conservative reference and full/coarse force labels."""
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from counterflow.io import load,save,dump
from .core import modes,jets,affine,matrices,context,subset,split


def overlap(n,m):
    def P(u):return np.where(u<-1,0,np.where(u<0,.5*(u+1)**2,np.where(u<1,1-.5*(1-u)**2,1)))
    a=np.arange(m)[:,None]*n/m-np.arange(n)[None];b=a+n/m
    return sum((P(b-s*n)-P(a-s*n))*m/n for s in (-1,0,1))


def coarse(v,W):
    return torch.einsum('iy,yxfc,jx->ijfc',W,v,W)


def project(v,K,U):
    n=len(v);xy,pp=modes(K);m=torch.fft.fftfreq(n,device=v.device,dtype=v.dtype)*n
    phase=torch.exp(-1j*torch.pi*(m[:,None]+m[None,:])/n)
    ft=torch.fft.fft2(v,dim=(0,1))/n**2*phase[:,:,None,None]
    zz=(ft[xy[:,1]%n,xy[:,0]%n]*v.new_tensor(pp)[:,None,:]).sum(-1).T/U
    return torch.stack((2*zz.imag,2*zz.real),-1).flatten(1)


def sample(v,x,L):
    n=len(v);u=(x%L)*n/L;ij=u.long();r=u-ij;out=v.new_zeros((len(x),)+v.shape[2:])
    for ox in (0,1):
        for oy in (0,1):
            w=(r[:,0] if ox else 1-r[:,0])*(r[:,1] if oy else 1-r[:,1])
            out+=w.reshape((-1,)+(1,)*(v.ndim-2))*v[(ij[:,1]+oy)%n,(ij[:,0]+ox)%n]
    return out


def full_jets(left,right,x,L):
    v=.5*(left+right);n=len(v)
    k=torch.fft.fftfreq(n,device=v.device,dtype=v.dtype)*2*torch.pi*n/L
    fv=torch.fft.fft2(v,dim=(0,1))
    g=torch.stack([torch.fft.ifft2(fv*(1j*kk[:,:,None,None]),dim=(0,1)).real for kk in
        (k[None,:].expand(n,n),k[:,None].expand(n,n))],-1)
    acc=(right-left)/.001+torch.einsum('yxfi,yxfij->yxfj',v,g)
    return sample(v,x,L),sample(g,x,L),sample(acc,x,L)


@torch.no_grad()
def prepare_case(package,run,name,cfg,device,smoke=False):
    target=run/'prepared'/name;target.mkdir(parents=True,exist_ok=True)
    if (target/'complete.json').exists():return
    src=package/'data'/name;c=json.loads((src/'physics.json').read_text(encoding='utf8'))
    z=load(src/'observations.pt');labels=load(src/'latent_truth.pt') if name!=cfg['test'] else None
    frames=2 if smoke else cfg['end']-cfg['start'];keep=z['frame']<frames
    z=subset(z,keep,'cpu');labels=subset(labels,keep,'cpu') if labels else None
    nrow=len(z['pid']);ref=[];grids=[]
    fullv=torch.empty(nrow,2,2,dtype=torch.float64);fulla=torch.empty_like(fullv)
    with h5py.File(src/'fields.h5','r') as h:
        W=torch.tensor(overlap(h['v'].shape[1],cfg['N']),device=device,dtype=torch.float64)
        left=torch.as_tensor(h['v'][0],device=device)
        for j in range(frames+1):
            v=left if j==0 else torch.as_tensor(h['v'][j],device=device)
            cv=coarse(v,W);ref.append(project(cv,cfg['K'],c['physics']['vn_mean_m_s']).cpu())
            grids.append(cv.permute(2,0,1,3).cpu())
            if j>0 and labels is not None:
                ix=torch.where(z['frame']==j-1)[0];zz=subset(z,ix,device)
                vv,gg,aa=full_jets(left,v,zz['x'],c['physics']['box_m'])
                fullv[ix]=vv.cpu();fulla[ix]=aa.cpu()
            left=v
            if j%20==0:print('GPU reference preparation:',name,j,'/',frames,flush=True)
    a=torch.stack(ref);save(target/'reference.pt',dict(a=a,grid=torch.stack(grids)))
    # Runtime pack has no truth fields or latent labels.
    save(target/'observations.pt',dict(c=c,z=z,intervals=frames))
    if labels is not None:
        context_parts=[];ev=torch.zeros(nrow,2,dtype=torch.float64)
        jetparts=[[],[],[]]
        for ix in torch.arange(nrow).split(512):
            zz=subset(z,ix,device);vv,gg,aa=jets(a.to(device),zz,c,cfg['K'])
            context_parts.append(context(c,zz,vv,gg).cpu())
            for out,part in zip(jetparts,(vv,gg,aa)):out.append(part.cpu())
            for kind,s in [('free',0),('cap',1)]:
                m=zz['state']==s
                if not m.any():continue
                A,_,_,_=matrices(c,vv,kind)
                dif=affine(c,fullv[ix].to(device)[m],fulla[ix].to(device)[m],kind)-affine(c,vv[m],aa[m],kind)
                if kind=='cap':
                    q=labels['q'][ix].to(device)[m];idx=torch.arange(len(q),device=device)
                    dif=dif[idx,q];Am=A[q]
                else:dif=dif[:,0];Am=A[0].expand(len(dif),-1,-1)
                ev[ix[m.cpu()]]=torch.linalg.solve(Am,dif[...,None])[...,0].cpu()/c['physics']['vn_mean_m_s']
        save(target/'training.pt',dict(z=z,labels=labels,context=torch.cat(context_parts),force=ev,
             jets=tuple(torch.cat(p) for p in jetparts)))
    dump(target/'complete.json',dict(rows=nrow,intervals=frames,test_truth_not_used_for_labels=labels is None))


def fit_force(run,cfg):
    """Pooled fixed spatial-force moments; never fit a per-target residual."""
    models={};report={}
    for kind,s in [('free',0),('cap',1)]:
        means=[];seconds=[];counts=[]
        for name in cfg['train']:
            d=load(run/'prepared'/name/'training.pt');m=split(d['z']['pid'],'train')&(d['z']['state']==s)
            x=d['force'][m];pid=d['z']['pid'][m];N=len(pid.unique())
            if not N:raise ValueError('Empty force training state: '+name)
            _,inv,count=pid.unique(return_inverse=True,return_counts=True)
            w=1/(N*count[inv].double());means.append((w[:,None]*x).sum(0))
            seconds.append(torch.einsum('b,bi,bj->ij',w,x,x));counts.append(len(x))
        mu=torch.stack(means).mean(0);C=torch.stack(seconds).mean(0)-mu[:,None]*mu[None,:]
        C+=torch.eye(2,dtype=C.dtype)*1e-6
        models[kind]=dict(mean=mu.tolist(),covariance=C.tolist());report[kind]=dict(per_case_rows=counts)
    dump(run/'force.json',models);dump(run/'force_fit_audit.json',report)


def main(package,run,cfg,device,smoke=False):
    for name in cfg['train']+[cfg['validation'],cfg['test']]:prepare_case(package,run,name,cfg,device,smoke)
    fit_force(run,cfg)
