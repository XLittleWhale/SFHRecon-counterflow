"""Shared physical observation operator and compact Fourier queries.

All observations are original 1ms endpoints. No microscopic samples or target
endpoint enter the causal history or predictor query location.
"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from counterflow import observation as ob
from counterflow.io import load, save, dump
from counterflow.dynamics import make, rollout

DT = .001


def config():
    return json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf8'))


def modes(K):
    xy = np.array([(0,j) for j in range(1,K+1)]+
                  [(i,j) for i in range(1,K+1) for j in range(-K,K+1)])
    return xy, np.c_[xy[:,1], -xy[:,0]]/np.linalg.norm(xy, axis=1)[:,None]


def basis(x, c, K=12):
    xy, pp = modes(K)
    k = x.new_tensor(xy)*2*math.pi/c['physics']['box_m']
    perp = x.new_tensor(pp)
    ang = (x % c['physics']['box_m'])@k.T
    b = torch.stack((-ang.sin(),ang.cos()),-1)
    db = torch.stack((-ang.cos(),-ang.sin()),-1)
    U = c['physics']['vn_mean_m_s']
    V = (U*b[...,None]*perp[None,:,None]).flatten(1,2)
    G = (U*db[...,None,None]*perp[None,:,None,:,None]*k[None,:,None,None,:]).flatten(1,2)
    return V,G


def jets(a, z, c, K=12):
    V,G = basis(z['x'],c,K)
    frame = z['frame']
    mid = .5*(a[frame]+a[frame+1]);da = (a[frame+1]-a[frame])/DT
    v = a.new_tensor([[0,c['physics']['vn_mean_m_s']],[0,c['physics']['vs_mean_m_s']]])
    v = v+torch.einsum('bfm,bmj->bfj',mid,V)
    g = torch.einsum('bfm,bmij->bfij',mid,G)
    acc = torch.einsum('bfm,bmj->bfj',da,V)+torch.einsum('bfij,bfi->bfj',g,v)
    return v,g,acc


def matrices(c, like, kind):
    C=c['constants'];I=torch.eye(2,device=like.device,dtype=like.dtype)
    R=like.new_tensor([[0,-1],[1,0]])
    if kind=='cap':
        A=((C['drag']+C['eg'])*I[None]-(C['kf']+C['egp'])*like.new_tensor([-1,1])[:,None,None]*R)/C['mass']
    else:A=I[None]*C['drag']/C['mass']
    E=torch.matrix_exp(-DT*A)
    H=torch.linalg.solve(DT*A,I-E);B=torch.linalg.solve(DT*A,I-H)
    return A,H,B,DT*B@A


def affine(c, v, acc, kind):
    """Acceleration-independent affine RHS b in dvp/dt=-A_q vp+b.

    Source uses partial_t v + grad(|v|^2)/2, deliberately NOT (v.grad)v.
    """
    C=c['constants'];p=c['physics']
    body=1.5*C['vol']*(p['rho_n_kg_m3']*acc[:,0]+p['rho_s_kg_m3']*acc[:,1])
    body=body+v.new_tensor([0,C['gravity']])
    if kind=='free':return (C['drag']*v[:,0]+body)[:,None]/C['mass']
    rot=lambda x:torch.stack((-x[...,1],x[...,0]),-1)
    q=v.new_tensor([-1,1])[None,:,None]
    return ((C['drag']+C['eg'])*v[:,0,None]-C['kf']*q*rot(v[:,1,None])-
            C['egp']*q*rot(v[:,0,None])+body[:,None])/C['mass']


def context(c,z,v,g):
    C=c['constants'];U=c['physics']['vn_mean_m_s']
    f=v.new_tensor([math.log(U/.004),math.log(c['hr']),math.log(2),
                    math.log(DT*(C['drag']+C['eg'])/C['mass']),c['physics']['alpha']])
    return torch.cat((f.expand(len(v),-1),(v/U-z['w'][:,None]).flatten(1),(DT*g).flatten(1)),1)


def distribution(c,z,v,g,acc,kind,net,cal,force,degree):
    U=c['physics']['vn_mean_m_s'];A,H,B,M=matrices(c,v,kind)
    base=torch.einsum('qij,bj->bqi',H,z['w'])+DT/U*torch.einsum('qij,bqj->bqi',B,affine(c,v,acc,kind))
    mu=v.new_tensor(force[kind]['mean']);C=v.new_tensor(force[kind]['covariance'])
    base=base+torch.einsum('qij,j->qi',M,mu)[None]
    cov=M@C@M.transpose(-1,-2)
    sigma=cal['sigma'][kind]
    cov=cov+torch.eye(2,device=v.device,dtype=v.dtype)*sigma**2
    if kind=='free':return base[:,:,None],cov[None,:,None].expand(len(v),-1,1,-1,-1),v.new_zeros(len(v),1,1)
    ctx=context(c,z,v,g)
    lp,mu,L=ob.latent(net,z['h'],ctx,cal)
    eps,lw=ob.nodes(degree,str(v.device),v.dtype)
    delta=mu[:,:,None]+torch.einsum('bqij,nj->bqni',L,eps)
    eta=delta/(c['hr']**2+delta.square().sum(-1,keepdim=True)).sqrt()
    gain=DT*c['tension_twice_N']/(c['constants']['mass']*U)*B
    pred=base[:,:,None]+torch.einsum('qij,bqnj->bqni',gain,eta)
    return pred,cov[None,:,None].expand(len(v),-1,len(eps),-1,-1),lp[:,:,None]+lw


def scores(pred,cov,lw,target):
    L=torch.linalg.cholesky(cov)
    r=torch.linalg.solve_triangular(L,(target[:,None,None]-pred)[...,None],upper=False)[...,0]
    lp=-.5*r.square().sum(-1)-L.diagonal(dim1=-2,dim2=-1).log().sum(-1)-math.log(2*math.pi)
    return -torch.logsumexp((lp+lw).flatten(1),1)


def coverage(pred,cov,lw,target):
    """Exact Gaussian-mixture Rosenblatt diagnostic (quadrature approximation)."""
    mu=pred.flatten(1,2);C=cov.flatten(1,2);ww=lw.flatten(1)
    normal=lambda t:.5*(1+torch.erf(t/math.sqrt(2)))
    sx=C[...,0,0].sqrt();dx=(target[:,0,None]-mu[...,0])/sx
    fx=(ww.exp()*normal(dx)).sum(-1)
    weights=torch.softmax(ww-.5*dx.square()-sx.log(),-1)
    cy=mu[...,1]+C[...,1,0]/C[...,0,0]*(target[:,0,None]-mu[...,0])
    sy=(C[...,1,1]-C[...,1,0].square()/C[...,0,0]).clamp_min(1e-14).sqrt()
    fy=(weights*normal((target[:,1,None]-cy)/sy)).sum(-1)
    u=torch.stack((fx,fy),-1).clamp(1e-8,1-1e-8)
    return (math.sqrt(2)*torch.erfinv(2*u-1)).square().sum(-1)


def subset(z, ix, device=None, dtype=torch.float64):
    return {k:(v[ix].to(device=device,dtype=dtype if v.is_floating_point() else v.dtype)) for k,v in z.items() if torch.is_tensor(v)}


def split(pid, group):
    if group=='train':return pid%5>=2
    if group=='validation':return pid%5==1
    if group=='test':return pid%5==0
    if group=='all':return torch.ones_like(pid,dtype=torch.bool)
    raise ValueError(group)


def weights(z,group):
    """Equal states, equal physical PIDs within state, equal intervals within PID."""
    out=torch.zeros(len(z['pid']),dtype=torch.float64)
    good=split(z['pid'],group)
    present=[s for s in (0,1) if (good&(z['state']==s)).any()]
    for s in present:
        ix=torch.where(good&(z['state']==s))[0];pid=z['pid'][ix]
        _,inv,count=pid.unique(return_inverse=True,return_counts=True)
        out[ix]=1/(len(present)*len(count)*count[inv].double())
    return out


def operator(c, template, cfg, device):
    data=dict(template,c=c['physics'],U=c['physics']['vn_mean_m_s'],
              mean=torch.tensor([[0,c['physics']['vn_mean_m_s']],[0,c['physics']['vs_mean_m_s']]],dtype=torch.float64))
    assert data['n']==cfg['quadrature_grid'] and data['d']==len(modes(cfg['K'])[0])*2
    return make(data,device)


def penalty(a,cfg,factor):
    xy,_=modes(cfg['K']);r=torch.tensor(np.max(abs(xy),1).repeat(2),device=a.device,dtype=a.dtype)
    hi=r>2
    physical=.003*a.square().sum(-1).mean()/24+.03*(a[1:]-a[:-1]).square().sum(-1).mean()/24
    spectral=.03*(a[0,:,hi]*(r[hi]/2)**2).square().sum(-1).mean()/24
    return factor*(physical+spectral)


def grid(a,c,cfg):
    n=cfg['N'];x=(torch.arange(n,device=a.device,dtype=a.dtype)+.5)*c['physics']['box_m']/n
    yy,xx=torch.meshgrid(x,x,indexing='ij');V,_=basis(torch.stack((xx,yy),-1).reshape(-1,2),c,cfg['K'])
    v=a.new_tensor([[0,c['physics']['vn_mean_m_s']],[0,c['physics']['vs_mean_m_s']]])
    return (v[None,:,None]+torch.einsum('tfm,pmj->tfpj',a,V)).reshape(len(a),2,n,n,2)
