"""Causal full-segment filtering; current endpoints are scored before updating."""
import numpy as np
import torch
from common import EW,TM,M,S
import data as D
import latent_model as LM

def sequence_scores(z,c,base,head,cal,force,weights,jets,degree=20,mode='temporal',
                    metrics=False,backward=False,batch_pids=64):
    ids=(weights>0)&(z['state']==1)
    parts={}
    value=0.
    mass=float(weights[ids].sum())
    for pack in D.pack_sequences(z,ids,batch_pids):
        indices=pack['index']
        valid=pack['valid']
        B,nt=indices.shape
        # Dummy padding must also stay within the authorized I/V/training mask.
        # A global row zero may belong to H, even though its loss weight is zero.
        filler=indices[valid][0]
        safe_indices=torch.where(valid,indices,filler)
        flat=safe_indices.flatten()
        full=EW.subset(z,flat,'cpu')
        vv,gg,aa=jets(full,flat)
        previous=None
        previous_ctx=None
        losses=[]
        for frame in range(nt):
            if not bool(valid[:,frame].any()):
                previous=None
                previous_ctx=None
                continue
            take=torch.arange(B)*nt+frame
            zz={k:v[take] for k,v in full.items()}
            v,g,acc=vv[take],gg[take],aa[take]
            pred,cov,lw,posterior=LM.step(c,zz,v,g,acc,base,head,cal,force,degree,
                prevstate=previous,prevctx=previous_ctx,mode=mode,
                reset_mask=pack['reset'][:,frame])
            nll=-posterior['log_evidence']
            ww=weights[safe_indices[:,frame]]*valid[:,frame]
            losses.append((ww*nll).sum())
            if metrics:
                cover=(EW.coverage(pred,cov,lw,zz['target'])<=4.605170186).double()
                prob=lw.exp()
                mean=(prob[...,None]*pred).sum((1,2))
                trace=cov.diagonal(dim1=-2,dim2=-1).sum(-1)
                spread=(prob*(trace+(pred-mean[:,None,None]).square().sum(-1))).sum((1,2)).clamp_min(0).sqrt()*c['physics']['vn_mean_m_s']*.001*1e6
                for i in torch.where(valid[:,frame])[0].tolist():
                    pid=str(int(zz['pid'][i]))
                    row=parts.setdefault(pid,dict(count=0,nll=0.,coverage=0.,spread=0.))
                    row['count']+=1
                    for key,tensor in (('nll',nll),('coverage',cover),('spread',spread)):
                        row[key]+=float(tensor[i].detach())
            previous=posterior
            previous_ctx=TM.context(c,zz,v,g)
        loss=torch.stack(losses).sum()
        if backward: (.08*loss).backward()
        value+=float(loss.detach())
    for row in parts.values():
        for key in ('nll','coverage','spread'): row[key]/=row['count']
    return value,mass,parts

def score(a,es,en,d,net,cal,force,group,degree=20,metrics=False,backward=False,batch=256):
    z,c=d['z'],d['c']
    w=S.weights(z,group)
    result={'total':0.}
    if metrics: result.update(coverage={},spread={},per_pid={})
    ids=torch.where((w>0)&(z['state']==0))[0]
    value,mass,parts=0.,0.,{}
    for ix in ids.split(batch):
        zz=EW.subset(z,ix,a.device)
        v,g,acc=M.R.local_jets(a,es,en,zz,c)
        pred,cov,lw=TM.distribution(c,zz,v,g,acc,'free',net.base,cal,force,degree)
        nll=TM.scores(pred,cov,lw,zz['target'])
        ww=w[ix].to(a)
        loss=(ww*nll).sum()
        if backward: (.08*loss).backward()
        value+=float(loss.detach());mass+=float(ww.sum())
        if metrics:
            cover=(EW.coverage(pred,cov,lw,zz['target'])<=4.605170186).double()
            prob=lw.exp();mean=(prob[...,None]*pred).sum((1,2))
            trace=cov.diagonal(dim1=-2,dim2=-1).sum(-1)
            spread=(prob*(trace+(pred-mean[:,None,None]).square().sum(-1))).sum((1,2)).sqrt()*c['physics']['vn_mean_m_s']*.001*1e6
            for pid in zz['pid'].unique().tolist():
                use=zz['pid']==pid
                row=parts.setdefault(str(pid),dict(count=0,nll=0.,coverage=0.,spread=0.))
                row['count']+=int(use.sum())
                for key,tensor in (('nll',nll),('coverage',cover),('spread',spread)):
                    row[key]+=float(tensor[use].detach().sum())
    for row in parts.values():
        for key in ('nll','coverage','spread'): row[key]/=row['count']
    result['free']=value/mass;result['total']+=value
    if metrics:
        result['per_pid']['free']=parts
        for key in ('coverage','spread'): result[key]['free']=float(np.mean([r[key] for r in parts.values()]))
    jets=lambda zz,ix:M.R.local_jets(a,es,en,zz,c)
    value,mass,parts=sequence_scores(z,c,net.base,net.head,cal,force,w,jets,degree,net.mode,metrics,backward)
    result['cap']=value/mass;result['total']+=value
    if metrics:
        result['per_pid']['cap']=parts
        for key in ('coverage','spread'): result[key]['cap']=float(np.mean([r[key] for r in parts.values()]))
    return result
