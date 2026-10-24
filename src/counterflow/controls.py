import math
import torch
import numpy as np
from . import runtime as rt
def batches(c, split=None, group=2):
    """Sort each whole trajectory in time; never split its four observations."""
    out=[];T=c.get('observation_steps',c['intervals'])
    for kind in ('cap','free'):
        z=c[kind]
        mask=z['pid']%5!=0 if split is None else split[kind]==group
        ids=z['pid'][mask].unique()
        for ps in ids.split(40):
            ix=torch.where(mask & torch.isin(z['pid'],ps))[0]
            ix=ix[torch.argsort(z['pid'][ix]*1000+z['frame'][ix])]
            zz={k:v[ix] for k,v in z.items()}
            assert len(ix)==len(ps)*T
            assert bool((torch.diff(zz['frame'].reshape(-1,T),dim=1)==1).all())
            out.append((kind,zz))
    return out
def parts(c,group):
    split={k:torch.where(c[k]['pid']%c['pid_stride']%5==0,0,
                        torch.where(c[k]['pid']%c['pid_stride']%5==1,1,2)) for k in ('cap','free')}
    return batches(c,split,group)

def control_mask(c,K):
    mask=torch.zeros_like(c['initial'],dtype=torch.bool)
    if c['kind']=='periodic':mask[:,rt.indices(12,K)]=True
    else:mask[:,:((K+1)*(2*K+1))]=True
    return mask

def archive(path,c,op,z,summary):
    with torch.no_grad():a,q=rt.forward(c,op,z);v=rt.velocity(c,a)
    rt.save(path/'selected.pt',z)
    np.savez_compressed(path/'fields.npz',theta=z.numpy(),coefficients=a.numpy(),log_line_coefficients=q.numpy(),
        velocity=v.numpy(),time_s=c['start_s']+np.arange(c['intervals']+1)*.001,x=c['x'].numpy(),y=c['y'].numpy())
    rt.dump(path/'summary.json',summary)
