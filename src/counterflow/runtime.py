from .io import *
from .observation import move
from .dynamics import make,rollout
import numpy as np
import torch
def velocity(c,a):
    v=c['grid_v0'][None]+torch.einsum('tfm,fpmj->tfpj',a,c['grid_V'])
    return v.reshape(len(a),2,*c['grid_shape'],2)
def indices(K,Ksmall):
    pairs=[(0,j) for j in range(1,K+1)]+[(i,j) for i in range(1,K+1) for j in range(-K,K+1)]
    return torch.as_tensor(np.flatnonzero((np.max(np.abs(pairs),axis=1)<=Ksmall).repeat(2)))

def forward(c,op,z,substeps=2):
    return rollout(op,torch.einsum('fij,fj->fi',c['decode'],z),c['intervals'],substeps,False)
def penalties(c,z,a,warm,anchor_factor):
    if c['kind']=='periodic':physical=.003*a.square().sum(-1).mean()/24+.03*(a[1:]-a[:-1]).square().sum(-1).mean()/24
    else:physical=.002*z.square().sum()
    high=torch.ones(z.shape[-1],dtype=torch.bool);high[c['root_ids']]=False
    spectral=z.new_zeros(()) if c['K']==2 else .03*(z[:,high]*c['high_weight']).square().sum(-1).mean()/len(c['root_ids'])
    anchor=z.new_zeros(())
    if warm is not None and anchor_factor>0:
        previous=warm[:,c['prev_ids']];scale=previous.norm(dim=-1).clamp_min(.1)
        anchor=anchor_factor*.5*.03*((z[:,c['prev_ids']]-previous).square().sum(-1)/(.1*scale).square()).mean()
    return dict(physical=physical,spectral=spectral,anchor=anchor)
