import math
import numpy as np
import torch
from .observation import *
IDENTITY=dict(latent=[0.,0.],sign_temperature=1.)
def score(net,z,count):
    latent_nll=-net.log_prob(z['h'],z['c'],z['y'],z['q']).mean()
    pred,lw=decoded(net,z['h'],z['c'],z['base'],z['gain'],z['hr'],IDENTITY,sobol=count)
    obs=mixture_nll(pred,lw,z['target'],.3).mean()
    return latent_nll+.1*obs,latent_nll,obs

@torch.no_grad()
def validate(net,views,count,batch):
    net.eval();result=[]
    for z in views:
        totals=np.zeros(3);N=len(z['q'])
        for lo in range(0,N,batch):
            sub={k:v[lo:lo+batch] for k,v in z.items()};totals+=np.array([float(s) for s in score(net,sub,count)])*len(sub['q'])
        result.append((totals/N).tolist())
    return float(np.mean([a[0] for a in result])),result
