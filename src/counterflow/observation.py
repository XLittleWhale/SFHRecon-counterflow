"""Standalone, device-aware version of the frozen physical observation model.

Architecture and force conventions match ModalConditioning_20261009. No source
project imports, absolute paths, or reference fields are used during inversion.
"""
import math
from functools import lru_cache
import numpy as np
import torch
from torch import nn
DT=.001

def move(obj,device,dtype=torch.float64):
    if torch.is_tensor(obj):return obj.to(device=device,dtype=dtype if obj.is_floating_point() else obj.dtype)
    if isinstance(obj,dict):return {k:move(v,device,dtype) for k,v in obj.items()}
    if isinstance(obj,list):return [move(v,device,dtype) for v in obj]
    if isinstance(obj,tuple):return tuple(move(v,device,dtype) for v in obj)
    return obj

class Density(nn.Module):
    def __init__(self):
        super().__init__();self.gru=nn.GRU(4,32,batch_first=True)
        self.head=nn.Sequential(nn.Linear(53,48),nn.SiLU(),nn.Linear(48,32),nn.SiLU())
        self.sign=nn.Linear(32,2);self.output=nn.Linear(32,12)
        self.register_buffer('cm',torch.zeros(17));self.register_buffer('cs',torch.ones(17))
    def encode(self,h,c):
        _,s=self.gru(h);return self.head(torch.cat((s[-1],h[:,-1],(c-self.cm)/self.cs),-1))
    def parameters_at(self,e):
        out=self.output(e).reshape(len(e),2,1,6);mu=out[...,1:3]
        L=out.new_zeros(len(e),2,1,2,2)
        L[...,0,0]=torch.nn.functional.softplus(out[...,3])+.015
        L[...,1,1]=torch.nn.functional.softplus(out[...,5])+.015;L[...,1,0]=out[...,4]
        return torch.log_softmax(out[...,0],-1),mu,L
    def log_prob(self,h,c,y,q):
        e=self.encode(h,c);_,mu,L=self.parameters_at(e);ix=torch.arange(len(e),device=e.device)
        mu=mu[ix,q,0];L=L[ix,q,0];z=torch.linalg.solve_triangular(L,(y-mu)[...,None],upper=False)[...,0]
        return torch.log_softmax(self.sign(e),-1)[ix,q]-.5*z.square().sum(-1)-math.log(2*math.pi)-L.diagonal(dim1=-2,dim2=-1).log().sum(-1)

@lru_cache(maxsize=24)
def nodes(degree,device,dtype):
    x,w=np.polynomial.hermite.hermgauss(degree);xx,yy=np.meshgrid(x,x,indexing='ij')
    return torch.as_tensor(np.c_[xx.ravel(),yy.ravel()]*math.sqrt(2),device=device,dtype=dtype),torch.as_tensor(np.log(np.outer(w,w).ravel()/math.pi),device=device,dtype=dtype)

def law(par,U,lo,hi):return float(np.clip(np.exp(par[0]+par[1]*np.log(U/.004)),lo,hi))

def latent(net,h,c,cfg):
    e=net.encode(h,c);_,mu,L=net.parameters_at(e)
    tau=(cfg['latent'][0]+cfg['latent'][1]*c[:,0]).exp().clamp(.25,4)
    return torch.log_softmax(net.sign(e)/cfg['sign_temperature'],-1),mu[:,:,0],L[:,:,0]*tau[:,None,None,None]

def decoded(net,h,c,base,gain,hr,cfg,degree=16,sobol=None):
    logq,mu,L=latent(net,h,c,cfg)
    if sobol is None:eps,lw=nodes(degree,str(h.device),h.dtype)
    else:
        u=torch.quasirandom.SobolEngine(2,scramble=True,seed=815).draw(sobol).to(h)
        eps=math.sqrt(2)*torch.erfinv(2*u.clamp(1e-6,1-1e-6)-1);lw=h.new_full((sobol,),-math.log(sobol))
    delta=mu[:,:,None]+torch.einsum('pqij,nj->pqni',L,eps)
    force=delta/(hr[:,None,None,None].square()+delta.square().sum(-1,keepdim=True)).sqrt()
    pred=base[:,:,None]+torch.einsum('pqij,pqnj->pqni',gain,force)
    return pred,logq[:,:,None]+lw

def mixture_nll(pred,lw,target,sigma):
    return -torch.logsumexp((lw-.5*((target[:,None,None]-pred)/sigma).square().sum(-1)-math.log(2*math.pi*sigma**2)).flatten(1),-1)

def local(z,a):
    k=z['frame'];fraction=z.get('time_fraction',.5)
    if torch.is_tensor(fraction):fraction=fraction[:,None,None]
    mid=(1-fraction)*a[k]+fraction*a[k+1];diff=(a[k+1]-a[k])/DT
    v=z['v0']+torch.einsum('pfm,pfmj->pfj',mid,z['V'])
    g=z['g0']+torch.einsum('pfm,pfmij->pfij',mid,z['G'])
    acc=torch.einsum('pfm,pfmj->pfj',diff,z['V'])+torch.einsum('pfij,pfi->pfj',g,v)
    return v,g,acc

def particle_parts(case,a):
    z=case['cap'];c=case['physics'];C=case['constants'];U=c['vn_mean_m_s'];v,g,acc=local(z,a)
    context=torch.cat((z['context'],(v/U-z['w'][:,None]).flatten(1),(DT*g).flatten(1)),1)
    body=1.5*C['vol']*(c['rho_n_kg_m3']*acc[:,0]+c['rho_s_kg_m3']*acc[:,1])+a.new_tensor([0,C['gravity']])
    rot=lambda x:torch.stack((-x[...,1],x[...,0]),-1);q=a.new_tensor([-1,1])
    force=((C['drag']+C['eg'])*v[:,0,None]-C['kf']*q[None,:,None]*rot(v[:,1,None])-C['egp']*q[None,:,None]*rot(v[:,0,None])+body[:,None])*DT/(C['mass']*U)
    B,H=case['response'];base=torch.einsum('qij,pj->pqi',H,z['w'])+torch.einsum('qij,pqj->pqi',B,force)
    gain=(B*DT/(C['mass']*U)*case['tension_twice_N'])[None].expand(len(z['w']),-1,-1,-1)
    hr=a.new_full((len(z['w']),),case['hr']);return context,base,gain,hr

def predict_free(case,a):
    z=case['free'];c=case['physics'];C=case['constants'];U=c['vn_mean_m_s'];v,g,acc=local(z,a)
    body=1.5*C['vol']*(c['rho_n_kg_m3']*acc[:,0]+c['rho_s_kg_m3']*acc[:,1])+a.new_tensor([0,C['gravity']])
    H,B=case['free_response'];return H*z['w']+B*DT*(C['drag']*v[:,0]+body)/(C['mass']*U)

def likelihood(case,a,net,cfg,degree=16):
    ctx,base,gain,hr=particle_parts(case,a);pred,lw=decoded(net,case['cap']['h'],ctx,base,gain,hr,cfg,degree)
    pars=case.get('fixed_noise',cfg['sigma'][str(case['K'])]);U=case['physics']['vn_mean_m_s']
    sc,sf=[law(pars[k],U,.015,1.5) for k in ('captured','free')]
    cap=mixture_nll(pred,lw,case['cap']['target'],sc);free=predict_free(case,a)
    fn=.5*((free-case['free']['target'])/sf).square().sum(-1)+math.log(2*math.pi*sf**2)
    return cap,fn,(pred*lw.exp()[...,None]).sum((1,2)),free

def group_loss(case,a,net,cfg,group,degree=16):
    cn,fn,_,_=likelihood(case,a,net,cfg,degree);values=[]
    for kind,score in [('cap',cn),('free',fn)]:
        pid=case[kind]['pid'];mask=pid%5>=2 if group=='train' else pid%5==(1 if group=='validation' else 0)
        if mask.any():values.append(score[mask])
    if not values:raise ValueError('Empty particle split')
    return torch.cat(values).mean()
