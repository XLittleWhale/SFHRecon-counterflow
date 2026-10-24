import math
import numpy as np
import torch
from .observation import *
def optimize(loss,start,mask,cfg):
    p=torch.nn.Parameter(start.clone());opt=torch.optim.Adam([p],lr=.03);best=math.inf;last=0;trace=[];saved=start.clone();reason='max_steps'
    for i in range(cfg['max_steps']):
        opt.zero_grad();v=loss(p*mask);assert torch.isfinite(v);v.backward()
        assert torch.isfinite(p.grad).all();value=float(v.detach())
        if value<best-1e-8:best=value;saved=(p*mask).detach().clone();last=i
        if i%50==0:trace.append(dict(step=i,loss=value,gradient=float(p.grad.norm())))
        if i>=cfg['min_steps'] and (i-last>=cfg['patience'] or float(p.grad.norm())<1e-5):reason='training_objective_plateau_or_small_gradient';break
        opt.step()
    return saved,dict(steps=i+1,reason=reason,trace=trace)

def latent_score(rows,p):
    vals=[]
    for r in rows:
        q=r['q'];ix=torch.arange(len(q),device=q.device);L=r['L'][ix,q]
        z=torch.linalg.solve_triangular(L,(r['y']-r['mu'][ix,q])[...,None],upper=False)[...,0]
        logt=(p[0]+p[1]*math.log(r['U']/.004)).clamp(math.log(.25),math.log(4))
        sign=-torch.log_softmax(r['logits']/p[2].exp().clamp(.25,4),-1)[ix,q]
        vals.append((sign+.5*z.square().sum(-1)*(-2*logt).exp()+2*logt+L.diagonal(dim1=-2,dim2=-1).log().sum(-1)+math.log(2*math.pi)).mean())
    return torch.stack(vals).mean()

def residuals(rows,cfg,K,kind,degree):
    out=[]
    for r in rows:
        if r['K']!=K:continue
        x=math.log(r['U']/.004)
        if kind=='free':
            if len(r['free_error']):out.append((x,r['free_error'].square().sum(-1),None))
            continue
        eps,lw=nodes(degree,str(r['mu'].device),r['mu'].dtype);tau=law(cfg['latent'],r['U'],.25,4)
        delta=r['mu'][:,:,None]+tau*torch.einsum('pqij,nj->pqni',r['L'],eps)
        force=delta/(r['hr'][:,None,None,None].square()+delta.square().sum(-1,keepdim=True)).sqrt()
        pred=r['base'][:,:,None]+torch.einsum('pqij,pqnj->pqni',r['gain'],force)
        e=(r['target'][:,None,None]-pred).square().sum(-1).flatten(1)
        weights=(torch.log_softmax(r['logits']/cfg['sign_temperature'],-1)[:,:,None]+lw).flatten(1)
        out.append((x,e,weights))
    return out

def noise_score(rows,p):
    vals=[]
    for x,e,lw in rows:
        logsd=(p[0]+p[1]*x).clamp(math.log(.015),math.log(1.5));lp=-.5*e*(-2*logsd).exp()-2*logsd-math.log(2*math.pi)
        vals.append((-torch.logsumexp(lp+lw,-1) if lw is not None else -lp).mean())
    assert vals,'Empty calibration data';return torch.stack(vals).mean()
