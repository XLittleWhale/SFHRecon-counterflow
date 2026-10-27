"""Full-batch strong-constraint 4D-Var: initial velocities ONLY.

Observation chunks accumulate a gradient on a detached rollout leaf; a single
PDE reverse pass propagates their exact sum. This is memory scheduling, not a
truncated time window, stochastic particle subset, or stopped fluid gradient.
"""
import json
import time
import math
import numpy as np
import torch
from counterflow.io import load,save,dump,seed
from counterflow.dynamics import rollout
from .core import modes,jets,subset,weights,distribution,scores,operator,penalty,grid
from .train import model


def objective(a,c,z,net,cal,force,cfg,device,group,backward=False):
    p=cfg['inversion'];ww=weights(z,group);value=0.;kindvalues={}
    for kind,s in [('free',0),('cap',1)]:
        total=0.;mass=0.
        ids=torch.where((ww>0)&(z['state']==s))[0]
        for ix in ids.split(p['batch']):
            b=subset(z,ix,device);vv,gg,aa=jets(a,b,c,cfg['K'])
            pred,cov,lw=distribution(c,b,vv,gg,aa,kind,net,cal,force,p['degree'])
            w=ww[ix].to(device);loss=(w*scores(pred,cov,lw,b['target'])).sum()
            if backward:(p['observation_weight']*loss).backward()
            total+=float(loss.detach());mass+=float(w.sum())
        if mass:kindvalues[kind]=total/mass
        value+=total
    return value,kindvalues


def fit(package,run,cfg,name,factor,restart,device,path,smoke=False):
    path.mkdir(parents=True,exist_ok=True)
    if (path/'complete.json').exists():return load(path/'selected.pt')
    d=load(run/'prepared'/name/'observations.pt');c,z,T=d['c'],d['z'],d['intervals']
    # Deliberately no access to reference.pt or latent_truth.pt in this function.
    op=operator(c,load(package/'operator.pt'),cfg,device);net=model(run,device)
    cal=json.loads((run/'calibration.json').read_text());force=json.loads((run/'force.json').read_text())
    p=cfg['inversion'];xy,_=modes(cfg['K']);order=np.max(abs(xy),1).repeat(2)
    seed(261026+restart);theta=torch.nn.Parameter(torch.zeros(2,len(order),device=device,dtype=torch.float64))
    if restart:
        with torch.no_grad():theta[:,order<=2]=.03*torch.randn_like(theta[:,order<=2])
    prior=theta.detach().clone();trace=[];best_global=float('inf');global_state=None
    # Keep the untouched start as a legitimate candidate. Optimization is not
    # forced to beat the background by discarding it from the selection pool.
    if (path/'selected.pt').exists():
        old=load(path/'selected.pt');best_global=old['validation'];global_state=old['theta'].to(device)
    else:
        with torch.no_grad():
            baseline,_=rollout(op,theta,T,p['substeps'],False)
            best_global,_=objective(baseline,c,z,net,cal,force,cfg,device,'validation')
        global_state=theta.detach().clone()
        save(path/'selected.pt',dict(theta=global_state.cpu(),validation=best_global,K=2,step=-1,factor=factor,restart=restart))
    stages=[(2,2,p['lr'][0],1)] if smoke else list(zip(p['orders'],p['steps'],p['lr'],p['minimum_steps']))
    for stage,(K,maxsteps,lr,minsteps) in enumerate(stages):
        active=torch.tensor(order<=K,device=device);opt=torch.optim.Adam([theta],lr=lr)
        best=float('inf');beststate=theta.detach().clone();beststep=0;begin=0;stale=0
        last=path/f'K{K}_last.pt'
        if last.exists():
            ck=load(last,str(device));theta.data.copy_(ck['current']);beststate=ck['best_state'];best=ck['best'];beststep=ck['best_step'];stale=ck['stale']
            opt.load_state_dict(ck['optimizer']);begin=ck['step']+1;trace=ck['trace']
            best_global=ck['best_global'];global_state=ck['global_state']
            if (path/f'K{K}_status.json').exists():
                theta.data.copy_(beststate.to(device));continue
        started=time.perf_counter()
        for step in range(begin,maxsteps):
            opt.zero_grad(set_to_none=True)
            a,q=rollout(op,theta,T,p['substeps'],True)
            if not torch.isfinite(a).all() or not torch.isfinite(q).all():raise FloatingPointError('Nonfinite PDE rollout; no output promoted')
            leaf=a.detach().requires_grad_(True)
            nll,bykind=objective(leaf,c,z,net,cal,force,cfg,device,'train',True)
            reg=penalty(a,cfg,factor)
            torch.autograd.backward((a,reg),(leaf.grad,torch.ones_like(reg)))
            theta.grad[:,~active]=0
            gradnorm=float(torch.linalg.vector_norm(theta.grad))
            torch.nn.utils.clip_grad_norm_([theta],5.,error_if_nonfinite=True);opt.step()
            check=step==0 or (step+1)%p['check_every']==0 or step+1==maxsteps
            if check:
                with torch.no_grad():
                    av,qv=rollout(op,theta,T,p['substeps'],False)
                    val,vkind=objective(av,c,z,net,cal,force,cfg,device,'validation')
                if val<best-1e-5:best=val;beststate=theta.detach().clone();beststep=step;stale=0
                else:stale+=1
                if val<best_global:
                    best_global=val;global_state=theta.detach().clone()
                    save(path/'selected.pt',dict(theta=global_state.cpu(),validation=val,K=K,step=step,factor=factor,restart=restart))
                row=dict(K=K,step=step,train=nll,validation=val,train_states=bykind,validation_states=vkind,
                    regularization=float(reg.detach()),gradient_norm=gradnorm,seconds=time.perf_counter()-started)
                trace.append(row)
                save(last,dict(current=theta.detach().cpu(),best_state=beststate.cpu(),best=best,best_step=beststep,
                    stale=stale,step=step,optimizer=opt.state_dict(),trace=trace,best_global=best_global,global_state=global_state.cpu()))
                print(f'{name} lambda={factor} restart={restart} K={K} step={step+1}: train={nll:.5g}, valid={val:.5g}',flush=True)
                if step+1>=minsteps and stale>=p['patience_checks']:break
        theta.data.copy_(beststate.to(device))
        dump(path/f'K{K}_status.json',dict(last_step=step if begin<maxsteps else begin-1,selected_step=beststep,
            stopping='validation_patience' if stale>=p['patience_checks'] else 'iteration_budget_not_convergence_proof'))
    dump(path/'trace.json',trace);dump(path/'complete.json',dict(selected_validation=best_global,
        sole_controls='initial vn and vs coefficients',PDE_steps=T,all_training_intervals=True,smoke=smoke))
    return load(path/'selected.pt')


@torch.no_grad()
def tuning_metrics(package,run,cfg,name,selected,device):
    d=load(run/'prepared'/name/'observations.pt');c=d['c'];op=operator(c,load(package/'operator.pt'),cfg,device)
    ref=load(run/'prepared'/name/'reference.pt')['a'].to(device)
    a,_=rollout(op,selected['theta'].to(device),d['intervals'],cfg['inversion']['substeps'],False)
    # Parseval applies to the same orthogonal velocity basis and fixed means.
    err=(a-ref).square().mean((0,2)).sqrt();base=ref.square().mean((0,2)).sqrt().clamp_min(1e-12)
    return dict(relative_to_uniform=(err/base).tolist(),score=float((err/base).max()),
                calibration_truth_only=True,initial_error=(a[0]-ref[0]).square().mean(-1).sqrt().tolist())


def tune(package,run,cfg,device,smoke=False):
    if (run/'selection.json').exists():return
    reports=[]
    for factor in cfg['inversion']['prior_grid'][:1 if smoke else None]:
        selected=fit(package,run,cfg,cfg['validation'],factor,0,device,run/'tuning'/f'prior_{factor}',smoke)
        reports.append(dict(factor=factor,validation_nll=selected['validation'],**tuning_metrics(package,run,cfg,cfg['validation'],selected,device)))
    pick=min(reports,key=lambda r:(r['score'],r['validation_nll']))
    dump(run/'selection.json',dict(selected=pick,candidates=reports,criterion='min worst-fluid RMS relative to uniform, calibration case only',
        note='finite predeclared grid, not globally optimal parameters; targets never consulted',smoke=smoke))


def main(package,run,cfg,device,smoke=False):
    if (run/'inverse/complete.json').exists():return
    factor=json.loads((run/'selection.json').read_text())['selected']['factor'];rows=[]
    for restart in cfg['inversion']['restarts'][:1 if smoke else None]:
        result=fit(package,run,cfg,cfg['test'],factor,restart,device,run/'inverse'/f'restart_{restart}',smoke)
        rows.append(result)
    pick=min(rows,key=lambda r:r['validation']);save(run/'inverse/selected.pt',pick)
    dump(run/'inverse/complete.json',dict(selected_restart=pick['restart'],validation=pick['validation'],
        factors_fixed_before_target=True,reference_truth_used=False,smoke=smoke))
