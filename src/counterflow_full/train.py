"""Fresh-seed, full native-time physical-latent training and independent calibration."""
import json
import math
import time
import torch
from counterflow.observation import Density
from counterflow.io import load,save,dump,seed,cpu_state
from .core import subset,split,distribution,scores,weights


def default_cal():
    return dict(latent=[0.,0.],sign_temperature=1.,sigma=dict(free=.3,cap=.3))


def views(run,names,group):
    out=[]
    for name in names:
        d=load(run/'prepared'/name/'training.pt');c=load(run/'prepared'/name/'observations.pt')['c']
        m=split(d['z']['pid'],group)&(d['z']['state']==1)
        z=subset(d['z'],m,'cpu',torch.float32);z.update(subset(d['labels'],m,'cpu',torch.float32))
        z['c']=d['context'][m].float()
        for key,value in zip(('v','g','acc'),d['jets']):z[key]=value[m].float()
        # Equal case and PID weighting, not domination by long capture episodes.
        pid=z['pid'];_,inv,count=pid.unique(return_inverse=True,return_counts=True)
        z['weight']=(1/(len(names)*len(count)*count[inv].double())).float()
        out.append((name,c,z))
    return out


def loss(net,z,c,force,degree=6):
    latent=-net.log_prob(z['h'],z['c'],z['y'],z['q'])
    pred,cov,lw=distribution(c,z,z['v'],z['g'],z['acc'],'cap',net,default_cal(),force,degree)
    return latent+.1*scores(pred,cov,lw,z['target'])


@torch.no_grad()
def validation(net,sets,force,device,batch):
    net.eval();value=0.
    for name,c,z in sets:
        for ix in torch.arange(len(z['q'])).split(batch):
            b=subset(z,ix,device,torch.float32)
            value+=float((b['weight']*loss(net,b,c,force)).sum())
    return value


def main(package,run,cfg,device,smoke=False):
    out=run/'training';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():return
    tr=views(run,cfg['train'],'train');va=views(run,[cfg['validation']],'validation')
    force=json.loads((run/'force.json').read_text());p=cfg['training'];reports=[]
    epochs=2 if smoke else p['epochs'];min_ep=1 if smoke else p['minimum_epochs']
    # Concatenation is compact; basis matrices are NEVER materialized for all rows.
    C=torch.cat([z['c'] for _,_,z in tr]);W=torch.cat([z['weight'] for _,_,z in tr]);W=W/W.sum()
    cm=(W[:,None]*C).sum(0);cs=(W[:,None]*(C-cm).square()).sum(0).sqrt().clamp_min(.05)
    for sd in p['seeds'][:1 if smoke else None]:
        path=out/f'seed_{sd}';path.mkdir(exist_ok=True);seed(sd)
        net=Density().to(device);net.cm.copy_(cm.to(device));net.cs.copy_(cs.to(device))
        opt=torch.optim.AdamW(net.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=10,min_lr=1e-6)
        start=1;best=float('inf');bestep=0;trace=[]
        if (path/'last.pt').exists():
            ck=load(path/'last.pt',str(device));net.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
            start=ck['epoch']+1;best=ck['best'];bestep=ck['best_epoch'];trace=ck['trace']
            torch.set_rng_state(ck['rng'].cpu())
            if device.type=='cuda' and ck['cuda_rng'] is not None:torch.cuda.set_rng_state_all([v.cpu() for v in ck['cuda_rng']])
        if start>epochs:
            reports.append(dict(seed=sd,best=best,epoch=bestep,stopping='completed previously'));continue
        begun=time.perf_counter()
        for epoch in range(start,epochs+1):
            net.train();total=0.
            # Each row is visited once per epoch: no 3000-row cap, no sampled time window.
            batches=[(c,z,ix) for _,c,z in tr for ix in torch.randperm(len(z['q'])).split(p['batch'])]
            for j in torch.randperm(len(batches)).tolist():
                c,z,ix=batches[j];b=subset(z,ix,device,torch.float32)
                value=(b['weight']*loss(net,b,c,force)).sum()*len(batches)
                opt.zero_grad(set_to_none=True);value.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True);opt.step();total+=float(value.detach())/len(batches)
            val=validation(net,va,force,device,p['batch']);scheduler.step(val)
            if val<best-1e-5:
                best=val;bestep=epoch;save(path/'best.pt',dict(state=cpu_state(net),seed=sd,epoch=epoch,validation=val))
            trace.append(dict(epoch=epoch,train=total,validation=val,lr=opt.param_groups[0]['lr'],seconds=time.perf_counter()-begun))
            save(path/'last.pt',dict(state=cpu_state(net),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,
                best=best,best_epoch=bestep,trace=trace,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None))
            if epoch%5==0 or epoch==1:print(f'Train seed={sd} epoch={epoch}: train={total:.5g}, validation={val:.5g}',flush=True)
            if epoch>=min_ep and epoch-bestep>=p['patience']:break
        dump(path/'trace.json',trace);reports.append(dict(seed=sd,best=best,epoch=bestep,last_epoch=epoch,
               stopping='patience' if epoch<epochs else 'epoch_budget_not_proof_of_convergence'))
    winner=min(reports,key=lambda r:r['best']);save(out/'selected.pt',load(out/f"seed_{winner['seed']}"/'best.pt'))
    dump(out/'complete.json',dict(seeds=reports,selected=winner,initialization='fresh random weights; no legacy checkpoint',
        loss='physical latent NLL + 0.1 fixed-force observation NLL',field_outputs=False,
        conditioning='reference N40 K12 local slip and gradients; no position/time/PID input',smoke=smoke))


def model(run,device):
    net=Density().double().to(device);net.load_state_dict(load(run/'training/selected.pt')['state']);net.eval()
    for p in net.parameters():p.requires_grad_(False)
    return net


def calibration(run,cfg,device,smoke=False):
    if (run/'calibration.json').exists():return
    net=model(run,device);force=json.loads((run/'force.json').read_text())
    d=load(run/'prepared'/cfg['validation']/'training.pt');c=load(run/'prepared'/cfg['validation']/'observations.pt')['c']
    z=d['z'];ww=weights(z,'train');z=z|dict(weight=ww)
    p=cfg['calibration'];raw=torch.nn.Parameter(torch.zeros(4,device=device,dtype=torch.float64))
    opt=torch.optim.Adam([raw],lr=.03);trace=[]
    def decode():
        return dict(latent=[raw[0].clamp(-math.log(4),math.log(4)),0.],sign_temperature=raw[1].clamp(-1.4,1.4).exp(),
                    sigma={k:(.3*raw[2+j].exp()).clamp(.015,1.5) for j,k in enumerate(('free','cap'))})
    for step in range(2 if smoke else p['steps']):
        opt.zero_grad(set_to_none=True);val=0.
        for kind,s in [('free',0),('cap',1)]:
            ids=torch.where((ww>0)&(z['state']==s))[0]
            for ix in ids.split(p['batch']):
                b=subset(z,ix,device);vv,gg,aa=[v[ix].to(device) for v in d['jets']]
                pred,cov,lw=distribution(c,b,vv,gg,aa,kind,net,decode(),force,p['degree'])
                value=(b['weight']*scores(pred,cov,lw,b['target'])).sum();value.backward();val+=float(value.detach())
        opt.step();trace.append(dict(step=step,nll=val))
        if step%20==0:print('Calibration',step,val,flush=True)
    cal=decode();cal['latent']=[float(v.detach()) if torch.is_tensor(v) else v for v in cal['latent']]
    cal['sign_temperature']=float(cal['sign_temperature'].detach());cal['sigma']={k:float(v.detach()) for k,v in cal['sigma'].items()}
    dump(run/'calibration.json',cal);dump(run/'calibration_trace.json',trace)
