"""One field optimization/selection/evaluation controller for channel datasets."""
import math
import time
import numpy as np
from runtime import *
import field_score as SCORE
import workflow as protocol
import workflow as adaptation

def context(out,row,seed,device):
    pack=load(contained(out,row['path'])/'inversion_observations.pt')
    meta,z=pack['meta'],move(pack['z'],device)
    if not bool(((z['pid']%5>=3)|(z['pid']%5==1)).all()):
        raise ValueError('Field input may contain I/V only')
    c,op,cold,prior=F.prepare_physics(meta['physics_config'],meta['intervals'],device=device)
    field=F.make_field(c,cold,seed,duration_s=meta['intervals']*.001)
    net,cal,force,digest=adaptation.frozen(out,row['reference'],seed,device)
    return dict(c=c,z=z,op=op,prior=prior,field=field,net=net,cal=cal,force=force,digest=digest)

def score(ctx,group,degree,z=None,**kw):
    return SCORE.score(ctx['c'],ctx['z'] if z is None else z,ctx['net'],ctx['cal'],ctx['force'],
        lambda zz:F.queries(ctx['c'],ctx['field'],zz),group,degree,batch=128,batch_pids=32,**kw)

def step(ctx,opt,settings):
    field,z=ctx['field'],ctx['z']
    opt.zero_grad(set_to_none=True)
    fit=score(ctx,'I',settings['train_degree'],backward=True)
    times=torch.arange(2*ctx['prior'].intervals+1).to(field.cold)*.0005
    a,adot=field.coefficients_and_dt(times)
    nfit=float((z['end']-z['start'])[P.group_mask(z['pid'],'I')].sum())
    pde,detail=F.residual_loss(a,adot,ctx['op'],ctx['prior'],nfit)
    reg,parts=field.regularization()
    (pde+reg).backward()
    norm=torch.nn.utils.clip_grad_norm_(field.parameters(),settings['gradient_clip'],error_if_nonfinite=True)
    if not all(bool(torch.isfinite(v).all()) for v in (a,adot,pde,reg)):
        raise FloatingPointError('Nonfinite field objective')
    opt.step()
    return dict(inversion=fit,objective=.08*fit['total']+float((pde+reg).detach()),
        pde_loss=float(pde.detach()),regularization=float(reg.detach()),gradient_norm=float(norm))

def eligible(candidate,initial,s):
    return all(math.isfinite(candidate[k]) and candidate[k]<=initial[k]+s['guard'] for k in ('free','cap'))

def run(out,row,seed,device='cuda',max_new_steps=None):
    p=protocol.verify(out);s=p['config']['field'];budgets=s['budgets']
    folder=Path(out)/'fields'/row['name']/f'seed_{seed}'
    with lock(folder):
        ctx=context(out,row,seed,device);field,net=ctx['field'],ctx['net']
        before=state(net);opt=torch.optim.Adam(field.parameters(),lr=s['lr'])
        def snapshot(k):
            with torch.no_grad():v=score(ctx,'V',s['validation_degree'])
            return dict(step=k,field_state=state(field),validation=v,
                probability_sha256=ctx['digest'],protocol_fingerprint=p['fingerprint'])
        k=0;trace=[];wall=0.
        if (folder/'last.pt').exists():
            last=load(folder/'last.pt')
            if last['protocol_fingerprint']!=p['fingerprint'] or last['probability_sha256']!=ctx['digest']:
                raise RuntimeError('Field checkpoint protocol or probability changed')
            field.load_state_dict(last['field_state']);opt.load_state_dict(last['optimizer'])
            k,best,initial,trace,wall=[last[x] for x in ('step','best','initial','trace','wall_seconds')]
        else:
            started=time.perf_counter();best=snapshot(0);initial=best['validation']
            save(folder/'background.pt',best);wall+=time.perf_counter()-started
        start=k
        if k==max(budgets):
            dump(folder/'status.json',dict(step=k,complete=True,protocol_fingerprint=p['fingerprint'],
                probability_sha256=ctx['digest'],network_updates=0,wall_seconds=wall,selected_step=best['step']))
        while k<max(budgets):
            started=time.perf_counter();result=step(ctx,opt,s);k+=1
            if k%s['validate_every']==0 or k in budgets:
                candidate=snapshot(k);v=candidate['validation']
                chosen=eligible(v,initial,s) and v['total']<best['validation']['total']-s['selection_min_delta']
                if chosen:best=candidate
                result.update(validation=v,selected=chosen)
                if k in budgets:
                    save(folder/f'budget_{k}_primary.pt',best)
                    save(folder/f'budget_{k}_terminal.pt',candidate)
            if any(x.requires_grad or x.grad is not None for x in net.parameters()) or any(
                not torch.equal(v.detach().cpu(),before[n]) for n,v in net.state_dict().items()):
                raise RuntimeError('Frozen probability changed during field optimization')
            wall+=time.perf_counter()-started;trace.append(dict(step=k,**result,wall_seconds=wall))
            info=dict(step=k,complete=k==max(budgets),protocol_fingerprint=p['fingerprint'],
                probability_sha256=ctx['digest'],network_updates=0,wall_seconds=wall)
            save(folder/'last.pt',dict(**info,field_state=state(field),optimizer=opt.state_dict(),
                best=best,initial=initial,trace=trace))
            dump(folder/'status.json',dict(**info,selected_step=best['step']))
            dump(folder/'history.json',trace)
            print('field',row['name'],seed,k,'selected',best['step'],flush=True)
            if max_new_steps is not None and k-start>=max_new_steps:break
        return read(folder/'status.json')

def freeze(out):
    p=protocol.verify(out);records=[]
    prob=read(Path(out)/'probability_frozen.json')
    for row in p['targets']:
        for seed in p['config']['field_seeds']:
            folder=Path(out)/'fields'/row['name']/f'seed_{seed}'
            status=read(folder/'status.json')
            last=load(folder/'last.pt')
            model=next(x for x in prob['models'] if x['seed']==seed and x['case']==row['reference'])
            if not status['complete'] or status['protocol_fingerprint']!=p['fingerprint'] or status['probability_sha256']!=model['sha256']:
                raise RuntimeError('Every target/seed must finish before any H or truth evaluation')
            for label in ['background']+[f'budget_{b}_{which}' for b in p['config']['field']['budgets'] for which in ('primary','terminal')]:
                path=folder/(label+'.pt');ck=load(path)
                if ck['protocol_fingerprint']!=p['fingerprint'] or ck['probability_sha256']!=model['sha256']:
                    raise RuntimeError('Field snapshot binding mismatch')
                if label=='background':expected_step=0
                else:
                    budget=int(label.split('_')[1]);expected_step=budget
                    if label.endswith('_primary'):
                        expected_step=0;best=last['initial']
                        for item in last['trace']:
                            v=item.get('validation')
                            if item['step']<=budget and v is not None and eligible(v,last['initial'],p['config']['field']):
                                if v['total']<best['total']-p['config']['field']['selection_min_delta']:
                                    best=v;expected_step=item['step']
                if ck['step']!=expected_step:raise RuntimeError('Field snapshot differs from the V-only selection trace')
                records.append(dict(case=row['name'],seed=seed,label=label,step=ck['step'],
                    path=path.relative_to(out).as_posix(),sha256=sha(path),probability_sha256=model['sha256']))
    gate=dict(complete=True,protocol_fingerprint=p['fingerprint'],records=records,
        probability_gate_sha256=sha(Path(out)/'probability_frozen.json'),H_used_for_selection=False,
        direct_field_truth_used_in_pinn_loss=False,
        same_realization_reference_truth_used_for_probability_joint=True,
        data_role=p['config']['data_role'],target_truth_isolated_blind_reconstruction=False)
    path=Path(out)/'field_selection_frozen.json'
    if path.exists() and read(path)!=gate:raise RuntimeError('Field freeze gate changed')
    dump(path,gate);return gate

@torch.no_grad()
def grid(ctx,times,x,y):
    field=ctx['field'];yy,xx=np.meshgrid(y,x,indexing='ij')
    points=torch.as_tensor(np.stack((xx,yy),-1).reshape(-1,2)).to(field.cold)
    basis=F.geometry(ctx['c'],device=field.cold.device,dtype=field.cold.dtype).evaluate(points)
    a=field(torch.as_tensor(times).to(field.cold))
    v=basis['v0']+torch.einsum('tfm,pfmj->tpfj',a,basis['V'])
    return v.reshape(len(times),len(y),len(x),2,2).cpu().numpy()

def evaluate(out,device='cuda'):
    p=protocol.verify(out);gate=read(Path(out)/'field_selection_frozen.json')
    if not gate['complete'] or gate['protocol_fingerprint']!=p['fingerprint'] or gate['probability_gate_sha256']!=sha(Path(out)/'probability_frozen.json'):
        raise RuntimeError('Missing or changed selection freeze gate')
    for r in gate['records']:
        if sha(contained(out,r['path']))!=r['sha256']:raise RuntimeError('Frozen field snapshot changed')
    summaries=[]
    for row in p['targets']:
        path=contained(out,row['path'])
        heldout=load(path/'heldout_observations.pt')['z']
        if not bool((heldout['pid']%5==0).all()):raise ValueError('Held-out pack contains non-H rows')
        # Experimental data has no field truth and must never acquire a fake RMSE.
        truth=None
        if row['kind']=='cf_simulation':
            with np.load(path/'reference_evaluation.npz',allow_pickle=False) as d:truth=dict(d)
        for seed in p['config']['field_seeds']:
            ctx=context(out,row,seed,device)
            z=move(heldout,device)
            for r in [v for v in gate['records'] if v['case']==row['name'] and v['seed']==seed]:
                ctx['field'].load_state_dict(load(contained(out,r['path']))['field_state'])
                with torch.no_grad():metric=score(ctx,'H',p['config']['field']['validation_degree'],z=z,metrics=True)
                if truth is not None:
                    times,x,y=truth['time_s'],truth['x_m'],truth['y_m']
                    valid=(times>=0)&(times<=ctx['c']['intervals']*.001+1e-10)
                    times=times[valid]
                else:
                    times=np.linspace(0,ctx['c']['intervals']*.001,51)
                    x=np.linspace(0,.016,65);y=np.arange(32)*.008/32
                pred=grid(ctx,times,x,y)
                errors=None
                if truth is not None:
                    weights=np.ones((len(y),len(x)));weights[:,(0,-1)]=.5;weights/=weights.sum()
                    errors={}
                    for f,key in enumerate(('vn','vs')):
                        e=float(np.einsum('tyx,yx->t',((pred[...,f,:]-truth[key][valid])**2).sum(-1),weights).mean())
                        scale=float(np.einsum('tyx,yx->t',(truth[key][valid]**2).sum(-1),weights).mean())
                        errors[key]=dict(vector_rmse_m_s=math.sqrt(e),relative_L2=math.sqrt(e/scale))
                export=Path(out)/'evaluation'/row['name']/f'seed_{seed}'/(r['label']+'.npz')
                export.parent.mkdir(parents=True,exist_ok=True)
                with export.open('wb') as stream:np.savez_compressed(stream,time_s=times,x_m=x,y_m=y,vn=pred[...,0,:],vs=pred[...,1,:])
                summaries.append(dict(case=row['name'],kind=row['kind'],seed=seed,label=r['label'],
                    selected_step=r['step'],H=metric,field_errors=errors,probability_sha256=ctx['digest'],
                    export=export.relative_to(out).as_posix()))
                dump(Path(out)/'evaluation/summary.json',dict(complete=False,runs=summaries,
                    data_role=p['config']['data_role'],diagnostic_only=p['config']['diagnostic_only']))
    dump(Path(out)/'evaluation/summary.json',dict(complete=True,protocol_fingerprint=p['fingerprint'],runs=summaries,
        data_role=p['config']['data_role'],diagnostic_only=p['config']['diagnostic_only'],
        independent_field_generalization=False,target_truth_isolated_blind_reconstruction=False))
