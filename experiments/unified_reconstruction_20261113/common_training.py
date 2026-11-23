"""One A/B/closure/joint controller, independent of case geometry.

Geometry adapters own fields, quadrature, clocks and windows. Loss weights,
freezing, optimizers, stopping, selection and checkpoint semantics live here.
"""
import copy
import math
import random
import time
import numpy as np
from runtime import *
import baseline_api as A
import native_losses as N
import clock_contract as clock

STAGES=('A','B','closure','joint')


def settings(profile='baseline_retrain_all'):
    ab=dict(epochs=80,minimum_epochs=40,patience=20,batch=1024,lr=1e-4,weight_decay=1e-4,
        gradient_clip=5.,train_degree=6,validation_degree=20,selection_min_delta=1e-5,
        microbatch_rows=32,train_dtype='float32',validation_dtype='float64')
    joint=dict(epochs=40,minimum_epochs=20,patience=10,windows_per_case=2,inner_steps=2,inner_lr=.05,
        inner_gradient_clip=5.,lr_closure=1e-4,lr_probability=1e-5,weight_decay=0.,gradient_clip=5.,
        train_degree=6,validation_degree=12,minimum_state_pids=2,field_weight=1.,initial_weight=.5,
        endpoint_weight=.08,teacher_weight=1.,anchor_weight=.05,selection_min_delta=1e-5,substeps=2,
        full_window_amplitude_limit=1e6,full_window_growth_factor=4.,validation_windows=4)
    result=dict(A=ab.copy(),B=ab.copy(),closure=joint.copy(),joint=joint.copy())
    if profile=='continuation_short':
        for stage in ('A','B'):result[stage].update(epochs=20,minimum_epochs=8,patience=6)
        for stage in ('closure','joint'):result[stage].update(epochs=12,minimum_epochs=6,patience=4)
    elif profile=='smoke':
        for stage in STAGES:result[stage].update(epochs=1,minimum_epochs=1,patience=1,train_degree=2,validation_degree=2)
        for stage in ('A','B'):result[stage].update(batch=16,microbatch_rows=4)
        for stage in ('closure','joint'):result[stage].update(validation_windows=1)
    elif profile not in ('baseline_retrain_all','joint_continuation'):raise ValueError('Unknown shared training profile')
    return result


def rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
    if r['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(r['cuda'])


def values(metrics):
    return {k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in metrics.items()}


def copied(net,seed,device,stage='PINN'):
    model,*_=B.load(seed,device);model.load_state_dict(net.state_dict());A.configure_stage(model,stage)
    return model.double()


def _view(case,ids,dtype):
    return (N.subset(case['z'],ids,dtype),tuple(v[ids].to(dtype) for v in case['jets']),
        N.subset(case['labels'],ids,dtype),case['force_labels'][ids].to(dtype))


def _allowed(case,group):
    # A registered dataset supplies roles, never the geometry name.
    roles=case.get('roles',dict(F=lambda p:p%5==2,V=lambda p:p%5==1))
    return roles[group](case['z']['pid'])


def epoch_ab(case,net,stage,opt,s,smoke=False):
    z=case['z'];mask=_allowed(case,'F');device=z['pid'].device
    if stage=='A':
        jobs=[];weight=z['target'].new_zeros(len(z['pid']))
        for code,kind in ((0,'free'),(1,'cap')):
            ids=torch.where(mask&(z['state']==code))[0]
            if smoke:ids=ids[:8]
            weight[ids]=.5*N.pid_weights(z['pid'][ids]).to(weight)
            ids=ids[torch.randperm(len(ids),device=device)]
            jobs.extend((kind,ix) for ix in ids.split(s['batch']))
        random.shuffle(jobs);total=0.;grads=[]
        for kind,ids in jobs:
            opt.zero_grad(set_to_none=True)
            for ix in ids.split(s['microbatch_rows']):
                zz,jets,labels,force_labels=_view(case,ix,torch.float32)
                objective,_,_=N.marginal(case['c'],zz,jets,labels,force_labels,net,case['cal'],case['force'],kind,s['train_degree'])
                weighted=(weight[ix].float()*objective).sum()
                if not bool(torch.isfinite(weighted)):raise FloatingPointError('Nonfinite A objective')
                (weighted*len(jobs)).backward();total+=float(weighted.detach())
            grads.append(float(torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],s['gradient_clip'],error_if_nonfinite=True)))
            opt.step()
        return dict(objective=total,updates=len(jobs),gradient_max=max(grads))
    previous,current,_=clock.pairs({k:v.detach().cpu() for k,v in z.items()},mask.detach().cpu())
    previous,current=previous.to(device),current.to(device)
    if smoke:previous,current=previous[:16],current[:16]
    weight=N.pid_weights(z['pid'][current]).to(z['target'])
    order=torch.randperm(len(current),device=device);jobs=list(order.split(s['batch']))
    # Contexts/jets are fixed reference data; only the transition head learns.
    zz={k:v.float() if v.is_floating_point() else v for k,v in z.items()}
    labels={k:v.float() if v.is_floating_point() else v for k,v in case['labels'].items()}
    jets=tuple(v.float() for v in case['jets']);total=0.;grads=[]
    for job in jobs:
        opt.zero_grad(set_to_none=True)
        for ix in job.split(s['microbatch_rows']):
            loss=N.pair_loss(case['c'],zz,jets,labels,net,case['cal'],previous[ix],current[ix])
            weighted=(weight[ix].float()*loss).sum()
            if not bool(torch.isfinite(weighted)):raise FloatingPointError('Nonfinite B objective')
            (weighted*len(jobs)).backward();total+=float(weighted.detach())
        grads.append(float(torch.nn.utils.clip_grad_norm_(net.head.parameters(),s['gradient_clip'],error_if_nonfinite=True)))
        opt.step()
    return dict(objective=total,updates=len(jobs),gradient_max=max(grads),pairs=len(current))


def validation(case,adapter,net,closure,stage,s,seed,device,smoke=False):
    case=case.get('validation_case',case)
    model=copied(net,seed,device)
    if stage in ('A','B'):
        ids=torch.where(_allowed(case,'V'))[0]
        if smoke:
            ids=torch.cat([ids[case['z']['state'][ids]==state][:8] for state in (0,1)])
        zz,jets,_,_=_view(case,ids,torch.float64)
        with torch.no_grad():metric=N.window_nll(case['c'],zz,jets,model,case['cal'],case['force'],
            torch.ones(len(ids),device=ids.device,dtype=torch.bool),s['validation_degree'],marginals=stage=='A')
        metric=values(metric)
        return metric,metric['total'] if stage=='A' else metric['cap']
    starts=adapter.valid_starts(case,'V')
    selected=sorted(set(starts[round(i)] for i in np.linspace(0,len(starts)-1,s['validation_windows'])))
    op=adapter.closed(case,closure);metrics=[]
    for start in selected:
        w=adapter.window(case,start,'V',pid_limit=2 if smoke else None);kernels=adapter.kernels(w)
        with torch.enable_grad():a,inner=kernels['infer_initial'](w,op,model,s,create_graph=False)
        with torch.no_grad():
            field=kernels['field_errors'](a,w['truth'])
            likelihood=kernels['likelihood'](w,a,model,w['masks']['outer'],s['validation_degree'])
            teacher,_=kernels['teacher_loss'](w,op,s)
        metrics.append(values(dict(**field,free=likelihood['free'],cap=likelihood['cap'],teacher=teacher,
            inner_before=inner[0]['objective'],inner_after=inner[-1]['objective'])))
    result={k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
    result['windows']=[dict(start=k,**m) for k,m in zip(selected,metrics)]
    return result,None


def run_stage(out,case,adapter,seed,stage,protocol,parent=None,closure_parent=None,device='cuda',max_new_epochs=None):
    profile=protocol['config']['profile'];smoke=profile=='smoke';s=protocol['settings'][stage]
    folder=Path(out)/'probability'/case['case']/f'seed_{seed}'/stage
    training_seed=B.MODELS[seed][0]
    random.seed(training_seed);np.random.seed(training_seed);torch.manual_seed(training_seed)
    net,_,_,origin=B.load(seed,device)
    dependencies=dict(periodic=origin)
    if parent is not None:
        ck=load(parent)
        if ck['protocol_fingerprint']!=protocol['fingerprint'] or ck['case']!=case['case'] or ck['seed']!=seed:
            raise RuntimeError('Previous stage belongs to a different protocol/case/seed')
        net.load_state_dict(ck['net_state']);dependencies['parent_sha256']=sha(parent)
    closure=None
    if stage in ('closure','joint'):
        closure=adapter.ClosureNet().to(device)
        if stage=='joint':
            if closure_parent is None:raise RuntimeError('Joint requires same-protocol selected closure')
            ck=load(closure_parent);status=read(closure_parent.parent/'status.json')
            if not status['complete'] or ck['protocol_fingerprint']!=protocol['fingerprint'] or ck['case']!=case['case'] or ck['seed']!=seed:
                raise RuntimeError('Closure prerequisite is incomplete or mismatched')
            closure.load_state_dict(ck['closure_state']);dependencies['closure_sha256']=sha(closure_parent)
    A.configure_stage(net,stage)
    if stage in ('A','B'):net.float()
    frozen={k:v.detach().cpu().clone() for k,v in net.named_parameters() if not v.requires_grad}
    buffers={k:v.detach().cpu().clone() for k,v in net.named_buffers()}
    groups=[]
    if closure is not None:groups.append(dict(params=list(closure.parameters()),lr=s['lr_closure'],name='closure'))
    params=[p for p in net.parameters() if p.requires_grad]
    if params:groups.append(dict(params=params,lr=s['lr'] if stage in ('A','B') else s['lr_probability'],name='probability'))
    optimizer=(torch.optim.AdamW if stage in ('A','B') else torch.optim.Adam)(groups,weight_decay=s['weight_decay'])
    def snapshot(epoch,metric,score,stability):
        return dict(case=case['case'],seed=seed,training_seed=training_seed,stage=stage,epoch=epoch,net_state=state(net),
            closure_state={} if closure is None else state(closure),validation=copy.deepcopy(metric),
            selection_score=float(score),stability=stability,protocol_fingerprint=protocol['fingerprint'],
            dependencies=dependencies,profile=profile,normalization_inherited=True)
    with lock(folder):
        epoch=stale=0;wall=0.;stopped=False;trace=[]
        if (folder/'last.pt').exists():
            last=load(folder/'last.pt')
            if last['protocol_fingerprint']!=protocol['fingerprint'] or last['dependencies']!=dependencies:
                raise RuntimeError('Resume protocol/dependency mismatch')
            net.load_state_dict(last['net_state'])
            if closure is not None:closure.load_state_dict(last['closure_state'])
            optimizer.load_state_dict(last['optimizer']);restore_rng(last['rng'])
            epoch,stale,wall,stopped,trace,best,origin_validation=[last[k] for k in
                ('epoch','stale','wall_seconds','stopped_early','trace','best','origin_validation')]
            # A completed parent must retain its exact bytes. Re-serializing
            # an unchanged torch checkpoint can change its SHA and invalidate
            # an interrupted child stage's dependency record.
            if (epoch>=s['epochs'] or stopped) and (folder/'status.json').exists():
                status=read(folder/'status.json')
                if status.get('complete'):
                    selected=load(folder/'selected.pt')
                    if status['selected_sha256']!=sha(folder/'selected.pt') or selected['epoch']!=best['epoch']:
                        raise RuntimeError('Completed selected checkpoint changed')
                    for key in ('net_state','closure_state'):
                        if any(not torch.equal(v,best[key][k]) for k,v in selected[key].items()):raise RuntimeError('Completed selection differs from resume state')
                    return folder/'selected.pt'
        else:
            start=time.perf_counter();metric,score=validation(case,adapter,net,closure,stage,s,seed,device,smoke)
            origin_validation=metric
            vc=case.get('validation_case',case)
            stable=adapter.stability(vc,adapter.closed(vc,closure),s) if closure is not None else dict(passed=True,not_applicable=True)
            if not stable['passed']:raise RuntimeError('Epoch-zero closure stability failed')
            if score is None:score=1.
            best=snapshot(0,metric,score,stable);trace=[dict(epoch=0,validation=metric,score=score,selected=True,stability=stable)]
            wall+=time.perf_counter()-start
        def persist():
            complete=epoch>=s['epochs'] or stopped
            now=snapshot(epoch,trace[-1]['validation'],trace[-1]['score'],trace[-1].get('stability'))
            now.update(optimizer=optimizer.state_dict(),rng=rng(),stale=stale,wall_seconds=wall,
                stopped_early=stopped,trace=trace,best=best,origin_validation=origin_validation)
            save(folder/'last.pt',now);save(folder/'selected.pt',best)
            dump(folder/'history.json',trace)
            dump(folder/'status.json',dict(complete=complete,case=case['case'],seed=seed,training_seed=training_seed,stage=stage,epoch=epoch,
                selected_epoch=best['epoch'],selected_sha256=sha(folder/'selected.pt'),wall_seconds=wall,
                protocol_fingerprint=protocol['fingerprint'],dependencies=dependencies,profile=profile,
                diagnostic_only=smoke,data_role='same_realization_supervised_calibration'))
        persist();initial_epoch=epoch
        while epoch<s['epochs'] and not stopped:
            start=time.perf_counter()
            if stage in ('A','B'):training=epoch_ab(case,net,stage,optimizer,s,smoke)
            else:
                training=[];starts=adapter.valid_starts(case,'F');op=adapter.closed(case,closure)
                for _ in range(s['windows_per_case']):
                    w=adapter.window(case,random.choice(starts),'F',pid_limit=2 if smoke else None)
                    optimizer.zero_grad(set_to_none=True)
                    objective,details=adapter.kernels(w)['train_window'](w,op,net,stage,s)
                    if not bool(torch.isfinite(objective)):raise FloatingPointError('Nonfinite closure/joint objective')
                    objective.backward()
                    norms={g['name']:float(torch.nn.utils.clip_grad_norm_(g['params'],s['gradient_clip'],error_if_nonfinite=True)) for g in optimizer.param_groups}
                    optimizer.step();training.append(dict(start=w['start'],objective=float(objective.detach()),gradients=norms,**values(details)))
            epoch+=1
            for k,v in net.named_parameters():
                if k in frozen and not torch.equal(v.detach().cpu(),frozen[k]):raise RuntimeError('Frozen stage parameter changed: '+k)
                if not bool(torch.isfinite(v).all()):raise FloatingPointError('Nonfinite probability weight')
            for k,v in net.named_buffers():
                if not torch.equal(v.detach().cpu(),buffers[k]):raise RuntimeError('Inherited normalization changed')
            metric,score=validation(case,adapter,net,closure,stage,s,seed,device,smoke)
            if score is None:
                vc=case.get('validation_case',case)
                w=adapter.window(vc,adapter.valid_starts(vc,'V')[0],'V',pid_limit=2 if smoke else None)
                score=adapter.kernels(w)['composite'](metric,origin_validation)
            if not math.isfinite(score):raise FloatingPointError('Nonfinite validation score')
            better=score<best['selection_score']-s['selection_min_delta'];stable=None
            if better:
                vc=case.get('validation_case',case)
                stable=adapter.stability(vc,adapter.closed(vc,closure),s) if closure is not None else dict(passed=True,not_applicable=True)
            chosen=better and stable['passed']
            if chosen:best=snapshot(epoch,metric,score,stable);stale=0
            else:stale+=1
            stopped=epoch>=s['minimum_epochs'] and stale>=s['patience']
            wall+=time.perf_counter()-start
            trace.append(dict(epoch=epoch,validation=metric,score=score,selected=chosen,stability=stable,training=training,wall_seconds=wall))
            persist();print(case['case'],seed,stage,epoch,'selected',best['epoch'],'V',score,flush=True)
            if max_new_epochs is not None and epoch-initial_epoch>=max_new_epochs:break
    return folder/'selected.pt'
