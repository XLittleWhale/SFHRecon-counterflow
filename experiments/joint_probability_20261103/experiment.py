"""PID-isolated physics / probability-model adaptation experiment.

Historical implementations are read-only, version-hashed parent dependencies.
Target reference fields and heldout responses are scored only after global freeze.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'2')
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2]
OUT=BASE/'outputs/JointProbabilityPhysics_20261103'
PARENT=BASE/'outputs/PINNClosureFactorial_20261030'
sys.path.insert(0,str(PARENT))
import factorial_model as M
from prepare_external import load_case,case_name,observation_path
from physics.simulation_operator import Momentum
import network_finetune as NF
EW=M.EW
load,save=M.R.load,M.R.save
CASES=(.0015,.004)
SEEDS=(301030,301031,301032)
ARMS=('old_frozen','new_frozen','new_joint')
STEPS=200
VALIDATE_EVERY=10
WARM_STEPS=40
FINE_EVERY=5

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    os.replace(temporary,path)

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def group_mask(pid,group):
    r=pid%5
    if group=='F':return r==2
    if group=='I':return r>=3
    if group=='V':return r==1
    if group=='H':return r==0
    raise ValueError(group)

def weights(z,group):
    w=torch.zeros(len(z['pid']),dtype=torch.float64)
    mask=group_mask(z['pid'],group)
    for state in (0,1):
        ix=torch.where(mask & (z['state']==state))[0]
        if not len(ix):raise ValueError('Both particle states required in each group')
        _,inv,count=z['pid'][ix].unique(return_inverse=True,return_counts=True)
        w[ix]=1/(2*len(count)*count[inv].double())
    return w

def score(a,eta_s,eta_n,d,net,cal,force,group,degree=20,metrics=False,backward=False):
    z,c=d['z'],d['c'];w=weights(z,group);result={'total':0.}
    if metrics:result.update(coverage={},spread={},per_pid={})
    for kind,state in (('free',0),('cap',1)):
        ids=torch.where((w>0)&(z['state']==state))[0]
        value=0.;mass=0.;parts={}
        for ix in ids.split(256):
            zz=EW.subset(z,ix,a.device)
            v,g,acc=M.R.local_jets(a,eta_s,eta_n,zz,c)
            pred,cov,lw=EW.distribution(c,zz,v,g,acc,kind,net,cal,force,degree)
            nll=EW.fast_scores(pred,cov,lw,zz['target'])
            ww=w[ix].to(a.device);loss=(ww*nll).sum()
            if backward:(.08*loss).backward()
            value+=float(loss.detach());mass+=float(ww.sum())
            if metrics:
                cover=(EW.coverage(pred,cov,lw,zz['target'])<=4.605170186).double()
                prob=lw.exp();mean=(prob[...,None]*pred).sum((1,2))
                trace=cov.diagonal(dim1=-2,dim2=-1).sum(-1)
                spread=(prob*(trace+(pred-mean[:,None,None]).square().sum(-1))).sum((1,2)).clamp_min(0).sqrt()*c['physics']['vn_mean_m_s']*.001*1e6
                for pid in zz['pid'].unique().tolist():
                    use=zz['pid']==pid;r=parts.setdefault(str(pid),dict(count=0,nll=0.,coverage=0.,spread=0.))
                    r['count']+=int(use.sum())
                    for key,tensor in [('nll',nll),('coverage',cover),('spread',spread)]:r[key]+=float(tensor[use].detach().sum())
        result[kind]=value/mass;result['total']+=value
        if metrics:
            for row in parts.values():
                for key in ('nll','coverage','spread'):row[key]/=row['count']
            result['per_pid'][kind]=parts
            result['coverage'][kind]=float(np.mean([x['coverage'] for x in parts.values()]))
            result['spread'][kind]=float(np.mean([x['spread'] for x in parts.values()]))
    return result

def make_context(U,seed,arm):
    d,cfg,old,_=load_case(U,EW=EW)
    op=Momentum(old,friction_factor=2/3,nu_s=float(old.nu_s)) if arm=='old_frozen' else Momentum(old)
    theta=torch.zeros((2,624),dtype=torch.float64)
    if seed!=SEEDS[0]:
        gen=torch.Generator().manual_seed(seed)
        ring=M.TemporalNormalPrior(U,model='white').ring.to(theta)
        theta+=torch.randn(theta.shape,generator=gen,dtype=theta.dtype)*(.001/ring.pow(1.5))
    es=torch.zeros((d['intervals'],624),dtype=theta.dtype);en=es.clone()
    with torch.no_grad():a=M.R.rollout(op,theta,es,en)
    model=M.make_model('B',op,dict(U=U,a=a,eta_s=es,eta_n=en),seed)
    net,cal,force=EW.components('cpu')
    return d,cfg,model,net,cal,force

def plain_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def field_step(model,opt,d,cfg,net,cal,force):
    for p in net.parameters():p.requires_grad_(False);p.grad=None
    opt.zero_grad(set_to_none=True)
    a,es,en=model()
    if not all(torch.isfinite(x).all() for x in (a,es,en)):raise FloatingPointError('Nonfinite trajectory')
    leaf,sl,nl=[x.detach().requires_grad_() for x in (a,es,en)]
    fit=score(leaf,sl,nl,d,net,cal,force,'I',12,backward=True)
    reg,detail=M.regularization(model,a,es,en,cfg,int(group_mask(d['z']['pid'],'I').sum()))
    gradients=[x.grad if x.grad is not None else torch.zeros_like(x) for x in (leaf,sl,nl)]
    torch.autograd.backward([a,es,en,reg],[*gradients,torch.ones_like(reg)])
    norms={g['name']:float(torch.nn.utils.clip_grad_norm_(g['params'],5.,error_if_nonfinite=True)) for g in opt.param_groups}
    opt.step()
    return dict(inversion=fit,objective=.08*fit['total']+float(reg.detach()),
                regularization={k:float(v.detach()) for k,v in detail.items()},gradient_norms=norms)

def source_paths():
    paths={HERE/'experiment.py',HERE/'network_finetune.py',HERE/'physics/simulation_operator.py'}
    dynamic=[M.R,M.N,M.EW,M.A,getattr(NF,'_noise',None)]
    for module in list(sys.modules.values())+dynamic:
        value=getattr(module,'__file__',None)
        if value:
            path=Path(value).resolve()
            if path.suffix=='.py' and path.is_relative_to(BASE):paths.add(path)
    return paths

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():return verify_protocol()
    scopes={}
    for U in CASES:
        d,cfg,model,net,cal,force=make_context(U,SEEDS[0],'new_frozen')
        z=d['z'];sets={g:set(z['pid'][group_mask(z['pid'],g)].tolist()) for g in ('F','I','V','H')}
        assert len(set.union(*sets.values()))==len(z['pid'].unique())
        assert all(not sets[a]&sets[b] for i,a in enumerate(sets) for b in list(sets)[i+1:])
        groups={g:dict(pids=sorted(ids),rows=int(group_mask(z['pid'],g).sum()),states={kind:dict(pids=len(z['pid'][group_mask(z['pid'],g)&(z['state']==state)].unique()),rows=int((group_mask(z['pid'],g)&(z['state']==state)).sum())) for kind,state in [('free',0),('cap',1)]}) for g,ids in sets.items()}
        scopes[case_name(U)]=dict(U=U,frames=d['intervals']+1,first_ms=5,last_ms=100,groups=groups,physical_pid_disjoint=True)
    replay,manifest=NF.prepare_replay(EW.RUN)
    save(OUT/'replay.pt',replay);dump(OUT/'replay_manifest.json',manifest)
    inputs=[observation_path(U) for U in CASES]+[EW.RUN/'training/selected.pt',EW.RUN/'calibration.json',EW.RUN/'force.json',EW.PACKAGE/'operator.pt',PARENT/'calibration/temporal_prior.json',M.SOURCE/'normal_calibration.pt',M.SOURCE/'calibration.pt',OUT/'replay.pt']
    sources={str(p):sha(p) for p in source_paths()}
    protocol=dict(version='joint-probability-physics-development-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        cases=list(CASES),seeds=list(SEEDS),arms=list(ARMS),steps=STEPS,validation_every=VALIDATE_EVERY,
        warm_steps=WARM_STEPS,fine_every=FINE_EVERY,field_lr=dict(theta=.0003,super=.009,normal=.009),
        fine_lr=.0001,teacher_kl_weight=.1,replay_weight=.1,field_clip=5.,network_clip=1.,
        train_degree=12,validation_degree=20,evaluation_degree=20,evaluation_sensitivity_degree=28,guard=.01,selection_delta=1e-6,
        trainable_network='sign and output heads only; 462 parameters; GRU/shared head/normalization/calibration/force frozen',
        split='F pid%5==2; I>=3; V==1; H==0; all times/states of physical PID stay together',
        update_routing='I updates field with network frozen; F captured observations update heads with detached fields, plus frozen teacher KL and old-case supervised replay',
        initialization='Zero closure; zero theta for seed301030, small paired initial theta perturbations for other seeds; no historical target warm checkpoints',
        field_budget='Exactly 200 field updates each arm; joint additionally 32 full F-network updates; report cost, not equal-compute claim',
        new_physics=dict(nu_s=0.,Fns_factor=1.),old_physics=dict(nu_s='legacy template',Fns_factor=2/3),
        process_prior='Frozen B slow normal / white super calibration under legacy RHS; not recalibrated for new operator',
        target_truth_used_in_fit=False,test_used_in_fit=False,all_selections_frozen_before_new_truth_test_scoring=True,
        historical_target_truth_and_test_seen=True,confirmatory=False,
        raw_simulation_regenerated=False,CF_full_inverse_rerun=False,scopes=scopes,sources=sources,
        inputs={str(p):sha(p) for p in inputs},threads_per_process=2)
    protocol['fingerprint']=fingerprint(protocol);dump(OUT/'protocol.json',protocol)
    print(json.dumps(dict(prepared=True,fingerprint=protocol['fingerprint'],runs=len(CASES)*len(SEEDS)*len(ARMS))))
    return protocol

def verify_protocol():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf8'));fp=p.pop('fingerprint')
    assert fingerprint(p)==fp;p['fingerprint']=fp
    assert p['steps']==STEPS and p['warm_steps']==WARM_STEPS and p['fine_every']==FINE_EVERY
    for group in ('sources','inputs'):
        for path,h in p[group].items():assert sha(path)==h,path
    return p

@torch.no_grad()
def snapshot(model,net,d,cal,force,U,seed,arm,iteration,fp):
    a,es,en=model();val=score(a,es,en,d,net,cal,force,'V',20)
    return dict(a=a.clone(),eta_s=es.clone(),eta_n=en.clone(),theta=a[0].clone(),model_state=plain_state(model),net_state=plain_state(net),
                U=U,seed=seed,arm=arm,iteration=iteration,validation_components=val,fingerprint=fp)

def run(U,seed,arm,max_new_steps=None):
    p=verify_protocol();fp=p['fingerprint'];d,cfg,model,net,cal,force=make_context(U,seed,arm)
    dest=OUT/'runs'/case_name(U)/f'seed_{seed}'/arm;dest.mkdir(parents=True,exist_ok=True)
    groups=model.parameter_groups()
    for g in groups:g['lr']=p['field_lr'][g['name']]
    opt=torch.optim.Adam(groups)
    teacher=copy.deepcopy(net);fine_cfg=NF.configure_heads(net)
    nopt=torch.optim.Adam(fine_cfg['params'],lr=p['fine_lr'])
    replay=load(OUT/'replay.pt') if arm=='new_joint' else None
    cpu=wall=0.;iteration=0;trace=[];network_steps=0
    if (dest/'last.pt').exists():
        ck=load(dest/'last.pt');assert ck['fingerprint']==fp
        model.load_state_dict(ck['model_state']);net.load_state_dict(ck['net_state'])
        opt.load_state_dict(ck['optimizer']);nopt.load_state_dict(ck['network_optimizer'])
        iteration,trace,selected,warmval,network_steps,cpu,wall=[ck[k] for k in ('iteration','trace','selected','warm_validation','network_steps','cpu_seconds','wall_seconds')]
    else:
        t,w=time.process_time(),time.perf_counter()
        selected=snapshot(model,net,d,cal,force,U,seed,arm,0,fp);warmval=selected['validation_components']
        cpu+=time.process_time()-t;wall+=time.perf_counter()-w
        trace=[dict(iteration=0,validation=warmval,selected=True)]
    start=iteration
    def persist():
        save(dest/'selected.pt',selected)
        save(dest/'last.pt',dict(fingerprint=fp,model_state=plain_state(model),net_state=plain_state(net),optimizer=opt.state_dict(),network_optimizer=nopt.state_dict(),
            iteration=iteration,trace=trace,selected=selected,warm_validation=warmval,network_steps=network_steps,cpu_seconds=cpu,wall_seconds=wall))
        dump(dest/'history.json',trace)
        dump(dest/'status.json',dict(complete=iteration==STEPS,iteration=iteration,steps=STEPS,network_steps=network_steps,selected_iteration=selected['iteration'],
            validation=selected['validation_components'],cpu_seconds=cpu,wall_seconds=wall,case=case_name(U),arm=arm,seed=seed,fingerprint=fp))
    persist()
    while iteration<STEPS:
        t,w=time.process_time(),time.perf_counter()
        row=field_step(model,opt,d,cfg,net,cal,force);iteration+=1
        if arm=='new_joint' and iteration>WARM_STEPS and (iteration-WARM_STEPS)%FINE_EVERY==0:
            with torch.no_grad():a,es,en=model()
            for name,param in net.named_parameters():param.requires_grad_(name in fine_cfg['names'])
            row['fine']=NF.network_step(a=a,es=es,en=en,d=d,cal=cal,force=force,student=net,teacher=teacher,optimizer=nopt,replay=replay)
            network_steps+=1
        if iteration%VALIDATE_EVERY==0 or iteration==STEPS:
            candidate=snapshot(model,net,d,cal,force,U,seed,arm,iteration,fp);val=candidate['validation_components']
            eligible=all(math.isfinite(val[k]) for k in ('total','free','cap')) and all(val[k]<=warmval[k]+.01 for k in ('free','cap'))
            chosen=eligible and val['total']<selected['validation_components']['total']-1e-6
            if chosen:selected=candidate
            row.update(validation=val,eligible=eligible,selected=chosen,selected_iteration=selected['iteration'])
            print(case_name(U),seed,arm,'step',iteration,'V',round(val['total'],6),'selected',selected['iteration'],flush=True)
        cpu+=time.process_time()-t;wall+=time.perf_counter()-w
        row.update(iteration=iteration,cpu_seconds=cpu,wall_seconds=wall);trace.append(row);persist()
        if max_new_steps is not None and iteration-start>=max_new_steps:break
    return selected

def freeze():
    p=verify_protocol();runs=[]
    for U in CASES:
        for seed in SEEDS:
            for arm in ARMS:
                folder=OUT/'runs'/case_name(U)/f'seed_{seed}'/arm
                status=json.loads((folder/'status.json').read_text());assert status['complete'] and status['iteration']==STEPS
                ck=load(folder/'last.pt');selected=load(folder/'selected.pt')
                assert selected['fingerprint']==p['fingerprint'] and ck['network_steps']==(32 if arm=='new_joint' else 0)
                best=ck['trace'][0];warm=ck['warm_validation']
                for row in ck['trace'][1:]:
                    if 'validation' not in row:continue
                    val=row['validation'];eligible=all(val[k]<=warm[k]+.01 for k in ('free','cap'))
                    if eligible and val['total']<best['validation']['total']-1e-6:best=row
                assert best['iteration']==selected['iteration']
                d,cfg,model,net,cal,force=make_context(U,seed,arm)
                model.load_state_dict(selected['model_state'])
                net.load_state_dict(selected['net_state'])
                with torch.no_grad():
                    a,es,en=model()
                    assert all(torch.equal(x,y) for x,y in zip((a,es,en),(selected['a'],selected['eta_s'],selected['eta_n'])))
                    val=score(a,es,en,d,net,cal,force,'V',20)
                assert max(abs(val[k]-selected['validation_components'][k]) for k in val)<1e-10
                runs.append(dict(case=case_name(U),U=U,seed=seed,arm=arm,path=str(folder/'selected.pt'),sha256=sha(folder/'selected.pt'),selected_iteration=selected['iteration']))
    result=dict(all_required_selections_frozen=True,runs=runs,fingerprint=p['fingerprint'],test_used_for_selection=False,truth_used_for_selection=False)
    dump(OUT/'selection_frozen.json',result);print(json.dumps(dict(frozen=True,runs=len(runs))))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','freeze'])
    parser.add_argument('--U',type=float,default=.004);parser.add_argument('--seed',type=int,default=SEEDS[0]);parser.add_argument('--arm',choices=ARMS,default='new_frozen');parser.add_argument('--max-new-steps',type=int)
    args=parser.parse_args();M.R.setup('cpu')
    if args.action=='prepare':prepare()
    elif args.action=='freeze':freeze()
    else:run(args.U,args.seed,args.arm,args.max_new_steps)
