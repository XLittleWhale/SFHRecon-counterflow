"""Initial-state weak 4D-Var with frozen learned dynamics and observations."""
from common import *
import argparse
import time
import numpy as np

STEPS=200
VALIDATE_EVERY=10

def training_gate():
    p=read(OUT/'protocol.json');g=read(OUT/'training/selection_frozen.json')
    assert g['all_required_selections_frozen']
    assert g['protocol_fingerprint']==p['fingerprint']
    assert fingerprint({k:v for k,v in g.items() if k!='fingerprint'})==g['fingerprint']
    assert len(g['runs'])==9 and {(r['arm'],r['seed']) for r in g['runs']}=={(a,s) for a in NEW_ARMS for s in TRAIN_SEEDS}
    for r in g['runs']:
        assert sha(r['path'])==r['sha256']
        assert sha(r['last_path'])==r['last_sha256']
        assert sha(r['status_path'])==r['status_sha256']
        status=read(r['status_path'])
        assert status['complete'] and status['protocol_fingerprint']==p['fingerprint']
    return g

def trained_record(arm,seed):
    ts=TRAIN_SEEDS[SEEDS.index(seed)]
    if arm=='baseline':
        g=read(VOUT/'training/selection_frozen.json')
        r=next(r for r in g['runs'] if r['seed']==ts)
    else:
        g=read(OUT/'training/selection_frozen.json')
        r=next(r for r in g['runs'] if r['seed']==ts and r['arm']==arm)
    assert sha(r['path'])==r['sha256']
    return ts,r

def make_context(U,seed,arm,*,initial_network=False):
    assert arm in ARMS
    d,cfg,model,_,cal,force=S.P.make_context(U,seed,'new_frozen')
    net,_=baseline_network(seed)
    if arm!='baseline':
        from closure_model import ClosureNet,ClosureMomentum
        ck=load(trained_record(arm,seed)[1]['path'])
        net.load_state_dict(ck['net_state'])
        if arm in ('closure','joint'):
            settings=read(OUT/'protocol.json')['closure_settings']
            closure=ClosureNet(**settings).double()
            closure.load_state_dict(ck['closure_state'])
            closure.eval()
            for par in closure.parameters():par.requires_grad_(False)
            model.op=ClosureMomentum(model.op,closure)
    net.freeze();net.mode='temporal'
    return d,cfg,model,net,cal,force

def closure_state(model):
    return plain_state(model.op.net) if hasattr(model.op,'net') else {}

def field_step(model,opt,d,cfg,net,cal,force):
    assert not any(p.requires_grad for p in net.parameters())
    if hasattr(model.op,'net'):assert not any(p.requires_grad for p in model.op.net.parameters())
    before=model.theta.detach().clone()
    opt.zero_grad(set_to_none=True)
    a,es,en=model()
    if not all(torch.isfinite(v).all() for v in (a,es,en)):raise FloatingPointError('Nonfinite field')
    leaf,sl,nl=[x.detach().requires_grad_() for x in (a,es,en)]
    fit=score(leaf,sl,nl,d,net,cal,force,'I',12,backward=True)
    reg,detail=M.regularization(model,a,es,en,cfg,int(group_mask(d['z']['pid'],'I').sum()))
    grads=[x.grad if x.grad is not None else torch.zeros_like(x) for x in (leaf,sl,nl)]
    torch.autograd.backward([a,es,en,reg],[*grads,torch.ones_like(reg)])
    norms={g['name']:float(torch.nn.utils.clip_grad_norm_(g['params'],5.,error_if_nonfinite=True)) for g in opt.param_groups}
    opt.step()
    # Parseval on the solenoidal sin/cos basis: velocity-vector RMS per fluid.
    update=((model.theta.detach()-before).square().sum(-1)/2).sqrt()*model.op.U*1000
    return dict(inversion=fit,objective=.08*fit['total']+float(reg.detach()),
        regularization={k:float(v.detach()) for k,v in detail.items()},gradient_norms=norms,
        initial_velocity_update_rms_mm_s=update.tolist())

@torch.no_grad()
def snapshot(model,net,d,cal,force,U,seed,arm,k,p,ts,tr):
    a,es,en=model();v=score(a,es,en,d,net,cal,force,'V',20)
    return dict(a=a.clone(),eta_s=es.clone(),eta_n=en.clone(),theta=a[0].clone(),
        model_state=plain_state(model),net_state=plain_state(net),closure_state=closure_state(model),
        U=U,seed=seed,arm=arm,iteration=k,validation_components=v,fingerprint=p['fingerprint'],
        training_seed=ts,training_sha=tr['sha256'],
        query_convention='subtract residual endpoint increments only; deterministic closure integrated in RHS')

def run(U,seed,arm,max_new_steps=None):
    assert arm in NEW_ARMS
    p=verify_protocol();training_gate();ts,tr=trained_record(arm,seed)
    d,cfg,model,net,cal,force=make_context(U,seed,arm)
    net_before=plain_state(net);cl_before=closure_state(model)
    folder=OUT/'runs'/case_name(U)/f'seed_{seed}'/arm;folder.mkdir(parents=True,exist_ok=True)
    groups=model.parameter_groups()
    for g in groups:g['lr']=p['field_lr'][g['name']]
    opt=torch.optim.Adam(groups);iteration=0;cpu=wall=0.
    snap=lambda k:snapshot(model,net,d,cal,force,U,seed,arm,k,p,ts,tr)
    if (folder/'last.pt').exists():
        ck=load(folder/'last.pt');assert ck['fingerprint']==p['fingerprint'] and ck['training_sha']==tr['sha256']
        assert all(torch.equal(v,net_before[k]) for k,v in ck['net_state'].items())
        assert set(ck['closure_state'])==set(cl_before) and all(torch.equal(v,cl_before[k]) for k,v in ck['closure_state'].items())
        model.load_state_dict(ck['model_state']);opt.load_state_dict(ck['optimizer'])
        iteration,trace,selected,warm,cpu,wall=[ck[k] for k in ('iteration','trace','selected','warm_validation','cpu_seconds','wall_seconds')]
    else:
        c,w=time.process_time(),time.perf_counter();selected=snap(0);warm=selected['validation_components']
        trace=[dict(iteration=0,validation=warm,selected=True)]
        cpu+=time.process_time()-c;wall+=time.perf_counter()-w
    start=iteration
    def persist():
        assert all(torch.equal(v,net_before[k]) for k,v in net.state_dict().items())
        assert all(torch.equal(v,cl_before[k]) for k,v in closure_state(model).items())
        save(folder/'selected.pt',selected)
        save(folder/'last.pt',dict(fingerprint=p['fingerprint'],model_state=plain_state(model),net_state=net_before,
            closure_state=cl_before,optimizer=opt.state_dict(),iteration=iteration,trace=trace,selected=selected,
            warm_validation=warm,cpu_seconds=cpu,wall_seconds=wall,training_seed=ts,training_sha=tr['sha256'],
            U=U,seed=seed,arm=arm,network_steps=0))
        dump(folder/'history.json',trace)
        dump(folder/'status.json',dict(complete=iteration==STEPS,iteration=iteration,steps=STEPS,
            selected_iteration=selected['iteration'],validation=selected['validation_components'],network_steps=0,
            cpu_seconds=cpu,wall_seconds=wall,case=case_name(U),arm=arm,seed=seed,training_seed=ts,
            training_sha=tr['sha256'],fingerprint=p['fingerprint'],reused=False))
    persist()
    while iteration<STEPS:
        c,w=time.process_time(),time.perf_counter();row=field_step(model,opt,d,cfg,net,cal,force);iteration+=1
        if iteration%VALIDATE_EVERY==0:
            cand=snap(iteration);v=cand['validation_components']
            eligible=all(np.isfinite(v[k]) for k in ('free','cap','total')) and all(v[k]<=warm[k]+.01 for k in ('free','cap'))
            chosen=eligible and v['total']<selected['validation_components']['total']-1e-6
            if chosen:selected=cand
            row.update(validation=v,eligible=eligible,selected=chosen,selected_iteration=selected['iteration'],
                guard_failed={k:v[k]>warm[k]+.01 for k in ('free','cap')})
            print(case_name(U),seed,arm,iteration,'V',v['total'],'selected',selected['iteration'],flush=True)
        cpu+=time.process_time()-c;wall+=time.perf_counter()-w
        row.update(iteration=iteration,cpu_seconds=cpu,wall_seconds=wall);trace.append(row)
        if iteration%VALIDATE_EVERY==0 or (max_new_steps is not None and iteration-start>=max_new_steps):persist()
        if max_new_steps is not None and iteration-start>=max_new_steps:break

@torch.no_grad()
def reuse():
    p=verify_protocol();training_gate();oldgate=read(VOUT/'selection_frozen.json')
    for U in CASES:
        for seed in SEEDS:
            r=next(r for r in oldgate['runs'] if r['arm']=='temporal' and r['seed']==seed and r['case']==case_name(U))
            assert sha(r['path'])==r['sha256'] and sha(r['last_path'])==r['last_sha256']
            ts,tr=trained_record('baseline',seed);d,cfg,model,net,cal,force=make_context(U,seed,'baseline')
            oldselected,oldlast=load(r['path']),load(r['last_path'])
            assert all(torch.equal(v,net.state_dict()[k]) for k,v in oldselected['net_state'].items())
            changes=dict(arm='baseline',fingerprint=p['fingerprint'],training_seed=ts,training_sha=tr['sha256'],
                         net_state=plain_state(net),closure_state={},reuse_source=r)
            selected=dict(oldselected,**changes);last=dict(oldlast,**changes);last['selected']=selected
            for ck in (selected,last):
                model.load_state_dict(ck['model_state']);a,es,en=model()
                if ck is selected:
                    assert all(torch.equal(x,ck[k]) for x,k in ((a,'a'),(es,'eta_s'),(en,'eta_n')))
                expected=ck['validation_components'] if ck is selected else ck['trace'][-1]['validation']
                got=score(a,es,en,d,net,cal,force,'V',20)
                assert max(abs(got[k]-expected[k]) for k in got)<1e-9
            folder=OUT/'runs'/case_name(U)/f'seed_{seed}'/'baseline';folder.mkdir(parents=True,exist_ok=True)
            save(folder/'selected.pt',selected);save(folder/'last.pt',last);dump(folder/'history.json',last['trace'])
            dump(folder/'status.json',dict(complete=True,iteration=200,steps=200,selected_iteration=selected['iteration'],
                validation=selected['validation_components'],network_steps=0,cpu_seconds=0.,wall_seconds=0.,
                historical_cpu_seconds=oldlast['cpu_seconds'],historical_wall_seconds=oldlast['wall_seconds'],
                case=case_name(U),arm='baseline',seed=seed,training_seed=ts,training_sha=tr['sha256'],
                fingerprint=p['fingerprint'],reused=True,reuse_source=r))
            print('REUSED',case_name(U),seed,flush=True)

@torch.no_grad()
def freeze():
    p=verify_protocol();training_gate();rows=[]
    for U in CASES:
        for seed in SEEDS:
            for arm in ARMS:
                folder=OUT/'runs'/case_name(U)/f'seed_{seed}'/arm
                status,last,selected=read(folder/'status.json'),load(folder/'last.pt'),load(folder/'selected.pt')
                assert status['complete'] and last['iteration']==status['iteration']==200
                assert last['fingerprint']==selected['fingerprint']==status['fingerprint']==p['fingerprint']
                assert last['network_steps']==0
                best=last['trace'][0];warm=last['warm_validation']
                for row in last['trace'][1:]:
                    if 'validation' not in row:continue
                    v=row['validation']
                    if all(np.isfinite(v[k]) for k in ('free','cap','total')) and all(v[k]<=warm[k]+.01 for k in ('free','cap')) and v['total']<best['validation']['total']-1e-6:best=row
                assert best['iteration']==selected['iteration']==status['selected_iteration']
                ts,tr=trained_record(arm,seed);d,cfg,model,net,cal,force=make_context(U,seed,arm)
                assert all(item['training_sha']==tr['sha256'] and item['training_seed']==ts for item in (selected,last,status))
                cs=closure_state(model)
                for ck in (selected,last):
                    assert set(ck['closure_state'])==set(cs) and all(torch.equal(v,cs[k]) for k,v in ck['closure_state'].items())
                    assert all(torch.equal(v,net.state_dict()[k]) for k,v in ck['net_state'].items())
                    model.load_state_dict(ck['model_state']);a,es,en=model()
                    if ck is selected:
                        assert all(torch.equal(x,ck[k]) for x,k in ((a,'a'),(es,'eta_s'),(en,'eta_n')))
                    got=score(a,es,en,d,net,cal,force,'V',20)
                    expected=ck['validation_components'] if ck is selected else ck['trace'][-1]['validation']
                    assert max(abs(got[k]-expected[k]) for k in got)<1e-9
                rows.append(dict(case=case_name(U),U=U,seed=seed,arm=arm,path=str(folder/'selected.pt'),sha256=sha(folder/'selected.pt'),
                    last_path=str(folder/'last.pt'),last_sha256=sha(folder/'last.pt'),selected_iteration=selected['iteration'],
                    training_seed=ts,training_sha=tr['sha256'],training_epoch=load(tr['path'])['epoch'],reused=arm=='baseline'))
    gate=dict(all_required_selections_frozen=True,all_terminal_states_bound_before_evaluation=True,runs=rows,
        fingerprint=p['fingerprint'],training_gate_sha256=sha(OUT/'training/selection_frozen.json'),
        target_H_used_for_selection=False,target_truth_used_for_selection=False)
    dump(OUT/'selection_frozen.json',gate);print('FIELD_SELECTIONS_FROZEN',len(rows),flush=True)
    return gate

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','reuse','freeze'])
    parser.add_argument('--U',type=float,default=.004);parser.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0])
    parser.add_argument('--arm',choices=NEW_ARMS,default='joint');parser.add_argument('--max-new-steps',type=int)
    args=parser.parse_args();setup()
    if args.action=='run':run(args.U,args.seed,args.arm,args.max_new_steps)
    elif args.action=='reuse':reuse()
    else:freeze()
