"""Frozen transition priors with unchanged two-fluid initial/increment controls."""
from common import *
import argparse
import time
import numpy as np
import latent_model as LM
import train_latent as TNEW
from scoring import score

STEPS=200
VALIDATE_EVERY=10
verify_protocol=TNEW.verify_protocol
plain_state=TNEW.plain_state
group_mask=S.group_mask

def training_gate():
    p=read(OUT/'protocol.json');g=read(OUT/'training/selection_frozen.json')
    b={k:v for k,v in g.items() if k!='fingerprint'}
    assert fingerprint(b)==g['fingerprint'] and g['all_required_selections_frozen']
    assert g['protocol_fingerprint']==p['fingerprint']
    assert {r['seed'] for r in g['runs']}==set(TRAIN_SEEDS) and len(g['runs'])==3
    for row in g['runs']:
        assert sha(row['path'])==row['sha256'] and sha(row['status_path'])==row['status_sha256']
    return g

def trained_record(seed):
    ts=TRAIN_SEEDS[SEEDS.index(seed)]
    g=read(OUT/'training/selection_frozen.json')
    r=next(r for r in g['runs'] if r['seed']==ts)
    assert sha(r['path'])==r['sha256']
    return ts,r

def make_context(U,seed,arm,*,initial_network=False):
    assert arm in ARMS
    d,cfg,model,base,cal,force=S.make_context(U,seed,'independent')
    state=None if initial_network else load(trained_record(seed)[1]['path'])['state']
    head=LM.LatentTransitionHead(base,state=state).double()
    net=Network(base,head).freeze();net.mode=arm
    return d,cfg,model,net,cal,force

def field_step(model,opt,d,cfg,net,cal,force):
    if any(p.requires_grad for p in net.parameters()):raise RuntimeError('Target network must be frozen')
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
    return dict(inversion=fit,objective=.08*fit['total']+float(reg.detach()),
                regularization={k:float(v.detach()) for k,v in detail.items()},gradient_norms=norms)

@torch.no_grad()
def snapshot(model,net,d,cal,force,U,seed,arm,iteration,p,ts,tr):
    a,es,en=model();v=score(a,es,en,d,net,cal,force,'V',20)
    return dict(a=a.clone(),eta_s=es.clone(),eta_n=en.clone(),theta=a[0].clone(),
        model_state=plain_state(model),net_state=plain_state(net),U=U,seed=seed,arm=arm,
        iteration=iteration,validation_components=v,fingerprint=p['fingerprint'],training_seed=ts,training_sha=tr['sha256'])

def run(U,seed,arm,max_new_steps=None):
    if arm=='iid':raise ValueError('iid uses provenance-checked frozen field reuse')
    p=verify_protocol();training_gate();ts,tr=trained_record(seed)
    d,cfg,model,net,cal,force=make_context(U,seed,arm)
    net_before=plain_state(net)
    folder=OUT/'runs'/case_name(U)/f'seed_{seed}'/arm;folder.mkdir(parents=True,exist_ok=True)
    groups=model.parameter_groups()
    for g in groups:g['lr']=p['field_lr'][g['name']]
    opt=torch.optim.Adam(groups);iteration=0;cpu=wall=0.
    snap=lambda k:snapshot(model,net,d,cal,force,U,seed,arm,k,p,ts,tr)
    if (folder/'last.pt').exists():
        ck=load(folder/'last.pt');assert ck['fingerprint']==p['fingerprint'] and ck['training_sha']==tr['sha256']
        model.load_state_dict(ck['model_state']);opt.load_state_dict(ck['optimizer'])
        iteration,trace,selected,warm,cpu,wall=[ck[k] for k in ('iteration','trace','selected','warm_validation','cpu_seconds','wall_seconds')]
    else:
        c,w=time.process_time(),time.perf_counter();selected=snap(0);warm=selected['validation_components']
        trace=[dict(iteration=0,validation=warm,selected=True)]
        cpu+=time.process_time()-c;wall+=time.perf_counter()-w
    start=iteration
    def persist():
        assert all(torch.equal(v,net_before[k]) for k,v in net.state_dict().items())
        save(folder/'selected.pt',selected)
        save(folder/'last.pt',dict(fingerprint=p['fingerprint'],model_state=plain_state(model),net_state=net_before,
            optimizer=opt.state_dict(),iteration=iteration,trace=trace,selected=selected,warm_validation=warm,
            cpu_seconds=cpu,wall_seconds=wall,training_seed=ts,training_sha=tr['sha256'],U=U,seed=seed,arm=arm,network_steps=0))
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
            row.update(validation=v,eligible=eligible,selected=chosen,selected_iteration=selected['iteration'])
            print(case_name(U),seed,arm,iteration,'V',v['total'],'selected',selected['iteration'],flush=True)
        cpu+=time.process_time()-c;wall+=time.perf_counter()-w
        row.update(iteration=iteration,cpu_seconds=cpu,wall_seconds=wall);trace.append(row)
        if iteration%VALIDATE_EVERY==0 or (max_new_steps is not None and iteration-start>=max_new_steps):persist()
        if max_new_steps is not None and iteration-start>=max_new_steps:break

@torch.no_grad()
def reuse_iid():
    p=verify_protocol();training_gate()
    oldgate=read(OLD_OUT/'selection_frozen.json')
    for U in CASES:
        for seed in SEEDS:
            r=next(r for r in oldgate['runs'] if r['arm']=='independent' and r['seed']==seed and r['case']==case_name(U))
            assert sha(r['path'])==r['sha256'] and sha(r['last_path'])==r['last_sha256']
            ts,tr=trained_record(seed);d,cfg,model,net,cal,force=make_context(U,seed,'iid')
            oldselected,oldlast=load(r['path']),load(r['last_path'])
            changes=dict(arm='iid',fingerprint=p['fingerprint'],training_seed=ts,training_sha=tr['sha256'],
                         net_state=plain_state(net),reuse_source=r)
            selected=dict(oldselected,**changes);last=dict(oldlast,**changes)
            last['selected']=selected
            for ck in (selected,last):
                model.load_state_dict(ck['model_state']);a,es,en=model()
                expected=ck['validation_components'] if ck is selected else ck['trace'][-1]['validation']
                got=score(a,es,en,d,net,cal,force,'V',20)
                assert max(abs(got[k]-expected[k]) for k in got)<1e-9
            folder=OUT/'runs'/case_name(U)/f'seed_{seed}'/'iid';folder.mkdir(parents=True,exist_ok=True)
            save(folder/'selected.pt',selected);save(folder/'last.pt',last)
            dump(folder/'history.json',last['trace'])
            dump(folder/'status.json',dict(complete=True,iteration=200,steps=200,selected_iteration=selected['iteration'],
                validation=selected['validation_components'],network_steps=0,cpu_seconds=0.,wall_seconds=0.,
                historical_cpu_seconds=oldlast['cpu_seconds'],historical_wall_seconds=oldlast['wall_seconds'],
                case=case_name(U),arm='iid',seed=seed,training_seed=ts,training_sha=tr['sha256'],fingerprint=p['fingerprint'],
                reused=True,reuse_source=r))
            print('REUSED',case_name(U),seed,flush=True)

@torch.no_grad()
def freeze():
    p=verify_protocol();tg=training_gate();rows=[]
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
                ts,tr=trained_record(seed);d,cfg,model,net,cal,force=make_context(U,seed,arm)
                for ck in (selected,last):
                    assert all(torch.equal(v,net.state_dict()[k]) for k,v in ck['net_state'].items())
                    model.load_state_dict(ck['model_state']);a,es,en=model()
                    got=score(a,es,en,d,net,cal,force,'V',20)
                    expected=ck['validation_components'] if ck is selected else ck['trace'][-1]['validation']
                    assert max(abs(got[k]-expected[k]) for k in got)<1e-9
                rows.append(dict(case=case_name(U),U=U,seed=seed,arm=arm,path=str(folder/'selected.pt'),sha256=sha(folder/'selected.pt'),
                    last_path=str(folder/'last.pt'),last_sha256=sha(folder/'last.pt'),selected_iteration=selected['iteration'],
                    training_seed=ts,training_sha=tr['sha256'],training_epoch=load(tr['path'])['epoch'],reused=arm=='iid'))
    gate=dict(all_required_selections_frozen=True,all_terminal_states_bound_before_evaluation=True,runs=rows,
        fingerprint=p['fingerprint'],training_gate_sha256=sha(OUT/'training/selection_frozen.json'),
        target_H_used_for_selection=False,target_truth_used_for_selection=False)
    dump(OUT/'selection_frozen.json',gate);print('FIELD_SELECTIONS_FROZEN',len(rows),flush=True)
    return gate

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','reuse','freeze'])
    parser.add_argument('--U',type=float,default=.004);parser.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0])
    parser.add_argument('--arm',choices=NEW_ARMS,default='temporal');parser.add_argument('--max-new-steps',type=int)
    args=parser.parse_args();setup()
    if args.action=='run':run(args.U,args.seed,args.arm,args.max_new_steps)
    elif args.action=='reuse':reuse_iid()
    else:freeze()
