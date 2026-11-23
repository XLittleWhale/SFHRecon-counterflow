"""Globally parameterized continuous two-fluid inversion with a frozen likelihood."""
from pinn_common import *
from field_model import NeuralField,symmetric_regularization,residual_loss
from global_spline import GlobalSpline
from continuous_query import score_nodes
import argparse
import time
import math
from datetime import datetime,timezone

BUDGETS=(200,600)
STEPS=600
VALIDATE_EVERY=10

def folder_for(U,seed,arm):
    return OUT/'runs'/case_name(U)/arm/f'seed_{seed}'

def make_field(arm,old,seed):
    cold=old.theta.detach().clone()
    ring=old.prior.ring
    if arm=='pinn':return NeuralField(cold,ring,seed=seed).double()
    if arm=='global_spline':return GlobalSpline(cold,ring).double()
    raise ValueError(arm)

def restore_field(U,seed,arm,checkpoint_path):
    d,cfg,old,net,cal,force=make_context(U,seed)
    ck=load(checkpoint_path)
    assert ck['arm']==arm and ck['seed']==seed and ck['U']==U
    assert ck['probability_sha256']==probability_record(seed)['sha256']
    field=make_field(arm,old,seed);field.load_state_dict(ck['field_state'])
    return d,cfg,old,net,cal,force,field,ck

def field_step(field,opt,old,d,net,cal,force,group='I'):
    opt.zero_grad(set_to_none=True)
    times=torch.arange(191,dtype=torch.float64)*.0005
    a,adot=field.coefficients_and_dt(times)
    if not all(torch.isfinite(v).all() for v in (a,adot)):
        raise FloatingPointError('Nonfinite continuous field')
    am,dm=a[1::2],adot[1::2]
    al,dl=[x.detach().requires_grad_() for x in (am,dm)]
    fit=score_nodes(al,dl,d,net,cal,force,group,12,backward=True)
    pde,detail=residual_loss(a,adot,old.op,old.std,old.prior,int(group_mask(d['z']['pid'],group).sum()))
    reg,parts=symmetric_regularization(a[::2],old.prior.ring)
    gradients=[x.grad if x.grad is not None else torch.zeros_like(x) for x in (al,dl)]
    torch.autograd.backward([am,dm,pde+reg],[*gradients,torch.ones_like(reg)])
    norm=torch.nn.utils.clip_grad_norm_(field.parameters(),5.,error_if_nonfinite=True)
    opt.step()
    return dict(inversion=fit,objective=.08*fit['total']+float((pde+reg).detach()),
        pde_loss=float(pde.detach()),symmetric_regularization=float(reg.detach()),
        regularization={k:float(v.detach()) for k,v in parts.items()},
        normal_whitened_energy=float(detail['normal_whitened_energy'].detach()),
        super_whitened_energy=float(detail['super_whitened_energy'].detach()),
        residual_velocity_step_rms_mm_s=detail['residual_velocity_step_rms'].detach().tolist(),
        gradient_norm=float(norm))

@torch.no_grad()
def snapshot(field,net,d,cal,force,U,seed,arm,step,p):
    am,dm=field.coefficients_and_dt((torch.arange(95,dtype=torch.float64)+.5)*DT)
    val=score_nodes(am,dm,d,net,cal,force,'V',20)
    return dict(field_state=plain_state(field),a=field.states().clone(),a_mid=am.clone(),adot_mid=dm.clone(),
        U=U,seed=seed,arm=arm,step=step,val=val,protocol_fingerprint=p['fingerprint'],
        probability_sha256=probability_record(seed)['sha256'],network_steps=0,
        field_trainable={k:v.numel() for k,v in field.named_parameters() if v.requires_grad},
        query_convention='continuous midpoint value and exact SI time derivative; no residual subtraction')

def eligible(v,warm):
    return all(math.isfinite(v[k]) for k in ('total','free','cap')) and all(v[k]<=warm[k]+.01 for k in ('free','cap'))

def run(U,seed,arm,max_new_steps=None):
    p=verify_protocol();assert U in CASES and seed in SEEDS and arm in ARMS
    d,cfg,old,net,cal,force=make_context(U,seed)
    field=make_field(arm,old,seed);net_before=plain_state(net)
    folder=folder_for(U,seed,arm);folder.mkdir(parents=True,exist_ok=True)
    opt=torch.optim.Adam(field.parameters(),lr=p['field_lr_by_arm'][arm])
    step=0;cpu=wall=0.;snap=lambda k:snapshot(field,net,d,cal,force,U,seed,arm,k,p)
    if (folder/'last.pt').exists():
        ck=load(folder/'last.pt')
        assert ck['protocol_fingerprint']==p['fingerprint'] and ck['probability_sha256']==probability_record(seed)['sha256']
        field.load_state_dict(ck['field_state']);opt.load_state_dict(ck['optimizer'])
        step,trace,selected,warm,cpu,wall=[ck[k] for k in ('step','trace','selected','warm','cpu_seconds','wall_seconds')]
    else:
        c,w=time.process_time(),time.perf_counter();selected=snap(0);warm=selected['val']
        trace=[dict(step=0,validation=warm,selected=True)]
        cpu+=time.process_time()-c;wall+=time.perf_counter()-w
    start=step
    def persist():
        assert all(torch.equal(v,net_before[k]) for k,v in net.state_dict().items())
        assert not any(x.requires_grad or x.grad is not None for x in net.parameters())
        save(folder/'selected.pt',selected)
        save(folder/'last.pt',dict(field_state=plain_state(field),optimizer=opt.state_dict(),step=step,
            trace=trace,selected=selected,warm=warm,cpu_seconds=cpu,wall_seconds=wall,
            U=U,seed=seed,arm=arm,protocol_fingerprint=p['fingerprint'],
            probability_sha256=probability_record(seed)['sha256'],network_steps=0))
        dump(folder/'history.json',trace)
        dump(folder/'status.json',dict(complete=step==STEPS,step=step,steps=STEPS,selected_step=selected['step'],
            validation=selected['val'],cpu_seconds=cpu,wall_seconds=wall,network_steps=0,network_parameters_equal=True,
            case=case_name(U),U=U,seed=seed,arm=arm,protocol_fingerprint=p['fingerprint'],
            probability_sha256=probability_record(seed)['sha256'],
            field_parameters=sum(x.numel() for x in field.parameters()),
            updated_utc=datetime.now(timezone.utc).isoformat()))
    persist()
    while step<STEPS:
        c,w=time.process_time(),time.perf_counter()
        row=field_step(field,opt,old,d,net,cal,force);step+=1
        if step%VALIDATE_EVERY==0:
            cand=snap(step);v=cand['val']
            chosen=eligible(v,warm) and v['total']<selected['val']['total']-1e-6
            if chosen:selected=cand
            row.update(validation=v,eligible=eligible(v,warm),selected=chosen,selected_step=selected['step'])
            print(case_name(U),seed,arm,step,'V',v['total'],'selected',selected['step'],flush=True)
            if step in BUDGETS:
                save(folder/f'budget_{step}_primary.pt',selected)
                save(folder/f'budget_{step}_terminal.pt',cand)
        cpu+=time.process_time()-c;wall+=time.perf_counter()-w
        row.update(step=step,cpu_seconds=cpu,wall_seconds=wall);trace.append(row)
        if step%VALIDATE_EVERY==0 or (max_new_steps is not None and step-start>=max_new_steps):persist()
        if max_new_steps is not None and step-start>=max_new_steps:break

@torch.no_grad()
def freeze():
    p=verify_protocol();rows=[]
    for U in CASES:
        for seed in SEEDS:
            for arm in ARMS:
                folder=folder_for(U,seed,arm);status=read(folder/'status.json');last=load(folder/'last.pt')
                assert status['complete'] and status['step']==last['step']==STEPS and last['network_steps']==0
                assert status['protocol_fingerprint']==last['protocol_fingerprint']==p['fingerprint']
                assert status['network_parameters_equal'] and status['probability_sha256']==probability_record(seed)['sha256']
                budgets={}
                for budget in BUDGETS:
                    best=last['trace'][0]
                    for row in last['trace'][1:]:
                        if row['step']>budget or 'validation' not in row:continue
                        if eligible(row['validation'],last['warm']) and row['validation']['total']<best['validation']['total']-1e-6:best=row
                    record={}
                    for mode in ('primary','terminal'):
                        path=folder/f'budget_{budget}_{mode}.pt'
                        d,cfg,old,net,cal,force,field,ck=restore_field(U,seed,arm,path)
                        assert ck['protocol_fingerprint']==p['fingerprint'] and ck['network_steps']==0
                        assert ck['step']==(best['step'] if mode=='primary' else budget)
                        check=snapshot(field,net,d,cal,force,U,seed,arm,ck['step'],p)
                        assert all(torch.equal(ck[k],check[k]) for k in ('a','a_mid','adot_mid'))
                        assert max(abs(ck['val'][k]-check['val'][k]) for k in ('total','free','cap'))<1e-9
                        record[mode]=dict(path=str(path),sha256=sha(path),step=ck['step'])
                    budgets[str(budget)]=record
                rows.append(dict(U=U,case=case_name(U),seed=seed,arm=arm,budgets=budgets,
                    probability_sha256=probability_record(seed)['sha256'],status_path=str(folder/'status.json'),
                    status_sha256=sha(folder/'status.json'),last_path=str(folder/'last.pt'),last_sha256=sha(folder/'last.pt')))
    gate=dict(complete=True,all_required_selections_frozen=True,all_terminal_states_bound_before_evaluation=True,
        runs=rows,protocol_fingerprint=p['fingerprint'],probability_gate_sha256=sha(OUT/'probability_selection/selection_frozen.json'),
        target_H_used_for_selection=False,target_reference_used_for_selection=False,
        frozen_utc=datetime.now(timezone.utc).isoformat())
    path=OUT/'selection_frozen.json'
    if path.exists():
        existing=read(path)
        assert {k:v for k,v in existing.items() if k!='frozen_utc'}=={k:v for k,v in gate.items() if k!='frozen_utc'},'Existing global freeze differs'
    else:dump(path,gate)
    print('GLOBAL_SELECTIONS_FROZEN',len(rows),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('run','freeze'))
    parser.add_argument('--U',type=float,default=.004);parser.add_argument('--seed',type=int,default=301030)
    parser.add_argument('--arm',choices=ARMS,default='pinn');parser.add_argument('--max-new-steps',type=int)
    args=parser.parse_args();setup()
    if args.action=='run':run(args.U,args.seed,args.arm,args.max_new_steps)
    else:freeze()
