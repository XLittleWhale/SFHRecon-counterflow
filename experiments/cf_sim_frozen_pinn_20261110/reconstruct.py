"""Native-time CF field inversion using the original frozen joint probability law."""
from common import *
import argparse
import math
import field_adapter as F
import native_probability as Q

def context(case, seed):
    pack = load(prepared(case)/'inversion_observations.pt')
    meta, z = pack['meta'], move_tensors(pack['z'])
    c, op, cold, prior = F.prepare_physics(meta['physics_config'], meta['intervals'], device=device())
    field = F.make_field(c, cold, seed, duration_s=meta['intervals']*.001)
    net, cal, force, digest = probability(case, seed, device=device())
    return c, z, op, prior, field, net, cal, force, digest

def score(c, z, field, net, cal, force, group, degree, **kw):
    return Q.score(c, z, net, cal, force, lambda zz:F.queries(c, field, zz),
                   group, degree, batch=128, batch_pids=32, **kw)

def field_step(c, z, op, prior, field, opt, net, cal, force, degree=12):
    opt.zero_grad(set_to_none=True)
    fit = score(c,z,field,net,cal,force,'I',degree,backward=True)
    times = torch.arange(2*prior.intervals+1).to(field.cold)*.0005
    a, adot = field.coefficients_and_dt(times)
    nfit = float((z['end']-z['start'])[Q.group_mask(z['pid'],'I')].sum())
    pde, detail = F.residual_loss(a,adot,op,prior,nfit)
    reg, parts = field.regularization()
    (pde+reg).backward()
    norm = torch.nn.utils.clip_grad_norm_(field.parameters(),5.,error_if_nonfinite=True)
    if not all(torch.isfinite(x).all() for x in (a,adot,pde,reg)):
        raise FloatingPointError('Nonfinite field objective')
    opt.step()
    return dict(inversion=fit,objective=.08*fit['total']+float((pde+reg).detach()),
        pde_loss=float(pde.detach()),regularization=float(reg.detach()),
        regularization_parts={k:float(v.detach()) for k,v in parts.items()},
        **{k:float(v.detach()) for k,v in detail.items()},gradient_norm=float(norm))

@torch.no_grad()
def snapshot(c,z,field,net,cal,force,case,seed,step,p,digest):
    val = score(c,z,field,net,cal,force,'V',p['field']['validation_degree'])
    a,adot = field.coefficients_and_dt(torch.arange(2*c['intervals']+1).to(field.cold)*.0005)
    return dict(case=case,seed=seed,step=step,field_state=state(field),a=a[::2].cpu(),
        a_half=a.cpu(),adot_half=adot.cpu(),val=val,protocol_fingerprint=p['fingerprint'],
        probability_sha256=digest,network_steps_during_inversion=0)

def eligible(v,warm,p):
    return all(math.isfinite(v[k]) for k in ('total','free','cap')) and all(
        v[k] <= warm[k]+p['field']['guard_nll_per_ms'] for k in ('free','cap'))

def selected_from_trace(trace,warm,budget,p):
    best = trace[0]
    for row in trace[1:]:
        if row['step'] <= budget and 'validation' in row and eligible(row['validation'],warm,p):
            if row['validation']['total'] < best['validation']['total']-p['field']['selection_delta_nll_per_ms']:
                best = row
    return best

def run(case,seed,max_new_steps=None):
    p = verify_protocol()
    c,z,op,prior,field,net,cal,force,digest = context(case,seed)
    before = state(net)
    folder = field_folder(case,seed)
    folder.mkdir(parents=True,exist_ok=True)
    lock = folder/'run.lock'
    with lock.open('x',encoding='utf8') as s:
        json.dump(dict(pid=os.getpid(),started_utc=utc()),s)
    opt = torch.optim.Adam(field.parameters(),lr=p['field']['lr'])
    step = 0
    cpu = wall = 0.
    snap = lambda k:snapshot(c,z,field,net,cal,force,case,seed,k,p,digest)
    try:
        if (folder/'last.pt').exists():
            ck = load(folder/'last.pt')
            assert ck['protocol_fingerprint']==p['fingerprint'] and ck['probability_sha256']==digest
            field.load_state_dict(ck['field_state'])
            opt.load_state_dict(ck['optimizer'])
            step,trace,selected,warm,cpu,wall = [ck[k] for k in
                ('step','trace','selected','warm','cpu_seconds','wall_seconds')]
        else:
            started = time.perf_counter()
            selected = snap(0)
            warm = selected['val']
            save(folder/'background.pt',selected)
            trace = [dict(step=0,validation=warm,selected=True)]
            wall += time.perf_counter()-started

        def persist():
            assert all(torch.equal(v.cpu(),before[k]) for k,v in net.state_dict().items())
            assert not any(x.requires_grad or x.grad is not None for x in net.parameters())
            info = dict(case=case,seed=seed,step=step,steps=max(BUDGETS),
                protocol_fingerprint=p['fingerprint'],probability_sha256=digest,
                network_steps_during_inversion=0,cpu_seconds=cpu,wall_seconds=wall)
            save(folder/'last.pt',dict(**info,field_state=state(field),optimizer=opt.state_dict(),
                trace=trace,selected=selected,warm=warm))
            dump(folder/'history.json',trace)
            dump(folder/'status.json',dict(**info,complete=step==max(BUDGETS),pid=os.getpid(),
                selected_step=selected['step'],validation=selected['val'],network_parameters_equal=True,
                field_parameters=sum(x.numel() for x in field.parameters()),updated_utc=utc()))

        persist()
        start = step
        while step < max(BUDGETS):
            ct,wt = time.process_time(),time.perf_counter()
            row = field_step(c,z,op,prior,field,opt,net,cal,force,p['field']['train_degree'])
            step += 1
            if step % p['field']['validate_every'] == 0 or step in BUDGETS:
                cand = snap(step)
                v = cand['val']
                chosen = eligible(v,warm,p) and v['total'] < selected['val']['total']-p['field']['selection_delta_nll_per_ms']
                if chosen:
                    selected = cand
                row.update(validation=v,eligible=eligible(v,warm,p),selected=chosen,selected_step=selected['step'])
                if step in BUDGETS:
                    save(folder/f'budget_{step}_primary.pt',selected)
                    save(folder/f'budget_{step}_terminal.pt',cand)
            cpu += time.process_time()-ct
            wall += time.perf_counter()-wt
            row.update(step=step,cpu_seconds=cpu,wall_seconds=wall)
            trace.append(row)
            persist()  # every step: native camera likelihood steps are expensive
            print(case,seed,'field',step,'objective',row['objective'],'selected',selected['step'],flush=True)
            if max_new_steps is not None and step-start >= max_new_steps:
                break
        return read(folder/'status.json')
    except BaseException as error:
        dump(folder/'failure.json',dict(case=case,seed=seed,step=step,error=repr(error),updated_utc=utc()))
        raise
    finally:
        lock.unlink(missing_ok=True)

def freeze():
    p = verify_protocol()
    rows = []
    for case in CASES:
        for seed in SEEDS:
            folder = field_folder(case,seed)
            status,last = read(folder/'status.json'),load(folder/'last.pt')
            assert status['complete'] and status['step']==last['step']==max(BUDGETS)
            assert status['network_parameters_equal'] and last['network_steps_during_inversion']==0
            assert last['protocol_fingerprint']==p['fingerprint']
            probability_source = probability_record(seed)
            digest = sha(probability_source['path'])
            assert digest == probability_source['sha256']
            assert digest==last['probability_sha256']
            def bound(path,expected_step):
                ck = load(path)
                assert ck['step']==expected_step and ck['protocol_fingerprint']==p['fingerprint']
                assert ck['probability_sha256']==digest and ck['network_steps_during_inversion']==0
                return dict(path=path.relative_to(OUT).as_posix(),sha256=sha(path),step=ck['step'])
            budgets = {}
            for budget in BUDGETS:
                best = selected_from_trace(last['trace'],last['warm'],budget,p)
                budgets[str(budget)] = dict(primary=bound(folder/f'budget_{budget}_primary.pt',best['step']),
                    terminal=bound(folder/f'budget_{budget}_terminal.pt',budget))
            rows.append(dict(case=case,seed=seed,background=bound(folder/'background.pt',0),
                budgets=budgets,probability_sha256=digest))
    gate = dict(complete=True,protocol_fingerprint=p['fingerprint'],runs=rows,
        test_used_for_selection=False,reference_fields_used_in_inversion=False,
        probability_source='Nov5 frozen joint baseline',
        probability_training_uses_same_case_F_truth_context=False,
        probability_optimization_steps=0,frozen_utc=utc())
    if (OUT/'selection_frozen.json').exists():
        old=read(OUT/'selection_frozen.json')
        assert {k:v for k,v in old.items() if k!='frozen_utc'}=={k:v for k,v in gate.items() if k!='frozen_utc'}
    else:
        dump(OUT/'selection_frozen.json',gate)
    print('ALL_FROZEN_PROBABILITY_FIELD_SELECTIONS_FROZEN',len(rows),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=('run','freeze'))
    ap.add_argument('--case',choices=CASES,default=CASES[0])
    ap.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0])
    ap.add_argument('--max-new-steps',type=int)
    args=ap.parse_args()
    setup()
    if args.action=='freeze': freeze()
    else: run(args.case,args.seed,args.max_new_steps)
