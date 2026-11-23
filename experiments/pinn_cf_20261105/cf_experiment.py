"""Six CF inversions; all four budget snapshots freeze before test evaluation."""
from cf_common import *
import cf_field as F
import cf_probability as Q
from cf_history import attach_causal_history
import argparse
import math

STEPS = 600
VALIDATE_EVERY = 10

def folder_for(case, seed):
    return OUT/'runs'/case/f'seed_{seed}'

def context(case, seed):
    net, cal, force = probability(seed)
    c, z, op, cold, prior = F.load_case(case)
    z = attach_causal_history(c, z)
    field = F.make_field(c, cold, seed)
    return c, z, op, prior, net, cal, force, field

def score(c, z, field, net, cal, force, group, degree, **kw):
    return Q.score(c, z, net, cal, force, lambda zz: F.queries(c, field, zz), group, degree, **kw)

def field_step(c, z, op, prior, field, opt, net, cal, force, degree=12):
    opt.zero_grad(set_to_none=True)
    fit = score(c, z, field, net, cal, force, 'fit', degree, backward=True)
    t = torch.arange(201, dtype=field.cold.dtype)*.0005
    a, adot = field.coefficients_and_dt(t)
    nfit = float((z['end']-z['start'])[group_mask(z['pid'], 'fit')].sum())
    pde, detail = F.residual_loss(a, adot, op, prior, nfit)
    reg, parts = field.regularization()
    (pde+reg).backward()
    norm = torch.nn.utils.clip_grad_norm_(field.parameters(), 5., error_if_nonfinite=True)
    if not all(torch.isfinite(x).all() for x in (a, adot, pde, reg)):
        raise FloatingPointError('Nonfinite field objective')
    opt.step()
    return dict(inversion=fit, objective=.08*fit['total']+float((pde+reg).detach()),
                pde_loss=float(pde.detach()), regularization=float(reg.detach()),
                regularization_parts={k:float(v.detach()) for k,v in parts.items()},
                **{k:float(v.detach()) for k,v in detail.items()}, gradient_norm=float(norm))

@torch.no_grad()
def snapshot(c, z, field, net, cal, force, case, seed, step, protocol):
    val = score(c, z, field, net, cal, force, 'selection', protocol['validation_degree'])
    a, adot = field.coefficients_and_dt(torch.arange(201, dtype=field.cold.dtype)*.0005)
    return dict(case=case, seed=seed, step=step, field_state=state(field), a=a[::2].clone(),
                a_half=a.clone(), adot_half=adot.clone(), val=val,
                protocol_fingerprint=protocol['fingerprint'], probability_sha256=probability_record(seed)['sha256'],
                network_steps=0, initial_or_increment_parameters=False)

def eligible(v, warm, p):
    return all(math.isfinite(v[k]) for k in ('total','free','cap')) and all(
        v[k] <= warm[k]+p['guard_nll_per_ms'] for k in ('free','cap'))

def selected_from_trace(trace, warm, budget, p):
    best = trace[0]
    for row in trace[1:]:
        if row['step'] > budget or 'validation' not in row:
            continue
        v = row['validation']
        if eligible(v, warm, p) and v['total'] < best['validation']['total']-p['selection_delta_nll_per_ms']:
            best = row
    return best

def run(case, seed, max_new_steps=None):
    p = verify_protocol()
    assert case in CASES and seed in SEEDS
    c, z, op, prior, net, cal, force, field = context(case, seed)
    net_before = state(net)
    folder = folder_for(case, seed)
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder/'run.lock'
    with lock.open('x', encoding='utf8') as s:
        s.write(json.dumps(dict(pid=os.getpid(), started=utc())))
    opt = torch.optim.Adam(field.parameters(), lr=p['field_lr'])
    step = 0
    cpu = wall = 0.
    snap = lambda k: snapshot(c,z,field,net,cal,force,case,seed,k,p)
    try:
        if (folder/'last.pt').exists():
            ck = load(folder/'last.pt')
            assert ck['protocol_fingerprint'] == p['fingerprint']
            assert ck['probability_sha256'] == probability_record(seed)['sha256']
            field.load_state_dict(ck['field_state'])
            opt.load_state_dict(ck['optimizer'])
            step, trace, selected, warm, cpu, wall = [ck[k] for k in
                ('step','trace','selected','warm','cpu_seconds','wall_seconds')]
        else:
            ct, wt = time.process_time(), time.perf_counter()
            selected = snap(0)
            warm = selected['val']
            save(folder/'background.pt', selected)
            trace = [dict(step=0, validation=warm, selected=True)]
            cpu += time.process_time()-ct
            wall += time.perf_counter()-wt

        def persist():
            assert all(torch.equal(v, net_before[k]) for k,v in net.state_dict().items())
            assert not any(x.requires_grad or x.grad is not None for x in net.parameters())
            common = dict(case=case, seed=seed, step=step, steps=STEPS,
                protocol_fingerprint=p['fingerprint'], probability_sha256=probability_record(seed)['sha256'],
                network_steps=0, cpu_seconds=cpu, wall_seconds=wall)
            save(folder/'last.pt', dict(**common, field_state=state(field), optimizer=opt.state_dict(),
                                      trace=trace, selected=selected, warm=warm))
            dump(folder/'history.json', trace)
            dump(folder/'status.json', dict(**common, complete=step==STEPS, pid=os.getpid(),
                selected_step=selected['step'], validation=selected['val'], network_parameters_equal=True,
                field_parameters=sum(x.numel() for x in field.parameters()), updated_utc=utc()))

        persist()
        start = step
        while step < STEPS:
            ct, wt = time.process_time(), time.perf_counter()
            row = field_step(c,z,op,prior,field,opt,net,cal,force,p['train_degree'])
            step += 1
            if step % VALIDATE_EVERY == 0:
                cand = snap(step)
                v = cand['val']
                chosen = eligible(v,warm,p) and v['total'] < selected['val']['total']-p['selection_delta_nll_per_ms']
                if chosen:
                    selected = cand
                row.update(validation=v, eligible=eligible(v,warm,p), selected=chosen,
                           selected_step=selected['step'])
                print(case,seed,step,'V',v['total'],'selected',selected['step'],flush=True)
                if step in BUDGETS:
                    save(folder/f'budget_{step}_primary.pt', selected)
                    save(folder/f'budget_{step}_terminal.pt', cand)
            cpu += time.process_time()-ct
            wall += time.perf_counter()-wt
            row.update(step=step, cpu_seconds=cpu, wall_seconds=wall)
            trace.append(row)
            if step%VALIDATE_EVERY == 0 or (max_new_steps is not None and step-start >= max_new_steps):
                persist()
            if max_new_steps is not None and step-start >= max_new_steps:
                break
    except BaseException as error:
        dump(folder/'failure.json', dict(case=case,seed=seed,step=step,error=repr(error),updated_utc=utc()))
        raise
    finally:
        lock.unlink(missing_ok=True)

@torch.no_grad()
def freeze():
    p = verify_protocol()
    rows = []
    for case in CASES:
        for seed in SEEDS:
            folder = folder_for(case,seed)
            status, last = read(folder/'status.json'), load(folder/'last.pt')
            assert status['complete'] and status['step']==last['step']==STEPS
            assert status['network_parameters_equal'] and last['network_steps']==0
            assert status['protocol_fingerprint']==last['protocol_fingerprint']==p['fingerprint']
            c,z,op,prior,net,cal,force,field = context(case,seed)
            def bound(path, expected_step):
                ck = load(path)
                assert ck['protocol_fingerprint']==p['fingerprint'] and ck['step']==expected_step
                assert ck['probability_sha256']==probability_record(seed)['sha256'] and ck['network_steps']==0
                field.load_state_dict(ck['field_state'])
                check = snapshot(c,z,field,net,cal,force,case,seed,ck['step'],p)
                assert all(torch.equal(ck[k],check[k]) for k in ('a','a_half','adot_half'))
                assert max(abs(ck['val'][k]-check['val'][k]) for k in ('total','free','cap'))<1e-9
                return dict(path=str(path),sha256=sha(path),step=ck['step'])
            background = bound(folder/'background.pt',0)
            budgets = {}
            for budget in BUDGETS:
                best = selected_from_trace(last['trace'],last['warm'],budget,p)
                budgets[str(budget)] = {
                    'primary':bound(folder/f'budget_{budget}_primary.pt',best['step']),
                    'terminal':bound(folder/f'budget_{budget}_terminal.pt',budget)}
            rows.append(dict(case=case,seed=seed,background=background,budgets=budgets,
                probability_sha256=probability_record(seed)['sha256'],
                status_path=str(folder/'status.json'),status_sha256=sha(folder/'status.json'),
                last_path=str(folder/'last.pt'),last_sha256=sha(folder/'last.pt')))
    gate = dict(complete=True,protocol_fingerprint=p['fingerprint'],runs=rows,
                all_24_budget_snapshots_and_6_backgrounds_frozen=True,network_steps=0,
                test_used_for_selection=False,frozen_utc=utc())
    path = OUT/'selection_frozen.json'
    if path.exists():
        old = read(path)
        assert {k:v for k,v in old.items() if k!='frozen_utc'} == {k:v for k,v in gate.items() if k!='frozen_utc'}
    else:
        dump(path,gate)
    print('ALL_CF_SELECTIONS_FROZEN',len(rows),flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action',choices=('run','freeze'))
    ap.add_argument('--case',choices=CASES,default=CASES[0])
    ap.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0])
    ap.add_argument('--max-new-steps',type=int)
    args = ap.parse_args()
    setup()
    if args.action == 'run': run(args.case,args.seed,args.max_new_steps)
    else: freeze()
