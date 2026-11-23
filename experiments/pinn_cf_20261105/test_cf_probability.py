"""Bounded synthetic/non-target-V checks; never opens CF test observations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import torch
import cf_probability as P


def preconditioned_thirty_step_preflight(out):
    """Execute the already written one-candidate plan, without any selection."""
    import ctypes
    import gc
    import statistics
    import cf_common as C
    import cf_experiment as E
    import cf_field as F
    C.setup()
    plan_path = C.OUT/'preflight/B_preconditioning_30step_plan.json'
    plan = C.read(plan_path)
    assert plan['frozen'] and plan['field_seed'] == 301030
    assert plan['optimizer']['steps_per_case'] == 30 and plan['optimizer']['learning_rate'] == .001
    assert C.fingerprint({k:v for k,v in plan.items() if k != 'plan_fingerprint'}) == plan['plan_fingerprint']
    if out.exists():
        raise FileExistsError('Preserve a previous preflight result; this is not a retry/search entry')
    sources = {str(C.HERE/name):C.sha(C.HERE/name) for name in
               ('cf_common.py','cf_field.py','cf_probability.py','cf_history.py','cf_experiment.py','test_cf_probability.py')}
    inputs = {str(P.SIM/'probability_selection/selection_frozen.json'):P.sha(P.SIM/'probability_selection/selection_frozen.json')}
    for name in ('calibration.json','force.json'):
        path = P.CALIBRATION/name
        inputs[str(path)] = P.sha(path)
    for case in C.CASES:
        for path in (C.BASE/'github/data/inputs'/f'{case}.pt',
                     F.RUNTIME/'payload/server_package/inputs'/case/'observations.pt'):
            if path.exists():
                inputs[str(path)] = C.sha(path)
    class PMC(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong),
            ('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),
            ('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),
            ('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),
            ('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
    def memory():
        p=PMC();p.cb=ctypes.sizeof(p)
        ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(PMC),ctypes.c_ulong]
        ok=ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(p),p.cb)
        return dict(peak_working_set_bytes=int(p.PeakWorkingSetSize),working_set_bytes=int(p.WorkingSetSize),
                    peak_pagefile_bytes=int(p.PeakPagefileUsage)) if ok else None
    prior = C.read(C.OUT/'preflight/optimization.json')
    start_wall,start_cpu = time.perf_counter(),time.process_time()
    report = dict(complete=False,passed=False,started_utc=C.utc(),plan_path=str(plan_path),
        plan_sha256=C.sha(plan_path),plan_fingerprint=plan['plan_fingerprint'],sources=sources,inputs=inputs,
        rows=[],no_test_score=True,no_weight_or_checkpoint_selection=True,no_warmstart_weights_saved=True,
        prior_failed_two_step_attempt=dict(wall_seconds=prior['wall_seconds'],cpu_seconds=prior['cpu_seconds'],
            unsuccessful_import_wall_seconds=prior.get('unsuccessful_import_attempt_wall_seconds'),
            result_path=str(C.OUT/'preflight/optimization.json'),result_sha256=C.sha(C.OUT/'preflight/optimization.json')))
    C.dump(out,report)
    try:
        for case in plan['cases']:
            wt,ct=time.perf_counter(),time.process_time()
            c,z,op,priorB,net,cal,force,field=E.context(case,301030)
            frozen_net=C.state(net)
            inputs[C.probability_record(301030)['path']]=C.probability_record(301030)['sha256']
            def validation(label):
                result={}
                for degree in (20,28):
                    qw,qc=time.perf_counter(),time.process_time()
                    with torch.no_grad():
                        value=E.score(c,z,field,net,cal,force,'selection',degree)
                    result[str(degree)]=dict(score=value,wall_seconds=time.perf_counter()-qw,cpu_seconds=time.process_time()-qc)
                    print(case,label,degree,value['total'],flush=True)
                return result
            def physical_norms():
                with torch.no_grad():
                    a=field.states();q=field.normalized_coefficients(a)
                return dict(raw_a_norm=float(a.norm()),raw_a_max=float(a.abs().max()),
                            physical_q_norm=float(q.norm()),physical_q_max=float(q.abs().max()))
            row=dict(case=case,seed=301030,steps=[],initial_selection=validation('initial-selection'),
                initial_physical_norms=physical_norms(),preconditioner=c['pinn_output_preconditioning'])
            report['rows'].append(row)
            opt=torch.optim.Adam(field.parameters(),lr=.001)
            for step in range(1,31):
                qw,qc=time.perf_counter(),time.process_time()
                measured=E.field_step(c,z,op,priorB,field,opt,net,cal,force,degree=12)
                finite=all(bool(torch.isfinite(p).all()) and (p.grad is None or bool(torch.isfinite(p.grad).all())) for p in field.parameters())
                if not finite:
                    raise FloatingPointError('Post-update parameter/gradient is nonfinite')
                measured.update(step=step,objective_state=step-1,physical_norms_after_update=physical_norms(),
                    parameters_and_gradients_finite=finite,wall_seconds=time.perf_counter()-qw,
                    cpu_seconds=time.process_time()-qc,memory=memory())
                row['steps'].append(measured)
                if step%5==0:
                    print(case,'step',step,'state',step-1,'J',measured['objective'],'PDE',measured['pde_loss'],
                          'grad',measured['gradient_norm'],flush=True)
                    C.dump(out,report)
            qw,qc=time.perf_counter(),time.process_time()
            with torch.no_grad():
                fit=E.score(c,z,field,net,cal,force,'fit',12)
                aa,ad=field.coefficients_and_dt(torch.arange(201,dtype=field.cold.dtype)*.0005)
                nfit=float((z['end']-z['start'])[C.group_mask(z['pid'],'fit')].sum())
                pde,details=F.residual_loss(aa,ad,op,priorB,nfit)
                reg,parts=field.regularization()
            row['final_objective']=dict(state=30,inversion=fit,objective=.08*fit['total']+float(pde+reg),
                pde_loss=float(pde),regularization=float(reg),regularization_parts={k:float(v) for k,v in parts.items()},
                **{k:float(v) for k,v in details.items()},wall_seconds=time.perf_counter()-qw,cpu_seconds=time.process_time()-qc)
            row['final_selection']=validation('after-thirty-selection')
            row['probability_unchanged']=all(torch.equal(v,frozen_net[k]) for k,v in net.state_dict().items())
            row['probability_gradients_absent']=all(not p.requires_grad and p.grad is None for p in net.parameters())
            assert row['probability_unchanged'] and row['probability_gradients_absent']
            row['stability']={}
            for key in ('objective','pde_loss'):
                values=[r[key] for r in row['steps']]+[row['final_objective'][key]]
                early=statistics.median(values[11:21]);late=statistics.median(values[21:31]);initial=values[0]
                tests=dict(tail_median_bounded=late<=10*max(early,1e-12),
                           tail_max_bounded=max(values[21:31])<=100*max(initial,1e-12),
                           final_bounded=values[30]<=100*max(initial,1e-12))
                row['stability'][key]=dict(values_by_state=values,median_states11_to20=early,
                    median_states21_to30=late,tail_median_ratio=late/max(early,1e-12),
                    tail_max_ratio=max(values[21:31])/max(initial,1e-12),
                    final_ratio=values[30]/max(initial,1e-12),checks=tests,passed=all(tests.values()))
            row.update(implementation_passed=True,stability_passed=all(x['passed'] for x in row['stability'].values()),
                wall_seconds=time.perf_counter()-wt,cpu_seconds=time.process_time()-ct,memory=memory(),
                final_physical_norms=physical_norms())
            row['passed']=row['implementation_passed'] and row['stability_passed']
            C.dump(out,report)
            del c,z,op,priorB,net,field,opt
            gc.collect()
        source_equal=all(C.sha(p)==h for p,h in sources.items())
        input_equal=all(C.sha(p)==h for p,h in inputs.items())
        report.update(complete=True,passed=all(r['passed'] for r in report['rows']) and source_equal and input_equal,
            implementation_passed=all(r['implementation_passed'] for r in report['rows']),
            stability_passed=all(r['stability_passed'] for r in report['rows']),
            sources_unchanged=source_equal,inputs_unchanged=input_equal,finished_utc=C.utc())
    except Exception as error:
        report.update(complete=False,passed=False,failed=True,error=repr(error),failed_utc=C.utc())
        raise
    finally:
        report.update(wall_seconds=time.perf_counter()-start_wall,cpu_seconds=time.process_time()-start_cpu,memory=memory())
        report['known_optimization_preflight_cost']=dict(
            wall_seconds=report['wall_seconds']+prior['wall_seconds']+prior.get('unsuccessful_import_attempt_wall_seconds',0.),
            cpu_seconds=report['cpu_seconds']+prior['cpu_seconds'],
            note='Both optimization preflights plus recorded failed import wall time; separate geometry/history/manufactured audits not included; failed-import CPU unavailable')
        C.dump(out,report)
    print('COMPLETE',report['passed'],report['wall_seconds'],report['memory'],flush=True)


def camera_optimization_preflight(out):
    """Exactly two disposable full-fit updates per case at fixed .001; no H."""
    import ctypes
    import gc
    import cf_common as C
    import cf_experiment as E
    C.setup()
    class PMC(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
            ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
            ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
            ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
            ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
    def memory():
        p = PMC(); p.cb = ctypes.sizeof(p)
        ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(p), p.cb)
        return dict(peak_working_set_bytes=int(p.PeakWorkingSetSize), working_set_bytes=int(p.WorkingSetSize),
                    peak_pagefile_bytes=int(p.PeakPagefileUsage)) if ok else None
    wall, cpu, rows = time.perf_counter(), time.process_time(), []
    for case in C.CASES:
        wt, ct = time.perf_counter(), time.process_time()
        c, z, op, prior, net, cal, force, field = E.context(case, 301030)
        before = C.state(net)
        def selection_scores(label):
            values = {}
            for degree in (12, 20, 28):
                qw, qc = time.perf_counter(), time.process_time()
                with torch.no_grad():
                    value = E.score(c, z, field, net, cal, force, 'selection', degree)
                values[str(degree)] = dict(score=value, wall_seconds=time.perf_counter()-qw,
                                           cpu_seconds=time.process_time()-qc)
                print(case, label, degree, value['total'], flush=True)
            return values
        initial = selection_scores('initial-selection')
        steps = []
        opt = torch.optim.Adam(field.parameters(), lr=.001)
        for k in (1, 2):
            qw, qc = time.perf_counter(), time.process_time()
            row = E.field_step(c, z, op, prior, field, opt, net, cal, force, degree=12)
            row.update(step=k, wall_seconds=time.perf_counter()-qw, cpu_seconds=time.process_time()-qc, memory=memory())
            steps.append(row)
            print(case, 'fit-step', k, row['objective'], row['gradient_norm'], row['wall_seconds'], flush=True)
        after = selection_scores('after-two-selection')
        equal = all(torch.equal(t, before[n]) for n, t in net.state_dict().items())
        no_gradient = all(not p.requires_grad and p.grad is None for p in net.parameters())
        assert equal and no_gradient
        rows.append(dict(case=case, seed=301030, learning_rate=.001, full_fit=True, steps=steps,
            initial_selection=initial, after_two_steps_selection=after,
            probability_unchanged=equal, probability_gradients_absent=no_gradient,
            counts={group:{name:dict(intervals=int((P.group_mask(z['pid'],group)&(z['state']==kind)).sum()),
                pids=int(z['pid'][P.group_mask(z['pid'],group)&(z['state']==kind)].unique().numel()))
                for kind,name in ((0,'free'),(1,'cap'))} for group in ('fit','selection')},
            cpu_seconds=time.process_time()-ct, wall_seconds=time.perf_counter()-wt, memory=memory()))
        C.dump(out, dict(complete=False, rows=rows, no_test_score=True, no_warmstart_weights_saved=True))
        del c, z, op, prior, net, field, opt
        gc.collect()
    growth = {r['case']:r['steps'][1]['objective']/max(abs(r['steps'][0]['objective']), 1e-30) for r in rows}
    unstable = any(value > 1e3 for value in growth.values())
    report = dict(complete=True, passed=not unstable, implementation_passed=True,
        stability_passed=not unstable, objective_growth_second_over_first=growth,
        stability_note='Order-of-magnitude blow-up is a stability warning, not a field-quality or hyperparameter-selection score',
        acceptance='finite objective/gradient and frozen probabilities; stability reported separately',
        fixed_steps_per_case=2, fixed_lr=.001, no_hyperparameter_selection=True,
        no_test_score=True, no_warmstart_weights_saved=True, rows=rows,
        cpu_seconds=time.process_time()-cpu, wall_seconds=time.perf_counter()-wall, memory=memory())
    C.dump(out, report)
    print('COMPLETE', report['wall_seconds'], report['memory'], flush=True)


def one_ms_z(z, frame):
    B = len(z['target'])
    return dict(pid=z['pid'], state=torch.ones(B, dtype=torch.long), segment=z['pid'],
        start=torch.full((B,), float(frame), dtype=torch.float64),
        end=torch.full((B,), float(frame+1), dtype=torch.float64),
        time=torch.full((B, 1), frame+.5, dtype=torch.float64),
        dt=torch.full((B, 1), .001, dtype=torch.float64),
        w=z['w'], h=z['h'], target=z['target'])


def synthetic_pack(z, duration_ms):
    B = len(z['pid'])
    cuts = torch.cat((torch.arange(int(duration_ms)+1, dtype=torch.float64), torch.tensor([duration_ms], dtype=torch.float64)))
    cuts = cuts.unique(sorted=True)
    mids = (cuts[:-1]+cuts[1:])/2
    zz = one_ms_z(z, 0)
    zz.update(end=torch.full((B,), duration_ms, dtype=torch.float64),
              time=mids[None].expand(B, -1), dt=torch.diff(cuts)[None].expand(B, -1)*.001,
              h=z['h'][:, None].expand(-1, len(mids), -1, -1))
    return zz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    parser.add_argument('--cf-two-step', action='store_true')
    parser.add_argument('--cf-preconditioned-thirty', action='store_true')
    args = parser.parse_args()
    if args.cf_preconditioned_thirty:
        if args.out is None:
            raise ValueError('A separate planned preflight output path is required')
        return preconditioned_thirty_step_preflight(args.out)
    if args.cf_two_step:
        if args.out is None:
            raise ValueError('A separate preflight output path is required')
        return camera_optimization_preflight(args.out)
    torch.set_num_threads(2)
    start, cpu = time.perf_counter(), time.process_time()
    net, cal, force, record = P.load_probability(301030)
    source = P.CALIBRATION/'prepared/vn0_0065m_s/observations.pt'
    raw = torch.load(source, weights_only=False, map_location='cpu')
    allowed = torch.where((raw['z']['pid'] % 5 == 1) & (raw['z']['state'] == 1))[0][:2]
    z = P.subset(raw['z'], allowed)
    c = raw['c']
    del raw
    torch.manual_seed(98267)
    B = len(z['pid'])
    velocity = torch.randn(B, 2, 2, dtype=torch.float64)*.00015
    velocity[..., 1] += velocity.new_tensor([c['physics']['vn_mean_m_s'], c['physics']['vs_mean_m_s']])
    gradient = torch.randn(B, 2, 2, 2, dtype=torch.float64)*.01
    acceleration = torch.randn(B, 2, 2, dtype=torch.float64)*.002
    jets = tuple(x[:, None] for x in (velocity, gradient, acceleration))
    newprev, oldprev, oldctx = None, None, None
    errors = []
    for frame in (0, 1):
        zz = one_ms_z(z, frame)
        ours = P.captured_distribution(c, zz, jets, net, cal, force, 12, newprev,
                                      torch.zeros(B, dtype=torch.bool), True)
        parent = P.LM.step(c, z, velocity, gradient, acceleration, net.base, net.head,
                          cal, force, 12, oldprev, oldctx)
        errors.extend(float((x-y).abs().max()) for x, y in zip(ours[:3], parent[:3]))
        for key in ('qlogp', 'emean', 'ecov', 'log_evidence'):
            errors.append(float((ours[3][key]-parent[3][key]).abs().max()))
        newprev, oldprev, oldctx = ours[3], parent[3], parent[3]['context']
    assert max(errors) < 2e-9, max(errors)
    free_z = one_ms_z(z, 0)
    free_z['state'] = torch.zeros_like(free_z['state'])
    newfree = P.free_distribution(c, free_z, jets, net, cal, force)
    oldfree = P.TM.distribution(c, z, velocity, gradient, acceleration, 'free', net.base, cal, force, 12)
    free_error = max(float((x-y).abs().max()) for x, y in zip(newfree, oldfree))
    assert free_error < 2e-10
    long = synthetic_pack(z, 1000/90)
    M = long['time'].shape[1]
    # Independent innovation-to-endpoint response audit. Each old velocity
    # innovation affects every later displacement piece through its decay.
    free_long = dict(long, state=torch.zeros_like(long['state']))
    free_jets = tuple(x[:, None].expand(-1, M, *x.shape[1:]) for x in (velocity, gradient, acceleration))
    free_pred, free_cov, _ = P.free_distribution(c, free_long, free_jets, net, cal, force)
    sensitivities = []
    for j in range(M):
        A, F, gain = P._propagator(c, 'free', free_long['dt'][:, j])
        F, gain = F[:, 0], gain[:, 0]
        sensitivities = [F@old for old in sensitivities]
        ctx = P.core.context(c, {'w': z['w']}, velocity, gradient)
        _, force_L = net.base.free_parameters(P.history(free_long, j), ctx, force)
        sensitivities.append(gain@A[0]@force_L)
    reference_cov = sum(G[:, 2:]@G[:, 2:].transpose(-1, -2) for G in sensitivities)
    reference_cov += cal['sigma']['free']**2*torch.eye(2, dtype=torch.float64)
    long_cov_error = float((reference_cov-free_cov[:, 0, 0]).abs().max())
    assert long_cov_error < 2e-10
    scale = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    dv = torch.randn_like(velocity)*.0002
    def makejet(a):
        return ((velocity+a*dv)[:, None].expand(-1, M, -1, -1),
                gradient[:, None].expand(-1, M, -1, -1, -1),
                (acceleration+a*.003)[:, None].expand(-1, M, -1, -1))
    pp, cc, ll, ss = P.captured_distribution(c, long, makejet(scale), net, cal, force, 12)
    loss = -ss['log_evidence'].sum()
    analytical = float(torch.autograd.grad(loss, scale)[0])
    step = 1e-5
    numerical = []
    for sign in (-1, 1):
        r = P.captured_distribution(c, long, makejet(scale.detach()+sign*step), net, cal, force, 12)
        numerical.append(float(-r[3]['log_evidence'].sum()))
    fd = (numerical[1]-numerical[0])/(2*step)
    relative = abs(analytical-fd)/max(abs(fd), abs(analytical), 1e-8)
    assert relative < 3e-5, (analytical, fd, relative)
    assert bool(torch.linalg.eigvalsh(cc).min() > 0)
    changed = dict(long, target=long['target']+.05)
    a = P.captured_distribution(c, changed, makejet(scale.detach()), net, cal, force, 12)
    pred_change = max(float((x-y).detach().abs().max()) for x, y in zip(a[:3], (pp, cc, ll)))
    assert pred_change < 1e-12
    posterior_change = float((a[3]['emean']-ss['emean']).detach().abs().max())
    assert posterior_change > 0
    reset = P.captured_distribution(c, long, makejet(scale.detach()), net, cal, force, 12,
                                   previous=a[3], reset=torch.ones(B, dtype=torch.bool))
    reset_error = float((reset[0]-pp).detach().abs().max())
    assert reset_error < 1e-10
    assert all(p.grad is None and not p.requires_grad for p in net.parameters())
    report = dict(complete=True, passed=True, probability_record=record,
        one_ms_two_consecutive_steps_max_abs_error=max(errors), free_one_ms_max_abs_error=free_error,
        long_free_cross_piece_covariance_reference_error=long_cov_error,
        native_fraction_duration_ms=1000/90, internal_pieces=M,
        minimum_endpoint_covariance_eigenvalue=float(torch.linalg.eigvalsh(cc).detach().min()),
        context_gradient=analytical, context_gradient_finite_difference=fd,
        context_gradient_relative_error=relative, changed_endpoint_prediction_error=pred_change,
        changed_endpoint_posterior_difference=posterior_change, segment_reset_error=reset_error,
        network_gradient_absent=True, target_test_scored=False,
        source_scope='two non-target 6.5 mm/s validation captured rows plus synthetic durations/fields',
        wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu,
        migration_contract=P.CONTRACT)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
