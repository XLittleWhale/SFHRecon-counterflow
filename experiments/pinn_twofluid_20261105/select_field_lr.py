"""Prospectively registered, non-target-only numerical step-size selection.

Exactly three learning rates per field representation and 30 updates each.
The last ten pre-update full-objective values define selection. The resulting
weights are discarded: only the per-arm numerical learning rate is retained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from pinn_common import (HERE, OUT, T, N, make_context, setup, plain_state,
                         probability_record, read, dump, sha, fingerprint)
from experiment import field_step, make_field
from preflight_optimization import memory_bytes

ARMS = ('global_spline', 'pinn')
LRS = (.0001, .0003, .001)
STEPS = 30
SEED = 301030
PLAN = OUT/'preflight/lr_plan.json'


def function_sha(function):
    return hashlib.sha256(inspect.getsource(function).encode('utf8')).hexdigest()


def write_plan():
    if PLAN.exists():
        raise RuntimeError('LR plan already exists; refusing to overwrite or start a duplicate selection')
    N.verify_protocol()
    model = probability_record(SEED)
    if sha(model['path']) != model['sha256']:
        raise RuntimeError('Frozen selected probability checkpoint changed')
    bound = ('select_field_lr.py', 'pinn_common.py', 'field_model.py', 'global_spline.py',
             'continuous_query.py', 'preflight_optimization.py')
    plan = dict(frozen=True, created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='numerical learning-rate selection only; all diagnostic weights discarded',
        case='vn0_0065m_s', U=.0065, seed=SEED, group='V', full_intervals=95,
        arms=list(ARMS), candidates=list(LRS), steps=STEPS,
        optimizer='Adam default betas/eps, no weight decay', gradient_clip=5.,
        GH=12, pde_lambda=1., pde_nodes=191,
        field_parameterization='as bound source; equal cold state and scale_power=1',
        objective='.08*balanced_probability_NLL + B_whitened_Simpson_PDE + uniform_spatial_regularization',
        criterion='minimum mean of last 10 pre-update objectives (updates 21..30, states 20..29)',
        finite_requirement='all 30 pre-update objectives/components and all post-update field parameters finite',
        tie_absolute_tolerance=1e-9, tie_rule='choose smaller learning rate',
        max_parallel_workers=2, per_worker_torch_threads=2,
        no_candidate_or_budget_extension=True, target_reference_read=False,
        target_H_scored=False, field_step_source_sha256=function_sha(field_step),
        make_field_source_sha256=function_sha(make_field),
        experiment_file_sha256_at_plan=sha(HERE/'experiment.py'),
        sources={str(HERE/name): sha(HERE/name) for name in bound},
        inputs={str(T.SOURCE/'prepared/vn0_0065m_s/observations.pt'):
                    sha(T.SOURCE/'prepared/vn0_0065m_s/observations.pt'),
                str(OUT/'probability_selection/selection_frozen.json'):
                    sha(OUT/'probability_selection/selection_frozen.json'),
                model['path']: model['sha256']},
        probability=dict(family=model['family'], seed=model['training_seed'],
                         path=model['path'], sha256=model['sha256']))
    plan['fingerprint'] = fingerprint(plan)
    dump(PLAN, plan)
    return plan


def verify_plan():
    plan = read(PLAN)
    if not plan['frozen'] or fingerprint({k: v for k, v in plan.items() if k != 'fingerprint'}) != plan['fingerprint']:
        raise RuntimeError('LR plan fingerprint mismatch')
    for category in ('sources', 'inputs'):
        for path, digest in plan[category].items():
            if sha(path) != digest:
                raise RuntimeError('Bound LR '+category+' changed: '+path)
    if function_sha(field_step) != plan['field_step_source_sha256'] or function_sha(make_field) != plan['make_field_source_sha256']:
        raise RuntimeError('Bound field optimizer implementation changed')
    return plan


def numeric_finite(value):
    if isinstance(value, dict):
        return all(numeric_finite(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return all(numeric_finite(x) for x in value)
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return True


def candidate_summary(trace):
    tail = trace[-10:]
    objectives = [row['objective'] for row in tail]
    first = trace[0]['objective']
    differences = [b-a for a, b in zip(objectives[:-1], objectives[1:])]
    signs = [1 if x > 0 else -1 for x in differences if abs(x) > 1e-12]
    return dict(epoch0=dict(objective=first, pde_loss=trace[0]['pde_loss'],
                            balanced_nll=trace[0]['inversion']['total'],
                            free_nll=trace[0]['inversion']['free'], cap_nll=trace[0]['inversion']['cap']),
        tail_mean_objective=statistics.mean(objectives),
        tail_mean_pde_loss=statistics.mean(row['pde_loss'] for row in tail),
        tail_mean_regularization=statistics.mean(row['symmetric_regularization'] for row in tail),
        tail_mean_balanced_nll=statistics.mean(row['inversion']['total'] for row in tail),
        tail_mean_free_nll=statistics.mean(row['inversion']['free'] for row in tail),
        tail_mean_cap_nll=statistics.mean(row['inversion']['cap'] for row in tail),
        tail_objective_std=statistics.pstdev(objectives), tail_objective_min=min(objectives),
        tail_objective_max=max(objectives), tail_net_objective_change=objectives[-1]-objectives[0],
        tail_adjacent_change_sign_reversals=sum(a != b for a, b in zip(signs[:-1], signs[1:])),
        tail_mean_above_epoch0=statistics.mean(objectives) > first,
        last_pre_update_objective_above_epoch0=objectives[-1] > first,
        tail_contains_objective_increases=any(x > 0 for x in differences),
        final_pre_update_state_index=29,
        interpretation='Finite numerical diagnostic only; short-run objective ranking is not field accuracy or convergence')


def run_arm(arm):
    setup()
    plan = verify_plan()
    d, cfg, old, net, cal, force = make_context(.0065, SEED)
    if not bool((d['z']['pid'] % 5 == 1).all()) or d['intervals'] != 95:
        raise AssertionError('The sole authorized observation view is non-target V over 95 intervals')
    frozen_probability = plain_state(net)
    reports = []
    for index, lr in enumerate(LRS):
        verify_plan()
        dest = OUT/f'preflight/lr_{arm}_{index}.json'
        if dest.exists():
            raise RuntimeError('Candidate report exists; no duplicate diagnostic updates allowed')
        field = make_field(arm, old, SEED)
        optimizer = torch.optim.Adam(field.parameters(), lr=lr)
        trace = []
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        failure = None
        try:
            for step in range(1, STEPS+1):
                tick, cpu_tick = time.perf_counter(), time.process_time()
                row = field_step(field, optimizer, old, d, net, cal, force, group='V')
                if not numeric_finite(row) or any(not bool(torch.isfinite(p).all()) for p in field.parameters()):
                    raise FloatingPointError('Nonfinite candidate loss/gradient/updated parameter')
                if any(p.requires_grad or p.grad is not None for p in net.parameters()):
                    raise AssertionError('Probability parameter is trainable or acquired gradient')
                if any(not torch.equal(value, frozen_probability[key]) for key, value in net.state_dict().items()):
                    raise AssertionError('Frozen probability state changed')
                row.update(update=step, pre_update_state_index=step-1,
                           wall_seconds=time.perf_counter()-tick, cpu_seconds=time.process_time()-cpu_tick)
                trace.append(row)
                if step % 10 == 0:
                    print(arm, 'lr', lr, 'update', step, 'pre_objective', row['objective'], flush=True)
        except (FloatingPointError, RuntimeError) as exc:
            failure = f'{type(exc).__name__}: {exc}'
        verify_plan()
        report = dict(arm=arm, learning_rate=lr, complete=len(trace) == STEPS,
            finite=failure is None and len(trace) == STEPS, failure=failure,
            trace=trace, steps=len(trace), plan_fingerprint=plan['fingerprint'],
            probability_sha256=plan['probability']['sha256'], probability_bitwise_unchanged=True,
            weights_discarded=True, target_reference_read=False, target_H_scored=False,
            wall_seconds=time.perf_counter()-start_wall, cpu_seconds=time.process_time()-start_cpu,
            memory=memory_bytes())
        if report['finite']:
            report['summary'] = candidate_summary(trace)
        dump(dest, report)
        reports.append(dict(path=str(dest), sha256=sha(dest)))
        del field, optimizer
    dump(OUT/f'preflight/lr_{arm}_complete.json', dict(complete=True,
        plan_fingerprint=plan['fingerprint'], arm=arm, reports=reports))


def aggregate(wall_seconds):
    plan = verify_plan()
    selected, records = {}, []
    for arm in ARMS:
        arm_rows = []
        for index, lr in enumerate(LRS):
            path = OUT/f'preflight/lr_{arm}_{index}.json'
            row = read(path)
            if row['plan_fingerprint'] != plan['fingerprint'] or row['learning_rate'] != lr or row['arm'] != arm:
                raise AssertionError('Candidate identity mismatch')
            arm_rows.append(row)
            records.append(dict(arm=arm, learning_rate=lr, path=str(path), sha256=sha(path),
                finite=row['finite'], wall_seconds=row['wall_seconds'], cpu_seconds=row['cpu_seconds'],
                summary=row.get('summary'), failure=row['failure']))
        eligible = [row for row in arm_rows if row['finite']]
        if not eligible:
            raise RuntimeError('No finite learning-rate candidate for '+arm)
        best = eligible[0]
        for row in eligible[1:]:
            if row['summary']['tail_mean_objective'] < best['summary']['tail_mean_objective']-1e-9:
                best = row
        selected[arm] = best['learning_rate']
    gate = dict(complete=True, field_lr_by_arm=selected,
        plan_path=str(PLAN), plan_sha256=sha(PLAN), plan_fingerprint=plan['fingerprint'],
        candidates=records, probability_sha256=plan['probability']['sha256'],
        sources=plan['sources'], field_step_source_sha256=plan['field_step_source_sha256'],
        target_reference_read=False, target_H_scored=False, all_weights_discarded=True,
        total_cpu_seconds=sum(row['cpu_seconds'] for row in records),
        sum_candidate_worker_wall_seconds=sum(row['wall_seconds'] for row in records),
        total_wall_seconds=wall_seconds, completed_utc=datetime.now(timezone.utc).isoformat())
    gate['fingerprint'] = fingerprint(gate)
    dump(OUT/'preflight/lr_selection.json', gate)
    print('FIELD_LR_SELECTED', selected, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.arm:
        run_arm(args.arm)
        return
    write_plan()  # MUST precede both workers and every candidate computation.
    started = time.perf_counter()
    workers = []
    for arm in ARMS:
        logpath = OUT/f'preflight/lr_{arm}.log'
        stream = logpath.open('x', encoding='utf8')
        command = [sys.executable, str(Path(__file__).resolve()), '--arm', arm]
        workers.append((arm, subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT), stream))
    failed = []
    for arm, process, stream in workers:
        code = process.wait()
        stream.close()
        if code:
            failed.append((arm, code))
    if failed:
        raise RuntimeError('LR worker failure: '+str(failed))
    aggregate(time.perf_counter()-started)


if __name__ == '__main__':
    main()
