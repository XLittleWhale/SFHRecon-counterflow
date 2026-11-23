"""Evaluate globally frozen CF PINN snapshots; never select or fit a field.

CF has no field reference.  All field quantities below describe the candidate,
its difference from the fixed cold background, or equation/boundary consistency.
Only this post-gate entry point evaluates held-out native camera endpoints.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
import numpy as np
import torch

from cf_common import (OUT, CASES, SEEDS, BUDGETS, evaluation_gate, read, dump,
                       sha, load, setup, utc, fingerprint, group_mask, state)
import cf_field as F
from cf_experiment import context, score

REPRESENTATIVE_SEED = 301030
DEGREES = (20, 28)


def plain(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return value.tolist() if value.ndim else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def scenarios(run):
    yield 0, 'background', run['background']
    for budget in BUDGETS:
        for mode in ('primary', 'terminal'):
            yield budget, mode, run['budgets'][str(budget)][mode]


def row_id(case, seed, budget, mode):
    return f'{case}_seed{seed}_budget{budget}_{mode}'


def rhs_chunks(op, a, chunk=32):
    return torch.cat([F.batched_rhs(op, part) for part in a.split(chunk)])


def spatial_weights(op):
    # Same fixed 34-point x / 52-point periodic-y rule as the frozen operator.
    weights = np.tile(np.polynomial.legendre.leggauss(34)[1]/(2*52), 52)
    if len(weights) != op.v0.shape[1]:
        raise ValueError('Unexpected CF operator quadrature')
    return op.V.new_tensor(weights)


def raw_velocity_rms(op, a):
    """Area-weighted vector RMS for raw coefficient perturbations (no mean)."""
    w = spatial_weights(op)
    values = []
    for part in a.split(32):
        v = torch.einsum('fpm,tfm->tfp', op.V, part).reshape(len(part), *op.v0.shape)
        values.append(torch.einsum('tfp,p->tf', v.square().sum(-1), w))
    return torch.cat(values).clamp_min(0).sqrt()


@torch.no_grad()
def pde_audit(field, op, prior, exposure_ms):
    # All .25ms points are diagnostic only; optimization used the half-ms grid.
    t = torch.arange(401, dtype=field.cold.dtype, device=field.device)*.00025
    a, adot = field.coefficients_and_dt(t)
    residual = .001*(adot-rhs_chunks(op, a))
    paths = [residual[offset:400:4] for offset in range(4)] + [residual[4:401:4]]
    reports = []
    for fraction, r in zip((0., .25, .5, .75, 1.), paths):
        qn, qs = prior.quadratic(r[:, 0], r[:, 1])
        vrms = raw_velocity_rms(op, r)
        reports.append(dict(fraction=fraction, points=100,
            normal_B_energy=float(qn), super_B_energy=float(qs),
            dt_coefficient_rms=r.square().mean((0, 2)).sqrt().tolist(),
            dt_velocity_rms_mm_s=(vrms.square().mean(0).sqrt()*1000).tolist(),
            instantaneous_acceleration_rms_m_s2=(vrms.square().mean(0).sqrt()/.001).tolist(),
            maximum_absolute_dt_coefficient=float(r.abs().max())))

    def combine(indices, weights):
        out = dict(path_points=100*len(indices), weights=list(weights))
        for key in ('normal_B_energy', 'super_B_energy'):
            out[key] = sum(w*reports[i][key] for i, w in zip(indices, weights))
        out['pde_loss'] = .08/(2*exposure_ms)*(out['normal_B_energy']+out['super_B_energy'])
        out['dt_velocity_rms_mm_s'] = [math.sqrt(sum(w*reports[i]['dt_velocity_rms_mm_s'][f]**2
            for i, w in zip(indices, weights))) for f in range(2)]
        return out

    trained = combine((0, 2, 4), (1/6, 4/6, 1/6))
    quarter = combine((1, 3), (.5, .5))
    check, detail = F.residual_loss(a[::2], adot[::2], op, prior, exposure_ms)
    if not math.isclose(float(check), trained['pde_loss'], rel_tol=1e-10, abs_tol=1e-12):
        raise RuntimeError('Independent aggregation differs from trained PDE loss')
    return dict(paths=reports, trained_simpson=trained, independent_quarter=quarter,
        independent_to_trained_energy_ratio={key: quarter[key]/trained[key] if trained[key] else None
            for key in ('normal_B_energy', 'super_B_energy')},
        trained_unique_points=201, independent_points=200,
        normalization='sum_fit_native_durations_ms', exposure_ms=exposure_ms,
        physics=dict(nu_s=0., Fns_prefactor=1., fixed_L0=True),
        new_selection_or_optimization=False,
        interpretation='Equation consistency only; no field truth and no estimate of field accuracy')


def geometry(c, like):
    return F.runtime().core.channel_geometry(c['physics']['vn_mean_m_s'],
        c['physics']['vs_mean_m_s'], str(like.device), like.dtype)


def values_at(c, a, points, with_gradient=False):
    b = geometry(c, a).evaluate(points.to(a))
    v = b['v0'][None]+torch.einsum('tfm,pfmj->tpfj', a, b['V'])
    if with_gradient:
        g = b['g0'][None]+torch.einsum('tfm,pfmij->tpfij', a, b['G'])
        return v, g
    return v


@torch.no_grad()
def field_audit(c, field, op):
    a = field.states()
    w = spatial_weights(op)
    mean, second = [], []
    for part in a.split(24):
        v = op.v0+torch.einsum('fpm,tfm->tfp', op.V, part).reshape(len(part), *op.v0.shape)
        mean.append(torch.einsum('tfpj,p->tfj', v, w))
        second.append(torch.einsum('tfp,p->tf', v.square().sum(-1), w))
    departure = raw_velocity_rms(op, a-field.cold)
    mean, second = torch.cat(mean), torch.cat(second)
    reg, regparts = field.regularization(a)
    y = torch.linspace(0., .008, 65, dtype=a.dtype, device=a.device)
    walls = torch.cat([torch.stack((torch.full_like(y, x), y), -1) for x in (0., .016)])
    wall_v, wall_g = values_at(c, a, walls, True)
    x = torch.linspace(0., .016, 65, dtype=a.dtype, device=a.device)
    ends = torch.cat([torch.stack((x, torch.full_like(x, y0)), -1) for y0 in (0., .008)])
    end_v, end_g = values_at(c, a, ends, True)
    # Independent spatial grid includes all boundaries and mid-view interior.
    yy, xx = torch.meshgrid(torch.linspace(0., .008, 17, dtype=a.dtype, device=a.device),
                           torch.linspace(0., .016, 33, dtype=a.dtype, device=a.device), indexing='ij')
    _, grid_g = values_at(c, a, torch.stack((xx.flatten(), yy.flatten()), -1), True)
    divergence = grid_g[..., 0, 0]+grid_g[..., 1, 1]
    return dict(times_ms=list(range(101)),
        departure_from_cold_time_rms_mm_s=(departure*1000).tolist(),
        departure_from_cold_initial_rms_mm_s=(departure[0]*1000).tolist(),
        departure_from_cold_first5ms_rms_mm_s=(departure[:6].square().mean(0).sqrt()*1000).tolist(),
        departure_from_cold_full_rms_mm_s=(departure.square().mean(0).sqrt()*1000).tolist(),
        candidate_full_speed_rms_mm_s=(second.mean(0).sqrt()*1000).tolist(),
        candidate_spatial_mean_velocity_mm_s=(mean*1000).tolist(),
        regularization=float(reg), regularization_parts=plain(regparts),
        boundary=dict(normal_no_slip_max_mm_s=float(wall_v[:, :, 0].abs().max())*1000,
            super_no_penetration_max_mm_s=float(wall_v[:, :, 1, 0].abs().max())*1000,
            axial_periodic_velocity_max_mm_s=float((end_v[:, :65]-end_v[:, 65:]).abs().max())*1000,
            axial_periodic_gradient_max_s_inv=float((end_g[:, :65]-end_g[:, 65:]).abs().max()),
            divergence_max_s_inv=float(divergence.abs().max()),
            super_tangential_neumann_not_imposed=True),
        field_truth_available=False,
        interpretation='RMS is velocity magnitude or departure from the cold background, never reconstruction error')


@torch.no_grad()
def export_field(c, field, path, metadata):
    """Cell-centred visualization grid; domain edges stored separately."""
    nx, ny = 64, 32
    x = (torch.arange(nx, dtype=field.cold.dtype, device=field.device)+.5)*.016/nx
    y = (torch.arange(ny, dtype=field.cold.dtype, device=field.device)+.5)*.008/ny
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    points = torch.stack((xx.flatten(), yy.flatten()), -1)
    t = torch.arange(101, dtype=field.cold.dtype, device=field.device)*.001
    a, adot = field.coefficients_and_dt(t)
    vv, ww, dd = [], [], []
    for part in a.split(8):
        v, g = values_at(c, part, points, True)
        vv.append(v.permute(0, 2, 1, 3).reshape(-1, 2, ny, nx, 2))
        ww.append((g[..., 1, 0]-g[..., 0, 1]).permute(0, 2, 1).reshape(-1, 2, ny, nx))
        dd.append((g[..., 0, 0]+g[..., 1, 1]).permute(0, 2, 1).reshape(-1, 2, ny, nx))
    velocity, vorticity, divergence = torch.cat(vv), torch.cat(ww), torch.cat(dd)
    bg = values_at(c, field.cold[None], points)[0].permute(1, 0, 2).reshape(2, ny, nx, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, time_s=t.cpu().numpy(), physical_time_s=(t+c['start_s']).cpu().numpy(),
        x_m=x.cpu().numpy(), y_m=y.cpu().numpy(), x_edges_m=np.linspace(0., .016, nx+1),
        y_edges_m=np.linspace(0., .008, ny+1), velocity_m_s=velocity.cpu().numpy(),
        vorticity_s_inv=vorticity.cpu().numpy(), divergence_s_inv=divergence.cpu().numpy(),
        cold_velocity_m_s=bg.cpu().numpy(), departure_from_cold_m_s=(velocity-bg[None]).cpu().numpy(),
        coefficients=a.cpu().numpy(), coefficient_time_derivative_s_inv=adot.cpu().numpy())
    return dict(**metadata, kind='field_npz', path=str(path.resolve()), sha256=sha(path),
                frames=101, domain_m=[.016, .008], grid_cells=[nx, ny],
                has_reference_or_field_error=False, grid='cell centres with explicit physical edges')


def probability_difference(primary, refined):
    return dict(degrees=list(DEGREES), nll_per_ms={k: refined[k]-primary[k] for k in ('total', 'free', 'cap')},
        native_nll={k: refined['native_nll'][k]-primary['native_nll'][k] for k in ('total', 'free', 'cap')},
        coverage={k: refined['coverage'][k]-primary['coverage'][k] for k in ('free', 'cap')},
        spread_um={k: refined['spread'][k]-primary['spread'][k] for k in ('free', 'cap')},
        rmse_um={k: refined['rmse_um'][k]-primary['rmse_um'][k] for k in ('free', 'cap')},
        interpretation='Descriptive GH sensitivity; no post-hoc model selection or convergence guarantee')


def metric_values(row):
    h = row['heldout']
    values = {'H_total_nll_per_ms': h['total']}
    for kind in ('free', 'cap'):
        values.update({f'H_{kind}_nll_per_ms': h[kind],
            f'H_{kind}_native_nll': h['native_nll'][kind],
            f'H_{kind}_coverage90': h['coverage'][kind],
            f'H_{kind}_coverage_abs_error': abs(h['coverage'][kind]-.9),
            f'H_{kind}_spread_um': h['spread'][kind],
            f'H_{kind}_endpoint_rmse_um': h['rmse_um'][kind],
            f'H_{kind}_secant_rmse_mm_s': row['endpoint_metrics']['secant_rmse_mm_s'][kind]})
    for f, kind in enumerate(('normal', 'super')):
        values[f'{kind}_departure_from_cold_full_rms_mm_s'] = row['field']['departure_from_cold_full_rms_mm_s'][f]
        values[f'{kind}_departure_from_cold_initial_rms_mm_s'] = row['field']['departure_from_cold_initial_rms_mm_s'][f]
    values['trained_pde_loss'] = row['pde']['trained_simpson']['pde_loss']
    values['quarter_pde_loss'] = row['pde']['independent_quarter']['pde_loss']
    return values


def aggregate(records):
    out = {}
    for case in CASES:
        case_out = {}
        for budget, mode in [(0, 'background')]+[(b, m) for b in BUDGETS for m in ('primary', 'terminal')]:
            rows = [r for r in records if (r['case'], r['budget'], r['mode']) == (case, budget, mode)]
            if len(rows) != 3 or {r['seed'] for r in rows} != set(SEEDS):
                raise RuntimeError('Each descriptive aggregate requires all three fixed numerical seeds')
            values = [metric_values(r) for r in rows]
            case_out[f'{budget}_{mode}'] = {k: dict(mean=float(np.mean([v[k] for v in values])),
                sample_std=float(np.std([v[k] for v in values], ddof=1)),
                minimum=float(min(v[k] for v in values)), maximum=float(max(v[k] for v in values)))
                for k in values[0]}
        out[case] = case_out
    return out


def paired_background(records):
    out = []
    for r in records:
        if r['mode'] == 'background':
            continue
        bg = next(b for b in records if b['case'] == r['case'] and b['seed'] == r['seed'] and b['mode'] == 'background')
        values, bvalues = metric_values(r), metric_values(bg)
        out.append(dict(case=r['case'], seed=r['seed'], budget=r['budget'], mode=r['mode'],
            difference={k: values[k]-bvalues[k] for k in values},
            degree28_nll_per_ms_difference={k: r['heldout_degree28'][k]-bg['heldout_degree28'][k]
                for k in ('total', 'free', 'cap')},
            probability_sha256=r['probability_sha256'], paired_on_same_PID=True))
    return out


def support(c, z):
    report = dict(camera_fps=float(c['camera_fps']), start_s=float(c['start_s']), duration_ms=100,
        domain_m=[.016, .008], physics_mean_velocity_mm_s=[c['physics']['vn_mean_m_s']*1000,
        c['physics']['vs_mean_m_s']*1000], groups={})
    for group in ('fit', 'selection', 'test'):
        gm = group_mask(z['pid'], group)
        row = dict(unique_physical_PID=int(z['pid'][gm].unique().numel()))
        for state_number, name in ((0, 'free'), (1, 'cap')):
            use = gm & (z['state'] == state_number)
            row[name] = dict(PID=int(z['pid'][use].unique().numel()), intervals=int(use.sum()),
                segments=int(z['segment'][use].unique().numel()), exposure_ms=float((z['end']-z['start'])[use].sum()))
        report['groups'][group] = row
    return report


def run():
    setup()
    protocol, gate = evaluation_gate()
    if protocol.get('pde_normalizer') != 'sum_fit_native_durations_ms':
        raise RuntimeError('Expected the frozen native-exposure PDE normalizer')
    gate_path = OUT/'selection_frozen.json'
    gate_sha = sha(gate_path)
    binding = fingerprint(dict(protocol=protocol['fingerprint'], gate=gate_sha,
        evaluation_source=sha(Path(__file__)), degrees=DEGREES))
    started, cpu_start = time.perf_counter(), time.process_time()
    records, exports, run_costs, histories, support_records = [], [], [], [], {}
    for run_record in gate['runs']:
        case, seed = run_record['case'], run_record['seed']
        c, z, op, prior, net, cal, force, field = context(case, seed)
        before = state(net)
        exposure_ms = float((z['end']-z['start'])[group_mask(z['pid'], 'fit')].sum())
        support_records[case] = support(c, z)
        status = read(run_record['status_path'])
        if sha(run_record['status_path']) != run_record['status_sha256'] or not status['complete']:
            raise RuntimeError('Run status changed after global freeze')
        trace_path = Path(run_record['status_path']).parent/'history.json'
        trace = read(trace_path)
        histories.append(dict(case=case, seed=seed, source=str(trace_path), sha256=sha(trace_path), rows=trace))
        prefix = next(r for r in trace if r['step'] == 200)
        run_costs.append(dict(case=case, seed=seed, cpu_seconds=status['cpu_seconds'], wall_seconds=status['wall_seconds'],
            budget200_cpu_seconds=prefix['cpu_seconds'], budget200_wall_seconds=prefix['wall_seconds'],
            additional201_600_cpu_seconds=status['cpu_seconds']-prefix['cpu_seconds'],
            additional201_600_wall_seconds=status['wall_seconds']-prefix['wall_seconds']))
        for budget, mode, checkpoint in scenarios(run_record):
            name = row_id(case, seed, budget, mode)
            destination = OUT/'evaluation'/f'{name}.json'
            cached = read(destination) if destination.exists() else None
            if cached is not None:
                if cached.get('binding') != binding or cached['checkpoint_sha256'] != checkpoint['sha256']:
                    raise RuntimeError('Existing evaluation cache has a different frozen binding')
                row = cached
            else:
                tick, tick_cpu = time.perf_counter(), time.process_time()
                ck = load(checkpoint['path'])
                if ck['protocol_fingerprint'] != protocol['fingerprint'] or ck['probability_sha256'] != run_record['probability_sha256']:
                    raise RuntimeError('Checkpoint provenance mismatch')
                if ck['network_steps'] != 0:
                    raise RuntimeError('Particle network was updated in target inversion')
                field.load_state_dict(ck['field_state'])
                with torch.no_grad():
                    if not torch.equal(field.states(), ck['a']):
                        raise RuntimeError('Frozen field replay differs from saved 101 frames')
                    h20 = score(c, z, field, net, cal, force, 'test', DEGREES[0], metrics=True)
                    h28 = score(c, z, field, net, cal, force, 'test', DEGREES[1], metrics=True)
                    fa = field_audit(c, field, op)
                    pa = pde_audit(field, op, prior, exposure_ms)
                duration_ms = 1000/float(c['camera_fps'])
                row = dict(complete=True, binding=binding, case=case, seed=seed, budget=budget, mode=mode,
                    step=int(ck['step']), checkpoint_path=checkpoint['path'], checkpoint_sha256=checkpoint['sha256'],
                    probability_sha256=run_record['probability_sha256'], protocol_fingerprint=protocol['fingerprint'],
                    selection_gate_sha256=gate_sha, validation=ck['val'], validation_source='frozen checkpoint; already replayed by global freeze',
                    heldout=h20, heldout_degree28=h28, quadrature=probability_difference(h20, h28),
                    endpoint_metrics=dict(native_interval_ms=duration_ms,
                        secant_rmse_mm_s={k: h20['rmse_um'][k]/duration_ms for k in ('free', 'cap')},
                        interpretation='Conditional endpoint predictive-mean error, not instantaneous velocity truth or camera localization uncertainty'),
                    field=fa, pde=pa, evaluation_wall_seconds=time.perf_counter()-tick,
                    evaluation_cpu_seconds=time.process_time()-tick_cpu, network_steps=0,
                    reference_read=False, target_test_used_for_selection=False)
                dump(destination, plain(row))
                print('CF_EVALUATED', name, flush=True)
            records.append(row)
            if seed == REPRESENTATIVE_SEED:
                field.load_state_dict(load(checkpoint['path'])['field_state'])
                path = OUT/'fields'/f'{name}.npz'
                metadata = dict(case=case, seed=seed, budget=budget, mode=mode,
                    checkpoint_path=checkpoint['path'], checkpoint_sha256=checkpoint['sha256'],
                    probability_sha256=run_record['probability_sha256'])
                exports.append(export_field(c, field, path, metadata))
            if any(not torch.equal(v, before[k]) for k, v in net.state_dict().items()):
                raise RuntimeError('Evaluation modified particle model weights')
            dump(OUT/'evaluation_status.json', dict(complete=False, records_complete=len(records),
                expected_records=30, latest=name, updated_utc=utc(), binding=binding))
    if len(records) != 30 or len(exports) != 10:
        raise RuntimeError('Expected 30 evaluation records and 10 representative field series')
    summary = dict(complete=True, created_utc=utc(), binding=binding, protocol_fingerprint=protocol['fingerprint'],
        selection_gate_sha256=gate_sha, runs=records, summary=aggregate(records),
        paired_background=paired_background(records), support=support_records, histories=histories,
        cost=dict(unique_field_runs=run_costs, new_probability_training_runs=0,
            unique_inverse_cpu_seconds=sum(r['cpu_seconds'] for r in run_costs),
            sum_inverse_worker_wall_seconds=sum(r['wall_seconds'] for r in run_costs),
            budget200_cpu_seconds=sum(r['budget200_cpu_seconds'] for r in run_costs),
            budget200_worker_wall_seconds=sum(r['budget200_wall_seconds'] for r in run_costs),
            evaluation_current_invocation_wall_seconds=time.perf_counter()-started,
            evaluation_current_invocation_cpu_seconds=time.process_time()-cpu_start,
            interpretation='200-step prefixes are included in 600-step costs; worker wall sum overlaps in parallel'),
        probability_weights_updated=False, reference_field_available=False,
        observation_scope='Same-window native endpoint conditional on past observed positions; not free-running future trajectories',
        limitations=['Two experimental windows, no field truth', 'Three numerical seeds share physical data',
            'Frozen 1ms simulation probability model uses a new camera propagation adapter',
            'Long intervals rely on causal PCHIP/past-secant pilot and ADF moment closure',
            'Fractional latent kernels and force discretization are migration assumptions',
            'y-periodic wall-channel model approximates a finite experimental view',
            'Fixed L0, K12 and uncalibrated full Frozen B covariance push-forward',
            'Previous CF results use different probability/score/physics and are not direct numerical comparators',
            'NLL, calibration and PDE consistency do not establish true field accuracy'])
    dump(OUT/'CF_EVALUATION_SUMMARY.json', plain(summary))
    dump(OUT/'FIELD_MANIFEST.json', dict(complete=True, binding=binding, protocol_fingerprint=protocol['fingerprint'],
        evaluation_sha256=sha(OUT/'CF_EVALUATION_SUMMARY.json'), artifacts=exports,
        representative_seed_fixed_before_test=REPRESENTATIVE_SEED, frames_per_export=101))
    dump(OUT/'evaluation_status.json', dict(complete=True, records_complete=30, field_exports=10,
        updated_utc=utc(), binding=binding))
    write_brief(summary)


def write_brief(summary):
    lines = ['PINN基线 CF 两窗口重建：冻结后评价摘要',
        '无CF流场真值；本表仅报告条件粒子预测，不能解释为常/超真实场误差。',
        'NLL为PID平衡NLL/ms；覆盖应接近90%；宽度为位移协方差迹平方根μm。',
        '数值为三个种子均值，种子共享物理数据，不是三次独立实验。', '',
        'case | budget/mode | H free/cap NLL/ms | coverage | width μm | endpoint RMSE μm']
    for case in CASES:
        for label, row in summary['summary'][case].items():
            def pair(suffix):
                return '/'.join(f"{row[f'H_{kind}_{suffix}']['mean']:.7g}" for kind in ('free', 'cap'))
            lines.append(f"{case} | {label} | {pair('nll_per_ms')} | {pair('coverage90')} | "
                         f"{pair('spread_um')} | {pair('endpoint_rmse_um')}")
    lines += ['', '600步为200步同一轨迹的扩展；V主选与固定终态分别报告。',
        'GH20/28及200个quarter非配点PDE诊断为固定快照的描述性检查，不用于改选。',
        '新概率网络训练成本为0；固定背景和PINN均使用同一新相机适配。',
        '完整结果：CF_EVALUATION_SUMMARY.json；代表场：FIELD_MANIFEST.json。']
    (OUT/'CF_EVALUATION_REPORT.txt').write_text('\n'.join(lines)+'\n', encoding='utf8')


if __name__ == '__main__':
    run()
