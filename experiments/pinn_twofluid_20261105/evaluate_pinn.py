"""Globally gated continuous-field evaluation; never fits or chooses models.

All twelve runs, four snapshots per run, three frozen probability models and
twelve historical comparison snapshots are verified before H/reference access.
The old joint pipeline is an external step-200 reference, not a matched PINN
factorial arm: its likelihood query subtracts eta and its objective differs.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
OUT = BASE/'outputs/PINNTwoFluid_20261105'
HELPER = HERE.parent/'joint_probability_20261103/evaluate_joint.py'
CASES = (('vn0_0015m_s', .0015), ('vn0_004m_s', .004))
SEEDS = (301030, 301031, 301032)
TRAIN_SEEDS = (411040, 411041, 411042)
ARMS = ('global_spline', 'pinn')
HISTORICAL = 'historical_joint'
BUDGETS = (200, 600)
MODES = ('primary', 'terminal')
DEGREES = (20, 28)
KEYS = ('total', 'free', 'cap')
VERSION = 'pinn-twofluid-evaluation-v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    tmp.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def same_state(actual, expected, label):
    if set(actual) != set(expected):
        raise RuntimeError('Different tensor keys: '+label)
    for key, value in actual.items():
        if value.shape != expected[key].shape or not torch.equal(value, expected[key].to(value)):
            raise RuntimeError('Changed tensor: '+label+'/'+key)


def load_hashed(C, path, digest):
    if sha(path) != digest:
        raise RuntimeError('Changed checkpoint: '+str(path))
    value = C.load(path)
    if sha(path) != digest:
        raise RuntimeError('Checkpoint changed during reading: '+str(path))
    return value


def cheap_gate(out=OUT):
    out = Path(out).resolve()
    gate = read(out/'selection_frozen.json')
    expected = {(case, arm, seed) for case, _ in CASES for arm in ARMS for seed in SEEDS}
    rows = gate.get('runs', [])
    if gate.get('complete') is not True or len(rows) != 12:
        raise RuntimeError('All twelve runs must finish and freeze before target evaluation')
    if {(r['case'], r['arm'], int(r['seed'])) for r in rows} != expected:
        raise RuntimeError('Duplicate/missing twelve-run identity')
    entries = []
    for row in rows:
        if abs(float(row['U'])-dict(CASES)[row['case']]) > 1e-14:
            raise RuntimeError('Incorrect case velocity in freeze gate')
        if set(row['budgets']) != set(map(str, BUDGETS)):
            raise RuntimeError('Both 200 and 600 budgets must be frozen')
        for budget in BUDGETS:
            bucket = row['budgets'][str(budget)]
            if set(bucket) != set(MODES):
                raise RuntimeError('Both primary and terminal snapshots required')
            for mode in MODES:
                entry = bucket[mode]
                expected_path = out/'runs'/row['case']/row['arm']/f"seed_{row['seed']}"/f'budget_{budget}_{mode}.pt'
                if Path(entry['path']).resolve() != expected_path:
                    raise RuntimeError('Checkpoint escaped expected run directory')
                step = int(entry['step'])
                if (mode == 'terminal' and step != budget) or not 0 <= step <= budget:
                    raise RuntimeError('Incorrect snapshot budget/selection step')
                if sha(entry['path']) != entry['sha256']:
                    raise RuntimeError('Changed globally frozen snapshot')
                entries.append(dict(case=row['case'], U=float(row['U']), arm=row['arm'],
                    seed=int(row['seed']), budget=budget, mode=mode, source='new', **entry))
    if len(entries) != 48:
        raise RuntimeError('Forty-eight new snapshots required')
    return gate, entries


def bind_gates(C, out):
    gate, entries = cheap_gate(out)
    p = C.verify_protocol()
    if gate.get('protocol_fingerprint', gate.get('fingerprint')) != p['fingerprint']:
        raise RuntimeError('Field gate/protocol mismatch')
    sources = {str(Path(k).resolve()): v for k, v in p['sources'].items()}
    for path in (Path(__file__), HELPER):
        if sources.get(str(path.resolve())) != sha(path):
            raise RuntimeError('Evaluator/helper must be prospectively source-bound: '+str(path))
    pg = C.selection_gate()
    if pg.get('complete') is not True or pg.get('family') != 'joint':
        raise RuntimeError('This comparison prospectively binds the selected joint probability family')
    models = pg['models']
    if len(models) != 3 or {int(r['training_seed']) for r in models} != set(TRAIN_SEEDS):
        raise RuntimeError('All three probability model seeds must be frozen')
    states = {}
    for r in models:
        ck = load_hashed(C, r['path'], r['sha256'])
        states[int(r['field_seed'])] = (r, ck['net_state'])
    run_costs = {}
    for row in gate['runs']:
        if sha(row['status_path']) != row['status_sha256']:
            raise RuntimeError('Changed frozen run status')
        status = read(row['status_path'])
        last = load_hashed(C, row['last_path'], row['last_sha256'])
        if (status['complete'] is not True or status['step'] != 600 or last['step'] != 600
                or status['network_steps'] != 0 or last['network_steps'] != 0
                or status['network_parameters_equal'] is not True
                or status['protocol_fingerprint'] != p['fingerprint']):
            raise RuntimeError('Incomplete budget, frozen-network audit or protocol identity')
        for budget in BUDGETS:
            trace_row = next(v for v in last['trace'] if v['step'] == budget)
            run_costs[(row['case'], row['arm'], int(row['seed']), budget)] = dict(
                wall_seconds=float(trace_row['wall_seconds']), cpu_seconds=float(trace_row['cpu_seconds']))
    for entry in entries:
        entry.update(run_costs[(entry['case'], entry['arm'], entry['seed'], entry['budget'])])
        ck = load_hashed(C, entry['path'], entry['sha256'])
        if (ck['arm'] != entry['arm'] or int(ck['seed']) != entry['seed']
                or int(ck['step']) != entry['step'] or abs(float(ck['U'])-entry['U']) > 1e-14
                or ck['protocol_fingerprint'] != p['fingerprint']):
            raise RuntimeError('New checkpoint identity/protocol mismatch')
        if 'net_state' in ck:
            same_state(ck['net_state'], states[entry['seed']][1], 'new frozen probability')
        for key in ('probability_sha256', 'probability_sha'):
            if key in ck and ck[key] != states[entry['seed']][0]['sha256']:
                raise RuntimeError('New snapshot changed probability binding')
        if ck.get('network_steps', 0) != 0:
            raise RuntimeError('Target probability updates are forbidden')
    # Old read-only gate is checked in full before opening any target reference.
    old_gate_path = C.NOUT/'selection_frozen.json'
    old_gate = read(old_gate_path)
    if not old_gate.get('all_required_selections_frozen') or not old_gate.get('all_terminal_states_bound_before_evaluation'):
        raise RuntimeError('Historical selected/terminal gate is incomplete')
    histories = [r for r in old_gate['runs'] if r['arm'] == 'joint']
    if len(histories) != 6 or {(r['case'], int(r['seed'])) for r in histories} != {(c, s) for c, _ in CASES for s in SEEDS}:
        raise RuntimeError('Six historical joint reconstructions required')
    for r in histories:
        historical_last = load_hashed(C, r['last_path'], r['last_sha256'])
        for mode in MODES:
            path, digest = (r['path'], r['sha256']) if mode == 'primary' else (r['last_path'], r['last_sha256'])
            ck = load_hashed(C, path, digest)
            if ck['arm'] != 'joint' or int(ck['seed']) != int(r['seed']) or abs(float(ck['U'])-float(r['U'])) > 1e-14:
                raise RuntimeError('Historical reconstruction identity mismatch')
            if mode == 'terminal' and (ck['iteration'] != 200 or ck['network_steps'] != 0):
                raise RuntimeError('Historical terminal is not frozen step 200')
            same_state(ck['net_state'], states[int(r['seed'])][1], 'historical versus selected probability')
            entries.append(dict(case=r['case'], U=float(r['U']), arm=HISTORICAL, seed=int(r['seed']),
                budget=200, mode=mode, source='historical', path=path, sha256=digest, step=int(ck['iteration']),
                wall_seconds=float(historical_last['wall_seconds']), cpu_seconds=float(historical_last['cpu_seconds'])))
    manifest = dict(version=VERSION, protocol_fingerprint=p['fingerprint'],
        field_gate_sha256=sha(out/'selection_frozen.json'),
        probability_gate_sha256=sha(out/'probability_selection/selection_frozen.json'),
        historical_gate_sha256=sha(old_gate_path), evaluator_sha256=sha(__file__), helper_sha256=sha(HELPER),
        snapshots=entries, probability_models=models, primary_degree=20, sensitivity_degree=28,
        all_48_new_and_12_historical_snapshots_bound_before_target_evaluation=True,
        new_query='true continuous midpoint a/adot; particle acceleration ut+g^T u; no eta subtraction',
        historical_query='legacy midpoint; subtract eta_n/2 from normal velocity, both eta/dt from acceleration',
        historical_comparison='external step-200 reference; query, objective and parameterization differ')
    path = out/'evaluation_manifest.json'
    if path.exists() and read(path) != manifest:
        raise RuntimeError('Existing evaluation manifest differs')
    if not path.exists():
        dump(path, manifest)
    return p, entries, states, sha(path)


def checked_nodes(Q, am, dm, d, net, cal, force, group, degree, metrics=True):
    result = Q.score_nodes(am, dm, d, net, cal, force, group, degree, metrics=metrics)
    vals = [result[k] for k in KEYS]
    if metrics:
        vals += [result[k][s] for k in ('coverage', 'spread') for s in ('free', 'cap')]
    if not np.isfinite(vals).all():
        raise RuntimeError('Nonfinite continuous-query likelihood')
    return result


def extended_field_metrics(J, a, reference, U):
    result = J.field_metrics(a, reference, U)
    time_rms = np.asarray(result['time_rms_mm_s'])
    result.update(relative_t5ms_rms_mm_s=time_rms[5].tolist(),
        initial_5ms_window_rms_mm_s=np.sqrt(np.square(time_rms[:6]).mean(0)).tolist(),
        relative_time_s=(np.arange(96)*.001).tolist(), physical_time_s=(np.arange(5, 101)*.001).tolist(),
        initial_definition='relative t=0, historical physical time=5 ms; initial_5ms_window uses relative 0..5 ms (6 nodes)')
    return result


@torch.no_grad()
def pde_collocation_audit(C, field, old):
    """Postfreeze quarter-point check; does not feed optimization/selection.

    Each path has 95 nodes before the unchanged normal B whitening operation.
    Trained left/mid/right paths use Simpson 1/6,4/6,1/6; untrained quarter and
    three-quarter paths use equal weights for a separately named diagnostic.
    dt*residual is in the same coefficient units as historical increments.
    """
    if float(old.op.nu_s) != 0. or float(old.op.friction_factor) != 1.:
        raise RuntimeError('PDE audit physics differs from frozen nu_s=0/Fns=1')
    base = torch.arange(95, dtype=old.theta.dtype, device=old.theta.device)*.001
    fractions = (0., .25, .5, .75, 1.)
    paths = []
    for fraction in fractions:
        times = base+fraction*.001
        a, adot = field.coefficients_and_dt(times)
        residual = (adot-C.CURRENT.batched_rhs(old.op, a))*.001
        if not torch.isfinite(residual).all():
            raise RuntimeError('Nonfinite independent PDE collocation residual')
        paths.append(residual)
    metrics = []
    for fraction, residual in zip(fractions, paths):
        metrics.append(dict(interval_fraction=fraction, points=95,
            dt_coefficient_rms=torch.sqrt(residual.square().mean((0, 2))).cpu().tolist(),
            dt_velocity_rms_mm_s=(torch.sqrt(residual.square().sum(-1).mean(0)/2)*old.op.U*1000).cpu().tolist(),
            instantaneous_acceleration_rms_m_s2=(torch.sqrt(residual.square().sum(-1).mean(0)/2)*old.op.U/.001).cpu().tolist(),
            normal_B_energy=float(old.prior.whiten(residual[:, 0]).square().sum()),
            super_std_energy=float((residual[:, 1]/old.std[1]).square().sum()),
            max_abs_dt_coefficient=float(residual.abs().max())))
    def combined(indices, weights):
        coefficient_mse = sum(weight*paths[i].square().mean((0, 2)) for i, weight in zip(indices, weights))
        vector_mse = sum(weight*paths[i].square().sum(-1).mean(0)/2 for i, weight in zip(indices, weights))
        return dict(points=sum(metrics[i]['points'] for i in indices), path_weights=list(weights),
            dt_coefficient_rms=coefficient_mse.sqrt().cpu().tolist(),
            dt_velocity_rms_mm_s=(vector_mse.sqrt()*old.op.U*1000).cpu().tolist(),
            normal_B_energy=sum(weight*metrics[i]['normal_B_energy'] for i, weight in zip(indices, weights)),
            super_std_energy=sum(weight*metrics[i]['super_std_energy'] for i, weight in zip(indices, weights)))
    trained = combined((0, 2, 4), (1/6, 4/6, 1/6))
    independent = combined((1, 3), (.5, .5))
    ratio = {key: independent[key]/max(trained[key], 1e-30) for key in ('normal_B_energy', 'super_std_energy')}
    return dict(paths=metrics, trained_simpson=trained, independent_quarter=independent,
        independent_to_trained_energy_ratio=ratio, independent_points=190,
        trained_unique_points=191, trained_path_points_with_endpoint_reuse=285,
        selection_or_optimization_use=False, physics=dict(nu_s=0., Fns=1.),
        normal_B_definition='Frozen B whitening separately on each 95-node path',
        interpretation='Independent temporal collocation diagnostic; no post-hoc threshold or reselection')


def flatten_metrics(row, degree=20):
    h = row['observations']['H'] if degree == 20 else row['heldout_degree28']
    f = row['field']
    result = dict(normal_rms_mm_s=f['rms_mm_s'][0], superfluid_rms_mm_s=f['rms_mm_s'][1], heldout_total_nll=h['total'])
    for i, fluid in enumerate(('normal', 'superfluid')):
        for key, source in (('initial', 'initial_rms_mm_s'), ('relative_t5ms', 'relative_t5ms_rms_mm_s'),
                            ('initial_5ms_window', 'initial_5ms_window_rms_mm_s')):
            result[f'{key}_{fluid}_rms_mm_s'] = f[source][i]
    for state in ('free', 'cap'):
        result[f'heldout_{state}_nll'] = h[state]
        result[f'heldout_{state}_coverage'] = h['coverage'][state]
        result[f'heldout_{state}_coverage_abs_error'] = abs(h['coverage'][state]-.9)
        result[f'heldout_{state}_spread_um'] = h['spread'][state]
    return result


def describe(values):
    return dict(mean=float(np.mean(values)), min=float(np.min(values)), max=float(np.max(values)),
                by_seed={str(s): float(v) for s, v in zip(SEEDS, values)})


def summarize(rows):
    index = {(r['budget'], r['mode'], r['case'], r['arm'], r['seed']): r for r in rows}
    if len(index) != 60:
        raise RuntimeError('Sixty unique new/historical rows required')
    summaries, quadrature_directions, primary_terminal, budget_changes = {}, [], [], []
    for budget in BUDGETS:
        summaries[str(budget)] = {}
        for mode in MODES:
            summaries[str(budget)][mode] = {}
            for degree in DEGREES:
                section = {}
                for case, U in CASES:
                    values = {}
                    for arm in (*ARMS, HISTORICAL):
                        b = 200 if arm == HISTORICAL else budget
                        values[arm] = [flatten_metrics(index[(b, mode, case, arm, s)], degree) for s in SEEDS]
                    item = dict(U=U, arms={}, paired_changes={})
                    for arm in values:
                        item['arms'][arm] = {k: describe([v[k] for v in values[arm]]) for k in values[arm][0]}
                    for a, b in (('pinn', 'global_spline'), ('pinn', HISTORICAL), ('global_spline', HISTORICAL)):
                        changes = {k: describe([x[k]-y[k] for x, y in zip(values[a], values[b])]) for k in values[a][0]}
                        core = ('normal_rms_mm_s', 'superfluid_rms_mm_s', 'heldout_free_nll', 'heldout_cap_nll')
                        count = sum(all(x[k] < y[k] for k in core) for x, y in zip(values[a], values[b]))
                        item['paired_changes'][a+'_minus_'+b] = dict(metrics=changes, joint_four_metric_improvement_count=count,
                            all_four_mean_deltas_negative=all(changes[k]['mean'] < 0 for k in core),
                            historical_budget=200 if b == HISTORICAL else None)
                    section[case] = item
                summaries[str(budget)][mode][str(degree)] = section
            for case, _ in CASES:
                for pair in summaries[str(budget)][mode]['20'][case]['paired_changes']:
                    for metric in ('heldout_free_nll', 'heldout_cap_nll'):
                        deltas = [summaries[str(budget)][mode][str(d)][case]['paired_changes'][pair]['metrics'][metric]['mean'] for d in DEGREES]
                        quadrature_directions.append(dict(budget=budget, mode=mode, case=case, pair=pair,
                            metric=metric, delta20=deltas[0], delta28=deltas[1], sign_reversal=deltas[0]*deltas[1] < 0))
        for case, _ in CASES:
            for arm in ARMS:
                tt = [flatten_metrics(index[(budget, 'terminal', case, arm, s)]) for s in SEEDS]
                pp = [flatten_metrics(index[(budget, 'primary', case, arm, s)]) for s in SEEDS]
                primary_terminal.append(dict(budget=budget, case=case, arm=arm,
                    metrics={k: describe([t[k]-p[k] for t, p in zip(tt, pp)]) for k in tt[0]}))
    for mode in MODES:
        for case, _ in CASES:
            for arm in ARMS:
                hi = [flatten_metrics(index[(600, mode, case, arm, s)]) for s in SEEDS]
                lo = [flatten_metrics(index[(200, mode, case, arm, s)]) for s in SEEDS]
                budget_changes.append(dict(mode=mode, case=case, arm=arm,
                    metrics={k: describe([x[k]-y[k] for x, y in zip(hi, lo)]) for k in hi[0]}))
    return summaries, quadrature_directions, primary_terminal, budget_changes


def flatten_score(h):
    result = {k+'_nll': float(h[k]) for k in KEYS}
    for state in ('free', 'cap'):
        result[state+'_coverage'] = float(h['coverage'][state])
        result[state+'_spread_um'] = float(h['spread'][state])
    return result


@torch.no_grad()
def attribution(C, J, Q, rows, out, binding):
    """Fixed old coefficients, then continuous-query convention, then new field.

    The intermediate Hermite curve is deterministic, unoptimized, and exactly
    reproduces all 96 historical a nodes. Its derivative is the derivative of
    that interpolation. It is not a claim about the old model between nodes.
    """
    from global_spline import GlobalSpline
    index = {(r['budget'], r['mode'], r['case'], r['arm'], r['seed']): r for r in rows}
    result = []
    for case, U in CASES:
        for seed in SEEDS:
            oldrow = index[(200, 'primary', case, HISTORICAL, seed)]
            ck = load_hashed(C, oldrow['checkpoint_path'], oldrow['checkpoint_sha256'])
            d, _, model, net, cal, force = C.E.make_context(U, seed, 'joint')
            same_state(net.state_dict(), ck['net_state'], 'auxiliary historical probability')
            model.load_state_dict(ck['model_state'])
            a, _, _ = model()
            fixed = GlobalSpline(torch.zeros_like(a[0]), torch.ones_like(a[0, 0]), seed=seed)
            fixed.controls.copy_(a)
            if not torch.equal(fixed.states(), a):
                raise RuntimeError('Fixed historical interpolation changed nodal coefficients')
            am, dm = fixed.coefficients_and_dt((torch.arange(95, dtype=a.dtype)+.5)*.001)
            cache = out/'auxiliary'/case/f'seed_{seed}'/'historical_continuous_query.json'
            key = dict(manifest_sha256=binding, historical_checkpoint_sha256=oldrow['checkpoint_sha256'])
            cached = read(cache) if cache.exists() else None
            if cached is None or cached.get('cache_key') != key:
                cached = dict(cache_key=key, by_degree={str(degree): checked_nodes(Q, am, dm, d, net, cal, force, 'H', degree) for degree in DEGREES})
                dump(cache, cached)
            for budget in BUDGETS:
                for arm in ARMS:
                    own = index[(budget, 'primary', case, arm, seed)]
                    degrees = {}
                    for degree in DEGREES:
                        old = oldrow['observations']['H'] if degree == 20 else oldrow['heldout_degree28']
                        new = own['observations']['H'] if degree == 20 else own['heldout_degree28']
                        cross = cached['by_degree'][str(degree)]
                        b, x, n = map(flatten_score, (old, cross, new))
                        components = dict(query_definition_change_at_fixed_nodes={k: x[k]-b[k] for k in b},
                            field_change_under_continuous_query={k: n[k]-x[k] for k in b},
                            total_change={k: n[k]-b[k] for k in b})
                        assert all(abs(components['query_definition_change_at_fixed_nodes'][k]+components['field_change_under_continuous_query'][k]-components['total_change'][k]) < 1e-10 for k in b)
                        degrees[str(degree)] = dict(historical_original_query=old,
                            historical_fixed_nodes_continuous_query=cross, new_field_continuous_query=new, components=components)
                    result.append(dict(case=case, seed=seed, arm=arm, budget=budget, mode='primary',
                        fixed_historical_checkpoint_sha256=oldrow['checkpoint_sha256'],
                        own_checkpoint_sha256=own['checkpoint_sha256'], by_degree=degrees))
            print('ATTRIBUTION', case, seed, flush=True)
    if len(result) != 24:
        raise RuntimeError('Twenty-four auxiliary primary comparisons required')
    return dict(complete=True, rows=result, probability_unchanged=True, fitting_or_selection=False,
        interpretation='Descriptive path-dependent decomposition, not causal; fixed old 96 nodes are interpolated without fitting',
        fixed_query='Historical original a/eta query versus exact-nodal deterministic continuous Hermite interpolation; then new field')


def export_field(J, row, a, reference, c, out, binding, *, am=None, dm=None):
    x, velocity, means = J.resolve_fields(a, c)
    _, reference_v, _ = J.resolve_fields(reference, c)
    rms = np.sqrt(np.square((velocity-reference_v)*1000).sum(-1).mean((0, 2, 3)))
    error = float(np.max(np.abs(rms-np.asarray(row['field']['rms_mm_s']))))
    if error > 1e-9:
        raise RuntimeError('Exported grid RMS differs from coefficient RMS')
    path = out/'fields'/str(row['budget'])/row['mode']/row['case']/f"{row['arm']}_seed_{row['seed']}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {} if am is None else dict(midpoint_a=am.cpu().numpy(), midpoint_adot_per_s=dm.cpu().numpy())
    np.savez_compressed(path, time_s=np.arange(5, 101)*.001, relative_time_s=np.arange(96)*.001,
        x_m=x, y_m=x, a=a.cpu().numpy(), velocity_m_s=velocity, velocity_error_m_s=velocity-reference_v,
        means_m_s=means, checkpoint_sha256=row['checkpoint_sha256'], reference_sha256=row['reference_sha256'],
        evaluation_manifest_sha256=binding, iteration=row['iteration'], arm=row['arm'], mode=row['mode'], budget=row['budget'], **extra)
    with np.load(path) as check:
        assert check['velocity_m_s'].shape == (96, 2, 40, 40, 2)
        assert np.array_equal(check['velocity_m_s'], velocity)
    return dict(path=str(path.resolve()), sha256=sha(path), kind='field_npz', frames=96,
        case=row['case'], arm=row['arm'], mode=row['mode'], budget=row['budget'], seed=row['seed'],
        grid_vs_summary_rms_max_mm_s=error, npz_readback_exact=True)


def report_text(result):
    lines = ['连续双流场第一轮：全局样条与PINN，冻结同一粒子概率模型', '',
        '新增两臂不设独立初场参数、过程eta或闭合网络；全时间共同拟合确定性二流方程残差与粒子似然。',
        'nu_s=0，Fns=1。冻结B只提供方程残差尺度/相关性，不是可反演过程增量。',
        '同一joint概率家族及三个配对模型冻结；旧joint四维变分仅为200步外部参照，查询定义和目标函数也不同。',
        '新12run共48主/末态与3概率模型冻结后评分H/reference；旧6run主/末态另12行。',
        '初场relative0=physical5ms；initial_5ms_window是relative0..5ms的六帧均方RMS；全窗relative0..95ms。',
        'RMS mm/s；宽度sqrt(trace displacement covariance) μm；覆盖率名义90%。H为同窗因果一步条件预测，不是无观测滚动。', '']
    for budget in BUDGETS:
        for mode in MODES:
            lines += [f'budget={budget}, mode={mode}; historical始终200步',
                'case | arm | initial n/s | first5ms n/s | full n/s | H free/cap NLL | coverage free/cap | width free/cap']
            for case, _ in CASES:
                for arm in (*ARMS, HISTORICAL):
                    stats = result['summary'][str(budget)][mode]['20'][case]['arms'][arm]
                    v = lambda key: stats[key]['mean']
                    pairs = [(v('initial_normal_rms_mm_s'), v('initial_superfluid_rms_mm_s')),
                        (v('initial_5ms_window_normal_rms_mm_s'), v('initial_5ms_window_superfluid_rms_mm_s')),
                        (v('normal_rms_mm_s'), v('superfluid_rms_mm_s')),
                        (v('heldout_free_nll'), v('heldout_cap_nll')),
                        (v('heldout_free_coverage'), v('heldout_cap_coverage')),
                        (v('heldout_free_spread_um'), v('heldout_cap_spread_um'))]
                    lines.append(f'{case} | {arm} | '+' | '.join(f'{a:.8g}/{b:.8g}' for a, b in pairs))
            lines.append('')
    lines += ['限制：三个种子共享物理数据；历史目标已探索；单一非目标6.5工况概率家族选择。',
        '概率拟合改善不等于场改善；ADF、上下文偏移与PINN/样条表达能力及优化收敛限制仍需单独解释。',
        '方程软残差仍可承受偏差；用统一时间参数表示不保证初场和内点具有相同可辨识性或梯度条件。',
        '辅助分解将旧query到固定旧节点连续query的变化，与连续query下新场变化区分；不是纯因果归因。',
        '所有200/600预算及主/末态均预先保留，未根据H/真值再选择；GH20/28为敏感性，非积分收敛保证。', '']
    return '\n'.join(lines)


@torch.no_grad()
def evaluate(out=OUT):
    started = time.perf_counter(); out = Path(out).resolve()
    # Cheap gate runs before importing a runner that can load observations.
    cheap_gate(out)
    import pinn_common as C
    import continuous_query as Q
    import experiment as E
    if out != C.OUT.resolve():
        raise RuntimeError('Evaluator output differs from frozen experiment')
    C.setup()
    J = module('pinn_evaluation_pure_helpers', HELPER)
    p, entries, model_states, binding = bind_gates(C, out)
    # No target reference/H read is allowed above this global gate boundary.
    rows, references, artifacts = [], {}, []
    for case, U in CASES:
        rp = J.reference_path(case); rdigest = sha(rp)
        reference = load_hashed(C, rp, rdigest)['a'].double().cpu()
        references[case] = dict(path=str(rp.resolve()), sha256=rdigest)
        for entry in [e for e in entries if e['case'] == case]:
            arm, seed, budget, mode = entry['arm'], entry['seed'], entry['budget'], entry['mode']
            ck = load_hashed(C, entry['path'], entry['sha256'])
            if entry['source'] == 'new':
                d, cfg, old, net, cal, force, field, restored = E.restore_field(U, seed, arm, entry['path'])
                if restored['step'] != ck['step']:
                    raise RuntimeError('Restored checkpoint identity differs')
                same_state(field.state_dict(), ck['field_state'], 'replayed continuous field')
                a = field.states().detach().cpu()
                am, dm = field.coefficients_and_dt((torch.arange(95, dtype=a.dtype)+.5)*.001)
                am, dm = am.detach(), dm.detach()
                if not all(torch.equal(value, ck[key]) for value, key in ((a, 'a'), (am, 'a_mid'), (dm, 'adot_mid'))):
                    raise RuntimeError('Continuous field/value/time-derivative replay changed')
                same_state(net.state_dict(), model_states[seed][1], 'continuous frozen probability')
                validation = checked_nodes(Q, am, dm, d, net, cal, force, 'V', 20, metrics=False)
                expected_v = ck['val']
                compute_h = lambda degree: checked_nodes(Q, am, dm, d, net, cal, force, 'H', degree)
                query_convention = Q.QUERY_CONVENTION
            else:
                d, cfg, old, net, cal, force = C.E.make_context(U, seed, 'joint')
                same_state(net.state_dict(), model_states[seed][1], 'historical frozen probability')
                old.load_state_dict(ck['model_state'])
                same_state(C.E.closure_state(old), ck['closure_state'], 'historical frozen closure')
                a, es, en = [v.detach().cpu() for v in old()]
                am = dm = None
                replay = dict(a=a, eta_s=es, eta_n=en)
                validation = C.E.score(a, es, en, d, net, cal, force, 'V', 20)
                expected_v = ck['validation_components'] if mode == 'primary' else ck['trace'][-1]['validation']
                compute_h = lambda degree: J.checked_score(C.E, replay, d, net, cal, force, 'H', degree)
                query_convention = 'historical residual-subtracted endpoint query; neural deterministic closure retained'
            if any(parameter.requires_grad for parameter in net.parameters()):
                raise RuntimeError('Probability network became trainable')
            delta = max(abs(validation[k]-expected_v[k]) for k in KEYS)
            if not np.isfinite(delta) or delta > 1e-8:
                raise RuntimeError(f'Frozen V replay mismatch {case}/{arm}/{seed}/{budget}/{mode}: {delta}')
            if a.shape != (96, 2, 624) or not torch.isfinite(a).all():
                raise RuntimeError('Invalid reconstructed field')
            key = dict(manifest_sha256=binding, checkpoint_sha256=entry['sha256'], reference_sha256=rdigest)
            cache = out/'evaluation'/str(budget)/mode/case/arm/f'seed_{seed}.json'
            row = read(cache) if cache.exists() else None
            if row is None or row.get('cache_key') != key:
                h20, h28 = compute_h(20), compute_h(28)
                row = dict(case=case, U=U, arm=arm, seed=seed, budget=budget, mode=mode, source=entry['source'],
                    step=entry['step'], iteration=entry['step'], checkpoint_path=entry['path'], checkpoint_sha256=entry['sha256'],
                    wall_seconds=entry['wall_seconds'], cpu_seconds=entry['cpu_seconds'],
                    probability_checkpoint_sha256=model_states[seed][0]['sha256'], target_network_updates=0,
                    reference_path=str(rp.resolve()), reference_sha256=rdigest, replay_V_max_difference=delta, cache_key=key,
                    query_convention=query_convention, field=extended_field_metrics(J, a, reference, U),
                    observations=dict(V=validation, H=h20), heldout_degree28=h28, quadrature=J.quadrature_difference(h20, h28))
                row['pde_collocation_audit'] = pde_collocation_audit(C, field, old) if entry['source'] == 'new' else None
                dump(cache, row)
            rows.append(row)
            if seed == SEEDS[0]:
                artifacts.append(export_field(J, row, a, reference, d['c'], out, binding, am=am, dm=dm))
            print('EVALUATED', case, arm, seed, budget, mode, 'step', entry['step'], flush=True)
        x, refv, means = J.resolve_fields(reference, d['c'])
        rpout = out/'fields'/case/'reference_K12.npz'; rpout.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(rpout, time_s=np.arange(5, 101)*.001, relative_time_s=np.arange(96)*.001,
            x_m=x, y_m=x, a=reference.numpy(), velocity_m_s=refv, means_m_s=means, reference_sha256=rdigest)
        artifacts.append(dict(path=str(rpout.resolve()), sha256=sha(rpout), kind='reference_npz', frames=96, case=case))
    if len(rows) != 60 or len(artifacts) != 22:
        raise RuntimeError('Expected sixty result rows and twenty-two field NPZs')
    summary, directions, primary_terminal, budget_changes = summarize(rows)
    auxiliary = attribution(C, J, Q, rows, out, binding)
    costs = []
    for entry in [r for r in entries if r['source'] == 'new' and r['budget'] == 600 and r['mode'] == 'terminal']:
        path = Path(entry['path']).parent/'status.json'
        costs.append(dict(case=entry['case'], seed=entry['seed'], arm=entry['arm'], status_path=str(path),
            status_sha256=sha(path), status=read(path)))
    result = dict(version=VERSION, complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.perf_counter()-started, protocol_fingerprint=p['fingerprint'], manifest_sha256=binding,
        probability_gate_sha256=sha(out/'probability_selection/selection_frozen.json'), reference_sources=references,
        target_H_or_reference_before_global_gate=False, selected_or_trained_during_evaluation=False,
        runs=rows, summary=summary, quadrature_directions=directions, terminal_minus_primary=primary_terminal,
        budget600_minus200=budget_changes, auxiliary=auxiliary, cost=dict(field_runs=costs, new_probability_training_runs=0),
        prediction_scope='Causal one-step same-window endpoint prediction; previous observed endpoints allowed; score before ADF update')
    dump(out/'tables/individual_runs.json', rows)
    dump(out/'AUXILIARY_QUERY_ATTRIBUTION.json', auxiliary)
    dump(out/'FIELD_MANIFEST.json', dict(complete=True, evaluation_manifest_sha256=binding,
        representative_seed=SEEDS[0], artifacts=artifacts))
    dump(out/'PINN_EVALUATION_SUMMARY.json', result)
    text = report_text(result)
    (out/'PINN_EVALUATION_REPORT.txt').write_text(text, encoding='utf8')
    if not (out/'CONCLUSIONS.txt').exists():
        (out/'CONCLUSIONS.txt').write_text(text, encoding='utf8')
    print('EVALUATION COMPLETE', out/'PINN_EVALUATION_SUMMARY.json', flush=True)
    return result


def selftest():
    """Pure manufactured tensors only; no target/reference files are opened."""
    J = module('pinn_evaluation_selftest_helpers', HELPER)
    a = torch.ones(96, 2, 624, dtype=torch.float64)
    truth = torch.zeros_like(a)
    f = extended_field_metrics(J, a, truth, .004)
    expected = float(np.sqrt(312)*4)
    assert max(abs(x-expected) for x in f['rms_mm_s']) < 1e-12
    assert f['initial_5ms_window_rms_mm_s'] == f['rms_mm_s']
    h = dict(total=1., free=1., cap=1., coverage=dict(free=.9, cap=.9), spread=dict(free=2., cap=3.))
    rows = []
    for case, U in CASES:
        for seed in SEEDS:
            for mode in MODES:
                for arm, budgets in [(v, BUDGETS) for v in ARMS]+[(HISTORICAL, (200,))]:
                    for budget in budgets:
                        rows.append(dict(case=case, U=U, seed=seed, mode=mode, arm=arm, budget=budget,
                            field=f, observations=dict(H=h), heldout_degree28=h))
    summary, directions, changes, budget_changes = summarize(rows)
    assert len(rows) == 60 and all(not d['sign_reversal'] for d in directions)
    assert len(changes) == 8 and len(budget_changes) == 8
    assert all(v['mean'] == 0 for r in changes+budget_changes for v in r['metrics'].values())
    class ManufacturedField:
        def coefficients_and_dt(self, times):
            aa = torch.sin(times[:, None, None]*20).expand(-1, 2, 624)
            dd = (20*torch.cos(times[:, None, None]*20)).expand_as(aa)
            return aa, dd
    old = SimpleNamespace(theta=torch.zeros(2, 624, dtype=torch.float64),
        op=SimpleNamespace(nu_s=0., friction_factor=1., U=.004),
        prior=SimpleNamespace(whiten=lambda x: x*3), std=torch.ones(2, 624, dtype=torch.float64))
    fake = SimpleNamespace(CURRENT=SimpleNamespace(batched_rhs=lambda op, a: 2*a))
    audit = pde_collocation_audit(fake, ManufacturedField(), old)
    assert audit['independent_points'] == 190 and audit['trained_unique_points'] == 191
    assert abs(audit['trained_simpson']['normal_B_energy']/audit['trained_simpson']['super_std_energy']-9) < 1e-12
    return dict(passed=True, manufactured_rows=60, manufactured_rms_mm_s=expected,
                no_target_or_reference_files_opened=True, summary_and_pairing_passed=True,
                independent_PDE_quarter_point_audit_passed=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--check-gate-only', action='store_true')
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        print(json.dumps(selftest(), ensure_ascii=False, indent=2))
    elif args.check_gate_only:
        cheap_gate(args.out)
        print('All 48 new snapshot metadata/hashes verified; no target tensors opened')
    else:
        evaluate(args.out)
