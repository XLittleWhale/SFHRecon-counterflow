"""Prospective non-target closure / observation / joint meta-training.

C/D differentiate through two gradient steps optimizing a cold short-window
initial field against inner particles. Disjoint outer particles and non-target
field truth supervise the result. This is truncated optimization-aware training,
not a claim of converged bilevel inference. Formal fitting requires a frozen
parent protocol; the pilot never writes production model selections.
"""
from __future__ import annotations
import argparse
import copy
import math
import random
import time
import numpy as np
import torch
from common import *
from closure_model import ClosureNet, ClosureMomentum
import data_training as D
import probability as P

DEFAULT_SETTINGS = dict(epochs=40, minimum_epochs=20, patience=10, windows_per_case=2,
    window_steps=5, inner_steps=2, inner_lr=.05, inner_gradient_clip=5.,
    lr_closure=1e-4, lr_probability=1e-5, weight_decay=0., gradient_clip=5.,
    train_degree=6, validation_degree=12, validation_starts=[0, 30, 60, 90],
    minimum_state_pids=2, field_weight=1., initial_weight=.5,
    endpoint_weight=.08, teacher_weight=1., anchor_weight=.05,
    selection_min_delta=1e-5, substeps=2, dtype='float64', num_threads=2,
    candidate_full_window_check=True, full_window_amplitude_limit=1e6, full_window_growth_factor=4.)
SETTINGS = DEFAULT_SETTINGS
CLOSURE_SETTINGS = dict(width=16, coefficient_bound=1.0)


def validate_settings(s):
    if set(s) != set(DEFAULT_SETTINGS):
        raise ValueError('Training settings must explicitly match the prospective schema')
    if not (1 <= s['minimum_epochs'] <= s['epochs'] and s['patience'] >= 1):
        raise ValueError('Invalid fixed epoch budget')
    if s['window_steps'] != 5 or s['inner_steps'] != 2 or s['windows_per_case'] != 2:
        raise ValueError('First-round design requires 5 ms / 2 inner steps / 2 windows per case')
    if s['dtype'] != 'float64' or s['num_threads'] != 2 or not s['candidate_full_window_check']:
        raise ValueError('Require double precision, two threads, full-window selection stability')
    if s['minimum_state_pids'] < 2:
        raise ValueError('At least two physical PIDs in every scored state/split')


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())


def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'])


def initial_scale(like):
    modes, _ = P.core.modes(12)
    radius = np.max(np.abs(modes), axis=1).repeat(2)
    # Fixed spectral preconditioner, independent of truth and data.
    return 1/(1+(like.new_tensor(radius)/4)**2)


def field_errors(prediction, truth):
    energy = truth.square().sum(-1).mean(0).clamp_min(1e-12)
    per_fluid = (prediction-truth).square().sum(-1).mean(0)/energy
    initial = (prediction[0]-truth[0]).square().sum(-1)/truth[0].square().sum(-1).clamp_min(1e-12)
    return dict(field=per_fluid.mean(), normal=per_fluid[0], super=per_fluid[1],
                initial=initial.mean(), initial_normal=initial[0], initial_super=initial[1],
                whole_normal=per_fluid[0], whole_super=per_fluid[1])


def likelihood(w, a, net, mask, degree):
    return P.window_nll(w['c'], w['z'], D.query(w, a), net, w['cal'], w['force'],
                        mask, degree, w['length'])


def infer_initial(w, op, net, settings, *, create_graph):
    """Particle-only differentiable inner loop; truth never enters its updates."""
    x = torch.zeros_like(w['truth'][0], requires_grad=True)
    scale = initial_scale(x)
    history = []
    degree = settings['train_degree'] if create_graph else settings['validation_degree']
    for _ in range(settings['inner_steps']):
        a = D.trajectory(op, x*scale, w['length'], settings['substeps'])
        score = likelihood(w, a, net, w['masks']['inner'], degree)
        prior = P.core.penalty(a, P.core.config(), 1.)
        objective = settings['endpoint_weight']*score['total']+prior
        gradient = torch.autograd.grad(objective, x, create_graph=create_graph)[0]
        history.append(dict(objective=float(objective.detach()), nll=float(score['total'].detach()),
                            gradient_norm=float(gradient.detach().norm())))
        norm = gradient.norm().clamp_min(settings['inner_gradient_clip'])
        x = x-settings['inner_lr']*gradient*(settings['inner_gradient_clip']/norm)
        if not create_graph:
            x = x.detach().requires_grad_()
    a = D.trajectory(op, x*scale, w['length'], settings['substeps'])
    final = likelihood(w, a, net, w['masks']['inner'], degree)
    objective = settings['endpoint_weight']*final['total']+P.core.penalty(a, P.core.config(), 1.)
    history.append(dict(objective=float(objective.detach()), nll=float(final['total'].detach())))
    return a, history


def teacher_loss(w, op, settings):
    pred = D.trajectory(op, w['truth'][0], w['length'], settings['substeps'])
    # Normalize by the true finite-window change, separately for both fluids.
    # This targets dynamics, rather than the much larger unchanged initial field.
    scale = (w['truth'][1:]-w['truth'][0]).square().sum(-1).mean(0)
    floor = 1e-6*w['truth'].square().sum(-1).mean(0)+1e-12
    components = (pred[1:]-w['truth'][1:]).square().sum(-1).mean(0)/torch.maximum(scale, floor)
    return components.mean(), components


def anchors(w, net, settings):
    # The labels AND context are fixed under the current pure physical operator.
    # They deliberately cannot move with the learned closure to reduce the loss.
    with torch.no_grad():
        residual = D.reference_residual(w['op'], w['truth'], settings['substeps'])
        jets = tuple(x.detach() for x in D.query(w, w['truth'], residual))
        force_labels = D.relabel_force(w, jets).detach()
    return P.supervised_anchors(w['c'], w['z'], jets, w['labels'], force_labels, net,
        w['cal'], w['force'], w['masks']['permitted'], w['length'])


def train_window(w, op, net, arm, settings):
    teacher, teacher_components = teacher_loss(w, op, settings)
    if arm == 'closure':
        return settings['teacher_weight']*teacher, dict(teacher=teacher, teacher_normal=teacher_components[0],
            teacher_super=teacher_components[1])
    a, inner = infer_initial(w, op, net, settings, create_graph=True)
    errors = field_errors(a, w['truth'])
    endpoint = likelihood(w, a, net, w['masks']['outer'], settings['train_degree'])
    anchor = anchors(w, net, settings)
    loss = (settings['field_weight']*errors['field']+settings['initial_weight']*errors['initial']
            +settings['endpoint_weight']*endpoint['total']+settings['anchor_weight']*anchor['total'])
    if arm == 'joint':
        loss = loss+settings['teacher_weight']*teacher
    metrics = dict(**errors, free=endpoint['free'], cap=endpoint['cap'], anchor=anchor['total'],
                   anchor_free=anchor['free'], anchor_cap=anchor['cap'], anchor_pair=anchor['pair'],
                   teacher=teacher, teacher_normal=teacher_components[0], teacher_super=teacher_components[1],
                   inner_before=inner[0]['objective'], inner_after=inner[-1]['objective'])
    return loss, metrics


def scalar_metrics(values):
    return {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in values.items()}


def validate(case, op, net, settings):
    values = []
    for start in settings['validation_starts']:
        w = D.window(case, start, settings['window_steps'], settings['minimum_state_pids'])
        with torch.enable_grad():
            a, inner = infer_initial(w, op, net, settings, create_graph=False)
        with torch.no_grad():
            errors = field_errors(a.detach(), w['truth'])
            endpoint = likelihood(w, a.detach(), net, w['masks']['outer'], settings['validation_degree'])
            teacher, components = teacher_loss(w, op, settings)
            values.append(scalar_metrics(dict(**errors, free=endpoint['free'], cap=endpoint['cap'],
                teacher=teacher, teacher_normal=components[0], teacher_super=components[1],
                inner_before=inner[0]['objective'], inner_after=inner[-1]['objective'])))
    result = {key: float(np.mean([item[key] for item in values])) for key in values[0]}
    result['windows'] = [dict(start=start, **value) for start, value in zip(settings['validation_starts'], values)]
    return result


def composite(metrics, origin):
    if not all(math.isfinite(metrics[k]) for k in ('field', 'free', 'cap')):
        return float('inf')
    try:
        score = (metrics['field']/max(origin['field'], 1e-12)
                 +math.exp(metrics['free']-origin['free'])+math.exp(metrics['cap']-origin['cap']))/3
    except OverflowError:
        return float('inf')
    return score if math.isfinite(score) else float('inf')


@torch.no_grad()
def _full_rollout_stats(case, op, settings):
    a = case['a'][0]
    initial = float(a.square().sum(-1).mean().sqrt())
    maximum = float(a.abs().max())
    peak = a.square().sum(-1).sqrt()
    for frame in range(95):
        a = D.batched_step(op, a, settings['substeps'])
        if not bool(torch.isfinite(a).all()):
            return dict(passed=False, failed_frame=frame+1, reason='nonfinite',
                        case=case['case'], initial_modal_rms=initial)
        maximum = max(maximum, float(a.abs().max()))
        peak = torch.maximum(peak, a.square().sum(-1).sqrt())
        if maximum > settings['full_window_amplitude_limit']:
            return dict(passed=False, failed_frame=frame+1, reason='fixed_amplitude_limit',
                        max_abs_coefficient=maximum, case=case['case'])
    return dict(passed=True, frames=96, case=case['case'], max_abs_coefficient=maximum,
                initial_modal_rms=initial, final_modal_rms=float(a.square().sum(-1).mean().sqrt()),
                peak_modal_rms_per_fluid=peak.tolist(),
                initial_modal_rms_per_fluid=case['a'][0].square().sum(-1).sqrt().tolist(),
                truth_used_only_for_start=True, process_increments_zero=True)


@torch.no_grad()
def full_window_check(case, op, settings):
    key = (settings['substeps'], settings['full_window_amplitude_limit'])
    cache = case.setdefault('_pure_physics_full_window', {})
    if key not in cache:
        cache[key] = _full_rollout_stats(case, case['op'], settings)
    baseline = cache[key]
    if not baseline['passed']:
        raise RuntimeError('Non-target pure-physics full-window baseline failed: '+str(baseline))
    result = copy.deepcopy(baseline) if op is case['op'] else _full_rollout_stats(case, op, settings)
    result['baseline_peak_modal_rms_per_fluid'] = baseline['peak_modal_rms_per_fluid']
    result['growth_factor_limit'] = settings['full_window_growth_factor']
    if result['passed']:
        denominator = np.maximum(baseline['peak_modal_rms_per_fluid'], baseline['initial_modal_rms_per_fluid'])
        ratio = np.asarray(result['peak_modal_rms_per_fluid'])/np.maximum(denominator, 1e-12)
        result['peak_growth_ratio_per_fluid'] = ratio.tolist()
        if bool((ratio > settings['full_window_growth_factor']).any()):
            result.update(passed=False, reason='relative_per_fluid_growth_limit')
    return result


def build(arm, seed, closure_settings, *, protocol=None, pilot=False):
    net, origin = baseline_network(seed, trainable=arm in ('probability', 'joint'))
    closure = ClosureNet(**closure_settings).double() if arm in ('closure', 'joint') else None
    if arm == 'joint' and not pilot:
        path = OUT/'training/closure'/f'seed_{seed}'/'selected.pt'
        status = read(path.parent/'status.json')
        ck = load(path)
        if not status['complete'] or ck['protocol_fingerprint'] != protocol['fingerprint']:
            raise RuntimeError('Joint training requires complete same-protocol selected B')
        closure.load_state_dict(ck['closure_state'])
        origin['closure_start'] = dict(path=str(path), sha256=sha(path), epoch=ck['epoch'])
    return net, closure, origin


def optimizer_for(net, closure, settings):
    groups = []
    if closure is not None:
        groups.append(dict(params=list(closure.parameters()), lr=settings['lr_closure'], name='closure'))
    probability_parameters = [v for v in net.parameters() if v.requires_grad]
    if probability_parameters:
        groups.append(dict(params=probability_parameters, lr=settings['lr_probability'], name='probability'))
    return torch.optim.Adam(groups, weight_decay=settings['weight_decay'])


def checkpoint(net, closure, epoch, arm, seed, p, validation, score, source, stability):
    return dict(net_state=plain_state(net), closure_state={} if closure is None else plain_state(closure),
        epoch=epoch, arm=arm, seed=seed, protocol_fingerprint=p['fingerprint'], fingerprint=p['fingerprint'],
        validation=copy.deepcopy(validation), selection_score=score, origin=copy.deepcopy(source),
        stability=copy.deepcopy(stability), rng_state=rng_state())


def run(arm, seed, max_new_epochs=None):
    p = verify_protocol()
    settings = p['training_settings']; validate_settings(settings)
    folder = OUT/'training'/arm/f'seed_{seed}'; folder.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    net, closure, source = build(arm, seed, p['closure_settings'], protocol=p)
    opt = optimizer_for(net, closure, settings)
    cases = [D.load_case(name) for name in D.TRAIN_CASES]
    validation_case = D.load_case(D.VALIDATION_CASE)
    valop = validation_case['op'] if closure is None else ClosureMomentum(validation_case['op'], closure)
    epoch = stale = 0; cpu = wall = 0.; stopped = False
    if (folder/'last.pt').exists():
        old = load(folder/'last.pt')
        assert old['protocol_fingerprint'] == p['fingerprint'] and old['arm'] == arm and old['seed'] == seed
        assert old['origin'] == source, 'Training dependency checkpoint changed since the saved run'
        net.load_state_dict(old['net_state'])
        if closure is not None: closure.load_state_dict(old['closure_state'])
        opt.load_state_dict(old['optimizer']); restore_rng(old['rng_state'])
        epoch, stale, cpu, wall, trace, origin, selected, stopped = [old[k] for k in
            ('epoch', 'stale', 'cpu_seconds', 'wall_seconds', 'trace', 'origin_validation', 'selected', 'stopped_early')]
        if epoch >= settings['epochs'] or stopped:
            status_path = folder/'status.json'
            if status_path.exists() and (folder/'selected.pt').exists():
                status = read(status_path)
                if status.get('complete') and status.get('fingerprint') == p['fingerprint']:
                    return
            if (OUT/'training/selection_frozen.json').exists():
                raise RuntimeError('A frozen completed training artifact is missing or inconsistent')
            # Recovery after an interruption between last.pt and status.json:
            # regenerate the derived status/export, without another optimizer step.
    else:
        c, t = time.process_time(), time.perf_counter()
        origin = validate(validation_case, valop, net, settings)
        stable = full_window_check(validation_case, valop, settings)
        if not stable['passed']:
            raise RuntimeError('Epoch-zero full-window stability failed; report before changing protocol: '+str(stable))
        selected = checkpoint(net, closure, 0, arm, seed, p, origin, composite(origin, origin), source, stable)
        trace = [dict(epoch=0, validation=origin, selection_score=selected['selection_score'], selected=True, stability=stable)]
        cpu += time.process_time()-c; wall += time.perf_counter()-t
    def persist():
        state = checkpoint(net, closure, epoch, arm, seed, p, trace[-1]['validation'], trace[-1]['selection_score'],
                           source, trace[-1].get('stability'))
        state.update(optimizer=opt.state_dict(), trace=trace, selected=selected, origin_validation=origin,
                     stale=stale, cpu_seconds=cpu, wall_seconds=wall, stopped_early=stopped)
        save(folder/'selected.pt', selected); save(folder/'last.pt', state)
        dump(folder/'history.json', trace)
        dump(folder/'status.json', dict(complete=epoch >= settings['epochs'] or stopped, epoch=epoch, arm=arm, seed=seed,
            selected_epoch=selected['epoch'], selection_score=selected['selection_score'], stale=stale,
            cpu_seconds=cpu, wall_seconds=wall, stopped_early=stopped, protocol_fingerprint=p['fingerprint'],
            fingerprint=p['fingerprint'], target_data_used=False))
    persist(); start_epoch = epoch
    while epoch < settings['epochs'] and not stopped:
        c, t = time.process_time(), time.perf_counter()
        jobs = [(case, random.randrange(96-settings['window_steps'])) for case in cases
                for _ in range(settings['windows_per_case'])]
        random.shuffle(jobs); metrics = []
        for case, start in jobs:
            w = D.window(case, start, settings['window_steps'], settings['minimum_state_pids'])
            op = case['op'] if closure is None else ClosureMomentum(case['op'], closure)
            opt.zero_grad(set_to_none=True)
            loss, detail = train_window(w, op, net, arm, settings)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite training objective')
            loss.backward()
            norms = {g['name']: float(torch.nn.utils.clip_grad_norm_(g['params'], settings['gradient_clip'],
                    error_if_nonfinite=True)) for g in opt.param_groups}
            opt.step(); metrics.append(dict(case=case['case'], start=start, loss=float(loss.detach()),
                                           gradients=norms, **scalar_metrics(detail)))
        epoch += 1
        candidate = validate(validation_case, valop, net, settings)
        score = composite(candidate, origin)
        eligible = math.isfinite(score)
        better = eligible and score < selected['selection_score']-settings['selection_min_delta']
        stable = full_window_check(validation_case, valop, settings) if better else None
        chosen = better and stable['passed']
        if chosen:
            selected = checkpoint(net, closure, epoch, arm, seed, p, candidate, score, source, stable); stale = 0
        else: stale += 1
        stopped = epoch >= settings['minimum_epochs'] and stale >= settings['patience']
        cpu += time.process_time()-c; wall += time.perf_counter()-t
        trace.append(dict(epoch=epoch, validation=candidate, selection_score=score, eligible=eligible,
                          selected=chosen, stability=stable, training=metrics, cpu_seconds=cpu, wall_seconds=wall))
        persist()
        print(arm, seed, epoch, 'V', score, 'selected', selected['epoch'], flush=True)
        if max_new_epochs is not None and epoch-start_epoch >= max_new_epochs: break


def freeze():
    p = verify_protocol(); s = p['training_settings']; validate_settings(s)
    validation_case = D.load_case(D.VALIDATION_CASE)
    rows = []
    for arm in NEW_ARMS:
        for seed in TRAIN_SEEDS:
            folder = OUT/'training'/arm/f'seed_{seed}'
            status, last, selected = read(folder/'status.json'), load(folder/'last.pt'), load(folder/'selected.pt')
            assert status['complete'] and (last['epoch'] == s['epochs'] or last['stopped_early'])
            assert last['fingerprint'] == selected['fingerprint'] == status['fingerprint'] == p['fingerprint']
            best = last['trace'][0]
            for row in last['trace'][1:]:
                score = composite(row['validation'], last['origin_validation'])
                assert score == row['selection_score']
                if math.isfinite(score) and score < best['selection_score']-s['selection_min_delta']:
                    assert row['stability'] is not None
                    if row['stability']['passed']: best = row
            assert best['epoch'] == selected['epoch'] == status['selected_epoch']
            assert selected['stability']['passed'] and (selected['closure_state'] != {}) == (arm != 'probability')
            net, closure, source = build(arm, seed, p['closure_settings'], protocol=p)
            assert selected['origin'] == last['origin'] == source
            for key, value in net.named_buffers():
                assert torch.equal(value, selected['net_state'][key]) and torch.equal(value, last['net_state'][key]), 'Probability normalization buffer drift'
            if arm == 'closure':
                assert all(torch.equal(v, selected['net_state'][k]) and torch.equal(v, last['net_state'][k])
                           for k, v in net.state_dict().items())
            net.load_state_dict(selected['net_state'])
            if closure is not None: closure.load_state_dict(selected['closure_state'])
            valop = validation_case['op'] if closure is None else ClosureMomentum(validation_case['op'], closure)
            reproduced = validate(validation_case, valop, net, s)
            for key, value in reproduced.items():
                if key != 'windows': assert abs(value-selected['validation'][key]) < 1e-10
            assert full_window_check(validation_case, valop, s)['passed']
            for key in ('net_state', 'closure_state'):
                assert all(torch.equal(value, last['selected'][key][name]) for name, value in selected[key].items())
            for state in (selected['net_state'], selected['closure_state']):
                assert all(torch.isfinite(v).all() for v in state.values())
            rows.append(dict(arm=arm, seed=seed, path=str(folder/'selected.pt'), sha256=sha(folder/'selected.pt'),
                epoch=selected['epoch'], last_path=str(folder/'last.pt'), last_sha256=sha(folder/'last.pt'),
                status_path=str(folder/'status.json'), status_sha256=sha(folder/'status.json'),
                selection_score=selected['selection_score'], full_window_stable=True))
    gate = dict(all_required_selections_frozen=True, protocol_fingerprint=p['fingerprint'], runs=rows,
                target_data_used=False, anchor_labels_fixed=True)
    gate['fingerprint'] = fingerprint(gate); dump(OUT/'training/selection_frozen.json', gate)
    return gate


def pilot():
    """Discarded pilot using only non-target data, including actual hypergradients."""
    settings = dict(DEFAULT_SETTINGS)
    random.seed(TRAIN_SEEDS[0]); np.random.seed(TRAIN_SEEDS[0]); torch.manual_seed(TRAIN_SEEDS[0])
    net, closure, _ = build('joint', TRAIN_SEEDS[0], dict(width=16, coefficient_bound=1.), pilot=True)
    case = D.load_case(D.TRAIN_CASES[3])
    w = D.window(case, 30, settings['window_steps'], settings['minimum_state_pids'])
    op = ClosureMomentum(case['op'], closure)
    # Compare our trainable local law to the untouched frozen v2 arithmetic.
    with torch.no_grad():
        residual = D.reference_residual(case['op'], w['truth'], settings['substeps'])
        jets = D.query(w, w['truth'], residual)
        ids = torch.where((w['z']['state'] == 1) & (w['z']['frame'] == 0))[0][:12]
        z = P.core.subset(w['z'], ids)
        for parameter in net.parameters(): parameter.requires_grad_(False)
        first_states = [None, None]; contexts = [None, None]; parity = 0.
        for _ in range(2):
            outputs = []
            for i, function in enumerate((P.step, LM.step)):
                values = function(w['c'], z, *(v[ids] for v in jets), net.base, net.head,
                    w['cal'], w['force'], settings['train_degree'], first_states[i], contexts[i])
                first_states[i] = values[3]; contexts[i] = values[3]['context']; outputs.append(values)
            for left, right in zip(outputs[0][:3], outputs[1][:3]):
                parity = max(parity, float((left-right).abs().max()))
            for key in outputs[0][3]: parity = max(parity, float((outputs[0][3][key]-outputs[1][3][key]).abs().max()))
        for parameter in net.parameters(): parameter.requires_grad_(True)
        assert parity == 0.
        old_query = M.R.local_jets(w['truth'], residual[:, 1], residual[:, 0], w['z'], w['c'])
        query_parity = max(float((left-right).abs().max()) for left, right in zip(jets, old_query))
        assert query_parity < 1e-12
    validation = D.load_case(D.VALIDATION_CASE)
    valop = ClosureMomentum(validation['op'], closure)
    origin_validation = validate(validation, valop, net, settings)
    begun = time.perf_counter()
    a, inner = infer_initial(w, op, net, settings, create_graph=True)
    field = field_errors(a, w['truth'])['initial']
    gradients = torch.autograd.grad(field, tuple(net.parameters()), allow_unused=True)
    maxima = [float(v.abs().max()) for v in gradients if v is not None]
    assert maxima and all(math.isfinite(v) for v in maxima) and max(maxima) > 0
    hyper_seconds = time.perf_counter()-begun
    assert inner[-1]['objective'] < inner[0]['objective'], 'Cold-field inner objective failed to decrease'
    opt = optimizer_for(net, closure, settings)
    timing = []
    for step_index in range(2):
        began = time.perf_counter(); opt.zero_grad(set_to_none=True)
        loss, metrics = train_window(w, op, net, 'joint', settings)
        loss.backward()
        norms = {g['name']: float(torch.nn.utils.clip_grad_norm_(g['params'], settings['gradient_clip'],
                 error_if_nonfinite=True)) for g in opt.param_groups}
        opt.step()
        timing.append(dict(step=step_index, seconds=time.perf_counter()-began, loss=float(loss.detach()),
                           gradient_norms=norms, **scalar_metrics(metrics)))
    baseline_stability = full_window_check(validation, validation['op'], settings)
    closed_stability = full_window_check(validation, valop, settings)
    final_validation = validate(validation, valop, net, settings)
    assert baseline_stability['passed'], 'Zero closure is not stable; report before protocol freeze'
    result = dict(passed=True, formal_models_written=False, target_files_opened=False,
        settings=settings, case=case['case'], window_start=30, window_counts=w['counts'],
        inner=inner, initial_field_probability_hypergradient_max=max(maxima), hypergradient_seconds=hyper_seconds,
        updates=timing, pure_physics_96frame=baseline_stability, pilot_closure_96frame=closed_stability,
        probability_local_step_max_difference=parity, observation_query_max_difference=query_parity,
        validation_before=origin_validation, validation_after=final_validation,
        validation_composite_before=composite(origin_validation, origin_validation),
        validation_composite_after=composite(final_validation, origin_validation),
        anchor_context_and_labels_fixed_under_pure_physics=True,
        limitation='Two cold-start inner gradient steps are a truncated sensitivity proxy, not converged inversion.')
    dump(OUT/'preflight/training_preflight.json', result)
    print(result, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['run', 'freeze', 'pilot'])
    parser.add_argument('--arm', choices=NEW_ARMS, default='joint')
    parser.add_argument('--seed', type=int, choices=TRAIN_SEEDS, default=TRAIN_SEEDS[0])
    parser.add_argument('--max-new-epochs', type=int)
    args = parser.parse_args(); setup()
    if args.action == 'run': run(args.arm, args.seed, args.max_new_epochs)
    elif args.action == 'freeze': freeze()
    else: pilot()
