"""Frozen pretrained transition models followed by PID-isolated field inversion.

Targets are never used to retrain the probability model. Historical experiment
implementations are read-only dependencies with their original hash contract.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '2')
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
OUT = BASE / 'outputs/SharedTransition_20261104'
PARENT_SOURCE = HERE.parent / 'joint_probability_20261103'
sys.path.insert(0, str(PARENT_SOURCE))
spec = importlib.util.spec_from_file_location('shared_transition_parent', PARENT_SOURCE/'experiment.py')
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
sys.path.insert(0, str(HERE))
import transition_model as TM
M, EW = P.M, P.EW
load = P.load
CASES = (.0015, .004)
ARMS = ('cap_retrain', 'independent', 'shared_gru')
SEEDS = (301030, 301031, 301032)
TRAIN_SEEDS = (411040, 411041, 411042)
STEPS = 200
VALIDATE_EVERY = 10
case_name = P.case_name
group_mask, weights = P.group_mask, P.weights
sha, dump, fingerprint = P.sha, P.dump, P.fingerprint


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temp)
    temp.replace(path)


def training_gate():
    import training as T
    training_protocol = T.verify_protocol()
    path = OUT/'training/selection_frozen.json'
    gate = read(path)
    body = {k: v for k, v in gate.items() if k != 'fingerprint'}
    if fingerprint(body) != gate.get('fingerprint') or gate.get('protocol_fingerprint') != training_protocol['fingerprint']:
        raise RuntimeError('Network gate fingerprint/protocol mismatch')
    if not gate.get('all_required_selections_frozen'):
        raise RuntimeError('All nine network selections must be frozen before inversion')
    expected = {(arm, seed) for arm in ARMS for seed in TRAIN_SEEDS}
    records = gate['runs']
    if len(records) != 9 or {(r['arm'], int(r['seed'])) for r in records} != expected:
        raise RuntimeError('Incomplete/duplicate network selection gate')
    for r in records:
        target = OUT/'training'/r['arm']/f"seed_{r['seed']}"/'selected.pt'
        if Path(r['path']).resolve() != target.resolve() or sha(target) != r['sha256']:
            raise RuntimeError('Network selected checkpoint changed: '+str(target))
    return gate


def trained_record(arm, seed):
    ts = TRAIN_SEEDS[SEEDS.index(seed)]
    gate = read(OUT/'training/selection_frozen.json')
    record = next(r for r in gate['runs'] if r['arm'] == arm and int(r['seed']) == ts)
    if sha(record['path']) != record['sha256']:
        raise RuntimeError('Training checkpoint hash mismatch')
    return ts, record


def make_context(U, seed, arm, *, initial_network=False):
    if arm not in ARMS:
        raise ValueError(arm)
    d, cfg, model, base, cal, force = P.make_context(U, seed, 'new_frozen')
    state = None
    if not initial_network:
        training_seed, record = trained_record(arm, seed)
        ck = load(record['path'])
        assert ck['arm'] == arm and ck['seed'] == training_seed
        assert ck['protocol_fingerprint'] == read(OUT/'training/protocol.json')['fingerprint']
        state = ck['state']
    net = TM.load_model(arm, base, state=state, device='cpu').double()
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return d, cfg, model, net, cal, force


def score(a, eta_s, eta_n, d, net, cal, force, group, degree=20, metrics=False, backward=False, batch=256):
    z, c = d['z'], d['c']
    w = weights(z, group)
    result = {'total': 0.}
    if metrics:
        result.update(coverage={}, spread={}, per_pid={})
    for kind, state in (('free', 0), ('cap', 1)):
        ids = torch.where((w > 0) & (z['state'] == state))[0]
        value, mass, parts = 0., 0., {}
        for ix in ids.split(batch):
            zz = EW.subset(z, ix, a.device)
            v, g, acc = M.R.local_jets(a, eta_s, eta_n, zz, c)
            pred, cov, lw = TM.distribution(c, zz, v, g, acc, kind, net, cal, force, degree)
            # The legacy fast captured score assumes covariance is identical
            # across rows. Learned free covariance is observation-dependent.
            nll = (TM.scores if kind == 'free' else EW.fast_scores)(pred, cov, lw, zz['target'])
            ww = w[ix].to(a.device)
            loss = (ww*nll).sum()
            if backward:
                (.08*loss).backward()
            value += float(loss.detach())
            mass += float(ww.sum())
            if metrics:
                cover = (EW.coverage(pred, cov, lw, zz['target']) <= 4.605170186).double()
                prob = lw.exp()
                mean = (prob[..., None]*pred).sum((1, 2))
                trace = cov.diagonal(dim1=-2, dim2=-1).sum(-1)
                spread = (prob*(trace+(pred-mean[:, None, None]).square().sum(-1))).sum((1, 2))
                spread = spread.clamp_min(0).sqrt()*c['physics']['vn_mean_m_s']*.001*1e6
                for pid in zz['pid'].unique().tolist():
                    use = zz['pid'] == pid
                    row = parts.setdefault(str(pid), dict(count=0, nll=0., coverage=0., spread=0.))
                    row['count'] += int(use.sum())
                    for key, tensor in [('nll', nll), ('coverage', cover), ('spread', spread)]:
                        row[key] += float(tensor[use].detach().sum())
        result[kind] = value/mass
        result['total'] += value
        if metrics:
            for row in parts.values():
                for key in ('nll', 'coverage', 'spread'):
                    row[key] /= row['count']
            result['per_pid'][kind] = parts
            for key in ('coverage', 'spread'):
                result[key][kind] = float(np.mean([r[key] for r in parts.values()]))
    return result


def plain_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def field_step(model, optimizer, d, cfg, net, cal, force):
    if any(p.requires_grad for p in net.parameters()):
        raise RuntimeError('The probability network must remain frozen throughout target inversion')
    optimizer.zero_grad(set_to_none=True)
    a, es, en = model()
    if not all(torch.isfinite(v).all() for v in (a, es, en)):
        raise FloatingPointError('Nonfinite reconstruction')
    leaf, sl, nl = [v.detach().requires_grad_() for v in (a, es, en)]
    fit = score(leaf, sl, nl, d, net, cal, force, 'I', degree=12, backward=True)
    reg, detail = M.regularization(model, a, es, en, cfg, int(group_mask(d['z']['pid'], 'I').sum()))
    gradients = [v.grad if v.grad is not None else torch.zeros_like(v) for v in (leaf, sl, nl)]
    torch.autograd.backward([a, es, en, reg], [*gradients, torch.ones_like(reg)])
    norms = {g['name']: float(torch.nn.utils.clip_grad_norm_(g['params'], 5., error_if_nonfinite=True))
             for g in optimizer.param_groups}
    optimizer.step()
    return dict(inversion=fit, objective=.08*fit['total']+float(reg.detach()),
                regularization={k: float(v.detach()) for k, v in detail.items()}, gradient_norms=norms)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'protocol.json').exists():
        return verify_protocol()
    old = P.verify_protocol()
    tg = training_gate()
    sources = dict(old['sources'])
    for name in ('experiment.py', 'transition_model.py', 'training.py', 'evaluate_transition.py'):
        path = HERE/name
        sources[str(path)] = sha(path)
    helper = PARENT_SOURCE/'evaluate_joint.py'
    sources[str(helper)] = sha(helper)
    inputs = dict(old['inputs'])
    for path in (OUT/'training/protocol.json', OUT/'training/selection_frozen.json'):
        inputs[str(path)] = sha(path)
    for row in tg['runs']:
        inputs[str(Path(row['path']))] = row['sha256']
    for name in ('protocol.json', 'selection_frozen.json', 'JOINT_EVALUATION_SUMMARY.json',
                 'tables/individual_runs.json', 'secondary_terminal_manifest.json', 'secondary_terminal.json'):
        path = P.OUT/name
        inputs[str(path)] = sha(path)
    protocol = dict(version='shared-transition-retraining-inversion-v1',
        created_utc=datetime.now(timezone.utc).isoformat(), cases=list(CASES), arms=list(ARMS),
        seeds=list(SEEDS), training_seeds=list(TRAIN_SEEDS), steps=STEPS, validation_every=VALIDATE_EVERY,
        field_lr=dict(theta=.0003, super=.009, normal=.009), train_degree=12,
        evaluation_degree=20, evaluation_sensitivity_degree=28, guard=.01, selection_delta=1e-6,
        new_physics=dict(nu_s=0., Fns_factor=1.), process_prior='Frozen B under legacy RHS, unchanged',
        split='I pid%5>=3; V pid%5==1; H pid%5==0; F pid%5==2 unused in this target experiment',
        target_network_updates=0, target_truth_used_in_fit=False, target_H_used_for_selection=False,
        training='Seven non-target cases; all network weights retrained from common pretrained checkpoint; frozen before target inversion',
        validation='6.5mm/s held-out PIDs select networks; target V selects paired field checkpoint',
        initialization=old['initialization'], matched_legacy='Reuse six old experiment new_frozen runs under exactly the same I split, initial fields, steps and physics',
        secondary_terminal='Prospectively fixed all eighteen step200 terminal states; never replace validation-selected results',
        interpretation='Development comparison; seeds pair network shuffle with initial field perturbations; same physical data, no independent-replicate claim',
        historical_target_truth_and_test_seen=True, confirmatory=False, raw_simulation_regenerated=False,
        scopes=old['scopes'], sources=sources, inputs=inputs, threads_per_process=2)
    protocol['fingerprint'] = fingerprint(protocol)
    dump(OUT/'protocol.json', protocol)
    print(json.dumps(dict(prepared=True, runs=18, fingerprint=protocol['fingerprint'])), flush=True)
    return protocol


def verify_protocol():
    p = read(OUT/'protocol.json')
    fp = p.pop('fingerprint')
    assert fingerprint(p) == fp
    p['fingerprint'] = fp
    assert p['steps'] == STEPS and tuple(p['arms']) == ARMS and tuple(p['seeds']) == SEEDS
    for group in ('sources', 'inputs'):
        for path, expected in p[group].items():
            if sha(path) != expected:
                raise RuntimeError('Frozen experiment dependency changed: '+path)
    return p


@torch.no_grad()
def snapshot(model, net, d, cal, force, U, seed, arm, iteration, protocol, training_seed, training_sha):
    a, es, en = model()
    val = score(a, es, en, d, net, cal, force, 'V', 20)
    return dict(a=a.clone(), eta_s=es.clone(), eta_n=en.clone(), theta=a[0].clone(),
                model_state=plain_state(model), net_state=plain_state(net), U=U, seed=seed, arm=arm,
                iteration=iteration, validation_components=val, fingerprint=protocol['fingerprint'],
                training_seed=training_seed, training_sha=training_sha)


def run(U, seed, arm, max_new_steps=None):
    p = verify_protocol()
    training_seed, record = trained_record(arm, seed)
    d, cfg, model, net, cal, force = make_context(U, seed, arm)
    before_net = plain_state(net)
    dest = OUT/'runs'/case_name(U)/f'seed_{seed}'/arm
    dest.mkdir(parents=True, exist_ok=True)
    groups = model.parameter_groups()
    for g in groups:
        g['lr'] = p['field_lr'][g['name']]
    opt = torch.optim.Adam(groups)
    cpu = wall = 0.
    iteration = 0
    snap = lambda step: snapshot(model, net, d, cal, force, U, seed, arm, step, p, training_seed, record['sha256'])
    if (dest/'last.pt').exists():
        ck = load(dest/'last.pt')
        assert ck['fingerprint'] == p['fingerprint'] and ck['training_sha'] == record['sha256']
        model.load_state_dict(ck['model_state'])
        opt.load_state_dict(ck['optimizer'])
        iteration, trace, selected, warm, cpu, wall = [ck[k] for k in
            ('iteration', 'trace', 'selected', 'warm_validation', 'cpu_seconds', 'wall_seconds')]
    else:
        t, w = time.process_time(), time.perf_counter()
        selected = snap(0)
        warm = selected['validation_components']
        cpu += time.process_time()-t
        wall += time.perf_counter()-w
        trace = [dict(iteration=0, validation=warm, selected=True)]
    start = iteration
    def persist():
        if not all(torch.equal(value, before_net[key]) for key, value in net.state_dict().items()):
            raise AssertionError('Probability network changed during field inversion')
        save(dest/'selected.pt', selected)
        save(dest/'last.pt', dict(fingerprint=p['fingerprint'], model_state=plain_state(model),
            net_state=before_net, optimizer=opt.state_dict(), iteration=iteration, trace=trace, selected=selected,
            warm_validation=warm, cpu_seconds=cpu, wall_seconds=wall, training_seed=training_seed,
            training_sha=record['sha256'], U=U, seed=seed, arm=arm, network_steps=0))
        dump(dest/'history.json', trace)
        dump(dest/'status.json', dict(complete=iteration==STEPS, iteration=iteration, steps=STEPS,
            selected_iteration=selected['iteration'], validation=selected['validation_components'],
            network_steps=0, cpu_seconds=cpu, wall_seconds=wall, case=case_name(U), arm=arm, seed=seed,
            training_seed=training_seed, training_sha=record['sha256'], fingerprint=p['fingerprint']))
    persist()
    while iteration < STEPS:
        t, w = time.process_time(), time.perf_counter()
        row = field_step(model, opt, d, cfg, net, cal, force)
        iteration += 1
        if iteration % VALIDATE_EVERY == 0 or iteration == STEPS:
            candidate = snap(iteration)
            val = candidate['validation_components']
            eligible = all(np.isfinite(val[k]) for k in ('free', 'cap', 'total')) and all(val[k] <= warm[k]+.01 for k in ('free', 'cap'))
            chosen = eligible and val['total'] < selected['validation_components']['total']-1e-6
            if chosen:
                selected = candidate
            row.update(validation=val, eligible=eligible, selected=chosen, selected_iteration=selected['iteration'])
            print(case_name(U), seed, arm, 'step', iteration, 'V', round(val['total'], 6), 'selected', selected['iteration'], flush=True)
        cpu += time.process_time()-t
        wall += time.perf_counter()-w
        row.update(iteration=iteration, cpu_seconds=cpu, wall_seconds=wall)
        trace.append(row)
        if iteration % VALIDATE_EVERY == 0 or (max_new_steps is not None and iteration-start >= max_new_steps):
            persist()
        if max_new_steps is not None and iteration-start >= max_new_steps:
            break
    return selected


@torch.no_grad()
def freeze():
    p = verify_protocol()
    training_gate()
    runs = []
    for U in CASES:
        for seed in SEEDS:
            for arm in ARMS:
                folder = OUT/'runs'/case_name(U)/f'seed_{seed}'/arm
                status, last, selected = read(folder/'status.json'), load(folder/'last.pt'), load(folder/'selected.pt')
                assert status['complete'] and status['iteration'] == last['iteration'] == STEPS
                assert status['fingerprint'] == last['fingerprint'] == p['fingerprint']
                assert last['network_steps'] == 0 and selected['fingerprint'] == p['fingerprint']
                best, warm = last['trace'][0], last['warm_validation']
                for row in last['trace'][1:]:
                    if 'validation' not in row:
                        continue
                    v = row['validation']
                    eligible = all(np.isfinite(v[k]) for k in ('free', 'cap', 'total')) and all(v[k] <= warm[k]+.01 for k in ('free', 'cap'))
                    if eligible and v['total'] < best['validation']['total']-1e-6:
                        best = row
                assert selected['iteration'] == best['iteration'] == status['selected_iteration']
                ts, training = trained_record(arm, seed)
                assert selected['training_seed'] == ts and selected['training_sha'] == training['sha256']
                d, cfg, model, net, cal, force = make_context(U, seed, arm)
                for ck in (selected, last):
                    assert all(torch.equal(net.state_dict()[k], v) for k, v in ck['net_state'].items())
                    model.load_state_dict(ck['model_state'])
                    a, es, en = model()
                    expected_v = ck['validation_components'] if ck is selected else ck['trace'][-1]['validation']
                    val = score(a, es, en, d, net, cal, force, 'V', 20)
                    assert max(abs(val[k]-expected_v[k]) for k in val) < 1e-10
                    if ck is selected:
                        assert all(torch.equal(x, y) for x, y in zip((a, es, en), (ck['a'], ck['eta_s'], ck['eta_n'])))
                runs.append(dict(case=case_name(U), U=U, seed=seed, arm=arm,
                    path=str(folder/'selected.pt'), sha256=sha(folder/'selected.pt'),
                    last_path=str(folder/'last.pt'), last_sha256=sha(folder/'last.pt'),
                    selected_iteration=selected['iteration'], training_seed=ts, training_sha=training['sha256']))
    result = dict(all_required_selections_frozen=True, runs=runs, fingerprint=p['fingerprint'],
        all_terminal_states_bound_before_evaluation=True, target_H_used_for_selection=False, target_truth_used_for_selection=False,
        training_gate_sha256=sha(OUT/'training/selection_frozen.json'))
    dump(OUT/'selection_frozen.json', result)
    print(json.dumps(dict(frozen=True, runs=len(runs), terminal_bound=len(runs))), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'freeze'])
    parser.add_argument('--U', type=float, default=.004)
    parser.add_argument('--seed', type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument('--arm', choices=ARMS, default='shared_gru')
    parser.add_argument('--max-new-steps', type=int)
    args = parser.parse_args()
    M.R.setup('cpu')
    if args.action == 'prepare':
        prepare()
    elif args.action == 'freeze':
        freeze()
    else:
        run(args.U, args.seed, args.arm, args.max_new_steps)
