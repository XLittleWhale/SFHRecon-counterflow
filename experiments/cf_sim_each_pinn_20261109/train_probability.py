"""Each-case native endpoint training; F fits, V selects, I/H never load.

All F intervals are visited every epoch. Captured rows remain chronological
within physical PID/state segments, with four-interval truncated backprop and
detached causal posterior carried between chunks. Only observed endpoints
supervise the network; interpolated truth jets are fixed conditioning data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import torch

try:
    from . import native_probability as P
except ImportError:
    import native_probability as P

DEFAULT_SETTINGS = dict(epochs=80, minimum_epochs=40, patience=20, batch=1024,
    batch_pids=4, truncate_intervals=4, lr=1e-4, weight_decay=1e-4,
    gradient_clip=5., train_degree=6, validation_degree=20,
    selection_min_delta=1e-5, device='cpu', threads=2, dtype='float64', smoke=False)


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    os.replace(temp, path)


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def _state(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _utc():
    return datetime.now(timezone.utc).isoformat()


def settings_with_defaults(settings=None):
    result = DEFAULT_SETTINGS | (settings or {})
    unknown = set(result)-set(DEFAULT_SETTINGS)-{'max_new_epochs'}
    if unknown:
        raise ValueError('Unknown training settings: '+str(sorted(unknown)))
    if result['dtype'] != 'float64':
        raise ValueError('Native endpoint training requires float64')
    if not result['smoke']:
        for key in ('epochs', 'minimum_epochs', 'patience', 'lr', 'weight_decay',
                    'gradient_clip', 'train_degree', 'validation_degree', 'truncate_intervals'):
            if result[key] != DEFAULT_SETTINGS[key]:
                raise ValueError('Formal budget/physics setting differs from declared protocol: '+key)
    if not 1 <= result['minimum_epochs'] <= result['epochs'] or result['patience'] < 1:
        raise ValueError('Invalid epoch budget')
    if min(result['batch'], result['batch_pids'], result['truncate_intervals']) < 1:
        raise ValueError('Batch and truncation sizes must be positive')
    return result


def _move(z, device):
    return {k: v.detach().to(device=device, dtype=torch.float64 if v.is_floating_point() else v.dtype)
            for k, v in z.items() if torch.is_tensor(v)}


def load_data(case_dir, device='cpu'):
    case_dir = Path(case_dir)
    obs = _load(case_dir/'probability_observations.pt')
    pack = _load(case_dir/'training_jets.pt')
    if pack.get('meta', {}).get('probability_observation_sha256') != P.sha(case_dir/'probability_observations.pt'):
        raise ValueError('Truth jets are not hash-bound to these probability observations')
    z, meta = obs['z'], obs['meta']
    if not bool(((z['pid'] % 5 == 2) | (z['pid'] % 5 == 1)).all()):
        raise ValueError('Probability input pack contains forbidden I/H PIDs')
    if not torch.equal(z['row_id'].cpu(), pack['row_id'].cpu()):
        raise ValueError('Observation/jet row correspondence mismatch')
    if not torch.equal(z['row_id'].cpu(), torch.arange(len(z['pid']))):
        raise ValueError('Probability row ids must be local contiguous indices')
    z = _move(z, device)
    jets = tuple(pack[key].detach().to(device=device, dtype=torch.float64) for key in ('v','g','acc'))
    if any(len(a) != len(z['pid']) or not bool(torch.isfinite(a).all()) for a in jets):
        raise ValueError('Nonfinite or misaligned fixed training jets')
    try:
        from .field_adapter import probability_physics
    except ImportError:
        from field_adapter import probability_physics
    c = probability_physics(meta['physics_config'], meta['intervals'])
    c['name'] = str(meta.get('case', case_dir.name))
    for group in ('F','V'):
        P.weights(z, group)  # Explicitly require both states in both splits.
    return dict(c=c, z=z, jets=jets, meta=meta, case_dir=case_dir)


def query(data, z):
    return tuple(a[z['row_id']] for a in data['jets'])


@torch.no_grad()
def normalize_from_F(net, data):
    """Context moments use F only; each state/PID/interval has equal mass."""
    z, c = data['z'], data['c']
    w = P.weights(z, 'F')
    selected = torch.where(w > 0)[0]
    cm = net.base.cm.new_zeros(17)
    second = cm.clone()
    mass = cm.new_zeros(())
    for ix in selected.split(512):
        zz = P.subset(z, ix)
        v, g, _ = query(data, zz)
        active = zz['dt'] > 0
        count = active.sum(-1).clamp_min(1)
        for j in range(v.shape[1]):
            ctx = P.core.context(c, {'w': zz['w']}, v[:, j], g[:, j])
            ww = w[ix]*active[:, j]/count
            cm += (ww[:, None]*ctx).sum(0)
            second += (ww[:, None]*ctx.square()).sum(0)
            mass += ww.sum()
    cm = cm/mass
    cs = (second/mass-cm.square()).clamp_min(0).sqrt().clamp_min(.05)
    net.base.cm.copy_(cm)
    net.base.cs.copy_(cs)
    return dict(group='F', pids=z['pid'][selected].unique().tolist(), rows=len(selected),
                mean=cm.cpu().tolist(), scale=cs.cpu().tolist(), V_or_H_used=False)


@torch.no_grad()
def validate(net, data, cal, force, settings):
    was_training = net.training
    net.freeze()
    score = P.score(data['c'], data['z'], net, cal, force,
                    lambda z: query(data, z), 'V', settings['validation_degree'],
                    batch=settings['batch'], batch_pids=settings['batch_pids'])
    if was_training:
        net.enable_training()
    if not all(math.isfinite(score[k]) for k in ('total','free','cap')):
        raise FloatingPointError('Nonfinite validation endpoint NLL')
    return score


def _detach(state):
    return None if state is None else {k: v.detach() for k, v in state.items()}


def _update(net, opt, weighted_loss, multiplier, clip):
    objective = weighted_loss*multiplier
    if not bool(torch.isfinite(objective)):
        raise FloatingPointError('Nonfinite native endpoint training objective')
    opt.zero_grad(set_to_none=True)
    objective.backward()
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), clip, error_if_nonfinite=True)
    opt.step()
    if not all(bool(torch.isfinite(p).all()) for p in net.parameters()):
        raise FloatingPointError('Nonfinite probability parameter after update')
    return float(norm.detach())


def train_epoch(net, data, cal, force, opt, settings, generator):
    """All F rows, equal state/PID/interval weighting, causal truncated BPTT."""
    net.enable_training()
    c, z = data['c'], data['z']
    w = P.weights(z, 'F')
    free = torch.where((w > 0) & (z['state'] == 0))[0]
    order = torch.randperm(len(free), generator=generator).to(free.device)
    free_batches = list(free[order].split(settings['batch']))
    cap_packs = list(P.sequence_packs(z, (w > 0) & (z['state'] == 1), settings['batch_pids']))
    trunc = settings['truncate_intervals']
    updates = len(free_batches)+sum(math.ceil(index.shape[1]/trunc) for index, _, _ in cap_packs)
    tasks = [('free', batch) for batch in free_batches]+[('cap', pack) for pack in cap_packs]
    order = torch.randperm(len(tasks), generator=generator).tolist()
    totals = dict(free=0., cap=0., gradient_norm_max=0., updates=0, rows=0)
    for number in order:
        kind, task = tasks[number]
        if kind == 'free':
            ix = task
            zz = P.subset(z, ix)
            pred, covariance, lw = P.free_distribution(c, zz, query(data, zz), net, cal, force)
            nll = P.core.scores(pred, covariance, lw, zz['target'])
            loss = (w[ix]*nll/(zz['end']-zz['start'])).sum()
            norm = _update(net, opt, loss, updates, settings['gradient_clip'])
            totals['free'] += float(loss.detach())
            totals['rows'] += len(ix)
            totals['updates'] += 1
            totals['gradient_norm_max'] = max(totals['gradient_norm_max'], norm)
            continue
        index, valid, reset = task
        B, nt = index.shape
        previous = None
        for start in range(0, nt, trunc):
            losses = []
            for t in range(start, min(start+trunc, nt)):
                ix = index[:, t]
                zz = P.subset(z, ix)
                _, _, _, previous = P.captured_distribution(c, zz, query(data, zz), net, cal, force,
                    settings['train_degree'], previous, reset[:, t], True)
                ww = w[ix]*valid[:, t]
                losses.append((ww*(-previous['log_evidence'])/(zz['end']-zz['start'])).sum())
                totals['rows'] += int(valid[:, t].sum())
            loss = torch.stack(losses).sum()
            norm = _update(net, opt, loss, updates, settings['gradient_clip'])
            previous = _detach(previous)
            totals['cap'] += float(loss.detach())
            totals['updates'] += 1
            totals['gradient_norm_max'] = max(totals['gradient_norm_max'], norm)
    expected = int((w > 0).sum())
    if totals['rows'] != expected or totals['updates'] != updates:
        raise AssertionError('An epoch did not visit every authorized F row exactly once')
    totals['total'] = totals['free']+totals['cap']
    totals['free'] *= 2
    totals['cap'] *= 2
    return totals


def _protocol(case_dir, seed, settings):
    sources = P.source_paths()+[Path(__file__), Path(__file__).with_name('field_adapter.py')]
    inputs = [Path(case_dir)/name for name in ('probability_observations.pt','training_jets.pt')]
    inputs += P.calibration_paths()
    result = dict(version='native-each-probability-v1', seed=int(seed),
        settings={k:v for k,v in settings.items() if k != 'max_new_epochs'},
        sources={str(p.resolve()):P.sha(p) for p in sources},
        inputs={str(p.resolve()):P.sha(p) for p in inputs},
        objective='equal-state/PID/interval native endpoint NLL/ms; no latent or force supervision',
        initialization='fresh random marginal encoders/heads; physical-zero free/head outputs; no pretrained tensors',
        initialization_dtype='float64; independent of caller default dtype',
        partition='F pid%5==2 trains; V==1 selects; I>=3 and H==0 excluded at load',
        bptt='4 camera intervals; chronological causal posterior carried detached between chunks',
        truth_jet_use='fixed F/V camera-field interpolation only; no field optimizer or latent truth',
        calibration='original force anchors and observation noise frozen, no new microscopic labels')
    result['fingerprint'] = _fingerprint(result)
    return result


def _run(case_dir: Path, output: Path, seed: int, settings: dict | None = None):
    settings = settings_with_defaults(settings)
    torch.set_num_threads(settings['threads'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(case_dir, seed, settings)
    protocol_path = output/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding='utf8')) != protocol:
            raise RuntimeError('Probability settings, sources or input hashes changed; use a new output')
    else:
        _json(protocol_path, protocol)
    gate_path = output/'selection_frozen.json'
    if gate_path.exists():
        gate = json.loads(gate_path.read_text(encoding='utf8'))
        if gate['protocol_fingerprint'] != protocol['fingerprint'] or P.sha(output/'selected.pt') != gate['selected_sha256']:
            raise RuntimeError('Frozen probability checkpoint mismatch')
        return gate
    _json(output/'status.json', dict(complete=False, stage='loading', seed=seed,
        pid=os.getpid(), updated_utc=_utc(), protocol_fingerprint=protocol['fingerprint']))
    data = load_data(case_dir, settings['device'])
    cal, force = P.calibration()
    net = P.NativeNetwork(seed, settings['device'])
    normalization = normalize_from_F(net, data)
    opt = torch.optim.AdamW(net.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    generator = torch.Generator().manual_seed(seed+1009)
    epoch, last_improvement, elapsed, history = 0, 0, 0., []
    last_path = output/'last.pt'
    if last_path.exists():
        saved = _load(last_path)
        if saved['protocol_fingerprint'] != protocol['fingerprint']:
            raise RuntimeError('Resume protocol differs')
        net.load_state_dict(saved['net_state'], strict=True)
        opt.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator_state'])
        epoch, last_improvement, elapsed, history, best = [saved[k] for k in
            ('epoch','last_improvement','elapsed_seconds','history','best')]
    else:
        _json(output/'status.json', dict(complete=False, stage='initial_validation', epoch=0,
            seed=seed, pid=os.getpid(), updated_utc=_utc(), protocol_fingerprint=protocol['fingerprint']))
        started = time.perf_counter()
        initial = validate(net, data, cal, force, settings)
        best = dict(epoch=0, validation=initial['total'])
        history = [dict(epoch=0, validation=initial)]
        elapsed += time.perf_counter()-started
        _save(output/'selected.pt', dict(net_state=_state(net), seed=seed, epoch=0,
            validation=initial, protocol_fingerprint=protocol['fingerprint']))
        _save(last_path, dict(net_state=_state(net), optimizer=opt.state_dict(),
            generator_state=generator.get_state(), epoch=0, last_improvement=0,
            elapsed_seconds=elapsed, history=history, best=best, seed=seed,
            protocol_fingerprint=protocol['fingerprint']))
        _json(output/'history.json', history)

    def finish():
        selected = _load(output/'selected.pt')
        if selected['protocol_fingerprint'] != protocol['fingerprint'] or selected['epoch'] != best['epoch']:
            raise RuntimeError('Selected model identity differs from resume checkpoint')
        net.load_state_dict(selected['net_state'], strict=True)
        net.freeze()
        gate = dict(complete=True, seed=seed, epoch=epoch, maximum_epochs=settings['epochs'],
            selected_epoch=best['epoch'], elapsed_seconds=elapsed,
            protocol_fingerprint=protocol['fingerprint'], selected_sha256=P.sha(output/'selected.pt'),
            selected_path=str((output/'selected.pt').resolve()), settings=protocol['settings'],
            normalization=normalization, pid=os.getpid(), updated_utc=_utc(),
            stopping='maximum_epochs' if epoch == settings['epochs'] else 'validation_patience',
            probability_parameters_frozen=True, field_optimized=False, I_H_loaded=False,
            calibration_sha256={str(p):P.sha(p) for p in P.calibration_paths()},
            supervised_latent_or_force_labels_used=False)
        _json(gate_path, gate)
        _json(output/'complete.json', gate)
        _json(output/'status.json', gate)
        return gate

    if epoch >= settings['epochs'] or (epoch >= settings['minimum_epochs']
                                       and epoch-last_improvement >= settings['patience']):
        return finish()
    starting_epoch = epoch
    while epoch < settings['epochs']:
        _json(output/'status.json', dict(complete=False, stage='training', epoch=epoch,
            active_epoch=epoch+1, seed=seed, pid=os.getpid(), updated_utc=_utc(),
            elapsed_seconds=elapsed, selected_epoch=best['epoch'],
            protocol_fingerprint=protocol['fingerprint']))
        started = time.perf_counter()
        training = train_epoch(net, data, cal, force, opt, settings, generator)
        _json(output/'status.json', dict(complete=False, stage='validation', epoch=epoch,
            active_epoch=epoch+1, seed=seed, pid=os.getpid(), updated_utc=_utc(),
            elapsed_seconds=elapsed+time.perf_counter()-started, selected_epoch=best['epoch'],
            protocol_fingerprint=protocol['fingerprint']))
        validation = validate(net, data, cal, force, settings)
        epoch += 1
        chosen = validation['total'] < best['validation']-settings['selection_min_delta']
        if chosen:
            best = dict(epoch=epoch, validation=validation['total'])
            last_improvement = epoch
            _save(output/'selected.pt', dict(net_state=_state(net), seed=seed, epoch=epoch,
                validation=validation, protocol_fingerprint=protocol['fingerprint']))
        elapsed += time.perf_counter()-started
        history.append(dict(epoch=epoch, train=training, validation=validation, selected=chosen,
                            epoch_seconds=time.perf_counter()-started, elapsed_seconds=elapsed))
        complete = epoch == settings['epochs'] or (
            epoch >= settings['minimum_epochs'] and epoch-last_improvement >= settings['patience'])
        _save(last_path, dict(net_state=_state(net), optimizer=opt.state_dict(),
            generator_state=generator.get_state(), epoch=epoch, last_improvement=last_improvement,
            elapsed_seconds=elapsed, history=history, best=best, seed=seed,
            protocol_fingerprint=protocol['fingerprint']))
        _json(output/'history.json', history)
        status = dict(complete=complete, seed=seed, epoch=epoch, maximum_epochs=settings['epochs'],
            selected_epoch=best['epoch'], elapsed_seconds=elapsed,
            pid=os.getpid(), updated_utc=_utc(), stage='finished' if complete else 'epoch_complete',
            protocol_fingerprint=protocol['fingerprint'], selected_sha256=P.sha(output/'selected.pt'))
        _json(output/'status.json', status)
        print('PROBABILITY', data['c']['name'], seed, epoch, 'V', validation['total'],
              'selected', best['epoch'], 'seconds', round(elapsed, 2), flush=True)
        if complete:
            return finish()
        if settings.get('max_new_epochs') is not None and epoch-starting_epoch >= settings['max_new_epochs']:
            return status
    raise RuntimeError('Completed resume lacks frozen selection; inspect checkpoint before continuing')


def run(case_dir: Path, output: Path, seed: int, settings: dict | None = None):
    """Exclusive resumable training; caller may additionally own a job lock."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = output/'training.lock'
    with lock.open('x', encoding='utf8') as stream:
        json.dump(dict(pid=os.getpid(), seed=seed, started_utc=_utc()), stream)
    try:
        return _run(case_dir, output, seed, settings)
    except BaseException as error:
        _json(output/'failure.json', dict(seed=seed, error=repr(error),
            pid=os.getpid(), updated_utc=_utc()))
        raise
    finally:
        lock.unlink(missing_ok=True)


def profile(case_dir, output, seed=411040, settings=None):
    """One mixed free/captured optimizer step; no retained training weights."""
    settings = settings_with_defaults(settings)
    torch.set_num_threads(settings['threads'])
    data = load_data(case_dir, settings['device'])
    cal, force = P.calibration()
    net = P.NativeNetwork(seed, settings['device'])
    normalize_from_F(net, data)
    net.enable_training()
    opt = torch.optim.AdamW(net.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    z, c = data['z'], data['c']
    w = P.weights(z, 'F')
    started = time.perf_counter()
    losses = []
    fi = torch.where((w > 0) & (z['state'] == 0))[0][:min(settings['batch'], 16)]
    zz = P.subset(z, fi)
    pred, cov, lw = P.free_distribution(c, zz, query(data, zz), net, cal, force)
    losses.append((w[fi]*P.core.scores(pred, cov, lw, zz['target'])/(zz['end']-zz['start'])).sum())
    index, valid, reset = next(P.sequence_packs(z, (w > 0) & (z['state'] == 1), settings['batch_pids']))
    previous = None
    nt = min(index.shape[1], settings['truncate_intervals'])
    for t in range(nt):
        ix = index[:, t]
        zz = P.subset(z, ix)
        _, _, _, previous = P.captured_distribution(c, zz, query(data, zz), net, cal, force,
            settings['train_degree'], previous, reset[:, t], True)
        losses.append((w[ix]*valid[:, t]*(-previous['log_evidence'])/(zz['end']-zz['start'])).sum())
    norm = _update(net, opt, torch.stack(losses).sum(), 1., settings['gradient_clip'])
    if settings['device'].startswith('cuda'):
        torch.cuda.synchronize()
    wall = time.perf_counter()-started
    families = {name: any(p.grad is not None and bool(torch.isfinite(p.grad).all())
                 and float(p.grad.abs().max()) > 0 for key,p in net.named_parameters() if key.startswith(prefix))
                for name,prefix in [('cap','base.gru'),('free','base.free_output'),('transition','head.') ]}
    result = dict(complete=True, case=c['name'], optimizer_steps=1, wall_seconds=wall,
        free_rows=len(fi), captured_rows=int(valid[:, :nt].sum()), captured_pids=index.shape[0],
        camera_intervals_per_pid=nt, internal_pieces=z['time'].shape[1], train_degree=settings['train_degree'],
        gradient_norm=norm, finite_nonzero_gradient_families=families,
        probability_parameters=sum(p.numel() for p in net.parameters()),
        initialization_dtype='float64',
        all_F_rows=int((w > 0).sum()), formal_epochs=settings['epochs'],
        estimate_note='Single mixed step only; full GH20 validation and variable sequence lengths add cost.',
        weights_discarded=True, formal_training_started=False, settings=settings)
    _json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run','profile'])
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=411040)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    result = globals()[args.action](args.case_dir, args.output, args.seed, dict(device=args.device))
    print(json.dumps(result, ensure_ascii=False), flush=True)
