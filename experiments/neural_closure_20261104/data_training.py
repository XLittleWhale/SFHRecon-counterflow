"""Non-target training truth and strictly isolated short-window particle views.

Only the explicit seven training cases and 6.5 mm/s validation can be opened.
Old jets are used solely to reconstruct and relabel the full-force residual;
every context and resolved acceleration is recomputed with nu_s=0/Fns=1.
"""
from __future__ import annotations
import argparse
import torch
from common import *
from closure_model import CURRENT, ClosureMomentum, batched_step
from probability import core

TRAIN_CASES = T.TRAIN_CASES
VALIDATION_CASE = T.VALIDATION_CASE
ALLOWED = (*TRAIN_CASES, VALIDATION_CASE)


def input_paths():
    paths = [T.PACKAGE/'operator.pt', T.SOURCE/'calibration.json', T.SOURCE/'force.json']
    for name in ALLOWED:
        paths.extend(T.SOURCE/'prepared'/name/file for file in ('training.pt', 'observations.pt', 'reference.pt'))
    return paths


def load_case(name):
    if name not in ALLOWED:
        raise ValueError('Only non-target training/validation data may be loaded')
    folder = T.SOURCE/'prepared'/name
    d = load(folder/'training.pt')
    obs = load(folder/'observations.pt')
    a = load(folder/'reference.pt')['a'].double()
    assert a.shape == (96, 2, 624) and torch.isfinite(a).all()
    c = obs['c']
    operator = CURRENT.Momentum(core.operator(c, load(T.PACKAGE/'operator.pt'), core.config(), 'cpu'),
                                local=False, friction_factor=1.0, nu_s=0.0)
    z = core.subset(d['z'], torch.arange(len(d['z']['pid'])))
    allowed = z['pid'] % 5 == 1 if name == VALIDATION_CASE else z['pid'] % 5 >= 2
    ids = torch.where(allowed)[0]
    # Remove all forbidden particle rows at load time, before constructing padding.
    z = core.subset(z, ids)
    labels = {key: val[ids].double() if val.is_floating_point() else val[ids]
              for key, val in d['labels'].items()}
    oldjets = tuple(item[ids].double() for item in d['jets'])
    return dict(case=name, c=c, z=z, a=a, labels=labels, oldjets=oldjets,
                force_labels=d['force'][ids].double(), op=operator,
                cal=read(T.SOURCE/'calibration.json'), force=read(T.SOURCE/'force.json'),
                group='validation' if name == VALIDATION_CASE else 'train')


def split_masks(z, group):
    p = z['pid']
    if group == 'train':
        permitted = p % 5 >= 2
        inner, outer = p % 5 >= 3, p % 5 == 2
    elif group == 'validation':
        permitted = p % 5 == 1
        inner = permitted & ((p//5) % 2 == 0)
        outer = permitted & ((p//5) % 2 == 1)
    else:
        raise ValueError(group)
    if not bool(permitted.all()) or bool((inner & outer).any()) or not bool((inner | outer).all()):
        raise AssertionError('Physical PID isolation violation')
    return dict(inner=inner, outer=outer, permitted=permitted)


def window(case, start, length=5, minimum_state_pids=2):
    if not (0 <= start <= 95-length):
        raise ValueError('Invalid window bounds')
    ids = torch.where((case['z']['frame'] >= start) & (case['z']['frame'] < start+length))[0]
    z = core.subset(case['z'], ids)
    z['original_frame'] = z['frame'].clone()
    z['frame'] = z['frame']-start
    masks = split_masks(z, case['group'])
    counts = {}
    for split in ('inner', 'outer'):
        counts[split] = {}
        for state, kind in ((0, 'free'), (1, 'cap')):
            rows = masks[split] & (z['state'] == state)
            n = len(z['pid'][rows].unique())
            counts[split][kind] = dict(rows=int(rows.sum()), pids=n)
            if n < minimum_state_pids:
                raise ValueError(f'Insufficient {case["case"]} {start} {split}/{kind}: {n} PIDs')
    V, G = core.basis(z['x'], case['c'], 12)
    return dict(case=case['case'], c=case['c'], op=case['op'], cal=case['cal'], force=case['force'],
                z=z, labels={k: v[ids] for k, v in case['labels'].items()},
                oldjets=tuple(v[ids] for v in case['oldjets']), force_labels=case['force_labels'][ids],
                truth=case['a'][start:start+length+1], masks=masks, counts=counts,
                basis=(V, G), start=start, length=length, group=case['group'])


def trajectory(op, initial, length, substeps=2):
    result = [initial]
    for _ in range(length):
        result.append(batched_step(op, result[-1], substeps))
    return torch.stack(result)


def reference_residual(op, truth, substeps=2):
    return truth[1:]-batched_step(op, truth[:-1], substeps)


def query(w, a, residual=None):
    """Identical Frozen B query: subtract stochastic residual, never neural RHS."""
    if residual is None:
        residual = a.new_zeros(len(a)-1, 2, a.shape[-1])
    V, G = w['basis']
    frame = w['z']['frame']
    mid = .5*(a[frame]+a[frame+1])-.5*torch.stack((residual[frame, 0], torch.zeros_like(residual[frame, 0])), 1)
    derivative = (a[frame+1]-a[frame]-residual[frame])/core.DT
    c = w['c']
    mean = a.new_tensor([[0, c['physics']['vn_mean_m_s']], [0, c['physics']['vs_mean_m_s']]])
    velocity = mean+torch.einsum('bfm,bmj->bfj', mid, V)
    gradient = torch.einsum('bfm,bmij->bfij', mid, G)
    acceleration = torch.einsum('bfm,bmj->bfj', derivative, V)+torch.einsum('bfij,bfi->bfj', gradient, velocity)
    return velocity, gradient, acceleration


def relabel_force(w, newjets):
    """Preserve full physical force while changing the resolved query convention."""
    oldv, _, oldacc = w['oldjets']
    newv, _, newacc = newjets
    result = w['force_labels'].clone()
    U = w['c']['physics']['vn_mean_m_s']
    for state, kind in ((0, 'free'), (1, 'cap')):
        ids = torch.where(w['z']['state'] == state)[0]
        oldb = core.affine(w['c'], oldv[ids], oldacc[ids], kind)
        newb = core.affine(w['c'], newv[ids], newacc[ids], kind)
        A = core.matrices(w['c'], newv, kind)[0]
        q = torch.zeros(len(ids), dtype=torch.long) if state == 0 else w['labels']['q'][ids]
        ix = torch.arange(len(ids))
        correction = torch.linalg.solve(A[q], ((oldb-newb)[ix, q]/U)[..., None])[..., 0]
        result[ids] = result[ids]+correction
    return result


def audit():
    report = dict(target_files_opened=False, operator=dict(nu_s=0., Fns_factor=1.),
                  inputs={str(p): sha(p) for p in input_paths()}, cases=[])
    for name in ALLOWED:
        case = load_case(name)
        masks = split_masks(case['z'], case['group'])
        rows = dict(case=name, group=case['group'], truth_shape=list(case['a'].shape), particle_rows=len(case['z']['pid']),
                    splits={k: int(v.sum()) for k, v in masks.items()}, fixed_windows=[])
        for start in (0, 30, 60, 90):
            w = window(case, start)
            rows['fixed_windows'].append(dict(start=start, counts=w['counts']))
        report['cases'].append(rows)
    dump(OUT/'preflight/data_audit.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--audit', action='store_true')
    args = parser.parse_args(); setup()
    if args.audit: audit()
