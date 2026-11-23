"""Continuous Fourier-field queries for the unchanged particle likelihood.

Coordinates and time are SI. Coefficients use the existing divergence-free
sin/cos ordering, with the fixed physical mean supplied separately. Neither
the PDE right-hand side nor a process innovation enters these queries.

The particle equation has a specific acceleration convention: partial_t u +
grad(|u|**2)/2. This is *not* the material acceleration used by the two-fluid
PDE. Both are exposed explicitly to prevent an accidental transpose change.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
PACKAGE = BASE / 'outputs/FullWindowCalibration_20261026/server_package'
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))
from counterflow_full import core

DT_SECONDS = .001
QUERY_CONVENTION = ('continuous midpoint; true partial_t; particle acc=partial_t u+'
                    'grad(|u|^2)/2; no PDE-residual/process-increment subtraction')


def midpoint_times(z, *, dt=DT_SECONDS, origin=0.):
    """The interval frame k queries time origin+(k+1/2)*dt seconds.

    A short-window caller must pass its absolute origin explicitly, or retain
    original_frame and request that origin when making its query callback.
    Endpoint observations never determine these query times or positions.
    """
    if dt <= 0 or z['frame'].ndim != 1:
        raise ValueError('Expected positive SI interval duration and frame[B]')
    return (z['frame'].to(device=z['x'].device, dtype=z['x'].dtype)+.5)*dt+origin


def coefficient_jets(a, adot, z, c, K=12, *, basis=None, detailed=False):
    """Query row-aligned a/adot[B,2,M]; adot is derivative per second.

    Axis order is [normal,superfluid], [velocity component,coordinate] for G.
    Frozen probability parameters do not imply detached a/adot/context inputs.
    """
    expected = (len(z['x']), 2, 2*len(core.modes(K)[0]))
    if a.shape != expected or adot.shape != expected:
        raise ValueError(f'Expected a and adot shape {expected}, got {a.shape}, {adot.shape}')
    if a.device != adot.device or a.dtype != adot.dtype:
        raise ValueError('a and adot must share device/dtype')
    if z['x'].device != a.device or z['x'].dtype != a.dtype:
        raise ValueError('Spatial coordinates and coefficients must share device/dtype')
    V, G = core.basis(z['x'], c, K) if basis is None else basis
    if V.shape != (len(a), expected[-1], 2) or G.shape != (len(a), expected[-1], 2, 2):
        raise ValueError('Invalid Fourier basis shape')
    p = c['physics']
    fixed_mean = a.new_tensor([[0., p['vn_mean_m_s']], [0., p['vs_mean_m_s']]])
    velocity = fixed_mean+torch.einsum('bfm,bmj->bfj', a, V)
    gradient = torch.einsum('bfm,bmij->bfij', a, G)
    partial_t = torch.einsum('bfm,bmj->bfj', adot, V)
    energy_gradient = torch.einsum('bfij,bfi->bfj', gradient, velocity)
    advection = torch.einsum('bfij,bfj->bfi', gradient, velocity)
    particle_acceleration = partial_t+energy_gradient
    if detailed:
        return dict(velocity=velocity, gradient=gradient, partial_t=partial_t,
                    energy_gradient=energy_gradient, advection=advection,
                    particle_acceleration=particle_acceleration,
                    material_acceleration=partial_t+advection)
    return velocity, gradient, particle_acceleration


local_jets = coefficient_jets


def query(field, z, c, K=12, *, create_graph=True, dt=DT_SECONDS, origin=0., detailed=False):
    """Query a field implementing coefficients_and_dt(t_seconds).

    enable_grad is intentional: even evaluation requires the time derivative.
    create_graph=False permits a derivative-only evaluation; training uses True
    so the likelihood derivative through partial_t reaches all field weights.
    """
    with torch.enable_grad():
        t = midpoint_times(z, dt=dt, origin=origin)
        a, adot = field.coefficients_and_dt(t)
        if not create_graph:
            a, adot = a.detach(), adot.detach()
        return coefficient_jets(a, adot, z, c, K, detailed=detailed)


def make_query(field, c, K=12, *, create_graph=True, dt=DT_SECONDS, origin=0.):
    """Callback compatible with frozen causal sequence_scores(jets=...)."""
    def jets(z, indices=None):
        # indices belong to likelihood accounting; they never enter the field.
        return query(field, z, c, K, create_graph=create_graph, dt=dt, origin=origin)
    return jets


def score_nodes(a_mid, adot_mid, d, net, cal, force, group, degree=20,
                metrics=False, backward=False, batch=256, batch_pids=64):
    """Unchanged particle law on continuous field/time-derivative nodes.

    a_mid/adot_mid have shape [95,2,624] for the full 95 ms window, and are
    queried by each observation interval's frame. A shorter explicitly rebased
    window is also valid. No endpoint coefficients or eta arrays are accepted.

    The parent score returns detached scalar accounting while optionally
    accumulating .08*likelihood gradients into these input leaves per batch.
    A caller can then backpropagate their gradients through the field network
    once. This limits filter-graph memory without detaching the field inputs
    *inside* either particle model. Every captured current endpoint is scored
    by LM.step before it updates the posterior used at the next interval.
    """
    from pinn_common import S, TM, SCORING
    if a_mid.ndim != 3 or a_mid.shape[1:] != (2, 624) or adot_mid.shape != a_mid.shape:
        raise ValueError('Expected matching midpoint coefficient and SI derivative [T,2,624]')
    if any(p.requires_grad for p in net.parameters()):
        raise RuntimeError('Particle model parameters must remain frozen')
    if backward and (not a_mid.is_leaf or not adot_mid.is_leaf):
        raise ValueError('Batched backward requires independent a_mid/adot_mid leaves')
    z, c = d['z'], d['c']
    w = S.weights(z, group)
    result = dict(total=0.)
    if metrics:
        result.update(coverage={}, spread={}, per_pid={})

    def jets(zz, indices=None):
        frame = zz['frame']
        if bool(((frame < 0) | (frame >= len(a_mid))).any()):
            raise ValueError('Particle frame outside supplied midpoint time nodes')
        return coefficient_jets(a_mid[frame], adot_mid[frame], zz, c)

    ids = torch.where((w > 0) & (z['state'] == 0))[0]
    value, mass, parts = 0., 0., {}
    for ix in ids.split(batch):
        zz = core.subset(z, ix, a_mid.device, a_mid.dtype)
        v, g, acc = jets(zz)
        pred, cov, lw = TM.distribution(c, zz, v, g, acc, 'free', net.base, cal, force, degree)
        nll = TM.scores(pred, cov, lw, zz['target'])
        ww = w[ix].to(a_mid)
        loss = (ww*nll).sum()
        if backward:
            (.08*loss).backward()
        value += float(loss.detach())
        mass += float(ww.sum())
        if metrics:
            cover = (core.coverage(pred, cov, lw, zz['target']) <= 4.605170186).double()
            prob = lw.exp()
            mean = (prob[..., None]*pred).sum((1, 2))
            trace = cov.diagonal(dim1=-2, dim2=-1).sum(-1)
            spread = (prob*(trace+(pred-mean[:, None, None]).square().sum(-1))).sum((1, 2))
            spread = spread.clamp_min(0).sqrt()*c['physics']['vn_mean_m_s']*.001*1e6
            for pid in zz['pid'].unique().tolist():
                use = zz['pid'] == pid
                row = parts.setdefault(str(pid), dict(count=0, nll=0., coverage=0., spread=0.))
                row['count'] += int(use.sum())
                for key, tensor in (('nll', nll), ('coverage', cover), ('spread', spread)):
                    row[key] += float(tensor[use].detach().sum())
    if mass <= 0:
        raise ValueError('No free particles in requested group')
    for row in parts.values():
        for key in ('nll', 'coverage', 'spread'):
            row[key] /= row['count']
    result['free'] = value/mass
    result['total'] += value
    if metrics:
        result['per_pid']['free'] = parts
        for key in ('coverage', 'spread'):
            result[key]['free'] = float(np.mean([r[key] for r in parts.values()]))
    value, mass, parts = SCORING.sequence_scores(z, c, net.base, net.head, cal, force,
        w, jets, degree=degree, mode=net.mode, metrics=metrics, backward=backward,
        batch_pids=batch_pids)
    if mass <= 0:
        raise ValueError('No captured particles in requested group')
    result['cap'] = value/mass
    result['total'] += value
    if metrics:
        result['per_pid']['cap'] = parts
        for key in ('coverage', 'spread'):
            result[key]['cap'] = float(np.mean([r[key] for r in parts.values()]))
    return result
