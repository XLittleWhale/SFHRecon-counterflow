"""Differentiable two-state likelihood for non-target joint learning.

The frozen v2 probability law is preserved exactly.  The two local overrides
only remove its frozen-parameter guards: contexts, causal filtering, latent
quadrature, and posterior moments retain their complete differentiation graph.
Historical modules are never monkey-patched.
"""
from __future__ import annotations
import math
import torch
from common import LM, TM

core, ob = LM.core, LM.ob


class JointHead(LM.LatentTransitionHead):
    def parameters_at(self, h, ctx, prevctx):
        if ctx.shape != (len(h), 17) or prevctx.shape != ctx.shape:
            raise ValueError('Expected current and previous context [B,17]')
        base = self._base
        encoded = base.encode(h, ctx)
        inputs = torch.cat((encoded, (prevctx-base.cm)/base.cs, (ctx-base.cm)/base.cs), -1)
        raw = self.network(inputs)
        kappa = LM.bounded_kappa(raw[:, 0])
        A = 1.5*torch.tanh(raw[:, 1:5]).reshape(-1, 2, 2)
        norm = torch.linalg.matrix_norm(A, ord=2, dim=(-2, -1))
        A = A*(.995/norm.clamp_min(.995))[:, None, None]
        b = 2*torch.tanh(raw[:, 5:7])
        d0 = torch.exp(math.log(50.)*torch.tanh(raw[:, 7]))
        off = .5*torch.tanh(raw[:, 8])
        d1 = torch.exp(math.log(50.)*torch.tanh(raw[:, 9]))
        L = torch.stack((torch.stack((d0, torch.zeros_like(d0)), -1),
                         torch.stack((off, d1), -1)), -2)
        return kappa, A, b, L


def step(c, z, v, g, acc, base, head, cal, force, degree,
         prevstate=None, prevctx=None, mode='temporal', reset_mask=None, update=True):
    """Exact v2 predictive law, allowing base and transition parameters to learn."""
    if mode not in LM.MODES:
        raise ValueError('Unknown latent mode: '+str(mode))
    ctx = core.context(c, z, v, g)
    logq, mu, geometry_factor = ob.latent(base, z['h'], ctx, cal)
    qprior, emean, ecov = LM._predict_state(logq, z['h'], ctx, prevctx, prevstate, head, mode, reset_mask)
    eps, node_logw = ob.nodes(degree, str(v.device), v.dtype)
    if mode == 'iid':
        pred, cov, lw = core.distribution(c, z, v, g, acc, 'cap', base, cal, force, degree)
        e = eps[None, None].expand(len(v), 2, -1, -1)
    else:
        efactor = LM._cholesky(ecov)
        e = emean[:, :, None]+torch.einsum('bqij,nj->bqni', efactor, eps)
        delta = mu[:, :, None]+torch.einsum('bqij,bqnj->bqni', geometry_factor, e)
        tension_direction = delta/(c['hr']**2+delta.square().sum(-1, keepdim=True)).sqrt()
        U = c['physics']['vn_mean_m_s']
        _, H, B, M = core.matrices(c, v, 'cap')
        physical = torch.einsum('qij,bj->bqi', H, z['w'])
        physical = physical+core.DT/U*torch.einsum('qij,bqj->bqi', B, core.affine(c, v, acc, 'cap'))
        force_mean = v.new_tensor(force['cap']['mean'])
        force_cov = v.new_tensor(force['cap']['covariance'])
        physical = physical+torch.einsum('qij,j->qi', M, force_mean)[None]
        covariance = M@force_cov@M.transpose(-1, -2)
        covariance = covariance+torch.eye(2, device=v.device, dtype=v.dtype)*cal['sigma']['cap']**2
        gain = core.DT*c['tension_twice_N']/(c['constants']['mass']*U)*B
        pred = physical[:, :, None]+torch.einsum('qij,bqnj->bqni', gain, tension_direction)
        cov = covariance[None, :, None].expand(len(v), -1, len(eps), -1, -1)
        lw = qprior[:, :, None]+node_logw
    if not update:
        return pred, cov, lw, None
    if 'target' not in z:
        raise ValueError('Updating a filter requires the current observed endpoint')
    log_joint = lw+LM.emission_log_prob(pred, cov, z['target'])
    q_evidence = torch.logsumexp(log_joint, -1)
    evidence = torch.logsumexp(q_evidence, -1)
    posterior_q = q_evidence-evidence[:, None]
    conditional_w = torch.softmax(log_joint, -1)
    posterior_mean = (conditional_w[..., None]*e).sum(2)
    centered = e-posterior_mean[:, :, None]
    posterior_cov = torch.einsum('bqn,bqni,bqnj->bqij', conditional_w, centered, centered)
    posterior_cov = .5*(posterior_cov+posterior_cov.transpose(-1, -2))
    posterior_cov = posterior_cov+LM.POSTERIOR_COVARIANCE_JITTER*torch.eye(2, device=v.device, dtype=v.dtype)
    state = dict(qlogp=posterior_q, emean=posterior_mean, ecov=posterior_cov,
                 base_qlogp=logq, context=ctx, log_evidence=evidence,
                 predictive_qlogp=qprior, predictive_emean=emean, predictive_ecov=ecov)
    return pred, cov, lw, state


def pid_weights(pid):
    """Each physical PID has equal mass; its available rows share that mass."""
    if len(pid) == 0:
        raise ValueError('Empty particle state/split is forbidden')
    _, inv, counts = pid.unique(return_inverse=True, return_counts=True)
    return 1/(len(counts)*counts[inv].double())


def sequence_layout(z, mask, frames):
    ids = torch.where(mask)[0]
    if not len(ids):
        raise ValueError('No captured rows in this permitted split/window')
    pids = z['pid'][ids].unique(sorted=True)
    index = torch.full((len(pids), frames), -1, dtype=torch.long)
    ranks = torch.searchsorted(pids, z['pid'][ids])
    fr = z['frame'][ids]
    if bool(((fr < 0) | (fr >= frames)).any()):
        raise ValueError('Window frames must be local and inside the window')
    if len(torch.unique(ranks*frames+fr)) != len(ids):
        raise ValueError('Duplicate PID/frame rows')
    index[ranks, fr] = ids
    valid = index >= 0
    reset = torch.ones_like(valid)
    reset[:, 1:] = ~valid[:, :-1]
    return index, valid, reset


def window_nll(c, z, jets, net, cal, force, mask, degree, frames):
    """Equal two states/PIDs; all within-window cap filtering gradients survive.

    Each captured state segment resets at a missing frame or free/cap switch.
    The first local frame resets: no unavailable pre-window posterior is used.
    Padding is copied only from an authorized captured row and contributes zero.
    """
    v, g, acc = jets
    fi = torch.where(mask & (z['state'] == 0))[0]
    ci = torch.where(mask & (z['state'] == 1))[0]
    if min(len(fi), len(ci)) == 0:
        raise ValueError('Both states require observed rows in every scoring split')
    sub = core.subset(z, fi)
    pred, cov, lw = TM.distribution(c, sub, v[fi], g[fi], acc[fi], 'free', net.base, cal, force, degree)
    free = (core.scores(pred, cov, lw, sub['target'])*pid_weights(sub['pid'])).sum()
    index, valid, reset = sequence_layout(z, mask & (z['state'] == 1), frames)
    allw = v.new_zeros(len(v))
    allw[ci] = pid_weights(z['pid'][ci])
    filler = int(ci[0])
    previous, previous_context = None, None
    captured = v.new_zeros(())
    for t in range(frames):
        ids = torch.where(valid[:, t], index[:, t], filler)
        sub = core.subset(z, ids)
        _, _, _, previous = step(c, sub, v[ids], g[ids], acc[ids], net.base, net.head,
            cal, force, degree, previous, previous_context, 'temporal', reset[:, t], True)
        previous_context = previous['context']
        captured = captured+(-previous['log_evidence']*allw[ids]*valid[:, t]).sum()
    return dict(free=free, cap=captured, total=.5*(free+captured),
                free_rows=len(fi), cap_rows=len(ci), free_pids=len(z['pid'][fi].unique()),
                cap_pids=len(z['pid'][ci].unique()))


def supervised_anchors(c, z, jets, labels, force_labels, net, cal, force, mask, frames):
    """True-field context anchors; no target labels and no hidden endpoint input."""
    v, g, _ = jets
    ctx = core.context(c, z, v, g)
    fi = torch.where(mask & (z['state'] == 0))[0]
    ci = torch.where(mask & (z['state'] == 1))[0]
    if min(len(fi), len(ci)) == 0:
        raise ValueError('Both supervised states must be present')
    free = (TM.free_force_nll(net.base, z['h'][fi], ctx[fi], force_labels[fi], force)*pid_weights(z['pid'][fi])).sum()
    logq, _, _, e, factor = LM.whiten_latent(net.base, z['h'][ci], ctx[ci], labels['y'][ci], labels['q'][ci], cal)
    marginal = (-logq[torch.arange(len(ci)), labels['q'][ci]]+.5*e.square().sum(-1)
                +factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)+math.log(2*math.pi))
    cap = (marginal*pid_weights(z['pid'][ci])).sum()
    index, valid, _ = sequence_layout(z, mask & (z['state'] == 1), frames)
    paired = valid[:, 1:] & valid[:, :-1]
    previous, current = index[:, :-1][paired], index[:, 1:][paired]
    if not len(current):
        raise ValueError('No adjacent captured pair for temporal anchor')
    values, _ = LM.supervised_pair_nll(net.base, net.head, z['h'][previous], ctx[previous],
        labels['y'][previous], labels['q'][previous], z['h'][current], ctx[current],
        labels['y'][current], labels['q'][current], cal)
    pair = (values*pid_weights(z['pid'][current])).sum()
    return dict(free=free, cap=cap, pair=pair, total=(free+cap+pair)/3, pairs=len(current))
