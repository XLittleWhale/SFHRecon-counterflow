"""Native-camera transfer of the frozen PINNTwoFluid joint probability law.

There are no intermediate observed endpoints. Between camera endpoints we
propagate a conditional Gaussian joint state (e, particle velocity, cumulative
displacement), separately for each latent sign. Two-dimensional GH integration
retains nonlinear tension and its cross moments. A camera endpoint is scored
once, then updates only the q/e posterior used by the next camera interval.

This is an explicit transfer approximation, not a retrained camera model:
* the original four-channel history is a 5 ms causal PCHIP reconstruction
  from past camera points with secant extrapolation after the interval origin;
  these features are not new observations or likelihood factors;
* a fraction f of a 1 ms latent step is the mixture (1-f)I + f K, with Gaussian
  moment matching conditional on sign, not an asserted continuous-time law;
* unresolved force innovations are independent between internal integration
  pieces, while q/e/particle-velocity/displacement cross covariances persist;
* one unchanged calibrated observation-noise covariance is added at the sole
  actual camera endpoint. Its experimental calibration is not established.

All coefficients, contexts and filtering moments retain field gradients.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
SIM = BASE/'outputs/PINNTwoFluid_20261105'
PACKAGE = BASE/'outputs/FullWindowCalibration_20261026/server_package'
CF_PACKAGE = BASE/'outputs/JointProbabilityPhysics_20261103/cf_runtime/payload/server_package'
CALIBRATION = BASE/'outputs/FullWindowCalibration_20261026/fullwindow_4mm_results'
for relative in ('counterflow/observation.py', 'counterflow_full/core.py'):
    if (PACKAGE/relative).read_bytes() != (CF_PACKAGE/relative).read_bytes():
        raise RuntimeError('CF runtime copy differs from frozen simulation probability source: '+relative)
# The CF package contains additional channel geometry modules but byte-identical
# frozen probability sources. Import the complete package first to avoid Python
# caching the smaller simulation-only counterflow namespace.
if str(CF_PACKAGE) not in sys.path:
    sys.path.insert(0, str(CF_PACKAGE))
from counterflow import observation as ob
from counterflow_full import core


def _module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


TM = _module('cf_frozen_transition_density', HERE.parent/'shared_transition_20261104/transition_model.py')
LM = _module('cf_frozen_latent_v2', HERE.parent/'latent_transition_20261104_v2/latent_model.py')
DT = .001
POSTERIOR_JITTER = 1e-10
CONTRACT = dict(version=1, probability='frozen PINNTwoFluid joint net_state',
    history='per-internal-midpoint 5x4 causal 5ms PCHIP past; secant future pilot; no pseudo observations',
    latent_transfer='fractional identity/full 1ms kernel mixture; sign-conditional ADF',
    within_interval='joint e2/vp2/dx2 conditional Gaussian; GH2 nonlinear tension',
    across_camera_endpoint='only q/e posterior persists; vp resets to past observed secant, dx to zero; vp/e covariance discarded',
    endpoint_noise='one original calibrated sigma^2 I per actual endpoint',
    force_innovations='independent between internal pieces',
    target_units='dx/(U*0.001s)', velocity_units='vp/U',
    context_timescale_seconds=.001, target_updates_per_interval=1)


class FrozenNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = TM.TransitionDensity('independent', ob.Density()).double()
        self.head = LM.LatentTransitionHead(self.base).double()
        self.mode = 'temporal'

    def freeze(self):
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        return self


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_probability(field_seed, device='cpu'):
    """Load exactly the selected joint tensors; never load observations/truth."""
    gate_path = SIM/'probability_selection/selection_frozen.json'
    gate = json.loads(gate_path.read_text(encoding='utf8'))
    assert gate['complete'] and gate['family'] == 'joint'
    row = next(r for r in gate['models'] if r['field_seed'] == int(field_seed))
    if sha(row['path']) != row['sha256']:
        raise RuntimeError('Frozen joint probability checkpoint SHA mismatch')
    saved = torch.load(row['path'], map_location='cpu', weights_only=False)
    net = FrozenNetwork()
    net.load_state_dict(saved['net_state'], strict=True)
    net.to(device=device, dtype=torch.float64).freeze()
    cal = json.loads((CALIBRATION/'calibration.json').read_text(encoding='utf8'))
    force = json.loads((CALIBRATION/'force.json').read_text(encoding='utf8'))
    return net, cal, force, dict(path=row['path'], sha256=row['sha256'],
        training_seed=row['training_seed'], field_seed=field_seed,
        gate_path=str(gate_path), gate_sha256=sha(gate_path))


def group_mask(pid, group):
    if group in ('fit', 'I'):
        return pid % 5 >= 2
    if group in ('selection', 'V'):
        return pid % 5 == 1
    if group in ('test', 'H'):
        return pid % 5 == 0
    raise ValueError('Explicit fit/selection/test group required')


def weights(z, group):
    use = group_mask(z['pid'], group)
    out = torch.zeros(len(z['pid']), dtype=z['target'].dtype, device=z['pid'].device)
    for kind in (0, 1):
        ix = torch.where(use & (z['state'] == kind))[0]
        if not len(ix):
            raise ValueError('Both particle states must be present in the group')
        _, inv, count = z['pid'][ix].unique(return_inverse=True, return_counts=True)
        out[ix] = .5/(len(count)*count[inv].to(out))
    return out


def subset(z, ix):
    return {k: v[ix] for k, v in z.items() if torch.is_tensor(v)}


def history(z, j=0):
    h = z.get('h')
    if h is None:
        raise ValueError('Explicit causal 5ms history required; camera 5x6 truncation is forbidden')
    if h.ndim == 4 and h.shape[1] == z['time'].shape[1] and h.shape[2:] == (5, 4):
        return h[:, j]
    if h.ndim == 3 and h.shape[1:] == (5, 4) and z['time'].shape[1] == 1:
        return h
    raise ValueError('Expected per-midpoint causal history [B,M,5,4] (single step permits [B,5,4])')


def _validate(c, z, net, jets):
    if any(p.requires_grad for p in net.parameters()):
        raise RuntimeError('The selected probability weights must stay frozen')
    v, g, acc = jets
    B, M = z['time'].shape
    if v.shape != (B, M, 2, 2) or g.shape != (B, M, 2, 2, 2) or acc.shape != v.shape:
        raise ValueError('Expected camera quadrature jets [B,M,fluid,component]')
    h = z['dt']
    if bool((h < 0).any()) or bool((h > DT+1e-10).any()):
        raise ValueError('Internal physical pieces must lie in [0,1ms]')
    if not torch.allclose(h.sum(-1)/DT, z['end']-z['start'], atol=2e-7, rtol=0):
        raise ValueError('Internal pieces must exactly tile each native interval')
    if bool((h[:, 0] <= 0).any()) or bool((z['end'] <= z['start']).any()):
        raise ValueError('Every camera interval must contain a physical piece')
    if not bool(torch.isfinite(v).all() & torch.isfinite(g).all() & torch.isfinite(acc).all()):
        raise ValueError('Nonfinite candidate field query')
    history(z)


def response(c, kind, h):
    """Exact constant-affine particle response on each SI-duration piece."""
    A = core.matrices(c, h, kind)[0]
    I = torch.eye(2, dtype=h.dtype, device=h.device)
    E = torch.matrix_exp(-h[:, None, None, None]*A)
    hp = h.clamp_min(1e-15)[:, None, None, None]
    H = torch.linalg.solve(A, I-E)/hp
    B = torch.linalg.solve(A, I-H)/hp
    H = torch.where((h > 0)[:, None, None, None], H, I)
    B = torch.where((h > 0)[:, None, None, None], B, .5*I)
    return A, E, H, B


def _propagator(c, kind, h):
    A, E, H, B = response(c, kind, h)
    I = torch.eye(2, dtype=h.dtype, device=h.device)
    # State order is particle velocity/U, cumulative displacement/(U*1ms).
    top = torch.cat((E, torch.zeros_like(E)), -1)
    bottom = torch.cat((h[:, None, None, None]/DT*H, I.expand_as(E)), -1)
    F = torch.cat((top, bottom), -2)
    gain = torch.cat((h[:, None, None, None]*H,
                      h[:, None, None, None].square()/DT*B), -2)
    return A, F, gain


def free_distribution(c, z, jets, net, cal, force):
    """Analytic propagation of the learned free-force law to one real endpoint."""
    _validate(c, z, net, jets)
    v, g, acc = jets
    B, M = z['time'].shape
    mean = torch.cat((z['w'], torch.zeros_like(z['w'])), -1)
    covariance = v.new_zeros(B, 4, 4)
    for j in range(M):
        h = z['dt'][:, j]
        A, F, gain = _propagator(c, 'free', h)
        F, gain = F[:, 0], gain[:, 0]
        ctx = core.context(c, {'w': z['w']}, v[:, j], g[:, j])
        fm, fL = net.base.free_parameters(history(z, j), ctx, force)
        affine = core.affine(c, v[:, j], acc[:, j], 'free')[:, 0]/c['physics']['vn_mean_m_s']
        affine = affine+torch.einsum('ij,bj->bi', A[0], fm)
        mean = torch.einsum('bij,bj->bi', F, mean)+torch.einsum('bij,bj->bi', gain, affine)
        noise_gain = gain@A[0]@fL
        covariance = F@covariance@F.transpose(-1, -2)+noise_gain@noise_gain.transpose(-1, -2)
    cov = covariance[:, 2:, 2:]+cal['sigma']['free']**2*torch.eye(2, dtype=v.dtype, device=v.device)
    return mean[:, None, None, 2:], cov[:, None, None], v.new_zeros(B, 1, 1)


def _initial_joint(logq, z, previous=None):
    B = len(logq)
    if previous is None:
        e = logq.new_zeros(B, 2, 2)
        C = torch.eye(2, dtype=logq.dtype, device=logq.device).expand(B, 2, 2, 2)
        q = logq
    else:
        e, C, q = previous['emean'], previous['ecov'], previous['qlogp']
    r = torch.cat((z['w'], torch.zeros_like(z['w'])), -1)[:, None].expand(-1, 2, -1)
    mean = torch.cat((e, r), -1)
    top = torch.cat((C, C.new_zeros(B, 2, 2, 4)), -1)
    cov = torch.cat((top, C.new_zeros(B, 2, 4, 6)), -2)
    return q, mean, cov


def _latent_predict(q, mean, cov, logq, hist, ctx, prevctx, head, fraction):
    """Exact first two moments of (1-f) identity plus f learned 1ms kernel."""
    if bool(((fraction < -1e-8) | (fraction > 1+1e-8)).any()):
        raise ValueError('Fractional latent advance must be between zero and 1ms')
    fraction = fraction.clamp(0, 1)
    kappa, A, b, L = head.parameters_at(hist, ctx, prevctx)
    T = LM.sign_transition(logq, kappa).exp()
    B = len(q)
    I4 = torch.eye(4, dtype=q.dtype, device=q.device).expand(B, 4, 4)
    transform = torch.cat((torch.cat((A, A.new_zeros(B, 2, 4)), -1),
                           torch.cat((A.new_zeros(B, 4, 2), I4), -1)), -2)
    shift = torch.cat((b, b.new_zeros(B, 4)), -1)
    jump_mean = torch.einsum('bij,bpj->bpi', transform, mean)+shift[:, None]
    jump_cov = transform[:, None]@cov@transform[:, None].transpose(-1, -2)
    Q = L@L.transpose(-1, -2)
    Q = torch.cat((torch.cat((Q, Q.new_zeros(B, 2, 4)), -1), Q.new_zeros(B, 4, 6)), -2)
    jump_cov = jump_cov+Q[:, None]
    eye = torch.eye(2, dtype=q.dtype, device=q.device)
    stay_weight = q.exp()[:, :, None]*((1-fraction)[:, None, None]*eye[None])
    jump_weight = q.exp()[:, :, None]*(fraction[:, None, None]*T)
    ww = torch.cat((stay_weight, jump_weight), 1)
    mm = torch.cat((mean, jump_mean), 1)
    cc = torch.cat((cov, jump_cov), 1)
    mass = ww.sum(1)
    ww = ww/mass[:, None].clamp_min(torch.finfo(q.dtype).tiny)
    next_mean = torch.einsum('bpq,bpi->bqi', ww, mm)
    diff = mm[:, :, None]-next_mean[:, None]
    next_cov = torch.einsum('bpq,bpqij->bqij', ww, cc[:, :, None]+diff[..., :, None]*diff[..., None, :])
    return mass.clamp_min(torch.finfo(q.dtype).tiny).log(), next_mean, .5*(next_cov+next_cov.transpose(-1, -2))


def _particle_piece(c, q, mean, covariance, mu, geoL, v, acc, h, force, degree):
    """GH conditional propagation; return Gaussian state and final mixture."""
    eps, logw = ob.nodes(degree, str(v.device), v.dtype)
    Cee = covariance[..., :2, :2]
    Le = LM._cholesky(Cee)
    e = mean[..., None, :2]+torch.einsum('bqij,nj->bqni', Le, eps)
    cross = covariance[..., 2:, :2]
    gain_e = torch.cholesky_solve(cross.transpose(-1, -2), Le).transpose(-1, -2)
    cond_mean = mean[..., None, 2:]+torch.einsum('bqij,bqnj->bqni', gain_e, e-mean[..., None, :2])
    cond_cov = covariance[..., 2:, 2:]-gain_e@cross.transpose(-1, -2)
    cond_cov = .5*(cond_cov+cond_cov.transpose(-1, -2))
    delta = mu[:, :, None]+torch.einsum('bqij,bqnj->bqni', geoL, e)
    eta = delta/(c['hr']**2+delta.square().sum(-1, keepdim=True)).sqrt()
    A, F, response_gain = _propagator(c, 'cap', h)
    affine = core.affine(c, v, acc, 'cap')/c['physics']['vn_mean_m_s']
    fm, fcov = v.new_tensor(force['cap']['mean']), v.new_tensor(force['cap']['covariance'])
    affine = affine+torch.einsum('qij,j->qi', A, fm)[None]
    tension = c['tension_twice_N']/(c['constants']['mass']*c['physics']['vn_mean_m_s'])
    acceleration = affine[:, :, None]+tension*eta
    r = torch.einsum('bqij,bqnj->bqni', F, cond_mean)+torch.einsum('bqij,bqnj->bqni', response_gain, acceleration)
    force_gain = response_gain@A
    rcov = F@cond_cov@F.transpose(-1, -2)+force_gain@fcov@force_gain.transpose(-1, -2)
    nodes = torch.cat((e, r), -1)
    node_weights = logw.exp()
    new_mean = torch.einsum('n,bqni->bqi', node_weights, nodes)
    centered = nodes-new_mean[:, :, None]
    new_cov = torch.einsum('n,bqni,bqnj->bqij', node_weights, centered, centered)
    embedded = torch.cat((rcov.new_zeros(len(v), 2, 2, 6),
                         torch.cat((rcov.new_zeros(len(v), 2, 4, 2), rcov), -1)), -2)
    new_cov = new_cov+embedded
    mixture = dict(pred=r[..., 2:], cov=rcov[..., None, 2:, 2:].expand(-1, -1, len(eps), -1, -1),
                   lw=q[:, :, None]+logw, e=e)
    return new_mean, .5*(new_cov+new_cov.transpose(-1, -2)), mixture


def captured_distribution(c, z, jets, net, cal, force, degree=12,
                          previous=None, reset=None, update=True):
    """One native camera interval, score-before-update causal q/e filtering."""
    _validate(c, z, net, jets)
    v, g, acc = jets
    B, M = z['time'].shape
    reset = torch.ones(B, dtype=torch.bool, device=v.device) if previous is None else reset
    if reset is None:
        reset = torch.zeros(B, dtype=torch.bool, device=v.device)
    previous_ctx = previous['context'] if previous is not None else None
    previous_time = previous['time'] if previous is not None else None
    q = mean = covariance = last = lastctx = lastlogq = lasttime = None
    for j in range(M):
        active = z['dt'][:, j] > 0
        if not bool(active.any()):
            continue
        hist = history(z, j)
        ctx = core.context(c, {'w': z['w']}, v[:, j], g[:, j])
        logq, mu, geoL = ob.latent(net.base, hist, ctx, cal)
        if j == 0:
            q0, mean0, cov0 = _initial_joint(logq, z)
            if previous is None:
                q, mean, covariance = q0, mean0, cov0
            else:
                q, mean, covariance = _initial_joint(logq, z, previous)
                # Reset rows are supplied a zero advance; their result is then
                # replaced by the exact source-law segment-start distribution.
                fraction = torch.where(reset, torch.zeros_like(z['time'][:, 0]), z['time'][:, 0]-previous_time)
                q, mean, covariance = _latent_predict(q, mean, covariance, logq, hist, ctx, previous_ctx, net.head, fraction)
                q = torch.where(reset[:, None], q0, q)
                mean = torch.where(reset[:, None, None], mean0, mean)
                covariance = torch.where(reset[:, None, None, None], cov0, covariance)
        else:
            fraction = torch.where(active, z['time'][:, j]-lasttime, torch.zeros_like(lasttime))
            nq, nm, nc = _latent_predict(q, mean, covariance, logq, hist, ctx, lastctx, net.head, fraction)
            q = torch.where(active[:, None], nq, q)
            mean = torch.where(active[:, None, None], nm, mean)
            covariance = torch.where(active[:, None, None, None], nc, covariance)
        nm, nc, mixture = _particle_piece(c, q, mean, covariance, mu, geoL, v[:, j], acc[:, j], z['dt'][:, j], force, degree)
        mean = torch.where(active[:, None, None], nm, mean)
        covariance = torch.where(active[:, None, None, None], nc, covariance)
        if last is None:
            last, lastctx, lastlogq, lasttime = mixture, ctx, logq, z['time'][:, j]
        else:
            last = {k: torch.where(active.reshape((B,)+(1,)*(a.ndim-1)), a, last[k]) for k, a in mixture.items()}
            lastctx = torch.where(active[:, None], ctx, lastctx)
            lastlogq = torch.where(active[:, None], logq, lastlogq)
            lasttime = torch.where(active, z['time'][:, j], lasttime)
    pred, cov, lw = last['pred'], last['cov'], last['lw']
    cov = cov+cal['sigma']['cap']**2*torch.eye(2, dtype=v.dtype, device=v.device)
    if not update:
        return pred, cov, lw, None
    # This is the only occurrence of the current actual endpoint in the model.
    log_joint = lw+LM.emission_log_prob(pred, cov, z['target'])
    q_evidence = torch.logsumexp(log_joint, -1)
    evidence = torch.logsumexp(q_evidence, -1)
    conditional = torch.softmax(log_joint, -1)
    emean = (conditional[..., None]*last['e']).sum(2)
    diff = last['e']-emean[:, :, None]
    ecov = torch.einsum('bqn,bqni,bqnj->bqij', conditional, diff, diff)
    ecov = .5*(ecov+ecov.transpose(-1, -2))+POSTERIOR_JITTER*torch.eye(2, dtype=v.dtype, device=v.device)
    state = dict(qlogp=q_evidence-evidence[:, None], emean=emean, ecov=ecov,
        base_qlogp=lastlogq, context=lastctx, time=lasttime, log_evidence=evidence)
    return pred, cov, lw, state


def _metrics(c, z, pred, cov, lw, nll):
    p = lw.exp()
    mean = (p[..., None]*pred).sum((1, 2))
    trace = cov.diagonal(dim1=-2, dim2=-1).sum(-1)
    spread2 = (p*(trace+(pred-mean[:, None, None]).square().sum(-1))).sum((1, 2)).clamp_min(0)
    scale = c['physics']['vn_mean_m_s']*DT*1e6
    return dict(nll=nll/(z['end']-z['start']), native_nll=nll,
                coverage=(core.coverage(pred, cov, lw, z['target']) <= 4.605170186).to(nll),
                spread=spread2.sqrt()*scale, squared_error=(z['target']-mean).square().sum(-1)*scale**2)


def _record(parts, z, values, valid=None):
    for i, pid in enumerate(z['pid'].tolist()):
        if valid is not None and not bool(valid[i]):
            continue
        row = parts.setdefault(str(pid), dict(count=0, nll=0., native_nll=0., coverage=0., spread=0., squared_error=0.))
        row['count'] += 1
        for key, tensor in values.items():
            row[key] += float(tensor[i].detach())


def _sequence_packs(z, selected, batch_pids):
    pids = z['pid'][selected].unique(sorted=True)
    for pp in pids.split(batch_pids):
        sequences = []
        for pid in pp:
            ix = torch.where(selected & (z['pid'] == pid))[0]
            ix = ix[torch.argsort(z['start'][ix], stable=True)]
            sequences.append(ix)
        length = max(len(ix) for ix in sequences)
        filler = int(sequences[0][0])
        index = torch.full((len(pp), length), filler, dtype=torch.long, device=z['pid'].device)
        valid = torch.zeros_like(index, dtype=torch.bool)
        reset = torch.ones_like(valid)
        for b, ix in enumerate(sequences):
            index[b, :len(ix)] = ix
            valid[b, :len(ix)] = True
            if len(ix) > 1:
                reset[b, 1:len(ix)] = ((z['segment'][ix[1:]] != z['segment'][ix[:-1]]) |
                    ((z['start'][ix[1:]]-z['end'][ix[:-1]]).abs() > 1e-7))
        yield index, valid, reset


def score(c, z, net, cal, force, jets, group, degree=12, metrics=False,
          backward=False, batch=64, batch_pids=8, gradient_weight=.08):
    """Equal states/PIDs/intervals NLL/ms; optional batch backward.

    jets(authorized_subset) must return v,g,acc and may query independent field
    coefficient leaves. No held-out row is used as padding. An outer caller
    gates test/H access until all inversion snapshots have been frozen.
    """
    w = weights(z, group)
    out = dict(total=0., native_nll=dict(total=0.),
        units=dict(total_free_cap='NLL/ms; each native row divided by its duration, then PID/state averaged',
                   native_nll='NLL per native endpoint; equal PID/state weight',
                   spread='micrometre; sqrt(trace predictive displacement covariance)',
                   rmse_um='micrometre; sqrt(mean squared 2D predictive-mean displacement error)',
                   target='dimensionless displacement divided by U*0.001s'))
    allparts = {}
    for kind, name in ((0, 'free'), (1, 'cap')):
        selected = (w > 0) & (z['state'] == kind)
        total, native, mass, parts = 0., 0., float(w[selected].sum()), {}
        if kind == 0:
            for ix in torch.where(selected)[0].split(batch):
                zz = subset(z, ix)
                pred, cov, lw = free_distribution(c, zz, jets(zz), net, cal, force)
                nll = core.scores(pred, cov, lw, zz['target'])
                loss = (w[ix]*nll/(zz['end']-zz['start'])).sum()
                if backward:
                    (gradient_weight*loss).backward()
                total += float(loss.detach())
                native += float((w[ix]*nll).detach().sum())
                if metrics:
                    _record(parts, zz, _metrics(c, zz, pred, cov, lw, nll))
        else:
            for index, valid, reset in _sequence_packs(z, selected, batch_pids):
                B, nt = index.shape
                zzfull = subset(z, index.flatten())
                vv, gg, aa = jets(zzfull)
                previous, losses = None, []
                for t in range(nt):
                    take = torch.arange(B, device=index.device)*nt+t
                    zz = subset(zzfull, take)
                    pred, cov, lw, previous = captured_distribution(c, zz, (vv[take], gg[take], aa[take]),
                        net, cal, force, degree, previous, reset[:, t], True)
                    nll = -previous['log_evidence']
                    ww = w[index[:, t]]*valid[:, t]
                    losses.append((ww*nll/(zz['end']-zz['start'])).sum())
                    native += float((ww*nll).detach().sum())
                    if metrics:
                        _record(parts, zz, _metrics(c, zz, pred, cov, lw, nll), valid[:, t])
                loss = torch.stack(losses).sum()
                if backward:
                    (gradient_weight*loss).backward()
                total += float(loss.detach())
        out[name] = total/mass
        out['total'] += total
        out['native_nll'][name] = native/mass
        out['native_nll']['total'] += native
        if metrics:
            for row in parts.values():
                for key in ('nll', 'native_nll', 'coverage', 'spread', 'squared_error'):
                    row[key] /= row['count']
                row['rmse_um'] = math.sqrt(row['squared_error'])
            allparts[name] = parts
    if metrics:
        out['per_pid'] = allparts
        for key in ('coverage', 'spread'):
            out[key] = {name: float(np.mean([r[key] for r in parts.values()])) for name, parts in allparts.items()}
        out['rmse_um'] = {name: math.sqrt(float(np.mean([r['squared_error'] for r in parts.values()]))) for name, parts in allparts.items()}
    return out
