"""Frozen marginal law plus a learned captured-latent Markov transition.

The latent is (q, e), where e is the two-dimensional standard-normal residual
of the *calibrated*, q-conditional captured displacement-geometry law.  The
particle propagator and observation noise are unchanged.  Target filtering is
an assumed-density approximation: after each endpoint, retain a separate
Gaussian in e for each q.  Gauss--Hermite integration approximates the endpoint
evidence and its posterior moments; it is not an exact nonlinear HMM filter.

This module never loads data, checkpoints, target truth, or evaluation groups.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
PACKAGE = BASE / "outputs/FullWindowCalibration_20261026/server_package"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))
from counterflow import observation as ob
from counterflow_full import core

MODES = ("iid", "reset", "temporal")
POSTERIOR_COVARIANCE_JITTER = 1e-10


def bounded_kappa(raw):
    """Exactly clamp(raw, 0, 12), with an explicit boundary derivative.

    The zero initialization uses the admissible right derivative 1 at raw=0.
    Derivatives are 0 for raw<0, 1 for 0<=raw<12, and 0 for raw>=12.
    Values, bounds, the probability kernel, and the iid initialization match
    the original clamp exactly; only the lower-bound subgradient is fixed.
    Explicit where branches avoid torch-version-dependent clamp autograd.
    """
    return torch.where(raw < 0., torch.zeros_like(raw),
                       torch.where(raw >= 12., torch.full_like(raw, 12.), raw))


class LatentTransitionHead(nn.Module):
    """A 66 -> 32 -> 10 head with an exactly iid zero-output initialization.

    ``base`` is a frozen independent TransitionDensity.  It is held by a
    non-registered reference, so this module's state_dict/parameters contain
    only the transition head.  Context and history input gradients through
    base.encode remain enabled.  ``state`` is a head-only state_dict.
    Calling .to(...)/.double() also moves this referenced base, even though it
    is excluded from the head checkpoint.  The head initially uses the base's
    device and floating dtype.  Do not reuse that same base in a different
    dtype concurrently; deepcopy(base) and rebuild a head for validation.
    """

    def __init__(self, base, state=None, freeze=False):
        super().__init__()
        object.__setattr__(self, "_base", base)
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        base.eval()
        self.network = nn.Sequential(nn.Linear(66, 32), nn.SiLU(), nn.Linear(32, 10))
        self.to(device=base.cm.device, dtype=base.cm.dtype)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        if state is not None:
            self.load_state_dict(state, strict=True)
        if freeze:
            self.freeze()

    def freeze(self):
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    def _apply(self, fn, recurse=True):
        # Non-registered base must follow floating dtype/device conversions.
        self._base._apply(fn, recurse=recurse)
        return super()._apply(fn, recurse=recurse)

    def parameters_at(self, h, ctx, prevctx):
        """Return kappa[B], A[B,2,2], b[B,2], L[B,2,2].

        Inputs use current causal h[B,5,4] and previous/current context[B,17].
        No endpoint, PID, absolute position, time, or truth is a network input.
        """
        if ctx.shape != (len(h), 17) or prevctx.shape != ctx.shape:
            raise ValueError("Expected current and previous context [B,17]")
        base = self._base
        if any(parameter.requires_grad for parameter in base.parameters()):
            raise RuntimeError("The marginal network must remain frozen")
        encoded = base.encode(h, ctx)
        inputs = torch.cat((encoded, (prevctx-base.cm)/base.cs, (ctx-base.cm)/base.cs), -1)
        raw = self.network(inputs)
        kappa = bounded_kappa(raw[:, 0])
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


def sign_transition(logq, kappa):
    """Log T(q_current | q_previous), shape [B,2,2], row-normalized."""
    eye = torch.eye(2, device=logq.device, dtype=logq.dtype)
    return torch.log_softmax(logq[:, None, :]+kappa[:, None, None]*eye[None], -1)


def whiten_latent(base, h, ctx, y, q, cal):
    """Calibrated p0 quantities and labeled e, without detaching input grads."""
    logq, mu, L = ob.latent(base, h, ctx, cal)
    ids = torch.arange(len(h), device=h.device)
    if y.shape != (len(h), 2) or q.shape != (len(h),):
        raise ValueError("Expected y[B,2] and q[B] labels")
    selected_L = L[ids, q]
    e = torch.linalg.solve_triangular(selected_L, (y-mu[ids, q])[..., None], upper=False)[..., 0]
    return logq, mu, L, e, selected_L


def supervised_pair_nll(base, head, prev_h, prev_ctx, prev_y, prev_q,
                        h, ctx, y, q, cal):
    """Return per-pair NLL[B] and diagnostics for continuous captured pairs.

    The density is with respect to current physical dimensionless y=delta/r,
    so the current calibrated L Jacobian is included exactly once.  Previous
    y/q is conditioning information, not another likelihood factor.  Caller
    is responsible for PID continuity, frame difference one, and train split.
    """
    _, _, _, prev_e, _ = whiten_latent(base, prev_h, prev_ctx, prev_y, prev_q, cal)
    logq, _, _, current_e, current_L = whiten_latent(base, h, ctx, y, q, cal)
    kappa, A, b, L = head.parameters_at(h, ctx, prev_ctx)
    mean = torch.einsum("bij,bj->bi", A, prev_e)+b
    white = torch.linalg.solve_triangular(L, (current_e-mean)[..., None], upper=False)[..., 0]
    continuous = .5*white.square().sum(-1)+L.diagonal(dim1=-2, dim2=-1).log().sum(-1)+math.log(2*math.pi)
    jacobian = current_L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    ids = torch.arange(len(h), device=h.device)
    categorical = -sign_transition(logq, kappa)[ids, prev_q, q]
    nll = categorical+continuous+jacobian
    return nll, dict(categorical=categorical, continuous=continuous, jacobian=jacobian,
                     kappa=kappa, A=A, b=b, L=L, previous_e=prev_e, current_e=current_e)


def _standard_state(logq):
    count = len(logq)
    return (logq, logq.new_zeros(count, 2, 2),
            torch.eye(2, device=logq.device, dtype=logq.dtype).expand(count, 2, 2, 2))


def _cholesky(covariance):
    """Cholesky with a small, documented fallback only for numerical failure."""
    covariance = .5*(covariance+covariance.transpose(-1, -2))
    factor, info = torch.linalg.cholesky_ex(covariance)
    if bool((info != 0).any()):
        eye = torch.eye(2, device=covariance.device, dtype=covariance.dtype)
        jitter = max(POSTERIOR_COVARIANCE_JITTER, 20*torch.finfo(covariance.dtype).eps)
        covariance = covariance+(info != 0)[..., None, None]*jitter*eye
        factor = torch.linalg.cholesky(covariance)
    return factor


def _predict_state(logq, h, ctx, prevctx, prevstate, head, mode, reset_mask):
    first_logq, first_mean, first_cov = _standard_state(logq)
    if mode == "iid" or prevstate is None:
        return first_logq, first_mean, first_cov
    if prevctx is None:
        raise ValueError("A continuing segment requires previous context")
    if mode == "reset":
        previous_logq, previous_mean, previous_cov = _standard_state(prevstate["base_qlogp"])
    else:
        previous_logq, previous_mean, previous_cov = [prevstate[key] for key in ("qlogp", "emean", "ecov")]
    kappa, A, b, innovation = head.parameters_at(h, ctx, prevctx)
    transition = sign_transition(logq, kappa)
    joint = previous_logq[:, :, None]+transition
    next_logq = torch.logsumexp(joint, 1)
    mix = (joint-next_logq[:, None, :]).exp()
    component_mean = torch.einsum("bij,bqj->bqi", A, previous_mean)+b[:, None]
    component_cov = A[:, None]@previous_cov@A[:, None].transpose(-1, -2)
    component_cov = component_cov+(innovation@innovation.transpose(-1, -2))[:, None]
    # Axis p: previous q; axis q: current q.  Moment matching merges p only.
    next_mean = torch.einsum("bpq,bpi->bqi", mix, component_mean)
    centered = component_mean[:, :, None]-next_mean[:, None]
    next_cov = torch.einsum("bpq,bpqij->bqij", mix,
                           component_cov[:, :, None]+centered[..., :, None]*centered[..., None, :])
    if reset_mask is not None:
        if reset_mask.shape != (len(logq),) or reset_mask.dtype != torch.bool:
            raise ValueError("reset_mask must be bool[B]")
        next_logq = torch.where(reset_mask[:, None], first_logq, next_logq)
        next_mean = torch.where(reset_mask[:, None, None], first_mean, next_mean)
        next_cov = torch.where(reset_mask[:, None, None, None], first_cov, next_cov)
    return next_logq, next_mean, next_cov


def emission_log_prob(pred, cov, target):
    """Conditional endpoint log density [B,2,M], no latent weights.

    The physical captured covariance is constant over quadrature nodes.  Its
    Cholesky is evaluated once per row/sign, not D**2 redundant times.
    """
    factor = torch.linalg.cholesky(cov[:, :, 0])
    residual = torch.linalg.solve_triangular(factor[:, :, None], (target[:, None, None]-pred)[..., None], upper=False)[..., 0]
    logdet = factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    return -.5*residual.square().sum(-1)-logdet[:, :, None]-math.log(2*math.pi)


def step(c, z, v, g, acc, base, head, cal, force, degree,
         prevstate=None, prevctx=None, mode="temporal", reset_mask=None, update=True):
    """One vectorized captured step; samples in the batch are distinct PIDs.

    Returns ``pred[B,2,D**2,2], cov[B,2,D**2,2,2], lw[B,2,D**2], state``.
    These three predictive tensors NEVER use z['target']; score and coverage
    must use them before the posterior is propagated.  With update=True, the
    observed current endpoint is used only to produce the returned state:
    qlogp[B,2], emean[B,2,2], ecov[B,2,2,2], plus base_qlogp and context.
    State['log_evidence'][B] is the same single predictive endpoint factor.
    With update=False, target need not exist and state is None.

    First segment rows (prevstate=None or reset_mask=True) use original p0.
    iid always uses p0; reset uses previous p0 rather than endpoint posterior;
    temporal propagates the ADF posterior.  The caller aligns previous state
    and context by PID and resets at gaps, state switches, and segment starts.
    No state tensor is detached here: target inversion retains full causal
    derivatives through all past evidence, context, and physical propagation.
    """
    if mode not in MODES:
        raise ValueError("Unknown latent mode: "+str(mode))
    if any(parameter.requires_grad for parameter in base.parameters()):
        raise RuntimeError("The base probability model must be frozen")
    ctx = core.context(c, z, v, g)
    logq, mu, geometry_factor = ob.latent(base, z["h"], ctx, cal)
    qprior, emean, ecov = _predict_state(logq, z["h"], ctx, prevctx, prevstate, head, mode, reset_mask)
    eps, node_logw = ob.nodes(degree, str(v.device), v.dtype)
    if mode == "iid":
        # Exact old arithmetic, including quadrature layout and normalization.
        # This branch permits bitwise reuse of the previous independent runs.
        pred, cov, lw = core.distribution(c, z, v, g, acc, "cap", base, cal, force, degree)
        e = eps[None, None].expand(len(v), 2, -1, -1)
    else:
        efactor = _cholesky(ecov)
        e = emean[:, :, None]+torch.einsum("bqij,nj->bqni", efactor, eps)
        delta = mu[:, :, None]+torch.einsum("bqij,bqnj->bqni", geometry_factor, e)
        tension_direction = delta/(c["hr"]**2+delta.square().sum(-1, keepdim=True)).sqrt()
        U = c["physics"]["vn_mean_m_s"]
        _, H, B, M = core.matrices(c, v, "cap")
        physical = torch.einsum("qij,bj->bqi", H, z["w"])
        physical = physical+core.DT/U*torch.einsum("qij,bqj->bqi", B, core.affine(c, v, acc, "cap"))
        force_mean = v.new_tensor(force["cap"]["mean"])
        force_cov = v.new_tensor(force["cap"]["covariance"])
        physical = physical+torch.einsum("qij,j->qi", M, force_mean)[None]
        covariance = M@force_cov@M.transpose(-1, -2)
        covariance = covariance+torch.eye(2, device=v.device, dtype=v.dtype)*cal["sigma"]["cap"]**2
        gain = core.DT*c["tension_twice_N"]/(c["constants"]["mass"]*U)*B
        pred = physical[:, :, None]+torch.einsum("qij,bqnj->bqni", gain, tension_direction)
        cov = covariance[None, :, None].expand(len(v), -1, len(eps), -1, -1)
        lw = qprior[:, :, None]+node_logw
    if not update:
        return pred, cov, lw, None
    if "target" not in z:
        raise ValueError("Updating a filter requires the current observed endpoint")
    log_joint = lw+emission_log_prob(pred, cov, z["target"])
    q_evidence = torch.logsumexp(log_joint, -1)
    evidence = torch.logsumexp(q_evidence, -1)
    posterior_q = q_evidence-evidence[:, None]
    conditional_w = torch.softmax(log_joint, -1)
    posterior_mean = (conditional_w[..., None]*e).sum(2)
    centered = e-posterior_mean[:, :, None]
    posterior_cov = torch.einsum("bqn,bqni,bqnj->bqij", conditional_w, centered, centered)
    posterior_cov = .5*(posterior_cov+posterior_cov.transpose(-1, -2))
    posterior_cov = posterior_cov+POSTERIOR_COVARIANCE_JITTER*torch.eye(2, device=v.device, dtype=v.dtype)
    state = dict(qlogp=posterior_q, emean=posterior_mean, ecov=posterior_cov,
                 base_qlogp=logq, context=ctx, log_evidence=evidence,
                 predictive_qlogp=qprior, predictive_emean=emean, predictive_ecov=ecov)
    return pred, cov, lw, state
