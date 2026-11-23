"""Physically anchored free/captured transition densities.

Only a causal four-channel particle history and the existing 17-dimensional
local physical context reach these networks. Position, time, PID and target
displacement are not network inputs. A free-particle head predicts a bounded
distribution of unresolved force; the physical affine particle propagator
continues to determine its displacement mean and fluid derivatives.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import sys

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
PACKAGE = PROJECT / "outputs/FullWindowCalibration_20261026/server_package"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))
from counterflow.observation import Density
from counterflow_full import core

ARMS = ("cap_retrain", "independent", "shared_gru")
FREE_MEAN_BOUND = 2.0
FREE_LOG_DIAGONAL_BOUND = math.log(3.0)
FREE_OFF_DIAGONAL_BOUND = 1.0
scores = core.scores
coverage = core.coverage
context = core.context


def force_anchor(force, like):
    """Return the fixed global unresolved-free-force mean and Cholesky factor."""
    mean = torch.as_tensor(force["free"]["mean"], dtype=like.dtype, device=like.device)
    covariance = torch.as_tensor(force["free"]["covariance"], dtype=like.dtype, device=like.device)
    if mean.shape != (2,) or covariance.shape != (2, 2):
        raise ValueError("Expected a two-dimensional fixed free-force anchor")
    return mean, torch.linalg.cholesky(covariance)


class TransitionDensity(Density):
    """Three matched retraining arms, with the original captured API intact.

    cap_retrain: original captured network, fixed free distribution.
    independent: private captured/free GRUs, MLPs and output heads.
    shared_gru: one shared causal-history GRU, private context MLPs and heads.
    No module aliases are registered, so state_dict has no duplicate shared
    parameters. All actual parameters are initially trainable for retraining.
    """

    def __init__(self, arm, base_net):
        if arm not in ARMS:
            raise ValueError("Unknown transition arm: " + str(arm))
        # Do not randomly initialize/reseed the inherited captured network.
        nn.Module.__init__(self)
        self.arm = arm
        for name in ("gru", "head", "sign", "output"):
            setattr(self, name, copy.deepcopy(getattr(base_net, name)))
        self.register_buffer("cm", base_net.cm.detach().clone())
        self.register_buffer("cs", base_net.cs.detach().clone())
        if arm != "cap_retrain":
            if arm == "independent":
                self.free_gru = copy.deepcopy(base_net.gru)
            self.free_head = copy.deepcopy(base_net.head)
            # Zero initialization has no effect on the caller's shuffle RNG.
            with torch.random.fork_rng(devices=[]):
                self.free_output = nn.Linear(32, 5).to(base_net.cm)
                nn.init.zeros_(self.free_output.weight)
                nn.init.zeros_(self.free_output.bias)
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        self.train()

    @staticmethod
    def _validate_inputs(h, ctx):
        if h.ndim != 3 or h.shape[-1] != 4 or ctx.shape != (len(h), 17):
            raise ValueError("Network accepts causal history [B,T,4] and physical context [B,17] only")

    def encode(self, h, c):
        self._validate_inputs(h, c)
        return super().encode(h, c)

    def encode_free(self, h, ctx):
        if self.arm == "cap_retrain":
            raise RuntimeError("cap_retrain has no learned free encoder")
        self._validate_inputs(h, ctx)
        encoder = self.free_gru if self.arm == "independent" else self.gru
        _, hidden = encoder(h)
        inputs = torch.cat((hidden[-1], h[:, -1], (ctx - self.cm) / self.cs), -1)
        return self.free_head(inputs)

    def free_standardized_parameters(self, h, ctx):
        """Bounded mean m and triangular S in old-force whitened coordinates."""
        self._validate_inputs(h, ctx)
        if self.arm == "cap_retrain":
            mean = h.new_zeros(len(h), 2)
            factor = torch.eye(2, dtype=h.dtype, device=h.device).expand(len(h), -1, -1)
            return mean, factor
        raw = self.free_output(self.encode_free(h, ctx))
        mean = FREE_MEAN_BOUND * torch.tanh(raw[:, :2])
        d0 = torch.exp(FREE_LOG_DIAGONAL_BOUND * torch.tanh(raw[:, 2]))
        off = FREE_OFF_DIAGONAL_BOUND * torch.tanh(raw[:, 3])
        d1 = torch.exp(FREE_LOG_DIAGONAL_BOUND * torch.tanh(raw[:, 4]))
        factor = torch.stack((torch.stack((d0, torch.zeros_like(d0)), -1),
                              torch.stack((off, d1), -1)), -2)
        return mean, factor

    def free_parameters(self, h, ctx, force):
        """Return actual dimensionless-force mean/L, not displacement output."""
        mean0, factor0 = force_anchor(force, h)
        mean, factor = self.free_standardized_parameters(h, ctx)
        return mean0 + mean @ factor0.T, factor0[None] @ factor

    def parameter_audit(self):
        named = dict(self.named_parameters())
        counts = {name: p.numel() for name, p in named.items()}
        if self.arm != "cap_retrain":
            if {id(p) for p in self.head.parameters()} & {id(p) for p in self.free_head.parameters()}:
                raise AssertionError("Captured/free conditional MLPs must remain private")
        if self.arm == "independent":
            if {id(p) for p in self.gru.parameters()} & {id(p) for p in self.free_gru.parameters()}:
                raise AssertionError("Independent arm encoders unexpectedly share parameters")
        return dict(arm=self.arm, total_parameters=sum(counts.values()), per_tensor=counts,
                    parameter_ids_unique=len(named) == len({id(p) for p in named.values()}),
                    shared_history_gru=self.arm == "shared_gru", private_conditional_mlps=True,
                    input_contract="h[B,T,4], context[B,17]; no explicit x/time/PID/target displacement",
                    normalization_buffers_frozen=True)


def load_model(arm, base_net, state=None, device=None):
    model = TransitionDensity(arm, base_net)
    if state is not None:
        model.load_state_dict(state, strict=True)
    if device is not None:
        model.to(device)
    model.parameter_audit()
    return model


def free_force_nll(net, h, ctx, force_truth, force):
    """Normalized NLL of the actual dimensionless force, including anchor Jacobian.

    force_truth must be old non-target training/validation labels, never a
    target-inversion residual fitted separately for each particle.
    """
    mean, factor = net.free_parameters(h, ctx, force)
    if force_truth.shape != mean.shape:
        raise ValueError("Free-force labels must have shape [B,2]")
    whitened = torch.linalg.solve_triangular(factor, (force_truth - mean)[..., None], upper=False)[..., 0]
    return .5 * whitened.square().sum(-1) + factor.diagonal(dim1=-2, dim2=-1).log().sum(-1) + math.log(2 * math.pi)


def distribution(c, z, v, g, acc, kind, net, cal, force, degree):
    """Legacy-compatible normalized endpoint law, with optional learned free force.

    The free force covariance replaces the old force covariance. The calibrated
    observation noise is added exactly once. Free endpoints are analytic 2-D
    Gaussians; degree affects only the inherited captured quadrature.
    """
    if kind == "cap" or (kind == "free" and net.arm == "cap_retrain"):
        return core.distribution(c, z, v, g, acc, kind, net, cal, force, degree)
    if kind != "free":
        raise ValueError("Unknown particle state: " + str(kind))
    ctx = core.context(c, z, v, g)
    mean, factor = net.free_parameters(z["h"], ctx, force)
    U = c["physics"]["vn_mean_m_s"]
    _, H, B, M = core.matrices(c, v, "free")
    base = torch.einsum("qij,bj->bqi", H, z["w"])
    base = base + core.DT / U * torch.einsum("qij,bqj->bqi", B, core.affine(c, v, acc, "free"))
    base = base + torch.einsum("qij,bj->bqi", M, mean)
    propagated_factor = M[0][None] @ factor
    covariance = propagated_factor @ propagated_factor.transpose(-1, -2)
    covariance = covariance + torch.eye(2, dtype=v.dtype, device=v.device) * cal["sigma"]["free"] ** 2
    return base[:, :, None], covariance[:, None, None], v.new_zeros(len(v), 1, 1)
