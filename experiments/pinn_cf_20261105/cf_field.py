"""Continuous PINN field on the frozen CF channel geometry.

All times accepted by the field are physical seconds. Observation ``z['time']``
is in milliseconds relative to the case start and is converted explicitly.
The two fields share a time MLP but have separate output coefficients; there is
no trainable initial condition, closure, forcing, or process increment.

Frozen B is transferred by its complete L2 push-forward covariance, not by
truncating source coordinates or applying source whitening to wall coefficients.
The normal source prior has exactly two temporal kernels. A stable spatial SVD
and simultaneous covariance diagonalization reduce its exact quadratic form to
325 independent 100-by-100 temporal solves. All singular directions are retained.
"""
from __future__ import annotations

import importlib.util
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
RUNTIME = BASE / 'outputs/JointProbabilityPhysics_20261103/cf_runtime'
DT = .001
DURATION = .100


@lru_cache(maxsize=1)
def runtime():
    spec = importlib.util.spec_from_file_location('pinn_cf_frozen_transfer', RUNTIME / 'b_transfer.py')
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def wall_normalization(c, op):
    """Mass-normalize and diagonalize the original channel gradient energy.

    With area-normalized quadrature M=integral V.T V/U^2 and K=integral G.T
    G/U^2, the existing decoder D satisfies D.T M D=I. Diagonalizing D.T K D
    gives Q and k^2. Set raw a=(D Q/sqrt(2))q, so every normalized coefficient
    q has velocity energy U^2/2, exactly the sine/cosine convention of the
    simulation PINN. Orthogonalization is solely fixed geometry, never data fit.
    """
    U = float(c['physics']['vn_mean_m_s'])
    weights = op.V.new_tensor(np.tile(np.polynomial.legendre.leggauss(34)[1]/2/52, 52))
    M = torch.einsum('fpm,p,fpn->fmn', op.V, weights.repeat_interleave(2), op.V)/U**2
    K = torch.einsum('fpm,p,fpn->fmn', op.G, weights.repeat_interleave(4), op.G)/U**2
    old = c['decode'].to(op.V)
    identity = torch.eye(op.m).to(op.V)
    old_mass_error = float((old.transpose(-1, -2)@M@old-identity).abs().max())
    if old_mass_error > 1e-10:
        raise ValueError('Original channel decoder is not mass orthonormal')
    stiffness = old.transpose(-1, -2)@K@old
    eig, rotation = torch.linalg.eigh((stiffness+stiffness.transpose(-1, -2))/2)
    if not torch.isfinite(eig).all() or float(eig.min()) <= 0:
        raise ValueError('Expected positive channel gradient energy')
    decode = old@rotation/math.sqrt(2)
    # D_new.T M D_new=I/2 gives the exact inverse without a generic LU solve.
    encode = 2*decode.transpose(-1, -2)@M
    ring = (eig.sqrt()/(2*math.pi/.01)).clamp_min(1)
    normalized_mass = decode.transpose(-1, -2)@M@decode
    normalized_gradient = decode.transpose(-1, -2)@K@decode
    report = dict(
        coordinates='raw a=(D Q/sqrt(2)) q; Q diagonalizes mass-normalized gradient energy',
        velocity_energy_per_q_squared='U^2/2',
        spectral_radius='max(1,sqrt(gradient eigenvalue)/(2*pi/0.01 m))',
        spectral_radius_min=ring.min(-1).values.tolist(), spectral_radius_max=ring.max(-1).values.tolist(),
        original_decoder_mass_error=old_mass_error,
        normalized_mass_half_identity_error=float((normalized_mass-.5*identity).abs().max()),
        normalized_gradient_diagonal_relative=float((normalized_gradient-.5*torch.diag_embed(eig)).norm()/normalized_gradient.norm()),
        decode_inverse_error=float((encode@decode-identity).abs().max()),
        raw_mass_diagonal_min=M.diagonal(dim1=-2, dim2=-1).min(-1).values.tolist(),
        raw_mass_diagonal_max=M.diagonal(dim1=-2, dim2=-1).max(-1).values.tolist(),
        raw_mass_condition=(torch.linalg.eigvalsh(M)[:, -1]/torch.linalg.eigvalsh(M)[:, 0]).tolist(),
        fitting_or_selection=False)
    return decode, encode, ring, report


def wall_ring(c, like):
    return c['pinn_ring'].to(like)


def output_preconditioning(c, projected):
    """Absolute one-ms B spatial scale as an invertible optimizer coordinate.

    Raw coefficient innovations have covariance P diag(sigma^2) P.T. Mapping
    them by the fixed encode gives Cq. An SVD of Aq=encode P diag(sigma) gives
    its unique symmetric SPD root Lq=U diag(s) U.T without squaring condition.
    No singular direction is dropped, jittered, or rescaled. This zero-lag
    transform does not replace the full temporal B covariance in the PDE loss.
    """
    factors, inverses, diagnostics = [], [], []
    for f, std in enumerate((projected.normal_prior.std, projected.std_s)):
        A = c['pinn_encode'][f]@(projected.projection[f]*std)
        U, s, _ = torch.linalg.svd(A, full_matrices=False)
        if not torch.isfinite(s).all() or float(s[-1]) <= 0:
            raise ValueError('B output preconditioner must retain full positive rank')
        L, inverse = (U*s)@U.T, (U/s)@U.T
        covariance = A@A.T
        identity = torch.eye(len(s)).to(A)
        factors.append(L)
        inverses.append(inverse)
        diagnostics.append(dict(
            fluid='normal' if f == 0 else 'superfluid',
            minimum_singular_value=float(s[-1]), maximum_singular_value=float(s[0]),
            condition=float(s[0]/s[-1]),
            covariance_reconstruction_relative=float((L@L.T-covariance).norm()/covariance.norm()),
            symmetry_relative=float((L-L.T).norm()/L.norm()),
            inverse_identity_max_abs=float((inverse@L-identity).abs().max()),
            rms_q_step_per_unit_independent_output=float(s.square().sum().sqrt()),
            numerical_jitter=0., dropped_singular_directions=0, external_scale_multiplier=1.))
    report = dict(
        coordinates='q=Lq diag(1/ring) u; raw a=decode q',
        covariance='Cq=encode P diag(sigma^2) P.T encode.T at zero temporal lag',
        absolute_time_scale_seconds=DT, factors=diagnostics,
        modifies_physical_objective=False, modifies_optimizer_coordinates=True,
        full_temporal_B_loss_unchanged=True, probability_unchanged=True,
        fitting_or_selection=False)
    return torch.stack(factors), torch.stack(inverses), report


class NeuralField(nn.Module):
    def __init__(self, cold, ring, seed=0, duration=DURATION, width=128, depth=3, frequencies=8,
                 decode=None, encode=None, output_preconditioner=None):
        super().__init__()
        if decode is None or encode is None or decode.shape != (2, cold.shape[-1], cold.shape[-1]):
            raise ValueError('The audited mass-normalized channel decoder is required')
        if output_preconditioner is None or output_preconditioner.shape != decode.shape:
            raise ValueError('The full-rank absolute B output preconditioner is required')
        self.register_buffer('cold', cold.detach().clone())
        self.register_buffer('scale', ring.to(cold).reciprocal())
        self.register_buffer('ring', ring.to(cold).clone())
        self.register_buffer('decode', decode.to(cold).detach().clone())
        self.register_buffer('encode', encode.to(cold).detach().clone())
        self.register_buffer('output_preconditioner', output_preconditioner.to(cold).detach().clone())
        self.register_buffer('frequency', torch.arange(1, frequencies+1).to(cold)*math.pi)
        self.duration = float(duration)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.hidden = nn.ModuleList([nn.Linear(1+2*frequencies if i == 0 else width, width).to(cold)
                                         for i in range(depth)])
            self.output = nn.Linear(width, cold.numel()).to(cold)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @property
    def device(self):
        return self.cold.device

    def coefficients_and_dt(self, t_seconds):
        t = t_seconds.to(self.cold).reshape(-1, 1)
        tau, rate = 2*t/self.duration-1, 2/self.duration
        phase = tau*self.frequency
        x = torch.cat((tau, phase.sin(), phase.cos()), 1)
        dx = torch.cat((torch.ones_like(tau)*rate, phase.cos()*self.frequency*rate,
                        -phase.sin()*self.frequency*rate), 1)
        for layer in self.hidden:
            dx = F.linear(dx, layer.weight, None)
            x = layer(x).tanh()
            dx = dx*(1-x.square())
        value = self.output(x).reshape(-1, *self.cold.shape)
        tangent = F.linear(dx, self.output.weight, None).reshape_as(value)
        q = torch.einsum('fij,tfj->tfi', self.output_preconditioner, value*self.scale)
        dq = torch.einsum('fij,tfj->tfi', self.output_preconditioner, tangent*self.scale)
        return (self.cold + torch.einsum('fij,tfj->tfi', self.decode, q),
                torch.einsum('fij,tfj->tfi', self.decode, dq))

    def forward(self, t_seconds):
        return self.coefficients_and_dt(t_seconds)[0]

    def states(self):
        return self(torch.arange(round(self.duration/DT)+1).to(self.cold)*DT)

    def parameter_groups(self, lr=.001):
        return [dict(params=list(self.parameters()), lr=lr, name='entire_field')]

    def normalized_coefficients(self, a):
        return torch.einsum('fij,tfj->tfi', self.encode, a-self.cold)

    def regularization(self, a=None):
        if a is None:
            a = self.states()
        return symmetric_regularization(self.normalized_coefficients(a), self.ring)


class WallResidualPrior(nn.Module):
    """Exact push-forward quadratic energy of the complete frozen B prior.

    A=P diag(sigma)=U diag(s) Vh. In spatial coordinates b=r U/s,
    covariance is T0 tensor (V0 V0.T) + T1 tensor (I-V0 V0.T).
    Diagonalize V0 V0.T=Q diag(lambda) Q.T. Column j of bQ then has
    covariance lambda_j T0+(1-lambda_j) T1. Thus the method is exact up to
    floating-point error and does not select a pseudoinverse cutoff or jitter.
    """
    def __init__(self, projected):
        super().__init__()
        self.intervals = projected.intervals
        self.projection_checks = projected.projection_checks
        self.register_buffer('projection', projected.projection.detach().clone())
        normal = projected.normal_prior
        parameters = normal.parameters_by_ring
        unique = []
        groups = []
        for p in parameters:
            key = (float(p['ell_intervals']), float(p['white_variance_fraction']))
            if key not in unique:
                unique.append(key)
            groups.append(unique.index(key))
        if len(unique) != 2:
            raise ValueError('Frozen B must contain exactly the two audited temporal kernels')
        kernels = []
        for g in range(2):
            k = groups.index(g)
            L = normal.cholesky[k]
            kernels.append(L@L.T)
        A = projected.projection[0]*normal.std
        U, s, Vh = torch.linalg.svd(A, full_matrices=False)
        if not torch.isfinite(s).all() or float(s[-1]) <= 0:
            raise ValueError('Normal push-forward is not full row rank')
        source_group = normal.ring.new_tensor(groups)[normal.ring-1]
        low = Vh[:, source_group == 0]
        fraction = low@low.T
        lam, Q = torch.linalg.eigh((fraction+fraction.T)/2)
        spill = float(torch.maximum((-lam).clamp_min(0).max(), (lam-1).clamp_min(0).max()))
        if spill > 1e-10:
            raise ValueError('Invalid covariance partition after spatial SVD')
        # Correct only eigensolver roundoff at mathematical endpoints 0 and 1.
        lam = lam.clamp(0, 1)
        temporal = lam[:, None, None]*kernels[0] + (1-lam[:, None, None])*kernels[1]
        self.register_buffer('normal_transform', (U/s)@Q)
        self.register_buffer('normal_temporal_cholesky', torch.linalg.cholesky(temporal))
        self.register_buffer('normal_fraction', lam)
        As = projected.projection[1]*projected.std_s
        Us, ss, _ = torch.linalg.svd(As, full_matrices=False)
        if not torch.isfinite(ss).all() or float(ss[-1]) <= 0:
            raise ValueError('Superfluid push-forward is not full row rank')
        self.register_buffer('super_transform', Us/ss)
        self.diagnostics = dict(
            method='Full-rank SVD push-forward with two-kernel simultaneous diagonalization',
            wall_modes=len(s), source_modes=A.shape[1], intervals=self.intervals,
            temporal_kernels=[dict(ell_intervals=e, white_variance_fraction=w) for e, w in unique],
            normal_covariance_condition=float((s[0]/s[-1]).square()),
            super_covariance_condition=float((ss[0]/ss[-1]).square()),
            normal_min_singular_value=float(s[-1]), super_min_singular_value=float(ss[-1]),
            endpoint_eigenvalue_roundoff=spill, numerical_jitter=0., dropped_singular_directions=0,
            covariance_reestimated_on_cf=False, energy_renormalized=False,
            interpretation='Exact projected prior; potentially stiff because the cropped prior is anisotropic')

    def whiten_normal(self, residual):
        if residual.shape[-2:] != (self.intervals, self.normal_transform.shape[0]):
            raise ValueError('Normal residual must be (..., intervals, wall_modes)')
        spatial = residual@self.normal_transform
        return torch.linalg.solve_triangular(self.normal_temporal_cholesky,
                                             spatial.transpose(-1, -2).unsqueeze(-1),
                                             upper=False).squeeze(-1).transpose(-1, -2)

    def whiten_super(self, residual):
        return residual@self.super_transform

    def quadratic(self, normal, superfluid):
        return self.whiten_normal(normal).square().sum(), self.whiten_super(superfluid).square().sum()


def load_case(name, device='cpu'):
    c, z, op, cold = runtime().load_case(name, device)
    if op.nu_s != 0 or op.friction_factor != 1 or c['intervals'] != 100:
        raise ValueError('Expected the frozen nu_s=0, Fns=1, 100 ms channel case')
    # Existing initial is exactly zero: wall-boundary-respecting prescribed background.
    if torch.count_nonzero(cold):
        raise ValueError('PINN migration requires the fixed original cold background')
    decode, encode, ring, normalization = wall_normalization(c, op)
    c = dict(c, pinn_decode=decode, pinn_encode=encode, pinn_ring=ring,
             pinn_spatial_normalization=normalization)
    projected = runtime().ProjectedPrior(c, op, intervals=c['intervals'])
    output_factor, output_inverse, output_checks = output_preconditioning(c, projected)
    c.update(pinn_output_preconditioner=output_factor, pinn_output_preconditioner_inverse=output_inverse,
             pinn_output_preconditioning=output_checks)
    return c, z, op, cold, WallResidualPrior(projected)


def make_field(c, cold, seed=0):
    return NeuralField(cold, c['pinn_ring'], seed=seed, decode=c['pinn_decode'], encode=c['pinn_encode'],
                       output_preconditioner=c['pinn_output_preconditioner'])


def batched_rhs(op, a):
    """Frozen weak Galerkin momentum RHS, vectorized over time nodes."""
    if a.ndim != 3 or a.shape[1:] != (2, op.m):
        raise ValueError('Expected (nodes,2,wall_modes)')
    v = op.v0 + torch.einsum('fpm,tfm->tfp', op.V, a).reshape(len(a), *op.v0.shape)
    g = op.g0 + torch.einsum('fpm,tfm->tfp', op.G, a).reshape(len(a), *op.g0.shape)
    force = op.ck*op.L0*(v[:, 0]-v[:, 1])
    conv = torch.einsum('tfpj,tfpij->tfpi', v, g)
    stress = (v[..., :, None]*v[..., None, :]).flatten(2)
    loads = -.5*torch.einsum('fpm,tfp->tfm', op.Pv, conv.flatten(2))
    loads = loads + .5*torch.einsum('fpm,tfp->tfm', op.Pg, stress)
    ff = torch.einsum('fpm,tp->tfm', op.Pv, force.flatten(1))
    dn = loads[:, 0] + ff[:, 0]/op.p['rho_n_kg_m3'] - a[:, 0]@op.D.T - op.d0
    ds = loads[:, 1] - ff[:, 1]/op.p['rho_s_kg_m3']
    return torch.stack((dn, ds), 1)


def residual_loss(a, adot, op, prior, nfit):
    """Simpson-weighted three 1 ms residual paths, from 201 half-ms nodes."""
    if len(a) != 2*prior.intervals+1 or nfit <= 0:
        raise ValueError('Expected complete half-ms nodes and positive fixed I count')
    residual = (adot-batched_rhs(op, a))*DT
    paths = torch.stack((residual[:-1:2], residual[1::2], residual[2::2]))
    wn, ws = prior.whiten_normal(paths[:, :, 0]), prior.whiten_super(paths[:, :, 1])
    weights = a.new_tensor([1/6, 4/6, 1/6])
    qn = (weights*wn.square().sum((1, 2))).sum()
    qs = (weights*ws.square().sum((1, 2))).sum()
    return .08/(2*nfit)*(qn+qs), dict(normal_whitened_energy=qn, super_whitened_energy=qs)


def symmetric_regularization(q, ring):
    """Simulation PINN weights on normalized q, uniform over all 101 times.

    Unlike raw wall coefficients, each q coordinate has exactly U^2/2 spatial
    velocity energy. Equivalent wall radii are fixed stiffness eigenvalues.
    """
    ring = ring.to(q)
    high = ring > 2
    energy = .003*q.square().sum(-1).mean()/24
    temporal = .03*(q[1:]-q[:-1]).square().sum(-1).mean()/24
    spectral = .03*(q*torch.where(high, (ring/2)**2, 0)).square().sum(-1).mean()/24
    return energy+temporal+spectral, dict(energy=energy, temporal=temporal, spectral_uniform=spectral)


def queries_from_coefficients(c, a, adot, z):
    """Query arbitrary continuous values and tangents; useful with detached leaves.

    a/adot contain one (2,modes) row per flattened z['time'] entry. Geometry G
    is G[velocity component, derivative coordinate]. Particle acceleration
    follows the established ut + G.T u convention, distinct from PDE advection.
    """
    shape = z['time'].shape
    x = z['x'].to(a).reshape(-1, 2)
    a, adot = a.reshape(-1, 2, a.shape[-1]), adot.reshape(-1, 2, adot.shape[-1])
    geo = runtime().core.channel_geometry(c['physics']['vn_mean_m_s'], c['physics']['vs_mean_m_s'],
                                          str(a.device), a.dtype)
    b = geo.evaluate(x)
    v = b['v0'] + torch.einsum('pfm,pfmj->pfj', a, b['V'])
    g = b['g0'] + torch.einsum('pfm,pfmij->pfij', a, b['G'])
    acc = torch.einsum('pfm,pfmj->pfj', adot, b['V']) + torch.einsum('pfij,pfi->pfj', g, v)
    return tuple(q.reshape(*shape, *q.shape[1:]) for q in (v, g, acc))


def queries(c, field, z):
    a, adot = field.coefficients_and_dt(z['time'].reshape(-1)*DT)
    return queries_from_coefficients(c, a, adot, z)
