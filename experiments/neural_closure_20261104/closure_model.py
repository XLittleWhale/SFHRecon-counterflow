"""Small, shared two-fluid closure inside the current deterministic RK solver.

No data files are opened by this module.  ``ClosureNet`` maps nine local,
dimensionless invariants to eight bounded tensor-basis coefficients.  Two
trace-free stresses and one action/reaction force produce solenoidal modal
time derivatives.  The explicit resolved operator retains nu_s=0 and F_ns=1.

The neural term is a deterministic RHS correction, NOT an endpoint jump.
``ClosureMomentum`` implements the old rhs/step interface so that the existing
rollout may still add separately inferred stochastic endpoint innovations.
Particle acceleration queries must include the deterministic neural term and
subtract only those innovations; that integration is the caller's concern.

Public API:
  ClosureNet(width=16, coefficient_bound=1.0, dtype=torch.float64)
  closure_tendencies(op, a, net, diagnostics=False): [...,2,624] -> same shape
      With diagnostics=True returns (modal_derivative, differentiable_dict).
  ClosureMomentum(op, net).rhs(a,b=None), .step(a,b=None,substeps=2)
      Supports scalar or arbitrary leading batch dimensions, float64 states.
  batched_step(op,a,substeps=2): accepts either wrapped or current base operator.
  closed_step_components(op,a,substeps=2): dictionary with closed_mean (the
      one-step difference), phys_step, closed_step; not a continuous RHS force.
  tendency_loss(op,pred,target,mode_weight=None): dimensionless MSE components.

Run ``python closure_model.py --selftest [--output path]`` for synthetic-only
checks.  No target observations, hidden labels, or reference fields are read.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn


HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
PACKAGE = BASE / "outputs/FullWindowCalibration_20261026/server_package"
CURRENT_SOURCE = HERE.parent / "joint_probability_20261103/physics/simulation_operator.py"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))
_spec = importlib.util.spec_from_file_location("_neural_closure_resolved_operator", CURRENT_SOURCE)
CURRENT = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = CURRENT
_spec.loader.exec_module(CURRENT)

DT = .001


class ClosureNet(nn.Module):
    """568-parameter default network; final layer starts identically at zero.

    The coefficient bound is an explicit protocol hyperparameter, not a
    stability guarantee.  Tanh has nonzero derivative at zero initialization.
    Inputs are asinh-transformed dimensionless invariants and log(U/.004).
    Proper-rotation tensor structure is explicit; no reflection claim is made.
    """
    def __init__(self, width=16, coefficient_bound=1.0, dtype=torch.float64):
        super().__init__()
        if width < 1 or not math.isfinite(coefficient_bound) or coefficient_bound <= 0:
            raise ValueError("Positive width and finite coefficient_bound required")
        self.width = int(width)
        self.register_buffer("coefficient_bound", torch.tensor(float(coefficient_bound), dtype=dtype))
        self.layers = nn.Sequential(nn.Linear(9, width, dtype=dtype), nn.SiLU(),
                                    nn.Linear(width, width, dtype=dtype), nn.SiLU(),
                                    nn.Linear(width, 8, dtype=dtype))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features):
        if features.shape[-1] != 9:
            raise ValueError("Expected nine invariant features")
        if features.dtype != self.coefficient_bound.dtype:
            raise TypeError("Move closure network to the state dtype before use")
        return self.coefficient_bound * torch.tanh(self.layers(features))


def _check_state(op, a):
    if a.ndim < 2 or a.shape[-2:] != (2, 2 * len(op.x)):
        raise ValueError("Expected [...,2,2*number_of_modes] modal state")
    if not a.is_floating_point():
        raise TypeError("Real floating modal coefficients required")


def _fields(op, a):
    _check_state(op, a)
    cdtype = torch.complex128 if a.dtype == torch.float64 else torch.complex64
    vh = torch.zeros((*a.shape[:-2], 2, op.n, op.n, 2), device=a.device, dtype=cdtype)
    perp = op.perp.to(a)
    z = .5 * op.U * (a[..., 1::2] + 1j * a[..., ::2])[..., None] * perp
    vh[..., op.y, op.x, :] = z
    vh[..., op.my, op.mx, :] = z.conj()
    vh[..., 0, 0, :] = op.mean.to(vh)
    inv = lambda q: torch.fft.ifft2(q, dim=(-3, -2)).real * op.n**2
    v = inv(vh)
    dx = inv(vh * (1j * op.kx.to(a))[..., None])
    dy = inv(vh * (1j * op.ky.to(a))[..., None])
    return v, dx, dy


def invariant_features(op, v, dx, dy):
    """Return features and tensor bases, retaining all spatial derivatives."""
    ell = float(op.c["box_m"]) / (2 * math.pi)
    scale = ell / float(op.U)
    grad = torch.stack((dx, dy), -1) * scale
    strain = .5 * (grad + grad.transpose(-1, -2))
    eye = torch.eye(2, dtype=v.dtype, device=v.device)
    strain = strain - .5 * strain.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None] * eye
    sn, ss = strain[..., 0, :, :, :, :], strain[..., 1, :, :, :, :]
    w = (v[..., 0, :, :, :] - v[..., 1, :, :, :]) / float(op.U)
    w2 = w.square().sum(-1)
    ww = w[..., :, None] * w[..., None, :] - .5 * w2[..., None, None] * eye
    omega = (dx[..., 1] - dy[..., 0]) * scale
    wsnw = torch.einsum("...i,...ij,...j->...", w, sn, w)
    wssw = torch.einsum("...i,...ij,...j->...", w, ss, w)
    raw = torch.stack((w2, sn.square().sum((-2, -1)), ss.square().sum((-2, -1)),
                       (sn * ss).sum((-2, -1)), omega[..., 0, :, :], omega[..., 1, :, :],
                       wsnw, wssw), -1)
    speed = raw.new_full((*raw.shape[:-1], 1), math.log(float(op.U) / .004))
    features = torch.cat((raw.asinh(), speed), -1)
    return features, (sn, ss, ww, w), ell


def _project(op, acceleration):
    """Existing nonzero solenoidal modal projection, with fixed mean modes."""
    fh = torch.fft.fft2(acceleration, dim=(-3, -2)) / op.n**2
    z = fh[..., op.y, op.x, :]
    z = (z * op.perp.to(acceleration)).sum(-1) / float(op.U)
    return torch.stack((2 * z.imag, 2 * z.real), -1).flatten(-2)


def closure_tendencies(op, a, net, diagnostics=False):
    """Deterministic modal acceleration from stresses and internal exchange.

    Stress bases: dimensionless S_n, S_s and trace-free w*w. Coefficients 0:3
    form the normal kinematic stress, 3:6 the superfluid kinematic stress.
    Coefficients 6 and 7 multiply w and its positive 90-degree rotation.
    Internal force density uses rho_reduced, and accelerations are +f/rho_n,
    -f/rho_s. Thus the internal force cancels pointwise before projection.
    All mean modes are excluded, as in the parent model with prescribed means.
    Raw power diagnostics include work on those prescribed means. The spatial
    average of resolved_fluctuation_power_density excludes that mean work and
    equals the work on the retained nonzero solenoidal velocity modes.
    """
    v, dx, dy = _fields(op, a)
    features, (sn, ss, ww, w), ell = invariant_features(op, v, dx, dy)
    coefficients = net(features)
    basis = torch.stack((sn, ss, ww), -3)
    stress_n = float(op.U)**2 * torch.einsum("...k,...kij->...ij", coefficients[..., :3], basis)
    stress_s = float(op.U)**2 * torch.einsum("...k,...kij->...ij", coefficients[..., 3:6], basis)
    stress = torch.stack((stress_n, stress_s), -5)
    stress_h = torch.fft.fft2(stress, dim=(-4, -3)) / op.n**2
    div_h = (1j * op.kx.to(a)[..., None]) * stress_h[..., 0] + (1j * op.ky.to(a)[..., None]) * stress_h[..., 1]
    stress_acc = -torch.fft.ifft2(div_h, dim=(-3, -2)).real * op.n**2
    rn, rs = float(op.c["rho_n_kg_m3"]), float(op.c["rho_s_kg_m3"])
    rho_reduced = rn * rs / (rn + rs)
    jw = torch.stack((-w[..., 1], w[..., 0]), -1)
    force = rho_reduced * float(op.U)**2 / ell * (coefficients[..., 6, None] * w + coefficients[..., 7, None] * jw)
    exchange_acc = torch.stack((force / rn, -force / rs), -4)
    modal = _project(op, stress_acc + exchange_acc)
    if not diagnostics:
        return modal
    stress_modal = _project(op, stress_acc)
    exchange_modal = _project(op, exchange_acc)
    density = a.new_tensor((rn, rs)).reshape(*([1] * (v.ndim - 4)), 2, 1, 1, 1)
    mean_shape = (*([1] * (v.ndim - 4)), 2, 1, 1, 2)
    fluctuation = v - op.mean.to(a).reshape(mean_shape)
    diagnostic = dict(coefficients=coefficients, features=features,
                      stress_acceleration=stress_acc, exchange_acceleration=exchange_acc,
                      force_density=force, kinematic_stress=stress,
                      stress_tendency=stress_modal, exchange_tendency=exchange_modal,
                      exchange_total_force=rn * exchange_acc[..., 0, :, :, :] + rs * exchange_acc[..., 1, :, :, :],
                      stress_mean_acceleration=stress_acc.mean((-3, -2)),
                      exchange_power_density=(density * v * exchange_acc).sum((-4, -1)),
                      total_power_density=(density * v * (stress_acc + exchange_acc)).sum((-4, -1)),
                      resolved_fluctuation_power_density=(density * fluctuation * (stress_acc + exchange_acc)).sum((-4, -1)))
    return modal, diagnostic


def tendency_loss(op, prediction, target, mode_weight=None):
    """Dimensionless modal MSE; caller chooses any scientific loss weighting.

    Divide coefficient time derivatives by U/(box/(2*pi)); no fitted scale or
    target statistic is used. Return separate fluid errors and their average.
    Optional nonnegative mode_weight has length 624 and is normalized to mean 1.
    """
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target must have identical shapes")
    _check_state(op, prediction)
    rate = float(op.U) / (float(op.c["box_m"]) / (2 * math.pi))
    error = ((prediction - target) / rate).square()
    if mode_weight is not None:
        weight = torch.as_tensor(mode_weight, dtype=error.dtype, device=error.device)
        if weight.shape != error.shape[-1:] or bool((weight < 0).any()) or not bool(weight.mean() > 0):
            raise ValueError("Expected nonnegative mode weights with positive mean")
        error = error * weight / weight.mean()
    per_fluid = error.mean(-1).reshape(-1, 2).mean(0)
    return dict(total=per_fluid.mean(), normal=per_fluid[0], super=per_fluid[1])


class ClosureMomentum:
    """Resolved Momentum plus a neural RHS; preserves separate jump semantics.

    The wrapped current operator is read-only. Inference must freeze ``net``
    parameters externally while keeping derivatives with respect to fields.
    The wrapper does not inherit nn.Module; optimize/freeze ``net`` explicitly.
    ``rhs`` matches the current scalar operator bitwise when the head is zero.
    A batch uses the current operator's vectorized constant-L helper.
    """
    def __init__(self, op, net):
        if float(op.nu_s) != 0.0 or float(op.friction_factor) != 1.0:
            raise ValueError("This experiment requires explicit nu_s=0 and F_ns factor=1")
        if bool(op.local):
            raise ValueError("First-round implementation uses the frozen constant-L operator")
        self.base = op
        self.net = net

    def __getattr__(self, name):
        return getattr(self.base, name)

    def rhs(self, a, b=None):
        if a.dtype != torch.float64:
            raise TypeError("Current resolved solver and wrapper require float64 states")
        if b is None:
            b = a.new_zeros(self.base.B.shape[-1])
        if a.ndim == 2:
            resolved = self.base.rhs(a, b)
        else:
            if bool((b != 0).any()):
                raise ValueError("Batched resolved helper supports only frozen b=0")
            resolved = CURRENT.batched_rhs(self.base, a)
        return resolved + closure_tendencies(self.base, a, self.net)

    def step(self, a, b=None, substeps=2):
        if not isinstance(substeps, int) or substeps < 1:
            raise ValueError("Positive integer substeps required")
        h = DT / substeps
        for _ in range(substeps):
            k1 = self.rhs(a, b)
            k2 = self.rhs(a + h * k1 / 2, b)
            k3 = self.rhs(a + h * k2 / 2, b)
            k4 = self.rhs(a + h * k3, b)
            a = a + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return a

    def tendencies(self, a, diagnostics=False):
        return closure_tendencies(self.base, a, self.net, diagnostics)


def batched_step(op, a, substeps=2):
    """Current resolved/closed RK map with arbitrary leading batch dimensions.

    This helper deliberately never calls the historical factorial-model RHS.
    It is usable during differentiable training and frozen-network evaluation.
    """
    if not isinstance(substeps, int) or substeps < 1:
        raise ValueError("Positive integer substeps required")
    if isinstance(op, ClosureMomentum):
        return op.step(a, substeps=substeps)
    if float(op.nu_s) != 0. or float(op.friction_factor) != 1. or bool(op.local):
        raise ValueError("Expected current constant-L nu_s=0/F_ns=1 operator")
    if a.ndim == 2:
        return op.step(a, a.new_zeros(op.B.shape[-1]), substeps)
    return CURRENT.batched_step(op, a, substeps)


def closed_step_components(op, a, substeps=2):
    """Separate deterministic neural one-step change from stochastic jumps.

    ``closed_mean = closed_step - phys_step`` compares two RK maps starting
    from the same state. It must not be passed as eta to local_jets: it already
    contributes to the deterministic acceleration in the closed RK dynamics.
    """
    phys_op = op.base if isinstance(op, ClosureMomentum) else op
    physical = batched_step(phys_op, a, substeps)
    closed = batched_step(op, a, substeps) if isinstance(op, ClosureMomentum) else physical
    return dict(closed_mean=closed-physical, phys_step=physical, closed_step=closed)


def _synthetic_operator():
    """Construct synthetic metadata and modes; never open a case or checkpoint."""
    n, U, box = 40, .004, .001
    xy, perp = CURRENT.modes(12)
    xy = torch.as_tensor(xy, dtype=torch.long)
    freq = torch.fft.fftfreq(n, d=box/n, dtype=torch.float64) * (2 * math.pi)
    ky, kx = torch.meshgrid(freq, freq, indexing="ij")
    old = SimpleNamespace(n=n, U=U, nu_s=0., mean=torch.tensor([[U, 0.], [-.002, 0.]], dtype=torch.float64),
        c=dict(box_m=box, rho_n_kg_m3=100., rho_s_kg_m3=45., alpha=.1,
               kappa_m2_s=1e-7, mu_n_Pa_s=3e-6, alpha_v=1e6, beta_v=1.),
        perp=torch.as_tensor(perp, dtype=torch.float64), kx=kx, ky=ky, k2=kx.square()+ky.square(),
        x=xy[:, 0] % n, y=xy[:, 1] % n, mx=(-xy[:, 0]) % n, my=(-xy[:, 1]) % n,
        lap=(xy.double().square().sum(-1)*(2*math.pi/box)**2).repeat_interleave(2))
    return CURRENT.Momentum(old)


def selftest():
    torch.set_num_threads(2)
    torch.manual_seed(411040)
    op = _synthetic_operator()
    net = ClosureNet()
    wrapper = ClosureMomentum(op, net)
    ring = torch.maximum(op.x.minimum(op.n-op.x), op.y.minimum(op.n-op.y)).double().repeat_interleave(2)
    a = .002 * torch.randn(2, 624, dtype=torch.float64) / ring
    b = a.new_zeros(op.B.shape[-1])
    zero_rhs = torch.equal(wrapper.rhs(a, b), op.rhs(a, b))
    zero_step = torch.equal(wrapper.step(a, b), op.step(a, b))
    assert zero_rhs and zero_step, "Zero-head identity must be exact"
    zero_components = closed_step_components(wrapper, a)
    assert torch.equal(zero_components["closed_mean"], torch.zeros_like(a))
    batch = torch.stack((a, .8*a)).requires_grad_()
    zero_batch = wrapper.rhs(batch)
    number = lambda x: float(x.detach())
    batch_difference = number((zero_batch - torch.stack([op.rhs(x, b) for x in batch])).abs().max())
    assert batch_difference < 1e-12
    zero_batch_components = closed_step_components(wrapper, batch)
    assert torch.equal(zero_batch_components["closed_mean"], torch.zeros_like(batch))
    signal = torch.randn_like(batch)
    probe = (closure_tendencies(op, batch, net) * signal).mean()
    first_weight_gradient = torch.autograd.grad(probe, net.layers[-1].weight)[0]
    zero_head_learns = number(first_weight_gradient.abs().max())
    assert zero_head_learns > 1e-10
    with torch.no_grad():
        net.layers[-1].weight.normal_(0., .0003)
        net.layers[-1].bias.normal_(0., .0003)
    tendency, diag = closure_tendencies(op, batch, net, True)
    objective = (tendency * signal).mean() + .0001*tendency.square().mean()
    grad = torch.autograd.grad(objective, batch, create_graph=True)[0]
    second = torch.autograd.grad((grad.square()).sum(), batch)[0]
    assert bool(torch.isfinite(grad).all() and torch.isfinite(second).all())
    assert number(grad.abs().max()) > 0 and number(second.abs().max()) > 0
    pointwise_force_error = number(diag["exchange_total_force"].abs().max())
    modal_force_error = number((100*diag["exchange_tendency"][..., 0, :] + 45*diag["exchange_tendency"][..., 1, :]).abs().max())
    stress_mean_error = number(diag["stress_mean_acceleration"].abs().max())
    assert pointwise_force_error < 1e-12 and modal_force_error < 1e-9 and stress_mean_error < 1e-12
    decomposition_error = number((tendency-diag["stress_tendency"]-diag["exchange_tendency"]).abs().max())
    assert decomposition_error < 1e-12
    with torch.no_grad():
        single_difference = float((tendency-torch.stack([closure_tendencies(op, x, net) for x in batch])).abs().max())
        assert single_difference < 1e-12
        test32 = closure_tendencies(op, batch.float(), net.float())
        assert bool(torch.isfinite(test32).all())
        net.double()
        state = a.clone()
        for _ in range(12):
            state = wrapper.step(state, b)
        assert bool(torch.isfinite(state).all())
        initial_rms, end_rms = float(a.square().mean().sqrt()), float(state.square().mean().sqrt())
        assert end_rms < max(.1, initial_rms*20)
        # Local tensor operations must commute with exact grid translations.
        phase = torch.exp(-2j * math.pi * (op.x + 2*op.y).double() / op.n)
        def translated(x):
            z = (x[..., 1::2] + 1j*x[..., ::2]) * phase
            return torch.stack((z.imag, z.real), -1).flatten(-2)
        shift_error = number((closure_tendencies(op, translated(a), net)
                              - translated(closure_tendencies(op, a, net))).abs().max())
        boosted = copy.copy(op)
        boosted.mean = op.mean + op.mean.new_tensor([.003, -.001])[None]
        boost_error = number((closure_tendencies(boosted, a, net)
                              - closure_tendencies(op, a, net)).abs().max())
        uniform_error = number(closure_tendencies(op, torch.zeros_like(a), net).abs().max())
        assert shift_error < 1e-11 and boost_error < 1e-12 and uniform_error < 1e-12
    detached = batch.detach().requires_grad_()
    rhs = wrapper.rhs(detached)
    rhs_grad = torch.autograd.grad(rhs.square().mean(), detached, create_graph=True)[0]
    rhs_second = torch.autograd.grad(rhs_grad.square().sum(), detached)[0]
    assert bool(torch.isfinite(rhs_second).all())
    # The actual RK graph must support gradients through an inner state update,
    # as required when outer closure training differentiates through 4DVar.
    inner_state = a.detach().clone().requires_grad_()
    target = .97*a.detach()
    inner_prediction = wrapper.step(inner_state, b)
    inner_loss = (inner_prediction-target).square().mean()
    inner_gradient = torch.autograd.grad(inner_loss, inner_state, create_graph=True)[0]
    adapted_state = inner_state - .2*inner_gradient
    outer_prediction = wrapper.step(adapted_state, b)
    outer_loss = (outer_prediction-target).square().mean()
    mixed = torch.autograd.grad(outer_loss, net.layers[-1].weight)[0]
    assert bool(torch.isfinite(mixed).all()) and number(mixed.abs().max()) > 0
    # A separate small synthetic regression checks learnability, not just
    # nonzero derivatives. It never touches production networks or datasets.
    teacher, learner = ClosureNet(), ClosureNet()
    with torch.no_grad():
        teacher.layers[-1].bias.copy_(a.new_tensor([-.01, 0., 0., 0., -.005, 0., -.005, 0.]))
        teacher_target = closure_tendencies(op, a, teacher)
    optimizer = torch.optim.Adam(learner.parameters(), lr=.0005)
    synthetic_losses = []
    for _ in range(21):
        optimizer.zero_grad(set_to_none=True)
        pred = closure_tendencies(op, a, learner)
        loss = tendency_loss(op, pred, teacher_target)["total"]
        synthetic_losses.append(number(loss))
        if len(synthetic_losses) <= 20:
            loss.backward()
            optimizer.step()
    assert min(synthetic_losses[1:]) < .7*synthetic_losses[0]
    report = dict(passed=True, synthetic_only=True, target_files_read=False,
        parameter_count=sum(x.numel() for x in net.parameters()), zero_rhs_bitwise=zero_rhs,
        zero_step_bitwise=zero_step, batch_resolved_max_difference=batch_difference,
        zero_head_output_weight_gradient_max=zero_head_learns,
        closure_first_gradient_max=number(grad.abs().max()), closure_second_gradient_max=number(second.abs().max()),
        full_rhs_second_gradient_max=number(rhs_second.abs().max()),
        rk_inner_outer_head_gradient_max=number(mixed.abs().max()),
        internal_force_density_max_error=pointwise_force_error,
        projected_internal_force_max_error=modal_force_error,
        stress_mean_acceleration_max_error=stress_mean_error,
        tendency_decomposition_max_error=decomposition_error, batch_scalar_max_error=single_difference,
        grid_translation_max_error=shift_error, common_velocity_boost_max_error=boost_error,
        uniform_state_nonzero_mode_error=uniform_error,
        float32_closure_finite=True, short_rollout_steps=12, initial_coefficient_rms=initial_rms,
        final_coefficient_rms=end_rms, coefficient_bound=float(net.coefficient_bound),
        synthetic_learning_steps=20, synthetic_learning_loss_initial=synthetic_losses[0],
        synthetic_learning_loss_final=synthetic_losses[-1], synthetic_learning_loss_best=min(synthetic_losses),
        limitation="Small nonzero synthetic head only; bounded coefficients do not guarantee arbitrary-head or long-rollout stability.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.selftest:
        parser.error("Use --selftest; this module does not launch training")
    result = selftest()
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf8")
    print(encoded)
