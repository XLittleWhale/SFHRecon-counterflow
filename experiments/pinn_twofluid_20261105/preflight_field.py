"""Synthetic independent checks of continuous field units and derivatives.

No optimization, target field, target H scoring, or scientific selection occurs.
Only non-target physical metadata are read for the original PDE parity check.
"""
from __future__ import annotations

import torch
from global_spline import GlobalSpline, DT_SECONDS, DURATION_SECONDS, slope_matrix
from field_model import NeuralField, residual_loss, symmetric_regularization
from pinn_common import OUT, CURRENT, M, T, EW, load, dump, setup


def number(x):
    return float(x.detach()) if torch.is_tensor(x) else float(x)


def close(actual, expected, atol, name):
    error = number((actual-expected).abs().max())
    if not torch.allclose(actual, expected, atol=atol, rtol=1e-8):
        raise AssertionError(f'{name}: maximum error {error}')
    return error


def test_spline(cold, ring):
    field = GlobalSpline(cold, ring)
    points = torch.tensor([0., .00017, .001, .00183, .0167, .05432, .0932, .0948, .095],
                          dtype=torch.float64)
    values, derivative = field.coefficients_and_dt(points)
    constant = close(values, cold[None].expand_as(values), 1e-13, 'constant spline')
    constant_derivative = close(derivative, torch.zeros_like(derivative), 1e-10,
                                'constant spline derivative')
    grid = field.node_times
    poly = lambda t: .03+.2*t-3*t*t+17*t*t*t
    dpoly = lambda t: .2-6*t+51*t*t
    with torch.no_grad():
        field.controls.zero_()
        field.controls[:, 0, 0] = (poly(grid)-cold[0, 0])/field.scale[0]
    got, got_dt = field.coefficients_and_dt(points)
    polynomial = close(got[:, 0, 0], poly(points), 1e-12, 'cubic exact values')
    polynomial_dt = close(got_dt[:, 0, 0], dpoly(points), 1e-10, 'cubic exact SI derivatives')
    nodes = field.states()
    knot_exact = close(field(grid), nodes, 1e-12, 'spline knot interpolation')
    # Shared nodal slopes make the representation C1 at every interior knot.
    at = grid[1:-1]
    before, after = field.coefficients_and_dt(at-1e-10)[1], field.coefficients_and_dt(at+1e-10)[1]
    continuity = close(before, after, 2e-8, 'C1 left/right derivatives')
    slopes = torch.einsum('ij,jfm->ifm', field.derivative_matrix, nodes)
    boundary = close(got_dt[[0, -1]], slopes[[0, -1]], 1e-10, 'free endpoint derivatives')
    if not bool(got_dt[[0, -1], 0, 0].abs().min() > .01):
        raise AssertionError('Endpoint derivatives were spuriously constrained to zero')
    params = list(field.named_parameters())
    if len(params) != 1 or params[0][0] != 'controls' or params[0][1].shape != (96, 2, 624):
        raise AssertionError('Spline must have one uniform all-time parameter block')
    return field, dict(constant_value_error=constant, constant_dt_error=constant_derivative,
        cubic_value_error=polynomial, cubic_dt_error=polynomial_dt,
        knot_error=knot_exact, c1_derivative_error=continuity,
        endpoint_derivative_error=boundary, parameters=params[0][1].numel(),
        independent_initial_or_derivative_parameters=False)


def derivative_test(field, label):
    t = torch.tensor([0., .00723, .03172, .06731, .095], dtype=torch.float64)
    value, tangent = field.coefficients_and_dt(t)
    _, automatic = torch.autograd.functional.jvp(lambda x: field(x), t,
        torch.ones_like(t), create_graph=True, strict=True)
    exact = close(tangent, automatic, 1e-9, label+' analytical tangent versus AD')
    h = 1e-7
    ti = t[1:-1]
    difference = (field(ti+h)-field(ti-h))/(2*h)
    fd = close(field.coefficients_and_dt(ti)[1], difference, 2e-5,
               label+' physical-second central difference')
    # Independent second-order one-sided differences at both physical boundaries.
    endpoints = torch.stack(((-3*field(t[:1])+4*field(t[:1]+h)-field(t[:1]+2*h))[0]/(2*h),
                             (3*field(t[-1:])-4*field(t[-1:]-h)+field(t[-1:]-2*h))[0]/(2*h)))
    endpoint_fd = close(tangent[[0, -1]], endpoints, 3e-5, label+' endpoint physical derivative')
    parameter = next(field.parameters())
    field.zero_grad(set_to_none=True)
    loss = (value.square()+(.001*tangent).square()).mean()
    gradient = torch.autograd.grad(loss, parameter)[0]
    ix_flat = int(gradient.abs().argmax())
    ix = tuple(int(x) for x in torch.unravel_index(torch.tensor(ix_flat), parameter.shape))
    analytic = gradient[ix].detach().clone()
    original = parameter[ix].detach().clone()
    eps = 1e-6
    estimates = []
    try:
        for direction in (1., -1.):
            with torch.no_grad():
                parameter[ix] = original+direction*eps
            v, d = field.coefficients_and_dt(t)
            estimates.append((v.square()+(.001*d).square()).mean().detach())
    finally:
        with torch.no_grad():
            parameter[ix] = original
    numerical = (estimates[0]-estimates[1])/(2*eps)
    mixed_error = number((analytic-numerical).abs())
    if not torch.allclose(analytic, numerical, atol=1e-10, rtol=2e-4):
        raise AssertionError(f'{label} mixed field/time parameter gradient: {analytic} vs {numerical}')
    if not bool(torch.isfinite(gradient).all()) or not bool(gradient.abs().max() > 0):
        raise AssertionError('Zero or nonfinite parameter gradient')
    return dict(time_AD_error=exact, time_FD_error=fd, endpoint_FD_error=endpoint_fd,
                mixed_parameter_gradient=number(analytic), mixed_parameter_FD=number(numerical),
                mixed_parameter_error=mixed_error, parameter_index=ix)


def physics_test(cold, ring):
    name = T.VALIDATION_CASE
    observations = load(T.SOURCE/'prepared'/name/'observations.pt')
    c = observations['c']
    op = CURRENT.Momentum(EW.operator(c, load(T.PACKAGE/'operator.pt'), EW.config(), 'cpu'),
                          local=False, friction_factor=1., nu_s=0.)
    if op.n != 64 or op.nu_s != 0. or op.friction_factor != 1. or op.local:
        raise AssertionError('Incorrect original deterministic physics')
    gen = torch.Generator().manual_seed(20261105)
    a = torch.randn((3, 2, 624), generator=gen, dtype=torch.float64)*.001
    b = a.new_zeros(op.B.shape[-1])
    rhs = CURRENT.batched_rhs(op, a)
    parity = close(rhs, torch.stack([op.rhs(x, b) for x in a]), 1e-11, 'batch physical rhs')
    zero_rhs = CURRENT.batched_rhs(op, torch.zeros_like(a))
    zero = close(zero_rhs, torch.zeros_like(a), 1e-11, 'uniform fixed mean solution')
    U = c['physics']['vn_mean_m_s']
    std = M.scales(U).double()
    prior = M.TemporalNormalPrior(U, model='slow_innovation', std_override=std[0])
    field = GlobalSpline(cold, ring)
    with torch.no_grad():
        field.controls.copy_(torch.randn(field.controls.shape, generator=gen, dtype=torch.float64)*.0001)
    half_grid = torch.arange(191, dtype=torch.float64)*.0005
    values, derivatives = field.coefficients_and_dt(half_grid)
    loss, details = residual_loss(values, derivatives, op, std, prior, nfit=1000)
    residual = (derivatives-CURRENT.batched_rhs(op, values))*.001
    manual = residual.new_zeros(())
    for weight, phase in zip((1/6, 4/6, 1/6), (residual[:-1:2], residual[1::2], residual[2::2])):
        manual = manual+weight*(prior.whiten(phase[:, 0]).square().sum()+
                                (phase[:, 1]/std[1]).square().sum())
    scaled = close(loss, .08/(2*1000)*manual, 1e-9, 'B-scaled Simpson residual')
    gradient = torch.autograd.grad(loss, field.controls)[0]
    if not bool(torch.isfinite(gradient).all()) or not bool(gradient[[0, -1]].abs().max() > 0):
        raise AssertionError('PDE loss fails to reach endpoint field controls')
    # The spatial regularizer treats both endpoints and every interior node alike.
    identical = torch.zeros((96, 2, 624), dtype=torch.float64)
    high = int(torch.where(ring > 2)[0][0])
    penalties = []
    for j in (0, 23, 95):
        x = identical.clone()
        x[j, 0, high] = .02
        penalties.append(symmetric_regularization(x, ring)[1]['spectral_uniform'])
    uniform = close(torch.stack(penalties), penalties[0].expand(3), 1e-15,
                    'time-symmetric spectral regularization')
    return dict(non_target_metadata_case=name, target_reference_read=False,
        target_H_scored=False, operator_n=op.n, nu_s=op.nu_s, Fns=op.friction_factor,
        mean_fixed=True, batch_rhs_error=parity, uniform_mean_rhs_error=zero,
        residual_scaling_error=scaled, spatial_penalty_time_uniform_error=uniform,
        pde_gradient_first_norm=number(gradient[0].norm()),
        pde_gradient_last_norm=number(gradient[-1].norm()))


def main():
    setup()
    xy, _ = EW.modes(12)
    ring = torch.tensor(abs(xy).max(1).repeat(2), dtype=torch.float64)
    cold = torch.zeros((2, 624), dtype=torch.float64)
    spline, spline_report = test_spline(cold, ring)
    spline_report['derivatives'] = derivative_test(spline, 'global_spline')
    neural = NeuralField(cold, ring, seed=20261105)
    grid = torch.arange(96, dtype=torch.float64)*.001
    value, derivative = neural.coefficients_and_dt(grid)
    neural_report = dict(cold_value_error=close(value, cold[None].expand_as(value), 0., 'neural cold'),
        cold_dt_error=close(derivative, torch.zeros_like(derivative), 0., 'neural cold derivative'),
        parameter_count=sum(p.numel() for p in neural.parameters()))
    # A deliberately nonzero synthetic network tests derivatives beyond its zero head.
    with torch.no_grad():
        gen = torch.Generator().manual_seed(12345)
        neural.output.weight.copy_(torch.randn(neural.output.weight.shape, generator=gen,
                                               dtype=torch.float64)*.001)
    neural_report['derivatives'] = derivative_test(neural, 'pinn')
    report = dict(passed=True, scientific_optimization_steps=0, target_reference_read=False,
        target_H_scored=False, global_spline=spline_report, neural=neural_report,
        physics=physics_test(cold, ring))
    dump(OUT/'preflight/field_independent.json', report)
    print(report, flush=True)
    return report


if __name__ == '__main__':
    main()
