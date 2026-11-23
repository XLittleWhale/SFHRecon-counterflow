"""Read-only analytic/non-target checks for the continuous particle interface.

No target observations, target H, reference field, fitting, or selection is
accessed. The probability checks use at most four intervals from non-target
6.5 mm/s validation PIDs. Any supplied checkpoint is frozen throughout.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
import continuous_query as Q

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]


def maxdiff(a, b):
    return float((a.detach()-b.detach()).abs().max())


def analytic_checks():
    torch.set_num_threads(2)
    gen = torch.Generator().manual_seed(511050)
    c = dict(physics=dict(vn_mean_m_s=.0065, vs_mean_m_s=-.0032, box_m=.001))
    K, count = 3, 7
    modes = 2*len(Q.core.modes(K)[0])
    rand = lambda *shape: torch.randn(*shape, generator=gen, dtype=torch.float64)
    x = torch.rand(count, 2, generator=gen, dtype=torch.float64)*.0008+.0001
    x.requires_grad_()
    z = dict(x=x, frame=torch.arange(count))
    t = Q.midpoint_times(z)
    a0, beta, gamma = .02*rand(count, 2, modes), rand(count, 2, modes), rand(count, 2, modes)
    a = a0+t[:, None, None]*beta+t[:, None, None]**2*gamma
    adot = beta+2*t[:, None, None]*gamma
    detail = Q.coefficient_jets(a, adot, z, c, K, detailed=True)
    gradient = torch.stack([torch.stack([
        torch.autograd.grad(detail['velocity'][:, f, j].sum(), x, retain_graph=True)[0]
        for j in range(2)], 1) for f in range(2)], 1)
    spatial_error = maxdiff(gradient, detail['gradient'])
    energy_gradient = torch.stack([torch.autograd.grad(
        .5*detail['velocity'][:, f].square().sum(), x, retain_graph=True)[0]
        for f in range(2)], 1)
    energy_error = maxdiff(energy_gradient, detail['energy_gradient'])
    transpose_distinction = maxdiff(detail['energy_gradient'], detail['advection'])
    eps = 1e-7
    ap = a0+(t+eps)[:, None, None]*beta+(t+eps)[:, None, None]**2*gamma
    an = a0+(t-eps)[:, None, None]*beta+(t-eps)[:, None, None]**2*gamma
    vp = Q.coefficient_jets(ap, adot, z, c, K)[0]
    vn = Q.coefficient_jets(an, adot, z, c, K)[0]
    time_error = maxdiff((vp-vn)/(2*eps), detail['partial_t'])
    zp = dict(x=x.detach()+x.new_tensor([3., -2.])*c['physics']['box_m'], frame=z['frame'])
    periodic = Q.coefficient_jets(a, adot, zp, c, K)
    periodic_error = max(maxdiff(j, k) for j, k in zip(periodic,
        (detail['velocity'], detail['gradient'], detail['particle_acceleration'])))
    divergence = float(detail['gradient'].diagonal(dim1=-2, dim2=-1).sum(-1).detach().abs().max())
    n = 2*K+2
    xx = (torch.arange(n, dtype=torch.float64)+.5)*c['physics']['box_m']/n
    yy, xx = torch.meshgrid(xx, xx, indexing='ij')
    grid = dict(x=torch.stack((xx, yy), -1).reshape(-1, 2))
    vg = Q.coefficient_jets(a[:1].expand(n*n, -1, -1), adot[:1].expand(n*n, -1, -1), grid, c, K)[0]
    mean_error = maxdiff(vg.mean(0), vg.new_tensor([[0., .0065], [0., -.0032]]))

    # A trajectory affine in SI time has the same midpoint and derivative as
    # the previous zero-innovation finite-difference observation operator.
    astart, slope = .02*rand(2, modes), rand(2, modes)
    endpoints = astart+torch.arange(count+1, dtype=torch.float64)[:, None, None]*.001*slope
    old = Q.core.jets(endpoints, z, c, K)
    new = Q.coefficient_jets(astart+t[:, None, None]*slope, slope.expand(count, -1, -1), z, c, K)
    affine_error = max(maxdiff(j, k) for j, k in zip(old, new))

    # Differentiate a loss through both a(t) and its genuine time derivative.
    beta_scalar = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    direction = rand(count, 2, modes)
    def objective(b):
        aa = a.detach()+b*t[:, None, None]*direction
        dd = adot.detach()+b*direction
        v, g, acc = Q.coefficient_jets(aa, dd, dict(x=x.detach()), c, K)
        return (v.square().sum()+1e-6*g.square().sum()+acc.square().sum())
    scalar = objective(beta_scalar)
    auto = torch.autograd.grad(scalar, beta_scalar)[0]
    fd = (objective(beta_scalar.detach()+1e-5)-objective(beta_scalar.detach()-1e-5))/2e-5
    gradient_error = float((auto-fd).abs()/auto.abs().clamp_min(1.))
    assert spatial_error < 1e-12 and energy_error < 1e-12
    assert time_error < 1e-9 and periodic_error < 1e-10 and divergence < 1e-12
    assert mean_error < 1e-14 and affine_error < 1e-11 and gradient_error < 1e-7
    assert transpose_distinction > 1e-4
    return dict(spatial_autograd_max_error=spatial_error,
                energy_gradient_autograd_max_error=energy_error,
                particle_vs_material_acceleration_max_difference=transpose_distinction,
                true_time_derivative_finite_difference_error=time_error,
                periodic_max_error=periodic_error, divergence_max=divergence,
                fixed_mean_max_error=mean_error, affine_zero_eta_legacy_max_error=affine_error,
                field_parameter_directional_gradient_relative_error=gradient_error,
                interval_seconds=.001, probability_acceleration='ut + transpose(grad_u) @ u',
                pde_advection='grad_u @ u', residual_subtracted=False, passed=True)


def probability_checks(net, c, z, cal, force, *, group='V', degree=4):
    from pinn_common import N, TM
    for parameter in net.parameters():
        assert not parameter.requires_grad
        parameter.grad = None
    state_before = {k: v.detach().clone() for k, v in net.state_dict().items()}
    gen = torch.Generator().manual_seed(511051)
    frames = int(z['frame'].max())+1
    start = .002*torch.randn(2, 624, generator=gen, dtype=torch.float64)
    slope = .02*torch.randn(2, 624, generator=gen, dtype=torch.float64)
    ends = (start+torch.arange(frames+1, dtype=torch.float64)[:, None, None]*.001*slope).requires_grad_()
    am = (.5*(ends[:-1]+ends[1:])).detach().requires_grad_()
    dm = ((ends[1:]-ends[:-1])/.001).detach().requires_grad_()
    d = dict(z=z, c=c)
    zeros = ends.new_zeros(frames, 624)
    old = N.score(ends, zeros, zeros, d, net, cal, force, group, degree, metrics=True, backward=True, batch=3)
    new = Q.score_nodes(am, dm, d, net, cal, force, group, degree, metrics=True, backward=True, batch=3, batch_pids=1)
    chained = torch.zeros_like(ends)
    chained[:-1] += .5*am.grad-dm.grad/.001
    chained[1:] += .5*am.grad+dm.grad/.001
    value_error = max(abs(old[k]-new[k]) for k in ('free', 'cap', 'total'))
    metric_error = max(abs(old[k][state]-new[k][state]) for k in ('coverage', 'spread') for state in ('free', 'cap'))
    input_gradient_error = maxdiff(ends.grad, chained)
    grad_scale = max(float(ends.grad.abs().max()), 1.)
    assert value_error < 1e-9 and metric_error < 1e-9 and input_gradient_error/grad_scale < 1e-10
    assert am.grad.norm() > 0 and dm.grad.norm() > 0

    capids = torch.where(z['state'] == 1)[0][:2]
    zz = Q.core.subset(z, capids)
    vv, gg, aa = Q.coefficient_jets(am[zz['frame']], dm[zz['frame']], zz, c)
    LM = N.LM
    first = LM.step(c, zz, vv, gg, aa, net.base, net.head, cal, force, degree)
    altered = dict(zz, target=zz['target']+zz['target'].new_tensor([.3, -.2]))
    second = LM.step(c, altered, vv, gg, aa, net.base, net.head, cal, force, degree)
    current_prediction_error = max(maxdiff(first[i], second[i]) for i in range(3))
    posterior_difference = max(maxdiff(first[3][k], second[3][k]) for k in ('qlogp', 'emean', 'ecov'))
    context = TM.context(c, zz, vv, gg)
    follow1 = LM.step(c, zz, vv, gg, aa, net.base, net.head, cal, force, degree,
                     prevstate=first[3], prevctx=context)
    follow2 = LM.step(c, zz, vv, gg, aa, net.base, net.head, cal, force, degree,
                     prevstate=second[3], prevctx=context)
    next_difference = max(maxdiff(follow1[i], follow2[i]) for i in (0, 2))
    assert current_prediction_error == 0 and posterior_difference > 1e-8 and next_difference > 1e-8
    assert all(torch.equal(value, net.state_dict()[key]) for key, value in state_before.items())
    assert all(parameter.grad is None for parameter in net.parameters())
    return dict(rows=len(z['pid']), pids=int(z['pid'].unique().numel()), frames=frames,
                quadrature_degree=degree, legacy_zero_eta_likelihood_max_error=value_error,
                legacy_zero_eta_coverage_spread_max_error=metric_error,
                batched_input_gradient_max_error=input_gradient_error,
                batched_input_gradient_relative_error=input_gradient_error/grad_scale,
                midpoint_gradient_norm=float(am.grad.norm()), time_derivative_gradient_norm=float(dm.grad.norm()),
                current_endpoint_perturbation_prediction_error=current_prediction_error,
                current_endpoint_perturbation_posterior_difference=posterior_difference,
                next_prediction_difference=next_difference, frozen_network_unchanged=True,
                frozen_network_parameter_gradients_absent=True, passed=True)


def run(checkpoint=None, analytic_only=False):
    report = dict(analytic=analytic_checks(), target_files_opened=False, reference_files_opened=False,
                  scientific_fitting=False, query_convention=Q.QUERY_CONVENTION)
    if not analytic_only:
        from pinn_common import N
        N.setup()
        net, record = N.baseline_network(411041)
        if checkpoint is not None:
            state = N.load(checkpoint)
            net.load_state_dict(state['net_state'])
            record = dict(path=str(Path(checkpoint).resolve()), sha256=N.sha(checkpoint))
        net.freeze()
        net.mode = 'temporal'
        source = N.T.SOURCE
        path = source/'prepared/vn0_0065m_s/observations.pt'
        data = N.load(path)
        z = data['z']
        permitted = (z['pid'] % 5 == 1) & (z['frame'] < 4)
        mask = torch.zeros_like(permitted)
        for state in (0, 1):
            ids = torch.where(permitted & (z['state'] == state))[0]
            pids = z['pid'][ids].unique(sorted=True)[:2]
            mask |= permitted & (z['state'] == state) & torch.isin(z['pid'], pids)
        z = Q.core.subset(z, torch.where(mask)[0])
        assert bool((z['pid'] % 5 == 1).all()) and len(z['pid']) > 0
        report['probability'] = probability_checks(net, data['c'], z,
            N.read(source/'calibration.json'), N.read(source/'force.json'))
        report['probability']['source'] = str(path)
        report['probability']['source_sha256'] = N.sha(path)
        report['probability']['network'] = record
    report['sources'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (HERE/'continuous_query.py', Path(__file__).resolve())}
    report['passed'] = True
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--analytic-only', action='store_true')
    args = parser.parse_args()
    result = run(args.checkpoint, args.analytic_only)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps(result, indent=2, ensure_ascii=False))
