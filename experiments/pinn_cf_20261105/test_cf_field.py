"""Finite field/operator checks only; no H score, optimization, or CF selection."""
import json
import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import cf_field as cf


def run(case_name='cf_113_camera100_f0184'):
    started = time.perf_counter()
    cpu_started = time.process_time()
    torch.set_num_threads(2)
    torch.manual_seed(21371)
    c, z, op, cold, prior = cf.load_case(case_name)
    ring = cf.wall_ring(c, cold)
    field = cf.make_field(c, cold, seed=301030)
    assert ring.shape == (2, op.m) and op.m == 325
    assert len(field.states()) == 101
    assert torch.count_nonzero(field.states()) == 0
    assert all(p['external_scale_multiplier'] == 1 and p['dropped_singular_directions'] == 0
               for p in c['pinn_output_preconditioning']['factors'])
    with torch.no_grad():
        field.output.weight.normal_(std=1e-5)
        field.output.bias.normal_(std=1e-5)
    t = cold.new_tensor([.0001, .0152, .0513, .0999])
    a, d = field.coefficients_and_dt(t)
    eps = 1e-7
    fd = (field(t+eps)-field(t-eps))/(2*eps)
    derivative_relative = float(((d-fd).norm()/d.norm()).detach())
    assert derivative_relative < 1e-7
    rows = cold.new_empty(3, 2, op.m).normal_(std=1e-3)
    batch = cf.batched_rhs(op, rows)
    scalar = torch.stack([op.rhs(row) for row in rows])
    rhs_relative = float((batch-scalar).norm()/scalar.norm())
    assert rhs_relative < 1e-11
    # Both walls and both periodic-y edges, at random nonzero coefficients.
    W, H = c['domain_m']
    x = cold.new_tensor([[0., .002], [W, .004], [.004, 0.], [.004, H]])
    zz = dict(time=t.reshape(1, -1)*1000, x=x.reshape(1, -1, 2))
    aa = rows[0].expand(4, -1, -1).clone()
    dd = rows[1].expand(4, -1, -1).clone()
    v, g, acc = cf.queries_from_coefficients(c, aa, dd, zz)
    boundary = max(float(v[0, :2, 0].abs().max()), float(v[0, :2, 1, 0].abs().max()))
    divergence = float(g.diagonal(dim1=-2, dim2=-1).sum(-1).abs().max())
    periodic = max(float((v[0, 2]-v[0, 3]).abs().max()), float((g[0, 2]-g[0, 3]).abs().max()))
    assert boundary < 1e-12 and divergence < 1e-12 and periodic < 1e-12
    # A reduced full-rank wall space permits an independent explicit dense C^-1 check.
    raw = cf.runtime().ProjectedPrior(c, op, intervals=7)
    small = SimpleNamespace(intervals=7, projection_checks={}, projection=raw.projection[:, :6],
                            normal_prior=raw.normal_prior, std_s=raw.std_s)
    p = cf.WallResidualPrior(small)
    rn = cold.new_empty(7, 6).normal_(std=1e-5).requires_grad_()
    rs = cold.new_empty(7, 6).normal_(std=1e-5).requires_grad_()
    qn, qs = p.quadratic(rn, rs)
    cn = rn.new_zeros(42, 42)
    for k, ix in enumerate(raw.normal_prior.indices):
        A = small.projection[0, :, ix]*raw.normal_prior.std[ix]
        L = raw.normal_prior.cholesky[k]
        cn += torch.kron((L@L.T).contiguous(), (A@A.T).contiguous())
    As = small.projection[1]*raw.std_s
    cs = torch.kron(torch.eye(7, dtype=As.dtype), (As@As.T).contiguous())
    exact_n = rn.flatten()@torch.linalg.solve(cn, rn.flatten())
    exact_s = rs.flatten()@torch.linalg.solve(cs, rs.flatten())
    dense_relative = max(float(((qn-exact_n).abs()/exact_n).detach()), float(((qs-exact_s).abs()/exact_s).detach()))
    assert dense_relative < 1e-8
    gn, gs = torch.autograd.grad(qn+qs, (rn, rs))
    grad_relative = max(float(((gn.flatten()-2*torch.linalg.solve(cn, rn.flatten())).norm()/gn.norm()).detach()),
                        float(((gs.flatten()-2*torch.linalg.solve(cs, rs.flatten())).norm()/gs.norm()).detach()))
    assert grad_relative < 1e-8
    # Full 325D whitening inversion without forming a squared-condition covariance.
    white = cold.new_empty(100, op.m).normal_()
    col = torch.matmul(prior.normal_temporal_cholesky, white.T.unsqueeze(-1)).squeeze(-1).T
    residual = torch.linalg.solve(prior.normal_transform.T, col.T).T
    roundtrip = float((prior.whiten_normal(residual)-white).norm()/white.norm())
    assert roundtrip < 1e-8
    # Parameter gradients of SI residual and spectral terms at the full 201-node grid.
    t0 = time.perf_counter()
    a, d = field.coefficients_and_dt(torch.arange(201).to(cold)*.0005)
    loss, details = cf.residual_loss(a, d, op, prior, nfit=100)
    reg, _ = field.regularization(a[::2])
    (loss+reg).backward()
    gradient_finite = all(p.grad is not None and torch.isfinite(p.grad).all() for p in field.parameters())
    assert gradient_finite
    # The same physical field has exactly the same original objective in the
    # archived unpreconditioned optimizer coordinates. This checks the last
    # layer's invertible reparameterization rather than comparing two fits.
    archive = cf.BASE/'outputs/PINNCF_20261105/preflight/archive/before_B_output_preconditioning/cf_field.py'
    spec = importlib.util.spec_from_file_location('cf_field_before_output_preconditioning', archive)
    before = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(before)
    old = before.NeuralField(cold, ring, seed=301030, decode=c['pinn_decode'], encode=c['pinn_encode'])
    old.load_state_dict({key:value for key,value in field.state_dict().items() if key != 'output_preconditioner'})
    with torch.no_grad():
        W = field.output.weight.reshape(2, op.m, -1)
        b = field.output.bias.reshape(2, op.m)
        old.output.weight.copy_((torch.einsum('fij,fjk->fik', c['pinn_output_preconditioner'], W*field.scale[...,None])
                                /field.scale[...,None]).reshape_as(old.output.weight))
        old.output.bias.copy_((torch.einsum('fij,fj->fi',c['pinn_output_preconditioner'],b*field.scale)
                              /field.scale).reshape_as(old.output.bias))
    old_a, old_d = old.coefficients_and_dt(torch.arange(201).to(cold)*.0005)
    old_loss, _ = cf.residual_loss(old_a, old_d, op, prior, nfit=100)
    old_reg, _ = old.regularization(old_a[::2])
    equivalent_field_error = max(float(((old_a-a).norm()/a.norm()).detach()),
                                 float(((old_d-d).norm()/d.norm()).detach()))
    equivalent_objective_error = float(((old_loss+old_reg-loss-reg).abs()/(loss+reg).abs()).detach())
    equivalent_regularization_error = float(((old_reg-reg).abs()/reg.abs()).detach())
    assert equivalent_field_error < 1e-10 and equivalent_objective_error < 1e-8 and equivalent_regularization_error < 1e-10
    q = field.normalized_coefficients(a)
    result = dict(case=case_name, complete=True, no_target_H_scored=True, no_optimization=True,
                  field_parameters=sum(p.numel() for p in field.parameters()),
                  source_basis='original CF Shen325, K12, 16x8mm',
                  time_derivative_relative=derivative_relative, rhs_relative=rhs_relative,
                  boundary_velocity_max_abs_m_s=boundary, divergence_max_abs_s_inv=divergence,
                  periodic_value_gradient_max_abs=periodic, dense_prior_energy_relative=dense_relative,
                  dense_prior_gradient_relative=grad_relative, full_prior_whitening_roundtrip_relative=roundtrip,
                  finite_parameter_gradients=bool(gradient_finite),
                  one_full_residual_forward_backward_seconds=time.perf_counter()-t0,
                  perturbed_test_residual_loss=float(loss.detach()),
                  prior_diagnostics=prior.diagnostics, projection_checks=prior.projection_checks,
                  spatial_normalization=c['pinn_spatial_normalization'],
                  output_preconditioning=c['pinn_output_preconditioning'],
                  same_physical_field_relative_error=equivalent_field_error,
                  same_physical_objective_relative_error=equivalent_objective_error,
                  same_physical_regularization_relative_error=equivalent_regularization_error,
                  random_1e_5_output_weight_field_rms_mm_s=(q.square().sum(-1).mean(0)/2).sqrt().detach().mul(c['physics']['vn_mean_m_s']*1000).tolist(),
                  wall_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu_started)
    return result


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = dict(complete=True, cases=[run(name) for name in ('cf_113_camera100_f0184', 'cf_217_camera100_f0333')])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
        print(json.dumps(dict(complete=True, path=str(args.out.resolve()),
                              wall_seconds=sum(c['wall_seconds'] for c in result['cases']),
                              cpu_seconds=sum(c['cpu_seconds'] for c in result['cases']))))
    else:
        print(json.dumps(result, indent=2))
