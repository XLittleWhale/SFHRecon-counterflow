"""Synthetic-only checks of the captured Markov model and ADF implementation.

Does not open any particle pack, target H, target reference, or checkpoint.
Run with the project Python; optionally save a JSON report using --output.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

import latent_model as LM
from counterflow import observation as ob
from counterflow_full import core


def q_boundary_and_learning_regression():
    """Probe the actual v2 function and train only synthetic disposable heads.

    No formal model is loaded or changed.  The optimization uses the same
    AdamW learning rate, decay and gradient clipping as the formal training.
    These short synthetic runs are checks, not selected experiment networks.
    """
    results = {}
    for dtype in (torch.float32, torch.float64):
        torch.manual_seed(2041104)
        points = torch.tensor([-100., -1e-6, 0., 1e-6, .5, 11.99999, 12., 12.00001, 100.],
                              dtype=dtype, requires_grad=True)
        actual = LM.bounded_kappa(points)
        expected = points.clamp(0., 12.)
        gradients = torch.autograd.grad(actual.sum(), points)[0]
        expected_gradients = points.new_tensor([0., 0., 1., 1., 1., 1., 0., 0., 0.])
        if not torch.equal(actual, expected) or not torch.equal(gradients, expected_gradients):
            raise AssertionError(f"bounded_kappa value/boundary-gradient regression failed for {dtype}")
        broad = torch.linspace(-100., 100., 10001, dtype=dtype)
        if not torch.equal(LM.bounded_kappa(broad), broad.clamp(0., 12.)):
            raise AssertionError("bounded_kappa changed the forward probability parametrization")

        count = 256
        base = ob.Density().to(dtype=dtype)
        with torch.no_grad():
            # The marginal q law is deliberately uncertain; only continuity
            # can explain the synthetic q labels' 95% persistence.
            base.sign.weight.zero_()
            base.sign.bias.zero_()
        head = LM.LatentTransitionHead(base)
        frozen_base = {name: value.detach().clone() for name, value in base.state_dict().items()}
        h = .1*torch.randn(count, 5, 4, dtype=dtype)
        previous_h = h+.01*torch.randn_like(h)
        ctx = .2*torch.randn(count, 17, dtype=dtype)
        previous_ctx = ctx+.02*torch.randn_like(ctx)
        previous_q = torch.arange(count) % 2
        current_q = previous_q.clone()
        current_q[::20] = 1-current_q[::20]
        cal = dict(latent=[0., 0.], sign_temperature=1.)
        ids = torch.arange(count)
        with torch.no_grad():
            _, previous_mu, previous_L = ob.latent(base, previous_h, previous_ctx, cal)
            _, mu, L = ob.latent(base, h, ctx, cal)
            previous_e = torch.randn(count, 2, dtype=dtype)
            current_e = .8*previous_e+.4*torch.randn_like(previous_e)
            previous_y = previous_mu[ids, previous_q]+torch.einsum("bij,bj->bi", previous_L[ids, previous_q], previous_e)
            y = mu[ids, current_q]+torch.einsum("bij,bj->bi", L[ids, current_q], current_e)

        def pair_loss():
            return LM.supervised_pair_nll(base, head, previous_h, previous_ctx, previous_y, previous_q,
                                           h, ctx, y, current_q, cal)

        initial_loss, initial_info = pair_loss()
        initial_categorical = float(initial_info["categorical"].mean().detach())
        initial_kappa = float(initial_info["kappa"].mean().detach())
        if initial_kappa != 0.:
            raise AssertionError("The repaired q head is not exactly iid at initialization")
        initial_loss.mean().backward()
        initial_q_bias_gradient = float(head.network[-1].bias.grad[0])
        initial_q_weight_gradient_norm = float(head.network[-1].weight.grad[0].norm())
        if initial_q_bias_gradient >= 0 or initial_q_weight_gradient_norm <= 0:
            raise AssertionError("Synthetic q persistence did not reach the zero-initialized output row")
        head.zero_grad(set_to_none=True)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-4)
        updates = 32
        for _ in range(updates):
            optimizer.zero_grad(set_to_none=True)
            nll, _ = pair_loss()
            nll.mean().backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        with torch.no_grad():
            final_loss, final_info = pair_loss()
        final_categorical = float(final_info["categorical"].mean())
        final_kappa = float(final_info["kappa"].mean())
        q_weight_norm = float(head.network[-1].weight[0].detach().norm())
        q_bias = float(head.network[-1].bias[0].detach())
        if not (q_weight_norm > 0 and final_kappa > 0 and final_categorical < initial_categorical-1e-5):
            raise AssertionError(f"Synthetic q-learning regression failed for {dtype}")
        if not all(torch.equal(value, frozen_base[name]) for name, value in base.state_dict().items()):
            raise AssertionError("Synthetic q-learning regression modified the frozen base")
        if any(parameter.grad is not None for parameter in base.parameters()):
            raise AssertionError("Synthetic q-learning regression differentiated base weights")
        results[str(dtype).removeprefix("torch.")] = dict(
            probe_input=points.detach().tolist(), probe_output=actual.detach().tolist(), probe_gradient=gradients.tolist(),
            forward_identical_to_clamp=True, lower_boundary_derivative=1., upper_boundary_derivative=0.,
            mini_training=dict(samples=count, updates=updates, objective="complete supervised calibrated latent-pair NLL",
                observed_q_stay_rate=float((current_q == previous_q).double().mean()),
                marginal_q_probability=.5, initial_q_output_weight_gradient_norm=initial_q_weight_gradient_norm,
                initial_q_output_bias_gradient=initial_q_bias_gradient, final_q_output_weight_norm=q_weight_norm,
                final_q_output_bias=q_bias, initial_mean_kappa=initial_kappa, final_mean_kappa=final_kappa,
                initial_categorical_nll=initial_categorical, final_categorical_nll=final_categorical,
                categorical_nll_change=final_categorical-initial_categorical,
                frozen_base_unchanged=True))
    return results


def run():
    torch.set_num_threads(2)
    torch.manual_seed(1041104)
    dtype = torch.float64
    batch = 5
    base = ob.Density().to(dtype=dtype)
    head = LM.LatentTransitionHead(base)
    c = dict(physics=dict(vn_mean_m_s=.004, vs_mean_m_s=-.003, box_m=.001,
                          rho_n_kg_m3=90., rho_s_kg_m3=55., alpha=.2),
             constants=dict(drag=4e-12, eg=1e-12, kf=8e-13, egp=2e-13,
                            mass=5e-15, vol=2e-18, gravity=-1e-15),
             hr=1.5, tension_twice_N=2e-14)
    cal = dict(latent=[.1, -.05], sign_temperature=1.2, sigma=dict(cap=.13, free=.12))
    force = {kind: dict(mean=[.02, -.01], covariance=[[.02, .003], [.003, .03]])
             for kind in ("cap", "free")}
    z = dict(h=.1*torch.randn(batch, 5, 4, dtype=dtype),
             w=.3*torch.randn(batch, 2, dtype=dtype),
             target=.3*torch.randn(batch, 2, dtype=dtype))
    v = .004*torch.randn(batch, 2, 2, dtype=dtype)
    g = torch.randn(batch, 2, 2, 2, dtype=dtype)
    acc = .01*torch.randn(batch, 2, 2, dtype=dtype)
    degree = 12
    errors = {}

    def close(name, actual, expected, tolerance=2e-11):
        error = float((actual-expected).abs().max().detach())
        errors[name] = error
        if not error <= tolerance:
            raise AssertionError(f"{name}: {error} > {tolerance}")

    original = core.distribution(c, z, v, g, acc, "cap", base, cal, force, degree)
    first = LM.step(c, z, v, g, acc, base, head, cal, force, degree)
    for name, actual, expected in zip(("first_pred", "first_cov", "first_logweights"), first[:3], original):
        close(name, actual, expected)
    close("first_evidence", -first[3]["log_evidence"], core.scores(*original, z["target"]))
    for mode in LM.MODES:
        result = LM.step(c, z, v, g, acc, base, head, cal, force, degree,
                         prevstate=first[3], prevctx=first[3]["context"], mode=mode)
        for name, actual, expected in zip(("pred", "cov", "logweights"), result[:3], original):
            close("zero_head_"+mode+"_"+name, actual, expected)

    # Supervised current latent density has exactly the calibrated p0 law at A=0.
    ctx = core.context(c, z, v, g)
    logq, mu, L = ob.latent(base, z["h"], ctx, cal)
    q = torch.tensor([0, 1, 0, 1, 1])
    prevq = 1-q
    ids = torch.arange(batch)
    e = torch.randn(batch, 2, dtype=dtype)
    preve = torch.randn(batch, 2, dtype=dtype)
    y = mu[ids, q]+torch.einsum("bij,bj->bi", L[ids, q], e)
    prevy = mu[ids, prevq]+torch.einsum("bij,bj->bi", L[ids, prevq], preve)
    pair_nll, pair_info = LM.supervised_pair_nll(base, head, z["h"], ctx, prevy, prevq,
                                               z["h"], ctx, y, q, cal)
    expected = -logq[ids, q]+.5*e.square().sum(-1)+L[ids, q].diagonal(dim1=-2, dim2=-1).log().sum(-1)
    expected = expected+torch.log(torch.tensor(2*torch.pi, dtype=dtype))
    close("zero_head_pair_law", pair_nll, expected)
    pair_nll.mean().backward()
    if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in head.parameters()):
        raise AssertionError("Nonfinite gradients at zero spectral matrix")
    if not any(parameter.grad is not None and bool((parameter.grad != 0).any()) for parameter in head.parameters()):
        raise AssertionError("Transition head did not receive supervised gradients")
    if any(parameter.grad is not None for parameter in base.parameters()):
        raise AssertionError("Frozen base received a parameter gradient")
    head.zero_grad(set_to_none=True)

    # Exercise actual time dependence.  All values are synthetic and fixed.
    with torch.no_grad():
        head.network[-1].bias.copy_(torch.tensor([.7, .25, .07, -.04, .2, .02, -.03, -.08, .04, -.1], dtype=dtype))
    changed_z = dict(z, target=z["target"]+torch.tensor([.4, -.3], dtype=dtype))
    state_a = LM.step(c, z, v, g, acc, base, head, cal, force, degree)[3]
    changed = LM.step(c, changed_z, v, g, acc, base, head, cal, force, degree)
    state_b = changed[3]
    for name, actual, expected in zip(("pred", "cov", "logweights"), changed[:3], original):
        close("current_endpoint_cannot_change_"+name, actual, expected)
    if torch.equal(state_a["emean"], state_b["emean"]):
        raise AssertionError("Endpoint did not update posterior latent means")
    temporal_a = LM.step(c, z, v, g, acc, base, head, cal, force, degree,
                         prevstate=state_a, prevctx=ctx, mode="temporal")
    temporal_b = LM.step(c, z, v, g, acc, base, head, cal, force, degree,
                         prevstate=state_b, prevctx=ctx, mode="temporal")
    temporal_difference = float((temporal_a[0]-temporal_b[0]).abs().max().detach())
    if temporal_difference <= 1e-8:
        raise AssertionError("Temporal filter ignored past endpoint posterior")
    reset_a = LM.step(c, z, v, g, acc, base, head, cal, force, degree,
                     prevstate=state_a, prevctx=ctx, mode="reset")
    reset_b = LM.step(c, z, v, g, acc, base, head, cal, force, degree,
                     prevstate=state_b, prevctx=ctx, mode="reset")
    for name, actual, expected in zip(("pred", "cov", "logweights"), reset_a[:3], reset_b[:3]):
        close("reset_ignores_past_endpoint_"+name, actual, expected)
    mask = torch.tensor([True, False, True, False, True])
    partial = LM.step(c, z, v, g, acc, base, head, cal, force, degree,
                     prevstate=state_a, prevctx=ctx, mode="temporal", reset_mask=mask)
    for name, actual, expected in zip(("pred", "cov", "logweights"), partial[:3], original):
        close("segment_reset_"+name, actual[mask], expected[mask])
    for name, actual, expected in zip(("pred", "cov", "logweights"), partial[:3], temporal_a[:3]):
        close("segment_continuation_"+name, actual[~mask], expected[~mask])
    kappa, A, b, innovation = head.parameters_at(z["h"], ctx, ctx)
    close("q_transition_row_normalized", LM.sign_transition(logq, kappa).exp().sum(-1), torch.ones(batch, 2, dtype=dtype))
    close("predictive_weights_normalized", temporal_a[2].exp().sum((1, 2)), torch.ones(batch, dtype=dtype))
    close("posterior_q_normalized", temporal_a[3]["qlogp"].exp().sum(-1), torch.ones(batch, dtype=dtype))
    min_covariance_eigenvalue = float(torch.linalg.eigvalsh(temporal_a[3]["ecov"]).min().detach())
    if min_covariance_eigenvalue <= 0:
        raise AssertionError("Posterior covariance is not positive definite")
    no_target = dict(z)
    del no_target["target"]
    predictive_only = LM.step(c, no_target, v, g, acc, base, head, cal, force, degree, update=False)
    if predictive_only[3] is not None:
        raise AssertionError("update=False must not return a posterior")

    # Frozen head/base still pass derivatives from two endpoints into prior jets.
    head.freeze()
    va = v.clone().requires_grad_()
    vb = (v*.9).clone().requires_grad_()
    previous = LM.step(c, z, va, g, acc, base, head, cal, force, degree)[3]
    nxt = LM.step(c, z, vb, g, acc, base, head, cal, force, degree,
                  prevstate=previous, prevctx=previous["context"], mode="temporal")
    loss = -nxt[3]["log_evidence"].mean()
    gradients = torch.autograd.grad(loss, (va, vb))
    gradient_norms = [float(gradient.norm()) for gradient in gradients]
    if not all(bool(torch.isfinite(gradient).all()) and bool((gradient != 0).any()) for gradient in gradients):
        raise AssertionError("Missing/nonfinite cross-time field derivatives")
    direction = torch.randn_like(va)
    direction /= direction.norm()
    analytic = float((gradients[0]*direction).sum())
    def evaluate(velocity):
        prior = LM.step(c, z, velocity, g, acc, base, head, cal, force, degree)[3]
        after = LM.step(c, z, vb.detach(), g, acc, base, head, cal, force, degree,
                        prevstate=prior, prevctx=prior["context"], mode="temporal")
        return float(-after[3]["log_evidence"].mean())
    epsilon = 1e-7
    numeric = (evaluate(va.detach()+epsilon*direction)-evaluate(va.detach()-epsilon*direction))/(2*epsilon)
    relative = abs(analytic-numeric)/max(1., abs(analytic), abs(numeric))
    if relative > 2e-5:
        raise AssertionError(f"Cross-time finite difference failed: {relative}")
    errors["cross_time_directional_relative_error"] = relative
    if any("base" in name for name in head.state_dict()):
        raise AssertionError("Head checkpoint unexpectedly includes marginal model")

    # Float32 supervised fitting and float64 validation must each use a
    # matching base/head dtype without changing the original frozen base.
    float_head = copy.deepcopy(head).float()
    float_base = float_head._base
    for parameter in float_head.parameters():
        parameter.requires_grad_(True)
    float_nll, _ = LM.supervised_pair_nll(float_base, float_head,
        z["h"].float(), ctx.float(), prevy.float(), prevq,
        z["h"].float(), ctx.float(), y.float(), q, cal)
    float_nll.mean().backward()
    if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in float_head.parameters()):
        raise AssertionError("Float32 supervised fitting has nonfinite gradients")
    if base.cm.dtype != torch.float64 or float_base.cm.dtype != torch.float32:
        raise AssertionError("Deepcopied training/validation base dtype was not isolated")
    converted = copy.deepcopy(float_head).double()
    if converted._base.cm.dtype != torch.float64:
        raise AssertionError("head.double() did not convert its nonregistered base")
    extreme = copy.deepcopy(head)
    with torch.no_grad():
        extreme.network[-1].weight.zero_()
        extreme.network[-1].bias.copy_(torch.tensor([100., 100., 100., 100., 100.,
                                                    100., -100., -100., 100., -100.], dtype=dtype))
    ex_kappa, ex_A, ex_b, ex_L = extreme.parameters_at(z["h"], ctx, ctx)
    max_singular = float(torch.linalg.matrix_norm(ex_A, ord=2).max())
    innovation_min = float(torch.linalg.eigvalsh(ex_L@ex_L.transpose(-1, -2)).min())
    if max_singular > .995+1e-12 or innovation_min <= 0 or not bool((ex_kappa == 12.).all()):
        raise AssertionError("Extreme transition parameters violate the frozen bounds")
    extreme_result = LM.step(c, z, v, g, acc, base, extreme, cal, force, degree,
                            prevstate=state_a, prevctx=ctx, mode="temporal")
    if not all(bool(torch.isfinite(tensor).all()) for tensor in extreme_result[:3]):
        raise AssertionError("Extreme bounded transition is not numerically finite")
    q_regression = q_boundary_and_learning_regression()
    return dict(passed=True, synthetic_only=True, target_H_read=False, target_reference_read=False,
                approximation="per-q Gaussian assumed-density posterior; GH moments",
                degree=degree, batch=batch, checks=errors,
                temporal_prediction_change_from_previous_endpoint=temporal_difference,
                minimum_posterior_covariance_eigenvalue=min_covariance_eigenvalue,
                frozen_network_cross_time_field_gradient_norms=gradient_norms,
                float32_supervised_gradients_finite=True, dtype_copy_isolated=True,
                extreme_A_spectral_norm=max_singular, extreme_innovation_minimum_eigenvalue=innovation_min,
                q_boundary_and_learning_regression=q_regression,
                head_parameters=sum(parameter.numel() for parameter in head.parameters()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered+"\n", encoding="utf8")
    print(rendered)
