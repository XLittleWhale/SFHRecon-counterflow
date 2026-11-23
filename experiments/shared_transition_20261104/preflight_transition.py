"""Focused double-precision transition-law checks on old non-target data only."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

import transition_model as T
from counterflow_full.train import model as old_model


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(actual, expected, tolerance=1e-12):
    value = float((actual.detach() - expected.detach()).abs().max())
    if value > tolerance:
        raise AssertionError(f"Mismatch {value} > {tolerance}")
    return value


def field_derivative(net, z, c, jets, cal, force):
    v, g, acc = [x.detach().clone().requires_grad_(True) for x in jets]
    generator = torch.Generator().manual_seed(411104)
    directions = [torch.randn(x.shape, generator=generator, dtype=x.dtype) * scale
                  for x, scale in zip((v, g, acc), (c["physics"]["vn_mean_m_s"] * .05, .1, .01))]
    def loss(vv, gg, aa):
        pred, cov, lw = T.distribution(c, z, vv, gg, aa, "free", net, cal, force, 8)
        return T.scores(pred, cov, lw, z["target"]).mean()
    gradients = torch.autograd.grad(loss(v, g, acc), (v, g, acc), allow_unused=True)
    gradients = [torch.zeros_like(x) if grad is None else grad for x, grad in zip((v, g, acc), gradients)]
    ad = sum(float((grad * direction).sum()) for grad, direction in zip(gradients, directions))
    epsilon = 1e-5
    with torch.no_grad():
        plus = loss(*[x + epsilon * direction for x, direction in zip((v, g, acc), directions)])
        minus = loss(*[x - epsilon * direction for x, direction in zip((v, g, acc), directions)])
        fd = float((plus - minus) / (2 * epsilon))
    error = abs(ad - fd)
    if error > 1e-7 + 2e-5 * max(abs(ad), abs(fd)):
        raise AssertionError(f"Field derivative mismatch: AD {ad}, FD {fd}")
    return dict(autograd=ad, central_difference=fd, epsilon=epsilon, absolute_error=error,
                tolerance="1e-7 + 2e-5*max(abs(AD),abs(FD))",
                gradient_norms={name: float(grad.norm()) for name, grad in zip(("v", "g", "acc"), gradients)})


def batch_partition_check(net, z, c, jets, cal, force):
    """Row-dependent Gaussian covariance must survive arbitrary batch splits."""
    fields = [x.detach().clone().requires_grad_(True) for x in jets]
    n = len(z["pid"])
    def calculate(batch):
        values = []
        for ix in torch.arange(n).split(batch):
            zz = T.core.subset(z, ix, "cpu")
            pred, cov, lw = T.distribution(c, zz, *[x[ix] for x in fields],
                                           "free", net, cal, force, 8)
            values.append(T.scores(pred, cov, lw, zz["target"]))
        values = torch.cat(values)
        gradients = torch.autograd.grad(values.mean(), fields, allow_unused=True)
        gradients = [torch.zeros_like(x) if g is None else g for x, g in zip(fields, gradients)]
        return values.detach(), [g.detach() for g in gradients]
    expected, expected_gradients = calculate(n)
    checks = {}
    for batch in (1, 3):
        values, gradients = calculate(batch)
        checks[str(batch)] = dict(score_error=compare(values, expected, tolerance=1e-11),
                                 gradient_errors={key: compare(a, b, tolerance=1e-9)
                                                  for key, a, b in zip(("v", "g", "acc"), gradients, expected_gradients)})
    with torch.no_grad():
        _, covariance, _ = T.distribution(c, z, *jets, "free", net, cal, force, 8)
        covariance_variation = float((covariance - covariance[:1]).abs().max())
    if net.arm != "cap_retrain" and covariance_variation <= 1e-14:
        raise AssertionError("The learned free-law batch check did not exercise heterogeneous covariances")
    return dict(row_covariance_max_difference=covariance_variation, partitions=checks,
                implementation="Per-row core.scores; legacy fast_scores assumes constant covariance and is not permitted for the learned free branch")


def preflight(output):
    torch.set_num_threads(2)
    run = T.PROJECT / "outputs/FullWindowCalibration_20261026/fullwindow_4mm_results"
    training_path = run / "prepared/vn0_002m_s/training.pt"
    observations_path = training_path.with_name("observations.pt")
    data = torch.load(training_path, map_location="cpu", weights_only=False)
    c = torch.load(observations_path, map_location="cpu", weights_only=False)["c"]
    cal = json.loads((run / "calibration.json").read_text(encoding="utf-8"))
    force = json.loads((run / "force.json").read_text(encoding="utf-8"))
    baseline = old_model(run, "cpu")
    report = dict(version="shared-transition-preflight-v1", created_utc=datetime.now(timezone.utc).isoformat(),
                  precision="torch.float64", training_case="vn0_002m_s",
                  target_observations_or_reference_read=False, H_scored=False,
                  description="Only four old-training free and four old-training captured rows; synthetic parameter perturbations for local derivative checks; no optimization or formal training.", arms={})
    batches = {}
    for kind, state in (("free", 0), ("cap", 1)):
        mask = (data["z"]["state"] == state) & (data["z"]["pid"] % 5 >= 2)
        ix = torch.where(mask)[0][:4]
        if len(ix) != 4:
            raise AssertionError("Insufficient old training records")
        z = T.core.subset(data["z"], ix, "cpu")
        jets = tuple(x[ix].double() for x in data["jets"])
        batches[kind] = (z, jets, data["force"][ix].double())
    for arm in T.ARMS:
        net = T.load_model(arm, baseline).double()
        row = dict(parameter_audit=net.parameter_audit(), initial_parity={}, normalized_density={})
        for kind in ("free", "cap"):
            z, jets, labels = batches[kind]
            with torch.no_grad():
                expected = T.core.distribution(c, z, *jets, kind, baseline, cal, force, 8)
                actual = T.distribution(c, z, *jets, kind, net, cal, force, 8)
                row["initial_parity"][kind] = {key: compare(x, y) for key, x, y in zip(("prediction", "covariance", "log_weight"), actual, expected)}
                pred, cov, lw = actual
                minimum_eigenvalue = float(torch.linalg.eigvalsh(cov).min())
                if minimum_eigenvalue <= 0:
                    raise AssertionError("Endpoint covariance is not positive definite")
                weight_error = float((lw.exp().sum((1, 2)) - 1).abs().max())
                if weight_error > 1e-12:
                    raise AssertionError("Mixture mass is not one")
                info = dict(minimum_covariance_eigenvalue=minimum_eigenvalue, mixture_weight_sum_error=weight_error)
                if kind == "free":
                    independent = -torch.distributions.MultivariateNormal(pred[:, 0, 0], covariance_matrix=cov[:, 0, 0]).log_prob(z["target"])
                    info["endpoint_nll_error_against_torch_gaussian"] = compare(T.scores(pred, cov, lw, z["target"]), independent)
                    ctx = T.context(c, z, jets[0], jets[1])
                    mean, factor = net.free_parameters(z["h"], ctx, force)
                    expected_force = -torch.distributions.MultivariateNormal(mean, scale_tril=factor).log_prob(labels)
                    info["force_nll_error_including_anchor_jacobian"] = compare(T.free_force_nll(net, z["h"], ctx, labels, force), expected_force)
                    changed = dict(z, target=z["target"] + 17., pid=z["pid"] + 100000,
                                   frame=z["frame"] + 1234, x=z["x"] + 100.)
                    altered = T.distribution(c, changed, *jets, kind, net, cal, force, 8)
                    info["no_explicit_target_pid_time_position_input"] = all(compare(x, y) == 0 for x, y in zip(actual, altered))
                row["normalized_density"][kind] = info
        # Round-trip reconstruction does not alias any source model parameters.
        restored = T.load_model(arm, baseline, state=copy.deepcopy(net.state_dict()))
        row["state_roundtrip_exact"] = all(torch.equal(x, restored.state_dict()[k]) for k, x in net.state_dict().items())
        if not row["state_roundtrip_exact"]:
            raise AssertionError("State serialization identity failed")
        z, jets, labels = batches["free"]
        ctx = T.context(c, z, jets[0], jets[1])
        initial_loss = T.free_force_nll(net, z["h"], ctx, labels, force).mean()
        if arm == "cap_retrain":
            if initial_loss.requires_grad:
                raise AssertionError("Fixed free branch unexpectedly depends on captured parameters")
            row["gradient_routing"] = dict(fixed_free_has_no_network_gradient=True)
        else:
            net.zero_grad(set_to_none=True)
            initial_loss.backward()
            row["zero_output_first_step"] = dict(output_receives_gradient=net.free_output.weight.grad is not None,
                                                 free_encoder_initial_gradient_norm=sum(float(p.grad.norm()) for p in net.free_head.parameters() if p.grad is not None),
                                                 note="A zero linear output initially blocks gradients to its encoder; this is expected, not a routing failure.")
            with torch.no_grad():
                net.free_output.weight.copy_(torch.linspace(-.015, .015, net.free_output.weight.numel(), dtype=torch.float64).reshape_as(net.free_output.weight))
                net.free_output.bias.copy_(torch.tensor([-.03, .02, .01, -.02, .02], dtype=torch.float64))
            net.zero_grad(set_to_none=True)
            T.free_force_nll(net, z["h"], ctx, labels, force).mean().backward()
            norm = lambda module: sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None)
            row["gradient_routing"] = dict(captured_gru_norm=norm(net.gru),
                                           private_free_gru_norm=norm(net.free_gru) if arm == "independent" else None,
                                           captured_private_head_norm=norm(net.head),
                                           free_private_head_norm=norm(net.free_head))
            if norm(net.head) != 0 or norm(net.free_head) <= 0:
                raise AssertionError("Private conditional MLP gradient routing failed")
            if arm == "independent" and (norm(net.gru) != 0 or norm(net.free_gru) <= 0):
                raise AssertionError("Independent free branch leaked into captured GRU")
            if arm == "shared_gru" and norm(net.gru) <= 0:
                raise AssertionError("Shared free branch did not differentiate shared GRU")
            m, S = net.free_standardized_parameters(z["h"], ctx)
            diag = S.diagonal(dim1=-2, dim2=-1)
            if not ((m.abs() < 2).all() and (diag >= 1 / 3).all() and (diag <= 3).all() and (S[:, 1, 0].abs() < 1).all()):
                raise AssertionError("Free-force parameter bounds violated")
            row["free_bounds"] = dict(mean_max_abs=float(m.detach().abs().max()), minimum_diagonal=float(diag.detach().min()),
                                      maximum_diagonal=float(diag.detach().max()), maximum_off_diagonal_abs=float(S[:, 1, 0].detach().abs().max()))
        row["field_gradient"] = field_derivative(net, z, c, jets, cal, force)
        row["heteroscedastic_batch_partition"] = batch_partition_check(net, z, c, jets, cal, force)
        report["arms"][arm] = row
    report["source_hashes"] = {str(path.resolve()): sha(path) for path in
        [Path(__file__), Path(T.__file__), Path(T.core.__file__), training_path,
         observations_path, run / "training/selected.pt", run / "calibration.json", run / "force.json"]}
    report["passed"] = True
    report["limitations"] = ["Focused local algebra/gradient checks do not establish reconstruction performance or global identifiability.",
                             "No target case H/reference was opened; old non-target training jets and force labels contain allowed simulator supervision.",
                             "The bound on triangular diagonals is not a bound on covariance eigenvalues because an off-diagonal term is allowed."]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(passed=True, output=str(output.resolve()),
                         parameter_counts={a: r["parameter_audit"]["total_parameters"] for a, r in report["arms"].items()},
                         field_gradient_errors={a: r["field_gradient"]["absolute_error"] for a, r in report["arms"].items()}), indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=T.PROJECT / "outputs/SharedTransition_20261104/preflight/transition_checks.json")
    args = parser.parse_args()
    preflight(args.out)
