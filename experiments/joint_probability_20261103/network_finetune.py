"""Restricted observation-network adaptation, separate from field inversion.

F = physical PID modulo 5 equal to 2; I = modulo 5 in {3, 4}.
Only captured F responses train the network. All field/closure inputs are
detached. Replay is drawn once from the seven historical training cases,
never from either reconstruction target, validation case, or held-out PIDs.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parents[3]
OUTPUTS = PROJECT / "outputs"
PACKAGE = OUTPUTS / "FullWindowCalibration_20261026" / "server_package"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))
from counterflow_full.core import context, distribution, scores, subset
from counterflow.observation import latent

_spec = importlib.util.spec_from_file_location(
    "joint_finetune_fixed_noise", OUTPUTS / "NormalNoiseAblation_20261030" / "noise_model.py")
_noise = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_noise)
local_jets = _noise.local_jets

TRAIN_CASES = ("vn0_0005m_s", "vn0_001m_s", "vn0_002m_s", "vn0_0025m_s",
               "vn0_003m_s", "vn0_005m_s", "vn0_0085m_s")
TARGET_CASES = {"vn0_0015m_s", "vn0_004m_s"}


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def configure_heads(student):
    """Return Adam-ready params and serializable names/counts; freeze all else."""
    names, parameters = [], []
    for name, parameter in student.named_parameters():
        enabled = name.startswith("sign.") or name.startswith("output.")
        parameter.requires_grad_(enabled)
        parameter.grad = None
        if enabled:
            names.append(name)
            parameters.append(parameter)
    count = sum(p.numel() for p in parameters)
    if count != 462:
        raise ValueError(f"Expected the fixed 462-parameter sign/output heads, got {count}")
    return dict(params=parameters, names=names, trainable_count=count,
                frozen_count=sum(p.numel() for p in student.parameters()) - count,
                ineffective_output_rows=[0, 6],
                note="Two single-component mixture-logit rows are unused by the inherited density.")


def preflight_contract(d):
    """Metadata-only checks; no V/H responses or target reference are accessed."""
    pid, state = d["z"]["pid"], d["z"]["state"]
    f = pid.remainder(5) == 2
    inv = pid.remainder(5) >= 3
    fp, ip = set(pid[f].tolist()), set(pid[inv].tolist())
    if fp & ip or not fp or not ip or not bool((f & (state == 1)).any()):
        raise ValueError("Invalid or empty physical-PID-separated F/I partition")
    return dict(F_I_disjoint=True, F_physical_pids=len(fp), I_physical_pids=len(ip),
                F_captured_rows=int((f & (state == 1)).sum()),
                I_label="inverse-fit set, not a held-out prediction test")


def prepare_replay(run_dir, seed=311103, rows_per_case=32):
    """Pick one captured record from each of 32 distinct train PIDs per case.

    The fixed generator chooses PIDs and a row within each selected PID.
    Equal case/PID weighting gives every selected row weight 1/(7*32).
    Returns (list of compact replay batches, auditable selection manifest).
    """
    run_dir = Path(run_dir)
    if rows_per_case <= 0:
        raise ValueError("rows_per_case must be positive")
    generator = torch.Generator().manual_seed(seed)
    replay, reports = [], []
    for name in TRAIN_CASES:
        if name in TARGET_CASES:
            raise AssertionError("Target case cannot enter historical replay")
        path = run_dir / "prepared" / name / "training.pt"
        observations_path = path.with_name("observations.pt")
        data = torch.load(path, map_location="cpu", weights_only=False)
        c = torch.load(observations_path, map_location="cpu", weights_only=False)["c"]
        z = data["z"]
        good = (z["pid"].remainder(5) >= 2) & (z["state"] == 1)
        pids = z["pid"][good].unique(sorted=True)
        if len(pids) < rows_per_case:
            raise ValueError(f"Insufficient distinct training capture PIDs in {name}")
        chosen = pids[torch.randperm(len(pids), generator=generator)[:rows_per_case]]
        rows = []
        for pid in chosen:
            candidates = torch.where(good & (z["pid"] == pid))[0]
            index = int(torch.randint(len(candidates), (1,), generator=generator))
            rows.append(int(candidates[index]))
        ix = torch.tensor(rows, dtype=torch.long)
        compact = subset(z, ix, "cpu")
        compact.update(subset(data["labels"], ix, "cpu"))
        compact["context"] = data["context"][ix].double()
        compact["weight"] = torch.full((len(ix),), 1 / (len(TRAIN_CASES) * len(ix)), dtype=torch.float64)
        for key, value in zip(("v", "g", "acc"), data["jets"]):
            compact[key] = value[ix].double()
        replay.append(dict(case=name, c=c, z=compact))
        reports.append(dict(case=name, rows=rows, pids=chosen.tolist(),
                            training_path=str(path.resolve()), training_sha256=_sha(path),
                            observations_sha256=_sha(observations_path)))
    manifest = dict(seed=seed, rows_per_case=rows_per_case, cases=reports,
                    total_rows=len(TRAIN_CASES) * rows_per_case,
                    selection="one random captured interval per distinct old-training physical PID",
                    old_training_pid_mask="pid % 5 >= 2", target_truth_read=False,
                    historical_supervision="old training labels q/y and original reference-conditioned jets only")
    return replay, manifest


def joint_latent_kl(student, teacher, h, ctx, cal):
    """KL(teacher(q,delta|h,ctx) || student(q,delta|h,ctx)), exact per row."""
    with torch.no_grad():
        p0, m0, l0 = latent(teacher, h, ctx, cal)
    p1, m1, l1 = latent(student, h, ctx, cal)
    transformed_covariance = torch.linalg.solve_triangular(l1, l0, upper=False)
    transformed_mean = torch.linalg.solve_triangular(l1, (m0 - m1)[..., None], upper=False)
    logdet_ratio = (l1.diagonal(dim1=-2, dim2=-1).log().sum(-1)
                    - l0.diagonal(dim1=-2, dim2=-1).log().sum(-1))
    gaussian = logdet_ratio + .5 * (transformed_covariance.square().sum((-1, -2))
                                    + transformed_mean.square().sum((-1, -2)) - 2)
    return (p0.exp() * (p0 - p1 + gaussian)).sum(-1)


def _replay_loss(student, replay, force, device, degree):
    # This is precisely the original supervised objective, before its outer weight.
    calibration = dict(latent=[0., 0.], sign_temperature=1., sigma=dict(free=.3, cap=.3))
    result = next(student.parameters()).new_zeros(())
    for item in replay:
        z = subset(item["z"], torch.arange(len(item["z"]["pid"])), device)
        nll_latent = -student.log_prob(z["h"], z["context"], z["y"], z["q"])
        pred, cov, lw = distribution(item["c"], z, z["v"], z["g"], z["acc"],
                                     "cap", student, calibration, force, degree)
        result = result + (z["weight"] * (nll_latent + .1 * scores(pred, cov, lw, z["target"]))).sum()
    return result


def network_step(a, es, en, d, cal, force, student, teacher, optimizer, replay,
                 batch=256, degree=12, replay_degree=6, lambda_kl=.1,
                 lambda_replay=.1, clip=1.):
    """One complete F-only network update with frozen-teacher KL and old replay.

    Call configure_heads (or restore its named requires_grad mask) before use.
    Inputs a/es/en are never differentiated or modified. Both regularizers are
    additive to equal-PID captured F NLL. No .08 field-loss scaling is applied.
    """
    partition = preflight_contract(d)
    if any(p.requires_grad for p in teacher.parameters()):
        raise ValueError("Teacher must remain frozen")
    trainable = {n: p for n, p in student.named_parameters() if p.requires_grad}
    if not trainable or any(not (n.startswith("sign.") or n.startswith("output.")) for n in trainable):
        raise ValueError("Only the sign/output heads may be trainable")
    opt_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if opt_ids != {id(p) for p in trainable.values()}:
        raise ValueError("Optimizer must contain exactly the trainable network heads")
    original_fields = [x.detach().clone() for x in (a, es, en)]
    fields = [x.detach() for x in (a, es, en)]
    before = {n: p.detach().clone() for n, p in student.named_parameters()}
    teacher_before = {n: p.detach().clone() for n, p in teacher.named_parameters()}
    device = next(student.parameters()).device
    if any(x.device != device for x in fields):
        raise ValueError("Network and field inputs must use the same device")
    optimizer.zero_grad(set_to_none=True)
    z = d["z"]
    ids = torch.where((z["pid"].remainder(5) == 2) & (z["state"] == 1))[0]
    _, inverse, counts = z["pid"][ids].unique(return_inverse=True, return_counts=True)
    weight = (1. / (len(counts) * counts[inverse].double())).to(device)
    f_nll, kl = 0., 0.
    student.eval()  # eval() leaves autograd enabled and preserves the fixed encoder convention.
    teacher.eval()
    for start in range(0, len(ids), batch):
        zz = subset(z, ids[start:start + batch], device)
        ww = weight[start:start + batch]
        with torch.no_grad():
            v, g, acc = local_jets(*fields, zz, d["c"])
            ctx = context(d["c"], zz, v, g)
        pred, cov, lw = distribution(d["c"], zz, v, g, acc, "cap", student, cal, force, degree)
        current_nll = (ww * scores(pred, cov, lw, zz["target"])).sum()
        current_kl = (ww * joint_latent_kl(student, teacher, zz["h"], ctx, cal)).sum()
        (current_nll + lambda_kl * current_kl).backward()
        f_nll += float(current_nll.detach())
        kl += float(current_kl.detach())
    replay_loss = _replay_loss(student, replay, force, device, replay_degree)
    (lambda_replay * replay_loss).backward()
    gradient = float(torch.nn.utils.clip_grad_norm_(list(trainable.values()), clip, error_if_nonfinite=True))
    optimizer.step()
    if not all(torch.equal(x.detach(), old) for x, old in zip((a, es, en), original_fields)):
        raise AssertionError("Network step modified an input flow/closure tensor")
    if not all(torch.equal(p.detach(), teacher_before[n]) for n, p in teacher.named_parameters()):
        raise AssertionError("Network step modified its frozen teacher")
    changes = {n: float((p.detach() - before[n]).abs().max()) for n, p in student.named_parameters()}
    if any(changes[n] != 0 for n in changes if n not in trainable):
        raise AssertionError("Network step modified a frozen encoder parameter")
    if not all(torch.isfinite(p).all() for p in student.parameters()):
        raise FloatingPointError("Nonfinite adapted network parameter")
    replay_value = float(replay_loss.detach())
    return dict(F_captured_nll=f_nll, teacher_student_KL=kl, replay_supervised_loss=replay_value,
                objective=f_nll + lambda_kl * kl + lambda_replay * replay_value,
                lambda_kl=lambda_kl, lambda_replay=lambda_replay, gradient_norm_before_clip=gradient,
                gradient_clip=clip, changed_weights={n: v for n, v in changes.items() if v != 0},
                maximum_weight_change=max(changes.values()), fields_unchanged=True,
                teacher_unchanged=True, frozen_encoder_unchanged=True,
                F_rows=len(ids), F_PIDs=len(counts), replay_rows=sum(len(x["z"]["pid"]) for x in replay),
                partition=partition)
