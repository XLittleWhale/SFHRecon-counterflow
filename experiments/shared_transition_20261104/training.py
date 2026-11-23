"""Non-target, physical-PID-isolated shared transition-model retraining.

Preparation/profile never authorize formal fitting. Formal runs require an
explicitly frozen protocol; selection freeze verifies all nine complete runs.
No target-case observation or reference file is opened by this module.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
OUT = BASE / "outputs/SharedTransition_20261104"
TRAINING = OUT / "training"
PACKAGE = BASE / "outputs/FullWindowCalibration_20261026/server_package"
SOURCE = BASE / "outputs/FullWindowCalibration_20261026/fullwindow_4mm_results"
sys.path.insert(0, str(PACKAGE))
from counterflow.observation import Density
from counterflow.io import cpu_state, dump, load, save, seed as set_seed, setup
from counterflow_full.core import scores, subset

TRAIN_CASES = ("vn0_0005m_s", "vn0_001m_s", "vn0_002m_s", "vn0_0025m_s",
               "vn0_003m_s", "vn0_005m_s", "vn0_0085m_s")
VALIDATION_CASE = "vn0_0065m_s"
ARMS = ("cap_retrain", "independent", "shared_gru")
SEEDS = (411040, 411041, 411042)
DEFAULT_SETTINGS = dict(epochs=80, minimum_epochs=40, patience=20, batch=1024,
                        lr=1e-4, weight_decay=1e-4, gradient_clip=5.,
                        train_degree=6, validation_degree=20,
                        validation_every=1, dtype="float32", validation_dtype="float64",
                        endpoint_weight=.1, selection_min_delta=1e-5,
                        num_threads=2, scheduler=None, device="cpu")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def model_module():
    import transition_model as T
    return T


def input_paths():
    result = [SOURCE / "training/selected.pt", SOURCE / "calibration.json", SOURCE / "force.json"]
    for case in (*TRAIN_CASES, VALIDATION_CASE):
        result += [SOURCE / "prepared" / case / name for name in ("training.pt", "observations.pt")]
    return result


def source_paths():
    return [HERE / "training.py", HERE / "transition_model.py",
            PACKAGE / "counterflow/observation.py", PACKAGE / "counterflow/io.py",
            PACKAGE / "counterflow_full/core.py", PACKAGE / "counterflow_full/build.py",
            PACKAGE / "counterflow_full/prepare.py"]


def manifest(paths):
    return [dict(path=str(path.resolve()), sha256=sha(path), bytes=path.stat().st_size) for path in paths]


def verify_manifest(items):
    for item in items:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen input/source changed: {item['path']}")


def make_views(names, group):
    """Separate states before accessing captured labels; free latent y is NaN."""
    dtype = torch.float32 if group == "train" else torch.float64
    result, reports = [], []
    for case in names:
        if case not in (*TRAIN_CASES, VALIDATION_CASE):
            raise ValueError("Target or unknown case cannot be loaded for network training")
        path = SOURCE / "prepared" / case
        data = load(path / "training.pt")
        c = load(path / "observations.pt")["c"]
        z = data["z"]
        mask = z["pid"] % 5 >= 2 if group == "train" else z["pid"] % 5 == 1
        report = dict(case=case, group=group, all_rows=len(z["pid"]), states={})
        for state, kind in ((0, "free"), (1, "cap")):
            ix = torch.where(mask & (z["state"] == state))[0]
            if not len(ix):
                raise ValueError(f"Empty {case}/{group}/{kind}")
            b = subset(z, ix, "cpu", dtype)
            if state == 1:
                b.update(subset(data["labels"], ix, "cpu", dtype))
                if not bool(torch.isfinite(b["y"]).all()):
                    raise ValueError("Captured latent training labels must be finite")
            b["context"] = data["context"][ix].to(dtype)
            b["force"] = data["force"][ix].to(dtype)
            for key, value in zip(("v", "g", "acc"), data["jets"]):
                b[key] = value[ix].to(dtype)
            for key in ("h", "target", "context", "force", "v", "g", "acc"):
                if not bool(torch.isfinite(b[key]).all()):
                    raise ValueError(f"Nonfinite {case}/{kind}/{key}")
            pids, inverse, counts = b["pid"].unique(return_inverse=True, return_counts=True)
            b["weight"] = (1 / (len(names) * 2 * len(pids) * counts[inverse].double())).to(dtype)
            result.append(dict(case=case, kind=kind, c=c, z=b))
            report["states"][kind] = dict(rows=len(ix), pids=len(pids),
                                          pid_values=pids.tolist(), weight=float(b["weight"].double().sum()))
        reports.append(report)
    return result, reports


def prepare(settings=None):
    """Write a candidate only; freeze-protocol is a separate explicit operation."""
    if (TRAINING / "protocol.json").exists():
        return verify_protocol()
    settings = DEFAULT_SETTINGS | (settings or {})
    validate_settings(settings)
    _, tr = make_views(TRAIN_CASES, "train")
    _, va = make_views((VALIDATION_CASE,), "validation")
    candidate = dict(version="shared-transition-training-20261104-v1", created_utc=utc(),
                     frozen=False, arms=list(ARMS), seeds=list(SEEDS), settings=settings,
                     train_cases=list(TRAIN_CASES), validation_case=VALIDATION_CASE,
                     train_pid_mask="pid % 5 >= 2", validation_pid_mask="pid % 5 == 1",
                     train_counts=tr, validation_counts=va,
                     initialization="same historical cap network checkpoint; all model parameters retrained",
                     normalization="historical cm/cs buffers retained; no target statistics",
                     calibration="historical calibration.json and force.json frozen",
                     training_loss="equal case/state/PID: free force Gaussian NLL + 0.1 endpoint NLL; cap latent NLL + 0.1 endpoint NLL",
                     constant_arm="cap_retrain free loss is reported but cannot update parameters",
                     selection="minimum balanced validation endpoint NLL, evaluated every epoch including epoch 0",
                     prediction_task="known-state, given-history 1 ms conditional displacement; stable-state intervals only",
                     unused_target_groups="all target F/I/V/H unused by network training/selection",
                     data_loading="non-target packs include other PID splits; only explicit masks enter objectives",
                     target_reference_read_or_H_scored=False,
                     inputs=manifest(input_paths()), sources=manifest(source_paths()))
    candidate["fingerprint"] = fingerprint(candidate)
    dump(TRAINING / "protocol_candidate.json", candidate)
    return candidate


def validate_settings(settings):
    if settings["dtype"] != "float32" or settings["validation_dtype"] != "float64":
        raise ValueError("Training float32 and copied validation float64 are required")
    if settings["device"] != "cpu" or settings["num_threads"] != 2:
        raise ValueError("This local training protocol requires CPU with two threads")
    if not 0 < settings["minimum_epochs"] <= settings["epochs"] or settings["patience"] < 1:
        raise ValueError("Invalid epoch budget")
    if settings["validation_every"] != 1 or settings["scheduler"] is not None:
        raise ValueError("Every-epoch validation and fixed LR are required")


def freeze_protocol():
    path = TRAINING / "protocol.json"
    if path.exists():
        return verify_protocol()
    p = read_json(TRAINING / "protocol_candidate.json")
    original_fp = p.pop("fingerprint")
    if fingerprint(p) != original_fp:
        raise ValueError("Candidate protocol fingerprint mismatch")
    verify_manifest(p["inputs"] + p["sources"])
    p.update(frozen=True, frozen_utc=utc())
    p["fingerprint"] = fingerprint(p)
    dump(path, p)
    return p


def verify_protocol():
    path = TRAINING / "protocol.json"
    if not path.exists():
        raise RuntimeError("No frozen network training protocol; run prepare/profile and freeze-protocol first")
    p = read_json(path)
    body = {k: v for k, v in p.items() if k != "fingerprint"}
    if not p.get("frozen") or fingerprint(body) != p["fingerprint"]:
        raise ValueError("Training protocol is not frozen or fingerprint is invalid")
    validate_settings(p["settings"])
    verify_manifest(p["inputs"] + p["sources"])
    return p


def new_model(arm, sd):
    set_seed(sd)
    base = Density().float()
    base.load_state_dict(load(SOURCE / "training/selected.pt")["state"])
    net = model_module().load_model(arm, base, device="cpu").float()
    for parameter in net.parameters():
        parameter.requires_grad_(True)
    return net


def loss_parts(net, item, b, cal, force, settings):
    T = model_module()
    if item["kind"] == "free":
        label = T.free_force_nll(net, b["h"], b["context"], b["force"], force)
    else:
        label = -net.log_prob(b["h"], b["context"], b["y"], b["q"])
    pred, cov, lw = T.distribution(item["c"], b, b["v"], b["g"], b["acc"],
                                  item["kind"], net, cal, force, settings["train_degree"])
    endpoint = scores(pred, cov, lw, b["target"])
    return label + settings["endpoint_weight"] * endpoint, label, endpoint


@torch.no_grad()
def validation(net, views, cal, force, settings):
    # Never change dtype/device/mode of the training model in-place.
    model = copy.deepcopy(net).double().eval()
    T = model_module()
    components = {kind: dict(nll=0., weight=0.) for kind in ("free", "cap")}
    for item in views:
        z = item["z"]
        for ix in torch.arange(len(z["pid"])).split(settings["batch"]):
            b = subset(z, ix, "cpu", torch.float64)
            pred, cov, lw = T.distribution(item["c"], b, b["v"], b["g"], b["acc"],
                                          item["kind"], model, cal, force, settings["validation_degree"])
            nll = scores(pred, cov, lw, b["target"])
            part = components[item["kind"]]
            part["nll"] += float((b["weight"] * nll).sum())
            part["weight"] += float(b["weight"].sum())
    result = {kind: value["nll"] / value["weight"] for kind, value in components.items()}
    result["total"] = .5 * (result["free"] + result["cap"])
    if not all(math.isfinite(v) for v in result.values()):
        raise FloatingPointError("Nonfinite validation endpoint NLL")
    return result


def train_epoch(net, opt, views, cal, force, settings, generator):
    net.train()
    batches = [(item, ix) for item in views
               for ix in torch.randperm(len(item["z"]["pid"]), generator=generator).split(settings["batch"])]
    report = dict(objective=0., label_nll=0., endpoint_nll=0., batches=len(batches),
                  optimizer_steps=0, rows=0, gradient_norm_max=0., states={})
    for j in torch.randperm(len(batches), generator=generator).tolist():
        item, ix = batches[j]
        b = subset(item["z"], ix, "cpu", torch.float32)
        objective, label, endpoint = loss_parts(net, item, b, cal, force, settings)
        weighted = (b["weight"] * objective).sum()
        value = weighted * len(batches)
        if not bool(torch.isfinite(value)):
            raise FloatingPointError("Nonfinite network training loss")
        opt.zero_grad(set_to_none=True)
        if value.requires_grad:
            value.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), settings["gradient_clip"], error_if_nonfinite=True))
            opt.step()
            report["optimizer_steps"] += 1
            report["gradient_norm_max"] = max(report["gradient_norm_max"], norm)
        report["objective"] += float(weighted.detach())
        report["label_nll"] += float((b["weight"] * label).sum().detach())
        report["endpoint_nll"] += float((b["weight"] * endpoint).sum().detach())
        report["rows"] += len(ix)
        part = report["states"].setdefault(item["kind"], dict(rows=0, objective=0.))
        part["rows"] += len(ix)
        part["objective"] += float(weighted.detach())
    if not all(bool(torch.isfinite(p).all()) for p in net.parameters()):
        raise FloatingPointError("Nonfinite network parameter")
    return report


def rng_state(generator):
    return dict(torch=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state(),
                batch_generator=generator.get_state())


def restore_rng(value, generator):
    torch.set_rng_state(value["torch"])
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    generator.set_state(value["batch_generator"])


def selected_payload(net, arm, sd, epoch, val, p):
    return dict(state=cpu_state(net), arm=arm, seed=sd, epoch=epoch,
                validation=val["total"], validation_components=val,
                protocol_fingerprint=p["fingerprint"],
                pretrained_sha256=sha(SOURCE / "training/selected.pt"),
                initialization="historical cap checkpoint; initialized free head equals historical force law",
                target_reference_read_or_H_scored=False)


def run(arm, sd):
    p = verify_protocol()
    if arm not in ARMS or sd not in SEEDS:
        raise ValueError("Arm/seed not in frozen protocol")
    settings = p["settings"]
    setup("cpu")
    path = TRAINING / arm / f"seed_{sd}"
    status_path = path / "status.json"
    if status_path.exists() and read_json(status_path).get("complete"):
        status = read_json(status_path)
        if status["protocol_fingerprint"] != p["fingerprint"] or sha(path / "selected.pt") != status["selected_sha256"]:
            raise ValueError("Completed run binding mismatch")
        return status
    begun = time.perf_counter()
    tr, _ = make_views(TRAIN_CASES, "train")
    va, _ = make_views((VALIDATION_CASE,), "validation")
    cal, force = read_json(SOURCE / "calibration.json"), read_json(SOURCE / "force.json")
    net = new_model(arm, sd)
    opt = torch.optim.AdamW(net.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    generator = torch.Generator().manual_seed(sd + 173)
    start, best_epoch, trace, elapsed_previous = 1, 0, [], 0.
    if (path / "last.pt").exists():
        ck = load(path / "last.pt")
        if ck["protocol_fingerprint"] != p["fingerprint"] or ck["arm"] != arm or ck["seed"] != sd:
            raise ValueError("Resume checkpoint binding mismatch")
        net.load_state_dict(ck["state"])
        opt.load_state_dict(ck["optimizer"])
        restore_rng(ck["rng"], generator)
        start, best, best_epoch = ck["epoch"] + 1, ck["best"], ck["best_epoch"]
        trace, elapsed_previous = ck["trace"], ck["wall_seconds"]
        # Recover an interrupted best/selected write from the authoritative last state.
        save(path / "selected.pt", ck["selected"])
    else:
        initial = validation(net, va, cal, force, settings)
        best = initial["total"]
        initial_payload = selected_payload(net, arm, sd, 0, initial, p)
        save(path / "initial.pt", initial_payload)
        save(path / "selected.pt", initial_payload)
        trace.append(dict(epoch=0, validation=initial, train=None, wall_seconds=time.perf_counter() - begun))
    dump(status_path, dict(complete=False, arm=arm, seed=sd, protocol_fingerprint=p["fingerprint"],
                          state="running", resume_epoch=start, started_utc=utc()))
    stop = "epoch_budget_not_proof_of_convergence"
    epoch = start - 1
    if epoch >= settings["minimum_epochs"] and epoch - best_epoch >= settings["patience"]:
        stop = "patience"
    else:
        for epoch in range(start, settings["epochs"] + 1):
            epoch_begun = time.perf_counter()
            train = train_epoch(net, opt, tr, cal, force, settings, generator)
            train_seconds = time.perf_counter() - epoch_begun
            val = validation(net, va, cal, force, settings)
            if val["total"] < best - settings["selection_min_delta"]:
                best, best_epoch = val["total"], epoch
                chosen = selected_payload(net, arm, sd, epoch, val, p)
            else:
                chosen = load(path / "selected.pt")
            elapsed = elapsed_previous + time.perf_counter() - begun
            trace.append(dict(epoch=epoch, train=train, validation=val,
                              train_seconds=train_seconds, epoch_seconds=time.perf_counter() - epoch_begun,
                              wall_seconds=elapsed, lr=opt.param_groups[0]["lr"]))
            # Last includes selection so restart is transactional at epoch boundaries.
            save(path / "last.pt", dict(state=cpu_state(net), optimizer=opt.state_dict(), rng=rng_state(generator),
                                        epoch=epoch, best=best, best_epoch=best_epoch, selected=chosen,
                                        arm=arm, seed=sd, trace=trace, wall_seconds=elapsed,
                                        protocol_fingerprint=p["fingerprint"]))
            save(path / "selected.pt", chosen)
            dump(path / "trace.json", trace)
            dump(status_path, dict(complete=False, state="running", arm=arm, seed=sd, epoch=epoch,
                                  selected_epoch=best_epoch, validation=best, wall_seconds=elapsed,
                                  protocol_fingerprint=p["fingerprint"]))
            print(f"TRAIN {arm} seed={sd} epoch={epoch} train={train['objective']:.6f} V={val['total']:.6f} best={best:.6f}@{best_epoch} seconds={trace[-1]['epoch_seconds']:.2f}", flush=True)
            if epoch >= settings["minimum_epochs"] and epoch - best_epoch >= settings["patience"]:
                stop = "patience"
                break
    # Recheck immutable bindings before a completed run can be published.
    verify_protocol()
    status = dict(complete=True, state="complete", arm=arm, seed=sd, last_epoch=epoch,
                  selected_epoch=best_epoch, validation=best, stopping=stop,
                  wall_seconds=elapsed_previous + time.perf_counter() - begun, finished_utc=utc(),
                  selected_path=str((path / "selected.pt").resolve()), selected_sha256=sha(path / "selected.pt"),
                  last_sha256=sha(path / "last.pt"), trace_sha256=sha(path / "trace.json"),
                  protocol_fingerprint=p["fingerprint"], converged_claim=False)
    dump(status_path, status)
    return status


def profile(arm="shared_gru", sd=SEEDS[0]):
    """One disposable epoch, never usable as a final training checkpoint."""
    setup("cpu")
    p = read_json(TRAINING / "protocol_candidate.json") if (TRAINING / "protocol_candidate.json").exists() else prepare()
    if fingerprint({k: v for k, v in p.items() if k != "fingerprint"}) != p["fingerprint"]:
        raise ValueError("Profile candidate fingerprint mismatch")
    verify_manifest(p["inputs"] + p["sources"])
    settings = p["settings"]
    begun = time.perf_counter()
    tr, _ = make_views(TRAIN_CASES, "train")
    va, _ = make_views((VALIDATION_CASE,), "validation")
    cal, force = read_json(SOURCE / "calibration.json"), read_json(SOURCE / "force.json")
    net = new_model(arm, sd)
    opt = torch.optim.AdamW(net.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    generator = torch.Generator().manual_seed(sd + 173)
    first = time.perf_counter()
    initial = validation(net, va, cal, force, settings)
    initial_seconds = time.perf_counter() - first
    first = time.perf_counter()
    train = train_epoch(net, opt, tr, cal, force, settings, generator)
    train_seconds = time.perf_counter() - first
    first = time.perf_counter()
    val = validation(net, va, cal, force, settings)
    validation_seconds = time.perf_counter() - first
    result = dict(profile_only=True, excluded_from_formal_parameters_and_selection=True,
                  arm=arm, seed=sd, settings=settings, initial_validation=initial, validation=val,
                  train=train, train_seconds=train_seconds, validation_seconds=validation_seconds,
                  initial_validation_seconds=initial_seconds, total_seconds=time.perf_counter() - begun,
                  estimated_80epoch_run_seconds=80 * (train_seconds + validation_seconds),
                  parameter_count=sum(v.numel() for v in net.parameters()),
                  torch_version=torch.__version__, num_threads=torch.get_num_threads(),
                  protocol_candidate_fingerprint=p["fingerprint"], finished_utc=utc())
    dump(TRAINING / "profiles" / f"{arm}_seed_{sd}.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def freeze_selections():
    p = verify_protocol()
    runs = []
    for arm in ARMS:
        for sd in SEEDS:
            path = TRAINING / arm / f"seed_{sd}"
            status = read_json(path / "status.json")
            if not status.get("complete") or status["protocol_fingerprint"] != p["fingerprint"]:
                raise RuntimeError(f"Incomplete or mismatched training run {arm}/{sd}")
            for filename, key in (("selected.pt", "selected_sha256"), ("last.pt", "last_sha256"), ("trace.json", "trace_sha256")):
                if sha(path / filename) != status[key]:
                    raise ValueError(f"Completed run {filename} hash changed")
            ck = load(path / "selected.pt")
            if (ck["protocol_fingerprint"], ck["arm"], ck["seed"], ck["epoch"]) != (p["fingerprint"], arm, sd, status["selected_epoch"]):
                raise ValueError("Selected checkpoint metadata mismatch")
            trace = read_json(path / "trace.json")
            if [row["epoch"] for row in trace] != list(range(status["last_epoch"] + 1)):
                raise ValueError("Incomplete or duplicated epoch trace")
            expected = trace[0]
            for row in trace[1:]:
                if row["validation"]["total"] < expected["validation"]["total"] - p["settings"]["selection_min_delta"]:
                    expected = row
            if (ck["epoch"] != expected["epoch"] or ck["validation_components"] != expected["validation"]
                    or ck["validation"] != expected["validation"]["total"]
                    or ck["pretrained_sha256"] != sha(SOURCE / "training/selected.pt")):
                raise ValueError("Selection differs from frozen validation rule or pretrained binding")
            runs.append(dict(arm=arm, seed=sd, selected_epoch=ck["epoch"], validation=ck["validation"],
                             path=str((path / "selected.pt").resolve()), sha256=sha(path / "selected.pt"),
                             status_path=str((path / "status.json").resolve()), status_sha256=sha(path / "status.json")))
    gate = dict(all_required_selections_frozen=True, protocol_fingerprint=p["fingerprint"],
                runs=runs, frozen_utc=utc(), target_reference_read_or_H_scored=False)
    gate["fingerprint"] = fingerprint(gate)
    gate_path = TRAINING / "selection_frozen.json"
    if gate_path.exists():
        old = read_json(gate_path)
        if fingerprint({k: v for k, v in old.items() if k != "fingerprint"}) != old["fingerprint"]:
            raise ValueError("Existing training selection gate fingerprint mismatch")
        if old["runs"] != runs or old["protocol_fingerprint"] != p["fingerprint"]:
            raise ValueError("An existing training selection freeze cannot be replaced")
        return old
    dump(gate_path, gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "profile", "freeze-protocol", "run", "run-all", "freeze-selections", "verify"))
    parser.add_argument("--arm", choices=ARMS, default="shared_gru")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--minimum-epochs", type=int)
    parser.add_argument("--patience", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        settings = {key: value for key, value in vars(args).items() if key in ("epochs", "minimum_epochs", "patience") and value is not None}
        p = prepare(settings)
        print(json.dumps(dict(path=str(TRAINING / "protocol_candidate.json"), fingerprint=p["fingerprint"], settings=p["settings"]), indent=2))
    elif args.command == "profile":
        profile(args.arm, args.seed)
    elif args.command == "freeze-protocol":
        print(json.dumps(freeze_protocol(), indent=2))
    elif args.command == "verify":
        print(verify_protocol()["fingerprint"])
    elif args.command == "run":
        print(json.dumps(run(args.arm, args.seed), indent=2))
    elif args.command == "run-all":
        for arm in ARMS:
            for sd in SEEDS:
                run(arm, sd)
    elif args.command == "freeze-selections":
        print(json.dumps(freeze_selections(), indent=2))


if __name__ == "__main__":
    main()
