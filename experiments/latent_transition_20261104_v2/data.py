"""Non-target latent transition data and complete-PID sequence layouts.

This module never loads target reference fields, target latent labels or target
held-out responses. Sequence utilities accept already-authorized observation
tensors but do not themselves open a target-case file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
PARENT = HERE.parent / "shared_transition_20261104"
OUT = BASE / "outputs/LatentTransition_20261104_v2"
FRAMES = 95
_MODULES = {}


def _source_module(name, path):
    """Import parent sources under unique names, without a `training` alias."""
    if name not in _MODULES:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODULES[name] = module
    return _MODULES[name]


def parent_training():
    return _source_module("_latent_transition_parent_shared_training", PARENT / "training.py")


def load_views(group):
    """Return captured-only non-target views, with equal-case pair mass.

    Each returned view retains the parent's case/kind/c/z contract. ``z`` has
    current-time q/y labels and causal histories. Its old row ``weight`` is
    retained only for compatibility; use ``pairs(view)`` for transition weights.
    """
    old = parent_training()
    if group == "train":
        names = old.TRAIN_CASES
    elif group == "validation":
        names = (old.VALIDATION_CASE,)
    else:
        raise ValueError("Only non-target train or validation views are allowed")
    views, _ = old.make_views(names, group)
    return [dict(view, case_weight=1.0 / len(names), group=group)
            for view in views if view["kind"] == "cap"]


def _consecutive(z, mask=None):
    n = len(z["pid"])
    mask = torch.ones(n, dtype=torch.bool) if mask is None else torch.as_tensor(mask, dtype=torch.bool).cpu()
    if mask.shape != (n,):
        raise ValueError("mask must have one boolean entry per observation")
    indices = torch.where(mask)[0].tolist()
    pids = z["pid"].detach().cpu().tolist()
    frames = z["frame"].detach().cpu().tolist()
    states = z["state"].detach().cpu().tolist()
    lookup = {}
    for i in indices:
        key = (int(pids[i]), int(frames[i]))
        if key in lookup:
            raise ValueError("Duplicate PID/frame observation")
        lookup[key] = i
    edges = []
    for (pid, frame), i in sorted(lookup.items()):
        j = lookup.get((pid, frame + 1))
        if j is not None and states[i] == states[j]:
            edges.append((i, j))
    if not edges:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty.clone()
    previous, current = torch.tensor(edges, dtype=torch.long).T
    return previous, current


def pairs(view):
    """Return source-row indices and equal-case/PID/edge probability weights.

    A PID's mass is divided by its number of available consecutive edges, then
    by the number of PIDs having at least one edge in this case. The view's mass
    is ``case_weight``. q changes are retained; q is not used to join/split data.
    """
    z = view["z"]
    if view.get("kind") != "cap" or not bool((z["state"] == 1).all()):
        raise ValueError("Latent transition pairs require captured-only views")
    previous, current = _consecutive(z)
    dtype = z["h"].dtype
    if not len(previous):
        return previous, current, torch.empty(0, dtype=dtype)
    _, inverse, counts = z["pid"].cpu()[current].unique(return_inverse=True, return_counts=True)
    weight = float(view.get("case_weight", 1.0)) / (len(counts) * counts[inverse].double())
    return previous, current, weight.to(dtype)


def sequence_layout(z, mask):
    """Pack captured rows into original frame 0..94, one whole row per PID.

    Returns pid[B], index[B,95], valid[B,95], reset[B,95]. ``index`` addresses
    the *original* z, including if z contains both states. Missing/free frames
    are -1. A valid frame starts a new chain exactly when its predecessor is
    missing; no target latent truth, q value or future response enters resets.
    """
    n = len(z["pid"])
    mask = torch.as_tensor(mask, dtype=torch.bool).cpu()
    if mask.shape != (n,):
        raise ValueError("mask must have one boolean entry per observation")
    chosen = torch.where(mask & (z["state"].detach().cpu() == 1))[0]
    pid_all = z["pid"].detach().cpu()
    frame_all = z["frame"].detach().cpu()
    if len(chosen) and not bool(((frame_all[chosen] >= 0) & (frame_all[chosen] < FRAMES)).all()):
        raise ValueError("Expected original observation frames 0..94 (5..99 ms)")
    pids = pid_all[chosen].unique(sorted=True)
    index = torch.full((len(pids), FRAMES), -1, dtype=torch.long)
    if len(chosen):
        row = torch.searchsorted(pids, pid_all[chosen])
        key = row * FRAMES + frame_all[chosen]
        if len(key.unique()) != len(key):
            raise ValueError("Duplicate captured PID/frame observation")
        index[row, frame_all[chosen]] = chosen
    valid = index >= 0
    reset = valid.clone()
    reset[:, 1:] &= ~valid[:, :-1]
    return dict(pid=pids, index=index, valid=valid, reset=reset)


def pack_sequences(z, mask, batch_pids=16):
    """Return a list of complete-PID layout dictionaries; never split time."""
    if not isinstance(batch_pids, int) or batch_pids < 1:
        raise ValueError("batch_pids must be a positive integer")
    full = sequence_layout(z, mask)
    return [{key: tensor[start:start + batch_pids] for key, tensor in full.items()}
            for start in range(0, len(full["pid"]), batch_pids)]


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _manifest(paths):
    return [dict(path=str(Path(p).resolve()), sha256=_sha(p), bytes=Path(p).stat().st_size)
            for p in dict.fromkeys(paths)]


def _correlation(previous, current):
    return [float(np.corrcoef(previous[:, j], current[:, j])[0, 1]) for j in range(2)]


def write_audit():
    """Reproduce allowed-data counts and seed-411040 calibrated lag1 audit."""
    old = parent_training()
    tm = _source_module("_latent_transition_parent_density", PARENT / "transition_model.py")
    from counterflow.observation import Density, latent
    torch.set_num_threads(2)
    base_checkpoint = old.SOURCE / "training/selected.pt"
    selected = BASE / "outputs/SharedTransition_20261104/training/independent/seed_411040/selected.pt"
    base = Density().double()
    base.load_state_dict(torch.load(base_checkpoint, weights_only=False)["state"])
    net = tm.load_model("independent", base).double().eval()
    net.load_state_dict(torch.load(selected, weights_only=False)["state"])
    cal_path = old.SOURCE / "calibration.json"
    cal = json.loads(cal_path.read_text(encoding="utf8"))
    inputs = [base_checkpoint, selected, cal_path]
    result = dict(version="latent-transition-data-audit-v1", created_utc=datetime.now(timezone.utc).isoformat(),
                  native_dt_seconds=.001, interval_frame_range=[0, 94], interval_start_ms_range=[5, 99],
                  latent_pair_time_ms_range=[5, 99], observed_particle_count_per_case=512,
                  train_cases=list(old.TRAIN_CASES), validation_case=old.VALIDATION_CASE,
                  train_pid_mask="pid % 5 >= 2", validation_pid_mask="pid % 5 == 1",
                  pair_rule="same physical PID; frame increments by one; same observed state; retain q flips",
                  weighting="equal case, equal PID with at least one edge, equal edges within that PID",
                  latent_definition="q=current attached vortex sign; y=periodic shortest endpoint anchor displacement / particle radius",
                  force_definition="dimensionless unresolved full/coarse physical force discrepancy, not native white-noise truth",
                  exclusions="No target observations, reference fields, target latent labels, or target H scoring",
                  caveat="Observed same-state intervals do not rule out unresolved microscopic release/recapture",
                  standardized_definition="e=L_calibrated^{-1}(y-mu), independent selected seed 411040, existing calibration",
                  cases=[], pooled={}, target_reference_read_or_H_scored=False)
    pooled = {"train": [], "validation": []}
    for group, names in (("train", old.TRAIN_CASES), ("validation", (old.VALIDATION_CASE,))):
        views, reports = old.make_views(names, group)
        by_case = {r["case"]: r for r in reports}
        for case in names:
            inputs += [old.SOURCE / "prepared" / case / name for name in ("training.pt", "observations.pt")]
            original = torch.load(old.SOURCE / "prepared" / case / "training.pt", weights_only=False)
            record = dict(case=case, group=group, rows=sum(s["rows"] for s in by_case[case]["states"].values()), states={})
            for view in [v for v in views if v["case"] == case]:
                state = 1 if view["kind"] == "cap" else 0
                split = original["z"]["pid"] % 5 == 1 if group == "validation" else original["z"]["pid"] % 5 >= 2
                ix = torch.where(split & (original["z"]["state"] == state))[0]
                # Keep original float64 labels/context for reproducible auditing;
                # load_views intentionally retains the parent's float32 training.
                z = old.subset(original["z"], ix, "cpu", torch.float64)
                z["force"] = original["force"][ix].double()
                z["context"] = original["context"][ix].double()
                if state == 1:
                    z.update(old.subset(original["labels"], ix, "cpu", torch.float64))
                previous, current = _consecutive(z)
                lookup = {(int(pid), int(frame)) for pid, frame in zip(z["pid"], z["frame"])}
                starts = [(pid, frame) for pid, frame in lookup if (pid, frame - 1) not in lookup]
                lengths = []
                for pid, frame in starts:
                    length = 1
                    while (pid, frame + length) in lookup:
                        length += 1
                    lengths.append(length)
                entry = dict(rows=len(z["pid"]), pids=len(z["pid"].unique()), pairs=len(previous), segments=len(starts),
                             length_mean=float(np.mean(lengths)), length_median=float(np.median(lengths)),
                             length_max=max(lengths), segments_ge5=sum(v >= 5 for v in lengths),
                             segments_ge10=sum(v >= 10 for v in lengths), force_finite=bool(torch.isfinite(z["force"]).all()),
                             force_std=z["force"].double().std(0).tolist())
                if view["kind"] == "cap":
                    y = z["y"].double()
                    eparts = []
                    with torch.no_grad():
                        for ix in torch.arange(len(y)).split(1024):
                            _, mu, factor = latent(net, z["h"][ix].double(), z["context"][ix].double(), cal)
                            qq, jj = z["q"][ix], torch.arange(len(ix))
                            eparts.append(torch.linalg.solve_triangular(factor[jj, qq],
                                (y[ix] - mu[jj, qq])[..., None], upper=False)[..., 0])
                    e = torch.cat(eparts)
                    same = z["q"][previous] == z["q"][current]
                    entry.update(y_finite=bool(torch.isfinite(y).all()), y_std=y.std(0).tolist(),
                                 q_flips=int((~same).sum()), q_flip_rate=float((~same).double().mean()),
                                 raw_delta_lag1_all=_correlation(y[previous], y[current]),
                                 raw_delta_lag1_sameq=_correlation(y[previous[same]], y[current[same]]),
                                 standardized_delta_lag1_all=_correlation(e[previous], e[current]),
                                 standardized_delta_lag1_sameq=_correlation(e[previous[same]], e[current[same]]),
                                 standardized_delta_mean=e.mean(0).tolist(), standardized_delta_std=e.std(0, unbiased=False).tolist())
                    pooled[group].append((y[previous], y[current], e[previous], e[current], same))
                    layout = sequence_layout(z, torch.ones(len(y), dtype=torch.bool))
                    if int(layout["valid"].sum()) != len(y) or int(layout["reset"].sum()) != len(starts):
                        raise AssertionError("Sequence packing lost rows or changed chain boundaries")
                    pv, cu, weight = pairs(dict(view, z=z, case_weight=1.0 / len(names)))
                    if not torch.equal(previous, pv) or not torch.equal(current, cu):
                        raise AssertionError("Pair utility differs from audit")
                    if abs(float(weight.double().sum()) - 1.0 / len(names)) > 1e-7:
                        raise AssertionError("Pair weighting mass mismatch")
                record["states"][view["kind"]] = entry
            result["cases"].append(record)
        values = [torch.cat([r[i] for r in pooled[group]]) for i in range(5)]
        yp, yc, ep, ec, same = values
        rows = [r for r in result["cases"] if r["group"] == group]
        result["pooled"][group] = dict(rows=sum(r["rows"] for r in rows),
            free_pairs=sum(r["states"]["free"]["pairs"] for r in rows), captured_pairs=len(yp),
            q_flips=int((~same).sum()), q_flip_rate=float((~same).double().mean()),
            raw_delta_lag1=_correlation(yp, yc), standardized_delta_lag1=_correlation(ep, ec),
            pooling_note="Descriptive edge-pooled correlations; training loss uses equal case/PID/edge weights")
    result["inputs"] = _manifest(inputs)
    result["sources"] = _manifest([Path(__file__), PARENT / "training.py", PARENT / "transition_model.py",
        old.PACKAGE / "counterflow/observation.py", old.PACKAGE / "counterflow_full/build.py",
        old.PACKAGE / "counterflow_full/prepare.py", old.PACKAGE / "counterflow_full/core.py"])
    target = OUT / "preflight/data_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(write_audit()["pooled"], indent=2))
