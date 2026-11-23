"""One diagnostic field update, using authorized I rows and a fresh frozen net.

This never trains/selects a scientific probability model or field checkpoint.
An optional 10 ms prior is a partial-PDE smoke test only. The default uses the
entire prepared case's prior and field duration while limiting native rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

import field_adapter as F
import native_probability as Q
import reconstruct


def network_digest(net):
    h = hashlib.sha256()
    for key, value in sorted(net.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def memory():
    try:
        import psutil
        info = psutil.Process().memory_info()
        return {"rss_GiB": info.rss/2**30,
                "peak_GiB": getattr(info, "peak_wset", info.rss)/2**30}
    except ImportError:
        return {}


def run(prepared, output=None, captured_rows=4, pde_intervals=None, degree=6):
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    started = time.perf_counter()
    pack = torch.load(Path(prepared)/"inversion_observations.pt", map_location="cpu", weights_only=False)
    meta, source = pack["meta"], pack["z"]
    indices = []
    for state in (0, 1):
        allowed = Q.group_mask(source["pid"], "I") & (source["state"] == state)
        # Longest actual paths expose sequence memory; never include V/H rows.
        pids, counts = source["pid"][allowed].unique(return_counts=True)
        if len(pids) < 2:
            raise ValueError("Benchmark requires at least two I PIDs in each state")
        for pid in pids[counts.argsort(descending=True)[:2]]:
            ix = torch.where(allowed & (source["pid"] == pid))[0]
            ix = ix[source["start"][ix].argsort(stable=True)]
            limit = (2 if state == 0 else captured_rows)
            indices.append(ix if limit == 0 else ix[:limit])
    ix = torch.cat(indices)
    z = {k: v[ix].clone() for k, v in source.items()}
    intervals = int(meta["intervals"]) if pde_intervals is None else pde_intervals
    c, op, cold, prior = F.prepare_physics(meta["physics_config"], intervals)
    field = F.make_field(c, cold, 301030, duration_s=meta["intervals"]*.001)
    net = Q.NativeNetwork(411040).freeze()
    cal, force = Q.calibration()
    before = network_digest(net)
    opt = torch.optim.Adam(field.parameters(), lr=.001)
    prepared_seconds = time.perf_counter()-started
    print(json.dumps(dict(stage="prepared", rows=len(ix), intervals=intervals,
                          captured_rows=int((z["state"] == 1).sum()),
                          seconds=prepared_seconds, **memory())), flush=True)
    update_started = time.perf_counter()
    score_seconds = []
    original_score = reconstruct.score
    def measured_score(*args, **kwargs):
        start = time.perf_counter()
        value = original_score(*args, **kwargs)
        score_seconds.append(time.perf_counter()-start)
        return value
    reconstruct.score = measured_score
    try:
        result = reconstruct.field_step(c, z, op, prior, field, opt, net, cal, force, degree)
    finally:
        reconstruct.score = original_score
    update_seconds = time.perf_counter()-update_started
    if network_digest(net) != before or any(p.grad is not None or p.requires_grad for p in net.parameters()):
        raise AssertionError("Frozen probability changed or acquired parameter gradients")
    if not all(bool(torch.isfinite(p).all()) for p in field.parameters()):
        raise AssertionError("Nonfinite field parameter after one update")
    report = dict(diagnostic_only=True, scientific_result=False, prepared=str(Path(prepared).resolve()),
                  rows=len(ix), captured_rows=int((z["state"] == 1).sum()),
                  selected_physical_pids=z["pid"].unique().tolist(),
                  pde_intervals=intervals, full_pde=intervals == int(meta["intervals"]),
                  field_duration_s=meta["intervals"]*.001, degree=degree,
                  prepare_seconds=prepared_seconds, update_seconds=update_seconds,
                  likelihood_seconds=sum(score_seconds),
                  pde_regularization_adam_seconds=update_seconds-sum(score_seconds),
                  probability_frozen_exact=True, probability_sha256=before,
                  finite_parameters=True, **memory(), result=result)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--captured-rows", type=int, default=4,
                        help="per captured PID; 0 retains complete paths")
    parser.add_argument("--pde-intervals", type=int, default=None)
    parser.add_argument("--degree", type=int, default=6)
    args = parser.parse_args()
    run(args.prepared, args.output, args.captured_rows, args.pde_intervals, args.degree)
