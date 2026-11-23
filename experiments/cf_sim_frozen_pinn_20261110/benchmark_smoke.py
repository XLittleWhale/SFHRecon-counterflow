"""One diagnostic field update, using I rows and the original frozen joint net.

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

import common
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
    result = {}
    try:
        import psutil
        info = psutil.Process().memory_info()
        result.update(rss_GiB=info.rss/2**30,
                      peak_GiB=getattr(info, "peak_wset", info.rss)/2**30)
    except ImportError:
        pass
    if torch.cuda.is_available() and common.device().startswith('cuda'):
        target = common.device()
        result.update(cuda_peak_allocated_GiB=torch.cuda.max_memory_allocated(target)/2**30,
                      cuda_peak_reserved_GiB=torch.cuda.max_memory_reserved(target)/2**30)
    return result


def synchronize():
    if common.device().startswith('cuda'):
        torch.cuda.synchronize(common.device())


def run(prepared, output=None, captured_rows=4, pde_intervals=None, degree=6, captured_pids=2):
    common.setup()
    target = common.device()
    if target.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(target)
    synchronize()
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
        for pid in pids[counts.argsort(descending=True)[:(2 if state == 0 else captured_pids)]]:
            ix = torch.where(allowed & (source["pid"] == pid))[0]
            ix = ix[source["start"][ix].argsort(stable=True)]
            limit = (2 if state == 0 else captured_rows)
            indices.append(ix if limit == 0 else ix[:limit])
    ix = torch.cat(indices)
    z = common.move_tensors({k: v[ix].clone() for k, v in source.items()}, target)
    intervals = int(meta["intervals"]) if pde_intervals is None else pde_intervals
    c, op, cold, prior = F.prepare_physics(meta["physics_config"], intervals, device=target)
    field = F.make_field(c, cold, 301030, duration_s=meta["intervals"]*.001)
    case = Path(prepared).name
    net, cal, force, source_digest = common.probability(case, 301030, device=target)
    before = network_digest(net)
    opt = torch.optim.Adam(field.parameters(), lr=.001)
    synchronize()
    prepared_seconds = time.perf_counter()-started
    print(json.dumps(dict(stage="prepared", rows=len(ix), intervals=intervals,
                          captured_rows=int((z["state"] == 1).sum()),
                          seconds=prepared_seconds, **memory())), flush=True)
    update_started = time.perf_counter()
    score_seconds = []
    original_score = reconstruct.score
    def measured_score(*args, **kwargs):
        synchronize()
        start = time.perf_counter()
        value = original_score(*args, **kwargs)
        synchronize()
        score_seconds.append(time.perf_counter()-start)
        return value
    reconstruct.score = measured_score
    try:
        result = reconstruct.field_step(c, z, op, prior, field, opt, net, cal, force, degree)
    finally:
        reconstruct.score = original_score
    synchronize()
    update_seconds = time.perf_counter()-update_started
    if network_digest(net) != before or any(p.grad is not None or p.requires_grad for p in net.parameters()):
        raise AssertionError("Frozen probability changed or acquired parameter gradients")
    if not all(bool(torch.isfinite(p).all()) for p in field.parameters()):
        raise AssertionError("Nonfinite field parameter after one update")
    report = dict(diagnostic_only=True, scientific_result=False, prepared=str(Path(prepared).resolve()),
                  device=target, precision='float64',
                  gpu_name=torch.cuda.get_device_name(target) if target.startswith('cuda') else None,
                  rows=len(ix), captured_rows=int((z["state"] == 1).sum()),
                  selected_physical_pids=z["pid"].unique().tolist(),
                  pde_intervals=intervals, full_pde=intervals == int(meta["intervals"]),
                  field_duration_s=meta["intervals"]*.001, degree=degree,
                  prepare_seconds=prepared_seconds, update_seconds=update_seconds,
                  likelihood_seconds=sum(score_seconds),
                  pde_regularization_adam_seconds=update_seconds-sum(score_seconds),
                  probability_frozen_exact=True, probability_state_sha256=before,
                  probability_checkpoint_sha256=source_digest,
                  probability_source='Nov5 frozen joint baseline', probability_optimization_steps=0,
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
    parser.add_argument("--captured-pids", type=int, default=2)
    args = parser.parse_args()
    run(args.prepared, args.output, args.captured_rows, args.pde_intervals, args.degree, args.captured_pids)
