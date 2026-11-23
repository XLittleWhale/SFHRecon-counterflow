"""Read-only CF simulation adapter with native endpoints and PID isolation.

Reference jets are *camera-rate interpolants*, not substep flow truth. Only F/V
particle rows get such jets. Inversion files never contain reference fields.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
import torch

DT = .001
VERSION = "cf-sim-native-adapter-v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def _pieces(start_s, end_s):
    """Tile one interval on the global 1 ms grid, without rounding endpoints."""
    first = math.floor(start_s / DT + 1e-10) + 1
    last = math.ceil(end_s / DT - 1e-10) - 1
    middle = np.arange(first, last + 1, dtype=np.float64) * DT
    edges = np.r_[start_s, middle, end_s]
    duration = np.diff(edges)
    if np.any(duration <= 0) or np.any(duration > DT + 1e-12):
        raise ValueError("Invalid native interval subdivision")
    return (edges[:-1] + edges[1:]) / 2, duration


def build_observations(observed: dict, hidden: dict, configuration: dict):
    """Return all eligible native intervals; force/anchor truth is never read.

    ``configuration`` is the full source configuration JSON. Row filtering sees
    native endpoint state/attachment labels, as explicitly required by this
    synthetic calibration protocol. It cannot certify sub-camera state stability.
    """
    physics = configuration["config"]
    fps = float(configuration["camera_fps"])
    speed = float(physics["vn_mean_m_s"])
    time = np.asarray(observed["time_s"], np.float64)
    xy = np.asarray(observed["xy_m"], np.float64)
    pid = np.asarray(observed["particle_id"], np.int64)
    segment = np.asarray(observed["segment_id"], np.int64)
    state = np.asarray(hidden["state"], np.int64)
    attached = np.asarray(hidden["attached_vortex"], np.int64)
    bounds = np.array([physics["width_m"], physics["height_m"]], np.float64)
    if speed <= 0 or len(time) < 3 or xy.shape != (len(time), len(pid), 2):
        raise ValueError("Invalid velocity, time count or observation shape")
    if any(a.shape != xy.shape[:2] for a in (segment, state, attached)):
        raise ValueError("State/segment shape does not match observations")
    if len(np.unique(pid)) != len(pid) or np.any(pid < 0):
        raise ValueError("Physical particle IDs must be unique and nonnegative")
    if not np.array_equal(pid, hidden["particle_id"]) or not np.array_equal(time, hidden["time_s"]):
        raise ValueError("Observation/truth identity or time mismatch")
    if not np.allclose(time, np.arange(len(time)) / fps, rtol=0, atol=1e-12):
        raise ValueError("Source timestamps do not match the native camera clock")
    if not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy > bounds):
        raise ValueError("Source positions are nonfinite or outside the channel")
    if not np.isin(state, [0, 1]).all() or not np.array_equal(state == 1, attached >= 0):
        raise ValueError("Invalid state/attachment labels")
    origin = float(time[1])
    duration = float(time[-1] - origin)
    intervals = int(math.ceil(duration / DT - 1e-10))
    rejected = Counter()
    rows = []
    # Rows remain chronological within each physical PID; frame is the original
    # camera interval index. State changes in the past are deliberately allowed.
    for j, particle in enumerate(pid):
        for i in range(1, len(time) - 1):
            if segment[i, j] != segment[i + 1, j]:
                rejected["target_crosses_periodic_segment"] += 1
                continue
            if segment[i - 1, j] != segment[i, j]:
                rejected["insufficient_same_segment_past"] += 1
                continue
            if state[i, j] != state[i + 1, j]:
                rejected["endpoint_state_change"] += 1
                continue
            if state[i, j] == 1 and attached[i, j] != attached[i + 1, j]:
                rejected["captured_endpoint_vortex_change"] += 1
                continue
            lo, hi = time[i:i + 2]
            query_s, widths = _pieces(float(lo - origin), float(hi - origin))
            query_abs = query_s + origin
            velocity = (xy[i, j] - xy[i - 1, j]) / (time[i] - time[i - 1])
            pilot = xy[i, j] + (query_abs - lo)[:, None] * velocity
            if np.any(query_abs < time[0]) or np.any(query_abs > time[-1]):
                rejected["pilot_time_outside_source"] += 1
                continue
            if np.any(pilot < 0) or np.any(pilot > bounds) or not np.isfinite(pilot).all():
                rejected["pilot_position_outside_domain"] += 1
                continue
            begin = max(0, i - 4)
            while segment[begin, j] != segment[i, j]:
                begin += 1
            past_t = time[begin:i + 1]
            if len(past_t) < 2:
                rejected["insufficient_same_segment_past"] += 1
                continue
            curve = PchipInterpolator(past_t, xy[begin:i + 1, j], axis=0, extrapolate=False)
            history_times = query_abs[:, None] + np.arange(-5, 1) * DT
            # Causal PCHIP is evaluated only through the interval origin. Later
            # pilot samples use the previous measured secant, never the target.
            past = curve(np.minimum(history_times, lo))
            future = history_times > lo
            linear = xy[i, j] + (history_times - lo)[..., None] * velocity
            past[future] = linear[future]
            if not np.isfinite(past).all():
                rejected["insufficient_five_ms_history"] += 1
                continue
            history = np.concatenate((np.diff(past, axis=1) / (speed * DT),
                                      (past[:, 1:] - past[:, -1:]) / (5 * speed * DT)), axis=-1)
            rows.append(dict(pid=int(particle), state=int(state[i, j]), segment=int(segment[i, j]),
                             frame=i, start=(lo - origin) / DT, end=(hi - origin) / DT,
                             time=query_s / DT, dt=widths, x=pilot, w=velocity / speed,
                             target=(xy[i + 1, j] - xy[i, j]) / (speed * DT), h=history))
    if not rows:
        raise ValueError("No eligible native camera intervals")
    pieces = max(len(row["dt"]) for row in rows)
    result = {}
    for key in ("pid", "state", "segment", "frame"):
        result[key] = torch.tensor([row[key] for row in rows], dtype=torch.int64)
    for key in ("start", "end", "w", "target"):
        result[key] = torch.from_numpy(np.asarray([row[key] for row in rows], np.float64))
    for key in ("time", "dt", "x", "h"):
        tail = rows[0][key].shape[1:]
        array = np.empty((len(rows), pieces) + tail, np.float64)
        for n, row in enumerate(rows):
            length = len(row[key])
            array[n, :length] = row[key]
            array[n, length:] = 0 if key == "dt" else row[key][-1]
        result[key] = torch.from_numpy(array)
    result["row_id"] = torch.arange(len(rows), dtype=torch.int64)
    result["source_row"] = result["row_id"].clone()
    if not torch.allclose(result["dt"].sum(1) / DT, result["end"] - result["start"],
                          rtol=0, atol=1e-9):
        raise ValueError("Native intervals are not exactly tiled")
    meta = dict(version=VERSION, camera_fps=fps, source_camera_frames=len(time),
                first_target_interval_frame=1, last_target_interval_frame=len(time) - 2,
                origin_s=origin, duration_s=duration, intervals=intervals,
                frame_zero_role="causal history only", physics_config=physics,
                time_units="start/end/time in milliseconds relative to origin_s; dt in seconds",
                positions_units="x in metres; w=previous secant/U; target=raw displacement/(U*0.001s)",
                history="5x4 causal 5ms PCHIP features; mixed past states allowed; previous secant beyond origin",
                state_filter="same camera endpoint state; captured endpoint vortex IDs must also agree",
                state_filter_limit="Endpoint agreement cannot establish sub-camera state stability",
                split_rule="F:pid%5==2; I:pid%5>=3; V:pid%5==1; H:pid%5==0")
    audit = dict(candidate_intervals=(len(time) - 2) * len(pid), accepted_intervals=len(rows),
                 rejected=dict(rejected), maximum_internal_pieces=pieces,
                 mixed_past_state_history_allowed=True, target_endpoint_used_in_history=False)
    return result, meta, audit


def _select(z, mask):
    out = {key: value[mask].clone() for key, value in z.items()}
    out["row_id"] = torch.arange(int(mask.sum()), dtype=torch.int64)
    return out


def _bilinear(values, x_grid, y_grid, points, height):
    """Bilinear value and its exact spatial gradient; y wraps, x does not.

    values: [ny,nx,fluid,component]; gradient ends in [component,derivative].
    """
    points = np.asarray(points, np.float64)
    nx, ny = len(x_grid), len(y_grid)
    dx, dy = x_grid[1] - x_grid[0], y_grid[1] - y_grid[0]
    if (not np.allclose(np.diff(x_grid), dx, rtol=0, atol=1e-13)
            or not np.allclose(np.diff(y_grid), dy, rtol=0, atol=1e-13)
            or abs(ny * dy - height) > 1e-12):
        raise ValueError("Reference interpolation requires a uniform, periodic-y grid")
    if np.any(points[..., 0] < x_grid[0]) or np.any(points[..., 0] > x_grid[-1]):
        raise ValueError("Reference query lies outside x walls")
    xx = (points[..., 0] - x_grid[0]) / dx
    yy = np.remainder(points[..., 1], height) / dy
    ix = np.minimum(np.floor(xx).astype(np.int64), nx - 2)
    iy = np.floor(yy).astype(np.int64) % ny
    tx, ty = (xx - ix)[..., None, None], (yy - np.floor(yy))[..., None, None]
    a, b = values[iy, ix], values[iy, ix + 1]
    c, d = values[(iy + 1) % ny, ix], values[(iy + 1) % ny, ix + 1]
    value = (1 - ty) * ((1 - tx) * a + tx * b) + ty * ((1 - tx) * c + tx * d)
    gx = ((1 - ty) * (b - a) + ty * (d - c)) / dx
    gy = ((1 - tx) * (c - a) + tx * (d - b)) / dy
    return value, np.stack((gx, gy), axis=-1)


class ReferenceFields:
    """Two-frame cache; no source file is edited, and hashes are checked."""
    def __init__(self, source_case, configuration):
        self.source = Path(source_case) / "particles"
        self.configuration = configuration
        self.manifest = _json(self.source / "frame_manifest.json")
        self.cache = OrderedDict()
        self.checked = set()

    def get(self, frame):
        if frame in self.cache:
            self.cache.move_to_end(frame)
            return self.cache[frame]
        name = f"field_truth/frame_{frame:06d}.npz"
        path = self.source / name
        if name not in self.manifest or sha256(path) != self.manifest[name]:
            raise ValueError("Reference field hash mismatch: " + name)
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: np.asarray(loaded[key], np.float64)
                      for key in ("x_m", "y_m", "vn", "vs", "time_s")}
        value = np.stack((arrays.pop("vn"), arrays.pop("vs")), axis=-2)
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite reference velocity field")
        expected = frame / float(self.configuration["camera_fps"])
        if abs(float(arrays["time_s"]) - expected) > 1e-12:
            raise ValueError("Reference field time mismatch")
        arrays["value"] = value
        self.cache[frame] = arrays
        self.checked.add(name)
        while len(self.cache) > 2:
            self.cache.popitem(last=False)
        return arrays

    def spatial(self, frame, points):
        f = self.get(frame)
        return _bilinear(f["value"], f["x_m"], f["y_m"], points,
                         self.configuration["config"]["height_m"])

    def jets(self, frame, query_absolute_s, points):
        first, second = self.get(frame), self.get(frame + 1)
        lo, hi = float(first["time_s"]), float(second["time_s"])
        query = np.asarray(query_absolute_s, np.float64)
        if np.any(query < lo - 1e-12) or np.any(query > hi + 1e-12):
            raise ValueError("Reference query is outside its native camera interval")
        v0, g0 = self.spatial(frame, points)
        v1, g1 = self.spatial(frame + 1, points)
        fraction = ((query - lo) / (hi - lo))[..., None, None]
        velocity = v0 + fraction * (v1 - v0)
        gradient = g0 + fraction[..., None] * (g1 - g0)
        temporal = (v1 - v0) / (hi - lo)
        acceleration = temporal + np.einsum("...fij,...fj->...fi", gradient, velocity)
        return velocity, gradient, acceleration


def prepare_case(source_case: Path, out: Path) -> dict:
    """Prepare isolated observation files, F/V reference jets and evaluation truth."""
    source_case, out = Path(source_case).resolve(), Path(out).resolve()
    if out == source_case or out.is_relative_to(source_case):
        raise ValueError("Prepared output must be outside the source case")
    configuration = _json(source_case / "configuration.json")
    metadata = _json(source_case / "particles/metadata.json")
    if metadata.get("smoke_only"):
        raise ValueError("Smoke-only input cannot train a scientific reconstruction")
    paths = {"configuration": source_case / "configuration.json",
             "observations": source_case / "particles/particle_trajectories.npz",
             "hidden_labels": source_case / "particles/particle_hidden_truth.npz",
             "frame_manifest": source_case / "particles/frame_manifest.json"}
    hashes = {key: sha256(path) for key, path in paths.items()}
    for key in ("observations", "hidden_labels", "frame_manifest"):
        if metadata["output_sha256"][paths[key].name] != hashes[key]:
            raise ValueError("Input hash mismatch: " + paths[key].name)
    with np.load(paths["observations"], allow_pickle=False) as loaded:
        observed = {key: loaded[key] for key in ("time_s", "particle_id", "xy_m", "segment_id")}
    with np.load(paths["hidden_labels"], allow_pickle=False) as loaded:
        hidden = {key: loaded[key] for key in ("time_s", "particle_id", "state", "attached_vortex")}
    z, meta, audit = build_observations(observed, hidden, configuration)
    meta.update(source_case=str(source_case), source_sha256=hashes,
                source_flow_status=metadata.get("source_flow_status"),
                steady_verified=metadata.get("steady_verified", False))
    masks = {"F": z["pid"] % 5 == 2, "I": z["pid"] % 5 >= 3,
             "V": z["pid"] % 5 == 1, "H": z["pid"] % 5 == 0}
    audit["groups"] = {}
    for group, mask in masks.items():
        audit["groups"][group] = dict(particles=int(z["pid"][mask].unique().numel()),
                                      intervals=int(mask.sum()),
                                      free=int((mask & (z["state"] == 0)).sum()),
                                      captured=int((mask & (z["state"] == 1)).sum()))
    probability = _select(z, masks["F"] | masks["V"])
    inversion = _select(z, masks["I"] | masks["V"])
    heldout = _select(z, masks["H"])
    out.mkdir(parents=True, exist_ok=True)
    for filename, data, groups in (("observations.pt", z, ["F", "I", "V", "H"]),
                                   ("probability_observations.pt", probability, ["F", "V"]),
                                   ("inversion_observations.pt", inversion, ["I", "V"]),
                                   ("heldout_observations.pt", heldout, ["H"])):
        _save(out / filename, dict(z=data, meta=dict(meta, permitted_groups=groups)))
    reader = ReferenceFields(source_case, configuration)
    n, m = probability["time"].shape
    v = np.empty((n, m, 2, 2), np.float64)
    g = np.empty((n, m, 2, 2, 2), np.float64)
    acc = np.empty_like(v)
    width, height = meta["physics_config"]["width_m"], meta["physics_config"]["height_m"]
    evaluation_x = np.linspace(0, width, 65)
    evaluation_y = np.arange(32) * height / 32
    yy, xx = np.meshgrid(evaluation_y, evaluation_x, indexing="ij")
    evaluation_points = np.stack((xx, yy), -1)
    reference = []
    for frame in range(1, len(observed["time_s"])):
        reference.append(reader.spatial(frame, evaluation_points)[0])
        ix = np.flatnonzero(probability["frame"].numpy() == frame)
        if len(ix):
            vv, gg, aa = reader.jets(frame, probability["time"][ix].numpy() * DT + meta["origin_s"],
                                     probability["x"][ix].numpy())
            v[ix], g[ix], acc[ix] = vv, gg, aa
    if not all(np.isfinite(value).all() for value in (v, g, acc)):
        raise ValueError("Nonfinite interpolated training jets")
    jet_contract = dict(version=VERSION, permitted_groups=["F", "V"],
                        probability_observation_sha256=sha256(out / "probability_observations.pt"),
                        fluid_axis=["normal", "superfluid"], vector_axis=["x", "y"],
                        gradient_axes="[fluid,velocity_component,spatial_derivative]",
                        interpolation="bilinear space, linear native-camera time; derivative of that interpolant",
                        acceleration="camera secant Eulerian derivative + G @ u",
                        limitation="camera-rate reference interpolants, not microstep ground truth; no snapshot force used")
    _save(out / "training_jets.pt", dict(row_id=probability["row_id"].clone(),
                                        v=torch.from_numpy(v), g=torch.from_numpy(g),
                                        acc=torch.from_numpy(acc), meta=jet_contract))
    reference = np.stack(reference)
    temp = out / "reference_evaluation.npz.tmp"
    with temp.open("wb") as stream:
        np.savez_compressed(stream, time_s=observed["time_s"][1:] - meta["origin_s"],
                            source_time_s=observed["time_s"][1:],
                            source_frame=np.arange(1, len(observed["time_s"])),
                            x_m=evaluation_x, y_m=evaluation_y,
                            vn=reference[..., 0, :].astype(np.float32),
                            vs=reference[..., 1, :].astype(np.float32))
    temp.replace(out / "reference_evaluation.npz")
    audit.update(meta=meta, output=str(out), input_modified=False,
                 field_frames_verified=len(reader.checked),
                 reference_evaluation=dict(frames=len(reference), shape_yx=[32, 65],
                                           use="evaluation only; never a field optimization target"),
                 training_jet_contract=jet_contract,
                 output_sha256={p.name: sha256(p) for p in out.iterdir()
                                if p.is_file() and p.suffix in (".pt", ".npz")})
    temporary = out / "data_audit.json.tmp"
    temporary.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(out / "data_audit.json")
    return audit


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_case", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(prepare_case(arguments.source_case, arguments.output), indent=2))
