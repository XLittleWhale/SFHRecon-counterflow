"""Two-stage CUDA workflow for the 1 cm periodic box.

Stage 1 ends only after a documented rolling-statistics stationarity test.
Stage 2 starts from its saved flow state, then inserts particles.  CPU mode
is intentionally reserved for the tiny smoke test unless explicitly enabled.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from .config import Config
from .flow import TorchFlow
from .tethered import VortexInertialTetheredParticles
from .vtk import write_field, write_points, write_series
from ..paths import RUN
HERE = RUN / 'simulation'

def _device(name: str, smoke: bool, threads: int) -> str:
    if threads:
        torch.set_num_threads(threads)
    if name == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA device unavailable: run this 1 cm case on a PyTorch/CUDA server')
    if name == 'cpu' and (not smoke):
        raise RuntimeError('1 cm production run on CPU is disabled (hours-long); use --device cuda')
    return name

def _period_steps(seconds: float, dt: float) -> int:
    return max(1, round(seconds / dt))

def _json(path: Path, content: dict) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(content, indent=2), encoding='utf-8')
    temporary.replace(path)

def _flow_snapshot(flow: TorchFlow, out: Path, index: int, time: float) -> None:
    f = flow.snapshot_fields()
    write_field(out / f'field_{index:04d}.vtkhdf', f, flow.c.box_m)
    write_points(out / f'vortices_{index:04d}.vtkhdf', flow.pos.cpu().numpy(), {'id': np.arange(len(flow.sign), dtype=np.int32), 'sign': flow.sign.cpu().numpy().astype(np.int8)})

def _window_drift(history: list[dict], end: float, window: float) -> dict | None:
    left = [r for r in history if end - 2 * window <= r['time_s'] < end - window]
    right = [r for r in history if end - window <= r['time_s'] <= end]
    if len(left) < 3 or len(right) < 3:
        return None
    keys = ('vn_fluct_rms_m_s', 'vs_fluct_rms_m_s', 'Fns_rms_N_m3')
    result = {}
    for key in keys:
        a = float(np.mean([r[key] for r in left]))
        b = float(np.mean([r[key] for r in right]))
        result[key] = abs(b - a) / max(abs(a), abs(b), 1e-12)

    def pair_rate(records: list[dict]) -> float:
        return (records[-1]['reconnection_pairs_total'] - records[0]['reconnection_pairs_total']) / max(records[-1]['time_s'] - records[0]['time_s'], 1e-12)
    rate_a, rate_b = (pair_rate(left), pair_rate(right))
    result['reconnection_rate'] = abs(rate_b - rate_a) / max(rate_a, rate_b, 1e-12)
    result['max_relative_drift'] = max(result.values())
    return result

def steady_main() -> None:
    p = argparse.ArgumentParser(description='Stage 1: evolve a 1 cm box until rolling diagnostics stabilize')
    p.add_argument('--output', type=Path, default=HERE / 'outputs' / 'large_steady')
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--precision', choices=('float32', 'float64'), default='float32')
    p.add_argument('--threads', type=int, default=0)
    p.add_argument('--grid', type=int, default=1024)
    p.add_argument('--core-cells', type=float, default=1.5)
    p.add_argument('--dt-us', type=float, default=20.0)
    p.add_argument('--max-time-s', type=float, default=10.0)
    p.add_argument('--min-time-s', type=float, default=1.0)
    p.add_argument('--window-s', type=float, default=0.2)
    p.add_argument('--diagnostic-ms', type=float, default=5.0)
    p.add_argument('--save-ms', type=float, default=100.0)
    p.add_argument('--checkpoint-ms', type=float, default=100.0)
    p.add_argument('--tolerance', type=float, default=0.05)
    p.add_argument('--consecutive', type=int, default=3)
    p.add_argument('--seed', type=int, default=2718)
    p.add_argument('--temperature-K', type=float, default=2.0)
    p.add_argument('--vn-mean-mm-s', type=float, default=10.0)
    p.add_argument('--resume', type=Path, default=None)
    p.add_argument('--smoke', action='store_true', help='tiny CPU/CUDA test only; no steady-state claim')
    args = p.parse_args()
    device = _device(args.device, args.smoke, args.threads)
    if args.smoke:
        c = Config(box_m=0.001, grid=64, dt_s=2e-05, seed=args.seed, temperature_K=args.temperature_K, vn_mean_m_s=args.vn_mean_mm_s * 0.001, initial_trapped_particles=0)
        args.max_time_s = 0.0002
        args.min_time_s = 0.0001
        args.window_s = 0.0001
        args.diagnostic_ms = 0.02
        args.save_ms = 0.1
        args.checkpoint_ms = 0.1
    else:
        c = Config(box_m=0.01, grid=args.grid, dt_s=args.dt_us * 1e-06, seed=args.seed, temperature_K=args.temperature_K, vn_mean_m_s=args.vn_mean_mm_s * 0.001, n_particles=512, initial_trapped_particles=128, coherent_fraction=0.02, core_cells=args.core_cells)
    if args.max_time_s <= 0 or args.window_s <= 0 or args.consecutive < 1:
        raise ValueError('invalid steady-state duration/window/consecutive')
    flow = TorchFlow.load_checkpoint(args.resume, device, args.precision) if args.resume else TorchFlow(c, device, args.precision)
    if args.resume and flow.c.box_m != c.box_m:
        raise ValueError('resume checkpoint box size differs from selected workflow')
    if args.resume and (flow.c.temperature_K != c.temperature_K or not np.isclose(flow.c.vn_mean_m_s, c.vn_mean_m_s, rtol=0, atol=1e-12)):
        raise ValueError('resume checkpoint temperature or mean normal speed differs')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(out.glob('field_*.vtkhdf')):
        raise FileExistsError('output already contains field snapshots; use --resume or a new output directory')
    history_path = out / 'steady_history.json'
    if args.resume and history_path.exists():
        history = json.loads(history_path.read_text(encoding='utf-8'))
    else:
        history = []
    times_path = out / 'field.vtkhdf.series'
    if args.resume and times_path.exists():
        times = [float(x['time']) for x in json.loads(times_path.read_text())['files']]
    else:
        times = []
    if times and times[-1] > flow.time + 0.25 * flow.c.dt_s:
        raise RuntimeError('VTK output is newer than resume checkpoint; resume into a fresh output directory')
    diag_every = _period_steps(args.diagnostic_ms * 0.001, flow.c.dt_s)
    save_every = _period_steps(args.save_ms * 0.001, flow.c.dt_s)
    checkpoint_every = _period_steps(args.checkpoint_ms * 0.001, flow.c.dt_s)
    next_check = max(args.min_time_s, flow.time) + 2 * args.window_s
    good_windows = 0
    detected = False
    last_drift = None
    while flow.time < args.max_time_s - 0.5 * flow.c.dt_s:
        flow.step(flow.c.dt_s)
        flow.reconnect()
        if flow.step_index % diag_every == 0:
            record = flow.diagnostics()
            history.append(record)
            if flow.time >= next_check and (not args.smoke):
                last_drift = _window_drift(history, flow.time, args.window_s)
                good_windows = good_windows + 1 if last_drift is not None and last_drift['max_relative_drift'] < args.tolerance else 0
                print(f'steady check t={flow.time:.3f}s drift={last_drift} pass={good_windows}/{args.consecutive}', flush=True)
                next_check += args.window_s / 2
                if good_windows >= args.consecutive:
                    detected = True
        if flow.step_index % save_every == 0:
            _flow_snapshot(flow, out, len(times), flow.time)
            times.append(flow.time)
            write_series(out, 'field', times)
            write_series(out, 'vortices', times)
            print(f'snapshot t={flow.time:.3f}s pairs={flow.reconnection_pairs_total}', flush=True)
        if flow.step_index % checkpoint_every == 0:
            flow.save_checkpoint(out / 'checkpoint_latest.npz')
            _json(history_path, history)
        if detected:
            break
    if not times or abs(times[-1] - flow.time) > 0.25 * flow.c.dt_s:
        _flow_snapshot(flow, out, len(times), flow.time)
        times.append(flow.time)
        write_series(out, 'field', times)
        write_series(out, 'vortices', times)
    flow.save_checkpoint(out / 'checkpoint_latest.npz')
    _json(history_path, history)
    if detected:
        flow.save_checkpoint(out / 'steady_state.npz')
    status = {'status': 'steady' if detected else 'incomplete', 'smoke_only': bool(args.smoke), 'time_s': flow.time, 'step_index': flow.step_index, 'reconnection_pairs_total': flow.reconnection_pairs_total, 'last_window_drift': last_drift, 'consecutive_passes': good_windows, 'threshold': args.tolerance, 'required_consecutive': args.consecutive, 'min_time_s': args.min_time_s, 'window_s': args.window_s, 'config': flow.c.metadata(), 'precision': args.precision, 'device': device, 'steady_checkpoint': 'steady_state.npz' if detected else None}
    _json(out / 'steady_status.json', status)
    print(json.dumps({k: status[k] for k in ('status', 'time_s', 'step_index', 'reconnection_pairs_total')}), flush=True)

def particles_main() -> None:
    mode = 'vortex_on_particle_tethered_inertial'
    p = argparse.ArgumentParser(description='Stage 2: seed particles into an accepted steady checkpoint')
    p.add_argument('--steady-checkpoint', type=Path, default=HERE / 'outputs' / 'large_steady' / 'steady_state.npz')
    default_output = 'particles'
    p.add_argument('--output', type=Path, default=HERE / 'outputs' / default_output)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--precision', choices=('float32', 'float64'), default='float32')
    p.add_argument('--threads', type=int, default=0)
    p.add_argument('--duration-ms', type=float, default=100.0)
    p.add_argument('--save-ms', type=float, default=1.0)
    p.add_argument('--particles', type=int, default=512)
    p.add_argument('--initial-trapped', type=int, default=128)
    p.add_argument('--effective-segment-um', type=float, default=None, help='out-of-plane segment length in micrometers; default particle diameter')
    p.add_argument('--anchor-separation-um', type=float, default=None, help='out-of-plane anchor offset h in um; default midpoint of (a_p, 2.5*a_p); require a_p < h < 2.5*a_p')
    p.add_argument('--tension-scale', type=float, default=1.0, help='scale the logarithmic line tension; zero is a baseline ablation')
    p.add_argument('--break-extension-um', type=float, default=None, help='optional anchor/particle separation for tension-induced release')
    p.add_argument('--anchor-inertia-radius-nm', type=float, default=0.1, help='xi in m_anchor=mass_factor*rho_s*xi^2*(segment/2); default vortex core radius')
    p.add_argument('--anchor-mass-factor', type=float, default=math.pi, help='dimensionless mass prefactor; pi is the cylindrical-core estimate')
    p.add_argument('--smoke', action='store_true', help='allow a tiny incomplete checkpoint for test only')
    args = p.parse_args()
    device = _device(args.device, args.smoke, args.threads)
    source = args.steady_checkpoint.resolve()
    if not source.exists():
        raise FileNotFoundError(f'steady checkpoint missing: {source}')
    if args.output.resolve() == source.parent:
        raise ValueError('particle output must not overwrite the steady source directory')
    status_path = source.parent / 'steady_status.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    if not args.smoke and (status['status'] != 'steady' or status['smoke_only']):
        raise RuntimeError('source has not passed the steady-state acceptance test')
    flow = TorchFlow.load_checkpoint(source, device, args.precision)
    if not args.smoke and (flow.c.box_m != 0.01 or len(flow.sign) != flow.c.n_vortices):
        raise ValueError('expected a self-consistent 1 cm steady checkpoint')
    flow.c.n_particles = args.particles
    flow.c.initial_trapped_particles = args.initial_trapped
    flow.c.validate()
    segment = 2 * flow.c.particle_radius_m if args.effective_segment_um is None else args.effective_segment_um * 1e-06
    tether_class = VortexInertialTetheredParticles
    particles = tether_class(flow, segment, anchor_separation_m=None if args.anchor_separation_um is None else args.anchor_separation_um * 1e-06, tension_scale=args.tension_scale, break_extension_m=None if args.break_extension_um is None else args.break_extension_um * 1e-06, **{'inertia_radius_m': args.anchor_inertia_radius_nm * 1e-09, 'mass_factor': args.anchor_mass_factor})
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob('field_*.vtkhdf')):
        raise FileExistsError('particle output already contains snapshots; select a new output directory')
    start_time = flow.time
    n_steps = round(args.duration_ms * 0.001 / flow.c.dt_s)
    save_every = _period_steps(args.save_ms * 0.001, flow.c.dt_s)
    times = []
    xy, xy_unwrapped, state, attached = ([], [], [], [])
    magnus, friction = ([], [])
    tension, anchors = ([], [])
    anchor_velocities, anchor_accelerations, anchor_reactions = ([], [], [])
    tension_z, tangents = ([], [])
    diagnostics = []

    def save() -> None:
        i = len(times)
        t = flow.time - start_time
        times.append(t)
        _flow_snapshot(flow, out, i, t)
        pos = particles.x.cpu().numpy()
        point_data = {'id': np.arange(len(pos), dtype=np.int32), 'state': particles.state, 'attached_vortex': particles.attached_vortex}
        point_data['force_tension_N'] = np.linalg.norm(particles.last_tension.cpu().numpy(), axis=1)
        write_points(out / f'particles_{i:04d}.vtkhdf', pos, point_data)
        xy.append(pos.copy())
        xy_unwrapped.append(particles.x_unwrapped.cpu().numpy().copy())
        state.append(particles.state.copy())
        attached.append(particles.attached_vortex.copy())
        magnus.append(particles.last_magnus.cpu().numpy().copy())
        friction.append(particles.last_friction.cpu().numpy().copy())
        tension.append(particles.last_tension.cpu().numpy().copy())
        anchor_positions = particles.anchors_unwrapped.cpu().numpy().copy()
        anchor_positions[particles.state != particles.TRAPPED] = np.nan
        anchors.append(anchor_positions)
        for source, target in ((particles.anchor_velocity, anchor_velocities), (particles.last_anchor_acceleration, anchor_accelerations), (particles.last_anchor_reaction, anchor_reactions)):
            values = source.cpu().numpy().copy()
            values[particles.state != particles.TRAPPED] = np.nan
            target.append(values)
        bound = particles.state == particles.TRAPPED
        if np.any(bound):
            delta = anchor_positions[bound] - particles.x_unwrapped.cpu().numpy()[bound, None, :]
            delta -= flow.c.box_m * np.round(delta / flow.c.box_m)
            length = np.sqrt(particles.anchor_separation_m ** 2 + np.sum(delta ** 2, axis=-1))
            max_anchor_length = float(np.max(length))
            tolerance = max(1e-15, particles.max_anchor_length_m * 1e-12)
            if max_anchor_length > particles.max_anchor_length_m + tolerance:
                raise AssertionError('virtual anchor exceeded the 2.5*a_p cap')
        else:
            max_anchor_length = None
        diagnostics.append({'time_s': t, 'trapped': int((particles.state == 1).sum()), 'captures_total': particles.captures_total, 'releases_total': particles.releases_total, **{'tension_releases_total': particles.tension_releases_total}, **{'anchor_cap_hits_total': particles.cap_hits_total}, **{'max_anchor_length_m': max_anchor_length}, **{}, 'reconnection_pairs_total': flow.reconnection_pairs_total})
        print(f"particle snapshot t={t:.4f}s trapped={diagnostics[-1]['trapped']} capture={particles.captures_total} release={particles.releases_total}", flush=True)
    save()
    for local_step in range(1, n_steps + 1):
        old = flow.fields()
        flow.step(flow.c.dt_s)
        new = flow.fields()
        particles.step_bound(old, new, flow, flow.c.dt_s)
        new = flow.fields()
        particles.step_free(old, new, flow, flow.c.dt_s)
        event = flow.reconnect()
        particles.reconcile_and_capture(flow, event)
        if local_step % save_every == 0 or local_step == n_steps:
            save()
    for prefix in ('field', 'vortices', 'particles'):
        write_series(out, prefix, times)
    np.savez_compressed(out / 'particle_trajectories.npz', time_s=np.asarray(times), particle_id=np.arange(args.particles), xy_m=np.asarray(xy), xy_unwrapped_m=np.asarray(xy_unwrapped))
    hidden = {'time_s': np.asarray(times), 'particle_id': np.arange(args.particles), 'state': np.asarray(state), 'attached_vortex': np.asarray(attached)}
    hidden.update(force_magnus_xy_N=np.asarray(magnus), force_friction_xy_N=np.asarray(friction))
    hidden.update(force_tension_xy_N=np.asarray(tension), anchor_xy_unwrapped_m=np.asarray(anchors))
    hidden.update(anchor_velocity_xy_m_s=np.asarray(anchor_velocities), anchor_step_acceleration_xy_m_s2=np.asarray(anchor_accelerations), anchor_tension_reaction_xy_N=np.asarray(anchor_reactions))
    np.savez_compressed(out / 'particle_hidden_truth.npz', **hidden)
    metadata = {'source_steady_checkpoint': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'source_steady_status': status['status'], 'smoke_only': bool(args.smoke), 'config': flow.c.metadata(), 'precision': args.precision, 'capture_mode': mode, 'snapshots': diagnostics}
    metadata['effective_vortex_segment_m'] = segment
    metadata['bound_integrator'] = 'particle trapezoidal velocity; pinned-vortex RK4 predictor-corrector'
    metadata['tension_closure'] = {'model': 'two inertial virtual out-of-plane anchors with opposite tension reaction', 'anchor_separation_m': particles.anchor_separation_m, 'intervortex_spacing_m': particles.intervortex_spacing_m, 'max_anchor_length_particle_radius_multiple': particles.MAX_ANCHOR_RADIUS_MULTIPLE, 'max_anchor_length_m': particles.max_anchor_length_m, 'max_anchor_in_plane_m': particles.max_anchor_in_plane_m, 'vortex_core_m': particles.vortex_core_m, 'tension_outer_m': particles.tension_outer_m, 'base_line_tension_N': particles.base_tension_N, 'tension_scale': particles.tension_scale, 'effective_line_tension_N': particles.tension_N, 'break_extension_m': particles.break_extension_m, 'tension_releases_total': particles.tension_releases_total, 'coupling': 'equal-and-opposite tension on particle and anchor substeps', 'limitation': 'virtual anchors, frozen-flow force substep and hard geometry cap are reduced closures; cap impulses are not fed back to particle/flow'}
    metadata['tension_closure'].update({'inertia_radius_m': particles.inertia_radius_m, 'mass_factor': particles.mass_factor, 'anchor_segment_m': particles.anchor_segment_m, 'anchor_mass_kg': particles.anchor_mass_kg, 'anchor_drag_Ns_m': particles.anchor_drag_Ns_m, 'anchor_gyro_Ns_m': particles.anchor_gyro_Ns_m, 'anchor_relaxation_s': particles.anchor_relaxation_s, 'flow_dt_over_anchor_relaxation': flow.c.dt_s / particles.anchor_relaxation_s, 'anchor_dissipative_mobility_m_Ns': particles.anchor_drag_Ns_m / (particles.anchor_drag_Ns_m ** 2 + particles.anchor_gyro_Ns_m ** 2), 'anchor_transverse_mobility_m_Ns': particles.anchor_gyro_Ns_m / (particles.anchor_drag_Ns_m ** 2 + particles.anchor_gyro_Ns_m ** 2), 'anchor_cap_hits_total': particles.cap_hits_total, 'anchor_integrator': 'exact frozen-force linear velocity and position; one midpoint force correction'})
    _json(out / 'metadata.json', metadata)
    flow.save_checkpoint(out / 'final_flow_state.npz')
