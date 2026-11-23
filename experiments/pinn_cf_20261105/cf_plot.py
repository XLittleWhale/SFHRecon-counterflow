"""Plot frozen CF evaluation products, without reading or scoring observations."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cf_common import OUT, CASES, SEEDS, evaluation_gate, read, dump, sha, utc

REPRESENTATIVE = 301030
FLUIDS = ('normal', 'superfluid')
SCENARIOS = ((0, 'background'), (200, 'primary'), (200, 'terminal'), (600, 'primary'), (600, 'terminal'))
SHORT = ('Cold', '200 V', '200 last', '600 V', '600 last')
COLORS = ('#777777', '#0072B2', '#56B4E9', '#D55E00', '#E69F00')


def guarded_inputs():
    protocol, gate = evaluation_gate()
    summary = read(OUT/'CF_EVALUATION_SUMMARY.json')
    manifest = read(OUT/'FIELD_MANIFEST.json')
    expected = {(case, seed, budget, mode) for case in CASES for seed in SEEDS for budget, mode in SCENARIOS}
    actual = {(r['case'], r['seed'], r['budget'], r['mode']) for r in summary['runs']}
    if not summary['complete'] or len(summary['runs']) != 30 or actual != expected:
        raise RuntimeError('Need all 30 globally frozen CF evaluation records')
    if summary['protocol_fingerprint'] != protocol['fingerprint'] or manifest['binding'] != summary['binding']:
        raise RuntimeError('Evaluation/field bindings disagree')
    if manifest['evaluation_sha256'] != sha(OUT/'CF_EVALUATION_SUMMARY.json'):
        raise RuntimeError('Evaluation report changed since field manifest')
    if len(manifest['artifacts']) != 10:
        raise RuntimeError('Expected 10 fixed representative field series')
    for item in manifest['artifacts']:
        if sha(item['path']) != item['sha256']:
            raise RuntimeError('Field export changed: '+item['path'])
        if item['seed'] != REPRESENTATIVE or item['frames'] != 101:
            raise RuntimeError('Unexpected representative or time window')
    return protocol, summary, manifest


def field_files(manifest, case):
    return {(item['budget'], item['mode']): Path(item['path'])
            for item in manifest['artifacts'] if item['case'] == case}


def save(fig, name, description, artifacts):
    fig.suptitle(description, fontsize=12)
    for extension in ('png', 'svg'):
        path = OUT/'figures'/f'{name}.{extension}'
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        artifacts.append(dict(name=name, format=extension, path=str(path.resolve()),
                              sha256=sha(path), description=description))
    plt.close(fig)


def maps(manifest, artifacts):
    frames = (0, 50, 100)
    scenarios = ((0, 'background'), (200, 'primary'), (600, 'primary'))
    labels = ('Fixed cold background', 'PINN 200 V-selected', 'PINN 600 V-selected')
    for case in CASES:
        paths = field_files(manifest, case)
        arrays = {key: dict(np.load(paths[key], allow_pickle=False)) for key in scenarios}
        for fluid, fluid_name in enumerate(FLUIDS):
            for kind, unit, cmap in (('speed', 'Speed [mm/s]', 'viridis'),
                                     ('vorticity', 'Vorticity [1/s]', 'RdBu_r'),
                                     ('background_difference', 'Departure from cold [mm/s]', 'magma')):
                values = {}
                for key, data in arrays.items():
                    if kind == 'speed':
                        values[key] = np.linalg.norm(data['velocity_m_s'][:, fluid], axis=-1)*1000
                    elif kind == 'vorticity':
                        values[key] = data['vorticity_s_inv'][:, fluid]
                    else:
                        values[key] = np.linalg.norm(data['departure_from_cold_m_s'][:, fluid], axis=-1)*1000
                hi = max(float(np.abs(v[list(frames)]).max()) for v in values.values())
                hi = max(hi, 1e-12)
                lo = -hi if kind == 'vorticity' else 0.
                fig, axes = plt.subplots(3, 3, figsize=(12, 8), layout='constrained')
                for i, (key, label) in enumerate(zip(scenarios, labels)):
                    data = arrays[key]
                    # Data are cell-centred.  Explicit grid edges, not min/max
                    # centre coordinates, place the displayed walls at 0/16 mm.
                    extent = tuple(v*1000 for v in (data['x_edges_m'][0], data['x_edges_m'][-1],
                                                    data['y_edges_m'][0], data['y_edges_m'][-1]))
                    for j, frame in enumerate(frames):
                        ax = axes[i, j]
                        im = ax.imshow(values[key][frame], origin='lower', extent=extent,
                                       vmin=lo, vmax=hi, cmap=cmap, aspect='equal')
                        ax.set_title(f'{label}; t={frame} ms', fontsize=9)
                        ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
                fig.colorbar(im, ax=axes, shrink=.85, label=unit)
                fig.supxlabel('No measured field reference. All rows share the color scale; relative time in the fixed 100 ms window.', fontsize=8)
                save(fig, f'{case}_{fluid_name}_{kind}',
                     f'{case}; {fluid_name}; seed {REPRESENTATIVE}', artifacts)


def profiles(manifest, artifacts):
    for case in CASES:
        arrays = {key: dict(np.load(path, allow_pickle=False)) for key, path in field_files(manifest, case).items()}
        for fluid, fluid_name in enumerate(FLUIDS):
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout='constrained')
            for frame, ax in zip((0, 50, 100), axes):
                for (key, label, color) in zip(SCENARIOS, SHORT, COLORS):
                    data = arrays[key]
                    profile = data['velocity_m_s'][frame, fluid, :, :, 1].mean(0)*1000
                    ax.plot(data['x_m']*1000, profile, label=label, color=color,
                            linestyle='--' if key[1] == 'terminal' else '-')
                ax.set(xlabel='x [mm]', ylabel='Mean over y of axial velocity [mm/s]', title=f't={frame} ms', xlim=(0, 16))
                ax.grid(alpha=.2)
            axes[-1].legend(fontsize=8)
            fig.supxlabel('Profiles are reconstructed fields, not measured reference profiles. Dashed: fixed terminal snapshots.', fontsize=8)
            save(fig, f'{case}_{fluid_name}_axial_profiles', f'{case}; {fluid_name}; seed {REPRESENTATIVE}', artifacts)


def field_time(summary, artifacts):
    for case in CASES:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        for fluid, ax in enumerate(axes):
            for (budget, mode), label, color in zip(SCENARIOS, SHORT, COLORS):
                if budget == 0:
                    continue
                rows = [r for r in summary['runs'] if (r['case'], r['budget'], r['mode']) == (case, budget, mode)]
                values = np.array([r['field']['departure_from_cold_time_rms_mm_s'] for r in rows])[:, :, fluid]
                ax.plot(np.arange(101), values.mean(0), color=color, label=label,
                        linestyle='--' if mode == 'terminal' else '-')
                ax.fill_between(np.arange(101), values.min(0), values.max(0), color=color, alpha=.10)
            ax.set(xlabel='Relative time [ms]', ylabel='RMS departure from cold [mm/s]', title=FLUIDS[fluid])
            ax.grid(alpha=.2)
        axes[-1].legend(fontsize=8)
        fig.supxlabel('Shading is the range across three numerical seeds, not a physical confidence interval or field-error band.', fontsize=8)
        save(fig, f'{case}_time_departure', case, artifacts)


def optimization(summary, artifacts):
    for case in CASES:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
        for history in summary['histories']:
            if history['case'] != case:
                continue
            trace = history['rows']
            fit = [r for r in trace if 'objective' in r]
            val = [r for r in trace if 'validation' in r]
            label = str(history['seed'])
            axes[0].plot([r['step'] for r in fit], [r['inversion']['total'] for r in fit], label=label)
            axes[1].plot([r['step'] for r in val], [r['validation']['total'] for r in val], label=label)
            axes[2].plot([r['step'] for r in fit], [r['pde_loss'] for r in fit], label=label)
        for ax, title, ylabel in zip(axes, ('Fit likelihood', 'Selection likelihood', 'Weighted PDE residual'),
                                     ('Equal-state/PID NLL/ms', 'Equal-state/PID NLL/ms', 'PDE loss')):
            ax.set(xlabel='Adam updates', ylabel=ylabel, title=title)
            ax.axvline(200, color='black', linestyle=':', alpha=.4)
            ax.grid(alpha=.2)
        axes[-1].legend(title='Numerical seed', fontsize=8)
        fig.supxlabel('Fit/PDE are measured before the update; selection is scored at saved post-update checkpoints. Test scores do not select checkpoints.', fontsize=8)
        save(fig, f'{case}_optimization', case, artifacts)


def calibration(summary, artifacts):
    xx = np.arange(len(SCENARIOS))
    for case in CASES:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
        for i, kind in enumerate(('free', 'cap')):
            for j, (suffix, ylabel) in enumerate((('nll_per_ms', 'NLL/ms'),
                ('coverage90', 'Coverage of nominal 90% region'), ('spread_um', 'Width [micrometres]'))):
                ax = axes[i, j]
                mean, low, high = [], [], []
                for budget, mode in SCENARIOS:
                    stats = summary['summary'][case][f'{budget}_{mode}'][f'H_{kind}_{suffix}']
                    mean.append(stats['mean']); low.append(stats['minimum']); high.append(stats['maximum'])
                mean, low, high = map(np.asarray, (mean, low, high))
                ax.errorbar(xx, mean, yerr=[mean-low, high-mean], fmt='o-', capsize=4, color='#0072B2')
                if suffix == 'coverage90':
                    ax.axhline(.9, color='#D55E00', linestyle='--'); ax.set_ylim(0., 1.02)
                ax.set_xticks(xx, SHORT, rotation=25)
                ax.set(ylabel=ylabel, title=kind)
                ax.grid(alpha=.2)
        fig.supxlabel('Held-out native endpoints, conditional on past observed positions. Bars show numerical-seed range. Higher coverage is not always better.', fontsize=8)
        save(fig, f'{case}_heldout_calibration', case, artifacts)


def pde_sensitivity(summary, artifacts):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    xx = np.arange(len(SCENARIOS))
    for j, case in enumerate(CASES):
        for f, fluid in enumerate(('normal', 'super')):
            ax = axes[f, j]
            for source, offset, color, label in (('trained_simpson', -.1, '#0072B2', '201 trained nodes'),
                                                 ('independent_quarter', .1, '#D55E00', '200 independent quarter nodes')):
                values = []
                for budget, mode in SCENARIOS:
                    rr = [r for r in summary['runs'] if (r['case'], r['budget'], r['mode']) == (case, budget, mode)]
                    values.append([r['pde'][source][f'{fluid}_B_energy'] for r in rr])
                values = np.maximum(np.asarray(values), 1e-30)
                mean = values.mean(1)
                ax.errorbar(xx+offset, mean, yerr=[mean-values.min(1), values.max(1)-mean],
                            fmt='o', capsize=3, label=label, color=color)
            ax.set_yscale('log')
            ax.set_xticks(xx, SHORT, rotation=25)
            ax.set(ylabel='Frozen-B quadratic energy', title=f'{case}; {fluid}')
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.supxlabel('Diagnostic points were never used for optimization or checkpoint selection. Low PDE residual is not evidence of field accuracy.', fontsize=8)
    save(fig, 'pde_noncollocation_sensitivity', 'Trained and independent temporal residual samples', artifacts)


def costs(summary, artifacts):
    costs = summary['cost']['unique_field_runs']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, base, ylabel in ((axes[0], 'cpu_seconds', 'CPU hours (summed)'),
                             (axes[1], 'wall_seconds', 'Worker wall hours (summed)')):
        first, additional = [], []
        for case in CASES:
            rows = [r for r in costs if r['case'] == case]
            first.append(sum(r[f'budget200_{base}'] for r in rows)/3600)
            additional.append(sum(r[f'additional201_600_{base}'] for r in rows)/3600)
        ax.bar(np.arange(2), first, label='0-200 prefix', color='#0072B2')
        ax.bar(np.arange(2), additional, bottom=first, label='201-600 additional', color='#D55E00')
        ax.set_xticks(np.arange(2), ['CF113', 'CF217'])
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    fig.supxlabel('Six unique inversion trajectories; primary/terminal snapshots add no training runs. Worker wall times overlap in parallel. New probability training: zero.', fontsize=8)
    save(fig, 'compute_cost', 'CF inversion cost', artifacts)


def run():
    protocol, summary, manifest = guarded_inputs()
    artifacts = []
    maps(manifest, artifacts)
    profiles(manifest, artifacts)
    field_time(summary, artifacts)
    optimization(summary, artifacts)
    calibration(summary, artifacts)
    pde_sensitivity(summary, artifacts)
    costs(summary, artifacts)
    dump(OUT/'PLOT_MANIFEST.json', dict(complete=True, created_utc=utc(),
        protocol_fingerprint=protocol['fingerprint'], evaluation_sha256=sha(OUT/'CF_EVALUATION_SUMMARY.json'),
        field_manifest_sha256=sha(OUT/'FIELD_MANIFEST.json'), script_sha256=sha(Path(__file__)),
        representative_seed=REPRESENTATIVE, figure_groups=len(artifacts)//2, artifacts=artifacts,
        observation_scoring_or_field_selection=False))
    print('CF_PLOTS_COMPLETE', len(artifacts)//2, flush=True)


if __name__ == '__main__':
    run()
