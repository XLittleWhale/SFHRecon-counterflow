"""Static, gate-bound figures for the complete four-arm transition comparison.

Primary selections and fixed step-200 terminals remain separate. No scoring,
training, checkpoint selection, or change to the frozen scientific code occurs.
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
OUT = HERE.parents[2] / "outputs/SharedTransition_20261104"
CASES = (("vn0_0015m_s", .0015), ("vn0_004m_s", .004))
ARMS = ("legacy", "cap_retrain", "independent", "shared_gru")
SEEDS = (301030, 301031, 301032)
TRAIN_SEEDS = (411040, 411041, 411042)
MODES = ("primary", "terminal")
REPRESENTATIVE_SEED = 301030
REPRESENTATIVE_TRAIN_SEED = 411040
LABELS = dict(legacy="Legacy net", cap_retrain="Cap retrain", independent="Independent", shared_gru="Shared GRU")
COLORS = dict(legacy="#64748b", cap_retrain="#d97706", independent="#0e7490", shared_gru="#a21caf")
PARAMETERS = dict(legacy=8270, cap_retrain=8270, independent=16243, shared_gru=12595)
MODE_LABELS = dict(primary="Validation-selected primary result", terminal="Fixed step-200 exploratory terminal diagnostic")
SCOPE = "H: same-window 1 ms prediction given observed history; not future free-running trajectories."
PARAMETER_NOTE = "Parameters: legacy/cap 8,270; independent 16,243; shared 12,595 (3,648 shared GRU)."


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def verify_complete(out):
    """No checkpoint, target reference, or new target-response scoring here."""
    path = out / "TRANSITION_EVALUATION_SUMMARY.json"
    if not path.is_file():
        raise RuntimeError("Complete transition evaluation is required before plots or target reference access")
    result = read(path)
    if (result.get("complete") is not True or result.get("selected_or_trained_during_evaluation") is not False
            or result.get("terminal_preplanned") is not True):
        raise RuntimeError("Evaluation incomplete or analysis/selection scope changed")
    manifest_path = out / "evaluation_manifest.json"
    if not manifest_path.is_file() or sha(manifest_path) != result.get("manifest_sha256"):
        raise RuntimeError("Completed evaluation manifest changed")
    expected = {(mode, case, seed, arm) for mode in MODES for case, _ in CASES for seed in SEEDS for arm in ARMS}
    rows = result.get("runs", [])
    if len(rows) != 48 or {(r["mode"], r["case"], int(r["seed"]), r["arm"]) for r in rows} != expected:
        raise RuntimeError("Both complete 24-run modes are required; selective plotting is forbidden")
    if any(r["iteration"] != 200 for r in rows if r["mode"] == "terminal"):
        raise RuntimeError("Terminal summary must refer to iteration 200")
    return result, read(manifest_path)


def bind_all(out):
    result, manifest = verify_complete(out)
    V = module("transition_plot_evaluation", HERE / "evaluate_transition.py")
    training_gate, field_gate = V.cheap_gate(out)
    E = module("transition_plot_experiment", HERE / "experiment.py")
    T = module("transition_plot_training", HERE / "training.py")
    J = module("transition_plot_helpers", V.HELPER)
    if out.resolve() != Path(E.OUT).resolve():
        raise RuntimeError("Plot destination must match the verified experiment output")
    E.M.R.setup("cpu")
    protocol, tp, records = V.bind_gates(E, T, out, training_gate, field_gate)
    legacy, old_rows, old_artifacts = V.bind_legacy(E, protocol)
    # Existing manifest must compare equal; no new manifest is created here.
    if not (out / "evaluation_manifest.json").is_file():
        raise RuntimeError("A completed existing evaluation manifest is required")
    rebound, binding = V.bind_manifest(out, protocol, tp, training_gate, records, legacy, old_artifacts)
    if (rebound != manifest or binding != result["manifest_sha256"]
            or result["protocol_fingerprint"] != protocol["fingerprint"]):
        raise RuntimeError("Summary is not bound to the current gates and dependencies")
    source_records = {(r["case"], int(r["seed"]), r["arm"]): r for r in records + legacy}
    training_records = {(r["arm"], int(r["seed"])): r for r in training_gate["runs"]}
    for row in result["runs"]:
        record = source_records[(row["case"], int(row["seed"]), row["arm"])]
        path_key, hash_key = ("path", "sha256") if row["mode"] == "primary" else ("last_path", "last_sha256")
        if (Path(row["checkpoint_path"]).resolve() != Path(record[path_key]).resolve()
                or row["checkpoint_sha256"] != record[hash_key]
                or row["primary_selected_iteration"] != record["selected_iteration"]):
            raise RuntimeError("Plot row checkpoint/selection provenance mismatch")
        ref = result["reference_sources"][row["case"]]
        if (Path(row["reference_path"]).resolve() != Path(ref["path"]).resolve()
                or row["reference_sha256"] != ref["sha256"]):
            raise RuntimeError("Row and summary reference bindings disagree")
        if row["arm"] != "legacy":
            ts = TRAIN_SEEDS[SEEDS.index(int(row["seed"]))]
            tr = training_records[(row["arm"], ts)]
            if row["training_seed"] != ts or row["training_epoch"] != tr["selected_epoch"]:
                raise RuntimeError("Plot row does not match the fixed network training selection")
            if row.get("cache_key", {}).get("manifest_sha256") != binding:
                raise RuntimeError("Per-run cache is not bound to the complete evaluation")
    if read(out / "tables/individual_runs.json") != result["runs"]:
        raise RuntimeError("Individual table differs from complete evaluation")
    for mode in MODES:
        rows = [r for r in result["runs"] if r["mode"] == mode]
        for degree in (20, 28):
            if V.summarize(rows, degree) != result["summary"][mode][str(degree)]:
                raise RuntimeError("Summary statistics do not match all individual runs")
    for arm in ARMS[1:]:
        counts = {r["parameter_count"] for r in result["cost"]["training_runs"] if r["arm"] == arm}
        if counts != {PARAMETERS[arm]}:
            raise RuntimeError("Figure parameter-count caption differs from evaluated model")
    return result, E, T, V, J, source_records, training_records


def plot_save(fig, path, plt, artifacts):
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        destination = path.with_suffix(suffix)
        fig.savefig(destination, dpi=175, bbox_inches="tight", facecolor="white")
        artifacts.append(dict(path=str(destination.resolve()), sha256=sha(destination), kind="figure"))
    plt.close(fig)


def epoch_note(result, seed=None):
    pieces = []
    for arm in ARMS[1:]:
        entries = [r for r in result["cost"]["training_runs"] if r["arm"] == arm and (seed is None or r["seed"] == seed)]
        entries.sort(key=lambda r: r["seed"])
        pieces.append(LABELS[arm] + " e=" + "/".join(str(r["selected_epoch"]) for r in entries))
    return "; ".join(pieces)


def summary_figures(result, out, plt, artifacts):
    metric_specs = (("normal_rms_mm_s", "Normal RMS [mm/s]"),
                    ("superfluid_rms_mm_s", "Superfluid RMS [mm/s]"),
                    ("heldout_free_nll", "H free NLL / interval"),
                    ("heldout_cap_nll", "H captured NLL / interval"))
    for mode in MODES:
        fig, axs = plt.subplots(2, 4, figsize=(17.4, 7.2), layout="constrained")
        for row, (case, U) in enumerate(CASES):
            for col, (metric, label) in enumerate(metric_specs):
                ax = axs[row, col]
                for i, arm in enumerate(ARMS):
                    entry = result["summary"][mode]["20"][case]["arms"][arm][metric]
                    vals = [entry["by_seed"][str(s)] for s in SEEDS]
                    ax.bar(i, entry["mean"], width=.67, color=COLORS[arm], alpha=.8)
                    ax.scatter(np.arange(3) * .13 + i - .13, vals, c="black", s=20, zorder=4)
                ax.set_xticks(range(4), [LABELS[a] for a in ARMS], rotation=23, ha="right")
                ax.set_title(f"{U*1000:g} mm/s | {label}", fontsize=10)
                ax.grid(axis="y", alpha=.2)
        fig.suptitle(MODE_LABELS[mode] + " | mean and all three paired seeds\n"
                     "Full 5–100 ms; N40/K12; all arms use zero superfluid viscosity and Fns factor 1", fontsize=13)
        fig.supxlabel(PARAMETER_NOTE + "\n" + epoch_note(result) + "\n" + SCOPE, fontsize=8)
        plot_save(fig, out / "figures" / mode / "01_summary", plt, artifacts)
        fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2), layout="constrained")
        for row, (case, U) in enumerate(CASES):
            for fluid, name in enumerate(("Normal fluid", "Superfluid")):
                ax = axs[row, fluid]
                for arm in ARMS:
                    rr = [r for r in result["runs"] if r["case"] == case and r["mode"] == mode and r["arm"] == arm]
                    curves = np.asarray([r["field"]["time_rms_mm_s"] for r in rr])[:, :, fluid]
                    if curves.shape != (3, 96):
                        raise RuntimeError("Expected all three 96-frame error curves")
                    ax.plot(np.arange(5, 101), curves.mean(0), color=COLORS[arm], label=LABELS[arm])
                    ax.fill_between(np.arange(5, 101), curves.min(0), curves.max(0), color=COLORS[arm], alpha=.13)
                ax.set(title=f"{U*1000:g} mm/s | {name}", xlabel="Physical time [ms]", ylabel="Velocity RMS [mm/s]")
                ax.grid(alpha=.2)
        axs[0, 0].legend(fontsize=9)
        fig.suptitle(MODE_LABELS[mode] + "\nLine: paired-seed mean; band: full seed range", fontsize=12)
        fig.supxlabel("The same physical data are reused across seeds; ranges are not independent-sample confidence intervals.", fontsize=9)
        plot_save(fig, out / "figures" / mode / "02_time_errors", plt, artifacts)


def training_figures(result, T, training_records, out, plt, artifacts):
    fig, axs = plt.subplots(3, 4, figsize=(17.6, 10.2), layout="constrained")
    trace_sources = []
    for row, arm in enumerate(ARMS[1:]):
        for si, sd in enumerate(TRAIN_SEEDS):
            record = training_records[(arm, sd)]
            status = read(record["status_path"])
            trace_path = Path(record["path"]).parent / "trace.json"
            if sha(trace_path) != status["trace_sha256"]:
                raise RuntimeError("Network trace changed after selection freeze")
            trace = read(trace_path)
            epochs = [r["epoch"] for r in trace]
            if epochs != list(range(status["last_epoch"] + 1)):
                raise RuntimeError("Incomplete network optimization trace")
            trace_sources.append(dict(path=str(trace_path), sha256=sha(trace_path)))
            style = ("-", "--", ":")[si]
            label = f"seed {sd}, selected e{record['selected_epoch']}"
            axs[row, 0].plot(epochs[1:], [r["train"]["objective"] for r in trace[1:]],
                             color=COLORS[arm], linestyle=style, label=label)
            for col, kind in enumerate(("free", "cap", "total"), 1):
                yy = [r["validation"][kind] for r in trace]
                axs[row, col].plot(epochs, yy, color=COLORS[arm], linestyle=style)
                ep = record["selected_epoch"]
                axs[row, col].scatter(ep, yy[ep], marker="x", color="black", s=42, zorder=5)
        axs[row, 0].legend(fontsize=8)
        for col, title in enumerate(("Training objective", "V free endpoint NLL", "V captured endpoint NLL", "V balanced endpoint NLL")):
            ax = axs[row, col]
            ax.set(title=f"{LABELS[arm]} ({PARAMETERS[arm]:,} parameters)\n{title}", xlabel="Network epoch")
            ax.grid(alpha=.2)
    fig.suptitle("Non-target probability-model retraining | all seeds, including epoch-0 validation\n"
                 "Crosses mark each frozen network checkpoint selected by balanced V endpoint NLL", fontsize=13)
    fig.supxlabel("Training: seven non-target cases, PID % 5 >= 2; V: 6.5 mm/s, PID % 5 == 1. "
                  "Shared GRU: 3,648 parameters; state-specific MLPs/heads remain private.\n"
                  "Training loss is an online epoch aggregate (force/latent NLL + 0.1 endpoint NLL); "
                  "validation uses a separate float64 model copy and degree 20.", fontsize=8)
    plot_save(fig, out / "figures" / "03_network_training", plt, artifacts)
    return trace_sources


def field_optimization(result, E, V, source_records, out, plt, artifacts):
    histories = {}
    for key, record in source_records.items():
        ck = V.load_hashed(E, record["last_path"], record["last_sha256"])
        trace = ck["trace"]
        if [r["iteration"] for r in trace] != list(range(201)):
            raise RuntimeError("Target trace must cover all 200 updates")
        histories[key] = trace
    fig, axs = plt.subplots(4, 2, figsize=(12.4, 12.3), layout="constrained")
    for col, (case, U) in enumerate(CASES):
        for arm in ARMS:
            rr = [histories[(case, seed, arm)] for seed in SEEDS]
            values = np.asarray([[x["objective"] for x in h[1:]] for h in rr])
            axs[0, col].plot(np.arange(1, 201), values.mean(0), color=COLORS[arm], label=LABELS[arm])
            axs[0, col].fill_between(np.arange(1, 201), values.min(0), values.max(0), color=COLORS[arm], alpha=.13)
            for row, kind in enumerate(("free", "cap", "total"), 1):
                x = [v["iteration"] for v in rr[0] if "validation" in v]
                y = np.asarray([[v["validation"][kind] for v in h if "validation" in v] for h in rr])
                axs[row, col].plot(x, y.mean(0), color=COLORS[arm])
                axs[row, col].fill_between(x, y.min(0), y.max(0), color=COLORS[arm], alpha=.13)
                for seed, h in zip(SEEDS, rr):
                    it = source_records[(case, seed, arm)]["selected_iteration"]
                    selected = next(v for v in h if v["iteration"] == it)
                    axs[row, col].scatter(it, selected["validation"][kind], marker="x", color=COLORS[arm], s=40, zorder=5)
        for row, title in enumerate(("0.08 x I NLL + regularization (pre-update)", "V free NLL", "V captured NLL", "V balanced NLL")):
            ax = axs[row, col]
            ax.set(title=f"{U*1000:g} mm/s | {title}", xlabel="Field optimization update")
            ax.grid(alpha=.2)
    axs[0, 0].legend(fontsize=9)
    fig.suptitle("Target inversion with fully frozen probability networks\n"
                 "Lines: three-seed means; bands: seed ranges; crosses: primary selected checkpoints", fontsize=13)
    fig.supxlabel("All runs reach update 200. I fits the fields; V selects; F is unused. "
                  "A terminal diagnostic never replaces a primary selection.\n" + epoch_note(result), fontsize=9)
    plot_save(fig, out / "figures" / "04_field_optimization", plt, artifacts)


def select_row(result, mode, case, arm):
    return next(r for r in result["runs"] if (r["mode"], r["case"], r["arm"], r["seed"])
                == (mode, case, arm, REPRESENTATIVE_SEED))


@torch.no_grad()
def representative_fields(result, E, V, J, out, artifacts):
    exports, data = {}, {}
    for case, U in CASES:
        # All model, selection and evaluation bindings were validated first.
        source = result["reference_sources"][case]
        if sha(source["path"]) != source["sha256"]:
            raise RuntimeError("Reference changed after completed evaluation")
        reference = E.load(source["path"])["a"].double().cpu()
        if sha(source["path"]) != source["sha256"]:
            raise RuntimeError("Reference changed during plotting read")
        fields = {mode: {} for mode in MODES}
        c = None
        reports = []
        for arm in ARMS:
            if arm == "legacy":
                d, _, model, _, _, _ = E.P.make_context(U, REPRESENTATIVE_SEED, "new_frozen")
            else:
                d, _, model, _, _, _ = E.make_context(U, REPRESENTATIVE_SEED, arm)
            c = d["c"]
            for mode in MODES:
                row = select_row(result, mode, case, arm)
                ck = V.load_hashed(E, row["checkpoint_path"], row["checkpoint_sha256"])
                model.load_state_dict(ck["model_state"], strict=True)
                a, es, en = [x.detach().cpu() for x in model()]
                if mode == "primary" and not all(torch.equal(v, ck[k]) for v, k in zip((a, es, en), ("a", "eta_s", "eta_n"))):
                    raise RuntimeError("Selected coefficients disagree with replayed model state")
                if int(ck["iteration"]) != row["iteration"]:
                    raise RuntimeError("Replayed checkpoint iteration differs from evaluation row")
                x, value, means = J.resolve_fields(a, c)
                _, refv, _ = J.resolve_fields(reference, c)
                fields[mode][arm] = value
                rms = np.sqrt(np.square((value - refv) * 1000).sum(-1).mean((0, 2, 3)))
                delta = float(np.max(np.abs(rms - np.asarray(row["field"]["rms_mm_s"]))))
                if delta > 1e-10:
                    raise RuntimeError("Exported field grid RMS differs from completed evaluation")
                folder = out / "fields" / mode / case
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"seed_{REPRESENTATIVE_SEED}_{arm}.npz"
                np.savez_compressed(path, time_s=np.arange(5, 101) * .001, x_m=x, y_m=x,
                    a=a.numpy(), eta_s=es.numpy(), eta_n=en.numpy(), velocity_m_s=value,
                    velocity_error_m_s=value-refv, means_m_s=means,
                    field_seed=REPRESENTATIVE_SEED,
                    training_seed=-1 if arm == "legacy" else REPRESENTATIVE_TRAIN_SEED,
                    training_epoch=-1 if arm == "legacy" else row["training_epoch"],
                    iteration=row["iteration"], mode=mode, arm=arm,
                    checkpoint_path=row["checkpoint_path"], checkpoint_sha256=row["checkpoint_sha256"],
                    reference_sha256=source["sha256"], evaluation_manifest_sha256=result["manifest_sha256"])
                with np.load(path) as saved:
                    if saved["velocity_m_s"].shape != (96, 2, 40, 40, 2) or not np.array_equal(saved["velocity_m_s"], value):
                        raise RuntimeError("Complete 96-frame NPZ readback mismatch")
                record = dict(path=str(path.resolve()), sha256=sha(path), kind="field_npz", case=case,
                              mode=mode, arm=arm, iteration=row["iteration"], training_epoch=row["training_epoch"],
                              checkpoint_sha256=row["checkpoint_sha256"], reference_sha256=source["sha256"],
                              grid_vs_summary_rms_max_mm_s=delta, frames=96, npz_readback_exact=True)
                artifacts.append(record)
                reports.append(record)
        x, refv, means = J.resolve_fields(reference, c)
        path = out / "fields" / case / "reference_K12.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, time_s=np.arange(5, 101) * .001, x_m=x, y_m=x,
                            a=reference.numpy(), velocity_m_s=refv, means_m_s=means,
                            reference_path=source["path"], reference_sha256=source["sha256"])
        artifacts.append(dict(path=str(path.resolve()), sha256=sha(path), kind="reference_npz"))
        fields["reference"] = refv
        data[case] = dict(fields=fields, x=x, c=c)
        exports[case] = reports
    return exports, data


def primary_field_figures(result, data, out, plt, artifacts):
    for case, U in CASES:
        fields = data[case]["fields"]
        c = data[case]["c"]
        width = c["physics"]["box_m"] * 1000
        extent = (0, width, 0, width)
        for fluid, name in enumerate(("normal", "superfluid")):
            for kind in ("speed", "error"):
                keys = ("reference",) + ARMS if kind == "speed" else ARMS
                fig, axs = plt.subplots(3, len(keys), figsize=(16.5 if kind == "speed" else 14.2, 9.8), layout="constrained")
                images = {}
                for key in keys:
                    value = fields["reference"][:, fluid] if key == "reference" else fields["primary"][key][:, fluid]
                    if kind == "error":
                        value = value - fields["reference"][:, fluid]
                    images[key] = np.linalg.norm(value, axis=-1) * 1000
                maximum = max(float(v.max()) for v in images.values())
                for row, frame in enumerate((0, 45, 95)):
                    for col, key in enumerate(keys):
                        ax = axs[row, col]
                        im = ax.imshow(images[key][frame], origin="lower", extent=extent,
                                       vmin=0, vmax=maximum or 1., cmap="viridis" if kind == "speed" else "magma")
                        if row == 0:
                            if key == "reference":
                                title = "K12 reference"
                            else:
                                rr = select_row(result, "primary", case, key)
                                title = f"{LABELS[key]} | step {rr['iteration']}"
                                if key != "legacy":
                                    title += f"\nnetwork epoch {rr['training_epoch']}"
                            ax.set_title(title, fontsize=9)
                        if col == 0:
                            ax.set_ylabel(f"{frame+5} ms\ny [mm]")
                        if row == 2:
                            ax.set_xlabel("x [mm]")
                        ax.set_aspect("equal")
                fig.colorbar(im, ax=axs, shrink=.8,
                             label="Speed [mm/s]" if kind == "speed" else "Velocity-vector error norm [mm/s]")
                fig.suptitle(f"{U*1000:g} mm/s | {name} | validation-selected primary fields\n"
                             f"Fixed field seed {REPRESENTATIVE_SEED}; matched retraining seed {REPRESENTATIVE_TRAIN_SEED}", fontsize=13)
                fig.supxlabel("One common color scale across all shown arms and physical times; fixed fluid means included.\n" + PARAMETER_NOTE, fontsize=9)
                plot_save(fig, out / "figures/primary" / f"{case}_{name}_{kind}", plt, artifacts)


def terminal_figures(result, data, out, plt, artifacts):
    for case, U in CASES:
        fields, c = data[case]["fields"], data[case]["c"]
        width = c["physics"]["box_m"] * 1000
        extent = (0, width, 0, width)
        # Keep both modes and all four arms; columns cannot be selected by results.
        for kind in ("speed", "error"):
            keys = ("reference",) + ARMS
            fig, axs = plt.subplots(4, 5, figsize=(16.8, 12.2), layout="constrained")
            for fluid, name in enumerate(("Normal", "Superfluid")):
                images = {}
                for mode in MODES:
                    for key in keys:
                        value = fields["reference"][-1, fluid] if key == "reference" else fields[mode][key][-1, fluid]
                        if kind == "error":
                            value = value - fields["reference"][-1, fluid]
                        images[(mode, key)] = np.linalg.norm(value, axis=-1) * 1000
                maximum = max(float(v.max()) for v in images.values())
                for mi, mode in enumerate(MODES):
                    row = 2 * fluid + mi
                    for col, key in enumerate(keys):
                        ax = axs[row, col]
                        im = ax.imshow(images[(mode, key)], origin="lower", extent=extent,
                                       vmin=0, vmax=maximum or 1., cmap="viridis" if kind == "speed" else "magma")
                        if key != "reference":
                            rr = select_row(result, mode, case, key)
                            ax.set_title(f"{LABELS[key]} | step {rr['iteration']}", fontsize=9)
                        else:
                            ax.set_title("K12 reference", fontsize=9)
                        if col == 0:
                            ax.set_ylabel(name + (" | selected" if mode == "primary" else " | terminal") + "\ny [mm]")
                        if row == 3:
                            ax.set_xlabel("x [mm]")
                        ax.set_aspect("equal")
                fig.colorbar(im, ax=axs[2*fluid:2*fluid+2].ravel().tolist(), shrink=.8,
                             label="Speed [mm/s]" if kind == "speed" else "Vector error norm [mm/s]")
            fig.suptitle(f"{U*1000:g} mm/s | t = 100 ms | fixed field seed {REPRESENTATIVE_SEED}\n"
                         "Exploratory step-200 diagnostic; does not replace validation-selected results", fontsize=13)
            fig.supxlabel("A common color scale for each fluid across both modes and all arms.\n"
                          + epoch_note(result, REPRESENTATIVE_TRAIN_SEED) + "\n" + PARAMETER_NOTE, fontsize=9)
            plot_save(fig, out / "figures/terminal" / f"{case}_t100_selected_vs_terminal_{kind}", plt, artifacts)


@torch.no_grad()
def plot(out=OUT):
    out = Path(out).resolve()
    result, E, T, V, J, records, training_records = bind_all(out)
    # All gates and completed evaluations passed before this point. Reference
    # fields are loaded below; original packs may contain H but no H is scored.
    plt = J.prepare_plotting()
    artifacts = []
    summary_figures(result, out, plt, artifacts)
    traces = training_figures(result, T, training_records, out, plt, artifacts)
    field_optimization(result, E, V, records, out, plt, artifacts)
    exports, data = representative_fields(result, E, V, J, out, artifacts)
    primary_field_figures(result, data, out, plt, artifacts)
    terminal_figures(result, data, out, plt, artifacts)
    report = dict(complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
                  plotter_sha256=sha(__file__), evaluation_summary_sha256=sha(out / "TRANSITION_EVALUATION_SUMMARY.json"),
                  evaluation_manifest_sha256=result["manifest_sha256"], protocol_fingerprint=result["protocol_fingerprint"],
                  representative_field_seed=REPRESENTATIVE_SEED, representative_training_seed=REPRESENTATIVE_TRAIN_SEED,
                  all_four_arms_and_both_modes_retained=True, scoring_or_fitting_or_selection_performed=False,
                  target_reference_reads_before_complete_evaluation_and_gate_verification=False,
                  reference_sources=result["reference_sources"], training_trace_sources=traces,
                  parameter_counts=PARAMETERS, shared_gru_parameter_count=3648,
                  fields=exports, artifacts=artifacts)
    dump(out / "PLOT_MANIFEST.json", report)
    lines = ["共享free/cap转移概率对照：绘图与全时窗场导出", "",
             "四臂legacy/cap_retrain/independent/shared_gru全部保留；主选与固定200步末态分开，不按误差或H择优。",
             "legacy指历史冻结概率网络、复用上一轮new_frozen反演；四臂均使用超流黏度0、Fns系数1。",
             "先核对完整评价summary、9个训练选解、18个反演主/末态及legacy历史哈希链，再读取目标reference。",
             "原观测包会整体反序列化，H不进入训练、反演梯度或选解；本绘图程序不进行任何H响应评分。",
             "代表场固定field seed301030，对应training seed411040；未按照结果挑选样本。",
             "fields/primary与fields/terminal各含2工况×4臂的完整96帧NPZ，5–100ms、N40/K12；速度单位m/s。",
             "图中速度、向量误差单位mm/s；误差为速度矢量差的模，而不是速度模之差。",
             "主选场图显示5/50/100ms；终态图在100ms同时列出四臂主选与末态，共用每种流体色标。",
             "NPZ导出的网格RMS逐一与评价summary的系数RMS核对，误差不超过1e-10mm/s，并完成完整数组回读。",
             "训练曲线标注网络选中epoch；目标优化曲线的叉号标注场选中步数，二者不可混用。",
             "参数量：legacy/cap_retrain=8270，independent=16243，shared_gru=12595，其中共享历史GRU=3648。",
             "H是同窗、给定已观测历史的1ms条件预测，不是未来自由外推；三seed共享物理数据，非独立重复实验。",
             "PNG和SVG在figures下，全部路径、SHA与checkpoint/reference绑定见PLOT_MANIFEST.json。", ""]
    (out / "PLOT_README.txt").write_text("\n".join(lines), encoding="utf8")
    print("PLOTS COMPLETE", out / "PLOT_MANIFEST.json", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    plot(args.out)
