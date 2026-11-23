"""Plot already-completed exploratory terminal diagnostics, without selection.

The complete secondary report and all eighteen terminal-byte hashes must pass
before any checkpoint replay, target reference read, or figure generation.
No likelihood evaluation, fitting, promotion or representative selection occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parents[2] / "outputs/JointProbabilityPhysics_20261103"
CASES = (("vn0_0015m_s", .0015), ("vn0_004m_s", .004))
SEEDS = (301030, 301031, 301032)
ARMS = ("old_frozen", "new_frozen", "new_joint")
FIXED_SEED = 301030
CAUTION_EN = "Exploratory terminal diagnostic; does not replace validation-selected results"
CAUTION_ZH = "探索性终态诊断，不替代选解"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def verify_completed_secondary(out):
    """Byte/metadata checks only; deliberately callable without experiment imports."""
    path = out / "secondary_terminal.json"
    if not path.exists():
        raise RuntimeError("Completed secondary terminal report missing; plots may not access checkpoints or reference")
    result = read(path)
    if result.get("complete") is not True or result.get("primary_selection_unchanged") is not True:
        raise RuntimeError("Secondary terminal analysis is incomplete or has altered primary selection")
    if result.get("exploratory") is not True or result.get("fitting_or_selection_performed") is not False:
        raise RuntimeError("Unexpected secondary-analysis scope")
    mp = out / "secondary_terminal_manifest.json"
    if not mp.exists() or sha(mp) != result.get("manifest_sha256"):
        raise RuntimeError("Terminal manifest differs from completed secondary report")
    manifest = read(mp)
    expected = {(c, s, a) for c, _ in CASES for s in SEEDS for a in ARMS}
    for label, entries in (("manifest", manifest.get("runs", [])), ("result", result.get("runs", []))):
        observed = {(r["case"], int(r["seed"]), r["arm"]) for r in entries}
        if observed != expected or len(entries) != 18:
            raise RuntimeError("Exactly eighteen unique fixed runs required in terminal " + label)
    if manifest.get("all_terminal_hashes_bound_before_H_reference_reads") is not True:
        raise RuntimeError("Terminal pre-read hash binding is missing")
    if manifest.get("global_gate_sha256") != sha(out / "selection_frozen.json"):
        raise RuntimeError("Primary selection gate changed after terminal binding")
    if manifest.get("plan_sha256") != sha(out / "secondary_analysis_plan.json"):
        raise RuntimeError("Secondary analysis plan changed after terminal binding")
    records = {(r["case"], int(r["seed"]), r["arm"]): r for r in manifest["runs"]}
    for row in result["runs"]:
        record = records[(row["case"], int(row["seed"]), row["arm"])]
        if (row.get("iteration") != 200 or row.get("manifest_sha256") != result["manifest_sha256"]
                or row.get("last_checkpoint_sha256") != record["sha256"]):
            raise RuntimeError("Completed terminal row does not match manifest")
    # Validate every terminal run, not just the fixed representative seed.
    for record in manifest["runs"]:
        folder = out / "runs" / record["case"] / f"seed_{record['seed']}" / record["arm"]
        target = folder / "last.pt"
        declared = Path(record["path"])
        if (declared.resolve() != target.resolve() or record.get("iteration") != 200
                or sha(target) != record["sha256"]):
            raise RuntimeError("Terminal checkpoint differs from bound manifest: " + str(target))
        if sha(folder / "status.json") != record["status_sha256"]:
            raise RuntimeError("Terminal status changed: " + str(folder))
        if sha(folder / "selected.pt") != record["primary_selected_sha256"]:
            raise RuntimeError("Primary selection changed: " + str(folder))
    return result, manifest


def render_comparison(J, out, case, U, x, fields, selected_iteration):
    """At t=100 ms, one common color scale per fluid row and figure."""
    plt = J.prepare_plotting()
    labels = (("reference", "K12 reference"),
              ("selected_new_joint", f"Selected new joint\nstep {selected_iteration}"),
              ("terminal_new_frozen", "Terminal new frozen\nstep 200"),
              ("terminal_new_joint", "Terminal new joint\nstep 200"))
    width_mm = float((x[1]-x[0])*len(x)*1000)
    extent = (0, width_mm, 0, width_mm)
    generated = []
    for kind in ("speed", "error"):
        fig, axs = plt.subplots(2, 4, figsize=(14.3, 7.1), layout="constrained")
        for fluid, fluid_name in enumerate(("Normal fluid", "Superfluid")):
            values = {}
            for key, _ in labels:
                value = fields[key][-1, fluid]
                if kind == "error":
                    value = value-fields["reference"][-1, fluid]
                values[key] = np.linalg.norm(value, axis=-1)*1000
            maximum = max(float(value.max()) for value in values.values())
            for col, (key, label) in enumerate(labels):
                ax = axs[fluid, col]
                im = ax.imshow(values[key], origin="lower", extent=extent, vmin=0,
                               vmax=maximum if maximum > 0 else 1., cmap="viridis" if kind == "speed" else "magma")
                if fluid == 0:
                    ax.set_title(label, fontsize=10)
                if col == 0:
                    ax.set_ylabel(fluid_name + "\ny [mm]")
                ax.set_xlabel("x [mm]")
                ax.set_aspect("equal")
            fig.colorbar(im, ax=axs[fluid].tolist(), shrink=.88,
                         label="Speed [mm/s]" if kind == "speed" else "Velocity-vector error norm [mm/s]")
        fig.suptitle(f"{U*1000:g} mm/s | t = 100 ms | fixed seed {FIXED_SEED}\n{CAUTION_EN}", fontsize=12)
        fig.supxlabel("Common color scale within each fluid row. " +
                      ("Errors are relative to the K12 reference; its own error is zero." if kind == "error"
                       else "Known fixed mean velocities are included."), fontsize=9)
        path = out / "terminal_figures" / f"{case}_t100_{kind}"
        J.save_figure(fig, path, plt)
        generated.extend([path.with_suffix(".png"), path.with_suffix(".svg")])
    return generated


def render_overview(J, out, secondary):
    plt = J.prepare_plotting()
    metrics = (("normal_rms_mm_s", "Normal RMS [mm/s]"),
               ("superfluid_rms_mm_s", "Superfluid RMS [mm/s]"),
               ("heldout_free_nll", "Held-out free NLL / interval"),
               ("heldout_cap_nll", "Held-out captured NLL / interval"))
    fig, axs = plt.subplots(2, 4, figsize=(16.3, 7.2), layout="constrained")
    for row, (case, U) in enumerate(CASES):
        for col, (metric, label) in enumerate(metrics):
            ax = axs[row, col]
            for i, arm in enumerate(ARMS):
                value = secondary["summary"][case]["arms"][arm][metric]
                samples = [float(value["by_seed"][str(seed)]) for seed in SEEDS]
                if not np.all(np.isfinite(samples)):
                    raise RuntimeError("Nonfinite cached secondary overview value")
                ax.bar(i, float(value["mean"]), width=.62, color=J.COLORS[arm], alpha=.75)
                ax.scatter(np.arange(3)*.12+i-.12, samples, c="black", s=22, zorder=4)
            ax.set_xticks(range(3), ["Old / frozen", "New / frozen", "New / tuned"], rotation=20)
            ax.set_title(f"{U*1000:g} mm/s | {label}", fontsize=10)
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("Fixed step-200 terminal states | mean and all three initialization seeds\n" + CAUTION_EN, fontsize=12)
    fig.supxlabel("H: same-window 1 ms prediction conditioned on observed particle history; not future free-running prediction.",
                  fontsize=9)
    path = out / "terminal_figures/terminal_overview"
    J.save_figure(fig, path, plt)
    return [path.with_suffix(".png"), path.with_suffix(".svg")]


@torch.no_grad()
def plot(out=DEFAULT_OUT):
    out = Path(out).resolve()
    secondary, manifest = verify_completed_secondary(out)
    # No checkpoint/target deserialization or image output precedes the guard.
    import experiment as E
    import evaluate_joint as J
    import evaluate_terminal as T
    E.M.R.setup("cpu")
    if out != Path(E.OUT).resolve():
        raise RuntimeError("Terminal plot directory must match frozen experiment OUT")
    protocol = E.verify_protocol()
    gate, _ = T.cheap_gate(out)
    T.verify_plan(out)
    J.verify_gate(out, protocol["fingerprint"])
    if manifest["protocol_fingerprint"] != protocol["fingerprint"] or manifest["evaluator_sha256"] != sha(T.__file__):
        raise RuntimeError("Terminal evaluation provenance changed")
    outputs, field_checks = [], {}
    for case, U in CASES:
        reference_record = secondary["reference_sources"][case]
        rp = J.reference_path(case)
        if Path(reference_record["path"]).resolve() != rp.resolve() or sha(rp) != reference_record["sha256"]:
            raise RuntimeError("Reference differs from completed terminal evaluation")
        reference = E.load(rp)["a"].double().cpu()
        fields, c, x = {}, None, None
        field_checks[case] = {}
        for arm in ARMS:
            record = next(r for r in manifest["runs"]
                          if (r["case"], int(r["seed"]), r["arm"]) == (case, FIXED_SEED, arm))
            ck, terminal, d, _, _, _ = T.load_terminal(E, out, record, protocol["fingerprint"])
            c = d["c"]
            x, velocity, means = J.resolve_fields(terminal["a"], c)
            if "reference" not in fields:
                _, fields["reference"], _ = J.resolve_fields(reference, c)
            fields["terminal_"+arm] = velocity
            terminal_row = next(r for r in secondary["runs"]
                               if (r["case"], int(r["seed"]), r["arm"]) == (case, FIXED_SEED, arm))
            grid_rms = np.sqrt(np.square((velocity-fields["reference"])*1000).sum(-1).mean((0, 2, 3)))
            difference = float(np.max(np.abs(grid_rms-np.asarray(terminal_row["field"]["rms_mm_s"]))))
            if difference > 1e-10:
                raise RuntimeError("Replayed terminal grid RMS differs from completed diagnostic")
            destination = out / "terminal_fields" / case / f"seed_{FIXED_SEED}_{arm}_step200.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, time_s=np.arange(5, 101)*.001, x_m=x, y_m=x,
                                a=terminal["a"].double().cpu().numpy(), velocity_m_s=velocity,
                                velocity_error_m_s=velocity-fields["reference"], means_m_s=means,
                                seed=np.int64(FIXED_SEED), iteration=np.int64(200),
                                purpose=np.array(CAUTION_EN),
                                axis_order=np.array(["time", "fluid", "y", "x", "component"]))
            with np.load(destination) as saved:
                if saved["velocity_m_s"].shape != (96, 2, 40, 40, 2) or not np.array_equal(saved["velocity_m_s"], velocity):
                    raise AssertionError("Terminal velocity NPZ readback failed")
            outputs.append(destination)
            field_checks[case][arm] = {"frames": 96, "first_time_s": .005, "last_time_s": .1,
                                      "terminal_sha256": record["sha256"],
                                      "grid_vs_saved_metric_max_difference_mm_s": difference,
                                      "npz_readback_exact": True}
        selected_record = next(r for r in gate["runs"]
                               if (r["case"], int(r["seed"]), r["arm"]) == (case, FIXED_SEED, "new_joint"))
        selected = J.load_selected(E, out, selected_record, protocol["fingerprint"])
        _, fields["selected_new_joint"], _ = J.resolve_fields(selected["a"], c)
        field_checks[case]["selected_new_joint"] = {"iteration": int(selected["iteration"]),
                                                    "selected_sha256": selected_record["sha256"]}
        outputs.extend(render_comparison(J, out, case, U, x, fields, int(selected["iteration"])))
    outputs.extend(render_overview(J, out, secondary))
    explanation = [
        CAUTION_ZH, "",
        "这些图在secondary_terminal.json完整、其manifest以及全部18个末态SHA校验通过后生成。",
        "训练时observations.pt按整包加载，包含F/I/V/H；损失与选解严格使用分组mask，H响应不进入梯度或选解。gate之后才对H响应评分/分析，并读取目标参考场。",
        "固定seed301030展示场，不按真值或H重新挑选代表样本。所有主选解保持不变。",
        "速度/误差图：t=100 ms，两行常流/超流，四列K12参考/selected new_joint/terminal new_frozen/terminal new_joint。",
        "每行使用同一色标，图示单位mm/s；误差为速度向量差的模，参考列自身误差为零。",
        "NPZ：每工况三臂第200步的完整96帧，覆盖5–100 ms；速度单位m/s、坐标m、时间s。",
        "NPZ轴顺序time,fluid,y,x,component；fluid=normal/superfluid，component=vx/vy；包含固定已知均值。",
        "terminal_overview展示所有三个初始化种子的终态RMS和已完成H NLL；没有新增预测评分。",
        "H是同窗内给定已观测历史的1ms条件预测，不是未来自由轨迹外推；三个seed不是独立物理数据重复。",
        "终态属于事后探索性诊断，不可替代或择优覆盖冻结的V选解结果。",
    ]
    txt = out / "terminal_plots.txt"
    txt.write_text("\n".join(explanation)+"\n", encoding="utf-8")
    outputs.append(txt)
    report = {"version": "exploratory-terminal-plots-v1", "complete": True,
              "created_utc": datetime.now(timezone.utc).isoformat(), "purpose": CAUTION_ZH,
              "fixed_representative_seed": FIXED_SEED, "primary_selection_unchanged": True,
              "new_prediction_scores_computed": False, "fits_or_selection_performed": False,
              "data_isolation_semantics": {
                  "observation_package_loaded_in_full_during_fit": True,
                  "package_contains_groups": ["F", "I", "V", "H"],
                  "heldout_H_responses_used_in_gradients_or_selection": False,
                  "target_reference_read_or_H_scored_before_global_gate": False,
                  "note": "整包反序列化不等于H响应参与拟合；隔离边界是梯度、选解和评分/分析。"},
              "protocol_fingerprint": protocol["fingerprint"],
              "secondary_report_sha256": sha(out / "secondary_terminal.json"),
              "terminal_manifest_sha256": sha(out / "secondary_terminal_manifest.json"),
              "plot_source_sha256": sha(__file__), "field_checks": field_checks,
              "outputs": [{"path": str(p.relative_to(out)), "sha256": sha(p), "bytes": p.stat().st_size} for p in outputs]}
    dump(out / "terminal_plots.json", report)
    print(json.dumps({"complete": True, "outputs": len(outputs), "report": str(out/"terminal_plots.json")}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check-gate-only", action="store_true")
    args = parser.parse_args()
    if args.check_gate_only:
        verify_completed_secondary(args.out.resolve())
        print("Completed terminal report and all eighteen bound hashes verified; no checkpoint/reference deserialized")
    else:
        plot(args.out)
