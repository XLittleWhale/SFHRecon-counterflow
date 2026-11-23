"""Post-selection evaluation of the physics/probability development comparison.

No target reference or held-out score is accessed before all eighteen required
selections have been frozen and their checkpoint hashes have been verified.
This program never fits, chooses a checkpoint, or changes scientific settings.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARMS = ("old_frozen", "new_frozen", "new_joint")
SEEDS = (301030, 301031, 301032)
CASES = (("vn0_0015m_s", .0015), ("vn0_004m_s", .004))
ARM_LABELS = {"old_frozen": "Old physics / frozen net",
              "new_frozen": "New physics / frozen net",
              "new_joint": "New physics / fine-tuned net"}
COLORS = {"old_frozen": "#66788a", "new_frozen": "#d28b22", "new_joint": "#176e83"}
REPRESENTATIVE_SEED = 301030
VERSION = "joint-probability-evaluation-v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, obj):
    def cast(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                                   allow_nan=False, default=cast), encoding="utf-8")
    temporary.replace(path)


def verify_gate(out, protocol_fingerprint):
    """Only public metadata and checkpoint bytes are inspected in this gate."""
    path = out / "selection_frozen.json"
    if not path.exists():
        raise RuntimeError("No global selection gate: held-out/reference evaluation is prohibited")
    gate = read(path)
    if gate.get("all_required_selections_frozen") is not True:
        raise RuntimeError("Global selection gate is not complete")
    if gate.get("fingerprint") != protocol_fingerprint:
        raise RuntimeError("Global selection gate is bound to a different protocol")
    expected = {(case, seed, arm) for case, _ in CASES for seed in SEEDS for arm in ARMS}
    entries = gate.get("runs", [])
    observed = {(r["case"], int(r["seed"]), r["arm"]) for r in entries}
    if observed != expected or len(entries) != len(expected):
        raise RuntimeError("The gate must contain exactly both cases, three seeds and three arms")
    for row in entries:
        target = out / "runs" / row["case"] / f"seed_{row['seed']}" / row["arm"] / "selected.pt"
        actual = Path(row["path"])
        if not actual.is_absolute():
            actual = out / actual
        if actual.resolve() != target.resolve():
            raise RuntimeError("Unexpected selected checkpoint path: " + str(actual))
        if sha(actual) != row["sha256"]:
            raise RuntimeError("Selected checkpoint changed after global gate: " + str(actual))
        target_u = dict(CASES)[row["case"]]
        if abs(float(row["U"]) - target_u) > 1e-14:
            raise RuntimeError("Case velocity in global gate is inconsistent")
    return gate, {"gate_path": str(path), "gate_sha256": sha(path),
                  "protocol_fingerprint": protocol_fingerprint,
                  "required_runs": len(expected), "verified_checkpoint_hashes": len(entries)}


def reference_path(case):
    if case == "vn0_0015m_s":
        return ROOT / "outputs/PINNClosureFactorial_20261030/data" / case / "reference.pt"
    return ROOT / "outputs/FullWindowCalibration_20261026/fullwindow_4mm_results/prepared" / case / "reference.pt"


def field_metrics(a, reference, U):
    a, reference = a.double().cpu(), reference.double().cpu()
    if a.shape != reference.shape or a.shape != (96, 2, 624):
        raise RuntimeError("Expected matching 96-frame, two-fluid, K12 coefficient trajectories")
    error = (a - reference) * (U * 1000)
    rms = (.5 * error.square().sum(-1).mean(0)).sqrt()
    time_rms = (.5 * error.square().sum(-1)).sqrt()
    denominator = (reference * (U * 1000)).square().sum(-1).mean(0)
    relative = (error.square().sum(-1).mean(0) / denominator).sqrt()
    return {"rms_mm_s": rms.tolist(), "time_rms_mm_s": time_rms.tolist(),
            "initial_rms_mm_s": time_rms[0].tolist(),
            "relative_fluctuation_rms": relative.tolist(),
            "reference": "N40/K12 matched-bandwidth reference; known fixed means cancel in errors"}


def load_selected(E, out, row, protocol_fingerprint):
    path = out / "runs" / row["case"] / f"seed_{row['seed']}" / row["arm"] / "selected.pt"
    if sha(path) != row["sha256"]:
        raise RuntimeError("Selected checkpoint changed after gate verification: " + str(path))
    ck = E.load(path)
    if ck.get("fingerprint") != protocol_fingerprint:
        raise RuntimeError("Selected checkpoint is bound to a different protocol: " + str(path))
    if ck["arm"] != row["arm"] or int(ck["seed"]) != int(row["seed"]):
        raise RuntimeError("Checkpoint identity does not match the global gate")
    if abs(float(ck["U"]) - float(row["U"])) > 1e-14:
        raise RuntimeError("Checkpoint velocity does not match the global gate")
    return ck


def network_from_checkpoint(base, ck):
    net = copy.deepcopy(base)
    net.load_state_dict(ck["net_state"], strict=True)
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net


def checked_score(E, ck, d, net, cal, force, group, degree):
    result = E.score(ck["a"], ck["eta_s"], ck["eta_n"], d, net, cal, force,
                     group, degree=degree, metrics=True)
    for key in ("total", "free", "cap"):
        if not np.isfinite(result[key]):
            raise RuntimeError(f"Nonfinite score: {group}/{degree}/{key}")
    for key in ("coverage", "spread"):
        for kind in ("free", "cap"):
            if not np.isfinite(result[key][kind]):
                raise RuntimeError(f"Nonfinite {key}: {group}/{degree}/{kind}")
    return result


def quadrature_difference(primary, higher):
    return {"degree_primary": 20, "degree_comparison": 28,
            "nll_28_minus_20": {k: higher[k] - primary[k] for k in ("total", "free", "cap")},
            "coverage_28_minus_20": {k: higher["coverage"][k] - primary["coverage"][k]
                                      for k in ("free", "cap")},
            "spread_um_28_minus_20": {k: higher["spread"][k] - primary["spread"][k]
                                       for k in ("free", "cap")},
            "pass_threshold": None, "interpretation": "Descriptive quadrature sensitivity; no new acceptance threshold"}


def decomposition(nf, jj, jn, nj):
    """2x2 fixed-field/fixed-network comparison; effects are descriptive."""
    out = {}
    for kind in ("total", "free", "cap"):
        out[kind] = {"total_joint_minus_new_frozen": jj[kind] - nf[kind],
                     "field_change_at_frozen_network": jn[kind] - nf[kind],
                     "network_change_at_new_frozen_field": nj[kind] - nf[kind],
                     "interaction": jj[kind] - jn[kind] - nj[kind] + nf[kind]}
        reconstructed = (out[kind]["field_change_at_frozen_network"] +
                         out[kind]["network_change_at_new_frozen_field"] + out[kind]["interaction"])
        if abs(reconstructed - out[kind]["total_joint_minus_new_frozen"]) > 1e-10:
            raise AssertionError("Factorial score decomposition does not reconstruct total change")
    return out


def summarize(rows):
    summary = {}
    index = {(r["case"], r["seed"], r["arm"]): r for r in rows}
    for case, U in CASES:
        item = {"U": U, "arms": {}, "paired_changes": {}}
        for arm in ARMS:
            rr = [index[(case, seed, arm)] for seed in SEEDS]
            values = {
                "normal_rms_mm_s": [r["field"]["rms_mm_s"][0] for r in rr],
                "superfluid_rms_mm_s": [r["field"]["rms_mm_s"][1] for r in rr],
                "heldout_free_nll": [r["observations"]["H"]["free"] for r in rr],
                "heldout_cap_nll": [r["observations"]["H"]["cap"] for r in rr],
                "heldout_free_coverage": [r["observations"]["H"]["coverage"]["free"] for r in rr],
                "heldout_cap_coverage": [r["observations"]["H"]["coverage"]["cap"] for r in rr],
                "heldout_free_spread_um": [r["observations"]["H"]["spread"]["free"] for r in rr],
                "heldout_cap_spread_um": [r["observations"]["H"]["spread"]["cap"] for r in rr],
            }
            item["arms"][arm] = {key: {"mean": float(np.mean(v)), "min": float(np.min(v)),
                                       "max": float(np.max(v)), "by_seed": dict(zip(map(str, SEEDS), v))}
                                  for key, v in values.items()}
        for first, second in (("new_frozen", "old_frozen"), ("new_joint", "new_frozen"),
                              ("new_joint", "old_frozen")):
            name = first + "_minus_" + second
            item["paired_changes"][name] = {}
            for metric in item["arms"][first]:
                aa, bb = item["arms"][first][metric], item["arms"][second][metric]
                vv = [aa["by_seed"][str(s)] - bb["by_seed"][str(s)] for s in SEEDS]
                item["paired_changes"][name][metric] = {"mean": float(np.mean(vv)),
                    "min": float(np.min(vv)), "max": float(np.max(vv)),
                    "by_seed": dict(zip(map(str, SEEDS), vv))}
        summary[case] = item
    return summary


def resolve_fields(a, c):
    from counterflow_full.core import basis
    a = torch.as_tensor(a, dtype=torch.float64).cpu()
    L, N = c["physics"]["box_m"], 40
    axis = (torch.arange(N, dtype=a.dtype) + .5) * L / N
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    V, _ = basis(torch.stack((xx, yy), -1).reshape(-1, 2), c, 12)
    means = a.new_tensor([[0, c["physics"]["vn_mean_m_s"]], [0, c["physics"]["vs_mean_m_s"]]])
    v = (torch.einsum("tfm,pmj->tfpj", a, V) + means[None, :, None]).reshape(96, 2, N, N, 2)
    return axis.numpy(), v.numpy(), means.numpy()


def prepare_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    return plt


def save_figure(fig, path, plt):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=170, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def overview_figures(rows, summary, out):
    plt = prepare_plotting()
    metric_specs = [("normal_rms_mm_s", "Normal velocity RMS [mm/s]"),
                    ("superfluid_rms_mm_s", "Superfluid velocity RMS [mm/s]"),
                    ("heldout_free_nll", "Held-out free NLL / interval"),
                    ("heldout_cap_nll", "Held-out captured NLL / interval")]
    fig, axs = plt.subplots(2, 4, figsize=(16.2, 7.2), layout="constrained")
    for row, (case, U) in enumerate(CASES):
        for col, (metric, label) in enumerate(metric_specs):
            ax = axs[row, col]
            for i, arm in enumerate(ARMS):
                values = summary[case]["arms"][arm][metric]
                ax.bar(i, values["mean"], width=.62, color=COLORS[arm], alpha=.75)
                vv = [values["by_seed"][str(s)] for s in SEEDS]
                ax.scatter(np.arange(3)*.12 + i - .12, vv, c="black", s=22, zorder=4)
            ax.set_xticks(range(3), ["Old / frozen", "New / frozen", "New / tuned"], rotation=20)
            ax.set_title(f"{U*1000:g} mm/s | {label}")
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("Physics and probability-model comparison | mean and all three selected seeds\n"
                 "Full 5–100 ms; K12 reference; independent held-out PIDs", fontsize=14)
    save_figure(fig, out / "figures/01_summary", plt)
    fig, axs = plt.subplots(2, 2, figsize=(11.3, 8), layout="constrained")
    for row, (case, U) in enumerate(CASES):
        for fluid, name in enumerate(("Normal", "Superfluid")):
            ax = axs[row, fluid]
            for arm in ARMS:
                curves = np.array([r["field"]["time_rms_mm_s"] for r in rows
                                   if r["case"] == case and r["arm"] == arm])[:, :, fluid]
                ax.plot(np.arange(5, 101), curves.mean(0), color=COLORS[arm], label=ARM_LABELS[arm])
                ax.fill_between(np.arange(5, 101), curves.min(0), curves.max(0), color=COLORS[arm], alpha=.12)
            ax.set(title=f"{U*1000:g} mm/s | {name}", xlabel="Time [ms]", ylabel="Velocity RMS [mm/s]")
            ax.grid(alpha=.2)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Full-window field errors | line: seed mean; band: seed range")
    save_figure(fig, out / "figures/02_time_errors", plt)


def export_representative(E, out, case, U, d, reference, records, protocol_fingerprint):
    """Fixed seed 301030; no representative selection based on test or truth."""
    plt = prepare_plotting()
    directory = out / "fields" / case
    directory.mkdir(parents=True, exist_ok=True)
    x, refv, means = resolve_fields(reference, d["c"])
    fields = {"reference": refv}
    np.savez_compressed(directory / "reference_K12.npz", time_s=np.arange(5, 101)*.001,
                        x_m=x, y_m=x, a=reference.cpu().numpy(), velocity_m_s=refv, means_m_s=means)
    checks = {}
    for arm in ARMS:
        row = next(r for r in records if r["case"] == case and r["seed"] == REPRESENTATIVE_SEED and r["arm"] == arm)
        ck = load_selected(E, out, row, protocol_fingerprint)
        _, value, _ = resolve_fields(ck["a"], d["c"])
        fields[arm] = value
        metric = field_metrics(ck["a"], reference, U)
        numerical = np.sqrt(np.square((value-refv)*1000).sum(-1).mean((0, 2, 3)))
        difference = float(np.max(np.abs(numerical - metric["rms_mm_s"])))
        if difference > 1e-10:
            raise AssertionError("Grid RMS disagrees with Fourier coefficient RMS")
        path = directory / f"seed_{REPRESENTATIVE_SEED}_{arm}.npz"
        np.savez_compressed(path, time_s=np.arange(5, 101)*.001, x_m=x, y_m=x,
                            a=ck["a"].cpu().numpy(), velocity_m_s=value,
                            velocity_error_m_s=value-refv, means_m_s=means)
        with np.load(path) as saved:
            if not np.array_equal(saved["velocity_m_s"], value):
                raise AssertionError("NPZ field readback mismatch")
        checks[arm] = {"file": str(path.relative_to(out)), "sha256": sha(path),
                       "grid_vs_coefficient_rms_max_mm_s": difference,
                       "npz_readback_exact": True}
    extent = (0, d["c"]["physics"]["box_m"]*1000, 0, d["c"]["physics"]["box_m"]*1000)
    frames = (0, 45, 95)
    for fluid, fluid_name in enumerate(("normal", "superfluid")):
        speed_max = max(float(np.linalg.norm(v[:, fluid], axis=-1).max()*1000) for v in fields.values())
        fig, axs = plt.subplots(3, 4, figsize=(13.5, 9.3), layout="constrained")
        for i, t in enumerate(frames):
            for j, key in enumerate(("reference",) + ARMS):
                ax = axs[i, j]
                speed = np.linalg.norm(fields[key][t, fluid], axis=-1)*1000
                im = ax.imshow(speed, origin="lower", extent=extent, vmin=0, vmax=speed_max, cmap="viridis")
                if i == 0:
                    ax.set_title("K12 reference" if key == "reference" else ARM_LABELS[key], fontsize=9)
                if j == 0:
                    ax.set_ylabel(f"{t+5} ms\ny [mm]")
                if i == 2:
                    ax.set_xlabel("x [mm]")
        fig.colorbar(im, ax=axs, shrink=.8, label="Speed [mm/s]")
        fig.suptitle(f"{U*1000:g} mm/s | {fluid_name} | fixed representative seed {REPRESENTATIVE_SEED}")
        save_figure(fig, out / "figures" / f"{case}_{fluid_name}_speed", plt)
        error_max = max(float(np.linalg.norm(v[:, fluid]-refv[:, fluid], axis=-1).max()*1000)
                        for key, v in fields.items() if key != "reference")
        fig, axs = plt.subplots(3, 3, figsize=(10.6, 9.3), layout="constrained")
        for i, t in enumerate(frames):
            for j, arm in enumerate(ARMS):
                ax = axs[i, j]
                error = np.linalg.norm(fields[arm][t, fluid]-refv[t, fluid], axis=-1)*1000
                im = ax.imshow(error, origin="lower", extent=extent, vmin=0, vmax=error_max, cmap="magma")
                if i == 0:
                    ax.set_title(ARM_LABELS[arm], fontsize=9)
                if j == 0:
                    ax.set_ylabel(f"{t+5} ms\ny [mm]")
                if i == 2:
                    ax.set_xlabel("x [mm]")
        fig.colorbar(im, ax=axs, shrink=.8, label="Velocity-vector error norm [mm/s]")
        fig.suptitle(f"{U*1000:g} mm/s | {fluid_name} error relative to K12 reference")
        save_figure(fig, out / "figures" / f"{case}_{fluid_name}_error", plt)
    return checks


def write_report(summary, crosses, rows, out):
    lines = ["联合概率模型微调／流场反演：事后开发对照评价", "",
             "所有18个checkpoint先由全局gate冻结并核对SHA256，再读取留出H与目标参考场。",
             "本实验使用已在历史诊断中检查过的工况，因此不声称全新盲测。",
             "三臂：old_frozen=原演化算子+冻结网络；new_frozen=新算子+冻结网络；new_joint=新算子+微调网络。",
             "时间窗5–100 ms，N40/K12参考，两个流体的固定均值相同；误差不包括0–4 ms。",
             "F用于概率模型微调，I仅用于流场反演，V用于选解，H只用于冻结后的评价；同一物理PID不跨组。",
             "每个工况/初值独立从同一预训练模型开始微调；这是工况内的粒子留出验证，不是未参与适配的新工况外推。",
             "NLL按原生1ms区间等状态、等PID加权；coverage为名义90%区域覆盖率；spread为预测终点位移sqrt(trace Cov)，单位um。",
             "H检验同一5–100 ms时间窗内未参与微调/反演的PID：每个1ms预测给定该粒子的已观测历史；它不是未来时间段的自由轨迹外推。",
             "所有结果报告三个指定种子；代表场图固定seed301030，未按真值或H选图。",
             "三个种子共享同一批观测及PID划分，只衡量初始化敏感性，不是三个独立数据集或独立物理重复实验。", ""]
    for case, U in CASES:
        lines.extend([f"工况 {U*1000:g} mm/s（三种子均值）", "arm | normal RMS mm/s | superfluid RMS mm/s | H free NLL | H cap NLL | H free coverage | H cap coverage | free/cap spread um"])
        for arm in ARMS:
            a = summary[case]["arms"][arm]
            mean = lambda k: a[k]["mean"]
            lines.append(f"{arm} | {mean('normal_rms_mm_s'):.7g} | {mean('superfluid_rms_mm_s'):.7g} | "
                         f"{mean('heldout_free_nll'):.7g} | {mean('heldout_cap_nll'):.7g} | "
                         f"{mean('heldout_free_coverage'):.2%} | {mean('heldout_cap_coverage'):.2%} | "
                         f"{mean('heldout_free_spread_um'):.7g}/{mean('heldout_cap_spread_um'):.7g}")
        for comparison, changes in summary[case]["paired_changes"].items():
            lines.append(comparison + ": " + "; ".join(f"{key} mean={changes[key]['mean']:+.7g}, "
                         f"seed range=[{changes[key]['min']:+.7g},{changes[key]['max']:+.7g}]"
                         for key in ("normal_rms_mm_s", "superfluid_rms_mm_s", "heldout_free_nll", "heldout_cap_nll")))
        subset = [r for r in rows if r["case"] == case]
        for state in ("free", "cap"):
            gap = max(abs(r["quadrature"]["nll_28_minus_20"][state]) for r in subset)
            lines.append(f"H {state}: 各臂/种子degree28相对20的最大绝对NLL差={gap:.7g}；仅描述，无新增通过阈值。")
        lines.append("交叉评分分解（H NLL，new_joint相对new_frozen，三种子平均）：")
        cc = [r for r in crosses if r["case"] == case]
        for state in ("free", "cap"):
            terms = cc[0]["decomposition"][state]
            lines.append(state + ": " + "; ".join(f"{key}={np.mean([r['decomposition'][state][key] for r in cc]):+.7g}" for key in terms))
        lines.append("")
    lines.extend(["解释边界：", "RMS/NLL降低表示改善；覆盖率应结合其对名义90%的距离与预测spread解释，不能靠变宽单独宣称改善。",
                  "微调网络在当前模型中只进入捕获态latent分布，自由态的变化由反演流场引起。",
                  "交叉评分是固定场/固定网络的事后代入分解，不是另一次拟合，也不证明因果可辨识性。",
                  "degree20/28差是积分敏感性诊断；本评价不设置事后通过阈值，不改变保护选解。",
                  "未改原Frozen B与正式基线；新物理参数和数据划分的公平控制以本实验protocol为准。", ""])
    (out / "JOINT_EVALUATION_REPORT.txt").write_text("\n".join(lines), encoding="utf-8")


@torch.no_grad()
def evaluate(out=None, plots=True):
    import experiment as E
    E.M.R.setup("cpu")
    out = Path(out) if out is not None else Path(E.OUT)
    out = out.resolve()
    if out != Path(E.OUT).resolve():
        raise RuntimeError("Evaluation directory must match the experiment's verified OUT directory")
    # Always reverify current source/input hashes, including on a fully cached run.
    # No cache lookup or target reference/H access is permitted before this check.
    protocol = E.verify_protocol()
    protocol_fingerprint = protocol["fingerprint"]
    if protocol.get("evaluation_degree") != 20 or protocol.get("evaluation_sensitivity_degree") != 28:
        raise RuntimeError("Protocol must declare primary degree20 and sensitivity degree28")
    gate, gate_audit = verify_gate(out, protocol_fingerprint)
    fingerprint = {"version": VERSION, "evaluator_sha256": sha(__file__),
                   "experiment_sha256": sha(Path(E.__file__)),
                   "protocol_sha256": sha(out / "protocol.json"), **gate_audit}
    records = sorted(gate["runs"], key=lambda r: (r["case"], int(r["seed"]), ARMS.index(r["arm"])))
    # Verify every embedded checkpoint identity/fingerprint before opening even
    # the first target reference. Repeat the check on each later checkpoint load.
    for record in records:
        load_selected(E, out, record, protocol_fingerprint)
    base_net, cal, force = E.EW.components("cpu")
    rows, crosses, reference_audit, export_audit = [], [], {}, {}
    began = time.perf_counter()
    for case, U in CASES:
        d, cfg, _, _ = E.load_case(U, EW=E.EW)
        rp = reference_path(case)
        reference = E.load(rp)["a"].double().cpu()
        reference_audit[case] = {"path": str(rp), "sha256": sha(rp)}
        for seed in SEEDS:
            ck_by_arm, row_by_arm = {}, {}
            for arm in ARMS:
                record = next(r for r in records if r["case"] == case and r["seed"] == seed and r["arm"] == arm)
                ck = load_selected(E, out, record, protocol_fingerprint)
                ck_by_arm[arm] = ck
                path = out / "evaluation" / case / f"seed_{seed}" / f"{arm}.json"
                cache_key = {**fingerprint, "checkpoint_sha256": record["sha256"], "reference_sha256": sha(rp)}
                if path.exists() and read(path).get("fingerprint") == cache_key:
                    row = read(path)
                else:
                    net = network_from_checkpoint(base_net, ck)
                    observations = {g: checked_score(E, ck, d, net, cal, force, g, 20) for g in ("F", "I", "V", "H")}
                    higher = checked_score(E, ck, d, net, cal, force, "H", 28)
                    row = {"case": case, "U": U, "seed": seed, "arm": arm,
                           "selected_iteration": int(ck["iteration"]), "checkpoint_path": record["path"],
                           "fingerprint": cache_key, "field": field_metrics(ck["a"], reference, U),
                           "observations": observations, "heldout_degree28": higher,
                           "quadrature": quadrature_difference(observations["H"], higher)}
                    dump(path, row)
                rows.append(row)
                row_by_arm[arm] = row
                print(f"EVALUATED {case} seed={seed} arm={arm}", flush=True)
            path = out / "evaluation" / case / f"seed_{seed}" / "cross_scores.json"
            key = {**fingerprint, "selected_sha256": {r["arm"]: r["sha256"] for r in records
                    if r["case"] == case and r["seed"] == seed}}
            if path.exists() and read(path).get("fingerprint") == key:
                cross = read(path)
            else:
                frozen_net = network_from_checkpoint(base_net, ck_by_arm["new_frozen"])
                joint_net = network_from_checkpoint(base_net, ck_by_arm["new_joint"])
                joint_field_frozen_net = checked_score(E, ck_by_arm["new_joint"], d, frozen_net, cal, force, "H", 20)
                frozen_field_joint_net = checked_score(E, ck_by_arm["new_frozen"], d, joint_net, cal, force, "H", 20)
                nf, jj = row_by_arm["new_frozen"]["observations"]["H"], row_by_arm["new_joint"]["observations"]["H"]
                cross = {"case": case, "U": U, "seed": seed, "fingerprint": key, "group": "H", "degree": 20,
                         "new_frozen_field_frozen_net": nf, "new_joint_field_joint_net": jj,
                         "new_joint_field_frozen_net": joint_field_frozen_net,
                         "new_frozen_field_joint_net": frozen_field_joint_net,
                         "decomposition": decomposition(nf, jj, joint_field_frozen_net, frozen_field_joint_net)}
                dump(path, cross)
            crosses.append(cross)
        if plots:
            export_audit[case] = export_representative(E, out, case, U, d, reference, records, protocol_fingerprint)
    summary = summarize(rows)
    dump(out / "tables/joint_summary.json", summary)
    dump(out / "tables/individual_runs.json", rows)
    dump(out / "tables/cross_scores.json", crosses)
    if plots:
        overview_figures(rows, summary, out)
    write_report(summary, crosses, rows, out)
    result = {"version": VERSION, "complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
              "wall_seconds": time.perf_counter()-began, "fingerprint": fingerprint,
              "target_truth_or_H_read_before_global_gate": False, "reference_sources": reference_audit,
              "runs": len(rows), "cross_evaluations": len(crosses), "summary": summary,
              "heldout_prediction_scope": "Same-window 1ms conditional endpoint prediction given observed particle history; not future free-running trajectory extrapolation",
              "seed_interpretation": "Initialization sensitivity on identical observations and PID splits; not independent datasets or physical repeats",
              "representative_seed": REPRESENTATIVE_SEED, "field_exports": export_audit,
              "plots_generated": bool(plots), "selection_or_fitting_performed": False}
    dump(out / "JOINT_EVALUATION_SUMMARY.json", result)
    print("EVALUATION COMPLETE", out / "JOINT_EVALUATION_REPORT.txt", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    evaluate(args.out, plots=not args.no_plots)
