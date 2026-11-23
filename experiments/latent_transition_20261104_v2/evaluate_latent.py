"""Gate-bound evaluation of iid/reset/temporal captured latent transitions.

No target H response is scored and no target reference is opened until all
three transition heads and all eighteen primary/terminal field states pass
the metadata, byte-hash and frozen-parameter checks. This script performs no
optimization or model selection. Reused iid checkpoints are independently
rescored under their original marginal likelihood after the global gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "outputs/LatentTransition_20261104_v2"
HELPER = HERE.parent / "joint_probability_20261103/evaluate_joint.py"
CASES = (("vn0_0015m_s", .0015), ("vn0_004m_s", .004))
SEEDS = (301030, 301031, 301032)
TRAIN_SEEDS = (411040, 411041, 411042)
ARMS = ("iid", "reset", "temporal")
NEW_ARMS = ARMS[1:]
MODES = ("primary", "terminal")
PAIRS = (("reset", "iid"), ("temporal", "reset"), ("temporal", "iid"))
DEGREES = (20, 28)
KEYS = ("total", "free", "cap")
REPRESENTATIVE_SEED = SEEDS[0]
VERSION = "latent-transition-evaluation-v1"
COLORS = dict(iid="#777777", reset="#1f77b4", temporal="#d95f02")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def same_state(actual, expected, label):
    if set(actual) != set(expected):
        raise RuntimeError("Different state keys: " + label)
    for key, value in actual.items():
        ref = expected[key].to(value)
        if value.shape != ref.shape or not torch.equal(value, ref):
            raise RuntimeError("Frozen tensor changed: " + label + "/" + key)


def load_hashed(E, path, digest):
    if sha(path) != digest:
        raise RuntimeError("Changed checkpoint: " + str(path))
    value = E.load(path)
    if sha(path) != digest:
        raise RuntimeError("Checkpoint changed during reading: " + str(path))
    return value


def cheap_gate(out):
    """Strict metadata-only check; does not import experiment or load tensors."""
    train_path, field_path = out / "training/selection_frozen.json", out / "selection_frozen.json"
    if not train_path.is_file() or not field_path.is_file():
        raise RuntimeError("All three network and eighteen field selections/terminal states must be frozen")
    training, fields = read(train_path), read(field_path)
    tr, fr = training.get("runs", []), fields.get("runs", [])
    if (training.get("all_required_selections_frozen") is not True or len(tr) != 3
            or {int(r["seed"]) for r in tr} != set(TRAIN_SEEDS)):
        raise RuntimeError("Incomplete three-head training gate")
    expected = {(c, s, a) for c, _ in CASES for s in SEEDS for a in ARMS}
    if (fields.get("all_required_selections_frozen") is not True
            or fields.get("all_terminal_states_bound_before_evaluation") is not True
            or len(fr) != 18 or {(r["case"], int(r["seed"]), r["arm"]) for r in fr} != expected):
        raise RuntimeError("All eighteen field states, including six reused iid runs, must be bound")
    for row in tr:
        path = Path(row.get("status_path", str(Path(row["path"]).parent / "status.json")))
        status = read(path)
        if status.get("complete") is not True:
            raise RuntimeError("Incomplete training status: " + str(path))
    for row in fr:
        path = Path(row.get("status_path", str(Path(row["path"]).parent / "status.json")))
        status = read(path)
        if (status.get("complete") is not True or status.get("iteration") != 200 or status.get("network_steps") != 0
                or status.get("arm") != row["arm"] or int(status.get("seed", -1)) != int(row["seed"])
                or status.get("case") != row["case"] or status.get("selected_iteration") != row["selected_iteration"]
                or status.get("fingerprint") != fields["fingerprint"]):
            raise RuntimeError("Incomplete or unfrozen field status: " + str(path))
    return training, fields


def bind_gates(E, T, out, training, fields):
    protocol = E.verify_protocol()
    tp = protocol
    if E.fingerprint({k: v for k, v in training.items() if k != "fingerprint"}) != training["fingerprint"]:
        raise RuntimeError("Training gate self-fingerprint mismatch")
    sources = {str(Path(p).resolve()): h for p, h in protocol["sources"].items()}
    for path in (Path(__file__), HELPER):
        if sources.get(str(path.resolve())) != sha(path):
            raise RuntimeError("Evaluator/helper source missing from prospective protocol: " + str(path))
    if (protocol.get("evaluation_degree") != 20 or protocol.get("evaluation_sensitivity_degree") != 28
            or fields["fingerprint"] != protocol["fingerprint"]
            or fields.get("training_gate_sha256") != sha(out / "training/selection_frozen.json")
            or training["protocol_fingerprint"] != tp["fingerprint"]):
        raise RuntimeError("Global gate/protocol linkage mismatch")
    train = {}
    for row in training["runs"]:
        status_path = Path(row.get("status_path", str(Path(row["path"]).parent / "status.json")))
        if row.get("status_sha256") != sha(status_path):
            raise RuntimeError("Training status not bound or changed after freezing")
        status = read(status_path)
        if (status.get("seed") != row["seed"] or status.get("selected_epoch") != row["selected_epoch"]
                or status.get("selected_sha256") != row["sha256"] or status.get("fingerprint") != tp["fingerprint"]):
            raise RuntimeError("Training status/checkpoint linkage mismatch")
        ck = load_hashed(E, row["path"], row["sha256"])
        if ck["seed"] != row["seed"] or ck["epoch"] != row["selected_epoch"] or ck["fingerprint"] != tp["fingerprint"]:
            raise RuntimeError("Training checkpoint identity mismatch")
        base, base_record = E.base_network(int(row["seed"]))
        if ck["base_sha256"] != row["base_sha256"] or ck["base_sha256"] != base_record["sha256"]:
            raise RuntimeError("Training head attached to a different frozen marginal network")
        train[int(row["seed"])] = (row, ck, base.state_dict())
    records = []
    for row in fields["runs"]:
        selected = load_hashed(E, row["path"], row["sha256"])
        last = load_hashed(E, row["last_path"], row["last_sha256"])
        seed, arm = int(row["seed"]), row["arm"]
        ts = TRAIN_SEEDS[SEEDS.index(seed)]
        tr, tc, base_state = train[ts]
        if (selected["iteration"] != row["selected_iteration"] or last["iteration"] != 200
                or last["network_steps"] != 0 or last["trace"][-1]["iteration"] != 200
                or selected["seed"] != seed or last["seed"] != seed
                or abs(float(selected["U"])-dict(CASES)[row["case"]]) > 1e-14):
            raise RuntimeError("Field identity/iteration mismatch")
        same_state(last["selected"]["model_state"], selected["model_state"], "embedded primary field")
        for ck in (selected, last, last["selected"]):
            if (ck["arm"] != arm or ck["fingerprint"] != protocol["fingerprint"]
                    or ck["training_seed"] != ts or ck["training_sha"] != tr["sha256"]):
                raise RuntimeError("Field/head linkage mismatch")
            head = {k.removeprefix("head."): v for k, v in ck["net_state"].items() if k.startswith("head.")}
            same_state(head, tc["state"], "trained/frozen transition head")
            base = {k.removeprefix("base."): v for k, v in ck["net_state"].items() if k.startswith("base.")}
            same_state(base, base_state, "unchanged independent marginal network")
            if set(ck["net_state"]) != {"base."+k for k in base_state} | {"head."+k for k in tc["state"]}:
                raise RuntimeError("Unexpected network parameter or buffer")
        if arm == "iid":
            source = selected["reuse_source"]
            if source != last["reuse_source"] or source["arm"] != "independent" or source["case"] != row["case"] or int(source["seed"]) != seed:
                raise RuntimeError("iid reuse provenance changed")
            old_selected = load_hashed(E, source["path"], source["sha256"])
            old_last = load_hashed(E, source["last_path"], source["last_sha256"])
            same_state(selected["model_state"], old_selected["model_state"], "reused iid primary field")
            same_state(last["model_state"], old_last["model_state"], "reused iid terminal field")
            if last["trace"] != old_last["trace"] or selected["iteration"] != old_selected["iteration"]:
                raise RuntimeError("iid historical trace or selection changed")
            for key in ("a", "eta_s", "eta_n"):
                if not torch.equal(selected[key], old_selected[key]):
                    raise RuntimeError("iid historical primary coefficients changed")
        records.append(dict(row, training_seed=ts, training_epoch=int(tc["epoch"]),
                            transition_training_sha256=tr["sha256"],
                            cpu_seconds=0. if arm == "iid" else float(last["cpu_seconds"]),
                            wall_seconds=0. if arm == "iid" else float(last["wall_seconds"]),
                            historical_cpu_seconds=float(last["cpu_seconds"]) if arm == "iid" else None,
                            historical_wall_seconds=float(last["wall_seconds"]) if arm == "iid" else None))
    manifest = dict(version=VERSION, protocol_fingerprint=protocol["fingerprint"],
        training_protocol_fingerprint=tp["fingerprint"],
        training_gate_sha256=sha(out / "training/selection_frozen.json"),
        field_gate_sha256=sha(out / "selection_frozen.json"), evaluator_sha256=sha(__file__),
        helper_sha256=sha(HELPER), training_runs=training["runs"], runs=records,
        all_selected_and_terminal_checkpoints_bound_before_target_evaluation=True,
        target_H_scoring_or_reference_reads_before_global_gate=False, primary_degree=20,
        sensitivity_degree=28, terminal_iteration=200, representative_seed=REPRESENTATIVE_SEED)
    path = out / "evaluation_manifest.json"
    if path.exists() and read(path) != manifest:
        raise RuntimeError("Existing evaluation manifest differs")
    if not path.exists():
        dump(path, manifest)
    return protocol, tp, records, sha(path)


def check_network(net, ck, arm):
    if any(p.requires_grad for p in net.parameters()):
        raise RuntimeError("A target probability parameter is trainable")
    same_state(net.state_dict(), ck["net_state"], "target network " + arm)


def metrics(row, degree=20):
    h = row["observations"]["H"] if degree == 20 else row["heldout_degree28"]
    values = dict(normal_rms_mm_s=row["field"]["rms_mm_s"][0], superfluid_rms_mm_s=row["field"]["rms_mm_s"][1],
                  heldout_total_nll=h["total"])
    for kind in ("free", "cap"):
        values.update({f"heldout_{kind}_nll": h[kind], f"heldout_{kind}_coverage": h["coverage"][kind],
                       f"heldout_{kind}_coverage_abs_error": abs(h["coverage"][kind]-.9),
                       f"heldout_{kind}_spread_um": h["spread"][kind]})
    return values


def descriptive(values):
    return dict(mean=float(np.mean(values)), min=float(np.min(values)), max=float(np.max(values)),
                by_seed=dict(zip(map(str, SEEDS), map(float, values))))


def summarize(rows, degree=20):
    index = {(r["case"], int(r["seed"]), r["arm"]): metrics(r, degree) for r in rows}
    if len(rows) != 18 or set(index) != {(c, s, a) for c, _ in CASES for s in SEEDS for a in ARMS}:
        raise RuntimeError("Each evaluation mode must have exactly eighteen rows")
    result = {}
    core = ("normal_rms_mm_s", "superfluid_rms_mm_s", "heldout_free_nll", "heldout_cap_nll")
    for case, U in CASES:
        item = dict(U=U, degree=degree, arms={}, paired_changes={})
        for arm in ARMS:
            item["arms"][arm] = {k: descriptive([index[(case, s, arm)][k] for s in SEEDS]) for k in index[(case, SEEDS[0], arm)]}
        for first, second in PAIRS:
            change = {k: descriptive([index[(case, s, first)][k]-index[(case, s, second)][k] for s in SEEDS])
                      for k in index[(case, SEEDS[0], first)]}
            joint = [all(index[(case, s, first)][k] < index[(case, s, second)][k] for k in core) for s in SEEDS]
            item["paired_changes"][first + "_minus_" + second] = dict(metrics=change,
                joint_four_metric_improvement_count=sum(joint), all_four_mean_deltas_negative=all(change[k]["mean"] < 0 for k in core))
        result[case] = item
    return result


def flatten(score):
    values = {f"{k}_nll": float(score[k]) for k in KEYS}
    for kind in ("free", "cap"):
        values.update({f"{kind}_coverage": float(score["coverage"][kind]), f"{kind}_spread_um": float(score["spread"][kind])})
    return values


def attribution(E, J, rows, out, binding):
    index = {(r["mode"], r["case"], r["seed"], r["arm"]): r for r in rows}
    result = []
    for case, U in CASES:
        for seed in SEEDS:
            iid = index[("primary", case, seed, "iid")]
            ck = load_hashed(E, iid["checkpoint_path"], iid["checkpoint_sha256"])
            fixed = {k: ck[k] for k in ("a", "eta_s", "eta_n")}
            for arm in NEW_ARMS:
                own = index[("primary", case, seed, arm)]
                d, _, _, net, cal, force = E.make_context(U, seed, arm)
                own_ck = load_hashed(E, own["checkpoint_path"], own["checkpoint_sha256"])
                check_network(net, own_ck, arm)
                key = dict(manifest_sha256=binding, iid_checkpoint_sha256=iid["checkpoint_sha256"], own_checkpoint_sha256=own["checkpoint_sha256"])
                path = out / "auxiliary" / case / f"seed_{seed}" / f"{arm}.json"
                item = read(path) if path.exists() else None
                if item is None or item.get("binding") != key:
                    bd = {}
                    for degree in DEGREES:
                        old = iid["observations"]["H"] if degree == 20 else iid["heldout_degree28"]
                        new = own["observations"]["H"] if degree == 20 else own["heldout_degree28"]
                        cross = J.checked_score(E, fixed, d, net, cal, force, "H", degree)
                        b, x, n = map(flatten, (old, cross, new))
                        components = dict(probability_change_at_iid_field={k: x[k]-b[k] for k in b},
                            field_change_at_new_probability={k: n[k]-x[k] for k in b}, total_change={k: n[k]-b[k] for k in b})
                        for k in b:
                            if abs(components["probability_change_at_iid_field"][k]+components["field_change_at_new_probability"][k]-components["total_change"][k]) > 1e-10:
                                raise RuntimeError("Attribution telescoping mismatch")
                        bd[str(degree)] = dict(iid_probability_iid_field=old, new_probability_iid_field=cross,
                                               new_probability_own_field=new, components=components)
                    item = dict(case=case, seed=seed, arm=arm, binding=key, by_degree=bd,
                        own_minus_iid_field_rms_mm_s=(np.asarray(own["field"]["rms_mm_s"])-iid["field"]["rms_mm_s"]).tolist())
                    dump(path, item)
                result.append(item)
                print("ATTRIBUTION", case, seed, arm, flush=True)
    if len(result) != 12:
        raise RuntimeError("Incomplete auxiliary comparisons")
    return dict(complete=True, fixed_mode="primary", rows=result,
                descriptive_path_dependent_not_causal=True, fitting_or_selection=False)


def plot_save(fig, path, plt, artifacts):
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        p = path.with_suffix(suffix)
        fig.savefig(p, dpi=160, bbox_inches="tight")
        artifacts.append(dict(path=str(p.resolve()), sha256=sha(p), kind="figure"))
    plt.close(fig)


def plots(E, J, result, out):
    plt, artifacts = J.prepare_plotting(), []
    rows = result["runs"]
    index = {(r["mode"], r["case"], r["seed"], r["arm"]): r for r in rows}
    keys = ("normal_rms_mm_s", "superfluid_rms_mm_s", "heldout_free_nll", "heldout_cap_nll")
    for mode in MODES:
        fig, axs = plt.subplots(2, 4, figsize=(15, 6.2), layout="constrained")
        for ri, (case, U) in enumerate(CASES):
            for ci, key in enumerate(keys):
                ax = axs[ri, ci]
                for ai, arm in enumerate(ARMS):
                    values = [metrics(index[(mode, case, s, arm)])[key] for s in SEEDS]
                    ax.bar(ai, np.mean(values), color=COLORS[arm], alpha=.6)
                    ax.scatter(np.arange(3)*.07+ai-.07, values, color="black", s=17, zorder=3)
                ax.set_xticks(range(3), ARMS)
                ax.set_title(f"{U*1000:g} mm/s | {key.replace('_', ' ')}", fontsize=9)
                ax.grid(axis="y", alpha=.2)
        fig.suptitle(f"Latent temporal transition | {mode} | three paired seeds")
        fig.supxlabel("RMS: mm/s. H: causal one-step conditional predictions, current endpoint scored before filtering. All seeds use the same physical data.", fontsize=8)
        plot_save(fig, out / "figures" / mode / "overview", plt, artifacts)
    for case, U in CASES:
        source = result["reference_sources"][case]
        ref = load_hashed(E, source["path"], source["sha256"])["a"].double().cpu()
        velocities = {}
        for mode in MODES:
            for arm in ARMS:
                row = index[(mode, case, REPRESENTATIVE_SEED, arm)]
                d, _, model, _, _, _ = E.make_context(U, REPRESENTATIVE_SEED, arm)
                ck = load_hashed(E, row["checkpoint_path"], row["checkpoint_sha256"])
                model.load_state_dict(ck["model_state"], strict=True)
                a, es, en = [x.detach().cpu() for x in model()]
                x, value, means = J.resolve_fields(a, d["c"])
                _, refv, _ = J.resolve_fields(ref, d["c"])
                rms = np.sqrt(np.square((value-refv)*1000).sum(-1).mean((0, 2, 3)))
                delta = float(np.max(np.abs(rms-np.asarray(row["field"]["rms_mm_s"]))))
                if delta > 1e-10:
                    raise RuntimeError("Exported grid RMS differs from reported coefficient RMS")
                velocities[(mode, arm)] = value
                p = out / "fields" / mode / case / f"seed_{REPRESENTATIVE_SEED}_{arm}.npz"
                p.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(p, time_s=np.arange(5, 101)*.001, x_m=x, y_m=x, a=a.numpy(), eta_s=es.numpy(),
                    eta_n=en.numpy(), velocity_m_s=value, velocity_error_m_s=value-refv, means_m_s=means,
                    checkpoint_sha256=row["checkpoint_sha256"], reference_sha256=source["sha256"],
                    evaluation_manifest_sha256=result["manifest_sha256"], iteration=row["iteration"], arm=arm, mode=mode)
                with np.load(p) as saved:
                    if saved["velocity_m_s"].shape != (96, 2, 40, 40, 2) or not np.array_equal(saved["velocity_m_s"], value):
                        raise RuntimeError("96-frame NPZ readback mismatch")
                artifacts.append(dict(path=str(p.resolve()), sha256=sha(p), kind="field_npz", frames=96,
                    case=case, mode=mode, arm=arm, grid_vs_summary_rms_max_mm_s=delta, npz_readback_exact=True))
        p = out / "fields" / case / "reference_K12.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, time_s=np.arange(5, 101)*.001, x_m=x, y_m=x, velocity_m_s=refv, a=ref.numpy(), reference_sha256=source["sha256"])
        artifacts.append(dict(path=str(p.resolve()), sha256=sha(p), kind="reference_npz", frames=96))
        extent = (0, d["c"]["physics"]["box_m"]*1000, 0, d["c"]["physics"]["box_m"]*1000)
        for mode in MODES:
            for fi, fluid in enumerate(("normal", "superfluid")):
                for kind in ("speed", "error"):
                    labels = ("reference",)+ARMS if kind == "speed" else ARMS
                    images = {}
                    for arm in labels:
                        value = refv if arm == "reference" else velocities[(mode, arm)]
                        if kind == "error":
                            value = value-refv
                        images[arm] = np.linalg.norm(value[:, fi], axis=-1)*1000
                    maximum = max(float(v.max()) for v in images.values()) or 1.
                    fig, axs = plt.subplots(3, len(labels), figsize=(13 if kind == "speed" else 10, 9), layout="constrained")
                    for ri, frame in enumerate((0, 45, 95)):
                        for ci, arm in enumerate(labels):
                            ax = axs[ri, ci]
                            im = ax.imshow(images[arm][frame], origin="lower", extent=extent, vmin=0, vmax=maximum,
                                           cmap="viridis" if kind == "speed" else "magma")
                            if ri == 0:
                                title = "K12 reference" if arm == "reference" else f"{arm} | step {index[(mode, case, REPRESENTATIVE_SEED, arm)]['iteration']}"
                                ax.set_title(title)
                            if ci == 0:
                                ax.set_ylabel(f"{frame+5} ms\ny [mm]")
                            if ri == 2:
                                ax.set_xlabel("x [mm]")
                    fig.colorbar(im, ax=axs, shrink=.8, label="Speed [mm/s]" if kind == "speed" else "Vector error norm [mm/s]")
                    fig.suptitle(f"{U*1000:g} mm/s | {fluid} | {mode} | fixed seed {REPRESENTATIVE_SEED}")
                    fig.supxlabel("Common color scale across all shown arms and times; known fluid means included. Primary and fixed step-200 results remain separate.", fontsize=8)
                    plot_save(fig, out / "figures" / mode / f"{case}_{fluid}_{kind}", plt, artifacts)
    manifest = dict(complete=True, evaluation_manifest_sha256=result["manifest_sha256"],
        representative_seed=REPRESENTATIVE_SEED, representative_training_seed=TRAIN_SEEDS[0],
        all_arms_modes_and_96_frames_retained=True, artifacts=artifacts)
    dump(out / "PLOT_MANIFEST.json", manifest)
    return manifest


def write_report(result, out):
    lines = ["捕获态隐变量时序约束：第一轮完整对照结果", "",
        "固定 νs=0、Fns系数1、Frozen B 初始场/过程增量先验及原有粒子观测动力学。",
        "iid：复用上一轮 independent 的六个冻结反演；reset/temporal：分别以同一新转移头重做六个反演。",
        "三颗转移头在非目标训练数据上训练并由非目标V选择；旧 independent 的自由/捕获边缘网络始终冻结。",
        "reset 每一步从旧边缘先验开始；temporal 采用过去已观测位移更新后的状态，并在连续捕获片段内传播。",
        "temporal 先预测并评分当前位移，之后才将该位移用于后验更新；不使用当前/未来响应预测它自身。",
        "本轮为离散递推加近似 assumed-density filtering；没有引入连续时空PINN、显式自由粒子隐路径优化或完整非线性平滑。",
        "同一位置观测只进入一次端点似然；保留原物理传播/观测噪声，未另叠加位置MSE。",
        "所有三网络与十八反演主选/200步末态经过全局gate和SHA检查后才评分H或读取reference。",
        "主分析保持预先V保护规则；次要分析固定200步，不因H/场误差替换主选结果。",
        "RMS单位mm/s，5–100ms、N40/K12参考；NLL越低越好，coverage以90%为目标，spread为位移协方差迹平方根(μm)。",
        "H为同窗连续PID的因果一步条件预测，允许该H粒子此前已观测端点作滤波；不是整段无观测自由滚动。",
        "三种子使用相同物理数据；训练shuffle与初场扰动配对，不是三次独立物理实验，也不是全新盲测工况。", ""]
    for mode in MODES:
        lines.extend([mode, "case | arm | normal RMS | super RMS | H free NLL | H cap NLL | free/cap coverage | free/cap spread"])
        for case, _ in CASES:
            item = result["summary"][mode]["20"][case]
            for arm in ARMS:
                v = lambda k: item["arms"][arm][k]["mean"]
                lines.append(f"{case} | {arm} | {v('normal_rms_mm_s'):.8g} | {v('superfluid_rms_mm_s'):.8g} | "
                    f"{v('heldout_free_nll'):.8g} | {v('heldout_cap_nll'):.8g} | {v('heldout_free_coverage'):.6f}/{v('heldout_cap_coverage'):.6f} | "
                    f"{v('heldout_free_spread_um'):.8g}/{v('heldout_cap_spread_um'):.8g}")
            for pair, change in item["paired_changes"].items():
                core = ("normal_rms_mm_s", "superfluid_rms_mm_s", "heldout_free_nll", "heldout_cap_nll")
                lines.append(pair + ": " + "; ".join(f"{k}={change['metrics'][k]['mean']:+.7g} [{change['metrics'][k]['min']:+.7g},{change['metrics'][k]['max']:+.7g}]" for k in core)
                    + f"; 四指标同时下降={change['joint_four_metric_improvement_count']}/3")
        lines.append("")
    lines.extend(["主选结果的直接结论（均值与配对结果，不作显著性推断）："])
    for case, _ in CASES:
        item = result["summary"]["primary"]["20"][case]
        baseline = item["arms"]["reset"]
        change = item["paired_changes"]["temporal_minus_reset"]
        lines.append(case + "，temporal相对reset：" + "; ".join(
            f"{name} {100*change['metrics'][key]['mean']/baseline[key]['mean']:+.4f}%"
            for key, name in (("normal_rms_mm_s", "常流RMS"), ("superfluid_rms_mm_s", "超流RMS"))) + "; " + "; ".join(
            f"{name} {change['metrics'][key]['mean']:+.8g}"
            for key, name in (("heldout_free_nll", "自由H NLL"), ("heldout_cap_nll", "捕获H NLL")))
            + f"；两场误差和两态NLL同降={change['joint_four_metric_improvement_count']}/3种子。")
        lines.append("该工况四个均值" + ("均下降；仍需结合覆盖率、积分敏感性和样本限制。" if change['all_four_mean_deltas_negative']
                     else "未同时下降，不能声称时序隐变量约束在该工况获得全面改善。"))
    lines.extend(["", "固定200步相对主选的差异："])
    for case, _ in CASES:
        for arm in ARMS:
            item = result["terminal_minus_primary"][case][arm]
            core = ("normal_rms_mm_s", "superfluid_rms_mm_s", "heldout_free_nll", "heldout_cap_nll")
            lines.append(f"{case} {arm}: " + "; ".join(f"{k}={item[k]['mean']:+.8g}" for k in core))
    lines.extend(["", "积分敏感性：20阶为主，28阶只作敏感性分析；以下报告所有NLL对比均值方向。"])
    for row in result["quadrature_directions"]:
        lines.append(f"{row['mode']} {row['case']} {row['pair']} {row['metric']}: Δ20={row['delta20']:+.8g}, Δ28={row['delta28']:+.8g}, 方向反转={row['sign_reversal']}")
    lines.extend(["", "固定iid主选场辅助归因：先在同一旧场换概率，再固定新概率换成新场。两项可加，但路径相关，不是因果贡献率。"])
    for case, _ in CASES:
        for arm in NEW_ARMS:
            rr = [r for r in result["auxiliary"]["rows"] if r["case"] == case and r["arm"] == arm]
            for kind in ("free_nll", "cap_nll"):
                cs = ("probability_change_at_iid_field", "field_change_at_new_probability", "total_change")
                lines.append(f"{case} {arm} {kind}: " + "; ".join(f"{c}={np.mean([r['by_degree']['20']['components'][c][kind] for r in rr]):+.8g}" for c in cs))
    lines.extend(["", "成本与限制："])
    for row in result["cost"]["training_runs"]:
        lines.append(f"训练seed {row['seed']}：选中epoch={row['selected_epoch']}，status={json.dumps(row['status'], ensure_ascii=False)}")
    for arm in ARMS:
        rr = [r for r in result["cost"]["field_runs"] if r["arm"] == arm]
        lines.append(f"{arm}: 本轮6反演累计wall={sum(r['wall_seconds'] for r in rr)/60:.3f}min，CPU={sum(r['cpu_seconds'] for r in rr)/60:.3f}min" + (f"；历史复用wall={sum(r['historical_wall_seconds'] for r in rr)/60:.3f}min（本轮未重跑）" if arm == "iid" else ""))
    lines.extend(["assumed-density filtering将每个q的连续隐状态后验压缩为高斯，丢失高阶/多峰信息；20/28阶变化不能穷尽该近似误差。",
        "历史网络和动力学训练标签来自有限分辨率参考；候选场上下文漂移及场/隐变量补偿不可辨识仍可能存在。",
        "新头训练收益、V收益或H概率收益均不能直接当作场改善。reset−iid同时包含转移头带来的分布改变；temporal−reset更接近历史后验传播的增量作用。",
        "没有按本轮H结果改选解、调参或启动CF。主/末态差、逐种子、逐PID、哈希、敏感性和辅助结果保存在JSON中。", ""])
    text = "\n".join(lines)
    (out / "LATENT_EVALUATION_REPORT.txt").write_text(text, encoding="utf8")
    (out / "CONCLUSIONS.txt").write_text(text, encoding="utf8")


@torch.no_grad()
def evaluate(out=OUT, make_plots=True):
    begun = time.perf_counter()
    out = Path(out).resolve()
    training, fields = cheap_gate(out)
    sys.path.insert(0, str(HERE))
    E = module("latent_evaluation_experiment", HERE / "experiment_latent.py")
    T = module("latent_evaluation_training", HERE / "train_latent.py")
    J = module("latent_evaluation_helpers", HELPER)
    if Path(E.OUT).resolve() != out:
        raise RuntimeError("Output differs from experiment.OUT")
    E.M.R.setup("cpu")
    protocol, tp, records, binding = bind_gates(E, T, out, training, fields)
    # All network and field bytes and identities are verified before this line.
    rows, references = [], {}
    for case, U in CASES:
        rp = J.reference_path(case)
        digest = sha(rp)
        reference = load_hashed(E, rp, digest)["a"].double().cpu()
        references[case] = dict(path=str(rp), sha256=digest)
        for record in [r for r in records if r["case"] == case]:
            seed, arm = int(record["seed"]), record["arm"]
            d, _, model, net, cal, force = E.make_context(U, seed, arm)
            for mode in MODES:
                path = record["path"] if mode == "primary" else record["last_path"]
                digest_ck = record["sha256"] if mode == "primary" else record["last_sha256"]
                ck = load_hashed(E, path, digest_ck)
                check_network(net, ck, arm)
                model.load_state_dict(ck["model_state"], strict=True)
                a, es, en = model()
                if not all(torch.isfinite(x).all() for x in (a, es, en)):
                    raise RuntimeError("Nonfinite replayed field")
                if mode == "primary" and not all(torch.equal(x, ck[k]) for x, k in zip((a, es, en), ("a", "eta_s", "eta_n"))):
                    raise RuntimeError("Primary coefficients disagree with replay")
                validation = E.score(a, es, en, d, net, cal, force, "V", degree=20, metrics=False)
                saved = ck["validation_components"] if mode == "primary" else ck["trace"][-1]["validation"]
                delta = max(abs(validation[k]-saved[k]) for k in KEYS)
                if not np.isfinite(delta) or delta > 1e-9:
                    raise RuntimeError(f"Replayed V differs ({delta}) for {case}/{seed}/{arm}/{mode}")
                key = dict(manifest_sha256=binding, checkpoint_sha256=digest_ck, reference_sha256=digest)
                cache = out / "evaluation" / mode / case / f"seed_{seed}" / f"{arm}.json"
                row = read(cache) if cache.exists() else None
                if row is None or row.get("cache_key") != key:
                    replay = dict(a=a, eta_s=es, eta_n=en)
                    observations = {g: E.score(a, es, en, d, net, cal, force, g, degree=20, metrics=False) for g in ("I", "V")}
                    observations["H"] = J.checked_score(E, replay, d, net, cal, force, "H", 20)
                    higher = J.checked_score(E, replay, d, net, cal, force, "H", 28)
                    row = dict(case=case, U=U, seed=seed, arm=arm, mode=mode, iteration=int(ck["iteration"]),
                        primary_selected_iteration=record["selected_iteration"], checkpoint_path=str(path), checkpoint_sha256=digest_ck,
                        training_seed=record["training_seed"], training_epoch=record["training_epoch"],
                        transition_training_sha256=record["transition_training_sha256"], target_network_updates=0,
                        reference_path=str(rp), reference_sha256=digest, replay_V_max_difference=delta, cache_key=key,
                        field=J.field_metrics(a, reference, U), observations=observations, heldout_degree28=higher,
                        quadrature=J.quadrature_difference(observations["H"], higher))
                    dump(cache, row)
                rows.append(row)
                print("EVALUATED", mode, case, seed, arm, "step", ck["iteration"], flush=True)
    if len(rows) != 36:
        raise RuntimeError("Both complete modes must have 36 rows total")
    summaries = {m: {str(degree): summarize([r for r in rows if r["mode"] == m], degree) for degree in DEGREES} for m in MODES}
    index = {(r["mode"], r["case"], r["seed"], r["arm"]): metrics(r) for r in rows}
    changes = {c: {a: {k: descriptive([index[("terminal", c, s, a)][k]-index[("primary", c, s, a)][k] for s in SEEDS])
                       for k in index[("primary", c, SEEDS[0], a)]} for a in ARMS} for c, _ in CASES}
    directions = []
    for mode in MODES:
        for case, _ in CASES:
            for first, second in PAIRS:
                pair = first+"_minus_"+second
                for metric in ("heldout_free_nll", "heldout_cap_nll"):
                    v20, v28 = [summaries[mode][str(d)][case]["paired_changes"][pair]["metrics"][metric]["mean"] for d in DEGREES]
                    directions.append(dict(mode=mode, case=case, pair=pair, metric=metric, delta20=v20, delta28=v28, sign_reversal=v20*v28 < 0))
    auxiliary = attribution(E, J, rows, out, binding)
    costs = []
    for r in training["runs"]:
        ck = load_hashed(E, r["path"], r["sha256"])
        costs.append(dict(seed=r["seed"], selected_epoch=r["selected_epoch"],
            transition_head_parameter_count=sum(v.numel() for v in ck["state"].values()),
            status=read(r.get("status_path", str(Path(r["path"]).parent/"status.json")))))
    result = dict(version=VERSION, complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.perf_counter()-begun, protocol_fingerprint=protocol["fingerprint"], manifest_sha256=binding,
        reference_sources=references, selected_or_trained_during_evaluation=False, terminal_preplanned=True,
        target_H_or_reference_before_global_gate=False, runs=rows, summary=summaries,
        terminal_minus_primary=changes, quadrature_directions=directions, auxiliary=auxiliary,
        cost=dict(training_runs=costs, field_runs=records), prediction_scope="Causal one-step same-window endpoint prediction, previous observed endpoints allowed; no future conditioning")
    dump(out / "tables/individual_runs.json", rows)
    dump(out / "AUXILIARY_PROBABILITY_ATTRIBUTION.json", auxiliary)
    dump(out / "LATENT_EVALUATION_SUMMARY.json", result)
    write_report(result, out)
    if make_plots:
        plots(E, J, result, out)
    print("EVALUATION COMPLETE", out / "LATENT_EVALUATION_REPORT.txt", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check-gate-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.check_gate_only:
        cheap_gate(args.out.resolve())
        print("Metadata gates complete; no target tensors or checkpoints opened")
    else:
        evaluate(args.out, not args.no_plots)
