"""Gate-bound evaluation of pretrained free/captured transition models.

All nine training selections and all eighteen selected/terminal field states
must be frozen before target H scoring or reference reads. The historical
new_frozen arm is reused with its checkpoint and evaluation provenance intact.
This program performs no training, selection, or changes to old modules.
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
BASE = HERE.parents[2]
OUT = BASE / "outputs/SharedTransition_20261104"
OLD = BASE / "outputs/JointProbabilityPhysics_20261103"
HELPER = HERE.parent / "joint_probability_20261103/evaluate_joint.py"
CASES = (("vn0_0015m_s", .0015), ("vn0_004m_s", .004))
SEEDS = (301030, 301031, 301032)
TRAIN_SEEDS = (411040, 411041, 411042)
NEW_ARMS = ("cap_retrain", "independent", "shared_gru")
ARMS = ("legacy",) + NEW_ARMS
MODES = ("primary", "terminal")
PAIRS = (("cap_retrain", "legacy"), ("independent", "cap_retrain"),
         ("shared_gru", "independent"), ("shared_gru", "legacy"))
VERSION = "shared-transition-evaluation-v1"
KEYS = ("total", "free", "cap")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def expected_runs():
    return [(case, U, seed, arm) for case, U in CASES for seed in SEEDS for arm in NEW_ARMS]


def checked_path(recorded, expected):
    actual, expected = Path(recorded).resolve(), Path(expected).resolve()
    if actual != expected:
        raise RuntimeError(f"Unexpected artifact path: {actual}; expected {expected}")
    return actual


def cheap_gate(out):
    """Read metadata only; incomplete studies cannot even deserialize checkpoints."""
    paths = (out / "training/selection_frozen.json", out / "selection_frozen.json")
    if not all(p.is_file() for p in paths):
        raise RuntimeError("Nine training and eighteen field selection/terminal gates are required before target evaluation")
    training, fields = [read(p) for p in paths]
    want_training = {(a, s) for a in NEW_ARMS for s in TRAIN_SEEDS}
    tr = training.get("runs", [])
    if (training.get("all_required_selections_frozen") is not True or len(tr) != 9
            or {(r["arm"], int(r["seed"])) for r in tr} != want_training):
        raise RuntimeError("Training gate must contain exactly all nine preselected networks")
    expected = {(c, s, a) for c, _, s, a in expected_runs()}
    fr = fields.get("runs", [])
    if (fields.get("all_required_selections_frozen") is not True
            or fields.get("all_terminal_states_bound_before_evaluation") is not True
            or len(fr) != 18 or {(r["case"], int(r["seed"]), r["arm"]) for r in fr} != expected):
        raise RuntimeError("Field gate must bind all eighteen primary selections and step-200 terminal states")
    for r in tr:
        folder = out / "training" / r["arm"] / f"seed_{r['seed']}"
        checked_path(r["path"], folder / "selected.pt")
        path = checked_path(r["status_path"], folder / "status.json")
        status = read(path)
        if (status.get("complete") is not True or status.get("arm") != r["arm"]
                or status.get("seed") != r["seed"] or status.get("selected_sha256") != r["sha256"]
                or status.get("protocol_fingerprint") != training["protocol_fingerprint"]):
            raise RuntimeError("Incomplete or inconsistent training status: " + str(path))
    for r in fr:
        folder = out / "runs" / r["case"] / f"seed_{r['seed']}" / r["arm"]
        checked_path(r["path"], folder / "selected.pt")
        checked_path(r["last_path"], folder / "last.pt")
        status = read(folder / "status.json")
        if (status.get("complete") is not True or status.get("iteration") != 200
                or status.get("steps") != 200 or status.get("network_steps") != 0
                or status.get("case") != r["case"] or status.get("seed") != r["seed"]
                or status.get("arm") != r["arm"] or status.get("training_sha") != r["training_sha"]
                or status.get("fingerprint") != fields["fingerprint"]
                or status.get("selected_iteration") != r["selected_iteration"]
                or status.get("training_seed") != TRAIN_SEEDS[SEEDS.index(int(r["seed"]))]):
            raise RuntimeError("Incomplete or inconsistent field status: " + str(folder))
    return training, fields


def same_state(actual, expected, label):
    if set(actual) != set(expected):
        raise RuntimeError("Different state_dict keys: " + label)
    for key, value in actual.items():
        ref = expected[key].to(device=value.device, dtype=value.dtype)
        if value.shape != ref.shape or not torch.equal(value, ref):
            raise RuntimeError("Frozen state changed: " + label + "/" + key)


def load_hashed(E, path, digest):
    if sha(path) != digest:
        raise RuntimeError("Checkpoint bytes changed: " + str(path))
    result = E.load(path)
    if sha(path) != digest:
        raise RuntimeError("Checkpoint changed during deserialization: " + str(path))
    return result


def source_bound(protocol, path):
    recorded = {str(Path(p).resolve()): h for p, h in protocol["sources"].items()}
    if recorded.get(str(Path(path).resolve())) != sha(path):
        raise RuntimeError("Evaluation source must be prospectively bound in inversion protocol: " + str(path))


def bind_gates(E, T, out, training, fields):
    """Bind every checkpoint and its network identity before any target scoring."""
    protocol, tp = E.verify_protocol(), T.verify_protocol()
    source_bound(protocol, __file__)
    source_bound(protocol, HELPER)
    if protocol.get("evaluation_degree") != 20 or protocol.get("evaluation_sensitivity_degree") != 28:
        raise RuntimeError("Required primary/sensitivity quadrature degrees are 20/28")
    tf = dict(training)
    tfp = tf.pop("fingerprint")
    if T.fingerprint(tf) != tfp or training["protocol_fingerprint"] != tp["fingerprint"]:
        raise RuntimeError("Training gate/protocol fingerprint mismatch")
    if fields["fingerprint"] != protocol["fingerprint"]:
        raise RuntimeError("Field gate belongs to a different protocol")
    if fields.get("training_gate_sha256") != sha(out / "training/selection_frozen.json"):
        raise RuntimeError("Field gate does not bind the current training gate")
    training_states = {}
    for r in training["runs"]:
        if sha(r["status_path"]) != r["status_sha256"]:
            raise RuntimeError("Training status changed after freeze")
        ck = load_hashed(E, r["path"], r["sha256"])
        if ((ck["arm"], ck["seed"], ck["epoch"], ck["protocol_fingerprint"])
                != (r["arm"], r["seed"], r["selected_epoch"], tp["fingerprint"])):
            raise RuntimeError("Training checkpoint identity mismatch")
        training_states[(r["arm"], int(r["seed"]))] = (r, ck)
    bound = []
    for r in fields["runs"]:
        seed = int(r["seed"])
        ts = TRAIN_SEEDS[SEEDS.index(seed)]
        training_record, training_ck = training_states[(r["arm"], ts)]
        if r["training_sha"] != training_record["sha256"] or r["training_seed"] != ts:
            raise RuntimeError("Field gate training seed/hash is inconsistent")
        selected = load_hashed(E, r["path"], r["sha256"])
        last = load_hashed(E, r["last_path"], r["last_sha256"])
        for ck in (selected, last, last["selected"]):
            if (ck["fingerprint"] != protocol["fingerprint"] or ck["arm"] != r["arm"]
                    or ck["seed"] != seed or abs(float(ck["U"]) - dict(CASES)[r["case"]]) > 1e-14
                    or ck["training_seed"] != ts or ck["training_sha"] != r["training_sha"]):
                raise RuntimeError("Field checkpoint protocol/identity/training linkage mismatch")
            same_state(ck["net_state"], training_ck["state"], "trained network/field " + r["arm"])
        if (last["iteration"] != 200 or last["network_steps"] != 0
                or not last["trace"] or last["trace"][-1]["iteration"] != 200
                or selected["iteration"] != r["selected_iteration"]
                or last["selected"]["iteration"] != selected["iteration"]):
            raise RuntimeError("Terminal/selected iteration mismatch")
        same_state(last["selected"]["model_state"], selected["model_state"], "embedded selected field")
        status_path = Path(r["path"]).parent / "status.json"
        bound.append(dict(r, status_sha256=sha(status_path), training_epoch=int(training_ck["epoch"]),
                          cpu_seconds=float(last["cpu_seconds"]), wall_seconds=float(last["wall_seconds"])))
    return protocol, tp, bound


def training_cost(E, training):
    """Read only gate-bound metadata and weights; count parameters separately from buffers."""
    rows = []
    for r in training["runs"]:
        status = read(r["status_path"])
        if sha(r["status_path"]) != r["status_sha256"]:
            raise RuntimeError("Training cost metadata changed after freezing")
        ck = load_hashed(E, r["path"], r["sha256"])
        parameters = {k: int(v.numel()) for k, v in ck["state"].items() if k not in ("cm", "cs")}
        rows.append(dict(r, parameter_count=sum(parameters.values()), parameter_counts=parameters,
                         last_epoch=int(status["last_epoch"]), stopping=status["stopping"],
                         wall_seconds=float(status["wall_seconds"]), converged_claim=False))
    return rows


def legacy_sources():
    return {name: OLD / name for name in (
        "protocol.json", "selection_frozen.json", "JOINT_EVALUATION_SUMMARY.json",
        "tables/individual_runs.json", "secondary_terminal_manifest.json", "secondary_terminal.json")}


def bind_legacy(E, protocol):
    """Validate historical metrics against their actual frozen primary/terminal bytes."""
    paths = legacy_sources()
    # Parent protocol records these immutable result artifacts before new fits.
    inputs = {str(Path(p).resolve()): h for p, h in protocol["inputs"].items()}
    for path in paths.values():
        if inputs.get(str(path.resolve())) != sha(path):
            raise RuntimeError("Historical reuse artifact missing/changed in input contract: " + str(path))
    old_protocol = read(paths["protocol.json"])
    gate = read(paths["selection_frozen.json"])
    summary = read(paths["JOINT_EVALUATION_SUMMARY.json"])
    primary = read(paths["tables/individual_runs.json"])
    manifest = read(paths["secondary_terminal_manifest.json"])
    terminal = read(paths["secondary_terminal.json"])
    oldfp = old_protocol["fingerprint"]
    if (gate.get("all_required_selections_frozen") is not True or gate["fingerprint"] != oldfp
            or summary.get("complete") is not True or terminal.get("complete") is not True
            or summary["fingerprint"]["protocol_fingerprint"] != oldfp
            or old_protocol.get("evaluation_degree") != 20 or old_protocol.get("evaluation_sensitivity_degree") != 28
            or manifest["protocol_fingerprint"] != oldfp
            or manifest["global_gate_sha256"] != sha(paths["selection_frozen.json"])
            or terminal["manifest_sha256"] != sha(paths["secondary_terminal_manifest.json"])):
        raise RuntimeError("Historical evaluation gate/manifest chain is broken")
    records, rows = [], []
    for case, U in CASES:
        for seed in SEEDS:
            key = (case, seed, "new_frozen")
            def one(items):
                matches = [r for r in items if (r["case"], int(r["seed"]), r["arm"]) == key]
                if len(matches) != 1:
                    raise RuntimeError("Missing/duplicate legacy record: " + str(key))
                return matches[0]
            sr, lr = one(gate["runs"]), one(manifest["runs"])
            selected = load_hashed(E, sr["path"], sr["sha256"])
            last = load_hashed(E, lr["path"], lr["sha256"])
            if (selected["fingerprint"] != oldfp or last["fingerprint"] != oldfp
                    or selected["arm"] != "new_frozen" or selected["seed"] != seed
                    or abs(float(selected["U"]) - U) > 1e-14 or last["iteration"] != 200
                    or last["network_steps"] != 0 or lr["primary_selected_sha256"] != sr["sha256"]):
                raise RuntimeError("Historical checkpoint identity mismatch")
            for mode, row in (("primary", one(primary)), ("terminal", one(terminal["runs"]))):
                reference = summary["reference_sources"][case] if mode == "primary" else terminal["reference_sources"][case]
                if mode == "primary":
                    if (row["fingerprint"]["checkpoint_sha256"] != sr["sha256"]
                            or row["fingerprint"]["reference_sha256"] != reference["sha256"]
                            or row["fingerprint"]["gate_sha256"] != sha(paths["selection_frozen.json"])
                            or row["selected_iteration"] != selected["iteration"]):
                        raise RuntimeError("Legacy primary row provenance mismatch")
                    iteration, ckpath, ckh = int(selected["iteration"]), sr["path"], sr["sha256"]
                else:
                    if (row["last_checkpoint_sha256"] != lr["sha256"]
                            or row["manifest_sha256"] != sha(paths["secondary_terminal_manifest.json"])
                            or row["reference_sha256"] != reference["sha256"]):
                        raise RuntimeError("Legacy terminal row provenance mismatch")
                    iteration, ckpath, ckh = 200, lr["path"], lr["sha256"]
                normalized = dict(case=case, U=U, seed=seed, arm="legacy", mode=mode,
                    iteration=iteration, primary_selected_iteration=int(selected["iteration"]),
                    checkpoint_path=ckpath, checkpoint_sha256=ckh,
                    training_seed=None, training_epoch=None, reused_historical_arm="new_frozen",
                    reference_path=reference["path"], reference_sha256=reference["sha256"],
                    field=row["field"], observations=row["observations"], heldout_degree28=row["heldout_degree28"],
                    quadrature=row["quadrature"], legacy_source_sha256=sha(paths["tables/individual_runs.json"] if mode == "primary" else paths["secondary_terminal.json"]))
                rows.append(normalized)
            records.append(dict(case=case, U=U, seed=seed, arm="legacy", path=sr["path"], sha256=sr["sha256"],
                last_path=lr["path"], last_sha256=lr["sha256"], selected_iteration=int(selected["iteration"]),
                cpu_seconds=float(last["cpu_seconds"]), wall_seconds=float(last["wall_seconds"])))
    return records, rows, {str(path): sha(path) for path in paths.values()}


def bind_manifest(out, protocol, tp, training, records, legacy, legacy_artifacts):
    value = dict(version=VERSION, protocol_fingerprint=protocol["fingerprint"],
        training_protocol_fingerprint=tp["fingerprint"],
        training_gate_sha256=sha(out / "training/selection_frozen.json"),
        field_gate_sha256=sha(out / "selection_frozen.json"), evaluator_sha256=sha(__file__),
        helper_sha256=sha(HELPER), training_runs=training["runs"], runs=records, legacy=legacy,
        legacy_artifacts=legacy_artifacts, primary_degree=20, sensitivity_degree=28,
        terminal_iteration=200, all_new_training_selected_and_terminal_states_bound_before_target_evaluation=True,
        binding_scope="This new comparison only; historical target data and historical H evaluations already existed.",
        H_scoring_or_reference_reads_by_this_evaluator_before_binding=False)
    path = out / "evaluation_manifest.json"
    if path.exists():
        if read(path) != value:
            raise RuntimeError("Existing evaluation manifest differs; refusing to mix analyses")
    else:
        dump(path, value)
    return value, sha(path)


def metrics(row, degree=20):
    h = row["observations"]["H"] if degree == 20 else row["heldout_degree28"]
    values = dict(normal_rms_mm_s=row["field"]["rms_mm_s"][0],
                  superfluid_rms_mm_s=row["field"]["rms_mm_s"][1], heldout_total_nll=h["total"])
    for kind in ("free", "cap"):
        values.update({f"heldout_{kind}_nll": h[kind], f"heldout_{kind}_coverage": h["coverage"][kind],
                       f"heldout_{kind}_coverage_abs_error": abs(h["coverage"][kind]-.9),
                       f"heldout_{kind}_spread_um": h["spread"][kind]})
    return values


def descriptive(values):
    return dict(mean=float(np.mean(values)), min=float(np.min(values)), max=float(np.max(values)),
                by_seed=dict(zip(map(str, SEEDS), map(float, values))))


def summarize(rows, degree=20):
    index = {(r["case"], r["seed"], r["arm"]): metrics(r, degree) for r in rows}
    expected = {(c, s, a) for c, _ in CASES for s in SEEDS for a in ARMS}
    if set(index) != expected or len(rows) != 24:
        raise RuntimeError("Every analysis mode must contain exactly 24 rows, including legacy")
    result = {}
    for case, U in CASES:
        item = dict(U=U, degree=degree, arms={}, paired_changes={})
        for arm in ARMS:
            item["arms"][arm] = {key: descriptive([index[(case, s, arm)][key] for s in SEEDS])
                                  for key in index[(case, SEEDS[0], arm)]}
        for first, second in PAIRS:
            change = {key: descriptive([index[(case, s, first)][key]-index[(case, s, second)][key] for s in SEEDS])
                      for key in index[(case, SEEDS[0], first)]}
            core = ("normal_rms_mm_s", "superfluid_rms_mm_s", "heldout_free_nll", "heldout_cap_nll")
            joint = [all(index[(case, s, first)][k] < index[(case, s, second)][k] for k in core) for s in SEEDS]
            item["paired_changes"][first + "_minus_" + second] = dict(metrics=change,
                joint_four_metric_improvements_by_seed=dict(zip(map(str, SEEDS), joint)),
                joint_four_metric_improvement_count=sum(joint),
                all_four_mean_deltas_negative=all(change[k]["mean"] < 0 for k in core))
        result[case] = item
    return result


def mode_changes(rows):
    index = {(r["mode"], r["case"], r["seed"], r["arm"]): metrics(r) for r in rows}
    return {case: {arm: {key: descriptive([index[("terminal", case, s, arm)][key]-index[("primary", case, s, arm)][key]
                    for s in SEEDS]) for key in index[("primary", case, SEEDS[0], arm)]}
                    for arm in ARMS} for case, _ in CASES}


def quadrature_summary(summaries):
    result = {}
    for mode in MODES:
        result[mode] = {}
        for case, _ in CASES:
            result[mode][case] = {}
            for first, second in PAIRS:
                pair = first + "_minus_" + second
                item = {}
                for key in ("heldout_free_nll", "heldout_cap_nll"):
                    d20 = summaries[mode]["20"][case]["paired_changes"][pair]["metrics"][key]["mean"]
                    d28 = summaries[mode]["28"][case]["paired_changes"][pair]["metrics"][key]["mean"]
                    item[key] = dict(mean_delta_degree20=d20, mean_delta_degree28=d28,
                                     sign_reversal=d20*d28 < 0)
                result[mode][case][pair] = item
    return result


def write_report(result, out):
    lines = ["共享／独立概率演化模型：冻结训练后流场反演对照", "",
        "固定 νs=0、Fns系数1、原 Frozen B 过程先验；未重标定过程先验、未重生成模拟原始数据。",
        "legacy复用上一轮new_frozen六个反演；cap_retrain、independent、shared_gru各训练三种子并冻结后从第0步用于目标反演。",
        "主分析为V保护规则选解；次要分析为预先固定全部第200步终态。两者完整报告，不能择优替换。",
        "9个训练选解及18个新反演的主/末态通过完整gate与SHA核验后才读取本轮reference和评分H。",
        "常/超流RMS单位mm/s，时间窗5–100ms；已知均流相同，使用N40/K12匹配带宽参考。",
        "H是同窗口、给定观测历史的原生1ms条件位移预测；不是未来自由滚动轨迹。",
        "NLL降低较好；coverage应接近名义90%，同时检查spread；spread为位移总协方差迹平方根，单位μm。",
        "三种子将训练shuffle与场初值配对；同一数据与划分，不是三次独立物理实验；不计算显著性或置信区间。",
        "cap_retrain−legacy为重训效应；independent−cap_retrain为新增free分支的整个管线效应；shared_gru−independent为共享方式增量。",
        "cap选解epoch可能不同，因此新增free对比不是固定同一cap权重的纯替换；参数容量和训练成本随arm分别报告。", ""]
    for mode in MODES:
        lines.extend([mode, "case | arm | normal RMS | super RMS | H free NLL | H cap NLL | free/cap coverage | free/cap spread"])
        for case, _ in CASES:
            summary = result["summary"][mode]["20"][case]
            for arm in ARMS:
                values = summary["arms"][arm]
                v = lambda key: values[key]["mean"]
                lines.append(f"{case} | {arm} | {v('normal_rms_mm_s'):.8g} | {v('superfluid_rms_mm_s'):.8g} | "
                    f"{v('heldout_free_nll'):.8g} | {v('heldout_cap_nll'):.8g} | "
                    f"{v('heldout_free_coverage'):.6f}/{v('heldout_cap_coverage'):.6f} | "
                    f"{v('heldout_free_spread_um'):.8g}/{v('heldout_cap_spread_um'):.8g}")
            for pair, item in summary["paired_changes"].items():
                pieces = [f"{k}={item['metrics'][k]['mean']:+.7g} [{item['metrics'][k]['min']:+.7g},{item['metrics'][k]['max']:+.7g}]"
                          for k in ("normal_rms_mm_s", "superfluid_rms_mm_s", "heldout_free_nll", "heldout_cap_nll")]
                lines.append(pair + ": " + "; ".join(pieces) +
                    f"; 两场RMS及两态NLL同时下降={item['joint_four_metric_improvement_count']}/3配对种子")
        lines.append("")
    lines.extend(["degree20为主，degree28仅敏感性检查；报告所有配对均值方向是否反转，不事后设阈值。",
        "目标工况在历史研究中已被探索；本轮是开发对照，不声称全新盲测或确认性推广证据。",
        "完整JSON同时记录逐种子、逐PID的H诊断、训练选中epoch、运行成本、主/末态差及全部哈希链。", ""])
    (out / "TRANSITION_EVALUATION_REPORT.txt").write_text("\n".join(lines), encoding="utf-8")


@torch.no_grad()
def evaluate(out=OUT):
    begun = time.perf_counter()
    out = Path(out).resolve()
    training, field_gate = cheap_gate(out)
    sys.path.insert(0, str(HERE))
    E = module_from_file("transition_evaluation_experiment", HERE / "experiment.py")
    T = module_from_file("transition_evaluation_training", HERE / "training.py")
    J = module_from_file("transition_evaluation_legacy_helpers", HELPER)
    if out != Path(E.OUT).resolve():
        raise RuntimeError("Evaluation output must match experiment.OUT")
    E.M.R.setup("cpu")
    protocol, tp, records = bind_gates(E, T, out, training, field_gate)
    legacy, rows, legacy_artifacts = bind_legacy(E, protocol)
    manifest, binding = bind_manifest(out, protocol, tp, training, records, legacy, legacy_artifacts)
    # No target reference read or target H scoring occurs above this boundary.
    references = {}
    for case, U in CASES:
        rp = J.reference_path(case)
        digest = sha(rp)
        reference = E.load(rp)["a"].double().cpu()
        if sha(rp) != digest:
            raise RuntimeError("Reference changed during deserialization")
        references[case] = dict(path=str(rp), sha256=digest)
        for row in rows:
            if row["case"] == case and (row["reference_sha256"] != digest or Path(row["reference_path"]).resolve() != rp.resolve()):
                raise RuntimeError("Legacy/new reference differs; comparison is invalid")
        for record in [r for r in records if r["case"] == case]:
            seed, arm = record["seed"], record["arm"]
            d, cfg, model, net, cal, force = E.make_context(U, seed, arm)
            for mode in MODES:
                ckpath = record["path"] if mode == "primary" else record["last_path"]
                cksha = record["sha256"] if mode == "primary" else record["last_sha256"]
                ck = load_hashed(E, ckpath, cksha)
                same_state(net.state_dict(), ck["net_state"], "network context " + arm)
                model.load_state_dict(ck["model_state"], strict=True)
                a, es, en = model()
                if not all(torch.isfinite(x).all() for x in (a, es, en)):
                    raise RuntimeError("Nonfinite field replay")
                if mode == "primary" and not all(torch.equal(x, ck[k]) for x, k in zip((a, es, en), ("a", "eta_s", "eta_n"))):
                    raise RuntimeError("Primary coefficients disagree with model-state replay")
                replay = dict(a=a, eta_s=es, eta_n=en)
                saved_v = ck["validation_components"] if mode == "primary" else ck["trace"][-1]["validation"]
                val = E.score(a, es, en, d, net, cal, force, "V", degree=20, metrics=False)
                delta = max(abs(val[k]-saved_v[k]) for k in KEYS)
                if delta > 1e-10 or not np.isfinite(delta):
                    raise RuntimeError("Replayed field/network disagrees with saved V score")
                cache_key = dict(manifest_sha256=binding, checkpoint_sha256=cksha,
                    reference_path=str(rp), reference_sha256=digest)
                cache_path = out / "evaluation" / mode / case / f"seed_{seed}" / f"{arm}.json"
                row = read(cache_path) if cache_path.is_file() else None
                if row is None or row.get("cache_key") != cache_key:
                    observations = {g: E.score(a, es, en, d, net, cal, force, g, degree=20, metrics=False) for g in ("F", "I", "V")}
                    if not all(np.isfinite(v[k]) for v in observations.values() for k in KEYS):
                        raise RuntimeError("Nonfinite diagnostic score")
                    observations["H"] = J.checked_score(E, replay, d, net, cal, force, "H", 20)
                    higher = J.checked_score(E, replay, d, net, cal, force, "H", 28)
                    row = dict(case=case, U=U, seed=seed, arm=arm, mode=mode,
                        iteration=int(ck["iteration"]), primary_selected_iteration=record["selected_iteration"],
                        checkpoint_path=ckpath, checkpoint_sha256=cksha, cache_key=cache_key,
                        reference_path=str(rp), reference_sha256=digest, replay_V_max_difference=delta,
                        training_seed=record["training_seed"], training_sha256=record["training_sha"],
                        training_epoch=record["training_epoch"], target_network_updates=0,
                        field=J.field_metrics(a, reference, U), observations=observations,
                        heldout_degree28=higher, quadrature=J.quadrature_difference(observations["H"], higher))
                    dump(cache_path, row)
                rows.append(row)
                print(f"EVALUATED {mode} {case} seed={seed} arm={arm} step={ck['iteration']}", flush=True)
    if len(rows) != 48:
        raise RuntimeError("Both modes must include all 24 runs")
    summaries = {mode: {str(degree): summarize([r for r in rows if r["mode"] == mode], degree)
                       for degree in (20, 28)} for mode in MODES}
    result = dict(version=VERSION, complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.perf_counter()-begun, manifest_sha256=binding,
        protocol_fingerprint=protocol["fingerprint"], reference_sources=references,
        selected_or_trained_during_evaluation=False, terminal_preplanned=True,
        target_reference_or_H_scoring_before_complete_new_gates=False,
        seed_interpretation="Paired training shuffle and field initialization on the same physical data; not independent physical replicates",
        prediction_scope="Same-window 1ms conditional endpoint given observed particle history; not free-running future trajectories",
        units=dict(field_rms="mm/s", spread="sqrt(trace(total displacement covariance)), micrometres",
                   NLL="normalized displacement NLL per native 1ms interval", coverage="nominal 90%, 2-D Rosenblatt threshold"),
        runs=rows, summary=summaries, terminal_minus_primary=mode_changes(rows),
        quadrature_paired_direction=quadrature_summary(summaries),
        cost=dict(field_runs=records+legacy, training_runs=training_cost(E, training),
                  training_settings=tp["settings"], legacy_parameter_count=8270,
                  note="Different parameter counts and wall/CPU cost are reported; equal optimizer epoch budget is not equal FLOP budget"))
    dump(out / "tables/individual_runs.json", rows)
    dump(out / "TRANSITION_EVALUATION_SUMMARY.json", result)
    write_report(result, out)
    print("EVALUATION COMPLETE", out / "TRANSITION_EVALUATION_REPORT.txt", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check-gate-only", action="store_true")
    args = parser.parse_args()
    if args.check_gate_only:
        cheap_gate(args.out.resolve())
        print("Nine training and eighteen field metadata gates complete; no checkpoints or target data opened")
    else:
        evaluate(args.out)
