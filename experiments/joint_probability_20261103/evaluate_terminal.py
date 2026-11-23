"""Exploratory terminal-state diagnostics, without changing primary selection.

This analysis was added after early validation-selected checkpoints remained
before adaptation. Every prespecified run's step-200 state is evaluated; no
terminal checkpoint replaces the primary validation-selected result.

Observation containers can contain every PID partition when deserialized.
H responses do not enter fitting or selection; H scoring and target-reference
reading in this evaluator are permitted only after the complete global gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parents[2] / "outputs/JointProbabilityPhysics_20261103"
CASES = (("vn0_0015m_s", .0015), ("vn0_004m_s", .004))
SEEDS = (301030, 301031, 301032)
ARMS = ("old_frozen", "new_frozen", "new_joint")
VERSION = "exploratory-terminal-diagnostic-v1"
PURPOSE = "次要终态诊断，非主选解，不做再选择"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def expected_runs():
    return [(case, U, seed, arm) for case, U in CASES for seed in SEEDS for arm in ARMS]


def cheap_gate(out):
    """No checkpoint, observation, reference, or inference before this guard."""
    path = out / "selection_frozen.json"
    if not path.exists():
        raise RuntimeError("Global selection gate missing: terminal evaluation cannot read checkpoints/reference or score H")
    gate = read(path)
    expected = {(c, s, a) for c, _, s, a in expected_runs()}
    entries = gate.get("runs", [])
    observed = {(r["case"], int(r["seed"]), r["arm"]) for r in entries}
    if gate.get("all_required_selections_frozen") is not True or observed != expected or len(entries) != 18:
        raise RuntimeError("Exactly eighteen globally frozen primary selections are required")
    statuses = {}
    for case, U, seed, arm in expected_runs():
        path = out / "runs" / case / f"seed_{seed}" / arm / "status.json"
        status = read(path)
        if (status.get("complete") is not True or status.get("iteration") != 200
                or status.get("steps") != 200 or status.get("case") != case
                or status.get("arm") != arm or status.get("seed") != seed):
            raise RuntimeError("All eighteen runs must be complete at step 200 before checkpoint reads: " + str(path))
        statuses[(case, seed, arm)] = status
    return gate, statuses


def verify_plan(out):
    path = out / "secondary_analysis_plan.json"
    if not path.exists():
        raise RuntimeError("The secondary analysis plan must be recorded before evaluation")
    plan = read(path)
    if plan.get("version") != VERSION or plan.get("script_sha256") != sha(__file__):
        raise RuntimeError("Secondary analysis script differs from its recorded plan")
    if plan.get("primary_selection_unchanged") is not True or plan.get("terminal_iteration") != 200:
        raise RuntimeError("Unexpected secondary analysis scope")
    if plan.get("heldout_degrees") != [20, 28] or plan.get("fit_validation_degree") != 20:
        raise RuntimeError("Quadrature differs from the recorded secondary plan")
    expected = [dict(case=c, U=u, seed=s, arm=a, iteration=200) for c, u, s, a in expected_runs()]
    if plan.get("runs") != expected:
        raise RuntimeError("Terminal diagnostics must include every fixed run without selection")
    for path_name, expected_hash in plan["helper_sources"].items():
        if sha(path_name) != expected_hash:
            raise RuntimeError("Secondary evaluation helper changed: " + path_name)
    return plan


def bind_terminal_checkpoints(E, J, out, protocol, gate, statuses, plan):
    """Register all completed terminal bytes before any target inference/read."""
    fingerprint = protocol["fingerprint"]
    J.verify_gate(out, fingerprint)  # verifies every primary-selected byte hash
    records = []
    for case, U, seed, arm in expected_runs():
        folder = out / "runs" / case / f"seed_{seed}" / arm
        status = statuses[(case, seed, arm)]
        path = folder / "last.pt"
        digest = sha(path)
        ck = E.load(path)
        selected = E.load(folder / "selected.pt")
        if (ck.get("fingerprint") != fingerprint or selected.get("fingerprint") != fingerprint
                or status.get("fingerprint") != fingerprint or ck.get("iteration") != 200):
            raise RuntimeError("Completed terminal/selected checkpoint protocol mismatch: " + str(path))
        expected_updates = 32 if arm == "new_joint" else 0
        if ck.get("network_steps") != expected_updates or status.get("network_steps") != expected_updates:
            raise RuntimeError("Unexpected terminal network update count")
        for pick in (selected, ck["selected"]):
            if (pick["arm"] != arm or int(pick["seed"]) != seed or abs(float(pick["U"]) - U) > 1e-14
                    or pick["fingerprint"] != fingerprint or pick["iteration"] != status["selected_iteration"]):
                raise RuntimeError("Terminal checkpoint embedded primary identity mismatch")
        if not ck.get("trace") or ck["trace"][-1].get("iteration") != 200:
            raise RuntimeError("Terminal trace does not end at the fixed step")
        if sha(path) != digest:
            raise RuntimeError("Terminal checkpoint changed while binding")
        records.append(dict(case=case, U=U, seed=seed, arm=arm, iteration=200,
                            network_steps=expected_updates, path=str(path), sha256=digest,
                            status_sha256=sha(folder / "status.json"),
                            primary_selected_sha256=sha(folder / "selected.pt"),
                            primary_selected_iteration=int(selected["iteration"])))
    manifest = dict(version=VERSION, purpose=PURPOSE, protocol_fingerprint=fingerprint,
                    global_gate_sha256=sha(out / "selection_frozen.json"),
                    plan_sha256=sha(out / "secondary_analysis_plan.json"),
                    evaluator_sha256=sha(__file__), all_terminal_hashes_bound_before_H_scoring_and_reference_reads=True,
                    all_terminal_hashes_bound_before_H_reference_reads=True,
                    binding_scope="This terminal analysis only: hashes precede its H scoring/reference reads. Primary post-gate evaluation may already have scored H/read reference; the legacy before_H_reference_reads key has this same limited scope.",
                    H_scoring_or_target_reference_reads_performed_at_binding=False, runs=records)
    path = out / "secondary_terminal_manifest.json"
    if path.exists():
        existing = read(path)
        if existing != manifest:
            raise RuntimeError("Existing secondary manifest differs; refusing to mix terminal states")
    else:
        dump(path, manifest)
    return manifest


def load_terminal(E, out, record, fingerprint):
    path = Path(record["path"])
    if sha(path) != record["sha256"]:
        raise RuntimeError("Terminal bytes differ from pre-read manifest")
    ck = E.load(path)
    if ck["fingerprint"] != fingerprint or ck["iteration"] != 200:
        raise RuntimeError("Terminal checkpoint identity changed")
    d, cfg, model, net, cal, force = E.make_context(record["U"], record["seed"], record["arm"])
    model.load_state_dict(ck["model_state"], strict=True)
    net.load_state_dict(ck["net_state"], strict=True)
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    net.eval()
    a, es, en = model()
    if not all(torch.isfinite(x).all() for x in (a, es, en)):
        raise RuntimeError("Nonfinite terminal field replay")
    fields = dict(a=a, eta_s=es, eta_n=en)
    return ck, fields, d, net, cal, force


def write_text(result, out):
    lines = [PURPOSE, "", "分析性质：探索性、事后追加。",
             "该计划在观察到早期 V 选解停留于微调开始前之后提出；制定计划时未计算本轮 H 分数或读取目标参考场。",
             "同一观测数据包可整体反序列化并包含 H 行；H 响应不进入拟合或选解，H 评分仅在全局 gate 通过后进行。",
             "固定评价两工况 × 三种子 × 三臂的全部第 200 步终态，无重新优化、无挑选、无替换主 V 选解。",
             "主结论仍以冻结的 V 选解报告为准；不得从主选解与终态结果中择优报告。",
             "", "case | seed | arm | primary选中步 | 常流RMS(mm/s) | 超流RMS(mm/s) | H free NLL | H cap NLL | H free/cap coverage"]
    for row in result["runs"]:
        h = row["observations"]["H"]
        lines.append(f"{row['case']} | {row['seed']} | {row['arm']} | {row['primary_selected_iteration']} | "
                     f"{row['field']['rms_mm_s'][0]:.8g} | {row['field']['rms_mm_s'][1]:.8g} | "
                     f"{h['free']:.8g} | {h['cap']:.8g} | {h['coverage']['free']:.6f}/{h['coverage']['cap']:.6f}")
    lines.extend(["", "各工况 new_joint − new_frozen 的三种子均值："])
    for case, summary in result["summary"].items():
        difference = summary["paired_changes"]["new_joint_minus_new_frozen"]
        lines.append(case + ": " + "; ".join(f"{metric}={difference[metric]['mean']:.8g}"
                     for metric in ("normal_rms_mm_s", "superfluid_rms_mm_s", "heldout_free_nll", "heldout_cap_nll")))
    lines.extend(["", "完整 JSON 记录所有终态的 F/I/V NLL，H 两态 NLL/90%覆盖率/预测尺度，degree20→28变化，"
                  "以及每工况每种子 new_frozen/new_joint 的2×2场与网络交叉评分分解。",
                  "积分次数敏感性仅作描述，没有新增验收阈值。覆盖率是原二维混合分布诊断；"
                  "预测尺度为位移总方差迹平方根（μm），不是区间宽度。",
                  "H为同窗口内原生1ms条件预测，非自由运行未来轨迹预测；三种子是相同数据的初始化敏感性。",
                  "目标工况在历史研究中已探索，本分析不是全新盲测或确认性证据。",
                  "所有18个末态以预先记录的SHA绑定，记录后才允许 H 评分及读取目标参考场。每项行缓存同时绑定"
                  "终态manifest与对应reference SHA；引用SHA缺失或改变须重新计算。该绑定先后关系仅针对本终态分析，"
                  "主分析可能已在全局gate后评分H/读取reference；不表示全实验首次访问。历史/主实验文件未修改。"])
    (out / "secondary_terminal.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def evaluate(out=DEFAULT_OUT):
    begun = time.perf_counter()
    out = Path(out).resolve()
    gate, statuses = cheap_gate(out)
    plan = verify_plan(out)
    import experiment as E
    import evaluate_joint as J
    E.M.R.setup("cpu")
    if out != Path(E.OUT).resolve():
        raise RuntimeError("Terminal evaluation must use the frozen experiment output directory")
    protocol = E.verify_protocol()
    manifest = bind_terminal_checkpoints(E, J, out, protocol, gate, statuses, plan)
    binding = sha(out / "secondary_terminal_manifest.json")
    # Target-reference reads and H scoring may occur only below. Loading a whole
    # observation container is distinct from using H responses in a loss/score.
    references = {}
    rows, crosses = [], []
    for case, U in CASES:
        reference_path = J.reference_path(case)
        reference_digest = sha(reference_path)
        reference = E.load(reference_path)["a"].double().cpu()
        if sha(reference_path) != reference_digest:
            raise RuntimeError("Reference file changed during deserialization")
        references[case] = dict(path=str(reference_path), sha256=reference_digest)
        for seed in SEEDS:
            contexts, rows_by_arm = {}, {}
            for arm in ARMS:
                record = next(r for r in manifest["runs"] if (r["case"], r["seed"], r["arm"]) == (case, seed, arm))
                ck, fields, d, net, cal, force = load_terminal(E, out, record, protocol["fingerprint"])
                contexts[arm] = (fields, d, net, cal, force)
                row_path = out / "secondary_terminal_rows" / case / f"seed_{seed}" / f"{arm}.json"
                cached_row = read(row_path) if row_path.exists() else None
                if (cached_row is not None and cached_row.get("manifest_sha256") == binding
                        and cached_row.get("reference_sha256") == reference_digest
                        and cached_row.get("reference_path") == str(reference_path)):
                    row = cached_row
                else:
                    observations = {g: E.score(fields["a"], fields["eta_s"], fields["eta_n"],
                                    d, net, cal, force, g, degree=20, metrics=False) for g in ("F", "I", "V")}
                    for group, score in observations.items():
                        if not all(np.isfinite(score[k]) for k in ("total", "free", "cap")):
                            raise RuntimeError("Nonfinite terminal diagnostic score: " + group)
                    saved_v = ck["trace"][-1]["validation"]
                    delta_v = max(abs(observations["V"][k] - saved_v[k]) for k in ("total", "free", "cap"))
                    if delta_v > 1e-10:
                        raise RuntimeError("Terminal model/network replay disagrees with final saved V")
                    observations["H"] = J.checked_score(E, fields, d, net, cal, force, "H", 20)
                    higher = J.checked_score(E, fields, d, net, cal, force, "H", 28)
                    row = dict(case=case, U=U, seed=seed, arm=arm, iteration=200,
                               network_steps=record["network_steps"], primary_selected_iteration=record["primary_selected_iteration"],
                               manifest_sha256=binding, last_checkpoint_sha256=record["sha256"],
                               reference_path=str(reference_path), reference_sha256=reference_digest,
                               terminal_replay_V_max_difference=delta_v, field=J.field_metrics(fields["a"], reference, U),
                               observations=observations, heldout_degree28=higher,
                               quadrature=J.quadrature_difference(observations["H"], higher))
                    dump(row_path, row)
                rows.append(row)
                rows_by_arm[arm] = row
                print(f"TERMINAL {case} seed={seed} arm={arm} fixed_step=200", flush=True)
            cross_path = out / "secondary_terminal_rows" / case / f"seed_{seed}" / "cross.json"
            cached_cross = read(cross_path) if cross_path.exists() else None
            if (cached_cross is not None and cached_cross.get("manifest_sha256") == binding
                    and cached_cross.get("reference_sha256") == reference_digest
                    and cached_cross.get("reference_path") == str(reference_path)):
                cross = cached_cross
            else:
                ff, d, frozen_net, cal, force = contexts["new_frozen"]
                jf, _, joint_net, _, _ = contexts["new_joint"]
                by_degree = {}
                for degree in (20, 28):
                    own_key = "observations" if degree == 20 else "heldout_degree28"
                    nf = rows_by_arm["new_frozen"][own_key]
                    jj = rows_by_arm["new_joint"][own_key]
                    if degree == 20:
                        nf, jj = nf["H"], jj["H"]
                    jn = J.checked_score(E, jf, d, frozen_net, cal, force, "H", degree)
                    nj = J.checked_score(E, ff, d, joint_net, cal, force, "H", degree)
                    by_degree[str(degree)] = dict(new_frozen_field_frozen_net=nf, new_joint_field_joint_net=jj,
                                                  new_joint_field_frozen_net=jn, new_frozen_field_joint_net=nj,
                                                  decomposition=J.decomposition(nf, jj, jn, nj))
                cross = dict(case=case, U=U, seed=seed, group="H", iteration=200,
                             manifest_sha256=binding, reference_path=str(reference_path),
                             reference_sha256=reference_digest, by_degree=by_degree)
                dump(cross_path, cross)
            crosses.append(cross)
    if len(rows) != 18 or len(crosses) != 6:
        raise RuntimeError("Incomplete fixed terminal analysis")
    result = dict(version=VERSION, complete=True, purpose=PURPOSE,
                  exploratory=True, added_after_observing_early_V_selection_before_adaptation=True,
                  H_scored_at_plan_creation=False, target_reference_read_at_plan_creation=False,
                  observation_container_may_include_H=True, H_used_in_fit_or_selection=False,
                  primary_selection_unchanged=True,
                  fitting_or_selection_performed=False, created_utc=datetime.now(timezone.utc).isoformat(),
                  wall_seconds=time.perf_counter() - begun, manifest_sha256=binding,
                  terminal_binding_scope=manifest["binding_scope"],
                  reference_sources=references,
                  cache_binding="Every row/cross binds terminal manifest SHA plus case reference path/SHA; missing reference SHA requires recomputation",
                  runs=rows, cross_scores=crosses,
                  summary=J.summarize(rows), plan=plan)
    dump(out / "secondary_terminal.json", result)
    write_text(result, out)
    print("TERMINAL DIAGNOSTIC COMPLETE", out / "secondary_terminal.json", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check-gate-only", action="store_true",
                        help="Check only gate and completed statuses, without opening any checkpoint or target data")
    args = parser.parse_args()
    if args.check_gate_only:
        cheap_gate(args.out.resolve())
        print("All eighteen metadata gates passed; no checkpoint or target data opened")
    else:
        evaluate(args.out)
