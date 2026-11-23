"""Read-only fit-integrity audit, permitted only after the global selection gate.

Reads protocol, completed status/last/selected checkpoints and the pretrained
network. Never loads target observations/references or scores any prediction.
Historical and frozen experiment sources are never changed by this program.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parents[2] / "outputs/JointProbabilityPhysics_20261103"
CASES = ("vn0_0015m_s", "vn0_004m_s")
SEEDS = (301030, 301031, 301032)
ARMS = ("old_frozen", "new_frozen", "new_joint")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def require_completed_gate(out):
    """Do not open any checkpoint before all eighteen completion statuses pass."""
    path = out / "selection_frozen.json"
    if not path.exists():
        raise RuntimeError("Global selection gate missing: audit may not read active checkpoints")
    gate = read(path)
    if gate.get("all_required_selections_frozen") is not True:
        raise RuntimeError("Global selection gate incomplete: audit may not read active checkpoints")
    expected = {(c, s, a) for c in CASES for s in SEEDS for a in ARMS}
    entries = gate.get("runs", [])
    observed = {(r["case"], r["seed"], r["arm"]) for r in entries}
    if observed != expected or len(entries) != 18:
        raise RuntimeError("Global selection gate must contain exactly eighteen unique required runs")
    statuses = {}
    for r in entries:
        key = (r["case"], r["seed"], r["arm"])
        status_path = out / "runs" / r["case"] / f"seed_{r['seed']}" / r["arm"] / "status.json"
        status = read(status_path)
        if status.get("complete") is not True or status.get("iteration") != 200:
            raise RuntimeError("A run is not complete; checkpoint audit prohibited: " + str(status_path))
        statuses[key] = status
    return gate, statuses


def nested_differences(left, right, path="", differences=None):
    """Exact trace comparison: timing removed by caller, no fitted tolerance."""
    if differences is None:
        differences = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            here = path + "." + str(key)
            if key not in left or key not in right:
                differences.append({"path": here, "reason": "key mismatch"})
            else:
                nested_differences(left[key], right[key], here, differences)
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            differences.append({"path": path, "reason": "length mismatch", "left": len(left), "right": len(right)})
        for i, (a, b) in enumerate(zip(left, right)):
            nested_differences(a, b, path + f"[{i}]", differences)
    elif isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if not torch.equal(left.cpu(), right.cpu()):
            differences.append({"path": path, "reason": "tensor mismatch"})
    elif left != right:
        item = {"path": path, "reason": "value mismatch", "left": left, "right": right}
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            item["absolute_difference"] = abs(left-right)
        differences.append(item)
    return differences


def weight_delta(state, baseline):
    missing, extra = sorted(set(baseline)-set(state)), sorted(set(state)-set(baseline))
    changed, norms, squares, outside_squares = [], {}, 0., 0.
    shapes = []
    for name in sorted(set(state) & set(baseline)):
        a, b = state[name].detach().cpu(), baseline[name].detach().cpu()
        if a.shape != b.shape:
            shapes.append(name)
            continue
        if not torch.equal(a, b):
            changed.append(name)
        value = float((a.double()-b.double()).square().sum())
        norms[name] = math.sqrt(value)
        squares += value
        if not (name.startswith("sign.") or name.startswith("output.")):
            outside_squares += value
    outside = [n for n in changed if not (n.startswith("sign.") or n.startswith("output."))]
    return {"exactly_equal_to_pretrained": not (changed or missing or extra or shapes),
            "changed_names": changed, "changed_names_outside_heads": outside,
            "missing_names": missing, "extra_names": extra, "shape_mismatch_names": shapes,
            "delta_l2": math.sqrt(squares), "non_head_delta_l2": math.sqrt(outside_squares),
            "per_tensor_delta_l2": norms}


def optimizer_steps(state):
    steps = []
    for entry in state.get("state", {}).values():
        value = entry.get("step")
        if value is not None:
            steps.append(int(value.item()) if torch.is_tensor(value) else int(value))
    return steps


def scientific_trace(trace, last_iteration=40):
    # Only these keys are execution-time bookkeeping; all other scientific and
    # checkpoint-selection fields must be identical before adaptation begins.
    return [{k: v for k, v in r.items() if k not in ("cpu_seconds", "wall_seconds")}
            for r in trace if int(r["iteration"]) <= last_iteration]


def write_text_report(result, out):
    lines = ["联合微调实验：运行完整性审计", "",
             f"总检查结果：{'通过' if result['passed'] else '失败（见下列失败项）'}；18个选解已冻结。",
             "本审计未读取目标观测/参考场，未计算H或其他预测分数，未重新拟合。",
             "冻结证据来自版本绑定源码、network更新计数/optimizer状态和last/selected网络权重；未保存每步网络快照，不能声称逐步复原全部历史网络。",
             "每臂固定200场更新；new_joint额外32次网络更新。相同场步数不等于相同计算成本。", "",
             "case | seed | arm | 场步/网络步 | 选中步/所含微调次数 | selected权重delta L2 | CPU s | wall s | 末尾训练目标 | selected V"]
    for r in result["runs"]:
        lines.append(f"{r['case']} | {r['seed']} | {r['arm']} | {r['field_steps']}/{r['network_steps']} | "
                     f"{r['selected_iteration']}/{r['selected_network_updates']} | {r['selected_network_delta']['delta_l2']:.8g} | "
                     f"{r['cpu_seconds']:.3f} | {r['wall_seconds']:.3f} | {r['final_training_objective']:.8g} | "
                     f"{r['selected_validation']['total']:.8g}")
    lines.extend(["", "各臂累计计算用时（sum of per-run timers）："])
    for arm, timing in result["timing"]["by_arm"].items():
        lines.append(f"{arm}: CPU={timing['sum_cpu_seconds']:.3f}s，wall累计={timing['sum_run_wall_seconds']:.3f}s")
    timing = result["timing"]
    lines.extend([f"全部运行 CPU累计={timing['sum_cpu_seconds']:.3f}s；wall累计={timing['sum_run_wall_seconds']:.3f}s。",
                  "每run计时覆盖初始验证及迭代计算，按原runner语义不含持久化I/O/初始化；wall累计不能当作并行批次历时。",
                  "实际批次起止历时：本审计不从文件时间戳推算。", "", "new_frozen与new_joint前40步："])
    for pair in result["pre_adaptation_parity"]:
        lines.append(f"{pair['case']} seed={pair['seed']}: {'完全一致' if pair['passed'] else '失败'}，差异项={pair['difference_count']}")
    lines.extend(["", "停止原因是固定更新预算，未设置或证明收敛；末尾目标、验证差和目标变化均只作描述。",
                  "三种子共享数据及PID划分，体现初始化敏感性，不是独立物理重复。"])
    if result["failures"]:
        lines.extend(["", "失败项："] + ["- " + x for x in result["failures"]])
    (out / "run_integrity.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(out=DEFAULT_OUT):
    out = Path(out).resolve()
    gate, statuses = require_completed_gate(out)
    # Deliberately import and verify only after the cheap gate/status preflight.
    import experiment as E
    E.M.R.setup("cpu")
    if out != Path(E.OUT).resolve():
        raise RuntimeError("Audit OUT must match the frozen experiment OUT")
    protocol = E.verify_protocol()
    fingerprint = protocol["fingerprint"]
    if gate.get("fingerprint") != fingerprint:
        raise RuntimeError("Global gate fingerprint differs from verified protocol")
    initial_path = E.EW.RUN / "training/selected.pt"
    pretrained = E.load(initial_path)["state"]
    failures, rows, traces, audits = [], [], {}, []

    def check(condition, label):
        if not condition:
            failures.append(label)
        return bool(condition)

    for item in sorted(gate["runs"], key=lambda r: (r["case"], r["seed"], ARMS.index(r["arm"]))):
        case, seed, arm = item["case"], item["seed"], item["arm"]
        key, label = (case, seed, arm), f"{case}/seed_{seed}/{arm}"
        folder = out / "runs" / case / f"seed_{seed}" / arm
        selected_path, last_path = folder / "selected.pt", folder / "last.pt"
        declared = Path(item["path"])
        if not declared.is_absolute():
            declared = out / declared
        if declared.resolve() != selected_path.resolve() or sha(selected_path) != item["sha256"]:
            raise RuntimeError("Selected checkpoint identity/hash differs from gate: " + label)
        status, last, selected = statuses[key], E.load(last_path), E.load(selected_path)
        if any(x.get("fingerprint") != fingerprint for x in (status, last, selected)):
            raise RuntimeError("Run fingerprint differs from verified protocol: " + label)
        trace = last["trace"]
        traces[key] = scientific_trace(trace)
        field_steps = sum("inversion" in x for x in trace)
        expected_fine = list(range(45, 201, 5)) if arm == "new_joint" else []
        actual_fine = [int(x["iteration"]) for x in trace if "fine" in x]
        check([x["iteration"] for x in trace] == list(range(201)), label + ": trace iterations are not exactly 0..200")
        check(field_steps == 200 and last["iteration"] == 200 and status["iteration"] == 200,
              label + ": expected 200 field updates")
        check(actual_fine == expected_fine, label + ": network update schedule differs from protocol")
        check(last["network_steps"] == status["network_steps"] == len(expected_fine), label + ": network step count mismatch")
        field_opt_steps = optimizer_steps(last["optimizer"])
        network_opt_steps = optimizer_steps(last["network_optimizer"])
        check(bool(field_opt_steps) and all(x == 200 for x in field_opt_steps), label + ": field Adam step mismatch")
        check((bool(network_opt_steps) and all(x == 32 for x in network_opt_steps)) if arm == "new_joint" else not network_opt_steps,
              label + ": network Adam step mismatch")
        selected_iteration = int(selected["iteration"])
        selected_updates = sum(i <= selected_iteration for i in actual_fine)
        check(status["selected_iteration"] == selected_iteration, label + ": selected iteration differs from status")
        embedded_difference = nested_differences(last["selected"], selected)
        check(not embedded_difference, label + ": selected.pt differs from last.pt embedded selection")
        matching_trace = [x for x in trace if x["iteration"] == selected_iteration and "validation" in x]
        check(len(matching_trace) == 1 and not nested_differences(matching_trace[0]["validation"], selected["validation_components"]),
              label + ": selected validation does not match recorded trace")
        check(not nested_differences(status["validation"], selected["validation_components"]), label + ": selected validation differs from status")
        final_delta = weight_delta(last["net_state"], pretrained)
        selected_delta = weight_delta(selected["net_state"], pretrained)
        for state_name, delta in (("final", final_delta), ("selected", selected_delta)):
            check(not delta["missing_names"] and not delta["extra_names"] and not delta["shape_mismatch_names"],
                  label + f": {state_name} network structure mismatch")
            check(not delta["changed_names_outside_heads"], label + f": {state_name} non-head network weights changed")
            if arm != "new_joint" or (state_name == "selected" and selected_updates == 0):
                check(delta["exactly_equal_to_pretrained"], label + f": {state_name} supposedly frozen network differs from pretrained")
        final_row = trace[-1]
        final_objective = float(final_row["objective"])
        check(math.isfinite(final_objective), label + ": nonfinite final objective")
        objective_tail = [float(x["objective"]) for x in trace[-20:]]
        cpu, wall = float(last["cpu_seconds"]), float(last["wall_seconds"])
        check(cpu == status["cpu_seconds"] and wall == status["wall_seconds"], label + ": status/last timer mismatch")
        checks = {"case": case, "seed": seed, "arm": arm, "field_steps": field_steps,
                  "network_steps": last["network_steps"], "network_update_iterations": actual_fine,
                  "selected_iteration": selected_iteration, "selected_network_updates": selected_updates,
                  "selected_network_delta": selected_delta, "final_network_delta": final_delta,
                  "field_optimizer_steps": field_opt_steps, "network_optimizer_steps": network_opt_steps,
                  "cpu_seconds": cpu, "wall_seconds": wall,
                  "final_training_objective": final_objective, "final_inverse_nll": final_row["inversion"],
                  "final_regularization": final_row["regularization"],
                  "final_gradient_norms": final_row["gradient_norms"],
                  "selected_validation": selected["validation_components"],
                  "final_validation": final_row["validation"],
                  "last_20_objectives": {"first": objective_tail[0], "last": objective_tail[-1],
                                         "min": min(objective_tail), "max": max(objective_tail),
                                         "last_minus_first": objective_tail[-1]-objective_tail[0]},
                  "stopping": "Fixed 200 field-update budget; convergence not established"}
        rows.append(checks)
        audits.append({"case": case, "seed": seed, "arm": arm,
                       "last_sha256": sha(last_path), "selected_sha256": item["sha256"],
                       "status_sha256": sha(folder / "status.json")})
    pairs = []
    for case in CASES:
        for seed in SEEDS:
            left, right = traces[(case, seed, "new_frozen")], traces[(case, seed, "new_joint")]
            differences = nested_differences(left, right)
            passed = check(not differences, f"{case}/seed_{seed}: new_frozen/new_joint first 40 scientific steps differ")
            pairs.append({"case": case, "seed": seed, "passed": passed,
                          "compared_iterations": list(range(41)), "exact_comparison": True,
                          "ignored_fields": ["cpu_seconds", "wall_seconds"],
                          "difference_count": len(differences), "differences": differences})
    timing = {"sum_cpu_seconds": sum(r["cpu_seconds"] for r in rows),
              "sum_run_wall_seconds": sum(r["wall_seconds"] for r in rows),
              "batch_elapsed_wall_seconds": None, "by_arm": {}}
    for arm in ARMS:
        rr = [r for r in rows if r["arm"] == arm]
        timing["by_arm"][arm] = {"sum_cpu_seconds": sum(r["cpu_seconds"] for r in rr),
                                 "sum_run_wall_seconds": sum(r["wall_seconds"] for r in rr)}
    result = {"version": "joint-run-integrity-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "passed": not failures, "failures": failures, "runs": rows,
              "pre_adaptation_parity": pairs, "timing": timing,
              "protocol_fingerprint": fingerprint, "protocol_sha256": sha(out / "protocol.json"),
              "gate_sha256": sha(out / "selection_frozen.json"), "audit_source_sha256": sha(__file__),
              "pretrained_network": {"path": str(initial_path), "sha256": sha(initial_path)},
              "checkpoint_audit": audits, "target_observations_deserialized": False,
              "target_reference_read": False, "prediction_scores_computed": False,
              "source_and_input_hashes_verified": True,
              "network_history_scope": "Frozen-source update path, optimizer/trace counters and final/selected equality; no per-step network snapshots exist",
              "convergence_established": False}
    dump(out / "run_integrity.json", result)
    write_text_report(result, out)
    print(json.dumps({"passed": result["passed"], "runs": len(rows), "failures": len(failures)}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = audit(args.out)
    raise SystemExit(0 if result["passed"] else 3)
