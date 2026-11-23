"""Auxiliary fixed-legacy-field probability attribution after complete evaluation.

No optimization, checkpoint selection, reference-field read, or change to the
frozen experiment is performed. H scoring is prohibited until the complete
primary/terminal evaluation and every training/field gate have been verified.
The two additive terms are descriptive path-dependent substitutions, not causal
effects, new primary metrics, or a rule for choosing any subsequent change.
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
OUT = HERE.parents[2] / "outputs/SharedTransition_20261104"
VERSION = "fixed-legacy-field-probability-attribution-v1"
DEGREES = (20, 28)
COMPONENTS = ("network_change_at_legacy_field", "field_change_at_new_network", "total_change")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def require_completed_main(out):
    """A metadata-only guard, deliberately before experiment imports/checkpoints."""
    path = out / "TRANSITION_EVALUATION_SUMMARY.json"
    if not path.is_file():
        raise RuntimeError("Complete main evaluation is required before auxiliary H scoring")
    digest = sha(path)
    result = read(path)
    if sha(path) != digest:
        raise RuntimeError("Main summary changed while being read")
    if (result.get("complete") is not True or result.get("selected_or_trained_during_evaluation") is not False
            or result.get("terminal_preplanned") is not True):
        raise RuntimeError("Main evaluation incomplete or scope changed")
    if not (out / "evaluation_manifest.json").is_file() or sha(out / "evaluation_manifest.json") != result.get("manifest_sha256"):
        raise RuntimeError("Completed main evaluation manifest missing or changed")
    return result, digest


def bind_completed_main(out):
    """Reuse audited gates and hash contracts; no target-response scoring here."""
    result, summary_sha = require_completed_main(out)
    V = module("auxiliary_transition_evaluation", HERE / "evaluate_transition.py")
    training, field_gate = V.cheap_gate(out)
    E = module("auxiliary_transition_experiment", HERE / "experiment.py")
    T = module("auxiliary_transition_training", HERE / "training.py")
    J = module("auxiliary_transition_helpers", V.HELPER)
    if out.resolve() != Path(E.OUT).resolve():
        raise RuntimeError("Auxiliary output must match frozen experiment.OUT")
    E.M.R.setup("cpu")
    protocol, tp, records = V.bind_gates(E, T, out, training, field_gate)
    legacy, old_rows, old_artifacts = V.bind_legacy(E, protocol)
    manifest, binding = V.bind_manifest(out, protocol, tp, training, records, legacy, old_artifacts)
    if binding != result["manifest_sha256"] or protocol["fingerprint"] != result["protocol_fingerprint"]:
        raise RuntimeError("Complete main summary no longer matches all current gates")
    rows = result.get("runs", [])
    expected = {(mode, case, seed, arm) for mode in V.MODES for case, _ in V.CASES for seed in V.SEEDS for arm in V.ARMS}
    index = {(r["mode"], r["case"], int(r["seed"]), r["arm"]): r for r in rows}
    if len(rows) != 48 or set(index) != expected or read(out / "tables/individual_runs.json") != rows:
        raise RuntimeError("Both complete 24-run main modes must be present and identical to the saved table")
    sources = {(r["case"], int(r["seed"]), r["arm"]): r for r in records + legacy}
    old_index = {(r["mode"], r["case"], int(r["seed"]), r["arm"]): r for r in old_rows}
    for key, row in index.items():
        record = sources[key[1:]]
        pk, hk = ("path", "sha256") if row["mode"] == "primary" else ("last_path", "last_sha256")
        if (Path(row["checkpoint_path"]).resolve() != Path(record[pk]).resolve()
                or row["checkpoint_sha256"] != record[hk]
                or row["primary_selected_iteration"] != record["selected_iteration"]):
            raise RuntimeError("Completed main row has inconsistent checkpoint provenance")
        if row["arm"] == "legacy":
            if row != old_index[key]:
                raise RuntimeError("Main legacy row differs from verified historical evaluation")
        else:
            cache = out / "evaluation" / row["mode"] / row["case"] / f"seed_{row['seed']}" / f"{row['arm']}.json"
            if (read(cache) != row or row["cache_key"]["manifest_sha256"] != binding
                    or row["training_sha256"] != record["training_sha"]):
                raise RuntimeError("Main row is not the gate-bound cached evaluation")
    for mode in V.MODES:
        for degree in DEGREES:
            if V.summarize([r for r in rows if r["mode"] == mode], degree) != result["summary"][mode][str(degree)]:
                raise RuntimeError("Main aggregate differs from individual evaluated rows")
    if sha(out / "TRANSITION_EVALUATION_SUMMARY.json") != summary_sha:
        raise RuntimeError("Main summary changed during dependency verification")
    return result, summary_sha, E, V, J, index, sources


def flatten(score):
    value = {f"{kind}_nll": float(score[kind]) for kind in ("total", "free", "cap")}
    for kind in ("free", "cap"):
        value[f"{kind}_coverage"] = float(score["coverage"][kind])
        value[f"{kind}_spread_um"] = float(score["spread"][kind])
    if not all(np.isfinite(v) for v in value.values()):
        raise RuntimeError("Nonfinite auxiliary score")
    return value


def attribution(baseline, new_at_baseline, new_own):
    """Telescoping path: legacy/legacy -> new/legacy -> new/new."""
    old, fixed, own = map(flatten, (baseline, new_at_baseline, new_own))
    values = {
        COMPONENTS[0]: {k: fixed[k] - old[k] for k in old},
        COMPONENTS[1]: {k: own[k] - fixed[k] for k in old},
        COMPONENTS[2]: {k: own[k] - old[k] for k in old},
    }
    for key in old:
        if not np.isclose(values[COMPONENTS[0]][key] + values[COMPONENTS[1]][key],
                          values[COMPONENTS[2]][key], rtol=1e-12, atol=1e-10):
            raise RuntimeError("Auxiliary additive decomposition does not reconstruct total delta")
    return values


def summarize(V, rows):
    index = {(r["case"], int(r["seed"]), r["arm"]): r for r in rows}
    expected = {(c, s, a) for c, _ in V.CASES for s in V.SEEDS for a in V.NEW_ARMS}
    if len(rows) != 18 or set(index) != expected:
        raise RuntimeError("Attribution must include all eighteen fixed primary comparisons")
    result = {}
    for case, _ in V.CASES:
        result[case] = {}
        for arm in V.NEW_ARMS:
            rr = [index[(case, s, arm)] for s in V.SEEDS]
            by_degree = {}
            for degree in DEGREES:
                dd = [r["by_degree"][str(degree)]["attribution"] for r in rr]
                by_degree[str(degree)] = {component: {metric: V.descriptive([d[component][metric] for d in dd])
                    for metric in dd[0][component]} for component in COMPONENTS}
            result[case][arm] = dict(by_degree=by_degree,
                own_minus_legacy_field_rms_mm_s={name: V.descriptive([r["own_minus_legacy_field_rms_mm_s"][i] for r in rr])
                                               for i, name in enumerate(("normal", "superfluid"))})
    return result


def write_text(result, out):
    lines = ["辅助概率收益归因：固定legacy主选解流场", "",
        "主评价全部完成后运行。未训练、未选解、未修改冻结源码、未读取目标reference；仅追加固定checkpoint的H交叉评分。",
        "每个工况/种子固定legacy主选a、eta_s、eta_n，再代入各新臂已选概率网络。",
        "network = newNet@legacyField − legacyNet@legacyField；field = newNet@ownField − newNet@legacyField。",
        "两项之和等于新臂自身主结果−legacy主结果；这是特定替换路径下的描述性分解，不是因果证明或独立可辨识性证明。",
        "不会替代原主指标，也不以H决定选解、训练或任何模型改动。未分析终态，以免混淆主选解归因。",
        "NLL负变化较好；coverage变化本身没有统一好坏方向，应结合名义90%和spread解释；spread单位μm。",
        "field项只表示固定新网络下的H分数变化，不能单凭它宣称流场改善；同时列出主评价已有的常/超流RMS变化（mm/s）。",
        "三seed是同数据上的训练shuffle/场初值配对，不是独立物理重复；报告均值、范围，不作显著性推断。", ""]
    for case, arms in result["summary"].items():
        for arm, values in arms.items():
            lines.append(case + " | " + arm)
            lines.append("主评价场RMS own−legacy: " + "; ".join(f"{k}={v['mean']:+.7g} [{v['min']:+.7g},{v['max']:+.7g}] mm/s"
                         for k, v in values["own_minus_legacy_field_rms_mm_s"].items()))
            for degree in DEGREES:
                for metric in ("total_nll", "free_nll", "cap_nll", "free_coverage", "cap_coverage", "free_spread_um", "cap_spread_um"):
                    pieces = []
                    for component in COMPONENTS:
                        v = values["by_degree"][str(degree)][component][metric]
                        pieces.append(f"{component}={v['mean']:+.8g} [{v['min']:+.8g},{v['max']:+.8g}]")
                    lines.append(f"degree{degree} {metric}: " + "; ".join(pieces))
            lines.append("")
    (out / "AUXILIARY_PROBABILITY_ATTRIBUTION.txt").write_text("\n".join(lines), encoding="utf8")


@torch.no_grad()
def evaluate(out=OUT):
    begun = time.perf_counter()
    out = Path(out).resolve()
    main, main_sha, E, V, J, index, sources = bind_completed_main(out)
    destination = out / "auxiliary_probability_attribution"
    bindings = dict(version=VERSION, script_sha256=sha(__file__), main_summary_sha256=main_sha,
        main_manifest_sha256=main["manifest_sha256"], protocol_fingerprint=main["protocol_fingerprint"],
        helper_sources={str(p): sha(p) for p in (Path(V.__file__), Path(E.__file__), V.HELPER, HERE / "training.py")},
        training_gate_sha256=sha(out / "training/selection_frozen.json"), field_gate_sha256=sha(out / "selection_frozen.json"),
        fixed_mode="primary", heldout_degrees=list(DEGREES),
        all_training_and_field_gates_and_complete_main_verified_before_auxiliary_H_scoring=True,
        reference_files_read=False, causal_interpretation=False, new_primary_metric=False,
        fitting_or_selection_or_H_driven_model_change=False,
        checkpoints=[dict(case=c, seed=s, arm=a, path=r["path"], sha256=r["sha256"])
                     for (c, s, a), r in sorted(sources.items())])
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        if read(manifest_path) != bindings:
            raise RuntimeError("Existing auxiliary manifest differs; refusing to mix attributions")
    else:
        V.dump(manifest_path, bindings)
    binding = sha(manifest_path)
    # All gates, main completion, every checkpoint and source hash passed above.
    # No target reference file is needed anywhere in this auxiliary program.
    rows = []
    for case, U in V.CASES:
        for seed in V.SEEDS:
            legacy_row = index[("primary", case, seed, "legacy")]
            legacy_ck = V.load_hashed(E, legacy_row["checkpoint_path"], legacy_row["checkpoint_sha256"])
            fixed = {k: legacy_ck[k] for k in ("a", "eta_s", "eta_n")}
            for arm in V.NEW_ARMS:
                own = index[("primary", case, seed, arm)]
                own_ck = V.load_hashed(E, own["checkpoint_path"], own["checkpoint_sha256"])
                d, cfg, model, net, cal, force = E.make_context(U, seed, arm)
                V.same_state(net.state_dict(), own_ck["net_state"], "auxiliary frozen network " + arm)
                cache_path = destination / "rows" / case / f"seed_{seed}" / f"{arm}.json"
                key = dict(manifest_sha256=binding, legacy_checkpoint_sha256=legacy_row["checkpoint_sha256"],
                           own_checkpoint_sha256=own["checkpoint_sha256"], training_sha256=own["training_sha256"])
                row = read(cache_path) if cache_path.is_file() else None
                if row is None or row.get("binding") != key:
                    by_degree = {}
                    for degree in DEGREES:
                        baseline = legacy_row["observations"]["H"] if degree == 20 else legacy_row["heldout_degree28"]
                        new_own = own["observations"]["H"] if degree == 20 else own["heldout_degree28"]
                        cross = J.checked_score(E, fixed, d, net, cal, force, "H", degree)
                        by_degree[str(degree)] = dict(legacy_network_legacy_field=baseline,
                            new_network_legacy_field=cross, new_network_own_field=new_own,
                            attribution=attribution(baseline, cross, new_own))
                    row = dict(case=case, U=U, seed=seed, arm=arm, mode="primary", binding=key,
                        legacy_selected_iteration=legacy_row["iteration"], own_selected_iteration=own["iteration"],
                        training_seed=own["training_seed"], training_epoch=own["training_epoch"],
                        by_degree=by_degree, legacy_field_rms_mm_s=legacy_row["field"]["rms_mm_s"],
                        own_field_rms_mm_s=own["field"]["rms_mm_s"],
                        own_minus_legacy_field_rms_mm_s=[a-b for a, b in zip(own["field"]["rms_mm_s"], legacy_row["field"]["rms_mm_s"])])
                    V.dump(cache_path, row)
                rows.append(row)
                print("AUXILIARY ATTRIBUTION", case, seed, arm, flush=True)
    if sha(out / "TRANSITION_EVALUATION_SUMMARY.json") != main_sha or sha(__file__) != bindings["script_sha256"]:
        raise RuntimeError("Auxiliary source or completed main summary changed during scoring")
    result = dict(version=VERSION, complete=True, auxiliary_only=True, causal_interpretation=False,
        fitting_or_selection_or_H_driven_model_change=False, reference_files_read=False,
        main_primary_metrics_unchanged=True, created_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.perf_counter()-begun, manifest_sha256=binding, main_summary_sha256=main_sha,
        interpretation="Descriptive additive substitution along legacy/legacy -> new/legacy -> new/new; not a causal decomposition",
        prediction_scope=main["prediction_scope"], seed_interpretation=main["seed_interpretation"],
        runs=rows, summary=summarize(V, rows))
    V.dump(out / "AUXILIARY_PROBABILITY_ATTRIBUTION.json", result)
    write_text(result, out)
    print("AUXILIARY ATTRIBUTION COMPLETE", out / "AUXILIARY_PROBABILITY_ATTRIBUTION.json", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check-gate-only", action="store_true")
    args = parser.parse_args()
    if args.check_gate_only:
        bind_completed_main(args.out.resolve())
        print("Complete main evaluation and all gates verified; no H scoring or reference read performed")
    else:
        evaluate(args.out)
