"""Independent, post-completion audit; never trains or scores observations.

This reporting utility is intentionally outside the frozen scientific source
manifest. It reads completed summaries/checkpoints/exports, recomputes their
arithmetic independently, and writes only FINAL_INDEPENDENT_AUDIT.json/.txt.
It must not run before both the evaluation summary and plot manifest complete.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parents[2] / "outputs/LatentTransition_20261104_v2"
OLD_OUT = HERE.parents[2] / "outputs/SharedTransition_20261104"
CASES = {"vn0_0015m_s": .0015, "vn0_004m_s": .004}
SEEDS = (301030, 301031, 301032)
TRAIN_SEEDS = (411040, 411041, 411042)
ARMS = ("iid", "reset", "temporal")
MODES = ("primary", "terminal")
PAIRS = (("reset", "iid"), ("temporal", "reset"), ("temporal", "iid"))
CORE = ("normal_rms_mm_s", "superfluid_rms_mm_s", "heldout_free_nll", "heldout_cap_nll")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Auditor:
    def __init__(self, out):
        self.out = out
        self.result = dict(version="independent-latent-transition-audit-v1", complete=False,
            passed=False, created_utc=datetime.now(timezone.utc).isoformat(),
            script_path=str(Path(__file__).resolve()), script_sha256=digest(__file__),
            scientific_protocol_modified=False, new_fitting_or_selection=False,
            observation_likelihood_recomputed=False, checks=[], hash_checks=[],
            numeric_comparisons=0, numeric_max_abs_error=0., tensor_comparisons=0,
            tensor_elements_compared=0, failures=[])

    def need(self, condition, label):
        if not condition:
            raise AssertionError(label)

    def checked_hash(self, path, expected=None):
        path = Path(path)
        actual = digest(path)
        self.need(expected is None or actual == expected, "SHA256 mismatch: " + str(path))
        self.result["hash_checks"].append(dict(path=str(path.resolve()), sha256=actual,
                                                bound=expected is not None))
        return actual

    def load(self, path, expected=None):
        actual = self.checked_hash(path, expected)
        value = torch.load(path, map_location="cpu", weights_only=False)
        self.need(digest(path) == actual, "Checkpoint changed while reading: " + str(path))
        return value

    def number(self, actual, expected, label, atol=5e-12, rtol=5e-12):
        a, b = float(actual), float(expected)
        self.need(math.isfinite(a) and math.isfinite(b), "Nonfinite numeric value: " + label)
        error = abs(a-b)
        self.result["numeric_comparisons"] += 1
        self.result["numeric_max_abs_error"] = max(self.result["numeric_max_abs_error"], error)
        self.need(error <= atol + rtol*abs(b), f"Numeric mismatch {label}: {a} versus {b}")

    def tree(self, actual, expected, label):
        if isinstance(expected, dict):
            self.need(isinstance(actual, dict) and set(actual) == set(expected), "Keys differ: " + label)
            for key in expected:
                self.tree(actual[key], expected[key], label + "/" + str(key))
        elif isinstance(expected, (list, tuple)):
            self.need(len(actual) == len(expected), "Lengths differ: " + label)
            for i, value in enumerate(expected):
                self.tree(actual[i], value, label + "/" + str(i))
        elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
            self.need(actual == expected, "Value differs: " + label)
        elif isinstance(expected, (int, float, np.number)):
            self.number(actual, expected, label)
        else:
            raise TypeError(type(expected))

    def state(self, actual, expected, label):
        self.need(set(actual) == set(expected), "State keys differ: " + label)
        for key in expected:
            a, b = actual[key], expected[key]
            self.need(torch.is_tensor(a) and torch.is_tensor(b), "Non-tensor state: " + label)
            self.need(a.shape == b.shape and torch.equal(a, b.to(a)), "Tensor changed: " + label + "/" + key)
            self.result["tensor_comparisons"] += 1
            self.result["tensor_elements_compared"] += a.numel()

    def done(self, label):
        self.result["checks"].append(label)

    def prerequisites(self):
        summary_path = self.out / "LATENT_EVALUATION_SUMMARY.json"
        plot_path = self.out / "PLOT_MANIFEST.json"
        self.need(summary_path.is_file() and plot_path.is_file(), "Evaluation summary and plot manifest must both exist")
        summary, plots = read(summary_path), read(plot_path)
        self.need(summary.get("complete") is True and plots.get("complete") is True,
                  "Evaluation summary and plot manifest must both be complete")
        self.checked_hash(summary_path)
        self.checked_hash(plot_path)
        self.done("完成门禁：主评价与绘图清单均已完成，之后才读取结果和checkpoint")
        return summary, plots

    def gates(self, summary, plots):
        p = read(self.out / "protocol.json")
        fp = p["fingerprint"]
        self.need(p.get("frozen") is True and fingerprint({k: v for k, v in p.items() if k != "fingerprint"}) == fp,
                  "Scientific protocol fingerprint invalid")
        self.need(str(Path(__file__).resolve()) not in {str(Path(x).resolve()) for x in p["sources"]},
                  "Post-completion auditor must not be part of the frozen scientific protocol")
        for kind in ("sources", "inputs"):
            for path, expected in p[kind].items():
                self.checked_hash(path, expected)
        self.need(p["new_physics"] == {"nu_s": 0., "Fns_factor": 1.} and "Frozen B" in p["process_prior"],
                  "Physical operator/process-prior contract changed")
        tg_path, fg_path = self.out / "training/selection_frozen.json", self.out / "selection_frozen.json"
        tg, fg = read(tg_path), read(fg_path)
        self.need(tg["all_required_selections_frozen"] is True and len(tg["runs"]) == 3,
                  "Three transition heads required")
        self.need({r["seed"] for r in tg["runs"]} == set(TRAIN_SEEDS), "Training seed set differs")
        self.need(fingerprint({k: v for k, v in tg.items() if k != "fingerprint"}) == tg["fingerprint"],
                  "Training gate fingerprint invalid")
        self.need(tg["protocol_fingerprint"] == fp and fg["fingerprint"] == fp, "Gate/protocol binding differs")
        self.checked_hash(tg_path, fg["training_gate_sha256"])
        self.need(fg["all_required_selections_frozen"] is True and fg["all_terminal_states_bound_before_evaluation"] is True,
                  "Field primary/terminal global freeze incomplete")
        expected = {(c, s, a) for c in CASES for s in SEEDS for a in ARMS}
        self.need(len(fg["runs"]) == 18 and {(r["case"], r["seed"], r["arm"]) for r in fg["runs"]} == expected,
                  "Eighteen distinct field runs required")
        self.need(summary["protocol_fingerprint"] == fp and not summary["selected_or_trained_during_evaluation"]
                  and summary["terminal_preplanned"], "Evaluation protocol/selection flag differs")
        manifest_path = self.out / "evaluation_manifest.json"
        self.checked_hash(manifest_path, summary["manifest_sha256"])
        manifest = read(manifest_path)
        self.checked_hash(tg_path, manifest["training_gate_sha256"])
        self.checked_hash(fg_path, manifest["field_gate_sha256"])
        self.need(plots["evaluation_manifest_sha256"] == summary["manifest_sha256"], "Plot/evaluation binding differs")
        self.result["protocol_fingerprint"] = fp
        self.result["early_preflight_padding_defect_disclosed"] = bool(p.get("early_preflight_zero_weight_padding_defect_disclosed"))
        self.result["protocol_note"] = "正式冻结前的零权H哑行预检缺陷已由原协议披露；本审计不将历史事件改写为未发生。"
        oldgate = read(OLD_OUT / "training/selection_frozen.json")
        heads, bases, training_records = {}, {}, {}
        for row in tg["runs"]:
            seed = row["seed"]
            ck = self.load(row["path"], row["sha256"])
            self.checked_hash(row["status_path"], row["status_sha256"])
            st = read(row["status_path"])
            self.need(st["complete"] and ck["epoch"] == st["selected_epoch"] == row["selected_epoch"], "Training selection identity differs")
            self.need(ck["fingerprint"] == st["fingerprint"] == fp and ck["seed"] == st["seed"] == seed,
                      "Training checkpoint identity differs")
            self.need(st["selected_sha256"] == row["sha256"], "Training selected hash differs")
            original = next(r for r in oldgate["runs"] if r["arm"] == "independent" and r["seed"] == seed)
            self.need(ck["base_sha256"] == row["base_sha256"] == original["sha256"], "Frozen marginal binding differs")
            base_ck = self.load(original["path"], original["sha256"])
            history = read(Path(row["path"]).parent / "history.json")
            best = history[0]
            for entry in history[1:]:
                if entry["validation"] < best["validation"] - p["training_settings"]["selection_min_delta"]:
                    best = entry
            self.need(best["epoch"] == ck["epoch"], "Training validation selection cannot be reproduced")
            self.number(ck["validation"], best["validation"], "training selected validation")
            self.need(st["last_epoch"] == p["training_settings"]["epochs"] or
                      (st["last_epoch"] >= p["training_settings"]["minimum_epochs"] and
                       st["last_epoch"] - best["epoch"] >= p["training_settings"]["patience"]),
                      "Training completion does not satisfy fixed budget/patience")
            heads[seed], bases[seed], training_records[seed] = ck["state"], base_ck["state"], row
        expected_rows, selection_audit = {}, []
        for row in fg["runs"]:
            selected = self.load(row["path"], row["sha256"])
            last = self.load(row["last_path"], row["last_sha256"])
            status = read(Path(row["path"]).parent / "status.json")
            seed, arm, case = row["seed"], row["arm"], row["case"]
            ts = TRAIN_SEEDS[SEEDS.index(seed)]
            tr = training_records[ts]
            self.need(status["complete"] and last["iteration"] == status["iteration"] == 200 and last["network_steps"] == 0,
                      "Incomplete field run or target network updates")
            self.need(status["case"] == case and status["arm"] == arm and status["seed"] == seed and status["fingerprint"] == fp,
                      "Field status identity differs")
            self.state(last["selected"]["model_state"], selected["model_state"], "embedded primary field")
            for ck in (selected, last, last["selected"]):
                self.need(ck["arm"] == arm and ck["seed"] == seed and ck["fingerprint"] == fp and
                          ck["training_seed"] == ts and ck["training_sha"] == tr["sha256"], "Field/head identity differs")
                self.number(ck["U"], CASES[case], "checkpoint U")
                net = ck["net_state"]
                self.need(set(net) == {"base."+k for k in bases[ts]} | {"head."+k for k in heads[ts]}, "Unexpected network state keys")
                self.state({k[5:]: v for k, v in net.items() if k.startswith("base.")}, bases[ts], "frozen base")
                self.state({k[5:]: v for k, v in net.items() if k.startswith("head.")}, heads[ts], "frozen head")
            trace, warm = last["trace"], last["warm_validation"]
            self.need(trace[0]["iteration"] == 0 and trace[-1]["iteration"] == 200, "Incomplete field history")
            best = trace[0]
            eligible_steps = [0]
            for entry in trace[1:]:
                if "validation" not in entry:
                    continue
                val = entry["validation"]
                eligible = all(math.isfinite(val[k]) for k in ("total", "free", "cap")) and all(
                    val[k] <= warm[k] + p["guard"] for k in ("free", "cap"))
                chosen = eligible and val["total"] < best["validation"]["total"] - p["selection_delta"]
                self.need(entry["eligible"] == eligible and entry["selected"] == chosen, "Saved field guard/selection flag differs")
                if eligible:
                    eligible_steps.append(entry["iteration"])
                if chosen:
                    best = entry
            self.need(best["iteration"] == selected["iteration"] == row["selected_iteration"] == status["selected_iteration"],
                      "Validation-selected field cannot be reproduced")
            self.tree(selected["validation_components"], best["validation"], "selected validation metadata")
            if arm == "iid":
                source = selected["reuse_source"]
                self.need(source == last["reuse_source"] and source["arm"] == "independent" and source["seed"] == seed and source["case"] == case,
                          "iid provenance differs")
                old_selected = self.load(source["path"], source["sha256"])
                old_last = self.load(source["last_path"], source["last_sha256"])
                self.state(selected["model_state"], old_selected["model_state"], "iid selected field reuse")
                self.state(last["model_state"], old_last["model_state"], "iid terminal field reuse")
                self.need(last["trace"] == old_last["trace"], "iid optimization history changed")
                self.state({k: selected[k] for k in ("a", "eta_s", "eta_n")},
                           {k: old_selected[k] for k in ("a", "eta_s", "eta_n")}, "iid selected coefficient reuse")
            for mode, ck, key, hashkey in (("primary", selected, "path", "sha256"), ("terminal", last, "last_path", "last_sha256")):
                expected_rows[(mode, case, seed, arm)] = dict(checkpoint_path=row[key], checkpoint_sha256=row[hashkey],
                    iteration=ck["iteration"], validation=selected["validation_components"] if mode == "primary" else trace[-1]["validation"],
                    training_seed=ts, transition_training_sha256=tr["sha256"], selected_iteration=selected["iteration"])
            selection_audit.append(dict(case=case, seed=seed, arm=arm, selected_iteration=selected["iteration"], eligible_steps=eligible_steps))
        self.result["training_heads"] = 3
        self.result["field_runs"] = 18
        self.result["selection_replay"] = selection_audit
        self.done("三头和18场gate、SHA、base/head逐元素冻结、iid原场复用、V保护/主选规则复核")
        return expected_rows

    @staticmethod
    def metrics(row, degree):
        score = row["observations"]["H"] if degree == 20 else row["heldout_degree28"]
        values = {"normal_rms_mm_s": row["field"]["rms_mm_s"][0],
                  "superfluid_rms_mm_s": row["field"]["rms_mm_s"][1], "heldout_total_nll": score["total"]}
        for kind in ("free", "cap"):
            values[f"heldout_{kind}_nll"] = score[kind]
            values[f"heldout_{kind}_coverage"] = score["coverage"][kind]
            values[f"heldout_{kind}_coverage_abs_error"] = abs(score["coverage"][kind] - .9)
            values[f"heldout_{kind}_spread_um"] = score["spread"][kind]
        return values

    @staticmethod
    def descriptive(values):
        return dict(mean=math.fsum(values)/len(values), min=min(values), max=max(values),
                    by_seed={str(seed): value for seed, value in zip(SEEDS, values)})

    def numeric(self, summary, expected_rows):
        rows = summary["runs"]
        index = {(r["mode"], r["case"], r["seed"], r["arm"]): r for r in rows}
        self.need(len(rows) == 36 and set(index) == set(expected_rows), "Exactly 36 unique primary/terminal rows required")
        self.tree(read(self.out / "tables/individual_runs.json"), rows, "individual rows mirror")
        for key, row in index.items():
            ex = expected_rows[key]
            for name in ("checkpoint_sha256", "iteration", "training_seed", "transition_training_sha256"):
                self.need(row[name] == ex[name], "Evaluation row checkpoint identity differs: " + name)
            self.need(Path(row["checkpoint_path"]).resolve() == Path(ex["checkpoint_path"]).resolve(), "Evaluation checkpoint path differs")
            self.need(row["target_network_updates"] == 0 and row["primary_selected_iteration"] == ex["selected_iteration"], "Evaluation metadata differs")
            self.tree(row["observations"]["V"], ex["validation"], "row V versus selected metadata")
            self.need(row["replay_V_max_difference"] <= 1e-9, "Evaluator V replay tolerance exceeded")
            for degree, scores in ((20, row["observations"]["H"]), (28, row["heldout_degree28"])):
                self.number(scores["total"], .5*(scores["free"]+scores["cap"]), "balanced H total")
                for kind in ("free", "cap"):
                    pidrows = list(scores["per_pid"][kind].values())
                    self.need(len(pidrows) > 0, "Empty held-out PID group")
                    for name, reported in (("nll", scores[kind]), ("coverage", scores["coverage"][kind]), ("spread", scores["spread"][kind])):
                        self.number(reported, math.fsum(r[name] for r in pidrows)/len(pidrows), "H equal-PID " + name, atol=1e-10)
                    self.need(0 <= scores["coverage"][kind] <= 1 and scores["spread"][kind] >= 0, "Invalid probability diagnostic")
            h20, h28 = row["observations"]["H"], row["heldout_degree28"]
            for kind in ("total", "free", "cap"):
                self.number(row["quadrature"]["nll_28_minus_20"][kind], h28[kind]-h20[kind], "row degree difference")
            for report, source in (("coverage_28_minus_20", "coverage"), ("spread_um_28_minus_20", "spread")):
                for kind in ("free", "cap"):
                    self.number(row["quadrature"][report][kind], h28[source][kind]-h20[source][kind], "row diagnostic degree difference")
        recomputed = {}
        for mode in MODES:
            recomputed[mode] = {}
            for degree in (20, 28):
                bycase = {}
                for case, U in CASES.items():
                    item = dict(U=U, degree=degree, arms={}, paired_changes={})
                    metric = {(seed, arm): self.metrics(index[(mode, case, seed, arm)], degree) for seed in SEEDS for arm in ARMS}
                    keys = list(metric[(SEEDS[0], ARMS[0])])
                    for arm in ARMS:
                        item["arms"][arm] = {k: self.descriptive([metric[(seed, arm)][k] for seed in SEEDS]) for k in keys}
                    for first, second in PAIRS:
                        changes = {k: self.descriptive([metric[(seed, first)][k]-metric[(seed, second)][k] for seed in SEEDS]) for k in keys}
                        count = sum(all(metric[(seed, first)][k] < metric[(seed, second)][k] for k in CORE) for seed in SEEDS)
                        item["paired_changes"][first+"_minus_"+second] = dict(metrics=changes,
                            joint_four_metric_improvement_count=count, all_four_mean_deltas_negative=all(changes[k]["mean"] < 0 for k in CORE))
                    bycase[case] = item
                recomputed[mode][str(degree)] = bycase
        self.tree(summary["summary"], recomputed, "independent summaries and paired deltas")
        terminal = {}
        for case in CASES:
            terminal[case] = {}
            for arm in ARMS:
                before = {s: self.metrics(index[("primary", case, s, arm)], 20) for s in SEEDS}
                after = {s: self.metrics(index[("terminal", case, s, arm)], 20) for s in SEEDS}
                terminal[case][arm] = {k: self.descriptive([after[s][k]-before[s][k] for s in SEEDS]) for k in before[SEEDS[0]]}
        self.tree(summary["terminal_minus_primary"], terminal, "terminal minus primary")
        directions = []
        for mode in MODES:
            for case in CASES:
                for first, second in PAIRS:
                    pair = first+"_minus_"+second
                    for metric in ("heldout_free_nll", "heldout_cap_nll"):
                        a, b = [recomputed[mode][str(d)][case]["paired_changes"][pair]["metrics"][metric]["mean"] for d in (20, 28)]
                        directions.append(dict(mode=mode, case=case, pair=pair, metric=metric, delta20=a, delta28=b, sign_reversal=a*b < 0))
        self.tree(summary["quadrature_directions"], directions, "20/28 paired directions")
        self.result["evaluation_rows"] = len(rows)
        self.result["quadrature_directions"] = directions
        self.result["quadrature_direction_reversals"] = [r for r in directions if r["sign_reversal"]]
        self.done("36行、逐PID概率汇总、主/末态均值极值配对差、20/28阶方向独立重算")
        return index

    @staticmethod
    def flat_scores(score):
        out = {f"{k}_nll": score[k] for k in ("total", "free", "cap")}
        for state in ("free", "cap"):
            out[state+"_coverage"] = score["coverage"][state]
            out[state+"_spread_um"] = score["spread"][state]
        return out

    def auxiliary(self, summary, index):
        aux = summary["auxiliary"]
        self.need(aux["complete"] and aux["fixed_mode"] == "primary" and not aux["fitting_or_selection"], "Auxiliary analysis incomplete")
        self.tree(read(self.out / "AUXILIARY_PROBABILITY_ATTRIBUTION.json"), aux, "auxiliary file mirror")
        keys = {(r["case"], r["seed"], r["arm"]) for r in aux["rows"]}
        self.need(len(aux["rows"]) == 12 and keys == {(c, s, a) for c in CASES for s in SEEDS for a in ("reset", "temporal")},
                  "Exactly twelve auxiliary comparisons required")
        max_telescoping = 0.
        for row in aux["rows"]:
            case, seed, arm = row["case"], row["seed"], row["arm"]
            iid, own = index[("primary", case, seed, "iid")], index[("primary", case, seed, arm)]
            self.need(row["binding"] == dict(manifest_sha256=summary["manifest_sha256"],
                iid_checkpoint_sha256=iid["checkpoint_sha256"], own_checkpoint_sha256=own["checkpoint_sha256"]), "Auxiliary binding differs")
            for degree in (20, 28):
                item = row["by_degree"][str(degree)]
                score_key = "observations" if degree == 20 else "heldout_degree28"
                old = iid[score_key]["H"] if degree == 20 else iid[score_key]
                new = own[score_key]["H"] if degree == 20 else own[score_key]
                self.tree(item["iid_probability_iid_field"], old, "auxiliary iid score")
                self.tree(item["new_probability_own_field"], new, "auxiliary own score")
                a, b, c = [self.flat_scores(x) for x in (old, item["new_probability_iid_field"], new)]
                for key in a:
                    p, f, t = b[key]-a[key], c[key]-b[key], c[key]-a[key]
                    for name, expected in (("probability_change_at_iid_field", p), ("field_change_at_new_probability", f), ("total_change", t)):
                        self.number(item["components"][name][key], expected, "auxiliary recomputed " + key)
                    max_telescoping = max(max_telescoping, abs((p+f)-t))
                    self.number(p+f, t, "auxiliary telescoping")
                self.number(b["free_nll"], a["free_nll"], "unchanged free network at fixed iid field", atol=1e-10)
            self.tree(row["own_minus_iid_field_rms_mm_s"],
                      (np.array(own["field"]["rms_mm_s"])-np.array(iid["field"]["rms_mm_s"])).tolist(), "auxiliary field delta")
        self.result["auxiliary_rows"] = 12
        self.result["auxiliary_max_telescoping_error"] = max_telescoping
        self.done("12行固定iid场归因及20/28阶两项之和、来源绑定独立复核")

    def exports(self, summary, plots, index):
        self.need(plots["representative_seed"] == SEEDS[0] and plots["representative_training_seed"] == TRAIN_SEEDS[0]
                  and plots["all_arms_modes_and_96_frames_retained"], "Plot representative selection contract differs")
        artifacts = plots["artifacts"]
        self.need(len({r["path"] for r in artifacts}) == len(artifacts), "Duplicate plot artifact paths")
        for row in artifacts:
            self.checked_hash(row["path"], row["sha256"])
        fields = [r for r in artifacts if r["kind"] == "field_npz"]
        refs = [r for r in artifacts if r["kind"] == "reference_npz"]
        figures = [r for r in artifacts if r["kind"] == "figure"]
        self.need(len(fields) == 12 and len(refs) == 2 and len(figures) == 36 and len(artifacts) == 50,
                  "Expected twelve reconstruction NPZ, two reference NPZ, thirty-six PNG/SVG")
        self.need({(r["mode"], r["case"], r["arm"]) for r in fields} == {(m, c, a) for m in MODES for c in CASES for a in ARMS},
                  "Field export combinations incomplete")
        reference_arrays = {}
        time_axis = np.arange(5, 101)*.001
        for case in CASES:
            path = self.out / "fields" / case / "reference_K12.npz"
            self.need(str(path.resolve()) in {str(Path(r["path"]).resolve()) for r in refs}, "Reference export missing")
            source = summary["reference_sources"][case]
            self.checked_hash(source["path"], source["sha256"])
            with np.load(path, allow_pickle=False) as saved:
                velocity, a = saved["velocity_m_s"].copy(), saved["a"].copy()
                self.need(velocity.shape == (96, 2, 40, 40, 2) and a.shape == (96, 2, 624), "Reference export dimensions invalid")
                self.need(np.array_equal(saved["time_s"], time_axis) and np.isfinite(velocity).all() and np.isfinite(a).all(), "Reference time/finite check failed")
                self.need(str(saved["reference_sha256"].item()) == source["sha256"], "Reference source binding differs")
                source_a = self.load(source["path"], source["sha256"])["a"].double().numpy()
                self.need(np.array_equal(a, source_a), "Exported reference coefficients differ from source")
                reference_arrays[case] = (velocity, a, saved["x_m"].copy(), saved["y_m"].copy())
        checked = []
        max_error = 0.
        for artifact in fields:
            case, mode, arm = artifact["case"], artifact["mode"], artifact["arm"]
            row = index[(mode, case, SEEDS[0], arm)]
            ref, ref_a, xx, yy = reference_arrays[case]
            with np.load(artifact["path"], allow_pickle=False) as saved:
                velocity, a = saved["velocity_m_s"], saved["a"]
                self.need(velocity.shape == (96, 2, 40, 40, 2) and a.shape == (96, 2, 624), "Reconstruction export dimensions invalid")
                self.need(np.isfinite(velocity).all() and np.isfinite(a).all() and np.array_equal(saved["time_s"], time_axis), "Reconstruction finite/time check failed")
                self.need(np.array_equal(saved["x_m"], xx) and np.array_equal(saved["y_m"], yy), "Field/reference grid differs")
                for key, expected in (("checkpoint_sha256", row["checkpoint_sha256"]), ("reference_sha256", row["reference_sha256"]),
                                      ("evaluation_manifest_sha256", summary["manifest_sha256"]), ("mode", mode), ("arm", arm), ("iteration", row["iteration"])):
                    self.need(saved[key].item() == expected, "NPZ provenance differs: " + key)
                error = velocity-ref
                self.need(np.array_equal(saved["velocity_error_m_s"], error), "Saved velocity error differs from readback difference")
                rms = np.sqrt(np.sum(error**2, axis=-1).mean(axis=(0, 2, 3)))*1000
                time_rms = np.sqrt(np.sum(error**2, axis=-1).mean(axis=(2, 3)))*1000
                coefficient_rms = np.sqrt(.5*np.sum((a-ref_a)**2, axis=-1).mean(axis=0))*CASES[case]*1000
                expect = np.asarray(row["field"]["rms_mm_s"])
                discrepancy = float(np.max(np.abs(rms-expect)))
                max_error = max(max_error, discrepancy)
                for actual, target in zip(rms, expect):
                    self.number(actual, target, "NPZ grid RMS versus reported", atol=1e-10)
                for actual, target in zip(coefficient_rms, expect):
                    self.number(actual, target, "NPZ coefficient RMS versus reported", atol=1e-10)
                for actual, target in zip(time_rms.flat, np.asarray(row["field"]["time_rms_mm_s"]).flat):
                    self.number(actual, target, "NPZ time RMS versus reported", atol=1e-10)
                if mode == "primary":
                    ck = self.load(row["checkpoint_path"], row["checkpoint_sha256"])
                    for key in ("a", "eta_s", "eta_n"):
                        self.need(np.array_equal(saved[key], ck[key].double().numpy()), "Primary export coefficient differs from checkpoint")
                checked.append(dict(case=case, mode=mode, arm=arm, frames=96,
                    grid_rms_mm_s=rms.tolist(), coefficient_rms_mm_s=coefficient_rms.tolist(), max_report_difference_mm_s=discrepancy))
        self.result["field_exports"] = checked
        self.result["npz_files"] = 14
        self.result["figure_files"] = 36
        self.result["field_rms_max_abs_error_mm_s"] = max_error
        self.done("14个96帧NPZ和36图SHA、场/参考绑定、逐时刻与总RMS独立回读复算")

    def run(self):
        started = time.perf_counter()
        try:
            summary, plots = self.prerequisites()
            expected = self.gates(summary, plots)
            index = self.numeric(summary, expected)
            self.auxiliary(summary, index)
            self.exports(summary, plots, index)
            self.result["complete"] = self.result["passed"] = True
        except Exception as exc:
            self.result["failures"].append(dict(type=type(exc).__name__, message=str(exc)))
        self.result["wall_seconds"] = time.perf_counter()-started
        self.write()
        return self.result

    def write(self):
        folder = self.out / "preflight"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "FINAL_INDEPENDENT_AUDIT.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
        tmp.replace(path)
        r = self.result
        lines = ["潜变量时间转移第一轮：独立结果审计", "", "状态：" + ("全部通过" if r["passed"] else "未通过/尚未完成"),
                 "本工具未训练、未选解、未重新评分观察数据，未修改任何冻结科学文件。",
                 "逐元素冻结比较覆盖参数和buffer；V选解复核使用已记录的验证元数据，不替代重新运行似然。", ""]
        lines += ["通过：" + text for text in r["checks"]]
        lines += ["", f"数值比较：{r['numeric_comparisons']}；最大绝对差：{r['numeric_max_abs_error']:.12g}",
                  f"张量比较：{r['tensor_comparisons']}；元素总数：{r['tensor_elements_compared']}",
                  f"SHA检查次数：{len(r['hash_checks'])}"]
        if r.get("protocol_note"):
            lines += [r["protocol_note"]]
        for entry in r.get("quadrature_direction_reversals", []):
            lines.append("积分方向反转：" + json.dumps(entry, ensure_ascii=False))
        for failure in r["failures"]:
            lines.append("失败：" + failure["type"] + ": " + failure["message"])
        lines += ["", "限制：本审计核对已完成结果的一致性与来源，不构成新独立物理重复，也不验证ADF近似的统计充分性。"]
        (folder / "FINAL_INDEPENDENT_AUDIT.txt").write_text("\n".join(lines)+"\n", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    output = Auditor(args.out.resolve()).run()
    print(json.dumps({k: output[k] for k in ("complete", "passed", "numeric_comparisons", "numeric_max_abs_error", "failures")}, ensure_ascii=False))
    sys.exit(0 if output["passed"] else 1)
