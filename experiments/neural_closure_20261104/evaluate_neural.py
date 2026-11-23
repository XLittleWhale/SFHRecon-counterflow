"""Gate-bound four-pipeline learned-closure and probability evaluation.

No target H response is scored and no target reference is opened until all
all nine new trained models and all twenty-four primary/terminal field states pass
the metadata, byte-hash and frozen-parameter checks. This script performs no
optimization or model selection. Reused baseline checkpoints are independently
rescored under the original v2 temporal likelihood after the global gate.
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
OUT = HERE.parents[2] / "outputs/NeuralClosure4DVar_20261104"
HELPER = HERE.parent / "joint_probability_20261103/evaluate_joint.py"
CASES = (("vn0_0015m_s", .0015), ("vn0_004m_s", .004))
SEEDS = (301030, 301031, 301032)
TRAIN_SEEDS = (411040, 411041, 411042)
ARMS = ("baseline", "closure", "probability", "joint")
NEW_ARMS = ARMS[1:]
MODES = ("primary", "terminal")
PAIRS = (("closure", "baseline"), ("probability", "baseline"), ("joint", "closure"), ("joint", "probability"), ("joint", "baseline"))
CROSSES = (("probability", "baseline"), ("joint", "baseline"), ("joint", "closure"))
DEGREES = (20, 28)
KEYS = ("total", "free", "cap")
REPRESENTATIVE_SEED = SEEDS[0]
VERSION = "neural-closure-four-pipeline-evaluation-v1"
COLORS = dict(baseline="#777777", closure="#1f77b4", probability="#2ca02c", joint="#d95f02")


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
    """Metadata only: all nine new training selections and 24 fields required."""
    tp, fp = out / "training/selection_frozen.json", out / "selection_frozen.json"
    if not tp.is_file() or not fp.is_file():
        raise RuntimeError("Nine training selections and all 24 primary/terminal fields must be frozen")
    training, fields = read(tp), read(fp)
    tr, fr = training.get("runs", []), fields.get("runs", [])
    if (training.get("all_required_selections_frozen") is not True or len(tr) != 9
            or {(r["arm"], int(r["seed"])) for r in tr} != {(a,s) for a in NEW_ARMS for s in TRAIN_SEEDS}):
        raise RuntimeError("Incomplete nine-model training gate")
    expected = {(c,s,a) for c,_ in CASES for s in SEEDS for a in ARMS}
    if (fields.get("all_required_selections_frozen") is not True
            or fields.get("all_terminal_states_bound_before_evaluation") is not True
            or len(fr) != 24 or {(r["case"],int(r["seed"]),r["arm"]) for r in fr} != expected):
        raise RuntimeError("Incomplete 24-field selected/terminal gate")
    for r in tr + fr:
        status = read(Path(r.get("status_path", str(Path(r["path"]).parent / "status.json"))))
        if status.get("complete") is not True:
            raise RuntimeError("Incomplete status: " + r["path"])
        if r in fr and (status.get("iteration") != 200 or status.get("network_steps") != 0):
            raise RuntimeError("Incomplete field budget or target network updates")
    return training, fields


def selected_epoch(record):
    return int(record["epoch"] if "epoch" in record else record["selected_epoch"])


def bind_gates(E, T, out, training, fields):
    protocol = E.verify_protocol()
    E.training_gate()
    sources = {str(Path(p).resolve()): h for p,h in protocol["sources"].items()}
    for path in (Path(__file__), HELPER):
        if sources.get(str(path.resolve())) != sha(path):
            raise RuntimeError("Evaluator/helper must be prospectively source-bound: " + str(path))
    if (protocol["evaluation_degree"] != 20 or protocol["evaluation_sensitivity_degree"] != 28
            or fields["fingerprint"] != protocol["fingerprint"]
            or training["protocol_fingerprint"] != protocol["fingerprint"]
            or fields["training_gate_sha256"] != sha(out / "training/selection_frozen.json")):
        raise RuntimeError("Protocol/global-gate mismatch")
    states = {}
    for tr in training["runs"]:
        ck = load_hashed(E, tr["path"], tr["sha256"])
        sp = Path(tr.get("status_path", str(Path(tr["path"]).parent / "status.json")))
        status = read(sp)
        if tr.get("status_sha256") != sha(sp):
            raise RuntimeError("Training status must be hash-bound")
        if (ck["arm"] != tr["arm"] or ck["seed"] != tr["seed"] or ck["epoch"] != selected_epoch(tr)
                or ck["fingerprint"] != protocol["fingerprint"] or status["fingerprint"] != protocol["fingerprint"]):
            raise RuntimeError("Training identity mismatch")
        if not {"net_state", "closure_state"} <= set(ck):
            raise RuntimeError("Training must save full probability and closure states")
        states[(tr["arm"],int(tr["seed"]))] = (tr,ck)
    records=[]
    for r in fields["runs"]:
        selected=load_hashed(E,r["path"],r["sha256"])
        last=load_hashed(E,r["last_path"],r["last_sha256"])
        seed,arm=int(r["seed"]),r["arm"]
        ts=TRAIN_SEEDS[SEEDS.index(seed)]
        if arm == "baseline":
            source=selected["reuse_source"]
            if source != last["reuse_source"] or source["arm"] != "temporal" or source["case"] != r["case"] or int(source["seed"]) != seed:
                raise RuntimeError("Baseline must reuse matching v2 temporal states")
            old=load_hashed(E,source["path"],source["sha256"])
            oldlast=load_hashed(E,source["last_path"],source["last_sha256"])
            for new,prior in ((selected,old),(last,oldlast)):
                same_state(new["model_state"],prior["model_state"],"baseline field reuse")
                same_state(new["net_state"],prior["net_state"],"baseline probability reuse")
            if last["trace"] != oldlast["trace"] or selected["iteration"] != old["iteration"]:
                raise RuntimeError("Historical baseline trace or selection changed")
            tr0=E.trained_record(arm,seed)
            tr=tr0[-1] if isinstance(tr0,tuple) else tr0
            tc=load_hashed(E,tr["path"],tr["sha256"])
            expected_net,expected_closure=old["net_state"],{}
            epoch=int(tc["epoch"])
        else:
            tr,tc=states[(arm,ts)]
            expected_net,expected_closure=tc["net_state"],tc["closure_state"]
            epoch=int(tc["epoch"])
        if (selected["iteration"] != r["selected_iteration"] or last["iteration"] != 200
                or last["network_steps"] != 0 or last["trace"][-1]["iteration"] != 200):
            raise RuntimeError("Field selection/terminal budget mismatch")
        same_state(last["selected"]["model_state"],selected["model_state"],"embedded primary field")
        for ck in (selected,last,last["selected"]):
            if (ck["arm"] != arm or ck["seed"] != seed or ck["fingerprint"] != protocol["fingerprint"]
                    or ck["training_sha"] != tr["sha256"] or ck["training_seed"] != ts
                    or abs(float(ck["U"])-dict(CASES)[r["case"]]) > 1e-14):
                raise RuntimeError("Field/model/protocol identity mismatch")
            same_state(ck["net_state"],expected_net,"frozen probability "+arm)
            same_state(ck["closure_state"],expected_closure,"frozen mean closure "+arm)
        records.append(dict(r,training_seed=ts,training_epoch=epoch,training_sha256=tr["sha256"],
            wall_seconds=0. if arm=="baseline" else float(last["wall_seconds"]),
            cpu_seconds=0. if arm=="baseline" else float(last["cpu_seconds"]),
            historical_wall_seconds=float(last["wall_seconds"]) if arm=="baseline" else None,
            historical_cpu_seconds=float(last["cpu_seconds"]) if arm=="baseline" else None))
    manifest=dict(version=VERSION,protocol_fingerprint=protocol["fingerprint"],
        training_gate_sha256=sha(out/"training/selection_frozen.json"),field_gate_sha256=sha(out/"selection_frozen.json"),
        evaluator_sha256=sha(__file__),helper_sha256=sha(HELPER),training_runs=training["runs"],runs=records,
        all_selected_and_terminal_checkpoints_bound_before_target_evaluation=True,
        target_H_scoring_or_reference_reads_before_global_gate=False,primary_degree=20,sensitivity_degree=28,
        terminal_iteration=200,representative_seed=REPRESENTATIVE_SEED,
        physical_query_convention="Learned mean enters RHS; subtract only residual endpoint innovations in particle query",
        comparison_scope="Four trained pipelines, not a strict factorial or a causal decomposition")
    path=out/"evaluation_manifest.json"
    if path.exists() and read(path)!=manifest:
        raise RuntimeError("Existing evaluation manifest differs")
    if not path.exists():dump(path,manifest)
    return protocol,protocol,records,sha(path)


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
    for t, index in ((5, 0), (50, 45), (100, 95)):
        for i, name in enumerate(("normal", "superfluid")):
            values[f"t{t}_{name}_rms_mm_s"] = row["field"]["time_rms_mm_s"][index][i]
    return values


def process_decomposition(model, a, es, en):
    """Exact discrete integrated mean contribution, not instantaneous C*DT."""
    from closure_model import batched_step
    op = model.op
    physical_op = op.base if hasattr(op, "net") else op
    physical = batched_step(physical_op, a[:-1])
    closed = batched_step(op, a[:-1]) if hasattr(op, "net") else physical
    mean = closed-physical
    residual = torch.stack((en, es), dim=1)
    total = a[1:]-physical
    error = float((total-mean-residual).abs().max())
    if not np.isfinite(error) or error > 1e-8:
        raise RuntimeError(f"Integrated closure/residual decomposition mismatch: {error}")
    def norms(x):
        time_rms = (x.square().sum(-1)/2).sqrt()*op.U*1000
        return dict(rms_mm_s=time_rms.square().mean(0).sqrt().cpu().tolist(),
                    time_rms_mm_s=time_rms.cpu().tolist())
    result = dict(integrated_mean=norms(mean), residual=norms(residual), total_defect=norms(total),
        decomposition_max_coefficient_error=error,
        residual_whitened_energy=dict(normal=float(model.prior.whiten(en).square().sum()),
                                     superfluid=float((es/model.std[1]).square().sum())),
        definition="mean=Phi_closed(a_k)-Phi_phys(a_k); total=a_{k+1}-Phi_phys(a_k)=mean+residual",
        query_convention="Particle query subtracts residual eta only; deterministic mean remains in acceleration")
    arrays = dict(integrated_closure_mean=mean.detach().cpu().numpy(),
                  residual_increment=residual.detach().cpu().numpy(),
                  total_process_defect=total.detach().cpu().numpy())
    return result, arrays


def descriptive(values):
    return dict(mean=float(np.mean(values)), min=float(np.min(values)), max=float(np.max(values)),
                by_seed=dict(zip(map(str, SEEDS), map(float, values))))


def summarize(rows, degree=20):
    index = {(r["case"], int(r["seed"]), r["arm"]): metrics(r, degree) for r in rows}
    if len(rows) != 24 or set(index) != {(c, s, a) for c, _ in CASES for s in SEEDS for a in ARMS}:
        raise RuntimeError("Each evaluation mode must have exactly twenty-four rows")
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


def attribution(E,J,rows,out,binding):
    index={(r["mode"],r["case"],r["seed"],r["arm"]):r for r in rows}
    result=[]
    for case,U in CASES:
        for seed in SEEDS:
            for arm,fixed_arm in CROSSES:
                fixed_row=index[("primary",case,seed,fixed_arm)]
                own=index[("primary",case,seed,arm)]
                fixed_ck=load_hashed(E,fixed_row["checkpoint_path"],fixed_row["checkpoint_sha256"])
                fixed={k:fixed_ck[k] for k in ("a","eta_s","eta_n")}
                d,_,model,net,cal,force=E.make_context(U,seed,arm)
                own_ck=load_hashed(E,own["checkpoint_path"],own["checkpoint_sha256"])
                check_network(net,own_ck,arm)
                same_state(E.closure_state(model),own_ck["closure_state"],"auxiliary frozen closure")
                key=dict(manifest_sha256=binding,fixed_checkpoint_sha256=fixed_row["checkpoint_sha256"],
                         own_checkpoint_sha256=own["checkpoint_sha256"])
                path=out/"auxiliary"/case/f"seed_{seed}"/f"{arm}_at_{fixed_arm}.json"
                item=read(path) if path.exists() else None
                if item is None or item.get("binding")!=key:
                    by_degree={}
                    for degree in DEGREES:
                        old=fixed_row["observations"]["H"] if degree==20 else fixed_row["heldout_degree28"]
                        new=own["observations"]["H"] if degree==20 else own["heldout_degree28"]
                        cross=J.checked_score(E,fixed,d,net,cal,force,"H",degree)
                        b,x,n=map(flatten,(old,cross,new))
                        components=dict(probability_change_at_fixed_query={k:x[k]-b[k] for k in b},
                            query_change_at_new_probability={k:n[k]-x[k] for k in b},total_change={k:n[k]-b[k] for k in b})
                        for k in b:
                            if abs(components["probability_change_at_fixed_query"][k]+components["query_change_at_new_probability"][k]-components["total_change"][k])>1e-10:
                                raise RuntimeError("Auxiliary telescoping mismatch")
                        by_degree[str(degree)]=dict(original_probability_fixed_query=old,
                            new_probability_fixed_query=cross,new_probability_own_query=new,components=components)
                    item=dict(case=case,seed=seed,arm=arm,fixed_arm=fixed_arm,binding=key,by_degree=by_degree,
                        own_minus_fixed_field_rms_mm_s=(np.asarray(own["field"]["rms_mm_s"])-fixed_row["field"]["rms_mm_s"]).tolist())
                    dump(path,item)
                result.append(item)
                print("ATTRIBUTION",case,seed,arm,"AT",fixed_arm,flush=True)
    if len(result)!=18:raise RuntimeError("Exactly eighteen cross-query comparisons required")
    return dict(complete=True,fixed_mode="primary",rows=result,fitting_or_selection=False,
        descriptive_path_dependent_not_causal=True,
        fixed_query="All a and residual eta_s/eta_n fixed; conditional particle emission query unchanged")


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
                ax.set_xticks(range(4), ARMS, rotation=18)
                ax.set_title(f"{U*1000:g} mm/s | {key.replace('_', ' ')}", fontsize=9)
                ax.grid(axis="y", alpha=.2)
        fig.suptitle(f"Neural closure / probability pipelines | {mode} | three paired seeds")
        fig.supxlabel("RMS: mm/s. H: causal one-step conditional predictions, current endpoint scored before filtering. All seeds use the same physical data.", fontsize=8)
        plot_save(fig, out / "figures" / mode / "overview", plt, artifacts)
        for case, U in CASES:
            fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
            for fluid, title in enumerate(("Normal", "Superfluid")):
                for arm in ARMS:
                    values = np.asarray([index[(mode, case, s, arm)]["field"]["time_rms_mm_s"] for s in SEEDS])[:, :, fluid]
                    axs[fluid].plot(np.arange(5, 101), values.mean(0), label=arm, color=COLORS[arm])
                    axs[fluid].fill_between(np.arange(5, 101), values.min(0), values.max(0), color=COLORS[arm], alpha=.12)
                axs[fluid].set(title=title, xlabel="Physical time [ms]", ylabel="Velocity-vector RMS error [mm/s]")
                axs[fluid].grid(alpha=.2)
            axs[0].legend()
            fig.suptitle(f"{U*1000:g} mm/s | {mode} | all 96 frames; mean and seed range")
            plot_save(fig, out / "figures" / mode / f"{case}_time_rms", plt, artifacts)
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
                same_state(E.closure_state(model), ck["closure_state"], "export closure")
                a, es, en = [x.detach().cpu() for x in model()]
                process, process_arrays = process_decomposition(model, a, es, en)
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
                    evaluation_manifest_sha256=result["manifest_sha256"], iteration=row["iteration"], arm=arm, mode=mode,
                    **process_arrays)
                with np.load(p) as saved:
                    if saved["velocity_m_s"].shape != (96, 2, 40, 40, 2) or not np.array_equal(saved["velocity_m_s"], value):
                        raise RuntimeError("96-frame NPZ readback mismatch")
                artifacts.append(dict(path=str(p.resolve()), sha256=sha(p), kind="field_npz", frames=96,
                    case=case, mode=mode, arm=arm, grid_vs_summary_rms_max_mm_s=delta, npz_readback_exact=True,
                    process_decomposition_max_coefficient_error=process["decomposition_max_coefficient_error"]))
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
                    fig, axs = plt.subplots(3, len(labels), figsize=(16 if kind == "speed" else 13, 9), layout="constrained")
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


def write_report(result,out):
    lines=["神经闭合与冻结概率四管线：完整第一轮结果", "",
        "baseline=原B过程+v2 temporal概率；closure=神经闭合+冻结旧概率；probability=原B过程+重训概率；joint=神经闭合与概率联合重训。",
        "closure和joint的闭合权重允许不同，因此四管线不是严格2×2因果实验。baseline复用六组v2 temporal；另18组新反演。",
        "保持nu_s=0、Fns系数1和冻结B残余先验；神经均值进入演化RHS。粒子查询只扣残余endpoint innovation，神经均值会进入粒子加速度：这是明确的物理模型改动。",
        "目标网络及闭合网络完全冻结，仅反演初始场和残余过程增量。九个新训练选解与全部24场主/末态冻结后才评分H/reference。",
        "主选用既定V每态initial+0.01保护与balanced改善；固定200步另报，不能按H或真值替换。",
        "RMS单位mm/s，N40/K12，5–100ms；宽度为sqrt(trace displacement covariance)，单位μm。覆盖率以90%为目标。",
        "H是同窗因果一步条件预测，允许同一H粒子先前已观测位移，不是未来无观测滚动。三种子共享物理数据，不是独立物理重复。", ""]
    for mode in MODES:
        lines += [mode,"case | arm | normal RMS | super RMS | H free NLL | H cap NLL | coverage free/cap | spread free/cap"]
        for case,_ in CASES:
            item=result["summary"][mode]["20"][case]
            for arm in ARMS:
                v=lambda k:item["arms"][arm][k]["mean"]
                lines.append(f"{case} | {arm} | {v('normal_rms_mm_s'):.8g} | {v('superfluid_rms_mm_s'):.8g} | {v('heldout_free_nll'):.8g} | {v('heldout_cap_nll'):.8g} | {v('heldout_free_coverage'):.6f}/{v('heldout_cap_coverage'):.6f} | {v('heldout_free_spread_um'):.8g}/{v('heldout_cap_spread_um'):.8g}")
                lines.append("  5/50/100ms RMS n/s: "+"; ".join(f"{t}ms={v(f't{t}_normal_rms_mm_s'):.8g}/{v(f't{t}_superfluid_rms_mm_s'):.8g}" for t in (5,50,100)))
                rr=[z for z in result['runs'] if z['mode']==mode and z['case']==case and z['arm']==arm]
                lines.append("  integrated mean / residual / total defect RMS n/s [mm/s]: "+"; ".join(
                    name+"="+"/".join(f"{x:.8g}" for x in np.mean([z['process'][name]['rms_mm_s'] for z in rr],axis=0))
                    for name in ('integrated_mean','residual','total_defect')))
            for pair,z in item["paired_changes"].items():
                lines.append(pair+": "+"; ".join(f"{k}={z['metrics'][k]['mean']:+.8g}" for k in ("normal_rms_mm_s","superfluid_rms_mm_s","heldout_free_nll","heldout_cap_nll"))+f"; four metrics improve={z['joint_four_metric_improvement_count']}/3")
        lines.append("")
    lines += ["20/28阶配对方向："]
    for z in result["quadrature_directions"]:
        lines.append(f"{z['mode']} {z['case']} {z['pair']} {z['metric']}: {z['delta20']:+.8g}/{z['delta28']:+.8g}, reverse={z['sign_reversal']}")
    lines += ["", "辅助归因：固定完整查询(a,残余eta_s,eta_n)后换概率；再固定新概率换查询。后者同时包含场与残余创新，不是纯速度场因果效应。"]
    for case,_ in CASES:
        for arm,fixed_arm in CROSSES:
            rr=[z for z in result["auxiliary"]["rows"] if z['case']==case and z['arm']==arm and z['fixed_arm']==fixed_arm]
            for kind in ("free_nll","cap_nll"):
                cs=("probability_change_at_fixed_query","query_change_at_new_probability","total_change")
                lines.append(f"{case} {arm} vs {fixed_arm} {kind}: "+"; ".join(f"{c}={np.mean([z['by_degree']['20']['components'][c][kind] for z in rr]):+.8g}" for c in cs))
    lines += ["", "成本："]
    for row in result['cost']['training_runs']:
        lines.append(json.dumps(row,ensure_ascii=False))
    for arm in ARMS:
        rr=[z for z in result['cost']['field_runs'] if z['arm']==arm]
        lines.append(f"{arm}: new wall={sum(z['wall_seconds'] for z in rr)/60:.4f}min, CPU={sum(z['cpu_seconds'] for z in rr)/60:.4f}min")
    lines += ["", "闭合均值贡献使用Phi_closed(a_k)-Phi_phys(a_k)的精确离散积分差，非瞬时闭合乘DT。均值与残余向量可加，二者RMS不可直接相加。",
        "限制：旧B残余先验并未因学习均值重新校准；context偏移、场/闭合/隐变量补偿及ADF近似仍存在。",
        "不能将概率验证改善、同场NLL改善或过程能量下降直接当作流场改善。初场与后期误差分别报告。",
        "目标工况和H在历史开发已被探索；没有通过本轮得到全新盲测、场后验不确定性或CF迁移结论。",
        "没有在评价期间训练、调参、改选解或启动下一轮。完整逐种子、逐PID、逐帧、主/末态差和来源见JSON。", ""]
    text="\n".join(lines)
    (out/"NEURAL_EVALUATION_REPORT.txt").write_text(text,encoding="utf8")
    (out/"CONCLUSIONS.txt").write_text(text,encoding="utf8")


@torch.no_grad()
def evaluate(out=OUT, make_plots=True):
    begun = time.perf_counter()
    out = Path(out).resolve()
    training, fields = cheap_gate(out)
    sys.path.insert(0, str(HERE))
    E = module("latent_evaluation_experiment", HERE / "experiment.py")
    T = E
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
                if hasattr(model.op, "net") and any(p.requires_grad for p in model.op.net.parameters()):
                    raise RuntimeError("Target mean-closure network must be frozen")
                same_state(E.closure_state(model), ck["closure_state"], "replayed frozen closure")
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
                    process, _ = process_decomposition(model, a, es, en)
                    row = dict(case=case, U=U, seed=seed, arm=arm, mode=mode, iteration=int(ck["iteration"]),
                        primary_selected_iteration=record["selected_iteration"], checkpoint_path=str(path), checkpoint_sha256=digest_ck,
                        training_seed=record["training_seed"], training_epoch=record["training_epoch"],
                        training_sha256=record["training_sha256"], target_network_updates=0,
                        reference_path=str(rp), reference_sha256=digest, replay_V_max_difference=delta, cache_key=key,
                        field=J.field_metrics(a, reference, U), process=process, observations=observations, heldout_degree28=higher,
                        quadrature=J.quadrature_difference(observations["H"], higher))
                    dump(cache, row)
                rows.append(row)
                print("EVALUATED", mode, case, seed, arm, "step", ck["iteration"], flush=True)
    if len(rows) != 48:
        raise RuntimeError("Both complete modes must have 48 rows total")
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
        costs.append(dict(arm=r["arm"], seed=r["seed"], selected_epoch=selected_epoch(r),
            closure_state_elements=sum(v.numel() for v in ck["closure_state"].values()),
            probability_state_elements=sum(v.numel() for v in ck["net_state"].values()),
            status=read(r.get("status_path", str(Path(r["path"]).parent/"status.json")))))
    result = dict(version=VERSION, complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.perf_counter()-begun, protocol_fingerprint=protocol["fingerprint"], manifest_sha256=binding,
        reference_sources=references, selected_or_trained_during_evaluation=False, terminal_preplanned=True,
        target_H_or_reference_before_global_gate=False, runs=rows, summary=summaries,
        terminal_minus_primary=changes, quadrature_directions=directions, auxiliary=auxiliary,
        cost=dict(training_runs=costs, field_runs=records), prediction_scope="Causal one-step same-window endpoint prediction, previous observed endpoints allowed; no future conditioning")
    dump(out / "tables/individual_runs.json", rows)
    dump(out / "AUXILIARY_PROBABILITY_ATTRIBUTION.json", auxiliary)
    dump(out / "NEURAL_EVALUATION_SUMMARY.json", result)
    write_report(result, out)
    if make_plots:
        plots(E, J, result, out)
    print("EVALUATION COMPLETE", out / "NEURAL_EVALUATION_REPORT.txt", flush=True)
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
