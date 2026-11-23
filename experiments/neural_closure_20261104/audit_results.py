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
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parents[2] / "outputs/NeuralClosure4DVar_20261104"
OLD_OUT = HERE.parents[2] / "outputs/LatentTransition_20261104_v2"
CASES = {"vn0_0015m_s": .0015, "vn0_004m_s": .004}
SEEDS = (301030, 301031, 301032)
TRAIN_SEEDS = (411040, 411041, 411042)
ARMS = ("baseline", "closure", "probability", "joint")
NEW_ARMS = ARMS[1:]
MODES = ("primary", "terminal")
PAIRS = (("closure", "baseline"), ("probability", "baseline"), ("joint", "closure"), ("joint", "probability"), ("joint", "baseline"))
CROSSES = (("probability", "baseline"), ("joint", "baseline"), ("joint", "closure"))
CORE = ("normal_rms_mm_s", "superfluid_rms_mm_s", "heldout_free_nll", "heldout_cap_nll")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def training_composite(metrics, origin):
    """Independent arithmetic only; deliberately does not import the trainer."""
    if not all(math.isfinite(metrics[key]) for key in ("field", "free", "cap")):
        return float("inf")
    try:
        value = (metrics["field"]/max(origin["field"], 1e-12)
                 + math.exp(metrics["free"]-origin["free"])
                 + math.exp(metrics["cap"]-origin["cap"]))/3
    except OverflowError:
        return float("inf")
    return value if math.isfinite(value) else float("inf")


class Auditor:
    def __init__(self, out):
        self.out = out
        self.result = dict(version="independent-neural-closure-audit-v1", complete=False,
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

    def training_guard(self, stability, settings, label):
        """Verify saved guard arithmetic; no dynamical rollout is rerun here."""
        self.need(stability["case"] == "vn0_0065m_s", label+" validation-only guard case")
        self.need(isinstance(stability.get("passed"), bool), label+" stability decision")
        self.number(stability["growth_factor_limit"], settings["full_window_growth_factor"],
                    label+" growth threshold")
        if stability.get("reason") == "nonfinite":
            self.need(not stability["passed"] and 1 <= stability["failed_frame"] <= 95,
                      label+" recorded nonfinite rollout failure")
            return
        maximum = stability["max_abs_coefficient"]
        if stability.get("reason") == "fixed_amplitude_limit":
            self.need(not stability["passed"] and maximum > settings["full_window_amplitude_limit"],
                      label+" fixed amplitude rejection")
            return
        self.need(stability["frames"] == 96 and stability["process_increments_zero"],
                  label+" full-window guard scope")
        self.need(math.isfinite(maximum) and maximum <= settings["full_window_amplitude_limit"],
                  label+" fixed amplitude guard")
        denominator = np.maximum(stability["baseline_peak_modal_rms_per_fluid"],
                                 stability["initial_modal_rms_per_fluid"])
        ratio = np.asarray(stability["peak_modal_rms_per_fluid"])/np.maximum(denominator, 1e-12)
        self.tree(stability["peak_growth_ratio_per_fluid"], ratio.tolist(), label+" growth ratios")
        passed = bool((ratio <= settings["full_window_growth_factor"]).all())
        self.need(stability["passed"] == passed, label+" per-fluid growth decision")
        if not passed:
            self.need(stability["reason"] == "relative_per_fluid_growth_limit", label+" growth rejection reason")

    def training_history(self, record, selected, status, protocol):
        """Recompute the selected epoch from already-frozen metadata, not data."""
        last = self.load(record["last_path"], record["last_sha256"])
        arm, seed = record["arm"], record["seed"]
        label = f"training/{arm}/{seed}"
        settings = protocol["training_settings"]
        self.need(last["fingerprint"] == last["protocol_fingerprint"] == protocol["fingerprint"],
                  label+" last fingerprint")
        self.need(last["arm"] == arm and last["seed"] == seed, label+" last identity")
        self.need(selected["origin"] == last["origin"], label+" source lineage")
        trace, origin = last["trace"], last["origin_validation"]
        self.need([row["epoch"] for row in trace] == list(range(last["epoch"]+1)),
                  label+" complete contiguous epoch history")
        self.need(0 <= last["epoch"] <= settings["epochs"], label+" fixed epoch budget")
        self.tree(trace[0]["validation"], origin, label+" epoch-zero origin")
        self.need(trace[0]["selected"] and trace[0]["stability"]["passed"],
                  label+" valid epoch-zero selection")
        self.training_guard(trace[0]["stability"], settings, label+" epoch-zero")
        best, stale, stopped = trace[0], 0, False
        eligible_epochs, improved_epochs, rejected_unstable = [0], [0], []
        for row in trace:
            score = training_composite(row["validation"], origin)
            if math.isfinite(score):
                self.number(row["selection_score"], score, label+" composite")
            else:
                self.need(row["selection_score"] == float("inf"), label+" ineligible composite")
            if row["epoch"] == 0:
                self.number(score, 1., label+" epoch-zero score")
                continue
            self.need(not stopped, label+" training continued after fixed early-stop rule")
            eligible = math.isfinite(score)
            better = eligible and score < best["selection_score"]-settings["selection_min_delta"]
            self.need(row["eligible"] == eligible, label+" saved eligibility")
            stability = row["stability"]
            self.need((stability is not None) == better, label+" candidate-only stability check")
            if better:
                self.training_guard(stability, settings, label+" candidate")
                if not stability["passed"]:
                    rejected_unstable.append(row["epoch"])
            chosen = better and stability["passed"]
            self.need(row["selected"] == chosen, label+" selected flag")
            if eligible:
                eligible_epochs.append(row["epoch"])
            if chosen:
                best, stale = row, 0
                improved_epochs.append(row["epoch"])
            else:
                stale += 1
            stopped = row["epoch"] >= settings["minimum_epochs"] and stale >= settings["patience"]
        self.need(last["epoch"] == settings["epochs"] or stopped, label+" training completion")
        self.need(last["stopped_early"] == status["stopped_early"] == stopped,
                  label+" early-stop decision")
        self.need(last["stale"] == status["stale"] == stale and status["epoch"] == last["epoch"],
                  label+" terminal epoch/stale")
        self.need(best["epoch"] == selected["epoch"] == status["selected_epoch"],
                  label+" independently selected epoch")
        self.tree(selected["validation"], best["validation"], label+" selected validation")
        self.tree(selected["stability"], best["stability"], label+" selected guard")
        self.number(selected["selection_score"], best["selection_score"], label+" selected score")
        self.number(status["selection_score"], best["selection_score"], label+" status selected score")
        for key in ("net_state", "closure_state"):
            self.state(last["selected"][key], selected[key], label+" embedded selected "+key)
        for key in ("epoch", "arm", "seed", "fingerprint", "origin"):
            self.need(last["selected"][key] == selected[key], label+" embedded metadata "+key)
        for key in ("cpu_seconds", "wall_seconds"):
            self.need(last[key] >= 0, label+" nonnegative "+key)
            self.number(status[key], last[key], label+" cost "+key)
        return dict(arm=arm, seed=seed, terminal_epoch=last["epoch"], selected_epoch=best["epoch"],
                    selected_score=best["selection_score"], eligible_epochs=eligible_epochs,
                    improved_epochs=improved_epochs, rejected_unstable_epochs=rejected_unstable,
                    early_stopped=stopped, stale=stale,
                    cpu_seconds=last["cpu_seconds"], process_wall_seconds=last["wall_seconds"],
                    origin=selected["origin"], checkpoint_path=record["path"],
                    checkpoint_sha256=record["sha256"], terminal_sha256=record["last_sha256"])

    def training_costs(self, history, summary):
        """Count nine unique runs once, and expose D's B dependency separately."""
        index = {(row["arm"], row["seed"]): row for row in history}
        recorded = {(row["arm"], row["seed"]): row for row in summary["cost"]["training_runs"]}
        self.need(len(recorded) == len(history) == 9 and set(recorded) == set(index),
                  "Training cost record identities")
        for key, row in index.items():
            status = recorded[key]["status"]
            self.number(status["cpu_seconds"], row["cpu_seconds"], "reported training CPU")
            self.number(status["wall_seconds"], row["process_wall_seconds"], "reported training wall")
            self.need(recorded[key]["selected_epoch"] == row["selected_epoch"], "reported training epoch")
        def totals(rows):
            return dict(runs=len(rows), cpu_seconds=math.fsum(x["cpu_seconds"] for x in rows),
                        process_wall_seconds_sum=math.fsum(x["process_wall_seconds"] for x in rows))
        by_arm = {arm: totals([r for r in history if r["arm"] == arm]) for arm in NEW_ARMS}
        joint = []
        for seed in TRAIN_SEEDS:
            own, upstream = index[("joint", seed)], index[("closure", seed)]
            source = own["origin"]["closure_start"]
            self.need(Path(source["path"]).resolve() == Path(upstream["checkpoint_path"]).resolve()
                      and source["sha256"] == upstream["checkpoint_sha256"]
                      and source["epoch"] == upstream["selected_epoch"], "Joint upstream closure lineage")
            joint.append(dict(seed=seed, joint_own=totals([own]), prerequisite_closure=totals([upstream]),
                              standalone_lineage_total=totals([own, upstream]),
                              upstream_already_in_unique_new_training_total=True))
        costs = dict(unique_new_training=totals(history), by_arm_own_training=by_arm,
                     joint_lineages=joint, joint_own_total=by_arm["joint"],
                     joint_upstream_closure_total=by_arm["closure"],
                     joint_standalone_lineage_total=totals([r for r in history if r["arm"] in ("joint", "closure")]),
                     convention="Nine distinct new training runs counted once. Joint depends on already-counted same-seed closure; standalone joint cost includes it, global unique cost does not add it again.",
                     wall_time_scope="Sum of per-process measured training wall seconds, not concurrent pipeline elapsed time. CPU seconds are additive process CPU time.",
                     excluded_from_new_training_total=["historical baseline probability training", "historical reused field inversions", "discarded pilots", "new target inversions", "evaluation and audit"])
        self.result["training_selection_replay"] = history
        self.result["training_cost_accounting"] = costs
        self.done("独立重算9组训练组合分数、稳定门槛、选中epoch及早停；新训练成本逐运行计一次，joint上游closure单列")

    def prerequisites(self):
        summary_path = self.out / "NEURAL_EVALUATION_SUMMARY.json"
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
        p=read(self.out/"protocol.json");fp=p["fingerprint"]
        self.need(p["frozen"] and fingerprint({k:v for k,v in p.items() if k!="fingerprint"})==fp,"Protocol fingerprint mismatch")
        for kind in ("sources","inputs"):
            for path,expected in p[kind].items():self.checked_hash(path,expected)
        self.need(p["new_physics"]=={"nu_s":0.,"Fns_factor":1.},"Required physics changed")
        tp,fgp=self.out/"training/selection_frozen.json",self.out/"selection_frozen.json"
        tg,fg=read(tp),read(fgp)
        self.need(tg["all_required_selections_frozen"] and tg["protocol_fingerprint"]==fp,"Training gate incomplete")
        self.need(len(tg["runs"])==9 and {(r['arm'],r['seed']) for r in tg['runs']}=={(a,s) for a in NEW_ARMS for s in TRAIN_SEEDS},"Nine training selections required")
        self.need(fingerprint({k:v for k,v in tg.items() if k!="fingerprint"})==tg["fingerprint"],"Training gate fingerprint mismatch")
        self.need(fg["fingerprint"]==fp and fg["all_required_selections_frozen"] and fg["all_terminal_states_bound_before_evaluation"],"Field gate incomplete")
        self.checked_hash(tp,fg["training_gate_sha256"])
        self.need(len(fg["runs"])==24 and {(r['case'],r['seed'],r['arm']) for r in fg['runs']}=={(c,s,a) for c in CASES for s in SEEDS for a in ARMS},"24 field runs required")
        self.need(summary['protocol_fingerprint']==fp and not summary['selected_or_trained_during_evaluation'] and summary['terminal_preplanned'],"Evaluation scope changed")
        mp=self.out/'evaluation_manifest.json';self.checked_hash(mp,summary['manifest_sha256']);manifest=read(mp)
        self.checked_hash(tp,manifest['training_gate_sha256']);self.checked_hash(fgp,manifest['field_gate_sha256'])
        self.need(plots['evaluation_manifest_sha256']==summary['manifest_sha256'],"Plot binding mismatch")
        trained={};training_history=[]
        for tr in tg['runs']:
            ck=self.load(tr['path'],tr['sha256'])
            sp=Path(tr.get('status_path',str(Path(tr['path']).parent/'status.json')))
            self.checked_hash(sp,tr['status_sha256']);st=read(sp)
            epoch=tr['epoch'] if 'epoch' in tr else tr['selected_epoch']
            self.need(st['complete'] and ck['epoch']==epoch and ck['seed']==tr['seed'] and ck['arm']==tr['arm'] and ck['fingerprint']==fp,"Training selection identity mismatch")
            training_history.append(self.training_history(tr,ck,st,p))
            trained[(tr['arm'],tr['seed'])]=(tr,ck)
        self.training_costs(training_history,summary)
        # Import only after complete metadata gates; the auditor never scores observations.
        sys.path.insert(0,str(HERE))
        spec=importlib.util.spec_from_file_location('neural_audit_experiment',HERE/'experiment.py')
        E=importlib.util.module_from_spec(spec);sys.modules[spec.name]=E;spec.loader.exec_module(E)
        E.setup();E.verify_protocol()
        from closure_model import batched_step
        expected_rows={};selection_audit=[];replayed={};process_energy={}
        for r in fg['runs']:
            selected=self.load(r['path'],r['sha256']);last=self.load(r['last_path'],r['last_sha256'])
            case,seed,arm=r['case'],r['seed'],r['arm'];ts=TRAIN_SEEDS[SEEDS.index(seed)]
            status=read(Path(r['path']).parent/'status.json')
            self.need(status['complete'] and last['iteration']==status['iteration']==200 and last['network_steps']==0,"Incomplete target inverse or network updates")
            self.state(last['selected']['model_state'],selected['model_state'],'embedded primary')
            if arm=='baseline':
                src=selected['reuse_source'];self.need(src==last['reuse_source'] and src['arm']=='temporal' and src['case']==case and src['seed']==seed,'Baseline source mismatch')
                old=self.load(src['path'],src['sha256']);oldlast=self.load(src['last_path'],src['last_sha256'])
                self.state(selected['model_state'],old['model_state'],'baseline primary reuse');self.state(last['model_state'],oldlast['model_state'],'baseline terminal reuse')
                self.need(last['trace']==oldlast['trace'],'Historical trace modified')
                _,tr=E.trained_record(arm,seed)
                expected_net,expected_closure=old['net_state'],{}
            else:
                tr,tc=trained[(arm,ts)];expected_net,expected_closure=tc['net_state'],tc['closure_state']
            for ck in (selected,last,last['selected']):
                self.need(ck['arm']==arm and ck['seed']==seed and ck['fingerprint']==fp and ck['training_seed']==ts and ck['training_sha']==tr['sha256'],'Checkpoint model linkage mismatch')
                self.state(ck['net_state'],expected_net,'frozen probability '+arm)
                self.state(ck['closure_state'],expected_closure,'frozen closure '+arm)
                self.number(ck['U'],CASES[case],'checkpoint U')
            trace,warm=last['trace'],last['warm_validation']
            self.need([z['iteration'] for z in trace]==list(range(201)),'Complete 201-row inverse history required')
            best=trace[0];eligible_steps=[0]
            for z in trace[1:]:
                if 'validation' not in z:continue
                v=z['validation'];eligible=all(math.isfinite(v[k]) for k in ('total','free','cap')) and all(v[k]<=warm[k]+p['guard'] for k in ('free','cap'))
                chosen=eligible and v['total']<best['validation']['total']-p['selection_delta']
                self.need(z['eligible']==eligible and z['selected']==chosen,'Saved guard/selection mismatch')
                if eligible:eligible_steps.append(z['iteration'])
                if chosen:best=z
            self.need(best['iteration']==selected['iteration']==r['selected_iteration']==status['selected_iteration'],'Validation-selected step mismatch')
            self.tree(selected['validation_components'],best['validation'],'selected validation')
            d,_,model,net,_,_=E.make_context(CASES[case],seed,arm)
            self.need(not any(q.requires_grad for q in net.parameters()),'Probability weights trainable')
            if hasattr(model.op,'net'):
                self.need(not any(q.requires_grad for q in model.op.net.parameters()),'Mean-closure weights trainable')
            self.state(net.state_dict(),expected_net,'actual probability context')
            self.state(E.closure_state(model),expected_closure,'actual closure context')
            for mode,ck,pk,hk in (('primary',selected,'path','sha256'),('terminal',last,'last_path','last_sha256')):
                with torch.no_grad():
                    model.load_state_dict(ck['model_state']);a,es,en=model()
                    if mode=='primary':self.state(dict(a=a,eta_s=es,eta_n=en),{k:ck[k] for k in ('a','eta_s','eta_n')},'forward primary replay')
                    physical_op=model.op.base if hasattr(model.op,'net') else model.op
                    physical=batched_step(physical_op,a[:-1]);closed=batched_step(model.op,a[:-1]) if hasattr(model.op,'net') else physical
                    mean=closed-physical;residual=torch.stack((en,es),1);total=a[1:]-physical
                    self.need(float((total-mean-residual).abs().max())<1e-8,'Integrated closure decomposition identity')
                    replayed[(mode,case,seed,arm)]=(a.detach().cpu(),es.detach().cpu(),en.detach().cpu(),mean.detach().cpu(),total.detach().cpu())
                    process_energy[(mode,case,seed,arm)]=dict(normal=float(model.prior.whiten(en).square().sum()),superfluid=float((es/model.std[1]).square().sum()))
                expected_rows[(mode,case,seed,arm)]=dict(checkpoint_path=r[pk],checkpoint_sha256=r[hk],iteration=ck['iteration'],
                    validation=selected['validation_components'] if mode=='primary' else trace[-1]['validation'],training_seed=ts,
                    training_sha256=tr['sha256'],selected_iteration=selected['iteration'])
            selection_audit.append(dict(case=case,seed=seed,arm=arm,selected_iteration=selected['iteration'],eligible_steps=eligible_steps))
        self.replayed=replayed
        self.process_energy=process_energy
        self.result.update(protocol_fingerprint=fp,training_selections=9,field_runs=24,selection_replay=selection_audit,field_forward_replays=48)
        self.done('Nine training selections, 24 fields, network/closure freezing, historical reuse, V-selection and 48 forward field replays')
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
        for t,i in ((5,0),(50,45),(100,95)):
            for f,name in enumerate(('normal','superfluid')):
                values[f't{t}_{name}_rms_mm_s']=row['field']['time_rms_mm_s'][i][f]
        return values

    @staticmethod
    def descriptive(values):
        return dict(mean=math.fsum(values)/len(values), min=min(values), max=max(values),
                    by_seed={str(seed): value for seed, value in zip(SEEDS, values)})

    def numeric(self, summary, expected_rows):
        rows = summary["runs"]
        index = {(r["mode"], r["case"], r["seed"], r["arm"]): r for r in rows}
        self.need(len(rows) == 48 and set(index) == set(expected_rows), "Exactly 48 unique primary/terminal rows required")
        self.tree(read(self.out / "tables/individual_runs.json"), rows, "individual rows mirror")
        references = {c:self.load(v['path'],v['sha256'])['a'].double().cpu()
                      for c,v in summary['reference_sources'].items()}
        for key, row in index.items():
            a,es,en,mean,total=self.replayed[key]
            ref=references[row['case']]
            error=(a-ref)*(CASES[row['case']]*1000)
            time_rms=(.5*error.square().sum(-1)).sqrt()
            full_rms=(.5*error.square().sum(-1).mean(0)).sqrt()
            rel=(error.square().sum(-1).mean(0)/(ref*(CASES[row['case']]*1000)).square().sum(-1).mean(0)).sqrt()
            for name,value in (('rms_mm_s',full_rms.tolist()),('time_rms_mm_s',time_rms.tolist()),
                               ('initial_rms_mm_s',time_rms[0].tolist()),('relative_fluctuation_rms',rel.tolist())):
                self.tree(row['field'][name],value,'all-seed replay field '+name)
            for name,value in (('integrated_mean',mean),('residual',torch.stack((en,es),1)),('total_defect',total)):
                tt=(value.square().sum(-1)/2).sqrt()*CASES[row['case']]*1000
                self.tree(row['process'][name]['time_rms_mm_s'],tt.tolist(),'integrated process time '+name)
                self.tree(row['process'][name]['rms_mm_s'],tt.square().mean(0).sqrt().tolist(),'integrated process RMS '+name)
            self.need(row['process']['decomposition_max_coefficient_error']<1e-8,'Process decomposition tolerance exceeded')
            self.tree(row['process']['residual_whitened_energy'],self.process_energy[key],'residual whitened prior energy')
            cache=self.out/'evaluation'/row['mode']/row['case']/f"seed_{row['seed']}"/f"{row['arm']}.json"
            self.tree(read(cache),row,'cache matches completed summary')
            ex = expected_rows[key]
            for name in ("checkpoint_sha256", "iteration", "training_seed", "training_sha256"):
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
        self.done("48行、逐PID概率汇总、主/末态均值极值配对差、20/28阶方向独立重算")
        return index

    @staticmethod
    def flat_scores(score):
        out = {f"{k}_nll": score[k] for k in ("total", "free", "cap")}
        for state in ("free", "cap"):
            out[state+"_coverage"] = score["coverage"][state]
            out[state+"_spread_um"] = score["spread"][state]
        return out

    def auxiliary(self,summary,index):
        aux=summary['auxiliary'];self.need(aux['complete'] and aux['fixed_mode']=='primary' and not aux['fitting_or_selection'],'Auxiliary incomplete')
        self.tree(read(self.out/'AUXILIARY_PROBABILITY_ATTRIBUTION.json'),aux,'auxiliary mirror')
        self.need(len(aux['rows'])==18 and {(r['case'],r['seed'],r['arm'],r['fixed_arm']) for r in aux['rows']}=={(c,s,a,b) for c in CASES for s in SEEDS for a,b in CROSSES},'18 cross-query rows required')
        error=0.
        for r in aux['rows']:
            c,s,a,b=r['case'],r['seed'],r['arm'],r['fixed_arm'];old=index['primary',c,s,b];own=index['primary',c,s,a]
            self.need(r['binding']==dict(manifest_sha256=summary['manifest_sha256'],fixed_checkpoint_sha256=old['checkpoint_sha256'],own_checkpoint_sha256=own['checkpoint_sha256']),'Auxiliary provenance mismatch')
            for degree in (20,28):
                z=r['by_degree'][str(degree)];o=old['observations']['H'] if degree==20 else old['heldout_degree28'];n=own['observations']['H'] if degree==20 else own['heldout_degree28']
                self.tree(z['original_probability_fixed_query'],o,'aux original score');self.tree(z['new_probability_own_query'],n,'aux new own score')
                q,x,y=[self.flat_scores(v) for v in (o,z['new_probability_fixed_query'],n)]
                for k in q:
                    values={'probability_change_at_fixed_query':x[k]-q[k],'query_change_at_new_probability':y[k]-x[k],'total_change':y[k]-q[k]}
                    for name,value in values.items():self.number(z['components'][name][k],value,'auxiliary delta '+k)
                    err=abs(values['probability_change_at_fixed_query']+values['query_change_at_new_probability']-values['total_change']);error=max(error,err)
                    self.number(values['probability_change_at_fixed_query']+values['query_change_at_new_probability'],values['total_change'],'aux telescope')
            self.tree(r['own_minus_fixed_field_rms_mm_s'],(np.array(own['field']['rms_mm_s'])-old['field']['rms_mm_s']).tolist(),'aux field delta')
        self.result.update(auxiliary_rows=18,auxiliary_max_telescoping_error=error)
        self.done('18 fixed-query substitutions, all degree20/28 components and additive identities independently recomputed')

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
        self.need(len(fields) == 16 and len(refs) == 2 and len(figures) == 44 and len(artifacts) == 62,
                  "Expected sixteen reconstruction NPZ, two reference NPZ, forty-four PNG/SVG")
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
                replay_a,replay_es,replay_en,replay_mean,replay_total=self.replayed[(mode,case,SEEDS[0],arm)]
                for name,value in (('a',replay_a),('eta_s',replay_es),('eta_n',replay_en),
                                   ('integrated_closure_mean',replay_mean),
                                   ('residual_increment',torch.stack((replay_en,replay_es),1)),
                                   ('total_process_defect',replay_total)):
                    self.need(np.allclose(saved[name],value.numpy(),rtol=1e-12,atol=1e-12),'Export versus independent replay '+name)
                checked.append(dict(case=case, mode=mode, arm=arm, frames=96,
                    grid_rms_mm_s=rms.tolist(), coefficient_rms_mm_s=coefficient_rms.tolist(), max_report_difference_mm_s=discrepancy))
        self.result["field_exports"] = checked
        self.result["npz_files"] = 18
        self.result["figure_files"] = 44
        self.result["field_rms_max_abs_error_mm_s"] = max_error
        self.done("18个96帧NPZ和44图SHA、场/参考绑定、逐时刻与总RMS独立回读复算")

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
        lines = ["神经闭合与概率四管线第一轮：独立结果审计", "", "状态：" + ("全部通过" if r["passed"] else "未通过/尚未完成"),
                 "本工具未训练、未选解、未重新评分观察数据，未修改任何冻结科学文件。",
                 "逐元素冻结比较覆盖参数和buffer；V选解复核使用已记录的验证元数据，不替代重新运行似然。", ""]
        lines += ["通过：" + text for text in r["checks"]]
        lines += ["", f"数值比较：{r['numeric_comparisons']}；最大绝对差：{r['numeric_max_abs_error']:.12g}",
                  f"张量比较：{r['tensor_comparisons']}；元素总数：{r['tensor_elements_compared']}",
                  f"SHA检查次数：{len(r['hash_checks'])}"]
        if r.get("training_selection_replay"):
            lines += ["", "训练选解独立复算（使用冻结历史元数据，未重跑训练或验证似然）："]
            for row in r["training_selection_replay"]:
                lines.append(f"{row['arm']} seed{row['seed']}: selected epoch={row['selected_epoch']}, "
                             f"terminal epoch={row['terminal_epoch']}, score={row['selected_score']:.10g}, "
                             f"early_stop={row['early_stopped']}, unstable_candidates={row['rejected_unstable_epochs']}")
        if r.get("training_cost_accounting"):
            cost = r["training_cost_accounting"]
            total = cost["unique_new_training"]
            lines += ["", "新训练成本（各运行耗时相加，不等于并发流程的实际经过时间）：",
                      f"9个唯一新训练：CPU={total['cpu_seconds']/60:.6f} min；运行墙钟之和={total['process_wall_seconds_sum']/60:.6f} min。"]
            for arm, value in cost["by_arm_own_training"].items():
                lines.append(f"{arm} 自身训练：CPU={value['cpu_seconds']/60:.6f} min；运行墙钟之和={value['process_wall_seconds_sum']/60:.6f} min。")
            dep, standalone = cost["joint_upstream_closure_total"], cost["joint_standalone_lineage_total"]
            lines += [f"joint前置closure：CPU={dep['cpu_seconds']/60:.6f} min；运行墙钟之和={dep['process_wall_seconds_sum']/60:.6f} min，已包含在唯一新训练总量中，不再重复相加。",
                      f"若单独从旧模型获得joint，closure+joint依赖链：CPU={standalone['cpu_seconds']/60:.6f} min；运行墙钟之和={standalone['process_wall_seconds_sum']/60:.6f} min。",
                      "以上不含历史概率训练/复用反演、丢弃pilot、新目标反演及评价审计；目标反演成本见主评价的独立记录。"]
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
