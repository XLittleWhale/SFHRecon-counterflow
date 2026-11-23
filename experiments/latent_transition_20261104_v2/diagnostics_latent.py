"""Post-selection, non-target diagnostics; never changes scientific selections.

This report helper is deliberately outside the frozen fitting source set.
It requires all three transition selections to be frozen and verifies their
hashes before reading them.  All particle data come from data.load_views,
which admits only the seven training cases and the 6.5 mm/s validation case.
No target case data, target H, latent truth, or reference field is opened.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from common import OUT, TRAIN_SEEDS, T, base_network, dump, fingerprint, load, read, sha, setup
import data as D
import latent_model as LM
from scoring import sequence_scores
from train_latent import verify_protocol

DEST = OUT / "diagnostics"
GATE = OUT / "training/selection_frozen.json"
QUANTILES = (.01, .05, .25, .5, .75, .95, .99)


def gate():
    protocol = verify_protocol()
    selected = read(GATE)
    body = {key: value for key, value in selected.items() if key != "fingerprint"}
    if fingerprint(body) != selected["fingerprint"]:
        raise RuntimeError("Selection gate fingerprint mismatch")
    if not selected.get("all_required_selections_frozen") or selected["protocol_fingerprint"] != protocol["fingerprint"]:
        raise RuntimeError("All transition selections must be frozen before diagnostics")
    rows = selected["runs"]
    if len(rows) != len(TRAIN_SEEDS) or {row["seed"] for row in rows} != set(TRAIN_SEEDS):
        raise RuntimeError("Incomplete/duplicate selected transition records")
    for row in rows:
        if sha(row["path"]) != row["sha256"] or sha(row["status_path"]) != row["status_sha256"]:
            raise RuntimeError("Selected transition artifact changed")
    return protocol, selected


def weighted_summary(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("Expected finite one-dimensional parameter observations")
    weights = weights / weights.sum()
    mean = np.dot(weights, values)
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    at = np.searchsorted(cumulative, QUANTILES, side="left").clip(0, len(values)-1)
    return dict(mean=float(mean), standard_deviation=float(np.sqrt(np.dot(weights, (values-mean)**2))),
                minimum=float(values.min()), maximum=float(values.max()),
                quantiles={f"p{int(100*q):02d}": float(values[order[i]]) for q, i in zip(QUANTILES, at)})


def gaussian_nll(residual, factor):
    white = torch.linalg.solve_triangular(factor, residual[..., None], upper=False)[..., 0]
    return .5*white.square().sum(-1)+factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)+math.log(2*math.pi)


def collect_pairs(base, head, views, cal):
    """Return raw sufficient diagnostic arrays; no fitting or resampling."""
    chunks = []
    cases = []
    for item in views:
        previous, current, weight = D.pairs(item)
        z = item["z"]
        parts = []
        for rows in torch.arange(len(current)).split(1024):
            pi, ci = previous[rows], current[rows]
            h, ctx = z["h"][ci].double(), z["context"][ci].double()
            nll, info = LM.supervised_pair_nll(base, head,
                z["h"][pi].double(), z["context"][pi].double(), z["y"][pi].double(), z["q"][pi],
                h, ctx, z["y"][ci].double(), z["q"][ci], cal)
            current_logq = LM.ob.latent(base, h, ctx, cal)[0]
            previous_q, current_q = z["q"][pi], z["q"][ci]
            ids = torch.arange(len(ci))
            e, ep = info["current_e"], info["previous_e"]
            A, b, L = info["A"], info["b"], info["L"]
            mean = torch.einsum("bij,bj->bi", A, ep)+b
            variance = L@L.transpose(-1, -2)
            residual = e-mean
            identity = torch.eye(2, dtype=e.dtype).expand(len(e), -1, -1)
            zero_continuous = gaussian_nll(e, identity)
            mean_continuous = gaussian_nll(residual, identity)
            variance_continuous = gaussian_nll(e, L)
            zero_categorical = -current_logq[ids, current_q]
            categorical = info["categorical"]
            jacobian = info["jacobian"]
            transition = LM.sign_transition(current_logq, info["kappa"])
            predicted_flip = transition[ids, previous_q, 1-previous_q].exp()
            zero_flip = current_logq[ids, 1-previous_q].exp()
            innovations = torch.linalg.solve_triangular(L, residual[..., None], upper=False)[..., 0]
            values = dict(weight=weight[rows].double(), kappa=info["kappa"],
                A_spectral_norm=torch.linalg.matrix_norm(A, ord=2),
                A_00=A[:, 0, 0], A_01=A[:, 0, 1], A_10=A[:, 1, 0], A_11=A[:, 1, 1],
                b0=b[:, 0], b1=b[:, 1], b_norm=b.norm(dim=-1),
                innovation_L00=L[:, 0, 0], innovation_L10=L[:, 1, 0], innovation_L11=L[:, 1, 1],
                innovation_variance0=variance[:, 0, 0], innovation_variance1=variance[:, 1, 1],
                innovation_covariance01=variance[:, 0, 1], innovation_trace=variance.diagonal(dim1=-2, dim2=-1).sum(-1),
                innovation_eigenvalue_min=torch.linalg.eigvalsh(variance)[:, 0],
                current_e0=e[:, 0], current_e1=e[:, 1], previous_e0=ep[:, 0], previous_e1=ep[:, 1],
                standardized_innovation0=innovations[:, 0], standardized_innovation1=innovations[:, 1],
                e_squared=e.square().sum(-1), residual_squared=residual.square().sum(-1),
                actual_flip=(previous_q != current_q).double(), predicted_flip=predicted_flip, zero_head_flip=zero_flip,
                nll=nll, zero_nll=zero_categorical+zero_continuous+jacobian,
                categorical_nll=categorical, zero_categorical_nll=zero_categorical,
                continuous_nll=info["continuous"], zero_continuous_nll=zero_continuous,
                jacobian=jacobian,
                q_only_nll=categorical+zero_continuous+jacobian,
                mean_only_nll=categorical+mean_continuous+jacobian,
                variance_only_nll=categorical+variance_continuous+jacobian,
                mean_with_learned_variance_gain=variance_continuous-info["continuous"],
                variance_with_learned_mean_gain=mean_continuous-info["continuous"],
                kappa_near_lower=(info["kappa"] <= .05).double(),
                kappa_near_upper=(info["kappa"] >= 11.95).double(),
                A_near_upper=(torch.linalg.matrix_norm(A, ord=2) >= .99).double(),
                b_near_bound=(b.abs().amax(-1) >= 1.98).double(),
                innovation_diagonal_near_lower=(L.diagonal(dim1=-2, dim2=-1).amin(-1) <= .0202).double(),
                innovation_diagonal_near_upper=(L.diagonal(dim1=-2, dim2=-1).amax(-1) >= 49.5).double(),
                innovation_offdiagonal_near_bound=(L[:, 1, 0].abs() >= .495).double())
            parts.append({key: tensor.detach().cpu().numpy() for key, tensor in values.items()})
        combined = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
        cases.append(dict(case=item["case"], pairs=len(current), pids=len(z["pid"][current].unique()),
                          results=summarize(combined)))
        chunks.append(combined)
    return {key: np.concatenate([part[key] for part in chunks]) for key in chunks[0]}, cases


def summarize(raw):
    weight = raw["weight"]
    mean = lambda key: float(np.dot(weight, raw[key])/weight.sum())
    parameter_names = ["kappa", "A_spectral_norm", "A_00", "A_01", "A_10", "A_11", "b0", "b1", "b_norm",
                       "innovation_L00", "innovation_L10", "innovation_L11", "innovation_variance0",
                       "innovation_variance1", "innovation_covariance01", "innovation_trace", "innovation_eigenvalue_min"]
    parameters = {key: weighted_summary(raw[key], weight) for key in parameter_names}
    boundary = {key: mean(key) for key in raw if "near_" in key}
    scores = {key: mean(key) for key in raw if "nll" in key or key.endswith("_gain")}
    scores["selected_minus_zero_latent_nll"] = scores["nll"]-scores["zero_nll"]
    scores["categorical_gain"] = scores["zero_categorical_nll"]-scores["categorical_nll"]
    scores["continuous_gain"] = scores["zero_continuous_nll"]-scores["continuous_nll"]
    dynamics = dict(actual_q_flip_rate=mean("actual_flip"), predicted_q_flip_probability=mean("predicted_flip"),
                    zero_head_q_flip_probability=mean("zero_head_flip"),
                    conditional_mean_squared_residual=mean("residual_squared"),
                    zero_mean_squared_residual=mean("e_squared"),
                    relative_squared_residual_reduction=1-mean("residual_squared")/mean("e_squared"),
                    standardized_innovations={key: weighted_summary(raw[key], weight)
                        for key in ("standardized_innovation0", "standardized_innovation1")})
    normw = weight/weight.sum()
    covariance = np.zeros((2, 2))
    correlation = np.zeros((2, 2))
    for i in range(2):
        left = raw[f"previous_e{i}"]; left = left-np.dot(normw, left)
        for j in range(2):
            right = raw[f"current_e{j}"]; right = right-np.dot(normw, right)
            covariance[i, j] = np.dot(normw, left*right)
            correlation[i, j] = covariance[i, j]/math.sqrt(np.dot(normw, left*left)*np.dot(normw, right*right))
    dynamics["standardized_e_lag1_weighted_correlation_matrix"] = correlation.tolist()
    return dict(pair_count=len(weight), pair_weight_mass=float(weight.sum()),
                parameters=parameters, near_boundary_weighted_fractions=boundary,
                supervised_pair_scores=scores, dynamics=dynamics)


def endpoint_validation(base, head, views, cal, force):
    result = {}
    for mode in ("iid", "reset", "temporal"):
        values = {}
        for degree in (20, 28):
            total = mass = 0.
            for item in views:
                z = item["z"]
                weights = z["weight"].double()*2
                jets = lambda zz, ix: (z["v"][ix].double(), z["g"][ix].double(), z["acc"][ix].double())
                value, contribution, _ = sequence_scores(z, item["c"], base, head, cal, force,
                    weights, jets, degree, mode, metrics=False, backward=False)
                total += value; mass += contribution
            values[str(degree)] = total/mass
        values["degree28_minus20"] = values["28"]-values["20"]
        result[mode] = values
    for degree in ("20", "28"):
        result["temporal_minus_reset_"+degree] = result["temporal"][degree]-result["reset"][degree]
        result["reset_minus_iid_"+degree] = result["reset"][degree]-result["iid"][degree]
        result["temporal_minus_iid_"+degree] = result["temporal"][degree]-result["iid"][degree]
    return result


def text_report(report):
    lines = ["冻结后的隐变量转移诊断（仅非目标训练/验证数据）", "",
        "本文件只作解释，不重新选择网络或流场，也不读取目标H、目标真值或目标粒子数据。",
        "训练统计按等工况/等PID/等连续配对加权；验证仅6.5 mm/s既定验证PID。",
        "概率评分单位为nat/连续捕获配对；端点评分为nat/捕获区间，均越低越好。",
        "ADF每个q保留一个高斯，GH计算只是数值积分；20/28差异不包含ADF近似误差。", ""]
    for run in report["runs"]:
        lines += [f"seed {run['seed']}；选中epoch {run['selected_epoch']}"]
        for group in ("train", "validation"):
            row = run["groups"][group]["pooled"]
            pars, bounds, scores, dynamics = [row[key] for key in
                ("parameters", "near_boundary_weighted_fractions", "supervised_pair_scores", "dynamics")]
            lines += [f"  {group}：{row['pair_count']} 对；latent NLL {scores['zero_nll']:.6f} → {scores['nll']:.6f}"
                      f"（差 {scores['selected_minus_zero_latent_nll']:+.6f}）。",
                f"    kappa均值 {pars['kappa']['mean']:.4f}；A谱范数均值 {pars['A_spectral_norm']['mean']:.4f}；"
                f"A≥.99比例 {bounds['A_near_upper']:.2%}。",
                f"    A加权均值 [[{pars['A_00']['mean']:.4f},{pars['A_01']['mean']:.4f}],"
                f"[{pars['A_10']['mean']:.4f},{pars['A_11']['mean']:.4f}]]；"
                f"b均值 [{pars['b0']['mean']:.4f},{pars['b1']['mean']:.4f}]。",
                f"    innovation L对角均值 [{pars['innovation_L00']['mean']:.4f},{pars['innovation_L11']['mean']:.4f}]；"
                f"条件创新协方差迹均值 {pars['innovation_trace']['mean']:.4f}（零头=2）。",
                f"    真实q翻转率 {dynamics['actual_q_flip_rate']:.4%}；预测 {dynamics['predicted_q_flip_probability']:.4%}；"
                f"零头预测 {dynamics['zero_head_q_flip_probability']:.4%}。",
                f"    条件均值预测使标准化e平方残差下降 {dynamics['relative_squared_residual_reduction']:.2%}。",
                f"    固定已学q核：mean-only NLL {scores['mean_only_nll']:.6f}，variance-only {scores['variance_only_nll']:.6f}，"
                f"full {scores['nll']:.6f}；给定已学方差后均值贡献 {scores['mean_with_learned_variance_gain']:+.6f}。"]
        endpoint = run["validation_endpoint"]
        lines.append("  6.5 mm/s完整因果段端点NLL：")
        for mode in ("iid", "reset", "temporal"):
            row = endpoint[mode]
            lines.append(f"    {mode}: D20={row['20']:.8f}, D28={row['28']:.8f}, Δ={row['degree28_minus20']:+.8f}")
        lines.append(f"    temporal−reset: D20={endpoint['temporal_minus_reset_20']:+.8f}, "
                     f"D28={endpoint['temporal_minus_reset_28']:+.8f}；负值表示吸收过去端点后的记忆有增益。")
        lines.append("")
    lines += ["解释边界：",
        "1. mean-only/variance-only是固定已选参数的监督密度替换诊断，二者有交互，不能把差值相加或当作因果比例。",
        "2. A高持续性、q翻转概率接近真值、条件均值残差下降支持学到了时间结构；只有方差缩放或NLL下降不足以证明。",
        "3. reset保留同一个头造成的当前边缘改变，但每步丢弃上一端点后验；temporal−reset才直接检验该近似过滤的记忆增益。",
        "4. 所有结果均用非目标真场生成的context；实际未知场反演的分布偏移和可辨识性仍需目标场对照回答。",
        "5. 这些训练/验证结果不能作为目标常流、超流或留出粒子改善的证据。",
        "6. 更完整的参数加权分位数、靠边比例、逐工况分解和来源SHA见同名JSON。"]
    return "\n".join(lines)+"\n"


@torch.no_grad()
def run():
    setup()
    started = time.perf_counter()
    protocol, selected = gate()
    cal, force = [read(T.SOURCE/name) for name in ("calibration.json", "force.json")]
    views = {group: D.load_views(group) for group in ("train", "validation")}
    report = dict(version="latent-transition-non-target-diagnostics-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        frozen_protocol_fingerprint=protocol["fingerprint"], training_selection_gate_sha256=sha(GATE),
        helper_source=dict(path=str(Path(__file__).resolve()), sha256=sha(Path(__file__))),
        target_data_opened=False, target_H_scored=False, target_reference_read=False,
        changes_to_fitting_or_selection=False,
        weighting="equal case/PID/consecutive captured edge for pair summaries; original equal PID/interval for endpoint scoring",
        boundary_definitions=dict(kappa_low="<=0.05", kappa_high=">=11.95", A_high=">=0.99 (bound 0.995)",
            b="any |component|>=1.98", innovation_diagonal_low="any <=0.0202", innovation_diagonal_high="any >=49.5",
            innovation_offdiagonal="|L10|>=0.495"),
        attribution_note="Fixed selected parameters; q held at learned value for mean-only/variance-only. Interactions prevent causal/additive interpretation.",
        runs=[])
    for record in sorted(selected["runs"], key=lambda value: value["seed"]):
        seed = record["seed"]
        base, base_record = base_network(seed, torch.float64)
        checkpoint = load(record["path"])
        if checkpoint["fingerprint"] != protocol["fingerprint"] or base_record["sha256"] != record["base_sha256"]:
            raise RuntimeError("Selected head/base binding mismatch")
        head = LM.LatentTransitionHead(base, state=checkpoint["state"], freeze=True).double()
        item = dict(seed=seed, selected_epoch=checkpoint["epoch"], selected_path=record["path"],
                    selected_sha256=record["sha256"], base_path=base_record["path"], base_sha256=base_record["sha256"], groups={})
        for group in ("train", "validation"):
            raw, cases = collect_pairs(base, head, views[group], cal)
            item["groups"][group] = dict(pooled=summarize(raw), cases=cases)
        item["validation_endpoint"] = endpoint_validation(base, head, views["validation"], cal, force)
        item["stored_selected_validation20"] = checkpoint["validation"]
        item["selected_validation20_reproduction_difference"] = item["validation_endpoint"]["temporal"]["20"]-checkpoint["validation"]
        if abs(item["selected_validation20_reproduction_difference"]) > 1e-10:
            raise RuntimeError("Selected validation score was not reproduced")
        report["runs"].append(item)
        print(f"Non-target transition diagnostics complete for seed {seed}", flush=True)
    # Verify the frozen gate/artifacts again after all read-only computations.
    gate()
    report["elapsed_seconds"] = time.perf_counter()-started
    DEST.mkdir(parents=True, exist_ok=True)
    dump(DEST/"TRAINED_TRANSITION_DIAGNOSTICS.json", report)
    (DEST/"TRAINED_TRANSITION_DIAGNOSTICS.txt").write_text(text_report(report), encoding="utf8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-training-gate", action="store_true")
    args = parser.parse_args()
    if args.wait_for_training_gate:
        while not GATE.exists():
            time.sleep(30)
    elif not GATE.exists():
        raise SystemExit("Training selection gate is not frozen; diagnostics did not read data")
    result = run()
    print(json.dumps(dict(complete=True, runs=len(result["runs"]), elapsed_seconds=result["elapsed_seconds"])))
