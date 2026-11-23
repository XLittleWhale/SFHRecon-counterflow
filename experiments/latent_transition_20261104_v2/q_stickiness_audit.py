"""Independent read-only audit of the selected q-stickiness clamp.

Uses only frozen selected heads and allowed non-target train/validation labels.
Counterfactual fixed scalar kappa values are diagnostics, never checkpoints or
new selections. No target data or field reconstruction is read or changed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from common import OUT, T, base_network, dump, load, read, sha, setup
import data as D
import latent_model as LM
from diagnostics_latent import gate, weighted_summary

KAPPAS = (0., .01, .1, 1.)


def summarize(raw):
    weights = raw["weight"]
    mean = lambda key: float(np.dot(weights, raw[key])/weights.sum())
    derivative = mean("right_derivative_at_zero")
    zero = mean("nll_0")
    return dict(pairs=len(weights), pair_weight_mass=float(weights.sum()),
        raw_kappa=weighted_summary(raw["raw_kappa"], weights),
        effective_kappa=weighted_summary(raw["effective_kappa"], weights),
        raw_negative_fraction=mean("negative"), raw_exact_zero_fraction=mean("exact_zero"),
        raw_above_twelve_fraction=mean("above_twelve"),
        observed_flip_rate=mean("actual_flip"), zero_head_predicted_flip_probability=mean("predicted_flip"),
        categorical_nll_right_derivative_at_zero=derivative,
        zero_is_local_minimum_along_uniform_nonnegative_kappa=bool(derivative >= -1e-10),
        categorical_nll_by_fixed_kappa={format(kappa, "g"): mean("nll_"+format(kappa, "g")) for kappa in KAPPAS},
        categorical_nll_change_from_zero={format(kappa, "g"): mean("nll_"+format(kappa, "g"))-zero for kappa in KAPPAS})


def group_audit(base, head, views, cal):
    all_cases, all_arrays = [], []
    for item in views:
        z = item["z"]
        previous, current, weights = D.pairs(item)
        chunks = []
        for rows in torch.arange(len(current)).split(1024):
            pi, ci = previous[rows], current[rows]
            h, ctx, prevctx = z["h"][ci].double(), z["context"][ci].double(), z["context"][pi].double()
            enc = base.encode(h, ctx)
            raw_kappa = head.network(torch.cat((enc, (prevctx-base.cm)/base.cs, (ctx-base.cm)/base.cs), -1))[:, 0]
            current_logq = torch.log_softmax(base.sign(enc)/cal["sign_temperature"], -1)
            qp, qc = z["q"][pi], z["q"][ci]
            ids = torch.arange(len(ci))
            actual_flip = (qp != qc).double()
            predicted_flip = current_logq[ids, 1-qp].exp()
            values = dict(weight=weights[rows].double(), raw_kappa=raw_kappa,
                effective_kappa=LM.bounded_kappa(raw_kappa), negative=(raw_kappa < 0).double(),
                exact_zero=(raw_kappa == 0).double(), above_twelve=(raw_kappa > 12).double(),
                actual_flip=actual_flip, predicted_flip=predicted_flip,
                right_derivative_at_zero=actual_flip-predicted_flip)
            for kappa in KAPPAS:
                transition = LM.sign_transition(current_logq, torch.full((len(ci),), kappa, dtype=torch.float64))
                values["nll_"+format(kappa, "g")] = -transition[ids, qp, qc]
            chunks.append({key: value.cpu().numpy() for key, value in values.items()})
        arrays = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
        all_cases.append(dict(case=item["case"], results=summarize(arrays)))
        all_arrays.append(arrays)
    pooled = {key: np.concatenate([part[key] for part in all_arrays]) for key in all_arrays[0]}
    return dict(pooled=summarize(pooled), cases=all_cases)


def text_report(report):
    lines = ["q持续性参数的独立只读诊断", "",
        "仅使用冻结选中模型和既定非目标train/validation数据；没有改参数、重选epoch或读取目标场/H。",
        "右导数是对有效kappa从0向正侧统一增加的监督分类NLL导数：实际翻转率−p0预测翻转概率。",
        "它绕过raw到[0,12]的clamp，仅用来检验kappa=0是否为这个受限方向的局部最优。", "",
        f"当前torch版本：{report['torch_version']}。直接自动微分探针："]
    for dtype, probe in report["clamp_autograd_probe"].items():
        lines.append(f"  {dtype}: 输入 {probe['input']}，实际bounded_kappa实现梯度 {probe['gradient']}。")
    lines += ["v2保持与clamp相同的数值函数，在0边界显式取右导数1；这是对v1已确认初始化梯度阻断的修复。", ""]
    for run in report["runs"]:
        lines.append(f"seed {run['seed']}，selected epoch {run['selected_epoch']}")
        for group in ("train", "validation"):
            row = run["groups"][group]["pooled"]
            raw = row["raw_kappa"]
            lines += [f"  {group}: raw均值 {raw['mean']:.8g}，p01/50/99 "
                      f"{raw['quantiles']['p01']:.8g}/{raw['quantiles']['p50']:.8g}/{raw['quantiles']['p99']:.8g}，"
                      f"负侧比例 {row['raw_negative_fraction']:.2%}。",
                f"    分类NLL右导数 {row['categorical_nll_right_derivative_at_zero']:+.10g}；"
                f"真实翻转 {row['observed_flip_rate']:.6%}，p0预测 {row['zero_head_predicted_flip_probability']:.6%}。",
                "    固定kappa的分类NLL："+", ".join(f"{key}: {value:.10f}" for key, value in row["categorical_nll_by_fixed_kappa"].items()),
                "    相对kappa=0的差："+", ".join(f"{key}: {value:+.10f}" for key, value in row["categorical_nll_change_from_zero"].items())]
            if (row["raw_exact_zero_fraction"] > .99999
                    and report["clamp_autograd_probe"]["float32"]["gradient"][0] == 0
                    and not row["zero_is_local_minimum_along_uniform_nonnegative_kappa"]):
                lines.append("    结论：raw全部精确0，当前环境clamp在0边界导数为0；这是零初始化后的边界梯度阻断。"
                             "有效kappa正向导数为负，因此不能解释为该方向的局部最优。")
            elif row["raw_negative_fraction"] > .99999 and not row["zero_is_local_minimum_along_uniform_nonnegative_kappa"]:
                lines.append("    结论：raw全落在负侧clamp死区，同时有效kappa正向的局部导数为负；"
                             "不能把有效kappa=0解释为非负持续性参数的局部最优。")
            elif row['effective_kappa']['mean'] > 0:
                lines.append("    结论：选中头已产生非零符号持续性；下列固定kappa网格只诊断原零核，不用于替换已选模型。")
            elif not row["zero_is_local_minimum_along_uniform_nonnegative_kappa"]:
                lines.append("    结论：有效kappa=0并非统一正向扰动下的局部最优；需结合实际输出与负区比例判断优化限制。")
            else:
                lines.append("    结论：统一提高kappa没有负的局部导数，但这不能证明所有context条件化函数均无增益。")
        lines.append("")
    lines += ["限制：",
        "1. 该诊断只证明局部优化/参数化行为，不证明修改clamp后目标流场、粒子预测或完整段过滤一定改善。",
        "2. p0本身已通过观测历史和场context预测当前q；分类NLL的绝对增益必须与该已有解释能力同时判断。",
        "3. 固定kappa网格不是新的训练或选解，不得据此替换已选模型；本轮正式结果保持原样。",
        "4. 加权方式为等工况、等PID、每PID等连续捕获配对；原数据不观测微步内的所有释放/再捕获事件。"]
    return "\n".join(lines)+"\n"


@torch.no_grad()
def run():
    setup()
    started = time.perf_counter()
    protocol, selected = gate()
    views = {group: D.load_views(group) for group in ("train", "validation")}
    cal = read(T.SOURCE/"calibration.json")
    clamp_probe = {}
    with torch.enable_grad():
        for dtype in (torch.float32, torch.float64):
            inputs = torch.tensor([0., -1e-6, 1e-6], dtype=dtype, requires_grad=True)
            outputs = LM.bounded_kappa(inputs)
            gradients = torch.autograd.grad(outputs.sum(), inputs)[0]
            clamp_probe[str(dtype).removeprefix("torch.")] = dict(input=inputs.detach().tolist(),
                output=outputs.detach().tolist(), gradient=gradients.detach().tolist())
    result = dict(version="q-stickiness-clamp-read-only-audit-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_fingerprint=protocol["fingerprint"], selection_gate_sha256=sha(OUT/"training/selection_frozen.json"),
        helper_source=dict(path=str(Path(__file__).resolve()), sha256=sha(Path(__file__))),
        target_data_read=False, target_H_scored=False, target_reference_read=False, model_or_selection_changed=False,
        fixed_kappa_grid=list(KAPPAS), torch_version=torch.__version__, clamp_autograd_probe=clamp_probe, runs=[])
    for record in sorted(selected["runs"], key=lambda row: row["seed"]):
        base, base_record = base_network(record["seed"], torch.float64)
        ck = load(record["path"])
        head = LM.LatentTransitionHead(base, state=ck["state"], freeze=True).double()
        q_output_weight = ck["state"]["network.2.weight"][0]
        q_output_bias = ck["state"]["network.2.bias"][0]
        result["runs"].append(dict(seed=record["seed"], selected_epoch=ck["epoch"],
            selected_sha256=record["sha256"], base_sha256=base_record["sha256"],
            q_output_layer_weight_all_exact_zero=bool((q_output_weight == 0).all()),
            q_output_layer_bias=float(q_output_bias),
            groups={group: group_audit(base, head, views[group], cal) for group in ("train", "validation")}))
    gate()
    result["elapsed_seconds"] = time.perf_counter()-started
    folder = OUT/"diagnostics"
    folder.mkdir(parents=True, exist_ok=True)
    dump(folder/"Q_STICKINESS_AUDIT.json", result)
    (folder/"Q_STICKINESS_AUDIT.txt").write_text(text_report(result), encoding="utf8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps(dict(complete=True, elapsed_seconds=result["elapsed_seconds"], runs=len(result["runs"]))))
