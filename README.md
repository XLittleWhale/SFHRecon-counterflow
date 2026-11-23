# Two-fluid counterflow reconstruction

## Manuscript baseline · 2026-11-22

The current manuscript compares the **periodic two-fluid PINN at 1.5 and 4.0 mm/s**
with the **calibrated CF217 channel PINN at 2 K**. The paper reports validation-selected
fields from the 600-update periodic runs and 200-update CF217 runs. The particle
observation model is frozen during field optimization. The CF217 reconstruction uses
a channel basis, a reference-calibrated directional residual tolerance, and a
separate scale for the field output.

- [Methods, case selection and workspace commands](docs/MANUSCRIPT_BASELINE.md)
- [Version and artifact fingerprints](config/manuscript_baseline.json)
- `python -m workspace_runtime.manuscript --list` lists the manuscript cases from this checkout.
- `python -m workspace_runtime.manuscript --workspace <SFHFlow>` verifies the external assets in the organized workspace.

This is the simulation version reported in the manuscript. The older CF camera
workflow below is a historical development record; experimental reconstruction
is not a result of this manuscript version. New runs use their own numbered output
directories and do not replace the selected fields.

<details>
<summary>Historical development and release notes</summary>

## 当前选择：恢复固定 κL 的 PINN 基线（2026-11-06）

按用户决定，后续模拟与 CF 对照继续采用确定性互摩擦中的 **固定 κL0**，
模拟使用 `PINNTwoFluid_20261105 / pinn`，CF 使用 `PINNCF_20261105`。
粒子概率权重、`nu_s=0`、`Fns_factor=1`、Frozen B 和既有预算/选解规则保持各自冻结定义。
`OmegaFrictionPINN_20261106` 的 `|omega_s|` 替换仅保留为已完成的对照，不提升为默认方案。
本次恢复不重跑、不覆盖上述实验，也不改写已冻结的科学源码或检查点。

CF 已完成两个既有窗口的评价，但结果仍接近“常流 Poiseuille + 超流常量”的冷背景，
概率覆盖不足，不能把其当作已验证的双类抛物线真场重建。
当前工作转向解释剖面自由度、固定 κL 方程、Frozen B 优化尺度与 V 选解的影响。
完整结果见 [CF 结论](../outputs/PINNCF_20261105/CONCLUSIONS.txt)、
[旋度对照结论](../outputs/OmegaFrictionPINN_20261106/CONCLUSIONS.txt)。
近背景剖面的源码、选解、保存场与固定解析剖面诊断见
[CF 近背景诊断](../outputs/CFBaselineDiagnosis_20261106/DIAGNOSIS.txt)；该诊断没有新增反演或H评分。
下文模拟基线继续有效；`config/simulation_baseline.json` 是旧 FullWindow 发布快照的身份记录，
不作为本地 PINN 实验的自动启动或新结果提升开关。

## 当前模拟重建基线：PINN 二流场（2026-11-05）

项目以 **`PINNTwoFluid_20261105` 的 `pinn` 分支**作为后续模拟重建与方法对照的基线。
粒子概率模型保持冻结，整个时间窗的常流/超流场由同一连续时间神经网络参数化，
通过**粒子概率负对数似然 + 确定性二流体方程软残差 + 全时域正则**联合反演。
初场是该连续场在窗口起点的取值，不再另设独立初场、过程增量或闭合网络参数。
场仍需针对每个观测窗口优化；它不是直接跨工况输出流场的预训练预测器。

### 基线身份与固定规则

| 项目 | 冻结定义 |
|---|---|
| 版本与实现 | `PINNTwoFluid_20261105`；[源码](experiments/pinn_twofluid_20261105/) |
| 协议指纹 | `44f9d65161bd5a7835aebf72465cff542a8e067e35765f73949c40db334625d4` |
| 概率模型 | 非目标6.5 mm/s验证工况选出的 `joint` 家族；自由态概率模型与捕获态因果ADF隐变量转移均冻结，仅使用其概率权重，不载入旧joint闭合网络 |
| 种子配对 | 场 `301030/301031/301032` ↔ 概率 `411040/411041/411042`；报告三种子均值，不按H或真场选种子 |
| 场表示 | 3×128 Tanh时间网络、8组正余弦特征；输出两流各624个无散Fourier系数，K12，196320个参数；物理计算网格M64 |
| 物理与先验 | `nu_s=0`、`Fns_factor=1`、固定L0与已知平均流；Frozen B仅提供PDE残差尺度/时间相关性 |
| 时间窗 | 物理5–100 ms，对应相对0–95 ms、96帧；PDE约束使用191个不同时间节点 |
| 优化 | 原冷初值，Adam学习率0.001、梯度裁剪5；全时间统一正则，候选场上下文的梯度穿过冻结概率网络 |
| 主比较预算 | **200步预算内的V主选**；保留固定第200步终态作敏感性对照 |
| 扩展预算 | 同一轨迹延长至600步，另存V主选与固定第600步终态；不与200步结果混称等预算 |
| 选解与积分 | V每10步检查，遵守冻结的两状态保护及均衡NLL规则；反演积分阶数12，V/H主评价20，敏感性28 |
| 数据隔离 | 本轮全部概率权重及场主/末态冻结后才评分目标H/reference；H及真场不参与本轮优化或选解 |

基线方法、概率权重和各工况的场检查点共同定义版本；`global_spline` 是同轮对照臂，
不是另一个默认基线。相对t=0即物理5 ms；“首5ms”是相对0–5 ms六个节点的RMS，
不同于单个t=5 ms值。空间图固定展示种子301030，不能代替三种子评价。

### 已完成的基线结果

下表均为**V主选、三种子均值**。RMS单位mm/s，数值对为常流/超流；
H NLL数值对为自由态/捕获态，越低越好。H是同窗因果一步条件预测，允许使用先前观测，
不是无观测的长时滚动预测。参考场使用N40/K12匹配带宽。

| 工况(mm/s) | 预算 | 初场RMS 常/超 | 首5ms RMS 常/超 | 全窗RMS 常/超 | H NLL 自由/捕获 |
|---|---|---|---|---|---|
| 1.5 | **200（主）** | 0.422981 / 0.405919 | 0.422449 / 0.404555 | 0.413475 / 0.380434 | −2.226628 / −1.938293 |
| 4 | **200（主）** | 0.698216 / 0.874350 | 0.692348 / 0.872860 | 0.643316 / 0.850083 | −2.290847 / −1.642030 |
| 1.5 | 600（扩展） | 0.197726 / 0.341597 | 0.197403 / 0.339272 | 0.195510 / 0.338224 | −2.803936 / −2.094652 |
| 4 | 600（扩展） | 0.645713 / 0.862148 | 0.637957 / 0.850359 | 0.580700 / 0.809472 | −2.357802 / −1.667512 |

本轮PINN相对全局样条及旧joint 4D-Var外部参照，在初场、首5ms、全窗两流RMS与
两态H NLL上共同改善；优势在三个数值种子及固定终态中均成立。
旧joint的闭合、查询及正则不同，因此不是仅改变参数化的一因素对照。
**本轮没有按相同协议重跑2026-10-27的FullWindow发布基线，不能将两版表格直接相减来宣称全面胜出。**

概率校准仍有限制：PINN200的名义90%覆盖率为1.5工况91.48%/95.52%、4工况94.66%/92.02%；
600步升至98.42%/97.31%和95.95%/92.82%，存在过覆盖。NLL与场误差下降不表示校准也改善。
三个种子共享物理实现，两个目标工况已有历史探索，概率家族只经单一非目标工况选择；
不能视为全新盲测或跨独立流动实现的泛化保证。K12、固定L0、ADF及上下文偏移限制仍在。
本次登记仅更新模拟重建基线，CF入口及其验证状态仍独立保留。

### 本地检查点、证据与运行入口

源码位于本仓库的 `experiments/pinn_twofluid_20261105/`；完整结果位于相邻目录
`../outputs/PINNTwoFluid_20261105/`，下列证据链接要求本地完整工作区，**不包含在旧的6 MiB发布快照中**。
当前PINN尚未制作独立可移植发布包；本次基线登记不代表已上传新的GitHub资产。

- [中文结论与完整对照](../outputs/PINNTwoFluid_20261105/CONCLUSIONS.txt)
- [协议及输入/源码SHA](../outputs/PINNTwoFluid_20261105/protocol.json)、[概率家族与权重冻结](../outputs/PINNTwoFluid_20261105/probability_selection/selection_frozen.json)、[场选解冻结](../outputs/PINNTwoFluid_20261105/selection_frozen.json)
- [60行主/末态原始结果](../outputs/PINNTwoFluid_20261105/tables/individual_runs.json)、[交付及独立复核清单](../outputs/PINNTwoFluid_20261105/DELIVERY_MANIFEST.json)
- [已完成的完整结果审计](../outputs/PINNTwoFluid_20261105/FINAL_INDEPENDENT_AUDIT.txt)
- [200步总览](../outputs/PINNTwoFluid_20261105/figures/overview_budget200.png)、[600步总览](../outputs/PINNTwoFluid_20261105/figures/overview_budget600.png)
- [22个96帧场导出清单](../outputs/PINNTwoFluid_20261105/FIELD_MANIFEST.json)、[坐标边界修正后的流场图](../outputs/PINNTwoFluid_20261105/figures_display_corrected/)、[展示修正来源](../outputs/PINNTwoFluid_20261105/DISPLAY_CORRECTION_MANIFEST.json)

概率文件为结果目录中的 `probability_selection/models/seed_411040.pt`（另两种子同理）；
场文件为 `runs/<case>/pinn/seed_<seed>/budget_<200或600>_<primary或terminal>.pt`，
其中case为 `vn0_0015m_s` 或 `vn0_004m_s`。确切SHA与配对关系以上述冻结清单为准。
新两臂共12条600步轨迹、48个快照，加12行历史joint外部参照；这些不是60次独立反演。
本轮概率网络新增训练为0，场网络的优化计入反演成本：CPU合计26.98小时，
启动至评价/绘图/自动审计完成实际约5小时36分钟；预检和步长扫描成本另见完整报告。

本轮实现需 **Python 3.11或更新版本**（历史依赖使用 `hashlib.file_digest`）；
下面旧流程的Python 3.10要求不适用于本轮。当前Windows完整工作区可从
`F:/file_xjy/SFHFlow` 执行只读协议核验：

```powershell
& .\.venv\Scripts\python.exe -B -c "import sys; sys.path.insert(0, r'VFM/2DCounterflow/github/experiments/pinn_twofluid_20261105'); from pinn_common import verify_protocol; print(verify_protocol()['fingerprint'])"
```

该命令只核对协议指纹及61个源码、138个输入的SHA，不反演、不评分、不写结果，也不代替60行结果算术检查。
完整结果检查已记录于上面的审计报告；显式运行
[audit_results.py](experiments/pinn_twofluid_20261105/audit_results.py) 会重写审计报告，但不执行反演或重新评分原始目标H/reference。
完整编排入口为 [run_pipeline.py](experiments/pinn_twofluid_20261105/run_pipeline.py)，依次执行反演、全局冻结、评价、绘图与审计。
它绑定当前输出路径和历史依赖，**不是接受任意新输出目录的通用CLI，也不是仅查看结果的命令**。
新机器需要完整数据、历史依赖源码/标定/权重及冻结清单；仅安装源码包不能复现本轮。
当前协议及冻结清单还绑定 `F:/file_xjy/SFHFlow` 下的绝对路径，不能仅复制 `github/` 后假定原指纹可直接复用。
新的实验应先准备独立路径与协议，不改写本基线的源码、划分、预算、选解或结果。
下面的 `counterflow_full` 与 `run.py` 命令分别属于历史发布模拟和历史/CF流程，均不启动当前PINN基线。

## 历史发布模拟基线：FullWindow（2026-10-27）

此前发布的训练及模拟重建基线为 **FullWindow 4 mm/s** 完整 GPU 结果：原始 1 ms 观测、
5–100 ms 单初态、512 PID、N40/K12/M64，七工况重训、独立 6.5 mm/s 标定、
动量–Vinen 强约束 4D-Var。网络输出物理符号/锚点分布，只有窗口首帧两流系数被反演。

该历史发布基线的首帧 vn/vs 向量 RMS 为 **0.595727/1.236360 mm/s**，全窗为
**0.489406/0.891793 mm/s**。这不是高精度或唯一性保证，也不是 CF 模型升级。
旧 CF 权重和流程保留，不能将其与本次周期结果称为一个已经统一验证的新版本。

- [流程、重训命令、数学边界与完整验收](docs/FULLWINDOW_BASELINE.md)
- [全部方程与目标函数](src/counterflow_full/MATHEMATICS.md)
- [冻结模型、首帧、配置及证据](artifacts/fullwindow_4mm_20261026/)
- [13 组正式结果图（PNG/PDF/SVG）](artifacts/fullwindow_4mm_20261026/figures/)

```bash
python -m pip install -e '.[test]'
pytest -q
python -m counterflow_full.verify --baseline artifacts/fullwindow_4mm_20261026
```

此验证不需要大数据包，重新推进完整 PDE 并重算已报告误差；完整重训需另备原始模拟数据。
`src/counterflow_full/` 保留为该历史版本的模拟入口，下面的 `run.py train/invert` 是更早的历史/CF入口。
该版本发布快照约6 MiB；它不包含当前PINN基线。原始场、粒子观测、服务器大数据包和完整运行目录不进入Git。

## 历史统一基线与保留的 CF 路径

以下说明对应此前发布的统一基线；“唯一入口／统一方法”等表述仅描述该历史流程，
不能替代当前PINN模拟协议。既有数据清单也仅覆盖历史数据包。

本目录是本项目**唯一的源码发布入口**。只将本目录的受版本管理文件上传 GitHub；上一级的历史 `outputs`、旧脚本和原始数据不属于本仓库。

统一方法：候选场条件化的涡符号/锚点概率网络，捕获态物理混合似然，自由态随机截距相关似然，共享窗口初态，动量–Vinen PDE 正向推进。周期场和 CF 使用相同的网络、优化及选解规则；几何和采样率作为显式物理输入。没有按算例切换方法的开关。

```text
.
├── run.py                    # 唯一命令入口
├── pyproject.toml            # Python 依赖与可选安装入口
├── src/counterflow/          # 唯一源码实现
│   ├── observation.py       # GRU、潜变量分布、物理受力解码
│   ├── likelihood.py        # 捕获态/自由态及相机区间似然
│   ├── dynamics.py          # 周期/固壁动量–Vinen 推进
│   ├── train.py, calibrate.py
│   ├── invert.py            # 共享初态、模态继续、验证选解
│   ├── evaluate.py, plot.py # 留出评价、VTKHDF、论文图组
│   └── forward/             # 点涡＋惯性锚点粒子数据生成
├── config/                  # 固定统一策略、数据哈希清单
├── artifacts/               # 当前网络、初始化网络、标定和训练记录
├── tests/                   # 概率、可迁移性、梯度与短重建检查
├── docs/                    # 数学方法、数据契约、验证和论文
├── scripts/                 # 源码发布包与数据包工具
├── data/                    # 本地数据包；Git 忽略
└── runs/                    # 各次运行的结果；Git 忽略
```

## 历史/CF流程的安装与运行

本历史流程的受控诊断：现有72/141训练粒子的信息账本、两流补偿与B模型梯度通路分解。
原网络/PDE/初态均未更新，不是一次新反演。结果支持先使用完整训练队列，但自由态的
参考方向偏差仍在；不能宣称已修复重建。见[验证记录](docs/VALIDATION.md)。

```bash
python -m counterflow.information_flow --oracle-root /path/to/OracleInitialInverse --output /path/to/InformationFlow --phase all
python -m counterflow.information_flow_figures --output /path/to/InformationFlow
```

使用原始已保存轨迹、真实涡符号及冻结B检查点；三处线性化均保持12 ms PDE和6 ms观测。
只分析训练及既有验证PID，不评分测试PID；实际内部路径/参考方向为特权诊断。
输出五组PNG/PDF/SVG，不输出冒充新重建结果的场文件。

上一轮受控诊断：固定真实粒子受力后的正则强度/初态子空间消融，结果及局限见
[验证记录](docs/VALIDATION.md)。20个有限预算对照没有得到更优方案，验证规则仍保留
原K12、原正则。上一轮发现的先验偏差不意味着减弱正则能降低场误差。
**不是新的生产模型**，不替换下面的统一方法。

```bash
python -m counterflow.oracle_inverse --model-root /path/to/SeparatedClosure_20261025 --source /path/to/vn0_004m_s --output /path/to/OracleInitialInverse --phase all
python -m counterflow.oracle_inverse_figures --output /path/to/OracleInitialInverse
```

需要先前独立训练/冻结输入及原始场数据；每条反演固定64更新，七条CPU对照。
`--phase prepare/invert/evaluate/checks`可分别运行；不要复用已有结果目录改变策略。
实际路径、符号、锚点、遗漏受力及参考启动均为显式标注的特权诊断。

在上述冻结输入上复现最新消融：

```bash
python -m counterflow.oracle_regularization --oracle-root /path/to/OracleInitialInverse --output /path/to/OraclePriorSubspace --phase all
python -m counterflow.oracle_regularization_figures --output /path/to/OraclePriorSubspace
```

固定12 ms窗口与两段6 ms观测，比较K4/K12、三种正则强度及原始/先验度量下的
64/128维训练Jacobian子空间。各64更新，两条原基线复用；不训练网络。
`--phase prepare/information/invert/extend/evaluate/checks`支持分步。
先冻结验证选解，再计算参考场/测试误差；结果含五组图和同分辨率VTKHDF。
扩展子空间是顺序验证探索，不是独立盲测；不能将其或已知受力输入直接用于CF。

Python 3.10 或更新版本。先按硬件安装 PyTorch，再安装其余依赖；CPU 可用于当前有时间上限的训练/反演，生产级正向模拟需要 CUDA。

```bash
pip install -e ".[test]"
python run.py --help
python -m pytest -q
```

也可以直接使用已具备依赖的 Python 执行 `run.py`，不需要安装本项目。`counterflow` 命令是同一入口的别名。以源码目录为工作目录；使用安装入口从别处启动时，可设置 `COUNTERFLOW_HOME` 指向该源码目录。

### 重现保存的历史统一基线

本地已整理好 `data/` 和 `runs/baseline/`。新机器需要另行拷贝数据包与基线运行记录，不能只下载 GitHub 源码便假定已拥有研究数据。

```bash
python run.py --run runs/baseline verify
python run.py --run runs/baseline evaluate
python run.py --run runs/baseline plot
```

`verify` 重放五个保存的初态，比较场及验证似然并检查数据哈希；不执行优化。`evaluate` 仅在选解冻结后读取参考真值，重新生成场文件和概率评价。`plot` 输出八组标准图及汇总 PDF：`runs/baseline/Results/comparison_figures.pdf`。其中旧对照场和壁面敏感性图是冻结的比较证据，不是可选的另一套重建实现。

### 重新反演

```bash
python run.py --run runs/new invert
python run.py --run runs/new evaluate
python run.py --run runs/new plot
```

默认使用随仓库提供的同一份模型与标定。可用 `invert --case periodic_4 cf_261` 短测两个算例；标准整体评估需要全部五个算例完成并冻结选解。每次新试验使用新运行目录，避免混用已经生成的检查点。

### 续训后反演

```bash
python run.py --run runs/retrained train
python run.py --run runs/retrained calibrate
python run.py --run runs/retrained invert
python run.py --run runs/retrained evaluate
python run.py --run runs/retrained plot
```

训练沿用当前 H18 初始化、六个条件场模态和两个种子，是有限预算续训，并非从零训练或收敛保证。模型直接监督量是涡符号和锚点偏移，参考速度场用于条件与物理解码。自由态解码器没有神经权重；只有自由态标定数据与噪声律保持一致时，才能复用随包提供的相关参数，代码会校验这一条件。

### 生成新的二维模拟

```bash
python run.py --run runs/simulation steady --device cuda --max-time-s 10 --vn-mean-mm-s 4
python run.py --run runs/simulation particles --device cuda --steady-checkpoint runs/simulation/simulation/outputs/large_steady/steady_state.npz --duration-ms 100
```

只有通过稳态判据后才生成 `steady_state.npz`；达到 10 s 上限不代表已经稳态。唯一粒子生成模型是惯性双锚点 tethered 模型。CPU 功能检查使用 `--smoke`，其结果明确标记为未完成稳态，不能当作生产数据。

新增原始模拟训练张量构建尚未纳入独立发布接口。CF 原始 Excel 已可通过下面的入口进入同一套冻结网络与反演流程，详见 [数据说明](docs/DATA.md)。

### 新热流 CF 实验

```bash
python -m counterflow.cf_raw /path/to/2.0_K_113mWcm2_90fps.xlsx data/experimental/cf_113/source_rows.json
python -m counterflow.cf_prepare --rows data/experimental/cf_113/source_rows.json
python run.py --run runs/cf_113 cf --audit data/experimental/cf_113/audit.json --figures runs/cf_113/Results
```

前两条命令需要已安装本包（`pip install -e .`），或将 `src` 加入 `PYTHONPATH`。文件名须含温度、热流、帧率；工作表字段为 `trajectory_num,time,pos_x,pos_y,category`。读取不修改 Excel。沿用 0.01265 mm/pixel、16×8 mm 通道，g1/g3 捕获态、g2 自由态；不适用于其他相机标定或温度。需要本地数据包中的 `cf_261.pt` 和其几何算子作为固定离散基底，不读取旧算例拟合初态。速度归一化、基础 L 和线张力依热流重新计算，几何算子缩放已用独立积分重算检验。

上述 `cf` 入口仅用于复现历史的两起始帧结果（捕获态为平滑 1 ms 复合似然），不再作为新 CF 计算的推荐入口。

### CF 单起始帧、真实相机观测（2026-10-25）

```bash
python -m counterflow.cf_camera_prepare --rows data/experimental/cf_113/source_rows.json
python -m counterflow.cf_camera_prepare --rows data/experimental/cf_217/source_rows.json
python run.py --run runs/cf_camera100 cf-camera --audit data/experimental/cf_113_camera100_f0184/audit.json data/experimental/cf_217_camera100_f0333/audit.json --figures runs/cf_camera100/Results --seconds 600
python run.py --run runs/cf_camera100 cf-camera-check --case cf_113_camera100_f0184 cf_217_camera100_f0333 --output runs/cf_camera100/Results/implementation_checks.json
```

准备命令打印实际生成的审计路径；上例帧号对应当前数据。每个工况仅一个 100 ms 窗口、一个待反演首帧；允许粒子中途进入/离开，按完整原始相机区间取观测。113 和 217 的观测间隔分别为 11.111… 和 5.555… ms，PDE 仍为 1 ms。g1/g3 和 g2 均直接比较原始位置增量，不制造平滑后的 1 ms 标签。缺帧、状态切换不跨越；不足三个点的段、缺少前一速度区间、窗口外区间、积分试探路径越界区间被显式排除。按物理粒子身份划分训练/验证/测试，各身份可包含多个不连续段。

沿用同一冻结网络、动量/Vinen 方程、壁面与 K2→4→8→12 路径。捕获态在每个相机区间保持符号及标准正态创新一致，通过时变网络边缘分布传递锚点；这是**新增的区间内相关假设，不是已经学得或验证过的时间联合分布**。自由态沿用段级偏差相关。白噪声位移方差按实际持续时间缩放，不随积分划分增加观测数。PCHIP/1 ms 历史只用于内部网络适配和积分，且不读取当前目标端点；网络并未以真实相机帧率重训。新似然及时间假设需重新评价校准，不能沿用旧校准保证。

仅保留单窗的验证选解模态延续，不再重复执行“首窗初始化”分支。`--seconds` 为每工况优化软预算；达到时限不表示收敛。结果独立保存，不覆盖历史图。窗口、粒子数量及观测算子均改变，新旧分数不能作为单因素配对消融直接相减；背景始终是 PDE 推进而非实验真场。

### 冻结模型的约束消融与采样回放

```bash
python run.py --run runs/cf_camera_ablation cf-camera-ablation --figures runs/cf_camera_ablation/Results
python run.py camera-replay --source /path/to/vn0_004m_s --figures runs/replay24 --start-ms 24
python run.py camera-replay --source /path/to/vn0_004m_s --figures runs/replay60 --start-ms 60
```

第一条要求上节两个相机观测包已准备好。自由态、捕获态、联合约束均执行固定 6 次更新，保留完整粒子总体的原始归一化权重、相同网络/PDE/正则/初态基底；删除一类约束不放大另一类权重。保存每个候选初态及验证粒子得分。额外用逐状态配对 bootstrap 上界进行零劣化边际的验证选解；未通过则返回零控制背景。它是探索性保护，不保证所有指标改善，尤其不能保证实验场误差。所有算例、路径和保护选解冻结后才读取测试评分。该短试验不替代收敛反演。

回放读取原模拟的 1 ms 轨迹、隐变量和 VTKHDF 场快照，只比较保存端点的真实增量。1/5/6/11/12 ms 覆盖相机附近的采样间隔，并不冒充精确 90/180 fps 数据。完整源场、源场的锐截断 K12 和均匀场，以及参考张力/网络预测分别对照；实际内部轨迹和稠密历史只作诊断上界，不用于 CF 重建。原求解器微步长未保存，故这不是精确的微步积分复现；还需区分源场截断与特定粗粒化滤波。结果独立保存，不覆盖现有研究基线。

## 结果的适用范围

### 粗粒化与似然方向验收

```bash
python run.py scale-acceptance --source-root /path/to/large_particles_vortex_tethered_inertial --output runs/scale_acceptance
```

该入口只生成独立诊断，不覆盖发布权重或既有反演。依次执行数据一致性检查、三种参考网格的配对短续训、原始采样位移回放、PDE 积分网格检查、参考初态到冻结反演初态的方向检查和四组图。可用 `--phase prepare/train/accept/pde/line/plot` 单独执行阶段。需要本地原始模拟、标准数据包及 `runs/baseline` 冻结结果。

显式区分参考粗粒化 N、模态 K、PDE 积分网格 M 和绘图网格。训练试验固定 K12，用同一初始化和 2/5 mm/s 的六帧子集、6.5 mm/s 验证集，比较 N=32/40/80；4 mm/s 仅用于诊断。真实 q/锚点和位移标签不变，标定冻结，不是生产重训或收敛保证。K24 只作带宽对照，不声称 K<=18 的原网络已在此带宽校准。`figure_manifest.json` 给出各图完整定义，验收数值见 `docs/VALIDATION.md`。

这是统一、可复现的研究基线，尚未证明所有工况和指标稳定优于其他方法。10 mm/s 周期算例误差仍较大；CF 没有场真值。概率校准、轨迹拟合和二流场误差是不同指标，不能相互替代。见 [数学方法](docs/METHOD.md)、[局限性](docs/LIMITATIONS.md) 和 [验证记录](docs/VALIDATION.md)。

### 受力尺度与 CF 相容性诊断

```bash
python -m counterflow.cf_force_support --fit-root runs/cf_camera_ablation --output runs/force_compatibility
python -m counterflow.force_scale --source /path/to/large_particles_vortex_tethered_inertial --output runs/force_compatibility/spatial
python -m counterflow.force_figures --output runs/force_compatibility
```

需已安装本包或设置 `PYTHONPATH=src`，并准备相机观测包与短消融候选。第一条仅用自由态验证分数冻结候选，再检查留出捕获态的有界受力可达性及原网络概率区域；不把逐区间最优张力用作预测。第二条独立比较同位置、同符号、同张力下完整源场与 N40K12 场的受力差，2/5 mm/s 标定，6.5 mm/s 验证，4 mm/s 测试；相关过程只预测空间遗漏受力，不代替完整轨迹似然。三组图保存为 PDF/PNG/SVG。此诊断不训练或替换发布网络，不修改 PDE、CF 初态或原标定。完整定义及局限见 `docs/VALIDATION.md` 的受力诊断章节。

### 分离物理隐变量和空间遗漏受力：有限预算重训与反演

```bash
python -m counterflow.closure_trial --source /path/to/large_particles_vortex_tethered_inertial --force-root runs/force_compatibility --output runs/separated_closure --phase all --training-seconds 45 --steps 64
python -m counterflow.closure_checks --output runs/separated_closure
python -m counterflow.closure_figures --output runs/separated_closure
```

实验入口，**不替换发布模型**。需要上节 `force_scale` 生成的 `spatial/force_sequences.pt`、参考场系数/局部受力缓存和标准数据包。也可按 `--phase prepare/train/accept/invert/evaluate` 单独执行。每种训练执行两个种子，45 s 为每个种子的软上限；64 为每种反演的梯度更新总数，不代表收敛。概率积分的数值复核可重复执行，不更新模型或选解。

四组为：A 原物理网络与原噪声边缘；B 原网络＋状态常数的时间相关遗漏受力；C 原网络＋条件遗漏受力网络；D 联合微调物理网络和遗漏受力网络。空间标签固定为同位置的完整源场受力减参考 N40K12 受力，不能随反演候选重定义。B/C/D 另行标定剩余误差，避免与原噪声重复计入；因此 A→B 是结构化受力及噪声核算的联合改动，不是单纯增添一个网络的消融。

训练使用 2/5 mm/s、物理 PID 划分；验证含独立 PID 和 6.5 mm/s 捕获态；4 mm/s 仅作开发测试。端到端对照固定 24--36 ms 单窗、两个原始 6 ms 增量、同一 K2→4→8→12 初态延续与原 PDE。两态均使用区间复合似然；A 的自由态保留原单区间边缘，**本试验没有保留生产版跨区间的段级联合密度**。它不是旧论文结果的逐项复现，也尚非 CF 部署版本。

输出包括训练/选解审计、概率校准、共同辨场方向、场误差，以及 `fields/` 下同分辨率参考/背景/四种重建的 VTKHDF 序列。五组诊断图为 PDF/PNG/SVG。具体结果和未通过的验收项见 [验证记录](docs/VALIDATION.md)；不要仅凭位移均值误差下降或 NLL 变小宣布场重建成功。

### 冻结模型的辨场验证：新增时间块与均值/协方差消融

```bash
python -m counterflow.closure_discrimination --source /path/to/large_particles_vortex_tethered_inertial/vn0_004m_s --model-root runs/separated_closure --output runs/field_discrimination --phase all
python -m counterflow.discrimination_figures --output runs/field_discrimination
```

使用上节冻结 A/B/C/D，增加“条件均值＋固定协方差”和“固定均值＋条件协方差”两项诊断，不重训、不反演、不替换生产模型。新增 42/84 ms 两个起点，比较真实 1/6/12 ms 保存端点；固定 N40K12，检验均匀化、两流分别改变涨落幅值、相位平移，以及初态扰动后的 PDE 回放。按物理 PID 聚合和配对 bootstrap，不能把同一粒子跨时间重复计为独立样本。仍是同一工况的开发诊断，不是外部盲测。

可分别执行 `prepare/score/refine/report/check/attribution`；最终汇总前必须先完成 `refine`，排除原模型在低概率候选场下的积分误差。`attribution` 是事后链式法则分析，截断导数不用于实际优化。模型和候选定义冻结在 `policy.json`，六组 PDF/PNG/SVG 的图注及来源哈希保存在 `figure_manifest.json`；结论与适用范围见 [验证记录](docs/VALIDATION.md)。运行 `all` 会复用同目录的评分缓存，改变模型或候选设计应使用新目录。

### 固定协方差的六参数均值校准

```bash
python -m counterflow.directional_mean --model-root runs/separated_closure --diagnostic-root runs/field_discrimination --output runs/directional_mean --phase all
python -m counterflow.directional_figures --output runs/directional_mean
```

冻结已有物理/条件网络，只调整每种粒子状态的条件均值缩放和二维小偏移，协方差保持 B 不变。G 仅拟合原物理受力标签，H 再加入组级方向得分约束；训练/选择均不使用 4 mm/s 诊断数据。可分阶段执行 `prepare/train/check/discriminate/invert/evaluate/cross/audit`。原 B 反演按哈希复用，E/G/H 使用同一 64 更新预算；未知初态反演不读取场/受力真值。`cross` 固定模型交叉评分全部候选场，不作新选解；`audit` 核对 VTKHDF 和原始场张量。该入口是有限预算研究对照，不自动替换生产权重或 CF 结果。

输出包括六参数、选解轨迹、辨场和概率检验、端到端场误差、同分辨率 VTKHDF 与五组 PNG/PDF/SVG。数学定义见 [方法说明](docs/METHOD.md)，有效性与不足必须结合 [验证记录](docs/VALIDATION.md) 判断，而非只看方向指标下降。

### 路径与初速度隔离诊断（不训练、不反演）

```bash
python -m counterflow.path_velocity_audit --model-root runs/separated_closure --diagnostic-root runs/field_discrimination --output runs/path_velocity_audit
python -m counterflow.path_velocity_figures --output runs/path_velocity_audit
```

固定 B 和已有 42/84 ms 数据，比较因果 pilot / 保存实际路径，以及相机差分 / 密集位置差分初速度；保留相同网络历史、PDE 和协方差。数据未保存瞬时速度，中心差分只能作为代理，不能称为真值。实际路径及中心差分利用未来观测，只能做特权诊断，不用于实际反演。输出指定参考/PDE 回放评分、PID 配对区间、差分敏感性和四组 PNG/PDF/SVG。`--phase check` 仅重查模型哈希及梯度，`--phase report` 仅重新汇总已有分数。有限测试结果不支持优先扩建路径积分器，详见 [验证记录](docs/VALIDATION.md)。

### 物理潜变量与空间遗漏受力的 oracle 对照

```bash
python -m counterflow.force_oracle_audit --model-root runs/separated_closure --diagnostic-root runs/field_discrimination --path-root runs/path_velocity_audit --output runs/force_oracle_audit
python -m counterflow.force_oracle_figures --output runs/force_oracle_audit
```

固定 B、PDE 和原始相机端点，主对照分别替换真实符号/锚点和固定参考场处的空间受力差；主组保留B受力协方差，额外检查仅真符号、保存张力及噪声精度。标签绝不随候选场重新计算。可分阶段执行 `prepare/score/report/information`；`prepare` 读取高分辨率场快照并缓存局部量，后续不重跑正向模拟。真实路径与隐变量仅用于诊断，不是可部署的重建输入。

输出含源一致恒等式核验、辨场分数、PID配对区间、两幅值方向信息矩阵、PDE兼容性与七组 PNG/PDF/SVG。该控制已恢复捕获态局部辨场能力，但同时暴露自由态vs弱敏感和PDE回放误差；尚未证明闭合可预测或完成oracle初态反演。详情见 [验证记录](docs/VALIDATION.md)，公式及下一步决策依据见 [方法说明](docs/METHOD.md)。

### 完整训练队列与自由态梯度阻断对照

```bash
python -m counterflow.gradient_routing --information-root runs/information_flow --output runs/gradient_routing --steps 64
python -m counterflow.gradient_routing_figures --output runs/gradient_routing
```

依赖既有 `InformationFlow` 输入及其冻结父目录。四组为 72/141 个训练 PID × 完整/阻断自由态向超流初态的梯度，固定两态权重 2:1、网络、PDE 和验证选解。默认每组 64 次更新，为有界 CPU 诊断，不代表收敛。可分阶段执行 `prepare/check/invert/evaluate`；`invert` 冻结四组选解后 `evaluate` 才计算参考场误差和测试指标。只改变研究输出，不覆盖生产结果。

输出含梯度有限差分、原基线复现、逐状态留出评价、同分辨率 40×40 参考/重建 VTKHDF，以及六组无标题的 PNG/PDF/SVG。实际中点路径仍是特权诊断输入；优化数据不含 oracle 力、符号、锚点或参考场监督。阻断在完整 PDE 反传之后实施，不删除前向物理，也不保证仍是原标量目标的梯度。方法和结果分别见 `docs/METHOD.md`、`docs/VALIDATION.md`。

### CF 型稀疏相机观测：软先验与时间信息汇集

```bash
python -m counterflow.sparse_strategy --information-root runs/information_flow --prior-root runs/force_scale/spatial/cache --output runs/sparse_strategy --steps 32
python -m counterflow.sparse_diagnostics --output runs/sparse_strategy --phase check
python -m counterflow.sparse_diagnostics --output runs/sparse_strategy --phase information
python -m counterflow.sparse_figures --output runs/sparse_strategy
```

研究检验，**不是新的生产模型或实际 CF 重建**。从既有4 mm/s周期模拟抽取两起点，导入113/217的相机间隔、短轨长度和空间覆盖模板；只有原相机时刻的保存端点进入历史和目标，不使用真实内部路径。因源数据只保存1 ms，模板时刻取最近保存时刻，目标不插值。113型部分模板因同折状态兼容粒子不足而跳过，具体数量在审计中报告。

三组为原正则、训练工况2/5 mm/s提供的两流软初态先验、相同软先验但仅使用前12 ms观测；均预测24 ms，K2→4→8→12，每组32更新。保留冻结B、原动量/Vinen PDE及完整梯度；全部选解冻结后才评分参考场和测试PID。完整实验12条反演，默认是有界CPU检验而非充分收敛。

输出包含十组无标题PNG/PDF/SVG、600帧同分辨率参考/候选VTKHDF、因果性和梯度检查，以及局部混合矩近似的信息量/空间方差比。后者不是精确混合似然Fisher或校准后验；只验证了线性化下的序贯信息加法，**没有实现非线性滑窗协方差传递**。本轮软先验没有稳定明显增益，延长窗口也不保证场误差下降，不自动晋级CF；数学定义和实测结论见方法、验证文档。

## 上传 GitHub

```bash
python scripts/package.py source --output ../counterflow-source.zip
```

源码包排除 `data/`、`runs/`、临时文件及原始实验文件，包含小型模型权重。数据可用同一工具的 `data` 模式单独打包，存储于数据仓库/服务器；发布地址尚未设置。此目录尚未自动建立远程仓库、上传或发布；未替作者选定开放许可证。

</details>
