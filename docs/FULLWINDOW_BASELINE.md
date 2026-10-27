# 正式模拟基线：4 mm/s 全窗口重训与初态反演

2026-10-27 将已回传的 `FullWindowCalibration_20261026/fullwindow_4mm_results`
确认为正式周期模拟基线，ID 为 `fullwindow_4mm_20261026`。采用原运行选中的结果，
不重新挑选“更好看”的时段、初态或网络，不覆盖旧 CF 权重。正式表示可追溯的比较基准，
**不表示高精度初态恢复、全局最优或所有工况稳定优于旧方案**。

## 入口与冻结结果

- 训练、标定、模拟反演：[`counterflow_full`](../src/counterflow_full/)。
- 完整数学契约：[`MATHEMATICS.md`](../src/counterflow_full/MATHEMATICS.md)。
- 固定参数：[`config.json`](../src/counterflow_full/config.json)。
- 网络、首帧、算子、物性、标定与 SHA256：
  [`artifacts/fullwindow_4mm_20261026`](../artifacts/fullwindow_4mm_20261026/)。
- 原始机器可读评估：[`report.json`](../artifacts/fullwindow_4mm_20261026/evidence/Results/report.json)。
- 13 组无标题 PNG/PDF/SVG：[`figures`](../artifacts/fullwindow_4mm_20261026/figures/)，
  图注在 [`figure_manifest.json`](../artifacts/fullwindow_4mm_20261026/evidence/Results/figure_manifest.json)。
- 旧 `counterflow` 的训练/反演命令仍保留，用于历史基线和 CF；不是本次全窗口训练入口。

快照约 6 MiB，不含原始粒子数据、约 12.54 GiB 的服务器输入包、优化器状态或完整 VTKHDF。
原始本地回传目录保持不变。`manifest.json` 保留原服务器包、运行策略及输出哈希；
发布时只调整便携构建、文档和验证入口，不更改正式运行的科学内核或数值结果。

## 固定实验协议

1. 1 cm × 1 cm 二维周期场；5–100 ms，共 95 个原始 1 ms 区间、96 帧。
   512 个物理 PID，46,939 个有效区间；排除 1,701 个端点状态变化区间，非有限区间为零。
   没有固定粒子数截断，也没有将微步长或插值位置当观测。
2. 物理 PID 按模 5 固定分组：2/3/4 拟合、1 选解、0 测试，分别 306/103/103 个 PID。
   同一粒子转态不改变分组。自由态有效区间为 7,501/2,065/2,487，捕获态为
   20,566/7,369/6,951；所有有效粒子参加其所属组，不将测试粒子用于优化。
3. 训练工况为 0.5、1、2、2.5、3、5、8.5 mm/s；6.5 mm/s 负责标定与正则选择；
   4 mm/s 仅作最终目标反演及评估。七工况有 110,000 个拟合组捕获态训练样本。
   4 mm/s 曾用于历史方法开发，因此不是全新盲测。
4. 保守粗粒化网格 N=40，条件场与反演均为 K=12 无散傅里叶模态；PDE 求积网格 M=64。
   两流共有 1,248 个未知首帧系数，均值、物性及初始线密度已知。
5. 动量–Vinen 强约束 4D-Var，只优化一个首帧；后续场由 PDE 推进。
   保留 \(\nu_s=\alpha\kappa\) 与 \(F_{ns}=-(2/3)\alpha\rho_s\kappa L(v_n-v_s)\)，
   不反演 \(\delta F_{ns}\)，不训练流体额外闭合。每毫秒两个 RK4 子步。

## 训练与反演的数学边界

网络不是直接输出 \(v_n,v_s\) 的 RealNVP。它输出涡符号与锚点偏移的条件分布：

\[
q_\phi(q,\delta\mid h,c)=\pi_\phi(q\mid h,c)
\mathcal N(\delta;\mu_{\phi q},S_{\phi q}S_{\phi q}^{\mathsf T}).
\]

\(h\) 只含当前及过去的原始采样轨迹；\(c\) 包含候选场局部滑移速度、梯度和物性。
训练用参考场构造条件，标签是符号/锚点而非流体速度；该设计不自动保证场可辨识性或泛化。
额外的粗场受力误差用训练工况“全分辨率减粗场”的仿射受力差拟合固定分状态高斯矩；
不将张力或粒子阻力重复分配到该误差，不在目标反演中自由拟合误差均值。

解析积分冻结系数下的粒子速度与位置，边缘化符号、锚点和受力误差，得到原始位置增量
\(Y_j=(x_{j+1}-x_j)/(U\Delta)\) 的概率密度。自由态去除锚点和捕获受力项，
保留源方程中真实存在的超流加速度依赖；没有人为截断自由态到超流的梯度。

\[
J_{\rm train}=\mathbb E_{\rm case,PID,time}
[-\log q_\phi(q^*,\delta^*\mid h,c_{\rm ref})-0.1\log p_\phi(Y\mid h,c_{\rm ref})].
\]

两个随机初始化分别为 261026/261027。最大 240 轮、最少 60 轮，验证早停耐心 30 轮。
实际选中 seed 261026 第 224 轮；该种子运行至 240 轮，另一个在 60 轮停止。
随后冻结网络，用 6.5 mm/s 标定潜变量宽度、符号温度及两态噪声。

目标反演的唯一控制为 \(\theta=(a_n(t_0),a_s(t_0))\)，目标函数为

\[
J(\theta)=0.08\,\mathcal L_{\rm fit}(\theta)+\lambda R(\theta,S_{\rm PDE}(\theta)),
\qquad
\mathcal L_G=\frac1{|S|}\sum_s\frac1{|P_s|}\sum_{p\in P_s}
\frac1{|I_{ps}|}\sum_{j\in I_{ps}}-\log p_\phi(Y_{pj}\mid h_{pj},c_{pj}(S_{\rm PDE}(\theta))).
\]

这是分状态、分 PID 平衡的复合似然，不是假设每个增量独立的精确轨迹联合概率。
\(R\) 的全部系数与定义见数学契约。6.5 mm/s 在预先给定的 0.3/1/3 中选定
\(\lambda=1\)。K2→4→8→12 逐步释放首帧控制，PDE 始终保留 K12；全窗口反传、
全部拟合区间参与每次梯度，不是截断时间反传。两次初始化的最终选解只依据留出 PID NLL。
正式选中 restart 1、K12、零基 step 39；没有用 4 mm/s 真场选择初态。

## 完整结果与局限性

RMS 定义为 \(\sqrt{\langle|v-v_{\rm ref}|^2\rangle}\)，单位 mm/s，
是**二维向量 RMS**，不能与旧报告的分量 RMS 或不同时间窗直接比较。
参考场为相同 N40/K12 粗粒化模态场；全窗指标对 96 个保存时刻等权。

| 对照 | 首帧 vn | 首帧 vs | 全窗 vn | 全窗 vs |
|---|---:|---:|---:|---:|
| 均匀背景 | 2.072525 | 2.417771 | 2.158647 | 2.499602 |
| 正式反演 | **0.595727** | **1.236360** | **0.489406** | **0.891793** |
| 参考首帧经同一 PDE 推进 | 0 | 0 | 0.387012 | 0.963367 |

反演全窗相对背景误差降低约 77.33%/64.32%；首帧相对涨落误差仍约 28.74%/51.14%。
模态投影到未投影粗网格的全窗差异为 0.312109/0.417274 mm/s；反演相对未投影粗场
为 0.580457/0.984587 mm/s。不能将表示误差混同于反演误差。

| 测试 PID 指标 | 自由态 | 捕获态 |
|---|---:|---:|
| 均匀背景 NLL | 5.477724 | 3.277486 |
| 逐时刻参考场 NLL | -2.137333 | -1.725402 |
| 参考初态 PDE 回放 NLL | -1.855805 | -1.242925 |
| 反演 NLL | -1.763741 | -1.257022 |
| 反演增量向量 RMS（μm） | 0.594550 | 0.845453 |
| 反演 90% 覆盖率 | 0.905705 | 0.861900 |

关键限制：

- 参考初态经当前 PDE 推进也会漂移，说明 95 ms 强约束存在流体模型偏差。
  它不是所有初态可能达到的误差下界；反演会用初态补偿后续偏差。
- 参考初态→反演初态诊断线上，拟合 NLL 持续改善，而测试分数在中间更优，仍有过拟合。
  这条线使用真值，只用于诊断，不能用来部署选解。
- 两个训练种子的最佳验证目标为 -4.338421/-1.123301，训练仍有明显种子敏感性；
  两次初态优化较接近不能证明整个训练过程稳定。
- 超流初态误差不仅是高阶不足。高阶及弱敏感方向未被充分恢复，不能声称唯一辨识或
  给出已校准的场后验置信区间。CF 尚未经过此全窗口模型的验证。
- 保存的 Vinen 网格无触发裁剪；时间子步 2→4 的全窗速度差约
  \(2.26\times10^{-10}/3.22\times10^{-10}\) mm/s。因而当前误差不主要来自时间积分精度。

后续优化优先区分持续 PDE 偏差与真实首帧信息，并提高训练种子稳定性；本次只确立基线，
不修改上述模型，也不把“增加 K 或训练轮数”当作已证实有效的修正。

## 从干净仓库验证／回放

```bash
python -m pip install -e '.[test]'
pytest -q
python -m counterflow_full.verify --baseline artifacts/fullwindow_4mm_20261026
# 可选：输出 96 帧同网格参考/反演场，目录必须尚不存在
python -m counterflow_full.verify --baseline artifacts/fullwindow_4mm_20261026 --export runs/formal_4mm_replay
```

验证器检查冻结文件哈希、加载网络，从保存的首帧重新推进完整 95 ms PDE，比较回传系数并
独立重算上述场误差。它不进行训练或再反演，也不把回放一致性称为独立科学验证。
部署的代码兼容性另由测试覆盖；原服务器源码哈希原样保存在运行策略中。

发布前的干净源码目录验证：13 项测试通过，3 项依赖旧私有数据的测试按预期跳过；
10 个科学内核的规范化源码哈希与原服务器代码一致。CPU 完整回放系数最大差
\(5.56\times10^{-17}\)，1.1.0 wheel 构建成功。哈希快照保留原始 JSON/图文件字节，
避免跨平台换行转换破坏校验。这些检查不代替多工况或 CF 科学验收。

## 从原始模拟数据重训与完整反演

需要九个 `vn...m_s` 原始算例目录，每个包含 `metadata.json`、
`particle_trajectories.npz`、`particle_hidden_truth.npz`、`field_*.vtkhdf`、
`vortices_*.vtkhdf`，覆盖 0–100 ms 原始保存时刻。数据未上传 GitHub，仓库不宣称
仅靠源码即可重新训练。原始 20 μs 步长仅属于模拟器，不作为神经网络观测。

```bash
python -m counterflow_full.build --source /path/to/large_particles_vortex_tethered_inertial --dest /path/to/new_fullwindow_package
# 将该完整输入包传到 CUDA 服务器，在包内执行
python -m pip install -r requirements.txt
python run_all.py --phase check
python run_all.py --out /root/autodl-tmp/fullwindow_4mm_new_run
```

构建器内置便携物性模板与算子，不依赖旧的 `data/inputs/periodic_4.pt`。
依次执行 prepare→train→calibrate→tune→invert→evaluate，产物包含全部图与五组 VTKHDF。
正式运行要求 CUDA；`--device cpu --smoke` 仅为两个区间的连通性检查。
改变源码/配置须用新结果目录，不修改原冻结回传包；新训练结果不得自动覆盖本基线。
