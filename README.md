# Two-fluid counterflow reconstruction

## 当前正式模拟基线（2026-10-27）

训练及模拟算例重建采用 **FullWindow 4 mm/s** 完整 GPU 结果：原始 1 ms 观测、
5–100 ms 单初态、512 PID、N40/K12/M64，七工况重训、独立 6.5 mm/s 标定、
动量–Vinen 强约束 4D-Var。网络输出物理符号/锚点分布，只有窗口首帧两流系数被反演。

正式基线的首帧 vn/vs 向量 RMS 为 **0.595727/1.236360 mm/s**，全窗为
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
新增 `src/counterflow_full/` 是当前模拟入口，下面的 `run.py train/invert` 是历史/CF 入口。
正式快照约 6 MiB；原始场、粒子观测、服务器大数据包和完整运行目录不进入 Git。

## 历史统一基线与保留的 CF 路径

以下说明对应此前发布的统一基线；“唯一入口／统一方法”等表述仅描述该历史流程，
不能替代上面的正式模拟协议。既有数据清单也仅覆盖历史数据包。

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

## 安装与运行

Python 3.10 或更新版本。先按硬件安装 PyTorch，再安装其余依赖；CPU 可用于当前有时间上限的训练/反演，生产级正向模拟需要 CUDA。

```bash
pip install -e ".[test]"
python run.py --help
python -m pytest -q
```

也可以直接使用已具备依赖的 Python 执行 `run.py`，不需要安装本项目。`counterflow` 命令是同一入口的别名。以源码目录为工作目录；使用安装入口从别处启动时，可设置 `COUNTERFLOW_HOME` 指向该源码目录。

### 重现保存的基线

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

新增原始模拟数据和 Excel 的预处理/训练张量构建尚未纳入独立发布接口。当前支持的端到端复现起点是经过校验的五个观测包和已有训练张量，详见 [数据说明](docs/DATA.md)。

## 结果的适用范围

这是统一、可复现的研究基线，尚未证明所有工况和指标稳定优于其他方法。10 mm/s 周期算例误差仍较大；CF 没有场真值。概率校准、轨迹拟合和二流场误差是不同指标，不能相互替代。见 [数学方法](docs/METHOD.md)、[局限性](docs/LIMITATIONS.md) 和 [验证记录](docs/VALIDATION.md)。

## 上传 GitHub

```bash
python scripts/package.py source --output ../counterflow-source.zip
```

源码包排除 `data/`、`runs/`、临时文件及原始实验文件，包含小型模型权重。数据可用同一工具的 `data` 模式单独打包，存储于数据仓库/服务器；发布地址尚未设置。此目录尚未自动建立远程仓库、上传或发布；未替作者选定开放许可证。
