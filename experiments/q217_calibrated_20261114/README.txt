CF217 正式离线GPU包：标定残差容差＋解耦输出尺度，三种子×200步

运行
在解压后的包目录执行：
  bash run_gpu.sh /root/autodl-tmp/cf217_calibrated_3seeds_200_20261114
必须使用新的输出目录，不能续写旧包或本地20步验证的目录。
对同一输出目录重跑同一命令即可断点续跑；模型、优化器、V选解历史均恢复。

任务范围
仅2 K、217 mW/cm^2，180 fps原生轨迹、完整267 ms重建窗口。
PINN种子301030/301031/301032，每个200步，共600步。
分别复用已有CF概率训练种子411040/411041/411042的selected_joint检查点。
这些概率模型此前已完成周期初始化→CF的A/B/closure/joint训练。
本包保持三个概率模型冻结，不重复概率训练；与本地四组验证的控制方法一致。
不包含113/175/481训练，不执行600步扩展。

算法
新残差容差：参考场前32个原生间隔预标定，后17间隔仅检查；完整旧时间相关核保留。
在能量坐标中令s_new=max(s_old,各方向训练残差RMS,0.1×训练残差全局RMS)。
该量是包括持续模型差异的二阶矩容差，没有用减均值冒充消除动力学偏差。
输出尺度：q=网络输出/ring，再经固定通道decode；不再乘1ms创新的绝对标准差。
几何encode/decode/ring、残差白化矩阵、完整时间Cholesky和三种子初始网络均封存，
GPU启动后直接载入，不重新进行会改变简并模态坐标的特征分解。
固定方程、PDE权重、时间离散、轨迹似然、GH阶数与选解规则均沿用验证设置。
三种子共享同一套已冻结的几何/残差容差，概率模型和网络初始参数按种子配对。

设置
float64，Adam lr=0.001，梯度裁剪5。
训练GH12，验证/评价GH20；535个半毫秒PDE节点和完整捕获序列反向传播。
每10步用V集验证；以初始状态为保护阈值，两类状态均不得退化超过0.01，
总V分数至少改善1e-6才更新选解。0步背景允许成为选定解。
全部三个种子的选解冻结之后，才计算H粒子与完整网格真实场误差。
本次继续标注“同实现监督校准”，不是独立实现泛化或CF实验成功的声明。
不把目标真实场直接作为PINN拟合项，但参考真实场已参与概率训练和残差容差标定。

环境
Python>=3.10、CUDA版PyTorch>=2.2、numpy、scipy、psutil；推荐沿用上次成功的AutoDL镜像。
包内不联网安装依赖，不包含PyTorch wheels或CUDA驱动。
启动时校验包内SHA，随后在实际GPU上执行几何、概率冻结、PINN损失/梯度对照检查。
GPU验收失败会停止，不会继续正式训练。CPU打包验证不能代替服务器的CUDA验收。

其他入口
  python launch.py plan --output /root/autodl-tmp/cf217_plan
  python launch.py validate --device cuda --output /root/autodl-tmp/cf217_cuda_check
  python launch.py fields --device cuda --output /root/autodl-tmp/cf217_calibrated_3seeds_200_20261114
  python launch.py evaluate --device cuda --output /root/autodl-tmp/cf217_calibrated_3seeds_200_20261114
  python launch.py smoke --device cpu --output /tmp/cf217_smoke
smoke为单种子2步，但保持完整数据、时间窗口与GH阶数，必须使用单独目录。
可用--max-new-steps 10限制每次调用的新增步数，再用同一命令续跑。
正式run/fields/evaluate禁止CPU；不隐式退回CPU长时间训练。

结果
complete.json和evaluation/summary.json均complete=true表示全部完成。
fields/cf_2K_Q217/seed_*/：background、budget_200_primary、budget_200_terminal及last/history/status。
evaluation/cf_2K_Q217/seed_*/：选定/末端/背景的重建网格npz。
evaluation/summary.json：各种子H似然、覆盖率、位移RMSE以及常/超流场误差。
probability_frozen.json：原CF概率检查点SHA及原训练协议，不改写其来源。
field_selection_frozen.json：V选解门禁及所有输出绑定。
acceptance_cuda.json：实际服务器数值验收；每个输出目录独立产生。
下载时保留整个输出目录，尤其protocol、两个冻结门禁、fields和evaluation。

时间参考
此前相同217数据的GPU重建三种子×200步约9.29小时，可作为排期参考。
本包省去重复概率训练；新容差/输出尺度的正式GPU耗时尚无实测，不保证与旧值相同。

打包证据
validation/中为本版本CPU检查、完整数据2步冒烟、断点恢复和本地方法回归证据。
provenance/local_validation/保留此前4组×20步诊断的报告与图，不能当作本包200步结果。
provenance/source_probability/保留原概率训练协议和冻结记录。
旧代码仅作为数值依赖保留，正式入口是q217_calibrated_20261114的新控制器。
本包不导入此前20步或200步PINN的训练状态；三个PINN都从已封存的0步初始化开始。
