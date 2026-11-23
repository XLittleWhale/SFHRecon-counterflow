统一基线方法：完整 each 训练与重建 GPU 离线包，2026-11-13 v2

运行
本包执行完整流程，不是此前只运行诊断的 MethodRevision 包，也不调用20261112
endpoint-only/pooled控制器。正式入口按顺序执行服务器数值验收、A、B、closure、
joint、概率选解冻结、PINN训练、全部场选解冻结、H与真值评估、重建网格导出。
默认2K的113/175/217三个工况，每个工况三个种子，独立训练。不含481。
概率训练随机种子411040/411041/411042，分别对应PINN种子301030/301031/301032。

解压到任意目录，在已有CUDA PyTorch的AutoDL镜像中运行：

  bash run_gpu.sh /root/autodl-tmp/unified_cf_full_20261113

默认沿用原基准预算：A/B最多80轮、最少40轮、patience20；closure/joint最多40轮、
最少20轮、patience10；PINN主预算200步、扩展600步。阶段epoch0也参与验证选择。
中断后对同一输出目录重跑同一命令会恢复模型、优化器、随机状态和选模历史。
每个阶段结束写status.json、history.json、last.pt、selected.pt；不要只凭日志末行
判断全流程完成。最终标志是complete.json及evaluation/summary.json的complete=true。

可选统一续训配置（所有几何使用相同语义，不由CF类型隐式改变）：
  bash run_gpu.sh /root/autodl-tmp/unified_cf_short continuation_short
  bash run_gpu.sh /root/autodl-tmp/unified_cf_joint joint_continuation
continuation_short仍重训全部阶段：A/B 20/8/patience6，closure/joint 12/6/patience4。
joint_continuation明确继承周期A/B谱系，重训CF closure/joint，不宣称已完成CF A/B复训。
不同profile必须使用不同输出目录。默认baseline_retrain_all不跳过任何训练阶段。

分阶段及诊断命令：
  python launch.py plan --output /root/autodl-tmp/plan
  python launch.py validate --device cuda --output /root/autodl-tmp/validation
  python launch.py probability --device cuda --output /root/autodl-tmp/unified_cf_full_20261113
  python launch.py fields --device cuda --output /root/autodl-tmp/unified_cf_full_20261113
  python launch.py evaluate --device cuda --output /root/autodl-tmp/unified_cf_full_20261113
  python launch.py smoke --device cpu --output /tmp/unified_cf_smoke
smoke使用217一个种子、完整阶段但缩小数据和预算；其检查点标记diagnostic_only，
不能冒充正式结果。正式入口要求CUDA，并先在当前设备执行数值验收。
run.lock防止两个进程同时写同一输出。相同主机的失效进程锁可自动恢复；若输出移至
另一主机，先确认旧进程已停，再将其run.lock改名保留，随后续跑。

离线环境
Python>=3.10，已安装CUDA版PyTorch>=2.2、numpy、scipy、psutil；建议至少24GB显存。
不联网安装，不依赖旧包目录。本包不含CUDA驱动或依赖wheels，应选择这些库已安装的镜像。
环境变量PYTHON_BIN可指定解释器；CF_EACH_PDE_CHUNK默认64，缩小只改变显存分块。
A/B优化器batch仍为1024，内部分32行累积同一次更新的梯度，不改变样本权重或步数。
文件SHA校验、原始基线权重/校准、生成器源码证书及三例准备数据均包含在包内。

共享方法与几何适配
common_training.py唯一负责阶段冻结、优化器、停止、验证选模与断点恢复。
A为自由force监督NLL或俘获y/q监督NLL，加0.1真实端点NLL；只训练边缘网络。
B冻结边缘网络，训练连续同PID的校准隐状态转移NLL，只用因果端点预测选模。
A/B训练float32、复制模型后float64验证，lr=1e-4、AdamW、weight_decay=1e-4；
A训练GH6/验证GH20；B验证GH20。归一化与物理校准继承原周期检查点，不重估。
closure先训练teacher目标。joint要求同协议同case/seed的完整closure选解，使用
两步可微内层、互斥inner/outer粒子、参考场/初值误差、端点NLL、监督anchor、teacher。
联合损失权重仍是1、0.5、0.08、0.05、1；joint为float64，GH6/GH12，原复合验证评分。
候选closure/joint选解必须通过整个50帧窗口稳定性检查，不能只查短训练窗口。
周期与CF调用同一原始joint目标函数；periodic_joint.py保留旧G.T@u约定，
channel_joint.py使用CF材料加速度G@u和通道边界。周期完整joint值/超梯度做数值回归。
通道closure复用原568参数、9不变量、8张量基系数网络，以Pv/Pg弱投影到通道基。
应力乘sin²(pi*x/W)保证侧壁零神经应力牵引；内部交换力逐点等量反向。
状态使用固定几何的质量归一化坐标，每个系数对应U²/2能量，不用目标数据归一化。
PINN沿用原基线固定物理算子和Frozen-B先验，不把学到的closure另加进PINN物理项。

显式时间适配
113/175保持90fps，217保持180fps；原相机终点没有插值增样。
神经历史由过去相机点构造，物理查询在积分中点；y/q标签仍在原生区间起点。
隐状态时钟统一定义在积分片段起点。长间隔通过真实片段dt/1ms组合原核，
不足1ms用(1-f)I+fK并做按符号条件的高斯矩匹配；B训练和端点推断使用相同时钟。
该映射是显式近似，不把1ms核直接当作11.11/5.56ms核。没有新增隐状态真值标签。
原1ms监督核的损失/梯度保持不变；另检查2.5ms解析组合及真实CF训练的下降性。
共同窗口规则max(5ms,2原生区间)：周期仍5ms，CF为22.22/11.11ms。
共同内层时间预条件为inner_lr * min(1,5ms/window_duration)^2，inner_lr基值0.05。
因此周期5ms完全保持原步长；113/175有效步长0.00253125，217为0.010125。
这项明确适配用于避免长窗口的两步梯度过冲，不改变损失权重或内层步数。

数据与溯源
复用20261108三例已有模拟。历史q来自播撒前检查点中的实际符号表和历史附着索引；
精确哈希绑定的生成器源码证明涡槽位和符号不改变，不用索引奇偶、局部涡量或最终猜测。
y为anchor0-particle除以半径，只在流向y周期取差；两锚点对称性已经核查。
自由force=solve(A,affine(full)-affine(resolved))/U，不采用力快照替代。
full使用已保存相机帧场的空间双线性、时间线性插值，resolved为K12通道投影；
不是微步真实场的声明。C的固定anchor由纯物理残差扣除查询重新计算相同力标签。
不把相机起点q当成所有子步q真值；未认证的cap子步force保持NaN且不进入监督损失。
只附训练F/V、重建I/V、留出H和评估网格所需数据；无需上传全部高分辨率原始场。
若需重做数据准备，prepare --source指向原cf_reference_transient1s_50frames_results_20261108。

协议与选择边界
F:pid%5==2训练概率；I:pid%5>=3训练PINN；V:pid%5==1验证；H:pid%5==0留出。
CF joint的inner/outer在F内部按(pid//5)%2划分；V也独立划分，不能借用I/H凑数。
共同窗口构造只保留两组各至少两个自由/俘获PID且有连续俘获标签对的窗口。
joint读取同一个参考实现的流场真值，故结果严格标为“同实现监督校准”，不是独立场
泛化，也不能称作目标真值隔离的盲重建。最终评估文件不参与选模。
全部概率模型冻结后才进入PINN；全部case/seed及预算选解冻结后才评分H并读取评估真值。
周期原始非目标参考协议和CF同实现协议在报告中分别记录，不混称同一种泛化实验。
CF实验可使用匹配模拟的冻结模型及相同通道PINN接口；实验不生成y/q、force或真值误差。
本发布的正式批次是上述三个CF模拟；不自动将未注册实验或其他几何塞入通道适配器。

验收与结果
包内validation/保存本机检查证据。服务器入口会重跑数值验收，检查真实CF A/B损失下降、
两步joint超梯度、内层/外层损失下降、全窗口稳定性、周期回归、冻结概率后的PINN反传。
本机没有CUDA，CPU证据不冒充GPU实测；完整GPU预算训练结果由AutoDL运行产生。
训练耗时分别写入阶段status/history；不沿用旧endpoint-only小时数估计。
evaluation/summary.json含H指标和vn/vs场误差；evaluation/<case>/seed_*/包含重建npz。
原规范原样封存在specification/，其历史implementation_complete=false描述修订规范
创建时的状态；本发布执行状态以bundle_manifest、acceptance和complete记录为准。
