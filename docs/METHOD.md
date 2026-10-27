# 固定的统一方法

> 2026-10-27：模拟训练与重建的当前正式入口是
> [FullWindow 基线](FULLWINDOW_BASELINE.md)，完整方程见
> [数学契约](../src/counterflow_full/MATHEMATICS.md)。下文保留历史统一/CF 方法，
> 不是新模拟基线的目标函数，尤其不能混用随机截距似然与新分状态受力误差复合似然。

设历史二维轨迹特征为 \(H\)，候选场局部特征为 \(c(v_n,v_s,\nabla v_n,\nabla v_s)\)。GRU 编码历史，网络给出

\[
p_\phi(q,\delta\mid H,c)=\pi_{\phi,q}(H,c)
\mathcal N_2(\delta;\mu_{\phi,q},L_{\phi,q}L_{\phi,q}^{T}),\qquad q\in\{-1,+1\}.
\]

\(\delta\) 是以粒子半径归一化的平面锚点偏移；\(q\) 是涡符号。固定受力响应将其映射为无量纲位移均值 \(m(q,\delta;v_n,v_s)\)。捕获态观测为高斯混合：

\[
p_c(y_t\mid H_t,c_t)=\sum_q\int\pi_q\mathcal N_2(\delta;\mu_q,L_qL_q^T)
\mathcal N_2(y_t;m(q,\delta),\sigma_c^2 I)\,d\delta,
\quad y_t=\Delta x_t/(U\Delta t).
\]

数值积分使用 Gauss–Hermite 节点。训练使用涡符号与锚点标签，并加上物理解码的位移评分；场值作为条件输入而非网络输出监督。历史窗口和捕获态插值含有复合观测假设，不宣称该条件密度是完整生成模型。

自由态采用解析阻力响应。周期场的每个轨迹片段、每个坐标轴 \(a\) 的协方差为

\[
C_a=\sigma_f^2(I+\beta_a\mathbf1\mathbf1^T).
\]

CF 在原始相机间隔内连续传递粒子速度状态；内部场步长仍为 1 ms。令 \(W\) 为相机区间与内部步的重叠权重，则

\[
C_a^{\rm cam}=\sigma_{\rm CF}^2[WW^T+\beta_a(W\mathbf1)(W\mathbf1)^T].
\]

\(\beta\) 为固定标定量，不逐粒子优化。捕获态继续使用 1 ms 平滑插值的混合似然。

流体由无散基投影后的动量方程与 Vinen 方程推进。作用于正常流体的互摩擦是

\[
F_{ns}=-\frac23\alpha\rho_s\kappa L(v_n-v_s),
\]

并以等反力进入超流方程。周期场 \(\nu_s=\alpha\kappa\)；CF 为无超流有效黏度、Shen 固壁基、正常流无滑移、超流无穿透，不规定超流切向零梯度。已知平均流量与初始 \(L\) 固定。压力由无散投影消去，不意味着压力为零。

仅反演共同初态控制 \(\theta\)，不反演 \(\delta F_{ns}\) 或额外流体闭合：

\[
(a(t),L(t))=\mathcal S_t(D\theta,L_0),\qquad
J(\theta)=0.08\frac{1}{N_{\rm seg}}\sum_j\frac{-\log p(Y_j\mid\mathcal S(D\theta),H_j)}{T_j}
+R_{\rm phys}+R_{\rm spec}.
\]

\(T_j\) 以 ms 计。周期物理正则为

\[
R_{\rm phys}=\frac{0.003}{24}\langle\|a_{tf}\|^2\rangle_{t,f}
+\frac{0.03}{24}\langle\|a_{t+1,f}-a_{tf}\|^2\rangle_{t,f};
\]

CF 使用 \(0.002\|\theta\|^2\)。谱正则为
\(R_{\rm spec}=0.03d_2^{-1}\langle\|w_{>2}\odot\theta_{f,>2}\|^2\rangle_f\)，其中 \(d_2\) 是 K2 控制数。

PDE 始终以 K12 推进；初态控制依次开放 K2、K4、K8、K12。共同初态连续覆盖两个观测窗。整个物理粒子按 `pid % 5` 分为测试 0、验证 1、训练 2/3/4。冷起点和首窗训练所得起点使用同一规则；验证评分按状态、窗口等权，配对粒子标准误差用于约束起点替换。真场及测试评分不参与选解。
