# Flow Reconstruction of Superfluid Counterflow

## Abstract

超流氦中的示踪粒子同时受到常流、超流及局部涡结构作用，轨迹不能直接等价为任一流体速度。本文使用条件隐变量分布描述涡符号和虚拟锚点位移，通过源一致的粒子受力解码构造离散位移约束，再由动量–Vinen 方程推进共享初态，实现二维粗粒化二流场的变分重建。自由态采用段内相关误差的解析边缘化，捕获态保留物理混合似然；周期与 CF 使用同一网络和优化版本。三个周期短窗算例的常流/超流分量 RMS 误差为 0.097/0.315、0.512/0.793、5.664/6.323 mm/s。模态截断仅占总误差平方约 1.5%–18%。实验不同时间段表现不一致，且缺少场真值。本方法已证明条件性的重建能力，但尚未证明跨工况稳定精度、完整可辨识性或场不确定性的概率校准。

## 1. Introduction

PIV 和 PTV 为超流氦流动提供了重要观测手段，但自由粒子与涡捕获粒子的运动机制不同。自由粒子的常流阻力提供较直接的常流约束；捕获粒子还受到张力、Magnus 力和涡致互摩擦作用，不能直接作为超流速度示踪。连续二流场的重建必须同时处理流体动力学与不可直接观测的粒子受力。

宏观二流体方程适合流道尺度推进，却不能从速度场和标量涡线密度确定某一粒子的局部涡几何。本文学习轨迹历史和候选场条件下的物理隐变量分布，再由粒子方程生成观测概率约束；待反演量仅为共享时间起点的两流场模态系数。

二维参考数据来自平面点涡–粒子系统，不再将三维场投影直接视为满足二维方程的真值。但点涡源系统与粗粒化重建 PDE 仍有闭合差异。周期有真值验证与 CF 无场真值验证提供不同层次的证据，必须分别解释。

## 2. Theory Background

### 2.1 二流体场与互摩擦约定

记 $\rho=\rho_n+\rho_s$。等温、低马赫数实现中分别要求

$$
\nabla\cdot\mathbf V_n=\nabla\cdot\mathbf V_s=0,
\qquad \rho_n\langle\mathbf V_n\rangle_S+\rho_s\langle\mathbf V_s\rangle_S=0.
$$

给定热流时，以 SI 单位计算 $U=Q_h/(\rho sT)$，取 $\overline{\mathbf V}_n=U\mathbf e_y$、$\overline{\mathbf V}_s=-(\rho_n/\rho_s)U\mathbf e_y$。这些均值是输入，不是本轮识别的未知量。重建推进方程为

$$
\begin{aligned}
\partial_t\mathbf V_n+(\mathbf V_n\cdot\nabla)\mathbf V_n
&=-\nabla\Pi_n+\nu_n\Delta\mathbf V_n+\mathbf F_{ns}/\rho_n+\mathbf g_n,\\
\partial_t\mathbf V_s+(\mathbf V_s\cdot\nabla)\mathbf V_s
&=-\nabla\Pi_s+\nu_s^{\rm eff}\Delta\mathbf V_s-\mathbf F_{ns}/\rho_s+\mathbf g_s.
\end{aligned}
$$

$\Pi_f$ 是运动学压力，$\mathbf g_f$ 维持已知平均驱动；不将含梯度的向量写成标量有效压力。$\nu_n=\mu_n/\rho_n$；周期采用 $\nu_s^{\rm eff}=\alpha\kappa$，CF 采用 $\nu_s^{\rm eff}=0$，仍保留互摩擦耗散。

约定 $\mathbf F_{ns}$ 为作用于常流的单位体积力，$\mathbf V_{ns}=\mathbf V_n-\mathbf V_s$。三维无极化各向同性涡切向分布满足 $\langle\mathbf t\mathbf t^T\rangle=I_3/3$，给出

$$
\mathbf F_{ns}=-\frac23\alpha\rho_s\kappa L\mathbf V_{ns},
\qquad \mathbf F_{ns}\cdot\mathbf V_{ns}\leq0.
$$

系数 $2/3$ 来自三维取向平均，不是纯二维点涡恒等式；CF 壁面附近的各向同性仍为假设。

### 2.2 涡线密度与源系统差异

有限平均体积内，$L=|\mathcal V_\Delta|^{-1}\int_{\mathcal L\cap\mathcal V_\Delta}d\xi$，$\mathbf F_{ns}=L\langle\mathbf f_{ns}\rangle_{\mathcal L,\Delta}$。$L$ 的量纲为 $\mathrm m^{-2}$；从单位长度力到体积力不能遗漏 $L$。重建中推进

$$
\partial_tL+\nabla\cdot(L\mathbf V_s)
=\alpha_v|\mathbf V_{ns}|L^{3/2}-\beta_vL^2+\gamma_v|\mathbf V_{ns}|^{5/2}.
$$

最后一项是额外现象学源项，不是经典两项 Vinen 平衡。使用 $\chi=\log(L/L_0)$，$L_0=[(\alpha_v/\beta_v)(\rho/\rho_s)U]^2$，联合时间段开始时 $\chi=0$。$L_0$ 不参与反演，第二窗口不重置；$\gamma_v\ne0$ 时该初值并非精确稳态。实现中以 $L=L_0\exp[\operatorname{clip}(\chi,-12,8)]$ 计算互摩擦和反应项。

源点涡系统采用

$$
\mathbf F_{ns}^{h}=-\alpha\rho_s\kappa L^h\mathbf v_{ns}^h
+\alpha'\rho_s\omega_\ell^hJ\mathbf v_{ns}^h,
\qquad J=\begin{pmatrix}0&-1\\1&0\end{pmatrix}.
$$

源耗散系数为 1，并含有有符号横向项，与重建的 $2/3$ 闭合不同。参考粗粒化场未必精确属于重建 PDE 解空间。统一约定 $J(x,y)=(-y,x)$。

### 2.3 粒子受力与物理隐变量

令 $\Lambda_p=4\pi a_p^3/3$、$m_p^*=(\rho_p+\rho/2)\Lambda_p$、$D_p=6\pi\mu_na_p$。常规附加质量推导使用物质加速度，但当前源程序和解码器实际使用

$$
\mathbf A_f=\partial_t\mathbf u_f+\nabla\frac{|\mathbf u_f|^2}{2},\qquad
\mathbf b_p=\frac32\Lambda_p\sum_f\rho_f\mathbf A_f+(\rho_p-\rho)\Lambda_p\mathbf g.
$$

若 $\omega_f=\partial_xu_{f,y}-\partial_yu_{f,x}$，则

$$
\mathbf A_f-(\partial_t+\mathbf u_f\cdot\nabla)\mathbf u_f=-\omega_fJ\mathbf u_f.
$$

因此有旋流中不能将其直接写成物质导数。保持源一致性不证明其对真实实验完全准确。自由态满足

$$
\dot{\mathbf x}_p=\mathbf v_p,\qquad m_p^*\dot{\mathbf v}_p=D_p(\mathbf u_n-\mathbf v_p)+\mathbf b_p.
$$

捕获态增加

$$
\mathbf F_{\rm vortex}=G_pqJ(\mathbf v_p-\mathbf u_s)-D_v(\mathbf v_p-\mathbf u_n)
+G_vqJ(\mathbf v_p-\mathbf u_n)+\mathbf F_T.
$$

$q=\pm1$，$G_p=\rho_s\kappa\ell_e$，$D_v=\ell_e\Gamma$，$G_v=\ell_e\Gamma'$，其中

$$
d_\alpha=(1-\alpha')^2+\alpha^2,\quad
\Gamma=\frac{\rho_s\kappa\alpha}{d_\alpha},\quad
\Gamma'=\frac{\rho_s\kappa[\alpha'(1-\alpha')-\alpha^2]}{d_\alpha}.
$$

双叉乘阻尼项不携带额外 $q$，因为 $q^2=1$。两锚点高度为 $\pm h$，当前数据平面偏移相同，故

$$
\mathbf F_T=2T_\ell\frac{\boldsymbol\delta}{\sqrt{h^2+|\boldsymbol\delta|^2}},\qquad
z=(q,\mathbf d),\quad\mathbf d=\boldsymbol\delta/a_p.
$$

锚点按照受载弛豫方程 $m_a\dot{\mathbf u}_a=-D_a(\mathbf u_a-\mathbf u_0)+qG_aJ(\mathbf u_a-\mathbf u_0)-\mathbf f_{T,a}$ 演化。几何限幅、捕获、释放和再注入是约化规则；几何修正冲量不全部反馈至粒子和流体。学习只使用稳定状态区间，不建模状态切换概率。$z$ 只代表虚拟锚点模型，而非任意三维涡丝构型。

## 3. Methodology

### 3.1 离散位移物理解码

不将位置二阶差分作为瞬时加速度标签。周期轨迹先展开，定义

$$
\mathbf w_{p,k}=\frac{\mathbf x_{p,k}-\mathbf x_{p,k-1}}{U\Delta t},\qquad
\mathbf y_{p,k}=\frac{\mathbf x_{p,k+1}-\mathbf x_{p,k}}{U\Delta t}.
$$

$U\mathbf w$ 是区间平均速度，近似而非精确瞬时初速。区间内冻结场与隐变量，令

$$
\begin{aligned}
M_q&=[-(D_p+D_v)I+(G_p+G_v)qJ]/m_p^*,\\
\mathbf f_q&=(D_p+D_v)\mathbf u_n-G_pqJ\mathbf u_s-G_vqJ\mathbf u_n+\mathbf b_p,\\
H_q&=\int_0^1e^{u\Delta tM_q}du,\qquad B_q=\int_0^1(1-u)e^{u\Delta tM_q}du.
\end{aligned}
$$

捕获态无量纲位移均值为

$$
\boldsymbol\mu^C(q,\mathbf d;\mathbf V)=H_q\mathbf w+
\frac{\Delta t}{m_p^*U}B_q[\mathbf f_q(\mathbf V)+\mathbf F_T(a_p\mathbf d)].
$$

自由态令涡致系数及张力为零，取 $b_F=D_p\Delta t/m_p^*$、$H_F=(1-e^{-b_F})/b_F$、$B_F=(1-H_F)/b_F$，有

$$
\boldsymbol\mu^F=H_F\mathbf w+\frac{\Delta t}{m_p^*U}B_F(D_p\mathbf V_n+\mathbf b_p).
$$

该响应对冻结线性系统精确，对变化的受力、位置和初速只是近似。候选场使用区间两端平均及时间差分，在已观测位置采样。

### 3.2 条件隐变量模型与训练

输入包含五帧归一化位移速度及相对位置历史 $\mathcal H$、两流相对速度 $(\mathbf V_f/U-\mathbf w)$、梯度 $\Delta t\nabla\mathbf V_f$ 和五个已知物性条件。GRU 隐维数 32，条件向量维数 17，输出

$$
p_\theta(q,\mathbf d\mid\mathcal H,c)=\pi_{\theta,q}
\mathcal N_2(\mathbf d;\boldsymbol\mu_{\theta,q},C_{\theta,q}C_{\theta,q}^T).
$$

边缘化隐变量获得捕获态位移密度

$$
p^C_{\theta,K}(\mathbf y\mid\mathcal H,\mathbf V)=\sum_q\int
\mathcal N_2(\mathbf y;\boldsymbol\mu^C(q,\mathbf d;\mathbf V),\sigma_{C,K}^2I)
p_\theta(q,\mathbf d\mid\mathcal H,c(\mathbf V))d\mathbf d.
$$

这是涡符号与高斯锚点的物理混合模型，不是当前重训的 RealNVP。高斯锚点未截断至源几何上限；张力映射有界不等于所有隐变量样本都符合生成器几何约束。

训练速度为 0.5、1、2、2.5、3、5、8.5 mm/s；6.5 mm/s 验证；1.5、4、10 mm/s 用于本轮重建评价。源数据采用 1 cm×1 cm 周期域、512 个粒子（初始捕获 128 个）、100 ms 记录和 1 ms 保存间隔。粒子半径 2.3 μm、密度 145.6 kg/m³。不能将初始捕获数写为总粒子数。

在 $\mathcal K=\{2,4,8,12,16,18\}$ 中对每个样本随机选择阶数，同一参考投影场同时用于条件与受力解码：

$$
\mathcal L_{\rm train}=\frac1N\sum_i\mathbb E_K\left[
-\log p_\theta(q_i^*,\mathbf d_i^*\mid\mathcal H_i,c(P_K\mathbf V_i^*))
-0.1\log p^C_{\theta,K}(\mathbf y_i\mid\mathcal H_i,P_K\mathbf V_i^*)\right].
$$

位移训练项采用固定无量纲噪声 0.3。六个视图不是六份独立数据；参考场不是输出标签，但确实进入训练条件和受力，因此不自动保证对错误候选场或实验域泛化。

当前为已有 H18 检查点的两种子续训，而非从零训练。分别完成 13/14 轮、533/574 次更新，选中第 1/2 轮，以 K12/16/18 验证目标均值选取第二种子供所有算例使用。早停不是收敛证明。随后重标定符号温度、锚点尺度与位移噪声；周期反演固定使用 K12 噪声律，CF 保留历史实验噪声律。历史数据已反复用于开发，不是新的盲测。

### 3.3 自由态相关误差及 CF 采样

自由态无需涡隐变量网络。一个含 $m$ 个 1 ms 增量的片段，其轴向残差向量为 $\mathbf e_{p,a}$。解析边缘化零均值随机截距，得到

$$
C_{p,a}=\sigma_F^2(I_m+\beta_a\mathbf1\mathbf1^T),\qquad
\ell_p^F=\frac1{2m}\sum_{a=x,y}
[\mathbf e_{p,a}^TC_{p,a}^{-1}\mathbf e_{p,a}+\log\det C_{p,a}+m\log(2\pi)].
$$

当前 $\beta_x=0.129478$、$\beta_y=1.423373$，不逐粒子拟合偏差。常量残差方向的精度降低为独立模型的 $1/(1+m\beta_a)$，其他方向不变。它减少持续误差重复计权，但也削弱同形真实场变化的证据，不修正受力均值。捕获态仍使用 $\ell_p^C=-m^{-1}\sum_k\log p^C_{\theta,12}$，未增加捕获态相关修正。

CF 以 g1/g3 为捕获态、g2 为自由态。自由态按原始 180 fps 相机位移评分，前一原始割线提供初速及因果路径预测；相机区间内组合完整/部分 1 ms 响应，不在未观测中间点重置速度。令 $W_{ij}$ 为相机区间与第 $j$ 个 PDE 输出区间的重叠时长除以 1 ms：

$$
C^{\rm cam}_{p,a}=\sigma_{F,\rm CF}^2[WW^T+\beta_a(W\mathbf1)(W\mathbf1)^T].
$$

位移以 $U\times1\,\mathrm{ms}$ 归一化，负对数密度除以 $\mathbf1^TW\mathbf1$。$W=I$ 时恢复原 1 ms 算子。此为迁移的误差模型，不是已校准的相机噪声；初速及共享端点定位误差尚未独立标定。

捕获态保留平滑插值后的 1 ms 回顾性复合约束，其原始相机位移误差只作积分诊断。该状态间非对称性仍是局限。插值不增加独立信息，1 ms 输出不代表 1 ms 实验可辨识分辨率。

### 3.4 共享初态动量反演与选解

周期使用无散 Fourier 基；CF 使用壁面兼容流函数基，常流无滑移、超流不可穿透。CF 的 $\nu_s^{\rm eff}=0$，不施加超流零切向梯度。局部域 16 mm×8 mm，轴向 Fourier 周期性只是有限视场近似，不是已验证的入口/出口条件。

无散、边界法向为零的测试函数使 $\langle\mathbf b_i,\nabla\Pi\rangle=0$，故压力由动量投影消去，不需要压力观测，也不是令压力为零。令 $\mathbf V_f=\mathbf V_f^0+\sum_i a_{f,i}\mathbf b_{f,i}$，只优化初始能量坐标 $\boldsymbol\zeta_0$。周期 K12 总计 1248 个初始速度控制系数。

全程使用 K12 PDE，仅将初始控制模态依次开放 K2→K4→K8→K12；低阶仍可调整，后续 PDE 可生成所有保留阶数。当前不使用历史跨阶软锚定项或高阶硬幅值上限。输出间隔 1 ms，内部 RK4 子步 0.5 ms。

周期 [5,9] 与 [13,17] ms 两观测窗共享 5 ms 初态并连续推进，无观测间隙不伪造数据。CF 每块的两个连续 12 ms 窗口共享初态。同步推进 $L$，不增加 $\delta F_{ns}$ 或神经流体闭合。

对拟合粒子–窗口片段集合 $\mathcal P_{\rm fit}$，将每段负对数评分按 1 ms 观测时长归一化为 $\ell_p$，目标为

$$
\mathcal J=\frac{0.08}{N_{\rm seg}}\sum_{p\in\mathcal P_{\rm fit}}\ell_p
+R_{\rm base}+R_{\rm high}.
$$

两种状态按实际段数进入训练目标，不人为各占一半；不额外加入位移均值 RMS。正则为

$$
\begin{aligned}
R_{\rm base}^{\rm per}&=\frac{0.003}{24}\operatorname{mean}_{k,f}\|\mathbf a_{f,k}\|^2
+\frac{0.03}{24}\operatorname{mean}_{k,f}\|\mathbf a_{f,k+1}-\mathbf a_{f,k}\|^2,\\
R_{\rm base}^{\rm CF}&=0.002\sum_f\|\boldsymbol\zeta_{f,0}\|^2,\\
R_{\rm high}&=\frac{0.03}{d_2}\operatorname{mean}_f\sum_{j\in H}w_{f,j}^2\zeta_{f,j,0}^2.
\end{aligned}
$$

$H$ 是初态 K2 以上模态；周期/CF 的 $d_2$ 为 24/15。周期 $w_j=(|\mathbf k_j|/2)^2$；CF 为速度梯度 Rayleigh 商与根模态带最大商之比。这些系数不是校准的场先验协方差。

物理粒子 ID 跨窗与状态统一划分：模 5 余数 2/3/4 为拟合、1 为验证、0 为测试，无最终全数据 refit。比较背景起点和首窗独立反演初态起点的两条联合路径；验证按状态/窗口等权，并检查每类粒子配对评分退化不超过一个配对标准误。这是启发式选解，不保证场误差不退化。

所有算例使用相同预算规则。周期完成计划更新，CF 触及时间上限，未证明达到驻点。神经参数冻结，但梯度仍同时经过候选场条件和解码器。流体向前推进，伴随向后传播；不是流体向后演化。

## 4. Results

### 4.1 周期场误差

源周期节点分片线性插值后作保守面积积分，统一到 40×40 网格，再作模态投影。评价覆盖 5–17 ms 的 13 帧，而非完整 100 ms。分量 RMS 为 $E_f=[(2N_tN_g)^{-1}\sum_{k,i}|\widehat{\mathbf V}_{f,k,i}-\mathbf V^*_{f,k,i}|^2]^{1/2}$。

| 平均常流速度（mm/s） | 常流/超流误差（mm/s） | 模态截断误差（mm/s） | 相对真场波动 RMS 的误差 |
|---|---|---|---|
| 1.5 | 0.0966 / 0.3153 | 0.0394 / 0.1089 | 15.1% / 47.6% |
| 4 | 0.5124 / 0.7934 | 0.2173 / 0.2885 | 34.4% / 45.1% |
| 10 | 5.6640 / 6.3233 | 0.7005 / 0.7999 | 57.2% / 63.2% |

相对误差分母是真场相对已知均匀均值的波动 RMS，不是平均常流速度。正交投影满足

$$
\|\widehat{\mathbf V}_K-\mathbf V^*\|^2
=\|\widehat{\mathbf V}_K-P_K\mathbf V^*\|^2+\|(I-P_K)\mathbf V^*\|^2.
$$

截断只占总误差平方约 1.5%–18%，主要误差在现有空间内部。截断是表示下界，而非 PDE 和观测下保证可达的精度。提高 K 可以影响优化，但缺少高阶表示并非误差的唯一原因。

相对同一自由相关旧版，1.5/4 mm/s 场误差略升约 0.05%–1.30%。1.5 mm/s 两种状态的留出 NLL 未通过单标准误门槛，不等价于显著退化。10 mm/s 拟合组没有自由态片段，且超过最高训练速度 8.5 mm/s，不是双状态信息充分的测试。

### 4.2 CF 观测一致性

CF 为 2.0 K、217 mW/cm²，$U\simeq7.466$ mm/s。261 与 505 是同一记录的两个 24 ms 时间段，不是独立实验工况。

| 时间段 | 场 | 捕获态相机位移均值 RMS（μm） | 自由态相机位移均值 RMS（μm） |
|---|---|---|---|
| 261 | 重建 | 16.491 | 20.375 |
| 261 | 背景推进 | 23.561 | 24.107 |
| 505 | 重建 | 29.986 | 19.930 |
| 505 | 背景推进 | 30.635 | 14.695 |

捕获态列为回顾性积分诊断，自由态列为原始相机因果响应。测试物理粒子数分别为捕获/自由 4/6 和 5/3，样本较少。第二时段自由态差于背景，不能以捕获态改善掩盖。CF 没有场真值，背景不是参考真值。混合均值可能落在概率分支之间；均值误差和似然不可互换，二者均不能直接认证全场精度。

### 4.3 历史比较与数值验证

历史同预算测试支持在所测 4 mm/s 工况共享初态优于独立窗口。后续捕获态相关修正改善部分周期场，却恶化 CF 自由态预测，故当前所有算例统一采用自由态相关版，不拼接不同版本最优结果。

本轮完整梯度检查、物理粒子隔离和场文件读回通过。内部步长减半造成的场差小于 0.01 mm/s，远低于实际误差。这验证所测数值链路，不证明物理闭合正确或全局收敛。绘图组中的零粒子壁面/黏度对照是正向敏感性计算，不是四次新反演。

## 5. Identifiability and limitations

### 5.1 两流场约束不等强

在正确指定的统计模型下，可辨识性要求

$$
p(Y\mid\mathcal S(a_0))=p(Y\mid\mathcal S(b_0))\Longrightarrow a_0=b_0.
$$

唯一正则化最优解不证明该命题。当前复合评分不是完整原始轨迹生成概率，也未完成全参数信息谱或唯一性证明。局部响应为

$$
\delta Y=A_n\delta\mathbf V_n+A_s\delta\mathbf V_s+B\delta z.
$$

能被隐变量受力变化抵消的场方向较弱；独立潜变量与测量误差的局部高斯近似给出 $\Sigma_{\rm eff}=\Sigma_\epsilon+B\Sigma_zB^T$。训练分布提供统计假设，不增加独立测量。这不证明经 PDE 和边缘化后存在精确退化；比较观测分量数与速度未知数也不足以证明无解。

此前冻结模型下，仅将参考超流初态波动替换为均匀值，捕获态留出 NLL 在 4/1.5 mm/s 分别变差约 4.244/7.786，自由态变化很小。因此捕获轨迹含有超流信息，但不证明全部模态可恢复。这些是历史同工况诊断，不是当前重训的新测量，不跨工况比较绝对信息量。此前低秩均值响应探测也未覆盖完整信息空间，不能将未保留方向直接判为无信息。

### 5.2 模型偏差与流场误差的混淆

令 $Y=\mathcal G(a_\star)+b+\epsilon$，其中 $b$ 包括粗粒化 PDE、受力与采样近似偏差。对恒定协方差的局部高斯目标，记 $J=D\mathcal G(a_\star)$、$\Lambda=\nabla^2R(a_\star)$，有近似

$$
\widehat a-a_\star\simeq(J^T\Sigma^{-1}J+\Lambda)^{-1}
[J^T\Sigma^{-1}(b+\epsilon)-\nabla R(a_\star)].
$$

这不是实际混合目标的精确公式，但说明场敏感方向上的持续误差会造成场偏差。相关降权不消除非零均值误差，增强正则则权衡稳定性与偏差。

历史“参考初态→原重建初态”连线中，每个候选均经同一 PDE 回放，训练目标改善却伴随留出评分变差。这支持过拟合和目标偏离，不把所有误差都归于物理闭合，也不排除优化盆地问题。现有证据不支持“条件网络必然吞掉超流响应”，换网络本身也不是已证实的解决办法。

### 5.3 可达程度

定义 $\mathcal M_K=\{\mathcal S(a_0):a_0\in\mathcal A_K\}$ 及

$$
E_{\rm dyn}=\inf_{a_0\in\mathcal A_K}\|\mathcal S(a_0)-P_K\mathbf V^*\|.
$$

参考不在该轨道集合时，仅改初态不能精确匹配。当前未全局求出此量；单次参考初态回放是一个可行误差，不是最优误差下界。延长共享窗口既增加信息，也可能累积动力学误差。

当前能确认的是给定均值、边界、物性、类别和闭合下的短窗粗场恢复。已有低/中速测试中常流更准确，超流仍有显著波动误差；高流速与 CF 未获稳定验证。表中为已达精度，不是理论极限或新工况承诺。

基底之外的小尺度未表示；弱响应、稀疏覆盖方向更多依赖 PDE 和正则，不能直接称为数学上不可观测。平均流量与初始 L 是给定量；压力被投影消去；完整三维涡结构及出平面动力学不在恢复范围。粒子预测覆盖率不是流场后验区间。当前不能给出有证明的最大可辨识阶数或统一最终误差百分比。

## 6. Conclusion

物理隐变量解码、自由/捕获态互补观测与共享初态动量反演构成了可复现统一基线，具有条件性二维二流场重建能力。但统一版本不等于各工况稳定最优，改善粒子拟合不等于改善流场。现有主要误差发生在已保留模态内。后续应优先确定独立轨迹支持的场方向、区分持续观测偏差与场扰动，使实验两类算子匹配原始采样，再对预先留出的粒子与时间块验证。增加网络规模、K 或迭代长度不能替代这些工作。

## 版本与结果来源

2026-10-23 更新。基线为 `../outputs/UnifiedFreeCorrelation_20261014`，绘图为 `../outputs/Results/UnifiedFreeCorrelation_20261014`。历史诊断来自 `LikelihoodDirections_20261012`、`Identifiability_20261012`，不与本轮重训结果混记。完整源文件对应关系见 `UPDATE_NOTES_20261023.txt`。本次仅修订文稿，不新增训练/重建。原稿及原 PDF 保存在 `archive_before_20261023`；完整英文推导保留在同目录 LaTeX 文件中。
