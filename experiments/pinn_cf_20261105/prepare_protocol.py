"""Freeze the CF transfer after implementation and fixed preconditioning checks."""
from cf_common import *
import cf_field as F
import cf_probability as Q
from cf_history import source_paths, attach_causal_history

DESIGN = '''PINNCF_20261105：当前模拟PINN基线向CF113/CF217的首次受控迁移

目的
在两个既有典型CF实验窗口，用冻结joint粒子概率模型约束统一连续二流场PINN，完成重建及留出评价。
不是直接把模拟输入换文件：壁边界、相机时间尺度和优化坐标必须作明确的数学适配。
所有历史数据、源模型和结果只读；不重训/校准概率网络，不修改Frozen B，不新增自由初场、过程增量或闭合网络。

数据与隔离
固定cf_113_camera100_f0184（90fps）和cf_217_camera100_f0333（180fps），原100ms窗口。
使用真实相机端点位移，各560/1071个区间；fit分别326/594个区间。
物理PID%5>=2为fit，==1为selection，==0为test；同一PID切段或换状态不跨组。
每态先逐PID等权、PID内逐原生区间等权，两态再等权。三数值种子不构成三个独立实验流场。
历史窗口及test在旧方法开发中已探索，不能称首次盲测。
来自原始位置的过去5ms因果PCHIP/割线pilot仅形成网络历史特征；未把插值点当新监督。
每个内部查询历史在真实区间起点后只用过去割线外推，不利用当前端点或未来位置；字段映射逐行校验。
初始粒子速度是上一相机区间割线速度，非观测到的瞬时速度。查询位置是因果pilot，非未知真实中间轨迹。
重建域外历史可作既有条件数据；所有候选场查询仍限定在相对0–100ms。

冻结概率与真实时长适配
沿用模拟基线选定joint家族三个权重，field301030/31/32配对prob411040/41/42，不挑幸运单种子。
网络在1ms律下训练，本轮11.111/5.556ms区间内部按已有最长1ms分片预测，只在真实端点评分一次。
自由态使用原物理仿射传播与冻结learned force均值/方差；保留速度记忆导致的跨片段位移协方差。
捕获态每个符号q条件下，内部联合高斯状态包含(e2,vp2,累计dx2)，二维GH处理非线性张力并保留交叉矩。
不满1ms的隐状态步使用(1-f)恒等核+f完整1ms核，再按符号矩匹配；这是新增迁移假设，不是唯一连续时间推广。
该fractional混合不构成连续时间半群，改变内部步长也会改变创新律，不能称其为纯积分收敛检验。
未建模力创新在内部片段之间独立，片段内按常力传播；与真实CF噪声对应关系尚未实证校准。
真实相机端点只加一次原calibration sigma^2 I；不得把它称为已测CF定位精度。
当前端点先评分再更新(q,e)后验；只有(q,e)跨相机区间保留。vp每新真实区间重置为过去割线，累计dx归零。
因此相机端点间vp/e协方差有意舍弃；这是条件一步预测，不是长时自由滚动轨迹预测。
候选场v、空间梯度、真实时间导数经过概率上下文及滤波完整回传场参数，概率权重和校准参数不更新。
保留粒子方程原加速度约定partial_t u+(grad u)^T u；二流体PDE对流为(grad u)u，不混淆两者。

空间、物理与统一PINN
16×8mm通道，常流壁面无滑移、超流不穿透；视场y周期仍为模型近似，非真实入口出口条件。
两流固定热流均速，常流背景为Poiseuille、超流背景为常量；原冷系数为零，六组均从相同物理背景开始。
K12壁相容无散Shen基每流325个系数；3层128宽Tanh、8组固定时间正余弦网络统一输出两流所有时刻。
同一网络参数控制初始场和后续场，无独立初场步长/正则/优化器。统一表示不保证初始时刻可辨识性相等。
质量归一并对角化固定梯度能量，raw a=Dq，使每个q系数速度能量为U^2/2，与模拟Fourier尺度匹配。
谱半径为max(1,sqrt(梯度特征值)/(2pi/0.01m))，并沿用原1/r输出尺度。
nu_s=0、Fns系数1、均匀固定平衡L0；常流黏度、两流密度和摩擦参数沿原CF标定。
使用冻结弱Galerkin空间动量算子及连续时间真导数；压力由无散投影消去，不训练新的平均流或压力自由度。

损失和固定先验
实际端点二维位移用U*1ms无量纲化，得到原生NLL；每行再除真实毫秒间隔，按PID/状态平均形成NLL/ms。
观测损失=.08*balanced NLL/ms；同时报告原生NLL，不能把NLL/ms解释为虚构1ms独立观测似然。
PDE在201个0.5ms节点计算r=(da/dt-RHS(a))*1ms；左/中/右100段路径分别白化平方后按1/6、4/6、1/6平均。
Frozen B保留完整624 Fourier潜在方向投影到325壁模的推前协方差；常流保留两组时间核，超流保留原白噪声尺度。
完整SVD/两核同时对角化计算精确二次型，不截小奇异方向、不加jitter、不按CF重估能量。
PDE=.08/(2*Nfit_exposure_ms)*(Qn+Qs)，N分别3622.222222和3300ms，而不是原生行数326/594。
物理q上101节点均匀施加幅值.003、邻时差.03、高频.03系数和原/24归一，不对初场单设惩罚。

预条件与预检披露
初次沿用模拟Adam .001、质量谱坐标时，仅两步全fit预检就使PDE代价放大约10^5–10^6倍，记录完整保留。
原因是壁面裁切投影B协方差条件数约10^12，原优化坐标可沿极小容差方向作过大更新；并非概率网络被训练。
修正只用冻结B零滞后协方差：Cq=encode P diag(sigma^2) P^T encode^T，SVD构造完整SPD平方根Lq。
网络输出u先乘1/r，再q=Lq(u/r)。绝对1ms B尺度，不按最大特征值放大、无奇异方向截断。
Lq固定、满秩，不改变可表示场空间、物理q正则或raw PDE/NLL定义；改变Adam坐标及有限预算优化轨迹。
因此这是与CF边界先验相容的优化适配，不宣称与模拟版完全相同优化问题/训练轨迹。
该唯一预条件方案按事先登记30步×两案例×seed301030验证，不做步长搜索、不选预检权重，全部丢弃。
制造导数/壁面/散度/白化/1ms概率回归/长段协方差/因果/reset/padding/梯度和真实fit预检逐项保存。
GH12训练、GH20选择与主评价、GH28敏感性；GH差仅判断证据数值稳健性，不能按test差重新选场或概率。

正式预算与选解
仅PINN，两case×三paired seed共6组，每组同轨迹600次Adam更新，lr=.001，统一clip5，CPU每进程2线程。
200步为主要预算，600步为预先安排延长；每预算保存selection主选与固定终态，共24个快照。
每10步以GH20 selection评分。每态不高于本组冷背景+.01 NLL/ms且balanced优于已选1e-6 NLL/ms才更新。
始终保留step0；即使没有非零解通过保护也完成预定预算并诚实报告选择背景。
6个冷背景和24快照、概率权重及所有源码/数据/规则全局冻结后才评分test。test不参与步长、停止、选解或校准。
唯一主要配对参照为同一新概率/相机适配下的冷背景；旧CF模型使用不同概率/query/Fns，不能直接跨轮相减分数。

评价与交付
30行背景/预算/选解评价及三种子均值：test两态NLL/ms和原生NLL、名义90%二维Rosenblatt区域覆盖、宽度、端点RMSE。
宽度=sqrt(trace全部预测位移协方差))，单位微米；RMSE是预测均值的真实端点位移误差，不是流体场误差。
覆盖率应接近90%而非越高越好；更宽分布或似然改善不能单独证明场更真实。
报告V/test、200/600、主选/终态、GH20/28变化，PDE配点与200个1/4、3/4ms非配点残差、壁面/散度及种子差异。
固定seed301030导出两case各背景+四快照，共10个101帧NPZ及物理单位清楚的流场/涡量/剖面/差分图。
源与checkpoint SHA、网络冻结、选解规则、全部指标/图和成本需独立复核；本轮网络新训练成本为0。
CF没有独立两流真场，不报告真实常流/超流RMS或初场准确度。固定热流均速和PDE小残差不能替代真值检验。
单窗、历史test已探索、稀疏相机历史与模拟分布偏移、固定L0、K12、ADF及相机速度重置均限制结论。
本轮结束后不自行重训概率、重新校准B、改变窗口、追加新实验或CF调参。
'''

def main():
    setup()
    if (OUT/'protocol.json').exists():
        return verify_protocol()
    required = ['field.json','history.json','probability_unit.json','probability_independent_review.json',
                'field_B_output_preconditioning.json','optimization_B_preconditioned_30.json','pipeline_review.json']
    reports = {name:read(OUT/'preflight'/name) for name in required}
    assert reports['probability_unit.json']['passed']
    assert reports['optimization_B_preconditioned_30.json']['passed']
    assert reports['field_B_output_preconditioning.json']['complete']
    assert all(x['complete'] and x['finite_parameter_gradients'] for x in reports['field_B_output_preconditioning.json']['cases'])
    assert reports['history.json']['passed'] and reports['probability_independent_review.json']['passed']
    assert reports['pipeline_review.json']['passed']
    rt = F.runtime()
    contract = rt.verify_contract()
    gate = read(PARENT_OUT/'probability_selection/selection_frozen.json')
    assert gate['complete'] and gate['family']=='joint'
    sources, inputs = {}, {}
    def bind(path, dest):
        path = Path(path).resolve()
        dest[str(path)] = sha(path)
    for row in contract['runtime_sources']:
        path = F.RUNTIME/row['relative_path']
        assert sha(path)==row['sha256']
        bind(path,sources if path.suffix=='.py' else inputs)
    bind(F.RUNTIME/'b_transfer.py',sources)
    bind(F.RUNTIME/'SOURCE_CONTRACT.json',inputs)
    for path in (F.RUNTIME/'B_prior_assets').glob('*'):
        if path.is_file(): bind(path,inputs)
    for path in HERE.glob('*.py'):
        if path.name not in ('audit_results.py',): bind(path,sources)
    for path in (Q.TM.__file__,Q.LM.__file__,Q.CALIBRATION/'calibration.json',Q.CALIBRATION/'force.json'):
        bind(path,sources if Path(path).suffix=='.py' else inputs)
    for path in (PARENT_OUT/'protocol.json',PARENT_OUT/'probability_selection/selection_frozen.json'):
        bind(path,inputs)
    for record in gate['models']:
        assert sha(record['path'])==record['sha256']
        bind(record['path'],inputs)
    # Exact comparison files used by the probability import assertion.
    for relative in ('counterflow/observation.py','counterflow_full/core.py'):
        bind(BASE/'outputs/FullWindowCalibration_20261026/server_package'/relative,sources)
    for path in (OUT/'preflight').rglob('*'):
        if path.is_file() and path.suffix in ('.json','.py','.txt'): bind(path,inputs)
    scopes = {}
    for case in CASES:
        c,z,op,cold,prior = F.load_case(case)
        z = attach_causal_history(c,z)
        for path in source_paths(case).values(): bind(path,inputs)
        flux = '113' if case == CASES[0] else '217'
        fps = '90' if case == CASES[0] else '180'
        bind(BASE.parent/'data/cf'/f'10.10classified_k2_2.0 K_{flux}mWcm2_{fps}fps_Trajectory_v3.xlsx',inputs)
        for filename in ('observations.pt','operator.pt'):
            bind(F.RUNTIME/'payload/server_package/inputs'/case/filename,inputs)
        scopes[case] = dict(fps=c['camera_fps'],exposure_ms=float((z['end']-z['start'])[group_mask(z['pid'],'fit')].sum()),
            groups={g:dict(rows=int(group_mask(z['pid'],g).sum()),pids=z['pid'][group_mask(z['pid'],g)].unique().tolist())
                    for g in ('fit','selection','test')},physics=dict(U=c['physics']['vn_mean_m_s'],L0=op.L0),
            residual_prior=prior.diagnostics)
    design = OUT/'EXPERIMENT_DESIGN.txt'
    design.write_text(DESIGN,encoding='utf8')
    bind(design,inputs)
    for name in ('cf_experiment.py','cf_evaluate.py','cf_plot.py','run_pipeline.py'):
        assert (HERE/name).exists()
    p = dict(frozen=True,version='PINNCF-first-transfer-v1',created_utc=utc(),cases=list(CASES),seeds=list(SEEDS),
        training_seeds=list(TRAIN_SEEDS),steps=600,budgets=[200,600],validation_every=10,field_lr=.001,gradient_clip=5.,
        train_degree=12,validation_degree=20,evaluation_degree=20,evaluation_sensitivity_degree=28,
        guard_nll_per_ms=.01,selection_delta_nll_per_ms=1e-6,probability_family='joint',network_steps=0,
        probability_contract=Q.CONTRACT,physics=dict(nu_s=0.,Fns_factor=1.,L='fixed equilibrium'),
        pde_normalizer='sum_fit_native_durations_ms',pde_nodes=201,field_nodes=101,off_collocation_nodes=200,
        output_preconditioner='full-rank absolute Frozen B zero-lag covariance SPD square root in physical q coordinates',
        inversion_workers=3,threads_per_process=2,evaluation_rows=30,field_npz_count=10,
        all_snapshots_before_test=True,test_used_for_selection=False,no_cf_field_truth=True,
        historical_test_seen=True,confirmatory=False,scopes=scopes,
        environment=dict(python=sys.version,torch=torch.__version__,platform=sys.platform),sources=sources,inputs=inputs)
    p['fingerprint'] = fingerprint(p)
    dump(OUT/'protocol.json',p)
    print('PINNCF_PROTOCOL_FROZEN',p['fingerprint'],flush=True)
    return p

if __name__=='__main__': main()
