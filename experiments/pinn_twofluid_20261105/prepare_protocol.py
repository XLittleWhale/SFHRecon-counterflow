"""Freeze the continuous-field experiment only after non-target preflight."""
from pinn_common import *
from datetime import datetime,timezone

DESIGN='''冻结粒子概率、统一连续二流场：PINNTwoFluid_20261105 第一轮

问题与范围
本轮按用户新指令，把“初场 + 闭合网络 + 剩余过程增量”的目标反演改为整个时空二流场的统一表示。
概率网络维持已训练权重，目标反演只更新场参数。没有目标专属概率微调、神经RHS闭合或自由过程增量。
意图是缓解初场与后续场的参数/优化方式差异；统一参数化不保证初始时刻信息量、梯度或可辨识性相同。
旧实验全部只读；不改正式基线，不自动开展下一轮或CF。

概率模型选择
预先登记selection_plan，以非目标6.5mm/s、PID%5==1、同一个完整95区间物理锚定查询的GH20 balanced NLL，
比较baseline/probability/joint各三种子家族平均，选joint；GH28与零残余差分仅做敏感性，均不改变选择。
joint平均-1.804926，比probability的-1.803320仅好0.001606；不存在逐状态、逐种子或统计意义上的全面支配。
保持三套网络与原场种子配对，不选择最幸运单种子。只加载joint的概率参数，不带入它的旧闭合网络。
原概率包含自由态分布与捕获态因果ADF隐状态转移；先预测当前端点再更新后验，不使用未来观测。
模型参数冻结仍保留概率损失通过候选速度、梯度和时间导数返回场参数的梯度。

两个新的配对场表示
pinn：时间输入MLP(3层128宽Tanh，8组固定正余弦时间特征)输出常/超流各624个Fourier系数。
连续场u(x,t)由同一系数网络与无散周期Fourier K12基合成；共196320个可训练参数。
直接链式求导得到连续时间导数，并用现有64×64谱投影算子求原确定性二流体残差。
这属于连续强时间、空间Galerkin投影的PINN；不是任意空间坐标MLP，也不是过去独立theta加离散DCT过程的弱PINN。
global_spline：统一优化96×2×624节点控制，共119808参数；固定三次Hermite插值，斜率由节点差分导出。
首尾斜率自由，斜率不是新增可训练量；作为同连续概率、同PDE、同正则、同预算的机制对照。
两者全时域都从原冷初值常场出发，seed301030为零扰动，301031/32使用原小扰动；1/ring为统一输出尺度。
没有独立初场参数、初场专属步长、过程增量或神经闭合。NN与Spline参数数目及有效更新尺度不同，不能称等容量。

物理、概率与正则
显式nu_s=0，Fns乘数1，保留恒定L0与既定密度/摩擦/常流黏度，已知两流均值固定。
压力由无散谱投影消去；非零K12模态上约束原确定性动量方程，64网格求非线性项，均值模式不拟合。
相对时间0–95ms对应数据绝对时间5–100ms；191个0.5ms节点计算r=da/dt−RHS(a)。
r乘1ms后使用原Frozen B常流时间相关/超流白噪声尺度作为固定残差容许度；B不训练、不重新标定。
对95个左/中/右残差路径分别白化、平方后按1/6、4/6、1/6加权，避免先平均导致残差抵消。
PDE项=.08/(2*nI)*(Qn+Qs)，全局权重1；仅为软约束，不能宣称原方程被精确满足。
全96时刻均施加原幅值、时间平滑和谱能量系数；把原来只施加初场的谱项推广为全时域平均。
粒子查询使用相对中点的真实连续场和真实时间导数，不扣除PDE残差或过程增量。
严格保留原粒子方程加速度约定：partial_t u+(grad u)^T u；流体对流项则为(grad u)u，二者不同。
两种粒子的概率负对数似然为观测约束，不追加位置MSE、粒子速度等于流体速度或隐变量真值损失。
既有概率在固定锚定query下训练，而此处用候选连续query，仍可能存在context偏移；冻结模型不能自动解决该问题。

非目标检查与冻结
解析空间/时间导数、混合参数梯度、端点、无散/周期、原方程批量一致性、B缩放、概率冻结/因果检查均通过。
首次2步6.5/V试算发现lr=.001可使PDE代价上升，说明目标刚性，不能把finite检查当作收敛。
因此在正式目标运行前写lr_plan，两个臂各用固定[.0001,.0003,.001]、30步非目标V似然+PDE目标；
按第21–30次更新前总目标均值选择各臂步长，不读6.5场真值，不读目标H/reference，不扩展候选或预算。
所有试算权重丢弃；最终步长、代价、震荡情况及成本写lr_selection并与正式协议绑定。
这是单验证工况的工程选择，不能当独立泛化证据。

正式反演和选解
两例1.5/4mm/s×三个配对种子×两场表示，共12个新反演，各600步、Adam、整体梯度clip5、CPU双线程。
I PID%5>=3只拟合；V==1每10步评分选解；H==0仅全局冻结后评价；F==2不用。物理PID所有时间/状态不跨组。
拟合GH12；V GH20；每个候选两态NLL均不高于本组初始值+.01，且balanced低于已选值1e−6才更新。
保留epoch0；在预算200和600分别冻结V主选与固定终态，每组4个checkpoint，合计48个。
200步是与历史200步比较的主要预算，600步仅为预先安排的延长优化诊断，不声称与历史等计算资源。
旧NeuralClosure4DVar的6组joint主/末态共12行作外部参照：概率权重相同，但旧神经闭合、残差query和初场谱正则不同。
因此新旧差异不能只归因于PINN参数化；主要机制对比应为本轮PINN与Spline，且需报告各自成本。
三个概率模型和全部12组场两预算主/末态冻结、SHA核验后，才可读取目标reference或评分H。

评价与交付
共60行(48新+12历史)：绝对5ms初场、相对5ms节点、初5ms六节点均方、全窗常/超流速度RMS，
H自由/捕获NLL、名义90%覆盖、预测分布宽度及逐PID指标；20/28阶敏感性不参与选解。
同时报告200→600、主选→末态变化和计算成本，不能把概率拟合改善当场真值改善。
固定旧场系数，先比较旧残差query与连续样条query，再比较新连续场query，区分query定义和场变化的描述性贡献。
这些贡献依赖分解路径，不是因果识别；不把历史闭合影响误称新概率收益。
全局冻结后在每1ms间隔1/4与3/4位置检查190个非配点PDE残差，判断配点外误差，结果不回流优化。
预定seed301030代表图，导出20个重建和2个参考96帧NPZ；保存PNG/SVG，不按真值挑好看的种子。
最终独立复核哈希、全表、冻结网络、预算选解、敏感性与图，撰写中文CONCLUSIONS。

限制
目标1.5/4mm/s及H在历史开发中已探索，本轮不是真正新盲测；三种子不是独立流场实现。
统一场表示不增加初始时刻观测，仍有概率/动力学/场的可辨识性问题、有限谱分辨率、固定L0和ADF近似。
真实模拟没有重生成；nu_s/Fns变化与数据生成物理可能存在差异；B仍是历史校准残差尺度。
更小PDE残差、验证NLL或更平滑场均不能单独证明真实场更准确。
'''

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():return verify_protocol()
    parent=N.verify_protocol()
    for name in ('query_interface.json','field_independent.json','optimization.json'):
        report=read(OUT/'preflight'/name)
        assert report.get('passed',report.get('analytic',{}).get('passed',False)),name
    lr=read(OUT/'preflight/lr_selection.json')
    assert lr['complete'] and lr['all_weights_discarded']
    from select_field_lr import verify_plan
    lrplan=verify_plan()
    assert fingerprint({k:v for k,v in lr.items() if k!='fingerprint'})==lr['fingerprint']
    assert lr['plan_fingerprint']==lrplan['fingerprint'] and sha(lr['plan_path'])==lr['plan_sha256']
    for item in lr['candidates']:assert sha(item['path'])==item['sha256']
    chosen_lr=lr['field_lr_by_arm']
    assert set(chosen_lr)==set(ARMS) and all(x in (.0001,.0003,.001) for x in chosen_lr.values())
    selection=selection_gate();assert selection['complete'] and selection['family']=='joint'
    assert not selection['target_H_scored'] and not selection['target_reference_read'] and not selection['probability_parameters_updated']
    sources=dict(parent['sources']);inputs=dict(parent['inputs'])
    for path in HERE.glob('*.py'):
        if path.name!='audit_results.py':sources[str(path.resolve())]=sha(path)
    for name in ('experiment.py','evaluate_pinn.py','plot_pinn.py','run_pipeline.py'):
        assert (HERE/name).exists()
    helper=HERE.parent/'joint_probability_20261103/evaluate_joint.py';sources[str(helper)]=sha(helper)
    for path in (NOUT/'protocol.json',NOUT/'selection_frozen.json',NOUT/'training/selection_frozen.json',
                 OUT/'probability_selection/selection_plan.json',OUT/'probability_selection/selection_frozen.json'):
        inputs[str(path)]=sha(path)
    for row in selection['models']:inputs[row['path']]=row['sha256']
    for row in read(NOUT/'selection_frozen.json')['runs']:
        if row['arm']=='joint':
            inputs[row['path']]=row['sha256'];inputs[row['last_path']]=row['last_sha256']
    for path in (OUT/'preflight').glob('*.json'):inputs[str(path)]=sha(path)
    scopes={}
    for U in CASES:
        d,cfg,old,net,cal,force=make_context(U,SEEDS[0]);z=d['z']
        scopes[case_name(U)]={g:dict(rows=int(group_mask(z['pid'],g).sum()),
            pids=sorted(z['pid'][group_mask(z['pid'],g)].unique().tolist())) for g in ('I','V','H','F')}
    design=OUT/'EXPERIMENT_DESIGN.txt';design.write_text(DESIGN,encoding='utf8');inputs[str(design)]=sha(design)
    p=dict(version='continuous-twofluid-pinn-first-round',frozen=True,created_utc=datetime.now(timezone.utc).isoformat(),
        cases=list(CASES),seeds=list(SEEDS),arms=list(ARMS),training_seeds=list(TRAIN_SEEDS),
        budgets=[200,600],steps=600,validation_every=10,field_lr_by_arm=chosen_lr,gradient_clip=5.,
        probability_family='joint',probability_parameter_updates=0,network_steps=0,train_degree=12,
        validation_degree=20,evaluation_degree=20,evaluation_sensitivity_degree=28,guard=.01,selection_delta=1e-6,
        field_architecture=dict(pinn=dict(width=128,depth=3,frequencies=8,parameters=196320),
            global_spline=dict(nodes=96,parameters=119808,slopes='fixed fourth order interior and cubic boundary')),
        spatial_basis='solenoidal Fourier K12; 624 coefficients per fluid; fixed means',physics_grid=64,
        new_physics=dict(nu_s=0.,Fns_factor=1.),process_prior='Frozen B only as PDE residual tolerance/correlation',
        collocation=dict(nodes=191,dt=.0005,quadrature=[1/6,4/6,1/6],residual_increment_scale=.001,pde_weight=1.),
        query='continuous midpoint + actual SI time derivative; no increment subtraction',
        regularization='time-uniform energy/temporal/spectral penalties; no initial-only term',
        target_reference_used_in_fit=False,target_H_used_for_selection=False,all_selections_before_target_evaluation=True,
        historical_target_truth_and_test_seen=True,confirmatory=False,scopes=scopes,
        inversion_workers=3,threads_per_process=2,evaluation_rows=60,auxiliary_rows=24,field_npz_count=22,
        external_reference='Six old joint 4DVar runs; same probabilities but learned closure/query/prior differ',
        sources=sources,inputs=inputs)
    p['fingerprint']=fingerprint(p);dump(OUT/'protocol.json',p)
    print('PROTOCOL_FROZEN',p['fingerprint'],flush=True)
    return p

if __name__=='__main__':main()
