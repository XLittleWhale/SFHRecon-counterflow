"""Freeze scientific sources, allowed data and budgets after non-target checks."""
from common import *
from datetime import datetime,timezone
import training as N
import data_training as D

DESIGN='''学习二流体闭合与粒子概率模型：第一轮本机四管线对照

目的
初始场仍通过全窗弱约束4D-Var优化；跨非目标算例学习常/超流确定性闭合和两态粒子概率。
核心问题是5ms初始场、全窗两流场、两态留出粒子预测能否共同改善，而非只看训练似然。

四条管线与来源
baseline：复用LatentTransition_20261104_v2的temporal概率和6组场，逐SHA/张量/V重放验证。
closure：学习确定性闭合，原temporal概率完全冻结。
probability：原显式物理与B随机过程不变，通过非目标短窗反演任务训练粒子概率。
joint：从同种子closure已选模型出发，与原temporal概率一起接受短窗反演外层训练。
三个训练seed411040/41/42与场seed301030/31/32配对。9个新训练结果，18个新目标反演，复用6个baseline。
joint的闭合可不同于closure；四管线不是严格2x2因子设计，joint-closure不能解释为纯概率效应。

物理和模型
显式nu_s=0、Fns系数1，恒定L与现有无散Fourier K12/N40；常/超流已知均值固定。
568参数局部不变量网络，输出两流应力及内部交换力张量基系数；内部交换按密度加权抵消。
神经闭合进入每个RK4子步的RHS；旧B的零均值端点创新与协方差保留、没有重新标定。
粒子local_jets只扣剩余随机端点创新，不扣神经确定性闭合；均值贡献Phi_closed-Phi_phys单独导出。
参数冻结并不截断候选场context、闭合或时间传播的梯度；目标反演网络更新严格为0。
不引入每目标时空MLP、单粒子任意补偿、重复位置MSE、粒子速度等于流体速度约束。
学习闭合可能产生有效耗散或附加摩擦；显式参数保持不变不代表有效动力学未改变。

非目标数据与训练
训练0.5/1/2/2.5/3/5/8.5 mm/s；验证仅6.5 mm/s；只使用非目标reference.a[96,2,624]作场监督。
七例只有665个1ms全场转移；196988粒子训练行不是独立流场样本。40网格真场不等于K12投影，标签不混用。
训练PID%5>=2，其中inner用>=3、outer用==2；验证仅PID%5==1，再按floor(PID/5)%2分内外。
端点似然按状态/PID均衡；捕获使用完整短窗因果过滤，先评分再更新，窗口首端重置，不使用未来端点作为输入。
闭合先接受真场推进监督；probability/joint通过短窗口两步可微初场反演、outer初场/整段场及独立粒子评分联合学习。
两步内层不是充分收敛反演；短窗首端重置不等于目标95区间的完整记忆，属于本轮明确近似。
训练内层是无随机创新、谱预条件SGD的5ms初场代理；目标为含原B创新、原坐标Adam的95ms弱约束反演。
本轮保留原B过程自由度以隔离学习确定性均值的作用，不检验低维过程替代；代理超梯度非零不保证长窗迁移。
监督概率锚定从固定纯物理新算子重建非目标真场query及free-force标签，不让标签随可学习闭合漂移。
固定真场锚定和可学习候选场query属于不同评估条件，必须分别解释；不直接复用旧prepared context作新模型输入。
选模型仅用非目标验证，包含epoch0；无量纲组合选择分数的各项单列，不把组合改善称为各项同时改善。
训练验证场分数是归一化模态MSE，不是目标评价的mm/s RMS；组合分数按各臂自己的epoch0归一，不能跨臂直接排名。
改进候选还需通过非目标完整95步无创新滚动的有限性/稳定门槛；具体设置见protocol.training_settings。
所有正式训练预算、采样、超参、停止和选择规则在本次冻结中绑定。pilot权重丢弃且成本单列。

目标反演及评价
1.5/4mm/s、5–100ms96帧；I PID%5>=3拟合，V%5==1选解，H%5==0只在全局冻结后评价，F不用。
原冷初值、原B初场/过程正则与Adam组步长；每组200步，每10步V，两态均不得超过本组初始NLL+.01且balanced改善1e-6。
主要结果按V选择；固定step200另列。全部9训练选择与24场主/末态冻结之前不评分目标H，不读取目标reference。
训练/反演积分阶数与敏感性分别绑定；目标反演12阶、V/主评价20阶、敏感性28阶。
48行主/末态：初始5ms、50ms、100ms和全窗常/超RMS；H自由/捕获NLL、名义90%覆盖、预测宽度和逐PID数据。
辅助18行：probability概率@baseline查询、joint概率@baseline查询、joint概率@closure查询；固定完整(a,eta_s,eta_n)。
代表seed301030导出两工况四臂两模式16组场加2参考，共18个96帧NPZ和预定图，不按真值挑种子。
训练/反演/复用成本分开；检查闭合均值、残余创新、实际初场更新量、验证保护和200步收敛迹象。
joint继承closure训练，单臂部署成本应含该前置成本；全实验资源总和则不重复累计这一已执行的训练。
概率校准、短窗训练收益不能代替流场重建结论；不自动升级正式基线。

限制与范围
1.5/4mm/s历史真值和H已在开发中探索，本轮不是全新盲测；三个种子也不是三个独立物理实现。
ADF后验近似、候选context偏移、初场/过程/粒子隐变量可辨识性仍需据结果判断。
本轮只做获授权的模拟对照，不自动启动下一轮调参、连续PINN或CF。
所有历史源码/数据/结果只读保留；本轮为独立新模型分支。
'''

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():return verify_protocol()
    required=('closure_model_selftest.json','integration.json','data_audit.json','window_coverage.json',
              'training_preflight.json','independent_numerics.json','independent_review.json','evaluation_integration_review.json')
    reports={name:read(OUT/'preflight'/name) for name in required}
    for name in ('closure_model_selftest.json','integration.json','training_preflight.json','independent_numerics.json','independent_review.json'):
        assert reports[name].get('passed',reports[name].get('complete',False)),name
    for path,digest in reports['independent_review.json']['source_shas'].items():
        assert sha(path)==digest,'Source changed after independent review: '+path
    parent=read(VOUT/'protocol.json')
    assert fingerprint({k:v for k,v in parent.items() if k!='fingerprint'})==parent['fingerprint']
    sources=dict(parent['sources'])
    # Read-only result audits/reports may be refined after fitting; model/evaluator may not.
    for path in HERE.glob('*.py'):
        if path.name!='audit_results.py':sources[str(path.resolve())]=sha(path)
    helper=HERE.parent/'joint_probability_20261103/evaluate_joint.py'
    sources[str(helper.resolve())]=sha(helper)
    inputs=dict(parent['inputs'])
    for path in (VOUT/'protocol.json',VOUT/'training/selection_frozen.json',VOUT/'selection_frozen.json'):
        inputs[str(path)]=sha(path)
    for r in read(VOUT/'training/selection_frozen.json')['runs']:
        inputs[r['path']]=r['sha256']
    for r in read(VOUT/'selection_frozen.json')['runs']:
        if r['arm']=='temporal':
            inputs[r['path']]=r['sha256'];inputs[r['last_path']]=r['last_sha256']
    for path in D.input_paths():inputs[str(path)]=sha(path)
    for name in required:
        path=OUT/'preflight'/name;inputs[str(path)]=sha(path)
    design=OUT/'EXPERIMENT_DESIGN.txt';design.write_text(DESIGN,encoding='utf8');inputs[str(design)]=sha(design)
    p=dict(version='neural-closure-4dvar-first-round',frozen=True,created_utc=datetime.now(timezone.utc).isoformat(),
        cases=list(CASES),arms=list(ARMS),seeds=list(SEEDS),training_seeds=list(TRAIN_SEEDS),
        training_settings=N.SETTINGS,closure_settings=N.CLOSURE_SETTINGS,steps=200,validation_every=10,
        field_lr=dict(theta=.0003,super=.009,normal=.009),train_degree=12,evaluation_degree=20,
        evaluation_sensitivity_degree=28,guard=.01,selection_delta=1e-6,
        new_physics=dict(nu_s=0.,Fns_factor=1.),process_prior='Frozen B residual innovations unchanged',
        closure_mean='continuous deterministic RHS, integrated at every RK4 stage',
        query_increment='residual stochastic endpoint increments only',target_network_updates=0,
        target_truth_used_in_fit=False,target_H_used_for_selection=False,
        formal_H_reference_only_after_all_models_fields_frozen=True,historical_target_truth_and_test_seen=True,
        confirmatory=False,sources=sources,inputs=inputs,threads_per_process=2,training_workers=3,inversion_workers=4,
        baseline_reuse='Six v2 temporal selected/terminal fields with exact frozen-law regression',
        secondary_terminal='All fixed step200 states; not replacement for V selections',
        evaluation_rows=48,auxiliary_rows=18,field_npz_count=18)
    p['fingerprint']=fingerprint(p);dump(OUT/'protocol.json',p)
    print('PROTOCOL_FROZEN',p['fingerprint'],flush=True)
    return p

if __name__=='__main__':main()
