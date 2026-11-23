"""Freeze the first latent-transition round before any formal fitting."""
from common import *
from datetime import datetime,timezone
import train_latent as N

DESIGN='''捕获态隐变量时间转移：完整第一轮本机对照（v2梯度修复重跑）

版本缘由
v1零初始化kappa使用torch.clamp；本机torch2.14在零边界梯度为0，导致q持续性输出没有学习。
用户明确授权“同意停止并修复重跑”；v1已停止，源码/协议/全部检查点保留，未进行目标H/reference评价。
v2的kappa数值仍严格等于clamp(raw,0,12)，只显式约定raw=0取右导数1、raw>=12取0。
这项缺陷修复不改变任何概率公式、split、学习率、预算、选解、物理过程或判据。
所有三个head及十二个新反演从原冷初态重新运行；不沿用v1权重，不将两版结果混合。
v1已消耗计算成本另外保留；v2版间审计与q分支学习回归在正式冻结前完成。

目的与三臂
iid：上一轮已经冻结的 independent 自由/捕获边缘概率网络及六组反演。
reset：同一新增转移头，每个非首区间从上一时刻的边缘隐状态先验传播，不吸收以往位移后验。
temporal：同一转移头，从同一PID此前位移过滤得到的隐状态后验连续传播。
reset和temporal使用相同训练选中head，差别仅在历史隐状态后验是否传播；并不各自为reset重新调网络。
两种新臂均冷启动反演，不能直接把同场NLL改善称为场改善。

模型与推断
保持原物理位移emission：捕获符号q、几何delta、未解析力协方差、观测噪声只出现一次。
e=L_q(context)^(-1)(delta-mu_q(context))使用已校准旧条件Gaussian白化。
新增66→32→10转移头2474参数，以当前因果历史编码与前后候选场context为输入。
符号转移softmax(logp0(q_current)+kappa*I_same), kappa∈[0,12]；不假设q永不翻转。
e_current=A e_previous+b+Linnovation epsilon；A谱范数≤.995，b分量∈[-2,2]，
创新Cholesky对角∈[.02,50]、下三角off∈[-.5,.5]。零输出头精确恢复iid。
连续潜变量采用assumed-density filtering：每个q保留一个Gaussian；按预测混合对当前位移评分，再更新后验。
缺帧或可见状态切换重置；完整PID95区间保留跨时间梯度，不进行时间截断。
不增加位置MSE、重复端点NLL、粒子速度真值、任意单粒子补偿参数或目标网络微调。
这是一轮在原离散物理粒子传播上的潜变量时间模型检验，不是已经实施连续Euler–Lagrange PINN。

训练、验证和成本预算
复用上一轮三个independent selected网络：free、cap两个边缘分支完全冻结。
只训练三个新转移头，种子411040/411041/411042。
非目标0.5/1/2/2.5/3/5/8.5mm/s，PID%5>=2；105888连续cap边；等case/PID/PID内边。
监督损失为真正相邻潜变量pair的条件NLL（包括delta Jacobian）；每epoch遍历所有边，随机排列全局小批。
6.5mm/s PID%5==1仅做选解，按完整cap段的因果端点NLL；包含epoch0，不用目标工况选择。
80epochs上限、至少40、20无改善停止，lr1e-4 AdamW，batch1024，clip5，2线程CPU，训练float32/验证float64。
预算及停止规则在正式训练前固定；两个预检pilot epoch权重不复用。

反演与留出
1.5/4mm/s完整5–100ms96帧；K12/N40，νs=0，Fns系数1，冻结B过程先验及原粒子加速度解释。
每新臂×两工况×三种子=12次新反演，另严格哈希/数值校验复用6次iid。
I PID%5>=3拟合初场/常超过程增量；V%5==1选场；H%5==0只在全局冻结后评价；F%5==2不用。
场seed301030/301031/301032配对上述训练seed，零闭合/原冷初值；全网络冻结且保留context梯度。
固定200场更新，lr(theta=.0003,super=.009,normal=.009)，每10步检查V，guard每态initial+.01，balanced降低1e-6才选。
主选与固定第200步均预先报告，共36行；训练3头及全部18场主/末态冻结前不进行正式H/reference评价。
反演GH12，验证/主评价GH20，GH28敏感性；GH差异不等于高斯后验压缩误差的完整评估。
H是同窗因果一步预测，可使用H此前已观测的位移；不是未来无观测自由滚动轨迹。
评价常/超流RMS、H free/cap NLL、90%覆盖率和宽度；固定iid主选场换新模型的12行辅助归因单列。
固定代表seed301030导出14个96帧场文件（含2参考），绘制5/50/100ms图，不按真值挑图。

解释边界
先前目标工况和H在历史开发中被探索；此轮是开发对照，三个种子不代表三次独立物理重复。
低速隐变量强持续、旧边缘概率偏宽：必须将reset−iid的分布变化与temporal−reset的历史后验增益分开。
模型验证收益或位移概率收益不直接代表流场改善；不自动提升为正式基线，不启动CF或第二轮调参。
候选场context漂移、隐变量/场补偿不可辨识及ADF多峰压缩限制仍存在。
预检曾发现零权缺帧填充错误（可能计算全局行0/H哑行），未输出H指标或用于选择；已在正式冻结前修复，详见preflight/integration.json。
'''

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():return N.verify_protocol()
    audit=read(OUT/'preflight/integration.json')
    assert audit['complete']
    synthetic=read(OUT/'preflight/model_synthetic.json')
    parent=S.verify_protocol()
    sources=dict(parent['sources'])
    # As in v1, post-completion read-only audits are not scientific dependencies.
    reports={'audit_results.py','diagnostics_latent.py','q_stickiness_audit.py'}
    for path in HERE.glob('*.py'):
        if path.name not in reports:sources[str(path)]=sha(path)
    helper=HERE.parent/'joint_probability_20261103/evaluate_joint.py';sources[str(helper)]=sha(helper)
    inputs=dict(parent['inputs'])
    for path in (OLD_OUT/'protocol.json',OLD_OUT/'training/selection_frozen.json',OLD_OUT/'selection_frozen.json',
                 OUT/'preflight/integration.json',OUT/'preflight/model_synthetic.json',OUT/'preflight/data_audit.json',
                 OUT/'preflight/version_change_audit.json'):
        inputs[str(path)]=sha(path)
    retired=BASE/'outputs/LatentTransition_20261104'
    for path in (retired/'protocol.json',retired/'STOPPED_NOTICE.txt',retired/'diagnostics/Q_STICKINESS_AUDIT.json'):
        inputs[str(path)]=sha(path)
    for path in T.input_paths():inputs[str(path)]=sha(path)
    for record in read(OLD_OUT/'selection_frozen.json')['runs']:
        if record['arm']=='independent':
            inputs[record['path']]=record['sha256'];inputs[record['last_path']]=record['last_sha256']
    for record in read(OLD_OUT/'training/selection_frozen.json')['runs']:
        if record['arm']=='independent':inputs[record['path']]=record['sha256']
    (OUT/'EXPERIMENT_DESIGN.txt').write_text(DESIGN,encoding='utf8')
    inputs[str(OUT/'EXPERIMENT_DESIGN.txt')]=sha(OUT/'EXPERIMENT_DESIGN.txt')
    p=dict(version='latent-transition-first-round-v2',frozen=True,created_utc=datetime.now(timezone.utc).isoformat(),
        cases=list(CASES),arms=list(ARMS),seeds=list(SEEDS),training_seeds=list(TRAIN_SEEDS),
        training_settings=N.SETTINGS,steps=200,validation_every=10,
        field_lr=dict(theta=.0003,super=.009,normal=.009),train_degree=12,evaluation_degree=20,
        evaluation_sensitivity_degree=28,guard=.01,selection_delta=1e-6,
        new_physics=dict(nu_s=0.,Fns_factor=1.),process_prior='Frozen B, unchanged legacy RHS calibration',
        target_network_updates=0,target_truth_used_in_fit=False,target_H_used_for_selection=False,
        formal_H_reference_only_after_all_heads_fields_frozen=True,
        historical_target_truth_and_test_seen=True,confirmatory=False,
        superseded_version=dict(path=str(retired),reason='zero-boundary clamp derivative blocked q-head learning',
            user_authorized_stop_and_restart=True,old_target_evaluation_completed=False,
            change='identical kappa values; explicit derivative=1 at raw=0; original data, budget and rules'),
        early_preflight_zero_weight_padding_defect_disclosed=True,
        sources=sources,inputs=inputs,threads_per_process=2,workers=4,
        particle_batch_pids=64,continuous_posterior='ADF per-q Gaussian; no time-gradient truncation',
        baseline_reuse='Six original independent frozen reconstructions, complete provenance; no replacement selection',
        secondary_terminal='All fixed step200 states reported; not substitutes for V selections')
    p['fingerprint']=fingerprint(p);dump(OUT/'protocol.json',p)
    print('PROTOCOL_FROZEN',p['fingerprint'],flush=True)
    return p

if __name__=='__main__':main()
