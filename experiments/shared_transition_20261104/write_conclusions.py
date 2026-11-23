"""Write the human interpretation from immutable completed experiment outputs."""
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / 'outputs/SharedTransition_20261104'


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf8'))


def main():
    s = read('TRANSITION_EVALUATION_SUMMARY.json')
    a = read('AUXILIARY_PROBABILITY_ATTRIBUTION.json')
    p = read('PLOT_MANIFEST.json')
    assert s['complete'] and a['complete'] and p['complete']
    assert len(s['runs']) == 48 and len(a['runs']) == 18
    for name in ['preflight/FINAL_NUMERIC_AUDIT.txt', 'preflight/FINAL_PLOT_AUDIT.txt',
                 'diagnostics/FINAL_SELECTION_INTERPRETATION.txt']:
        assert (OUT / name).is_file(), name
    arms = ('legacy', 'cap_retrain', 'independent', 'shared_gru')
    cases = ('vn0_0015m_s', 'vn0_004m_s')
    names = dict(legacy='原冻结网络', cap_retrain='仅重训捕获态',
                 independent='新增自由态独立网络', shared_gru='两态共享GRU')
    lines = [
        '自由态／捕获态转移概率联合建模：重训与反演完整对照结论',
        '2026-11-04，全部本机CPU计算完成；原始协议、已选网络及反演选解保持冻结。', '',
        '结论',
        '新增自由态条件概率模型值得保留为候选，但目前只在4 mm/s算例观察到两相场误差和两态留出NLL同时改善。1.5 mm/s的主选解虽然自由态NLL大幅改善，两相场RMS却变差。硬共享历史GRU没有带来稳定的额外流场收益，不宜直接替换冻结B或据此全量迁移CF。',
        '若下一阶段必须选一个候选，优先保留独立自由态分支作为研究候选；这不是已验证跨工况优于原方案的正式基线升级。共享分支仅保留为消融结果。', '',
        '实验到底做了什么',
        '1. 共完成9组概率网络重训、18组新流场反演，复用上一轮相同条件下6组new_frozen反演作为legacy。每个新臂3个训练shuffle种子，与3个场初值种子固定配对；两目标工况各完整5–100 ms、96帧、K12/N40。',
        '2. 四臂均使用超流黏度νs=0、Fns系数1、原冻结B过程先验。没有重标定先验、重生成原始模拟数据或在本轮启动CF。legacy是上一轮匹配的冷初始化对照，不等同于所有设置都相同的已发布正式基线。',
        '3. 网络在0.5、1、2、2.5、3、5、8.5 mm/s七个非目标工况训练，196988个1 ms区间；6.5 mm/s的9034个区间作网络验证。目标1.5/4 mm/s完全不参与网络重训或选网络。',
        '4. 全部可训练权重从同一历史预训练checkpoint继续训练，至多80轮；不是随机从零训练。网络选解后冻结，再独立反演流场，目标网络更新次数为0。本轮不是在目标反演中同步微调网络。',
        '5. 目标I组（PID%5>=3，204个PID）只拟合场，V组（%5==1，103个PID）选场，H组（%5==0，103个PID）只做最终预测评价；F组（%5==2，102个PID）本轮不用。9个网络与18组主/末态全部冻结后才评分本轮H或读目标真场。',
        '6. 自由态原本已有物理条件位移概率；新增的是历史条件化的未解析力均值和协方差，经原物理响应映射到位移，并非让网络任意输出位移。捕获态仍是物理解码的隐变量混合密度。shared_gru只共享历史GRU，两态条件MLP和输出头保持私有。',
        '7. cap_retrain→independent比较新增自由态学习的整条流程；各臂选中的捕获权重/训练轮次可能不同，不是固定同一捕获网络的纯单因素替换。independent→shared_gru比较本次具体共享方式，且参数容量不同。', '',
        '主结果：按预定验证规则选择的流场',
        '下表均为三配对种子平均；RMS单位mm/s，NLL为原生1 ms条件位移密度的负对数似然，越低越好。连续密度NLL可以为负。',
        '工况 | 模型 | 常流RMS | 超流RMS | H自由态NLL | H捕获态NLL',
    ]
    for case in cases:
        vals = s['summary']['primary']['20'][case]['arms']
        for arm in arms:
            v = vals[arm]
            keys = ('normal_rms_mm_s', 'superfluid_rms_mm_s', 'heldout_free_nll', 'heldout_cap_nll')
            lines.append(f"{s['summary']['primary']['20'][case]['U']*1000:g} mm/s | {names[arm]} | " +
                         ' | '.join(f"{v[k]['mean']:.6f}" for k in keys))
    lines += ['',
        '4 mm/s：新增独立自由态网络，相对仅重训捕获态，常流RMS下降1.10%、超流RMS下降2.08%，自由态NLL下降1.92494，捕获态NLL下降0.04264；三个配对种子均同时改善这四项指标。相对legacy，两场RMS下降1.31%和2.81%。收益存在，但幅度有限，不能称已解决流场重建。',
        '1.5 mm/s：新增独立自由态网络，相对仅重训捕获态，常流RMS上升0.72%、超流RMS上升2.98%；自由态NLL下降3.05264，而捕获态NLL平均上升0.07140。两场退化在三个种子中均发生，因此这里的概率收益不能当作重建收益。',
        '共享GRU相对独立网络：两工况主选解的常/超流RMS均值均略高，没有一个配对种子实现四项主指标同时优于独立网络。1.5 mm/s自由态NLL继续下降0.37387，但捕获态NLL上升0.16703，捕获态退化在三个种子中方向一致。4 mm/s自由态NLL继续下降0.03083；捕获态degree20仅改善0.00545，degree28反为退化0.00317，这个很小的捕获态优势不稳健。',
        '这里“主选解共享无四项同时改善”并不意味着所有场指标在所有种子都变坏，也不覆盖固定200步终态或其他共享结构。', '',
        '概率改善来自哪里',
        '保持同一legacy主选流场不动，仅替换概率网络，再与新网络自己的反演场比较。网络项 = 新网络@旧场 − 旧网络@旧场；场项 = 新网络@新场 − 新网络@旧场。两项相加严格还原总变化。这是给定替换路径的描述性分解，不是因果证明。',
        '工况 | 模型 | H自由态NLL网络项 | H自由态NLL场项 | H捕获态NLL网络项 | H捕获态NLL场项',
    ]
    for case in cases:
        for arm in ('independent', 'shared_gru'):
            v = a['summary'][case][arm]['by_degree']['20']
            nums = [v[component][metric]['mean'] for metric in ('free_nll', 'cap_nll')
                    for component in ('network_change_at_legacy_field', 'field_change_at_new_network')]
            lines.append(f"{s['summary']['primary']['20'][case]['U']*1000:g} mm/s | {names[arm]} | " +
                         ' | '.join(f'{v:+.6f}' for v in nums))
    lines += ['',
        '自由态NLL的大幅下降几乎都已发生在“固定旧场，仅换网络”这一步：独立网络两工况约99.2%和99.6%，共享网络约99.7%和100.2%。大于100%表示随后场变化略微抵消概率收益，不能理解为额外物理信息。',
        '4 mm/s新增自由态网络仍有真实场收益，因为常/超RMS独立评价均下降；同时新网络下的捕获态H分数因场变化进一步改善。1.5 mm/s的场变化反而使捕获态H NLL上升。网络适配成功与流场反演成功必须分开判定。', '',
        '覆盖率和分布宽度',
        '名义90%覆盖率采用既有二维Rosenblatt诊断；spread为位移总协方差迹的平方根，单位μm。',
        '工况 | 模型 | H自由/捕获覆盖率(%) | H自由/捕获spread(μm)',
    ]
    for case in cases:
        for arm in arms:
            v = s['summary']['primary']['20'][case]['arms'][arm]
            cov = '/'.join(f"{100*v['heldout_'+k+'_coverage']['mean']:.2f}" for k in ('free', 'cap'))
            spread = '/'.join(f"{v['heldout_'+k+'_spread_um']['mean']:.6f}" for k in ('free', 'cap'))
            lines.append(f"{s['summary']['primary']['20'][case]['U']*1000:g} mm/s | {names[arm]} | {cov} | {spread}")
    lines += ['',
        '新增网络后，自由态覆盖率上升且平均spread下降，因此本次NLL收益不是简单靠整体放大方差取得的。但所有主结果仍明显低于名义90%，尚不能声称不确定性已校准。场误差、均值偏差、分布形状和训练/反演输入差异都可能影响覆盖率，不能仅据此归咎于方差大小。', '',
        '1.5 mm/s为什么仍差：选解与两相权衡',
        '1.5 mm/s：legacy/cap_retrain三个种子均选第20步；independent/shared_gru均选第10步。各网络从第0步已经生效，没有选回重训前网络的问题。新增自由态的6组只有第0/10步满足保护条件，第20步起自由态V均超出初值+0.01，捕获态保护则始终通过。第20步的均衡V本来也比第10步差；继续优化到140–170步后，均衡V才重新优于第10步，但仍被自由态保护挡住。到200步，均衡V比选中态下降0.1703–0.2571，而自由态V超保护阈值0.2616–0.4448。',
        '200步终态提前约定为诊断，完整保留，不能用H或真值重新选为主结果。1.5 mm/s终态如下：',
        '模型 | 常流RMS | 超流RMS | H自由态NLL | H捕获态NLL',
    ]
    for arm in arms:
        v = s['summary']['terminal']['20']['vn0_0015m_s']['arms'][arm]
        lines.append(names[arm] + ' | ' + ' | '.join(f"{v[k]['mean']:.6f}" for k in
                     ('normal_rms_mm_s', 'superfluid_rms_mm_s', 'heldout_free_nll', 'heldout_cap_nll')))
    lines += [
        '固定200步下，独立自由态网络相对cap_retrain改善常流RMS和两态H NLL，但超流RMS仍更差（0.693974对0.671204）。共享相对独立的两场均值有所改善，却仍有捕获态NLL平均退化；2/3种子四项同时改善也不等于跨种子一致。相对legacy，共享终态仍是常流改善、超流变差。因此早选解影响结果，但不能把全部退化归因于早停。另需注意，新增自由态的六组在200步时，H自由态NLL均比各自10步主选解更差，H捕获态NLL则均更好；保护条件对应真实的两态预测取舍，不能仅把它当作阻碍而事后放宽。',
        '4 mm/s所有臂、所有种子的主选解均为200步，主/终态结果完全相同。',
        '实际场图也显示1.5 mm/s主选常流仍接近初始近均匀场，尚未恢复参考场的主要空间结构；4 mm/s中后期超流已恢复更多结构，但初始帧与整体RMS仍有限。不能只依据几百分点的相对改善宣称重建充分。', '',
        '共享冲突诊断及解释边界',
        '已选共享网络在七个非目标训练工况固定均衡抽样（894行）的两态GRU梯度余弦为+0.1270、−0.1749、+0.2357，自由态/捕获态梯度范数比0.523、0.403、0.609；逐工况负余弦分别1/7、4/7、2/7。2 mm/s在三个种子中均出现负余弦。',
        '可据此判断存在局部、工况相关的冲突，而不能断言全过程持续全局冲突。共享后1.5 mm/s自由态改善、捕获态退化，与负迁移现象相容，但该单快照不能证明唯一机制；参数容量、训练轮数及两态损失尺度也同时影响结果。',
        '训练时的物理context来自非目标模拟参考，而反演时来自尚不准确的候选场，二者差异可能解释“概率适配很强但场改善很小”。这是由流程和结果提出的待检验解释，本轮未单独证明。', '',
        '预算、成本与复核',
        '本机Python 3.12.14 / PyTorch 2.14.0+cpu，无CUDA；批调度最多4任务，每任务2线程。',
        '模型 | 参数量 | 选中epoch（三种子） | 实际末epoch | 平均单次重训分钟 | 平均单次反演分钟',
    ]
    for arm in ('cap_retrain', 'independent', 'shared_gru'):
        tr = [r for r in s['cost']['training_runs'] if r['arm'] == arm]
        fr = [r for r in s['cost']['field_runs'] if r['arm'] == arm]
        lines.append(f"{names[arm]} | {tr[0]['parameter_count']} | " + '/'.join(str(r['selected_epoch']) for r in tr) +
            ' | ' + '/'.join(str(r['last_epoch']) for r in tr) +
            f" | {mean(r['wall_seconds'] for r in tr)/60:.2f} | {mean(r['wall_seconds'] for r in fr)/60:.2f}")
    tb = read('training_batch_status.json')
    pipeline = read('pipeline_status.json')
    stage = {r['stage']: r['seconds'] for r in pipeline['records']}
    lines += [
        f"9组重训实际批耗时{tb['elapsed_seconds']/60:.2f}分钟；18组反演批耗时{stage['field_inversions']/60:.2f}分钟；冻结场{stage['freeze_fields']/60:.2f}分钟、主评价{stage['evaluate']/60:.2f}分钟、绘图{stage['plot']/60:.2f}分钟、辅助归因{a['wall_seconds']/60:.2f}分钟。以上不是把所有并行单次耗时相加。",
        '共享网络比独立网络少3648个参数（22.46%）。单次耗时受同时运行任务数和CPU竞争影响，不能作为严格性能基准。共享/独立选解集中在79–80轮，4 mm/s场选解也到预算终点，均不声称完全收敛。',
        '48行主/末态和18行归因、9/18选解链、冻结网络、汇总代数、20/28阶敏感性及场导出由独立代理复核；具体检查与误差界见preflight/FINAL_NUMERIC_AUDIT.txt、preflight/FINAL_PLOT_AUDIT.txt。选解保护条件逐种子证据见diagnostics/FINAL_SELECTION_INTERPRETATION.txt。',
        '18个NPZ含两套各8个完整重建场与两个参考场；代表种子301030预先固定，没有按误差挑选。图中5/50/100 ms为展示，全窗指标覆盖96帧。', '',
        '建议与下一步边界',
        '保留冻结B和本轮完整对照结果。新增自由态独立分支是更值得继续检验的候选；暂不把硬共享GRU作为默认网络，也不依据目标H调损失权重、保护阈值或挑选终态。',
        '若下一阶段继续研发，优先在新的、非目标开发划分上验证候选场误差下的概率训练，以及两态验证保护与场识别之间的关系，再用尚未参与设计的新工况检验；本轮不自动执行这类新实验。只有在独立分支稳定后，才有理由比较更弱的共享或两态梯度平衡，并重新冻结评价协议。',
        'H仅为同时间窗、给定历史和已知稳定自由/捕获状态的1 ms条件预测；未学习状态切换概率，不是未来自由滚动轨迹。三个种子共享物理样本，不是三个独立物理实验；本轮目标工况历史上已探索，属于开发对照而非全新盲测。', '',
        '结果位置（均相对于本输出目录）',
        '完整逐种子指标与绑定：TRANSITION_EVALUATION_SUMMARY.json、TRANSITION_EVALUATION_REPORT.txt、evaluation_manifest.json',
        '固定旧场替换网络的归因：AUXILIARY_PROBABILITY_ATTRIBUTION.json / .txt',
        '主结果总览：figures/primary/01_summary.png；场优化曲线：figures/04_field_optimization.png',
        '两工况两相5/50/100 ms重建与向量误差图：figures/primary/*_speed.png、*_error.png',
        '200步终态与主选解比较：figures/terminal/；全部96帧数值场：fields/primary/、fields/terminal/',
        '图与场来源：PLOT_MANIFEST.json、PLOT_README.txt；设计：EXPERIMENT_DESIGN.txt',
    ]
    (OUT / 'CONCLUSIONS.txt').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(OUT / 'CONCLUSIONS.txt')


if __name__ == '__main__':
    main()
