"""Derive a scope-only Q217/three-seed/200-step release from the verified full ZIP."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import uuid
import zipfile

BASE = Path(__file__).resolve().parents[3]
ROOT = BASE/'outputs/UnifiedReconstruction_20261113_GPU'
NAME = 'UnifiedCF_Q217_3Seeds_200Steps_GPU_20261113'
MODULE = 'github/experiments/unified_reconstruction_20261113'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf8'))
def put(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf8', newline='\n')
def dump(p, value): put(p, json.dumps(value, ensure_ascii=False, indent=2))
def replace(p, before, after):
    s=p.read_text(encoding='utf8')
    if s.count(before)!=1: raise RuntimeError('Unexpected parent source: '+str(p))
    put(p,s.replace(before,after))

def build():
    parent=read(ROOT/'build_validation_full.json');archive=Path(parent['archive'])
    if sha(archive)!=parent['sha256']:raise RuntimeError('Parent ZIP changed')
    stage=ROOT/'stage'/uuid.uuid4().hex[:8]/NAME
    with zipfile.ZipFile(archive) as z:
        manifest_name=next(n for n in z.namelist() if n.endswith('/bundle_manifest.json'))
        prefix=manifest_name.rsplit('/',1)[0]+'/'
        manifest=json.loads(z.read(manifest_name))
        for name,digest in manifest['sha256'].items():
            data=z.read(prefix+name)
            if hashlib.sha256(data).hexdigest()!=digest:raise RuntimeError('Parent member changed: '+name)
            if any(case in name.split('/') for case in ('cf_2K_Q113','cf_2K_Q175')):continue
            if name==MODULE+'/build_bundle.py':continue
            target=(stage/name).resolve()
            if not target.is_relative_to(stage.resolve()):raise RuntimeError('Path escapes release')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    # Only scope, entry-point routing and documentation change. Numerical kernels
    # and all A/B/closure/joint settings remain byte-identical to the full release.
    workflow=stage/MODULE/'workflow.py'
    replace(workflow,"version='unified-baseline-full-v2-20261113'","version='unified-q217-three-seed-200-v1-20261113'")
    replace(workflow,"cases=[217] if smoke else [113,175,217]","cases=[217]")
    replace(workflow,"budgets=[1,2] if smoke else [200,600]","budgets=[1,2] if smoke else [200]")
    pipeline=stage/MODULE/'pipeline.py'
    replace(pipeline,'for q in (113,175,217):',"for q in workflow.defaults(args.profile)['cases']:")
    replace(pipeline,"acceptance.run(args.device));return","acceptance.run(args.device,tuple(workflow.defaults(args.profile)['cases'])));return")
    replace(stage/MODULE/'acceptance.py','cases=(113,175,217)','cases=(217,)')
    config=read(stage/MODULE/'config.json')
    config.update(version='unified-q217-three-seed-200-v1-20261113',cases=[217])
    config['field']['budgets']=[200];dump(stage/MODULE/'config.json',config)
    put(stage/'run_gpu.sh','''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-/root/autodl-tmp/unified_cf_q217_3seeds_200_20261113}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "Q217 only; three seeds; full A/B/closure/joint; PINN ends at 200 steps"
exec "${PYTHON_BIN:-python}" -u "$ROOT/launch.py" run --device cuda --profile baseline_retrain_all --output "$OUT"
''')
    original_readme=(stage/'README.txt').read_text(encoding='utf8')
    put(stage/'specification/PARENT_FULL_RELEASE_README.txt',original_readme)
    readme='''CF 217 子任务：三种子完整概率复训 + 200步 PINN 重建

本包范围
仅2K、217 mW/cm^2，保留180fps原生轨迹，三个独立种子。
概率种子411040/411041/411042分别对应PINN种子301030/301031/301032。
默认完整A -> B -> closure -> joint -> 概率冻结 -> PINN -> 全部选解冻结 -> H/真值评估与重建导出。
每个种子PINN仅200步；不运行600步，不包含113/175的训练及重建数据。
A/B仍最多80轮、最少40轮、patience20；closure/joint仍最多40轮、最少20轮、patience10。
只缩小工况范围与PINN预算，未降低概率训练预算、GH阶数或改变训练损失。
沿用周期selected_joint初始化；本批次结果为同实现监督校准，不是独立场泛化或盲重建。

运行（在解压后的包目录内）
  bash run_gpu.sh /root/autodl-tmp/unified_cf_q217_3seeds_200_20261113

需已安装CUDA版PyTorch>=2.2、Python>=3.10、numpy、scipy、psutil的AutoDL镜像。
完全离线运行，不在线安装库；本包不包含依赖wheels或CUDA驱动。
启动后自动校验SHA、封存协议、执行217 CUDA数值验收，然后完成三种子训练与重建。
断点续跑：对同一输出目录重跑同一命令；完整模型、优化器、随机状态及选解记录恢复。
必须使用本子任务的新输出目录；不能续写原三工况/600步协议的输出目录。

检查计划
  python launch.py plan --output /root/autodl-tmp/q217_plan
分阶段（相同输出目录；默认同一个完整概率训练profile）
  python launch.py probability --device cuda --output /root/autodl-tmp/unified_cf_q217_3seeds_200_20261113
  python launch.py fields --device cuda --output /root/autodl-tmp/unified_cf_q217_3seeds_200_20261113
  python launch.py evaluate --device cuda --output /root/autodl-tmp/unified_cf_q217_3seeds_200_20261113
smoke仍是217单种子、缩小数据和1/2步预算的诊断，不代表正式200步结果。

结果
最终完成标志：complete.json及evaluation/summary.json的complete=true。
probability/cf_2K_Q217/seed_*/下保存A/B/closure/joint的状态和检查点。
fields/cf_2K_Q217/seed_*/下保存background、budget_200_primary、budget_200_terminal。
evaluation/cf_2K_Q217/seed_*/下保存重建网格；summary.json汇总H及两流体场误差。
三个概率模型先全部冻结；三个PINN选解全部冻结后才进行H和最终真值评价。

成本参考
历史RTX5090的217冻结模型PINN约54.32秒/步（含验证）。
据此本任务三种子各200步约9.05小时，仅是历史单步成本外推，不是本包GPU实测。
还需另加完整概率训练、CUDA验收、最终评估与导出时间；概率复训目前没有匹配GPU计时。

验证与溯源
release_scope.json记录完整母包SHA与本次范围变更。
validation/subtask_scope_checks.json记录本子包正式配置、三种子权重加载、协议封存与重复校验。
validation/end_to_end_cpu_smoke.json和full_method_acceptance_cpu.json是母包数值/端到端证据；
这些历史证据不替代本子包服务器的CUDA验收，程序不会将它们当作正式训练完成记录。
本子包数值核及训练控制器与母包逐字节一致；修改仅涉及范围、入口路由和文档。
specification/PARENT_FULL_RELEASE_README.txt是完整母包说明，范围以本README及release_scope.json为准。
原方法规范保留在specification/；其历史未实现状态不代表当前包状态。
'''
    put(stage/'README.txt',readme);put(stage/MODULE/'README.txt',readme)
    scope=dict(name=NAME,cases=[217],probability_seeds=[411040,411041,411042],
        field_seeds=[301030,301031,301032],pinn_budgets=[200],profile='baseline_retrain_all',
        parent_archive=archive.name,parent_sha256=parent['sha256'],
        probability_training_budget_unchanged=True,numerical_kernels_unchanged=True,
        excluded_cases=[113,175,481],parent_validation_is_inherited_evidence=True)
    dump(stage/'release_scope.json',scope)
    changed=[]
    for name,digest in manifest['sha256'].items():
        if (stage/name).exists() and sha(stage/name)!=digest:changed.append(name)
    expected={MODULE+'/'+n for n in ('workflow.py','pipeline.py','acceptance.py','config.json','README.txt')}|{'README.txt','run_gpu.sh'}
    if set(changed)!=expected:raise RuntimeError('Unexpected changed files: '+str(changed))
    scope['modified_parent_files']=changed;dump(stage/'release_scope.json',scope)
    def seal_manifest():
        files={p.relative_to(stage).as_posix():sha(p) for p in sorted(stage.rglob('*'))
            if p.is_file() and p.name!='bundle_manifest.json' and '__pycache__' not in p.parts}
        m=dict(manifest,name=NAME,sha256=files,cases=[217],field_seeds=scope['field_seeds'],
            probability_seeds=scope['probability_seeds'],pinn_budgets=[200],parent_archive_sha256=parent['sha256'])
        dump(stage/'bundle_manifest.json',m);return files
    seal_manifest()
    validation=ROOT/'validation'/('q217_subtask_'+stage.parent.name);validation.mkdir(parents=True)
    probe='''import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/'github/experiments/unified_reconstruction_20261113'))
import workflow,common_training,method_contract
from runtime import *
root=Path(sys.argv[1]);out=Path(sys.argv[2]);config=workflow.defaults()
assert config['cases']==[217] and config['field']['budgets']==[200]
assert config['field_seeds']==[301030,301031,301032]
assert config['probability_seeds']==[411040,411041,411042]
assert set(config['stages'].values())=={'retrain'}
assert config==read(HERE/'config.json')
assert common_training.settings()['A']['epochs']==80
assert common_training.settings()['joint']['epochs']==40
method_contract.verify_spec()
seeds=[]
for seed in config['field_seeds']:
    net,cal,force,digest=B.load(seed,'cpu')
    seeds.append(dict(field_seed=seed,probability_seed=B.MODELS[seed][0],sha256=digest))
p=workflow.seal(out);again=workflow.seal(out)
assert p==again==workflow.verify(out)
assert len(p['references'])==len(p['targets'])==1
assert p['targets'][0]['name']=='cf_2K_Q217'
assert len(p['config']['field_seeds'])*len(p['targets'])==3
for path in root.rglob('*'):
    assert not any(x in path.parts for x in ('cf_2K_Q113','cf_2K_Q175'))
dump(root/'validation/subtask_scope_checks.json',dict(passed=True,config=config,
    three_seed_models_loaded=seeds,seal_and_resume_protocol_equal=True,formal_runs=3,
    field_steps_each=200,formal_gpu_training_executed=False))
print('SCOPE_AND_THREE_SEED_BINDINGS_PASS')
'''
    commands=[('plan',[sys.executable,str(stage/'launch.py'),'plan','--output',str(validation/'plan')],0),
        ('scope',[sys.executable,'-c',probe,str(stage),str(validation/'sealed_scope')],0),
        ('formal_cpu_guard',[sys.executable,str(stage/'launch.py'),'run','--device','cpu','--output',str(validation/'cpu_guard')],1)]
    # Scope check writes evidence, so refresh manifest before each subsequent entry.
    for label,cmd,expected_code in commands:
        seal_manifest()
        with (validation/(label+'.log')).open('w',encoding='utf8') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode!=expected_code:raise RuntimeError('Check failed: '+str(validation/(label+'.log')))
        if label=='formal_cpu_guard' and 'formal runs use CUDA' not in (validation/(label+'.log')).read_text():
            raise RuntimeError('Unexpected guard failure')
    files=seal_manifest();output=ROOT/(NAME+'.zip')
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in [*files,'bundle_manifest.json']:
            p=stage/name;info=zipfile.ZipInfo.from_file(p,arcname=NAME+'/'+name)
            info.compress_type=zipfile.ZIP_DEFLATED
            if p.suffix=='.sh':info.external_attr=0o100755<<16
            z.writestr(info,p.read_bytes(),compresslevel=6)
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:raise RuntimeError('ZIP CRC failed')
        for name,digest in files.items():
            if hashlib.sha256(z.read(NAME+'/'+name)).hexdigest()!=digest:raise RuntimeError('Final archive hash mismatch')
    report=dict(archive=str(output),sha256=sha(output),bytes=output.stat().st_size,stage=str(stage),
        cases=[217],seeds=scope['field_seeds'],budgets=[200],profile=scope['profile'],
        zip_crc_and_member_sha_verified=True,scope_checks_passed=True,
        parent_numerical_kernels_unchanged=True,cuda_locally_tested=False,
        validation_directory=str(validation))
    dump(ROOT/'build_validation_q217_3seeds_200.json',report)
    put(output.with_suffix('.zip.sha256'),report['sha256']+'  '+output.name+'\n')
    put(ROOT/'README_Q217_3Seeds_200Steps.txt',readme)
    print(json.dumps(report,indent=2))

if __name__=='__main__':build()
