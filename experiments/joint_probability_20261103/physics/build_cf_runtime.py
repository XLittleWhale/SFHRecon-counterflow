"""Build a new, independently hashed CF runtime; never edit historical bundles."""
from pathlib import Path
from datetime import datetime, timezone
import difflib
import hashlib
import json
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[3]
SOURCE = BASE/'outputs/FrozenB_CFLocalFull_20261031'
OUT = BASE/'outputs/JointProbabilityPhysics_20261103'
TARGET = OUT/'cf_runtime'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError('Expected a unique source anchor: '+repr(before))
    return text.replace(before, after)


def main():
    if TARGET.exists():
        raise RuntimeError('Target already exists; never overwrite a runtime variant')
    old_manifest = json.loads((SOURCE/'bundle_manifest.json').read_text(encoding='utf8'))
    for rel, item in old_manifest['files'].items():
        src, dst = SOURCE/rel, TARGET/rel
        if sha(src) != item['sha256']:
            raise RuntimeError('Historical source hash mismatch: '+rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    patches = []
    changed = []

    def update(rel, value):
        path = TARGET/rel
        previous = path.read_text(encoding='utf8') if path.exists() else ''
        if previous == value:
            return
        patches.extend(difflib.unified_diff(previous.splitlines(True), value.splitlines(True),
                                          fromfile='source/'+rel, tofile='variant/'+rel))
        path.write_text(value, encoding='utf8')
        changed.append(rel)

    rel = 'payload/server_package/counterflow/dynamics.py'
    text = (TARGET/rel).read_text(encoding='utf8')
    text = replace_once(text,
        '        self.__dict__.update(move(data,device));self.device=device',
        '        self.__dict__.update(move(data,device));self.device=device\n'
        '        self.legacy_nu_s=float(self.nu_s);self.nu_s=0.0;self.friction_factor=1.0')
    text = replace_once(text, "F=-(2/3)*c['alpha']", "F=-self.friction_factor*c['alpha']")
    text = replace_once(text,
        '    def __init__(self,data,device):self.__dict__.update(move(data,device));self.d=self.m',
        '    def __init__(self,data,device):\n'
        '        self.__dict__.update(move(data,device));self.d=self.m\n'
        '        if self.cr != 0 or torch.count_nonzero(self.Ds) or torch.count_nonzero(self.ds0):\n'
        "            raise ValueError('Expected existing isotropic nu_s=0 CF geometry')\n"
        '        self.legacy_ck=float(self.ck);self.nu_s=0.0;self.friction_factor=1.0\n'
        "        self.ck=-self.friction_factor*self.p['alpha']*self.p['rho_s_kg_m3']*self.p['kappa_m2_s']")
    update(rel, text)

    rel = 'b_transfer.py'
    text = (TARGET/rel).read_text(encoding='utf8')
    text = replace_once(text, 'Original channel weak momentum, nu_s=0, constant equilibrium L, no q.',
                        '2026-11-03 channel momentum: nu_s=0, F_ns prefactor=1, fixed equilibrium L.')
    text = replace_once(text, '        p = self.p\n',
        '        p = self.p\n'
        '        self.nu_s=0.0;self.friction_factor=1.0\n'
        "        self.ck=-self.friction_factor*p['alpha']*p['rho_s_kg_m3']*p['kappa_m2_s']\n")
    update(rel, text)

    rel = 'run_cf.py'
    text = (TARGET/rel).read_text(encoding='utf8')
    text = replace_once(text, "VERSION='frozen-B-CF-migration-pilot-v1'",
                        "VERSION='B-prior-CF-nus0-Fns1-20261103-v1'")
    text = replace_once(text, '        prior_recalibrated_on_cf=False,automatic_promotion=False,',
        '        prior_recalibrated_on_cf=False,automatic_promotion=False,\n'
        '        evolution_physics=dict(nu_s_m2_s=0.0,Fns_prefactor=1.0),')
    update(rel, text)

    rel = 'ADAPTER_READ_ME.txt'
    text = (TARGET/rel).read_text(encoding='utf8')
    text = text.replace('2/3互摩擦', '系数1的互摩擦（2026-11-03变体）')
    update(rel, '2026-11-03 新物理变体：nu_s=0；F_ns系数=1。B闭合先验与网络暂保持冻结。\n'
                '这改变了闭合所依附的确定性算子；先验尚未在新算子下重估。历史结果不被替换。\n\n'+text)
    for rel in ('run_platform.sh', 'run_platform_smoke.sh', 'RUN_READ_ME.txt'):
        text = (TARGET/rel).read_text(encoding='utf8')
        text = text.replace('FrozenB_CFMigration_20261031', 'CF_nus0_Fns1_20261103')
        text = text.replace('frozen_B_cf_migration_results_20261031', 'cf_nus0_Fns1_results_20261103')
        text = text.replace('frozen_B_cf_migration_smoke_20261031', 'cf_nus0_Fns1_smoke_20261103')
        if rel == 'RUN_READ_ME.txt':
            text = ('2026-11-03 算子变更：超流黏度0，F_ns乘子1。以下B指冻结闭合先验，不表示历史B的同一演化方程。\n'
                    '本次仅完成算子验证，没有启动完整CF再反演。\n\n'+text)
        update(rel, text)

    provenance = dict(version='nus0-Fns1-CF-variant-v1', date='2026-11-03',
        created_utc=datetime.now(timezone.utc).isoformat(),
        historical_parent=str(SOURCE), historical_parent_manifest_sha256=sha(SOURCE/'bundle_manifest.json'),
        historical_contract_sha256=sha(SOURCE/'SOURCE_CONTRACT.json'),
        physics=dict(nu_s_m2_s=0.0,Fns_prefactor=1.0,Fns='-alpha*rho_s*kappa*L*(vn-vs)'),
        previous_CF=dict(nu_s_m2_s=0.0,Fns_prefactor=2/3),
        B_closure_prior_reestimated=False, network_finetuned=False, full_CF_reconstruction_run=False,
        original_observations_operator_tensors_network_calibration_unchanged=True,
        materialized_tensors_note='Channel.ck is overridden from physics at construction; original operator.pt retains its historical baked coefficient.',
        source_files={rel:dict(sha256=item['sha256'],bytes=item['bytes']) for rel,item in old_manifest['files'].items()},
        source_builder=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)))
    update('PHYSICS_CHANGE.json', json.dumps(provenance, indent=2, ensure_ascii=False)+'\n')
    update('LOCAL_EXECUTION.json', json.dumps(dict(
        kind='local-execution-capable-new-physics-variant', inherited_explicit_CPU_flag='--allow-full-cpu',
        execution_performed=False, physics_change_file='PHYSICS_CHANGE.json',
        previous_execution_metadata_sha256=sha(SOURCE/'LOCAL_EXECUTION.json')), indent=2)+'\n')
    update('LOCAL_RUN_READ_ME.txt',
        '新演化算子变体：nu_s=0，F_ns系数=1；B的闭合先验/网络/观测标定冻结。\n'
        '完整CF计算尚未运行。本机若执行完整协议，仍需显式--device cpu --allow-full-cpu。\n'
        '必须使用新输出目录；任何历史选解、积分缓存或resume不可混入本变体。\n')

    contract = json.loads((SOURCE/'SOURCE_CONTRACT.json').read_text(encoding='utf8'))
    contract.pop('contract_fingerprint')
    contract['version'] = 'B-prior-CF-nus0-Fns1-source-v1'
    contract['created_utc'] = provenance['created_utc']
    contract['historical_source_contract_sha256'] = sha(SOURCE/'SOURCE_CONTRACT.json')
    contract['physics_adaptation'].update(Fns_prefactor=1.0,nu_s=0.0,
        prior_reestimated_for_new_deterministic_operator=False)
    for item in contract['runtime_sources']:
        path = TARGET/item['relative_path']
        item['parent_sha256'] = item['sha256']
        item['sha256'] = sha(path)
        item['bytes'] = path.stat().st_size
        if item['sha256'] != item['parent_sha256']:
            item['original_source'] = '2026-11-03 adaptation of '+item['original_source']
    contract['contract_fingerprint'] = hashlib.sha256(json.dumps(contract, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False).encode('utf8')).hexdigest()
    update('SOURCE_CONTRACT.json', json.dumps(contract,indent=2,ensure_ascii=False)+'\n')
    artifact = OUT/'physics'
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact/'cf_runtime_changes.patch').write_text(''.join(patches), encoding='utf8')
    subprocess.run([sys.executable, str(TARGET/'package_tool.py'), 'build'],check=True)
    # Reverify parent after construction, proving no historical file was edited.
    if any(sha(SOURCE/rel)!=item['sha256'] for rel,item in old_manifest['files'].items()):
        raise RuntimeError('Historical source unexpectedly changed')
    result = dict(target=str(TARGET), changed_files=changed,
        new_bundle_manifest_sha256=sha(TARGET/'bundle_manifest.json'),
        parent_all_files_unchanged=True, full_CF_reconstruction_started=False)
    (artifact/'cf_build_report.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
