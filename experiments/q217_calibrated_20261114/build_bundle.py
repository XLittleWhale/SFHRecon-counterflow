"""Build a new immutable release around the completed local four-arm validation."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import uuid
import zipfile

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2]
ROOT=BASE/'outputs/UnifiedReconstruction_20261113_GPU'
LOCAL=ROOT/'local_validation_q217_20261114'
RUN=ROOT/'unified_cf_q217_3seeds_200_cuda_fix1/unified_cf_q217_3seeds_200_cuda_fix1'
OLD_STAGE=ROOT/'stage/d7eab5f2/UnifiedCF_Q217_3Seeds_200Steps_GPU_20261113'
NAME='CF217_Calibrated_GPU_20261114'
MODULE='github/experiments/q217_calibrated_20261114'
OLD_MODULE='github/experiments/unified_reconstruction_20261113'
BUILD_RECORD=ROOT/'calibrated_q217_build_work.json'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf8'))
def put(path,text):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf8',newline='\n')
def dump(path,value):put(path,json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))
def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)


def seal_manifest(stage):
    files={p.relative_to(stage).as_posix():sha(p) for p in sorted(stage.rglob('*'))
        if p.is_file() and p.name!='bundle_manifest.json' and '__pycache__' not in p.parts}
    dump(stage/'bundle_manifest.json',dict(name=NAME,version='20261114-v1',sha256=files,
        cases=[217],field_seeds=[301030,301031,301032],pinn_budgets=[200],
        probability_policy='reuse_completed_frozen_CF_models',data_role='same_realization_supervised_calibration'))
    return files


def prepare():
    parent=read(ROOT/'build_validation_q217_3seeds_200.json');archive=Path(parent['archive'])
    if sha(archive)!=parent['sha256']:raise RuntimeError('Original Q217 archive changed')
    final=read(LOCAL/'final_acceptance.json')
    if not final['complete']:raise RuntimeError('Local validation must have completed')
    for name,digest in final['artifacts'].items():
        if sha(LOCAL/name)!=digest:raise RuntimeError('Local validation artifact changed: '+name)
    stage=BASE/'outputs/cf217_build'/uuid.uuid4().hex[:8]/'package'
    with zipfile.ZipFile(archive) as z:
        member=next(n for n in z.namelist() if n.endswith('/bundle_manifest.json'))
        prefix=member.rsplit('/',1)[0]+'/'
        original=json.loads(z.read(member))
        for name,digest in original['sha256'].items():
            data=z.read(prefix+name)
            if hashlib.sha256(data).hexdigest()!=digest:raise RuntimeError('Parent member changed: '+name)
            target=(stage/name).resolve()
            if not target.is_relative_to(stage.resolve()):raise RuntimeError('Unsafe ZIP path')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    # Keep inherited validation under provenance, separate from this release's tests.
    (stage/'provenance').mkdir(exist_ok=True)
    if (stage/'validation').exists():
        (stage/'validation').rename(stage/'provenance/parent_validation')
    copy(stage/'README.txt',stage/'provenance/parent_release_README.txt')
    copy(stage/'release_scope.json',stage/'provenance/parent_release_scope.json')
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in ('.py','.txt'):copy(path,stage/MODULE/path.name)
    code=(stage/OLD_MODULE/'fields.py').read_text(encoding='utf8')
    code=code.replace('import workflow as protocol','import calibrated_workflow as protocol')
    code=code.replace('import workflow as adaptation','import calibrated_workflow as adaptation\nimport calibrated_physics as PHYSICS')
    start=code.index('def context(');end=code.index('\ndef score(',start)
    code=code[:start]+'''def context(out,row,seed,device):
    return PHYSICS.context(out,row,seed,device)

'''+code[end:]
    code=code.replace('direct_field_truth_used_in_pinn_loss=False,','direct_field_truth_used_in_pinn_loss=False,\n        reference_truth_used_for_residual_tolerance=True,geometry_frozen=True,')
    put(stage/MODULE/'calibrated_fields.py',code)
    put(stage/'launch.py','''from pathlib import Path
import sys,runpy
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from verify_bundle import verify
verify()
sys.path.insert(0,str(ROOT/'github/experiments/unified_reconstruction_20261113'))
sys.path.insert(0,str(ROOT/'github/experiments/q217_calibrated_20261114'))
runpy.run_module('calibrated_pipeline',run_name='__main__')
''')
    put(stage/'run_gpu.sh','''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-/root/autodl-tmp/cf217_calibrated_3seeds_200_20261114}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CF_EACH_PDE_CHUNK="${CF_EACH_PDE_CHUNK:-64}"
exec "${PYTHON_BIN:-python}" -u "$ROOT/launch.py" run --device cuda --output "$OUT"
''')
    copy(HERE/'README.txt',stage/'README.txt')
    gate=read(RUN/'probability_frozen.json');source_protocol=read(RUN/'protocol.json')
    if not gate['complete'] or gate['protocol_fingerprint']!=source_protocol['fingerprint']:
        raise RuntimeError('Original probability selection incomplete')
    models=[]
    for row in gate['models']:
        if row['case']!='cf_2K_Q217':continue
        path=RUN/row['path']
        if sha(path)!=row['sha256']:raise RuntimeError('CF probability hash mismatch')
        asset_path=f"revision_assets/probability/seed_{row['seed']}.pt"
        copy(path,stage/asset_path)
        models.append({k:row[k] for k in ('case','seed','training_seed','sha256','parent_sha256')})
        models[-1]['asset_path']=asset_path
        for part in row['stages']:
            for file in ('status.json','history.json'):
                source=(RUN/part['path']).parent/file
                if source.exists():copy(source,stage/'provenance/source_probability'/f"seed_{row['seed']}"/part['stage']/file)
    if [r['seed'] for r in models]!=[301030,301031,301032]:raise RuntimeError('Three source models required')
    for name in ('protocol.json','probability_frozen.json','acceptance.json'):
        copy(RUN/name,stage/'provenance/source_probability'/name)
    for name in ('final_acceptance.json','audit_summary.json','calibration.json','numerical_checks.json',
        'geometry_platform_audit.json','验证报告.txt','结论.txt','validation.png','fields_middle_frame.png'):
        copy(LOCAL/name,stage/'provenance/local_validation'/name)
    for name in ('calibration.json','calibrated_tolerance.pt'):
        copy(LOCAL/name,stage/'revision_assets'/name)
    datasets={}
    for name in ('inversion_observations.pt','heldout_observations.pt','reference_evaluation.npz'):
        relative='outputs/CFSimEachPINN_20261109/prepared/cf_2K_Q217/'+name
        copy(RUN/'datasets/cf_2K_Q217'/name,stage/relative);datasets[name]=relative
    print('EXPORTING_FROZEN_GEOMETRY_OPERATOR_AND_TOLERANCE',flush=True)
    export_physics(stage)
    asset_files=[*datasets.values(),*[r['asset_path'] for r in models],
        'revision_assets/frozen_physics.pt','revision_assets/calibration.json','revision_assets/calibrated_tolerance.pt',
        'provenance/source_probability/protocol.json','provenance/source_probability/probability_frozen.json']
    dump(stage/'revision_assets/asset_manifest.json',dict(datasets=datasets,models=models,
        source_protocol_fingerprint=gate['protocol_fingerprint'],sha256={n:sha(stage/n) for n in asset_files}))
    sources={p.relative_to(stage).as_posix():sha(p) for p in stage.rglob('*.py')
        if '__pycache__' not in p.parts and 'provenance' not in p.parts and 'validation' not in p.parts}
    dump(stage/'revision_assets/runtime_contract.json',dict(source_sha256=sources,
        asset_manifest_sha256=sha(stage/'revision_assets/asset_manifest.json')))
    dump(stage/'release_scope.json',dict(name=NAME,cases=[217],field_seeds=[301030,301031,301032],
        probability_seeds=[411040,411041,411042],pinn_budgets=[200],probability_updates=0,
        probability_policy='reuse completed CF selected_joint',
        parent_archive=archive.name,parent_sha256=parent['sha256'],
        local_validation_manifest_sha256=sha(LOCAL/'final_acceptance.json'),
        numerical_changes=['CF second-moment residual tolerance','identity output scale in energy coordinates',
            'frozen geometry and initial parameters across devices'],
        data_role='same_realization_supervised_calibration',independent_generalization=False,
        cuda_locally_tested=False))
    seal_manifest(stage)
    validation=ROOT/'validation'/('calibrated_q217_'+stage.parent.name)
    validation.mkdir(parents=True,exist_ok=True)
    dump(BUILD_RECORD,dict(stage=str(stage),validation=str(validation),name=NAME))
    print('PREPARED',stage,flush=True)


def export_physics(stage):
    import torch
    sys.path.insert(0,str(OLD_STAGE/OLD_MODULE))
    from runtime import F,load,read as read_original,state
    import fields
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    p=read_original(RUN/'protocol.json');ctx=fields.context(RUN,p['targets'][0],301030,'cpu')
    saved=load(LOCAL/'newB_newScale/background.pt')['field_state']
    for key in ('decode','encode','ring'):
        if not torch.equal(ctx['field'].state_dict()[key],saved[key]):raise RuntimeError('Validated geometry changed')
    tol=load(LOCAL/'calibrated_tolerance.pt');meta=load(RUN/'datasets/cf_2K_Q217/inversion_observations.pt')['meta']
    c=ctx['c'];eye=torch.eye(325).expand(2,-1,-1).clone()
    c['pinn_output_preconditioner']=eye;c['pinn_output_preconditioner_inverse']=eye.clone()
    c['pinn_output_preconditioning']=dict(coordinates='q=output/ring; frozen geometry decode',
        absolute_innovation_scale_used=False,frozen_geometry=True)
    initial={}
    for seed in (301030,301031,301032):
        initial[seed]=state(F.make_field(c,ctx['field'].cold,seed,duration_s=meta['intervals']*.001))
    if any(not torch.equal(v,saved[k]) for k,v in initial[301030].items()):
        raise RuntimeError('Seed 301030 initial state differs from validated experiment')
    data=dict(intervals=meta['intervals'],physics_config=meta['physics_config'],context=c,
        cold=ctx['field'].cold,operator=vars(ctx['op']),initial_fields=initial,
        prior=dict(intervals=meta['intervals'],normal_transform=tol['transforms'][0],
            super_transform=tol['transforms'][1],normal_temporal_cholesky=ctx['prior'].normal_temporal_cholesky),
        data_role='same_realization_supervised_calibration',source_tolerance_sha256=sha(LOCAL/'calibrated_tolerance.pt'))
    torch.save(data,stage/'revision_assets/frozen_physics.pt')


def finalize():
    work=read(BUILD_RECORD);stage=Path(work['stage']);validation=Path(work['validation'])
    evidence=read(validation/'build_checks.json')
    if not evidence['passed']:raise RuntimeError('Build checks incomplete')
    copy(validation/'build_checks.json',stage/'validation/build_checks.json')
    for name in ('acceptance_cpu.json','complete.json','evaluation/summary.json'):
        copy(validation/'smoke'/name,stage/'validation/cpu_smoke'/name)
    copy(validation/'smoke/fields/cf_2K_Q217/seed_301030/history.json',stage/'validation/cpu_smoke/history.json')
    files=seal_manifest(stage);output=ROOT/(NAME+'.zip')
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in [*files,'bundle_manifest.json']:
            path=stage/name;info=zipfile.ZipInfo.from_file(path,arcname=NAME+'/'+name)
            info.compress_type=zipfile.ZIP_DEFLATED
            if path.suffix=='.sh':info.external_attr=0o100755<<16
            z.writestr(info,path.read_bytes(),compresslevel=6)
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:raise RuntimeError('Archive CRC failed')
        for name,digest in files.items():
            if hashlib.sha256(z.read(NAME+'/'+name)).hexdigest()!=digest:raise RuntimeError('Archive member mismatch')
    report=dict(archive=str(output),sha256=sha(output),bytes=output.stat().st_size,stage=str(stage),
        files=len(files)+1,zip_crc_and_member_sha_verified=True,build_checks_passed=True,
        cases=[217],field_seeds=[301030,301031,301032],budgets=[200],probability_updates=0,
        cpu_full_data_two_step_smoke_passed=True,cuda_locally_tested=False,
        server_cuda_acceptance_required=True,validation_directory=str(validation))
    dump(ROOT/'build_validation_q217_calibrated_20261114.json',report)
    put(output.with_suffix('.zip.sha256'),report['sha256']+'  '+output.name+'\n')
    copy(stage/'README.txt',ROOT/'README_Q217_Calibrated_20261114.txt')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','finalize'])
    args=parser.parse_args()
    prepare() if args.action=='prepare' else finalize()
