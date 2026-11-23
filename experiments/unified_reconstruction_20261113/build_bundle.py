"""Build and exercise the complete offline each baseline-method pipeline."""
import shutil
import subprocess
import zipfile
from runtime import *
import method_contract as contract
import baseline_api

ROOT=BASE/'outputs/UnifiedReconstruction_20261113_GPU'
NAME='UnifiedCF_FullBaseline_GPU_20261113'
OLD=BASE/'outputs/CFSimFrozenPINN_20261110/stage/CFSimFrozenPINN_GPU_20261110'
SOLVER=BASE/'outputs/CFReference_2K_20261106/CFReference_2K_AutoDL_Galantucci_1s_50frames_20261108'

def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)

def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value,encoding='utf8',newline='\n')

def build():
    contract.verify_spec();dump(HERE/'config.json',contract.config())
    stage=ROOT/'stage'/uuid.uuid4().hex[:8]/NAME;stage.mkdir(parents=True)
    original=read(OLD/'bundle_manifest.json')['sha256']
    prefixes=('outputs/JointProbabilityPhysics_20261103/cf_runtime/',
        'outputs/FullWindowCalibration_20261026/',
        'outputs/PINNTwoFluid_20261105/probability_selection/',
        'github/experiments/pinn_cf_20261105/',
        'github/experiments/shared_transition_20261104/',
        'github/experiments/latent_transition_20261104_v2/')
    for name,digest in original.items():
        if name.startswith(prefixes):
            if sha(OLD/name)!=digest:raise RuntimeError('Historical dependency changed: '+name)
            copy(OLD/name,stage/name)
    for name in ('native_probability.py','fast_score.py','field_adapter.py','frozen_probability.py','probability_test_fixtures.py'):
        copy(FROZEN/name,stage/(FROZEN/name).relative_to(BASE))
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in ('.py','.json','.pt','.txt'):copy(path,stage/path.relative_to(BASE))
    for name,digest in read(HERE/'source_contract.json')['sha256'].items():copy(BASE/name,stage/name)
    source=HERE.parent/'cf_sim_each_pinn_20261109/data_adapter.py';copy(source,stage/source.relative_to(BASE))
    for name,digest in read(HERE/'fixed_slot_certificate.json')['implementation_sha256'].items():
        if sha(SOLVER/name)!=digest:raise RuntimeError('Solver certificate mismatch')
        copy(SOLVER/name,stage/'reference_solver'/name)
    for q in contract.config()['cases']:
        case=f'cf_2K_Q{q}'
        path=BASE/'outputs/CFSimEachPINN_20261109/prepared'/case/'probability_observations.pt'
        audit=read(path.parent/'data_audit.json')
        if sha(path)!=audit['output_sha256'][path.name]:raise RuntimeError('Prepared observations changed')
        copy(path,stage/path.relative_to(BASE))
        for name in ('inversion_observations.pt','heldout_observations.pt','reference_evaluation.npz','data_audit.json'):
            source_path=path.parent/name
            if name!='data_audit.json' and sha(source_path)!=audit['output_sha256'][name]:raise RuntimeError('Target input changed: '+name)
            copy(source_path,stage/source_path.relative_to(BASE))
        source=ROOT/'validation/existing_data'/case
        for name in ('latent_labels.pt','label_provenance.json','window_audit.json',
                     'full_resolved_context.pt','training_reference_fields.pt','force_context_audit.json'):
            copy(source/name,stage/'prepared_supervision'/case/name)
        evidence=read(source/'label_provenance.json')
        context=load(source/'full_resolved_context.pt')
        if evidence['observation_sha256']!=sha(path) or context['meta']['latent_label_sha256']!=sha(source/'latent_labels.pt'):
            raise RuntimeError('Supervision row bindings changed')
        valid=context['free_force_valid']
        if not bool(torch.isfinite(context['free_force'][valid]).all()) or not bool(torch.isnan(context['free_force'][~valid]).all()):
            raise RuntimeError('Invalid force-target mask or fabricated cap force')
    for name in ('README.txt','training_contract.json','audit_existing.py','audit_results.json'):
        copy(BASE/'outputs/UnifiedReconstruction_20261113_revision'/name,stage/'specification'/name)
    copy(HERE/'README.txt',stage/'README.txt')
    local_acceptance=ROOT/'validation/full_method_acceptance_cpu.json'
    if local_acceptance.exists():copy(local_acceptance,stage/'validation/full_method_acceptance_cpu.json')
    put(stage/'verify_bundle.py','''from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
def verify():
    m=json.loads((ROOT/'bundle_manifest.json').read_text(encoding='utf8'))
    for name,digest in m['sha256'].items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT):raise RuntimeError('Path escapes bundle')
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1048576),b''):h.update(b)
        if h.hexdigest()!=digest:raise RuntimeError('Bundle SHA mismatch: '+name)
    print('BUNDLE_VERIFIED',len(m['sha256']),flush=True)
if __name__=='__main__':verify()
''')
    put(stage/'launch.py','''from pathlib import Path
import sys,runpy
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from verify_bundle import verify
verify()
sys.path.insert(0,str(ROOT/'github/experiments/unified_reconstruction_20261113'))
runpy.run_module('pipeline',run_name='__main__')
''')
    put(stage/'run_gpu.sh','''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-/root/autodl-tmp/unified_cf_full_20261113}"
PROFILE="${2:-baseline_retrain_all}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "Each supervised A/B/closure/joint + frozen probability + PINN; profile=$PROFILE"
exec "${PYTHON_BIN:-python}" -u "$ROOT/launch.py" run --device cuda --profile "$PROFILE" --output "$OUT"
''')
    put(stage/'requirements_runtime.txt','Python>=3.10; installed CUDA PyTorch>=2.2, numpy>=1.26, scipy>=1.11, psutil>=5.9\nNo network installation; no wheels included.\n')
    files={p.relative_to(stage).as_posix():sha(p) for p in sorted(stage.rglob('*')) if p.is_file()}
    manifest=dict(name=NAME,sha256=files,scope='each',formal_training_enabled=True,
        package_purpose='complete supervised continuation, closure/joint and frozen-probability PINN reconstruction',
        capabilities=contract.capabilities(),source_norm_sha256=sha(BASE/'outputs/UnifiedReconstruction_20261113_revision/README.txt'),
        forced_new_references=False,endpoint_only_fallback=False,raw_high_resolution_fields_included=False,
        original_prepared_FV_and_derived_supervision_included=True)
    dump(stage/'bundle_manifest.json',manifest)
    validation=ROOT/'validation'/('full_package_'+stage.parent.name);validation.mkdir(parents=True,exist_ok=True)
    for name,args,expected in [('plan',['plan'],0),('formal_cpu_guard',['train','--device','cpu'],1),('end_to_end',['smoke','--device','cpu'],0),('resume',['smoke','--device','cpu'],0)]:
        output=validation/('end_to_end' if name=='resume' else name)
        cmd=[sys.executable,str(stage/'launch.py'),*args,'--output',str(output)]
        print('PACKAGE_CHECK',name,str(validation),flush=True)
        with (validation/(name+'.log')).open('w',encoding='utf8') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode!=expected:raise RuntimeError('Unexpected '+name+' exit; inspect '+str(validation/(name+'.log')))
        if name=='formal_cpu_guard' and 'formal runs use CUDA' not in (validation/(name+'.log')).read_text(encoding='utf8'):raise RuntimeError('Wrong device guard failure')
    complete=read(validation/'end_to_end/complete.json')
    if not complete['complete'] or not complete['diagnostic_only']:raise RuntimeError('End-to-end smoke did not complete')
    smoke_report=dict(complete=complete,probability_gate=read(validation/'end_to_end/probability_frozen.json'),
        field_gate=read(validation/'end_to_end/field_selection_frozen.json'),
        evaluation=read(validation/'end_to_end/evaluation/summary.json'))
    dump(stage/'validation/end_to_end_cpu_smoke.json',smoke_report)
    files['validation/end_to_end_cpu_smoke.json']=sha(stage/'validation/end_to_end_cpu_smoke.json')
    if local_acceptance.exists():
        copy(local_acceptance,stage/'validation/full_method_acceptance_cpu.json')
        files['validation/full_method_acceptance_cpu.json']=sha(stage/'validation/full_method_acceptance_cpu.json')
    manifest['sha256']=files;dump(stage/'bundle_manifest.json',manifest)
    archive=ROOT/(NAME+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in [*files,'bundle_manifest.json']:
            p=stage/name;info=zipfile.ZipInfo.from_file(p,arcname=NAME+'/'+name);info.compress_type=zipfile.ZIP_DEFLATED
            if p.suffix=='.sh':info.external_attr=0o100755<<16
            z.writestr(info,p.read_bytes(),compresslevel=6)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise RuntimeError('ZIP CRC mismatch')
    digest=sha(archive);put(archive.with_suffix('.zip.sha256'),digest+'  '+archive.name+'\n')
    report=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=digest,stage=str(stage),files=len(files),
        zip_crc_verified=True,portable_cpu_end_to_end_passed=True,resume_idempotence_passed=True,
        formal_training_implementation_complete=True,cuda_locally_tested=False,validation_directory=str(validation))
    dump(ROOT/'build_validation_full.json',report);copy(HERE/'README.txt',ROOT/'README_full.txt');print(json.dumps(report,indent=2))

if __name__=='__main__':build()
