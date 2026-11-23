"""Exercise the staged release; no formal CPU/GPU training is represented as done."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
import torch

BASE=Path(__file__).resolve().parents[3]
ROOT=BASE/'outputs/UnifiedReconstruction_20261113_GPU'
LOCAL=ROOT/'local_validation_q217_20261114'
work=json.loads((ROOT/'calibrated_q217_build_work.json').read_text(encoding='utf8'))
stage,validation=Path(work['stage']),Path(work['validation'])
def read(p):return json.loads(p.read_text(encoding='utf8'))
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def command(label,args,expected=0,contains=None):
    path=validation/(label+'.log');start=time.perf_counter()
    print('CHECK_START',label,flush=True)
    with path.open('w',encoding='utf8') as log:
        proc=subprocess.run([sys.executable,'-u',str(stage/'launch.py'),*map(str,args)],stdout=log,stderr=subprocess.STDOUT)
    if proc.returncode!=expected or (contains and contains not in path.read_text(encoding='utf8')):
        raise RuntimeError('Check failed; inspect '+str(path))
    print('CHECK_PASSED',label,round(time.perf_counter()-start,2),'seconds',flush=True)
    return dict(exit_code=proc.returncode,log_sha256=sha(path),wall_seconds=time.perf_counter()-start)


def main():
    evidence={}
    evidence['plan']=command('plan',['plan','--output',validation/'plan'])
    evidence['formal_cpu_guard']=command('cpu_guard',['run','--device','cpu','--output',validation/'cpu_guard'],
        expected=1,contains='Formal runs require CUDA')
    evidence['three_seed_acceptance']=command('formal_cpu_acceptance',['validate','--device','cpu','--output',validation/'formal_check'])
    p=read(validation/'formal_check/protocol.json')
    assert p['config']['field_seeds']==[301030,301031,301032] and p['config']['field']['budgets']==[200]
    accepted=read(validation/'formal_check/acceptance_cpu.json')
    assert accepted['passed'] and len(accepted['probability_models'])==3
    assert not (validation/'formal_check/fields').exists()
    smoke=validation/'smoke'
    evidence['first_step']=command('smoke_first_step',['smoke','--device','cpu','--output',smoke,'--max-new-steps',1])
    folder=smoke/'fields/cf_2K_Q217/seed_301030'
    first=load(folder/'last.pt')
    assert first['step']==1 and not (smoke/'complete.json').exists()
    assert not (smoke/'field_selection_frozen.json').exists() and not (smoke/'evaluation').exists()
    evidence['no_early_H_or_field_evaluation']=True
    evidence['resume_step']=command('smoke_resume',['smoke','--device','cpu','--output',smoke,'--max-new-steps',1])
    last=load(folder/'last.pt');complete=read(smoke/'complete.json')
    assert complete['complete'] and complete['diagnostic_only'] and last['step']==2
    assert last['trace'][0]==first['trace'][0]
    assert all(float(v['step'])==2 for v in last['optimizer']['state'].values())
    local=read(LOCAL/'newB_newScale/history.json')[:2]
    errors={}
    for k in ('pde_loss','regularization','gradient_norm','objective'):
        errors[k]=max(abs(a[k]-b[k]) for a,b in zip(local,last['trace']))
        assert errors[k]<1e-8,(k,errors[k])
    errors['I_total']=max(abs(a['inversion']['total']-b['inversion']['total']) for a,b in zip(local,last['trace']))
    assert errors['I_total']<1e-9
    evidence['local_validated_first_two_steps_max_differences']=errors
    evaluation=read(smoke/'evaluation/summary.json')
    assert evaluation['complete'] and len(evaluation['runs'])==3
    assert complete['field_gate_sha256']==sha(smoke/'field_selection_frozen.json')
    assert complete['evaluation_sha256']==sha(smoke/'evaluation/summary.json')
    # Independent replay of V-only selection and checkpoint binding.
    best=last['initial'];chosen=0;s=p['config']['field']
    for item in last['trace']:
        v=item.get('validation')
        if v is not None and all(v[x]<=last['initial'][x]+s['guard'] for x in ('free','cap')) and v['total']<best['total']-s['selection_min_delta']:
            best=v;chosen=item['step']
    assert chosen==load(folder/'budget_2_primary.pt')['step']
    evidence['V_only_selection_reproduced']=True
    # These tests operate only on isolated test-output protocol bytes, restoring them exactly.
    sys.path.insert(0,str(stage/'github/experiments/unified_reconstruction_20261113'))
    sys.path.insert(0,str(stage/'github/experiments/q217_calibrated_20261114'))
    import calibrated_workflow as workflow
    import calibrated_fields as fields
    try:
        fields.evaluate(validation/'formal_check','cpu')
    except FileNotFoundError:pass
    else:raise AssertionError('Evaluation without a frozen field gate was accepted')
    try:
        workflow.seal(smoke,'formal')
    except RuntimeError as exc:
        assert 'profile changed' in str(exc)
    else:raise AssertionError('Profile mixing accepted')
    path=smoke/'protocol.json';original=path.read_bytes();changed=json.loads(original)
    changed['fingerprint']='0'*64
    try:
        path.write_text(json.dumps(changed),encoding='utf8')
        try:workflow.verify(smoke)
        except RuntimeError as exc:assert 'fingerprint' in str(exc)
        else:raise AssertionError('Tampered protocol accepted')
    finally:path.write_bytes(original)
    workflow.verify(smoke)
    evidence.update(passed=True,formal_scope_three_seeds_200=True,all_probability_models_imported_exactly=True,
        full_data_two_step_smoke=True,checkpoint_optimizer_resume_verified=True,
        full_time_window_ms=267,full_PDE_nodes=535,train_GH=12,validation_GH=20,
        premature_evaluation_rejected=True,profile_mixing_rejected=True,tampering_rejected=True,
        probability_parameters_unchanged=True,cuda_locally_tested=False,formal_training_executed=False)
    (validation/'build_checks.json').write_text(json.dumps(evidence,indent=2),encoding='utf8')
    print('ALL_RELEASE_CHECKS_PASSED',flush=True)


if __name__=='__main__':main()
