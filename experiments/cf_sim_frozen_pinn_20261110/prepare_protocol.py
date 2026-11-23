"""Predeclare the probability-only intervention before starting any field fit."""
from common import *
import native_probability as Q
import field_adapter as F
import frozen_probability as P

def seal():
    if (OUT/'protocol.json').exists(): return verify_protocol()
    sources=list(HERE.glob('*.py'))+Q.source_paths()+[F.BASELINE]
    runtime=BASE/'outputs/JointProbabilityPhysics_20261103/cf_runtime'
    sources+=list(runtime.rglob('*.py'))
    sources+=list((BASE/'outputs/FullWindowCalibration_20261026/server_package').rglob('*.py'))
    inputs=Q.calibration_paths()+[F.OPERATOR]
    inputs += [x for x in (runtime/'B_prior_assets').iterdir() if x.is_file()]
    inputs += P.source_paths()
    inputs += [COMPARISON/'protocol.json', COMPARISON/'analysis/result_summary.json',
               COMPARISON/'evaluation/summary.json']
    reference=read(COMPARISON/'protocol.json')
    if reference.get('fingerprint')!=fingerprint({k:v for k,v in reference.items() if k!='fingerprint'}):
        raise RuntimeError('Reference protocol fingerprint mismatch')
    summaries={}
    for case in CASES:
        folder=prepared(case)
        audit=read(folder/'data_audit.json')
        for name,digest in audit['output_sha256'].items():
            if sha(folder/name)!=digest: raise RuntimeError('Prepared input changed: '+str(folder/name))
        inputs += [x for x in folder.iterdir() if x.is_file()]
        summaries[case]=dict(groups=audit['groups'],meta=audit['meta'],
            accepted_intervals=audit['accepted_intervals'],rejected=audit['rejected'])
    # The completed day run supplies the exact field objective and batches.
    expected=dict(lr=.001,gradient_clip=5.,train_degree=12,validation_degree=20,
        validate_every=10,guard_nll_per_ms=.01,selection_delta_nll_per_ms=1e-6,
        likelihood_weight=.08,free_batch=128,captured_batch_pids=32)
    for key,value in expected.items():
        if reference['field'][key]!=value: raise RuntimeError('Unexpected reference field setting: '+key)
    if reference['budgets']!=[150] or reference['field_seeds']!=[301030]:
        raise RuntimeError('Expected completed one-seed, 150-step day comparison')
    records=[dict(path=asset_key(P.record(s)['path']),sha256=P.record(s)['sha256'],
        field_seed=s,training_seed=P.record(s)['training_seed']) for s in SEEDS]
    p=dict(version='cf-sim-frozen-pinn-20261110-v1',frozen=True,created_utc=utc(),
        cases=list(CASES),field_seeds=list(SEEDS),probability_seeds=list(TRAIN_SEEDS),budgets=list(BUDGETS),
        matched_budget=150,extended_budgets_reported_separately=[b for b in BUDGETS if b!=150],
        field=reference['field'],partitions=reference['partitions'],
        probability=dict(source='PINNTwoFluid_20261105 preselected joint, also used by PINNCF_20261105',
            records=records,training=False,normalization_refit=False,calibration_refit=False,
            new_CF_F_rows_used=False,probability_updates=0),
        execution=dict(workers=1,threads_per_worker=2,device=device(),resume=True),
        comparison=dict(baseline_protocol_sha256=sha(COMPARISON/'protocol.json'),
            baseline_protocol_fingerprint=reference['fingerprint'],
            intervention='replace only the probability model tensors, including its original cm/cs buffers',
            unchanged=['all prepared observations and partitions','cold initial field and field seed',
                'field architecture, fixed-L PDE, complete Frozen B and preconditioning',
                'physical parameters and Gu acceleration','native camera likelihood and GH degrees',
                'field optimizer, objective weights, complete captured sequence gradients','150-step matched budget'],
            primary_comparison='reference velocity RMSE at identical native times and coarse grid',
            likelihood_interpretation='each NLL is evaluated under its stated probability law; do not call differences a matched-model NLL improvement',
            exploratory=True,reason='existing H/reference results motivated this comparison; no new H fitting or selection'),
        sequence='load preselected frozen joint model without refitting; cold-start I field inversion; V selection; freeze all three fields; H and reference evaluation',
        prepared_cases=summaries,limitations=[x for x in reference['limitations']
            if 'select both probability and field' not in x]+[
            'V selects field checkpoints only; the Nov5 joint probability law is fixed by its original selection gate.',
            'Old probability-model domain transfer may fail; a successful loader or low training NLL does not establish reconstruction quality.',
            'One-seed 150-step results are an exploratory matched-budget comparison, not the original three-seed 200/600 baseline.'],
        sources={asset_key(x):sha(x) for x in sorted(set(sources))},
        inputs={asset_key(x):sha(x) for x in sorted(set(inputs))})
    p['fingerprint']=fingerprint(p)
    dump(OUT/'protocol.json',p)
    return p

if __name__=='__main__':
    setup();print(seal()['fingerprint'])
