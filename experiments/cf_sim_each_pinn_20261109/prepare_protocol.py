"""Seal the complete each experiment before any formal fitting."""
from common import *
import argparse
import native_probability as Q
import field_adapter as F
import train_probability as T

def seal():
    if (OUT/'protocol.json').exists():
        return verify_protocol()
    sources = list(HERE.glob('*.py'))+Q.source_paths()+[F.BASELINE]
    runtime = BASE/'outputs/JointProbabilityPhysics_20261103/cf_runtime'
    sources += list(runtime.rglob('*.py'))
    sources += list((BASE/'outputs/FullWindowCalibration_20261026/server_package').rglob('*.py'))
    inputs = Q.calibration_paths()+[F.OPERATOR]
    inputs += [p for p in (runtime/'B_prior_assets').iterdir() if p.is_file()]
    summaries = {}
    for case in CASES:
        folder = prepared(case)
        audit = read(folder/'data_audit.json')
        for name,digest in audit['output_sha256'].items():
            if sha(folder/name)!=digest:
                raise RuntimeError('Prepared input changed: '+str(folder/name))
        inputs += [p for p in folder.iterdir() if p.is_file()]
        summaries[case] = dict(groups=audit['groups'],meta=audit['meta'],
            accepted_intervals=audit['accepted_intervals'],rejected=audit['rejected'])
    p = dict(version='cf-sim-each-pinn-20261109-v1',frozen=True,created_utc=utc(),
        cases=list(CASES),field_seeds=list(SEEDS),probability_seeds=list(TRAIN_SEEDS),budgets=list(BUDGETS),
        probability=T.DEFAULT_SETTINGS,
        field=dict(lr=.001,gradient_clip=5.,train_degree=12,validation_degree=20,
            validate_every=10,guard_nll_per_ms=.01,selection_delta_nll_per_ms=1e-6,
            likelihood_weight=.08,free_batch=64,captured_batch_pids=2,epochs=None,
            architecture='K12 Shen 325 modes/fluid; 3x128 tanh; 8 sine/cosine time frequencies',
            initial='zero raw coefficients; prescribed Poiseuille normal / uniform superflow',
            PDE='baseline fixed equilibrium L0; nu_s=0; F_ns=1; complete Frozen B push-forward',
            time='all usable native intervals; 0.5ms PDE nodes to ceil(last observation / 1ms)',
            acceleration='partial_t u + G u (corrected to match reference particle generator)'),
        execution=dict(workers=3,threads_per_worker=2,device='cpu',resume=True),
        partitions=dict(F='pid%5==2',I='pid%5>=3',V='pid%5==1',H='pid%5==0'),
        sequence='each case independently trains F with its own truth contexts; V selects; freeze probability; I reconstructs; V selects fields; freeze all 9 runs; H and reference evaluate',
        interpretation='same-realization supervised diagnostic; not unseen-field or blind experimental generalization',
        differences_from_Nov5=[
            'fresh each-case probability network fitted on native endpoint likelihood, not reused Nov5 weights',
            'camera-rate interpolated reference jets; no unavailable microscopic force/latent supervision',
            'full available 50-frame recording with frame0 reserved for initial causal history',
            'material acceleration corrected from G^T u to G u to match this simulator',
            'four disjoint physical PID groups; mixed-state past history allowed',
            '50 native frames after 1 second evolution are transient, not certified steady'],
        limitations=[
            'True free/captured state labels and endpoint attachment identities screen eligible rows; this is state-conditional reconstruction, not position-only blind inference.',
            'Endpoint-stable state and attachment screening cannot exclude unobserved intraframe transitions.',
            'Inherited force/noise calibration is frozen; it was not recalibrated to the new camera simulations.',
            'Frozen L0 weak PDE and finite K12 basis differ from the fluctuating point-vortex reference.',
            'Reference fields and derivative jets are camera-rate interpolants.',
            'Validation PIDs select both probability and field; only H remains final held-out.'],
        prepared_cases=summaries,
        sources={str(x.resolve()):sha(x) for x in sorted(set(sources))},
        inputs={str(x.resolve()):sha(x) for x in sorted(set(inputs))})
    p['fingerprint']=fingerprint(p)
    dump(OUT/'protocol.json',p)
    return p

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepare-data',action='store_true')
    args=ap.parse_args()
    setup()
    if args.prepare_data:
        if (OUT/'protocol.json').exists():
            raise RuntimeError('Cannot regenerate inputs after protocol seal')
        from data_adapter import prepare_case
        for case in CASES: prepare_case(SOURCE/case,prepared(case))
    print(seal()['fingerprint'])
