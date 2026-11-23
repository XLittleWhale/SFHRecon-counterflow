"""Method declaration plus runtime numerical-acceptance guard."""
from runtime import *

def config():
    import workflow
    return workflow.defaults()

def verify_spec():
    provenance=read(HERE/'source_contract.json')
    for name,digest in provenance['sha256'].items():
        if sha(contained(BASE,name))!=digest:raise RuntimeError('Normative/source input changed: '+name)
    return provenance

def capabilities():
    return dict(original_baseline_loss_kernels=True,historical_q_from_certified_initial_slots=True,
        native_start_y=True,full_resolved_free_force_preparation=True,
        native_supervised_marginal_training_adapter=True,native_supervised_transition_training_adapter=True,
        channel_closure_joint_training_adapter=True,common_formal_training_controller=True,
        periodic_reference_training_adapter=True,cf_experimental_channel_inference_adapter=True,
        completed_cf_stage_adaptation=False,formal_training_enabled=True,
        runtime_numerical_acceptance_required=True,legacy_endpoint_only_fallback=False,forced_reference_generation=False)

def training_readiness():
    verify_spec()
    return dict(implementation_ready=True,capabilities=capabilities(),runtime_acceptance_required=True,
        formal_results_claimed=False,stages=['A','B','closure','joint','probability_freeze','PINN','field_freeze','evaluation'])

def require_training(out=None,device='cuda'):
    if out is None:raise RuntimeError('A sealed run and numerical acceptance record are required')
    import workflow
    p=workflow.verify(out);path=Path(out)/'acceptance.json'
    if not path.exists():raise RuntimeError('Numerical acceptance has not run')
    report=read(path)
    if not report.get('passed') or report.get('protocol_fingerprint')!=p['fingerprint']:
        raise RuntimeError('Acceptance record does not match this run')
    if not p['config']['diagnostic_only']:
        if device!='cuda' or not torch.cuda.is_available() or not report.get('cuda_tested'):
            raise RuntimeError('Formal training requires acceptance on the available CUDA device')
