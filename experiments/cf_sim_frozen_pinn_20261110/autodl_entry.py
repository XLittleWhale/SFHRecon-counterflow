"""Offline GPU entry for the fixed-Nov5 probability comparison."""
from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('environment','preflight','run','status','plots'))
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--extended',action='store_true')
    args=parser.parse_args()
    if args.action=='run' and not args.device.startswith('cuda'):
        raise SystemExit('Formal offline entry requires CUDA. CPU is supported only for diagnostics and plots.')
    suffix='_extended' if args.extended else ''
    output=args.output or Path(os.environ.get('OUT_DIR',
        '/root/autodl-tmp/cf_sim_frozen_pinn_results_20261110'+suffix))
    os.environ['CF_EACH_OUTPUT']=str(output.resolve())
    os.environ['CF_EACH_DEVICE']='cpu' if args.action=='plots' else args.device
    os.environ['CF_FROZEN_EXTENDED']='1' if args.extended else '0'
    if args.action=='status':
        path=output/'pipeline_status.json'
        print(path.read_text(encoding='utf8') if path.exists() else 'No pipeline status yet.')
        return
    missing=[];versions={}
    for name in ('numpy','scipy','torch','psutil'):
        try:
            module=importlib.import_module(name)
            versions[name]=getattr(module,'__version__','available')
        except ImportError:
            missing.append(name)
    if missing:
        raise SystemExit('Missing preinstalled dependencies: '+', '.join(missing)+
            '. This offline entry never downloads or installs packages.')
    try:
        import matplotlib
        from matplotlib import pyplot
        versions['matplotlib']=matplotlib.__version__
        plotting_error=None
    except Exception as error:
        plotting_error=repr(error)
    import common as C
    import torch
    C.setup()
    if args.action=='plots':
        import evaluate
        evaluate.plots_only()
        return
    target=C.device()
    environment=dict(python=platform.python_version(),platform=platform.platform(),versions=versions,
        device=target,cuda_available=torch.cuda.is_available(),cuda_runtime=torch.version.cuda,
        float_dtype='float64',network_access_required=False,plotting_error=plotting_error)
    if target.startswith('cuda'):
        prop=torch.cuda.get_device_properties(target)
        environment.update(gpu_name=prop.name,total_vram_bytes=prop.total_memory)
        probe=torch.randn(32,32,device=target,dtype=torch.float64,requires_grad=True)
        torch.linalg.cholesky(probe@probe.T+torch.eye(32,device=target)*.1).square().sum().backward()
        torch.cuda.synchronize()
    print(json.dumps(environment,indent=2),flush=True)
    if args.action=='environment': return
    C.OUT.mkdir(parents=True,exist_ok=True)
    preflight=C.OUT/'gpu_preflight.json'
    manifest=C.BASE/'bundle_manifest.json'
    manifest_digest=C.sha(manifest) if manifest.exists() else None
    existing=C.read(preflight) if preflight.exists() else {}
    passed=(existing.get('complete') and existing.get('bundle_sha256')==manifest_digest
            and existing.get('environment')==environment)
    if args.action=='preflight' or not passed:
        from test_fast_score import check_equivalence
        numerical=check_equivalence(target)
        import benchmark_smoke as B
        start=time.perf_counter();records=[]
        for index,case in enumerate(C.CASES):
            records.append(B.run(C.prepared(case),C.OUT/f'preflight_field_{case}.json',
                captured_rows=0 if index==0 else 4,pde_intervals=None,degree=12,
                captured_pids=32 if index==0 else 2))
        C.dump(preflight,dict(complete=True,diagnostic_only=True,bundle_sha256=manifest_digest,
            environment=environment,fields=records,numerical_equivalence=numerical,
            seconds=time.perf_counter()-start,weights_discarded=True,
            probability_optimization_steps=0,formal_training_complete=False,finished_utc=C.utc()))
    if args.action=='preflight': return
    import prepare_protocol
    prepare_protocol.seal()
    import run_pipeline
    run_pipeline.main()


if __name__=='__main__': main()
