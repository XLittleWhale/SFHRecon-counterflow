"""Read-only integrity and CPU replay of the frozen formal simulation baseline.

No training, observations, hidden particle labels, or legacy data bundle needed.
An explicit export directory optionally receives reference/inverse VTKHDF files.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from counterflow.io import load, sha, setup
from counterflow.dynamics import rollout
from counterflow.observation import Density
from counterflow.vtk import vtk_series
from .core import operator, grid
from .evaluate import metrics


def verify(baseline, export=None):
    baseline=Path(baseline).resolve()
    manifest=json.loads((baseline/'manifest.json').read_text(encoding='utf8'))
    if manifest['status']!='formal_periodic_baseline':
        raise ValueError('Not an adopted formal baseline')
    for name,digest in manifest['sha256'].items():
        path=(baseline/name).resolve()
        if not path.is_relative_to(baseline) or sha(path)!=digest:
            raise ValueError('Baseline integrity mismatch: '+name)
    cfg=json.loads((baseline/'config.json').read_text(encoding='utf8'))
    c=json.loads((baseline/'physics.json').read_text(encoding='utf8'))
    selected=load(baseline/'inverse/selected.pt')
    expected=load(baseline/'replay_coefficients.pt')
    setup('cpu')
    net=Density().double().eval()
    net.load_state_dict(load(baseline/'training/selected.pt')['state'])
    if not all(torch.isfinite(p).all() for p in net.parameters()):
        raise ValueError('Nonfinite network weights')
    op=operator(c,load(baseline/'operator.pt'),cfg,'cpu')
    with torch.no_grad():
        a,q=rollout(op,selected['theta'],cfg['end']-cfg['start'],
                    cfg['inversion']['substeps'],False)
        torch.testing.assert_close(a,expected['inverse'],rtol=1e-9,atol=1e-11)
        if not torch.isfinite(q).all():raise ValueError('Nonfinite Vinen state')
        fields={name:grid(value,c,cfg) for name,value in
                [('inverse',a),('reference',expected['reference'])]}
    result=metrics(fields['inverse'],fields['reference'])
    saved=json.loads((baseline/'evidence/Results/report.json').read_text(encoding='utf8'))
    for key in ('window_vector_RMS_mm_s','initial_vector_RMS_mm_s'):
        np.testing.assert_allclose(result[key],saved['field_metrics']['inverse'][key],
                                   rtol=1e-8,atol=1e-9)
    if export is not None:
        export=Path(export).resolve()
        if export.exists():raise FileExistsError('Use a new export directory: '+str(export))
        export.mkdir(parents=True)
        xy=(np.arange(cfg['N'])+.5)*c['physics']['box_m']/cfg['N']
        times=(cfg['start']+np.arange(len(a)))*cfg['dt']
        for name,value in fields.items():
            vtk_series(export/name,value.numpy(),xy,xy,times)
    return dict(status='verified',files_checked=len(manifest['sha256']),
                coefficient_max_abs=float((a-expected['inverse']).abs().max()),
                initial_vector_RMS_mm_s=result['initial_vector_RMS_mm_s'],
                window_vector_RMS_mm_s=result['window_vector_RMS_mm_s'],
                note='Frozen forward replay and artifact integrity, not retraining or independent validation')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--export',type=Path)
    a=p.parse_args()
    print(json.dumps(verify(a.baseline,a.export),indent=2,allow_nan=False))


if __name__=='__main__':main()
