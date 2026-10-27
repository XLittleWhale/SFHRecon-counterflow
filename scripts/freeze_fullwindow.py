"""Freeze an explicitly adopted full GPU result; never overwrite an existing snapshot.

Copies only small checkpoints, provenance, scientific reports and figure groups.
Raw observations/fields, optimizer states and the large server data pack stay local.
"""
import argparse
import json
import shutil
from pathlib import Path
from counterflow.io import dump,load,save,sha


def freeze(package,run,dest):
    package,run,dest=(Path(p).resolve() for p in (package,run,dest))
    if dest.exists():raise FileExistsError('Immutable snapshot already exists: '+str(dest))
    policy=json.loads((run/'policy.json').read_text(encoding='utf8'))
    completion=json.loads((run/'Results/complete.json').read_text(encoding='utf8'))
    if policy['smoke'] or not completion['full_run']:raise ValueError('Smoke is not a formal baseline')
    if sha(package/'manifest.json')!=policy['package_manifest_sha']:
        raise ValueError('Original package manifest mismatch')
    for name,digest in policy['source_hashes'].items():
        if sha(package/name)!=digest:raise ValueError('Original source mismatch: '+name)
    cfg=policy['config']
    targets={
        'training/selected.pt':run/'training/selected.pt',
        'inverse/selected.pt':run/'inverse/selected.pt',
        'calibration.json':run/'calibration.json',
        'force.json':run/'force.json',
        'operator.pt':package/'operator.pt',
        'physics.json':package/'data'/cfg['test']/'physics.json',
        'config.json':package/'counterflow_full/config.json',
        'evidence/input_manifest.json':package/'manifest.json',
    }
    for pattern in ('*.json','training/**/*.json','inverse/**/*.json','tuning/**/*.json','Results/*.json'):
        for path in run.glob(pattern):targets['evidence/'+path.relative_to(run).as_posix()]=path
    for ext in ('png','pdf','svg'):
        for path in (run/'Results').glob('*.'+ext):targets['figures/'+path.name]=path
    for case in cfg['train']+[cfg['validation'],cfg['test']]:
        targets['evidence/input_audits/'+case+'.json']=package/'data'/case/'audit.json'
    for name,path in targets.items():
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    fields=load(run/'Results/fields.pt')
    save(dest/'replay_coefficients.pt',{key:fields['coefficients'][key] for key in ('inverse','reference')})
    # A release manifest explicitly separates the returned run from the portable
    # publication builder/docs. Scientific kernels are checked separately in CI.
    manifest=dict(id='fullwindow_4mm_20261026',status='formal_periodic_baseline',adopted='2026-10-27',
        scope='Periodic 4 mm/s, native 1 ms, 5–100 ms; not a CF model promotion',
        immutable_source_result='FullWindowCalibration_20261026/fullwindow_4mm_results',
        original_package_manifest_sha256=policy['package_manifest_sha'],
        original_run_policy_sha256=sha(run/'policy.json'),
        archive_excludes=['raw training data','prepared observations','optimizer states','full VTKHDF series'],
        source_field_tensor_sha256=sha(run/'Results/fields.pt'),
        sha256={p.relative_to(dest).as_posix():sha(p) for p in sorted(dest.rglob('*')) if p.is_file()})
    dump(dest/'manifest.json',manifest)
    print(json.dumps(dict(destination=str(dest),files=len(manifest['sha256']),
        MiB=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())/1024**2),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('package','run','dest'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();freeze(a.package,a.run,a.dest)
