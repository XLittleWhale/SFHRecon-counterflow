"""Independent resumable stages and immutable run fingerprints."""
import argparse
import json
import shutil
import sys
from pathlib import Path
import torch
from counterflow.io import setup,dump,sha
from .core import config


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--phase',required=True);p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();package=a.package.resolve();run=a.out.resolve();cfg=config()
    if a.phase=='check':
        manifest=json.loads((package/'manifest.json').read_text(encoding='utf8'))
        failures=[name for name,digest in manifest['code_hashes'].items() if sha(package/name)!=digest]
        if failures:raise RuntimeError('Package code fingerprint mismatch: '+str(failures))
        if not torch.cuda.is_available() and not a.smoke:raise RuntimeError('CUDA PyTorch required for the full run')
        for name in cfg['train']+[cfg['validation'],cfg['test']]:
            for f in ('physics.json','observations.pt','fields.h5','audit.json'):
                if not (package/'data'/name/f).exists():raise FileNotFoundError(name+'/'+f)
        print(json.dumps(dict(torch=torch.__version__,cuda=torch.cuda.is_available(),package_GiB=manifest['GiB'],
            disk_free_GiB=shutil.disk_usage(package).free/1024**3,case_count=9),indent=2));return
    if not a.smoke and not a.device.startswith('cuda'):raise RuntimeError('Formal CPU calculation disabled; use the offline CUDA server')
    device=setup(a.device);run.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(run).free<4*1024**3:raise RuntimeError('Need 4 GiB output/checkpoint headroom')
    fingerprint=dict(config=cfg,smoke=a.smoke,package_manifest_sha=sha(package/'manifest.json'),
                     source_hashes={str(q.relative_to(package)):sha(q) for folder in ('counterflow_full','counterflow') for q in (package/folder).glob('*.py')})
    policy=run/'policy.json'
    if policy.exists() and json.loads(policy.read_text(encoding='utf8'))!=fingerprint:raise RuntimeError('Run policy/package changed; use a fresh output directory')
    dump(policy,fingerprint)
    if a.phase=='prepare':
        from .prepare import main as fn
    elif a.phase=='train':
        from .train import main as fn
    elif a.phase=='calibrate':
        from .train import calibration
        calibration(run,cfg,device,a.smoke);return
    elif a.phase=='tune':
        from .inverse import tune as fn
    elif a.phase=='invert':
        from .inverse import main as fn
    elif a.phase=='evaluate':
        from .evaluate import main as fn
    else:raise ValueError(a.phase)
    fn(package,run,cfg,device,a.smoke)


if __name__=='__main__':main()
