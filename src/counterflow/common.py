from pathlib import Path
import json,math,time,copy,gc
import numpy as np
import torch
from . import runtime as rt, likelihood as obs, controls
from .observation import Density,decoded,particle_parts,predict_free,law
from .paths import DATA, RUN, RESULTS, ASSETS, CONFIG
HERE=RUN
DEST=RESULTS
CASES=["periodic_1p5","periodic_4","periodic_10","cf_261","cf_505"]
POLICY=json.loads((CONFIG/"policy.json").read_text(encoding="utf8"))

def declare():
    rt.setup('cpu')
    HERE.mkdir(parents=True,exist_ok=True)
    DEST.mkdir(parents=True,exist_ok=True)
    path=HERE/'policy.json'
    if path.exists():
        if readjson(path)!=POLICY:raise ValueError('Run policy changed; use a new --run directory')
    else:rt.dump(path,POLICY)

def artifact(name):
    """A run is either a complete calibrated model or the supplied frozen model."""
    files={'model':'training/selected.pt','calibration':'calibration/calibration.json','correlation':'correlation.json'}
    local=HERE/files[name]
    if (HERE/'training/selected.pt').exists():
        if not all((HERE/p).exists() for p in files.values()):
            raise RuntimeError('New weights require calibration and correlation files before inversion. See docs/DATA.md.')
        return local
    return ASSETS/{'model':'model.pt','calibration':'calibration.json','correlation':'correlation.json'}[name]

def model():
    net=Density().double()
    net.load_state_dict(rt.load(artifact('model'))['state']);net.eval()
    for p in net.parameters():p.requires_grad_(False)
    cal=readjson(artifact('calibration'))
    if cal.get('checkpoint_sha256') and cal['checkpoint_sha256']!=rt.sha(artifact('model')):
        raise ValueError('Model and calibration hashes disagree')
    return net,cal

def bundle(label):
    from .paths import data_path
    if label not in CASES:raise ValueError(label)
    c,windows=rt.load(data_path(f'inputs/{label}.pt'))
    cal=readjson(artifact('calibration'))
    for obj in [c]+windows:
        for key in ('operator_source','refined_source'):
            if key in obj:obj[key]=str(data_path(obj[key]))
        if obj['kind']=='periodic':obj['fixed_noise']=cal['sigma']['12']
    obs.BETA=torch.tensor(readjson(artifact('correlation'))['beta'],dtype=torch.float64)
    return c,windows

def fingerprints():
    return dict(model_sha=rt.sha(artifact('model')),calibration_sha=rt.sha(artifact('calibration')),
                correlation_sha=rt.sha(artifact('correlation')),policy_sha=rt.sha(CONFIG/'policy.json'),
                likelihood_version=POLICY['version'])
def parts(c,group):
    return [(k,z) for k,z in controls.parts(c,group) if len(z['pid'])]

def observation(c,a,bs,net,cal,backward=False,degree=16):
    T=c['observation_steps']; count=sum(len(z['pid'])//T for _,z in bs)
    assert count>0
    result={}; total=0.
    for kind,z in bs:
        if kind=='cap': val=obs.captured_nll(c,a,z,net,cal,degree)
        else: val=obs.free_nll(c,a,z)
        loss=val.sum()/count; total+=float(loss.detach()); result.setdefault(kind,[]).append(val.detach())
        if backward: (.08*loss).backward()
    return total,{k:torch.cat(v) for k,v in result.items()}

@torch.no_grad()
def validation(c,a,bs,net,cal,degree=24):
    _,values=observation(c,a,bs,net,cal,degree=degree)
    records=[]
    for kind,v in values.items():
        ids=torch.cat([z['pid'][::c['observation_steps']] for k,z in bs if k==kind])
        for w in (ids//c['pid_stride']).unique():
            m=ids//c['pid_stride']==w
            records.append(dict(kind=kind,window=int(w),n=int(m.sum()),nll=float(v[m].mean())))
    return float(np.mean([r['nll'] for r in records])),records,values

def readjson(path): return json.loads(Path(path).read_text(encoding='utf8'))
