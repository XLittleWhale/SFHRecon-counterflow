"""Shared, read-only parent interfaces for learned-closure four-arm experiment."""
from pathlib import Path
import importlib.util
import sys
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'2')
import torch

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2]
OUT=BASE/'outputs/NeuralClosure4DVar_20261104'
VHERE=HERE.parent/'latent_transition_20261104_v2'
VOUT=BASE/'outputs/LatentTransition_20261104_v2'

def module(name,path):
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result
    spec.loader.exec_module(result)
    return result

V=module('closure_parent_common',VHERE/'common.py')
S,T,M,EW,TM=V.S,V.T,V.M,V.EW,V.TM
load,save,dump,sha,read,fingerprint=V.load,V.save,V.dump,V.sha,V.read,V.fingerprint
CASES,SEEDS,TRAIN_SEEDS=V.CASES,V.SEEDS,V.TRAIN_SEEDS
case_name=V.case_name
ARMS=('baseline','closure','probability','joint')
NEW_ARMS=ARMS[1:]
group_mask,weights=S.group_mask,S.weights
sys.path.insert(0,str(VHERE))
import latent_model as LM
import data as VD
SCORING=module('closure_parent_scoring',VHERE/'scoring.py')
score=SCORING.score
sys.path.insert(0,str(HERE))

def setup():
    M.R.setup('cpu');torch.set_num_threads(2)

def plain_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def baseline_network(seed,trainable=False):
    from probability import JointHead
    ts=seed if seed in TRAIN_SEEDS else TRAIN_SEEDS[SEEDS.index(seed)]
    base,br=V.base_network(ts)
    gate=read(VOUT/'training/selection_frozen.json')
    row=next(r for r in gate['runs'] if r['seed']==ts)
    assert sha(row['path'])==row['sha256']
    head=JointHead(base,state=load(row['path'])['state']).double()
    net=V.Network(base,head);net.mode='temporal'
    net.train(trainable)
    for p in net.parameters():p.requires_grad_(trainable)
    return net,dict(base=br,head=row)

def verify_protocol():
    p=read(OUT/'protocol.json')
    assert p['frozen'] and fingerprint({k:v for k,v in p.items() if k!='fingerprint'})==p['fingerprint']
    for category in ('sources','inputs'):
        for path,digest in p[category].items():
            if sha(path)!=digest:raise RuntimeError('Frozen '+category+' changed: '+path)
    return p
