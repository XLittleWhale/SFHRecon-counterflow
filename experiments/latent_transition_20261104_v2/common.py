"""Read-only reuse of the previously frozen independent probability experiment."""
from pathlib import Path
import importlib.util
import sys
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'2')
import torch

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2]
OUT=BASE/'outputs/LatentTransition_20261104_v2'
OLD=HERE.parent/'shared_transition_20261104'
OLD_OUT=BASE/'outputs/SharedTransition_20261104'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result
    spec.loader.exec_module(result)
    return result
sys.path.insert(0,str(OLD))
S=module('latent_shared_parent',OLD/'experiment.py')
T=module('latent_shared_training',OLD/'training.py')
sys.path.insert(0,str(HERE))
load,save,dump,sha,read,fingerprint=S.load,S.save,S.dump,S.sha,S.read,S.fingerprint
M,EW,TM=S.M,S.EW,S.TM
CASES=S.CASES
SEEDS=S.SEEDS
TRAIN_SEEDS=S.TRAIN_SEEDS
ARMS=('iid','reset','temporal')
NEW_ARMS=('reset','temporal')
case_name=S.case_name
def base_network(seed,dtype=torch.float64):
    ts=seed if seed in TRAIN_SEEDS else TRAIN_SEEDS[SEEDS.index(seed)]
    rows=read(OLD_OUT/'training/selection_frozen.json')['runs']
    record=next(r for r in rows if r['arm']=='independent' and r['seed']==ts)
    if sha(record['path'])!=record['sha256']:
        raise RuntimeError('Frozen base network hash changed')
    base=T.new_model('independent',ts).to(dtype=dtype)
    base.load_state_dict(load(record['path'])['state'])
    base.eval()
    for p in base.parameters(): p.requires_grad_(False)
    return base,record

class Network(torch.nn.Module):
    """Serializable frozen base plus new transition head; no target weight updates."""
    def __init__(self,base,head):
        super().__init__()
        self.base=base
        self.head=head
    def freeze(self):
        self.eval()
        for p in self.parameters(): p.requires_grad_(False)
        return self

def setup():
    M.R.setup('cpu')
    torch.set_num_threads(2)
