"""Frozen ancestry and shared interfaces; no historical scientific file is edited."""
from pathlib import Path
import os
import sys
import importlib.util
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'2')
import torch

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2]
OUT=BASE/'outputs/PINNTwoFluid_20261105'
NHERE=HERE.parent/'neural_closure_20261104'
NOUT=BASE/'outputs/NeuralClosure4DVar_20261104'
sys.path.insert(0,str(NHERE))
import common as N
assert Path(N.__file__).resolve()==(NHERE/'common.py').resolve()
E=N.module('pinn_parent_neural_experiment',NHERE/'experiment.py')
CURRENT=N.module('pinn_current_physics',HERE.parent/'joint_probability_20261103/physics/simulation_operator.py')
sys.path.insert(0,str(HERE))
S,T,M,EW,TM,SCORING=N.S,N.T,N.M,N.EW,N.TM,N.SCORING
load,save,dump,sha,read,fingerprint=N.load,N.save,N.dump,N.sha,N.read,N.fingerprint
CASES,SEEDS,TRAIN_SEEDS=N.CASES,N.SEEDS,N.TRAIN_SEEDS
case_name,group_mask,weights=N.case_name,N.group_mask,N.weights
ARMS=('global_spline','pinn')
DT=.001
TEND=.095

def setup():
    N.setup()

def selection_gate():
    path=OUT/'probability_selection/selection_frozen.json'
    return read(path)

def verify_protocol():
    p=read(OUT/'protocol.json')
    assert p['frozen'] and fingerprint({k:v for k,v in p.items() if k!='fingerprint'})==p['fingerprint']
    for category in ('sources','inputs'):
        for path,digest in p[category].items():
            if sha(path)!=digest:raise RuntimeError('Frozen '+category+' changed: '+path)
    return p

def probability_record(seed):
    gate=selection_gate()
    # The selected family and original three probability seeds are used unchanged.
    rows=gate.get('models',gate.get('selected_models',gate.get('runs',[])))
    ts=TRAIN_SEEDS[SEEDS.index(seed)]
    return next(r for r in rows if int(r.get('training_seed',r.get('seed',-1)))==ts)

def make_context(U,seed):
    if abs(U-.0065)<1e-12:
        # Preflight only: observations and metadata; exclude every non-V PID
        # before passing observations onward. Never open reference/training jets.
        from counterflow_full import core
        d=load(T.SOURCE/'prepared/vn0_0065m_s/observations.pt')
        d=dict(d,z=core.subset(d['z'],torch.where(d['z']['pid']%5==1)[0]))
        cfg=core.config()
        op=CURRENT.Momentum(core.operator(d['c'],load(T.PACKAGE/'operator.pt'),cfg,'cpu'))
        cold=torch.zeros(2,624,dtype=torch.float64)
        es=torch.zeros(95,624,dtype=torch.float64)
        old=M.make_model('B',op,dict(U=U,a=cold[None].expand(96,-1,-1),eta_s=es,eta_n=es.clone()),seed)
        cal,force=read(T.SOURCE/'calibration.json'),read(T.SOURCE/'force.json')
    else:
        d,cfg,old,_,cal,force=S.P.make_context(U,seed,'new_frozen')
    net,_=N.baseline_network(seed)
    record=probability_record(seed)
    path=record.get('path',record.get('checkpoint_path'))
    assert sha(path)==record['sha256']
    ck=load(path)
    net.load_state_dict(ck['net_state'])
    net.freeze();net.mode='temporal'
    assert float(old.op.nu_s)==0 and old.op.friction_factor==1
    return d,cfg,old,net,cal,force

def plain_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
