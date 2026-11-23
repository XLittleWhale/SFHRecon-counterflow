"""Local CF migration bookkeeping. Historical sources and weights stay read-only."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
from pathlib import Path
import sys
import json
import hashlib
import time
import uuid
from datetime import datetime, timezone
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
OUT = BASE/'outputs/PINNCF_20261105'
PARENT_HERE = HERE.parent/'pinn_twofluid_20261105'
PARENT_OUT = BASE/'outputs/PINNTwoFluid_20261105'
CASES = ('cf_113_camera100_f0184', 'cf_217_camera100_f0333')
SEEDS = (301030, 301031, 301032)
TRAIN_SEEDS = (411040, 411041, 411042)
BUDGETS = (200, 600)

def setup():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)

def utc():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode('utf8')).hexdigest()

def _replace(temp, path):
    for attempt in range(12):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(.05*(attempt+1))

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
        _replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        torch.save(value, temp)
        _replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)

def parent():
    if str(PARENT_HERE) not in sys.path:
        sys.path.insert(0, str(PARENT_HERE))
    import pinn_common
    assert Path(pinn_common.__file__).resolve() == PARENT_HERE/'pinn_common.py'
    return pinn_common

def probability_record(seed):
    return next(row for row in read(PARENT_OUT/'probability_selection/selection_frozen.json')['models']
                if row['field_seed'] == seed)

def probability(seed):
    """Load only the already-selected joint probability state, not its old closure."""
    from cf_probability import load_probability
    net, cal, force, _ = load_probability(seed)
    return net, cal, force

def state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def group_mask(pid, group):
    if group == 'fit': return pid % 5 >= 2
    if group == 'selection': return pid % 5 == 1
    if group == 'test': return pid % 5 == 0
    raise ValueError(group)

def verify_protocol():
    p = read(OUT/'protocol.json')
    assert p['frozen'] and p['fingerprint'] == fingerprint({k:v for k,v in p.items() if k!='fingerprint'})
    for category in ('sources', 'inputs'):
        for path, digest in p[category].items():
            if sha(path) != digest:
                raise RuntimeError(f'Frozen {category} changed: {path}')
    return p

def evaluation_gate():
    p = verify_protocol()
    gate = read(OUT/'selection_frozen.json')
    assert gate['complete'] and gate['protocol_fingerprint'] == p['fingerprint']
    assert len(gate['runs']) == len(CASES)*len(SEEDS)
    for row in gate['runs']:
        for record in [row['background'], *[v for b in row['budgets'].values() for v in b.values()]]:
            assert sha(record['path']) == record['sha256']
        assert probability_record(row['seed'])['sha256'] == row['probability_sha256']
    return p, gate
