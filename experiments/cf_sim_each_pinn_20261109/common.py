"""Bookkeeping for per-realization CF probability training and PINN inversion."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
from pathlib import Path
import hashlib
import json
import sys
import time
import uuid
from datetime import datetime, timezone
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
OUT = Path(os.environ.get('CF_EACH_OUTPUT', str(BASE/'outputs/CFSimEachPINN_20261109')))
SOURCE = BASE/'outputs/cf_reference_transient1s_50frames_results_20261108'
CASES = ('cf_2K_Q113', 'cf_2K_Q175', 'cf_2K_Q217')
SEEDS = (301030, 301031, 301032)
TRAIN_SEEDS = (411040, 411041, 411042)
BUDGETS = (200, 600)

def setup():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(int(os.environ.get('CF_EACH_THREADS', '2')))

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

def _atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        writer(temp)
        for attempt in range(12):
            try:
                os.replace(temp, path)
                return
            except PermissionError:
                if attempt == 11:
                    raise
                time.sleep(.05*(attempt+1))
    finally:
        temp.unlink(missing_ok=True)

def dump(path, value):
    _atomic(path, lambda p: p.write_text(json.dumps(value, indent=2, ensure_ascii=False,
        allow_nan=False), encoding='utf8'))

def save(path, value):
    _atomic(path, lambda p: torch.save(value, p))

def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)

def state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def prepared(case):
    if case not in CASES:
        raise ValueError(case)
    return OUT/'prepared'/case

def probability_folder(case, seed):
    return OUT/'probability'/case/f'seed_{TRAIN_SEEDS[SEEDS.index(seed)]}'

def field_folder(case, seed):
    return OUT/'runs'/case/f'seed_{seed}'

def verify_protocol():
    p = read(OUT/'protocol.json')
    assert p['frozen'] and p['fingerprint'] == fingerprint({k:v for k,v in p.items() if k!='fingerprint'})
    for category in ('sources', 'inputs'):
        for path, digest in p[category].items():
            if sha(path) != digest:
                raise RuntimeError(f'Frozen {category} changed: {path}')
    return p

def probability(case, seed, device='cpu'):
    import native_probability as Q
    folder = probability_folder(case, seed)
    gate = read(folder/'selection_frozen.json')
    assert gate['complete']
    path = folder/'selected.pt'
    expected = gate['selected_sha256']
    net = Q.load_frozen(path, device=device, expected_sha256=expected)
    cal, force = Q.calibration()
    return net, cal, force, expected

def evaluate_gate():
    p = verify_protocol()
    gate = read(OUT/'selection_frozen.json')
    assert gate['complete'] and gate['protocol_fingerprint'] == p['fingerprint']
    assert len(gate['runs']) == len(CASES)*len(SEEDS)
    for row in gate['runs']:
        for record in [row['background'], *[v for b in row['budgets'].values() for v in b.values()]]:
            if sha(record['path']) != record['sha256']:
                raise RuntimeError('Frozen field checkpoint changed')
        folder = probability_folder(row['case'], row['seed'])
        assert sha(folder/'selected.pt') == row['probability_sha256']
    return p, gate
