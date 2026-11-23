"""Controlled CF inversion with the preselected Nov5 frozen joint model."""
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
OUT = Path(os.environ.get('CF_EACH_OUTPUT', str(BASE/'results'))).resolve()
INPUT = Path(os.environ.get('CF_EACH_INPUT', str(BASE/'outputs/CFSimEachPINN_20261109/prepared'))).resolve()
SOURCE = BASE/'outputs/cf_reference_transient1s_50frames_results_20261108'
CASES = ('cf_2K_Q113', 'cf_2K_Q175', 'cf_2K_Q217')
SEEDS = (301030,)
TRAIN_SEEDS = (411040,)
BUDGETS = (150,200,600) if os.environ.get('CF_FROZEN_EXTENDED')=='1' else (150,)
COMPARISON = Path(os.environ.get('CF_FROZEN_COMPARISON', str(BASE/'comparison_baseline'))).resolve()

def setup():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(int(os.environ.get('CF_EACH_THREADS', '2')))

def device():
    name = os.environ.get('CF_EACH_DEVICE', 'cuda')
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this GPU package. Select a CUDA PyTorch image; CPU is only allowed explicitly for diagnostics.')
    return name

def asset_key(path):
    return Path(path).resolve().relative_to(BASE).as_posix()

def asset_path(key):
    path = (BASE/key).resolve()
    if not path.is_relative_to(BASE):
        raise ValueError('Asset path escapes the bundle')
    return path

def output_path(key):
    path = (OUT/key).resolve()
    if not path.is_relative_to(OUT):
        raise ValueError('Checkpoint path escapes the output directory')
    return path

def move_tensors(z, target=None):
    return {k:v.to(target or device()) for k,v in z.items()}

def utc():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def sha(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
        return digest.hexdigest()

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
    return INPUT/case

def probability_folder(case, seed):
    return OUT/'probability'/case/f'seed_{TRAIN_SEEDS[SEEDS.index(seed)]}'

def field_folder(case, seed):
    return OUT/'runs'/case/f'seed_{seed}'

def verify_protocol():
    p = read(OUT/'protocol.json')
    if p.get('version')!='cf-sim-frozen-pinn-20261110-v1':
        raise RuntimeError('Use a separate output directory for the frozen-probability comparison')
    if p['budgets']!=list(BUDGETS):
        raise RuntimeError('Snapshot budgets differ from the sealed comparison protocol')
    if p['probability']['training'] or p['probability']['probability_updates']!=0:
        raise RuntimeError('The original probability model must remain frozen')
    assert p['frozen'] and p['fingerprint'] == fingerprint({k:v for k,v in p.items() if k!='fingerprint'})
    for category in ('sources', 'inputs'):
        for path, digest in p[category].items():
            if sha(asset_path(path)) != digest:
                raise RuntimeError(f'Frozen {category} changed: {path}')
    return p

def probability(case, seed, device='cpu'):
    if case not in CASES:
        raise ValueError(case)
    import frozen_probability as Q
    return Q.load(seed,device=device)

def probability_record(seed):
    import frozen_probability as Q
    return Q.record(seed)

def evaluate_gate():
    p = verify_protocol()
    gate = read(OUT/'selection_frozen.json')
    assert gate['complete'] and gate['protocol_fingerprint'] == p['fingerprint']
    assert len(gate['runs']) == len(CASES)*len(SEEDS)
    for row in gate['runs']:
        for record in [row['background'], *[v for b in row['budgets'].values() for v in b.values()]]:
            if sha(output_path(record['path'])) != record['sha256']:
                raise RuntimeError('Frozen field checkpoint changed')
        assert probability_record(row['seed'])['sha256'] == row['probability_sha256']
    return p, gate
