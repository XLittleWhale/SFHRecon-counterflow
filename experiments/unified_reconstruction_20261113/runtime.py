"""Portable imports and atomic records for the staged reconstruction protocol."""
from pathlib import Path
import importlib.util
import hashlib
import json
import os
import sys
import uuid
import socket
from contextlib import contextmanager
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
FROZEN = HERE.parent/'cf_sim_frozen_pinn_20261110'

def module(name, path):
    path = Path(path).resolve()
    if name in sys.modules:
        if Path(sys.modules[name].__file__).resolve() != path:
            raise RuntimeError('Module collision; use a separate process: '+name)
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

# These historical modules intentionally expose a small, tested numerical API.
# Do not import their old experiment controllers or change their source files.
sys.path.insert(0, str(FROZEN))
P = module('native_probability', FROZEN/'native_probability.py')
F = module('field_adapter', FROZEN/'field_adapter.py')
B = module('frozen_probability', FROZEN/'frozen_probability.py')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()

def atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        writer(temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

def dump(path, value):
    atomic(path, lambda p:p.write_text(json.dumps(value, indent=2, ensure_ascii=False,
        allow_nan=False), encoding='utf8'))

def save(path, value):
    atomic(path, lambda p:torch.save(value, p))

def state(net):
    return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}

def move(z, device):
    return {k:v.to(device=device, dtype=torch.float64 if v.is_floating_point() else v.dtype)
        for k,v in z.items() if torch.is_tensor(v)}

@contextmanager
def lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'run.lock'
    import psutil
    if path.exists():
        owner=read(path)
        if owner.get('host')!=socket.gethostname():raise RuntimeError('Run lock belongs to another host; verify that run has stopped before moving its run.lock aside')
        try:
            active=abs(psutil.Process(owner['pid']).create_time()-owner['process_created'])<1e-3
        except (psutil.NoSuchProcess,KeyError):active=False
        if active:raise RuntimeError('This output is already in use by PID '+str(owner['pid']))
        path.rename(path.with_name('run.lock.stale.'+uuid.uuid4().hex[:8]))
    with path.open('x', encoding='utf8') as stream:
        json.dump(dict(pid=os.getpid(),host=socket.gethostname(),process_created=psutil.Process().create_time()),stream)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)

def setup(device):
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    if device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Formal pipeline requires CUDA; use tests for CPU diagnostics')

def contained(root, name):
    path = (Path(root)/name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Path escapes the run: '+str(name))
    return path
