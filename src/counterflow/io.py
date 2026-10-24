from pathlib import Path
import json,os,hashlib,random,time
import numpy as np
import torch
def json_scalar(value):
    """Convert numerical scalar wrappers; still reject NaN/Inf and unknown objects."""
    if isinstance(value,np.generic):return value.item()
    if torch.is_tensor(value) and value.numel()==1:return value.detach().cpu().item()
    raise TypeError(f'Object of type {type(value).__module__}.{type(value).__name__} is not JSON serializable')

def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False,default=json_scalar),encoding='utf8');os.replace(tmp,path)

def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(data,tmp);os.replace(tmp,path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def load(path,device='cpu'):return torch.load(path,map_location=device,weights_only=False)

def seed(n):random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)

def cpu_state(net):return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}

def setup(device):
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable. Use a CUDA PyTorch environment; no automatic long CPU fallback.')
    torch.set_num_threads(2)
    # Frozen GRU still needs input derivatives in inverse optimization.
    torch.backends.cudnn.enabled=False
    torch.backends.cuda.matmul.allow_tf32=False
    return torch.device(device)
