"""Fresh, native-camera endpoint training of the Nov5 joint architecture.

The original propagator, ADF, GH integration and endpoint scoring are imported
into an isolated module instance. Only its parameter-validation and PID-split
hooks are replaced here; neither the historical module nor its files change.
No microscopic force/sign/anchor labels or fictitious 1 ms endpoints are used.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent/'pinn_cf_20261105/cf_probability.py'
_name = '_cf_sim_each_native_probability_baseline_20261109'
if _name in sys.modules:
    _runtime = sys.modules[_name]
else:
    _spec = importlib.util.spec_from_file_location(_name, BASELINE)
    _runtime = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _runtime
    _spec.loader.exec_module(_runtime)

core, DT = _runtime.core, _runtime.DT


def group_mask(pid, group):
    if group in ('F', 'train', 'probability'):
        return pid % 5 == 2
    if group in ('I', 'fit'):
        return pid % 5 >= 3
    if group in ('V', 'validation', 'selection'):
        return pid % 5 == 1
    if group in ('H', 'test'):
        return pid % 5 == 0
    raise ValueError('Require explicit F/I/V/H particle group')


def _validate(c, z, net, jets):
    trainable = any(p.requires_grad for p in net.parameters())
    if trainable and not (net.training and getattr(net, '_endpoint_training_authorized', False)):
        raise RuntimeError('Trainable probability requires the explicit endpoint-training branch')
    v, g, acc = jets
    B, M = z['time'].shape
    if v.shape != (B, M, 2, 2) or g.shape != (B, M, 2, 2, 2) or acc.shape != v.shape:
        raise ValueError('Expected camera quadrature jets [B,M,fluid,component]')
    h = z['dt']
    if bool((h < 0).any()) or bool((h > DT+1e-10).any()):
        raise ValueError('Internal pieces must lie in [0,1ms]')
    if not torch.allclose(h.sum(-1)/DT, z['end']-z['start'], atol=2e-7, rtol=0):
        raise ValueError('Internal pieces must exactly tile the native interval')
    if bool((h[:, 0] <= 0).any()) or bool((z['end'] <= z['start']).any()):
        raise ValueError('Every interval requires a positive physical duration')
    if not bool(torch.isfinite(v).all() & torch.isfinite(g).all() & torch.isfinite(acc).all()):
        raise ValueError('Nonfinite candidate field query')
    _runtime.history(z)


class NativeLatentHead(_runtime.LM.LatentTransitionHead):
    """Same bounded head; permit base gradients only in explicit training."""

    def parameters_at(self, h, ctx, prevctx):
        if ctx.shape != (len(h), 17) or prevctx.shape != ctx.shape:
            raise ValueError('Expected current and previous context [B,17]')
        base = self._base
        if any(p.requires_grad for p in base.parameters()) and not (
                self.training and getattr(self, '_endpoint_training_authorized', False)):
            raise RuntimeError('The marginal network must remain frozen outside endpoint training')
        encoded = base.encode(h, ctx)
        inputs = torch.cat((encoded, (prevctx-base.cm)/base.cs, (ctx-base.cm)/base.cs), -1)
        raw = self.network(inputs)
        kappa = _runtime.LM.bounded_kappa(raw[:, 0])
        A = 1.5*torch.tanh(raw[:, 1:5]).reshape(-1, 2, 2)
        norm = torch.linalg.matrix_norm(A, ord=2, dim=(-2, -1))
        A = A*(.995/norm.clamp_min(.995))[:, None, None]
        b = 2*torch.tanh(raw[:, 5:7])
        d0 = torch.exp(math.log(50.)*torch.tanh(raw[:, 7]))
        off = .5*torch.tanh(raw[:, 8])
        d1 = torch.exp(math.log(50.)*torch.tanh(raw[:, 9]))
        L = torch.stack((torch.stack((d0, torch.zeros_like(d0)), -1),
                         torch.stack((off, d1), -1)), -2)
        return kappa, A, b, L


class NativeNetwork(_runtime.FrozenNetwork):
    """Identical tensor architecture, constructed without pretrained weights."""

    def __init__(self, seed=411040, device='cpu'):
        # Preserve unrelated caller RNG streams while making a reproducible init.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            previous_dtype = torch.get_default_dtype()
            try:
                # Historical layers have no dtype argument. Pin initialization
                # precision so CLI and imported workers produce identical seeds.
                torch.set_default_dtype(torch.float64)
                torch.nn.Module.__init__(self)
                self.base = _runtime.TM.TransitionDensity('independent', _runtime.ob.Density()).double()
                self.head = NativeLatentHead(self.base).double()
                self.mode = 'temporal'
            finally:
                torch.set_default_dtype(previous_dtype)
        self.to(device=device, dtype=torch.float64)
        self.freeze()

    def enable_training(self):
        self._endpoint_training_authorized = True
        self.head._endpoint_training_authorized = True
        self.train(True)
        for p in self.parameters():
            p.requires_grad_(True)
        return self

    def freeze(self):
        self._endpoint_training_authorized = False
        self.head._endpoint_training_authorized = False
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
            p.grad = None
        return self


# These hooks belong only to the private import above. The historical module
# name `cf_probability` and its functions retain their original frozen guards.
_runtime._validate = _validate
_runtime.group_mask = group_mask
weights = _runtime.weights
subset = _runtime.subset
free_distribution = _runtime.free_distribution
captured_distribution = _runtime.captured_distribution
sequence_packs = _runtime._sequence_packs
score = _runtime.score
FrozenNetwork = NativeNetwork


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_paths():
    return [Path(__file__), BASELINE, Path(_runtime.TM.__file__), Path(_runtime.LM.__file__),
            Path(_runtime.ob.__file__), Path(core.__file__)]


def calibration():
    """Fixed inherited force anchors/noise; no new true-force labels are made."""
    folder = _runtime.CALIBRATION
    return tuple(json.loads((folder/name).read_text(encoding='utf8'))
                 for name in ('calibration.json', 'force.json'))


def calibration_paths():
    return [_runtime.CALIBRATION/name for name in ('calibration.json', 'force.json')]


def load_frozen(path, device='cpu', expected_sha256=None):
    path = Path(path)
    if expected_sha256 is not None and sha(path) != expected_sha256:
        raise RuntimeError('Probability checkpoint SHA mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model = NativeNetwork(saved['seed'], device)
    model.load_state_dict(saved['net_state'], strict=True)
    return model.freeze()
