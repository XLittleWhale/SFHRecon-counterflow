"""Load the exact joint family selected for the November 5 PINN baselines.

The gate is an immutable provenance record. Its old absolute path strings are
never dereferenced: portable bundle paths below identify the same sealed bytes.
Neither per-case training nor context normalization is performed here.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import native_probability as P

BASE = Path(__file__).resolve().parents[3]
FOLDER = BASE/'outputs/PINNTwoFluid_20261105/probability_selection'
GATE_SHA256 = '7a5dbf66d8ea7551b9c75b983dde2d9a4998515fecdae0537558074be47d768e'
PLAN_SHA256 = 'a6de38ab49949374362222e72322bfae677637a2f8b8edafb03bf817326cb54c'
MODELS = {
    301030: (411040, '9962a81b8a0fd6708732415b3c17d62225db8f86c50299c994d482b459fb4a63'),
    301031: (411041, 'eb08f3f1f2a2d95350fd725403a73c15c2330fe06be4c9cbc8943807612cd342'),
    301032: (411042, 'da39909d765652c6f07839fbfd65ebd61dfc95e7809e4f5f19f5a42e92d80701'),
}
CALIBRATION_SHA256 = {
    'calibration.json': '6d31a91abf760b48c95ce55121da89350dde73661b766ef155783622ed0c7d40',
    'force.json': '89a941ac0d8c9e94eb3354ae8005d23f3e39441a606406ba586f082f565cb5ba',
}


def _verify(path, expected):
    if P.sha(path) != expected:
        raise RuntimeError('Original frozen probability asset hash mismatch: '+str(path))


def records():
    """Verify and return all fixed seed mappings without loading a model."""
    _verify(FOLDER/'selection_frozen.json', GATE_SHA256)
    _verify(FOLDER/'selection_plan.json', PLAN_SHA256)
    gate = json.loads((FOLDER/'selection_frozen.json').read_text(encoding='utf8'))
    if not gate['complete'] or gate['family'] != 'joint' or gate['probability_parameters_updated']:
        raise RuntimeError('Require the original completed joint family selection')
    if gate['selection_plan_sha256'] != PLAN_SHA256 or len(gate['models']) != 3:
        raise RuntimeError('Original family selection plan differs')
    rows = []
    for field_seed, (training_seed, digest) in MODELS.items():
        row = next(r for r in gate['models'] if r['field_seed'] == field_seed)
        if row['family'] != 'joint' or row['training_seed'] != training_seed or row['sha256'] != digest:
            raise RuntimeError('Original probability/field seed pairing differs')
        path = FOLDER/'models'/f'seed_{training_seed}.pt'
        _verify(path, digest)
        rows.append(dict(field_seed=field_seed, training_seed=training_seed, family='joint',
            path=path.relative_to(BASE).as_posix(), sha256=digest,
            source_selected_epoch=row['source']['selected_epoch']))
    for path in P.calibration_paths():
        _verify(path, CALIBRATION_SHA256[path.name])
    return rows


def source_paths():
    """Runtime assets, including the original unchanged selection records."""
    return [FOLDER/'selection_frozen.json', FOLDER/'selection_plan.json',
            *[FOLDER/'models'/f'seed_{seed}.pt' for seed, _ in MODELS.values()],
            *P.calibration_paths()]


def record(field_seed):
    """One verified record with an absolute Path for local callers."""
    if field_seed not in MODELS:
        raise ValueError('Unknown original PINN seed: '+str(field_seed))
    row = next(row for row in records() if row['field_seed'] == field_seed)
    return dict(row, path=BASE/row['path'])


def load(field_seed, device='cpu'):
    """Return (frozen network, original calibration, original force, SHA-256)."""
    if field_seed not in MODELS:
        raise ValueError('Unknown original PINN seed: '+str(field_seed))
    row = record(field_seed)
    saved = torch.load(row['path'], map_location='cpu', weights_only=False)
    if (saved['family'] != 'joint' or saved['mode'] != 'temporal'
            or saved['training_seed'] != row['training_seed'] or saved['field_seed'] != field_seed
            or saved['selection_plan_sha256'] != PLAN_SHA256
            or saved['target_data_used_for_selection'] or saved['network_trainable']):
        raise RuntimeError('Original joint checkpoint metadata differs from the selection')
    net = P.NativeNetwork(row['training_seed'], device)
    net.load_state_dict(saved['net_state'], strict=True)
    net.freeze()
    if any(not torch.equal(value.detach().cpu(), saved['net_state'][key])
           for key, value in net.state_dict().items()):
        raise RuntimeError('Original model parameters or normalization buffers were altered')
    cal, force = P.calibration()
    return net, cal, force, row['sha256']
