"""V-only comparison of original joint and each-trained endpoint models.

This diagnostic uses already selected V particles and interpolation of camera
field truth. V was used to select the each model, so the comparison is
exploratory. H and reconstructed field checkpoints are never loaded. A cold
Poiseuille/uniform context contrasts field-conditioning effects with model
effects; the truth-jet context is not an achievable reconstruction benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import uuid

import torch
import native_probability as P
import frozen_probability as FROZEN
from field_adapter import probability_physics

BASE = Path(__file__).resolve().parents[3]
CASES = ('cf_2K_Q113', 'cf_2K_Q175', 'cf_2K_Q217')
PREPARED = BASE/'outputs/CFSimEachPINN_20261109/prepared'
EACH = BASE/'outputs/cf_sim_each_pinn_day_results_20261109'
OUTPUT = BASE/'outputs/CFSimFrozenPINN_20261110/diagnostics'


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def _dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    os.replace(temporary, path)


def load_validation(case_dir):
    """Read the hash-bound F/V pack, retain only V rows for every score."""
    observation_path, jets_path = (case_dir/name for name in ('probability_observations.pt', 'training_jets.pt'))
    obs, pack = _load(observation_path), _load(jets_path)
    all_z = obs['z']
    if not bool(((all_z['pid'] % 5 == 2) | (all_z['pid'] % 5 == 1)).all()):
        raise RuntimeError('Input observation pack contains forbidden I/H particles')
    if pack['meta']['probability_observation_sha256'] != P.sha(observation_path):
        raise RuntimeError('Truth jets are not bound to these observations')
    if not torch.equal(all_z['row_id'], pack['row_id']):
        raise RuntimeError('Truth jet and observation rows differ')
    use = torch.where(all_z['pid'] % 5 == 1)[0]
    z = P.subset(all_z, use)
    jets = tuple(pack[name][use] for name in ('v', 'g', 'acc'))
    z['row_id'] = torch.arange(len(use))
    if not all(torch.isfinite(value).all() for value in jets):
        raise RuntimeError('Nonfinite V truth jets')
    P.weights(z, 'V')
    c = probability_physics(obs['meta']['physics_config'], obs['meta']['intervals'])
    return c, z, jets, obs['meta'], dict(probability_observations=P.sha(observation_path), training_jets=P.sha(jets_path))


def cold_queries(c, z):
    """Exact zero-coefficient background; no PDE/prior matrices are needed."""
    x = z['x']
    U, vs, width = (c['physics'][key] for key in ('vn_mean_m_s','vs_mean_m_s','width_m'))
    r = x[..., 0]/width
    v = x.new_zeros(*x.shape[:-1], 2, 2)
    g = x.new_zeros(*x.shape[:-1], 2, 2, 2)
    v[..., 0, 1] = 6*U*r*(1-r)
    v[..., 1, 1] = vs
    g[..., 0, 1, 0] = 6*U/width*(1-2*r)
    return v, g, torch.einsum('...fij,...fj->...fi', g, v)


def _load_each(case, each_root, input_hashes):
    folder = each_root/'probability'/case/'seed_411040'
    gate, protocol = _read(folder/'selection_frozen.json'), _read(folder/'protocol.json')
    if not gate['complete'] or gate['seed'] != 411040 or protocol['seed'] != 411040:
        raise RuntimeError('Require the completed per-case probability selection')
    if gate['protocol_fingerprint'] != protocol['fingerprint']:
        raise RuntimeError('Each selection and protocol differ')
    for key, name in [('probability_observations','probability_observations.pt'), ('training_jets','training_jets.pt')]:
        if protocol['inputs']['prepared/'+name] != input_hashes[key]:
            raise RuntimeError('Each model used different prepared input: '+name)
    for path in P.calibration_paths():
        source_key = next(k for k in protocol['inputs'] if k.endswith('/'+path.name))
        if protocol['inputs'][source_key] != P.sha(path):
            raise RuntimeError('Each and original models use different calibration')
    saved = _load(folder/'selected.pt')
    if saved['epoch'] != gate['selected_epoch'] or saved['protocol_fingerprint'] != gate['protocol_fingerprint']:
        raise RuntimeError('Each selected checkpoint identity differs')
    net = P.load_frozen(folder/'selected.pt', expected_sha256=gate['selected_sha256'])
    return net, dict(path=str(folder/'selected.pt'), sha256=gate['selected_sha256'],
        selected_epoch=gate['selected_epoch'], completed_epochs=gate['epoch'],
        selected_validation=saved['validation'])


@torch.no_grad()
def run(case, prepared_root=PREPARED, each_root=EACH, output=OUTPUT, threads=2):
    if case not in CASES:
        raise ValueError(case)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(threads)
    started = time.perf_counter()
    c, z, truth_jets, meta, input_hashes = load_validation(Path(prepared_root)/case)
    old, cal, force, old_sha = FROZEN.load(301030)
    each, each_record = _load_each(case, Path(each_root), input_hashes)
    result = dict(version='cf-sim-frozen-vs-each-V-diagnostic-v1', complete=False, case=case,
        field_seed=301030, probability_seed=411040, degree=20, threads=threads, device='cpu',
        V_rows=len(z['pid']), V_pids=len(z['pid'].unique()), input_sha256=input_hashes,
        original_joint_sha256=old_sha, each_model=each_record,
        scores={}, H_loaded=False, parameters_updated=False, normalization_updated=False,
        selection_changed=False, V_used_previously_to_select_each_model=True,
        interpretation='Exploratory V diagnostic; truth jets are camera-space/time interpolants, not exact microscopic jets. '
                       'Original-model transfer and each model differ in training data/objective and normalization. '
                       'Cold-vs-truth context contrasts do not prove reconstruction accuracy.',
        context_contrast_definition='cold NLL/ms minus reference-jet NLL/ms; positive means reference context scores better')
    path = Path(output)/(case+'.json')
    for family, net in [('old_joint', old), ('each', each)]:
        before = {k:v.detach().clone() for k,v in net.state_dict().items()}
        result['scores'][family] = {}
        for context in ('reference_jets', 'cold'):
            query = (lambda zz: tuple(a[zz['row_id']] for a in truth_jets)) if context == 'reference_jets' else (lambda zz: cold_queries(c, zz))
            tick = time.perf_counter()
            score = P.score(c, z, net, cal, force, query, 'V', degree=20, metrics=True,
                            backward=False, batch=512, batch_pids=32)
            score['wall_seconds'] = time.perf_counter()-tick
            result['scores'][family][context] = score
            result['wall_seconds'] = time.perf_counter()-started
            _dump(path, result)
            print(case, family, context, 'NLL/ms',score['total'],'free',score['free'],'cap',score['cap'],
                  'coverage',score['coverage'],'seconds',round(score['wall_seconds'],2),flush=True)
        if any(not torch.equal(v, before[k]) for k,v in net.state_dict().items()) or any(p.requires_grad for p in net.parameters()):
            raise RuntimeError('Diagnostic changed a probability model or normalization buffer')
    result['contrasts'] = {
        'cold_minus_reference_nll_per_ms': {family:{kind:result['scores'][family]['cold'][kind]-result['scores'][family]['reference_jets'][kind]
              for kind in ('total','free','cap')} for family in ('old_joint','each')},
        'old_minus_each_nll_per_ms': {context:{kind:result['scores']['old_joint'][context][kind]-result['scores']['each'][context][kind]
              for kind in ('total','free','cap')} for context in ('reference_jets','cold')}}
    result['each_saved_validation_max_abs_error'] = max(abs(result['scores']['each']['reference_jets'][key]-each_record['selected_validation'][key])
                                                       for key in ('total','free','cap'))
    result['complete'] = True
    result['wall_seconds'] = time.perf_counter()-started
    _dump(path, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--prepared-root', type=Path, default=PREPARED)
    parser.add_argument('--each-root', type=Path, default=EACH)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    run(args.case, args.prepared_root, args.each_root, args.output, args.threads)
