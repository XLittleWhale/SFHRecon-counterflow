"""Prospective fixed-query probability selection using non-target 6.5 only.

No training, no target observations/reference, no changes to historical outputs.
The plan is written and SHA-bound before the first predictive score is computed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
OUT = BASE / 'outputs/PINNTwoFluid_20261105/probability_selection'
NEURAL = HERE.parent / 'neural_closure_20261104'
sys.path.insert(0, str(NEURAL))
import common as C
import data_training as D
import probability as P

FAMILIES = ('baseline', 'probability', 'joint')
TRAIN_SEEDS = (411040, 411041, 411042)
FIELD_SEEDS = (301030, 301031, 301032)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    tmp.replace(path)


def bound(path):
    return dict(path=str(Path(path).resolve()), sha256=digest(path))


def candidate(family, seed, gate):
    net, origin = C.baseline_network(seed, trainable=False)
    if family == 'baseline':
        record = dict(family=family, training_seed=seed, origin=origin,
                      selected_epoch=origin['head']['selected_epoch'])
    else:
        row = next(r for r in gate['runs'] if r['arm'] == family and r['seed'] == seed)
        if digest(row['path']) != row['sha256']:
            raise RuntimeError('Changed selected network: ' + row['path'])
        ck = C.load(row['path'])
        if ck['arm'] != family or ck['seed'] != seed or ck['epoch'] != row['epoch']:
            raise RuntimeError('Network identity mismatch')
        net.load_state_dict(ck['net_state'], strict=True)
        record = dict(family=family, training_seed=seed, selected_model=row,
                      selected_epoch=ck['epoch'], origin=origin)
    net.freeze()
    if any(p.requires_grad for p in net.parameters()):
        raise RuntimeError('Probability network is not frozen')
    return net, record


def make_plan(gate):
    records = [candidate(family, seed, gate)[1] for family in FAMILIES for seed in TRAIN_SEEDS]
    inputs = [bound(p) for p in D.input_paths() if D.VALIDATION_CASE in str(p)
              or Path(p).name in ('operator.pt', 'calibration.json', 'force.json')]
    return dict(version='pinn-probability-selection-v1',
        case=D.VALIDATION_CASE, target_cases_allowed=False,
        particle_mask='PID % 5 == 1; all other particles removed at data load',
        interval_frames=list(range(95)), causal_filter='score endpoint before ADF update; reset at missing/cap segment',
        weights='each state mass 1/2; each physical PID equal mass inside each state',
        primary=dict(degree=20, query='fixed reference Fourier a[96,2,624]; residual=a[t+1]-Phi_phys(a[t]); old residual-subtracted local_jets',
                     physics='nu_s=0; Fns=1; no learned closure',
                     criterion='family mean across three seeds of 0.5*(free_NLL+cap_NLL)',
                     tie_tolerance=1e-12, tie_order=list(FAMILIES)),
        sensitivities=[dict(degree=28, query='same fixed pure-physics-anchor query', selects=False),
                       dict(degree=20, query='same fixed reference Fourier a; zero residual/raw finite time difference', selects=False)],
        deployment='one common winning family; keep three frozen networks paired to field seeds; no target-specific winner',
        seed_pairing=[dict(training_seed=t, field_seed=f) for t,f in zip(TRAIN_SEEDS,FIELD_SEEDS)],
        candidates=records, inputs=inputs,
        parent_protocol=bound(C.OUT/'protocol.json'), parent_training_gate=bound(C.OUT/'training/selection_frozen.json'),
        baseline_protocol=bound(C.VOUT/'protocol.json'), baseline_training_gate=bound(C.VOUT/'training/selection_frozen.json'),
        selector_source=bound(__file__),
        limitations=['6.5 validation was already used for historical training selection; this is not independent test data',
                     'Three seeds are numerical training realizations, not independent physical experiments',
                     'Historical 1.5/4 targets have been explored but are not read or used in this selection',
                     'Fixed-anchor and continuous PINN time-derivative queries differ; zero-residual sensitivity cannot change winner'])


def ensure_plan(plan):
    path = OUT/'selection_plan.json'
    if path.exists():
        if read(path) != plan:
            raise RuntimeError('Existing prospective plan differs; refuse to change selection protocol')
    else:
        dump(path, plan)
    return digest(path)


def evaluate(family, seed, gate, w, residual, degree, label):
    path = OUT/'scores'/f'{family}_{seed}_{label}.json'
    if path.exists():
        old = read(path)
        if old['selection_plan_sha256'] != digest(OUT/'selection_plan.json'):
            raise RuntimeError('Cached score bound to another plan')
        return old
    start = time.perf_counter()
    net, record = candidate(family, seed, gate)
    before = C.plain_state(net)
    with torch.no_grad():
        result = C.score(w['truth'], residual[:, 1], residual[:, 0],
                         dict(z=w['z'],c=w['c']), net, w['cal'], w['force'], 'V', degree, metrics=True)
    if any(not torch.equal(value, net.state_dict()[key]) for key,value in before.items()):
        raise RuntimeError('Probability parameters changed during scoring')
    row = dict(family=family, training_seed=seed, label=label, degree=degree,
               selection_plan_sha256=digest(OUT/'selection_plan.json'), model=record,
               free=float(result['free']), cap=float(result['cap']),
               balanced=.5*(result['free']+result['cap']), coverage=result['coverage'], spread=result['spread'],
               per_pid=result['per_pid'], wall_seconds=time.perf_counter()-start)
    if not all(np.isfinite(row[k]) for k in ('free','cap','balanced')):
        raise RuntimeError('Nonfinite predictive score')
    dump(path,row)
    print(label,family,seed,'free',row['free'],'cap',row['cap'],'balanced',row['balanced'],flush=True)
    return row


def main():
    C.setup()
    protocol = C.verify_protocol()
    gate = read(C.OUT/'training/selection_frozen.json')
    if not gate['all_required_selections_frozen'] or len(gate['runs']) != 9:
        raise RuntimeError('Historical models are not globally frozen')
    if gate['protocol_fingerprint'] != protocol['fingerprint']:
        raise RuntimeError('Training gate/protocol mismatch')
    plan = make_plan(gate)
    plan_sha = ensure_plan(plan)  # Must precede any score.
    final_path = OUT/'selection_frozen.json'
    if final_path.exists():
        final = read(final_path)
        if final['selection_plan_sha256'] != plan_sha:
            raise RuntimeError('Existing frozen selection has another plan')
        for row in final['models']:
            if digest(row['path']) != row['sha256']:
                raise RuntimeError('Changed frozen probability package')
        print('ALREADY_COMPLETE', final['family'], flush=True)
        return
    t0, cpu0 = time.perf_counter(), time.process_time()
    dump(OUT/'status.json',dict(complete=False,stage='loading_non_target',selection_plan_sha256=plan_sha))
    case = D.load_case(D.VALIDATION_CASE)
    w = D.window(case,0,95)
    if not bool((w['z']['pid'] % 5 == 1).all()):
        raise RuntimeError('Unauthorized validation PID')
    with torch.no_grad():
        residual = D.reference_residual(case['op'],w['truth'],substeps=2)
        expected_jets = D.query(w,w['truth'],residual)
        actual_jets = C.M.R.local_jets(w['truth'],residual[:,1],residual[:,0],w['z'],w['c'])
        jet_error = [float((a-b).abs().max()) for a,b in zip(expected_jets,actual_jets)]
    if max(jet_error) > 1e-10:
        raise RuntimeError('Fixed-query interface mismatch: '+str(jet_error))
    rows=[]
    for label,degree,eta in [('anchor_gh20',20,residual),('anchor_gh28',28,residual),
                             ('zero_residual_gh20',20,torch.zeros_like(residual))]:
        for family in FAMILIES:
            for seed in TRAIN_SEEDS:
                dump(OUT/'status.json',dict(complete=False,stage=label,family=family,training_seed=seed,
                                            completed_scores=len(rows),selection_plan_sha256=plan_sha))
                rows.append(evaluate(family,seed,gate,w,eta,degree,label))
    primary=[r for r in rows if r['label']=='anchor_gh20']
    summary=[]
    for family in FAMILIES:
        group=[r for r in primary if r['family']==family]
        summary.append(dict(family=family,**{k:float(np.mean([r[k] for r in group])) for k in ('free','cap','balanced')}))
    best=min(r['balanced'] for r in summary)
    family=next(r['family'] for r in summary if r['balanced'] <= best+plan['primary']['tie_tolerance'])
    # Cross-implementation audit uses the exact same full non-target window.
    net,_=candidate('baseline',TRAIN_SEEDS[0],gate)
    with torch.no_grad():
        alternate=P.window_nll(w['c'],w['z'],expected_jets,net,w['cal'],w['force'],w['masks']['permitted'],20,95)
    first=next(r for r in primary if r['family']=='baseline' and r['training_seed']==TRAIN_SEEDS[0])
    nll_error={key:abs(float(alternate[key])-first[key]) for key in ('free','cap')}
    if max(nll_error.values())>1e-9:
        raise RuntimeError('Causal scoring interface mismatch: '+str(nll_error))
    packages=[]
    for seed,field_seed in zip(TRAIN_SEEDS,FIELD_SEEDS):
        net,record=candidate(family,seed,gate)
        path=OUT/'models'/f'seed_{seed}.pt'
        C.save(path,dict(net_state=C.plain_state(net),mode='temporal',family=family,training_seed=seed,
                         field_seed=field_seed,selection_plan_sha256=plan_sha,origin=record,
                         target_data_used_for_selection=False,network_trainable=False))
        packages.append(dict(**bound(path),family=family,training_seed=seed,field_seed=field_seed,source=record))
    report=dict(complete=True,family=family,selection_plan_sha256=plan_sha,plan_path=str(OUT/'selection_plan.json'),
                source_sha256=digest(__file__),criterion=plan['primary']['criterion'],family_means=summary,
                models=packages,scores=[bound(OUT/'scores'/f"{r['family']}_{r['training_seed']}_{r['label']}.json") for r in rows],
                fixed_query_jet_max_abs_error=jet_error,alternate_causal_nll_abs_error=nll_error,
                target_H_scored=False,target_reference_read=False,probability_parameters_updated=False,
                parent_protocol_fingerprint=protocol['fingerprint'],wall_seconds=time.perf_counter()-t0,
                cpu_seconds=time.process_time()-cpu0,limitations=plan['limitations'])
    dump(final_path,report)
    dump(OUT/'status.json',dict(complete=True,stage='frozen',family=family,completed_scores=len(rows),
                               selection_plan_sha256=plan_sha,selection_sha256=digest(final_path),
                               wall_seconds=report['wall_seconds'],cpu_seconds=report['cpu_seconds']))
    print('SELECTION_FROZEN',family,summary,flush=True)


if __name__=='__main__':
    main()
