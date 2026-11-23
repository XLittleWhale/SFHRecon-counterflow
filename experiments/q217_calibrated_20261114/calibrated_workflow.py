"""New protocol; imported probability checkpoints retain their original lineage."""
from pathlib import Path
import shutil
import torch
from runtime import BASE, B, read, load, dump, sha, fingerprint, contained

HERE=Path(__file__).resolve().parent


def defaults(profile='formal'):
    smoke=profile=='smoke'
    if profile not in ('formal','smoke'):raise ValueError(profile)
    return dict(version='q217-calibrated-frozen-probability-v1-20261114',profile=profile,
        scope='each',cases=[217],field_seeds=[301030] if smoke else [301030,301031,301032],
        probability_seeds=[411040] if smoke else [411040,411041,411042],
        probability_policy='reuse_completed_CF_selected_joint; zero probability updates',
        data_role='same_realization_supervised_calibration',independent_field_generalization=False,
        diagnostic_only=smoke,geometry_policy='frozen_cpu_coordinates_and_initial_parameters',
        residual_policy='frozen_CF_second_moment_tolerance; original temporal kernels',
        output_policy='q=network_output/ring; fixed_geometry_decode; identity_output_factor',
        field=dict(budgets=[2] if smoke else [200],lr=.001,validate_every=1 if smoke else 10,
            train_degree=12,validation_degree=20,guard=.01,selection_min_delta=1e-6,
            likelihood_weight=.08,gradient_clip=5.,learned_closure_in_pinn=False),
        smoke_keeps_full_data_and_time_window=True,
        evaluation_truth_used_for_selection=False,reference_truth_used_for_tolerance_calibration=True)


def copy_exact(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if sha(source)!=sha(destination):raise RuntimeError('Existing sealed input differs: '+str(destination))
    else:shutil.copy2(source,destination)


def seal(out,profile='formal'):
    out=Path(out);config=defaults(profile)
    if (out/'protocol.json').exists():
        p=verify(out)
        if p['config']!=config:raise RuntimeError('Protocol/profile changed; use a new output directory')
        freeze_probabilities(out)
        return p
    if any((out/name).exists() for name in ('fields','probability','complete.json','probability_frozen.json')):
        raise RuntimeError('Output contains an earlier experiment without this protocol')
    source=read(BASE/'revision_assets/asset_manifest.json')
    target_assets={};case='cf_2K_Q217';destination=out/'datasets'/case
    for name,path in source['datasets'].items():
        copy_exact(BASE/path,destination/name)
        target_assets[(destination/name).relative_to(out).as_posix()]=sha(destination/name)
    models=[]
    for row in source['models']:
        if row['seed'] not in config['field_seeds']:continue
        target=out/'probability'/case/f"seed_{row['seed']}"/'frozen_selected.pt'
        copy_exact(BASE/row['asset_path'],target)
        target_assets[target.relative_to(out).as_posix()]=sha(target)
        models.append(dict(row,path=target.relative_to(out).as_posix()))
    sources=read(BASE/'revision_assets/runtime_contract.json')['source_sha256']
    p=dict(config=config,references=[dict(name=case,kind='cf_simulation',adapter='channel')],
        targets=[dict(name=case,kind='cf_simulation',adapter='channel',reference=case,path='datasets/'+case)],
        models=models,assets=source['sha256'],sources=sources,target_assets=target_assets,
        runtime_contract_sha256=sha(BASE/'revision_assets/runtime_contract.json'),
        source_probability_protocol_fingerprint=source['source_protocol_fingerprint'],frozen=True)
    p['fingerprint']=fingerprint(p)
    dump(out/'protocol.json',p)
    freeze_probabilities(out)
    return p


def verify(out):
    out=Path(out);p=read(out/'protocol.json')
    if p['fingerprint']!=fingerprint({k:v for k,v in p.items() if k!='fingerprint'}):
        raise RuntimeError('Protocol fingerprint mismatch')
    if sha(BASE/'revision_assets/runtime_contract.json')!=p['runtime_contract_sha256']:
        raise RuntimeError('Runtime contract changed since protocol sealing')
    for mapping,root in ((p['assets'],BASE),(p['sources'],BASE),(p['target_assets'],out)):
        for name,digest in mapping.items():
            if sha(contained(root,name))!=digest:raise RuntimeError('Sealed input/source changed: '+name)
    return p


def freeze_probabilities(out):
    out=Path(out);p=verify(out)
    for row in p['models']:
        ck=load(out/row['path'])
        if sha(out/row['path'])!=row['sha256'] or ck['protocol_fingerprint']!=p['source_probability_protocol_fingerprint']:
            raise RuntimeError('Imported CF probability lineage mismatch')
        if ck['dependencies']['periodic']!=row['parent_sha256']:
            raise RuntimeError('Original periodic parent differs')
        if any(not bool(torch.isfinite(v).all()) for v in ck['net_state'].values()):
            raise RuntimeError('Nonfinite imported probability model')
    gate=dict(complete=True,protocol_fingerprint=p['fingerprint'],models=p['models'],
        scope='each',data_role=p['config']['data_role'],diagnostic_only=p['config']['diagnostic_only'],
        network_updates=0,imported_from_completed_CF_training=True,
        source_protocol_fingerprint=p['source_probability_protocol_fingerprint'],
        independent_field_generalization=False,evaluation_truth_used_for_selection=False)
    path=out/'probability_frozen.json'
    if path.exists() and read(path)!=gate:raise RuntimeError('Probability gate changed')
    if not path.exists():dump(path,gate)
    return gate


def frozen(out,case,seed,device):
    out=Path(out);p=read(out/'protocol.json');gate=read(out/'probability_frozen.json')
    if not gate['complete'] or gate['protocol_fingerprint']!=p['fingerprint']:
        raise RuntimeError('Probability gate mismatch')
    row=next(r for r in gate['models'] if r['case']==case and r['seed']==seed)
    path=contained(out,row['path'])
    if sha(path)!=row['sha256']:raise RuntimeError('Frozen probability bytes changed')
    ck=load(path)
    if ck['protocol_fingerprint']!=p['source_probability_protocol_fingerprint']:
        raise RuntimeError('Original probability provenance changed')
    net,cal,force,parent=B.load(seed,device)
    if parent!=row['parent_sha256']:raise RuntimeError('Periodic parent mismatch')
    net.load_state_dict(ck['net_state'],strict=True);net.freeze()
    if any(not torch.equal(v.detach().cpu(),ck['net_state'][k]) for k,v in net.state_dict().items()):
        raise RuntimeError('Imported probability tensors differ')
    return net,cal,force,row['sha256']
