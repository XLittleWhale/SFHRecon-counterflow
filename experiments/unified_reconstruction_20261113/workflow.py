"""Sealed inputs, each-run lineage and all-selection-before-evaluation gates."""
import math
import shutil
from runtime import *
import common_training as T


def defaults(profile='baseline_retrain_all'):
    smoke=profile=='smoke'
    return dict(version='unified-baseline-full-v2-20261113',profile=profile,scope='each',
        cases=[217] if smoke else [113,175,217],field_seeds=[301030] if smoke else [301030,301031,301032],
        probability_seeds=[411040] if smoke else [411040,411041,411042],
        initialization='periodic_selected_joint',reference_policy='reuse_existing',
        data_role='same_realization_supervised_calibration',independent_field_generalization=False,
        same_realization_field_truth_enters_joint=True,evaluation_truth_used_for_selection=False,
        roles=dict(F='pid%5==2',I='pid%5>=3',V='pid%5==1',H='pid%5==0',inner='within F/V: (pid//5)%2==0',outer='within F/V: (pid//5)%2==1'),
        stages={k:('inherit' if profile=='joint_continuation' and k in ('A','B') else 'retrain') for k in T.STAGES},
        field=dict(budgets=[1,2] if smoke else [200,600],lr=.001,validate_every=1 if smoke else 10,
            train_degree=2 if smoke else 12,validation_degree=2 if smoke else 20,guard=.01,
            selection_min_delta=1e-6,likelihood_weight=.08,gradient_clip=5.,learned_closure_in_pinn=False),
        time_adapter='boundary_clock_v2: (1-f)I+fK, sign-conditional Gaussian moments; one real endpoint',
        window_rule='max(5ms,2 native intervals)',
        inner_time_preconditioner='inner_lr * min(1,5ms/window_duration)^2; shared by all geometries',diagnostic_only=smoke)


def _copy(source,target):
    if target.exists():
        if sha(source)!=sha(target):raise RuntimeError('Existing sealed input differs: '+str(target))
        return
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)


def _smoke_pack(source,target,name):
    pack=load(source/name);z=pack['z'];first=int(z['frame'].min())
    mask=(z['frame']>=first)&(z['frame']<first+2);pids=[]
    groups=('H',) if name=='heldout_observations.pt' else ('I','V')
    for group in groups:
        for code in (0,1):pids.extend(z['pid'][mask&P.group_mask(z['pid'],group)&(z['state']==code)].unique()[:2].tolist())
    ids=torch.where(mask&torch.isin(z['pid'],z['pid'].new_tensor(pids)))[0]
    if not len(ids):raise RuntimeError('Empty smoke target')
    zz={k:v[ids] for k,v in z.items()};meta=dict(pack['meta'])
    meta['intervals']=int(math.ceil(float(zz['end'].max())-1e-8));meta['diagnostic_subset_only']=True
    save(target/name,dict(meta=meta,z=zz))


def seal(out,profile='baseline_retrain_all'):
    from cf_label_diagnostics import directory
    out=Path(out);config=defaults(profile)
    if (out/'protocol.json').exists():
        p=verify(out)
        if p['config']!=config:raise RuntimeError('Run profile changed; use a separate output directory')
        return p
    references=[];targets=[];assets={};target_assets={}
    for q in config['cases']:
        name=f'cf_2K_Q{q}';source=BASE/'outputs/CFSimEachPINN_20261109/prepared'/name
        folder=directory(name)
        for path in [source/'probability_observations.pt',*[folder/f for f in ('latent_labels.pt','full_resolved_context.pt','training_reference_fields.pt','label_provenance.json','force_context_audit.json')]]:
            assets[path.relative_to(BASE).as_posix()]=sha(path)
        destination=out/'datasets'/name
        for file in ('inversion_observations.pt','heldout_observations.pt','reference_evaluation.npz'):
            if profile=='smoke' and file.endswith('.pt'):_smoke_pack(source,destination,file)
            else:_copy(source/file,destination/file)
            target_assets[(destination/file).relative_to(out).as_posix()]=sha(destination/file)
        references.append(dict(name=name,kind='cf_simulation',adapter='channel',data_role=config['data_role']))
        targets.append(dict(name=name,kind='cf_simulation',adapter='channel',reference=name,path=destination.relative_to(out).as_posix()))
    sources={path.relative_to(BASE).as_posix():sha(path) for path in HERE.glob('*.py')}
    for path in B.source_paths():sources[path.relative_to(BASE).as_posix()]=sha(path)
    for path in P.source_paths()+[F.BASELINE,F.OPERATOR]:sources[path.relative_to(BASE).as_posix()]=sha(path)
    from method_contract import verify_spec
    spec=verify_spec();sources.update(spec['sha256'])
    p=dict(config=config,settings=T.settings(profile),references=references,targets=targets,assets=assets,
        target_assets=target_assets,sources=sources,frozen=True)
    p['fingerprint']=fingerprint(p);dump(out/'protocol.json',p);return p


def verify(out):
    p=read(Path(out)/'protocol.json')
    if p['fingerprint']!=fingerprint({k:v for k,v in p.items() if k!='fingerprint'}):raise RuntimeError('Protocol fingerprint mismatch')
    for mapping,root in ((p['assets'],BASE),(p['sources'],BASE),(p['target_assets'],Path(out))):
        for name,digest in mapping.items():
            if sha(contained(root,name))!=digest:raise RuntimeError('Sealed input or code changed: '+name)
    return p


def inherit_stage(out,case,seed,stage,p,parent=None):
    folder=Path(out)/'probability'/case/f'seed_{seed}'/stage
    net,_,_,digest=B.load(seed)
    if parent is not None:net.load_state_dict(load(parent)['net_state'])
    record=dict(case=case,seed=seed,training_seed=B.MODELS[seed][0],stage=stage,epoch=0,net_state=state(net),closure_state={},
        inherited=True,inheritance='periodic selected_joint lineage; no CF adaptation claimed for this stage',
        dependencies=dict(periodic=digest,parent_sha256=None if parent is None else sha(parent)),
        protocol_fingerprint=p['fingerprint'],profile=p['config']['profile'],normalization_inherited=True)
    path=folder/'selected.pt'
    if path.exists():
        old=load(path)
        if old['protocol_fingerprint']!=p['fingerprint'] or old['dependencies']!=record['dependencies']:raise RuntimeError('Inherited stage changed')
        return path
    save(path,record);dump(folder/'status.json',dict(complete=True,inherited=True,stage=stage,
        protocol_fingerprint=p['fingerprint'],selected_sha256=sha(path)))
    return path


def train_probabilities(out,device='cuda',max_new_epochs=None):
    p=verify(out)
    if (Path(out)/'probability_frozen.json').exists():return freeze_probabilities(out)
    for row in p['references']:
        if row['adapter']=='channel':
            import channel_joint as J
        elif row['adapter']=='periodic':
            import periodic_joint as J
        else:raise ValueError('Unimplemented geometry adapter: '+row['adapter'])
        case=J.load_case(row['name'],device)
        for seed in p['config']['field_seeds']:
            parent=None;closure=None
            for stage in T.STAGES:
                if p['config']['stages'][stage]=='inherit':path=inherit_stage(out,row['name'],seed,stage,p,parent)
                else:path=T.run_stage(out,case,J,seed,stage,p,parent,closure,device,max_new_epochs)
                status=read(path.parent/'status.json')
                if not status['complete']:return dict(complete=False,case=row['name'],seed=seed,stage=stage)
                parent=path
                if stage=='closure':closure=path
        del case
    return freeze_probabilities(out)


def freeze_probabilities(out):
    p=verify(out);models=[]
    for row in p['references']:
        for seed in p['config']['field_seeds']:
            stages=[];previous=None
            for stage in T.STAGES:
                folder=Path(out)/'probability'/row['name']/f'seed_{seed}'/stage
                path=folder/'selected.pt';ck=load(path);status=read(folder/'status.json')
                if not status['complete'] or status['protocol_fingerprint']!=p['fingerprint'] or status['selected_sha256']!=sha(path):
                    raise RuntimeError('All probability stages must complete and match their selected hashes')
                if ck['protocol_fingerprint']!=p['fingerprint'] or ck['dependencies']['periodic']!=B.record(seed)['sha256']:
                    raise RuntimeError('Probability parent/protocol mismatch')
                if previous is not None and ck['dependencies'].get('parent_sha256')!=sha(previous):raise RuntimeError('Stage parent changed')
                if not ck.get('inherited'):
                    last=load(folder/'last.pt');best=last['best'];selected=last['trace'][0]
                    for item in last['trace'][1:]:
                        if item['score']<selected['score']-p['settings'][stage]['selection_min_delta'] and item.get('stability',{} ) is not None and item['stability']['passed']:
                            selected=item
                    if selected['epoch']!=ck['epoch'] or best['epoch']!=ck['epoch']:raise RuntimeError('Selection trace differs')
                    for key in ('net_state','closure_state'):
                        if any(not torch.equal(v,best[key][k]) for k,v in ck[key].items()):raise RuntimeError('Selected tensors differ from completed run')
                if any(not bool(torch.isfinite(v).all()) for v in ck['net_state'].values()):raise RuntimeError('Nonfinite frozen model')
                stages.append(dict(stage=stage,mode='inherit' if ck.get('inherited') else 'retrain',
                    selected_epoch=ck['epoch'],path=path.relative_to(out).as_posix(),sha256=sha(path)))
                previous=path
            models.append(dict(case=row['name'],seed=seed,training_seed=B.MODELS[seed][0],
                path=previous.relative_to(out).as_posix(),sha256=sha(previous),parent_sha256=B.record(seed)['sha256'],stages=stages))
    gate=dict(complete=True,protocol_fingerprint=p['fingerprint'],models=models,scope='each',
        data_role=p['config']['data_role'],independent_field_generalization=False,
        diagnostic_only=p['config']['diagnostic_only'],evaluation_truth_used_for_selection=False)
    path=Path(out)/'probability_frozen.json'
    if path.exists() and read(path)!=gate:raise RuntimeError('Frozen probability gate changed')
    dump(path,gate);return gate


def frozen(out,case,seed,device):
    p=read(Path(out)/'protocol.json');gate=read(Path(out)/'probability_frozen.json')
    if not gate['complete'] or gate['protocol_fingerprint']!=p['fingerprint']:raise RuntimeError('Probability gate mismatch')
    row=next(r for r in gate['models'] if r['case']==case and r['seed']==seed);path=contained(out,row['path'])
    if sha(path)!=row['sha256']:raise RuntimeError('Frozen probability checkpoint changed')
    net,cal,force,parent=B.load(seed,device);ck=load(path)
    if parent!=row['parent_sha256'] or ck['protocol_fingerprint']!=p['fingerprint']:raise RuntimeError('Frozen probability lineage mismatch')
    net.load_state_dict(ck['net_state']);net.freeze()
    return net,cal,force,row['sha256']
