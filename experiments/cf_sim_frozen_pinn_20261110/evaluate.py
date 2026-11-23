"""Held-out likelihood and reference-field errors, after all selections freeze."""
from common import *
import argparse
import csv
import numpy as np
import field_adapter as F
import reconstruct as R

SCORING_MODEL = 'Nov5 frozen joint probability; no each-case probability fitting'

@torch.no_grad()
def values(c,a,points,gradient=False):
    points=points.to(a)
    b=F.geometry(c, device=a.device, dtype=a.dtype).evaluate(points)
    v=b['v0']+torch.einsum('tfm,pfmj->tpfj',a,b['V'])
    if not gradient: return v
    g=b['g0']+torch.einsum('tfm,pfmij->tpfij',a,b['G'])
    return v,g

@torch.no_grad()
def field_metrics(c,field,reference):
    times=torch.as_tensor(reference['time_s'],dtype=field.cold.dtype,device=field.device)
    a=field(times)
    yy,xx=np.meshgrid(reference['y_m'],reference['x_m'],indexing='ij')
    points=torch.as_tensor(np.stack((xx,yy),-1).reshape(-1,2),device=field.device,dtype=field.cold.dtype)
    prediction=values(c,a,points).reshape(len(times),*xx.shape,2,2).cpu().numpy()
    cold=values(c,field.cold[None],points)[0].reshape(*xx.shape,2,2).cpu().numpy()
    weights=np.ones(xx.shape);weights[:,(0,-1)]=.5;weights/=weights.sum()
    result={'reference_grid':dict(x_nodes=len(reference['x_m']),y_nodes=len(reference['y_m']),
        frames=len(times),interpretation='coarse nodal grid sampling error, not full solver-grid error')}
    for f,name,key in ((0,'normal','vn'),(1,'superfluid','vs')):
        truth=reference[key].astype(np.float64)
        pred=prediction[...,f,:]
        error2=((pred-truth)**2).sum(-1)
        rmse=np.sqrt(np.einsum('tyx,yx->t',error2,weights))
        truth_rms=np.sqrt(np.einsum('tyx,yx->t',(truth**2).sum(-1),weights))
        background_error=np.sqrt(np.einsum('tyx,yx->t',((cold[...,f,:]-truth)**2).sum(-1),weights))
        result[name]=dict(vector_rmse_m_s=float(np.sqrt(np.mean(rmse**2))),
            relative_L2=float(np.sqrt(np.mean(rmse**2)/np.mean(truth_rms**2))),
            background_rmse_m_s=float(np.sqrt(np.mean(background_error**2))),
            rmse_over_background=float(np.sqrt(np.mean(rmse**2)/np.mean(background_error**2))),
            per_frame_rmse_m_s=rmse.tolist())
    y=torch.linspace(0,.008,65,device=field.device,dtype=field.cold.dtype)
    walls=torch.cat((torch.stack((0*y,y),-1),torch.stack((0*y+.016,y),-1)))
    wall=values(c,a,walls)
    x=torch.linspace(0,.016,65,device=field.device,dtype=field.cold.dtype)
    ends=torch.cat((torch.stack((x,0*x),-1),torch.stack((x,0*x+.008),-1)))
    end=values(c,a,ends)
    _,g=values(c,a[::max(1,len(a)//5)],points,True)
    result['boundary']=dict(normal_wall_max_m_s=float(wall[:,:,0].abs().max()),
        super_normal_wall_max_m_s=float(wall[:,:,1,0].abs().max()),
        periodic_velocity_max_m_s=float((end[:,:65]-end[:,65:]).abs().max()),
        divergence_max_s_inv=float(g.diagonal(dim1=-2,dim2=-1).sum(-1).abs().max()))
    return result,prediction

def plot(case,seed,reference,prediction,metrics,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,6.5),constrained_layout=True)
    frame=len(reference['time_s'])//2
    for f,name,key in ((0,'Normal','vn'),(1,'Superfluid','vs')):
        ref=reference[key][frame,...,1]*1e3
        rec=prediction[frame,...,f,1]*1e3
        limit=max(abs(ref).max(),abs(rec).max())
        for j,(data,title) in enumerate(((ref,'Reference'),(rec,'PINN'),(rec-ref,'Error'))):
            lim=limit if j<2 else max(abs(data).max(),1e-12)
            im=axes[f,j].pcolormesh(reference['x_m']*1e3,reference['y_m']*1e3,data,
                shading='auto',cmap='RdBu_r',vmin=-lim,vmax=lim)
            axes[f,j].set_xlim(0,16);axes[f,j].set_ylim(0,8);axes[f,j].set_aspect('equal')
            axes[f,j].set_title(f'{name}: {title} $u_y$')
            axes[f,j].set_xlabel('x (mm)');axes[f,j].set_ylabel('y (mm)')
            fig.colorbar(im,ax=axes[f,j],label='mm/s',shrink=.8)
    fig.suptitle(f'{case}, seed {seed}, t={reference["time_s"][frame]:.4f} s; selected {max(BUDGETS)}-step budget')
    fig.savefig(path,dpi=170)
    plt.close(fig)

def try_plot(case,seed,reference,prediction,metrics,path):
    """Plots are optional artifacts; their dependencies cannot lose numeric results."""
    try:
        plot(case,seed,reference,prediction,metrics,path)
        status=dict(case=case,seed=seed,complete=True,pending=False,path=path.name)
    except Exception as error:
        status=dict(case=case,seed=seed,complete=False,pending=True,error=repr(error),
                    recovery='Repair plotting dependencies, then run evaluate.py --plots-only')
        print('PLOT_PENDING',case,seed,repr(error),flush=True)
    dump(path.parent/'plot_status.json',status)
    return status

def comparison_150(rows,protocol,baseline_dir=None):
    """Compare common field-error metrics at a matched 150-step budget.

    The two runs use different probability laws, so their own-model NLLs are
    deliberately excluded from this comparison table.
    """
    root=Path(baseline_dir) if baseline_dir is not None else COMPARISON
    paths={name:root/rel for name,rel in (
        ('summary','evaluation/summary.json'),('analysis','analysis/result_summary.json'),
        ('protocol','protocol.json'))}
    report=dict(budget=150,current_scoring_model=SCORING_MODEL,
        baseline_scoring_model='Each-case probability trained on the same realization, then frozen',
        cross_model_NLL_comparison=False,
        interpretation='Compare velocity RMSE and relative L2 only; each NLL uses its own probability model.',
        rows=[])
    if 150 not in protocol.get('budgets',BUDGETS):
        return dict(report,complete=False,reason='This run has no 150-step budget')
    missing=[name for name,path in paths.items() if not path.is_file()]
    if missing:
        return dict(report,complete=False,reason='Bundled comparison files are missing',missing=missing)
    baseline={name:read(path) for name,path in paths.items()}
    bp,bs=baseline['protocol'],baseline['summary']
    if (bs.get('selected_budget') != 150 or not bs.get('complete') or
            bs.get('protocol_fingerprint') != bp.get('fingerprint') or
            baseline['analysis'].get('protocol_fingerprint') != bp.get('fingerprint')):
        return dict(report,complete=False,reason='Baseline is not a consistent completed 150-step evaluation')
    report['baseline_file_sha256']={name:sha(path) for name,path in paths.items()}
    report['protocol_fingerprints']=dict(current=protocol['fingerprint'],baseline=bp['fingerprint'])
    field_keys=('lr','gradient_clip','train_degree','validation_degree','validate_every',
                'guard_nll_per_ms','selection_delta_nll_per_ms','likelihood_weight',
                'free_batch','captured_batch_pids','architecture','initial','PDE','time','acceleration')
    differences={key:dict(current=protocol.get('field',{}).get(key),baseline=bp.get('field',{}).get(key))
        for key in field_keys if protocol.get('field',{}).get(key) != bp.get('field',{}).get(key)}
    if differences:
        return dict(report,complete=False,reason='Field protocols differ',field_differences=differences)
    def input_hash(p,case,name):
        found=[value for key,value in p.get('inputs',{}).items()
               if key.replace('\\','/').endswith('/'+case+'/'+name)]
        return found[0] if len(found)==1 else None
    source={(r['case'],r['seed'],r['label']):r for r in bs['runs']}
    unmatched=[]
    for row in rows:
        if row['label'] not in ('150_primary','150_terminal'):
            continue
        key=(row['case'],row['seed'],row['label'])
        old=source.get(key)
        files=('data_audit.json','inversion_observations.pt','heldout_observations.pt','reference_evaluation.npz')
        same_inputs=all(input_hash(protocol,row['case'],name) is not None and
                        input_hash(protocol,row['case'],name)==input_hash(bp,row['case'],name) for name in files)
        if old is None or not same_inputs:
            unmatched.append(dict(case=row['case'],seed=row['seed'],label=row['label'],
                                  reason='Missing paired baseline or changed prepared inputs'))
            continue
        entry=dict(case=row['case'],seed=row['seed'],label=row['label'],budget=150,
                   selected_step=row['step'],baseline_selected_step=old['step'],identical_prepared_inputs=True)
        for fluid in ('normal','superfluid'):
            rmse,relative=fluid+'_rmse_m_s',fluid+'_relative_L2'
            entry[fluid]=dict(frozen_probability_rmse_m_s=row[rmse],each_probability_rmse_m_s=old[rmse],
                rmse_ratio_frozen_over_each=row[rmse]/old[rmse] if old[rmse]>0 else None,
                rmse_change_percent=100*(row[rmse]/old[rmse]-1) if old[rmse]>0 else None,
                frozen_probability_relative_L2=row[relative],each_probability_relative_L2=old[relative])
        report['rows'].append(entry)
    return dict(report,complete=bool(report['rows']) and not unmatched,unmatched=unmatched)

def plots_only():
    """Regenerate PNGs from exported arrays; never reload or score field models."""
    p=verify_protocol()
    out=OUT/'evaluation'
    summary=read(out/'summary.json')
    if not summary.get('complete') or summary['protocol_fingerprint'] != p['fingerprint']:
        raise RuntimeError('A complete numeric evaluation under this protocol is required')
    statuses=[]
    label=f"{summary['selected_budget']}_primary"
    for row in summary['runs']:
        if row['label'] != label:
            continue
        case,seed=row['case'],row['seed']
        if case not in CASES or seed not in SEEDS:
            raise ValueError('Unexpected case/seed in evaluation summary')
        folder=out/case/f'seed_{seed}'
        with np.load(prepared(case)/'reference_evaluation.npz') as stream:
            reference={k:stream[k].copy() for k in stream.files}
        with np.load(folder/(label+'_fields.npz')) as stream:
            if not all(np.array_equal(stream[key],reference[key]) for key in ('time_s','x_m','y_m')):
                raise ValueError('Export/reference grids do not match')
            prediction=np.stack((stream['vn'],stream['vs']),axis=-2)
        metrics=read(folder/(label+'.json'))['field']
        statuses.append(try_plot(case,seed,reference,prediction,metrics,folder/'reconstruction.png'))
    summary.update(plots_complete=all(r['complete'] for r in statuses),plot_status=statuses,
                   plots_updated_utc=utc())
    dump(out/'summary.json',summary)
    return statuses

@torch.no_grad()
def main():
    p,gate=evaluate_gate()
    out=OUT/'evaluation';out.mkdir(exist_ok=True)
    rows=[]
    plot_status=[]
    for record in gate['runs']:
        case,seed=record['case'],record['seed']
        folder=out/case/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
        meta=load(prepared(case)/'inversion_observations.pt')['meta']
        c,op,cold,prior=F.prepare_physics(meta['physics_config'],meta['intervals'],device=device())
        field=F.make_field(c,cold,seed)
        net,cal,force,digest=probability(case,seed,device=device())
        heldout=move_tensors(load(prepared(case)/'heldout_observations.pt')['z'])
        with np.load(prepared(case)/'reference_evaluation.npz') as stream:
            reference={k:stream[k].copy() for k in stream.files}
        checkpoints={'background':record['background']}
        for budget,arms in record['budgets'].items():
            for arm,ck in arms.items(): checkpoints[f'{budget}_{arm}']=ck
        cache={}
        for label,ckrecord in checkpoints.items():
            ck=load(output_path(ckrecord['path']));field.load_state_dict(ck['field_state'])
            identity=ck['step']
            if identity not in cache:
                nll=R.score(c,heldout,field,net,cal,force,'H',p['field']['validation_degree'],metrics=True)
                errors,prediction=field_metrics(c,field,reference)
                cache[identity]=(nll,errors,prediction)
            nll,errors,prediction=cache[identity]
            report=dict(case=case,seed=seed,label=label,selected_step=ck['step'],heldout=nll,
                field=errors,protocol_fingerprint=p['fingerprint'],checkpoint_sha256=ckrecord['sha256'],
                probability_sha256=digest,reference_used_for_selection=False,
                scoring_model=SCORING_MODEL,probability_optimization_steps=0)
            dump(folder/(label+'.json'),report)
            np.savez_compressed(folder/(label+'_fields.npz'),time_s=reference['time_s'],
                x_m=reference['x_m'],y_m=reference['y_m'],vn=prediction[...,0,:].astype(np.float32),
                vs=prediction[...,1,:].astype(np.float32))
            rows.append(dict(case=case,seed=seed,label=label,step=ck['step'],H_NLL_ms=nll['total'],
                H_free_NLL_ms=nll['free'],H_cap_NLL_ms=nll['cap'],
                normal_rmse_m_s=errors['normal']['vector_rmse_m_s'],
                superfluid_rmse_m_s=errors['superfluid']['vector_rmse_m_s'],
                normal_relative_L2=errors['normal']['relative_L2'],
                superfluid_relative_L2=errors['superfluid']['relative_L2']))
            if label==f'{max(BUDGETS)}_primary':
                plot_status.append(try_plot(case,seed,reference,prediction,errors,folder/'reconstruction.png'))
            print('EVALUATED',case,seed,label,flush=True)
        del prior,op,field,net
    with (out/'summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    aggregate={}
    for case in CASES:
        use=[r for r in rows if r['case']==case and r['label']==f'{max(BUDGETS)}_primary']
        aggregate[case]={k:dict(mean=float(np.mean([r[k] for r in use])),
            std=float(np.std([r[k] for r in use],ddof=1)) if len(use)>1 else None,
            n=len(use)) for k in use[0] if k not in ('case','seed','label','step')}
    dump(out/'summary.json',dict(complete=True,runs=rows,primary_across_seeds=aggregate,
        selected_budget=max(BUDGETS),reduced_budget=True,original_nine_run_baseline_complete=False,
        numeric_evaluation_complete=True,plots_complete=all(r['complete'] for r in plot_status),
        plot_status=plot_status,scoring_model=SCORING_MODEL,probability_optimization_steps=0,
        protocol_fingerprint=p['fingerprint'],finished_utc=utc()))
    dump(out/'comparison_150.json',comparison_150(rows,p))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots-only',action='store_true',help='CPU-capable PNG regeneration from saved arrays; no scoring')
    args=parser.parse_args()
    setup()
    if args.plots_only: plots_only()
    else: main()
