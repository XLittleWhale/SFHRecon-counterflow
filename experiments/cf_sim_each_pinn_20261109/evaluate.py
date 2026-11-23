"""Held-out likelihood and reference-field errors, after all selections freeze."""
from common import *
import csv
import numpy as np
import field_adapter as F
import reconstruct as R

@torch.no_grad()
def values(c,a,points,gradient=False):
    b=F.geometry(c).evaluate(points)
    v=b['v0']+torch.einsum('tfm,pfmj->tpfj',a,b['V'])
    if not gradient: return v
    g=b['g0']+torch.einsum('tfm,pfmij->tpfij',a,b['G'])
    return v,g

@torch.no_grad()
def field_metrics(c,field,reference):
    times=torch.as_tensor(reference['time_s'],dtype=torch.float64)
    a=field(times)
    yy,xx=np.meshgrid(reference['y_m'],reference['x_m'],indexing='ij')
    points=torch.tensor(np.stack((xx,yy),-1).reshape(-1,2))
    prediction=values(c,a,points).reshape(len(times),*xx.shape,2,2).numpy()
    cold=values(c,field.cold[None],points)[0].reshape(*xx.shape,2,2).numpy()
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
    y=torch.linspace(0,.008,65)
    walls=torch.cat((torch.stack((0*y,y),-1),torch.stack((0*y+.016,y),-1)))
    wall=values(c,a,walls)
    x=torch.linspace(0,.016,65)
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
    fig.suptitle(f'{case}, seed {seed}, t={reference["time_s"][frame]:.4f} s; selected 600-step budget')
    fig.savefig(path,dpi=170)
    plt.close(fig)

@torch.no_grad()
def main():
    p,gate=evaluate_gate()
    out=OUT/'evaluation';out.mkdir(exist_ok=True)
    rows=[]
    for record in gate['runs']:
        case,seed=record['case'],record['seed']
        folder=out/case/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
        meta=load(prepared(case)/'inversion_observations.pt')['meta']
        c,op,cold,prior=F.prepare_physics(meta['physics_config'],meta['intervals'])
        field=F.make_field(c,cold,seed)
        net,cal,force,digest=probability(case,seed)
        heldout=load(prepared(case)/'heldout_observations.pt')['z']
        with np.load(prepared(case)/'reference_evaluation.npz') as stream:
            reference={k:stream[k].copy() for k in stream.files}
        checkpoints={'background':record['background']}
        for budget,arms in record['budgets'].items():
            for arm,ck in arms.items(): checkpoints[f'{budget}_{arm}']=ck
        cache={}
        for label,ckrecord in checkpoints.items():
            ck=load(ckrecord['path']);field.load_state_dict(ck['field_state'])
            identity=ck['step']
            if identity not in cache:
                nll=R.score(c,heldout,field,net,cal,force,'H',p['field']['validation_degree'],metrics=True)
                errors,prediction=field_metrics(c,field,reference)
                cache[identity]=(nll,errors,prediction)
            nll,errors,prediction=cache[identity]
            report=dict(case=case,seed=seed,label=label,selected_step=ck['step'],heldout=nll,
                field=errors,protocol_fingerprint=p['fingerprint'],checkpoint_sha256=ckrecord['sha256'],
                probability_sha256=digest,reference_used_for_selection=False)
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
            if label=='600_primary': plot(case,seed,reference,prediction,errors,folder/'reconstruction.png')
            print('EVALUATED',case,seed,label,flush=True)
        del prior,op,field,net
    with (out/'summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    aggregate={}
    for case in CASES:
        use=[r for r in rows if r['case']==case and r['label']=='600_primary']
        aggregate[case]={k:dict(mean=float(np.mean([r[k] for r in use])),
            std=float(np.std([r[k] for r in use],ddof=1))) for k in use[0] if k not in ('case','seed','label','step')}
    dump(out/'summary.json',dict(complete=True,runs=rows,primary_600_across_seeds=aggregate,
        protocol_fingerprint=p['fingerprint'],finished_utc=utc()))

if __name__=='__main__':
    setup();main()
