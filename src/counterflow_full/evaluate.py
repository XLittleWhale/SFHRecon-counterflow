"""Truth is opened only after final checkpoint selection; no feedback to fitting."""
import json
import math
import numpy as np
import torch
from counterflow.io import load,save,dump
from counterflow.dynamics import rollout
from counterflow.vtk import vtk_series
from .core import jets,grid,operator,distribution,scores,coverage,subset,split
from .train import model


def metrics(v,ref):
    err=(v-ref).square().sum(-1)
    e=err.mean((0,2,3)).sqrt()*1000
    r=ref-ref.mean((2,3),keepdim=True)
    denom=r.square().sum(-1).mean((0,2,3)).sqrt().clamp_min(1e-12)
    return dict(window_vector_RMS_mm_s=e.tolist(),initial_vector_RMS_mm_s=(err[0].mean((1,2)).sqrt()*1000).tolist(),
                relative_fluctuation_error=(e/1000/denom).tolist(),
                by_time_RMS_mm_s=(err.mean((2,3)).sqrt()*1000).tolist())


@torch.no_grad()
def observation_report(a,c,z,net,cal,force,cfg,device,degree):
    report={};p=cfg['inversion']
    for kind,s in [('free',0),('cap',1)]:
        values=[];radii=[];errors=[];pids=[]
        for ix in torch.where(z['state']==s)[0].split(p['batch']):
            b=subset(z,ix,device);v,g,ac=jets(a,b,c,cfg['K'])
            pred,cov,lw=distribution(c,b,v,g,ac,kind,net,cal,force,degree)
            values.append(scores(pred,cov,lw,b['target']).cpu());radii.append(coverage(pred,cov,lw,b['target']).cpu())
            mean=(pred*lw.exp()[...,None]).sum((1,2))
            errors.append((mean-b['target']).square().sum(-1).cpu());pids.append(b['pid'].cpu())
        val=torch.cat(values);rad=torch.cat(radii);err=torch.cat(errors);pid=torch.cat(pids)
        groups={}
        for group in ('train','validation','test'):
            ids=pid[split(pid,group)].unique();particle=[]
            for ident in ids:
                m=pid==ident
                particle.append([float(val[m].mean()),float(err[m].mean()),float((rad[m]<=-2*math.log(.1)).double().mean())])
            if not particle:continue
            x=np.asarray(particle);rng=np.random.default_rng(261026)
            boot=x[rng.integers(len(x),size=(1000,len(x))),0].mean(1)
            groups[group]=dict(particles=len(ids),intervals=int(split(pid,group).sum()),NLL=float(x[:,0].mean()),
                NLL_PID_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                increment_vector_RMS_um=float(np.sqrt(x[:,1].mean())*c['physics']['vn_mean_m_s']*.001*1e6),
                coverage90=float(x[:,2].mean()),particle_ids=ids.tolist(),particle_NLL=x[:,0].tolist())
        report[kind]=groups
    return report


def logL_range(op,q):
    values=[]
    for qq in q:
        ft=torch.zeros(op.n,op.n,device=q.device,dtype=torch.complex128)
        z=.5*(qq[1::2]-1j*qq[2::2]);ft[op.y,op.x]=z;ft[op.my,op.mx]=z.conj();ft[0,0]=qq[0]
        values.append(op.inv(ft))
    v=torch.stack(values)
    return dict(saved_grid_min=float(v.min()),saved_grid_max=float(v.max()),
        saved_grid_clipped_fraction=float(((v<=-12)|(v>=8)).double().mean()),note='saved RK endpoints, not every internal stage')


@torch.no_grad()
def main(package,run,cfg,device,smoke=False):
    dest=run/'Results';dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():return
    d=load(run/'prepared'/cfg['test']/'observations.pt');c,z,T=d['c'],d['z'],d['intervals']
    ref=load(run/'prepared'/cfg['test']/'reference.pt');a_ref=ref['a'].to(device)
    selected=load(run/'inverse/selected.pt');theta=selected['theta'].to(device)
    op=operator(c,load(package/'operator.pt'),cfg,device);p=cfg['inversion']
    a,q=rollout(op,theta,T,p['substeps'],False)
    ar,qr=rollout(op,a_ref[0],T,p['substeps'],False)
    ab,qb=rollout(op,torch.zeros_like(theta),T,p['substeps'],False)
    fine,_=rollout(op,theta,T,2*p['substeps'],False)
    coefficients=dict(reference=a_ref,reference_initial_PDE=ar,background=ab,inverse=a)
    fields={name:grid(av,c,cfg).cpu() for name,av in coefficients.items()}
    fields['coarse_unprojected']=ref['grid']
    report=dict(field_metrics={name:metrics(v,fields['reference']) for name,v in fields.items()},
        against_unprojected={name:metrics(v,fields['coarse_unprojected']) for name,v in fields.items() if name!='coarse_unprojected'},
        timestep_refinement=metrics(grid(fine,c,cfg).cpu(),fields['inverse']),
        logL=dict(inverse=logL_range(op,q),reference_initial_PDE=logL_range(op,qr)),
        start_ms=cfg['start'],end_ms=cfg['start']+T,intervals=T,N=cfg['N'],K=cfg['K'],M=op.n,
        selected_control_order=selected['K'],selected_restart=selected['restart'],prior_factor=selected['factor'],
        counts=json.loads((package/'data'/cfg['test']/'audit.json').read_text()),
        smoke=smoke,completed_full_window=not smoke and T==95,
        cautions=['Particle/bootstrap uncertainty is conditional on this one flow realization.',
                  'Reference-initial replay is a model-mismatch control, NOT a lower bound on all possible inverse errors.',
                  'Reference trajectory need not lie in the strong-constraint PDE solution manifold.',
                  'One-ms factors form a composite likelihood; temporal correlations are not fully modeled.',
                  '4mm/s has been inspected in earlier work; no fresh-blind-test claim.'])
    net=model(run,device);cal=json.loads((run/'calibration.json').read_text());force=json.loads((run/'force.json').read_text())
    report['observations']={}
    for name,av in coefficients.items():
        print('Final held-out evaluation:',name,flush=True)
        report['observations'][name]=observation_report(av,c,z,net,cal,force,cfg,device,p['eval_degree'])
    report['quadrature_check']=observation_report(a,c,z,net,cal,force,cfg,device,p['degree'])
    report['rays']=[]
    for lam in ([0.,1.] if smoke else [0.,.25,.5,.75,1.]):
        aa,_=rollout(op,(1-lam)*a_ref[0]+lam*theta,T,p['substeps'],False)
        report['rays'].append(dict(lam=lam,observations=observation_report(aa,c,z,net,cal,force,cfg,device,p['eval_degree'])))
    report['acceptance']={
        'beats_uniform_both_fluids':all(x<y for x,y in zip(report['field_metrics']['inverse']['window_vector_RMS_mm_s'],report['field_metrics']['background']['window_vector_RMS_mm_s'])),
        'no_saved_L_clipping':report['logL']['inverse']['saved_grid_clipped_fraction']==0,
        'integration_refinement_below_1pct_U':max(report['timestep_refinement']['window_vector_RMS_mm_s'])<.01*c['physics']['vn_mean_m_s']*1000,
        'selected_without_target_truth':True}
    n=cfg['N'];xy=(np.arange(n)+.5)*c['physics']['box_m']/n;tt=(cfg['start']+np.arange(T+1))*.001
    for name,v in fields.items():vtk_series(dest/'fields'/name,v.numpy(),xy,xy,tt)
    save(dest/'fields.pt',dict(fields=fields,coefficients={k:v.cpu() for k,v in coefficients.items()},times_s=tt))
    dump(dest/'report.json',report);plots(dest,fields,report,c,tt)
    dump(dest/'complete.json',dict(full_run=not smoke,selected_restart=selected['restart'],acceptance=report['acceptance']))


def plots(dest,fields,report,c,times):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':8,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'legend.fontsize':7,'axes.titlesize':9,'axes.linewidth':.65,'savefig.bbox':'tight','pdf.fonttype':42,'ps.fonttype':42})
    teal='#007F86';orange='#D77632';purple='#8056A0';gray='#6D7781';colors=[teal,orange,purple,gray]
    captions={}
    def finish(fig,name,caption):
        for ext in ('png','pdf','svg'):fig.savefig(dest/f'{name}.{ext}',dpi=250)
        plt.close(fig);captions[name]=caption
    def label(ax,i):ax.text(.01,1.02,f'({chr(97+i)})',transform=ax.transAxes,ha='left',va='bottom')
    extent=[0,c['physics']['box_m']*1000,0,c['physics']['box_m']*1000]
    for f,fluid in enumerate(('vn','vs')):
        for time_index,suffix in [(0,'initial'),(len(times)//2,'middle'),(-1,'final')]:
            fig,axs=plt.subplots(2,3,figsize=(7.1,4.5),layout='constrained')
            for j,comp in enumerate(('x','y')):
                r=fields['reference'][time_index,f,...,j].numpy()*1000;v=fields['inverse'][time_index,f,...,j].numpy()*1000
                lo=min(r.min(),v.min());hi=max(r.max(),v.max());e=v-r;lim=max(abs(e).max(),1e-8)
                for k,(value,cm,limits) in enumerate([(r,'viridis',(lo,hi)),(v,'viridis',(lo,hi)),(e,'RdBu_r',(-lim,lim))]):
                    ax=axs[j,k];im=ax.imshow(value,origin='lower',extent=extent,cmap=cm,vmin=limits[0],vmax=limits[1])
                    fig.colorbar(im,ax=ax,shrink=.8,label=fr'$v_{{{fluid[-1]},{comp}}}$ (mm/s)' if k<2 else 'Error (mm/s)')
                    ax.set(xlabel='$x$ (mm)',ylabel='$y$ (mm)');label(ax,j*3+k)
            finish(fig,f'F_{fluid}_{suffix}',f'Rows x/y; columns modal reference, inverse, signed error; t={times[time_index]*1000:g} ms. Shared reference/inverse scales within each row.')
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.6),layout='constrained')
    for j,ax in enumerate(axs):
        for color,name in zip(colors,['inverse','reference_initial_PDE','background','coarse_unprojected']):
            yy=np.asarray(report['field_metrics'][name]['by_time_RMS_mm_s'])[:,j]
            ax.plot(times*1000,yy,color=color,label=name.replace('_',' '))
        ax.set(xlabel='$t$ (ms)',ylabel=fr'$v_{{{("n","s")[j]}}}$ vector RMS error (mm/s)');label(ax,j)
    axs[0].legend();finish(fig,'F_time_error','Errors against identical N40 K12 modal reference. Coarse-unprojected curve quantifies representation discrepancy, not inversion error.')
    fig,axs=plt.subplots(2,2,figsize=(7.1,4.3),layout='constrained')
    for j,kind in enumerate(('free','cap')):
        for ii,key in enumerate(('NLL','coverage90')):
            ax=axs[ii,j]
            for offset,(name,color) in enumerate(zip(['reference','reference_initial_PDE','background','inverse'],colors)):
                vals=[report['observations'][name][kind][g][key] for g in ('train','validation','test')]
                ax.plot(np.arange(3)+offset*.035,vals,'o-',color=color,label=name.replace('_',' '))
            ax.set(xticks=range(3),xticklabels=['Fit','Selection','Test'],ylabel='90% coverage' if key=='coverage90' else 'NLL');label(ax,ii*2+j)
            ax.text(.98,.08,{'free':'Free','cap':'Captured'}[kind],transform=ax.transAxes,ha='right',va='bottom')
            if key=='coverage90':ax.axhline(.9,color=gray,lw=.6,ls='--');ax.set_ylim(-.02,1.05)
    axs[0,0].legend();finish(fig,'F_probability','Statewise PID-balanced scores and Rosenblatt 90% coverage. Fitting, selection and test particles are disjoint.')
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.6),layout='constrained')
    for j,kind in enumerate(('free','cap')):
        for group,style in [('train','--'),('test','-')]:
            yy=[r['observations'][kind][group]['NLL'] for r in report['rays']]
            axs[j].plot([r['lam'] for r in report['rays']],np.array(yy)-yy[0],style+'o',color=colors[int(group=='test')],label=group)
        axs[j].set(xlabel=r'Reference initial $\rightarrow$ fitted initial, $\lambda$',ylabel=r'$\Delta$NLL');label(axs[j],j);axs[j].legend()
    finish(fig,'F_likelihood_ray','Diagnostic-only initial-state line with full nonlinear PDE rollouts. No line value feeds checkpoint selection; reference initialization need not minimize misspecified likelihood.')
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.6),layout='constrained')
    for j in range(2):
        r=fields['reference'][0,j].numpy().ravel()*1000;v=fields['inverse'][0,j].numpy().ravel()*1000
        axs[j].hexbin(r,v,gridsize=35,mincnt=1,cmap='viridis');lo=min(r.min(),v.min());hi=max(r.max(),v.max());axs[j].plot([lo,hi],[lo,hi],color=orange,lw=.8)
        axs[j].set(xlabel='Reference component (mm/s)',ylabel='Reconstructed component (mm/s)');label(axs[j],j)
    finish(fig,'F_initial_scatter','Initial-field component comparison, vn left and vs right. Color encodes sample count, not probability density.')
    from .core import modes
    coeff=load(dest/'fields.pt')['coefficients'];xy,_=modes(12);rings=np.max(abs(xy),1).repeat(2)
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.6),layout='constrained')
    for j,ax in enumerate(axs):
        for name,color in [('reference',teal),('inverse',orange)]:
            a=coeff[name][0,j].numpy();e=[.5*c['physics']['vn_mean_m_s']**2*np.sum(a[rings==k]**2)*1e6 for k in range(1,13)]
            ax.semilogy(range(1,13),np.maximum(e,1e-12),'o-',color=color,label=name)
        ax.set(xlabel=r'Square-shell order $K$',ylabel=r'Velocity power (mm$^2$/s$^2$)');label(ax,j);ax.legend()
    finish(fig,'F_initial_spectrum','Initial fluctuation energy in square Fourier shells; prescribed mean excluded.')
    trainroot=dest.parent/'training';info=json.loads((trainroot/'complete.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.6),layout='constrained')
    for i,row in enumerate(info['seeds']):
        tr=json.loads((trainroot/f"seed_{row['seed']}"/'trace.json').read_text())
        for j,key in enumerate(('train','validation')):
            axs[j].plot([r['epoch'] for r in tr],[r[key] for r in tr],color=colors[i%4],label=f"Seed {row['seed']}")
            axs[j].set(xlabel='Epoch',ylabel=f'{key.capitalize()} objective');label(axs[j],j);axs[j].legend()
    finish(fig,'F_training','All-interval network training and validation objective; each curve is a fresh random initialization. Validation selects the checkpoint.')
    chosen=json.loads((dest.parent/'inverse/complete.json').read_text())['selected_restart']
    trace=json.loads((dest.parent/'inverse'/f'restart_{chosen}'/'trace.json').read_text())
    fig,ax=plt.subplots(figsize=(3.45,2.6),layout='constrained')
    for key,color in [('train',teal),('validation',orange)]:ax.plot(range(len(trace)),[r[key] for r in trace],color=color,label=key)
    for i,row in enumerate(trace):
        if i==0 or row['K']!=trace[i-1]['K']:ax.axvline(i,color=gray,lw=.6,ls=':');ax.text(i,.98,f"K{row['K']}",transform=ax.get_xaxis_transform(),va='top',fontsize=7)
    ax.set(xlabel='Saved validation checkpoint index',ylabel='Composite NLL');ax.legend(loc='lower left')
    finish(fig,'F_optimization','Selected restart: full-batch fitting and disjoint-PID selection scores through all initial-control orders; not truth-based checkpoint selection.')
    dump(dest/'figure_manifest.json',dict(captions=captions,formats=['png','pdf','svg'],titles_embedded=False))
