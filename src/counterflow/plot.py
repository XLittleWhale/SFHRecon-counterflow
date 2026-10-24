"""The seven named Results groups plus one uniform-protocol audit page."""
from .common import *
from . import figure_style as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse
from scipy.stats import chi2
DATA=DEST/'data';C=st.C;AUDIT=[];PDF=None;R=None
COL={'reference':C['ink'],'projected':C['purple'],'reconstruction':C['teal'],'previous':C['orange'],'background':C['gray']}
NAME={'reference':'Reference','projected':'K12 projection','reconstruction':'Unified retrained','previous':'Previous free-corr.','background':'PDE background'}
CAPTIONS=[]

def data(name):
    with np.load(DATA/name) as z:return {k:z[k].copy() for k in z.files}

def save(fig,folder,name,caption=''):
    path=DEST/folder;path.mkdir(exist_ok=True);assert all(not a.get_title() for a in fig.axes)
    fig.canvas.draw();PDF.savefig(fig,facecolor='white')
    fig.savefig(path/f'{name}.pdf',facecolor='white',metadata={'Title':'','Creator':'counterflow'})
    fig.savefig(path/f'{name}.png',dpi=600,facecolor='white')
    AUDIT.append(dict(folder=folder,name=name,page=len(AUDIT)+1,width_mm=180,height_mm=fig.get_figheight()*25.4,
        axes=len(fig.axes),font=st.FONT_NAME+' / STIX',labels_pt=9,ticks_pt=8,panel_pt=10,no_titles=True,png_dpi=600))
    CAPTIONS.append(name+'\n'+caption);plt.close(fig)

def panels(axs):
    for i,ax in enumerate(axs.flat):st.panel(ax,i)

def probability():
    ref=data('periodic_4_reference_probability.npz');rec=data('periodic_4_reconstruction_probability.npz')
    fig,axs=plt.subplots(2,3,figsize=(st.WIDTH,132/25.4),layout='constrained')
    extent=st.paired_extent([ref,rec],0)
    st.density_panel(axs[0,0],ref,0,extent,True);st.density_panel(axs[0,1],rec,0,extent)
    for col,kind in enumerate(('cap','free')):
        ax=axs[1,col]
        for tag in ('reference','reconstruction'):
            d=R['periodic_4']['variants'][tag]['particles'][kind];p=ref['nominal'];ci=np.array(d['CI95'])
            ax.plot(p,d['coverage'],'o-',color=COL[tag],label=NAME[tag]);ax.fill_between(p,ci[0],ci[1],color=COL[tag],alpha=.1,linewidth=0)
        ax.plot([.45,1],[.45,1],'--',color=C['gray'],lw=.7);ax.set(xlim=(.45,1),ylim=(0,1.03),xticks=[.5,.7,.9],yticks=[0,.5,1],xlabel='Nominal coverage',ylabel='Empirical coverage');ax.legend(fontsize=7)
    for row,metric in enumerate(('vector_RMS_um','NLL')):
        ax=axs[row,2]
        for kind,col,mk,label in [('cap',C['teal'],'o','Captured'),('free',C['orange'],'s','Free')]:
            y=[]
            for case in CASES[:3]:
                q=R[case]['variants']['reference']['particles'][kind];y.append(q[metric]['mean'] if metric=='NLL' else q[metric])
            ax.plot([1.5,4,10],y,mk+'-',color=col,label=label)
        ax.set(xticks=[1.5,4,10],xlabel=r'$\overline{v}_{n,y}\ (\mathrm{mm\,s^{-1}})$',ylabel='NLL / step' if metric=='NLL' else r'Decoder RMS $(\mu\mathrm{m})$');ax.legend(fontsize=7)
    panels(axs);save(fig,'periodic','P1_observation_probability',
        '(a,b) Same held-out 4 mm/s increment: reference-conditioned and reconstructed-conditioned physical mixture, 50/80/95% HPD contours. (c,f) Three test conditions evaluated at reference fields. (d,e) Captured/free one-step Rosenblatt/Gaussian radial coverage (NOT HPD coverage), with physical-particle bootstrap CI. The 10 mm/s free test has only one physical particle; no population-calibration claim. NLL is for dimensionless displacements, not a dimensional density.')

def maps():
    z=data('periodic_4_fields.npz');z['v_K12']=z['reconstruction'];z['projected_K12']=z['projected']
    st.load=lambda _:z
    st.save=lambda fig,folder,name:save(fig,folder,name,'Final frame, 17 ms, 4 mm/s test case. Rows vn and vs; columns conservative 40-cell reference, K12 projection, uniform-version reconstruction, vector error. Shared velocity colour scales within each row; arrows show fluctuations. Reference and reconstruction have identical 40x40 sample coordinates; no field truth entered inverse fitting.')
    st.periodic_maps()

def scales():
    z=data('periodic_4_fields.npz');fig,axs=plt.subplots(2,2,figsize=(st.WIDTH,125/25.4),layout='constrained')
    for f in (0,1):
        ax=axs[0,f]
        for metric,col,mk,label in [('full',C['ink'],'o','Total'),('inverse',C['orange'],'s','Inference'),('truncation',C['teal'],'^','Truncation')]:
            ax.plot([1.5,4,10],[R[c]['variants']['reconstruction']['error_mm_s'][metric][f] for c in CASES[:3]],mk+'-',color=col,label=label)
        ax.set(xticks=[1.5,4,10],xlabel=r'$\overline{v}_{n,y}\ (\mathrm{mm\,s^{-1}})$',ylabel=r'$E_'+('n' if f==0 else 's')+r'\ (\mathrm{mm\,s^{-1}})$');ax.legend(fontsize=7)
        ax=axs[1,f]
        for key in ('reference','projected','previous','reconstruction'):
            E=st.spectrum(z[key]);k=np.arange(1,21);m=E[f,1:21]>1e-18;ax.loglog(k[m],E[f,1:21][m],color=COL[key],label=NAME[key])
        ax.set(xlabel=r'$|\mathbf{k}|$',ylabel=r'$\mathcal{E}_'+('n' if f==0 else 's')+r'\ (\mathrm{mm^2\,s^{-2}})$');ax.legend(fontsize=6.8)
    panels(axs);save(fig,'periodic','P3_modal_error_spectra','Top: whole-13-frame component RMS errors in the three unseen training conditions (historically reused development tests). Bottom: 4 mm/s fluctuation energy spectra with Parseval-checked shell summation. This is NOT a new K-sweep. Previous means the same free-correlated version for every historical comparison, never a captured-correlated substitute.')

def cf_structure():
    z=data('cf_261_fields.npz');m=readjson(DATA/'cf_261_metadata.json');fig,axs=plt.subplots(3,2,figsize=(st.WIDTH,164/25.4),layout='constrained')
    for j,kind in enumerate(('cap','free')):
        ax=axs[0,j]
        for tr in m['tracks']:
            if tr['kind']!=kind:continue
            xy=np.array(tr['xy_mm']);col=C['teal'] if tr['pid']%5==0 else C['gold'] if tr['pid']%5==1 else C['gray']
            ax.plot(xy[:,0],xy[:,1],color=col,lw=.8);ax.scatter(*xy[0],s=8,color=col,linewidths=0)
        for lab,col in [('Fit',C['gray']),('Validation',C['gold']),('Test',C['teal'])]:ax.plot([],[],ls='none',marker='o',color=col,label=lab)
        ax.set(xlim=(0,16),ylim=(0,8),xlabel=r'$x\ (\mathrm{mm})$',ylabel=r'$y\ (\mathrm{mm})$',xticks=[0,8,16],yticks=[0,4,8]);ax.set_aspect('equal');ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.10),fontsize=7,columnspacing=.8)
    for f in (0,1):
        ax=axs[1,f];v=z['reconstruction'][-1,f,...,1]*1000;dx=16/(v.shape[1]-1);dy=8/v.shape[0]
        im=ax.imshow(np.concatenate((v,v[:1])),origin='lower',extent=[-dx/2,16+dx/2,-dy/2,8+dy/2],cmap=st.CMAP,interpolation='nearest')
        ax.set(xlim=(0,16),ylim=(0,8),xlabel=r'$x\ (\mathrm{mm})$',ylabel=r'$y\ (\mathrm{mm})$',xticks=[0,8,16],yticks=[0,4,8]);cb=fig.colorbar(im,ax=ax,pad=.025,fraction=.038,shrink=.67)
        cb.set_label(r'$v_{'+('n' if f==0 else 's')+r',y}\ (\mathrm{mm\,s^{-1}})$')
        ax=axs[2,f]
        for key in ('background','previous','reconstruction'):ax.plot(z['x']*1000,z[key][:,f,...,1].mean((0,1))*1000,color=COL[key],label=NAME[key],ls='--' if key=='background' else '-')
        ax.set(xlim=(0,16),xlabel=r'$x\ (\mathrm{mm})$',ylabel=r'$\langle v_{'+('n' if f==0 else 's')+r',y}\rangle_{y,t}\ (\mathrm{mm\,s^{-1}})$');ax.legend(fontsize=7)
    for i,ax in enumerate(axs.flat):st.panel(ax,i,inside=i<4)
    save(fig,'cf','C1_tracks_fields_profiles','CF frame261, one shared initial state across two 12ms windows. Top captured/free tracks, middle final-frame vn/vs y-components, bottom space-time averaged profiles. g1/g3 captured; g2 free. Shen wall basis; superfluid effective viscosity zero; no zero-tangential-gradient condition. Background is a PDE solution, NOT experimental field truth. Every reconstruction curve labelled previous uses the previous free-correlated version.')

def cf_probability():
    fig,axs=plt.subplots(2,3,figsize=(st.WIDTH,135/25.4),layout='constrained');p=data('cf_261_reconstruction_probability.npz')
    cap=data('cf_261_reconstruction_native_cap.npz')
    for j,kind in enumerate(('cap','free')):
        ax=axs[0,j];obs=cap['target_um'] if kind=='cap' else p['free_target'];pred=cap['pred_um'] if kind=='cap' else p['free_mean']
        for dim,col,mk in [(0,C['teal'],'o'),(1,C['orange'],'s')]:ax.scatter(obs[:,dim],pred[:,dim],s=15,color=col,marker=mk,label=['$x$','$y$'][dim])
        lo=min(obs.min(),pred.min());hi=max(obs.max(),pred.max());pad=.08*(hi-lo)
        ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],'--',color=C['gray']);ax.set(xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad),xlabel=r'Observed displacement $(\mu\mathrm{m})$',ylabel=r'Decoder displacement $(\mu\mathrm{m})$');ax.set_aspect('equal');ax.legend(fontsize=7);st.clean(ax)
    for j,metric in enumerate(('rms','nll')):
        ax=axs[1,j]
        for kind,col,mk,label in [('cap',C['teal'],'o','Captured'),('free',C['orange'],'s','Free')]:
            values=[]
            for tag in ('background','previous','reconstruction'):
                q=R['cf_261']['variants'][tag]
                values.append(q['particles'][kind]['NLL']['mean'] if metric=='nll' else q['captured_native']['vector_RMS_um'] if kind=='cap' else q['particles'][kind]['vector_RMS_um'])
            ax.plot(range(3),values,mk+'-',color=col,label=label)
        ax.set(xticks=range(3),xticklabels=['BG','Previous','Unified'],ylabel='NLL / ms' if metric=='nll' else r'Camera RMS $(\mu\mathrm{m})$');ax.legend(fontsize=7)
    st.density_panel(axs[0,2],p,0,legend=True);i=min(12,len(p['cap_nodes'])-1);st.density_panel(axs[1,2],p,i)
    panels(axs);save(fig,'cf','C2_withheld_displacement_probability','Held-out physical particles only. Top left/middle: original-camera displacement diagnostics. Captured camera mean is retrospectively integrated from the 1ms physical mixture, NOT a causal camera likelihood; free is the causal raw-camera response. Bottom scores use the same newly trained model for all candidate fields. Right: two predetermined held-out captured 1ms displacement mixture densities (not camera-interval densities). RMS is a mixture-mean diagnostic, not the optimization target or sole probability-quality criterion.')

def wall():
    z=data('S1_zero_particle_wall_forward.npz');fig,axs=plt.subplots(2,2,figsize=(st.WIDTH,122/25.4),layout='constrained')
    variants=['legacy_viscous','legacy_inviscid','shen_viscous','shen_inviscid']
    labels=['Legacy, viscous','Legacy, inviscid','Shen, viscous','Shen, inviscid'];cols=[C['gray'],C['purple'],C['orange'],C['teal']]
    for f in (0,1):
        ax=axs[0,f]
        for key,lab,col in zip(variants,labels,cols):ax.plot(z['x']*1000,z[key][-1,f,...,1].mean(0)*1000,color=col,label=lab)
        ax.set(xlabel=r'$x\ (\mathrm{mm})$',ylabel=r'$\langle v_{'+('n' if f==0 else 's')+r',y}\rangle_y\ (\mathrm{mm\,s^{-1}})$');ax.legend(fontsize=6.7)
    ax=axs[1,0]
    for key,lab,col in zip(variants,labels,cols):
        dy=np.gradient(z[key][:,1,...,1],z['x'],axis=-1)
        ax.plot(np.arange(25),dy[:,:,0].mean(1),color=col,label=lab)
    ax.set(xlabel=r'$t-t_0\ (\mathrm{ms})$',ylabel=r'$\langle\partial_x v_{s,y}\rangle_{y,x=0}\ (\mathrm{s^{-1}})$');ax.legend(fontsize=6.7)
    ax=axs[1,1]
    for f,col,mk,lab in [(0,C['teal'],'o','$v_n$'),(1,C['orange'],'s','$v_s$')]:ax.semilogy(range(5),[max(R[k]['time_refinement_RMS_mm_s'][f],1e-14) for k in CASES],mk+'-',color=col,label=lab)
    ax.set(xticks=range(5),xticklabels=['1.5','4','10','CF261','CF505'],ylabel=r'Half-step change $(\mathrm{mm\,s^{-1}})$');ax.legend(fontsize=7)
    panels(axs);save(fig,'supplement','S1_wall_basis_viscosity','Top and lower-left: new zero-particle, zero-perturbation, 24ms FORWARD-only wall-basis/viscosity diagnostic, NOT four alternative inversions and not a data-based selection. Main CF inversions all use Shen/inviscid. Wall tangential superfluid derivatives are shown, not forced to zero. Lower-right: two versus four PDE substeps for the selected initial states in all five cases.')

def latents():
    z=data('cf_261_fields.npz');p=data('cf_261_reconstruction_probability.npz');fig,axs=plt.subplots(2,2,figsize=(st.WIDTH,122/25.4),layout='constrained')
    for tag in ('previous','reconstruction'):
        v=z[tag]-z[tag].mean(2,keepdims=True);ft=np.fft.rfft(v,axis=2)/v.shape[2]
        E=.5*np.mean(np.sum(abs(ft)**2,-1),axis=(0,3))*1e6;E[:,1:-1]*=2
        assert np.allclose(E.sum(1),.5*np.mean(np.sum(v*v,-1),axis=(0,2,3))*1e6)
        for f in (0,1):axs[0,f].semilogy(range(1,13),E[f,1:13],'o-',color=COL[tag],label=NAME[tag])
    for f in (0,1):axs[0,f].set(xlabel=r'$k_y$',ylabel=r'$\mathcal{E}_'+('n' if f==0 else 's')+r'(k_y)\ (\mathrm{mm^2\,s^{-2}})$');axs[0,f].legend(fontsize=7)
    ax=axs[1,0];ax.scatter(p['observed_vp_y_mm_s'],p['latent_q'][:,1],s=12,color=C['teal'],alpha=.6,linewidths=0)
    ax.set(xlabel=r'$v_{p,y}^{\mathrm{obs}}\ (\mathrm{mm\,s^{-1}})$',ylabel=r'$P(q=+1\mid\mathcal{H},c)$',ylim=(-.02,1.02));st.clean(ax)
    ax=axs[1,1]
    for q,col,mk in [(0,C['purple'],'o'),(1,C['orange'],'s')]:
        m=p['latent_mu'][0,q];L=p['latent_L'][0,q];ev,Q=np.linalg.eigh(L@L.T);angle=np.degrees(np.arctan2(Q[1,1],Q[0,1]));r=np.sqrt(chi2.ppf(.95,2)*ev)
        ax.add_patch(Ellipse(m,2*r[1],2*r[0],angle=angle,facecolor=col,edgecolor=col,alpha=.18));ax.scatter(*m,marker=mk,color=col,s=22,label=f'$q={[-1,1][q]:+d}$')
    ax.autoscale_view();ax.set_aspect('equal',adjustable='datalim');ax.set(xlabel=r'$\delta_x/a_p$',ylabel=r'$\delta_y/a_p$');ax.legend(fontsize=7);st.clean(ax)
    panels(axs);save(fig,'supplement','S2_cf_spectra_latents','CF261 directional spectra and candidate-field-conditioned vortex-sign/anchor distributions. Lower panels use a fixed prefix of held-out captured observations and the first held-out observation, respectively. Ellipses are latent-conditional 95% Gaussian anchor regions; they are not velocity-field uncertainty or measured vortex geometry.')

def audit():
    fig,axs=plt.subplots(2,2,figsize=(st.WIDTH,130/25.4),layout='constrained')
    for f in (0,1):
        ax=axs[0,f]
        for tag in ('background','previous','reconstruction'):
            xx=[i for i,c in enumerate(CASES[:3]) if tag in R[c]['variants']]
            yy=[R[CASES[i]]['variants'][tag]['error_mm_s']['full'][f] for i in xx]
            ax.plot(xx,yy,'o-',color=COL[tag],label=NAME[tag])
        ax.set(xticks=range(3),xticklabels=['1.5','4','10'],xlabel=r'$\overline{v}_{n,y}\ (\mathrm{mm\,s^{-1}})$',ylabel=r'$E_'+('n' if f==0 else 's')+r'\ (\mathrm{mm\,s^{-1}})$');ax.legend(fontsize=7)
    ax=axs[1,0]
    for seed,col in [(141001,C['purple']),(141002,C['teal'])]:
        trace_path=HERE/f'training/seed_{seed}/trace.json'
        if not trace_path.exists():trace_path=ASSETS/f'training/seed_{seed}/trace.json'
        trace=readjson(trace_path);ax.plot([x['epoch'] for x in trace],[x['validation'] for x in trace],'o-',color=col,label=f'Seed {seed}')
    ax.set(xlabel='Retraining epoch',ylabel='Reserved-condition objective');ax.legend(fontsize=7)
    ax=axs[1,1]
    for kind,col,mk,lab in [('cap',C['teal'],'o','Captured'),('free',C['orange'],'s','Free')]:
        yy=[]
        for label in CASES[3:]:
            r=R[label]['variants'];yy.append(r['reconstruction']['particles'][kind]['NLL']['mean']-r['background']['particles'][kind]['NLL']['mean'])
        ax.plot([0,1],yy,mk+'-',color=col,label=lab)
    ax.axhline(0,color=C['gray'],lw=.7,ls='--');ax.set(xticks=[0,1],xticklabels=['CF261','CF505'],ylabel=r'NLL$_{\rm unified}$ - NLL$_{\rm BG}$');ax.legend(fontsize=7)
    panels(axs);save(fig,'diagnostics','D1_uniform_version_audit','Top: same-grid three-condition field errors; historical comparison exists for 1.5/4 only, and uses free-correlation for BOTH (no captured-correlation substitution). Lower-left: both local continuation training runs; one global validation-selected checkpoint. Lower-right: CF held-out NLL relative to the PDE background under the same new scoring law. No CF field truth exists. Missing free training at 10 mm/s is documented, not filled with test observations.')

def main():
    global PDF,R
    R=readjson(DEST/'evaluation.json')['results']
    with PdfPages(DEST/'comparison_figures.pdf',metadata={'Title':'','Creator':'counterflow'}) as pdf:
        PDF=pdf
        for task in (probability,maps,scales,cf_structure,cf_probability,wall,latents,audit):print('PLOT',task.__name__,flush=True);task()
    rt.dump(DEST/'figure_manifest.json',AUDIT)
    (DEST/'CAPTIONS.txt').write_text('\n\n'.join(CAPTIONS)+'\n',encoding='utf8')

if __name__=='__main__':main()
