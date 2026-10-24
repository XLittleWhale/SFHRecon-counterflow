"""Evaluation ONLY after every checkpoint is frozen. No scores feed inversion."""
from .common import *
import importlib.util
from scipy.stats import chi2
from .observation import latent,mixture_nll
from . import vtk, probability as cm
from .paths import DATA as INPUT_DATA
DATA=DEST/'data'
NOM=np.array([.5,.7,.8,.9,.95])

def reference(c):
    with np.load(INPUT_DATA/'evaluation'/f"{c['name']}_reference.npz") as z:
        return torch.as_tensor(z['coefficients'].copy()),z['velocity'].copy()


def rms(v):return np.sqrt(np.mean(v*v,axis=(0,2,3,4)))*1000

def grouped(v,ids):
    vv=np.array([np.mean(v[ids==p]) for p in np.unique(ids)])
    return dict(mean=float(vv.mean()),SE=float(vv.std(ddof=1)/np.sqrt(len(vv))) if len(vv)>1 else None,physical_particles=len(vv),segments=len(v))

def coverage(radius,pid):
    ids=np.unique(pid);hits=np.array([radius<=chi2.ppf(q,2) for q in NOM]).T
    by=np.array([hits[pid==p].mean(0) for p in ids]);rng=np.random.default_rng(141014)
    counts=np.array([(pid==p).sum() for p in ids]);sums=by*counts[:,None]
    draw=rng.integers(len(by),size=(400,len(by)))
    boot=sums[draw].sum(1)/counts[draw].sum(1)[:,None]
    return dict(coverage=hits.mean(0).tolist(),particle_balanced_coverage=by.mean(0).tolist(),CI95=np.quantile(boot,[.025,.975],axis=0).tolist())

@torch.no_grad()
def probability(c,a,net,cal):
    outputs={};report={};bs=parts(c,0);U=c['physics']['vn_mean_m_s'];scale=U*.001*1e6;T=c['observation_steps']
    for kind in ('cap','free'):
        selected=[z for k,z in bs if k==kind]
        if not selected:report[kind]=dict(available=False);continue
        vals=[];ids=[];means=[];targets=[];pids=[];frames=[];radius=[];nodes=[];weights=[];qq=[];mu=[];Ls=[];vpy=[]
        sigma=law(c['fixed_noise']['captured' if kind=='cap' else 'free'],U,.015,1.5)
        for z in selected:
            ids.append(z['pid'][::T].numpy()%c['pid_stride'])
            if kind=='cap':
                ctx,base,gain,hr=particle_parts(dict(c,cap=z),a)
                pp,lw=decoded(net,z['h'],ctx,base,gain,hr,cal,24)
                val=obs.captured_nll(c,a,z,net,cal,24);zz,_=cm.transform(pp,lw,z['target'],sigma)
                means.append((pp*lw.exp()[...,None]).sum((1,2)).numpy()*scale);targets.append(z['target'].numpy()*scale)
                radius.append(zz.square().sum(-1).numpy());pids.append(z['pid'].numpy()%c['pid_stride']);frames.append(z['frame'].numpy())
                # Only a fixed small prefix of held-out observations is needed for contour panels.
                if not nodes:
                    take=min(48,len(pp));nodes.append(pp[:take].flatten(1,2).numpy()*scale);weights.append(lw[:take].flatten(1).exp().numpy())
                    lq,m,L=latent(net,z['h'],ctx,cal);qq.append(lq.exp().numpy());mu.append(m.numpy());Ls.append(L.numpy());vpy.append(z['w'][:,1].numpy()*U*1000)
            elif c['kind']=='periodic':
                val=obs.free_nll(c,a,z);mean=predict_free(dict(c,free=z),a)
                err=(z['target']-mean)/sigma
                radius.append((err.square()/(1+obs.BETA)).sum(-1).numpy());means.append(mean.numpy()*scale);targets.append(z['target'].numpy()*scale)
                pids.append(z['pid'].numpy()%c['pid_stride']);frames.append(z['frame'].numpy())
            else:
                val=obs.free_nll(c,a,z)
                for pid in z['pid'][::T].tolist():
                    p=c['free_native'][pid];mean=obs.native_predictions(c,a,p);W=p['weights'];d=W.sum(1)
                    diag=(W.square().sum(1))[:,None]+d[:,None].square()*obs.BETA[None]
                    radius.append((((p['target']-mean)/sigma).square()/diag).sum(-1).numpy())
                    means.append(mean.numpy()*scale);targets.append(p['target'].numpy()*scale)
                    pids.append(np.full(len(mean),pid%c['pid_stride']));frames.append(p['native_times'][:,0].numpy())
            vals.append(val.numpy())
        vv=np.concatenate(vals);ii=np.concatenate(ids);pred=np.concatenate(means);tar=np.concatenate(targets);rr=np.concatenate(radius);pid=np.concatenate(pids)
        report[kind]=dict(available=True,NLL=grouped(vv,ii),vector_RMS_um=float(np.sqrt(np.mean(np.sum((pred-tar)**2,-1)))),
            intervals=len(pred),**coverage(rr,pid))
        for k,v in dict(mean=pred,target=tar,pid=pid,frame=np.concatenate(frames),radius=rr,nll=vv,segment_pid=ii).items():outputs[kind+'_'+k]=v
        if nodes:
            outputs.update(cap_nodes=nodes[0],cap_weights=weights[0],cap_sigma_um=sigma*scale,latent_q=qq[0],latent_mu=mu[0],latent_L=Ls[0],observed_vp_y_mm_s=vpy[0])
    outputs['nominal']=NOM
    return report,outputs

@torch.no_grad()
def cap_native(c,windows,a,net,cal):
    pred=[];tar=[];pid=[]
    for offset,w in zip(c['offsets'],windows):
        means=[]
        for lo in range(0,len(w['cap']['pid']),128):
            z={k:v[lo:lo+128] for k,v in w['cap'].items()};ctx,b,g,h=particle_parts(dict(w,cap=z),a[offset:offset+13])
            pp,lw=decoded(net,z['h'],ctx,b,g,h,cal,24);means.append((pp*lw.exp()[...,None]).sum((1,2)))
        mean=torch.cat(means);native=w['native']['cap']
        pred.append(((mean[native['ids']]*native['weights'][...,None]).sum(1)*w['physics']['vn_mean_m_s']*1e6).numpy())
        tar.append(native['obs_um'].numpy());pid.append(native['pid'].numpy())
    pred,tar,pid=map(np.concatenate,(pred,tar,pid))
    return dict(pred_um=pred,target_um=tar,pid=pid),dict(vector_RMS_um=float(np.sqrt(np.mean(np.sum((pred-tar)**2,-1)))),intervals=len(pred),physical_particles=len(np.unique(pid)),retrospective=True)

@torch.no_grad()
def main():
    declare();assert (HERE/'frozen_selections.json').exists(),'Freeze every case before reading truth'
    for p,h in readjson(HERE/'frozen_selections.json').items():assert rt.sha(HERE/p)==h
    DATA.mkdir(parents=True,exist_ok=True);net,cal=model();report={};sources={}
    for label in CASES:
        c,windows=bundle(label);op=rt.make(rt.load(c['operator_source']),'cpu');folder=HERE/'fits'/label
        choice=readjson(folder/'selected_joint.json');chosen=folder/choice['chosen'];theta=rt.load(chosen/'selected.pt')
        a,q=rt.forward(c,op,theta);bg,_=rt.forward(c,op,torch.zeros_like(theta))
        arrays=dict(x=c['x'].numpy(),y=c['y'].numpy(),time_s=c['start_s']+np.arange(c['intervals']+1)*.001)
        fields={'reconstruction':a,'background':bg};rr=dict(selection=choice,fit=readjson(chosen/'summary.json'),variants={})
        sources[(chosen/'selected.pt').relative_to(HERE).as_posix()]=rt.sha(chosen/'selected.pt')
        comparator=INPUT_DATA/'evaluation'/f'{label}_comparator.pt'
        if comparator.exists():
            fields['previous'],_=rt.forward(c,op,rt.load(comparator))
            sources['archived_comparator:'+label]=rt.sha(comparator)
        if c['kind']=='periodic':
            ref,truth=reference(c);fields['reference']=ref;arrays['reference']=truth
            arrays['projected']=rt.velocity(c,ref).numpy()
        for tag,coeff in fields.items():
            vel=rt.velocity(c,coeff).numpy();arrays[tag if tag!='reference' else 'projected']=vel
            diag,probs=probability(c,coeff,net,cal);meta=dict(particles=diag)
            np.savez_compressed(DATA/f'{label}_{tag}_probability.npz',**probs)
            if c['kind']=='periodic':
                meta['error_mm_s']=dict(full=rms(vel-truth).tolist(),inverse=rms(vel-arrays['projected']).tolist(),truncation=rms(arrays['projected']-truth).tolist())
                meta['window_errors_mm_s']=[rms(vel[o:o+5]-truth[o:o+5]).tolist() for o in c['offsets']]
            else:
                raw,diagnostic=cap_native(c,windows,coeff,net,cal);meta['captured_native']=diagnostic
                np.savez_compressed(DATA/f'{label}_{tag}_native_cap.npz',**raw)
                meta['wall_max_abs_m_s']=dict(normal_n=float(np.max(abs(vel[:,0,:,[0,-1],0]))),tangential_n=float(np.max(abs(vel[:,0,:,[0,-1],1]))),normal_s=float(np.max(abs(vel[:,1,:,[0,-1],0]))))
            rr['variants'][tag]=meta
            vtk.vtk_series(HERE/'fields'/label/tag,truth if tag=='reference' else vel,c['x'].numpy(),c['y'].numpy(),arrays['time_s'])
        refined,_=rt.forward(c,op,theta,4)
        rr['time_refinement_RMS_mm_s']=rms(rt.velocity(c,refined).numpy()-arrays['reconstruction']).tolist()
        rr['numerical_refinement_passed']=max(rr['time_refinement_RMS_mm_s'])<.01
        np.savez_compressed(DATA/f'{label}_fields.npz',**arrays)
        if c['kind']=='channel':rt.dump(DATA/f'{label}_metadata.json',dict(tracks=c['segments'],physical_truth_available=False))
        report[label]=rr;rt.dump(HERE/'evaluation_progress.json',report)
        print('EVALUATE',label,rr['variants']['reconstruction'].get('error_mm_s',rr['variants']['reconstruction']['particles']),flush=True);gc.collect()
    same=all(readjson(HERE/'fits'/label/'selected_joint.json')['model_sha']==fingerprints()['model_sha'] for label in CASES)
    assert same and all(r['selection']['likelihood_version']==POLICY['version'] for r in report.values())
    # Frozen forward-only sensitivity evidence, not selectable inverse versions.
    import shutil
    shutil.copy2(INPUT_DATA/'evaluation/S1_zero_particle_wall_forward.npz',DATA/'S1_zero_particle_wall_forward.npz')
    wallmeta=readjson(INPUT_DATA/'evaluation/wall_metadata.json')
    obj=dict(results=report,policy=POLICY,uniform_model_and_version=True,sources=sources,wall_diagnostic=wallmeta,**fingerprints())
    rt.dump(HERE/'evaluation.json',obj);rt.dump(DEST/'evaluation.json',obj)
    rt.dump(DEST/'data_manifest.json',{p.name:rt.sha(p) for p in DATA.glob('*')})

if __name__=='__main__':main()
