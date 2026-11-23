"""Channel geometry adapter for the unchanged optimization-aware objectives.

States are mass-normalized q (energy U^2/2 per coefficient), with the same
spectral/initial penalties as the periodic baseline. The original 568-parameter
tensor-basis closure is projected weakly; its stress has zero wall traction.
The material acceleration convention is G@u, deliberately distinct from the
historical periodic G.T@u adapter.
"""
import math
from types import SimpleNamespace
import numpy as np
from torch.utils.checkpoint import checkpoint
from runtime import *
import baseline_api
import native_losses as N
import clock_contract as clock
from force_context import force_correction,quadrature

_ns=dict(torch=torch,nn=torch.nn,math=math)
baseline_api.functions(baseline_api.SOURCES['closure'],['ClosureNet','invariant_features'],_ns)
ClosureNet=_ns['ClosureNet'];invariant_features=_ns['invariant_features']


class ChannelMomentum:
    def __init__(self,c,base,dt,net=None):
        self.c=c;self.base=base;self.dt=float(dt);self.net=net
        self.decode=c['pinn_decode'];self.encode=c['pinn_encode'];self.ring=c['pinn_ring']
        self.points=quadrature(base.V)
        self.envelope=torch.sin(math.pi*self.points[:,0]/c['domain_m'][0]).square()
        self.feature_op=SimpleNamespace(U=c['physics']['vn_mean_m_s'],
            c=dict(box_m=c['domain_m'][1]))

    def raw(self,q):return torch.einsum('fij,...fj->...fi',self.decode,q)
    def normalized(self,a):return torch.einsum('fij,...fj->...fi',self.encode,a)

    def tendency(self,a,diagnostics=False):
        if self.net is None:return torch.zeros_like(a)
        scalar=a.ndim==2;aa=a[None] if scalar else a
        op=self.base
        v=op.v0+torch.einsum('fpm,tfm->tfp',op.V,aa).reshape(len(aa),*op.v0.shape)
        g=op.g0+torch.einsum('fpm,tfm->tfp',op.G,aa).reshape(len(aa),*op.g0.shape)
        # Original invariant code expects two spatial dimensions; one can be 1.
        features,(sn,ss,ww,w),ell=invariant_features(self.feature_op,v.unsqueeze(2),g[...,0].unsqueeze(2),g[...,1].unsqueeze(2))
        coefficients=self.net(features);basis=torch.stack((sn,ss,ww),-3)
        U=self.feature_op.U
        stress=torch.stack((U**2*torch.einsum('...k,...kij->...ij',coefficients[...,:3],basis),
                            U**2*torch.einsum('...k,...kij->...ij',coefficients[...,3:6],basis)),1).squeeze(2)
        stress=stress*self.envelope[None,None,:,None,None]
        rn,rs=op.p['rho_n_kg_m3'],op.p['rho_s_kg_m3']
        jw=torch.stack((-w[...,1],w[...,0]),-1)
        force=rn*rs/(rn+rs)*U**2/ell*(coefficients[...,6,None]*w+coefficients[...,7,None]*jw)
        force=force.squeeze(1)
        exchange=torch.stack((force/rn,-force/rs),1)
        stress_modal=torch.einsum('fpm,tfp->tfm',op.Pg,stress.flatten(2))
        force_modal=torch.einsum('fpm,tfp->tfm',op.Pv,exchange.flatten(2))
        result=stress_modal+force_modal
        result=result[0] if scalar else result
        if diagnostics:return result,dict(internal_force_error=(rn*exchange[:,0]+rs*exchange[:,1]).abs().max(),
            stress=stress,coefficients=coefficients)
        return result

    def raw_rhs(self,a):
        physical=self.base.rhs(a) if a.ndim==2 else F.batched_rhs(self.base,a)
        return physical+self.tendency(a)

    def rhs(self,q):return self.normalized(self.raw_rhs(self.raw(q)))

    def step(self,q,substeps=2):
        # Same <=0.5 ms RK4 substep rule at every geometry/camera cadence.
        count=math.ceil(self.dt/(.001/substeps)-1e-10);h=self.dt/count
        def advance(a):
            k1=self.rhs(a);k2=self.rhs(a+h*k1/2);k3=self.rhs(a+h*k2/2);k4=self.rhs(a+h*k3)
            return a+h*(k1+2*k2+2*k3+k4)/6
        for _ in range(count):
            if torch.is_grad_enabled() and (q.requires_grad or self.net is not None):
                q=checkpoint(advance,q,use_reentrant=False,preserve_rng_state=False)
            else:q=advance(q)
        return q


def trajectory(op,initial,length,substeps=2):
    result=[initial]
    for _ in range(length):result.append(op.step(result[-1],substeps))
    return torch.stack(result)


def reference_residual(op,truth,substeps=2):
    return truth[1:]-op.step(truth[:-1],substeps)


def query(w,a,residual=None):
    z=w['z'];frame=z['frame'];dt=w['dt']
    fraction=(z['time']*.001-frame[:,None]*dt)/dt
    if bool(((fraction[z['dt']>0]<-1e-7)|(fraction[z['dt']>0]>1+1e-7)).any()):raise ValueError('Query outside native interval')
    if residual is None:residual=torch.zeros_like(a[:-1])
    # Preserve the baseline endpoint-innovation query semantics; learned
    # deterministic closure is already included in the trajectory difference.
    normal_only=torch.stack((residual[:,0],torch.zeros_like(residual[:,1])),1)
    aa=a[frame,None]+fraction[...,None,None]*(a[frame+1,None]-a[frame,None]-normal_only[frame,None])
    adot=((a[frame+1]-a[frame]-residual[frame])/dt)[:,None].expand_as(aa)
    raw=w['op'].raw(aa);rawdt=w['op'].raw(adot)
    if 'basis' not in w:
        w['basis']=F.geometry(w['c'],a.device,a.dtype).evaluate(z['x'].reshape(-1,2))
    b=w['basis'];raw=raw.reshape(-1,2,a.shape[-1]);rawdt=rawdt.reshape_as(raw)
    v=b['v0']+torch.einsum('pfm,pfmj->pfj',raw,b['V'])
    g=b['g0']+torch.einsum('pfm,pfmij->pfij',raw,b['G'])
    acc=torch.einsum('pfm,pfmj->pfj',rawdt,b['V'])+torch.einsum('pfij,pfj->pfi',g,v)
    return tuple(j.reshape(*z['time'].shape,*j.shape[1:]) for j in (v,g,acc))


def relabel_force(w,jets):
    n,m=w['z']['dt'].shape;result=jets[0].new_full((n,m,2),float('nan'))
    fi=torch.where(w['z']['state']==0)[0]
    flatten=lambda jj:tuple(j[fi].reshape(-1,*j.shape[2:]) for j in jj)
    state=torch.zeros(len(fi)*m,dtype=torch.long,device=result.device)
    result[fi]=force_correction(w['c'],flatten(w['full_jets']),flatten(jets),state,state).reshape(len(fi),m,2)
    return result


def penalty(a,ring,factor=1.):
    high=ring>2
    physical=.003*a.square().sum(-1).mean()/24+.03*(a[1:]-a[:-1]).square().sum(-1).mean()/24
    spectral=.03*(a[0]*torch.where(high,(ring/2)**2,0)).square().sum(-1).mean()/24
    return factor*(physical+spectral)


def kernels(w):
    ring=w['op'].ring
    core=SimpleNamespace(config=lambda:ring,penalty=penalty)
    prob=SimpleNamespace(core=core,window_nll=N.window_nll,supervised_anchors=N.supervised_anchors)
    data=SimpleNamespace(trajectory=trajectory,query=query,reference_residual=reference_residual,relabel_force=relabel_force)
    return baseline_api.joint_kernels(data,prob,lambda like:1/(1+(ring.to(like)/4)**2))


def load_case(name,device='cpu',supervision_root=None):
    from cf_label_diagnostics import directory
    folder=directory(name) if supervision_root is None else Path(supervision_root)/name
    path=BASE/'outputs/CFSimEachPINN_20261109/prepared'/name/'probability_observations.pt'
    obs=load(path);labels=load(folder/'latent_labels.pt');ref=load(folder/'full_resolved_context.pt')
    field=load(folder/'training_reference_fields.pt')
    if labels['meta']['observation_sha256']!=sha(path) or ref['meta']['latent_label_sha256']!=sha(folder/'latent_labels.pt'):
        raise RuntimeError('Reference supervision provenance mismatch')
    if not torch.equal(ref['row_id'],obs['z']['row_id']):raise RuntimeError('Reference rows misaligned')
    if not bool(((obs['z']['pid']%5==1)|(obs['z']['pid']%5==2)).all()):raise ValueError('Only F/V reference rows permitted')
    c,base,_,_=F.prepare_physics(obs['meta']['physics_config'],obs['meta']['intervals'],device=device)
    dt=1/obs['meta']['camera_fps'];op=ChannelMomentum(c,base,dt)
    z=move(obs['z'],device);cal,force=P.calibration()
    return dict(case=name,c=c,op=op,dt=dt,meta=obs['meta'],z=z,labels=move(labels,device),
        a=op.normalized(field['a'].to(device)),cal=cal,force=force,
        jets=tuple(v.to(device) for v in ref['resolved_jets']),full_jets=tuple(v.to(device) for v in ref['full_jets']),
        force_labels=ref['free_force'].to(device),window_audit=clock.audit_windows(obs['z'],obs['meta']['camera_fps']))


def window(case,start,group='F',pid_limit=None):
    length=case['window_audit']['native_intervals'];z=case['z']
    mask=(z['frame']>=start)&(z['frame']<start+length)&(z['pid']%5=={'F':2,'V':1}[group])
    if pid_limit is not None:
        # Only preflight: preserve >=2 PIDs/state in each inner/outer split.
        pids=[];splits=clock.split_reference(z['pid'],group)
        for split in ('inner','outer'):
            for state in (0,1):pids.extend(z['pid'][mask&splits[split]&(z['state']==state)].unique()[:pid_limit].tolist())
        mask=mask&torch.isin(z['pid'],z['pid'].new_tensor(pids))
    ids=torch.where(mask)[0];zz=N.subset(z,ids)
    zz['frame']=zz['frame']-start
    shift=(start*case['dt']-case['meta']['origin_s'])/.001
    for key in ('time','start','end'):zz[key]=zz[key]-shift
    masks=clock.split_reference(zz['pid'],group)
    for split in ('inner','outer'):
        for state in (0,1):
            if len(zz['pid'][masks[split]&(zz['state']==state)].unique())<2:raise ValueError('Insufficient physical PIDs for joint window')
    return dict(case=case['case'],c=case['c'],op=case['op'],dt=case['dt'],cal=case['cal'],force=case['force'],
        z=zz,labels=N.subset(case['labels'],ids),full_jets=tuple(j[ids] for j in case['full_jets']),
        truth=case['a'][start:start+length+1],masks=masks,start=start,length=length)


def valid_starts(case,group):return [r['start'] for r in case['window_audit']['groups'][group]['valid']]


def closed(case,net):return ChannelMomentum(case['c'],case['op'].base,case['dt'],net)


@torch.no_grad()
def stability(case,op,settings):
    def stats(operator):
        q=case['a'][0];peak=q.square().sum(-1).sqrt()
        for frame in range(len(case['a'])-1):
            q=operator.step(q,settings['substeps'])
            if not bool(torch.isfinite(q).all()) or float(q.abs().max())>settings['full_window_amplitude_limit']:
                return dict(passed=False,failed_frame=frame+1,reason='nonfinite_or_amplitude')
            peak=torch.maximum(peak,q.square().sum(-1).sqrt())
        return dict(passed=True,peak=peak.cpu().tolist(),frames=len(case['a']),duration_s=(len(case['a'])-1)*case['dt'])
    if '_stability' not in case:case['_stability']=stats(case['op'])
    baseline=case['_stability']
    if not baseline['passed']:raise RuntimeError('Pure physical full-window baseline is unstable: '+str(baseline))
    result=baseline.copy() if op is case['op'] else stats(op)
    if result['passed']:
        ratio=np.array(result['peak'])/np.maximum(np.array(baseline['peak']),1e-12)
        result.update(peak_growth_ratio=ratio.tolist(),passed=bool((ratio<=settings['full_window_growth_factor']).all()))
    return result
