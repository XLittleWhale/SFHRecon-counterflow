"""CF full/resolved force labels using the original dynamical-matrix formula.

This prepares query-aligned reference data; it does not declare a validated
native-cadence supervised transition/joint training adapter.
"""
import numpy as np
from runtime import *

DATA=module('revision_native_data_adapter',HERE.parent/'cf_sim_each_pinn_20261109/data_adapter.py')

def force_correction(c,full,resolved,state,q):
    vf,_,af=full;vr,_,ar=resolved
    output=torch.zeros((*state.shape,2),device=vr.device,dtype=vr.dtype)
    for kind,code in (('free',0),('cap',1)):
        ix=torch.where(state==code)[0]
        if not len(ix):continue
        delta=P.core.affine(c,vf[ix],af[ix],kind)-P.core.affine(c,vr[ix],ar[ix],kind)
        A=P.core.matrices(c,vr[ix],kind)[0]
        sign=torch.zeros(len(ix),device=vr.device,dtype=torch.long) if code==0 else q[ix]
        if bool(((sign<0)|(sign>1)).any()):raise ValueError('Captured rows require explicit q labels')
        ii=torch.arange(len(ix),device=vr.device)
        output[ix]=torch.linalg.solve(A[sign],delta[ii,sign,...,None])[...,0]/c['physics']['vn_mean_m_s']
    return output

def quadrature(like):
    x,_=np.polynomial.legendre.leggauss(34)
    y=np.arange(52)*.008/52
    yy,xx=np.meshgrid(y,(x+1)*.016/2,indexing='ij')
    return torch.as_tensor(np.stack((xx,yy),-1).reshape(-1,2)).to(like)

def project_reference(reader,c,device):
    template=load(F.OPERATOR);ratio=c['physics']['vn_mean_m_s']/template['p']['vn_mean_m_s']
    V=(template['V']*ratio).to(device);Pv=(template['Pv']/ratio).to(device)
    points=quadrature(V)
    geo=F.geometry(c,device,V.dtype).evaluate(points)
    # Confirm the exact quadrature ordering, scaling and dual normalization.
    expected=geo['V'].permute(1,0,3,2).reshape_as(V)
    if not torch.allclose(expected,V,atol=1e-13,rtol=1e-10):raise RuntimeError('Channel quadrature/basis mismatch')
    dual=torch.einsum('fpm,fpn->fmn',Pv,V)
    dual_error=float((dual-torch.eye(V.shape[-1],device=device)).abs().max())
    if dual_error>1e-9:raise RuntimeError('Channel velocity projection is not dual')
    coefficients=[]
    for frame in range(reader.configuration['camera_frames']):
        v=torch.as_tensor(reader.spatial(frame,points.cpu().numpy())[0]).to(V)
        delta=(v-geo['v0']).permute(1,0,2).flatten(1)
        coefficients.append(torch.einsum('fpm,fp->fm',Pv,delta).cpu())
    return torch.stack(coefficients),dict(dual_identity_max_abs=dual_error,
        projection='full saved field interpolated on fixed 34x52 Gauss-periodic grid; exact K12 dual',
        quadrature=[34,52],modes_per_fluid=325)

@torch.no_grad()
def prepare(source,prepared,output,device='cuda'):
    source,prepared,output=map(Path,(source,prepared,output))
    if output.resolve().is_relative_to(source.resolve()) or output.resolve().is_relative_to(prepared.resolve()):
        raise ValueError('Write preparation to a separate output directory')
    cfg=read(source/'configuration.json');obs=load(prepared/'probability_observations.pt')
    labels=load(output/'latent_labels.pt');z=obs['z'];meta=obs['meta']
    if labels['meta']['observation_sha256']!=sha(prepared/'probability_observations.pt'):
        raise RuntimeError('Latent labels and observation rows differ')
    c=F.probability_physics(meta['physics_config'],meta['intervals'])
    if not bool(((z['pid']%5==1)|(z['pid']%5==2)).all()):raise ValueError('Reference training requires F/V only')
    reader=DATA.ReferenceFields(source,cfg)
    a,projection=project_reference(reader,c,device)
    n,m=z['time'].shape
    full=[torch.zeros(n,m,2,2,dtype=torch.float64),torch.zeros(n,m,2,2,2,dtype=torch.float64),torch.zeros(n,m,2,2,dtype=torch.float64)]
    resolved=[torch.zeros_like(v) for v in full]
    force=torch.zeros(n,m,2,dtype=torch.float64)
    # The sign is needed only for cap force labels; native-start q is not a
    # claim that capture/sign stays unchanged at unobserved internal times.
    # Consequently only free rows are certified force-supervision targets here.
    force_valid=(z['state']==0)[:,None]&(z['dt']>0)
    for frame in z['frame'].unique(sorted=True).tolist():
        indices=torch.where(z['frame']==frame)[0]
        for ix in indices.split(64):
            zz=move({k:v[ix] for k,v in z.items()},device)
            t=zz['time']*.001+meta['origin_s']
            left,right=a[frame].to(device),a[frame+1].to(device)
            fraction=(t-frame/cfg['camera_fps'])*cfg['camera_fps']
            aa=left+fraction[...,None,None]*(right-left)
            adot=(right-left).expand_as(aa)*cfg['camera_fps']
            rj=F.queries_from_coefficients(c,aa,adot,zz)
            fj=tuple(torch.as_tensor(v,device=device) for v in reader.jets(frame,t.cpu().numpy(),zz['x'].cpu().numpy()))
            for dst,value in zip(full,fj):dst[ix]=value.cpu()
            for dst,value in zip(resolved,rj):dst[ix]=value.cpu()
            # Never fabricate cap q at sub-camera query times. Cap entries are
            # intentionally NaN; marginal/joint free-force loss uses only state0.
            free=torch.where(zz['state']==0)[0]
            force[ix]=float('nan')
            if len(free):
                flat=lambda jets:tuple(v[free].reshape(-1,*v.shape[2:]) for v in jets)
                nf=len(free)*m
                f=force_correction(c,flat(fj),flat(rj),torch.zeros(nf,dtype=torch.long,device=device),
                    torch.zeros(nf,dtype=torch.long,device=device))
                force[ix[free.cpu()]]=f.reshape(len(free),m,2).cpu()
    force[~force_valid]=float('nan')
    contract=dict(observation_sha256=sha(prepared/'probability_observations.pt'),
        latent_label_sha256=sha(output/'latent_labels.pt'),projection=projection,
        operator_sha256=sha(F.OPERATOR),full_field='saved native camera fields; bilinear space, linear time',
        resolved_field='same native field projected to fixed K12 channel basis; linear native time',
        query='same causal pilot location and subinterval midpoint for full and resolved',
        acceleration='partial_t u + G @ u (CF); legacy periodic used G^T @ u',
        force='solve(A, affine(full)-affine(resolved))/U; free only, no force snapshots',
        units='dimensionless force correction in original baseline definition',
        cap_substep_force_certified=False,microstep_truth_claim=False,
        training_method_accepted=False,original_observation_endpoints_unchanged=True,
        field_sha256={name:reader.manifest[name] for name in sorted(reader.checked)})
    save(output/'full_resolved_context.pt',dict(c=c,row_id=z['row_id'],full_jets=tuple(full),
        resolved_jets=tuple(resolved),free_force=force,free_force_valid=force_valid,meta=contract))
    save(output/'training_reference_fields.pt',dict(a=a,frame_time_s=torch.arange(len(a))/cfg['camera_fps'],meta=contract))
    dump(output/'force_context_audit.json',contract)
    return contract
