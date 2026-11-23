"""On-device field/gradient acceptance before any formal optimization."""
import math
import torch
from runtime import F,P,read,state
import calibrated_fields as fields
import calibrated_workflow as workflow
from calibrated_physics import asset


def probe(out,device):
    p=read(out/'protocol.json');ctx=fields.context(out,p['targets'][0],301030,device)
    before=state(ctx['net']);field=ctx['field'];z=ctx['z']
    with torch.no_grad():
        values=torch.arange(field.output.weight.numel(),dtype=torch.float64).sin().reshape_as(field.output.weight)*.001
        field.output.weight.copy_(values.to(device))
    first=int(z['frame'].min());mask=(z['frame']<first+2)&P.group_mask(z['pid'],'I');pids=[]
    for kind in (0,1):
        selected=z['pid'][mask&(z['state']==kind)].unique()[:2]
        if len(selected)<2:raise RuntimeError('Acceptance requires both particle states')
        pids.extend(selected.tolist())
    ix=torch.where(mask&torch.isin(z['pid'],z['pid'].new_tensor(pids)))[0]
    small={k:v[ix] for k,v in z.items()};trial=dict(ctx,z=small)
    fit=fields.score(trial,'I',12,backward=True)
    t=torch.arange(2*ctx['prior'].intervals+1,device=device)*.0005
    a,ad=field.coefficients_and_dt(t)
    nfit=float((z['end']-z['start'])[P.group_mask(z['pid'],'I')].sum())
    loss,_=F.residual_loss(a,ad,ctx['op'],ctx['prior'],nfit)
    reg,_=field.regularization();(loss+reg).backward()
    gradient=torch.cat([v.grad.reshape(-1) for v in field.parameters()]).detach().cpu()
    if not torch.isfinite(gradient).all():raise RuntimeError('Nonfinite PINN derivative')
    if any(not torch.equal(v.detach().cpu(),before[k]) for k,v in ctx['net'].state_dict().items()):
        raise RuntimeError('Frozen probability changed during acceptance')
    if any(v.requires_grad or v.grad is not None for v in ctx['net'].parameters()):
        raise RuntimeError('Frozen probability acquired gradients')
    with torch.no_grad():
        tt=torch.tensor([.0173,.0811,.2013],device=device)
        aa,da=field.coefficients_and_dt(tt);h=1e-7
        fd=(field(tt+h)-field(tt-h))/(2*h)
        derivative=float((fd-da).norm()/da.norm().clamp_min(1e-20))
        points=torch.tensor([[x,y] for x in (0.,.0012,.0073,.016) for y in (0.,.008)],device=device)
        b=F.geometry(ctx['c'],device).evaluate(points)
        v=b['v0'][None]+torch.einsum('tfm,pfmj->tpfj',aa,b['V'])
        g=b['g0'][None]+torch.einsum('tfm,pfmij->tpfij',aa,b['G'])
        wall=v[:,[0,1,6,7]]
        normal_wall=float(wall[:,:,0,:].abs().max());super_wall=float(wall[:,:,1,0].abs().max())
        periodic=float((v[:,::2]-v[:,1::2]).abs().max())
        divergence=float(torch.diagonal(g,dim1=-2,dim2=-1).sum(-1).abs().max())
        identity=float((field.encode@field.decode-torch.eye(325,device=device)).abs().max())
        if max(normal_wall,super_wall,periodic)>1e-10 or divergence>1e-8 or derivative>1e-6 or identity>1e-8:
            raise RuntimeError('Geometry or derivative acceptance failed')
    return dict(inversion=fit['total'],pde=float(loss.detach()),regularization=float(reg.detach()),
        gradient=gradient,rows=len(ix),full_PDE_nodes=len(t),boundary_max_m_s=max(normal_wall,super_wall),
        periodic_max_m_s=periodic,divergence_max_per_s=divergence,time_derivative_relative_error=derivative,
        coordinate_inverse_max_error=identity)


def run(out,device):
    p=workflow.verify(out);seeds=[]
    for seed in p['config']['field_seeds']:
        net,_,_,digest=workflow.frozen(out,'cf_2K_Q217',seed,device)
        if any(x.requires_grad for x in net.parameters()):raise RuntimeError('Trainable probability')
        seeds.append(dict(seed=seed,sha256=digest,all_parameters_frozen=True))
    reference=probe(out,'cpu')
    result=reference if device=='cpu' else probe(out,device)
    relative=float((result['gradient']-reference['gradient']).norm()/reference['gradient'].norm().clamp_min(1e-20))
    errors={k:abs(result[k]-reference[k]) for k in ('inversion','pde','regularization')}
    if relative>1e-5 or any(value>1e-7*max(1,abs(reference[k])) for k,value in errors.items()):
        raise RuntimeError('CPU/device objective or gradient mismatch: '+str((relative,errors)))
    public={k:v for k,v in result.items() if k!='gradient'}
    return dict(passed=True,device=device,torch_version=torch.__version__,probability_models=seeds,
        cuda_tested=device=='cuda',CPU_device_gradient_relative_difference=relative,
        CPU_device_loss_absolute_differences=errors,probe=public,
        unchanged_geometry_and_prior_loaded_from_asset=True,H_or_evaluation_truth_read=False,
        probe_model_discarded=True,formal_training_executed=False)
