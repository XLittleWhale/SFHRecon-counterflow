"""Real non-target regression, query semantics and adjoint integration checks."""
from common import *
from closure_model import ClosureNet,ClosureMomentum,closed_step_components
import time

def main():
    setup();start=time.perf_counter()
    case='vn0_0065m_s'
    d,cfg,old,_=EW.inputs(case,'cpu');op=S.P.Momentum(old)
    net,origin=baseline_network(TRAIN_SEEDS[0])
    base,_=V.base_network(TRAIN_SEEDS[0]);hr=origin['head']
    oldnet=V.Network(base,LM.LatentTransitionHead(base,state=load(hr['path'])['state']).double()).freeze()
    oldnet.mode='temporal'
    assert all(torch.equal(x,oldnet.state_dict()[k]) for k,x in net.state_dict().items())
    _,cal,force=EW.components('cpu')
    closure=ClosureNet().double()
    for par in closure.parameters():par.requires_grad_(False)
    wrapper=ClosureMomentum(op,closure)
    generator=torch.Generator().manual_seed(411040)
    theta=.001*torch.randn(2,624,generator=generator,dtype=torch.float64)
    es=torch.zeros(95,624,dtype=torch.float64);en=es.clone()
    with torch.no_grad():
        a0=M.R.rollout(op,theta,es,en);a1=M.R.rollout(wrapper,theta,es,en)
    assert torch.equal(a0,a1)
    results=[];derivatives=[]
    for a,n in ((a0,oldnet),(a1,net)):
        leaf,sl,nl=[v.clone().requires_grad_() for v in (a,es,en)]
        result=score(leaf,sl,nl,d,n,cal,force,'V',12,backward=True)
        results.append(result)
        derivatives.append([v.grad.detach().clone() for v in (leaf,sl,nl)])
    delta=max(abs(results[0][k]-results[1][k]) for k in results[0])
    gd=max(float((x-y).abs().max()) for x,y in zip(*derivatives))
    assert delta<1e-12 and gd<1e-10
    # A real non-target field is legal pre-freeze; only projected modal states used.
    reference=load(T.SOURCE/'prepared'/case/'reference.pt')['a']
    with torch.no_grad():
        closure.layers[-1].bias.fill_(.002)
        parts=closed_step_components(wrapper,reference[:2])
        residual=reference[1:3]-parts['closed_step']
        decomposition=(parts['closed_mean']+residual-(reference[1:3]-parts['phys_step'])).abs().max()
        assert float(decomposition)<1e-12
        z=d['z'];ids=torch.where((z['pid']%5==1)&(z['frame']<2))[0][:8]
        zz=EW.subset(z,ids,'cpu')
        jets=M.R.local_jets(reference[:3],residual[:,1],residual[:,0],zz,d['c'])
        assert all(torch.isfinite(v).all() for v in jets)
    report=dict(passed=True,target_reference_read_or_H_scored=False,case=case,zero_closure_full_rollout_bitwise=True,
        baseline_parameter_tensors_equal=True,baseline_probability_max_difference=delta,baseline_query_gradient_max_difference=gd,
        validation_baseline=results[0],integrated_mean_residual_identity_error=float(decomposition),
        query_convention='only residual endpoint innovations subtracted; neural deterministic mean stays in RHS/acceleration',
        wall_seconds=time.perf_counter()-start,training_origin=origin)
    dump(OUT/'preflight/integration.json',report)
    print(report,flush=True)

if __name__=='__main__':main()
