import copy
import json
from pathlib import Path
import numpy as np
import torch
from counterflow import observation as ob
from counterflow.io import load
from counterflow_full import core as fc,prepare as fp
from counterflow_full.build import records


def fixture():
    torch.set_num_threads(2)
    resources=Path(fc.__file__).parent/'resources'
    c=json.loads((resources/'physics_4mm.json').read_text(encoding='utf8'))
    like=torch.zeros((),dtype=torch.float64)
    _,H,B,_=fc.matrices(c,like,'cap');c['response']=(B,H)
    # Legacy fixtures store the isotropic free response as two scalars.
    _,H,B,_=fc.matrices(c,like,'free');c['free_response']=(H[0,0,0],B[0,0,0])
    return c,load(resources/'operator.pt')


def test_native_causal_records():
    cfg=dict(start=5,end=9)
    rng=np.random.default_rng(1);x=np.cumsum(rng.normal(size=(10,8,2))*1e-6,0);state=np.zeros((10,8),int)
    a,report=records(x,state,cfg);xx=x.copy();xx[6:]+=10
    b,_=records(xx,state,cfg);m=a['frame']==0
    for key in ('x','w','h'):np.testing.assert_array_equal(a[key][m],b[key][m])
    assert not np.array_equal(a['target'][m],b['target'][m])
    assert report['valid_intervals']==32 and report['unique_pids']==8


def test_coarse_conservation_and_phase():
    W=torch.tensor(fp.overlap(64,40));torch.testing.assert_close(W.sum(1),torch.ones(40,dtype=W.dtype))
    torch.testing.assert_close(W.sum(0),torch.full((64,),40/64,dtype=W.dtype))
    c,_=fixture();a=torch.randn(1,2,624,dtype=torch.float64)*.01
    v=fc.grid(a,c,fc.config())[0].permute(1,2,0,3)
    got=fp.project(v,12,c['physics']['vn_mean_m_s'])
    torch.testing.assert_close(got,a[0],atol=1e-14,rtol=1e-11)


def test_basis_and_force_parity():
    c,_=fixture();torch.manual_seed(2);a=torch.randn(3,2,624,dtype=torch.float64)*.002
    z=dict(frame=torch.tensor([0,1]),x=torch.rand(2,2,dtype=torch.float64)*.01,
           w=torch.randn(2,2,dtype=torch.float64),h=torch.randn(2,5,4,dtype=torch.float64))
    v,g,ac=fc.jets(a,z,c);V,G=fc.basis(z['x'],c)
    legacy=z|dict(V=V[:,None].expand(-1,2,-1,-1),G=G[:,None].expand(-1,2,-1,-1,-1),
        v0=torch.tensor([[0,c['physics']['vn_mean_m_s']],[0,c['physics']['vs_mean_m_s']]]),g0=torch.zeros_like(g))
    legacy['v0']=legacy['v0'].double();legacy['context']=fc.context(c,z,v,g)[:,:5]
    cc=c|dict(cap=legacy,free=legacy)
    for x,y in zip(ob.local(legacy,a),(v,g,ac)):torch.testing.assert_close(x,y,atol=1e-9,rtol=1e-7)
    ctx,b,gain,hr=ob.particle_parts(cc,a);torch.testing.assert_close(ctx,fc.context(c,z,v,g),atol=1e-6,rtol=1e-6)
    A,H,B,M=fc.matrices(c,a,'cap');base=torch.einsum('qij,bj->bqi',H,z['w'])+.001/c['physics']['vn_mean_m_s']*torch.einsum('qij,bqj->bqi',B,fc.affine(c,v,ac,'cap'))
    torch.testing.assert_close(base,b,atol=1e-7,rtol=1e-6)
    A,H,B,M=fc.matrices(c,a,'free');base=torch.einsum('qij,bj->bqi',H,z['w'])+.001/c['physics']['vn_mean_m_s']*torch.einsum('qij,bqj->bqi',B,fc.affine(c,v,ac,'free'))
    torch.testing.assert_close(base[:,0],ob.predict_free(cc,a),atol=1e-7,rtol=1e-6)


def test_pid_balancing_and_disjoint():
    z=dict(pid=torch.tensor([0,1,2,2,3,4,2]),state=torch.tensor([0,0,0,0,0,1,1]))
    w=fc.weights(z,'train');torch.testing.assert_close(w.sum(),torch.tensor(1.,dtype=w.dtype))
    assert w[0]==w[1]==0
    assert w[2]==w[3] and w[2]*2==w[4]
    assert float(w[z['state']==0].sum())==.5
    for a in ('train','validation','test'):
        for b in ('train','validation','test'):
            if a!=b:assert not (fc.split(z['pid'],a)&fc.split(z['pid'],b)).any()


def test_chunked_full_pde_gradient():
    from counterflow.dynamics import rollout
    from counterflow_full.inverse import objective
    from counterflow_full.train import default_cal
    c,data=fixture();cfg=copy.deepcopy(fc.config());cfg['inversion'].update(batch=2,degree=2)
    op=fc.operator(c,data,cfg,'cpu');torch.manual_seed(3)
    net=ob.Density().double().eval()
    for p in net.parameters():p.requires_grad_(False)
    force={k:dict(mean=[.02,-.03],covariance=[[.01,0],[0,.02]]) for k in ('cap','free')}
    z=dict(frame=torch.tensor([0,0,1,1]),pid=torch.tensor([2,3,2,3]),state=torch.tensor([0,0,1,1]),
        x=torch.rand(4,2,dtype=torch.float64)*.01,w=torch.randn(4,2,dtype=torch.float64),
        h=torch.randn(4,5,4,dtype=torch.float64),target=torch.randn(4,2,dtype=torch.float64))
    initial=(torch.randn(2,624,dtype=torch.float64)*1e-3).requires_grad_(True)
    a,_=rollout(op,initial,2,2,False);w=fc.weights(z,'train');loss=fc.penalty(a,cfg,1.)
    for kind,s in [('free',0),('cap',1)]:
        ix=torch.where(z['state']==s)[0];b=fc.subset(z,ix,'cpu');v,g,ac=fc.jets(a,b,c)
        pred,cov,lw=fc.distribution(c,b,v,g,ac,kind,net,default_cal(),force,2)
        loss=loss+.08*(w[ix]*fc.scores(pred,cov,lw,b['target'])).sum()
    grad,=torch.autograd.grad(loss,initial)
    a,_=rollout(op,initial,2,2,True);leaf=a.detach().requires_grad_(True)
    objective(leaf,c,z,net,default_cal(),force,cfg,'cpu','train',True)
    reg=fc.penalty(a,cfg,1.);torch.autograd.backward((a,reg),(leaf.grad,torch.ones_like(reg)))
    torch.testing.assert_close(initial.grad,grad,rtol=1e-10,atol=1e-10)
    # The free equation retains its genuine acceleration sensitivity to vs.
    assert torch.linalg.vector_norm(initial.grad[1])>0
