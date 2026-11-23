"""Executable numerical acceptance, using discarded models and permitted data."""
import gc
import math
import time
import tempfile
from runtime import *
import baseline_api as A
import native_losses as N
import common_training as T
import channel_joint as J
import clock_contract as clock


def periodic_joint_regression(device):
    import periodic_joint as Q
    case=Q.load_case('vn0_0025m_s',device,fixture=True);w=Q.window(case,30,'F',2)
    results=[]
    for old in (True,False):
        net,*_=B.load(301030,device);A.configure_stage(net,'joint')
        torch.manual_seed(401113);closure=Q.ClosureNet().to(device);op=Q.closed(case,closure)
        k=Q.legacy_kernels() if old else Q.kernels(w)
        loss,detail=k['train_window'](Q.legacy_window(w) if old else w,op,net,'joint',T.settings('smoke')['joint'])
        loss.backward()
        result=[loss.detach(),torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None]),
            torch.cat([p.grad.flatten() for p in closure.parameters() if p.grad is not None])]
        results.append([v.detach().cpu() for v in result])
    differences=[float((l-r).abs().max()) for l,r in zip(*results)]
    if max(differences)>1e-8:raise RuntimeError('Periodic complete joint value/hypergradient regression failed: '+str(differences))
    return dict(loss=float(results[0][0]),loss_max_difference=differences[0],
        probability_gradient_max_difference=differences[1],closure_gradient_max_difference=differences[2],
        native_duration_ms=5,actual_two_step_inner_outer=True)


def periodic_marginal_regression(device):
    fixture=load(HERE/'periodic_regression.pt');report={}
    for sample in fixture['marginals']:
        b=move(sample['batch'],device);n=len(b['w']);kind=sample['item']['kind'];results=[]
        for native in (False,True):
            net,cal,force,_=B.load(301030,device);A.configure_stage(net,'A')
            if native:
                z=dict(b,h=b['h'][:,None],time=b['w'].new_full((n,1),.5),dt=b['w'].new_full((n,1),.001),
                    start=b['w'].new_zeros(n),end=b['w'].new_ones(n))
                labels={k:b[k] for k in ('y','q') if k in b}
                loss=N.marginal(sample['item']['c'],z,tuple(b[k][:,None] for k in ('v','g','acc')),
                    labels,b['force'][:,None],net,cal,force,kind,6)[0].mean()
            else:loss=A.marginal_kernel()(net.base,sample['item'],b,cal,force,dict(train_degree=6,endpoint_weight=.1))[0].mean()
            loss.backward();results.append((loss.detach(),torch.cat([p.grad.flatten() for p in net.base.parameters() if p.grad is not None])))
        error=[float((l-r).abs().max()) for l,r in zip(*results)]
        if max(error)>1e-8:raise RuntimeError('Periodic A loss/gradient differs: '+str(error))
        report[kind]=dict(loss_difference=error[0],gradient_difference=error[1])
    return report


def transition_clock_test(device):
    sample=load(HERE/'periodic_regression.pt')['transition']['batch'];b=move(sample,device)
    net,cal,_,_=B.load(301030,device);A.configure_stage(net,'B')
    ref,_=A.LM.supervised_pair_nll(net.base,net.head,**b,cal=cal)
    got=N.pair_from_path(net.base,net.head,**b,cal=cal,path_h=b['h'][:,None],path_ctx=b['ctx'][:,None],fractions=b['h'].new_ones(len(b['h']),1))
    if not torch.equal(ref,got):raise RuntimeError('1ms transition identity failed')
    # Independent analytic AR(1) composition, followed by a half-step mixture.
    class Fixed:
        def parameters_at(self,h,ctx,prev):
            return h.new_zeros(len(h)),torch.eye(2).to(h).expand(len(h),2,2)*.5,h.new_zeros(len(h),2),torch.eye(2).to(h).expand(len(h),2,2)*.7
    n=len(b['h']);h=b['h'];ctx=b['ctx'];y=b['y'];q=b['q']
    _,_,_,e0,_=A.LM.whiten_latent(net.base,h,ctx,b['prev_y'],b['prev_q'],cal)
    logq,_,_,e,L=A.LM.whiten_latent(net.base,h,ctx,y,q,cal)
    # Two full advances + 0.5 I/K mixture: all sign rows are already stationary.
    m0=.25*e0;m1=.5*m0;m=.5*(m0+m1)
    I=torch.eye(2).to(h);C0=(.49+.25*.49)*I;C1=.25*C0+.49*I
    C=.5*(C0+C1)+.25*(m0-m1)[...,None]*(m0-m1)[...,None,:]
    chol=torch.linalg.cholesky(C);white=torch.linalg.solve_triangular(chol,(e-m)[...,None],upper=False)[...,0]
    expected=-logq[torch.arange(n,device=h.device),q]+.5*white.square().sum(-1)+math.log(2*math.pi)+chol.diagonal(dim1=-2,dim2=-1).log().sum(-1)+L.diagonal(dim1=-2,dim2=-1).log().sum(-1)
    actual=N.pair_from_path(net.base,Fixed(),h,ctx,b['prev_y'],b['prev_q'],h,ctx,y,q,cal,
        h[:,None].expand(-1,3,-1,-1),ctx[:,None].expand(-1,3,-1),h.new_tensor([1.,1.,.5]).expand(n,3))
    error=float((expected-actual).abs().max())
    if error>1e-9:raise RuntimeError('Native latent clock composition differs from analytic moment calculation')
    return dict(one_ms_bitwise=True,analytic_2_5ms_nll_error=error,pseudo_labels_used=False)


def cf_ab(case,device):
    report={};z=case['z']
    for stage in ('A','B'):
        net,cal,force,_=B.load(301030,device);net.float();A.configure_stage(net,stage)
        frozen={k:v.detach().clone() for k,v in net.named_parameters() if not v.requires_grad}
        if stage=='A':
            batches=[]
            for code,kind in ((0,'free'),(1,'cap')):
                ix=torch.where((z['pid']%5==2)&(z['state']==code))[0][:4]
                batches.append((kind,T._view(case,ix,torch.float32)))
            def loss():
                values=[]
                for kind,(zz,jets,labels,forces) in batches:
                    values.append(N.marginal(case['c'],zz,jets,labels,forces,net,cal,force,kind,6)[0].mean())
                return .5*sum(values)
        else:
            prev,cur,_=clock.pairs({k:v.detach().cpu() for k,v in z.items()},(z['pid']%5==2).cpu())
            prev,cur=prev[:8].to(device),cur[:8].to(device)
            zz={k:v.float() if v.is_floating_point() else v for k,v in z.items()}
            labels={k:v.float() if v.is_floating_point() else v for k,v in case['labels'].items()}
            jets=tuple(j.float() for j in case['jets'])
            def loss():return N.pair_loss(case['c'],zz,jets,labels,net,cal,prev,cur).mean()
        opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
        losses=[];gradients=[]
        for step in range(5):
            opt.zero_grad(set_to_none=True);value=loss();losses.append(float(value.detach()))
            if step==4:break
            value.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],5.,error_if_nonfinite=True)
            gradients.append(float(norm));opt.step()
        if not all(math.isfinite(x) for x in losses+gradients) or min(gradients)<=0 or losses[-1]>=losses[0]:
            raise RuntimeError('CF supervised '+stage+' loss/gradient check failed: '+str(losses))
        for k,v in net.named_parameters():
            if k in frozen and (v.grad is not None or not torch.equal(v,frozen[k])):raise RuntimeError('Frozen '+stage+' weights changed')
        report[stage]=dict(losses=losses,gradient_norms=gradients,train_dtype='float32',discarded=True,
            real_native_endpoints=True,no_pseudo_latent_labels=True)
    return report


def cf_joint(case,device,full_stability=True):
    net,*_=B.load(301030,device);A.configure_stage(net,'joint')
    torch.manual_seed(401113);closure=J.ClosureNet().to(device);op=J.closed(case,closure)
    start=J.valid_starts(case,'F')[0];w=J.window(case,start,'F',pid_limit=2);s=T.settings('smoke')['joint'];k=J.kernels(w)
    q=case['a'][start]
    if not torch.equal(op.raw_rhs(op.raw(q)),case['op'].raw_rhs(op.raw(q))):raise RuntimeError('Zero closure changed channel RHS')
    _,diagnostic=op.tendency(op.raw(q),True)
    opt=torch.optim.Adam([dict(params=list(net.parameters()),lr=1e-5),dict(params=list(closure.parameters()),lr=1e-4)])
    losses=[];metrics=[];norms=[]
    for step in range(2):
        opt.zero_grad(set_to_none=True);objective,detail=k['train_window'](w,op,net,'joint',s)
        losses.append(float(objective.detach()));metrics.append(T.values(detail))
        objective.backward()
        norms.append([float(torch.nn.utils.clip_grad_norm_(g['params'],5.,error_if_nonfinite=True)) for g in opt.param_groups])
        if step==0:opt.step()
    if not losses[-1]<losses[0] or any(not math.isfinite(v) or v<=0 for row in norms for v in row):
        raise RuntimeError('CF joint hypergradient/loss reduction failed: '+str(losses))
    if metrics[0]['inner_after']>=metrics[0]['inner_before']:raise RuntimeError('Two-step inner fitting failed to improve')
    stable=J.stability(case,op,s) if full_stability else None
    if stable is not None and not stable['passed']:raise RuntimeError('CF full-window closure stability failed')
    return dict(window_start=start,window_native_intervals=w['length'],duration_s=w['length']*case['dt'],
        effective_inner_lr=w['effective_inner_lr'],
        losses=losses,gradient_norms_probability_closure=norms,metrics=metrics,
        internal_force_cancellation=float(diagnostic['internal_force_error'].detach()),zero_closure_identity=True,
        full_window_stability=stable,two_inner_steps=True,disjoint_inner_outer=True)


def frozen_field(device):
    import field_score
    fixture=module('revision_fixture',FROZEN/'probability_test_fixtures.py').synthetic_data()
    z=move(fixture['z'],device);z['pid']=z['pid'].clone();z['pid'][z['pid']==2]=3;z['pid'][z['pid']==7]=8
    c,op,cold,prior=F.prepare_physics(fixture['meta']['physics_config'],6,device=device)
    field=F.make_field(c,cold,301030,duration_s=.006);net,cal,force,_=B.load(301030,device);before=state(net)
    metric=field_score.score(c,z,net,cal,force,lambda zz:F.queries(c,field,zz),'I',2,backward=True)
    a,adot=field.coefficients_and_dt(torch.arange(13).to(cold)*.0005)
    nfit=float((z['end']-z['start'])[P.group_mask(z['pid'],'I')].sum())
    residual,_=F.residual_loss(a,adot,op,prior,nfit);reg,_=field.regularization();(residual+reg).backward()
    norm=float(torch.stack([p.grad.square().sum() for p in field.parameters() if p.grad is not None]).sum().sqrt())
    if not math.isfinite(norm) or norm<=0:raise RuntimeError('Native-clock PINN gradient failed')
    if any(p.requires_grad or p.grad is not None for p in net.parameters()) or any(not torch.equal(v,before[k]) for k,v in state(net).items()):
        raise RuntimeError('Frozen probability changed in PINN backward')
    return dict(field_gradient_l2=norm,probability_unchanged=True,native_clock_adapter='boundary_v2')


def stage_resume(case,device):
    p=dict(config=dict(profile='smoke'),settings=T.settings('smoke'),fingerprint='discarded-resume-check')
    with tempfile.TemporaryDirectory(prefix='unified-stage-resume-') as folder:
        root=Path(folder).resolve()
        if not root.is_relative_to(Path(tempfile.gettempdir()).resolve()):raise RuntimeError('Invalid temporary test root')
        a=T.run_stage(root,case,J,301030,'A',p,device=device);a_sha=sha(a)
        b=T.run_stage(root,case,J,301030,'B',p,parent=a,device=device);b_sha=sha(b)
        T.run_stage(root,case,J,301030,'A',p,device=device)
        T.run_stage(root,case,J,301030,'B',p,parent=a,device=device)
        if sha(a)!=a_sha or sha(b)!=b_sha:raise RuntimeError('Completed stage changed bytes on resume')
    return dict(completed_parent_and_child_hashes_unchanged=True,actual_stage_checkpoints=True,discarded=True)


def run(device='cpu',cases=(113,175,217),full_stability=True):
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    start=time.perf_counter();report=dict(device=device,cuda_tested=device=='cuda',torch_version=torch.__version__,
        device_name=torch.cuda.get_device_name() if device=='cuda' else 'CPU',production_models_written=False)
    import validate
    report['legacy_periodic']=validate.periodic_regression(device)
    report['periodic_A_native_parity']=periodic_marginal_regression(device)
    report['transition_clock']=transition_clock_test(device)
    report['periodic_joint']=periodic_joint_regression(device)
    report['frozen_probability_pinn']=frozen_field(device)
    report['cases']=[]
    for q in cases:
        name=f'cf_2K_Q{q}';print('acceptance',name,'A/B',flush=True);case=J.load_case(name,device)
        ab=cf_ab(case,device);print('acceptance',name,'joint/full-window',flush=True)
        joint=cf_joint(case,device,full_stability)
        resume=stage_resume(case,device) if q==217 else None
        report['cases'].append(dict(case=name,supervised=ab,joint=joint,stage_resume=resume))
        del case;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    report.update(passed=True,seconds=time.perf_counter()-start,formal_budget_training_executed=False)
    return report
