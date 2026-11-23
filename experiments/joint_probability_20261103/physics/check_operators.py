"""Focused numerical checks of the new periodic and CF evolution operators."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[3]
OUT = BASE/'outputs/JointProbabilityPhysics_20261103'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def source_manifest(extra=()):
    paths = {Path(m.__file__).resolve() for m in sys.modules.values()
             if getattr(m, '__file__', None) and str(getattr(m, '__file__')).endswith('.py')
             and Path(m.__file__).resolve().is_relative_to(BASE)}
    paths.update(Path(p).resolve() for p in extra)
    return [dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(paths)]


def simulation():
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(20261103)
    history = BASE/'outputs/PINNClosureFactorial_20261030'
    sys.path.insert(0,str(history))
    import factorial_model as F
    import prepare_external as P
    from simulation_operator import Momentum, batched_rhs, batched_step
    report = dict(kind='simulation-evolution-checks', cases={}, no_fitting=True,
                  no_reference_fields=True, no_test_scoring=True)
    for U in (.0015,.004):
        d,cfg,old,legacy = P.load_case(U,'cpu',EW=F.EW)
        op = Momentum(old)
        compatibility = Momentum(old,friction_factor=2/3,nu_s=float(old.nu_s))
        a = torch.randn(2,old.d,dtype=torch.float64)*1e-3
        b = torch.randn(op.B.shape[-1],dtype=torch.float64)*1e-4
        legacy_error = float((legacy.rhs(a,b)-compatibility.rhs(a,b)).abs().max())
        b.zero_()
        delta_D = (1-2/3)*op.c['alpha']*op.c['rho_s_kg_m3']*op.c['kappa_m2_s']*op.L0
        slip = a[0]-a[1]
        expected = torch.stack((-delta_D/op.c['rho_n_kg_m3']*slip,
            delta_D/op.c['rho_s_kg_m3']*slip+float(old.nu_s)*op.lap*a[1]))
        rhs_delta_error = float((op.rhs(a,b)-legacy.rhs(a,b)-expected).abs().max())
        aa = torch.stack((a,.9*a))
        batch_error = float((batched_rhs(op,aa)-torch.stack([op.rhs(x,b) for x in aa])).abs().max())
        step_error = float((batched_step(op,aa)-torch.stack([op.step(x,b) for x in aa])).abs().max())
        zeros = torch.zeros(8,old.d,dtype=torch.float64)
        states = F.R.rollout(op,a,zeros,zeros,differentiable=False)
        assert torch.isfinite(states).all()
        # Differentiate the actual inherited closure jets through the new RK4.
        controls = torch.randn(3,2,old.d,dtype=torch.float64)*1e-7
        controls.requires_grad_()
        query = dict(frame=torch.tensor([0,2]),x=a.new_tensor([[.002,.003],[.007,.008]]))
        def objective(x):
            path=F.R.rollout(op,a,x[:,1],x[:,0],differentiable=False)
            velocity,gradient,acceleration=F.R.local_jets(path,x[:,1],x[:,0],query,d['c'])
            return velocity.square().sum()+1e-7*acceleration.square().sum()
        grad = torch.autograd.grad(objective(controls),controls)[0]
        direction=torch.randn_like(controls);direction/=direction.norm();eps=1e-7
        fd=float(((objective(controls+eps*direction)-objective(controls-eps*direction))/(2*eps)).detach())
        ad=float((grad*direction).sum())
        gradient_error=abs(fd-ad)
        report['cases'][P.case_name(U)]=dict(nu_s_m2_s=op.nu_s,Fns_prefactor=op.friction_factor,
            old_nu_s_m2_s=float(old.nu_s),old_Fns_prefactor=2/3,
            old_rhs_reproduction_max_abs=legacy_error,
            new_minus_old_vs_analytic_friction_and_viscosity_max_abs=rhs_delta_error,
            batched_rhs_max_abs=batch_error,batched_rk4_max_abs=step_error,
            short_forward_ms=8,short_forward_all_finite=True,
            observation_jet_directional_gradient_abs_error=gradient_error,
            observation_jet_fd=fd,observation_jet_autograd=ad)
        assert op.nu_s==0 and op.friction_factor==1
        assert max(legacy_error,rhs_delta_error,batch_error,step_error)<1e-10
        assert gradient_error<1e-9
    inherited=[F.__file__,P.__file__,F.R.__file__,F.N.__file__,F.EW.__file__,F.A.__file__]
    inherited += [BASE/'outputs/EquilibriumWeak4DVar_20261029/model.py',
                  BASE/'outputs/NormalNoiseAblation_20261030/noise_model.py']
    report['source_dependencies']=source_manifest(inherited)
    report['passed']=True
    return report


def cf():
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(20261103)
    runtime = OUT/'cf_runtime'
    sys.path.insert(0,str(runtime))
    import b_transfer as B
    import package_tool
    package_tool.verify(runtime)
    B.verify_contract()
    old_path=BASE/'outputs/FrozenB_CFLocalFull_20261031/payload/server_package/counterflow/dynamics.py'
    spec=importlib.util.spec_from_file_location('counterflow.legacy_dynamics_for_check',old_path)
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    report=dict(kind='CF-evolution-checks',cases={},no_fitting=True,no_test_scoring=True,
                full_CF_reconstruction_run=False,source_contract_verified=True,bundle_manifest_verified=True)
    for case in ('cf_113_camera100_f0184','cf_217_camera100_f0333'):
        c,z,op,theta=B.load_case(case)
        original=legacy.make(B.load(B.PACKAGE/'inputs'/case/'operator.pt'),'cpu')
        a=torch.einsum('fij,fj->fi',c['decode'],theta)+torch.randn(2,op.m,dtype=torch.float64)*1e-4
        q0=a.new_zeros(original.B.shape[1])
        old_rhs=original.rhs(torch.cat((a.flatten(),q0)))[:2*op.m].reshape(2,op.m)
        v=op.v0+torch.einsum('fpm,fm->fp',op.V,a).reshape(op.v0.shape)
        delta_force=(op.ck-original.ck)*op.L0*(v[0]-v[1])
        delta_load=torch.einsum('fpm,p->fm',op.Pv,delta_force.flatten())
        expected=torch.stack((delta_load[0]/op.p['rho_n_kg_m3'],-delta_load[1]/op.p['rho_s_kg_m3']))
        delta_error=float((op.rhs(a)-old_rhs-expected).abs().max())
        old_ck=op.ck;op.ck=original.ck
        legacy_error=float((op.rhs(a)-old_rhs).abs().max())
        op.ck=old_ck
        zeros=a.new_zeros(8,op.m)
        path=B.rollout(op,a,zeros,zeros,checkpointed=False)
        assert torch.isfinite(path.a).all()
        report['cases'][case]=dict(nu_s_m2_s=op.nu_s,Fns_prefactor=op.friction_factor,
            old_ck=float(original.ck),new_ck=float(op.ck),ck_ratio=float(op.ck/original.ck),
            super_viscous_matrix_max_abs=float(op.Ds.abs().max()),
            super_viscous_background_max_abs=float(op.ds0.abs().max()),
            old_rhs_reproduction_max_abs=legacy_error,new_minus_old_vs_analytic_friction_max_abs=delta_error,
            short_forward_ms=8,short_forward_all_finite=True)
        assert op.nu_s==0 and op.friction_factor==1
        assert abs(float(op.ck/original.ck)-1.5)<1e-12
        assert max(legacy_error,delta_error)<1e-10
    report['source_dependencies']=source_manifest([old_path])
    report['passed']=True
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('simulation','cf','all'),default='all')
    args=parser.parse_args()
    out=OUT/'physics';out.mkdir(parents=True,exist_ok=True)
    if args.mode=='all':
        for mode in ('simulation','cf'):
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--mode',mode],check=True)
        # The existing adapter test also checks direct eta observation VJPs,
        # original zero-closure compatibility, and spatial projection accuracy.
        subprocess.run([sys.executable,str(OUT/'cf_runtime/test_adapter.py'),
            '--out',str(out/'cf_adapter_checks.json')],check=True)
        reports={name:json.loads((out/name).read_text(encoding='utf8')) for name in
                 ('simulation_operator_checks.json','cf_operator_checks.json','cf_adapter_checks.json')}
        result=dict(passed=all(r['passed'] for r in reports.values()),reports=list(reports),
            historical_results_overwritten=False,full_CF_reconstruction_run=False)
        (out/'physics_checks.json').write_text(json.dumps(result,indent=2),encoding='utf8')
        print(json.dumps(result,indent=2))
    else:
        result=simulation() if args.mode=='simulation' else cf()
        (out/(args.mode+'_operator_checks.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
        print(json.dumps({k:v for k,v in result.items() if k!='source_dependencies'},indent=2))


if __name__=='__main__':
    main()
