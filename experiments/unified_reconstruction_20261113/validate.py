"""Numerical checks; reports coverage rather than claiming full method acceptance."""
import unittest
import tempfile
import numpy as np
from runtime import *
import labels
import clock_contract as clock
import force_context as force
import baseline_api as baseline
import method_contract as contract

def field_backward(device):
    fixture=module('revision_fixture',FROZEN/'probability_test_fixtures.py').synthetic_data()
    z=move(fixture['z'],device);z['pid']=z['pid'].clone()
    z['pid'][z['pid']==2]=3;z['pid'][z['pid']==7]=8
    c,op,cold,prior=F.prepare_physics(fixture['meta']['physics_config'],6,device=device)
    field=F.make_field(c,cold,301030,duration_s=.006)
    net,cal,force,_=B.load(301030,device);before=state(net)
    likelihood=P.score(c,z,net,cal,force,lambda zz:F.queries(c,field,zz),'I',2,backward=True)
    a,adot=field.coefficients_and_dt(torch.arange(13).to(cold)*.0005)
    nfit=float((z['end']-z['start'])[P.group_mask(z['pid'],'I')].sum())
    pde,_=F.residual_loss(a,adot,op,prior,nfit);reg,_=field.regularization();(pde+reg).backward()
    norm=float(torch.stack([v.grad.square().sum() for v in field.parameters() if v.grad is not None]).sum().sqrt())
    if not np.isfinite(norm) or norm<=0:raise RuntimeError('Field gradient failed')
    if any(v.requires_grad or v.grad is not None for v in net.parameters()) or any(
        not torch.equal(v,before[k]) for k,v in state(net).items()):raise RuntimeError('Frozen probability changed')
    return dict(field_gradient_l2=norm,probability_parameters_equal=True,diagnostic_GH=2)

def periodic_regression(device='cpu'):
    fixture=load(HERE/'periodic_regression.pt')
    report={}
    for sample in fixture['marginals']:
        net,cal,fixed,_=B.load(301030,device);baseline.configure_stage(net,'A')
        net.double();batch=move(sample['batch'],device)
        loss,label,endpoint=baseline.marginal_kernel()(net.base,sample['item'],batch,cal,fixed,dict(train_degree=6,endpoint_weight=.1))
        objective=loss.mean();objective.backward()
        values=dict(objective=float(objective.detach()),label=float(label.mean().detach()),endpoint=float(endpoint.mean().detach()),
                    grad_l2=float(torch.stack([p.grad.square().sum() for p in net.base.parameters() if p.grad is not None]).sum().sqrt()))
        if any(not np.isclose(values[k],v,atol=1e-7,rtol=1e-7) for k,v in sample['expected'].items()):
            raise RuntimeError('Original periodic marginal value/gradient regression failed')
        if any(p.grad is not None for p in net.head.parameters()):raise RuntimeError('Stage A updated the transition head')
        report[sample['item']['kind']]=values
    sample=fixture['transition'];net,cal,_,_=B.load(301030,device);baseline.configure_stage(net,'B')
    batch=move(sample['batch'],device)
    loss,_=baseline.LM.supervised_pair_nll(net.base,net.head,**batch,cal=cal)
    loss.mean().backward()
    values=dict(objective=float(loss.mean().detach()),grad_l2=float(torch.stack([
        p.grad.square().sum() for p in net.head.parameters() if p.grad is not None]).sum().sqrt()))
    if any(not np.isclose(values[k],v,atol=1e-7,rtol=1e-7) for k,v in sample['expected'].items()):
        raise RuntimeError('Original periodic transition value/gradient regression failed')
    if any(p.grad is not None for p in net.base.parameters()):raise RuntimeError('Stage B changed its frozen base')
    report['transition']=values
    return report

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2);torch.set_default_dtype(torch.float64)

    def test_signs_use_actual_table_not_index_parity(self):
        table=np.array([-1,-1,1,1,-1]);attached=np.array([0,1,2,3,4,-1]);state=np.array([1,1,1,1,1,0])
        np.testing.assert_array_equal(labels.sign_labels(table,attached,state),[0,0,1,1,0,-1])
        with self.assertRaises(ValueError):labels.sign_labels(table,np.array([5]),np.array([1]))

    def test_changed_solver_cannot_use_fixed_slot_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';solver=root/'solver';solver.mkdir()
            cert=read(HERE/'fixed_slot_certificate.json')
            dump(source/'configuration.json',dict(version=cert['version'],implementation_sha256=cert['implementation_sha256']))
            dump(source/'particles/metadata.json',{})
            first=next(iter(cert['implementation_sha256']));(solver/first).write_text('changed',encoding='utf8')
            with self.assertRaises(RuntimeError):labels.verify_source(source,solver)

    def test_only_streamwise_geometry_wraps(self):
        p=np.array([[.001,.0001]]);anchors=np.array([[[.015,.0079],[.015,.0079]]])
        y,sym=labels.geometry_labels(anchors,p,np.array([1]),.016,.008,1e-6)
        np.testing.assert_allclose(y,[[14000,-200]],atol=1e-9);self.assertEqual(sym,0)

    def test_native_windows_and_old_transition_rejection(self):
        self.assertEqual(clock.native_window_intervals(1000),5)
        self.assertEqual(clock.native_window_intervals(90),2)
        self.assertEqual(clock.native_window_intervals(180),2)
        clock.require_one_ms(torch.ones(3))
        for fps in (90,180):
            with self.assertRaises(RuntimeError):clock.require_one_ms(torch.full((3,),1000/fps))
        masks=clock.split_reference(torch.arange(100),'F')
        self.assertFalse(bool((masks['inner']&masks['outer']).any()))
        self.assertTrue(torch.equal(masks['inner']|masks['outer'],masks['permitted']))

    def test_force_definition_zero_and_reversal(self):
        fixture=module('revision_fixture',FROZEN/'probability_test_fixtures.py').synthetic_data()
        c=fixture['c'];full=tuple(v[:,0] for v in fixture['jets'])
        altered=(full[0]+.001,full[1],full[2]+.003)
        state=torch.zeros(len(full[0]),dtype=torch.long);q=state
        zero=force.force_correction(c,full,full,state,q)
        a=force.force_correction(c,full,altered,state,q);b=force.force_correction(c,altered,full,state,q)
        self.assertTrue(torch.equal(zero,torch.zeros_like(zero)))
        self.assertTrue(torch.allclose(a,-b,atol=1e-13,rtol=0))
        self.assertGreater(float(a.norm()),0)

    def test_freezing_is_stage_specific_and_explicit(self):
        net,*_=B.load(301030,'cpu')
        for stage in ('A','B','closure','joint','PINN'):
            baseline.configure_stage(net,stage);baseline.assert_stage_freezing(net,stage)
        with self.assertRaises(RuntimeError):baseline.assert_stage_freezing(net,'joint')

    def test_original_periodic_loss_and_gradient(self):periodic_regression()

    def test_formal_training_requires_runtime_acceptance(self):
        self.assertTrue(contract.training_readiness()['implementation_ready'])
        with self.assertRaises(RuntimeError):contract.require_training()

    def test_frozen_probability_allows_field_backward(self):field_backward('cpu')

def run(device='cpu'):
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise RuntimeError('Revision regression checks failed')
    regression=periodic_regression(device)
    field=field_backward(device) if device=='cuda' else dict(covered_by_cpu_unit_test=True)
    import cf_label_diagnostics
    labels=cf_label_diagnostics.run(device)
    return dict(passed=True,tests_run=result.testsRun,device=device,periodic_marginal_regression=regression,
        native_transition_training_passed=False,channel_joint_training_passed=False,
        cf_supervised_label_component_probes=labels,
        frozen_probability_field_backward=field,formal_training_accepted=False,cuda_tested=device=='cuda')

if __name__=='__main__':run()
