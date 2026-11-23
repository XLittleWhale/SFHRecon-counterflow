"""Synthetic probability gradient, causality, split and restart checks."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cf_reference_flow import ChannelConfig
import field_adapter as F
import native_probability as P
import train_probability as T


def synthetic_data():
    config = ChannelConfig(vn_mean_m_s=.00602109043344241).metadata()
    c = F.probability_physics(config, 6)
    pids = torch.tensor([2,7,1,6]).repeat_interleave(3)
    frame = torch.arange(3).repeat(4)
    start = frame.double()*2
    time = start[:, None]+torch.tensor([.5,1.5], dtype=torch.float64)[None]
    n, m = time.shape
    state = torch.tensor([0,1,0,1]).repeat_interleave(3)
    z = dict(pid=pids, state=state, frame=frame, segment=pids.clone(),
             start=start, end=start+2, time=time, dt=torch.full((n,m), .001, dtype=torch.float64),
             x=torch.full((n,m,2), .004, dtype=torch.float64), row_id=torch.arange(n),
             w=torch.tensor([[.03,.7]], dtype=torch.float64).repeat(n,1),
             h=torch.zeros(n,m,5,4, dtype=torch.float64),
             target=torch.tensor([[.03,1.3]], dtype=torch.float64).repeat(n,1))
    velocity = torch.zeros(n,m,2,2, dtype=torch.float64)
    velocity[...,0,1] = config['vn_mean_m_s']
    velocity[...,1,1] = config['vs_mean_m_s']
    gradient = torch.zeros(n,m,2,2,2, dtype=torch.float64)
    acceleration = torch.zeros_like(velocity)
    return dict(c=c,z=z,jets=(velocity,gradient,acceleration),
                meta=dict(case='synthetic', physics_config=config, intervals=6))


class NativeProbabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cal, cls.force = P.calibration()

    def test_initialization_is_independent_of_callers_default_dtype(self):
        initial_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            a = P.NativeNetwork(123)
            torch.set_default_dtype(torch.float64)
            b = P.NativeNetwork(123)
            self.assertEqual(torch.get_default_dtype(), torch.float64)
            for name,value in a.state_dict().items():
                torch.testing.assert_close(value,b.state_dict()[name],rtol=0,atol=0)
        finally:
            torch.set_default_dtype(initial_dtype)

    def test_frozen_distribution_exactly_matches_baseline(self):
        data = synthetic_data()
        spec = importlib.util.spec_from_file_location('_native_test_unmodified_baseline', P.BASELINE)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        net = P.NativeNetwork(42).freeze()
        ix = torch.where(data['z']['pid'] == 7)[0][:2]
        previous_a = previous_b = None
        for row in ix:
            zz = P.subset(data['z'], row[None])
            jets = T.query(data, zz)
            a = P.captured_distribution(data['c'],zz,jets,net,self.cal,self.force,3,previous_a)
            b = old.captured_distribution(data['c'],zz,jets,net,self.cal,self.force,3,previous_b)
            for aa,bb in zip(a[:3],b[:3]):
                torch.testing.assert_close(aa,bb,rtol=0,atol=0)
            previous_a,previous_b = a[3],b[3]
            for key in previous_a:
                torch.testing.assert_close(previous_a[key],previous_b[key],rtol=0,atol=0)
        net.enable_training()
        with self.assertRaisesRegex(RuntimeError,'frozen'):
            old.captured_distribution(data['c'],zz,jets,net,self.cal,self.force,3)

    def test_all_model_families_receive_endpoint_gradients(self):
        data = synthetic_data()
        net = P.NativeNetwork(42).enable_training()
        T.normalize_from_F(net, data)
        settings = T.settings_with_defaults(dict(smoke=True,epochs=1,minimum_epochs=1,
                                                 train_degree=3,validation_degree=3,batch=2,batch_pids=1))
        opt = torch.optim.AdamW(net.parameters(),lr=1e-4)
        original = T._state(net)
        result = T.train_epoch(net,data,self.cal,self.force,opt,settings,torch.Generator().manual_seed(3))
        self.assertEqual(result['rows'], 6)
        self.assertEqual(result['updates'], 3)
        for prefix in ('base.gru.','base.free_output.','head.network.'):
            self.assertTrue(any(not torch.equal(value,original[name])
                for name,value in net.state_dict().items() if name.startswith(prefix)),prefix)
        self.assertTrue(all(torch.isfinite(p).all() for p in net.parameters()))
        net.eval()
        zz = P.subset(data['z'],torch.tensor([0]))
        with self.assertRaisesRegex(RuntimeError,'explicit'):
            P.free_distribution(data['c'],zz,T.query(data,zz),net,self.cal,self.force)

    def test_endpoint_is_scored_before_posterior_update(self):
        data = synthetic_data()
        net = P.NativeNetwork(42).freeze()
        zz = P.subset(data['z'],torch.tensor([3]))
        altered = dict(zz,target=zz['target']+torch.tensor([[.2,-.1]]))
        a = P.captured_distribution(data['c'],zz,T.query(data,zz),net,self.cal,self.force,3)
        b = P.captured_distribution(data['c'],altered,T.query(data,altered),net,self.cal,self.force,3)
        for aa,bb in zip(a[:3],b[:3]):
            torch.testing.assert_close(aa,bb,rtol=0,atol=0)
        self.assertFalse(torch.equal(a[3]['emean'],b[3]['emean']))

    def test_normalization_and_sequences_do_not_use_V_or_future(self):
        data = synthetic_data()
        net = P.NativeNetwork(1)
        report = T.normalize_from_F(net,data)
        original_cm, original_cs = net.base.cm.clone(),net.base.cs.clone()
        vmask = P.group_mask(data['z']['pid'],'V')
        data['jets'][0][vmask] += 100
        data['jets'][1][vmask] += 100
        T.normalize_from_F(net,data)
        torch.testing.assert_close(net.base.cm,original_cm,rtol=0,atol=0)
        torch.testing.assert_close(net.base.cs,original_cs,rtol=0,atol=0)
        self.assertEqual(set(report['pids']),{2,7})
        index,valid,reset = next(P.sequence_packs(data['z'],
            P.group_mask(data['z']['pid'],'F') & (data['z']['state']==1),1))
        self.assertEqual(index.tolist(),[[3,4,5]])
        self.assertEqual(reset.tolist(),[[True,False,False]])
        self.assertTrue(valid.all())

    def test_hash_frozen_checkpoint_resume_and_completed_recovery(self):
        data = synthetic_data()
        settings = dict(smoke=True,epochs=2,minimum_epochs=2,patience=2,
                        train_degree=2,validation_degree=2,batch=2,batch_pids=1,threads=1)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'input'
            source.mkdir()
            torch.save(dict(z=data['z'],meta=data['meta']),source/'probability_observations.pt')
            torch.save(dict(row_id=data['z']['row_id'],v=data['jets'][0],g=data['jets'][1],acc=data['jets'][2],
                       meta=dict(probability_observation_sha256=P.sha(source/'probability_observations.pt'))),
                       source/'training_jets.pt')
            first = T.run(source,root/'resumed',17,settings|dict(max_new_epochs=1))
            self.assertFalse(first['complete'])
            self.assertTrue((root/'resumed/last.pt').exists())
            resumed = T.run(source,root/'resumed',17,settings)
            full = T.run(source,root/'full',17,settings)
            a = T._load(root/'resumed/last.pt')
            b = T._load(root/'full/last.pt')
            for name in a['net_state']:
                torch.testing.assert_close(a['net_state'][name],b['net_state'][name],rtol=0,atol=0)
            self.assertEqual(resumed['selected_epoch'],full['selected_epoch'])
            net = P.load_frozen(root/'resumed/selected.pt',expected_sha256=resumed['selected_sha256'])
            self.assertTrue(all(not p.requires_grad for p in net.parameters()))
            (root/'resumed/selection_frozen.json').unlink()
            (root/'resumed/complete.json').unlink()
            recovered = T.run(source,root/'resumed',17,settings)
            self.assertEqual(recovered['epoch'],2)
            self.assertEqual(recovered['selected_sha256'],resumed['selected_sha256'])
            self.assertFalse((root/'resumed/training.lock').exists())


if __name__ == '__main__':
    unittest.main()
