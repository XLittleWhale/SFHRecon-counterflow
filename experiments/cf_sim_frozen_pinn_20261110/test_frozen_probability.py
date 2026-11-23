"""Exact historical tensors, portable loading and frozen input gradients."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
import frozen_probability as F
import native_probability as P
from diagnose_probability import cold_queries


class FrozenProbabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_original_three_mappings_and_all_tensors_unchanged(self):
        for field_seed, (training_seed, expected) in F.MODELS.items():
            net, _, _, digest = F.load(field_seed)
            row = F.record(field_seed)
            saved = torch.load(row['path'], map_location='cpu', weights_only=False)
            self.assertEqual(row['training_seed'], training_seed)
            self.assertEqual(digest, expected)
            self.assertEqual(len(net.state_dict()),28)
            self.assertEqual(sum(p.numel() for p in net.parameters()),18717)
            self.assertFalse(net.training)
            self.assertTrue(all(not p.requires_grad for p in net.parameters()))
            for name,value in saved['net_state'].items():
                torch.testing.assert_close(value,net.state_dict()[name],rtol=0,atol=0)
            with self.assertRaises(KeyError):
                _ = saved['seed']  # Historical packages deliberately use training_seed.
        with self.assertRaises(ValueError):
            F.load(411040)

    def test_portable_gate_paths_and_tampered_asset_rejected(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            folder = base/'outputs/PINNTwoFluid_20261105/probability_selection'
            folder.mkdir(parents=True)
            (folder/'models').mkdir()
            for path in F.source_paths():
                if path.is_relative_to(F.FOLDER):
                    shutil.copyfile(path,folder/path.relative_to(F.FOLDER))
            with patch.object(F,'BASE',base), patch.object(F,'FOLDER',folder):
                # The archived gate still contains inaccessible Windows paths;
                # the loader resolves only the relocated sealed asset paths.
                model, _, _, _ = F.load(301030)
                self.assertFalse(model.training)
                checkpoint = folder/'models/seed_411040.pt'
                checkpoint.write_bytes(checkpoint.read_bytes()+b'changed')
                with self.assertRaisesRegex(RuntimeError,'asset hash mismatch'):
                    F.load(301030)

    def test_cpu_network_outputs_match_unmodified_original(self):
        spec = importlib.util.spec_from_file_location('_original_probability_for_frozen_test',P.BASELINE)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        for seed in F.MODELS:
            new, _, _, _ = F.load(seed)
            original = old.FrozenNetwork()
            original.load_state_dict(new.state_dict(),strict=True)
            original.freeze()
            h = torch.zeros(2,5,4,dtype=torch.float64)
            context = new.base.cm[None].repeat(2,1)
            torch.testing.assert_close(new.base.encode(h,context),original.base.encode(h,context),rtol=0,atol=0)
            torch.testing.assert_close(new.base.encode_free(h,context),original.base.encode_free(h,context),rtol=0,atol=0)
            for actual,expected in zip(new.head.parameters_at(h,context,context),original.head.parameters_at(h,context,context)):
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_frozen_context_gradient_and_normalization_unchanged(self):
        net, _, _, _ = F.load(301030)
        before = {name:value.clone() for name,value in net.state_dict().items()}
        context = net.base.cm[None].repeat(2,1).requires_grad_(True)
        h = torch.zeros(2,5,4,dtype=torch.float64,requires_grad=True)
        value = net.base.encode(h,context).sum()+net.base.encode_free(h,context).sum()
        value.backward()
        self.assertTrue(torch.isfinite(context.grad).all())
        self.assertGreater(float(context.grad.abs().max()),0.)
        self.assertTrue(all(p.grad is None for p in net.parameters()))
        for name,old in before.items():
            torch.testing.assert_close(old,net.state_dict()[name],rtol=0,atol=0)

    def test_cold_context_walls_center_and_material_acceleration(self):
        c = dict(physics=dict(vn_mean_m_s=.006,vs_mean_m_s=-.0074,width_m=.016))
        z = dict(x=torch.tensor([[[0.,.001],[.008,.002],[.016,.003]]],dtype=torch.float64))
        v,g,acc = cold_queries(c,z)
        torch.testing.assert_close(v[0,:,0,1],torch.tensor([0.,.009,0.],dtype=torch.float64))
        torch.testing.assert_close(v[0,:,1,1],torch.full((3,),-.0074,dtype=torch.float64))
        torch.testing.assert_close(g[0,:,0,1,0],torch.tensor([2.25,0.,-2.25],dtype=torch.float64))
        self.assertEqual(float(acc.abs().max()),0.)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA hardware unavailable')
    def test_cuda_frozen_history_and_context_backward(self):
        net, _, _, _ = F.load(301030,'cuda')
        context = net.base.cm[None].repeat(2,1).requires_grad_(True)
        h = torch.zeros(2,5,4,dtype=torch.float64,device='cuda',requires_grad=True)
        (net.base.encode(h,context).sum()+net.base.encode_free(h,context).sum()).backward()
        self.assertTrue(torch.isfinite(context.grad).all())
        self.assertTrue(torch.isfinite(h.grad).all())
        self.assertTrue(all(p.grad is None for p in net.parameters()))


if __name__ == '__main__':
    unittest.main()
