"""Fast numerical checks for new-data physics, without any field training."""
from pathlib import Path
import sys
import unittest

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cf_reference_flow import ChannelConfig
import field_adapter as F


class FieldAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = ChannelConfig(vn_mean_m_s=.00602109043344241).metadata()
        cls.c, cls.op, cls.cold, cls.prior = F.prepare_physics(cls.config, 100)

    def test_physics_and_prior(self):
        c, op = self.c, self.op
        self.assertEqual(op.m, 325)
        self.assertEqual(op.nu_s, 0.)
        self.assertEqual(op.friction_factor, 1.)
        self.assertAlmostEqual(op.L0, c["physics"]["line_density_m2"], places=5)
        self.assertEqual(tuple(self.prior.normal_temporal_cholesky.shape), (325, 100, 100))
        self.assertFalse(c["operator_source"]["historical_observations_read"])
        self.assertLess(c["pinn_spatial_normalization"]["normalized_mass_half_identity_error"], 1e-10)
        self.assertLess(c["pinn_spatial_normalization"]["decode_inverse_error"], 1e-10)
        self.assertEqual(float(self.cold.abs().max()), 0.)
        light = F.probability_physics(self.config, 534)
        self.assertNotIn("pinn_decode", light)
        self.assertEqual(light["constants"], c["constants"])

    def test_scalar_and_batched_rhs_match(self):
        rng = torch.Generator().manual_seed(10)
        a = torch.randn(3, 2, 325, dtype=torch.float64, generator=rng)*.0001
        actual = F.batched_rhs(self.op, a)
        expected = torch.stack([self.op.rhs(row) for row in a])
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_background_material_acceleration_is_zero(self):
        z = dict(time=torch.tensor([[3., 10.]], dtype=torch.float64),
                 x=torch.tensor([[[.003, .002], [.011, .004]]], dtype=torch.float64))
        field = F.make_field(self.c, self.cold, seed=4, duration_s=.1)
        v, g, acceleration = F.queries(self.c, field, z)
        self.assertEqual(float(acceleration.abs().max()), 0.)
        legacy = torch.einsum("...ij,...i->...j", g, v)
        self.assertGreater(float(legacy.abs().max()), .001)

    def test_wall_conditions_and_exact_tangent(self):
        field = F.make_field(self.c, self.cold, seed=4, duration_s=.1)
        with torch.no_grad():
            field.output.weight.normal_(0, .001)
        x = torch.tensor([[[0., .001], [.016, .003], [.008, .004]]], dtype=torch.float64)
        z = dict(time=torch.tensor([[23., 47., 51.]], dtype=torch.float64), x=x)
        v, g, a = F.queries(self.c, field, z)
        self.assertLess(float(v[0, :2, 0].abs().max()), 1e-12)
        self.assertLess(float(v[0, :2, 1, 0].abs().max()), 1e-12)
        self.assertLess(float((g[..., 0, 0]+g[..., 1, 1]).abs().max()), 1e-11)
        t = torch.tensor([.023, .047, .051], dtype=torch.float64)
        value, tangent = field.coefficients_and_dt(t)
        eps = 1e-7
        finite = (field(t+eps)-field(t-eps))/(2*eps)
        torch.testing.assert_close(tangent, finite, rtol=2e-6, atol=1e-10)
        (v.square().sum()+a.square().sum()+g.square().sum()).backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in field.parameters()))
        self.assertGreater(float(field.output.weight.grad.abs().max()), 0.)

    def test_full_100ms_residual_backward(self):
        field = F.make_field(self.c, self.cold, seed=4, duration_s=.1)
        times = torch.arange(201, dtype=torch.float64)*.0005
        values, tangents = field.coefficients_and_dt(times)
        loss, details = F.residual_loss(values, tangents, self.op, self.prior, 1000.)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in field.parameters()))

    def test_reject_wrong_geometry_and_material(self):
        with self.assertRaises(ValueError):
            F.probability_physics(dict(self.config, width_m=.01), 100)
        with self.assertRaises(ValueError):
            F.probability_physics(self.config, 535)
        with self.assertRaises(ValueError):
            F.prepare_physics(dict(self.config, mu_n_Pa_s=2*self.config["mu_n_Pa_s"]), 100)


if __name__ == "__main__":
    unittest.main()
