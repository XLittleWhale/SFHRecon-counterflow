"""Behavioral tests for native time, causal inputs, PID isolation and jets."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from data_adapter import DT, ReferenceFields, build_observations, prepare_case, sha256


def fixture_arrays():
    time = np.arange(8, dtype=np.float64) / 90
    pid = np.arange(10)
    xy = np.zeros((8, 10, 2), np.float64)
    xy[..., 0] = .003 + pid[None, :] * .0002 + time[:, None] * .0005
    xy[..., 1] = .0025 + time[:, None] * .0004
    segment = np.broadcast_to(pid, (8, 10)).copy()
    state = np.broadcast_to((pid >= 5).astype(np.int8), (8, 10)).copy()
    attached = np.where(state == 1, np.broadcast_to(pid, state.shape), -1)
    observed = dict(time_s=time, particle_id=pid, xy_m=xy, segment_id=segment)
    hidden = dict(time_s=time.copy(), particle_id=pid.copy(), state=state, attached_vortex=attached)
    config = dict(camera_fps=90, camera_frames=8,
                  config=dict(vn_mean_m_s=.004, width_m=.016, height_m=.008,
                              temperature_K=2., n_particles=10))
    return observed, hidden, config


def write_fixture(root):
    observed, hidden, configuration = fixture_arrays()
    root.mkdir()
    particles = root / "particles"
    fields = particles / "field_truth"
    fields.mkdir(parents=True)
    (root / "configuration.json").write_text(json.dumps(configuration), encoding="utf8")
    np.savez(particles / "particle_trajectories.npz", **observed)
    np.savez(particles / "particle_hidden_truth.npz", **hidden)
    x, y = np.linspace(0, .016, 9), np.arange(4) * .002
    yy, xx = np.meshgrid(y, x, indexing="ij")
    manifest = {}
    for i, t in enumerate(observed["time_s"]):
        # Locally affine fields with a nonsymmetric velocity gradient. Queries
        # stay away from the periodic seam, so bilinear interpolation is exact.
        vn = np.stack((2 * yy + .3 * t, 3 * xx - .2 * t), -1)
        vs = np.stack((-yy + .1 * t, 4 * xx + .5 * t), -1)
        file = fields / f"frame_{i:06d}.npz"
        np.savez(file, x_m=x, y_m=y, time_s=t, vn=vn, vs=vs)
        manifest[f"field_truth/{file.name}"] = sha256(file)
    (particles / "frame_manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    metadata = dict(smoke_only=False, source_flow_status="transient", steady_verified=False,
                    output_sha256={name: sha256(particles / name) for name in
                                   ("particle_trajectories.npz", "particle_hidden_truth.npz", "frame_manifest.json")})
    (particles / "metadata.json").write_text(json.dumps(metadata), encoding="utf8")
    return observed, hidden, configuration


class NativeObservationTests(unittest.TestCase):
    def test_time_tiling_and_full_camera_span(self):
        observed, hidden, config = fixture_arrays()
        z, meta, audit = build_observations(observed, hidden, config)
        self.assertEqual(audit["accepted_intervals"], 60)
        self.assertEqual(z["frame"].unique().tolist(), list(range(1, 7)))
        self.assertAlmostEqual(meta["origin_s"], 1 / 90)
        self.assertAlmostEqual(meta["duration_s"], 6 / 90)
        self.assertEqual(meta["intervals"], 67)
        torch.testing.assert_close(z["dt"].sum(1) / DT, z["end"] - z["start"], rtol=0, atol=1e-10)
        self.assertTrue(bool((z["dt"] <= DT + 1e-12).all()))
        self.assertTrue(bool((z["dt"][:, 0] > 0).all()))
        self.assertEqual(tuple(z["h"].shape[2:]), (5, 4))
        expected_w = torch.tensor([.0005, .0004], dtype=torch.float64) / .004
        torch.testing.assert_close(z["w"], expected_w.expand_as(z["w"]), atol=1e-12, rtol=0)

    def test_future_endpoint_cannot_change_causal_features(self):
        observed, hidden, config = fixture_arrays()
        before, _, _ = build_observations(observed, hidden, config)
        altered = {key: value.copy() for key, value in observed.items()}
        altered["xy_m"][4, 2] += np.array([1e-6, -2e-6])
        after, _, _ = build_observations(altered, hidden, config)
        first = (before["pid"] == 2) & (before["frame"] == 3)
        second = (after["pid"] == 2) & (after["frame"] == 3)
        for key in ("x", "w", "h", "time", "dt"):
            torch.testing.assert_close(before[key][first], after[key][second], rtol=0, atol=0)
        self.assertFalse(torch.equal(before["target"][first], after["target"][second]))

    def test_endpoint_gates_allow_mixed_past(self):
        observed, hidden, config = fixture_arrays()
        hidden["state"][0, 2] = 1
        hidden["attached_vortex"][0, 2] = 12
        hidden["attached_vortex"][4:, 7] = 123
        observed["segment_id"][4:, 1] = 1001
        z, _, audit = build_observations(observed, hidden, config)
        self.assertTrue(bool(((z["pid"] == 2) & (z["frame"] == 1)).any()))
        self.assertFalse(bool(((z["pid"] == 7) & (z["frame"] == 3)).any()))
        self.assertFalse(bool(((z["pid"] == 1) & (z["frame"] == 3)).any()))
        self.assertFalse(bool(((z["pid"] == 1) & (z["frame"] == 4)).any()))
        self.assertEqual(audit["rejected"]["captured_endpoint_vortex_change"], 1)

    def test_outside_causal_pilot_is_rejected(self):
        observed, hidden, config = fixture_arrays()
        observed["xy_m"][0, 2, 0] = .014
        observed["xy_m"][1, 2, 0] = .0159
        z, _, audit = build_observations(observed, hidden, config)
        self.assertFalse(bool(((z["pid"] == 2) & (z["frame"] == 1)).any()))
        self.assertGreater(audit["rejected"]["pilot_position_outside_domain"], 0)


class FileContractTests(unittest.TestCase):
    def test_jet_derivative_and_vector_material_acceleration(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            _, _, config = write_fixture(source)
            reader = ReferenceFields(source, config)
            point = np.array([[.004, .003]])
            query = np.array([1.5 / 90])
            velocity, gradient, acceleration = reader.jets(1, query, point)
            expected_v = np.array([[[2 * .003 + .3 * query[0], 3 * .004 - .2 * query[0]],
                                    [-.003 + .1 * query[0], 4 * .004 + .5 * query[0]]]])
            expected_g = np.array([[[[0, 2], [3, 0]], [[0, -1], [4, 0]]]], np.float64)
            expected_a = np.array([[[.3, -.2], [.1, .5]]]) + np.einsum("...fij,...fj->...fi", expected_g, expected_v)
            np.testing.assert_allclose(velocity, expected_v, rtol=0, atol=1e-14)
            np.testing.assert_allclose(gradient, expected_g, rtol=0, atol=1e-13)
            np.testing.assert_allclose(acceleration, expected_a, rtol=0, atol=1e-13)

    def test_physical_pid_split_reference_isolation_and_input_immutability(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "source", Path(temp) / "prepared"
            write_fixture(source)
            before = {str(p.relative_to(source)): sha256(p) for p in source.rglob("*") if p.is_file()}
            audit = prepare_case(source, output)
            after = {str(p.relative_to(source)): sha256(p) for p in source.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            packs = {kind: torch.load(output / f"{kind}_observations.pt", weights_only=False)
                     for kind in ("probability", "inversion", "heldout")}
            prob, inverse, heldout = (packs[k]["z"] for k in ("probability", "inversion", "heldout"))
            self.assertTrue(set((prob["pid"] % 5).tolist()) <= {1, 2})
            self.assertTrue(set((inverse["pid"] % 5).tolist()) <= {1, 3, 4})
            self.assertTrue(set((heldout["pid"] % 5).tolist()) == {0})
            self.assertEqual(set(prob["pid"].tolist()) & set(inverse["pid"].tolist()), {1, 6})
            self.assertFalse(set(prob["pid"].tolist()) & set(heldout["pid"].tolist()))
            self.assertFalse(set(inverse["pid"].tolist()) & set(heldout["pid"].tolist()))
            jets = torch.load(output / "training_jets.pt", weights_only=False)
            self.assertEqual(jets["v"].shape[:2], prob["time"].shape)
            torch.testing.assert_close(jets["row_id"], prob["row_id"])
            for pack in packs.values():
                self.assertNotIn("v", pack["z"])
                self.assertNotIn("last_tension", pack["z"])
            with np.load(output / "reference_evaluation.npz") as reference:
                self.assertEqual(reference["vn"].shape, (7, 32, 65, 2))
                self.assertEqual(float(reference["time_s"][0]), 0)
                self.assertAlmostEqual(float(reference["time_s"][-1]), 6 / 90)
            self.assertEqual(audit["field_frames_verified"], 7)
            self.assertFalse(audit["meta"]["steady_verified"])
            with self.assertRaises(ValueError):
                prepare_case(source, source / "modified")


if __name__ == "__main__":
    unittest.main()
