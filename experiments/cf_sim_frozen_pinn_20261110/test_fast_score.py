"""Numerical contract for interval recomputation and larger PID batches."""
import math
import unittest
import torch
import native_probability as P
from fast_score import score
from probability_test_fixtures import synthetic_data


def check_equivalence(device='cpu'):
    torch.set_num_threads(2)
    data = synthetic_data()
    # Three free rows; two captured PIDs with unequal path lengths, a state
    # segment reset and a missing camera interval. Includes real padded rows.
    ix = torch.tensor([0,1,2,3,4,5,3,5])
    z = P.subset(data['z'], ix)
    z['pid'][-2:] = 12
    z['segment'][5] += 100
    z['row_id'] = torch.arange(len(ix))
    z = {k:v.to(device) for k,v in z.items()}
    net = P.NativeNetwork(123, device)
    with torch.no_grad():
        # Exercise persistent posterior gradients beyond zero-initial heads.
        gen = torch.Generator(device=device).manual_seed(77)
        for par in net.head.parameters():
            par.add_(.015*torch.randn(par.shape, generator=gen, device=device, dtype=par.dtype))
    net.freeze()
    cal, force = P.calibration()
    before = {k:v.clone() for k,v in net.state_dict().items()}
    reports, gradients = [], []
    for function, batch_pids in ((P._runtime.score, 1), (score, 2)):
        leaves = tuple(a[ix].to(device).detach().requires_grad_() for a in data['jets'])
        def jets(zz):
            return tuple(a[zz['row_id']] for a in leaves)
        result = function(data['c'], z, net, cal, force, jets, 'F', degree=3,
                          backward=True, metrics=True, batch=2, batch_pids=batch_pids)
        reports.append(result)
        gradients.append(tuple(a.grad.detach().clone() for a in leaves))
    for key in ('total','free','cap'):
        assert math.isclose(reports[0][key],reports[1][key],rel_tol=2e-10,abs_tol=1e-10)
        assert math.isclose(reports[0]['native_nll'][key],reports[1]['native_nll'][key],rel_tol=2e-10,abs_tol=1e-10)
    for name in ('free','cap'):
        for pid, row in reports[0]['per_pid'][name].items():
            for key, value in row.items():
                assert math.isclose(reports[1]['per_pid'][name][pid][key],value,rel_tol=2e-9,abs_tol=1e-9)
    max_error = 0.
    for a,b in zip(*gradients):
        torch.testing.assert_close(a,b,rtol=2e-8,atol=2e-8)
        assert torch.isfinite(b).all()
        max_error = max(max_error, float((a-b).abs().max()))
    assert any(float(g.abs().max()) > 0 for g in gradients[1])
    assert all(torch.equal(v,before[k]) for k,v in net.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in net.parameters())
    return dict(passed=True, device=device, max_absolute_gradient_error=max_error,
                full_sequence_gradient=True, reset_gap_padding=True, frozen_exact=True)


class FastScoreTests(unittest.TestCase):
    def test_checkpointed_field_gradient_and_metrics(self):
        check_equivalence()

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_checkpointed_field_gradient_and_metrics(self):
        check_equivalence('cuda')
