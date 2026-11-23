"""Two discarded non-target optimization steps, with independent backprop wiring.

The two arms run sequentially in fresh processes so their Windows peak working
sets and costs are separately measurable. No target reference or H is scored;
no scientific checkpoint is retained. This is a finite-gradient/memory smoke
test, not hyperparameter selection or evidence of reconstruction improvement.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from pinn_common import OUT, HERE, make_context, setup, group_mask, plain_state, dump, read, sha
from field_model import NeuralField, residual_loss, symmetric_regularization
from global_spline import GlobalSpline
from continuous_query import score_nodes


def memory_bytes():
    if os.name == 'nt':
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return dict(working_set_bytes=int(counters.WorkingSetSize),
                    peak_working_set_bytes=int(counters.PeakWorkingSetSize),
                    private_commit_proxy_bytes=int(counters.PagefileUsage),
                    peak_commit_proxy_bytes=int(counters.PeakPagefileUsage))
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(peak_working_set_bytes=int(peak if sys.platform == 'darwin' else peak*1024))


def finite(value, label):
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(label+' is nonfinite')
    elif not math.isfinite(value):
        raise FloatingPointError(label+' is nonfinite')


def run_arm(arm, *, steps=2):
    setup()
    wall0, cpu0 = time.perf_counter(), time.process_time()
    d, cfg, old, probability, cal, force = make_context(.0065, 301030)
    if not math.isclose(d['c']['physics']['vn_mean_m_s'], .0065, rel_tol=0., abs_tol=1e-14) or d['intervals'] != 95:
        raise AssertionError('Only the complete non-target 6.5 mm/s window is permitted')
    if any(p.requires_grad for p in probability.parameters()):
        raise AssertionError('Probability model must be frozen')
    probability_before = plain_state(probability)
    cold, ring = old.theta.detach().clone(), old.prior.ring.detach().clone()
    cls = {'global_spline': GlobalSpline, 'pinn': NeuralField}[arm]
    field = cls(cold, ring, seed=301030, scale_power=1.0)
    optimizer = torch.optim.Adam(field.parameter_groups(lr=.001))
    nodes = torch.arange(191, dtype=torch.float64)*.0005
    if not bool(group_mask(d['z']['pid'], 'V').all()):
        raise AssertionError('Preflight observations must contain only non-target V PIDs')
    nfit = int(group_mask(d['z']['pid'], 'V').sum())
    report = dict(arm=arm, case='vn0_0065m_s', inversion_group='V',
        optimization_steps=steps, diagnostic_only=True, target_reference_read=False,
        target_H_scored=False, discarded_weights=True, learning_rate=.001,
        node_count=191, particle_intervals=95, nfit=nfit,
        field_parameters=sum(p.numel() for p in field.parameters()),
        probability_parameters_frozen=True, before_memory=memory_bytes(), trace=[])
    for iteration in range(1, steps+1):
        tick, cpu_tick = time.perf_counter(), time.process_time()
        optimizer.zero_grad(set_to_none=True)
        before_parameters = [p.detach().clone() for p in field.parameters()]
        before_states = field.states().detach().clone()
        a, adot = field.coefficients_and_dt(nodes)
        finite(a, 'field'); finite(adot, 'SI derivative')
        midpoint, midpoint_dt = a[1::2], adot[1::2]
        value_leaf = midpoint.detach().requires_grad_(True)
        derivative_leaf = midpoint_dt.detach().requires_grad_(True)
        likelihood = score_nodes(value_leaf, derivative_leaf, d, probability, cal, force,
                                 'V', degree=12, backward=True)
        likelihood_wall = time.perf_counter()-tick
        for key, value in likelihood.items():
            finite(value, 'NLL '+key)
        value_gradient = value_leaf.grad
        derivative_gradient = derivative_leaf.grad
        if value_gradient is None or derivative_gradient is None:
            raise AssertionError('Frozen probability failed to differentiate both value and derivative inputs')
        finite(value_gradient, 'probability value gradient')
        finite(derivative_gradient, 'probability derivative gradient')
        physics_loss, physics_detail = residual_loss(a, adot, old.op, old.std, old.prior, nfit)
        space_loss, space_detail = symmetric_regularization(a[::2], ring)
        regularization = physics_loss+space_loss
        finite(regularization, 'PDE+uniform regularization')
        torch.autograd.backward([midpoint, midpoint_dt, regularization],
                                [value_gradient, derivative_gradient, torch.ones_like(regularization)])
        parameters = list(field.parameters())
        if any(p.grad is None for p in parameters):
            raise AssertionError('A field parameter lacks a gradient tensor')
        for parameter in parameters:
            finite(parameter.grad, 'field parameter gradient')
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        if not bool(gradient_norm > 0):
            raise AssertionError('All field gradients are zero')
        optimizer.step()
        after_states = field.states().detach()
        finite(after_states, 'updated field')
        delta = sum((p.detach()-previous).square().sum() for p, previous in zip(parameters, before_parameters)).sqrt()
        if not bool(delta > 0):
            raise AssertionError('Optimizer did not update the field')
        if any(p.grad is not None for p in probability.parameters()):
            raise AssertionError('Probability model received a parameter gradient')
        if any(not torch.equal(value, probability_before[key]) for key, value in probability.state_dict().items()):
            raise AssertionError('Probability model changed during field-only optimization')
        spatial_update = ((after_states-before_states).square().sum(-1)/2).sqrt()*old.op.U*1000
        row = dict(iteration=iteration, likelihood=likelihood,
            objective=.08*likelihood['total']+float(regularization.detach()),
            pde_loss=float(physics_loss.detach()), uniform_regularization=float(space_loss.detach()),
            gradient_norm=float(gradient_norm), parameter_update_norm=float(delta),
            probability_value_gradient_norm=float(value_gradient.norm()),
            probability_derivative_gradient_norm=float(derivative_gradient.norm()),
            field_update_first_rms_mm_s=spatial_update[0].tolist(),
            field_update_last_rms_mm_s=spatial_update[-1].tolist(),
            physics_components={key: (value.detach().tolist() if value.ndim else float(value.detach()))
                                for key, value in physics_detail.items()},
            regularization_components={key: float(value.detach()) for key, value in space_detail.items()},
            likelihood_wall_seconds=likelihood_wall,
            wall_seconds=time.perf_counter()-tick, cpu_seconds=time.process_time()-cpu_tick,
            memory=memory_bytes(), probability_bitwise_unchanged=True)
        report['trace'].append(row)
        print(arm, iteration, 'wall', row['wall_seconds'], 'peak_MiB',
              row['memory']['peak_working_set_bytes']/2**20, flush=True)
        # Graph references are not retained in the scalar-only diagnostic report.
        del a, adot, midpoint, midpoint_dt, value_leaf, derivative_leaf
        del physics_loss, physics_detail, space_loss, space_detail, regularization
        del value_gradient, derivative_gradient, before_parameters, before_states, after_states
        gc.collect()
    report.update(passed=True, probability_bitwise_unchanged=True,
        wall_seconds=time.perf_counter()-wall0, cpu_seconds=time.process_time()-cpu0,
        final_memory=memory_bytes())
    dump(OUT/f'preflight/optimization_{arm}.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('global_spline', 'pinn'))
    args = parser.parse_args()
    if args.arm:
        run_arm(args.arm)
        return
    started = time.perf_counter()
    reports = []
    for arm in ('global_spline', 'pinn'):
        command = [sys.executable, str(Path(__file__).resolve()), '--arm', arm]
        result = subprocess.run(command, check=False)
        if result.returncode:
            raise RuntimeError(f'{arm} optimization preflight failed with code {result.returncode}')
        reports.append(read(OUT/f'preflight/optimization_{arm}.json'))
    report = dict(passed=all(row['passed'] for row in reports),
        target_reference_read=False, target_H_scored=False, steps_per_arm=2,
        diagnostic_weights_discarded=True, science_protocol_modified=False,
        independent_child_processes=True, arms=reports,
        total_wall_seconds=time.perf_counter()-started,
        total_cpu_seconds=sum(row['cpu_seconds'] for row in reports),
        source_sha256=sha(Path(__file__)))
    dump(OUT/'preflight/optimization.json', report)
    print('OPTIMIZATION_PREFLIGHT_PASSED', flush=True)


if __name__ == '__main__':
    main()
