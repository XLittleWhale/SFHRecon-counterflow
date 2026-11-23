"""A simultaneous whole-window Fourier field with no separate initial control.

All 96 nodal controls are optimized together. Fixed cubic Hermite interpolation
uses shared nodal slopes from polynomial-exact finite differences: five-point
centered internally, four-point one-sided at the first/last two nodes. Slopes
are derived from the same controls, never additional trainable parameters.
The endpoints are free; no zero-derivative boundary condition is imposed.
"""
from __future__ import annotations

import torch
from torch import nn

NODE_COUNT = 96
DT_SECONDS = .001
DURATION_SECONDS = (NODE_COUNT-1)*DT_SECONDS


def slope_matrix(count, dt, *, dtype=torch.float64, device=None):
    if count < 5 or dt <= 0:
        raise ValueError('At least five uniformly spaced nodes and positive dt required')
    matrix = torch.zeros((count, count), dtype=dtype, device=device)
    matrix[0, :4] = matrix.new_tensor([-11., 18., -9., 2.])/6
    matrix[1, :4] = matrix.new_tensor([-2., -3., 6., -1.])/6
    for j in range(2, count-2):
        matrix[j, j-2:j+3] = matrix.new_tensor([1., -8., 0., 8., -1.])/12
    matrix[-2, -4:] = -matrix[1, :4].flip(0)
    matrix[-1, -4:] = -matrix[0, :4].flip(0)
    return matrix/dt


class GlobalSpline(nn.Module):
    def __init__(self, cold, ring, seed=0, scale_power=1.0):
        super().__init__()
        cold = cold.detach().clone().double()
        ring = torch.as_tensor(ring, dtype=cold.dtype, device=cold.device)
        if cold.shape != (2, 624) or ring.shape != (624,):
            raise ValueError('Expected cold[2,624], ring[624]')
        if not bool(torch.isfinite(cold).all()) or not bool((ring > 0).all()):
            raise ValueError('Finite cold state and strictly positive rings required')
        self.seed = int(seed)  # Recorded for parity; interpolation has no random features.
        self.scale_power = float(scale_power)
        self.register_buffer('cold', cold)
        self.register_buffer('scale', ring.pow(-self.scale_power))
        self.register_buffer('node_times', torch.arange(NODE_COUNT, dtype=cold.dtype,
                                                       device=cold.device)*DT_SECONDS)
        self.register_buffer('derivative_matrix', slope_matrix(
            NODE_COUNT, DT_SECONDS, dtype=cold.dtype, device=cold.device))
        self.controls = nn.Parameter(cold.new_zeros(NODE_COUNT, *cold.shape))

    def states(self):
        return self.cold[None]+self.scale[None, None]*self.controls

    def coefficients_and_dt(self, t_seconds, create_graph=True):
        t = torch.as_tensor(t_seconds, dtype=self.cold.dtype, device=self.cold.device)
        if t.ndim != 1 or not bool(torch.isfinite(t).all()):
            raise ValueError('Expected finite 1-D SI time coordinates')
        tolerance = 16*torch.finfo(t.dtype).eps
        if bool(((t < -tolerance) | (t > DURATION_SECONDS+tolerance)).any()):
            raise ValueError('Time outside the relative [0,.095] second window')
        # Do not clamp differentiable time: clamp can kill the derivative exactly
        # at either endpoint. The range check permits only roundoff-sized drift;
        # the endpoint polynomial then gives the intended one-sided derivative.
        nodes = self.states()
        slopes = torch.einsum('ij,jfm->ifm', self.derivative_matrix, nodes)
        index = torch.floor(t/DT_SECONDS).long().clamp(0, NODE_COUNT-2)
        s = ((t-self.node_times[index])/DT_SECONDS)[:, None, None]
        y0, y1 = nodes[index], nodes[index+1]
        m0, m1 = slopes[index], slopes[index+1]
        h00, h10 = 2*s**3-3*s**2+1, s**3-2*s**2+s
        h01, h11 = -2*s**3+3*s**2, s**3-s**2
        value = h00*y0+DT_SECONDS*h10*m0+h01*y1+DT_SECONDS*h11*m1
        derivative = ((6*s**2-6*s)*y0+(-6*s**2+6*s)*y1)/DT_SECONDS
        derivative = derivative+(3*s**2-4*s+1)*m0+(3*s**2-2*s)*m1
        if not create_graph:
            return value.detach(), derivative.detach()
        return value, derivative

    def forward(self, t_seconds=None):
        if t_seconds is None:
            return self.states()
        return self.coefficients_and_dt(t_seconds)[0]

    def parameter_groups(self, lr=.001):
        return [dict(params=[self.controls], lr=float(lr), name='field')]
