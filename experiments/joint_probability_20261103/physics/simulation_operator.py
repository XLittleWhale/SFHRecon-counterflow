"""Explicit momentum variant for the 2026-11-03 joint-model experiment.

The historical algebraic-L Momentum discretization is preserved, with only
nu_s and the isotropic F_ns prefactor exposed. Defaults implement the requested
nu_s=0 and prefactor=1. Historical modules and exported tensors are untouched.
"""
import math
import torch
from counterflow_full.core import DT, modes


class Momentum:
    def __init__(self, old, local=False, *, friction_factor=1.0, nu_s=0.0):
        self.__dict__.update(old.__dict__)
        self.legacy_nu_s = float(old.nu_s)
        self.local = bool(local)
        self.friction_factor = float(friction_factor)
        self.nu_s = float(nu_s)
        if self.friction_factor <= 0 or self.nu_s < 0:
            raise ValueError('Positive F_ns prefactor and nonnegative viscosity required')
        c = self.c
        self.L0 = (c['alpha_v']/c['beta_v'])**2*(self.mean[0]-self.mean[1]).square().sum()
        xy, _ = modes(2)
        x = torch.arange(self.n, device=self.mean.device, dtype=self.mean.dtype)*c['box_m']/self.n
        yy, xx = torch.meshgrid(x, x, indexing='ij')
        phase = (torch.stack((xx, yy), -1)@self.mean.new_tensor(xy).T)*2*math.pi/c['box_m']
        self.B = torch.cat((torch.ones_like(xx)[..., None],
                           torch.stack((phase.cos(), phase.sin()), -1).flatten(-2)), -1)

    def inv(self, z):
        return torch.fft.ifft2(z, dim=(-3, -2)).real*self.n**2

    def rhs(self, a, b):
        c = self.c
        vh = torch.zeros(2, self.n, self.n, 2, device=a.device, dtype=torch.complex128)
        z = .5*self.U*(a[:, 1::2]+1j*a[:, ::2])[..., None]*self.perp
        vh[:, self.y, self.x] = z
        vh[:, self.my, self.mx] = z.conj()
        vh[:, 0, 0] = self.mean.to(vh.dtype)
        v = self.inv(vh)
        dx = self.inv(vh*(1j*self.kx)[None, ..., None])
        dy = self.inv(vh*(1j*self.ky)[None, ..., None])
        w = v[0]-v[1]
        L = (c['alpha_v']/c['beta_v'])**2*w.square().sum(-1) if self.local else self.L0
        multiplier = torch.exp(self.B@b)
        D = self.friction_factor*c['alpha']*c['rho_s_kg_m3']*c['kappa_m2_s']*L
        force = -D[..., None]*multiplier[..., None]*w
        r = -v[..., 0, None]*dx-v[..., 1, None]*dy
        rn = r[0]+force/c['rho_n_kg_m3']+c['mu_n_Pa_s']/c['rho_n_kg_m3']*self.inv(-self.k2[..., None]*vh[0])
        rs = r[1]-force/c['rho_s_kg_m3']
        zz = torch.fft.fft2(torch.stack((rn, rs)), dim=(1, 2))[:, self.y, self.x]/self.n**2
        zz = (zz*self.perp).sum(-1)/self.U
        da = torch.stack((2*zz.imag, 2*zz.real), -1).flatten(-2)
        return da+torch.stack((torch.zeros_like(da[0]), -self.nu_s*self.lap*a[1]))

    def step(self, a, b, substeps=2):
        h = DT/substeps
        for _ in range(substeps):
            k1 = self.rhs(a, b)
            k2 = self.rhs(a+h*k1/2, b)
            k3 = self.rhs(a+h*k2/2, b)
            k4 = self.rhs(a+h*k3, b)
            a = a+h*(k1+2*k2+2*k3+k4)/6
        return a


def batched_rhs(op, a):
    """Constant-L, b=0 vectorization; same operator for discrete residual use."""
    if op.local:
        raise ValueError('This vectorized helper is the constant-L B variant')
    c = op.c
    vh = torch.zeros((*a.shape[:-2], 2, op.n, op.n, 2), device=a.device, dtype=torch.complex128)
    z = .5*op.U*(a[..., 1::2]+1j*a[..., ::2])[..., None]*op.perp
    vh[..., op.y, op.x, :] = z
    vh[..., op.my, op.mx, :] = z.conj()
    vh[..., 0, 0, :] = op.mean.to(vh.dtype)
    v = op.inv(vh)
    dx = op.inv(vh*(1j*op.kx)[..., None])
    dy = op.inv(vh*(1j*op.ky)[..., None])
    w = v[..., 0, :, :, :]-v[..., 1, :, :, :]
    D = op.friction_factor*c['alpha']*c['rho_s_kg_m3']*c['kappa_m2_s']*op.L0
    force = -D*w
    r = -v[..., 0, None]*dx-v[..., 1, None]*dy
    rn = r[..., 0, :, :, :]+force/c['rho_n_kg_m3']+c['mu_n_Pa_s']/c['rho_n_kg_m3']*op.inv(-op.k2[..., None]*vh[..., 0, :, :, :])
    rs = r[..., 1, :, :, :]-force/c['rho_s_kg_m3']
    zz = torch.fft.fft2(torch.stack((rn, rs), -4), dim=(-3, -2))[..., op.y, op.x, :]/op.n**2
    zz = (zz*op.perp).sum(-1)/op.U
    da = torch.stack((2*zz.imag, 2*zz.real), -1).flatten(-2)
    return da+torch.stack((torch.zeros_like(da[..., 0, :]), -op.nu_s*op.lap*a[..., 1, :]), -2)


def batched_step(op, a, substeps=2):
    h = DT/substeps
    for _ in range(substeps):
        k1 = batched_rhs(op, a)
        k2 = batched_rhs(op, a+h*k1/2)
        k3 = batched_rhs(op, a+h*k2/2)
        k4 = batched_rhs(op, a+h*k3)
        a = a+h*(k1+2*k2+2*k3+k4)/6
    return a
