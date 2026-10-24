"""CUDA-ready particle-mesh counterpart of solver.Flow.

PyTorch FFT/scatter/interpolation remain on the selected device.  The
infrequent periodic pseudo-reconnection search uses SciPy cKDTree on host.
The same Config, signs, equations and RK4 stages as the CPU reference apply.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

from .config import Config


def turn(v: torch.Tensor) -> torch.Tensor:
    return torch.stack((-v[..., 1], v[..., 0]), dim=-1)


class TorchGrid:
    def __init__(self, c: Config, device: str, precision: str = "float32"):
        self.c = c
        self.device = torch.device(device)
        self.dtype = torch.float64 if precision == "float64" else torch.float32
        self.cdtype = torch.complex128 if precision == "float64" else torch.complex64
        n = c.grid
        self.dx = c.box_m/n
        k = 2*torch.pi*torch.fft.fftfreq(n, d=self.dx, device=self.device, dtype=self.dtype)
        self.kx, self.ky = torch.meshgrid(k, k, indexing="xy")
        self.k2 = self.kx.square()+self.ky.square()
        self.inv_k2 = torch.where(self.k2 > 0, 1.0/torch.clamp(self.k2, min=1.), 0.)
        modes = torch.fft.fftfreq(n, device=self.device, dtype=self.dtype)*n
        mx, my = torch.meshgrid(modes, modes, indexing="xy")
        self.dealias = (mx.abs() < n/3) & (my.abs() < n/3)
        self.core_filter = torch.exp(-0.5*(c.core_cells*self.dx)**2*self.k2)
        ell = c.box_m/np.sqrt(c.n_vortices)
        self.friction_filter = torch.exp(-0.5*(c.friction_smooth_ell*ell)**2*self.k2)

    def cic(self, pos: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        n = self.c.grid
        u = torch.remainder(pos, self.c.box_m)/self.dx
        ii = torch.floor(u).long()
        q = u-ii
        ix, iy = ii[:, 0], ii[:, 1]
        out = torch.zeros(n*n, dtype=self.dtype, device=self.device)
        w0 = weight/self.dx**2
        for ox in (0, 1):
            for oy in (0, 1):
                flat = torch.remainder(iy+oy, n)*n+torch.remainder(ix+ox, n)
                value = w0*(q[:, 0] if ox else 1-q[:, 0])*(q[:, 1] if oy else 1-q[:, 1])
                out.scatter_add_(0, flat, value)
        return out.view(n, n)

    def sample(self, f: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        n = self.c.grid
        u = torch.remainder(pos, self.c.box_m)/self.dx
        ii = torch.floor(u).long()
        q = u-ii
        ix, iy = ii[:, 0], ii[:, 1]
        out = torch.zeros((len(pos),)+tuple(f.shape[2:]), dtype=self.dtype, device=self.device)
        for ox in (0, 1):
            for oy in (0, 1):
                w = (q[:, 0] if ox else 1-q[:, 0])*(q[:, 1] if oy else 1-q[:, 1])
                val = f[torch.remainder(iy+oy, n), torch.remainder(ix+ox, n)]
                out += w.reshape((-1,)+(1,)*(val.ndim-1))*val
        return out

    def velocities(self, omega_hat: torch.Tensor, mean_y: float) -> torch.Tensor:
        psi = omega_hat*self.inv_k2
        return torch.stack((torch.fft.ifft2(1j*self.ky*psi).real,
                            torch.fft.ifft2(-1j*self.kx*psi).real+mean_y), dim=-1)

    def gradient(self, f: torch.Tensor) -> torch.Tensor:
        hat = torch.fft.fft2(f)
        return torch.stack((torch.fft.ifft2(1j*self.kx*hat).real,
                            torch.fft.ifft2(1j*self.ky*hat).real), dim=-1)


class TorchFlow:
    def __init__(self, c: Config, device: str = "cuda", precision: str = "float32"):
        c.validate()
        self.c = c
        self.g = TorchGrid(c, device, precision)
        self.rng = np.random.default_rng(c.seed)
        init = self._initial_positions()
        self.pos = torch.as_tensor(init, device=self.g.device, dtype=self.g.dtype)
        self.sign = torch.as_tensor(np.tile(np.array([1., -1.]), c.n_vortices//2),
                                    device=self.g.device, dtype=self.g.dtype)
        self.ones = torch.ones_like(self.sign)
        self.omega_hat = torch.zeros((c.grid, c.grid), device=self.g.device, dtype=self.g.cdtype)
        self.time = 0.0
        self.step_index = 0
        self.reconnection_pairs_total = 0
        self.pin_ids = torch.empty(0, device=self.g.device, dtype=torch.long)
        self.pin_velocity = torch.empty((0, 2), device=self.g.device, dtype=self.g.dtype)
        self.pin_acceleration = torch.empty((0, 2), device=self.g.device, dtype=self.g.dtype)
        self.pin_fraction = 0.0
        self.pin_step_dt = c.dt_s

    def set_pins(self, ids: torch.Tensor, velocity: torch.Tensor,
                 acceleration: torch.Tensor | None = None) -> None:
        self.pin_ids = ids.to(device=self.g.device, dtype=torch.long).clone()
        self.pin_velocity = velocity.to(device=self.g.device, dtype=self.g.dtype).clone()
        self.pin_acceleration = (torch.zeros_like(self.pin_velocity) if acceleration is None
                                 else acceleration.to(device=self.g.device, dtype=self.g.dtype).clone())

    def _initial_positions(self) -> np.ndarray:
        c = self.c
        pos = self.rng.uniform(0, c.box_m, (c.n_vortices, 2))
        m = int((c.n_vortices//2)*c.coherent_fraction)
        if m:
            pos[np.arange(0, 2*m, 2)] = np.mod(self.rng.normal(
                (0.30*c.box_m, 0.50*c.box_m), 0.09*c.box_m, (m, 2)), c.box_m)
            pos[np.arange(1, 2*m, 2)] = np.mod(self.rng.normal(
                (0.70*c.box_m, 0.50*c.box_m), 0.09*c.box_m, (m, 2)), c.box_m)
        return pos

    @property
    def reconnection_radius_m(self) -> float:
        return self.c.reconnection_ell_fraction*self.c.box_m/np.sqrt(len(self.sign))

    @property
    def capture_radius_m(self) -> float:
        return self.reconnection_radius_m+self.c.particle_radius_m+self.c.capture_extra_radius_m

    def fields(self, omega_hat=None, pos=None) -> dict:
        if omega_hat is None:
            omega_hat = self.omega_hat
        if pos is None:
            pos = self.pos
        c, g = self.c, self.g
        signed_hat = torch.fft.fft2(g.cic(pos, self.sign))
        count_hat = torch.fft.fft2(g.cic(pos, self.ones))
        super_hat = c.kappa_m2_s*signed_hat*g.core_filter
        vn = g.velocities(omega_hat, c.vn_mean_m_s)
        vs = g.velocities(super_hat, c.vs_mean_m_s)
        L = torch.fft.ifft2(count_hat*g.friction_filter).real
        Omega = torch.fft.ifft2(c.kappa_m2_s*signed_hat*g.friction_filter).real
        slip = vn-vs
        F = (-c.alpha*c.rho_s_kg_m3*c.kappa_m2_s*L[..., None]*slip
             + c.alpha_prime*c.rho_s_kg_m3*Omega[..., None]*turn(slip))
        if len(self.pin_ids):
            ids = self.pin_ids
            vp = self.pin_velocity+self.pin_fraction*self.pin_step_dt*self.pin_acceleration
            vn_p = g.sample(vn, pos[ids])
            vs_p = g.sample(vs, pos[ids])
            q = self.sign[ids, None]
            slip_p = vn_p-vs_p
            original = (-c.alpha*c.rho_s_kg_m3*c.kappa_m2_s*slip_p
                        +c.alpha_prime*c.rho_s_kg_m3*c.kappa_m2_s*q*turn(slip_p))
            den = (1-c.alpha_prime)**2+c.alpha**2
            gamma = c.rho_s_kg_m3*c.kappa_m2_s*c.alpha/den
            gamma_prime = (c.rho_s_kg_m3*c.kappa_m2_s
                           *(c.alpha_prime*(1-c.alpha_prime)-c.alpha**2)/den)
            # Reaction of the bound-vortex friction on the normal component.
            reaction = gamma*(vp-vn_p)-gamma_prime*q*turn(vp-vn_p)
            correction = reaction-original
            correction_grid = torch.stack((g.cic(pos[ids], correction[:, 0]),
                                           g.cic(pos[ids], correction[:, 1])), dim=-1)
            correction_hat = torch.fft.fft2(correction_grid, dim=(0, 1))
            F = F+torch.fft.ifft2(correction_hat*g.friction_filter[..., None],
                                  dim=(0, 1)).real
        return {"vn": vn, "vs": vs, "omega_n": torch.fft.ifft2(omega_hat).real,
                "omega_s": torch.fft.ifft2(super_hat).real, "line_density": L, "F_ns": F}

    def rhs(self, omega_hat, pos):
        c, g = self.c, self.g
        f = self.fields(omega_hat, pos)
        gradw = torch.stack((torch.fft.ifft2(1j*g.kx*omega_hat).real,
                             torch.fft.ifft2(1j*g.ky*omega_hat).real), dim=-1)
        adv = torch.sum(f["vn"]*gradw, dim=-1)
        Fx = torch.fft.fft2(f["F_ns"][..., 0])
        Fy = torch.fft.fft2(f["F_ns"][..., 1])
        curl = (1j*g.kx*Fy-1j*g.ky*Fx)/c.rho_n_kg_m3
        dw = (-torch.fft.fft2(adv)+curl
              -(c.mu_n_Pa_s/c.rho_n_kg_m3)*g.k2*omega_hat)*g.dealias
        dw[0, 0] = 0
        vs_j = g.sample(f["vs"], pos)
        slip_j = g.sample(f["vn"], pos)-vs_j
        dx = vs_j+c.alpha*self.sign[:, None]*turn(slip_j)+c.alpha_prime*slip_j
        if len(self.pin_ids):
            dx[self.pin_ids] = (self.pin_velocity
                                +self.pin_fraction*self.pin_step_dt*self.pin_acceleration)
        return dw, dx

    @torch.no_grad()
    def step(self, dt: float) -> None:
        w, x, L = self.omega_hat, self.pos, self.c.box_m
        self.pin_step_dt = dt
        self.pin_fraction = 0.0
        k1w, k1x = self.rhs(w, x)
        self.pin_fraction = 0.5
        k2w, k2x = self.rhs(w+dt*k1w/2, torch.remainder(x+dt*k1x/2, L))
        k3w, k3x = self.rhs(w+dt*k2w/2, torch.remainder(x+dt*k2x/2, L))
        self.pin_fraction = 1.0
        k4w, k4x = self.rhs(w+dt*k3w, torch.remainder(x+dt*k3x, L))
        self.omega_hat = (w+dt*(k1w+2*k2w+2*k3w+k4w)/6)*self.g.dealias
        self.omega_hat[0, 0] = 0
        self.pos = torch.remainder(x+dt*(k1x+2*k2x+2*k3x+k4x)/6, L)
        self.time += dt
        self.step_index += 1
        self.pin_fraction = 0.0

    @torch.no_grad()
    def reconnect(self) -> dict:
        empty = {"indices": np.empty(0, np.int64), "positions": np.empty((0, 2)),
                 "velocities": np.empty((0, 2)), "pairs": 0}
        if self.step_index % self.c.reconnection_every:
            return empty
        pos = self.pos.cpu().numpy()
        pairs = cKDTree(pos, boxsize=self.c.box_m).query_pairs(
            self.reconnection_radius_m, output_type="ndarray")
        if not len(pairs):
            return empty
        sign = self.sign.cpu().numpy()
        pairs = pairs[sign[pairs[:, 0]] != sign[pairs[:, 1]]]
        if not len(pairs):
            return empty
        delta = pos[pairs[:, 0]]-pos[pairs[:, 1]]
        delta -= self.c.box_m*np.round(delta/self.c.box_m)
        pairs = pairs[np.argsort(np.sum(delta**2, axis=1), kind="stable")]
        used = np.zeros(len(sign), bool)
        chosen = []
        for i, j in pairs:
            if not (used[i] or used[j]):
                chosen.extend((int(i), int(j)))
                used[i] = used[j] = True
        if not chosen:
            return empty
        indices = np.asarray(chosen, np.int64)
        _, velocities = self.rhs(self.omega_hat, self.pos)
        event = {"indices": indices, "positions": pos[indices].copy(),
                 "velocities": velocities[indices].cpu().numpy(), "pairs": len(indices)//2}
        self.pos[torch.as_tensor(indices, device=self.g.device)] = torch.as_tensor(
            self.rng.uniform(0, self.c.box_m, (len(indices), 2)),
            device=self.g.device, dtype=self.g.dtype)
        self.reconnection_pairs_total += event["pairs"]
        return event

    @torch.no_grad()
    def diagnostics(self) -> dict:
        f = self.fields()
        vn = f["vn"]-torch.as_tensor([0., self.c.vn_mean_m_s], device=self.g.device)
        vs = f["vs"]-torch.as_tensor([0., self.c.vs_mean_m_s], device=self.g.device)
        return {"time_s": self.time,
                "vn_fluct_rms_m_s": float(torch.sqrt(torch.mean(vn.square())).item()),
                "vs_fluct_rms_m_s": float(torch.sqrt(torch.mean(vs.square())).item()),
                "Fns_rms_N_m3": float(torch.sqrt(torch.mean(f["F_ns"].square())).item()),
                "reconnection_pairs_total": self.reconnection_pairs_total}

    @torch.no_grad()
    def snapshot_fields(self) -> dict:
        return {k: v.cpu().numpy() for k, v in self.fields().items()}

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name+".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, omega_hat=self.omega_hat.cpu().numpy(),
                                vortex_xy_m=self.pos.cpu().numpy(),
                                vortex_sign=self.sign.cpu().numpy(),
                                time_s=np.array(self.time), step_index=np.array(self.step_index),
                                reconnection_pairs_total=np.array(self.reconnection_pairs_total),
                                config_json=np.array(json.dumps(self.c.metadata())),
                                rng_state_json=np.array(json.dumps(self.rng.bit_generator.state)))
        temporary.replace(path)

    @classmethod
    def load_checkpoint(cls, path: Path, device: str = "cuda", precision: str = "float32") -> "TorchFlow":
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(str(z["config_json"]))
            names = (k for k, f in Config.__dataclass_fields__.items() if f.init)
            c = Config(**{k: metadata[k] for k in names if k in metadata})
            obj = cls(c, device, precision)
            obj.omega_hat = torch.as_tensor(z["omega_hat"], device=obj.g.device,
                                            dtype=obj.g.cdtype)
            obj.pos = torch.as_tensor(z["vortex_xy_m"], device=obj.g.device,
                                      dtype=obj.g.dtype)
            obj.sign = torch.as_tensor(z["vortex_sign"], device=obj.g.device,
                                       dtype=obj.g.dtype)
            obj.ones = torch.ones_like(obj.sign)
            obj.time = float(z["time_s"])
            obj.step_index = int(z["step_index"])
            obj.reconnection_pairs_total = int(z["reconnection_pairs_total"])
            obj.rng.bit_generator.state = json.loads(str(z["rng_state_json"]))
        return obj
