"""Alternative capture: a point vortex is pinned to a finite-mass particle.

The point vortex represents a short out-of-plane segment q*e_z.  Its unknown
3-D attached length is a model parameter, not something the 2-D data supply.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree
import torch

from .flow import TorchFlow, turn
from .particle_base import TorchParticles


class VortexAttachedParticles(TorchParticles):
    def __init__(self, flow: TorchFlow, effective_segment_m: float):
        if not np.isfinite(effective_segment_m) or effective_segment_m <= 0:
            raise ValueError("effective vortex segment must be positive and finite")
        super().__init__(flow)
        self.effective_segment_m = effective_segment_m
        self.a = torch.zeros_like(self.v)
        self.last_magnus = torch.zeros_like(self.v)
        self.last_friction = torch.zeros_like(self.v)
        self.refresh_pins(flow)

    @torch.no_grad()
    def refresh_pins(self, flow: TorchFlow) -> None:
        p = np.flatnonzero(self.state == self.TRAPPED)
        ids = torch.as_tensor(self.attached_vortex[p], device=flow.g.device, dtype=torch.long)
        pp = torch.as_tensor(p, device=flow.g.device, dtype=torch.long)
        flow.set_pins(ids, self.v[pp], self.a[pp])

    @torch.no_grad()
    def step_bound(self, old: dict, new: dict, flow: TorchFlow, dt: float,
                   extra_force: torch.Tensor | None = None) -> None:
        p = np.flatnonzero(self.state == self.TRAPPED)
        if not len(p):
            return
        c, g = self.c, flow.g
        pi = torch.as_tensor(p, device=g.device, dtype=torch.long)
        vi = torch.as_tensor(self.attached_vortex[p], device=g.device, dtype=torch.long)
        x = self.x[pi]
        vn = (old["vn"]+new["vn"])/2
        vs = (old["vs"]+new["vs"])/2
        an = (new["vn"]-old["vn"])/dt
        ass = (new["vs"]-old["vs"])/dt
        for j in range(2):
            an = an+vn[..., j, None]*g.gradient(vn[..., j])
            ass = ass+vs[..., j, None]*g.gradient(vs[..., j])
        un, us = g.sample(vn, x), g.sample(vs, x)
        anp, asp = g.sample(an, x), g.sample(ass, x)
        volume = 4*math.pi*c.particle_radius_m**3/3
        rho = c.rho_n_kg_m3+c.rho_s_kg_m3
        mass = (c.particle_density_kg_m3+rho/2)*volume
        drag = 6*math.pi*c.mu_n_Pa_s*c.particle_radius_m
        ell = self.effective_segment_m
        scale = c.rho_s_kg_m3*c.kappa_m2_s*ell
        den = (1-c.alpha_prime)**2+c.alpha**2
        gamma = c.rho_s_kg_m3*c.kappa_m2_s*c.alpha/den
        gamma_prime = (c.rho_s_kg_m3*c.kappa_m2_s
                       *(c.alpha_prime*(1-c.alpha_prime)-c.alpha**2)/den)
        q = flow.sign[vi, None]
        body = 1.5*volume*(c.rho_n_kg_m3*anp+c.rho_s_kg_m3*asp)
        body[:, 1] += volume*(c.particle_density_kg_m3-rho)*c.gravity_y_m_s2
        # m dv/dt = a*v + b*J_q(v) + constant; J_q^2 = -I.
        a_coef = -drag-ell*gamma
        b_coef = scale+ell*gamma_prime
        constant = (drag*un+body-scale*q*turn(us)
                    +ell*gamma*un-ell*gamma_prime*q*turn(un))
        if extra_force is not None:
            if extra_force.shape != constant.shape:
                raise ValueError("extra bound force must have one 2-D vector per trapped particle")
            constant = constant+extra_force
        previous = self.v[pi].clone()
        rhs = ((mass+dt*a_coef/2)*previous
               +dt*b_coef/2*q*turn(previous)+dt*constant)
        aa = mass-dt*a_coef/2
        bb = -dt*b_coef/2
        updated = (aa*rhs-bb*q*turn(rhs))/(aa**2+bb**2)
        self.v[pi] = updated
        self.a[pi] = (updated-previous)/dt
        self.x_unwrapped[pi] += dt*(previous+updated)/2
        flow.pos[vi] = self.x[pi]
        self.last_magnus[pi] = scale*q*turn(updated-us)
        self.last_friction[pi] = ell*(-gamma*(updated-un)
                                      +gamma_prime*q*turn(updated-un))
        self.refresh_pins(flow)

    @torch.no_grad()
    def reconcile_and_capture(self, flow: TorchFlow, event: dict) -> dict:
        c, device = self.c, flow.g.device
        self.cooldown = np.maximum(self.cooldown-1, 0)
        released = np.flatnonzero((self.state == self.TRAPPED)
                                  & np.isin(self.attached_vortex, event["indices"]))
        if len(released):
            self.state[released] = self.FREE
            self.attached_vortex[released] = -1
            self.cooldown[released] = c.release_cooldown_steps
            self.releases_total += len(released)
            self.last_magnus[released] = 0
            self.last_friction[released] = 0
            self.a[released] = 0
        captured = []
        if flow.step_index % c.capture_every == 0:
            free = np.flatnonzero((self.state == self.FREE) & (self.cooldown == 0))
            occupied = np.zeros(len(flow.sign), bool)
            occupied[self.attached_vortex[self.state == self.TRAPPED]] = True
            occupied[event["indices"]] = True
            eligible = np.flatnonzero(~occupied)
            if len(free) and len(eligible):
                vortex_pos = flow.pos.cpu().numpy()
                particle_pos = self.x[torch.as_tensor(free, device=device)].cpu().numpy()
                dist, near = cKDTree(vortex_pos[eligible], boxsize=c.box_m).query(
                    particle_pos, distance_upper_bound=flow.capture_radius_m)
                candidates = np.flatnonzero(np.isfinite(dist))
                for loc in candidates[np.argsort(dist[candidates], kind="stable")]:
                    node = int(eligible[near[loc]])
                    if occupied[node]:
                        continue
                    particle = int(free[loc])
                    self.state[particle] = self.TRAPPED
                    self.attached_vortex[particle] = node
                    # Particle position and velocity stay continuous: the vortex moves.
                    flow.pos[node] = self.x[particle]
                    self.a[particle] = 0
                    occupied[node] = True
                    captured.append((particle, node))
        self.captures_total += len(captured)
        self.refresh_pins(flow)
        return {"captured": len(captured), "released": len(released)}
