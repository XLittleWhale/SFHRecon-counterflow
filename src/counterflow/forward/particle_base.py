"""GPU free-particle dynamics plus discrete 2-D vortex capture/detrapping."""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree
import torch

from .flow import TorchFlow


class TorchParticles:
    FREE, TRAPPED = 0, 1

    def __init__(self, flow: TorchFlow):
        c, g = flow.c, flow.g
        rng = np.random.default_rng(c.seed+100003)
        self.x_unwrapped = torch.as_tensor(
            rng.uniform(0, c.box_m, (c.n_particles, 2)), device=g.device, dtype=g.dtype)
        self.v = g.sample(flow.fields()["vn"], self.x_unwrapped)
        self.state = np.full(c.n_particles, self.FREE, np.int8)
        self.attached_vortex = np.full(c.n_particles, -1, np.int32)
        self.cooldown = np.zeros(c.n_particles, np.int32)
        self.captures_total = 0
        self.releases_total = 0
        n = c.initial_trapped_particles
        if n:
            chosen = rng.choice(len(flow.sign), n, replace=False)
            self.state[:n] = self.TRAPPED
            self.attached_vortex[:n] = chosen
            ids = torch.as_tensor(chosen, device=g.device)
            self.x_unwrapped[:n] = flow.pos[ids]
            _, vv = flow.rhs(flow.omega_hat, flow.pos)
            self.v[:n] = vv[ids]
        self.c = c

    @property
    def x(self):
        return torch.remainder(self.x_unwrapped, self.c.box_m)

    @torch.no_grad()
    def step_free(self, old: dict, new: dict, flow: TorchFlow, dt: float) -> None:
        free_np = np.flatnonzero(self.state == self.FREE)
        if not len(free_np):
            return
        idx = torch.as_tensor(free_np, device=flow.g.device)
        g, c = flow.g, self.c
        x = self.x[idx]
        vn = (old["vn"]+new["vn"])/2
        vs = (old["vs"]+new["vs"])/2
        an = (new["vn"]-old["vn"])/dt
        ass = (new["vs"]-old["vs"])/dt
        for j in range(2):
            an = an+vn[..., j, None]*g.gradient(vn[..., j])
            ass = ass+vs[..., j, None]*g.gradient(vs[..., j])
        unp = g.sample(vn, x)
        anp = g.sample(an, x)
        asp = g.sample(ass, x)
        volume = 4*math.pi*c.particle_radius_m**3/3
        rho = c.rho_n_kg_m3+c.rho_s_kg_m3
        mass = (c.particle_density_kg_m3+rho/2)*volume
        drag = 6*math.pi*c.mu_n_Pa_s*c.particle_radius_m
        body = 1.5*volume*(c.rho_n_kg_m3*anp+c.rho_s_kg_m3*asp)
        body[:, 1] += volume*(c.particle_density_kg_m3-rho)*c.gravity_y_m_s2
        before = self.v[idx].clone()
        after = ((mass-dt*drag/2)*before+dt*(drag*unp+body))/(mass+dt*drag/2)
        self.v[idx] = after
        self.x_unwrapped[idx] += dt*(before+after)/2

    @torch.no_grad()
    def synchronize_and_capture(self, flow: TorchFlow, old_vortex_pos: torch.Tensor,
                                event: dict, dt: float) -> dict:
        c, device = self.c, flow.g.device
        self.cooldown = np.maximum(self.cooldown-1, 0)
        trapped = np.flatnonzero(self.state == self.TRAPPED)
        released = np.empty(0, np.int64)
        if len(trapped):
            node_np = self.attached_vortex[trapped]
            pidx = torch.as_tensor(trapped, device=device)
            nidx = torch.as_tensor(node_np, device=device)
            delta = flow.pos[nidx]-old_vortex_pos[nidx]
            delta -= c.box_m*torch.round(delta/c.box_m)
            lookup = {int(k): j for j, k in enumerate(event["indices"])}
            if lookup:
                for loc, node in enumerate(node_np):
                    if int(node) in lookup:
                        prior = torch.as_tensor(event["positions"][lookup[int(node)]],
                                                device=device, dtype=flow.g.dtype)
                        dd = prior-old_vortex_pos[int(node)]
                        delta[loc] = dd-c.box_m*torch.round(dd/c.box_m)
            self.x_unwrapped[pidx] += delta
            self.v[pidx] = delta/dt
            if lookup:
                released = trapped[np.isin(node_np, event["indices"])]
                for p in released:
                    node = int(self.attached_vortex[p])
                    self.v[int(p)] = torch.as_tensor(event["velocities"][lookup[node]],
                                                     device=device, dtype=flow.g.dtype)
                self.state[released] = self.FREE
                self.attached_vortex[released] = -1
                self.cooldown[released] = c.release_cooldown_steps
                self.releases_total += len(released)
        if flow.step_index % c.capture_every:
            return {"captured": 0, "released": len(released)}
        free = np.flatnonzero((self.state == self.FREE) & (self.cooldown == 0))
        if not len(free):
            return {"captured": 0, "released": len(released)}
        occupied = np.zeros(len(flow.sign), bool)
        occupied[self.attached_vortex[self.state == self.TRAPPED]] = True
        occupied[event["indices"]] = True
        eligible = np.flatnonzero(~occupied)
        if not len(eligible):
            return {"captured": 0, "released": len(released)}
        vortex_pos = flow.pos.cpu().numpy()
        particle_pos = self.x[torch.as_tensor(free, device=device)].cpu().numpy()
        dist, near = cKDTree(vortex_pos[eligible], boxsize=c.box_m).query(
            particle_pos, distance_upper_bound=flow.capture_radius_m)
        candidates = np.flatnonzero(np.isfinite(dist))
        if not len(candidates):
            return {"captured": 0, "released": len(released)}
        captured = []
        for loc in candidates[np.argsort(dist[candidates], kind="stable")]:
            node = int(eligible[near[loc]])
            if occupied[node]:
                continue
            p = int(free[loc])
            jump = vortex_pos[node]-particle_pos[loc]
            jump -= c.box_m*np.round(jump/c.box_m)
            self.x_unwrapped[p] += torch.as_tensor(jump, device=device, dtype=flow.g.dtype)
            self.state[p] = self.TRAPPED
            self.attached_vortex[p] = node
            occupied[node] = True
            captured.append((p, node))
        if captured:
            _, vv = flow.rhs(flow.omega_hat, flow.pos)
            pidx = torch.as_tensor([p for p, _ in captured], device=device)
            nidx = torch.as_tensor([j for _, j in captured], device=device)
            self.v[pidx] = vv[nidx]
        self.captures_total += len(captured)
        return {"captured": len(captured), "released": len(released)}
