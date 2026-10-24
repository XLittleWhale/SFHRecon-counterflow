"""Pinned point vortices with reduced two-anchor filament-tension closures.

The virtual anchors follow the local point-vortex kinematics. Tension acts on
the particle only; anchor positions are geometrically projected if necessary.
This is a one-way reduced closure, not a resolved filament calculation.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .particle_attached import VortexAttachedParticles
from .flow import TorchFlow, turn


class VortexTetheredParticles(VortexAttachedParticles):
    # Leave a small margin so the enforced cap is strictly below 2.5*a_p.
    MAX_ANCHOR_RADIUS_MULTIPLE = 2.5*(1.0-1.0e-9)

    def __init__(self, flow: TorchFlow, effective_segment_m: float,
                 anchor_separation_m: float | None = None,
                 tension_scale: float = 1.0,
                 break_extension_m: float | None = None):
        c = flow.c
        particle_radius = c.particle_radius_m
        spacing = c.box_m/math.sqrt(len(flow.sign))
        max_length = self.MAX_ANCHOR_RADIUS_MULTIPLE*particle_radius
        h = (0.5*(particle_radius+max_length) if anchor_separation_m is None
             else anchor_separation_m)
        if not np.isfinite(h) or not particle_radius < h < max_length:
            raise ValueError(
                "anchor separation h must satisfy particle_radius < h < 2.5*particle_radius")
        max_in_plane = math.sqrt(max_length**2-h**2)
        if not np.isfinite(tension_scale) or tension_scale < 0:
            raise ValueError("tension scale must be finite and nonnegative")
        if break_extension_m is not None and (not np.isfinite(break_extension_m)
                                               or not 0 < break_extension_m < max_in_plane):
            raise ValueError("break extension must lie below the allowed in-plane anchor radius")
        super().__init__(flow, effective_segment_m)
        # T is line energy per length (N). The free-core and outer cutoffs are
        # explicit modelling choices; both are recorded with the output.
        self.vortex_core_m = 1.0e-10
        self.tension_outer_m = spacing
        self.base_tension_N = (c.rho_s_kg_m3*c.kappa_m2_s**2/(4*math.pi)
                               *math.log(self.tension_outer_m/self.vortex_core_m))
        self.tension_N = tension_scale*self.base_tension_N
        self.anchor_separation_m = h
        self.intervortex_spacing_m = spacing
        self.max_anchor_length_m = max_length
        self.max_anchor_in_plane_m = max_in_plane
        self.tension_scale = tension_scale
        self.break_extension_m = break_extension_m
        self.tension_releases_total = 0
        self.anchors_unwrapped = self.x_unwrapped[:, None, :].repeat(1, 2, 1).clone()
        self.last_tension = torch.zeros_like(self.v)

    def _short_delta(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        delta = a-b
        return delta-self.c.box_m*torch.round(delta/self.c.box_m)

    def _limit_anchor_length(self, particle: torch.Tensor,
                             anchors: torch.Tensor) -> torch.Tensor:
        """Project 3-D tether length below 2.5 particle radii."""
        delta = self._short_delta(anchors, particle[:, None, :])
        radius = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        factor = torch.clamp(self.max_anchor_in_plane_m/
                             torch.clamp(radius, min=1e-30), max=1.)
        return anchors+delta*(factor-1.)

    def _line_velocity(self, fields: dict, anchor: torch.Tensor,
                       q: torch.Tensor, flow: TorchFlow) -> torch.Tensor:
        flat = anchor.reshape(-1, 2)
        us = flow.g.sample(fields["vs"], flat).reshape(-1, 2, 2)
        un = flow.g.sample(fields["vn"], flat).reshape(-1, 2, 2)
        slip = un-us
        return us+self.c.alpha*q[:, None, None]*turn(slip)+self.c.alpha_prime*slip

    def _tension_force(self, particle: torch.Tensor,
                       anchors: torch.Tensor) -> torch.Tensor:
        return self._anchor_tension_forces(particle, anchors).sum(dim=1)

    def _anchor_tension_forces(self, particle: torch.Tensor,
                               anchors: torch.Tensor) -> torch.Tensor:
        delta = self._short_delta(anchors, particle[:, None, :])
        length = torch.sqrt(self.anchor_separation_m**2+delta.square().sum(-1,
                                                                           keepdim=True))
        return self.tension_N*delta/length

    def _advance_anchors(self, particle: torch.Tensor,
                         drifted: torch.Tensor) -> torch.Tensor:
        """Keep point-vortex advection; project only if the geometry cap is hit."""
        return self._limit_anchor_length(particle, drifted)

    @torch.no_grad()
    def step_bound(self, old: dict, new: dict, flow: TorchFlow, dt: float) -> None:
        bound = np.flatnonzero(self.state == self.TRAPPED)
        if not len(bound):
            return
        ids = torch.as_tensor(bound, device=flow.g.device, dtype=torch.long)
        vortices = torch.as_tensor(self.attached_vortex[bound], device=flow.g.device,
                                  dtype=torch.long)
        q = flow.sign[vortices]
        anchors = self.anchors_unwrapped[ids]
        velocity = (self._line_velocity(old, anchors, q, flow)
                    +self._line_velocity(new, anchors, q, flow))/2
        midpoint = self._limit_anchor_length(
            self.x_unwrapped[ids], anchors+0.5*dt*velocity)
        force = self._tension_force(self.x_unwrapped[ids], midpoint)
        super().step_bound(old, new, flow, dt, extra_force=force)
        self.last_tension[ids] = force
        self.anchors_unwrapped[ids] = self._advance_anchors(
            self.x_unwrapped[ids], anchors+dt*velocity)

    @torch.no_grad()
    def reconcile_and_capture(self, flow: TorchFlow, event: dict) -> dict:
        broken = np.empty(0, dtype=np.int64)
        if self.break_extension_m is not None:
            bound = np.flatnonzero(self.state == self.TRAPPED)
            if len(bound):
                ids = torch.as_tensor(bound, device=flow.g.device, dtype=torch.long)
                distance = torch.linalg.vector_norm(
                    self._short_delta(self.anchors_unwrapped[ids],
                                      self.x_unwrapped[ids, None, :]), dim=-1)
                broken = bound[torch.any(distance > self.break_extension_m,
                                         dim=1).cpu().numpy()]
            if len(broken):
                self.state[broken] = self.FREE
                self.attached_vortex[broken] = -1
                self.cooldown[broken] = self.c.release_cooldown_steps
                self.releases_total += len(broken)
                self.tension_releases_total += len(broken)
                self.last_tension[broken] = 0
                self.last_magnus[broken] = 0
                self.last_friction[broken] = 0
                self.a[broken] = 0
                self.refresh_pins(flow)
        before = self.state.copy()
        result = super().reconcile_and_capture(flow, event)
        captured = np.flatnonzero((before == self.FREE) & (self.state == self.TRAPPED))
        if len(captured):
            ids = torch.as_tensor(captured, device=flow.g.device, dtype=torch.long)
            self.anchors_unwrapped[ids] = self.x_unwrapped[ids, None, :]
            self.last_tension[ids] = 0
        released = np.flatnonzero((before == self.TRAPPED) & (self.state == self.FREE))
        if len(released):
            self.last_tension[released] = 0
        result["released"] += len(broken)
        result["tension_released"] = len(broken)
        return result


class VortexInertialTetheredParticles(VortexTetheredParticles):
    """Each virtual anchor carries mass and the opposite particle-tension force.

    The unforced point-vortex velocity is the zero-force solution of the
    Magnus/mutual-friction balance.  The *incremental* force equation is

      m_a dv_a/dt = -d (v_a-u_0) + q*b*J(v_a-u_0) - f_particle,each.

    Its constant-force/constant-u_0 substep is integrated analytically. This
    is essential when the core-scale mass makes the relaxation much shorter
    than the flow time step. The geometric length cap is still a reduced-model
    constraint, not a resolved 3-D filament/contact impulse.
    """

    def __init__(self, flow: TorchFlow, effective_segment_m: float,
                 anchor_separation_m: float | None = None,
                 tension_scale: float = 1.0,
                 break_extension_m: float | None = None,
                 inertia_radius_m: float = 1.0e-10,
                 mass_factor: float = math.pi):
        if not np.isfinite(inertia_radius_m) or inertia_radius_m <= 0:
            raise ValueError("anchor inertia radius must be finite and positive")
        if not np.isfinite(mass_factor) or mass_factor <= 0:
            raise ValueError("anchor mass factor must be finite and positive")
        super().__init__(flow, effective_segment_m, anchor_separation_m,
                         tension_scale, break_extension_m)
        c = self.c
        self.inertia_radius_m = inertia_radius_m
        self.mass_factor = mass_factor
        self.anchor_segment_m = 0.5*effective_segment_m
        self.anchor_mass_kg = (mass_factor*c.rho_s_kg_m3*inertia_radius_m**2
                               *self.anchor_segment_m)
        den = (1-c.alpha_prime)**2+c.alpha**2
        line_gyro = c.rho_s_kg_m3*c.kappa_m2_s
        line_drag = line_gyro*c.alpha/den
        line_transverse = line_gyro*(1-c.alpha_prime)/den
        self.anchor_drag_Ns_m = self.anchor_segment_m*line_drag
        self.anchor_gyro_Ns_m = self.anchor_segment_m*line_transverse
        self.anchor_relaxation_s = self.anchor_mass_kg/self.anchor_drag_Ns_m
        self.anchor_velocity = torch.zeros_like(self.anchors_unwrapped)
        self.last_anchor_acceleration = torch.zeros_like(self.anchors_unwrapped)
        self.last_anchor_reaction = torch.zeros_like(self.anchors_unwrapped)
        self.cap_hits_total = 0
        bound = np.flatnonzero(self.state == self.TRAPPED)
        if len(bound):
            ids = torch.as_tensor(bound, device=flow.g.device, dtype=torch.long)
            vortices = torch.as_tensor(self.attached_vortex[bound],
                                      device=flow.g.device, dtype=torch.long)
            self.anchor_velocity[ids] = self._line_velocity(
                flow.fields(), self.anchors_unwrapped[ids], flow.sign[vortices], flow)

    def _integrate_anchors(self, anchors: torch.Tensor, velocity: torch.Tensor,
                           unforced_velocity: torch.Tensor, q: torch.Tensor,
                           particle_force: torch.Tensor, dt: float
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact affine velocity/position step for frozen forces and flow."""
        d, b, mass = (self.anchor_drag_Ns_m, self.anchor_gyro_Ns_m,
                      self.anchor_mass_kg)
        qj_force = q[:, None, None]*turn(particle_force)
        equilibrium = unforced_velocity-(d*particle_force+b*qj_force)/(d*d+b*b)
        error = velocity-equilibrium
        z = complex(-d, b)*dt/mass
        decay = np.exp(z)
        integral = np.expm1(z)/z
        qj_error = q[:, None, None]*turn(error)
        updated_velocity = (equilibrium+decay.real*error+decay.imag*qj_error)
        updated_position = (anchors+dt*equilibrium
                            +dt*(integral.real*error+integral.imag*qj_error))
        return updated_position, updated_velocity

    @torch.no_grad()
    def step_bound(self, old: dict, new: dict, flow: TorchFlow, dt: float) -> None:
        bound = np.flatnonzero(self.state == self.TRAPPED)
        if not len(bound):
            return
        ids = torch.as_tensor(bound, device=flow.g.device, dtype=torch.long)
        vortices = torch.as_tensor(self.attached_vortex[bound],
                                  device=flow.g.device, dtype=torch.long)
        q = flow.sign[vortices]
        anchors = self.anchors_unwrapped[ids].clone()
        velocity = self.anchor_velocity[ids].clone()
        particle = self.x_unwrapped[ids].clone()
        particle_mid = particle+0.5*dt*self.v[ids]
        unforced = (self._line_velocity(old, anchors, q, flow)
                    +self._line_velocity(new, anchors, q, flow))/2
        # One midpoint predictor/corrector. The final pair of forces is
        # exactly opposite on the particle and the two virtual anchors.
        guess = self._limit_anchor_length(particle_mid, anchors+0.5*dt*velocity)
        predicted_force = self._anchor_tension_forces(particle_mid, guess)
        predicted, _ = self._integrate_anchors(
            anchors, velocity, unforced, q, predicted_force, dt)
        midpoint = self._limit_anchor_length(particle_mid, 0.5*(anchors+predicted))
        force_each = self._anchor_tension_forces(particle_mid, midpoint)
        total_force = force_each.sum(dim=1)
        VortexAttachedParticles.step_bound(
            self, old, new, flow, dt, extra_force=total_force)
        advanced, updated_velocity = self._integrate_anchors(
            anchors, velocity, unforced, q, force_each, dt)
        projected = self._limit_anchor_length(self.x_unwrapped[ids], advanced)
        correction = self._short_delta(advanced, projected)
        self.cap_hits_total += int(torch.count_nonzero(
            torch.linalg.vector_norm(correction, dim=-1) > 1e-15).item())
        # A hard geometric cap must not leave an outward relative velocity.
        radial = self._short_delta(projected, self.x_unwrapped[ids, None, :])
        norm = torch.linalg.vector_norm(radial, dim=-1, keepdim=True)
        direction = radial/torch.clamp(norm, min=1e-30)
        outward = ((updated_velocity-self.v[ids, None, :])*direction).sum(
            dim=-1, keepdim=True).clamp(min=0)
        on_cap = norm >= self.max_anchor_in_plane_m*(1-1e-7)
        updated_velocity = updated_velocity-torch.where(
            on_cap, outward*direction, torch.zeros_like(updated_velocity))
        self.anchors_unwrapped[ids] = projected
        self.anchor_velocity[ids] = updated_velocity
        self.last_anchor_acceleration[ids] = (updated_velocity-velocity)/dt
        self.last_anchor_reaction[ids] = -force_each
        self.last_tension[ids] = total_force

    @torch.no_grad()
    def reconcile_and_capture(self, flow: TorchFlow, event: dict) -> dict:
        before = self.state.copy()
        result = super().reconcile_and_capture(flow, event)
        captured = np.flatnonzero((before == self.FREE) & (self.state == self.TRAPPED))
        if len(captured):
            ids = torch.as_tensor(captured, device=flow.g.device, dtype=torch.long)
            vortices = torch.as_tensor(self.attached_vortex[captured],
                                      device=flow.g.device, dtype=torch.long)
            self.anchor_velocity[ids] = self._line_velocity(
                flow.fields(), self.anchors_unwrapped[ids], flow.sign[vortices], flow)
            self.last_anchor_acceleration[ids] = 0
            self.last_anchor_reaction[ids] = 0
        released = np.flatnonzero((before == self.TRAPPED) & (self.state == self.FREE))
        if len(released):
            self.anchor_velocity[released] = 0
            self.last_anchor_acceleration[released] = 0
            self.last_anchor_reaction[released] = 0
        return result
