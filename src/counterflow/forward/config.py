from __future__ import annotations
from dataclasses import asdict,dataclass,field
import math
import numpy as np
from .helium import properties as helium_properties
@dataclass
class Config:
    box_m: float = 1.0e-3
    grid: int = 128
    temperature_K: float = 2.0
    vn_mean_m_s: float = 0.010
    line_density_m2: float | None = None
    kappa_m2_s: float = 9.97e-8
    rho_n_kg_m3: float = field(init=False)
    rho_s_kg_m3: float = field(init=False)
    mu_n_Pa_s: float = field(init=False)
    alpha: float = field(init=False)
    alpha_prime: float = field(init=False)
    alpha_v: float = field(init=False)
    beta_v: float = field(init=False)
    dt_s: float = 2.0e-5
    duration_s: float = 0.004
    save_interval_s: float = 0.001
    warmup_s: float = 0.020
    seed: int = 2718
    n_particles: int = 256
    particle_radius_m: float = 2.3e-6
    particle_density_kg_m3: float = 145.6
    gravity_y_m_s2: float = -9.81
    core_cells: float = 1.5
    friction_smooth_ell: float = 1.0
    coherent_fraction: float = 0.25
    reconnection_ell_fraction: float = 0.10
    reconnection_every: int = 5
    capture_every: int = 5
    capture_extra_radius_m: float = 0.0
    initial_trapped_particles: int = 64
    release_cooldown_steps: int = 5

    def __post_init__(self) -> None:
        for name, value in helium_properties(self.temperature_K).items():
            setattr(self, name, value)
        if self.line_density_m2 is None:
            rho = self.rho_n_kg_m3+self.rho_s_kg_m3
            self.line_density_m2 = ((self.alpha_v/self.beta_v)
                                    *(rho/self.rho_s_kg_m3)*abs(self.vn_mean_m_s))**2

    @property
    def n_vortices(self) -> int:
        n = int(round(self.line_density_m2 * self.box_m**2 / 2)) * 2
        return max(2, n)

    @property
    def vs_mean_m_s(self) -> float:
        return -self.rho_n_kg_m3 / self.rho_s_kg_m3 * self.vn_mean_m_s

    def validate(self) -> None:
        if self.grid < 16 or self.grid % 2 or self.box_m <= 0:
            raise ValueError("grid must be even >=16 and box_m positive")
        if self.dt_s <= 0 or self.duration_s <= 0 or self.save_interval_s <= 0:
            raise ValueError("time intervals must be positive")
        if min(self.rho_n_kg_m3, self.rho_s_kg_m3, self.mu_n_Pa_s) <= 0:
            raise ValueError("densities and viscosity must be positive")
        if not math.isfinite(self.line_density_m2) or self.line_density_m2 <= 0:
            raise ValueError("line density must be positive and finite; mean vn must be nonzero")
        if not 0 <= self.coherent_fraction <= 1 or self.n_particles < 0:
            raise ValueError("invalid particle count or coherent_fraction")
        if self.core_cells <= 0 or self.friction_smooth_ell <= 0:
            raise ValueError("regularization scales must be positive")
        if (self.reconnection_ell_fraction <= 0 or self.reconnection_every < 1
                or self.capture_every < 1
                or self.capture_extra_radius_m < 0 or self.release_cooldown_steps < 0
                or not 0 <= self.initial_trapped_particles <= self.n_particles):
            raise ValueError("invalid reconnection/capture configuration")

    def metadata(self) -> dict:
        out = asdict(self)
        out["n_vortices"] = self.n_vortices
        out["vs_mean_m_s"] = self.vs_mean_m_s
        out["nu_n_m2_s"] = self.mu_n_Pa_s / self.rho_n_kg_m3
        out["coordinate_mapping"] = "2D (x,y) = old 3D (y,z)"
        out["periodic_axes"] = ["x", "y"]
        return out
