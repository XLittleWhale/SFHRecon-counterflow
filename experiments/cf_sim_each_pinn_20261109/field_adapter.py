"""New-data adapter for the frozen K12 CF PINN and full Frozen-B prior.

Only the historical geometry operator and prior assets are loaded. No old
observations, learned fields or target truth enter preparation. Unlike the
historical observation query, material acceleration is du/dt + (u.grad)u,
matching the 2026-11-06 CF point-vortex reference generator.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
BASELINE = HERE.parent / "pinn_cf_20261105" / "cf_field.py"
OPERATOR = (BASE / "outputs/JointProbabilityPhysics_20261103/cf_runtime/payload/"
            "server_package/inputs/cf_113_camera100_f0184/operator.pt")
DT = .001


def _baseline():
    name = "each_cf_frozen_field_20261105"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, BASELINE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def probability_physics(config: dict, intervals: int) -> dict:
    """Small metadata-only observation context; allocates no operator/prior."""
    if not isinstance(intervals, int) or not 1 <= intervals <= 534:
        raise ValueError("intervals must be an integer in [1,534] at the fixed 1 ms prior cadence")
    p = dict(config)
    if not math.isclose(float(p["width_m"]), .016, rel_tol=0, abs_tol=1e-14) or not math.isclose(
            float(p["height_m"]), .008, rel_tol=0, abs_tol=1e-14):
        raise ValueError("The frozen K12 geometry requires the 16 mm by 8 mm channel")
    required = ("vn_mean_m_s", "rho_n_kg_m3", "rho_s_kg_m3", "mu_n_Pa_s", "kappa_m2_s",
                "alpha", "alpha_prime", "alpha_v", "beta_v", "particle_radius_m",
                "particle_density_kg_m3", "temperature_K", "gravity_y_m_s2")
    if any(k not in p or not math.isfinite(float(p[k])) for k in required):
        raise ValueError("Reference configuration is missing finite physical constants")
    if any(float(p[k]) <= 0 for k in ("vn_mean_m_s", "rho_n_kg_m3", "rho_s_kg_m3",
                                      "mu_n_Pa_s", "kappa_m2_s", "particle_radius_m", "beta_v")):
        raise ValueError("Positive mean normal velocity and material scales required")
    U = float(p["vn_mean_m_s"])
    vs = -p["rho_n_kg_m3"]/p["rho_s_kg_m3"]*U
    if "vs_mean_m_s" in p and not math.isclose(float(p["vs_mean_m_s"]), vs, rel_tol=1e-10, abs_tol=1e-14):
        raise ValueError("Mean velocities violate the zero total mass-flux convention")
    p["vs_mean_m_s"] = vs
    equilibrium_L = (p["alpha_v"]/p["beta_v"]*(U-vs))**2
    if "line_density_m2" in p and not math.isclose(float(p["line_density_m2"]), equilibrium_L, rel_tol=1e-9):
        raise ValueError("This fixed-L0 baseline requires the prescribed equilibrium line density")
    p["line_density_m2"] = equilibrium_L
    # Used solely by the source-Fourier Frozen-B projection, not by the domain.
    p["box_m"] = .01
    p["domain_m"] = [.016, .008]
    p["periodic_axes"] = ["y"]
    p["geometry"] = "channel"
    radius = p["particle_radius_m"]
    volume = 4*math.pi*radius**3/3
    rho = p["rho_n_kg_m3"]+p["rho_s_kg_m3"]
    ell_segment = 2*radius
    denominator = (1-p["alpha_prime"])**2+p["alpha"]**2
    gamma = p["rho_s_kg_m3"]*p["kappa_m2_s"]*p["alpha"]/denominator
    gamma_prime = (p["rho_s_kg_m3"]*p["kappa_m2_s"]
                   *(p["alpha_prime"]*(1-p["alpha_prime"])-p["alpha"]**2)/denominator)
    constants = dict(mass=(p["particle_density_kg_m3"]+rho/2)*volume, vol=volume,
                     drag=6*math.pi*p["mu_n_Pa_s"]*radius,
                     eg=ell_segment*gamma, egp=ell_segment*gamma_prime,
                     kf=p["rho_s_kg_m3"]*p["kappa_m2_s"]*ell_segment,
                     gravity=volume*(p["particle_density_kg_m3"]-rho)*p["gravity_y_m_s2"])
    n_vortices = int(p.get("n_vortices", max(2, 2*round(equilibrium_L*.016*.008/2))))
    spacing = math.sqrt(.016*.008/n_vortices)
    tension = p["rho_s_kg_m3"]*p["kappa_m2_s"]**2/(4*math.pi)*math.log(spacing/1e-10)
    return dict(name="cf_sim_reference", kind="channel", physics=p, constants=constants,
                hr=.5*(1+2.5*(1-1e-9)), tension_twice_N=2*tension, K=12,
                domain_m=[.016, .008], intervals=intervals, start_s=0.,
                duration_s=intervals*DT,
                acceleration_convention="partial_t u_i + sum_j u_j partial_j u_i",
                baseline_acceleration_transpose_corrected=True,
                effective_segment_m=ell_segment, tension_outer_m=spacing)


def prepare_physics(config: dict, intervals: int, device="cpu"):
    """Return (c, momentum_operator, zero_cold, full_frozen_B) for one case."""
    c = probability_physics(config, intervals)
    F = _baseline()
    runtime = F.runtime()
    from counterflow.cf_geometry import rescale_operator
    template = torch.load(OPERATOR, map_location="cpu", weights_only=False)
    # rescale_operator preserves viscosity matrices; all material parameters
    # affecting them or the geometry must equal the 2 K source operator.
    physical_invariants = ("temperature_K", "rho_n_kg_m3", "rho_s_kg_m3", "mu_n_Pa_s",
                           "kappa_m2_s", "alpha", "alpha_prime", "alpha_v", "beta_v",
                           "particle_radius_m", "particle_density_kg_m3")
    material_roundoff = {}
    for key in physical_invariants:
        # The helium interpolation polynomial evaluates with ~5e-11 relative
        # differences across the Linux reference run and historical Windows
        # export. This tolerance admits that roundoff, not different physics.
        if not math.isclose(float(c["physics"][key]), float(template["p"][key]), rel_tol=1e-9, abs_tol=1e-15):
            raise ValueError("Frozen operator cannot be speed-rescaled after changing "+key)
        material_roundoff[key] = float(c["physics"][key])/float(template["p"][key])-1
    data = rescale_operator(template, c["physics"])
    # Preserve the exact supplied viscosity and mean counterflow even across
    # the harmless source-platform property-polynomial roundoff above.
    viscosity_ratio = ((c["physics"]["mu_n_Pa_s"]/c["physics"]["rho_n_kg_m3"])
                       /(template["p"]["mu_n_Pa_s"]/template["p"]["rho_n_kg_m3"]))
    data["D"], data["d0"] = data["D"]*viscosity_ratio, data["d0"]*viscosity_ratio
    data["v0"] = data["v0"].clone()
    data["v0"][1, :, 1] = c["physics"]["vs_mean_m_s"]
    op = runtime.FrozenMomentum(runtime.make(data, device))
    if op.m != 325 or op.nu_s != 0 or op.friction_factor != 1:
        raise ValueError("Expected frozen K12/325, nu_s=0, F_ns=1 momentum operator")
    cold = torch.zeros((2, op.m), device=device, dtype=torch.float64)
    weights = op.V.new_tensor(np.tile(np.polynomial.legendre.leggauss(34)[1]/2/52, 52))
    mass = torch.einsum("fpm,p,fpn->fmn", op.V, weights.repeat_interleave(2), op.V)/c["physics"]["vn_mean_m_s"]**2
    chol = torch.linalg.cholesky(mass)
    eye = torch.eye(op.m, device=op.V.device, dtype=op.V.dtype).expand(2, -1, -1)
    c["decode"] = torch.linalg.solve_triangular(chol.transpose(-1, -2), eye, upper=True)
    decode, encode, ring, checks = F.wall_normalization(c, op)
    c.update(initial=cold.clone(), pinn_decode=decode, pinn_encode=encode,
             pinn_ring=ring, pinn_spatial_normalization=checks)
    projected = runtime.ProjectedPrior(c, op, intervals=intervals)
    factor, inverse, factor_checks = F.output_preconditioning(c, projected)
    c.update(pinn_output_preconditioner=factor, pinn_output_preconditioner_inverse=inverse,
             pinn_output_preconditioning=factor_checks,
             operator_source=dict(path=str(OPERATOR), sha256=hashlib.sha256(OPERATOR.read_bytes()).hexdigest(),
                                  speed_ratio=c["physics"]["vn_mean_m_s"]/template["p"]["vn_mean_m_s"],
                                  material_relative_roundoff=material_roundoff,
                                  viscosity_ratio=viscosity_ratio,
                                  historical_observations_read=False,
                                  decoder_source="fixed-geometry mass Cholesky; no observations"))
    return c, op, cold, F.WallResidualPrior(projected)


def make_field(c, cold, seed=0, duration_s=None):
    duration = c["intervals"]*DT if duration_s is None else float(duration_s)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Positive finite field duration required")
    return _baseline().NeuralField(
        cold, c["pinn_ring"], seed=seed, duration=duration,
        decode=c["pinn_decode"], encode=c["pinn_encode"],
        output_preconditioner=c["pinn_output_preconditioner"])


def geometry(c, device="cpu", dtype=torch.float64):
    return _baseline().runtime().core.channel_geometry(c["physics"]["vn_mean_m_s"],
                                                       c["physics"]["vs_mean_m_s"], str(device), dtype)


def queries_from_coefficients(c, a, adot, z):
    """Physical (v, grad v, material acceleration), preserving field gradients."""
    shape = z["time"].shape
    x = z["x"].to(a).reshape(-1, 2)
    a = a.reshape(-1, 2, a.shape[-1])
    adot = adot.reshape_as(a)
    b = geometry(c, a.device, a.dtype).evaluate(x)
    v = b["v0"]+torch.einsum("pfm,pfmj->pfj", a, b["V"])
    g = b["g0"]+torch.einsum("pfm,pfmij->pfij", a, b["G"])
    ut = torch.einsum("pfm,pfmj->pfj", adot, b["V"])
    acceleration = ut+torch.einsum("pfij,pfj->pfi", g, v)
    return tuple(value.reshape(*shape, *value.shape[1:]) for value in (v, g, acceleration))


def queries(c, field, z):
    a, adot = field.coefficients_and_dt(z["time"].reshape(-1)*DT)
    return queries_from_coefficients(c, a, adot, z)


def batched_rhs(op, a):
    return _baseline().batched_rhs(op, a)


def residual_loss(a, adot, op, prior, nfit):
    return _baseline().residual_loss(a, adot, op, prior, nfit)


def symmetric_regularization(q, ring):
    return _baseline().symmetric_regularization(q, ring)
