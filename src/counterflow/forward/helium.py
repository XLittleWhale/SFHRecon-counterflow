"""Standalone subset of the project's He-II temperature fits.

Coefficients match VFM/2DVFM/Reconstruction/physics/config.py.  Kept locally
so the 2DCounterflow folder remains directly runnable on a CUDA server.
Polynomials use ascending powers of temperature in kelvin and SI outputs.
"""
from __future__ import annotations

import math


MU_N = (7.42116853e-04, -2.72872406e-03, 4.18086013e-03,
        -3.40692439e-03, 1.55634875e-03, -3.77813201e-04,
        3.80801796e-05)
RHO_N = (1.80183130e-01, -2.48558984e+01, 4.01282654e+02,
         -2.48742413e+03, 7.85114896e+03, -1.42666651e+04,
         1.57974461e+04, -1.08279105e+04, 4.48686634e+03,
         -1.02983154e+03, 1.00498992e+02)
RHO_S = (144.94795, 25.18243857, -406.9535624, 2509.83455,
         -7886.879518, 14286.34985, -15784.94029, 10803.13498,
         -4471.875738, 1025.612443, -100.0317144)
ENTROPY = (6.57311375e+06, -4.38113590e+07, 1.30656739e+08,
           -2.29500549e+08, 2.62852401e+08, -2.05055435e+08,
           1.10322083e+08, -4.04131685e+07, 9.64576203e+06,
           -1.35448196e+06, 8.49760567e+04)
ALPHA = (1.40348955e+02, -5.78781222e+02, 9.86538957e+02,
         -8.89757600e+02, 4.47861205e+02, -1.19283945e+02,
         1.31360361e+01)
ALPHA_PRIME = (1.25434053e-01, -5.36094653e-01, 7.12246528e-01,
               -3.62191839e-01, 6.21935645e-02)
C1 = (-37875.4603432003, 161027.467671278, -292620.794399981,
      294653.759992904, -177568.113431148, 64045.2095387393,
      -12802.0813284325, 1094.15041473176)
C2 = (2421.03719768316, -7059.46514061684, 8208.38710509775,
      -4758.60939944746, 1376.23800786335, -158.752493718344)


def _poly(coefficients: tuple[float, ...], temperature_K: float) -> float:
    # Match the canonical source's evaluation order exactly.
    return sum(value*temperature_K**i for i, value in enumerate(coefficients))


def properties(temperature_K: float) -> dict[str, float]:
    if not math.isfinite(temperature_K) or not 1.3 <= temperature_K <= 2.1:
        raise ValueError("temperature_K must be finite and in this model's 1.3–2.1 K range")
    rho_n = _poly(RHO_N, temperature_K)
    rho_s = _poly(RHO_S, temperature_K)
    mu_n = _poly(MU_N, temperature_K)
    entropy = _poly(ENTROPY, temperature_K)
    alpha = _poly(ALPHA, temperature_K)
    alpha_prime = _poly(ALPHA_PRIME, temperature_K)
    alpha_v = (_poly(C1, temperature_K)/1e6)*rho_s*entropy*temperature_K
    beta_v = _poly(C2, temperature_K)/1e8
    values = dict(rho_n_kg_m3=rho_n, rho_s_kg_m3=rho_s,
                  mu_n_Pa_s=mu_n, alpha=alpha, alpha_prime=alpha_prime,
                  alpha_v=alpha_v, beta_v=beta_v)
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError("non-finite helium material property")
    if min(rho_n, rho_s, mu_n, alpha_v, beta_v) <= 0:
        raise ValueError("nonphysical helium material property")
    return values
