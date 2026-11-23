# Manuscript simulation results · 2026-11-22

All cases are at 2 K. The two periodic cases use the validation-selected PINN
within 600 updates; CF217 uses the calibrated channel PINN within 200 updates.
The plots show the representative seed 301030, while the performance table
averages three paired seeds. Particle observation weights were frozen during
field optimization.

| Case | Fields and error | Spectra | Validation-selected steps |
| --- | --- | --- | --- |
| Periodic, 1.5 mm/s | [PDF](periodic_1p5_fields.pdf) | [PDF](periodic_1p5_spectra.pdf) | 600, 600, 550 |
| Periodic, 4.0 mm/s | [PDF](periodic_4p0_fields.pdf) | [PDF](periodic_4p0_spectra.pdf) | 530, 540, 510 |
| CF217, 217 mW/cm² | [PDF](cf217_fields.pdf) · [mean profiles](cf217_mean_profile.pdf) | [PDF](cf217_spectra.pdf) | 60, 70, 60 |

| Case | Held-out free NLL/ms | Held-out captured NLL/ms | Normal RMS (mm/s) | Superfluid RMS (mm/s) |
| --- | ---: | ---: | ---: | ---: |
| Periodic, 1.5 mm/s | −2.804 | −2.095 | 0.196 | 0.338 |
| Periodic, 4.0 mm/s | −2.358 | −1.668 | 0.581 | 0.809 |
| CF217 | 4.369 | 0.857 | 2.554 | 3.477 |

Periodic errors use the band-matched reference, while CF217 errors use the full
saved evaluation grid. In CF217 the same simulated realization supplies
probability supervision, residual calibration and reference fields; held-out
particles are separated by identity. The figures and values are from simulations.

[Full-precision metrics](metrics.json) · [Performance table (LaTeX)](performance_table.tex)
· [Case definitions (LaTeX)](reference_cases_table.tex)
· [Version and artifact fingerprints](../../config/manuscript_baseline.json)
