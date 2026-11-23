# Manuscript baseline · 2026-11-22

The SFHFlow manuscript lives in `01_Documentation/manuscript/counterflow_reconstruction/`.
This Git version tracks the observation and inversion implementations, frozen
experiment source, and the versioned workspace runner. The manuscript, simulated
trajectories, observations, model checkpoints, evaluation grids and deployment
archives remain in their numbered workspace blocks. The
[baseline manifest](../config/manuscript_baseline.json) records their locations
relative to SFHFlow and their SHA-256 values.

| Case | Probability model used in inversion | Fields used for the manuscript |
| --- | --- | --- |
| Periodic 1.5 and 4.0 mm/s | Three frozen models from the periodic `joint` family, selected using the separate 6.5 mm/s validation case | Validation-selected PINN checkpoints within the 600-update budget; 200 updates retained as a comparison |
| CF217 at 2 K | Three frozen models from the earlier CF217 probability retraining, initialized from the periodic family | Validation-selected calibrated channel PINN within 200 updates; selected steps 60, 70 and 60 |

For both geometries the candidate continuous velocity fields enter a conditional
particle-displacement likelihood, and projected momentum residuals tie the fields
together. CF217 uses a wall-compatible spatial basis. Its directional residual
tolerance uses the second moment about zero of increments measured on the reference
trajectory; the field output amplitude is separately scaled. The CF217 supervision,
residual calibration and reference evaluation use the same simulated realization,
with distinct particle-ID splits for training, validation, and held-out scoring.
The manuscript presents no laboratory reconstruction.

The periodic implementation is in
[`experiments/pinn_twofluid_20261105/`](../experiments/pinn_twofluid_20261105/).
The CF training and calibrated-build sources are in
[`experiments/unified_reconstruction_20261113/`](../experiments/unified_reconstruction_20261113/)
and [`experiments/q217_calibrated_20261114/`](../experiments/q217_calibrated_20261114/).
The build derives the calibrated field controller from the pinned parent; the
actual shipped code and input assets are checked by the frozen bundle manifest.

From SFHFlow, enter a numbered block and run its own `run.py`:

```powershell
# 02: inspect a forward case configuration.
..\.venv\Scripts\python.exe -B run.py plan --q 217 --fps 180
# 03: verify the packaged CF probability workflow.
..\.venv\Scripts\python.exe -B run.py verify --q 217
# 04: verify the published periodic and calibrated channel inputs.
..\.venv\Scripts\python.exe -B run.py verify --geometry periodic --speed 4
..\.venv\Scripts\python.exe -B run.py verify --q 217
```

Use a new `--run-id` for work that writes results. Resume with `--resume` only
if the case, code and inputs still match the run's recorded contract. Paper
figures read the adopted results under 04, not the newest directory under
`runs/`. The CF release's CUDA results are preserved; local CPU diagnostics
are not substitutes for formal results.
