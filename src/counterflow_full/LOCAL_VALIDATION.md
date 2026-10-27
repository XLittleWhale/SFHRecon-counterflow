# Local validation status — 2026-10-26

Historical pre-server checks only. The full GPU result was subsequently returned
and adopted as the periodic simulation baseline on 2026-10-27. See the repository
`docs/FULLWINDOW_BASELINE.md` and its frozen metrics; the statements below record
what was known before that run, not its current completion status.

This is a prepared offline GPU experiment, **not a completed formal result**.

- Repository test suite: 90 tests passed (including five new full-window tests).
- New tests cover original-endpoint causality, conservative cell integration,
  Fourier phase/projection round-trip, source-compatible free/captured force,
  disjoint PID/state balancing, and chunked versus monolithic PDE gradients.
  The gradient equivalence tolerance is relative/absolute 1e-10.
- A single 95 ms background PDE forward pass was finite and retained exactly
  zero fluctuation coefficients; CPU time about 1.35 s (two threads). This is
  **not** a GPU timing estimate and does not predict full optimization runtime.
- A two-interval, two-epoch/two-update smoke execution passed all pipeline stages:
  prepare, fresh training, calibration, prior calibration, inversion, evaluation.
  It checks program connectivity, not scientific performance or convergence.
- Plot rendering was inspected; 13 PNG/PDF/SVG figure groups are supported.
  Smoke VTKHDF structure was checked: ImageData, 40×40, identical cell-centred
  coordinates and reference/reconstruction times. No ParaView GUI was available
  for an application-level reader check.
- Nine lossless input packs contain 96 fields each (5–100 ms), dimensions
  1024×1024×2 fluids×2 velocity components, float64. Total package ≈12.54 GiB.
- All conditions provide 421,242 valid original one-ms intervals. Seven training
  conditions contribute 110,000 fitting-PID captured samples. No fixed sample cap.
- Final 4 mm/s case: 512 PIDs; 46,939 eligible intervals, 1,701 state-transition
  intervals excluded, no nonfinite intervals. Fitting: 28,067; selection: 9,434;
  testing: 9,438. Unique PID split: 306/103/103 (free/captured overlap within PID).
- No CUDA device is available locally. GPU kernel execution, formal convergence,
  full-window reconstruction accuracy and full-run wall time remain unverified.

The original source metadata reports a 4 ms duration despite the saved trajectory
running to 100 ms; this package validates and uses actual saved timestamps.
The endpoint anchor geometry and saved last-microstep midpoint tension are not
identical quantities. In the 4 mm/s input their maximum discrepancy is 1.557%
of 2T; it is audited, not silently turned into a new latent label.

Do not use `local_smoke` figures in the paper as formal results. Formal outputs
are written by default to `fullwindow_4mm_results/Results` on the server. Return
that output folder for scientific acceptance before replacing existing figures.
