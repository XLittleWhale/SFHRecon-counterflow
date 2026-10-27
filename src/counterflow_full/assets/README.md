# Full-window calibration and 4 mm/s initial-field inversion

This folder is self-contained. It includes lossless source fields, native 1 ms
trajectories, physical labels, code and configuration. No earlier project tree
or checkpoints are needed. Existing production results are not overwritten.

## Server execution

Copy **this entire folder** to `/root/autodl-tmp/FullWindowCalibration_20261026`.
Use the server's CUDA-enabled PyTorch environment (PyTorch >= 2.2), then:

```bash
cd /root/autodl-tmp/FullWindowCalibration_20261026
python -m pip install -r requirements.txt
python run_all.py --phase check
python run_all.py --out /root/autodl-tmp/fullwindow_4mm_results
```

The default run requires CUDA; it never silently starts long CPU training.
Dependencies do not install/replace PyTorch. If `check` fails, install the CUDA
PyTorch build appropriate for your server before continuing. No external
datasets are downloaded. The manifest reports exact package size; keep at least
4 GiB free in addition to the package. Avoid retaining a second extracted copy
or compressed archive on a 50 GiB data disk.

A bounded wiring test is available separately:

```bash
python run_all.py --smoke --out /root/autodl-tmp/fullwindow_smoke
```

Smoke uses two intervals and two training/inversion updates. **It is not the
requested formal reconstruction.** A smoke output cannot be resumed as a formal
run. Local CPU smoke is permitted only with explicit `--device cpu --smoke`.

Rerun the original command after interruption. Completed phases and completed
cases are reused; training and inversion resume at saved checkpoints. Calibration
restarts its small 4-parameter fit if interrupted. Policy/source fingerprints
must remain unchanged; edits require a new output directory. Progress prints at
case/epoch/validation-check boundaries, not continuous disk polling. The first
inversion check reports actual timing in its trace; runtime is hardware-dependent
and the complete 95 ms optimization can take hours. It is deliberately not the
old wall-clock-truncated diagnostic. Iteration-limit stops are reported honestly.

## Fixed protocol

- All native intervals from 5 to 100 ms, original 512 PIDs, with a causal 5 ms
  observed history. Only nonfinite and observed state-changing intervals are
  ineligible. No fixed count cap, microsteps or interpolated observation targets.
- `N=40` coarse reference cells, `K=12` Fourier modes, `M=64` PDE integration
  quadrature. N, K and M are different quantities. Controls progress K2/4/8/12;
  the PDE and observation operator always retain the full K12 representation.
- Seven training conditions; 6.5 mm/s is calibration; 4 mm/s is final evaluation.
  Each physical PID uses residue modulo 5: 2/3/4 fitting, 1 selection, 0 testing.
  The 4 mm/s case has been examined historically: this is not a fresh blind test.
- Two freshly initialized physical sign/anchor networks, max 240 epochs, minimum
  60, patience 30. Reference velocities are conditions, never network outputs.
- Fixed coarse-force Gaussian moments from full-minus-coarse affine force on
  training conditions only. Not a learnable field closure or per-target residual.
- Calibration fits scalar latent/sign temperatures and two state noise scales.
  The prior grid 0.3/1/3 is selected using calibration-case two-fluid error, not
  4 mm/s truth. Two final initializations are selected by held-out trajectory NLL.
- Initial fields are the only 4D-Var unknowns. No delta-F, learned fluid closure,
  field supervision, hidden anchor labels, or truth initialization in final fitting.

See `counterflow_full/MATHEMATICS.md` for the full equations and qualifications.

## Deliverables to return

Return `fullwindow_4mm_results` (not the large input folder). It contains selected
weights, calibration, force moments, optimization traces, audits, initial and
full-window metrics, and `Results` with PNG/PDF/SVG figures and ParaView VTKHDF
series on identical coordinates. `Results/report.json` includes reference-initial
PDE replay, background, inverse, PID-held-out state scores, probability coverage,
integration-step refinement and data counts. High error is reported, not hidden
by substituting a short window. No result is called accurate based on NLL alone.
