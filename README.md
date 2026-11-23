# Two-fluid counterflow reconstruction

## Manuscript baseline · 2026-11-22

The current manuscript compares the **periodic two-fluid PINN at 1.5 and 4.0 mm/s**
with the **calibrated CF217 channel PINN at 2 K**. The paper reports validation-selected
fields from the 600-update periodic runs and 200-update CF217 runs. The particle
observation model is frozen during field optimization. The CF217 reconstruction uses
a channel basis, a reference-calibrated directional residual tolerance, and a
separate scale for the field output.

- [Methods, case selection and workspace commands](docs/MANUSCRIPT_BASELINE.md)
- [Published figures, tables and metrics](results/manuscript_20261122/)
- [Version and artifact fingerprints](config/manuscript_baseline.json)
- `python -m workspace_runtime.manuscript --list` lists the manuscript cases from this checkout.
- `python -m workspace_runtime.manuscript --workspace <SFHFlow>` verifies the external assets in the organized workspace.

This version reports simulation results. Experimental reconstruction remains future
work. New runs use their own numbered output directories; the published cases read
the validation-selected checkpoints identified in the manuscript manifest.
