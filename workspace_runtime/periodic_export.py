"""Export the selected periodic checkpoint on its original spatial grid."""
import numpy as np

from .paths import SECTIONS
from .release import sha


def export(engine, speed, seed, checkpoint, target):
    d,*_ = engine.make_context(speed,seed)
    value = engine.load(checkpoint)
    reference = SECTIONS[4] / "results/periodic_baseline/fields" / engine.case_name(speed) / "reference_K12.npz"
    with np.load(reference,allow_pickle=False) as ref:
        x = ref["x_m"]
    # Resolve the same diagnostic Fourier grid and modes used in the published baseline.
    import evaluate_pinn
    evaluator = evaluate_pinn.module("workspace_periodic_parent", evaluate_pinn.HELPER)
    _, velocity, means = evaluator.resolve_fields(value["a"],d["c"])
    np.savez_compressed(target,a=value["a"].cpu().numpy(),relative_time_s=np.arange(96)*.001,
                       x_m=x,y_m=x,velocity_m_s=velocity,means_m_s=means,
                       step=value["step"],checkpoint_sha256=sha(checkpoint))
