"""Self-contained native-endpoint tensors for frozen-model gradient checks.

This fixture has no dependency on the each-case probability training program
or on any prior experiment directory.
"""
import torch
import field_adapter as F


def synthetic_data():
    config = dict(width_m=.016, height_m=.008, temperature_K=2.,
        vn_mean_m_s=.00602109043344241, rho_n_kg_m3=80.31489032991522,
        rho_s_kg_m3=65.30311194021488, mu_n_Pa_s=1.4031953999983464e-6,
        kappa_m2_s=9.97e-8, alpha=.28088939999963713,
        alpha_prime=-.00020682100000024572, alpha_v=.17149642285510036,
        beta_v=6.50846808739152e-8, particle_radius_m=2.3e-6,
        particle_density_kg_m3=145.6, gravity_y_m_s2=-9.81)
    c = F.probability_physics(config, 6)
    config = c['physics']
    pids = torch.tensor([2,7,1,6]).repeat_interleave(3)
    frame = torch.arange(3).repeat(4)
    start = frame.double()*2
    time = start[:, None]+torch.tensor([.5,1.5], dtype=torch.float64)[None]
    n, m = time.shape
    state = torch.tensor([0,1,0,1]).repeat_interleave(3)
    z = dict(pid=pids, state=state, frame=frame, segment=pids.clone(),
             start=start, end=start+2, time=time, dt=torch.full((n,m), .001, dtype=torch.float64),
             x=torch.full((n,m,2), .004, dtype=torch.float64), row_id=torch.arange(n),
             w=torch.tensor([[.03,.7]], dtype=torch.float64).repeat(n,1),
             h=torch.zeros(n,m,5,4, dtype=torch.float64),
             target=torch.tensor([[.03,1.3]], dtype=torch.float64).repeat(n,1))
    velocity = torch.zeros(n,m,2,2, dtype=torch.float64)
    velocity[...,0,1] = config['vn_mean_m_s']
    velocity[...,1,1] = config['vs_mean_m_s']
    gradient = torch.zeros(n,m,2,2,2, dtype=torch.float64)
    acceleration = torch.zeros_like(velocity)
    return dict(c=c,z=z,jets=(velocity,gradient,acceleration),
                meta=dict(case='synthetic', physics_config=config, intervals=6))
