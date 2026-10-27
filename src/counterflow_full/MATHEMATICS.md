# Full-window periodic baseline: mathematical contract

The completed 2026-10-26 GPU run was adopted on 2026-10-27. This contract describes
that simulation baseline, not an upgrade of the separate CF implementation.

## 1. Target, representation and separation of information

The data are one centimetre-square periodic simulation per imposed normal-fluid
mean U. We use only saved positions at Δ=1 ms and their free/captured state;
the source's 20 μs substeps are never training histories or inversion observations.
For each of 512 physical PIDs, the first five saved intervals supply history.
The reconstruction window is [5,100] ms, with one pair of unknown initial fields.
Finite, same-state endpoint intervals are eligible. A state switch inside an
interval whose endpoints have the same state cannot be detected from these
observations; this remains a source of model discrepancy.

Let C₄₀ be conservative integration of the periodic piecewise-linear 1024-node
source interpolant into 40×40 cells. Let P₁₂ be the divergence-free square Fourier
projection, with phase correction for cell-centred coordinates. The reference is

    u_ref(t) = prescribed_mean + P₁₂ C₄₀ (v_source(t) − prescribed_mean).

The unprojected C₄₀ source is retained for a separate representation-error audit.
N=40 reference cells, K=12 modal order and M=64 PDE quadrature are NOT identical
resolution parameters. There are d=4K(K+1)=624 real coefficients per fluid,
1248 unknown initial coefficients. The imposed means are fixed and not inferred.

For integer wavevectors k in the unique half of the square |kx|,|ky|≤K, e_k is
the unit vector perpendicular to k. With dimensionless coefficient vector a_f,

    u_f(x,t) = mean_f + U Σ_k e_k[-a^sin_fk(t) sin(2πk·x/ℓ)
                                         +a^cos_fk(t) cos(2πk·x/ℓ)].

Controls are θ=(a_n(t₀),a_s(t₀)); all later a(t) are deterministic PDE outputs.

## 2. Strong-constraint momentum/Vinen evolution

The unchanged periodic momentum system is projected onto the divergence-free
basis (pressure disappears under this projection):

    ∂t u_n + P(u_n·∇u_n) = ν_n Δu_n + P F_ns/ρ_n,
    ∂t u_s + P(u_s·∇u_s) = ν_s Δu_s − P F_ns/ρ_s,
    F_ns = −(2/3) α ρ_s κ L (u_n−u_s),
    ν_n = μ_n/ρ_n,   ν_s = ακ.

The zero modes are prescribed counterflow means; the implicit mean driving
balances the discarded mean force. Thus these are NOT unforced mean-decay
equations. L obeys

    ∂t L + u_s·∇L = α_v |u_n−u_s| L^(3/2) − β_v L²
                     + γ |u_n−u_s|^(5/2),
    L(t₀)=L_ref,   q=log(L/L_ref).

γ and all physical constants come from the existing exported operator. q is
evolved, not inverted; exp(q) retains the existing [-12,8] safety clamp, which is
audited in outputs. RK4 uses two substeps per saved millisecond and is checked
against four. No δF_ns, added learned fluid closure, reference L(t), or field
supervision is present in the final inverse objective.

Reference-initial replay solves this PDE from θ_ref. Its discrepancy measures
model compatibility over the long window. It is not a rigorous lower bound on
the error of every possible alternative initial state.

## 3. Causal particle observation and physical latents

At saved index j, use w_j=(x_j−x_(j−1))/(UΔ), pilot query
X_j=x_j+(x_j−x_(j−1))/2, and a five-secant history h_j. History at half-step times
is linearly interpolated using x≤j; its final half-step is a constant-velocity
extrapolation. x_(j+1) appears ONLY in the target

    Y_j=(x_(j+1)−x_j)/(UΔ).

The source force convention, confirmed against the existing implementation, is

    D_f = ∂t u_f + ∇(|u_f|²)/2,

not the usual material derivative (u_f·∇)u_f. At the pilot query, evaluate u_f,
∇u_f at the mean of the two PDE coefficients, and ∂t u_f by their secant.

Let R(x,y)=(-y,x), D=Stokes drag, E/E′=dissipative/reactive trapped friction,
K_m=Magnus coefficient, m=effective particle mass, V=particle volume. Then

    dv_p/dt = −A_q v_p + b_q + (2T/m) η(δ),
    A_q = [(D+E)I − q(K_m+E′)R]/m,
    b_q = [(D+E)u_n − q K_m R u_s − q E′ R u_n
            + (3V/2)(ρ_n D_n+ρ_s D_s) + F_g]/m,
    η(δ)=δ/sqrt((h/a_p)²+|δ|²),   q∈{-1,+1}.

δ is the common planar anchor offset divided by particle radius a_p. The two
virtual anchors have equal planar offsets and opposite out-of-plane offsets;
this symmetry is checked over the input data. Labels use instantaneous saved
geometry, not saved `last_tension` (the latter is a microscopic midpoint force).

For free particles, A_f=(D/m)I, b_f=[D u_n+(3V/2)(ρ_n D_n+ρ_s D_s)+F_g]/m,
and the anchor/Magnus/trapped-friction terms vanish. Its genuine acceleration
sensitivity to u_s is retained: we do not artificially stop that gradient.

The GRU model, identical in size to the established physical model, gives

    q_φ(q,δ|h,c) = π_φ(q|h,c) N(δ; μ_φq(h,c), S_φq S_φqᵀ),
    c=[log(U/.004), log(h/a_p), log 2, log(Δ(D+E)/m), α,
       vec(u_f/U−w), vec(Δ∇u_f)].

Neither absolute position, absolute time, PID nor case identity is an input.
Reference fields are training CONDITIONS, not velocity-output labels. This
does not by itself guarantee generalization or likelihood identifiability.

## 4. Explicit coarse-force discrepancy, not a free fitted correction

At the same saved-time pilot location, calculate full-resolution and reference
coarse b_q. The full-grid source acceleration is assembled on its grid BEFORE
bilinear sampling. Train the following fixed statistical correction separately:

    e = A_q^(-1)(b_q^full − b_q^coarse)/U,
    e|state ~ N(μ_state, Σ_state).

Moments are balanced by condition and physical PID, using training conditions
and fitting PIDs only. The −A_q v_p and tension terms are excluded from this
label, preventing double allocation of these known physical contributions.
This is a low-parameter marginalized missing-force model, NOT learned fluid
closure and NOT an arbitrary residual fitted separately at each target point.
It captures marginal moments only; conditional/state-transition/model biases
may remain. Extrapolation of its law from reference to candidate fields is a
model assumption, explicitly tested by the final likelihood ray.

Define response matrices for one frozen-coefficient saved interval:

    H_q=(ΔA_q)^(-1)(I−exp(−ΔA_q)),
    B_q=(ΔA_q)^(-1)(I−H_q),   M_q=Δ B_q A_q.

Analytically integrating v_p and position gives

    m_qδ = H_q w + (Δ/U) B_q[b_q+(2T/m)η(δ)] + M_q μ_state,
    C_q = M_q Σ_state M_qᵀ + σ_state² I,
    p(Y|h,c,state=cap) = Σ_q π_q ∫ N(Y;m_qδ,C_q) N(δ;μ_q,S_qS_qᵀ)dδ.

For free state there is one Gaussian, without η. Thus the likelihood is on
original position increments, not unstable twice-differentiated observations.
Gauss–Hermite quadrature approximates the two-dimensional integral; inversion
uses degree 12, final scoring degree 20, with numerical agreement reported.

## 5. Training, calibration and inverse objective

Fresh random seeds train φ with all eligible captured samples and the weighted
loss (reference conditions only):

    J_train(φ)=E_case,PID,time[−log q_φ(q*,δ*|h,c_ref)
                              −0.1 log p_φ(Y|h,c_ref)].

The training decoder uses fixed σ=.3 and untempered latent/sign probabilities.
Weight decay is 10^(-4). Validation selects a seed/checkpoint. Subsequent frozen-
network calibration on other PIDs fits scalar latent width τ, sign temperature
T_q and σ_free,σ_cap by position NLL, with τ∈[.25,4], T_q∈[exp(-1.4),exp(1.4)],
σ∈[.015,1.5]. Slopes versus U are NOT fitted from one validation condition.

For an inverse group G, S denotes the two observed states, P_s its physical PIDs,
and I_(p,s) all eligible intervals. Define the composite score

    L_G(θ)=1/|S| Σ_s 1/|P_s| Σ_p 1/|I_(p,s)|
                  Σ_(j∈I_(p,s)) −log p_φ(Y_pj|h_pj,c_pj(S_PDE(θ))).

Equal state/PID weighting prevents a long captured episode from dominating
simply because it has more saved frames. This is a COMPOSITE likelihood, not
an exact time-independent joint trajectory model; PID bootstrap is diagnostic
and no calibrated field-posterior uncertainty is claimed.

The actual optimization objective, dimensionless, is

    J(θ)=0.08 L_fit(θ)+λ R(θ,S_PDE(θ)),
    R=.003/24 mean_(t,f)||a_f(t)||²
      +.03/24 mean_(t,f)||a_f(t+Δ)−a_f(t)||²
      +.03/24 mean_f Σ_(r_m>2) [(r_m/2)² θ_fm]²,
    r_m=max(|kx_m|,|ky_m|).

K2→K4→K8→K12 activates more INITIAL controls; previous coefficients remain
adjustable. The PDE itself always contains K12 modes. Each step differentiates
the full 95 ms rollout and visits all fitting intervals, including all eligible
PIDs. Chunking sums an exact rollout gradient before one reverse PDE pass;
it is not temporal truncation or sparse minibatch inverse optimization.

The best stage/checkpoint and final restart are chosen solely by L_selection,
never by 4 mm/s field error. λ∈{.3,1,3} is fixed beforehand by a full 6.5 mm/s
calibration reconstruction, minimizing the worst of the two velocity RMS ratios
relative to its uniform baseline. This is a finite calibrated comparison, not
a proof of a globally best model/hyperparameter choice.

All PID splits are fixed: residues 2/3/4 fit, 1 select, 0 test (modulo 5). No
particle crosses splits even if it switches state. Seven other flow conditions
train φ/force moments. 4 mm/s references/latents never enter network learning,
noise calibration, inverse objective or checkpoint/restart selection.

## 6. Acceptance and limits

Report separately: initial and full-window vector RMS for both fluids, error
relative to fluctuations, modal-to-cell representation gap, spectrum, full
reference-initial PDE drift, timestep refinement, q-clamp activity, per-state
held-out NLL, predictive-mean displacement RMS, Rosenblatt 90% coverage, and the
reference-initial→fitted-initial likelihood line. Export same-coordinate reference,
inverse, background, reference-initial PDE and unprojected coarse VTKHDF series.

Low NLL alone is insufficient. A longer window and more particles can increase
information only along genuinely sensitive directions. They can also accumulate
persistent PDE/observation-model bias. This experiment quantifies both; it cannot
promise uniformly superior reconstruction or uniqueness of both recovered fields.
The 4 mm/s condition has been explored in previous work, so its evaluation is a
held-out fit assessment, not a newly untouched benchmark.
