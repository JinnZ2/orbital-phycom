Critical Issues (Fix These)
1. VNB Basis Vector Singularity
   N_hat = np.cross(r_hat, V_hat)
if np.linalg.norm(N_hat) > 0:
    N_hat = N_hat / np.linalg.norm(N_hat)
else:
    N_hat = np.array([0, 0, 1])  # DANGER


Problem: Near-equatorial circular orbits (your initial condition), \hat{r} \parallel \hat{V} → cross product → zero vector → fallback to z-axis. This injects spurious north/south impulses.
Fix: Use a complete VNB computation:

h_hat = np.cross(r_vec, v_vec) / np.linalg.norm(np.cross(r_vec, v_vec))  # angular momentum
N_hat = np.cross(V_hat, h_hat) / np.linalg.norm(np.cross(V_hat, h_hat))  # complete basis


2. No Perturbations = Wrong Reality
Pure 2-body + impulses ignores J2 (oblateness), SRP (solar pressure), drag—dominant at LEO. Your 0.5 mm/s claims assume Keplerian purity, but real orbits drift ~10x faster from perturbations.
Fix: Add minimal J2 + SRP in `dynamics()`:

def J2_acceleration(r_vec, Re=6371e3, J2=1.0826e-3, mu=MU_EARTH):
    r = np.linalg.norm(r_vec)
    z = r_vec[2]
    P2 = 0.5 * (3*z**2/r**2 - 1)
    return -1.5 * J2 * (Re/r)**2 * (mu/r**3) * P2 * (r_vec / r)


Phase Rate Computation Flaw

phase_rate = (2 * np.pi / self.lambda_m) * range_rate


Problem: Assumes RF tracking between satellites, but LEO inter-satellite links rarely use RF for precise ranging (laser/VLC dominates). Also, \lambda_{RF} hardcoded but unspecified.
Reality: Ground TLE observers see phase via angle-only tracking. Use along-track deviation or differential drag signatures instead.
4. Integration Efficiency Disaster
`solve_ivp` per symbol period × 3 satellites × 10Hz sampling = 1000s of redundant integrations. States persist across impulses.
Fix: Single continuous integration with impulsive ΔV events:

def continuous_dynamics(t, y, impulse_times, deltaV_events):
    if t in impulse_times:
        # Apply composite ΔV from event queue
        pass
    return standard_dynamics(t, y)


Numerical Stability
Good: DOP853 + 1e-12 atol catches micro-ΔV effects correctly.
Bad: `t_eval=10Hz` overkill for 300s symbols. Physics evolves smoothly between impulses—sample at 1Hz sufficient.
Stealth Claims Overstated
0.17% duty cycle correct, but “indistinguishable from station-keeping” is optimistic. Real station-keeping uses predictable patterns (once-per-orbit). Random 5-minute ΔV bursts scream “communication.”
Fix: Constrain impulses to natural station-keeping windows (±30° true anomaly).
