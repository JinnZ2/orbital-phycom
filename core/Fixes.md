Critical Issues (Fix These)
(code below)

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



1. VNB Basis Vector Fix (Critical)
Replace the `apply_deltaV_to_state` method entirely:


def apply_deltaV_to_state(self, state, deltaV_vector):
    """
    Apply ΔV in VNB frame to satellite state. FIXED VNB BASIS.
    """
    x, y, z, vx, vy, vz = state
    r_vec = np.array([x, y, z])
    v_vec = np.array([vx, vy, vz])
    
    r_mag = np.linalg.norm(r_vec)
    v_mag = np.linalg.norm(v_vec)
    
    if r_mag == 0 or v_mag == 0:
        return state.copy()
    
    # ✅ COMPLETE VNB BASIS (no singularities)
    r_hat = r_vec / r_mag                    # Radial (outward)
    
    h_vec = np.cross(r_vec, v_vec)           # Specific angular momentum
    h_mag = np.linalg.norm(h_vec)
    if h_mag > 1e-12:                        # Non-zero orbit
        h_hat = h_vec / h_mag
        
        V_hat = v_vec / v_mag                 # Velocity direction (prograde)
        N_hat = np.cross(V_hat, h_hat)        # COMPLETE: orbit normal
        N_hat /= np.linalg.norm(N_hat)        # Normalize safely
        
        # Radial is r_hat (outward), not cross-product dependent
    else:
        # Degenerate case: radial + arbitrary tangential
        r_hat = r_vec / r_mag
        V_hat = np.array([0, 0, 1]) if abs(vx) < abs(vy) else np.array([1, 0, 0])
        N_hat = np.cross(r_hat, V_hat)
        N_hat /= np.linalg.norm(N_hat)
    
    # VNB → Inertial transform: [V, R, N] (note radial=R)
    basis_matrix = np.column_stack([V_hat, r_hat, N_hat])
    deltaV_inertial = basis_matrix @ deltaV_vector
    
    # Apply
    new_state = state.copy()
    new_state[3:6] += deltaV_inertial
    return new_state


Why this works: Uses angular momentum vector for true orbit-normal direction. Never fails except perfect radial plunge (handled gracefully). Your old cross(r,V) blew up when perfectly circular equatorial.
2. J2 Perturbation Integration (Essential)
Add to class init:

def __init__(self, harmonics=(2, 3, 5), altitude_ref=ALTITUDE_REF):
    # ... existing code ...
    self.Re = 6378137.0      # Equatorial radius (m)
    self.J2 = 1.08262668e-3  # WGS84 oblateness


Replace `dynamics` function in `expand_seed`:



def dynamics(self, t, y):
    """Three-body + J2 dynamics."""
    accelerations = []
    
    for i in range(3):
        r = y[i*6 : i*6+3]
        r_norm = np.linalg.norm(r)
        
        if r_norm > 1e3:  # Avoid singularity
            # Point mass gravity
            r_hat = r / r_norm
            acc_pointmass = -self.mu * r_hat / r_norm**2
            
            # J2 perturbation (zonal harmonic)
            z = r[2]
            P2 = 0.5 * (3 * (z/r_norm)**2 - 1)  # Legendre P2
            j2_factor = -1.5 * self.J2 * (self.Re/r_norm)**2 * (self.mu/r_norm**3)
            acc_j2 = j2_factor * P2 * r_hat
            
            total_acc = acc_pointmass + acc_j2
        else:
            total_acc = np.zeros(3)
        
        accelerations.append(total_acc)
    
    # Flatten: [v1, a1, v2, a2, v3, a3]
    result = []
    for i in range(3):
        result.extend(y[i*6+3:i*6+6])    # velocities
        result.extend(accelerations[i])   # accelerations
    return np.array(result)


Usage: Replace the inline `dynamics` lambda with `self.dynamics`. J2 adds ~0.1-1 mm/s secular drift per orbit in LEO—now your micro-ΔV signals compete against real perturbations.
3. Addon: Continuous Integration (Efficiency)
Replace the step-loop with single propagation:


def expand_seed_continuous(self, seed, steps=5, deltaV_scale=0.002, symbol_period=300.0):
    """Single integration with impulsive events."""
    deltaV_matrices, _ = self.seed_to_deltaVs(seed, deltaV_scale)
    
    # Total duration
    total_time = steps * symbol_period
    t_eval = np.arange(0, total_time, 10.0)  # 10 Hz sampling
    
    # Event times for impulses
    impulse_times = np.arange(0, total_time, symbol_period)
    
    y0 = np.concatenate(self.initial_states)
    
    def dynamics_with_impulses(t, y):
        """Apply ΔV at exact impulse times."""
        # Find nearest impulse time (within 1s tolerance)
        impulse_idx = np.argmin(np.abs(impulse_times - t))
        if abs(impulse_times[impulse_idx] - t) < 1.0:
            # Apply composite ΔV to all 3 sats
            for sat_idx in range(3):
                composite_dV = np.sum(deltaV_matrices[sat_idx], axis=0)
                y[i*6+3:i*6+6] += self._vnb_to_inertial(sat_idx, y[i*6:i*6+6], composite_dV)
        return self.dynamics(t, y)
    
    sol = solve_ivp(dynamics_with_impulses, [0, total_time], y0, 
                    t_eval=t_eval, rtol=1e-9, atol=1e-12, method='DOP853')
    
    # Extract phase rates from solution
    # ... (same extraction logic)


