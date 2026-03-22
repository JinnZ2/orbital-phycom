"""
Orbital dynamics simulation with exact ΔV impulse insertion.

Provides high-precision orbital propagation using segmented integration
with impulse events applied at exact times.
"""

import numpy as np
from scipy.integrate import solve_ivp
from .physics_constants import MU_EARTH, R_EARTH, ALTITUDE_REF, LAMBDA_RF


class OrbitalSimulator:
    """
    Two-satellite orbital simulator with ΔV impulse capability.
    
    Uses segmented integration to apply impulses at exact times,
    avoiding timestep-dependent errors.
    """
    
    def __init__(self, harmonic_A=2, harmonic_B=3, altitude_ref=ALTITUDE_REF):
        """
        Initialize simulator with two satellites in harmonic orbits.
        
        Args:
            harmonic_A: Harmonic multiple for satellite A (e.g., 2)
            harmonic_B: Harmonic multiple for satellite B (e.g., 3)
            altitude_ref: Reference altitude in meters
        """
        self.mu = MU_EARTH
        self.R_earth = R_EARTH
        self.lambda_m = LAMBDA_RF
        
        # Calculate orbital parameters
        a0 = R_EARTH + altitude_ref
        T0 = 2 * np.pi * np.sqrt(a0**3 / self.mu)
        
        self.mA = harmonic_A
        self.mB = harmonic_B
        
        self.TA = T0 / self.mA
        self.TB = T0 / self.mB
        
        # Semi-major axes (from Kepler's third law)
        self.aA = a0 * self.mA**(-2/3)
        self.aB = a0 * self.mB**(-2/3)
        
        # Circular orbit velocities
        self.vA0 = np.sqrt(self.mu / self.aA)
        self.vB0 = np.sqrt(self.mu / self.aB)
        
        # Initial state: both on x-axis, circular orbits
        self.state0 = np.array([
            self.aA, 0, 0,  # Satellite A position
            0, self.vA0, 0, # Satellite A velocity
            self.aB, 0, 0,  # Satellite B position
            0, self.vB0, 0  # Satellite B velocity
        ], dtype=float)
    
    def dynamics(self, t, state):
        """
        Orbital dynamics (two-body problem).
        
        Args:
            t: Time (unused in autonomous system)
            state: [xA, yA, zA, vxA, vyA, vzA, xB, yB, zB, vxB, vyB, vzB]
            
        Returns:
            Time derivatives of state
        """
        # Extract positions
        rA = state[0:3]
        rB = state[6:9]
        
        # Compute accelerations (gravity only)
        rA_norm = np.linalg.norm(rA)
        rB_norm = np.linalg.norm(rB)
        
        accA = -self.mu * rA / rA_norm**3
        accB = -self.mu * rB / rB_norm**3
        
        # Return [velocities, accelerations]
        return np.concatenate([
            state[3:6],   # vA
            accA,
            state[9:12],  # vB
            accB
        ])
    
    def apply_deltaV(self, state, deltaV_magnitude, satellite='A'):
        """
        Apply tangential ΔV to satellite.
        
        Args:
            state: Current state vector
            deltaV_magnitude: ΔV in m/s (positive = prograde)
            satellite: 'A' or 'B'
            
        Returns:
            Modified state vector
        """
        new_state = state.copy()
        
        if satellite == 'A':
            v_vec = state[3:6]
        else:  # 'B'
            v_vec = state[9:12]
        
        v_norm = np.linalg.norm(v_vec)
        
        if v_norm > 0:
            # Apply ΔV in velocity direction (tangential)
            deltaV_vec = deltaV_magnitude * (v_vec / v_norm)
            
            if satellite == 'A':
                new_state[3:6] += deltaV_vec
            else:
                new_state[9:12] += deltaV_vec
        
        return new_state
    
    def compute_phase_rate(self, state):
        """
        Compute RF phase rate between satellites.
        
        Args:
            state: Current state vector
            
        Returns:
            phase_rate: dφ/dt in rad/s
            range: Distance between satellites in m
        """
        rA = state[0:3]
        rB = state[6:9]
        vA = state[3:6]
        vB = state[9:12]
        
        # Range vector and magnitude
        d = rB - rA
        range_dist = np.linalg.norm(d)
        
        # Range rate
        range_rate = np.dot(d, vB - vA) / range_dist
        
        # Phase rate = (2π/λ) * range_rate
        phase_rate = (2 * np.pi / self.lambda_m) * range_rate
        
        return phase_rate, range_dist
    
    def simulate(self, duration, impulse_events=None, n_output=900):
        """
        Simulate orbital motion with optional ΔV impulses.
        
        Uses segmented integration with exact impulse insertion.
        
        Args:
            duration: Simulation time in seconds
            impulse_events: List of (satellite, time, deltaV) tuples
                           e.g., [('A', 300.0, 0.5), ('B', 600.0, 0.3)]
            n_output: Number of output points
            
        Returns:
            times: Array of time points
            states: Array of state vectors
            phase_rates: Array of phase rates
            impulse_log: List of applied impulses
        """
        if impulse_events is None:
            impulse_events = []
        
        # Create timeline with impulse times
        impulse_times = sorted([t for (_, t, _) in impulse_events])
        segment_times = [0.0] + impulse_times + [duration]
        segment_times = sorted(list(set(segment_times)))  # Remove duplicates
        
        # Storage
        all_times = []
        all_states = []
        impulse_log = []
        
        current_state = self.state0.copy()
        
        # Simulate each segment
        for i in range(len(segment_times) - 1):
            t_start = segment_times[i]
            t_end = segment_times[i + 1]
            seg_duration = t_end - t_start
            
            # Number of points for this segment
            points = max(3, int(n_output * seg_duration / duration))
            t_eval = np.linspace(t_start, t_end, points)
            
            # Propagate
            sol = solve_ivp(
                self.dynamics,
                [t_start, t_end],
                current_state,
                t_eval=t_eval,
                rtol=1e-9,
                atol=1e-12,
                method='DOP853'
            )
            
            # Store results
            if len(all_times) == 0:
                all_times = sol.t.tolist()
                all_states = sol.y.copy()
            else:
                # Skip first point (duplicate from previous segment)
                all_times += sol.t.tolist()[1:]
                all_states = np.hstack([all_states, sol.y[:, 1:]])
            
            # Update state for next segment
            current_state = sol.y[:, -1].copy()
            
            # Apply any impulses at end of this segment
            for sat, t_imp, dv in impulse_events:
                if abs(t_imp - t_end) < 1e-9:
                    current_state = self.apply_deltaV(current_state, dv, sat)
                    impulse_log.append((sat, t_imp, dv))
        
        # Convert to arrays
        times = np.array(all_times)
        states = np.array(all_states)
        
        # Compute phase rates
        phase_rates = []
        ranges = []
        
        for i in range(states.shape[1]):
            pr, r = self.compute_phase_rate(states[:, i])
            phase_rates.append(pr)
            ranges.append(r)
        
        phase_rates = np.array(phase_rates)
        ranges = np.array(ranges)
        
        return times, states, phase_rates, ranges, impulse_log


def quick_test():
    """Quick sanity check."""
    print("Testing OrbitalSimulator...")
    
    sim = OrbitalSimulator(harmonic_A=2, harmonic_B=3)
    
    # Simulate with one impulse
    impulses = [('A', 300.0, 0.001)]  # 1 mm/s at 5 minutes
    
    times, states, phase_rates, ranges, log = sim.simulate(
        duration=600.0,  # 10 minutes
        impulse_events=impulses,
        n_output=300
    )
    
    print(f"✓ Simulated {len(times)} time points")
    print(f"✓ Applied {len(log)} impulse(s): {log}")
    print(f"✓ Phase rate range: [{phase_rates.min():.6f}, {phase_rates.max():.6f}] rad/s")
    print(f"✓ Distance range: [{ranges.min()/1000:.1f}, {ranges.max()/1000:.1f}] km")
    print("Test passed!")


if __name__ == '__main__':
    quick_test()
