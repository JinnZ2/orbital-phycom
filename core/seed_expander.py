"""
Geometric Seed Expansion for Orbital Networks

Maps compact seeds to orbital schedules using physics as decompressor.
Based on fractal intelligence framework.
"""

import numpy as np
from scipy.integrate import solve_ivp
from .physics_constants import MU_EARTH, R_EARTH, ALTITUDE_REF, LAMBDA_RF


class ThreeSatelliteExpander:
    """
    Three-satellite constellation with prime harmonic periods.
    
    Seeds expand into orbital evolution via physics.
    Compression ratio: ~7000× for daily schedules.
    """
    
    def __init__(self, harmonics=(2, 3, 5), altitude_ref=ALTITUDE_REF):
        """
        Initialize three-satellite system.
        
        Args:
            harmonics: Tuple of prime harmonic numbers (e.g., (2, 3, 5))
            altitude_ref: Reference altitude in meters
        """
        self.mu = MU_EARTH
        self.R_earth = R_EARTH
        self.lambda_m = LAMBDA_RF
        
        # Prime harmonics for constellation
        self.harmonics = harmonics
        
        # Calculate orbital parameters
        a0 = R_EARTH + altitude_ref
        T0 = 2 * np.pi * np.sqrt(a0**3 / self.mu)
        
        self.periods = [T0 / h for h in self.harmonics]
        self.semi_major = [a0 * h**(-2/3) for h in self.harmonics]
        self.velocities = [np.sqrt(self.mu / a) for a in self.semi_major]
        
        # 6 orbital directions in VNB frame
        # (Velocity, Normal, Binormal)
        self.directions = np.array([
            [1, 0, 0],   # prograde
            [-1, 0, 0],  # retrograde
            [0, 1, 0],   # outward (radial)
            [0, -1, 0],  # inward
            [0, 0, 1],   # north (orbit normal)
            [0, 0, -1]   # south
        ])
        
        # Initial states: all on x-axis, circular orbits
        self.initial_states = []
        for i in range(3):
            state = np.array([
                self.semi_major[i], 0, 0,  # position
                0, self.velocities[i], 0   # velocity (prograde)
            ])
            self.initial_states.append(state)
    
    def seed_to_deltaVs(self, seed, deltaV_scale=0.002):
        """
        Convert 15-value seed to ΔV components for all satellites.
        
        Seed structure: 5 values per satellite × 3 satellites = 15 values
        Each 5-value block: [prograde, retrograde, outward, inward, north]
        6th value (south) implicit from energy conservation (sum = 1.0)
        
        Args:
            seed: List/array of 15 values (0.0-1.0 each)
            deltaV_scale: Maximum ΔV magnitude in m/s
            
        Returns:
            deltaV_matrices: List of 3 matrices (one per satellite)
            deltaV_mags: List of 3 magnitude arrays
        """
        seed = np.array(seed)
        if len(seed) != 15:
            raise ValueError(f"Seed must have 15 values, got {len(seed)}")
        
        # Reshape: 3 satellites × 5 values each
        seed_matrix = seed.reshape(3, 5)
        
        deltaV_matrices = []
        deltaV_mags_list = []
        
        for sat_idx in range(3):
            sat_seed = seed_matrix[sat_idx]
            
            # Add 6th implicit component (energy conservation)
            s = np.array(list(sat_seed) + [1.0 - sum(sat_seed)])
            s = s / s.sum()  # Ensure normalization
            
            # Scale to ΔV magnitudes
            deltaV_mags = s * deltaV_scale
            
            # Convert to vectors in VNB frame
            deltaV_vectors = deltaV_mags[:, np.newaxis] * self.directions
            
            deltaV_matrices.append(deltaV_vectors)
            deltaV_mags_list.append(deltaV_mags)
        
        return deltaV_matrices, deltaV_mags_list
    
    def apply_deltaV_to_state(self, state, deltaV_vector):
        """
        Apply ΔV in VNB frame to satellite state.
        
        Args:
            state: [x, y, z, vx, vy, vz]
            deltaV_vector: [dV_velocity, dV_normal, dV_binormal] in VNB frame
            
        Returns:
            new_state: Modified state vector
        """
        x, y, z, vx, vy, vz = state
        v_vec = np.array([vx, vy, vz])
        r_vec = np.array([x, y, z])
        
        # Compute VNB basis vectors
        # V: velocity direction
        V_hat = v_vec / np.linalg.norm(v_vec)
        
        # N: radial direction (outward from Earth)
        r_hat = r_vec / np.linalg.norm(r_vec)
        
        # B: orbit normal (cross product)
        N_hat = np.cross(r_hat, V_hat)
        if np.linalg.norm(N_hat) > 0:
            N_hat = N_hat / np.linalg.norm(N_hat)
        else:
            N_hat = np.array([0, 0, 1])  # Default to z-axis
        
        # Transform ΔV from VNB to inertial frame
        deltaV_inertial = (deltaV_vector[0] * V_hat + 
                          deltaV_vector[1] * r_hat +  # radial = normal in orbital sense
                          deltaV_vector[2] * N_hat)
        
        # Apply to velocity
        new_state = state.copy()
        new_state[3:6] += deltaV_inertial
        
        return new_state
    
    def compute_phase_rates(self, states):
        """
        Compute phase rates between all satellite pairs.
        
        Args:
            states: List of 3 state vectors
            
        Returns:
            phase_rates: Array of 3 phase rates (AB, BC, AC)
            ranges: Array of 3 ranges
        """
        # Three pairs: (0,1), (1,2), (0,2)
        pairs = [(0, 1), (1, 2), (0, 2)]
        phase_rates = []
        ranges = []
        
        for i, j in pairs:
            rA = states[i][:3]
            rB = states[j][:3]
            vA = states[i][3:6]
            vB = states[j][3:6]
            
            # Range vector and magnitude
            d = rB - rA
            range_dist = np.linalg.norm(d)
            
            # Range rate
            range_rate = np.dot(d, vB - vA) / range_dist if range_dist > 0 else 0
            
            # Phase rate = (2π/λ) * range_rate
            phase_rate = (2 * np.pi / self.lambda_m) * range_rate
            
            phase_rates.append(phase_rate)
            ranges.append(range_dist)
        
        return np.array(phase_rates), np.array(ranges)
    
    def expand_seed(self, seed, steps=5, deltaV_scale=0.002, symbol_period=300.0):
        """
        Expand seed into orbital evolution for 3 satellites.
        
        Physics acts as decompressor: 15 values → complete orbital schedule.
        
        Args:
            seed: 15-value seed (5 per satellite)
            steps: Number of maneuver steps
            deltaV_scale: Maximum ΔV magnitude in m/s
            symbol_period: Time between maneuvers in seconds
            
        Returns:
            times: Array of time points
            phase_rates: Array of phase rates (N×3 for 3 pairs)
            states_history: List of state snapshots
        """
        # Convert seed to ΔV components
        deltaV_matrices, _ = self.seed_to_deltaVs(seed, deltaV_scale)
        
        # Initialize states
        states = [s.copy() for s in self.initial_states]
        
        all_phase_rates = []
        all_times = []
        states_history = []
        
        for step in range(steps):
            t_impulse = step * symbol_period
            
            # Apply composite ΔV to each satellite
            new_states = []
            for sat_idx in range(3):
                # Sum all ΔV components for this satellite
                composite_deltaV = np.sum(deltaV_matrices[sat_idx], axis=0)
                new_state = self.apply_deltaV_to_state(states[sat_idx], composite_deltaV)
                new_states.append(new_state)
            
            states = new_states
            
            # Propagate all satellites for one symbol period
            duration = symbol_period
            
            # Combine all states into single vector for integration
            y0 = np.concatenate(states)
            
            def dynamics(t, y):
                """Three-body orbital dynamics."""
                accelerations = []
                for i in range(3):
                    r = y[i*6 : i*6+3]
                    r_norm = np.linalg.norm(r)
                    if r_norm > 0:
                        acc = -self.mu * r / r_norm**3
                    else:
                        acc = np.zeros(3)
                    accelerations.append(acc)
                
                # Return velocities + accelerations
                result = []
                for i in range(3):
                    result.extend(y[i*6+3 : i*6+6])  # velocities
                    result.extend(accelerations[i])   # accelerations
                
                return np.array(result)
            
            # Sample during this period (10 Hz)
            t_eval = np.linspace(0, duration, int(duration/10) + 1)
            
            sol = solve_ivp(
                dynamics,
                [0, duration],
                y0,
                t_eval=t_eval,
                rtol=1e-9,
                atol=1e-12,
                method='DOP853',
                max_step=10.0
            )
            
            # Extract phase rates at each sample
            for t_idx in range(len(t_eval)):
                t = t_impulse + t_eval[t_idx]
                state_vector = sol.y[:, t_idx]
                
                # Extract individual states
                current_states = []
                for i in range(3):
                    current_states.append(state_vector[i*6 : (i+1)*6])
                
                # Compute phase rates
                phase_rates, _ = self.compute_phase_rates(current_states)
                
                all_phase_rates.append(phase_rates)
                all_times.append(t)
            
            # Update states for next impulse
            states = []
            for i in range(3):
                states.append(sol.y[i*6 : (i+1)*6, -1])
            
            # Store state snapshot
            states_history.append([s.copy() for s in states])
        
        return np.array(all_times), np.array(all_phase_rates), states_history


def quick_test():
    """Quick sanity check."""
    print("Testing ThreeSatelliteExpander...")
    
    expander = ThreeSatelliteExpander(harmonics=(2, 3, 5))
    
    # Test seed: different emphasis for each satellite
    test_seed = [
        # Satellite A (harmonic 2): mostly prograde
        0.6, 0.0, 0.2, 0.0, 0.2,
        # Satellite B (harmonic 3): balanced
        0.2, 0.2, 0.2, 0.2, 0.2,
        # Satellite C (harmonic 5): radial emphasis
        0.1, 0.1, 0.5, 0.1, 0.2
    ]
    
    print(f"Test seed (15 values):")
    print(f"  Sat A: {test_seed[0:5]}")
    print(f"  Sat B: {test_seed[5:10]}")
    print(f"  Sat C: {test_seed[10:15]}")
    
    # Expand seed
    times, phase_rates, _ = expander.expand_seed(
        test_seed,
        steps=3,
        deltaV_scale=0.001,
        symbol_period=300.0
    )
    
    print(f"✓ Expanded to {len(times)} time samples")
    print(f"✓ Phase rates shape: {phase_rates.shape} (3 pairs: AB, BC, AC)")
    print(f"✓ Time range: [0, {times[-1]:.0f}] seconds")
    print(f"✓ Phase rate ranges:")
    for i, pair in enumerate(['AB', 'BC', 'AC']):
        pr = phase_rates[:, i]
        print(f"    {pair}: [{pr.min():.6f}, {pr.max():.6f}] rad/s")
    
    print("Test passed!")


if __name__ == '__main__':
    quick_test()
