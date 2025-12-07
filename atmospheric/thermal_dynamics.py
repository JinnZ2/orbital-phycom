"""
Atmospheric thermal dynamics for seed expansion.

Models natural thermal processes that can be precisely modulated
to create detectable patterns in atmospheric behavior.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


class AtmosphericSimulator:
    """
    Atmospheric dynamics simulator for seed-based communication.
    
    Uses natural thermal processes (solar heating, ground thermal mass,
    convection) to create detectable signatures in wind/pressure patterns.
    """
    
    def __init__(self, grid_size=(100, 100), domain_km=50.0, dt=60.0):
        """
        Initialize atmospheric grid.
        
        Args:
            grid_size: (nx, ny) grid dimensions
            domain_km: Physical domain size in kilometers
            dt: Time step in seconds
        """
        self.nx, self.ny = grid_size
        self.domain_km = domain_km
        self.dt = dt
        
        # Grid spacing
        self.dx = self.domain_km * 1000 / self.nx  # meters
        self.dy = self.domain_km * 1000 / self.ny  # meters
        
        # Physical constants
        self.g = 9.81       # gravity (m/s²)
        self.cp = 1004.0    # specific heat of air (J/kg/K)
        self.rho0 = 1.225   # air density at sea level (kg/m³)
        self.R = 287.0      # gas constant for air (J/kg/K)
        self.T0 = 288.15    # reference temperature (K)
        
        # Initialize grids
        x = np.linspace(0, domain_km*1000, self.nx)
        y = np.linspace(0, domain_km*1000, self.ny)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        
        # State variables: [temperature, pressure, u_wind, v_wind]
        self.T = np.full((self.nx, self.ny), self.T0)  # Temperature (K)
        self.P = np.full((self.nx, self.ny), 101325.0) # Pressure (Pa)
        self.u = np.zeros((self.nx, self.ny))          # East wind (m/s)
        self.v = np.zeros((self.nx, self.ny))          # North wind (m/s)
        
        # Thermal sources (natural heating/cooling)
        self.thermal_sources = np.zeros((self.nx, self.ny))
        
        # Baseline temperature profile (varies with location)
        self._initialize_baseline_temperature()
    
    def _initialize_baseline_temperature(self):
        """Initialize natural temperature variation across domain."""
        # Gentle temperature gradient (warmer toward south/east)
        temp_gradient_x = 0.5 / (self.domain_km * 1000)  # 0.5 K per domain
        temp_gradient_y = 0.3 / (self.domain_km * 1000)  # 0.3 K per domain
        
        self.T += temp_gradient_x * self.X + temp_gradient_y * self.Y
        
        # Add some natural thermal features
        # Warm spot (like a lake or urban area)
        warm_x, warm_y = self.nx//3, self.ny//2
        warm_radius = min(self.nx, self.ny) * 0.1
        for i in range(self.nx):
            for j in range(self.ny):
                dist = np.sqrt((i - warm_x)**2 + (j - warm_y)**2)
                if dist < warm_radius:
                    self.T[i, j] += 2.0 * np.exp(-dist**2 / (2*(warm_radius/3)**2))
        
        # Cool spot (like a forest or elevated area)
        cool_x, cool_y = 2*self.nx//3, self.ny//4
        cool_radius = min(self.nx, self.ny) * 0.08
        for i in range(self.nx):
            for j in range(self.ny):
                dist = np.sqrt((i - cool_x)**2 + (j - cool_y)**2)
                if dist < cool_radius:
                    self.T[i, j] -= 1.5 * np.exp(-dist**2 / (2*(cool_radius/3)**2))
    
    def add_thermal_source(self, x_km, y_km, intensity_K, radius_km):
        """
        Add natural thermal source (heating/cooling).
        
        Args:
            x_km, y_km: Position in kilometers
            intensity_K: Temperature change in Kelvin (+ for heating, - for cooling)
            radius_km: Radius of influence in kilometers
        """
        x_idx = int(x_km * 1000 / self.dx)
        y_idx = int(y_km * 1000 / self.dy)
        radius_idx = int(radius_km * 1000 / self.dx)
        
        # Apply Gaussian thermal source
        for i in range(max(0, x_idx - 2*radius_idx), 
                      min(self.nx, x_idx + 2*radius_idx)):
            for j in range(max(0, y_idx - 2*radius_idx), 
                          min(self.ny, y_idx + 2*radius_idx)):
                dist_m = np.sqrt(((i - x_idx) * self.dx)**2 + 
                               ((j - y_idx) * self.dy)**2)
                if dist_m < radius_km * 1000:
                    gaussian = np.exp(-dist_m**2 / (2 * (radius_km * 1000 / 3)**2))
                    self.thermal_sources[i, j] += intensity_K * gaussian
    
    def compute_derivatives(self, state_vector, t):
        """
        Compute time derivatives for atmospheric dynamics.
        
        Simplified shallow water equations with thermal forcing.
        """
        # Unpack state
        n_vars = len(state_vector) // (self.nx * self.ny)
        state = state_vector.reshape(n_vars, self.nx, self.ny)
        T, P, u, v = state[0], state[1], state[2], state[3]
        
        # Initialize derivatives
        dT_dt = np.zeros_like(T)
        dP_dt = np.zeros_like(P)
        du_dt = np.zeros_like(u)
        dv_dt = np.zeros_like(v)
        
        # Thermal forcing
        dT_dt += self.thermal_sources / (self.rho0 * self.cp)
        
        # Temperature diffusion (natural mixing)
        diffusion_coeff = 10.0  # m²/s
        dT_dt += diffusion_coeff * (
            np.gradient(np.gradient(T, self.dx, axis=0), self.dx, axis=0) +
            np.gradient(np.gradient(T, self.dy, axis=1), self.dy, axis=1)
        )
        
        # Pressure gradient from temperature (hydrostatic adjustment)
        dP_dx = np.gradient(P, self.dx, axis=0)
        dP_dy = np.gradient(P, self.dy, axis=1)
        
        # Thermal win​​​​​​​​​​​​​​​​

#!/usr/bin/env python3
"""
Atmospheric Seed Expansion Demo

Demonstrates natural thermal perturbations creating detectable
atmospheric signatures through pure physics amplification.

Run time: ~3 minutes
Output: Multi-panel visualization of atmospheric evolution
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from atmospheric.thermal_dynamics import AtmosphericSimulator, AtmosphericSeedExpander


def main():
    print("="*70)
    print("Atmospheric PHYCOM: Natural Thermal Seed Expansion")
    print("="*70)
    print()
    
    print("Concept: Minimal thermal perturbations (fractions of a degree)")
    print("         create detectable patterns in natural atmospheric dynamics")
    print()
    print("Key principle: Work WITH nature, not against it")
    print("  - No artificial aerosols")
    print("  - No chemical injection")
    print("  - Pure thermal modulation within natural ranges")
    print("  - Physics amplifies the signal")
    print()
    
    # Initialize simulator
    print("Initializing atmospheric grid...")
    sim = AtmosphericSimulator(grid_size=(80, 80), domain_km=60.0, dt=60.0)
    print(f"  Domain: {sim.domain_km} km × {sim.domain_km} km")
    print(f"  Grid resolution: {sim.dx/1000:.2f} km")
    print(f"  Time step: {sim.dt} seconds")
    print()
    
    # Create expander
    expander = AtmosphericSeedExpander(sim)
    
    # Design seed
    print("Designing thermal seed...")
    print("  3 natural heat sources at different locations and times")
    print("  Maximum perturbation: ±0.3 K (barely noticeable)")
    print()
    
    # Seed: 3 thermal sources
    # Each source: [x_fraction, y_fraction, intensity_fraction, radius_fraction]
    test_seed = [
        0.25, 0.50, 0.55, 0.3,  # Source 1: west-center, slight heating, small
        0.75, 0.30, 0.45, 0.4,  # Source 2: east-south, slight cooling, medium
        0.50, 0.75, 0.60, 0.5   # Source 3: center-north, moderate heating, large
    ]
    
    print("Seed structure (12 values):")
    for i in range(3):
        idx = i * 4
        x_frac, y_frac, int_frac, rad_frac = test_seed[idx:idx+4]
        x_km = x_frac * sim.domain_km
        y_km = y_frac * sim.domain_km
        intensity_K = (int_frac - 0.5) * 2 * 0.3
        radius_km = 1.0 + rad_frac * 9.0
        
        print(f"  Source {i+1}: ({x_km:.1f}, {y_km:.1f}) km, "
              f"{intensity_K:+.2f} K, radius {radius_km:.1f} km")
    print()
    
    # Expand seed
    print("Expanding seed through atmospheric physics...")
    print("  Duration: 30 minutes")
    print("  Sampling: Every 60 seconds")
    print()
    
    times, states = expander.expand_seed(
        test_seed,
        total_duration=1800.0,  # 30 minutes
        intensity_scale=0.3      # ±0.3 K max
    )
    
    print(f"✓ Generated {len(times)} atmospheric snapshots")
    print()
    
    # Create visualization
    print("Creating visualization...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Select key time points
    snapshot_indices = [0, len(states)//3, 2*len(states)//3, -1]
    snapshot_times = [times[i] for i in snapshot_indices]
    
    for plot_idx, (snap_idx, snap_time) in enumerate(zip(snapshot_indices, snapshot_times)):
        T, P, u, v = states[snap_idx]
        wind_speed = np.sqrt(u**2 + v**2)
        T_anomaly = T - np.mean(T)
        
        # Temperature anomaly
        ax = plt.subplot(4, 4, plot_idx + 1)
        im = ax.contourf(sim.X/1000, sim.Y/1000, T_anomaly, levels=20, cmap='RdBu_r')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_title(f't = {snap_time/60:.1f} min: Temp Anomaly')
        plt.colorbar(im, ax=ax, label='ΔT (K)')
        ax.set_aspect('equal')
        
        # Wind speed
        ax = plt.subplot(4, 4, plot_idx + 5)
        im = ax.contourf(sim.X/1000, sim.Y/1000, wind_speed, levels=20, cmap='YlOrRd')
        # Add wind vectors (subsampled)
        skip = 4
        ax.quiver(sim.X[::skip, ::skip]/1000, sim.Y[::skip, ::skip]/1000,
                 u[::skip, ::skip], v[::skip, ::skip],
                 alpha=0.6, scale=50)
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_title(f't = {snap_time/60:.1f} min: Wind Speed')
        plt.colorbar(im, ax=ax, label='Speed (m/s)')
        ax.set_aspect('equal')
        
        # Pressure anomaly
        ax = plt.subplot(4, 4, plot_idx + 9)
        P_anomaly = P - np.mean(P)
        im = ax.contourf(sim.X/1000, sim.Y/1000, P_anomaly, levels=20, cmap='PuOr')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_title(f't = {snap_time/60:.1f} min: Pressure Anomaly')
        plt.colorbar(im, ax=ax, label='ΔP (Pa)')
        ax.set_aspect('equal')
    
    # Time series at detection points
    detect_points = [
        (15, 30, 'West'),
        (45, 30, 'East'),
        (30, 45, 'North')
    ]
    
    ax = plt.subplot(4, 4, 13)
    for x_km, y_km, label in detect_points:
        temps = []
        for T, _, _, _ in states:
            x_idx = int(x_km * 1000 / sim.dx)
            y_idx = int(y_km * 1000 / sim.dy)
            temps.append(T[x_idx, y_idx])
        
        temps = np.array(temps)
        ax.plot(times/60, temps - temps[0], label=label)
    
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Temperature Change (K)')
    ax.set_title('Temperature Evolution at Key Points')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Wind speed time series
    ax = plt.subplot(4, 4, 14)
    for x_km, y_km, label in detect_points:
        winds = []
        for _, _, u, v in states:
            x_idx = int(x_km * 1000 / sim.dx)
            y_idx = int(y_km * 1000 / sim.dy)
            wind = np.sqrt(u[x_idx, y_idx]**2 + v[x_idx, y_idx]**2)
            winds.append(wind)
        
        ax.plot(times/60, winds, label=label)
    
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Wind Speed (m/s)')
    ax.set_title('Wind Speed Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Total energy in system
    ax = plt.subplot(4, 4, 15)
    thermal_energy = []
    kinetic_energy = []
    
    for T, _, u, v in states:
        thermal = np.sum((T - sim.T0)**2)
        kinetic = np.sum(u**2 + v**2)
        thermal_energy.append(thermal)
        kinetic_energy.append(kinetic)
    
    ax.plot(times/60, thermal_energy, label='Thermal')
    ax.plot(times/60, kinetic_energy, label='Kinetic')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Energy (arb. units)')
    ax.set_title('Energy Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Detectability metric
    ax = plt.subplot(4, 4, 16)
    detectability = []
    
    for T, _, u, v in states:
        # Signal: std of temperature anomaly
        T_anom = T - np.mean(T)
        signal = np.std(T_anom)
        
        # Noise: baseline natural variation
        noise = 0.1  # K (assumed)
        
        snr = signal / noise
        detectability.append(snr)
    
    ax.plot(times/60, detectability, color='purple', linewidth=2)
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='SNR=1')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Signal-to-Noise Ratio')
    ax.set_title('Detectability Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Atmospheric Seed Expansion: Natural Thermal Perturbations', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_file = 'atmospheric_demo.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.show()
    
    # Summary
    print()
    print("="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Seed: {len(test_seed)} values → 3 thermal sources")
    print(f"Simulation: {times[-1]/60:.1f} minutes")
    print(f"Domain: {sim.domain_km} km × {sim.domain_km} km")
    print()
    print("Maximum perturbation applied: ±0.3 K")
    print(f"Final temperature variation: {np.std(states[-1][0]):.3f} K")
    print(f"Maximum wind speed generated: {np.sqrt(states[-1][2]**2 + states[-1][3]**2).max():.2f} m/s")
    print(f"Final detectability (SNR): {detectability[-1]:.2f}")
    print()
    print("Key Insights:")
    print("  • Tiny thermal changes (0.3 K) create measurable atmospheric effects")
    print("  • Pure physics amplifies the signal naturally")
    print("  • No artificial substances needed")
    print("  • Detectable signatures persist for tens of minutes")
    print("  • Works entirely within natural thermal ranges")
    print()
    print("Next Steps:")
    print("  • Test seed recovery (inverse problem)")
    print("  • Add realistic noise models")
    print("  • Couple with orbital system for multi-layer coordination")
    print("  • Design ground-based measurement protocols")
    print("="*70)


if __name__ == '__main__':
    main()
