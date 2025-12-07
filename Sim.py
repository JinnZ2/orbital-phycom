#!/usr/bin/env python3
"""
Three-Satellite Prime Harmonic Network Demo

Demonstrates physics-based seed expansion with three satellites.
Shows how 15 seed values expand into complete orbital schedules.

Run time: ~2 minutes
Output: Multi-panel visualization of network behavior
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.seed_expander import ThreeSatelliteExpander


def main():
    print("="*70)
    print("Orbital PHYCOM: Three-Satellite Prime Harmonic Network")
    print("="*70)
    print()
    
    # Initialize expander
    print("Initializing three-satellite constellation...")
    print("  Prime harmonics: 2, 3, 5")
    print("  This creates three independent phase-rate channels")
    print()
    
    expander = ThreeSatelliteExpander(harmonics=(2, 3, 5))
    
    # Display orbital parameters
    print("Orbital Parameters:")
    for i, h in enumerate(expander.harmonics):
        alt = (expander.semi_major[i] - expander.R_earth) / 1000
        period = expander.periods[i] / 60
        print(f"  Satellite {chr(65+i)} (harmonic {h}): "
              f"altitude = {alt:.1f} km, period = {period:.1f} min")
    print()
    
    # Design a test seed that demonstrates different strategies
    print("Seed Design:")
    print("  Satellite A: Heavy prograde bias (fast orbit changes)")
    print("  Satellite B: Balanced (stable reference)")
    print("  Satellite C: Radial emphasis (eccentricity modulation)")
    print()
    
    test_seed = [
        # Satellite A: prograde emphasis
        0.6, 0.0, 0.2, 0.0, 0.2,  # [prog, retro, out, in, north]
        # Satellite B: balanced
        0.2, 0.2, 0.2, 0.2, 0.2,
        # Satellite C: radial emphasis
        0.1, 0.1, 0.5, 0.1, 0.2
    ]
    
    print("Seed values (15 total):")
    for i in range(3):
        sat_name = chr(65 + i)
        seed_slice = test_seed[i*5:(i+1)*5]
        print(f"  {sat_name}: {seed_slice}")
    print()
    
    # Expand seed
    print("Expanding seed via orbital physics...")
    print("  Steps: 5 maneuvers")
    print("  Symbol period: 300 seconds (5 minutes)")
    print("  ΔV scale: 1.0 mm/s maximum")
    print()
    
    times, phase_rates, states_history = expander.expand_seed(
        test_seed,
        steps=5,
        deltaV_scale=0.001,  # 1 mm/s
        symbol_period=300.0   # 5 minutes
    )
    
    print(f"✓ Generated {len(times)} time samples")
    print(f"✓ Time span: {times[-1]/60:.1f} minutes")
    print(f"✓ Three phase-rate channels (AB, BC, AC)")
    print()
    
    # Also run baseline (no seed, no ΔV) for comparison
    print("Generating baseline (no maneuvers)...")
    baseline_seed = [0.0] * 15  # No ΔV
    times_base, phase_rates_base, _ = expander.expand_seed(
        baseline_seed,
        steps=5,
        deltaV_scale=0.0,
        symbol_period=300.0
    )
    
    print()
    
    # Create comprehensive visualization
    print("Creating visualization...")
    
    fig = plt.figure(figsize=(16, 12))
    
    pair_names = ['AB (2-3)', 'BC (3-5)', 'AC (2-5)']
    pair_colors = ['blue', 'green', 'red']
    
    # 1-3. Phase rates for each pair
    for i in range(3):
        ax = plt.subplot(4, 3, i+1)
        ax.plot(times_base/60, phase_rates_base[:, i], '--', 
                color='gray', alpha=0.5, linewidth=1, label='Baseline')
        ax.plot(times/60, phase_rates[:, i], '-', 
                color=pair_colors[i], linewidth=1.5, label='With seed')
        
        # Mark maneuver times
        for step in range(5):
            t_maneuver = step * 300.0
            ax.axvline(t_maneuver/60, color='k', linestyle=':', alpha=0.3)
        
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Phase Rate (rad/s)')
        ax.set_title(f'Pair {pair_names[i]}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 4-6. Phase rate deviations (signal)
    for i in range(3):
        ax = plt.subplot(4, 3, i+4)
        
        # Interpolate baseline to match times
        from scipy.interpolate import interp1d
        if len(times) == len(times_base) and np.allclose(times, times_base):
            deviation = phase_rates[:, i] - phase_rates_base[:, i]
        else:
            f = interp1d(times_base, phase_rates_base[:, i], 
                        bounds_error=False, fill_value=0)
            deviation = phase_rates[:, i] - f(times)
        
        ax.plot(times/60, deviation, color=pair_colors[i], linewidth=1.5)
        ax.axhline(0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        
        # Mark maneuvers
        for step in range(5):
            t_maneuver = step * 300.0
            ax.axvline(t_maneuver/60, color='k', linestyle=':', alpha=0.3)
        
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Δ Phase Rate (rad/s)')
        ax.set_title(f'Deviation: {pair_names[i]}')
        ax.grid(True, alpha=0.3)
    
    # 7. Combined spectrum (all three channels)
    ax = plt.subplot(4, 3, 7)
    for i in range(3):
        # FFT of phase rate deviation
        from scipy.interpolate import interp1d
        if len(times) == len(times_base) and np.allclose(times, times_base):
            deviation = phase_rates[:, i] - phase_rates_base[:, i]
        else:
            f = interp1d(times_base, phase_rates_base[:, i], 
                        bounds_error=False, fill_value=0)
            deviation = phase_rates[:, i] - f(times)
        
        # Compute FFT
        dt = times[1] - times[0]
        freq = np.fft.fftfreq(len(deviation), dt)
        fft_vals = np.abs(np.fft.fft(deviation))
        
        # Plot positive frequencies only
        pos_mask = freq > 0
        ax.semilogy(freq[pos_mask]*1000, fft_vals[pos_mask], 
                   color=pair_colors[i], alpha=0.7, label=pair_names[i])
    
    ax.set_xlabel('Frequency (mHz)')
    ax.set_ylabel('FFT Magnitude')
    ax.set_title('Frequency Spectrum of Deviations')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0, 50)  # Focus on 0-50 mHz
    
    # 8. Phase space (AB vs BC)
    ax = plt.subplot(4, 3, 8)
    ax.plot(phase_rates[:, 0], phase_rates[:, 1], 
           color='purple', alpha=0.6, linewidth=0.8)
    ax.scatter(phase_rates[0, 0], phase_rates[0, 1], 
              color='green', s=100, marker='o', zorder=5, label='Start')
    ax.scatter(phase_rates[-1, 0], phase_rates[-1, 1], 
              color='red', s=100, marker='s', zorder=5, label='End')
    ax.set_xlabel('Phase Rate AB (rad/s)')
    ax.set_ylabel('Phase Rate BC (rad/s)')
    ax.set_title('Phase Space Trajectory')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 9. Information content (entropy over time)
    ax = plt.subplot(4, 3, 9)
    # Sliding window entropy
    window_size = 50
    entropies = []
    window_times = []
    
    for start_idx in range(0, len(times) - window_size, 10):
        end_idx = start_idx + window_size
        window = phase_rates[start_idx:end_idx, :]
        
        # Flatten and histogram
        flat = window.flatten()
        hist, _ = np.histogram(flat, bins=20, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        
        # Shannon entropy
        entropy = -np.sum(hist * np.log2(hist))
        entropies.append(entropy)
        window_times.append(times[start_idx + window_size//2])
    
    ax.plot(np.array(window_times)/60, entropies, 
           color='orange', linewidth=2)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Shannon Entropy (bits)')
    ax.set_title('Information Content Over Time')
    ax.grid(True, alpha=0.3)
    
    # 10-12. Orbital positions (XY plane) at three snapshots
    snapshot_indices = [0, len(states_history)//2, -1]
    snapshot_times = [0, len(states_history)//2 * 300, (len(states_history)-1) * 300]
    
    for plot_idx, (snap_idx, snap_time) in enumerate(zip(snapshot_indices, snapshot_times)):
        ax = plt.subplot(4, 3, 10 + plot_idx)
        
        if snap_idx < len(states_history):
            states = states_history[snap_idx]
            
            # Plot orbits and positions
            for sat_idx, color in enumerate(['blue', 'green', 'red']):
                state = states[sat_idx]
                x, y = state[0], state[1]
                
                # Plot position
                ax.plot(x/1000, y/1000, 'o', color=color, markersize=10,
                       label=f'Sat {chr(65+sat_idx)}')
                
                # Plot velocity vector (scaled for visibility)
                vx, vy = state[3], state[4]
                scale = 500  # km
                ax.arrow(x/1000, y/1000, vx*scale/1000, vy*scale/1000,
                        head_width=200, head_length=300, fc=color, ec=color, alpha=0.5)
            
            # Earth
            earth_circle = plt.Circle((0, 0), expander.R_earth/1000, 
                                     color='cyan', alpha=0.3, label='Earth')
            ax.add_patch(earth_circle)
            
            ax.set_xlabel('X (km)')
            ax.set_ylabel('Y (km)')
            ax.set_title(f't = {snap_time/60:.1f} min')
            ax.legend(fontsize=7, loc='upper right')
            ax.axis('equal')
            ax.grid(True, alpha=0.3)
            
            # Set limits
            max_radius = max(expander.semi_major) * 1.2 / 1000
            ax.set_xlim(-max_radius, max_radius)
            ax.set_ylim(-max_radius, max_radius)
    
    plt.suptitle('Three-Satellite Network: Physics-Based Seed Expansion', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_file = 'three_satellite_demo.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.show()
    
    # Summary statistics
    print()
    print("="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Seed: 15 values (5 per satellite)")
    print(f"Expansion: {len(times)} phase rate samples over {times[-1]/60:.1f} minutes")
    print(f"  Compression ratio: {len(times)*3*8}/{15*8:.0f} = {len(times)*3/15:.0f}× raw data")
    print()
    print("Phase Rate Ranges (with seed):")
    for i, pair in enumerate(pair_names):
        pr = phase_rates[:, i]
        print(f"  {pair}: [{pr.min():.6f}, {pr.max():.6f}] rad/s")
    print()
    print("Maximum Deviations from Baseline:")
    for i, pair in enumerate(pair_names):
        if len(times) == len(times_base) and np.allclose(times, times_base):
            deviation = phase_rates[:, i] - phase_rates_base[:, i]
        else:
            from scipy.interpolate import interp1d
            f = interp1d(times_base, phase_rates_base[:, i], 
                        bounds_error=False, fill_value=0)
            deviation = phase_rates[:, i] - f(times)
        
        max_dev = np.max(np.abs(deviation))
        print(f"  {pair}: {max_dev:.8f} rad/s = {max_dev*1e6:.2f} µrad/s")
    print()
    print("Key Insights:")
    print("  • Three independent channels provide redundancy")
    print("  • Prime harmonics (2:3:5) prevent aliasing")
    print("  • ΔV as small as 1 mm/s creates detectable signatures")
    print("  • Physics deterministically expands seed → schedule")
    print("  • No synchronization needed (shared physics is clock)")
    print()
    print("Next Steps:")
    print("  • Add noise: 03_sensitivity_test.py")
    print("  • Seed recovery: Test inverse mapping")
    print("  • Protocol: Full PHYCOM implementation")
    print("="*70)


if __name__ == '__main__':
    main()
