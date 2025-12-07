#!/usr/bin/env python3
"""
Basic Orbital Deviation Simulation

Demonstrates orbital communication via small ΔV impulses.
Shows phase rate modulation with and without impulses.

Run time: ~30 seconds
Output: Graph showing detectable phase rate changes from 1 mm/s ΔV
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.orbital_dynamics import OrbitalSimulator


def main():
    print("="*60)
    print("Orbital PHYCOM: Basic Simulation")
    print("="*60)
    print()
    
    # Create simulator (harmonics 2:3)
    print("Initializing two-satellite system...")
    print("  Satellite A: 2× harmonic (faster orbit)")
    print("  Satellite B: 3× harmonic (slower orbit)")
    print()
    
    sim = OrbitalSimulator(harmonic_A=2, harmonic_B=3)
    
    print(f"Orbital parameters:")
    print(f"  A: altitude = {(sim.aA - sim.R_earth)/1000:.1f} km, period = {sim.TA/60:.1f} min")
    print(f"  B: altitude = {(sim.aB - sim.R_earth)/1000:.1f} km, period = {sim.TB/60:.1f} min")
    print()
    
    # Simulation parameters
    duration = 2 * sim.TA  # Two orbits of satellite A
    
    # Define impulse events
    impulses = [
        ('A', 300.0, 0.001),  # 1 mm/s at 5 min
        ('B', 600.0, 0.001),  # 1 mm/s at 10 min
        ('A', 900.0, 0.001),  # 1 mm/s at 15 min
    ]
    
    print(f"Simulating {duration/60:.1f} minutes...")
    print(f"Three ΔV impulses: {[f'{dv*1000:.1f} mm/s @ {t/60:.1f} min' for _, t, dv in impulses]}")
    print()
    
    # Run simulation WITH impulses
    print("Running with impulses...")
    t_on, states_on, pr_on, ranges_on, log = sim.simulate(
        duration=duration,
        impulse_events=impulses,
        n_output=900
    )
    
    # Run simulation WITHOUT impulses (baseline)
    print("Running baseline (no impulses)...")
    t_off, states_off, pr_off, ranges_off, _ = sim.simulate(
        duration=duration,
        impulse_events=[],
        n_output=900
    )
    
    print()
    print("Impulses applied:", log)
    print()
    
    # Create plots
    print("Generating plots...")
    
    fig = plt.figure(figsize=(14, 10))
    
    # 1. Orbits (XY plane)
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(states_off[0]/1000, states_off[1]/1000, 'b-', alpha=0.5, label='A (baseline)')
    ax1.plot(states_off[6]/1000, states_off[7]/1000, 'r-', alpha=0.5, label='B (baseline)')
    ax1.plot(states_on[0]/1000, states_on[1]/1000, 'b--', alpha=0.7, label='A (with ΔV)')
    ax1.plot(states_on[6]/1000, states_on[7]/1000, 'r--', alpha=0.7, label='B (with ΔV)')
    ax1.plot(0, 0, 'ko', markersize=8, label='Earth')
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_title('Satellite Orbits (XY Plane)')
    ax1.legend(fontsize=8)
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # 2. Inter-satellite distance
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(t_off/60, ranges_off/1000, 'b-', alpha=0.6, label='Baseline')
    ax2.plot(t_on/60, ranges_on/1000, 'r-', alpha=0.8, label='With ΔV')
    for _, t, _ in impulses:
        ax2.axvline(t/60, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Distance (km)')
    ax2.set_title('Inter-Satellite Range')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Phase rate (full)
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(t_off/60, pr_off, 'b-', alpha=0.6, label='Baseline')
    ax3.plot(t_on/60, pr_on, 'r-', alpha=0.8, label='With ΔV')
    for _, t, dv in impulses:
        ax3.axvline(t/60, color='k', linestyle='--', alpha=0.4)
        ax3.text(t/60, ax3.get_ylim()[1]*0.9, f'{dv*1000:.1f}mm/s', 
                fontsize=8, ha='center', rotation=90, alpha=0.7)
    ax3.set_xlabel('Time (min)')
    ax3.set_ylabel('Phase Rate (rad/s)')
    ax3.set_title('RF Phase Rate (Full Timeline)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Phase rate (zoomed near impulses)
    ax4 = plt.subplot(3, 2, 4)
    zoom_start = 4  # minutes
    zoom_end = 16   # minutes
    zoom_mask_off = (t_off/60 >= zoom_start) & (t_off/60 <= zoom_end)
    zoom_mask_on = (t_on/60 >= zoom_start) & (t_on/60 <= zoom_end)
    
    ax4.plot(t_off[zoom_mask_off]/60, pr_off[zoom_mask_off], 'b-', alpha=0.6, label='Baseline')
    ax4.plot(t_on[zoom_mask_on]/60, pr_on[zoom_mask_on], 'r-', alpha=0.8, label='With ΔV')
    for _, t, dv in impulses:
        if zoom_start <= t/60 <= zoom_end:
            ax4.axvline(t/60, color='k', linestyle='--', alpha=0.4)
            ax4.text(t/60, ax4.get_ylim()[1]*0.85, f'{dv*1000:.1f}mm/s', 
                    fontsize=8, ha='center', rotation=90, alpha=0.7)
    ax4.set_xlabel('Time (min)')
    ax4.set_ylabel('Phase Rate (rad/s)')
    ax4.set_title('Phase Rate (Zoom: Impulses)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Phase rate difference (signal)
    ax5 = plt.subplot(3, 2, 5)
    # Interpolate to match time grids if needed
    if len(t_on) == len(t_off) and np.allclose(t_on, t_off):
        pr_diff = pr_on - pr_off
    else:
        from scipy.interpolate import interp1d
        f = interp1d(t_off, pr_off, bounds_error=False, fill_value=0)
        pr_diff = pr_on - f(t_on)
    
    ax5.plot(t_on/60, pr_diff, 'g-', linewidth=1.5)
    for _, t, _ in impulses:
        ax5.axvline(t/60, color='r', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Time (min)')
    ax5.set_ylabel('Δ Phase Rate (rad/s)')
    ax5.set_title('Phase Rate Deviation (Signal)')
    ax5.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax5.grid(True, alpha=0.3)
    
    # 6. Phase rate derivative (for detection)
    ax6 = plt.subplot(3, 2, 6)
    pr_accel = np.gradient(pr_on, t_on)
    ax6.plot(t_on/60, np.abs(pr_accel), 'purple', linewidth=1)
    for _, t, _ in impulses:
        ax6.axvline(t/60, color='r', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Time (min)')
    ax6.set_ylabel('|d²φ/dt²| (rad/s²)')
    ax6.set_title('Phase Acceleration (Detection Signal)')
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3, which='both')
    
    plt.suptitle('Orbital Deviation Communication: 1 mm/s ΔV Impulses', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_file = 'basic_orbital_sim.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.show()
    
    # Summary
    print()
    print("="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Total simulation time: {duration/60:.1f} minutes")
    print(f"Applied ΔV impulses: {len(impulses)}")
    print(f"  Total ΔV: {sum([dv for _, _, dv in impulses])*1000:.2f} mm/s")
    print()
    print(f"Phase rate range (baseline): [{pr_off.min():.6f}, {pr_off.max():.6f}] rad/s")
    print(f"Phase rate range (with ΔV): [{pr_on.min():.6f}, {pr_on.max():.6f}] rad/s")
    print()
    print(f"Maximum phase rate deviation: {np.max(np.abs(pr_diff)):.8f} rad/s")
    print(f"  That's {np.max(np.abs(pr_diff))*1e6:.2f} µrad/s")
    print()
    print("Key insight:")
    print("  A 1 mm/s ΔV (0.000013% of orbital velocity)")
    print("  creates a detectable phase rate signature")
    print("  that persists for the entire orbit.")
    print()
    print("Next: Run 02_three_satellite_demo.py for the full network")
    print("="*60)


if __name__ == '__main__':
    main()
