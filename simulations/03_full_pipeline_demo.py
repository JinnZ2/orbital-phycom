#!/usr/bin/env python3
"""
Full Communication Pipeline Demo: Encode → Noise → Detect → Recover

Demonstrates the complete orbital seed communication system:
    1. Encode: Map seed to orbital maneuvers (forward model with J2)
    2. Noise: Corrupt observations with realistic noise sources
    3. Detect: Assess signal detectability (matched filter analysis)
    4. Recover: Reconstruct seed from noisy observations (inverse problem)

Run time: ~3-5 minutes (depends on recovery search depth)
Output: Multi-panel visualization + printed diagnostics
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.seed_expander import ThreeSatelliteExpander
from core.noise_model import PhaseRateNoiseModel
from core.seed_recovery import SeedRecoverer
from core.detection import DetectionAnalyzer


def main():
    print("=" * 70)
    print("Orbital PHYCOM: Full Communication Pipeline")
    print("  Encode → Noise → Detect → Recover")
    print("=" * 70)
    print()

    rng = np.random.default_rng(42)

    # --- Configuration ---
    HARMONICS = (2, 3, 5)
    STEPS = 3
    DELTA_V_SCALE = 0.001  # 1 mm/s
    SYMBOL_PERIOD = 300.0  # 5 minutes

    true_seed = [
        0.6, 0.0, 0.2, 0.0, 0.2,   # Sat A: prograde-heavy
        0.2, 0.2, 0.2, 0.2, 0.2,   # Sat B: balanced
        0.1, 0.1, 0.5, 0.1, 0.2    # Sat C: radial-heavy
    ]

    # ===== STAGE 1: ENCODE (Forward Model) =====
    print("[1/4] ENCODING: Seed → Orbital Evolution (with J2)")
    print(f"  Seed: {true_seed}")
    print(f"  ΔV scale: {DELTA_V_SCALE*1000:.1f} mm/s")
    print(f"  Steps: {STEPS}, Symbol period: {SYMBOL_PERIOD:.0f}s")

    expander = ThreeSatelliteExpander(harmonics=HARMONICS, include_j2=True)
    times, clean_pr, states_hist = expander.expand_seed(
        true_seed, steps=STEPS, deltaV_scale=DELTA_V_SCALE,
        symbol_period=SYMBOL_PERIOD
    )

    # Baseline (no maneuver)
    baseline_seed = [0.166] * 15
    times_b, baseline_pr, _ = expander.expand_seed(
        baseline_seed, steps=STEPS, deltaV_scale=DELTA_V_SCALE,
        symbol_period=SYMBOL_PERIOD
    )

    n = min(len(clean_pr), len(baseline_pr))
    deviation = clean_pr[:n] - baseline_pr[:n]

    print(f"  Generated {len(times)} time samples across 3 channels")
    print(f"  Max phase-rate deviation: {np.max(np.abs(deviation)):.2e} rad/s")
    print()

    # ===== STAGE 2: NOISE (Corrupt Observations) =====
    print("[2/4] NOISE: Adding realistic observation noise")

    noise_model = PhaseRateNoiseModel(carrier_snr_db=30.0)
    budget = noise_model.link_budget(delta_v=DELTA_V_SCALE,
                                     symbol_period=SYMBOL_PERIOD)

    print(f"  Noise breakdown:")
    print(f"    Thermal:       {budget['thermal_noise']:.2e} rad/s")
    print(f"    Clock:         {budget['clock_noise']:.2e} rad/s")
    print(f"    Perturbation:  {budget['perturbation_noise']:.2e} rad/s")
    print(f"    Ionospheric:   {budget['iono_noise']:.2e} rad/s")
    print(f"    Total:         {budget['noise_std']:.2e} rad/s")
    print(f"  Single-sample SNR: {budget['snr_single_sample_db']:.1f} dB")
    print(f"  Effective SNR:     {budget['effective_snr_db']:.1f} dB")

    # Generate noisy observation
    noisy_pr = clean_pr.copy()
    for ch in range(3):
        noise = noise_model.generate_noise(
            len(clean_pr), integration_time=SYMBOL_PERIOD, rng=rng
        )
        noisy_pr[:, ch] += noise

    print()

    # ===== STAGE 3: DETECT (Signal Detection Analysis) =====
    print("[3/4] DETECTION: Matched filter analysis")

    analyzer = DetectionAnalyzer(harmonics=HARMONICS, include_j2=True)
    perf = analyzer.matched_filter_performance(
        true_seed, delta_v_scale=DELTA_V_SCALE,
        steps=STEPS, symbol_period=SYMBOL_PERIOD
    )

    print(f"  Matched filter SNR: {perf['matched_filter_snr_db']:.1f} dB")
    print(f"  Detection probabilities:")
    for pfa_label, pd in perf['pd_at_pfa'].items():
        print(f"    {pfa_label}: Pd = {pd:.4f}")
    print(f"  Min ΔV for Pd>0.9 (Pfa=1e-4): "
          f"{perf['min_delta_v_for_pd90']*1000:.2f} mm/s")
    print(f"  Channel SNRs: "
          f"{[f'{s:.1f} dB' for s in perf['channel_snrs_db']]}")

    # ΔV sweep
    dv_range, pd_sweep, snr_sweep = analyzer.delta_v_sweep(
        true_seed, steps=STEPS, symbol_period=SYMBOL_PERIOD
    )
    print()

    # ===== STAGE 4: RECOVER (Inverse Problem) =====
    print("[4/4] RECOVERY: Seed reconstruction from noisy observations")
    print(f"  Coarse search: 100 random candidates...")

    recoverer = SeedRecoverer(
        harmonics=HARMONICS, include_j2=True,
        steps=STEPS, delta_v_scale=DELTA_V_SCALE,
        symbol_period=SYMBOL_PERIOD
    )

    recovered_seed, cost, correlation, diagnostics = recoverer.recover(
        noisy_pr, n_candidates=100, refine_top_k=3, rng=rng
    )

    eval_result = recoverer.evaluate_recovery(true_seed, recovered_seed)

    print(f"  Final correlation: {correlation:.4f}")
    print(f"  Seed RMSE: {eval_result['rmse']:.4f}")
    print(f"  Max error: {eval_result['max_error']:.4f}")
    print(f"  Per-satellite RMSE: "
          f"{[f'{e:.4f}' for e in eval_result['per_satellite_rmse']]}")
    print()

    # Print seed comparison
    print("  Seed Comparison:")
    print(f"  {'Value':>6}  {'True':>8}  {'Recovered':>10}  {'Error':>8}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*8}")
    labels = ['pro', 'ret', 'out', 'in', 'nor'] * 3
    for i in range(15):
        sat = 'ABC'[i // 5]
        label = f"{sat}.{labels[i]}"
        print(f"  {label:>6}  {true_seed[i]:8.4f}  "
              f"{recovered_seed[i]:10.4f}  "
              f"{eval_result['per_value_error'][i]:+8.4f}")
    print()

    # ===== VISUALIZATION =====
    print("Generating visualization...")

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Orbital PHYCOM: Full Communication Pipeline\n'
                 'Encode → Noise → Detect → Recover',
                 fontsize=14, fontweight='bold')

    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

    pair_labels = ['AB (2:3)', 'BC (3:5)', 'AC (2:5)']
    colors = ['#2196F3', '#FF5722', '#4CAF50']

    # Row 1: Phase rates (clean vs noisy) for each channel
    for ch in range(3):
        ax = fig.add_subplot(gs[0, ch])
        t_min = times[:n] / 60
        ax.plot(t_min, clean_pr[:n, ch], color=colors[ch],
                alpha=0.8, linewidth=1.5, label='Clean')
        ax.plot(t_min, noisy_pr[:n, ch], color='gray',
                alpha=0.3, linewidth=0.5, label='Noisy')
        ax.set_title(f'Channel {pair_labels[ch]}', fontsize=10)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Phase rate (rad/s)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Row 1, col 4: Link budget bar chart
    ax_budget = fig.add_subplot(gs[0, 3])
    noise_sources = ['Thermal', 'Clock', 'Perturb', 'Iono']
    noise_vals = [budget['thermal_noise'], budget['clock_noise'],
                  budget['perturbation_noise'], budget['iono_noise']]
    bar_colors = ['#2196F3', '#FF9800', '#F44336', '#9C27B0']
    ax_budget.barh(noise_sources, noise_vals, color=bar_colors, alpha=0.7)
    ax_budget.set_xlabel('Noise (rad/s)')
    ax_budget.set_title('Noise Budget', fontsize=10)
    ax_budget.set_xscale('log')
    ax_budget.axvline(budget['signal_phase_rate'], color='green',
                      linestyle='--', label=f'Signal={budget["signal_phase_rate"]:.3f}')
    ax_budget.legend(fontsize=7)

    # Row 2: Deviation from baseline
    for ch in range(3):
        ax = fig.add_subplot(gs[1, ch])
        ax.plot(t_min, deviation[:, ch], color=colors[ch], linewidth=1.5)
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
        ax.set_title(f'Deviation {pair_labels[ch]}', fontsize=10)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Δ Phase rate (rad/s)')
        ax.grid(True, alpha=0.3)

    # Row 2, col 4: Detection probability vs ΔV
    ax_det = fig.add_subplot(gs[1, 3])
    ax_det.semilogx(dv_range * 1000, pd_sweep, 'b-', linewidth=2)
    ax_det.axhline(0.9, color='red', linestyle='--', alpha=0.5,
                   label='Pd = 0.9')
    ax_det.axvline(DELTA_V_SCALE * 1000, color='green', linestyle=':',
                   alpha=0.5, label=f'Design ΔV = {DELTA_V_SCALE*1000:.0f} mm/s')
    ax_det.set_xlabel('ΔV (mm/s)')
    ax_det.set_ylabel('Detection Probability')
    ax_det.set_title('Pd vs ΔV (Pfa=1e-4)', fontsize=10)
    ax_det.set_ylim(-0.05, 1.05)
    ax_det.legend(fontsize=7)
    ax_det.grid(True, alpha=0.3)

    # Row 3: Recovery results
    # Seed comparison bar chart
    ax_seed = fig.add_subplot(gs[2, 0:2])
    x_pos = np.arange(15)
    width = 0.35
    ax_seed.bar(x_pos - width/2, true_seed, width, label='True',
                color='#2196F3', alpha=0.7)
    ax_seed.bar(x_pos + width/2, recovered_seed, width, label='Recovered',
                color='#FF5722', alpha=0.7)
    ax_seed.set_xticks(x_pos)
    labels_short = [f"{'ABC'[i//5]}{i%5}" for i in range(15)]
    ax_seed.set_xticklabels(labels_short, fontsize=7)
    ax_seed.set_ylabel('Seed Value')
    ax_seed.set_title('Seed Recovery: True vs Recovered', fontsize=10)
    ax_seed.legend(fontsize=8)
    ax_seed.grid(True, alpha=0.3, axis='y')

    # Recovery error per value
    ax_err = fig.add_subplot(gs[2, 2])
    errors = eval_result['per_value_error']
    bar_colors_err = ['#2196F3'] * 5 + ['#FF5722'] * 5 + ['#4CAF50'] * 5
    ax_err.bar(x_pos, errors, color=bar_colors_err, alpha=0.7)
    ax_err.axhline(0, color='black', linewidth=0.5)
    ax_err.set_xticks(x_pos)
    ax_err.set_xticklabels(labels_short, fontsize=7)
    ax_err.set_ylabel('Error')
    ax_err.set_title(f'Recovery Error (RMSE={eval_result["rmse"]:.4f})',
                     fontsize=10)
    ax_err.grid(True, alpha=0.3, axis='y')

    # Summary text panel
    ax_txt = fig.add_subplot(gs[2, 3])
    ax_txt.axis('off')
    summary = (
        f"Pipeline Summary\n"
        f"{'─' * 28}\n"
        f"Constellation: {HARMONICS}\n"
        f"ΔV scale: {DELTA_V_SCALE*1000:.1f} mm/s\n"
        f"J2 perturbation: ON\n"
        f"Steps: {STEPS}\n"
        f"Symbol period: {SYMBOL_PERIOD:.0f}s\n"
        f"{'─' * 28}\n"
        f"Effective SNR: {budget['effective_snr_db']:.1f} dB\n"
        f"MF SNR: {perf['matched_filter_snr_db']:.1f} dB\n"
        f"Pd (Pfa=1e-4): {perf['pd_at_pfa'].get('Pfa=1e-04', 0):.3f}\n"
        f"{'─' * 28}\n"
        f"Recovery correlation: {correlation:.4f}\n"
        f"Seed RMSE: {eval_result['rmse']:.4f}\n"
        f"Max error: {eval_result['max_error']:.4f}\n"
    )
    ax_txt.text(0.05, 0.95, summary, transform=ax_txt.transAxes,
                fontsize=9, fontfamily='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig('pipeline_results.png', dpi=150, bbox_inches='tight')
    print(f"Saved: pipeline_results.png")

    print()
    print("=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()
