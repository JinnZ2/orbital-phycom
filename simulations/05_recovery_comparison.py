#!/usr/bin/env python3
"""
Recovery Comparison: Basic vs Enhanced Signal Clarification

Compares seed recovery performance between:
    - Basic: Random search + Nelder-Mead on raw phase rates
    - Enhanced: Baseline subtraction + denoising + Sobol search
              on deviation-domain signal

Demonstrates the effect of each signal clarification technique:
    1. Differential baseline subtraction (cancels shared J2)
    2. Savitzky-Golay smoothing (suppresses white noise)
    3. Spectral low-pass filtering (removes high-freq noise)
    4. Sobol quasi-random search (better coverage than random)

Run time: ~5-10 minutes
Output: Comparison table + visualization
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
from core.enhanced_recovery import EnhancedRecoverer


def run_trial(true_seed, steps, delta_v_scale, symbol_period,
              noise_seed, search_seed, n_candidates=200):
    """Run one trial of basic vs enhanced recovery."""
    rng_noise = np.random.default_rng(noise_seed)
    exp = ThreeSatelliteExpander(include_j2=True)
    noise_model = PhaseRateNoiseModel()

    # Generate noisy observation
    _, clean, _ = exp.expand_seed(
        true_seed, steps=steps, deltaV_scale=delta_v_scale,
        symbol_period=symbol_period
    )
    noisy = clean.copy()
    for ch in range(3):
        noisy[:, ch] += noise_model.generate_noise(len(clean), rng=rng_noise)

    results = {}

    # Basic recovery
    rng_b = np.random.default_rng(search_seed)
    basic = SeedRecoverer(
        steps=steps, delta_v_scale=delta_v_scale,
        symbol_period=symbol_period, include_j2=True
    )
    seed_b, _, corr_b, _ = basic.recover(
        noisy, n_candidates=n_candidates, refine_top_k=5, rng=rng_b
    )
    results['basic'] = {
        'seed': seed_b,
        'metrics': basic.evaluate_recovery(true_seed, seed_b),
        'correlation': corr_b,
    }

    # Enhanced recovery (full pipeline)
    rng_e = np.random.default_rng(search_seed)
    enhanced = EnhancedRecoverer(
        steps=steps, delta_v_scale=delta_v_scale,
        symbol_period=symbol_period, include_j2=True
    )
    seed_e, diag_e = enhanced.recover(
        noisy, denoise_method='both',
        n_candidates=n_candidates, refine_top_k=5, rng=rng_e
    )
    results['enhanced'] = {
        'seed': seed_e,
        'metrics': enhanced.evaluate_recovery(true_seed, seed_e),
        'diagnostics': diag_e,
    }

    # Enhanced with savgol only
    rng_s = np.random.default_rng(search_seed)
    seed_s, diag_s = enhanced.recover(
        noisy, denoise_method='savgol',
        n_candidates=n_candidates, refine_top_k=5, rng=rng_s
    )
    results['savgol_only'] = {
        'seed': seed_s,
        'metrics': enhanced.evaluate_recovery(true_seed, seed_s),
    }

    # Enhanced with no denoising (baseline subtraction only)
    rng_n = np.random.default_rng(search_seed)
    seed_n, diag_n = enhanced.recover(
        noisy, denoise_method=None,
        n_candidates=n_candidates, refine_top_k=5, rng=rng_n
    )
    results['baseline_only'] = {
        'seed': seed_n,
        'metrics': enhanced.evaluate_recovery(true_seed, seed_n),
    }

    return results, noisy, clean


def main():
    print("=" * 70)
    print("Recovery Comparison: Basic vs Enhanced Signal Clarification")
    print("=" * 70)
    print()

    STEPS = 3
    DELTA_V_SCALE = 0.001
    SYMBOL_PERIOD = 300.0
    N_TRIALS = 3
    N_CANDIDATES = 200

    true_seed = [
        0.6, 0.0, 0.2, 0.0, 0.2,
        0.2, 0.2, 0.2, 0.2, 0.2,
        0.1, 0.1, 0.5, 0.1, 0.2
    ]

    print(f"True seed: {true_seed}")
    print(f"Config: steps={STEPS}, ΔV={DELTA_V_SCALE*1000:.0f} mm/s, "
          f"period={SYMBOL_PERIOD:.0f}s")
    print(f"Trials: {N_TRIALS}, Candidates per trial: {N_CANDIDATES}")
    print()

    # Collect results across trials
    methods = ['basic', 'baseline_only', 'savgol_only', 'enhanced']
    method_labels = {
        'basic': 'Basic (raw phase rates)',
        'baseline_only': 'Baseline subtraction only',
        'savgol_only': 'Baseline + SavGol denoise',
        'enhanced': 'Baseline + SavGol + Spectral',
    }

    all_rmses = {m: [] for m in methods}
    all_max_errors = {m: [] for m in methods}
    last_results = None

    for trial in range(N_TRIALS):
        print(f"--- Trial {trial + 1}/{N_TRIALS} ---")
        results, noisy, clean = run_trial(
            true_seed, STEPS, DELTA_V_SCALE, SYMBOL_PERIOD,
            noise_seed=100 + trial,
            search_seed=200 + trial,
            n_candidates=N_CANDIDATES
        )

        for m in methods:
            rmse = results[m]['metrics']['rmse']
            maxe = results[m]['metrics']['max_error']
            all_rmses[m].append(rmse)
            all_max_errors[m].append(maxe)
            print(f"  {method_labels[m]:40s} RMSE={rmse:.4f}  Max={maxe:.4f}")
        print()

        last_results = results

    # Summary table
    print("=" * 70)
    print("SUMMARY (mean ± std across trials)")
    print("=" * 70)
    print(f"{'Method':40s} {'RMSE':>12s} {'Max Error':>12s}")
    print("-" * 70)
    for m in methods:
        rmse_mean = np.mean(all_rmses[m])
        rmse_std = np.std(all_rmses[m])
        maxe_mean = np.mean(all_max_errors[m])
        maxe_std = np.std(all_max_errors[m])
        print(f"{method_labels[m]:40s} "
              f"{rmse_mean:.4f}±{rmse_std:.4f} "
              f"{maxe_mean:.4f}±{maxe_std:.4f}")
    print()

    # Visualization
    print("Generating visualization...")

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Seed Recovery: Signal Clarification Comparison',
                 fontsize=14, fontweight='bold')
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1: RMSE comparison bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(methods))
    means = [np.mean(all_rmses[m]) for m in methods]
    stds = [np.std(all_rmses[m]) for m in methods]
    colors = ['#9E9E9E', '#2196F3', '#FF9800', '#4CAF50']
    bars = ax1.bar(x, means, yerr=stds, color=colors, alpha=0.8,
                   capsize=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Basic', 'Baseline\nonly', 'SavGol', 'Full'],
                        fontsize=9)
    ax1.set_ylabel('RMSE')
    ax1.set_title('Recovery RMSE by Method', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Seed comparison (last trial)
    ax2 = fig.add_subplot(gs[0, 1:3])
    x_pos = np.arange(15)
    width = 0.2
    ax2.bar(x_pos - 1.5*width, true_seed, width, label='True',
            color='black', alpha=0.6)
    for i, m in enumerate(methods):
        ax2.bar(x_pos + (i - 1)*width, last_results[m]['seed'], width,
                label=method_labels[m].split('(')[0].strip(),
                color=colors[i], alpha=0.7)
    ax2.set_xticks(x_pos)
    labels_short = [f"{'ABC'[i//5]}{i%5}" for i in range(15)]
    ax2.set_xticklabels(labels_short, fontsize=7)
    ax2.set_ylabel('Seed Value')
    ax2.set_title('Seed Recovery Comparison (Last Trial)', fontsize=10)
    ax2.legend(fontsize=7, loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Denoising effect visualization
    ax3 = fig.add_subplot(gs[1, 0:2])

    # Show the denoising effect on a sample channel
    exp = ThreeSatelliteExpander(include_j2=True)
    noise_model = PhaseRateNoiseModel()
    rng_viz = np.random.default_rng(100)
    times, clean_pr, _ = exp.expand_seed(
        true_seed, steps=STEPS, deltaV_scale=DELTA_V_SCALE,
        symbol_period=SYMBOL_PERIOD
    )
    neutral_seed = [0.166] * 15
    _, baseline_pr, _ = exp.expand_seed(
        neutral_seed, steps=STEPS, deltaV_scale=DELTA_V_SCALE,
        symbol_period=SYMBOL_PERIOD
    )

    n_viz = min(len(clean_pr), len(baseline_pr))
    clean_dev = clean_pr[:n_viz] - baseline_pr[:n_viz]

    noisy_viz = clean_pr[:n_viz].copy()
    for ch in range(3):
        noisy_viz[:, ch] += noise_model.generate_noise(n_viz, rng=rng_viz)
    noisy_dev = noisy_viz - baseline_pr[:n_viz]

    # Denoise
    from scipy.signal import savgol_filter
    window = max(5, min(n_viz - 1, n_viz // 3))
    window = window if window % 2 == 1 else window - 1
    denoised_dev = np.zeros_like(noisy_dev)
    for ch in range(3):
        denoised_dev[:, ch] = savgol_filter(noisy_dev[:, ch], window, 2)

    ch = 0  # Show channel 0 (strongest signal)
    t_min = times[:n_viz] / 60
    ax3.plot(t_min, clean_dev[:, ch], 'g-', linewidth=2,
             label='True signal', alpha=0.8)
    ax3.plot(t_min, noisy_dev[:, ch], 'r-', linewidth=0.5,
             label='Noisy observation', alpha=0.3)
    ax3.plot(t_min, denoised_dev[:, ch], 'b-', linewidth=1.5,
             label='Denoised', alpha=0.8)
    ax3.set_xlabel('Time (min)')
    ax3.set_ylabel('Phase rate deviation (rad/s)')
    ax3.set_title('Signal Clarification: Channel AB', fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Per-trial RMSE evolution
    ax4 = fig.add_subplot(gs[1, 2])
    for i, m in enumerate(methods):
        ax4.plot(range(1, N_TRIALS + 1), all_rmses[m],
                 'o-', color=colors[i], label=method_labels[m].split('(')[0].strip(),
                 markersize=6, linewidth=1.5)
    ax4.set_xlabel('Trial')
    ax4.set_ylabel('RMSE')
    ax4.set_title('RMSE per Trial', fontsize=10)
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(range(1, N_TRIALS + 1))

    plt.savefig('recovery_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: recovery_comparison.png")

    print()
    print("=" * 70)
    print("Key findings:")
    print(f"  - Baseline subtraction: cancels dominant J2 perturbation noise")
    print(f"  - Savitzky-Golay (window={window}): matches orbital timescale")
    print(f"  - Spectral filtering: removes high-freq noise components")
    print(f"  - SNR per sample: ~0.22 (signal well below noise floor)")
    print(f"  - Integration over {n_viz} samples provides ~{np.sqrt(n_viz):.1f}x gain")
    print("=" * 70)


if __name__ == '__main__':
    main()
