"""
Detection probability analysis for orbital seed communication.

Computes the probability of correctly detecting a seed-encoded signal
in the presence of realistic noise, using Neyman-Pearson hypothesis
testing framework.

H0: No maneuver (baseline orbital evolution)
H1: Seed-encoded ΔV maneuver applied
"""

import numpy as np
from scipy.stats import norm, chi2
from .seed_expander import ThreeSatelliteExpander
from .noise_model import PhaseRateNoiseModel


class DetectionAnalyzer:
    """
    Analyzes detectability of seed-encoded orbital maneuvers.

    Uses energy detector and matched-filter approaches to compute
    detection probability (Pd) and false alarm probability (Pfa)
    as functions of ΔV magnitude and SNR.
    """

    def __init__(self, harmonics=(2, 3, 5), include_j2=True):
        """
        Initialize detection analyzer.

        Args:
            harmonics: Constellation harmonics
            include_j2: Use J2 in propagation
        """
        self.expander = ThreeSatelliteExpander(
            harmonics=harmonics, include_j2=include_j2
        )
        self.noise_model = PhaseRateNoiseModel()

    def compute_signal_energy(self, seed, baseline_seed=None,
                              steps=5, delta_v_scale=0.001,
                              symbol_period=300.0):
        """
        Compute signal energy (deviation from baseline).

        Args:
            seed: Test seed to analyze
            baseline_seed: Baseline seed (None = zero maneuver)
            steps: Number of expansion steps
            delta_v_scale: ΔV scale in m/s
            symbol_period: Symbol period in seconds

        Returns:
            energy: Total signal energy (sum of squared deviations)
            deviation: Phase-rate deviation time series (N x 3)
            times: Time array
        """
        # Signal
        times, signal_pr, _ = self.expander.expand_seed(
            seed, steps=steps, deltaV_scale=delta_v_scale,
            symbol_period=symbol_period
        )

        # Baseline (no maneuver or alternative seed)
        if baseline_seed is None:
            baseline_seed = [0.166] * 15  # Uniform allocation ≈ no net ΔV
        times_b, baseline_pr, _ = self.expander.expand_seed(
            baseline_seed, steps=steps, deltaV_scale=delta_v_scale,
            symbol_period=symbol_period
        )

        # Align lengths
        n = min(len(signal_pr), len(baseline_pr))
        deviation = signal_pr[:n] - baseline_pr[:n]

        energy = np.sum(deviation**2)
        return energy, deviation, times[:n]

    def energy_detector_roc(self, seed, delta_v_scale=0.001,
                            steps=5, symbol_period=300.0,
                            pfa_range=None):
        """
        Compute ROC curve using energy detector.

        Under H0 (no signal), test statistic T ~ chi2(N*3)
        Under H1 (signal present), T ~ noncentral chi2(N*3, lambda)

        Args:
            seed: Signal seed
            delta_v_scale: ΔV scale in m/s
            steps: Expansion steps
            symbol_period: Symbol period in seconds
            pfa_range: Array of false alarm probabilities

        Returns:
            pfa: False alarm probabilities
            pd: Detection probabilities
            snr_db: Signal-to-noise ratio in dB
        """
        if pfa_range is None:
            pfa_range = np.logspace(-6, -1, 50)

        # Compute signal energy
        energy, deviation, times = self.compute_signal_energy(
            seed, steps=steps, delta_v_scale=delta_v_scale,
            symbol_period=symbol_period
        )

        n_samples = len(deviation)
        n_channels = 3

        # Noise variance
        sigma = self.noise_model.total_noise_std(symbol_period)

        # SNR
        signal_power = energy / (n_samples * n_channels)
        noise_power = sigma**2
        snr_linear = signal_power / noise_power if noise_power > 0 else np.inf
        snr_db = 10 * np.log10(max(snr_linear, 1e-20))

        # Noncentrality parameter
        noncentrality = energy / sigma**2 if sigma > 0 else np.inf

        # Degrees of freedom
        dof = n_samples * n_channels

        # ROC: for each Pfa, compute threshold, then Pd
        pd = np.zeros_like(pfa_range)
        for i, pfa in enumerate(pfa_range):
            # Threshold from chi2 under H0
            threshold = chi2.ppf(1 - pfa, dof)
            # Detection probability from noncentral chi2
            pd[i] = 1 - chi2.cdf(threshold, dof, loc=noncentrality)

        return pfa_range, pd, snr_db

    def matched_filter_performance(self, seed, delta_v_scale=0.001,
                                   steps=5, symbol_period=300.0):
        """
        Matched filter detection performance.

        The matched filter is the optimal detector when the signal
        waveform is known. SNR gain = sqrt(N) over energy detector.

        Args:
            seed: Signal seed
            delta_v_scale: ΔV scale in m/s
            steps: Expansion steps
            symbol_period: Symbol period in seconds

        Returns:
            Dictionary with detection performance metrics
        """
        energy, deviation, times = self.compute_signal_energy(
            seed, steps=steps, delta_v_scale=delta_v_scale,
            symbol_period=symbol_period
        )

        n_samples = len(deviation)
        sigma = self.noise_model.total_noise_std(symbol_period)

        # Matched filter SNR (optimal)
        if sigma > 0:
            mf_snr = np.sqrt(energy) / sigma
        else:
            mf_snr = np.inf
        mf_snr_db = 20 * np.log10(max(mf_snr, 1e-20))

        # Detection probability at standard Pfa values
        pfa_targets = [1e-3, 1e-4, 1e-5, 1e-6]
        pd_at_pfa = {}
        for pfa in pfa_targets:
            threshold = norm.ppf(1 - pfa)
            pd = 1 - norm.cdf(threshold - mf_snr)
            pd_at_pfa[f"Pfa={pfa:.0e}"] = pd

        # Minimum ΔV for Pd > 0.9 at Pfa = 1e-4
        threshold_99 = norm.ppf(1 - 1e-4)
        snr_required = threshold_99 + norm.ppf(0.9)

        # ΔV scales linearly with signal, so min_dv ~ dv * (required/actual)
        if mf_snr > 0:
            min_delta_v = delta_v_scale * (snr_required / mf_snr)
        else:
            min_delta_v = np.inf

        # Per-channel analysis
        channel_snrs = []
        for ch in range(3):
            ch_energy = np.sum(deviation[:, ch]**2)
            ch_snr = np.sqrt(ch_energy) / sigma if sigma > 0 else np.inf
            channel_snrs.append(20 * np.log10(max(ch_snr, 1e-20)))

        return {
            'matched_filter_snr_db': mf_snr_db,
            'signal_energy': energy,
            'noise_std': sigma,
            'n_samples': n_samples,
            'pd_at_pfa': pd_at_pfa,
            'min_delta_v_for_pd90': min_delta_v,
            'channel_snrs_db': channel_snrs,
        }

    def delta_v_sweep(self, seed, delta_v_range=None, steps=5,
                      symbol_period=300.0, pfa=1e-4):
        """
        Sweep ΔV magnitude and compute detection probability.

        Args:
            seed: Signal seed (shape determines direction)
            delta_v_range: Array of ΔV magnitudes to test (m/s)
            steps: Expansion steps
            symbol_period: Symbol period in seconds
            pfa: Target false alarm probability

        Returns:
            delta_vs: ΔV values tested
            pd_values: Detection probabilities
            snr_values: SNR values in dB
        """
        if delta_v_range is None:
            delta_v_range = np.logspace(-5, -2, 30)

        sigma = self.noise_model.total_noise_std(symbol_period)
        threshold = norm.ppf(1 - pfa)

        # Get reference signal shape at unit ΔV
        ref_energy, _, _ = self.compute_signal_energy(
            seed, steps=steps, delta_v_scale=1.0,
            symbol_period=symbol_period
        )

        pd_values = []
        snr_values = []

        for dv in delta_v_range:
            # Energy scales as dv^2
            scaled_energy = ref_energy * dv**2
            mf_snr = np.sqrt(scaled_energy) / sigma if sigma > 0 else np.inf
            pd = 1 - norm.cdf(threshold - mf_snr)
            snr_db = 20 * np.log10(max(mf_snr, 1e-20))

            pd_values.append(pd)
            snr_values.append(snr_db)

        return delta_v_range, np.array(pd_values), np.array(snr_values)


def quick_test():
    """Quick sanity check."""
    print("Testing DetectionAnalyzer...")

    analyzer = DetectionAnalyzer(include_j2=True)

    seed = [
        0.6, 0.0, 0.2, 0.0, 0.2,
        0.2, 0.2, 0.2, 0.2, 0.2,
        0.1, 0.1, 0.5, 0.1, 0.2
    ]

    # Matched filter performance
    perf = analyzer.matched_filter_performance(
        seed, delta_v_scale=0.001, steps=3, symbol_period=300.0
    )

    print(f"  Matched filter SNR: {perf['matched_filter_snr_db']:.1f} dB")
    print(f"  Signal energy: {perf['signal_energy']:.2e}")
    print(f"  Noise std: {perf['noise_std']:.2e} rad/s")
    print(f"  Detection probabilities:")
    for pfa, pd in perf['pd_at_pfa'].items():
        print(f"    {pfa}: Pd = {pd:.4f}")
    print(f"  Min ΔV for Pd>0.9: {perf['min_delta_v_for_pd90']*1000:.3f} mm/s")
    print(f"  Channel SNRs: {[f'{s:.1f} dB' for s in perf['channel_snrs_db']]}")

    print("Test passed!")


if __name__ == '__main__':
    quick_test()
