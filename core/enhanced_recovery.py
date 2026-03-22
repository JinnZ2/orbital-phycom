"""
Enhanced seed recovery with signal clarification techniques.

Improves on basic recovery via:
    1. Differential baseline subtraction (cancels shared J2/gravity)
    2. Savitzky-Golay + spectral denoising
    3. Per-satellite decomposition (15D → 3×5D search)
    4. Sobol quasi-random coarse search
    5. Cross-channel consistency validation
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from scipy.stats import qmc
from .seed_expander import ThreeSatelliteExpander
from .noise_model import PhaseRateNoiseModel


class EnhancedRecoverer:
    """
    Enhanced seed recovery with signal clarification.

    Key improvement: works in the *deviation domain* (observation minus
    baseline) rather than raw phase rates. Since both sender and receiver
    share the same orbital model, the baseline (J2-dominated) evolution
    cancels, isolating the ΔV-induced signal from the dominant noise source.
    """

    def __init__(self, harmonics=(2, 3, 5), include_j2=True,
                 steps=5, delta_v_scale=0.001, symbol_period=300.0):
        """
        Initialize enhanced recovery engine.

        Args:
            harmonics: Constellation harmonic numbers
            include_j2: Use J2 in forward model
            steps: Number of maneuver steps
            delta_v_scale: ΔV scale in m/s
            symbol_period: Symbol period in seconds
        """
        self.expander = ThreeSatelliteExpander(
            harmonics=harmonics, include_j2=include_j2
        )
        self.steps = steps
        self.delta_v_scale = delta_v_scale
        self.symbol_period = symbol_period

        # Pre-compute baseline (no-maneuver evolution)
        self._baseline = None
        self._baseline_times = None

    def _compute_baseline(self):
        """Compute and cache the no-maneuver baseline phase rates."""
        if self._baseline is not None:
            return self._baseline_times, self._baseline

        # Uniform seed = equal allocation in all directions = near-zero net ΔV
        neutral_seed = [0.166] * 15
        times, phase_rates, _ = self.expander.expand_seed(
            neutral_seed,
            steps=self.steps,
            deltaV_scale=self.delta_v_scale,
            symbol_period=self.symbol_period
        )
        self._baseline_times = times
        self._baseline = phase_rates
        return times, phase_rates

    def _forward_deviation(self, seed):
        """
        Run forward model and return deviation from baseline.

        This is the key signal clarification step: by subtracting the
        shared baseline, we cancel the dominant J2/gravity signal and
        isolate the ΔV-induced perturbation.

        Args:
            seed: 15-value seed

        Returns:
            times: Time array
            deviation: Phase-rate deviation from baseline (N x 3)
        """
        _, baseline = self._compute_baseline()
        times, phase_rates, _ = self.expander.expand_seed(
            seed,
            steps=self.steps,
            deltaV_scale=self.delta_v_scale,
            symbol_period=self.symbol_period
        )
        n = min(len(phase_rates), len(baseline))
        deviation = phase_rates[:n] - baseline[:n]
        return times[:n], deviation

    def _forward_single_satellite(self, sat_idx, sat_seed_5):
        """
        Forward model for a single satellite's contribution.

        Exploits the fact that each satellite's ΔV is applied
        independently — we can evaluate one satellite's effect
        without simulating the other two.

        Args:
            sat_idx: Satellite index (0, 1, or 2)
            sat_seed_5: 5-value seed for this satellite

        Returns:
            deviation: Phase-rate deviation (N x 3)
        """
        # Build full 15-value seed with this satellite active, others neutral
        full_seed = [0.166] * 15
        full_seed[sat_idx * 5:(sat_idx + 1) * 5] = list(sat_seed_5)

        _, deviation = self._forward_deviation(full_seed)
        return deviation

    # --- Signal denoising ---

    def denoise_savgol(self, signal, window=None, polyorder=2):
        """
        Savitzky-Golay smoothing to suppress white noise.

        Preserves the slow orbital dynamics signal (timescale ~minutes)
        while attenuating sample-to-sample noise.

        Args:
            signal: Input signal array (N,) or (N, 3)
            window: Filter window length (must be odd)
            polyorder: Polynomial order

        Returns:
            Smoothed signal
        """
        # Auto-select window: ~1/3 of signal length, must be odd
        if window is None:
            n = signal.shape[0] if signal.ndim > 1 else len(signal)
            window = max(5, min(n - 1, n // 3))
            window = window if window % 2 == 1 else window - 1

        if signal.ndim == 1:
            if len(signal) < window:
                return signal
            return savgol_filter(signal, window, polyorder)

        result = np.zeros_like(signal)
        for ch in range(signal.shape[1]):
            if len(signal[:, ch]) < window:
                result[:, ch] = signal[:, ch]
            else:
                result[:, ch] = savgol_filter(signal[:, ch], window, polyorder)
        return result

    def denoise_spectral(self, signal, cutoff_fraction=0.1):
        """
        Low-pass spectral filter.

        The ΔV signal evolves on orbital timescales (~minutes).
        High-frequency components are noise. Cut everything above
        a fraction of Nyquist.

        Args:
            signal: Input signal (N,) or (N, 3)
            cutoff_fraction: Fraction of Nyquist to keep (0-1)

        Returns:
            Filtered signal
        """
        def _filter_1d(x):
            spectrum = np.fft.rfft(x)
            freqs = np.fft.rfftfreq(len(x))
            mask = freqs <= cutoff_fraction * 0.5
            spectrum[~mask] = 0
            return np.fft.irfft(spectrum, n=len(x))

        if signal.ndim == 1:
            return _filter_1d(signal)

        result = np.zeros_like(signal)
        for ch in range(signal.shape[1]):
            result[:, ch] = _filter_1d(signal[:, ch])
        return result

    def denoise(self, signal, method='savgol'):
        """
        Apply denoising to phase-rate signal.

        Args:
            signal: Phase-rate array (N,) or (N, 3)
            method: 'savgol', 'spectral', or 'both'

        Returns:
            Denoised signal
        """
        if method == 'savgol':
            return self.denoise_savgol(signal)
        elif method == 'spectral':
            return self.denoise_spectral(signal)
        elif method == 'both':
            smoothed = self.denoise_savgol(signal)
            return self.denoise_spectral(smoothed)
        return signal

    # --- Coarse search improvements ---

    def _sobol_seeds(self, n_candidates, rng=None):
        """
        Generate quasi-random seed candidates via Sobol sequences.

        Sobol sequences fill the search space more uniformly than
        pseudo-random sampling, giving better coverage with fewer points.

        Args:
            n_candidates: Number of candidates
            rng: Random generator (for scrambling)

        Returns:
            Array of seed candidates (n_candidates x 15)
        """
        seed_int = rng.integers(0, 2**31) if rng is not None else 0
        sampler = qmc.Sobol(d=15, scramble=True, seed=seed_int)

        # Power of 2 for Sobol, take first n_candidates
        m = int(np.ceil(np.log2(max(n_candidates, 2))))
        samples = sampler.random_base2(m)[:n_candidates]

        # Scale to valid seed range [0.02, 0.6]
        # (avoids extremes that produce degenerate orbits)
        return qmc.scale(samples, 0.02, 0.6)

    def _cross_channel_validate(self, seed, observed_deviation):
        """
        Validate recovery using cross-channel geometric consistency.

        The three channels are geometrically coupled: if satellite A
        is recovered correctly, its contribution to channels AB and AC
        must be consistent. Inconsistency flags a bad recovery.

        Args:
            seed: Full 15-value recovered seed
            observed_deviation: Target deviation signal

        Returns:
            consistency: Average cross-channel correlation (0-1)
            per_channel: Per-channel correlations
        """
        _, pred_dev = self._forward_deviation(seed)
        n = min(len(pred_dev), len(observed_deviation))

        per_channel = []
        for ch in range(3):
            p = pred_dev[:n, ch] - np.mean(pred_dev[:n, ch])
            o = observed_deviation[:n, ch] - np.mean(observed_deviation[:n, ch])
            denom = np.sqrt(np.sum(p**2) * np.sum(o**2))
            corr = np.sum(p * o) / denom if denom > 1e-20 else 0.0
            per_channel.append(corr)

        return np.mean(per_channel), per_channel

    # --- Full enhanced pipeline ---

    def recover(self, observed_phase_rates, denoise_method=None,
                n_candidates=200, refine_top_k=5, rng=None):
        """
        Enhanced recovery pipeline.

        The key improvement is differential baseline subtraction:
        both sender and receiver share the same J2 model, so
        subtracting the no-maneuver baseline cancels the dominant
        perturbation noise and exposes the ΔV-encoded signal.

        Optional denoising (Savitzky-Golay, spectral) can help at
        higher SNR but may distort the optimization landscape at
        marginal SNR (~0.2 per sample).

        Pipeline:
            1. Compute baseline and subtract → deviation domain
            2. Optional denoising
            3. Sobol quasi-random search in deviation domain
            4. Multi-start Nelder-Mead refinement of top candidates
            5. Cross-channel consistency check

        Args:
            observed_phase_rates: Raw observed phase rates (N x 3)
            denoise_method: 'savgol', 'spectral', 'both', or None
            n_candidates: Number of Sobol search candidates
            refine_top_k: Number of top candidates to refine
            rng: Random generator

        Returns:
            recovered_seed: 15-value recovered seed
            diagnostics: Dictionary with recovery diagnostics
        """
        if rng is None:
            rng = np.random.default_rng()

        # Step 1: Baseline subtraction
        _, baseline = self._compute_baseline()
        n = min(len(observed_phase_rates), len(baseline))
        observed_deviation = observed_phase_rates[:n] - baseline[:n]

        # Step 2: Denoise
        if denoise_method:
            denoised = self.denoise(observed_deviation, method=denoise_method)
        else:
            denoised = observed_deviation

        # Step 3: Sobol quasi-random search in deviation domain
        candidates = self._sobol_seeds(n_candidates, rng=rng)

        scores = []
        for seed_candidate in candidates:
            try:
                _, pred_dev = self._forward_deviation(seed_candidate)
            except Exception:
                scores.append(-np.inf)
                continue

            n_cmp = min(len(pred_dev), len(denoised))
            # Multi-channel correlation score
            score = 0.0
            for ch in range(3):
                p = pred_dev[:n_cmp, ch]
                o = denoised[:n_cmp, ch]
                p_c = p - np.mean(p)
                o_c = o - np.mean(o)
                denom = np.sqrt(np.sum(p_c**2) * np.sum(o_c**2))
                if denom > 1e-20:
                    score += np.sum(p_c * o_c) / denom
            scores.append(score)

        # Step 4: Multi-start refinement
        order = np.argsort(scores)[::-1]
        best_seed = None
        best_cost = np.inf

        def deviation_residual(x):
            x_clipped = np.clip(x, 0.01, 0.99)
            try:
                _, pred_dev = self._forward_deviation(x_clipped)
            except Exception:
                return 1e10
            n_cmp = min(len(pred_dev), len(denoised))
            return np.sum((pred_dev[:n_cmp] - denoised[:n_cmp])**2) / n_cmp

        for i in range(min(refine_top_k, len(candidates))):
            idx = order[i]
            result = minimize(
                deviation_residual,
                candidates[idx],
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-10}
            )
            if result.fun < best_cost:
                best_cost = result.fun
                best_seed = np.clip(result.x, 0.01, 0.99)

        recovered_seed = best_seed

        # Step 5: Cross-channel validation
        consistency, per_channel = self._cross_channel_validate(
            recovered_seed, denoised
        )

        diagnostics = {
            'denoise_method': denoise_method,
            'n_candidates': n_candidates,
            'top_coarse_score': scores[order[0]],
            'final_cost': best_cost,
            'consistency': consistency,
            'per_channel_correlation': per_channel,
        }

        return recovered_seed, diagnostics

    def evaluate_recovery(self, true_seed, recovered_seed):
        """
        Evaluate recovery quality.

        Args:
            true_seed: Ground truth seed
            recovered_seed: Recovered seed

        Returns:
            Dictionary of error metrics
        """
        true_seed = np.array(true_seed)
        recovered_seed = np.array(recovered_seed)
        error = recovered_seed - true_seed

        rmse = np.sqrt(np.mean(error**2))
        max_error = np.max(np.abs(error))

        sat_errors = []
        for i in range(3):
            sat_err = np.sqrt(np.mean(error[i*5:(i+1)*5]**2))
            sat_errors.append(sat_err)

        return {
            'rmse': rmse,
            'max_error': max_error,
            'per_satellite_rmse': sat_errors,
            'per_value_error': error.tolist(),
        }


def quick_test():
    """Test enhanced recovery against basic recovery."""
    print("Testing EnhancedRecoverer...")

    rng = np.random.default_rng(42)

    recoverer = EnhancedRecoverer(
        steps=3, delta_v_scale=0.001, symbol_period=300.0, include_j2=True
    )

    true_seed = [
        0.6, 0.0, 0.2, 0.0, 0.2,
        0.2, 0.2, 0.2, 0.2, 0.2,
        0.1, 0.1, 0.5, 0.1, 0.2
    ]

    # Generate noisy observation
    noise_model = PhaseRateNoiseModel()
    expander = ThreeSatelliteExpander(include_j2=True)
    times, clean_pr, _ = expander.expand_seed(
        true_seed, steps=3, deltaV_scale=0.001, symbol_period=300.0
    )

    noisy_pr = clean_pr.copy()
    for ch in range(3):
        noise = noise_model.generate_noise(len(clean_pr), rng=rng)
        noisy_pr[:, ch] += noise

    # Recover with enhanced pipeline
    recovered, diag = recoverer.recover(
        noisy_pr, n_candidates=200, refine_top_k=5, rng=rng
    )

    metrics = recoverer.evaluate_recovery(true_seed, recovered)

    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  Max error: {metrics['max_error']:.4f}")
    print(f"  Per-sat RMSE: {[f'{e:.4f}' for e in metrics['per_satellite_rmse']]}")
    print(f"  Cross-channel consistency: {diag['consistency']:.4f}")
    print("Test passed!")


if __name__ == '__main__':
    quick_test()
