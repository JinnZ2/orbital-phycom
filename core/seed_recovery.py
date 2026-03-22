"""
Seed recovery from noisy phase-rate observations.

Solves the inverse problem: given observed phase-rate time series
(possibly corrupted by noise), recover the 15-value seed that
generated them.

Uses a two-stage approach:
    1. Coarse search via matched-filter correlation against a seed library
    2. Fine refinement via gradient-free optimization (Nelder-Mead)
"""

import numpy as np
from scipy.optimize import minimize
from .seed_expander import ThreeSatelliteExpander
from .noise_model import PhaseRateNoiseModel


class SeedRecoverer:
    """
    Recovers seeds from noisy phase-rate observations.

    The forward model (seed → phase rates) is deterministic, so
    recovery is a nonlinear least-squares problem: find the seed
    that minimizes the residual between predicted and observed
    phase rates.
    """

    def __init__(self, harmonics=(2, 3, 5), include_j2=True,
                 steps=5, delta_v_scale=0.001, symbol_period=300.0):
        """
        Initialize recovery engine.

        Args:
            harmonics: Constellation harmonic numbers
            include_j2: Use J2 in forward model
            steps: Number of maneuver steps to simulate
            delta_v_scale: ΔV scale for forward model (m/s)
            symbol_period: Symbol period in seconds
        """
        self.expander = ThreeSatelliteExpander(
            harmonics=harmonics, include_j2=include_j2
        )
        self.steps = steps
        self.delta_v_scale = delta_v_scale
        self.symbol_period = symbol_period

    def forward_model(self, seed):
        """
        Run forward model: seed → phase-rate time series.

        Args:
            seed: 15-value seed array

        Returns:
            times: Time array
            phase_rates: Phase-rate array (N x 3)
        """
        times, phase_rates, _ = self.expander.expand_seed(
            seed,
            steps=self.steps,
            deltaV_scale=self.delta_v_scale,
            symbol_period=self.symbol_period
        )
        return times, phase_rates

    def generate_observation(self, seed, noise_model=None, rng=None):
        """
        Generate synthetic noisy observation from a known seed.

        Args:
            seed: True 15-value seed
            noise_model: PhaseRateNoiseModel instance (or None for default)
            rng: Random generator

        Returns:
            times: Time array
            observed: Noisy phase-rate array (N x 3)
            truth: Clean phase-rate array (N x 3)
        """
        if noise_model is None:
            noise_model = PhaseRateNoiseModel()

        times, truth = self.forward_model(seed)
        n_samples = len(times)

        # Add independent noise to each channel
        observed = truth.copy()
        for ch in range(3):
            noise = noise_model.generate_noise(
                n_samples,
                integration_time=self.symbol_period,
                rng=rng
            )
            observed[:, ch] += noise

        return times, observed, truth

    def residual(self, seed_candidate, observed_phase_rates,
                 observed_times=None):
        """
        Compute weighted residual between candidate and observation.

        Args:
            seed_candidate: 15-value candidate seed
            observed_phase_rates: Observed phase-rate array (N x 3)
            observed_times: Time array (unused, for interface compat)

        Returns:
            Scalar cost (sum of squared residuals)
        """
        # Clip seed to valid range
        seed_clipped = np.clip(seed_candidate, 0.01, 0.99)

        try:
            _, predicted = self.forward_model(seed_clipped)
        except Exception:
            return 1e20

        # Ensure same length
        n = min(len(predicted), len(observed_phase_rates))
        pred = predicted[:n]
        obs = observed_phase_rates[:n]

        # Normalized residual (scale-invariant across channels)
        residuals = pred - obs
        cost = 0.0
        for ch in range(3):
            channel_std = np.std(obs[:, ch])
            if channel_std > 0:
                cost += np.sum((residuals[:, ch] / channel_std) ** 2)
            else:
                cost += np.sum(residuals[:, ch] ** 2)

        return cost / n

    def correlation_score(self, seed_candidate, observed_phase_rates):
        """
        Matched-filter correlation score between candidate and observation.

        Higher is better (1.0 = perfect match).

        Args:
            seed_candidate: 15-value candidate seed
            observed_phase_rates: Observed phase-rate array (N x 3)

        Returns:
            Average correlation across channels (0 to 1)
        """
        seed_clipped = np.clip(seed_candidate, 0.01, 0.99)

        try:
            _, predicted = self.forward_model(seed_clipped)
        except Exception:
            return 0.0

        n = min(len(predicted), len(observed_phase_rates))
        pred = predicted[:n]
        obs = observed_phase_rates[:n]

        correlations = []
        for ch in range(3):
            p = pred[:, ch] - np.mean(pred[:, ch])
            o = obs[:, ch] - np.mean(obs[:, ch])
            denom = np.sqrt(np.sum(p**2) * np.sum(o**2))
            if denom > 0:
                corr = np.abs(np.sum(p * o) / denom)
            else:
                corr = 0.0
            correlations.append(corr)

        return np.mean(correlations)

    def coarse_search(self, observed_phase_rates, n_candidates=500,
                      rng=None):
        """
        Stage 1: Coarse search over random seed candidates.

        Generates random seeds and scores them against observation
        via matched-filter correlation.

        Args:
            observed_phase_rates: Observed phase-rate array (N x 3)
            n_candidates: Number of random seeds to try
            rng: Random generator

        Returns:
            best_seed: Best seed found
            best_score: Correlation score of best seed
            top_seeds: Top 10 seeds with scores
        """
        if rng is None:
            rng = np.random.default_rng()

        scores = []
        candidates = []

        for _ in range(n_candidates):
            seed = rng.uniform(0.05, 0.5, 15)
            score = self.correlation_score(seed, observed_phase_rates)
            scores.append(score)
            candidates.append(seed)

        # Sort by score (descending)
        order = np.argsort(scores)[::-1]
        top_seeds = [(candidates[i], scores[i]) for i in order[:10]]

        return top_seeds[0][0], top_seeds[0][1], top_seeds

    def refine(self, initial_seed, observed_phase_rates, method='Nelder-Mead',
               maxiter=200):
        """
        Stage 2: Refine seed estimate via optimization.

        Uses gradient-free optimizer to minimize residual between
        forward model output and observed phase rates.

        Args:
            initial_seed: Starting seed (from coarse search)
            observed_phase_rates: Observed phase-rate array (N x 3)
            method: Optimization method
            maxiter: Maximum iterations

        Returns:
            recovered_seed: Optimized seed
            final_cost: Final residual cost
            result: Full scipy OptimizeResult
        """
        def objective(x):
            return self.residual(x, observed_phase_rates)

        result = minimize(
            objective,
            initial_seed,
            method=method,
            options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-8}
        )

        recovered = np.clip(result.x, 0.01, 0.99)
        return recovered, result.fun, result

    def recover(self, observed_phase_rates, n_candidates=200,
                refine_top_k=3, rng=None):
        """
        Full recovery pipeline: coarse search → refinement.

        Args:
            observed_phase_rates: Observed phase-rate array (N x 3)
            n_candidates: Number of coarse search candidates
            refine_top_k: Number of top candidates to refine
            rng: Random generator

        Returns:
            best_seed: Best recovered seed
            best_cost: Final cost
            correlation: Final correlation score
            diagnostics: Dictionary of recovery diagnostics
        """
        # Stage 1: Coarse search
        _, _, top_seeds = self.coarse_search(
            observed_phase_rates, n_candidates=n_candidates, rng=rng
        )

        # Stage 2: Refine top candidates
        best_seed = None
        best_cost = np.inf
        refinement_results = []

        for i in range(min(refine_top_k, len(top_seeds))):
            candidate, coarse_score = top_seeds[i]
            recovered, cost, opt_result = self.refine(
                candidate, observed_phase_rates
            )
            refinement_results.append({
                'seed': recovered,
                'cost': cost,
                'coarse_score': coarse_score,
                'converged': opt_result.success,
            })

            if cost < best_cost:
                best_cost = cost
                best_seed = recovered

        # Final correlation score
        correlation = self.correlation_score(best_seed, observed_phase_rates)

        diagnostics = {
            'n_candidates': n_candidates,
            'top_coarse_score': top_seeds[0][1],
            'refinement_results': refinement_results,
            'final_correlation': correlation,
        }

        return best_seed, best_cost, correlation, diagnostics

    def evaluate_recovery(self, true_seed, recovered_seed):
        """
        Evaluate quality of seed recovery.

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
        relative_error = rmse / (np.std(true_seed) + 1e-10)

        # Per-satellite errors
        sat_errors = []
        for i in range(3):
            sat_err = np.sqrt(np.mean(error[i*5:(i+1)*5]**2))
            sat_errors.append(sat_err)

        return {
            'rmse': rmse,
            'max_error': max_error,
            'relative_error': relative_error,
            'per_satellite_rmse': sat_errors,
            'per_value_error': error.tolist(),
        }


def quick_test():
    """Quick sanity check with known seed."""
    print("Testing SeedRecoverer...")

    # Use fewer steps for speed
    recoverer = SeedRecoverer(
        steps=3, delta_v_scale=0.001, symbol_period=300.0, include_j2=True
    )

    # Known seed
    true_seed = [
        0.6, 0.0, 0.2, 0.0, 0.2,
        0.2, 0.2, 0.2, 0.2, 0.2,
        0.1, 0.1, 0.5, 0.1, 0.2
    ]

    # Generate clean observation (no noise for basic test)
    times, truth = recoverer.forward_model(true_seed)
    print(f"  Forward model: {len(times)} samples, 3 channels")

    # Test correlation with self (should be ~1.0)
    self_corr = recoverer.correlation_score(true_seed, truth)
    print(f"  Self-correlation: {self_corr:.6f}")

    # Test with heavily perturbed seed (should be lower correlation)
    perturbed = np.array(true_seed) * 0.3 + 0.2
    perturbed = np.clip(perturbed, 0.01, 0.99)
    pert_corr = recoverer.correlation_score(perturbed, truth)
    print(f"  Perturbed correlation: {pert_corr:.6f}")

    assert self_corr > 0.99, "Self-correlation should be ~1.0"
    print("Test passed!")


if __name__ == '__main__':
    quick_test()
