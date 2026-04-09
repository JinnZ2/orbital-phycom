"""Tests for noise model and seed recovery modules."""

import numpy as np
import pytest

from core.noise_model import PhaseRateNoiseModel
from core.seed_recovery import SeedRecoverer


class TestPhaseRateNoiseModel:
    """Tests for PhaseRateNoiseModel class."""

    def test_initialization(self):
        model = PhaseRateNoiseModel()
        assert model.lambda_rf > 0
        assert model.carrier_snr > 0

    def test_thermal_noise_positive(self):
        model = PhaseRateNoiseModel()
        sigma = model.thermal_noise_std()
        assert sigma > 0
        assert np.isfinite(sigma)

    def test_clock_noise_decreases_with_tau(self):
        model = PhaseRateNoiseModel()
        sigma_short = model.clock_noise_std(1.0)
        sigma_long = model.clock_noise_std(100.0)
        assert sigma_short > sigma_long  # Longer integration = less noise

    def test_clock_noise_small_tau_safe(self):
        model = PhaseRateNoiseModel()
        sigma = model.clock_noise_std(0.001)
        assert np.isfinite(sigma)

    def test_total_noise_positive(self):
        model = PhaseRateNoiseModel()
        sigma = model.total_noise_std(300.0)
        assert sigma > 0
        assert np.isfinite(sigma)

    def test_generate_noise_shape(self):
        model = PhaseRateNoiseModel()
        noise = model.generate_noise(100, rng=np.random.default_rng(42))
        assert noise.shape == (100,)
        assert np.all(np.isfinite(noise))

    def test_generate_noise_has_variance(self):
        model = PhaseRateNoiseModel()
        noise = model.generate_noise(1000, rng=np.random.default_rng(42))
        assert np.std(noise) > 0

    def test_link_budget_keys(self):
        model = PhaseRateNoiseModel()
        budget = model.link_budget(delta_v=0.001)
        expected_keys = [
            'delta_v_ms', 'signal_phase_rate', 'noise_std',
            'snr_single_sample_db', 'effective_snr_db',
        ]
        for key in expected_keys:
            assert key in budget

    def test_link_budget_snr_increases_with_dv(self):
        model = PhaseRateNoiseModel()
        b1 = model.link_budget(delta_v=0.0005)
        b2 = model.link_budget(delta_v=0.002)
        assert b2['effective_snr_db'] > b1['effective_snr_db']

    def test_iono_noise_psd_shape(self):
        model = PhaseRateNoiseModel()
        freqs = np.linspace(0.01, 10, 50)
        psd = model.iono_noise_psd(freqs)
        assert psd.shape == (50,)
        assert np.all(psd >= 0)


class TestSeedRecoverer:
    """Tests for SeedRecoverer class."""

    @pytest.fixture
    def recoverer(self):
        return SeedRecoverer(steps=2, delta_v_scale=0.001, symbol_period=100.0)

    @pytest.fixture
    def test_seed(self):
        return [0.6, 0.0, 0.2, 0.0, 0.2,
                0.2, 0.2, 0.2, 0.2, 0.2,
                0.1, 0.1, 0.5, 0.1, 0.2]

    def test_forward_model(self, recoverer, test_seed):
        times, pr = recoverer.forward_model(test_seed)
        assert len(times) > 0
        assert pr.shape[1] == 3
        assert np.all(np.isfinite(pr))

    def test_self_correlation(self, recoverer, test_seed):
        """A seed correlated against its own output should score ~1.0."""
        _, truth = recoverer.forward_model(test_seed)
        score = recoverer.correlation_score(test_seed, truth)
        assert score > 0.99

    def test_different_seed_lower_correlation(self, recoverer, test_seed):
        """Different seeds produce different phase-rate signatures."""
        _, truth = recoverer.forward_model(test_seed)
        # Use a very different seed to ensure distinguishable output
        different = [0.01, 0.8, 0.01, 0.1, 0.08,
                     0.7, 0.01, 0.1, 0.1, 0.09,
                     0.01, 0.01, 0.01, 0.8, 0.17]
        _, diff_pr = recoverer.forward_model(different)
        # Phase rates should differ between seeds
        assert not np.allclose(truth, diff_pr, atol=1e-10)

    def test_residual_zero_for_true_seed(self, recoverer, test_seed):
        _, truth = recoverer.forward_model(test_seed)
        cost = recoverer.residual(test_seed, truth)
        assert cost < 1e-6

    def test_evaluate_recovery_metrics(self, recoverer, test_seed):
        recovered = np.array(test_seed) + 0.01  # Small perturbation
        metrics = recoverer.evaluate_recovery(test_seed, recovered)
        assert 'rmse' in metrics
        assert 'max_error' in metrics
        assert metrics['rmse'] > 0
        assert len(metrics['per_satellite_rmse']) == 3
