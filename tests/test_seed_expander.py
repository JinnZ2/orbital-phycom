"""Tests for core seed expander module."""

import numpy as np
import pytest

from core.seed_expander import ThreeSatelliteExpander


class TestThreeSatelliteExpander:
    """Tests for ThreeSatelliteExpander class."""

    def test_initialization(self):
        exp = ThreeSatelliteExpander(harmonics=(2, 3, 5))
        assert len(exp.periods) == 3
        assert len(exp.semi_major) == 3
        assert len(exp.initial_states) == 3

    def test_seed_to_deltaVs_shape(self):
        exp = ThreeSatelliteExpander()
        seed = [0.2] * 15
        matrices, mags = exp.seed_to_deltaVs(seed)
        assert len(matrices) == 3
        assert len(mags) == 3
        for m in matrices:
            assert m.shape == (6, 3)

    def test_seed_to_deltaVs_wrong_length(self):
        exp = ThreeSatelliteExpander()
        with pytest.raises(ValueError, match="15 values"):
            exp.seed_to_deltaVs([0.2] * 10)

    def test_seed_to_deltaVs_scale(self):
        exp = ThreeSatelliteExpander()
        seed = [0.2] * 15
        _, mags_small = exp.seed_to_deltaVs(seed, deltaV_scale=0.001)
        _, mags_large = exp.seed_to_deltaVs(seed, deltaV_scale=0.002)
        # Larger scale = larger magnitudes
        assert np.sum(mags_large[0]) > np.sum(mags_small[0])

    def test_apply_deltaV_to_state(self):
        exp = ThreeSatelliteExpander()
        state = exp.initial_states[0].copy()
        dv = np.array([0.001, 0.0, 0.0])  # Prograde only
        new_state = exp.apply_deltaV_to_state(state, dv)
        # Position unchanged
        assert np.allclose(new_state[:3], state[:3])
        # Velocity changed by ~0.001 m/s
        velocity_change = np.linalg.norm(new_state[3:6] - state[3:6])
        assert velocity_change > 1e-6  # dV was applied
        assert abs(velocity_change - 0.001) < 1e-6  # magnitude matches

    def test_compute_phase_rates(self):
        exp = ThreeSatelliteExpander()
        states = [s.copy() for s in exp.initial_states]
        pr, ranges = exp.compute_phase_rates(states)
        assert pr.shape == (3,)
        assert ranges.shape == (3,)
        assert np.all(np.isfinite(pr))
        assert np.all(ranges > 0)

    def test_expand_seed_deterministic(self):
        """Same seed always produces same output."""
        exp = ThreeSatelliteExpander()
        seed = [0.6, 0.0, 0.2, 0.0, 0.2,
                0.2, 0.2, 0.2, 0.2, 0.2,
                0.1, 0.1, 0.5, 0.1, 0.2]

        times1, pr1, _ = exp.expand_seed(seed, steps=2, symbol_period=100.0)
        exp2 = ThreeSatelliteExpander()
        times2, pr2, _ = exp2.expand_seed(seed, steps=2, symbol_period=100.0)

        assert np.allclose(times1, times2)
        assert np.allclose(pr1, pr2)

    def test_expand_seed_different_seeds_differ(self):
        exp = ThreeSatelliteExpander()
        seed_a = [0.6, 0.0, 0.2, 0.0, 0.2,
                  0.2, 0.2, 0.2, 0.2, 0.2,
                  0.1, 0.1, 0.5, 0.1, 0.2]
        seed_b = [0.1, 0.5, 0.2, 0.1, 0.1,
                  0.3, 0.1, 0.3, 0.1, 0.2,
                  0.4, 0.0, 0.2, 0.2, 0.2]

        _, pr_a, _ = exp.expand_seed(seed_a, steps=2, symbol_period=100.0)
        _, pr_b, _ = exp.expand_seed(seed_b, steps=2, symbol_period=100.0)

        assert not np.allclose(pr_a, pr_b)

    def test_expand_seed_output_shape(self):
        exp = ThreeSatelliteExpander()
        seed = [0.2] * 15
        times, pr, history = exp.expand_seed(seed, steps=3, symbol_period=100.0)
        assert pr.shape[1] == 3  # 3 satellite pairs
        assert len(times) == len(pr)
        assert len(history) == 3  # 3 steps
