"""Tests for detection analysis and physics constants modules."""

import numpy as np
import pytest

from core.physics_constants import (
    MU_EARTH, R_EARTH, ALTITUDE_REF, LAMBDA_RF,
    J2_EARTH, R_EARTH_EQUATORIAL, C
)
from core.detection import DetectionAnalyzer
from core.drag_force import get_air_density, compute_drag_acceleration


class TestPhysicsConstants:
    """Verify physics constants are reasonable values."""

    def test_mu_earth(self):
        assert 3.9e14 < MU_EARTH < 4.0e14

    def test_r_earth(self):
        assert 6.3e6 < R_EARTH < 6.4e6

    def test_altitude_ref(self):
        assert ALTITUDE_REF == 500e3

    def test_lambda_rf(self):
        assert 0 < LAMBDA_RF < 1.0

    def test_j2(self):
        assert 1e-3 < J2_EARTH < 2e-3

    def test_speed_of_light(self):
        assert abs(C - 299792458.0) < 1.0


class TestDragForce:
    """Tests for atmospheric drag module."""

    def test_air_density_sea_level(self):
        rho = get_air_density(0.0)
        assert abs(rho - 1.225) < 0.01

    def test_air_density_decreases_with_altitude(self):
        rho_low = get_air_density(100e3)
        rho_high = get_air_density(500e3)
        assert rho_low > rho_high

    def test_air_density_positive(self):
        for alt in [0, 100e3, 300e3, 500e3, 1000e3]:
            assert get_air_density(alt) > 0

    def test_drag_acceleration_direction(self):
        r = np.array([R_EARTH + 400e3, 0, 0])
        v = np.array([0, 7700, 0])
        acc = compute_drag_acceleration(r, v)
        # Drag opposes velocity
        assert np.dot(acc, v) < 0

    def test_drag_zero_velocity_safe(self):
        r = np.array([R_EARTH + 400e3, 0, 0])
        v = np.array([0.0, 0.0, 0.0])
        acc = compute_drag_acceleration(r, v)
        assert np.allclose(acc, 0.0)

    def test_drag_negative_altitude_safe(self):
        r = np.array([R_EARTH * 0.5, 0, 0])  # Below surface
        v = np.array([0, 7700, 0])
        acc = compute_drag_acceleration(r, v)
        assert np.all(np.isfinite(acc))


class TestDetectionAnalyzer:
    """Tests for DetectionAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        return DetectionAnalyzer(include_j2=True)

    @pytest.fixture
    def test_seed(self):
        return [0.6, 0.0, 0.2, 0.0, 0.2,
                0.2, 0.2, 0.2, 0.2, 0.2,
                0.1, 0.1, 0.5, 0.1, 0.2]

    def test_compute_signal_energy(self, analyzer, test_seed):
        energy, deviation, times = analyzer.compute_signal_energy(
            test_seed, steps=2, delta_v_scale=0.001, symbol_period=100.0
        )
        assert energy > 0
        assert deviation.shape[1] == 3
        assert len(times) == len(deviation)

    def test_matched_filter_performance(self, analyzer, test_seed):
        perf = analyzer.matched_filter_performance(
            test_seed, delta_v_scale=0.001, steps=2, symbol_period=100.0
        )
        assert 'matched_filter_snr_db' in perf
        assert 'signal_energy' in perf
        assert perf['signal_energy'] > 0
        assert perf['n_samples'] > 0

    def test_delta_v_sweep(self, analyzer, test_seed):
        dvs, pd_vals, snr_vals = analyzer.delta_v_sweep(
            test_seed, delta_v_range=np.array([0.0005, 0.001, 0.002]),
            steps=2, symbol_period=100.0
        )
        assert len(dvs) == 3
        assert len(pd_vals) == 3
        # Higher delta-V = higher detection probability
        assert pd_vals[-1] >= pd_vals[0]
