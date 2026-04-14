"""Tests for core orbital dynamics module."""

import numpy as np
import pytest

from core.orbital_dynamics import OrbitalSimulator


class TestOrbitalSimulator:
    """Tests for OrbitalSimulator class."""

    def test_initialization(self):
        sim = OrbitalSimulator(harmonic_A=2, harmonic_B=3)
        assert sim.mu > 0
        assert sim.aA > 0
        assert sim.aB > 0
        assert sim.aA != sim.aB  # Different harmonics = different orbits

    def test_initial_state_shape(self):
        sim = OrbitalSimulator()
        assert sim.state0.shape == (12,)

    def test_gravity_returns_finite(self):
        sim = OrbitalSimulator()
        r = np.array([7e6, 0, 0])
        acc = sim._gravity(r)
        assert acc.shape == (3,)
        assert np.all(np.isfinite(acc))
        assert np.linalg.norm(acc) > 0

    def test_gravity_zero_radius_safe(self):
        """Division by zero should not crash."""
        sim = OrbitalSimulator()
        acc = sim._gravity(np.array([0.0, 0.0, 0.0]))
        assert np.all(np.isfinite(acc))
        assert np.allclose(acc, 0.0)

    def test_j2_acceleration_zero_radius_safe(self):
        sim = OrbitalSimulator()
        acc = sim._j2_acceleration(np.array([0.0, 0.0, 0.0]))
        assert np.all(np.isfinite(acc))

    def test_apply_deltaV_prograde(self):
        sim = OrbitalSimulator()
        state = sim.state0.copy()
        v_before = np.linalg.norm(state[3:6])

        new_state = sim.apply_deltaV(state, 0.001, satellite='A')
        v_after = np.linalg.norm(new_state[3:6])

        # Prograde burn increases speed
        assert v_after > v_before
        # Satellite B unchanged
        assert np.allclose(new_state[6:12], state[6:12])

    def test_apply_deltaV_zero_velocity_safe(self):
        sim = OrbitalSimulator()
        state = sim.state0.copy()
        state[3:6] = 0.0  # Zero velocity for satellite A
        new_state = sim.apply_deltaV(state, 0.001, satellite='A')
        # Should not crash, velocity stays zero
        assert np.allclose(new_state[3:6], 0.0)

    def test_compute_phase_rate_finite(self):
        sim = OrbitalSimulator()
        pr, r = sim.compute_phase_rate(sim.state0)
        assert np.isfinite(pr)
        assert np.isfinite(r)
        assert r > 0

    def test_compute_phase_rate_coincident_safe(self):
        """Satellites at same position should not crash."""
        sim = OrbitalSimulator()
        state = sim.state0.copy()
        state[6:9] = state[0:3]  # Put B at same position as A
        pr, r = sim.compute_phase_rate(state)
        assert np.isfinite(pr)
        assert pr == 0.0

    def test_simulate_no_impulses(self):
        sim = OrbitalSimulator()
        times, states, phase_rates, ranges, log = sim.simulate(
            duration=300.0, n_output=100
        )
        assert len(times) > 0
        assert len(phase_rates) == len(times)
        assert len(log) == 0
        assert np.all(np.isfinite(phase_rates))

    def test_simulate_with_impulse(self):
        sim = OrbitalSimulator()
        impulses = [('A', 150.0, 0.001)]
        times, states, phase_rates, ranges, log = sim.simulate(
            duration=300.0, impulse_events=impulses, n_output=100
        )
        assert len(log) == 1
        assert log[0] == ('A', 150.0, 0.001)

    def test_simulate_deterministic(self):
        """Same inputs should produce identical results."""
        sim = OrbitalSimulator()
        impulses = [('A', 100.0, 0.001)]

        _, _, pr1, _, _ = sim.simulate(300.0, impulses, n_output=50)
        sim2 = OrbitalSimulator()
        _, _, pr2, _, _ = sim2.simulate(300.0, impulses, n_output=50)

        assert np.allclose(pr1, pr2)

    def test_j2_disabled(self):
        sim_j2 = OrbitalSimulator(include_j2=True)
        sim_no_j2 = OrbitalSimulator(include_j2=False)

        _, _, pr_j2, _, _ = sim_j2.simulate(300.0, n_output=50)
        _, _, pr_no, _, _ = sim_no_j2.simulate(300.0, n_output=50)

        # J2 should produce different results
        assert not np.allclose(pr_j2, pr_no)
