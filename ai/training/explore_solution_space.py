"""
Explore Solution Space - Interactive parameter sweep and discovery tool.

Maps the PHYCOM system's parameter space to help AI agents and researchers
build intuition about system behavior, boundaries, and trade-offs.

Usage:
    python ai/training/explore_solution_space.py                  # Full exploration
    python ai/training/explore_solution_space.py --list-exercises  # List exercises
    python ai/training/explore_solution_space.py --dimension seed  # Explore seed space
    python ai/training/explore_solution_space.py --dimension noise # Explore noise space
    python ai/training/explore_solution_space.py --dimension detection  # Detection space
    python ai/training/explore_solution_space.py --summary         # Quick summary only
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class SolutionSpaceExplorer:
    """Maps the parameter space of the PHYCOM orbital communication system.

    Explores how seed values, noise levels, detection thresholds, and
    constellation geometry interact to define the feasible operating region.
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = {}
        self._load_constants()

    def _load_constants(self):
        """Load physics constants from the core module."""
        try:
            from core.physics_constants import MU_EARTH, R_EARTH, J2_EARTH
            self.mu = MU_EARTH
            self.r_earth = R_EARTH
            self.j2 = J2_EARTH
            if self.verbose:
                print("[OK] Loaded physics constants from core module")
        except ImportError:
            # Fallback constants if core module unavailable
            self.mu = 3.986004418e14
            self.r_earth = 6.371e6
            self.j2 = 1.08263e-3
            if self.verbose:
                print("[FALLBACK] Using built-in physics constants")

    def explore_seed_space(self, num_samples=20):
        """Explore how seed variations affect orbital schedule outputs.

        Args:
            num_samples: Number of random seeds to evaluate.

        Returns:
            Dict with seed space analysis results.
        """
        if self.verbose:
            print("\n=== SEED SPACE EXPLORATION ===")
            print(f"Sampling {num_samples} random seeds (15 values each)...\n")

        rng = np.random.default_rng(42)
        seeds = rng.random((num_samples, 15))

        # Analyze seed structure: 5 values per satellite x 3 satellites
        # [prograde, retrograde, outward, inward, north] per satellite
        analysis = {
            "num_seeds": num_samples,
            "seed_dimension": 15,
            "satellites": 3,
            "values_per_satellite": 5,
            "components": ["prograde", "retrograde", "outward", "inward", "north"],
            "value_range": [0.0, 1.0],
        }

        # Measure seed sensitivity: how much does output change per unit seed change?
        base_seed = np.full(15, 0.5)
        perturbations = np.logspace(-4, -1, 10)
        sensitivity = []

        for eps in perturbations:
            perturbed = base_seed.copy()
            perturbed[0] += eps  # Perturb prograde component of satellite 1
            delta_v = self._seed_to_delta_v(perturbed) - self._seed_to_delta_v(base_seed)
            sensitivity.append({
                "perturbation": float(eps),
                "delta_v_change_m_s": float(np.linalg.norm(delta_v)),
            })

        analysis["sensitivity"] = sensitivity

        # Map delta-V distribution across random seeds
        delta_vs = []
        for seed in seeds:
            dv = self._seed_to_delta_v(seed)
            delta_vs.append(float(np.linalg.norm(dv)))

        analysis["delta_v_stats"] = {
            "min_m_s": float(np.min(delta_vs)),
            "max_m_s": float(np.max(delta_vs)),
            "mean_m_s": float(np.mean(delta_vs)),
            "std_m_s": float(np.std(delta_vs)),
        }

        if self.verbose:
            print(f"  Seed dimension: {analysis['seed_dimension']} values")
            print(f"  Per satellite:  {analysis['values_per_satellite']} components")
            stats = analysis["delta_v_stats"]
            print(f"  Delta-V range:  {stats['min_m_s']*1000:.3f} - {stats['max_m_s']*1000:.3f} mm/s")
            print(f"  Delta-V mean:   {stats['mean_m_s']*1000:.3f} mm/s")
            print(f"\n  Sensitivity (prograde perturbation):")
            for s in sensitivity[:5]:
                print(f"    eps={s['perturbation']:.4f} -> dV change={s['delta_v_change_m_s']*1000:.4f} mm/s")

        self.results["seed_space"] = analysis
        return analysis

    def explore_noise_space(self, num_levels=15):
        """Explore how noise levels affect seed recovery accuracy.

        Args:
            num_levels: Number of noise levels to test.

        Returns:
            Dict with noise space analysis results.
        """
        if self.verbose:
            print("\n=== NOISE SPACE EXPLORATION ===")
            print(f"Sweeping {num_levels} noise levels...\n")

        noise_amplitudes = np.logspace(-5, -2, num_levels)
        rng = np.random.default_rng(123)

        # Simulate seed -> delta-V -> noisy observation -> recovery error
        true_seed = rng.random(15)
        true_dv = self._seed_to_delta_v(true_seed)
        true_dv_mag = float(np.linalg.norm(true_dv))

        recovery_analysis = []
        for amp in noise_amplitudes:
            # Add Gaussian noise to delta-V observation
            noise = rng.normal(0, amp, size=true_dv.shape)
            noisy_dv = true_dv + noise
            snr = true_dv_mag / amp if amp > 0 else float("inf")

            # Recovery error (simplified: measure observation distortion)
            error = float(np.linalg.norm(noisy_dv - true_dv) / true_dv_mag)

            recovery_analysis.append({
                "noise_amplitude_m_s": float(amp),
                "snr": float(snr),
                "relative_error": error,
                "recoverable": error < 0.1,
            })

        analysis = {
            "true_delta_v_m_s": true_dv_mag,
            "num_levels": num_levels,
            "noise_range_m_s": [float(noise_amplitudes[0]), float(noise_amplitudes[-1])],
            "recovery_results": recovery_analysis,
            "recovery_threshold_snr": 10.0,
        }

        # Find crossover point
        recoverable_count = sum(1 for r in recovery_analysis if r["recoverable"])
        analysis["recoverable_fraction"] = recoverable_count / len(recovery_analysis)

        if self.verbose:
            print(f"  True delta-V: {true_dv_mag*1000:.3f} mm/s")
            print(f"  Noise range:  {noise_amplitudes[0]*1000:.4f} - {noise_amplitudes[-1]*1000:.2f} mm/s")
            print(f"  Recoverable:  {recoverable_count}/{len(recovery_analysis)} levels")
            print(f"\n  SNR vs Recovery:")
            for r in recovery_analysis[::3]:
                status = "OK" if r["recoverable"] else "FAIL"
                print(f"    SNR={r['snr']:.1f}  error={r['relative_error']:.3f}  [{status}]")

        self.results["noise_space"] = analysis
        return analysis

    def explore_detection_space(self, num_points=20):
        """Explore detection probability vs delta-V magnitude.

        Args:
            num_points: Number of delta-V values to test.

        Returns:
            Dict with detection space analysis results.
        """
        if self.verbose:
            print("\n=== DETECTION SPACE EXPLORATION ===")
            print(f"Sweeping {num_points} delta-V magnitudes...\n")

        delta_vs = np.linspace(0.0001, 0.005, num_points)  # 0.1 to 5.0 mm/s
        rng = np.random.default_rng(456)

        # Matched filter detection model
        # Detection probability based on SNR from delta-V magnitude
        noise_floor = 0.0003  # 0.3 mm/s background noise
        num_trials = 200

        detection_results = []
        for dv in delta_vs:
            snr = dv / noise_floor
            # Simplified matched filter: Pd ≈ Q(Q^-1(Pfa) - sqrt(2*SNR))
            # Approximate with sigmoid for demonstration
            pd = 1.0 / (1.0 + np.exp(-3.0 * (snr - 2.0)))

            # Monte Carlo verification
            detections = 0
            for _ in range(num_trials):
                signal = dv + rng.normal(0, noise_floor)
                if signal > 2.0 * noise_floor:  # Simple threshold
                    detections += 1
            mc_pd = detections / num_trials

            detection_results.append({
                "delta_v_m_s": float(dv),
                "delta_v_mm_s": float(dv * 1000),
                "snr": float(snr),
                "pd_analytical": float(pd),
                "pd_monte_carlo": float(mc_pd),
                "stealth": float(dv) < 0.0005,
            })

        analysis = {
            "num_points": num_points,
            "delta_v_range_mm_s": [float(delta_vs[0] * 1000), float(delta_vs[-1] * 1000)],
            "noise_floor_mm_s": noise_floor * 1000,
            "detection_results": detection_results,
        }

        # Find stealth boundary
        stealth_boundary = None
        for r in detection_results:
            if r["pd_monte_carlo"] > 0.5:
                stealth_boundary = r["delta_v_mm_s"]
                break
        analysis["stealth_boundary_mm_s"] = stealth_boundary

        if self.verbose:
            print(f"  Noise floor:      {noise_floor*1000:.1f} mm/s")
            print(f"  Delta-V range:    {delta_vs[0]*1000:.1f} - {delta_vs[-1]*1000:.1f} mm/s")
            if stealth_boundary:
                print(f"  Stealth boundary: {stealth_boundary:.2f} mm/s (Pd > 0.5)")
            print(f"\n  Delta-V (mm/s) | SNR   | Pd (analytical) | Pd (MC)")
            print(f"  {'-'*55}")
            for r in detection_results[::4]:
                print(f"  {r['delta_v_mm_s']:14.2f} | {r['snr']:5.1f} | {r['pd_analytical']:15.3f} | {r['pd_monte_carlo']:.3f}")

        self.results["detection_space"] = analysis
        return analysis

    def explore_constellation_space(self):
        """Explore how constellation parameters affect system performance.

        Returns:
            Dict with constellation configuration analysis.
        """
        if self.verbose:
            print("\n=== CONSTELLATION SPACE EXPLORATION ===\n")

        altitude_ref = 500e3  # 500 km reference altitude
        period_ratios = [
            {"name": "Prime (2:3:5)", "ratios": [2, 3, 5], "type": "prime"},
            {"name": "Sequential (1:2:3)", "ratios": [1, 2, 3], "type": "mixed"},
            {"name": "Fibonacci (2:3:5)", "ratios": [2, 3, 5], "type": "fibonacci"},
            {"name": "Power (1:2:4)", "ratios": [1, 2, 4], "type": "power"},
            {"name": "Co-prime (3:5:7)", "ratios": [3, 5, 7], "type": "prime"},
            {"name": "Non-prime (2:4:6)", "ratios": [2, 4, 6], "type": "composite"},
        ]

        constellation_results = []
        for config in period_ratios:
            ratios = config["ratios"]

            # Channel independence: GCD of all pairs
            from math import gcd
            pair_gcds = [
                gcd(ratios[0], ratios[1]),
                gcd(ratios[1], ratios[2]),
                gcd(ratios[0], ratios[2]),
            ]
            max_gcd = max(pair_gcds)
            independence = 1.0 / max_gcd  # Higher = more independent

            # Repeat period: LCM of all ratios
            from math import lcm
            repeat = lcm(ratios[0], lcm(ratios[1], ratios[2]))

            # Number of unique phase combinations before repeat
            unique_phases = repeat

            constellation_results.append({
                "name": config["name"],
                "ratios": ratios,
                "type": config["type"],
                "channel_independence": float(independence),
                "repeat_period_symbols": repeat,
                "unique_phase_combinations": unique_phases,
                "pair_gcds": pair_gcds,
                "recommended": independence == 1.0,
            })

        analysis = {
            "altitude_ref_km": altitude_ref / 1000,
            "num_configurations": len(period_ratios),
            "configurations": constellation_results,
        }

        if self.verbose:
            print(f"  Reference altitude: {altitude_ref/1000:.0f} km")
            print(f"\n  {'Config':<20} | {'Independence':>12} | {'Repeat':>8} | {'Rec':>3}")
            print(f"  {'-'*52}")
            for c in constellation_results:
                rec = "YES" if c["recommended"] else " no"
                print(f"  {c['name']:<20} | {c['channel_independence']:>12.2f} | {c['repeat_period_symbols']:>8} | {rec}")

        self.results["constellation_space"] = analysis
        return analysis

    def full_exploration(self):
        """Run all explorations and produce a comprehensive summary.

        Returns:
            Dict with all exploration results.
        """
        if self.verbose:
            print("=" * 60)
            print("  ORBITAL PHYCOM - SOLUTION SPACE EXPLORATION")
            print("=" * 60)

        self.explore_seed_space()
        self.explore_noise_space()
        self.explore_detection_space()
        self.explore_constellation_space()

        if self.verbose:
            self._print_summary()

        return self.results

    def _print_summary(self):
        """Print a concise summary of all explorations."""
        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)

        if "seed_space" in self.results:
            stats = self.results["seed_space"]["delta_v_stats"]
            print(f"\n  Seed Space:")
            print(f"    15-dimensional seed -> delta-V range: {stats['min_m_s']*1000:.2f}-{stats['max_m_s']*1000:.2f} mm/s")

        if "noise_space" in self.results:
            frac = self.results["noise_space"]["recoverable_fraction"]
            print(f"\n  Noise Space:")
            print(f"    Recoverable at {frac*100:.0f}% of tested noise levels")

        if "detection_space" in self.results:
            boundary = self.results["detection_space"].get("stealth_boundary_mm_s")
            if boundary:
                print(f"\n  Detection Space:")
                print(f"    Stealth boundary: {boundary:.2f} mm/s (below = undetectable)")

        if "constellation_space" in self.results:
            configs = self.results["constellation_space"]["configurations"]
            recommended = [c["name"] for c in configs if c["recommended"]]
            print(f"\n  Constellation Space:")
            print(f"    Recommended configs: {', '.join(recommended)}")

        print(f"\n  Feasible operating region:")
        print(f"    Delta-V:  0.5 - 2.0 mm/s (stealth + detectable by partners)")
        print(f"    SNR:      > 10 for reliable recovery")
        print(f"    Ratios:   Use co-prime period ratios for channel independence")
        print(f"    Capacity: 0.011 bps raw, 0.117 bps effective (post-expansion)")
        print()

    def _seed_to_delta_v(self, seed):
        """Convert a 15-value seed to delta-V impulse vector (simplified model).

        Args:
            seed: Array of 15 float values in [0, 1].

        Returns:
            Array of delta-V components (m/s).
        """
        seed = np.asarray(seed)
        dv_scale = 0.002  # 2 mm/s max delta-V per component

        # Each satellite: [prograde, retrograde, outward, inward, north]
        # Net delta-V per satellite = (prograde - retrograde, outward - inward, north - south)
        # where south = 1 - sum(others) normalized
        dvs = []
        for i in range(3):
            s = seed[i * 5:(i + 1) * 5]
            net_prograde = (s[0] - s[1]) * dv_scale
            net_radial = (s[2] - s[3]) * dv_scale
            net_normal = s[4] * dv_scale * 0.5
            dvs.extend([net_prograde, net_radial, net_normal])

        return np.array(dvs)

    def save_results(self, path=None):
        """Save exploration results to JSON.

        Args:
            path: Output file path. Defaults to ai/training/exploration_results.json.
        """
        if path is None:
            path = Path(__file__).parent / "exploration_results.json"

        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)

        if self.verbose:
            print(f"Results saved to {path}")


def list_exercises():
    """Print available training exercises."""
    exercises_path = Path(__file__).parent / "exercises.json"
    with open(exercises_path) as f:
        data = json.load(f)

    print("\n  AVAILABLE TRAINING EXERCISES")
    print("  " + "=" * 50)
    for ex in data["exercises"]:
        print(f"\n  [{ex['id']}] {ex['title']}")
        print(f"       Difficulty: {ex['difficulty']}")
        print(f"       {ex['description']}")
        print(f"       Files: {', '.join(ex['key_files'][:3])}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore PHYCOM solution space")
    parser.add_argument("--dimension", choices=["seed", "noise", "detection", "constellation"],
                        help="Explore a specific dimension only")
    parser.add_argument("--list-exercises", action="store_true",
                        help="List available training exercises")
    parser.add_argument("--summary", action="store_true",
                        help="Quick summary mode (fewer samples)")
    parser.add_argument("--save", action="store_true",
                        help="Save results to JSON")
    args = parser.parse_args()

    if args.list_exercises:
        list_exercises()
        sys.exit(0)

    explorer = SolutionSpaceExplorer(verbose=True)

    if args.dimension:
        method = getattr(explorer, f"explore_{args.dimension}_space")
        method()
    else:
        explorer.full_exploration()

    if args.save:
        explorer.save_results()
