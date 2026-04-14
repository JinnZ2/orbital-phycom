"""
Generate training data for AI agents learning the Orbital PHYCOM system.

Produces reproducible reference datasets so that AI agents can learn from
pre-computed examples instead of running expensive simulations for every
query.

Datasets produced:
    reference_seeds.json      - Canonical seeds with descriptions
    seed_output_pairs.json    - Reference seed -> phase-rate pairs
    exercise_solutions.json   - Expected outputs for training exercises
    exploration_snapshot.json - Pre-computed solution space map

Usage:
    python ai/training/data/generate_training_data.py          # Generate all
    python ai/training/data/generate_training_data.py --check  # Verify existing
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from core.seed_expander import ThreeSatelliteExpander
from core.noise_model import PhaseRateNoiseModel
from core.detection import DetectionAnalyzer
from core.seed_recovery import SeedRecoverer


DATA_DIR = Path(__file__).parent
RNG_SEED = 42  # Deterministic generation


# -----------------------------------------------------------------------------
# Canonical seeds: hand-picked reference vectors with known characteristics
# -----------------------------------------------------------------------------

CANONICAL_SEEDS = {
    "neutral": {
        "values": [1/6] * 15,  # Uniform allocation, ~zero net delta-V
        "description": "Uniform allocation - each of 6 directions gets equal weight. Produces minimal net delta-V (drift only).",
        "expected_behavior": "Low signal energy, near-baseline phase rates",
        "use_case": "Baseline reference for detection analysis",
    },
    "prograde_heavy": {
        "values": [
            0.6, 0.0, 0.1, 0.0, 0.15,  # Sat A: mostly prograde
            0.6, 0.0, 0.1, 0.0, 0.15,  # Sat B: mostly prograde
            0.6, 0.0, 0.1, 0.0, 0.15,  # Sat C: mostly prograde
        ],
        "description": "All satellites push prograde - raises all orbits in unison.",
        "expected_behavior": "Large positive range-rate change, high detectability",
        "use_case": "High-SNR reference, easy detection test",
    },
    "retrograde_heavy": {
        "values": [
            0.0, 0.6, 0.1, 0.0, 0.15,
            0.0, 0.6, 0.1, 0.0, 0.15,
            0.0, 0.6, 0.1, 0.0, 0.15,
        ],
        "description": "All satellites push retrograde - lowers all orbits.",
        "expected_behavior": "Large negative range-rate change, mirror of prograde_heavy",
        "use_case": "Sign symmetry validation",
    },
    "differential": {
        "values": [
            0.6, 0.0, 0.1, 0.0, 0.15,  # Sat A: prograde
            0.0, 0.6, 0.1, 0.0, 0.15,  # Sat B: retrograde
            0.2, 0.2, 0.2, 0.2, 0.1,   # Sat C: balanced
        ],
        "description": "Satellites maneuver differently - creates relative phase-rate signature.",
        "expected_behavior": "Strong differential phase rates between pairs",
        "use_case": "Test multi-channel independence",
    },
    "radial_push": {
        "values": [
            0.1, 0.1, 0.6, 0.0, 0.1,
            0.1, 0.1, 0.6, 0.0, 0.1,
            0.1, 0.1, 0.6, 0.0, 0.1,
        ],
        "description": "All satellites push radially outward.",
        "expected_behavior": "Altitude excursion, returns to original orbit per Kepler",
        "use_case": "Test radial vs tangential response",
    },
    "test_message_alpha": {
        "values": [
            0.6, 0.0, 0.2, 0.0, 0.2,
            0.2, 0.2, 0.2, 0.2, 0.2,
            0.1, 0.1, 0.5, 0.1, 0.2,
        ],
        "description": "The standard test seed used across core module quick_tests.",
        "expected_behavior": "Mixed signal - each satellite has distinct profile",
        "use_case": "Canonical regression test vector",
    },
}


# -----------------------------------------------------------------------------
# Dataset generators
# -----------------------------------------------------------------------------

def generate_reference_seeds():
    """Write the canonical seed catalog."""
    print("[1/4] Generating reference_seeds.json...")
    data = {
        "schema_version": "1.0",
        "description": "Canonical 15-value seeds with known characteristics. Each seed is a 3x5 matrix: 3 satellites x 5 directional components [prograde, retrograde, outward, inward, north].",
        "seed_format": {
            "length": 15,
            "structure": "3 satellites x 5 components",
            "components": ["prograde", "retrograde", "outward", "inward", "north"],
            "value_range": [0.0, 1.0],
            "note": "6th component (south) derived by normalization: 1.0 - sum(others)",
        },
        "seeds": CANONICAL_SEEDS,
    }
    output_path = DATA_DIR / "reference_seeds.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"      -> {output_path.name} ({len(CANONICAL_SEEDS)} seeds)")
    return data


def generate_seed_output_pairs():
    """Run forward model on each canonical seed and record outputs."""
    print("[2/4] Generating seed_output_pairs.json...")
    expander = ThreeSatelliteExpander(harmonics=(2, 3, 5), include_j2=True)

    pairs = {}
    for name, seed_info in CANONICAL_SEEDS.items():
        seed = seed_info["values"]

        # Use small step count for compact dataset
        times, phase_rates, states_history = expander.expand_seed(
            seed, steps=2, deltaV_scale=0.001, symbol_period=300.0
        )

        # Downsample to keep file compact - take every 6th sample
        stride = max(1, len(times) // 20)
        times_ds = times[::stride].tolist()
        pr_ds = phase_rates[::stride].tolist()

        # Summary statistics (full resolution)
        pair_names = ["AB", "BC", "AC"]
        channel_stats = {}
        for ch_idx, ch_name in enumerate(pair_names):
            pr_ch = phase_rates[:, ch_idx]
            channel_stats[ch_name] = {
                "min": float(np.min(pr_ch)),
                "max": float(np.max(pr_ch)),
                "mean": float(np.mean(pr_ch)),
                "std": float(np.std(pr_ch)),
                "total_energy": float(np.sum(pr_ch**2)),
            }

        pairs[name] = {
            "seed": seed,
            "num_samples": len(times),
            "duration_seconds": float(times[-1]),
            "symbol_period": 300.0,
            "delta_v_scale": 0.001,
            "channel_stats": channel_stats,
            "downsampled_times": times_ds,
            "downsampled_phase_rates_AB": [row[0] for row in pr_ds],
            "downsampled_phase_rates_BC": [row[1] for row in pr_ds],
            "downsampled_phase_rates_AC": [row[2] for row in pr_ds],
        }

    data = {
        "schema_version": "1.0",
        "description": "Pre-computed forward model outputs for canonical seeds. Use these as reference signatures to compare against live simulation or recovered outputs.",
        "generation_params": {
            "harmonics": [2, 3, 5],
            "include_j2": True,
            "steps": 2,
            "delta_v_scale": 0.001,
            "symbol_period": 300.0,
        },
        "pairs": pairs,
    }
    output_path = DATA_DIR / "seed_output_pairs.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"      -> {output_path.name} ({len(pairs)} pairs)")
    return data


def generate_exercise_solutions():
    """Compute reference answers for the training exercises."""
    print("[3/4] Generating exercise_solutions.json...")
    expander = ThreeSatelliteExpander(harmonics=(2, 3, 5), include_j2=True)
    noise_model = PhaseRateNoiseModel()
    analyzer = DetectionAnalyzer(include_j2=True)

    solutions = {}

    # ex01: Seed expansion basics
    test_seed = CANONICAL_SEEDS["test_message_alpha"]["values"]
    perturbed = test_seed.copy()
    perturbed[0] += 0.01

    times1, pr1, _ = expander.expand_seed(test_seed, steps=1, symbol_period=300.0)
    times2, pr2, _ = expander.expand_seed(perturbed, steps=1, symbol_period=300.0)
    max_deviation = float(np.max(np.abs(pr2 - pr1)))

    solutions["ex01_seed_expansion_basics"] = {
        "question": "Does a 0.01 perturbation to seed[0] produce measurable output change?",
        "answer": "Yes",
        "max_phase_rate_deviation": max_deviation,
        "is_deterministic": True,
        "insight": "The mapping is deterministic and sensitive to small seed changes",
    }

    # ex02: Noise sensitivity
    noise_levels = [1e-6, 1e-5, 1e-4, 1e-3]
    snr_at_levels = {}
    for level in noise_levels:
        snr = 0.001 / level  # 1 mm/s signal over noise level
        snr_at_levels[f"noise_{level:.0e}"] = float(snr)

    solutions["ex02_noise_sensitivity"] = {
        "question": "How does SNR scale with noise amplitude?",
        "answer": "SNR = signal / noise, linear in 1/noise",
        "total_noise_std": float(noise_model.total_noise_std(300.0)),
        "snr_table": snr_at_levels,
        "insight": "Matched filtering provides ~sqrt(N) SNR improvement over threshold detection",
    }

    # ex03: Detection threshold mapping
    dv_range = np.array([0.0001, 0.0005, 0.001, 0.002, 0.005])
    dvs, pd_vals, snr_vals = analyzer.delta_v_sweep(
        test_seed, delta_v_range=dv_range, steps=2, symbol_period=300.0
    )

    detection_table = {}
    for dv, pd, snr in zip(dvs, pd_vals, snr_vals):
        detection_table[f"dv_{dv*1000:.2f}mm_s"] = {
            "pd": float(pd),
            "snr_db": float(snr),
        }

    solutions["ex03_detection_threshold"] = {
        "question": "What is the minimum delta-V for reliable detection?",
        "answer": "Depends on noise floor and Pfa target",
        "detection_table": detection_table,
        "insight": "Detection probability rises sharply once SNR exceeds ~10 dB",
    }

    # ex04: Constellation geometry
    solutions["ex04_constellation_geometry"] = {
        "question": "Why do prime harmonic ratios minimize channel cross-talk?",
        "answer": "Co-prime periods have maximum LCM, so satellites revisit the same relative geometry only after many orbits",
        "prime_ratios_25": {"ratios": [2, 3, 5], "lcm": 30, "independent": True},
        "non_prime_246": {"ratios": [2, 4, 6], "lcm": 12, "independent": False},
        "insight": "Prime ratios produce longer repeat periods = more distinct phase combinations",
    }

    # ex05: End-to-end pipeline
    recoverer = SeedRecoverer(steps=2, delta_v_scale=0.001, symbol_period=100.0)
    _, truth = recoverer.forward_model(test_seed)
    self_corr = float(recoverer.correlation_score(test_seed, truth))

    solutions["ex05_end_to_end_pipeline"] = {
        "question": "Can a seed be recovered from its own clean output?",
        "answer": "Yes, with correlation ~1.0",
        "self_correlation": self_corr,
        "bottleneck": "Recovery degrades rapidly with noise; codebook approaches are the next step",
        "insight": "Seed recovery is the bottleneck, not detection or encoding",
    }

    data = {
        "schema_version": "1.0",
        "description": "Reference solutions for training exercises in ../exercises.json",
        "solutions": solutions,
    }
    output_path = DATA_DIR / "exercise_solutions.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"      -> {output_path.name} ({len(solutions)} solutions)")
    return data


def generate_exploration_snapshot():
    """Pre-compute the solution space explorer output so it's queryable without re-running."""
    print("[4/4] Generating exploration_snapshot.json...")

    # Import the explorer
    from ai.training.explore_solution_space import SolutionSpaceExplorer

    explorer = SolutionSpaceExplorer(verbose=False)
    explorer.explore_seed_space(num_samples=15)
    explorer.explore_noise_space(num_levels=12)
    explorer.explore_detection_space(num_points=15)
    explorer.explore_constellation_space()

    data = {
        "schema_version": "1.0",
        "description": "Pre-computed solution space map. AI agents should query this instead of re-running explore_solution_space.py unless they need a custom parameter sweep.",
        "generated_by": "ai/training/explore_solution_space.py",
        "results": explorer.results,
    }

    output_path = DATA_DIR / "exploration_snapshot.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"      -> {output_path.name} (4 dimensions mapped)")
    return data


def check_datasets():
    """Verify all expected datasets exist and are parseable."""
    print("Checking training datasets...")
    expected = [
        "reference_seeds.json",
        "seed_output_pairs.json",
        "exercise_solutions.json",
        "exploration_snapshot.json",
    ]
    all_ok = True
    for name in expected:
        path = DATA_DIR / name
        if not path.exists():
            print(f"  MISSING: {name}")
            all_ok = False
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            print(f"  OK: {name} (schema_version={data.get('schema_version', '?')})")
        except json.JSONDecodeError as e:
            print(f"  CORRUPT: {name} ({e})")
            all_ok = False
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Generate AI training data")
    parser.add_argument("--check", action="store_true",
                        help="Verify existing datasets instead of regenerating")
    args = parser.parse_args()

    if args.check:
        ok = check_datasets()
        sys.exit(0 if ok else 1)

    print("=" * 60)
    print("  Orbital PHYCOM - Training Data Generation")
    print("=" * 60)
    print(f"  Output directory: {DATA_DIR}")
    print(f"  RNG seed: {RNG_SEED}")
    print()

    # Fix RNG for reproducibility where applicable
    np.random.seed(RNG_SEED)

    generate_reference_seeds()
    generate_seed_output_pairs()
    generate_exercise_solutions()
    generate_exploration_snapshot()

    print()
    print("Done. All datasets generated.")


if __name__ == "__main__":
    main()
