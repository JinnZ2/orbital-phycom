# Training Data - Pre-Computed Reference Datasets

This folder contains reproducible reference data for AI agents learning
the Orbital PHYCOM system. The goal: **learn from pre-computed examples
instead of re-running expensive simulations for every query.**

All datasets here are regeneratable from source via
`generate_training_data.py`.

## Datasets

### `reference_seeds.json`

Catalog of canonical 15-value seeds with known characteristics. Each
seed has a human-readable description, expected behavior, and use case.

**When to use**: As input to any simulation, detection, or recovery test
when you need a known-good starting point.

**Canonical seeds included**:
| Name | Description |
|------|-------------|
| `neutral` | Uniform allocation (minimal net delta-V) |
| `prograde_heavy` | All satellites push prograde |
| `retrograde_heavy` | All satellites push retrograde |
| `differential` | Each satellite maneuvers differently |
| `radial_push` | All satellites push radially outward |
| `test_message_alpha` | Standard regression test vector |

### `seed_output_pairs.json`

Pre-computed forward model outputs for each canonical seed. For every
seed you get:
- Full channel statistics (min/max/mean/std/total_energy) at full resolution
- Downsampled phase-rate time series (~20 samples per channel)
- Generation parameters (harmonics, delta_v_scale, symbol_period)

**When to use**: To validate that your forward model or recovery produces
expected outputs without running a fresh simulation. Compare your output's
channel stats against the reference.

### `exercise_solutions.json`

Reference answers for the training exercises in `../exercises.json`.
Each solution includes the question, the answer, supporting data, and
the key insight.

**When to use**: After attempting an exercise yourself, compare your
findings against the reference solution to self-verify understanding.

### `exploration_snapshot.json`

Pre-computed output from `explore_solution_space.py` covering all four
dimensions:
- Seed space (delta-V range, sensitivity)
- Noise space (SNR vs recoverability)
- Detection space (delta-V vs detection probability)
- Constellation space (period ratios vs independence)

**When to use**: Whenever you need to answer "how does X affect Y?" for
parameter combinations already covered. Only re-run the explorer if you
need a custom parameter sweep not in the snapshot.

## Regenerating the Data

All datasets are deterministic and reproducible:

```bash
python ai/training/data/generate_training_data.py
```

The generator uses a fixed RNG seed (42) so outputs are bit-for-bit
identical across runs. To verify the existing files:

```bash
python ai/training/data/generate_training_data.py --check
```

## Design Principles

- **Small enough to commit**: All files are JSON, total well under 100KB
- **Regeneratable**: Single source of truth is the generator script
- **Deterministic**: Fixed RNG seed, no wall-clock dependencies
- **Versioned**: Each file has `schema_version` for forward compatibility
- **Self-documenting**: Every dataset includes a `description` field

## Extending

To add a new canonical seed, edit `CANONICAL_SEEDS` in
`generate_training_data.py` and re-run. The seed will automatically
propagate into `reference_seeds.json` and `seed_output_pairs.json`.

To add a new exercise solution, add a new entry inside
`generate_exercise_solutions()` following the existing schema.
