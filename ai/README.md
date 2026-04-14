# AI - Safe Entry Point

**START HERE.** This folder is the designated entry point for AI agents
(and humans onboarding AI tools) working with the Orbital PHYCOM codebase.

It provides:
- A clear, ordered onboarding path
- Explicit safety boundaries
- Machine-readable navigation
- Training tools AND training data

---

## STEP 0: Orient Yourself

You are in a physics-based communications research project. The code
simulates orbital satellites communicating via tiny delta-V impulses.
Read this file top-to-bottom before touching anything else.

## STEP 1: Read Safety Boundaries

Open `safety_boundaries.md`. It tells you:
- Which files are safe to read (all)
- Which scripts are safe to run (simulations, quick_tests)
- Which files to modify with caution (core physics)
- What never to modify without explicit permission

**Do not skip this step.** If you modify a core constant or integrator
tolerance without understanding the physics, you will silently break every
downstream simulation.

## STEP 2: Load Navigation

Open `navigation.json`. It is a machine-readable map of every module,
class, and key function, with a suggested 7-step learning path. Parse it
programmatically to answer "where is X?" without guessing.

## STEP 3: Study the Training Data

Open `training/data/`. This folder contains **pre-computed reference
examples**:
- Canonical seeds with known outputs
- Expected phase-rate time series for each seed
- Ground-truth solutions for exercises
- Exploration snapshots (no need to re-run long simulations)

Learn from the data first. Only run live simulations when you need
behavior that is not already captured in a reference dataset.

## STEP 4: Run Training Exercises

Open `training/exercises.json` and work through exercises from
beginner to advanced. Each exercise references specific files and has
an expected insight so you can self-check your understanding.

Use `training/explore_solution_space.py` and `training/codebase_map.py`
as interactive tools.

## STEP 5: Do Real Work

Only after steps 0-4 should you make modifications to the codebase.
When you do:
- Read the file you are modifying in full first
- Check `core/Fixes.md` for known issues in the area you are touching
- Run `pytest tests/` after any change to `core/`
- Prefer small, reviewable diffs

---

## Folder Layout

```
ai/
  README.md              <-- You are here. Read this first.
  safety_boundaries.md   <-- Step 1. Safe operating limits.
  navigation.json        <-- Step 2. Machine-readable codebase map.
  training/              <-- Steps 3-4. Training resources.
    README.md              Training overview
    exercises.json         Guided learning exercises
    explore_solution_space.py  Interactive parameter sweep
    codebase_map.py        Programmatic codebase discovery
    data/                  <-- Pre-computed training datasets
      README.md              Dataset catalog
      reference_seeds.json   Canonical seeds
      seed_output_pairs.json Reference seed -> output pairs
      exercise_solutions.json Ground truth for exercises
      exploration_snapshot.json Pre-computed solution space map
      generate_training_data.py Regenerate all datasets
```

## Core Principles

- **Safe by default**: Read-only orientation before any modification
- **Data-first learning**: Use pre-computed references, not live simulation
- **Progressive disclosure**: Start simple, deepen on demand
- **Machine-readable**: JSON indexes for programmatic navigation
- **Reproducible**: All training data is regeneratable from source

## For Human Collaborators

This folder lets AI assistants onboard in seconds instead of minutes, and
reduces the chance of unintended modifications by making the safety
boundaries and project structure explicit. The training data folder
means an AI can learn the system's behavior from pre-computed examples
without burning compute on redundant simulations.
