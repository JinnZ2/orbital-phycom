# AI Training - Explore Solution Space

This folder contains tools for AI agents to build deep familiarity with the
Orbital PHYCOM codebase through guided exploration and interactive discovery.

## Contents

| File | Purpose |
|------|---------|
| `explore_solution_space.py` | Interactive solution space explorer - maps parameters, constraints, and outcomes |
| `codebase_map.py` | Programmatic codebase discovery - scans modules, classes, functions |
| `exercises.json` | Guided learning exercises with increasing complexity |

## Quick Start

```bash
# Explore the solution space interactively
python ai/training/explore_solution_space.py

# Generate a codebase map
python ai/training/codebase_map.py

# List available exercises
python ai/training/explore_solution_space.py --list-exercises
```

## How the Solution Space Explorer Works

The explorer maps the **parameter space** of the PHYCOM system:

1. **Seed Space**: How different seed values produce different orbital schedules
2. **Noise Space**: How noise levels affect seed recovery accuracy
3. **Detection Space**: How delta-V magnitude relates to detection probability
4. **Constellation Space**: How satellite period ratios affect channel capacity

Each dimension can be swept independently or in combination, producing a map
of feasible operating regions and their trade-offs.

## Training Philosophy

- **Learn by doing**: Run simulations with varying parameters
- **Build intuition**: Visualize how inputs map to outputs
- **Discover boundaries**: Find where the system breaks and why
- **Connect modules**: Understand how core components interact
