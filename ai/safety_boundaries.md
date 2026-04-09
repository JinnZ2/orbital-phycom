# Safety Boundaries for AI Agents

## Safe to Read (no restrictions)

All files in the repository are safe to read. Start with these for orientation:

| Priority | Path | Purpose |
|----------|------|---------|
| 1 | `CLAUDE.md` | Project instructions and conventions |
| 2 | `ai/navigation.json` | Structured codebase map |
| 3 | `README.md` | Project overview |
| 4 | `core/physics_constants.py` | Fundamental constants used everywhere |
| 5 | `core/Fixes.md` | Known issues and implementation notes |

## Safe to Run

These scripts produce output only (plots, print statements) and make no
persistent changes:

```
simulations/01_basic_orbital_sim.py    # ~30 sec, basic 2-satellite demo
simulations/02_three_satellite_demo.py # ~2 min, 3-satellite network
simulations/03_full_pipeline_demo.py   # ~3-5 min, full encode-decode pipeline
simulations/04_ground_atmospheric_coupling.py
```

Individual module verification (each has a `quick_test()` or `__main__` block):

```
core/orbital_dynamics.py
core/seed_expander.py
core/noise_model.py
core/detection.py
core/seed_recovery.py
```

## Modify with Caution

| Path | Risk | Notes |
|------|------|-------|
| `core/physics_constants.py` | High | Constants propagate everywhere; changing values breaks all simulations |
| `core/orbital_dynamics.py` | High | ODE integrator is finely tuned; small changes cause divergence |
| `core/seed_expander.py` | High | Seed format is a protocol contract; changes break compatibility |
| `requirements.txt` | Medium | Dependency changes affect all environments |

## Do Not Modify Without Explicit Permission

- `.git/` directory (never)
- `.gitignore` (rarely needed)
- `LICENSE` (legal document)
- `fieldlink/*.fieldlink.json` (cross-repo contracts)

## Validation After Changes

After modifying any `core/` module, verify by running:

```bash
python simulations/01_basic_orbital_sim.py
```

If it completes without errors and produces reasonable plots, the core is intact.
