# CLAUDE.md — Orbital PHYCOM

## Project Overview

Orbital PHYCOM (Physics-Based Communications via Orbital Deviation) is an ultra-low-bandwidth stealth orbital communication system. It uses geometric seed expansion and orbital mechanics as the decompression algorithm — transmitting only 15-40 bit seeds that deterministically generate full 24-hour orbital schedules.

**Key metric**: 6,944x compression ratio (144 bits/day vs. 1,000,000 bits for full schedule).

## Tech Stack

- **Language**: Python 3
- **Core libraries**: NumPy, SciPy (ODE integration, signal processing, sparse matrices), Matplotlib
- **Optional**: sgp4 (TLE propagation), requests (CelesTrak data harvesting)
- **No build system** — pure Python, direct script execution
- **Dependencies**: Listed in `requirements.txt`

## Repository Structure

```
core/                       # Orbital mechanics & seed expansion engines
  orbital_dynamics.py         # Orbital propagation with J2 perturbation + segmented impulses
  seed_expander.py            # 3-satellite constellation seed expansion (VNB frame, J2-aware)
  noise_model.py              # Phase-rate noise sources + link budget analysis
  seed_recovery.py            # Inverse problem: noisy observations → seed reconstruction
  detection.py                # Signal detectability analysis (ROC, matched filter, Pd vs ΔV)
  physics_constants.py        # Fundamental constants (SI units, includes J2)
  celestrak.py                # TLE harvesting from CelesTrak catalog
  debris_tracker.py           # Orbital debris tracking via SGP4
  drag_force.py               # Exponential atmosphere + drag acceleration
  Fixes.md                    # Known issues & implementation notes
  theory/                     # Theoretical reference documents

atmospheric/                # Atmospheric seed expansion & thermal dynamics
  thermal_dynamics.py         # AtmosphericSimulator + AtmosphericSeedExpander
  ground_calibration.py       # GroundThermalController + GroundAtmosphericCoupler

nutrient_cycling/           # Biogeochemical nutrient cycling & narrative detection
  nutrient_constants.py       # Physical constants for N, P, K, soil, agriculture
  nitrogen_fixation.py        # Lightning, bacterial, atmospheric, compost N pathways
  phosphorus_recovery.py      # Sewage, food waste, dump, rock P recovery
  potassium_cycling.py        # Weathering, ocean spray, ash, rock dust, human waste K
  soil_biology.py             # SOM restoration, microbial biomass, yield improvement
  local_capacity.py           # Integrated food security calculator (Liebig's law)
  narrative_detector.py       # Narrative vs physics claim analysis

robotics/                   # Hydrodynamic joint mechanics
  synovial_joint.py           # GeometricSeedController + HydrodynamicJointSimulator
  joint_visualization.py      # Joint pattern comparison visualization

simulations/                # Executable demonstrations
  01_basic_orbital_sim.py     # 2-satellite system (~30 sec)
  02_three_satellite_demo.py  # 3-satellite harmonic network (~2 min)
  03_full_pipeline_demo.py    # Full encode→noise→detect→recover pipeline (~3-5 min)
  04_ground_atmospheric_coupling.py

fieldlink/                  # Cross-repository connections
  nexus_emergency_management.fieldlink.json  # Link to Nexus-emergency-management
  infrastructure_assistance.fieldlink.json   # Link to Infrastructure-assistance

ai/                         # AI agent entry point, navigation & training
  README.md                   # Safe entry point for AI agents
  safety_boundaries.md        # Operating limits and safe exploration zones
  navigation.json             # Machine-readable codebase map
  training/                   # AI training & solution space exploration
    explore_solution_space.py   # Interactive parameter sweep tool
    codebase_map.py             # Programmatic codebase discovery
    exercises.json              # Guided learning exercises

Thermopylae/                # Orbital thermal ecosystem architecture (JSON specs)
docs/                       # Documentation (QUICKSTART.md)
```

## Running the Code

```bash
# Quick start (~30 sec)
python simulations/01_basic_orbital_sim.py

# Full 3-satellite demo (~2 min)
python simulations/02_three_satellite_demo.py

# Full pipeline: encode → noise → detect → recover (~3-5 min)
python simulations/03_full_pipeline_demo.py

# Ground-atmospheric coupling
python simulations/04_ground_atmospheric_coupling.py

# Individual module verification
python core/orbital_dynamics.py
python core/seed_expander.py
python core/noise_model.py
python core/detection.py
python core/seed_recovery.py
```

## Code Conventions

### Naming

All names follow PEP 8:

- **Files**: snake_case (`orbital_dynamics.py`, `seed_expander.py`)
- **Classes**: PascalCase (`OrbitalSimulator`, `ThreeSatelliteExpander`)
- **Functions**: snake_case (`expand_seed`, `calculate_miss_distance`)
- **Constants**: UPPER_SNAKE_CASE (`MU_EARTH`, `R_EARTH`, `ALTITUDE_REF`)
- **Private methods**: `_leading_underscore`
- **Variables**: Use ASCII names only (no unicode like Greek letters)

### Style

- Google-style docstrings with `Args:`, `Returns:` sections
- SI units throughout (meters, seconds, kg)
- High-precision constants (e.g., `MU_EARTH = 3.986004418e14`)
- Imports at top of file — no inline/repeated imports

### Physics Patterns

- Seeds are lists of floats normalized 0.0-1.0; 5 values per satellite x 3 satellites = 15 values
- Each 5-tuple: `[prograde, retrograde, outward, inward, north]`; 6th (south) from normalization
- Delta-V impulses: 0.0005-0.002 m/s typical
- ODE integration via `scipy.integrate.solve_ivp` (DOP853 integrator)
- VNB frame: V (velocity/prograde), N (radial/normal), B (binormal/orbit-normal)
- Time in seconds (`symbol_period=300.0` = 5 minutes)

### Visualization

- Multi-panel `GridSpec` layouts, `viridis`/`plasma` colormaps
- Time series + 3D surface plots

## Architecture Notes

- `core/` modules are foundational; other packages build on seed expansion concepts
- All simulators are class-based with explicit parameter constructors
- Constants centralized in `core/physics_constants.py`
- Each package has `__init__.py` with `__all__` exports
- Simulation scripts use `sys.path.insert` for imports (run as standalone scripts)

## Known Issues (from core/Fixes.md)

1. ~~**VNB singularity**~~: Fixed — now uses angular momentum vector for orbit-normal basis
2. ~~**Two-body only**~~: Fixed — J2 oblateness integrated into both propagators (`include_j2=True` by default)
3. **Phase rate computation**: Assumes RF inter-satellite links; real systems use angle-only tracking
4. **Integration sampling**: 10 Hz may be overkill; 1 Hz sufficient for smooth orbital evolution
5. **Seed recovery convergence**: Coarse search with 100-500 candidates provides approximate recovery; higher fidelity requires more candidates or longer observation windows
6. **Signal clarification vs raw recovery** (2026-03-22): Benchmarked baseline subtraction, Savitzky-Golay denoising, and spectral filtering against raw phase-rate recovery. At current SNR (~0.22/sample, 93 samples), baseline subtraction is comparable on RMSE (0.221 vs 0.212) but reduces worst-case max error (0.415 vs 0.543). Denoising methods (SavGol, spectral) consistently hurt. **Follow-up needed**: the real path to better recovery is likely stronger ΔV, longer observation windows, or a discrete seed codebook — not post-hoc signal processing. Investigate codebook-based recovery as a next step.

## Testing

No formal test framework. Validation is done via:
- Simulation output verification (graphs + print statements)
- Visual inspection of matplotlib plots
- `quick_test()` functions in core modules (`python core/orbital_dynamics.py`)

## License

MIT
