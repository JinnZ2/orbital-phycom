# CLAUDE.md — Orbital PHYCOM

## Project Overview

Orbital PHYCOM (Physics-Based Communications via Orbital Deviation) is an ultra-low-bandwidth stealth orbital communication system. It uses geometric seed expansion and orbital mechanics as the decompression algorithm — transmitting only 15–40 bit seeds that deterministically generate full 24-hour orbital schedules.

**Key metric**: 6,944× compression ratio (144 bits/day vs. 1,000,000 bits for full schedule).

## Tech Stack

- **Language**: Python 3
- **Core libraries**: NumPy, SciPy (ODE integration, signal processing, sparse matrices), Matplotlib
- **Optional**: sgp4 (TLE propagation), requests (CelesTrak data harvesting)
- **No build system** — pure Python, direct script execution

## Repository Structure

```
core/                   # Orbital mechanics & seed expansion engines
  orbital_dynamics.py     # High-precision orbital propagation (Kepler + J2 + SRP + drag)
  seed_expander.py        # 3-satellite constellation seed expansion
  physics_constants.py    # Fundamental constants (Earth, gravity, RF)
  CelesTrak.py            # TLE harvesting from CelesTrak catalog
  debris-tracker.py       # Orbital debris tracking & rendezvous
  drag-force.py           # Atmospheric drag modeling
  theory/                 # Theoretical reference documents

atmospheric/            # Atmospheric seed expansion & thermal dynamics
  thermal_dynamics.py     # Atmospheric simulator with thermal processes
  ground_calibration.py   # Ground-based thermal control system

robotics/               # Hydrodynamic joint mechanics
  synovial_joint.py       # Geometric seed-based surface geometry
  joint_visualization.py  # Joint mechanics visualization

simulations/            # Executable demonstrations
  01_basic_orbital_sim.py   # 2-satellite system (~30 sec)
  02_three_satellite_demo.py # 3-satellite harmonic network (~2 min)
  04_ground_atmospheric_coupling.py

Thermopylae/            # Orbital thermal ecosystem architecture (JSON specs)
docs/                   # Documentation (QUICKSTART.md)
Sim.py                  # Main 3-satellite demonstration script
```

## Running the Code

```bash
# Quick start (~30 sec)
python simulations/01_basic_orbital_sim.py

# Full 3-satellite demo (~2 min)
python simulations/02_three_satellite_demo.py

# Main demonstration
python Sim.py

# Individual module verification
python core/orbital_dynamics.py
python core/seed_expander.py
```

## Code Conventions

### Naming
- **Classes**: PascalCase (`OrbitalSimulator`, `ThreeSatelliteExpander`)
- **Functions**: snake_case (`expand_seed`, `calculate_miss_distance`)
- **Constants**: UPPER_SNAKE_CASE (`MU_EARTH`, `R_EARTH`, `ALTITUDE_REF`)
- **Private methods**: `_leading_underscore`

### Style
- Google-style docstrings with `Args:`, `Returns:` sections
- PEP 8 formatting (informal — no linter configured)
- SI units throughout (meters, seconds, kg)
- High-precision constants (e.g., `MU_EARTH = 3.986004418e14`)

### Physics Patterns
- Seeds are lists of floats normalized 0.0–1.0; 5 values per satellite × 3 satellites = 15 values
- Each 5-tuple: `[prograde, retrograde, outward, inward, north]`
- ΔV impulses: 0.0005–0.002 m/s typical
- ODE integration via `scipy.integrate.solve_ivp` (DOP853 integrator)
- VNB (Velocity-Normal-Binormal) reference frame for impulse vectors
- Time in seconds (`symbol_period=300.0` = 5 minutes)

### Visualization
- Multi-panel `GridSpec` layouts, `viridis`/`plasma` colormaps
- Time series + 3D surface plots

## Architecture Notes

- `core/` modules are foundational; other packages build on seed expansion concepts
- All simulators are class-based with explicit parameter constructors
- No external config files — parameters are hardcoded or passed to constructors
- Constants centralized in `core/physics_constants.py`

## Known Issues (from core/Fixes.md)

1. **VNB singularity**: Equatorial circular orbits cause cross-product failures
2. **Phase rate computation**: Assumes RF inter-satellite links; real systems use angle-only tracking
3. **Integration inefficiency**: Per-symbol ODE solutions; should use continuous integration with events
4. **Numerical sampling**: 10 Hz is overkill; 1 Hz sufficient

## Testing

No formal test framework. Validation is done via:
- Simulation output verification (graphs + print statements)
- Visual inspection of matplotlib plots
- Examples in `core/Fixes.md`

## License

MIT
