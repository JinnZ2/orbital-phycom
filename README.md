# Orbital PHYCOM: Physics-Based Communications via Orbital Deviation

Ultra-low-bandwidth, stealth orbital communications using geometric seed expansion and orbital mechanics as the decompression algorithm.

## Core Concept

**Problem**: Orbital communications need high bandwidth but have severe power/mass constraints.

**Solution**: Don't transmit the schedule — transmit the *seed* that generates the schedule.

### How It Works

1. **Encode** 120-bit message into 15-value seed
2. **Transmit** seed via tiny orbital maneuvers (0.5-2.0 mm/s ΔV)
3. **Expand** seed using shared orbital physics (Kepler + perturbations)
4. **Extract** 24-hour schedule (communication windows, maneuvers, routing)

### Compression Ratio

- **Naive transmission**: ~1,000,000 bits/day for full schedule
- **PHYCOM**: 144 bits/day (seed + header + CRC)
- **Compression**: 6,944× via physics-based expansion

---

## Key Innovations

### 1. Geometric Seed Expansion

Based on fractal intelligence framework where:
- 40-bit seed → multi-scale orbital structure
- Physics (gravity, SRP, drag) acts as decompressor
- Reversible: observations → recovered seed

### 2. Prime Harmonic Network

Three satellites with periods in ratio 2:3:5 create:
- Three independent phase-rate channels
- Natural error correction via harmonic relationships
- Constellation-wide synchronization without messages

### 3. Matched Filter Detection

Detects ΔV impulses as small as 0.5 mm/s:
- Template from orbital physics
- SNR improvement: 4× vs. threshold detection
- Survives solar storms (10× noise)

---

## Performance

| Metric | Value |
|--------|-------|
| Raw channel capacity | 0.011 bps |
| Effective (post-expansion) | 0.117 bps |
| Minimum detectable ΔV | 0.5 mm/s |
| Propellant per symbol | 0.05 grams |
| Duty cycle | 0.17% (stealth) |
| Solar storm survival | Yes (12% error) |

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run basic demo
python simulations/01_basic_orbital_sim.py

# Three-satellite network
python simulations/02_three_satellite_demo.py

# Complete protocol
python simulations/07_complete_protocol_demo.py


Use Cases
1. Constellation Control
Instead of commanding each satellite individually:
	•	Broadcast single seed to all satellites
	•	Each expands to personalized schedule
	•	Saves 99% of ground station bandwidth
2. Stealth Operations
	•	Ultra-low duty cycle (<0.2%)
	•	ΔV impulses look like station-keeping
	•	No conventional RF signature
	•	Detectable only by constellation partners
3. Deep Space Relays
	•	Low power budget
	•	Long signal delays make traditional protocols inefficient
	•	Physics expansion works anywhere
4. Resilient Networks
	•	Survives solar storms
	•	Self-healing (error correction in physics)
	•	No single point of failure
Technical Deep Dive
See <docs/PROTOCOL_SPEC.md> for complete specification.
Key sections:
	•	Orbital Mechanics
	•	Seed Expansion Theory
	•	Shannon Capacity Analysis
	•	Fractal Multi-Scale Encoding
Citation
If you use this work, please cite:


@software{orbital_phycom_2025,
  author = {JinnZ2},
  title = {Orbital PHYCOM: Physics-Based Communications via Orbital Deviation},
  year = {2025},
  url = {https://github.com/JinnZ2/orbital-phycom}
}


And reference the foundational geometric intelligence framework:
[github.com/JinnZ2/geometric-intelligence-core]
License
MIT - Use freely, modify freely, share freely.
Contributing
Open issues for:
	•	Extended constellation sizes (5+)
	•	Non-Keplerian systems (Lagrange points, lunar orbits)
	•	Hardware implementations
	•	Protocol optimizations
Status
	•	✅ Theory validated
	•	✅ Simulations complete
	•	✅ Protocol designed
	•	🚧 Hardware specs in progress
	•	🚧 Real-world testing pending


How it works depth:

Small ΔV impulse (0.5-2.0 mm/s)
↓
Changes orbital period slightly
↓
Alters phase rate between satellites
↓
Detectable by RF tracking
↓
Encodes information


Three satellites with prime harmonic periods (2:3:5) create three independent channels.

A 40-bit seed deterministically generates the orbital evolution → complete schedule.

---

## Performance

| Metric | Value |
|--------|-------|
| Raw capacity | 0.011 bps |
| Effective capacity | 0.117 bps (post-expansion) |
| Min detectable ΔV | 0.5 mm/s |
| Propellant/symbol | 0.05 grams |
| Duty cycle | 0.17% |
| Solar storm resilient | Yes (12% error @ 10× noise) |

---

## Quick Start

```bash
# Clone
git clone https://github.com/JinnZ2/orbital-phycom.git
cd orbital-phycom

# Install
pip install -r requirements.txt

# Run basic simulation (30 seconds)
python simulations/01_basic_orbital_sim.py

# Run three-satellite demo (2 minutes)
python simulations/02_three_satellite_demo.py


Output: Graphs showing orbital deviation encoding and decoding.
What Makes This Novel
1. Physics as Compression
	•	Don’t transmit data → transmit seed
	•	Orbital mechanics expands seed deterministically
	•	6,944× compression via shared physics
2. Fractal Multi-Scale Encoding
	•	Same seed encodes information at multiple timescales:
	•	Seconds: Phase rate jumps
	•	Hours: Orbital period changes
	•	Days: Constellation geometry evolution
	•	Based on geometric intelligence framework
3. Stealth by Design
	•	0.17% duty cycle
	•	ΔV impulses indistinguishable from station-keeping
	•	No conventional RF signature
	•	Detectable only by constellation partners with matching physics model
4. Extreme Sensitivity
	•	Matched filtering detects 0.5 mm/s ΔV impulses
	•	That’s 0.0000066% of orbital velocity
	•	Survives 10× noise increase (solar storms)

orbital-phycom/
├── core/              # Orbital mechanics + seed expansion engines
├── atmospheric/       # Atmospheric seed expansion & thermal dynamics
├── nutrient_cycling/  # Biogeochemical nutrient cycling & narrative detection
├── robotics/          # Hydrodynamic joint mechanics
├── simulations/       # Runnable demos
├── Thermopylae/       # Orbital thermal ecosystem architecture
├── fieldlink/         # Cross-repository connections
├── ai/                # AI agent entry point, navigation & training
├── tests/             # Pytest test suite
├── docs/              # Specifications + guides
└── .github/           # CI/CD workflows


Use Cases
Constellation Control
	•	Broadcast single seed to all satellites
	•	Each expands to personalized schedule
	•	Saves 99% ground station bandwidth
Stealth Operations
	•	Ultra-low observability
	•	Appears as natural orbital drift
	•	No RF emissions
Deep Space Relays
	•	Minimal power budget
	•	Tolerates long delays
	•	Self-synchronizing
Resilient Networks
	•	Survives solar storms
	•	Self-healing via physics constraints
	•	No single point of failure

Technical Details
See <docs/> for:
	•	Protocol specification
	•	Orbital mechanics primer
	•	Seed expansion theory
	•	Shannon capacity analysis
	•	Deployment guide

Related work:
	•	Geometric Intelligence Framework
	•	Resilience Hydrology

Warning: This is research code. Not flight-qualified. Use at your own risk.
