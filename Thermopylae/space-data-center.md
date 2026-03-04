THERMOPYLAE

Orbital Thermal Ecosystem Architecture
Self-Replicating Space Data Center from Orbital Debris

Technical Reference v1.0

March 2026
 
 
 
Design Principle: No single-use gradients. Every joule serves 3–4 roles before radiating away.


THERMOPYLAE — Technical Reference
1. System Overview

Thermopylae is an orbital compute infrastructure that treats waste heat as a resource rather than a liability. The architecture bootstraps from a single 1-ton seed factory launched to sun-synchronous LEO, captures and refines orbital debris, and self-replicates into a distributed thermal economy. The design draws explicitly from biological ecosystems: eat your waste, excrete structure, reproduce from environment.
 
Parameter
Specification
Target scale
1 MW compute (10 nodes at 100 kW each)
Launch mass
1 metric ton (seed factory)
Built mass
15 metric tons (from orbital debris)
Orbit
Sun-synchronous LEO, 800 km
Radiator total
5,000 m² distributed across swarm
Self-funding target
Month 12–18
10 MW path
10 federated swarms, lunar relay
 
The system is organized in nine interdependent layers, each building on the previous: thermal phase space constraints, dual-use radiator design, five-stage energy cascade, 100 kW platform architecture, swarm federation, evolutionary bootstrap, torch generational evolution, compute rad-hardening, and propulsion co-evolution.

2. Thermal Phase Space Constraints

All downstream architecture is constrained by the Stefan-Boltzmann radiation law. In vacuum, heat rejection scales with T⁴ but compute density scales linearly with power. This fundamental tension shapes every design decision.
 
2.1 Core Physics
Radiative power density: q = εσ(T⁴_rad − T⁴_sink), where ε is emissivity, σ = 5.67×10⁻⁸ W/m²K⁴
Key insight: Going from 300K to 400K radiators buys more cooling per m² than 350K to 450K due to T⁴ nonlinearity. But material constraints push back hard above 500K.
Sink temperature dominance: LEO free-flyer (3K sink) vs. lunar surface (280K sink) completely reshapes the constraint geometry. The ΔT between emitter and sink determines everything.
 
Scenario
Rejection Rate
LEO free-flyer
350K rad, 3K sink → 840 W/m²
Lunar surface
400K rad, 280K sink → 500 W/m²
L2 deep space
300K rad, 3K sink → 420 W/m²
Mars surface
350K rad, 210K sink → 730 W/m²
 
3. Three-Layer Dual-Use Radiator

The core innovation: 1500 m² of radiator surface is not passive waste metal. It is a structured energy export system with three functional layers.
 
3.1 Layer 1: Bulk Radiator
Dumps waste heat via broadband IR. Lambertian emission. No modification to standard thermal design. This layer does its thermal job unchanged and represents 90% of radiator area.
 
3.2 Layer 2: Geometric Concentrator
Horn and trough structures on 10% of radiator area narrow the emission lobe from 2π steradians to a designed cone angle. Critically, this does not increase brightness temperature above the physical emitter temperature. The geometry redirects, it does not amplify. Same brightness, narrower beam, higher power per steradian on target.
 
Parameter
Specification
Hot segment fraction
5–10% of total radiator area
Operating temperature
600–1000K (dedicated thermal loop)
Trough design
λ/2 width (5–10 μm), 10–20λ depth, 1m length
Packing density
10,000 troughs/m²
Cone angle
15° half-angle
Geometric gain
2π/Ω_cone (does not exceed diffraction limit)
 
3.3 Layer 3: Active Modulator
Information rides on emissivity modulation at the horn throat, not on the thermal carrier itself. The waste heat provides free carrier power. A small active element (VO₂ phase-change film, MEMS shutter, or electrochromic layer) switches emissivity at kHz rates. This is where the data lives.
 
Spectral integration: Engineered emissivity ε(λ) suppresses emission in wasted bands and enhances it in the comms band. Shifts a slice of the blackbody integral into a designed spectral window.
Spatial integration: Corrugated geometry constrains allowed emission angles. Same local brightness, much smaller solid angle in preferred direction.
Temporal integration: Duty-cycled brightness via stored thermal mass (PCM). Short hot bursts followed by cooling. Reshapes the time integral of the blackbody curve into high-peak, low-duty pulses.

4. Five-Stage Energy Cascade

Every joule of input power traverses five stages of decreasing temperature gradient. At each stage, the gradient is deliberately spent on at least one piggyback service before the remaining energy passes to the next stage. No single-use gradients.
 
4.1 Stage 1: Compute + Local Thermal Work
Input: 1 J electric → 0.85 J heat + FLOPs
Hot dies feed node-level phase-change thermal storage (paraffin wax, 200 kJ/kg latent heat). Excess heat keeps batteries and gyroscopes at 20–25°C during 30-minute eclipses, replacing 50 W/node of dedicated survival heaters. Flow sensors on the cooling loop monitor per-node health; thermal transients flag faults before electrical symptoms appear.
 
4.2 Stage 2: Cluster Thermal Bus
Pumped single-phase loop (Dowtherm fluid) aggregates node waste heat. Loop taps warm the propellant depot (Xe or N₂, 10 kg reserve) above sublimation point, saving 2 kW of electrical heaters. A 5 kW diversion to a high-temperature stage (500–600K block) feeds thermophotovoltaic cells (15% efficient at 1.55 μm), recovering approximately 750 mW electrical. This recovered power runs the always-on 1 kbps IR beacon and station-keeping thruster valves. Residual heat pulses attitude louvers (thermal actuators, no motors) for passive roll bias.
 
4.3 Stage 3: Radiator + Structured Export
Distributed to 20 radiator tiles (25 m² each). 10% of tile area is corrugated hot gates: 1 m troughs at 8–12 μm guide thermal emission into 15° cones toward ground relay. Tile-edge thermal diodes monitor gradient uniformity for workload throttling. Outer tiles double as docking radiators for visiting cubesats.
 
4.4 Stage 4: Final Entropy Dump
Remaining heat radiates as broadband IR over 90% of tile area, shaped by edge shields to minimize albedo reflection back to the cluster. The known IR signature serves as a calibration source for onboard sensors during alignment checks.
 
4.5 Stage 5: Ecosystem Services
The thermal flow pattern itself encodes diagnostic information. Monitoring how heat flows and fluctuates provides health, workload, and environmental data. Every m² of radiator has a thermal API serving compute cooling, housekeeping, signaling, diagnostics, docking support, and calibration.
 
Service
Value
Eclipse heater replacement
500 W (10 nodes)
Propellant conditioning
2 kW
TPV electrical recovery
750 mW
Attitude thermal actuators
200 W
IR beacon
100 bps nav data
Docking thermal support
4 tiles active
Total secondary services
~21 kW equivalent

5. 100 kW Platform Specification

 
Parameter
Specification
Total power
100 kW electric (solar or beamed)
Compute payload
10 modular nodes, 10 kW class each
Waste heat
~85 kW steady, peaked by workload
Radiator area
500 m² at 320–350K, 4 faces
Orbit
Sun-synchronous LEO
Dry mass
~1,500 kg
Services per joule
3–4 before final dump
 
5.1 Mass Stack
Subsystem
Mass
Compute + power electronics
~500 kg
Thermal plumbing + TPV/gates
~200 kg
Radiators (500 m² @ 1.2 kg/m²)
600 kg
Structure + tanks + docking
~200 kg
Total dry
~1,500 kg
 
5.2 Scaling Wall at 1 MW
Single-vehicle architecture breaks above 200 kW. Single-phase loop pressure drop exceeds pump limits. Thermal customers (heaters, propellant, TPV) saturate. Hot gate coherence needs active management. Radiator structural mass exceeds single-vehicle deployment limits. The solution is fracturing into a swarm.

6. 1 MW Swarm Thermal Economy

The 1 MW scaling wall dissolves into 10 autonomous 100 kW nodes that form a shared cooling and communications fabric. Nodes are no longer thermally isolated; they trade thermal capacity like a market.
 
Parameter
Specification
Platform count
10 autonomous nodes
Total compute
1 MW electric (~850 kW waste heat)
Total radiator
5,000 m² distributed
Formation spacing
100–500 m
Operating margin
80% thermal capacity; 20% headroom
IR mesh baselines
45 (10 choose 2)
Ranging precision
Sub-millimeter from waste heat photons
 
6.1 Thermal Leasing
Excess radiator capacity is a tradeable asset. Nodes with cold radiators (post-eclipse) accept heat dumps from hot neighbors via IR beaming (10–50 W per link at 200 m) or physical docking (kW class). Ground stations schedule beacon windows in exchange for priority compute beam time.
 
6.2 Hot Gate Inter-Node Protocol
Frame format: 8-bit node ID + 8-bit thermal margin, repeated at 100 bps OOK.
Link budget: 200 m inter-node distance, 10⁴ photons/bit SNR, error-free 100 bps.
Function: Each node broadcasts: I am N(x), I have (y)% thermal margin.
 
6.3 Resilience
If one node loses 50% radiator capacity (micrometeorite), the swarm reallocates compute load across the remaining 9 nodes with less than 5% total capacity loss. Thermal leasing and workload federation absorb the shock. The damaged node becomes feedstock for two replacements whose excess radiator capacity enters the market. The system gets stronger from damage.
 
6.4 Navigation Mesh
All hot gates form a distributed IR interferometer. Mutual ranging provides cm-level formation control without RF. The thermal exhaust IS the navigation system. No dedicated ranging hardware required.

7. Evolutionary Bootstrap

The seed factory is a genome, not a blueprint. What it builds depends on what feedstock it encounters and what the thermal economy demands.
 
7.1 Genome: Five Invariant Capabilities
Capability
Implementation
MELT
Plasma torch. Al at 660°C. Waste heat feeds TPV.
SORT
Centrifuge + spectral ID. Alloys, composites, chips, glass.
PRINT
Wire-arc AM. Radiator panels, frames, troughs, horns.
ASSEMBLE
Robotic arms. Dock, connect thermal loops, integrate compute.
COMMUNICATE
Hot gates. 100 bps thermal Morse. Swarm state broadcast.
 
7.2 Phenotype Expression
The factory expresses different phenotypes based on environment. Find a Centaur tank: express a radiator-heavy node. Find dead Starlinks: express compute-dense nodes from salvaged solar cells and chips. Encounter debris too small to grapple: express a collector/aggregator phenotype.
 
Phenotype
Requirements and Yields
Radiator Node
1,200 kg Al needed. 500 m² rad, 5 kW compute, 85 kW cooling.
Compute Node
600 kg Al + 20 chips + 40 m² solar. 150 m² rad, 50 kW compute.
Collector
200 kg Al. Sweep + stockpile feedstock. 2x capture rate.
Seed Factory
500 kg Al + 8 chips. Self-replicates. Build rate 1.0.
 
7.3 Selection Pressure
Fitness function: revenue per kg per month. Nodes generating more value from the thermal economy survive and reproduce. Compute price drops with supply (self-correcting market). Low-fitness nodes are recycled, returning 60% of materials. Seed production is boosted when population is below 5 (early growth prioritizes reproduction). Collector production is boosted when aluminum stockpile is below 2 tons (starvation triggers foraging).

8. Plasma Torch Generational Evolution

The torch is the metabolic core. Everything else is downstream of the ability to melt aluminum in vacuum. The torch evolves through four generations, each printed from the output of the previous.
 
Gen 0: Seed Torch (launched, 30 kg)
 
Power
5 kW solar-fed arc (Ar vortex constriction)
Nozzle
Tungsten cathode + scrap Al anode ring
Electrodes
Graphite (5 kg launched). C solubility in Al <0.01%. Zero contamination.
Plasma
10,000K jet, 1 cm dia × 10 cm reach
Cut rate
1 kg Al/hour (Centaur tank skin)
Yield
70% (crude ingots)
Nozzle life
200 hours
Sensing
Time-gated: 100 Hz arc pulse, 1 ms IR exposure synced to off-cycle
Waste
Slag glass (SiO₂) cast into insulating nozzle bodies
Evolution trigger
Yield exceeds 90% → prints Gen 1 upgrades
 
Gen 1: Self-Forged (25 kg)
 
Power
15 kW (salvaged solar cells)
Nozzle
Al + 5% tungsten carbide grit (from dead sats)
Cut rate
5 kg Al/hour
Yield
80% (gradient-purified ingots)
Nozzle life
500 hours
Sensing
Spectral filtering: 900–1100 nm band isolates Al melt line from plasma
New capability
Wire-arc AM printing (2 kg/hr). IR auto-tracking.
Print success
~50% (print 5 nozzles → 2–3 survive). 60% material recovery on failures.
Reproduction
Each Gen 1 cuts material for 2× Gen 1. Doubles monthly.
 
Gen 2: Swarm-Specialized (20 kg)
 
Power
25 kW (TPV recovered from cuts)
Nozzle
Metamaterial-printed (hot gates geometry)
Cut rate
20 kg Al/hour
Yield
88% (infiltration-densified parts, 99% dense)
Nozzle life
1,000 hours
Sensing
Dual-wavelength ratiometric (λ1=800nm, λ2=1200nm). Works through 50% plume opacity.
Specialization
Forks into radiator cutter (1mm precision), compute reflower (300°C zone), frame welder (10kV e-beam)
Print success
~70%
 
Gen 3: Apex Drone (15 kg)
 
Power
40 kW (swarm thermal bus direct)
Nozzle
Self-healing (Al matrix + polymer veins from composites)
Cut rate
50 kg Al/hour
Yield
95% (100% material closure)
Nozzle life
5,000+ hours (microwave plasma, no electrode erosion)
Key physics change
Microwave plasma sustainment. Electrodeless = no tungsten contamination in feedstock. Required for downstream precision (chip reflow, metamaterial printing).
Endgame form
Mobile mining drone: grapple + torch + printer in 50 kg package. Carries thermal gates. Prints new swarm nodes while flying formation.
 
8.1 Metallurgy Solutions
Gradient Purification
Hot zone at 2,000°C vaporizes Fe/Cu impurities. Cold zone at 800°C condenses Al first (vapor pressure differential). Yields 85% pure Al ingots from 95% scrap. Nature analog: termite wet sieving, applied thermally.
 
Porosity Knockout
Print nozzles with 20% void fraction. Infiltrate with molten Al via capillary action. Result: 99% dense final part. Zero waste (infiltrant from same melt). Nature analog: wood heartwood self-densification.

9. Compute Rad-Hardening Co-Evolution

Salvaged chips (130–65 nm CMOS from 1990s–2010s dead sats) fail in weeks under LEO radiation without mitigation. The compute and torch systems co-evolve: neither survives without the other.
 
Technique
Implementation
Triple modular redundancy
3 identical chips → majority vote. SEU rate drops from 10⁻⁶ to 10⁻¹⁸ effective. 3× mass.
Boron carbide potting
Salvaged composites + thermal blankets. ¹⁰B(n,α) captures thermal neutrons. 1 mil reduces soft error rate 100×.
Clock throttling
20 MHz vs spec 200 MHz. Bit flip window shrinks 10×. 10× slower but survives.
Annealing cycles
125°C bake every 30 days (thermal bus). Resets total dose lattice damage. 80% chip recovery per cycle.
Junk shielding
5 cm Al plate (first plasma cuts). 10 g/cm². Halves heavy ion flux.
 
Net result: 65 nm SRAM survives 5+ years in LEO with 10× mass penalty. Acceptable for seed phase.
 
9.1 Compute Evolution Path
Gen 0: TMR + throttling → 10 MIPS, survives 6 months.
Gen 1: Print first rad-hard FPGA from pure Al gates → 100 MIPS.
Gen 2: Chip reflow + error correction → 1 GIPS.
Gen 3: Metamaterial gates → radiation-immune compute.

10. Propulsion and Δv Co-Evolution

Mining 100 targets in LEO requires ~30 km/s cumulative Δv. Chemical propulsion is dead. Multi-stage propulsion from junk solves this.
 
Stage
Implementation
Stage 1: Cold gas
Scavenged Xe/N₂ from dead comsats. Isp 70s. 1 km/s per 10 kg Xe. First 10 targets.
Stage 2: Hall thruster
Salvaged solar arrays (100 m²) → 10 kW → 1 N. Isp 2,000s. 20 km/s per 100 kg Xe.
Stage 3: Thermal resistojet
Radiator waste heat → water resistojet. Isp 200s. Continuous station-keeping.
Stage 4: Aerobraking
De-orbit dead sats → atmospheric graze → free Δv for capture reboost.

11. Bio-Inspired Grappler Design

Five-layer capture system inspired by spider webs, gecko adhesion, frog tongues, and ant chemical signaling. Total mass: 37 kg. Capture volume: 5 m diameter. 85% success probability.
 
Layer
Specification
Layer 1: Web
Kevlar/carbon net, 5 m dia (20 m²). Spin-deployed. 8 kg. Spider orb-weaver analog.
Layer 2: Electroadhesion
10×10 cm electrodes, 1 kV DC. 10 N/cm² on rough Al. Works on any surface. 15 kg. Gecko setae analog.
Layer 3: Magnetic pulse
4× coilgun darts with neodymium cores. 1 kJ pulse. For ferrous targets. 12 kg. Frog tongue analog.
Layer 4: Chemical tag
UV-fluorescent polymer dots. Survives plasma torch. Track cut pieces. 2 kg. Ant pheromone analog.
Layer 5: Thermal guidance
Waste heat warms grapple arm → melts frost/ablates oxide → improves adhesion. IR beacon illuminates target.

12. Bootstrap Economics

 
12.1 Investment
Item
Cost
Seed launch (rideshare)
$1–5M
Monthly ground ops
$50k/month
Total to self-funding
$5–8M
 
12.2 Revenue Streams
Stream
Description
Compute sales
$10–15k/MW/month
Thermal leasing
Excess cooling capacity → bilateral trades
Debris cleanup credits
$1M/ton (FCC/FAA Space Act Agreement)
Insurance credits
Path clearance for active operators
Calibration services
Known IR signature for sensor alignment
 
12.3 Projected Timeline
Milestone
Status
Month 1
1 seed torch → 10 tons cut → first Gen 1 torch
Month 3
4–8 torches → Gen 1 online → exponential cutting begins
Month 6
16 torches → first 100 kW node operational → revenue starts
Month 12
64–256 torches → full 10 MW swarm → self-funding
Month 18
Second-generation swarm → modern compute shipped into existing thermal envelope
 
12.4 Scaling Economics
Mass scales linearly. Utility compounds geometrically through sharing. Each new swarm node adds to the thermal market and the ranging mesh. 10 MW requires 10 federated swarms. 100 MW requires inter-swarm lunar relay. At no point does fundamental architecture change — only the number of loosely coupled participants.
 
Critical coupling: No rad-hard compute = no plasma evolution. No plasma evolution = no shielding/propulsion = no Δv budget. Both unlock simultaneously at the Month 2–3 inflection point. The seed survives its first winter through statistical redundancy (TMR + shielding + throttling), then explodes into exponential growth.

13. Design Principles

 
No single-use gradients. Every major temperature drop (electrical → thermal, thermal → radiative) carries at least one piggyback service.
 
Match scales. Big slow gradients (bulk radiator, orbital day/night) serve big slow needs (structure, tanks). Small fast gradients (hot gates, pulsed loops) serve signaling and attitude trims.
 
Local closure. Clusters share compute, power, and thermal plumbing so that waste from one module is directly useful to another before hitting the global sink.
 
Information from waste. The thermal maze itself is a sensor network. Flow patterns encode health, workload, and environment.
 
Geometry does the coherence work. Engineered phase space (horns, troughs, corrugations) instead of brute-force material coherence. Passive structures that shape emission without active phasing.
 
Eat your waste, excrete structure. Orbital junk is regolith. The seed factory is a genome. The thermal economy is natural selection. The swarm is an ecosystem. Death feeds new life.
 
 
Build this.
