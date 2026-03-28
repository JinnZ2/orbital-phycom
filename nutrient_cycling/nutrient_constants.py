"""
Physical and biogeochemical constants for nutrient cycling.

All values in SI units unless otherwise noted.
Sources cited inline. All constants are independently verifiable.
"""

# =============================================================================
# NITROGEN CONSTANTS
# =============================================================================

# Lightning fixation
# ~5 Tg N/yr globally (Galloway et al., 2004)
# Earth land area: 1.49e14 m^2
LIGHTNING_N_FIXATION_GLOBAL_KG_YR = 5e9          # kg N/yr globally
EARTH_LAND_AREA_M2 = 1.49e14                      # m^2
LIGHTNING_N_FIXATION_KG_M2_YR = (
    LIGHTNING_N_FIXATION_GLOBAL_KG_YR / EARTH_LAND_AREA_M2
)  # ~3.4e-5 kg N/m^2/yr

# Biological nitrogen fixation (BNF) rates
# Symbiotic (legumes): 50-300 kg N/ha/yr (Peoples et al., 2009)
BNF_LEGUME_LOW_KG_HA_YR = 50.0
BNF_LEGUME_HIGH_KG_HA_YR = 300.0
BNF_LEGUME_TYPICAL_KG_HA_YR = 150.0

# Free-living soil bacteria: 5-20 kg N/ha/yr (Cleveland et al., 1999)
BNF_FREELIVING_LOW_KG_HA_YR = 5.0
BNF_FREELIVING_HIGH_KG_HA_YR = 20.0
BNF_FREELIVING_TYPICAL_KG_HA_YR = 10.0

# Cyanobacteria (rice paddies, wetlands): 20-50 kg N/ha/yr
BNF_CYANOBACTERIA_LOW_KG_HA_YR = 20.0
BNF_CYANOBACTERIA_HIGH_KG_HA_YR = 50.0
BNF_CYANOBACTERIA_TYPICAL_KG_HA_YR = 30.0

# Atmospheric wet deposition (NH4+ and NO3- in rainfall)
# 2-10 kg N/ha/yr depending on region (Dentener et al., 2006)
ATMOS_DEPOSITION_LOW_KG_HA_YR = 2.0
ATMOS_DEPOSITION_HIGH_KG_HA_YR = 10.0
ATMOS_DEPOSITION_TYPICAL_KG_HA_YR = 5.0

# Compost nitrogen content
# Finished compost: 1-3% N by dry weight (USDA)
COMPOST_N_FRACTION_LOW = 0.01
COMPOST_N_FRACTION_HIGH = 0.03
COMPOST_N_FRACTION_TYPICAL = 0.02

# Compost mineralization rate (fraction available in first year)
COMPOST_MINERALIZATION_RATE_YR1 = 0.10  # 10% of total N available year 1
COMPOST_MINERALIZATION_RATE_YR2 = 0.05  # 5% in year 2

# Haber-Bosch reference: ~1.2 tonnes natural gas per tonne NH3
HABER_BOSCH_GAS_PER_NH3_KG = 1200.0  # kg natural gas per 1000 kg NH3
HABER_BOSCH_ENERGY_MJ_PER_KG_N = 35.0  # MJ per kg N (best practice)

# =============================================================================
# PHOSPHORUS CONSTANTS
# =============================================================================

# Human phosphorus excretion
# ~1.0-1.5 g P/person/day (Jonsson et al., 2004)
HUMAN_P_EXCRETION_G_DAY = 1.2  # grams P per person per day
HUMAN_P_EXCRETION_KG_YR = HUMAN_P_EXCRETION_G_DAY * 365.25 / 1000.0

# Sewage phosphorus concentration
# Raw sewage: 5-15 mg P/L (Metcalf & Eddy)
SEWAGE_P_CONCENTRATION_MG_L = 10.0  # mg P per liter
SEWAGE_P_CONCENTRATION_KG_M3 = SEWAGE_P_CONCENTRATION_MG_L / 1e6  # kg P/m^3

# Daily sewage volume per person
SEWAGE_VOLUME_L_DAY = 200.0  # liters per person per day

# Phosphorus in municipal solid waste (food waste, organics)
# ~0.3% P by wet weight in food waste
FOOD_WASTE_P_FRACTION = 0.003  # kg P per kg wet food waste
FOOD_WASTE_PER_PERSON_KG_DAY = 0.3  # kg food waste per person per day

# Landfill/dump phosphorus density
# Decomposed municipal waste: ~1-5 kg P/tonne (variable)
DUMP_P_DENSITY_KG_PER_TONNE = 2.0  # kg P per tonne of waste
DUMP_DENSITY_TONNES_PER_M3 = 0.8  # compacted waste density

# Phosphorus recovery efficiencies
RECOVERY_EFF_STRUVITE = 0.90    # struvite precipitation from sewage
RECOVERY_EFF_ASH = 0.70         # from sewage sludge incineration ash
RECOVERY_EFF_CHEMICAL = 0.80    # acid/alkali extraction from waste
RECOVERY_EFF_BIOLOGICAL = 0.50  # enhanced biological phosphorus removal
RECOVERY_EFF_COMPOSTING = 0.95  # composting (retains nearly all P)

# Bioavailability timelines (fraction plant-available)
P_BIOAVAIL_STRUVITE_YR1 = 0.80   # struvite: 80% available year 1
P_BIOAVAIL_COMPOST_YR1 = 0.40    # compost: 40% year 1
P_BIOAVAIL_ROCK_PHOSPHATE_YR1 = 0.10  # rock phosphate: 10% year 1
P_BIOAVAIL_ASH_YR1 = 0.60        # ash: 60% year 1

# Crop phosphorus requirements
CROP_P_REQUIREMENT_KG_HA_YR = 20.0  # typical cereal crop removal

# Morocco phosphate reserve (for narrative context)
MOROCCO_PHOSPHATE_FRACTION_GLOBAL = 0.70  # ~70% of known reserves

# =============================================================================
# POTASSIUM CONSTANTS
# =============================================================================

# Rock weathering rates
# Granite/gneiss: 0.5-5 kg K/ha/yr (White & Brantley, 2003)
ROCK_WEATHERING_K_LOW_KG_HA_YR = 0.5
ROCK_WEATHERING_K_HIGH_KG_HA_YR = 5.0
ROCK_WEATHERING_K_TYPICAL_KG_HA_YR = 2.0

# Ocean spray deposition (coastal areas)
# 1-10 kg K/ha/yr within 50 km of coast (Stallard & Edmond, 1981)
OCEAN_SPRAY_K_LOW_KG_HA_YR = 1.0
OCEAN_SPRAY_K_HIGH_KG_HA_YR = 10.0
OCEAN_SPRAY_K_COASTAL_KG_HA_YR = 5.0

# Wood ash potassium content
# Hardwood ash: 3-7% K2O by weight (~2.5-5.8% K)
WOOD_ASH_K_FRACTION = 0.04  # ~4% K by weight
WOOD_ASH_YIELD_KG_PER_TONNE_WOOD = 10.0  # kg ash per tonne of wood

# Crop potassium removal
CROP_K_REMOVAL_CEREAL_KG_HA_YR = 30.0   # wheat/rice
CROP_K_REMOVAL_POTATO_KG_HA_YR = 150.0  # potatoes (high K demand)
CROP_K_REMOVAL_TYPICAL_KG_HA_YR = 50.0

# Human potassium excretion
HUMAN_K_EXCRETION_G_DAY = 2.5  # grams K per person per day
HUMAN_K_EXCRETION_KG_YR = HUMAN_K_EXCRETION_G_DAY * 365.25 / 1000.0

# Rock dust potassium content (basalt, granite)
BASALT_K_FRACTION = 0.008   # 0.8% K in basite
GRANITE_K_FRACTION = 0.035  # 3.5% K in granite

# =============================================================================
# SOIL BIOLOGY CONSTANTS
# =============================================================================

# Microbial biomass carbon
# Healthy soil: 200-500 mg C/kg soil (Jenkinson & Ladd, 1981)
MICROBIAL_BIOMASS_C_HEALTHY_MG_KG = 350.0
MICROBIAL_BIOMASS_C_DEGRADED_MG_KG = 50.0
MICROBIAL_BIOMASS_C_TARGET_MG_KG = 300.0

# Soil organic matter (SOM)
# Healthy: 3-6% SOM, degraded: 0.5-1.5% SOM
SOM_HEALTHY_FRACTION = 0.04    # 4% target
SOM_DEGRADED_FRACTION = 0.01   # 1% typical degraded
SOM_CARBON_FRACTION = 0.58     # Van Bemmelen factor: SOM is ~58% C

# Soil bulk density
SOIL_BULK_DENSITY_KG_M3 = 1300.0  # typical
SOIL_DEPTH_M = 0.30               # topsoil depth for calculations

# Carbon input needed to rebuild SOM
# Each 1% increase in SOM over 30cm depth ≈ 38 tonnes C/ha
SOM_CARBON_PER_PERCENT_TONNES_HA = 38.0

# Soil restoration timelines
SOIL_RESTORATION_YEARS_MINIMUM = 3.0   # measurable improvement
SOIL_RESTORATION_YEARS_FUNCTIONAL = 7.0  # functional biology restored
SOIL_RESTORATION_YEARS_FULL = 20.0     # near-full ecosystem recovery

# Cover crop carbon input
COVER_CROP_C_INPUT_TONNES_HA_YR = 2.0  # typical above+below ground

# Yield improvement per unit soil health
# ~1-3% yield increase per 1% SOM increase (Lal, 2020)
YIELD_INCREASE_PER_SOM_PERCENT = 0.02  # 2% yield gain per 1% SOM

# =============================================================================
# FOOD AND AGRICULTURAL CONSTANTS
# =============================================================================

# Human nutritional requirements
HUMAN_CALORIES_PER_DAY = 2200.0  # kcal/day average
HUMAN_N_REQUIREMENT_G_DAY = 10.0  # ~62.5g protein * 16% N

# Crop yields and nutrient content
# Wheat: ~3 tonnes/ha, ~2% N, ~0.4% P, ~0.5% K (grain)
WHEAT_YIELD_TONNES_HA = 3.0
WHEAT_N_FRACTION = 0.02
WHEAT_P_FRACTION = 0.004
WHEAT_K_FRACTION = 0.005
WHEAT_CALORIES_PER_KG = 3400.0  # kcal/kg

# Rice: ~4 tonnes/ha
RICE_YIELD_TONNES_HA = 4.0
RICE_N_FRACTION = 0.012
RICE_P_FRACTION = 0.003
RICE_K_FRACTION = 0.003
RICE_CALORIES_PER_KG = 3600.0

# Potato: ~20 tonnes/ha (fresh weight)
POTATO_YIELD_TONNES_HA = 20.0
POTATO_N_FRACTION = 0.003
POTATO_P_FRACTION = 0.0006
POTATO_K_FRACTION = 0.005
POTATO_CALORIES_PER_KG = 770.0

# Hectares per square meter
HA_TO_M2 = 10000.0
M2_TO_HA = 1.0 / HA_TO_M2

# =============================================================================
# ECONOMIC CONSTANTS (for comparison only)
# =============================================================================

# Synthetic fertilizer costs (approximate, USD/tonne, 2024)
COST_UREA_USD_TONNE = 350.0          # ~46% N → ~$761/tonne N
COST_DAP_USD_TONNE = 600.0           # diammonium phosphate
COST_POTASH_USD_TONNE = 300.0        # muriate of potash (KCl)
COST_NATURAL_GAS_USD_MMBTU = 3.0     # US benchmark

# Biological fix costs (rough estimates)
COST_COMPOST_SYSTEM_USD_PER_PERSON = 50.0  # community-scale
COST_STRUVITE_REACTOR_USD_PER_PERSON = 200.0  # sewage P recovery
COST_COVER_CROP_SEED_USD_HA = 100.0  # annual cost

# Seconds per year (for rate conversions)
SECONDS_PER_YEAR = 365.25 * 86400.0
DAYS_PER_YEAR = 365.25
