"""
Potassium cycling equations: from rock and waste to soil.

Models all natural and recoverable potassium pathways:
    - Rock weathering rates (geology-dependent)
    - Ocean spray deposition (distance from coast)
    - Wood ash availability (from biomass)
    - Human waste recovery
    - Crop removal vs return balance

Potassium is the third macronutrient. It's abundant on Earth.
The "shortage" is a supply chain problem, not a physics problem.
"""

import numpy as np
from .nutrient_constants import (
    ROCK_WEATHERING_K_TYPICAL_KG_HA_YR,
    ROCK_WEATHERING_K_LOW_KG_HA_YR,
    ROCK_WEATHERING_K_HIGH_KG_HA_YR,
    OCEAN_SPRAY_K_COASTAL_KG_HA_YR,
    OCEAN_SPRAY_K_LOW_KG_HA_YR,
    WOOD_ASH_K_FRACTION,
    WOOD_ASH_YIELD_KG_PER_TONNE_WOOD,
    CROP_K_REMOVAL_TYPICAL_KG_HA_YR,
    HUMAN_K_EXCRETION_KG_YR,
    HUMAN_K_EXCRETION_G_DAY,
    BASALT_K_FRACTION,
    GRANITE_K_FRACTION,
    DAYS_PER_YEAR,
    HA_TO_M2,
)


class PotassiumCyclingModel:
    """
    Calculates available potassium from natural and recoverable sources.
    """

    def __init__(
        self,
        land_area_ha,
        population=0,
        distance_to_coast_km=100.0,
        wood_biomass_available_tonnes_yr=0.0,
        rock_type='mixed',
        rock_dust_available_tonnes_yr=0.0,
        crop_residue_return_fraction=0.50,
        urine_collection_fraction=0.0,
    ):
        """
        Initialize potassium cycling model.

        Args:
            land_area_ha: Agricultural land area in hectares.
            population: Community population (for waste K recovery).
            distance_to_coast_km: Distance to nearest coast.
            wood_biomass_available_tonnes_yr: Wood burned for ash per year.
            rock_type: 'basalt', 'granite', or 'mixed'.
            rock_dust_available_tonnes_yr: Crushed rock applied per year.
            crop_residue_return_fraction: Fraction of crop residues returned
                to field (vs. removed/burned).
            urine_collection_fraction: Fraction of urine collected for
                fertilizer use (0-1).
        """
        self.land_area_ha = land_area_ha
        self.population = population
        self.distance_to_coast_km = distance_to_coast_km
        self.wood_biomass_available_tonnes_yr = wood_biomass_available_tonnes_yr
        self.rock_type = rock_type
        self.rock_dust_available_tonnes_yr = rock_dust_available_tonnes_yr
        self.crop_residue_return_fraction = crop_residue_return_fraction
        self.urine_collection_fraction = urine_collection_fraction

    def rock_weathering_kg_yr(self):
        """
        Potassium released from natural rock weathering.

        Rate depends on rock type, climate, and rainfall.
        Granite weathers slower than basalt.

        Returns:
            dict with kg_k_per_year and details.
        """
        rate = ROCK_WEATHERING_K_TYPICAL_KG_HA_YR
        total = rate * self.land_area_ha

        return {
            'kg_k_per_year': total,
            'rate_kg_ha_yr': rate,
            'range': (ROCK_WEATHERING_K_LOW_KG_HA_YR,
                      ROCK_WEATHERING_K_HIGH_KG_HA_YR),
            'source': 'White & Brantley 2003',
        }

    def ocean_spray_deposition_kg_yr(self):
        """
        Potassium from sea spray aerosols.

        Significant within ~50 km of coast, negligible inland.
        Exponential decay with distance.

        Returns:
            dict with kg_k_per_year and details.
        """
        # Exponential decay: rate = max_rate * exp(-distance/scale)
        scale_km = 30.0  # characteristic decay distance
        if self.distance_to_coast_km < 200:
            rate = OCEAN_SPRAY_K_COASTAL_KG_HA_YR * np.exp(
                -self.distance_to_coast_km / scale_km
            )
        else:
            rate = 0.0

        total = rate * self.land_area_ha

        return {
            'kg_k_per_year': total,
            'rate_kg_ha_yr': rate,
            'distance_to_coast_km': self.distance_to_coast_km,
            'coastal_max_rate': OCEAN_SPRAY_K_COASTAL_KG_HA_YR,
            'source': 'Stallard & Edmond 1981',
        }

    def wood_ash_kg_yr(self):
        """
        Potassium from wood ash.

        Hardwood ash is ~4% K. One tonne of wood yields ~10 kg ash.
        So 1 tonne wood → ~0.4 kg K.

        Returns:
            dict with kg_k_per_year and details.
        """
        ash_produced = (
            self.wood_biomass_available_tonnes_yr
            * WOOD_ASH_YIELD_KG_PER_TONNE_WOOD
        )
        k_from_ash = ash_produced * WOOD_ASH_K_FRACTION

        return {
            'kg_k_per_year': k_from_ash,
            'ash_produced_kg_yr': ash_produced,
            'wood_tonnes_yr': self.wood_biomass_available_tonnes_yr,
            'ash_yield_kg_per_tonne': WOOD_ASH_YIELD_KG_PER_TONNE_WOOD,
            'k_fraction_in_ash': WOOD_ASH_K_FRACTION,
            'source': 'Etiegni & Campbell 1991',
        }

    def rock_dust_kg_yr(self):
        """
        Potassium from crushed rock dust application.

        Basalt: ~0.8% K, Granite: ~3.5% K.
        Slow release over 3-5 years.

        Returns:
            dict with kg_k_per_year and details.
        """
        k_fractions = {
            'basalt': BASALT_K_FRACTION,
            'granite': GRANITE_K_FRACTION,
            'mixed': (BASALT_K_FRACTION + GRANITE_K_FRACTION) / 2.0,
        }
        k_fraction = k_fractions.get(self.rock_type, k_fractions['mixed'])

        total_k = self.rock_dust_available_tonnes_yr * 1000.0 * k_fraction

        # First-year availability: ~10-20% of total K in rock dust
        first_year_release = 0.15
        available_yr1 = total_k * first_year_release

        return {
            'kg_k_per_year': available_yr1,
            'total_k_in_dust_kg': total_k,
            'first_year_release_fraction': first_year_release,
            'rock_type': self.rock_type,
            'k_fraction': k_fraction,
            'rock_dust_tonnes_yr': self.rock_dust_available_tonnes_yr,
            'source': 'Leonardos et al. 1987; Harley & Gilkes 2000',
        }

    def human_waste_kg_yr(self):
        """
        Potassium recoverable from human urine.

        ~90% of excreted K is in urine (not feces).
        Urine diversion is the most efficient K recovery path.

        Returns:
            dict with kg_k_per_year and details.
        """
        gross_k = self.population * HUMAN_K_EXCRETION_KG_YR
        recovered = gross_k * self.urine_collection_fraction

        return {
            'kg_k_per_year': recovered,
            'gross_excretion_kg_yr': gross_k,
            'collection_fraction': self.urine_collection_fraction,
            'per_person_g_day': HUMAN_K_EXCRETION_G_DAY,
            'note': '~90% of K is in urine; urine diversion is most efficient',
            'source': 'Jonsson et al. 2004',
        }

    def crop_residue_return_kg_yr(self):
        """
        Potassium returned via crop residues left on field.

        If you remove all crop residues, you export K.
        If you return them, you recycle ~50-70% of what the crop took up.

        Returns:
            dict with kg_k_per_year (negative = net loss, positive = return).
        """
        total_removal = self.land_area_ha * CROP_K_REMOVAL_TYPICAL_KG_HA_YR
        returned = total_removal * self.crop_residue_return_fraction
        net_loss = total_removal - returned

        return {
            'kg_k_returned_yr': returned,
            'kg_k_removed_yr': total_removal,
            'net_loss_kg_yr': net_loss,
            'return_fraction': self.crop_residue_return_fraction,
            'removal_rate_kg_ha': CROP_K_REMOVAL_TYPICAL_KG_HA_YR,
        }

    def total_potassium_budget(self):
        """
        Complete potassium budget: sources minus crop demand.

        Returns:
            dict with total supply, demand, and balance.
        """
        weathering = self.rock_weathering_kg_yr()
        ocean = self.ocean_spray_deposition_kg_yr()
        ash = self.wood_ash_kg_yr()
        dust = self.rock_dust_kg_yr()
        human = self.human_waste_kg_yr()
        residue = self.crop_residue_return_kg_yr()

        total_supply = (
            weathering['kg_k_per_year']
            + ocean['kg_k_per_year']
            + ash['kg_k_per_year']
            + dust['kg_k_per_year']
            + human['kg_k_per_year']
            + residue['kg_k_returned_yr']
        )

        total_demand = residue['kg_k_removed_yr']
        balance = total_supply - total_demand

        return {
            'total_supply_kg_yr': total_supply,
            'total_demand_kg_yr': total_demand,
            'balance_kg_yr': balance,
            'surplus': balance >= 0,
            'weathering': weathering,
            'ocean_spray': ocean,
            'wood_ash': ash,
            'rock_dust': dust,
            'human_waste': human,
            'crop_residue': residue,
        }

    def report(self):
        """Print a human-readable potassium budget."""
        r = self.total_potassium_budget()
        lines = []
        lines.append("=" * 60)
        lines.append("POTASSIUM BUDGET (kg K/year)")
        lines.append("=" * 60)
        lines.append(f"Land area: {self.land_area_ha:.1f} ha")
        lines.append(f"Population: {self.population:,}")
        lines.append(f"Coast distance: {self.distance_to_coast_km:.0f} km")
        lines.append("-" * 60)
        lines.append("SUPPLY:")
        lines.append(
            f"  Rock weathering:       {r['weathering']['kg_k_per_year']:>10.1f} kg K"
        )
        lines.append(
            f"  Ocean spray:           {r['ocean_spray']['kg_k_per_year']:>10.1f} kg K"
        )
        lines.append(
            f"  Wood ash:              {r['wood_ash']['kg_k_per_year']:>10.1f} kg K"
        )
        lines.append(
            f"  Rock dust (yr1):       {r['rock_dust']['kg_k_per_year']:>10.1f} kg K"
        )
        lines.append(
            f"  Human waste:           {r['human_waste']['kg_k_per_year']:>10.1f} kg K"
        )
        lines.append(
            f"  Crop residue return:   {r['crop_residue']['kg_k_returned_yr']:>10.1f} kg K"
        )
        lines.append(
            f"  TOTAL SUPPLY:          {r['total_supply_kg_yr']:>10.1f} kg K"
        )
        lines.append("-" * 60)
        lines.append("DEMAND:")
        lines.append(
            f"  Crop removal:          {r['total_demand_kg_yr']:>10.1f} kg K"
        )
        lines.append("-" * 60)
        bal = r['balance_kg_yr']
        status = "SURPLUS" if bal >= 0 else "DEFICIT"
        lines.append(f"  BALANCE:               {bal:>10.1f} kg K ({status})")
        lines.append("=" * 60)
        return "\n".join(lines)


def quick_test():
    """Verify potassium cycling calculations."""
    print("Potassium Cycling Model - Quick Test")
    print()

    model = PotassiumCyclingModel(
        land_area_ha=200.0,
        population=5000,
        distance_to_coast_km=30.0,
        wood_biomass_available_tonnes_yr=50.0,
        rock_type='basalt',
        rock_dust_available_tonnes_yr=20.0,
        crop_residue_return_fraction=0.60,
        urine_collection_fraction=0.50,
    )

    print(model.report())

    r = model.total_potassium_budget()
    assert r['total_supply_kg_yr'] > 0, "Total K supply must be positive"
    print("\nAll assertions passed.")


if __name__ == '__main__':
    quick_test()
