"""
Phosphorus recovery equations: from waste to soil.

Models all recoverable phosphorus pathways:
    - Human excretion rates (per capita, measurable)
    - Sewage concentration (kg/m^3, testable)
    - Landfill/dump phosphorus density (tonnes/ha, surveyable)
    - Recovery efficiencies (struvite, ash, chemical, biological)
    - Bioavailability timelines (how fast plants can use it)

The physics: phosphorus hasn't left the planet.
It's in sewage. It's in dumps. It's in rock.
The question is recovery, not scarcity.
"""

import numpy as np
from .nutrient_constants import (
    HUMAN_P_EXCRETION_KG_YR,
    HUMAN_P_EXCRETION_G_DAY,
    SEWAGE_P_CONCENTRATION_KG_M3,
    SEWAGE_VOLUME_L_DAY,
    FOOD_WASTE_P_FRACTION,
    FOOD_WASTE_PER_PERSON_KG_DAY,
    DUMP_P_DENSITY_KG_PER_TONNE,
    DUMP_DENSITY_TONNES_PER_M3,
    RECOVERY_EFF_STRUVITE,
    RECOVERY_EFF_ASH,
    RECOVERY_EFF_CHEMICAL,
    RECOVERY_EFF_BIOLOGICAL,
    RECOVERY_EFF_COMPOSTING,
    P_BIOAVAIL_STRUVITE_YR1,
    P_BIOAVAIL_COMPOST_YR1,
    P_BIOAVAIL_ROCK_PHOSPHATE_YR1,
    P_BIOAVAIL_ASH_YR1,
    CROP_P_REQUIREMENT_KG_HA_YR,
    MOROCCO_PHOSPHATE_FRACTION_GLOBAL,
    DAYS_PER_YEAR,
    HA_TO_M2,
)


class PhosphorusRecoveryModel:
    """
    Calculates recoverable phosphorus from local waste streams.

    Every input is something a community can measure:
    - Count your people
    - Measure your sewage flow
    - Survey your dump
    - Test your soil
    """

    def __init__(
        self,
        population,
        sewage_fraction_collected=0.80,
        food_waste_fraction_collected=0.50,
        dump_volume_m3=0.0,
        dump_age_years=20.0,
        rock_phosphate_available_tonnes=0.0,
        recovery_method='composting',
    ):
        """
        Initialize phosphorus recovery model.

        Args:
            population: Number of people in the community.
            sewage_fraction_collected: Fraction of sewage that is collected
                (vs. dispersed). 1.0 = full collection.
            food_waste_fraction_collected: Fraction of food waste recovered.
            dump_volume_m3: Volume of existing landfill/dump in cubic meters.
            dump_age_years: Age of dump in years (affects decomposition state).
            rock_phosphate_available_tonnes: Local rock phosphate if any.
            recovery_method: One of 'struvite', 'ash', 'chemical',
                'biological', 'composting'.
        """
        self.population = population
        self.sewage_fraction_collected = sewage_fraction_collected
        self.food_waste_fraction_collected = food_waste_fraction_collected
        self.dump_volume_m3 = dump_volume_m3
        self.dump_age_years = dump_age_years
        self.rock_phosphate_available_tonnes = rock_phosphate_available_tonnes
        self.recovery_method = recovery_method

        self._recovery_efficiencies = {
            'struvite': RECOVERY_EFF_STRUVITE,
            'ash': RECOVERY_EFF_ASH,
            'chemical': RECOVERY_EFF_CHEMICAL,
            'biological': RECOVERY_EFF_BIOLOGICAL,
            'composting': RECOVERY_EFF_COMPOSTING,
        }

        self._bioavailability = {
            'struvite': P_BIOAVAIL_STRUVITE_YR1,
            'ash': P_BIOAVAIL_ASH_YR1,
            'chemical': P_BIOAVAIL_ASH_YR1,  # similar to ash
            'biological': P_BIOAVAIL_COMPOST_YR1,
            'composting': P_BIOAVAIL_COMPOST_YR1,
        }

    def sewage_phosphorus_kg_yr(self):
        """
        Phosphorus recoverable from human sewage per year.

        The math:
            population * excretion_rate * collection_fraction * recovery_efficiency

        Returns:
            dict with kg_p_per_year and full calculation trace.
        """
        gross_p = self.population * HUMAN_P_EXCRETION_KG_YR
        collected_p = gross_p * self.sewage_fraction_collected
        recovery_eff = self._recovery_efficiencies[self.recovery_method]
        recovered_p = collected_p * recovery_eff
        bioavail = self._bioavailability[self.recovery_method]
        plant_available_p = recovered_p * bioavail

        return {
            'kg_p_per_year': recovered_p,
            'plant_available_kg_yr': plant_available_p,
            'gross_excretion_kg_yr': gross_p,
            'collected_kg_yr': collected_p,
            'recovery_efficiency': recovery_eff,
            'bioavailability_yr1': bioavail,
            'per_person_g_day': HUMAN_P_EXCRETION_G_DAY,
            'collection_fraction': self.sewage_fraction_collected,
            'method': self.recovery_method,
            'source': 'Jonsson et al. 2004; Metcalf & Eddy',
        }

    def food_waste_phosphorus_kg_yr(self):
        """
        Phosphorus recoverable from food waste per year.

        Returns:
            dict with kg_p_per_year and calculation trace.
        """
        daily_waste_kg = (
            self.population * FOOD_WASTE_PER_PERSON_KG_DAY
            * self.food_waste_fraction_collected
        )
        daily_p_kg = daily_waste_kg * FOOD_WASTE_P_FRACTION
        annual_p_kg = daily_p_kg * DAYS_PER_YEAR

        # Composting retains nearly all P
        recovery_eff = RECOVERY_EFF_COMPOSTING
        recovered = annual_p_kg * recovery_eff

        return {
            'kg_p_per_year': recovered,
            'gross_p_kg_yr': annual_p_kg,
            'recovery_efficiency': recovery_eff,
            'waste_collected_kg_day': daily_waste_kg,
            'p_fraction_in_waste': FOOD_WASTE_P_FRACTION,
            'collection_fraction': self.food_waste_fraction_collected,
            'source': 'USDA food waste composition data',
        }

    def dump_phosphorus_total(self):
        """
        Total phosphorus locked in an existing landfill/dump.

        This is a one-time stock, not a flow.
        Communities can survey dump volume and estimate this.

        Returns:
            dict with total_kg_p and extraction timeline.
        """
        dump_mass_tonnes = self.dump_volume_m3 * DUMP_DENSITY_TONNES_PER_M3
        total_p_kg = dump_mass_tonnes * DUMP_P_DENSITY_KG_PER_TONNE

        recovery_eff = self._recovery_efficiencies.get('chemical', 0.80)

        # Practical extraction rate: ~5-10% of dump per year
        extraction_rate_yr = 0.05  # conservative
        annual_extraction_kg = total_p_kg * recovery_eff * extraction_rate_yr

        years_supply = total_p_kg * recovery_eff / max(annual_extraction_kg, 1e-10)

        return {
            'total_kg_p': total_p_kg,
            'recoverable_kg_p': total_p_kg * recovery_eff,
            'annual_extraction_kg': annual_extraction_kg,
            'dump_mass_tonnes': dump_mass_tonnes,
            'dump_volume_m3': self.dump_volume_m3,
            'p_density_kg_per_tonne': DUMP_P_DENSITY_KG_PER_TONNE,
            'recovery_efficiency': recovery_eff,
            'years_at_extraction_rate': years_supply,
            'extraction_rate_per_year': extraction_rate_yr,
            'source': 'Municipal waste composition surveys',
        }

    def rock_phosphate_kg(self):
        """
        Phosphorus from local rock phosphate deposits.

        Rock phosphate: ~13% P (as P2O5 ~30%, P = 30% * 0.4364)
        Very slow release: ~10% bioavailable in year 1.

        Returns:
            dict with total and annual plant-available P.
        """
        p_fraction_in_rock = 0.13  # ~13% P in phosphate rock
        total_p = self.rock_phosphate_available_tonnes * 1000.0 * p_fraction_in_rock

        return {
            'total_kg_p': total_p,
            'plant_available_yr1_kg': total_p * P_BIOAVAIL_ROCK_PHOSPHATE_YR1,
            'bioavailability_yr1': P_BIOAVAIL_ROCK_PHOSPHATE_YR1,
            'rock_tonnes': self.rock_phosphate_available_tonnes,
            'p_fraction': p_fraction_in_rock,
            'note': 'Slow release; best combined with biological activation',
        }

    def total_recoverable_phosphorus(self):
        """
        Total phosphorus budget from all local sources.

        Returns:
            dict with total flows, stocks, and food security assessment.
        """
        sewage = self.sewage_phosphorus_kg_yr()
        food_waste = self.food_waste_phosphorus_kg_yr()
        dump = self.dump_phosphorus_total()
        rock = self.rock_phosphate_kg()

        # Annual flows (renewable)
        annual_flow = (
            sewage['kg_p_per_year']
            + food_waste['kg_p_per_year']
        )

        # One-time stocks
        total_stock = dump['recoverable_kg_p'] + rock['total_kg_p']

        # How many hectares can this support?
        ha_supported_flow = annual_flow / CROP_P_REQUIREMENT_KG_HA_YR
        ha_supported_stock_yr = (
            dump['annual_extraction_kg'] / CROP_P_REQUIREMENT_KG_HA_YR
        )

        return {
            'annual_flow_kg_p': annual_flow,
            'total_stock_kg_p': total_stock,
            'sewage': sewage,
            'food_waste': food_waste,
            'dump': dump,
            'rock': rock,
            'hectares_supported_by_flow': ha_supported_flow,
            'hectares_supported_by_dump_extraction': ha_supported_stock_yr,
            'crop_p_requirement_kg_ha_yr': CROP_P_REQUIREMENT_KG_HA_YR,
            'narrative_context': {
                'morocco_controls_fraction': MOROCCO_PHOSPHATE_FRACTION_GLOBAL,
                'local_recovery_possible': annual_flow > 0,
                'years_of_dump_supply': dump['years_at_extraction_rate'],
            },
        }

    def report(self):
        """Print a human-readable phosphorus budget."""
        r = self.total_recoverable_phosphorus()
        lines = []
        lines.append("=" * 60)
        lines.append("PHOSPHORUS BUDGET")
        lines.append("=" * 60)
        lines.append(f"Population: {self.population:,}")
        lines.append(f"Recovery method: {self.recovery_method}")
        lines.append("-" * 60)
        lines.append("ANNUAL FLOWS (renewable):")
        lines.append(
            f"  Sewage recovery:       {r['sewage']['kg_p_per_year']:>10.1f} kg P/yr"
        )
        lines.append(
            f"  Food waste recovery:   {r['food_waste']['kg_p_per_year']:>10.1f} kg P/yr"
        )
        lines.append(
            f"  Total annual flow:     {r['annual_flow_kg_p']:>10.1f} kg P/yr"
        )
        lines.append("-" * 60)
        lines.append("STOCKS (one-time, extractable):")
        lines.append(
            f"  Dump total P:          {r['dump']['total_kg_p']:>10.1f} kg P"
        )
        lines.append(
            f"  Dump recoverable:      {r['dump']['recoverable_kg_p']:>10.1f} kg P"
        )
        lines.append(
            f"  Rock phosphate P:      {r['rock']['total_kg_p']:>10.1f} kg P"
        )
        lines.append("-" * 60)
        lines.append("FOOD SECURITY:")
        lines.append(
            f"  Hectares fed by flow:  {r['hectares_supported_by_flow']:>10.1f} ha"
        )
        lines.append(
            f"  Crop P demand:         {r['crop_p_requirement_kg_ha_yr']:>10.1f} kg P/ha/yr"
        )
        lines.append(
            f"  Dump supply years:     {r['dump']['years_at_extraction_rate']:>10.0f} yr"
        )
        lines.append("-" * 60)
        lines.append("NARRATIVE CHECK:")
        lines.append(
            f"  Morocco controls {MOROCCO_PHOSPHATE_FRACTION_GLOBAL*100:.0f}% "
            f"of mined phosphate reserves."
        )
        lines.append(
            f"  This community has {r['annual_flow_kg_p']:.0f} kg P/yr "
            f"recoverable from its own waste."
        )
        lines.append(
            f"  That supports {r['hectares_supported_by_flow']:.1f} ha of crops."
        )
        lines.append("=" * 60)
        return "\n".join(lines)


def quick_test():
    """Verify phosphorus recovery calculations for a small community."""
    print("Phosphorus Recovery Model - Quick Test")
    print()

    # Scenario: 5,000-person town
    model = PhosphorusRecoveryModel(
        population=5000,
        sewage_fraction_collected=0.70,
        food_waste_fraction_collected=0.40,
        dump_volume_m3=50000.0,  # 50,000 m^3 old dump
        dump_age_years=30,
        rock_phosphate_available_tonnes=0.0,
        recovery_method='composting',
    )

    print(model.report())

    r = model.total_recoverable_phosphorus()
    assert r['annual_flow_kg_p'] > 0, "Annual P flow must be positive"
    assert r['dump']['total_kg_p'] > 0, "Dump P must be positive"
    print("\nAll assertions passed.")


if __name__ == '__main__':
    quick_test()
