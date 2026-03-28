"""
Local production capacity calculator.

Given:
    - Population size
    - Land area
    - Current soil state
    - Local waste streams (sewage, dumps)
    - Climate/geography

Calculate:
    - Maximum sustainable food production
    - Limiting nutrient (N, P, or K)
    - Food security threshold (% of food grown locally)
    - Time to reach self-sufficiency

This is the tool a co-op downloads.
They measure their inputs. The equations tell them what's possible.
"""

import numpy as np
from .nitrogen_fixation import NitrogenFixationModel
from .phosphorus_recovery import PhosphorusRecoveryModel
from .potassium_cycling import PotassiumCyclingModel
from .soil_biology import SoilBiologyCascade
from .nutrient_constants import (
    HUMAN_CALORIES_PER_DAY,
    DAYS_PER_YEAR,
    WHEAT_YIELD_TONNES_HA,
    WHEAT_N_FRACTION,
    WHEAT_P_FRACTION,
    WHEAT_K_FRACTION,
    WHEAT_CALORIES_PER_KG,
    RICE_YIELD_TONNES_HA,
    RICE_CALORIES_PER_KG,
    POTATO_YIELD_TONNES_HA,
    POTATO_CALORIES_PER_KG,
    CROP_P_REQUIREMENT_KG_HA_YR,
    HA_TO_M2,
)


class LocalProductionCapacity:
    """
    Integrated food security calculator for a local community.

    Connects all nutrient models to answer the question:
    "Can this community feed itself from local resources?"
    """

    def __init__(
        self,
        population,
        land_area_ha,
        current_som_percent=1.5,
        legume_fraction=0.25,
        wetland_fraction=0.0,
        annual_rainfall_mm=800.0,
        latitude_deg=40.0,
        distance_to_coast_km=100.0,
        sewage_fraction_collected=0.70,
        food_waste_fraction_collected=0.40,
        dump_volume_m3=0.0,
        compost_available_tonnes_yr=0.0,
        wood_biomass_available_tonnes_yr=0.0,
        rock_dust_available_tonnes_yr=0.0,
        urine_collection_fraction=0.0,
        crop_residue_return_fraction=0.50,
        tillage='reduced',
        primary_crop='wheat',
    ):
        """
        Initialize with all measurable local parameters.

        Args:
            population: Number of people to feed.
            land_area_ha: Available agricultural land in hectares.
            current_som_percent: Current soil organic matter (%).
            legume_fraction: Fraction of land in legume rotation.
            wetland_fraction: Fraction of land that is wetland.
            annual_rainfall_mm: Annual rainfall.
            latitude_deg: Latitude (affects lightning, climate).
            distance_to_coast_km: Distance to coast (affects K deposition).
            sewage_fraction_collected: Fraction of sewage collected.
            food_waste_fraction_collected: Fraction of food waste recovered.
            dump_volume_m3: Volume of existing dump/landfill.
            compost_available_tonnes_yr: Compost produced per year.
            wood_biomass_available_tonnes_yr: Wood available for ash.
            rock_dust_available_tonnes_yr: Crushed rock available.
            urine_collection_fraction: Fraction of urine collected.
            crop_residue_return_fraction: Residues returned to field.
            tillage: Tillage practice.
            primary_crop: 'wheat', 'rice', or 'potato'.
        """
        self.population = population
        self.land_area_ha = land_area_ha
        self.primary_crop = primary_crop

        # Build sub-models
        self.nitrogen = NitrogenFixationModel(
            land_area_ha=land_area_ha,
            legume_fraction=legume_fraction,
            wetland_fraction=wetland_fraction,
            annual_rainfall_mm=annual_rainfall_mm,
            compost_available_tonnes_yr=compost_available_tonnes_yr,
            latitude_deg=latitude_deg,
        )

        self.phosphorus = PhosphorusRecoveryModel(
            population=population,
            sewage_fraction_collected=sewage_fraction_collected,
            food_waste_fraction_collected=food_waste_fraction_collected,
            dump_volume_m3=dump_volume_m3,
            recovery_method='composting',
        )

        self.potassium = PotassiumCyclingModel(
            land_area_ha=land_area_ha,
            population=population,
            distance_to_coast_km=distance_to_coast_km,
            wood_biomass_available_tonnes_yr=wood_biomass_available_tonnes_yr,
            rock_dust_available_tonnes_yr=rock_dust_available_tonnes_yr,
            crop_residue_return_fraction=crop_residue_return_fraction,
            urine_collection_fraction=urine_collection_fraction,
        )

        self.soil = SoilBiologyCascade(
            land_area_ha=land_area_ha,
            current_som_percent=current_som_percent,
            cover_crop_fraction=legume_fraction,
            compost_input_tonnes_ha_yr=(
                compost_available_tonnes_yr / max(land_area_ha, 1e-10)
            ),
            crop_residue_return_fraction=crop_residue_return_fraction,
            tillage=tillage,
        )

        # Crop parameters
        self._crops = {
            'wheat': {
                'yield_tonnes_ha': WHEAT_YIELD_TONNES_HA,
                'n_fraction': WHEAT_N_FRACTION,
                'p_fraction': WHEAT_P_FRACTION,
                'k_fraction': WHEAT_K_FRACTION,
                'calories_per_kg': WHEAT_CALORIES_PER_KG,
            },
            'rice': {
                'yield_tonnes_ha': RICE_YIELD_TONNES_HA,
                'n_fraction': 0.012,
                'p_fraction': 0.003,
                'k_fraction': 0.003,
                'calories_per_kg': RICE_CALORIES_PER_KG,
            },
            'potato': {
                'yield_tonnes_ha': POTATO_YIELD_TONNES_HA,
                'n_fraction': 0.003,
                'p_fraction': 0.0006,
                'k_fraction': 0.005,
                'calories_per_kg': POTATO_CALORIES_PER_KG,
            },
        }

    def caloric_demand(self):
        """
        Total caloric demand for the population.

        Returns:
            dict with annual caloric demand.
        """
        annual_kcal = self.population * HUMAN_CALORIES_PER_DAY * DAYS_PER_YEAR

        return {
            'annual_kcal': annual_kcal,
            'daily_kcal_per_person': HUMAN_CALORIES_PER_DAY,
            'population': self.population,
        }

    def nutrient_limited_hectares(self):
        """
        How many hectares can each nutrient support?

        The limiting nutrient determines maximum production.
        Liebig's law of the minimum.

        Returns:
            dict with hectares supported by each nutrient and limiting factor.
        """
        crop = self._crops[self.primary_crop]

        # Get nutrient budgets
        n_budget = self.nitrogen.total_available_nitrogen()
        p_budget = self.phosphorus.total_recoverable_phosphorus()
        k_budget = self.potassium.total_potassium_budget()

        # Nutrient demand per hectare (based on crop removal)
        yield_kg_ha = crop['yield_tonnes_ha'] * 1000.0
        n_demand_ha = yield_kg_ha * crop['n_fraction']
        p_demand_ha = yield_kg_ha * crop['p_fraction']
        k_demand_ha = yield_kg_ha * crop['k_fraction']

        # Hectares each nutrient can support
        n_available = n_budget['total_kg_n_per_year']
        p_available = p_budget['annual_flow_kg_p']
        k_available = k_budget['total_supply_kg_yr']

        ha_by_n = n_available / max(n_demand_ha, 1e-10)
        ha_by_p = p_available / max(p_demand_ha, 1e-10)
        ha_by_k = k_available / max(k_demand_ha, 1e-10)

        # Limiting nutrient (Liebig's law)
        ha_limited = min(ha_by_n, ha_by_p, ha_by_k)
        if ha_limited == ha_by_n:
            limiting = 'nitrogen'
        elif ha_limited == ha_by_p:
            limiting = 'phosphorus'
        else:
            limiting = 'potassium'

        # Also limited by actual land area
        ha_actual = min(ha_limited, self.land_area_ha)

        return {
            'ha_by_nitrogen': ha_by_n,
            'ha_by_phosphorus': ha_by_p,
            'ha_by_potassium': ha_by_k,
            'ha_nutrient_limited': ha_limited,
            'ha_actual': ha_actual,
            'limiting_nutrient': limiting,
            'land_limited': self.land_area_ha < ha_limited,
            'n_available_kg': n_available,
            'p_available_kg': p_available,
            'k_available_kg': k_available,
            'n_demand_kg_ha': n_demand_ha,
            'p_demand_kg_ha': p_demand_ha,
            'k_demand_kg_ha': k_demand_ha,
        }

    def food_security_assessment(self):
        """
        The main output: can this community feed itself?

        Returns:
            dict with food security percentage, limiting factors,
            and recommendations.
        """
        demand = self.caloric_demand()
        nutrients = self.nutrient_limited_hectares()
        soil_state = self.soil.nutrient_cycling_capacity()
        timeline = self.soil.restoration_timeline()
        yields = self.soil.yield_improvement()

        crop = self._crops[self.primary_crop]
        ha_productive = nutrients['ha_actual']

        # Apply soil biology multiplier to yield
        # Current degraded soil reduces effective yield
        bio_multiplier = min(
            soil_state['biology_fraction'] + 0.3, 1.0
        )  # even degraded soil produces something

        effective_yield_kg_ha = (
            crop['yield_tonnes_ha'] * 1000.0 * bio_multiplier
        )
        total_production_kg = ha_productive * effective_yield_kg_ha
        total_calories = total_production_kg * crop['calories_per_kg']

        food_security_pct = (total_calories / demand['annual_kcal']) * 100.0

        # After soil restoration
        future_multiplier = yields['final_yield_multiplier']
        future_calories = (
            ha_productive * crop['yield_tonnes_ha'] * 1000.0
            * future_multiplier * crop['calories_per_kg']
        )
        future_security_pct = (future_calories / demand['annual_kcal']) * 100.0

        # Recommendations
        recommendations = []
        if nutrients['limiting_nutrient'] == 'nitrogen':
            recommendations.append(
                "Increase legume rotation to boost biological N fixation"
            )
        elif nutrients['limiting_nutrient'] == 'phosphorus':
            recommendations.append(
                "Implement sewage P recovery (struvite or composting)"
            )
        else:
            recommendations.append(
                "Apply rock dust or collect wood ash for K"
            )

        if self.soil.current_som_percent < 2.0:
            recommendations.append(
                "Priority: rebuild soil organic matter with cover crops + compost"
            )

        if food_security_pct < 50:
            recommendations.append(
                "Consider high-calorie crops (potato) to maximize food per hectare"
            )

        return {
            'food_security_percent': food_security_pct,
            'food_security_percent_after_restoration': future_security_pct,
            'years_to_full_restoration': timeline['years_to_target'],
            'limiting_nutrient': nutrients['limiting_nutrient'],
            'productive_hectares': ha_productive,
            'effective_yield_kg_ha': effective_yield_kg_ha,
            'total_production_kg': total_production_kg,
            'total_calories_produced': total_calories,
            'calories_needed': demand['annual_kcal'],
            'population': self.population,
            'soil_biology_multiplier': bio_multiplier,
            'future_yield_multiplier': future_multiplier,
            'primary_crop': self.primary_crop,
            'nutrient_hectares': nutrients,
            'recommendations': recommendations,
        }

    def report(self):
        """Print a comprehensive food security report."""
        r = self.food_security_assessment()
        lines = []
        lines.append("=" * 70)
        lines.append("LOCAL FOOD SECURITY ASSESSMENT")
        lines.append("=" * 70)
        lines.append(f"Population: {self.population:,}")
        lines.append(f"Land area: {self.land_area_ha:.1f} ha")
        lines.append(f"Primary crop: {self.primary_crop}")
        lines.append(
            f"Current soil: {self.soil.current_som_percent:.1f}% SOM"
        )
        lines.append("")
        lines.append("-" * 70)
        lines.append("NUTRIENT AVAILABILITY:")
        n = r['nutrient_hectares']
        lines.append(
            f"  Nitrogen:   {n['n_available_kg']:>8.0f} kg/yr "
            f"-> supports {n['ha_by_nitrogen']:>6.1f} ha"
        )
        lines.append(
            f"  Phosphorus: {n['p_available_kg']:>8.0f} kg/yr "
            f"-> supports {n['ha_by_phosphorus']:>6.1f} ha"
        )
        lines.append(
            f"  Potassium:  {n['k_available_kg']:>8.0f} kg/yr "
            f"-> supports {n['ha_by_potassium']:>6.1f} ha"
        )
        lines.append(
            f"  Limiting nutrient: {r['limiting_nutrient'].upper()}"
        )
        lines.append(
            f"  Productive hectares: {r['productive_hectares']:.1f} ha"
        )
        if n['land_limited']:
            lines.append("  (Land-limited, not nutrient-limited)")
        lines.append("")
        lines.append("-" * 70)
        lines.append("PRODUCTION:")
        lines.append(
            f"  Effective yield: {r['effective_yield_kg_ha']:.0f} kg/ha "
            f"(soil bio multiplier: {r['soil_biology_multiplier']:.2f})"
        )
        lines.append(
            f"  Total production: {r['total_production_kg']:,.0f} kg/yr"
        )
        lines.append(
            f"  Total calories: {r['total_calories_produced']:,.0f} kcal/yr"
        )
        lines.append(
            f"  Calories needed: {r['calories_needed']:,.0f} kcal/yr"
        )
        lines.append("")
        lines.append("-" * 70)
        pct = r['food_security_percent']
        lines.append(f"  FOOD SECURITY: {pct:.1f}%")
        if pct >= 100:
            lines.append("  STATUS: SELF-SUFFICIENT")
        elif pct >= 50:
            lines.append("  STATUS: PARTIALLY SECURE (>50%)")
        else:
            lines.append("  STATUS: FOOD INSECURE (<50%)")
        lines.append("")
        lines.append(
            f"  After soil restoration ({r['years_to_full_restoration']} yr): "
            f"{r['food_security_percent_after_restoration']:.1f}%"
        )
        lines.append("")
        lines.append("-" * 70)
        lines.append("RECOMMENDATIONS:")
        for i, rec in enumerate(r['recommendations'], 1):
            lines.append(f"  {i}. {rec}")
        lines.append("=" * 70)
        return "\n".join(lines)


def quick_test():
    """Run a full local capacity assessment."""
    print("Local Production Capacity - Quick Test")
    print()

    # Scenario: 2,000-person rural community
    model = LocalProductionCapacity(
        population=2000,
        land_area_ha=300.0,
        current_som_percent=1.5,
        legume_fraction=0.25,
        annual_rainfall_mm=900.0,
        latitude_deg=35.0,
        distance_to_coast_km=50.0,
        sewage_fraction_collected=0.60,
        food_waste_fraction_collected=0.40,
        dump_volume_m3=10000.0,
        compost_available_tonnes_yr=50.0,
        wood_biomass_available_tonnes_yr=30.0,
        rock_dust_available_tonnes_yr=10.0,
        urine_collection_fraction=0.30,
        crop_residue_return_fraction=0.60,
        tillage='no_till',
        primary_crop='wheat',
    )

    print(model.report())

    r = model.food_security_assessment()
    assert r['food_security_percent'] > 0, "Must produce some food"
    assert r['limiting_nutrient'] in ('nitrogen', 'phosphorus', 'potassium')
    print("\nAll assertions passed.")


if __name__ == '__main__':
    quick_test()
