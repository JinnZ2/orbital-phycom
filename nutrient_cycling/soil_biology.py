"""
Soil biology cascade model.

The living layer that makes everything else work.

Without soil biology:
    - Nutrients don't cycle (they leach or lock up)
    - Water doesn't infiltrate (it runs off)
    - Carbon doesn't accumulate (it oxidizes)
    - Plants can't access rock minerals

With soil biology:
    - Mycorrhizal networks mine phosphorus from rock
    - Free-living bacteria fix nitrogen from air
    - Earthworms create macropores for water
    - Organic matter holds 20x its weight in water

This module calculates:
    - Microbial biomass required to cycle nutrients
    - Carbon inputs needed to rebuild soil organic matter
    - Time to restore functional soil biology
    - Yield improvement per unit of soil health
"""

import numpy as np
from .nutrient_constants import (
    MICROBIAL_BIOMASS_C_HEALTHY_MG_KG,
    MICROBIAL_BIOMASS_C_DEGRADED_MG_KG,
    MICROBIAL_BIOMASS_C_TARGET_MG_KG,
    SOM_HEALTHY_FRACTION,
    SOM_DEGRADED_FRACTION,
    SOM_CARBON_FRACTION,
    SOIL_BULK_DENSITY_KG_M3,
    SOIL_DEPTH_M,
    SOM_CARBON_PER_PERCENT_TONNES_HA,
    SOIL_RESTORATION_YEARS_MINIMUM,
    SOIL_RESTORATION_YEARS_FUNCTIONAL,
    SOIL_RESTORATION_YEARS_FULL,
    COVER_CROP_C_INPUT_TONNES_HA_YR,
    YIELD_INCREASE_PER_SOM_PERCENT,
    HA_TO_M2,
)


class SoilBiologyCascade:
    """
    Models the biological restoration of soil health over time.

    Tracks:
    - Soil organic matter (SOM) accumulation
    - Microbial biomass carbon recovery
    - Nutrient cycling capacity as a function of biology
    - Yield response to soil health improvement
    """

    def __init__(
        self,
        land_area_ha,
        current_som_percent=1.0,
        target_som_percent=4.0,
        cover_crop_fraction=0.50,
        compost_input_tonnes_ha_yr=0.0,
        crop_residue_return_fraction=0.50,
        tillage='reduced',
    ):
        """
        Initialize soil biology model.

        Args:
            land_area_ha: Area under management.
            current_som_percent: Current soil organic matter (%).
                Degraded: ~1%, healthy: ~4-6%.
            target_som_percent: Target SOM (%).
            cover_crop_fraction: Fraction of land with cover crops.
            compost_input_tonnes_ha_yr: External compost applied.
            crop_residue_return_fraction: Fraction of residues returned.
            tillage: 'conventional', 'reduced', or 'no_till'.
        """
        self.land_area_ha = land_area_ha
        self.current_som_percent = current_som_percent
        self.target_som_percent = target_som_percent
        self.cover_crop_fraction = cover_crop_fraction
        self.compost_input_tonnes_ha_yr = compost_input_tonnes_ha_yr
        self.crop_residue_return_fraction = crop_residue_return_fraction
        self.tillage = tillage

        # Tillage affects carbon retention
        self._tillage_retention = {
            'conventional': 0.30,  # 30% of C input retained
            'reduced': 0.50,       # 50% retained
            'no_till': 0.70,       # 70% retained
        }

    def carbon_deficit(self):
        """
        Calculate the carbon gap between current and target SOM.

        Each 1% SOM increase over 30cm depth requires ~38 tonnes C/ha.

        Returns:
            dict with carbon deficit and timeline estimates.
        """
        som_gap = self.target_som_percent - self.current_som_percent
        if som_gap <= 0:
            return {
                'carbon_deficit_tonnes_ha': 0.0,
                'total_deficit_tonnes': 0.0,
                'som_gap_percent': 0.0,
                'status': 'At or above target SOM',
            }

        deficit_per_ha = som_gap * SOM_CARBON_PER_PERCENT_TONNES_HA
        total_deficit = deficit_per_ha * self.land_area_ha

        return {
            'carbon_deficit_tonnes_ha': deficit_per_ha,
            'total_deficit_tonnes': total_deficit,
            'som_gap_percent': som_gap,
            'current_som': self.current_som_percent,
            'target_som': self.target_som_percent,
            'tonnes_c_per_percent_som': SOM_CARBON_PER_PERCENT_TONNES_HA,
        }

    def annual_carbon_input(self):
        """
        Calculate total annual carbon input from all management practices.

        Sources:
        - Cover crops (roots + shoots)
        - Compost (external C input)
        - Crop residues (returned fraction)

        Returns:
            dict with total C input and breakdown.
        """
        retention = self._tillage_retention.get(
            self.tillage, self._tillage_retention['reduced']
        )

        # Cover crop input
        cover_crop_c = (
            self.cover_crop_fraction * self.land_area_ha
            * COVER_CROP_C_INPUT_TONNES_HA_YR
        )

        # Compost input (compost is ~25% C by dry weight)
        compost_c_fraction = 0.25
        compost_c = (
            self.compost_input_tonnes_ha_yr * self.land_area_ha
            * compost_c_fraction
        )

        # Crop residue input (~40% C, ~3 tonnes residue/ha for cereals)
        residue_production = 3.0  # tonnes dry residue/ha/yr (typical)
        residue_c_fraction = 0.40
        residue_c = (
            self.land_area_ha
            * residue_production
            * residue_c_fraction
            * self.crop_residue_return_fraction
        )

        gross_input = cover_crop_c + compost_c + residue_c
        net_retained = gross_input * retention

        return {
            'gross_c_input_tonnes_yr': gross_input,
            'net_retained_tonnes_yr': net_retained,
            'retention_fraction': retention,
            'tillage': self.tillage,
            'cover_crop_c_tonnes': cover_crop_c,
            'compost_c_tonnes': compost_c,
            'residue_c_tonnes': residue_c,
            'net_per_ha_tonnes_yr': net_retained / max(self.land_area_ha, 1e-10),
        }

    def restoration_timeline(self):
        """
        Estimate years to reach target SOM given current management.

        Uses a diminishing returns model: SOM accumulation slows as
        you approach the target (biological equilibrium).

        Returns:
            dict with timeline and year-by-year projection.
        """
        deficit = self.carbon_deficit()
        c_input = self.annual_carbon_input()

        if deficit['carbon_deficit_tonnes_ha'] <= 0:
            return {
                'years_to_target': 0,
                'status': 'Already at target',
                'yearly_som': [self.current_som_percent],
            }

        net_per_ha = c_input['net_per_ha_tonnes_yr']
        if net_per_ha <= 0:
            return {
                'years_to_target': float('inf'),
                'status': 'No net carbon input — cannot restore',
                'yearly_som': [self.current_som_percent],
            }

        # Simulate year by year with diminishing returns
        som = self.current_som_percent
        yearly_som = [som]
        max_years = 50

        for year in range(1, max_years + 1):
            # Diminishing returns: rate decreases as SOM approaches target
            gap_fraction = (
                (self.target_som_percent - som)
                / (self.target_som_percent - self.current_som_percent)
            )
            gap_fraction = max(gap_fraction, 0.0)

            # SOM increase this year
            som_increase = (
                net_per_ha / SOM_CARBON_PER_PERCENT_TONNES_HA
                * gap_fraction
            )
            som += som_increase
            som = min(som, self.target_som_percent)
            yearly_som.append(som)

            if som >= self.target_som_percent * 0.95:
                break

        years_to_95pct = len(yearly_som) - 1

        return {
            'years_to_target': years_to_95pct,
            'yearly_som': yearly_som,
            'net_c_per_ha_yr': net_per_ha,
            'final_som': som,
            'reference_timelines': {
                'minimum_measurable': SOIL_RESTORATION_YEARS_MINIMUM,
                'functional_biology': SOIL_RESTORATION_YEARS_FUNCTIONAL,
                'full_ecosystem': SOIL_RESTORATION_YEARS_FULL,
            },
        }

    def microbial_biomass_trajectory(self):
        """
        Estimate microbial biomass carbon recovery over time.

        Microbial biomass responds faster than total SOM.
        Measurable within 1-3 years of management change.

        Returns:
            dict with yearly microbial biomass estimates.
        """
        timeline = self.restoration_timeline()
        yearly_som = timeline['yearly_som']

        # Microbial biomass scales roughly linearly with SOM
        # (simplified — real relationship is non-linear)
        som_range = SOM_HEALTHY_FRACTION * 100 - SOM_DEGRADED_FRACTION * 100
        mbc_range = (
            MICROBIAL_BIOMASS_C_HEALTHY_MG_KG
            - MICROBIAL_BIOMASS_C_DEGRADED_MG_KG
        )

        yearly_mbc = []
        for som in yearly_som:
            som_progress = (som - SOM_DEGRADED_FRACTION * 100) / som_range
            som_progress = np.clip(som_progress, 0.0, 1.0)
            mbc = (
                MICROBIAL_BIOMASS_C_DEGRADED_MG_KG
                + mbc_range * som_progress
            )
            yearly_mbc.append(mbc)

        return {
            'yearly_mbc_mg_kg': yearly_mbc,
            'yearly_som': yearly_som,
            'healthy_target_mg_kg': MICROBIAL_BIOMASS_C_HEALTHY_MG_KG,
            'degraded_baseline_mg_kg': MICROBIAL_BIOMASS_C_DEGRADED_MG_KG,
        }

    def yield_improvement(self):
        """
        Estimate yield improvement from SOM increase.

        Literature: ~1-3% yield increase per 1% SOM increase.

        Returns:
            dict with yield multiplier trajectory.
        """
        timeline = self.restoration_timeline()
        yearly_som = timeline['yearly_som']

        yearly_yield_multiplier = []
        for som in yearly_som:
            som_gain = som - self.current_som_percent
            yield_gain = som_gain * YIELD_INCREASE_PER_SOM_PERCENT
            yearly_yield_multiplier.append(1.0 + yield_gain)

        return {
            'yearly_yield_multiplier': yearly_yield_multiplier,
            'yearly_som': yearly_som,
            'yield_gain_per_som_percent': YIELD_INCREASE_PER_SOM_PERCENT,
            'final_yield_multiplier': yearly_yield_multiplier[-1],
        }

    def nutrient_cycling_capacity(self):
        """
        Estimate how soil biology affects nutrient availability.

        Healthy soil biology:
        - Increases N mineralization rate (more N from organic matter)
        - Mycorrhizae extend P access by 100x root volume
        - Bacterial weathering of K from rock minerals

        Returns:
            dict with nutrient cycling multipliers.
        """
        # Current biology relative to healthy baseline
        bio_fraction = (
            (self.current_som_percent - SOM_DEGRADED_FRACTION * 100)
            / (SOM_HEALTHY_FRACTION * 100 - SOM_DEGRADED_FRACTION * 100)
        )
        bio_fraction = np.clip(bio_fraction, 0.0, 1.0)

        # N mineralization: 2-4% of soil N per year in healthy soil
        # drops to <1% in degraded soil
        n_mineralization_rate = 0.01 + 0.03 * bio_fraction

        # P access: mycorrhizae extend root reach by 100x
        # in degraded soil, minimal mycorrhizal network
        p_access_multiplier = 1.0 + 9.0 * bio_fraction  # 1x to 10x

        # K weathering: biological weathering speeds mineral K release
        k_bio_weathering_multiplier = 1.0 + 2.0 * bio_fraction  # 1x to 3x

        return {
            'biology_fraction': bio_fraction,
            'n_mineralization_rate': n_mineralization_rate,
            'p_access_multiplier': p_access_multiplier,
            'k_bio_weathering_multiplier': k_bio_weathering_multiplier,
            'current_som_percent': self.current_som_percent,
            'note': (
                'These multipliers show how dead soil reduces nutrient '
                'access even when nutrients are physically present.'
            ),
        }

    def report(self):
        """Print a human-readable soil biology report."""
        deficit = self.carbon_deficit()
        c_input = self.annual_carbon_input()
        timeline = self.restoration_timeline()
        yields = self.yield_improvement()
        cycling = self.nutrient_cycling_capacity()

        lines = []
        lines.append("=" * 60)
        lines.append("SOIL BIOLOGY CASCADE")
        lines.append("=" * 60)
        lines.append(f"Land area: {self.land_area_ha:.1f} ha")
        lines.append(
            f"Current SOM: {self.current_som_percent:.1f}% "
            f"(target: {self.target_som_percent:.1f}%)"
        )
        lines.append(f"Tillage: {self.tillage}")
        lines.append("-" * 60)
        lines.append("CARBON DEFICIT:")
        lines.append(
            f"  Gap: {deficit['som_gap_percent']:.1f}% SOM"
        )
        lines.append(
            f"  Carbon needed: {deficit['carbon_deficit_tonnes_ha']:.1f} "
            f"tonnes C/ha"
        )
        lines.append(
            f"  Total: {deficit['total_deficit_tonnes']:.0f} tonnes C"
        )
        lines.append("-" * 60)
        lines.append("CARBON INPUTS:")
        lines.append(
            f"  Gross input: {c_input['gross_c_input_tonnes_yr']:.1f} "
            f"tonnes C/yr"
        )
        lines.append(
            f"  Net retained: {c_input['net_retained_tonnes_yr']:.1f} "
            f"tonnes C/yr ({c_input['retention_fraction']*100:.0f}%)"
        )
        lines.append(
            f"    Cover crops: {c_input['cover_crop_c_tonnes']:.1f} tonnes"
        )
        lines.append(
            f"    Compost:     {c_input['compost_c_tonnes']:.1f} tonnes"
        )
        lines.append(
            f"    Residues:    {c_input['residue_c_tonnes']:.1f} tonnes"
        )
        lines.append("-" * 60)
        lines.append("RESTORATION TIMELINE:")
        lines.append(
            f"  Years to 95% target: {timeline['years_to_target']}"
        )
        lines.append(
            f"  Final SOM: {timeline['final_som']:.2f}%"
        )
        lines.append(
            f"  Yield gain at end: "
            f"{(yields['final_yield_multiplier']-1)*100:.1f}%"
        )
        lines.append("-" * 60)
        lines.append("NUTRIENT CYCLING (current state):")
        lines.append(
            f"  Biology fraction: {cycling['biology_fraction']*100:.0f}% "
            f"of healthy"
        )
        lines.append(
            f"  N mineralization rate: {cycling['n_mineralization_rate']*100:.1f}%/yr"
        )
        lines.append(
            f"  P access multiplier: {cycling['p_access_multiplier']:.1f}x"
        )
        lines.append(
            f"  K bio-weathering: {cycling['k_bio_weathering_multiplier']:.1f}x"
        )
        lines.append("=" * 60)
        return "\n".join(lines)


def quick_test():
    """Verify soil biology cascade calculations."""
    print("Soil Biology Cascade - Quick Test")
    print()

    model = SoilBiologyCascade(
        land_area_ha=200.0,
        current_som_percent=1.2,
        target_som_percent=4.0,
        cover_crop_fraction=0.60,
        compost_input_tonnes_ha_yr=5.0,
        crop_residue_return_fraction=0.70,
        tillage='no_till',
    )

    print(model.report())

    timeline = model.restoration_timeline()
    assert timeline['years_to_target'] > 0, "Restoration must take time"
    assert timeline['yearly_som'][-1] > model.current_som_percent, \
        "SOM must increase"
    print("\nAll assertions passed.")


if __name__ == '__main__':
    quick_test()
