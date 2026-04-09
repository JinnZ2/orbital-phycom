"""
Nitrogen fixation equations: from air to soil.

Models all natural and recoverable nitrogen pathways:
    - Lightning fixation (natural baseline)
    - Bacterial fixation (legumes, free-living, cyanobacteria)
    - Atmospheric deposition (rainfall washout)
    - Compost decomposition (mineralization over time)

Every equation is traceable to published values.
Every input is something a community can measure or look up.
"""

import numpy as np
from .nutrient_constants import (
    LIGHTNING_N_FIXATION_KG_M2_YR,
    BNF_LEGUME_TYPICAL_KG_HA_YR,
    BNF_LEGUME_LOW_KG_HA_YR,
    BNF_LEGUME_HIGH_KG_HA_YR,
    BNF_FREELIVING_TYPICAL_KG_HA_YR,
    BNF_CYANOBACTERIA_TYPICAL_KG_HA_YR,
    ATMOS_DEPOSITION_TYPICAL_KG_HA_YR,
    COMPOST_N_FRACTION_TYPICAL,
    COMPOST_MINERALIZATION_RATE_YR1,
    COMPOST_MINERALIZATION_RATE_YR2,
    HABER_BOSCH_ENERGY_MJ_PER_KG_N,
    HA_TO_M2,
    HUMAN_N_REQUIREMENT_G_DAY,
    DAYS_PER_YEAR,
)


class NitrogenFixationModel:
    """
    Calculates available nitrogen from all natural and recoverable sources
    for a given land area and management strategy.

    All inputs are measurable. All outputs show the work.
    """

    def __init__(
        self,
        land_area_ha,
        legume_fraction=0.25,
        wetland_fraction=0.0,
        annual_rainfall_mm=800.0,
        compost_available_tonnes_yr=0.0,
        latitude_deg=40.0,
        lightning_flash_density=None,
    ):
        """
        Initialize nitrogen fixation model.

        Args:
            land_area_ha: Total agricultural land area in hectares.
            legume_fraction: Fraction of land in legume rotation (0-1).
            wetland_fraction: Fraction of land that is wetland/paddy (0-1).
            annual_rainfall_mm: Annual rainfall in millimeters.
            compost_available_tonnes_yr: Tonnes of finished compost per year.
            latitude_deg: Latitude in degrees (affects lightning density).
            lightning_flash_density: Flashes per km^2 per year. If None,
                estimated from latitude.
        """
        self.land_area_ha = land_area_ha
        self.legume_fraction = legume_fraction
        self.wetland_fraction = wetland_fraction
        self.annual_rainfall_mm = annual_rainfall_mm
        self.compost_available_tonnes_yr = compost_available_tonnes_yr
        self.latitude_deg = latitude_deg

        if lightning_flash_density is not None:
            self.lightning_flash_density = lightning_flash_density
        else:
            self.lightning_flash_density = self._estimate_lightning_density(
                latitude_deg
            )

    def _estimate_lightning_density(self, latitude_deg):
        """
        Estimate lightning flash density from latitude.

        Tropical regions: ~10-50 flashes/km^2/yr
        Temperate: ~1-5 flashes/km^2/yr
        Polar: ~0.1 flashes/km^2/yr

        Args:
            latitude_deg: Absolute latitude in degrees.

        Returns:
            Estimated flashes per km^2 per year.
        """
        abs_lat = abs(latitude_deg)
        if abs_lat < 15:
            return 20.0
        elif abs_lat < 30:
            return 8.0
        elif abs_lat < 50:
            return 3.0
        elif abs_lat < 65:
            return 1.0
        else:
            return 0.1

    def lightning_fixation_kg_yr(self):
        """
        Nitrogen fixed by lightning over the land area.

        Physics: N2 + O2 → 2NO (thermal dissociation in lightning channel)
        Each flash fixes ~1 kg N (varying estimates: 0.5-5 kg)
        Global total: ~5 Tg N/yr from ~1.4 billion flashes/yr

        Returns:
            dict with kg_n_per_year and calculation details.
        """
        # Method 1: From global average rate
        area_m2 = self.land_area_ha * HA_TO_M2
        from_global_rate = LIGHTNING_N_FIXATION_KG_M2_YR * area_m2

        # Method 2: From local flash density
        # ~1 kg N per flash (conservative estimate)
        kg_n_per_flash = 1.0
        area_km2 = self.land_area_ha / 100.0
        from_local_flashes = (
            self.lightning_flash_density * area_km2 * kg_n_per_flash
        )

        # Use local estimate (more specific)
        result = from_local_flashes

        return {
            'kg_n_per_year': result,
            'method': 'local_flash_density',
            'flash_density_per_km2_yr': self.lightning_flash_density,
            'area_km2': area_km2,
            'kg_n_per_flash': kg_n_per_flash,
            'global_rate_comparison_kg': from_global_rate,
            'source': 'Galloway et al. 2004; Schumann & Huntrieser 2007',
        }

    def bacterial_fixation_kg_yr(self):
        """
        Nitrogen fixed by soil bacteria.

        Three pathways:
        1. Symbiotic (Rhizobium + legumes): 50-300 kg N/ha/yr
        2. Free-living (Azotobacter, Clostridium): 5-20 kg N/ha/yr
        3. Cyanobacteria (wetlands, paddies): 20-50 kg N/ha/yr

        Returns:
            dict with kg_n_per_year and breakdown by pathway.
        """
        legume_area = self.land_area_ha * self.legume_fraction
        non_legume_area = self.land_area_ha * (
            1.0 - self.legume_fraction - self.wetland_fraction
        )
        wetland_area = self.land_area_ha * self.wetland_fraction

        symbiotic = legume_area * BNF_LEGUME_TYPICAL_KG_HA_YR
        freeliving = non_legume_area * BNF_FREELIVING_TYPICAL_KG_HA_YR
        cyanobacteria = wetland_area * BNF_CYANOBACTERIA_TYPICAL_KG_HA_YR

        total = symbiotic + freeliving + cyanobacteria

        return {
            'kg_n_per_year': total,
            'symbiotic_kg': symbiotic,
            'symbiotic_area_ha': legume_area,
            'symbiotic_rate_kg_ha': BNF_LEGUME_TYPICAL_KG_HA_YR,
            'symbiotic_range': (BNF_LEGUME_LOW_KG_HA_YR, BNF_LEGUME_HIGH_KG_HA_YR),
            'freeliving_kg': freeliving,
            'freeliving_area_ha': non_legume_area,
            'freeliving_rate_kg_ha': BNF_FREELIVING_TYPICAL_KG_HA_YR,
            'cyanobacteria_kg': cyanobacteria,
            'cyanobacteria_area_ha': wetland_area,
            'cyanobacteria_rate_kg_ha': BNF_CYANOBACTERIA_TYPICAL_KG_HA_YR,
            'source': 'Peoples et al. 2009; Cleveland et al. 1999',
        }

    def atmospheric_deposition_kg_yr(self):
        """
        Nitrogen deposited via rainfall (wet deposition).

        NH4+ and NO3- dissolved in rain → deposited on soil.
        Rate scales roughly with rainfall amount.

        Returns:
            dict with kg_n_per_year and calculation details.
        """
        # Scale deposition with rainfall relative to 800mm baseline
        rainfall_factor = self.annual_rainfall_mm / 800.0
        rate = ATMOS_DEPOSITION_TYPICAL_KG_HA_YR * rainfall_factor
        total = rate * self.land_area_ha

        return {
            'kg_n_per_year': total,
            'rate_kg_ha_yr': rate,
            'rainfall_factor': rainfall_factor,
            'annual_rainfall_mm': self.annual_rainfall_mm,
            'source': 'Dentener et al. 2006',
        }

    def compost_nitrogen_kg_yr(self, years_composting=1):
        """
        Plant-available nitrogen from compost application.

        Compost N is mostly organic → mineralized slowly.
        Year 1: ~10% of total N becomes plant-available.
        Year 2+: ~5%/yr of remaining total N.

        Args:
            years_composting: How many years compost has been applied.

        Returns:
            dict with kg_n_per_year and mineralization details.
        """
        total_n_in_compost = (
            self.compost_available_tonnes_yr * 1000.0
            * COMPOST_N_FRACTION_TYPICAL
        )

        # First-year mineralization from this year's application
        available_yr1 = total_n_in_compost * COMPOST_MINERALIZATION_RATE_YR1

        # Residual from previous years' applications
        residual = 0.0
        if years_composting > 1:
            for yr in range(1, min(years_composting, 10)):
                residual += (
                    total_n_in_compost * COMPOST_MINERALIZATION_RATE_YR2
                )

        total_available = available_yr1 + residual

        return {
            'kg_n_per_year': total_available,
            'total_n_in_compost_kg': total_n_in_compost,
            'first_year_available_kg': available_yr1,
            'residual_from_prior_years_kg': residual,
            'compost_tonnes_yr': self.compost_available_tonnes_yr,
            'n_fraction': COMPOST_N_FRACTION_TYPICAL,
            'mineralization_rate_yr1': COMPOST_MINERALIZATION_RATE_YR1,
            'years_composting': years_composting,
            'source': 'USDA; Sullivan & Miller 2001',
        }

    def total_available_nitrogen(self, years_composting=1):
        """
        Total plant-available nitrogen from all natural/recoverable sources.

        Returns:
            dict with total and all component breakdowns.
        """
        lightning = self.lightning_fixation_kg_yr()
        bacterial = self.bacterial_fixation_kg_yr()
        deposition = self.atmospheric_deposition_kg_yr()
        compost = self.compost_nitrogen_kg_yr(years_composting)

        total = (
            lightning['kg_n_per_year']
            + bacterial['kg_n_per_year']
            + deposition['kg_n_per_year']
            + compost['kg_n_per_year']
        )

        # How many people can this feed? (N basis only)
        n_per_person_kg_yr = HUMAN_N_REQUIREMENT_G_DAY * DAYS_PER_YEAR / 1000.0
        people_fed_n_basis = total / n_per_person_kg_yr

        # Equivalent synthetic fertilizer (energy cost)
        equivalent_energy_mj = total * HABER_BOSCH_ENERGY_MJ_PER_KG_N

        return {
            'total_kg_n_per_year': total,
            'lightning': lightning,
            'bacterial': bacterial,
            'deposition': deposition,
            'compost': compost,
            'people_fed_n_basis': people_fed_n_basis,
            'equivalent_haber_bosch_energy_mj': equivalent_energy_mj,
            'land_area_ha': self.land_area_ha,
            'kg_n_per_ha': total / self.land_area_ha if self.land_area_ha > 0 else 0,
        }

    def report(self, years_composting=1):
        """Print a human-readable nitrogen budget."""
        r = self.total_available_nitrogen(years_composting)
        lines = []
        lines.append("=" * 60)
        lines.append("NITROGEN BUDGET (kg N/year)")
        lines.append("=" * 60)
        lines.append(f"Land area: {self.land_area_ha:.1f} ha")
        lines.append(f"Legume rotation: {self.legume_fraction*100:.0f}%")
        lines.append(f"Rainfall: {self.annual_rainfall_mm:.0f} mm/yr")
        lines.append("-" * 60)
        lines.append(
            f"  Lightning fixation:      {r['lightning']['kg_n_per_year']:>10.1f} kg N"
        )
        lines.append(
            f"  Bacterial fixation:      {r['bacterial']['kg_n_per_year']:>10.1f} kg N"
        )
        lines.append(
            f"    Symbiotic (legumes):   {r['bacterial']['symbiotic_kg']:>10.1f} kg N"
        )
        lines.append(
            f"    Free-living:           {r['bacterial']['freeliving_kg']:>10.1f} kg N"
        )
        lines.append(
            f"    Cyanobacteria:         {r['bacterial']['cyanobacteria_kg']:>10.1f} kg N"
        )
        lines.append(
            f"  Atmospheric deposition:  {r['deposition']['kg_n_per_year']:>10.1f} kg N"
        )
        lines.append(
            f"  Compost mineralization:  {r['compost']['kg_n_per_year']:>10.1f} kg N"
        )
        lines.append("-" * 60)
        lines.append(
            f"  TOTAL AVAILABLE:         {r['total_kg_n_per_year']:>10.1f} kg N/yr"
        )
        lines.append(
            f"  Per hectare:             {r['kg_n_per_ha']:>10.1f} kg N/ha/yr"
        )
        lines.append(
            f"  People fed (N basis):    {r['people_fed_n_basis']:>10.0f}"
        )
        lines.append(
            f"  Equiv. Haber-Bosch:      {r['equivalent_haber_bosch_energy_mj']:>10.0f} MJ"
        )
        lines.append("=" * 60)
        return "\n".join(lines)


def quick_test():
    """Verify nitrogen fixation calculations for a small community."""
    print("Nitrogen Fixation Model - Quick Test")
    print()

    # Scenario: 500-person village, 200 ha farmland
    model = NitrogenFixationModel(
        land_area_ha=200.0,
        legume_fraction=0.30,     # 30% in legume rotation
        wetland_fraction=0.05,    # 5% wetland
        annual_rainfall_mm=900.0,
        compost_available_tonnes_yr=100.0,  # 100 tonnes compost/yr
        latitude_deg=35.0,
    )

    print(model.report(years_composting=3))

    # Verify physics: dominant source should be bacterial fixation
    r = model.total_available_nitrogen(years_composting=3)
    assert r['bacterial']['kg_n_per_year'] > r['lightning']['kg_n_per_year'], \
        "Bacterial fixation should dominate lightning"
    assert r['total_kg_n_per_year'] > 0, "Total N must be positive"
    print("\nAll assertions passed.")


if __name__ == '__main__':
    quick_test()
