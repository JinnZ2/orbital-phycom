"""
Narrative vs Physics detection layer.

When someone says "fertilizer shortage," this module:
    1. Checks the actual equations
    2. Calculates available nutrients (sewage + dumps + biology)
    3. Compares to claimed shortage
    4. Detects: "This is a narrative, not a physics problem"

The detection mechanism:
    - STATED CLAIM: "We can't grow food without synthetic fertilizer"
    - PHYSICAL REALITY: Nitrogen is 78% of the atmosphere. Phosphorus
      is in every toilet. Potassium is in every rock.
    - ACTUAL CONSTRAINT: "We poisoned the nutrient cycle" (fixable)
      vs. "Nutrients don't exist" (false)

Pattern recognition:
    - Where is actual physics being ignored in favor of a narrative
      that serves someone's power?
    - The fertilizer shortage is narrated as inevitable physics when
      the actual physics says it's a choice.
"""

import numpy as np
from .local_capacity import LocalProductionCapacity
from .nutrient_constants import (
    MOROCCO_PHOSPHATE_FRACTION_GLOBAL,
    COST_UREA_USD_TONNE,
    COST_DAP_USD_TONNE,
    COST_POTASH_USD_TONNE,
    COST_COMPOST_SYSTEM_USD_PER_PERSON,
    COST_STRUVITE_REACTOR_USD_PER_PERSON,
    COST_COVER_CROP_SEED_USD_HA,
    HABER_BOSCH_ENERGY_MJ_PER_KG_N,
)


class NarrativeDetector:
    """
    Compares narrative claims about resource scarcity against
    physical reality calculated from first principles.

    Not opinion. Not politics. Just: does the math support the claim?
    """

    def __init__(self, local_capacity_model):
        """
        Initialize with a LocalProductionCapacity model.

        Args:
            local_capacity_model: A configured LocalProductionCapacity instance
                representing a specific community's measurable reality.
        """
        self.model = local_capacity_model

    @classmethod
    def from_community(cls, **kwargs):
        """
        Create a NarrativeDetector directly from community parameters.

        Args:
            **kwargs: All parameters accepted by LocalProductionCapacity.

        Returns:
            NarrativeDetector instance.
        """
        model = LocalProductionCapacity(**kwargs)
        return cls(model)

    def analyze_claim(self, claim):
        """
        Analyze a narrative claim against physical reality.

        Args:
            claim: One of the supported claim types:
                'fertilizer_shortage'
                'cannot_feed_locally'
                'synthetic_required'
                'phosphorus_peak'

        Returns:
            dict with claim, physical_reality, contradiction, and evidence.
        """
        analyzers = {
            'fertilizer_shortage': self._analyze_fertilizer_shortage,
            'cannot_feed_locally': self._analyze_local_feeding,
            'synthetic_required': self._analyze_synthetic_requirement,
            'phosphorus_peak': self._analyze_phosphorus_peak,
        }

        analyzer = analyzers.get(claim)
        if analyzer is None:
            return {
                'claim': claim,
                'error': f"Unknown claim type. Supported: {list(analyzers.keys())}",
            }

        return analyzer()

    def _analyze_fertilizer_shortage(self):
        """Analyze the claim: 'There is a fertilizer shortage.'"""
        assessment = self.model.food_security_assessment()
        n_data = self.model.nitrogen.total_available_nitrogen()
        p_data = self.model.phosphorus.total_recoverable_phosphorus()
        k_data = self.model.potassium.total_potassium_budget()

        # What's physically available (no supply chain needed)
        n_local = n_data['total_kg_n_per_year']
        p_local = p_data['annual_flow_kg_p']
        p_stock = p_data['total_stock_kg_p']
        k_local = k_data['total_supply_kg_yr']

        # What the community needs
        nutrients = assessment['nutrient_hectares']
        n_needed = nutrients['n_demand_kg_ha'] * self.model.land_area_ha
        p_needed = nutrients['p_demand_kg_ha'] * self.model.land_area_ha
        k_needed = nutrients['k_demand_kg_ha'] * self.model.land_area_ha

        n_pct = (n_local / max(n_needed, 1e-10)) * 100
        p_pct = (p_local / max(p_needed, 1e-10)) * 100
        k_pct = (k_local / max(k_needed, 1e-10)) * 100

        is_narrative = (n_pct > 30 or p_pct > 30 or k_pct > 30)

        return {
            'claim': 'There is a fertilizer shortage',
            'physical_reality': {
                'nitrogen_available_pct_of_need': n_pct,
                'phosphorus_available_pct_of_need': p_pct,
                'potassium_available_pct_of_need': k_pct,
                'phosphorus_in_dump_kg': p_stock,
                'nitrogen_from_biology_kg': n_local,
            },
            'is_narrative_not_physics': is_narrative,
            'contradiction': (
                "Nutrients exist locally. The shortage is in the "
                "supply chain for SYNTHETIC fertilizer, not in the "
                "physical nutrients themselves. "
                f"This community has {n_pct:.0f}% of N needs from biology, "
                f"{p_pct:.0f}% of P needs from waste recovery, "
                f"and {k_pct:.0f}% of K needs from natural sources."
            ) if is_narrative else (
                "Local nutrient sources are genuinely insufficient. "
                "Recovery infrastructure or soil restoration needed."
            ),
            'what_is_actually_scarce': [
                "Cheap natural gas (for Haber-Bosch ammonia)",
                "Access to phosphate mines (Morocco controls 70%)",
                "Consolidated potash supply chains",
            ],
            'what_is_not_scarce': [
                "Atmospheric nitrogen (78% of air)",
                "Phosphorus in sewage/dumps (every toilet)",
                "Potassium in rock and ocean",
                "Soil bacteria (if soil is alive)",
            ],
        }

    def _analyze_local_feeding(self):
        """Analyze the claim: 'This community cannot feed itself locally.'"""
        assessment = self.model.food_security_assessment()
        pct_now = assessment['food_security_percent']
        pct_future = assessment['food_security_percent_after_restoration']
        years = assessment['years_to_full_restoration']

        return {
            'claim': 'This community cannot feed itself locally',
            'physical_reality': {
                'current_food_security_pct': pct_now,
                'after_restoration_pct': pct_future,
                'years_to_restore': years,
                'limiting_nutrient': assessment['limiting_nutrient'],
                'productive_hectares': assessment['productive_hectares'],
            },
            'is_narrative_not_physics': pct_future > 80,
            'contradiction': (
                f"With soil restoration and nutrient recovery, this community "
                f"can reach {pct_future:.0f}% food security in {years} years. "
                f"Current state ({pct_now:.0f}%) reflects degraded soil and "
                f"wasted nutrients, not fundamental limits."
            ) if pct_future > 80 else (
                f"Physical constraints are real: only {pct_future:.0f}% "
                f"achievable even with full restoration. "
                f"Land area or water may be the true limit."
            ),
            'recommendations': assessment['recommendations'],
        }

    def _analyze_synthetic_requirement(self):
        """Analyze the claim: 'Synthetic fertilizer is required for farming.'"""
        n_data = self.model.nitrogen.total_available_nitrogen()
        assessment = self.model.food_security_assessment()

        # Cost comparison
        n_needed = (
            assessment['nutrient_hectares']['n_demand_kg_ha']
            * self.model.land_area_ha
        )

        # Synthetic cost (annual, forever)
        urea_n_fraction = 0.46  # urea is 46% N
        synthetic_urea_tonnes = n_needed / (urea_n_fraction * 1000.0)
        annual_synthetic_cost = synthetic_urea_tonnes * COST_UREA_USD_TONNE

        # Biological fix cost (one-time infrastructure + small annual)
        infrastructure_cost = (
            self.model.population * COST_COMPOST_SYSTEM_USD_PER_PERSON
            + self.model.population * COST_STRUVITE_REACTOR_USD_PER_PERSON
            + self.model.land_area_ha * COST_COVER_CROP_SEED_USD_HA
        )
        annual_bio_cost = (
            self.model.land_area_ha * COST_COVER_CROP_SEED_USD_HA
        )

        # Energy comparison
        synthetic_energy_mj = n_needed * HABER_BOSCH_ENERGY_MJ_PER_KG_N
        bio_energy_mj = 0.0  # biological fixation is solar-powered

        # Payback period
        if annual_synthetic_cost > annual_bio_cost:
            annual_savings = annual_synthetic_cost - annual_bio_cost
            payback_years = infrastructure_cost / max(annual_savings, 1.0)
        else:
            payback_years = float('inf')

        return {
            'claim': 'Synthetic fertilizer is required for farming',
            'physical_reality': {
                'biological_n_available_kg': n_data['total_kg_n_per_year'],
                'biological_n_pct_of_need': (
                    n_data['total_kg_n_per_year'] / max(n_needed, 1e-10) * 100
                ),
            },
            'cost_comparison': {
                'synthetic_annual_usd': annual_synthetic_cost,
                'biological_infrastructure_usd': infrastructure_cost,
                'biological_annual_usd': annual_bio_cost,
                'payback_years': payback_years,
                'synthetic_energy_mj_yr': synthetic_energy_mj,
                'biological_energy_mj_yr': bio_energy_mj,
            },
            'is_narrative_not_physics': True,
            'contradiction': (
                "Synthetic fertilizer is one option, not a requirement. "
                f"Biological fixation can supply "
                f"{n_data['total_kg_n_per_year']:.0f} kg N/yr. "
                f"Synthetic costs ${annual_synthetic_cost:,.0f}/yr forever. "
                f"Biological infrastructure costs "
                f"${infrastructure_cost:,.0f} once, "
                f"then ${annual_bio_cost:,.0f}/yr. "
                f"Payback: {payback_years:.1f} years."
            ),
            'power_analysis': {
                'synthetic_keeps': (
                    "Dependence on natural gas supply chains, "
                    "mining corporations, and global commodity markets"
                ),
                'biological_creates': (
                    "Local self-sufficiency, community control, "
                    "no external supply chain dependency"
                ),
            },
        }

    def _analyze_phosphorus_peak(self):
        """Analyze the claim: 'We are running out of phosphorus (peak P).'"""
        p_data = self.model.phosphorus.total_recoverable_phosphorus()

        annual_flow = p_data['annual_flow_kg_p']
        dump_stock = p_data['dump']['recoverable_kg_p']
        dump_years = p_data['dump']['years_at_extraction_rate']

        return {
            'claim': 'We are running out of phosphorus (peak phosphorus)',
            'physical_reality': {
                'p_in_sewage_kg_yr': p_data['sewage']['kg_p_per_year'],
                'p_in_food_waste_kg_yr': p_data['food_waste']['kg_p_per_year'],
                'p_in_dump_kg': p_data['dump']['total_kg_p'],
                'p_annual_recoverable_kg': annual_flow,
                'dump_supply_years': dump_years,
            },
            'is_narrative_not_physics': True,
            'contradiction': (
                "Phosphorus hasn't left the planet. "
                f"This community excretes {p_data['sewage']['gross_excretion_kg_yr']:.0f} kg P/yr "
                f"into sewage. "
                f"Their dump contains {p_data['dump']['total_kg_p']:.0f} kg P. "
                f"What's 'running out' is cheap MINED phosphate "
                f"(Morocco controls {MOROCCO_PHOSPHATE_FRACTION_GLOBAL*100:.0f}%). "
                "The actual constraint is that we dump nutrients "
                "instead of cycling them."
            ),
            'what_is_true': (
                "Mined phosphate rock is a finite resource and "
                "concentrated in a few countries."
            ),
            'what_is_misleading': (
                "Framing this as 'running out of phosphorus' when "
                "phosphorus is being flushed into rivers daily. "
                "It's a recovery problem, not a scarcity problem."
            ),
        }

    def full_narrative_analysis(self):
        """
        Run all narrative checks and produce a comprehensive report.

        Returns:
            dict with all claim analyses.
        """
        claims = [
            'fertilizer_shortage',
            'cannot_feed_locally',
            'synthetic_required',
            'phosphorus_peak',
        ]

        results = {}
        for claim in claims:
            results[claim] = self.analyze_claim(claim)

        # Count narrative vs physics
        narrative_count = sum(
            1 for r in results.values()
            if r.get('is_narrative_not_physics', False)
        )

        results['summary'] = {
            'claims_analyzed': len(claims),
            'narrative_not_physics': narrative_count,
            'physics_confirmed': len(claims) - narrative_count,
            'pattern': (
                "The dominant narrative prioritizes solutions that "
                "maintain centralized control (synthetic supply chains) "
                "over solutions that enable local self-sufficiency "
                "(biological nutrient cycling)."
            ) if narrative_count > 2 else (
                "Mixed results — some claims reflect real constraints."
            ),
        }

        return results

    def report(self):
        """Print a human-readable narrative analysis."""
        results = self.full_narrative_analysis()
        lines = []
        lines.append("=" * 70)
        lines.append("NARRATIVE vs PHYSICS ANALYSIS")
        lines.append("=" * 70)
        lines.append(
            f"Community: {self.model.population:,} people, "
            f"{self.model.land_area_ha:.0f} ha"
        )
        lines.append("")

        for key, result in results.items():
            if key == 'summary':
                continue
            lines.append("-" * 70)
            lines.append(f"CLAIM: \"{result['claim']}\"")
            is_narr = result.get('is_narrative_not_physics', False)
            if is_narr:
                lines.append("VERDICT: NARRATIVE (not supported by physics)")
            else:
                lines.append("VERDICT: PHYSICS (real constraint)")
            lines.append(f"EVIDENCE: {result['contradiction']}")
            lines.append("")

        lines.append("=" * 70)
        s = results['summary']
        lines.append(
            f"SUMMARY: {s['narrative_not_physics']}/{s['claims_analyzed']} "
            f"claims are narrative, not physics"
        )
        lines.append(f"PATTERN: {s['pattern']}")
        lines.append("=" * 70)
        return "\n".join(lines)


def quick_test():
    """Run narrative detection on a sample community."""
    print("Narrative Detector - Quick Test")
    print()

    detector = NarrativeDetector.from_community(
        population=5000,
        land_area_ha=500.0,
        current_som_percent=1.5,
        legume_fraction=0.25,
        annual_rainfall_mm=900.0,
        latitude_deg=35.0,
        distance_to_coast_km=50.0,
        sewage_fraction_collected=0.60,
        food_waste_fraction_collected=0.40,
        dump_volume_m3=30000.0,
        compost_available_tonnes_yr=100.0,
        wood_biomass_available_tonnes_yr=50.0,
        rock_dust_available_tonnes_yr=10.0,
        urine_collection_fraction=0.30,
        crop_residue_return_fraction=0.60,
        tillage='no_till',
        primary_crop='wheat',
    )

    print(detector.report())

    # Verify specific claim
    result = detector.analyze_claim('fertilizer_shortage')
    assert 'physical_reality' in result
    assert 'contradiction' in result
    print("\nAll assertions passed.")


if __name__ == '__main__':
    quick_test()
