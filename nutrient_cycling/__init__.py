"""
Nutrient cycling physics module.

Models real biogeochemical nutrient flows (nitrogen, phosphorus, potassium)
from first principles. Enables local food security assessment by calculating
recoverable nutrients from waste streams, natural fixation, and soil biology.

Designed to be:
    - Traceable (show the work)
    - Localizable (work for any geography)
    - Verifiable (people can measure the inputs themselves)
    - Honest (no hiding the math)
"""

from .nitrogen_fixation import NitrogenFixationModel
from .phosphorus_recovery import PhosphorusRecoveryModel
from .potassium_cycling import PotassiumCyclingModel
from .soil_biology import SoilBiologyCascade
from .local_capacity import LocalProductionCapacity
from .narrative_detector import NarrativeDetector

__all__ = [
    'NitrogenFixationModel',
    'PhosphorusRecoveryModel',
    'PotassiumCyclingModel',
    'SoilBiologyCascade',
    'LocalProductionCapacity',
    'NarrativeDetector',
]
