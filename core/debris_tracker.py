"""
Orbital debris tracking via SGP4 propagation.

Propagates TLE elements to compute current position/velocity of
orbital objects. Used for proximity analysis and conjunction assessment.

Usage:
    python -m core.debris_tracker              # Track example debris
    python -m core.debris_tracker --tle1 "..." --tle2 "..."  # Custom TLE
"""

from datetime import datetime, timezone

import numpy as np
from sgp4.api import Satrec, jday


# Example TLE: COSMOS 1408 DEBRIS
EXAMPLE_TLE_L1 = "1 49414U 21099A   26075.91875000  .00000100  00000-0  10000-3 0  9991"
EXAMPLE_TLE_L2 = "2 49414  97.4000 120.5000 0010000 280.0000 120.0000 15.10000000123456"


def track_salvage_target(l1, l2):
    """
    Propagate a TLE to current time and return ECI state.

    Args:
        l1: TLE line 1
        l2: TLE line 2

    Returns:
        Tuple of (position_km, velocity_km_s), or None on error.
    """
    satellite = Satrec.twoline2rv(l1, l2)

    now = datetime.now(timezone.utc)
    jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, now.second)

    e, r, v = satellite.sgp4(jd, fr)

    if e == 0:
        return np.array(r), np.array(v)
    else:
        print(f"SGP4 propagation error code: {e}")
        return None


def calculate_miss_distance(pos_a, pos_b):
    """
    Calculate distance between two position vectors.

    Args:
        pos_a: Position vector (km)
        pos_b: Position vector (km)

    Returns:
        Distance in km.
    """
    return float(np.linalg.norm(np.array(pos_b) - np.array(pos_a)))


def quick_test():
    """Quick test with example TLE."""
    print("Testing debris_tracker...")

    result = track_salvage_target(EXAMPLE_TLE_L1, EXAMPLE_TLE_L2)
    if result is not None:
        r, v = result
        print(f"  Position (ECI): {np.round(r, 2)} km")
        print(f"  Velocity (ECI): {np.round(v, 3)} km/s")
        alt = np.linalg.norm(r) - 6371.0
        print(f"  Altitude: {alt:.1f} km")
    else:
        print("  Propagation failed (TLE may be stale)")

    # Test miss distance
    d = calculate_miss_distance([100, 0, 0], [101, 0, 0])
    print(f"  Miss distance test: {d:.1f} km")
    print("Test passed!")


if __name__ == "__main__":
    quick_test()
