from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
import numpy as np

# A real piece of "Industrial Waste": COSMOS 1408 DEBRIS (Example TLE)
# This represents a "Resource Reservoir" in LEO
tle_line1 = "1 49414U 21099A   26075.91875000  .00000100  00000-0  10000-3 0  9991"
tle_line2 = "2 49414  97.4000 120.5000 0010000 280.0000 120.0000 15.10000000123456"

def track_salvage_target(l1, l2):
    satellite = Satrec.twoline2rv(l1, l2)
    
    # Get current "Tin Can" time in Julian Days
    now = datetime.utcnow()
    jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, now.second)
    
    # Propagate: Get position (km) and velocity (km/s)
    e, r, v = satellite.sgp4(jd, fr)
    
    if e == 0:
        print(f"--- Salvage Target Identified ---")
        print(f"Position (ECI): {np.round(r, 2)} km")
        print(f"Velocity (ECI): {np.round(v, 3)} km/s")
        return r, v
    else:
        print("Anxiety: Tracking error. Signal lost in the noise.")
        return None

# track_salvage_target(tle_line1, tle_line2)



def calculate_miss_distance(pos_forge, pos_junk):
    """Calculate the distance between the Salvage Ship and the Debris."""
    distance = np.linalg.norm(np.array(pos_junk) - np.array(pos_forge))
    
    # If distance < 2km, trigger "Handshake" mechanism
    if distance < 2.0:
        print("FELTSensor: High Efficiency. Intercept feasible.")
    return distance
