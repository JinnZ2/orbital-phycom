"""
Atmospheric drag modeling for LEO satellites.

Provides exponential atmosphere density and drag acceleration
for integration into orbital dynamics.
"""

import numpy as np
from .physics_constants import R_EARTH, SAT_CD_DEFAULT, SAT_AREA_DEFAULT, SAT_MASS_DEFAULT


# Exponential atmosphere parameters
RHO0_SEA_LEVEL = 1.225   # kg/m^3 at sea level
SCALE_HEIGHT = 8500.0     # meters


def get_air_density(altitude):
    """
    Simple exponential atmosphere model.

    Args:
        altitude: Height above Earth surface in meters

    Returns:
        Air density in kg/m^3
    """
    return RHO0_SEA_LEVEL * np.exp(-altitude / SCALE_HEIGHT)


def compute_drag_acceleration(r_vec, v_vec, cd=SAT_CD_DEFAULT,
                              area=SAT_AREA_DEFAULT, mass=SAT_MASS_DEFAULT):
    """
    Compute atmospheric drag acceleration for a satellite.

    a_drag = -1/2 * rho * v^2 * (Cd * A / m) * v_hat

    Args:
        r_vec: Position vector in ECI frame (meters)
        v_vec: Velocity vector in ECI frame (m/s)
        cd: Drag coefficient (dimensionless)
        area: Cross-sectional area (m^2)
        mass: Satellite mass (kg)

    Returns:
        Drag acceleration vector in m/s^2
    """
    altitude = np.linalg.norm(r_vec) - R_EARTH
    if altitude < 0:
        altitude = 0.0

    rho = get_air_density(altitude)
    v_mag = np.linalg.norm(v_vec)

    if v_mag == 0:
        return np.zeros(3)

    v_hat = v_vec / v_mag
    acc_drag = -0.5 * rho * v_mag**2 * (cd * area / mass) * v_hat

    return acc_drag
