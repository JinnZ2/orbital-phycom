"""
Physical constants for orbital mechanics.

All values in SI units.
"""

# Gravitational parameter (Earth)
MU_EARTH = 3.986004418e14  # m^3/s^2

# Earth radius
R_EARTH = 6371e3  # meters

# Reference altitude for standard orbit
ALTITUDE_REF = 500e3  # meters (500 km)

# Standard gravitational acceleration
G0 = 9.80665  # m/s^2

# RF wavelength for phase tracking
LAMBDA_RF = 0.1  # meters (3 GHz)

# Speed of light
C = 299792458.0  # m/s

# Solar radiation pressure (at 1 AU)
P_SOLAR = 4.56e-6  # N/m^2

# Typical satellite parameters
SAT_MASS_DEFAULT = 50.0  # kg (CubeSat-class)
SAT_AREA_DEFAULT = 1.0   # m^2 (cross-sectional)
SAT_CD_DEFAULT = 2.2     # drag coefficient
SAT_CR_DEFAULT = 1.2     # reflectivity coefficient

# Time constants
SECONDS_PER_DAY = 86400.0
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_MINUTE = 60.0
