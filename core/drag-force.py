def get_air_density(self, altitude):
    """Simple Exponential Atmosphere Model (Zone 3 Hardness)"""
    rho0 = 1.225 # kg/m^3 at sea level
    H = 8500     # scale height in meters
    return rho0 * np.exp(-altitude / H)

def dynamics_with_drag(self, t, state):
    """Full Physics: J2 Bulge + Atmospheric Friction"""
    # ... (Include J2 logic from before) ...
    
    # Drag Constants
    Cd = 2.2      # Drag coefficient for a 'boxy' tin can
    Area = 2.0    # Cross-sectional area in m^2
    Mass = 500.0  # Mass in kg
    
    res = []
    for i in [0, 6]: # For Sat A and Sat B
        r_vec = state[i:i+3]
        v_vec = state[i+3:i+6]
        alt = np.linalg.norm(r_vec) - self.R_earth
        
        # Calculate Drag: a_drag = -1/2 * rho * v^2 * (Cd*A/m) * unit_v
        rho = self.get_air_density(alt)
        v_mag = np.linalg.norm(v_vec)
        acc_drag = -0.5 * rho * v_mag**2 * (Cd * Area / Mass) * (v_vec / v_mag)
        
        # Total Acceleration = Gravity + J2 + Drag
        # ... (J2 calculation here) ...
        # acc_total = acc_grav + acc_j2 + acc_drag
