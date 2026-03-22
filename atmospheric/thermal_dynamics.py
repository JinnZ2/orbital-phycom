"""
Atmospheric thermal dynamics for seed expansion.

Models natural thermal processes that can be precisely modulated
to create detectable patterns in atmospheric behavior.
"""

import numpy as np
from scipy.integrate import solve_ivp


class AtmosphericSimulator:
    """
    Atmospheric dynamics simulator for seed-based communication.

    Uses natural thermal processes (solar heating, ground thermal mass,
    convection) to create detectable signatures in wind/pressure patterns.
    """

    def __init__(self, grid_size=(100, 100), domain_km=50.0, dt=60.0):
        """
        Initialize atmospheric grid.

        Args:
            grid_size: (nx, ny) grid dimensions
            domain_km: Physical domain size in kilometers
            dt: Time step in seconds
        """
        self.nx, self.ny = grid_size
        self.domain_km = domain_km
        self.dt = dt

        # Grid spacing
        self.dx = self.domain_km * 1000 / self.nx  # meters
        self.dy = self.domain_km * 1000 / self.ny  # meters

        # Physical constants
        self.g = 9.81       # gravity (m/s^2)
        self.cp = 1004.0    # specific heat of air (J/kg/K)
        self.rho0 = 1.225   # air density at sea level (kg/m^3)
        self.R = 287.0      # gas constant for air (J/kg/K)
        self.T0 = 288.15    # reference temperature (K)

        # Initialize grids
        x = np.linspace(0, domain_km * 1000, self.nx)
        y = np.linspace(0, domain_km * 1000, self.ny)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')

        # State variables: [temperature, pressure, u_wind, v_wind]
        self.T = np.full((self.nx, self.ny), self.T0)  # Temperature (K)
        self.P = np.full((self.nx, self.ny), 101325.0)  # Pressure (Pa)
        self.u = np.zeros((self.nx, self.ny))           # East wind (m/s)
        self.v = np.zeros((self.nx, self.ny))           # North wind (m/s)

        # Thermal sources (heating rate in W/m^3)
        self.thermal_sources = np.zeros((self.nx, self.ny))

        # Baseline temperature profile (varies with location)
        self._initialize_baseline_temperature()

    def _initialize_baseline_temperature(self):
        """Initialize natural temperature variation across domain."""
        # Gentle temperature gradient (warmer toward south/east)
        temp_gradient_x = 0.5 / (self.domain_km * 1000)  # 0.5 K per domain
        temp_gradient_y = 0.3 / (self.domain_km * 1000)  # 0.3 K per domain

        self.T += temp_gradient_x * self.X + temp_gradient_y * self.Y

        # Warm spot (like a lake or urban area) — vectorized
        warm_x, warm_y = self.nx // 3, self.ny // 2
        warm_radius = min(self.nx, self.ny) * 0.1
        ii, jj = np.ogrid[:self.nx, :self.ny]
        dist_warm = np.sqrt((ii - warm_x)**2 + (jj - warm_y)**2)
        mask_warm = dist_warm < warm_radius
        self.T[mask_warm] += 2.0 * np.exp(
            -dist_warm[mask_warm]**2 / (2 * (warm_radius / 3)**2)
        )

        # Cool spot (like a forest or elevated area) — vectorized
        cool_x, cool_y = 2 * self.nx // 3, self.ny // 4
        cool_radius = min(self.nx, self.ny) * 0.08
        dist_cool = np.sqrt((ii - cool_x)**2 + (jj - cool_y)**2)
        mask_cool = dist_cool < cool_radius
        self.T[mask_cool] -= 1.5 * np.exp(
            -dist_cool[mask_cool]**2 / (2 * (cool_radius / 3)**2)
        )

    def add_thermal_source(self, x_km, y_km, intensity_K, radius_km):
        """
        Add natural thermal source (heating/cooling).

        Args:
            x_km, y_km: Position in kilometers
            intensity_K: Temperature change in Kelvin (+ for heating, - for cooling)
            radius_km: Radius of influence in kilometers
        """
        x_idx = int(x_km * 1000 / self.dx)
        y_idx = int(y_km * 1000 / self.dy)
        radius_idx = int(radius_km * 1000 / self.dx)

        i_lo = max(0, x_idx - 2 * radius_idx)
        i_hi = min(self.nx, x_idx + 2 * radius_idx)
        j_lo = max(0, y_idx - 2 * radius_idx)
        j_hi = min(self.ny, y_idx + 2 * radius_idx)

        ii, jj = np.ogrid[i_lo:i_hi, j_lo:j_hi]
        dist_m = np.sqrt(((ii - x_idx) * self.dx)**2 +
                         ((jj - y_idx) * self.dy)**2)
        mask = dist_m < radius_km * 1000
        sigma = radius_km * 1000 / 3
        gaussian = np.exp(-dist_m**2 / (2 * sigma**2))
        # Convert intensity (K) to heating rate (W/m^3) for thermal forcing
        self.thermal_sources[i_lo:i_hi, j_lo:j_hi][mask] += (
            intensity_K * self.rho0 * self.cp * gaussian[mask]
        )

    def compute_derivatives(self, state_vector, t):
        """
        Compute time derivatives for atmospheric dynamics.

        Simplified shallow water equations with thermal forcing.

        Args:
            state_vector: Flattened state [T, P, u, v]
            t: Current time (unused)

        Returns:
            Flattened time derivatives
        """
        # Unpack state
        n_vars = len(state_vector) // (self.nx * self.ny)
        state = state_vector.reshape(n_vars, self.nx, self.ny)
        T, P, u, v = state[0], state[1], state[2], state[3]

        # Initialize derivatives
        dT_dt = np.zeros_like(T)
        dP_dt = np.zeros_like(P)
        du_dt = np.zeros_like(u)
        dv_dt = np.zeros_like(v)

        # Thermal forcing: dT/dt = Q / (rho * cp)
        dT_dt += self.thermal_sources / (self.rho0 * self.cp)

        # Temperature diffusion (natural mixing)
        diffusion_coeff = 10.0  # m^2/s
        dT_dt += diffusion_coeff * (
            np.gradient(np.gradient(T, self.dx, axis=0), self.dx, axis=0) +
            np.gradient(np.gradient(T, self.dy, axis=1), self.dy, axis=1)
        )

        # Pressure gradient from temperature (hydrostatic adjustment)
        dP_dx = np.gradient(P, self.dx, axis=0)
        dP_dy = np.gradient(P, self.dy, axis=1)

        # Thermal wind: pressure perturbation from temperature anomaly
        T_anomaly = T - self.T0
        dP_dt += -self.rho0 * self.R * np.gradient(
            T_anomaly, self.dx, axis=0
        ) * u - self.rho0 * self.R * np.gradient(
            T_anomaly, self.dy, axis=1
        ) * v

        # Momentum equations (simplified, inviscid)
        du_dt += -dP_dx / self.rho0
        dv_dt += -dP_dy / self.rho0

        # Advection of temperature by wind
        dT_dt += -(u * np.gradient(T, self.dx, axis=0) +
                   v * np.gradient(T, self.dy, axis=1))

        # Stack and flatten
        derivatives = np.stack([dT_dt, dP_dt, du_dt, dv_dt])
        return derivatives.flatten()

    def step(self, dt=None):
        """
        Advance atmospheric state by one time step using forward Euler.

        Args:
            dt: Time step in seconds (defaults to self.dt)

        Returns:
            Tuple of (T, P, u, v) arrays after the step
        """
        if dt is None:
            dt = self.dt

        # Pack current state
        state_vector = np.stack([self.T, self.P, self.u, self.v]).flatten()

        # Compute derivatives and advance
        deriv = self.compute_derivatives(state_vector, 0.0)
        state_vector += dt * deriv

        # Unpack updated state
        state = state_vector.reshape(4, self.nx, self.ny)
        self.T = state[0]
        self.P = state[1]
        self.u = state[2]
        self.v = state[3]

        return self.T.copy(), self.P.copy(), self.u.copy(), self.v.copy()


class AtmosphericSeedExpander:
    """
    Expands compact seeds into atmospheric evolution via thermal physics.

    Each seed encodes thermal source parameters; physics amplifies the
    signal into detectable atmospheric patterns.
    """

    def __init__(self, simulator):
        """
        Initialize with an atmospheric simulator.

        Args:
            simulator: AtmosphericSimulator instance
        """
        self.sim = simulator

    def expand_seed(self, seed, total_duration=1800.0, intensity_scale=0.3):
        """
        Expand seed into atmospheric evolution.

        Seed structure: groups of 4 values per thermal source
        [x_fraction, y_fraction, intensity_fraction, radius_fraction]

        Args:
            seed: List of seed values (multiple of 4)
            total_duration: Simulation duration in seconds
            intensity_scale: Maximum thermal perturbation in Kelvin

        Returns:
            times: Array of time points
            states: List of (T, P, u, v) tuples at each time step
        """
        n_sources = len(seed) // 4

        # Apply thermal sources from seed
        for i in range(n_sources):
            idx = i * 4
            x_frac = seed[idx]
            y_frac = seed[idx + 1]
            int_frac = seed[idx + 2]
            rad_frac = seed[idx + 3]

            x_km = x_frac * self.sim.domain_km
            y_km = y_frac * self.sim.domain_km
            intensity_K = (int_frac - 0.5) * 2 * intensity_scale
            radius_km = 1.0 + rad_frac * 9.0

            self.sim.add_thermal_source(x_km, y_km, intensity_K, radius_km)

        # Run simulation
        times = []
        states = []
        n_steps = int(total_duration / self.sim.dt)

        for step in range(n_steps):
            t = step * self.sim.dt
            T, P, u, v = self.sim.step()
            times.append(t)
            states.append((T, P, u, v))

        return np.array(times), states
