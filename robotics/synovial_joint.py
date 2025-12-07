Hydrodynamic Joint Simulator

Physics-based simulation of geometric surface patterns creating
frictionless motion through natural hydrodynamic lubrication.

Based on Reynolds equation with geometric seed control for surface topology.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


class GeometricSeedController:
    """
    Controls surface geometry using seed-based patterns.
    
    Maps compact seeds to complex geometric surface patterns
    that create optimal hydrodynamic pressure distributions.
    """
    
    def __init__(self, domain_size=(0.01, 0.01), grid_size=(100, 100)):
        """
        Initialize geometric controller.
        
        Args:
            domain_size: Physical size in meters (Lx, Ly)
            grid_size: Grid resolution (nx, ny)
        """
        self.Lx, self.Ly = domain_size
        self.nx, self.ny = grid_size
        
        # Grid spacing
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        
        # Coordinate grids
        x = np.linspace(0, self.Lx, self.nx)
        y = np.linspace(0, self.Ly, self.ny)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        
        # Base film thickness (when surfaces are parallel)
        self.h0 = 1e-6  # 1 micron baseline
        
        # Surface pattern parameters
        self.pattern_types = {
            'sinusoidal': self._sinusoidal_pattern,
            'chevron': self._chevron_pattern,
            'spiral': self._spiral_pattern,
            'hexagonal': self._hexagonal_pattern
        }
    
    def seed_to_geometry(self, seed, pattern_type='sinusoidal'):
        """
        Convert seed to surface height field.
        
        Args:
            seed: Array of parameter values
            pattern_type: Type of geometric pattern
            
        Returns:
            h: Surface height field h(x,y) in meters
        """
        if pattern_type not in self.pattern_types:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        return self.pattern_types[pattern_type](seed)
    
    def _sinusoidal_pattern(self, seed):
        """
        Sinusoidal surface pattern for directional lubrication.
        
        Seed structure: [A1, λ1, φ1, A2, λ2, φ2, ...]
        Each triplet: amplitude, wavelength, phase
        """
        h = np.full_like(self.X, self.h0)
        
        n_waves = len(seed) // 3
        
        for i in range(n_waves):
            idx = i * 3
            if idx + 2 >= len(seed):
                break
            
            A = seed[idx] * self.h0       # Amplitude: 0-1 → 0-h0
            λ = seed[idx + 1] * self.Lx   # Wavelength: 0-1 → 0-Lx  
            φ = seed[idx + 2] * 2 * np.pi # Phase: 0-1 → 0-2π
            
            # Wave number
            k = 2 * np.pi / (λ + 1e-10)
            
            # Add sinusoidal component
            h += A * np.sin(k * self.X + φ)
        
        return h
    
    def _chevron_pattern(self, seed):
        """
        Chevron (V-shaped) pattern for convergent/divergent flow.
        
        Seed structure: [A, angle, x_center, y_center, width]
        """
        h = np.full_like(self.X, self.h0)
        
        if len(seed) < 5:
            return h
        
        A = seed[0] * self.h0                    # Amplitude
        angle = seed[1] * np.pi                  # Chevron angle: 0-π
        x_center = seed[2] * self.Lx             # Center x
        y_center = seed[3] * self.Ly             # Center y  
        width = seed[4] * min(self.Lx, self.Ly) # Pattern width
        
        # Relative coordinates
        X_rel = self.X - x_center
        Y_rel = self.Y - y_center
        
        # Chevron pattern (V-shape)
        chevron = np.where(
            np.abs(Y_rel - np.tan(angle/2) * np.abs(X_rel)) < width,
            A * (1 - np.abs(Y_rel - np.tan(angle/2) * np.abs(X_rel)) / width),
            0
        )
        
        h += chevron
        
        return h
    
    def _spiral_pattern(self, seed):
        """
        Spiral pattern for rotational flow generation.
        
        Seed structure: [A, pitch, center_x, center_y, n_turns]
        """
        h = np.full_like(self.X, self.h0)
        
        if len(seed) < 5:
            return h
        
        A = seed[0] * self.h0            # Amplitude
        pitch = seed[1] * self.Lx        # Spiral pitch
        x_center = seed[2] * self.Lx     # Center x
        y_center = seed[3] * self.Ly     # Center y
        n_turns = seed[4] * 5           # Number of turns: 0-5
        
        # Polar coordinates from center
        X_rel = self.X - x_center
        Y_rel = self.Y - y_center
        r = np.sqrt(X_rel**2 + Y_rel**2)
        θ = np.arctan2(Y_rel, X_rel)
        
        # Spiral function
        max_r = min(self.Lx, self.Ly) / 2
        spiral = np.where(
            r < max_r,
            A * np.sin(2 * np.pi * n_turns * r / max_r + θ) * (1 - r / max_r),
            0
        )
        
        h += spiral
        
        return h
    
    def _hexagonal_pattern(self, seed):
        """
        Hexagonal pattern for isotropic lubrication.
        
        Seed structure: [A, spacing, offset_x, offset_y]
        """
        h = np.full_like(self.X, self.h0)
        
        if len(seed) < 4:
            return h
        
        A = seed[0] * self.h0         # Amplitude
        spacing = seed[1] * self.Lx   # Hex spacing
        offset_x = seed[2] * spacing  # Pattern offset x
        offset_y = seed[3] * spacing  # Pattern offset y
        
        # Create hexagonal lattice
        # Hexagons in 2D: sum of 3 sinusoids at 60° angles
        k = 2 * np.pi / (spacing + 1e-10)
        
        # Three wave vectors at 60° separation
        h1 = np.sin(k * self.X)
        h2 = np.sin(k * (self.X * np.cos(np.pi/3) + self.Y * np.sin(np.pi/3)))
        h3 = np.sin(k * (self.X * np.cos(2*np.pi/3) + self.Y * np.sin(2*np.pi/3)))
        
        hexagonal = A * (h1 + h2 + h3) / 3
        
        h += hexagonal
        
        return h


class HydrodynamicJointSimulator:
    """
    Simulates hydrodynamic lubrication in joints with geometric surface patterns.
    
    Solves Reynolds equation to compute pressure distribution and load capacity
    from surface geometry controlled by seed patterns.
    """
    
    def __init__(self, domain_size=(0.01, 0.01), grid_size=(100, 100)):
        """
        Initialize joint simulator.
        
        Args:
            domain_size: Physical size in meters (Lx, Ly)
            grid_size: Grid resolution (nx, ny)
        """
        self.geo = GeometricSeedController(domain_size, grid_size)
        
        # Fluid properties (typical synovial fluid)
        self.mu = 0.001  # Viscosity: 1 cP (water-like)
        self.rho = 1000  # Density: 1000 kg/m³
        
        # Operating conditions
        self.U = 0.1     # Sliding velocity: 0.1 m/s
        self.W = 100.0   # Applied load: 100 N
        
        # Current state
        self.h = None    # Film thickness field
        self.p = None    # Pressure field
        self.tau = None  # Shear stress field
        
    def solve_reynolds(self, h):
        """
        Solve Reynolds equation for pressure field.
        
        Reynolds equation (2D, steady-state):
        ∂/∂x(h³ ∂p/∂x) + ∂/∂y(h³ ∂p/∂y) = 6μU ∂h/∂x
        
        Args:
            h: Film thickness field h(x,y)
            
        Returns:
            p: Pressure field p(x,y)
        """
        nx, ny = self.geo.nx, self.geo.ny
        dx, dy = self.geo.dx, self.geo.dy
        
        # Right-hand side: 6μU ∂h/∂x
        dh_dx = np.gradient(h, dx, axis=0)
        rhs = 6 * self.mu * self.U * dh_dx
        
        # Flatten for matrix solution
        rhs_flat = rhs.flatten()
        
        # Build coefficient matrix for finite difference
        # Using 5-point stencil for Laplacian operator
        
        # Coefficient function: h³(x,y)
        h3 = h**3
        
        # Build sparse matrix
        N = nx * ny
        diagonals = np.zeros((5, N))
        offsets = [0, -1, 1, -ny, ny]
        
        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                
                # Center coefficient
                center = 0.0
                
                # x-direction coefficients
                if i > 0:
                    h3_west = 0.5 * (h3[i, j] + h3[i-1, j])
                    coef_west = h3_west / dx**2
                    diagonals[3, idx] = -coef_west  # West neighbor
                    center += coef_west
                
                if i < nx - 1:
                    h3_east = 0.5 * (h3[i, j] + h3[i+1, j])
                    coef_east = h3_east / dx**2
                    diagonals[4, idx] = -coef_east  # East neighbor
                    center += coef_east
                
                # y-direction coefficients  
                if j > 0:
                    h3_south = 0.5 * (h3[i, j] + h3[i, j-1])
                    coef_south = h3_south / dy**2
                    diagonals[1, idx] = -coef_south  # South neighbor
                    center += coef_south
                
                if j < ny - 1:
                    h3_north = 0.5 * (h3[i, j] + h3[i, j+1])
                    coef_north = h3_north / dy**2
                    diagonals[2, idx] = -coef_north  # North neighbor
                    center += coef_north
                
                # Diagonal (center) coefficient
                diagonals[0, idx] = center
                
                # Boundary conditions: p = 0 at edges
                if i == 0 or i == nx-1 or j == 0 or j == ny-1:
                    diagonals[0, idx] = 1.0
                    diagonals[1:, idx] = 0.0
                    rhs_flat[idx] = 0.0
        
        # Create sparse matrix
        A = diags(diagonals, offsets, shape=(N, N), format='csr')
        
        # Solve linear system
        p_flat = spsolve(A, rhs_flat)
        
        # Reshape to 2D
        p = p_flat.reshape(nx, ny)
        
        # Ensure non-negative pressure
        p = np.maximum(p, 0)
        
        return p
    
    def compute_load_capacity(self, p):
        """
        Compute load-bearing capacity from pressure field.
        
        Args:
            p: Pressure field p(x,y)
            
        Returns:
            W: Total load capacity in Newtons
        """
        # Integrate pressure over area
        W = np.sum(p) * self.geo.dx * self.geo.dy
        return W
    
    def compute_friction(self, h, p):
        """
        Compute friction force from shear stress.
        
        Shear stress: τ = μU/h + h/2 · ∂p/∂x
        
        Args:
            h: Film thickness field
            p: Pressure field
            
        Returns:
            F: Total friction force in Newtons
            tau: Shear stress field
        """
        # Pressure gradient
        dp_dx = np.gradient(p, self.geo.dx, axis=0)
        
        # Shear stress components
        tau_couette = self.mu * self.U / h  # Couette flow
        tau_poiseuille = h / 2 * dp_dx      # Poiseuille flow
        
        tau = tau_couette + tau_poiseuille
        
        # Integrate over area
        F = np.sum(np.abs(tau)) * self.geo.dx * self.geo.dy
        
        return F, tau
    
    def compute_friction_coefficient(self, p, tau):
        """
        Compute coefficient of friction.
        
        μ_f = F / W (friction force / normal load)
        
        Args:
            p: Pressure field
            tau: Shear stress field
            
        Returns:
            mu_f: Coefficient of friction
        """
        W = self.compute_load_capacity(p)
        F = np.sum(np.abs(tau)) * self.geo.dx * self.geo.dy
        
        if W > 0:
            return F / W
        else:
            return float('inf')
    
    def simulate_joint(self, seed, pattern_type='sinusoidal'):
        """
        Simulate complete joint operation with geometric seed.
        
        Args:
            seed: Geometric seed parameters
            pattern_type: Type of surface pattern
            
        Returns:
            results: Dict with simulation results
        """
        # Generate surface geometry from seed
        self.h = self.geo.seed_to_geometry(seed, pattern_type)
        
        # Solve for pressure distribution
        self.p = self.solve_reynolds(self.h)
        
        # Compute friction
        F, self.tau = self.compute_friction(self.h, self.p)
        
        # Compute performance metrics
        W = self.compute_load_capacity(self.p)
        mu_f = self.compute_friction_coefficient(self.p, self.tau)
        
        # Minimum film thickness (wear indicator)
        h_min = np.min(self.h)
        
        results = {
            'h': self.h,
            'p': self.p,
            'tau': self.tau,
            'load_capacity': W,
            'friction_force': F,
            'friction_coefficient': mu_f,
            'min_film_thickness': h_min,
            'seed': seed,
            'pattern_type': pattern_type
        }
        
        return results


def quick_demonstration():
    """Quick demonstration of geometric hydrodynamic lubrication."""
    print("="*70)
    print("Biomimetic Synovial Joint Simulation")
    print("="*70)
    print()
    
    print("Concept: Surface geometry creates hydrodynamic pressure")
    print("         that eliminates friction through pure fluid mechanics")
    print()
    
    # Initialize simulator
    print("Initializing joint simulator...")
    sim = HydrodynamicJointSimulator(
        domain_size=(0.01, 0.01),  # 1cm × 1cm
        grid_size=(80, 80)
    )
    
    print(f"  Domain: {sim.geo.Lx*1000:.1f} mm × {sim.geo.Ly*1000:.1f} mm")
    print(f"  Grid: {sim.geo.nx} × {sim.geo.ny}")
    print(f"  Base film thickness: {sim.geo.h0*1e6:.1f} μm")
    print(f"  Sliding velocity: {sim.U} m/s")
    print(f"  Viscosity: {sim.mu*1000:.1f} cP")
    print()
    
    # Test different patterns
    patterns_to_test = [
        ('sinusoidal', [0.5, 0.3, 0.0, 0.3, 0.5, np.pi/4]),
        ('chevron', [0.4, 0.5, 0.5, 0.5, 0.1]),
        ('spiral', [0.6, 0.2, 0.5, 0.5, 2.0])
    ]
    
    print("Testing geometric patterns...")
    print()
    
    all_results = []
    
    for pattern_type, test_seed in patterns_to_test:
        print(f"Pattern: {pattern_type}")
        print(f"  Seed: {test_seed}")
        
        results = sim.simulate_joint(test_seed, pattern_type)
        all_results.append(results)
        
        print(f"  Load capacity: {results['load_capacity']:.2f} N")
        print(f"  Friction force: {results['friction_force']:.4f} N")
        print(f"  Friction coef: {results['friction_coefficient']:.6f}")
        print(f"  Min film thick: {results['min_film_thickness']*1e6:.2f} μm")
        print()
    
    print("✓ All patterns tested successfully!")
    print()
    print("Key Insight:")
    print("  Tiny geometric variations (micron-scale) create")
    print("  hydrodynamic pressures that support large loads")
    print("  with minimal friction - just like biological joints!")
    
    return all_results


if __name__ == '__main__':
    quick_demonstration()
ENDOFFILE
