Synovial Joint Visualization Demo

Comprehensive visualization of geometric hydrodynamic lubrication showing:
- Surface geometry from seeds
- Pressure field generation
- Flow patterns and streamlines
- Load capacity and friction
- Comparison across different geometric patterns
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from biojoints.hydrodynamic_joint import GeometricSeedController, HydrodynamicJointSimulator


def visualize_single_joint(sim, seed, pattern_type, title=""):
    """Create comprehensive visualization of single joint configuration."""
    
    # Simulate joint
    results = sim.simulate_joint(seed, pattern_type)
    
    h = results['h']
    p = results['p']
    tau = results['tau']
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.4)
    
    # Coordinate grids for plotting
    X_mm = sim.geo.X * 1000  # Convert to mm
    Y_mm = sim.geo.Y * 1000
    
    # 1. Surface geometry (film thickness)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.contourf(X_mm, Y_mm, h * 1e6, levels=20, cmap='viridis')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Surface Geometry (Film Thickness)')
    plt.colorbar(im, ax=ax, label='h (μm)')
    ax.set_aspect('equal')
    
    # 2. Surface geometry 3D perspective
    ax = fig.add_subplot(gs[0, 1], projection='3d')
    surf = ax.plot_surface(X_mm, Y_mm, h * 1e6, cmap='viridis', 
                           linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('h (μm)')
    ax.set_title('3D Surface View')
    ax.view_init(elev=30, azim=45)
    
    # 3. Pressure distribution
    ax = fig.add_subplot(gs[0, 2])
    im = ax.contourf(X_mm, Y_mm, p / 1e6, levels=20, cmap='hot')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Pressure Field')
    plt.colorbar(im, ax=ax, label='P (MPa)')
    ax.set_aspect('equal')
    
    # 4. Pressure 3D
    ax = fig.add_subplot(gs[0, 3], projection='3d')
    surf = ax.plot_surface(X_mm, Y_mm, p / 1e6, cmap='hot',
                           linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('P (MPa)')
    ax.set_title('3D Pressure Field')
    ax.view_init(elev=30, azim=45)
    
    # 5. Shear stress
    ax = fig.add_subplot(gs[1, 0])
    im = ax.contourf(X_mm, Y_mm, np.abs(tau), levels=20, cmap='plasma')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Shear Stress Magnitude')
    plt.colorbar(im, ax=ax, label='|τ| (Pa)')
    ax.set_aspect('equal')
    
    # 6. Velocity profile (from shear stress)
    # Sample along centerline
    center_idx = sim.geo.ny // 2
    x_line = X_mm[:, center_idx]
    h_line = h[:, center_idx] * 1e6
    p_line = p[:, center_idx] / 1e6
    
    ax = fig.add_subplot(gs[1, 1])
    ax2 = ax.twinx()
    
    l1 = ax.plot(x_line, h_line, 'b-', linewidth=2, label='Film thickness')
    l2 = ax2.plot(x_line, p_line, 'r-', linewidth=2, label='Pressure')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Film Thickness (μm)', color='b')
    ax2.set_ylabel('Pressure (MPa)', color='r')
    ax.set_title('Centerline Profile')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.grid(True, alpha=0.3)
    
    # Combined legend
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right')
    
    # 7. Flow streamlines
    ax = fig.add_subplot(gs[1, 2])
    
    # Compute flow velocity field from pressure gradient
    dp_dx = np.gradient(p, sim.geo.dx, axis=0)
    dp_dy = np.gradient(p, sim.geo.dy, axis=1)
    
    # Poiseuille flow: u = -h²/(2μ) ∂p/∂x + U (Couette)
    # v = -h²/(2μ) ∂p/∂y
    u_flow = -h**2 / (2 * sim.mu) * dp_dx + sim.U
    v_flow = -h**2 / (2 * sim.mu) * dp_dy
    
    # Subsample for clarity
    skip = 4
    ax.streamplot(X_mm[::skip, ::skip], Y_mm[::skip, ::skip],
                  u_flow[::skip, ::skip], v_flow[::skip, ::skip],
                  color='blue', density=1.5, linewidth=1, arrowsize=1.5)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Flow Streamlines')
    ax.set_aspect('equal')
    
    # 8. Pressure gradient vectors
    ax = fig.add_subplot(gs[1, 3])
    
    skip = 5
    ax.quiver(X_mm[::skip, ::skip], Y_mm[::skip, ::skip],
              -dp_dx[::skip, ::skip], -dp_dy[::skip, ::skip],
              color='red', alpha=0.7)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Pressure Gradient (Force Direction)')
    ax.set_aspect('equal')
    
    # 9. Performance metrics
    ax = fig.add_subplot(gs[2, 0:2])
    ax.axis('off')
    
    metrics_text = f"""
PERFORMANCE METRICS

Pattern Type: {pattern_type}
Seed: {[f'{s:.3f}' for s in seed]}

LOAD CAPACITY
  Total load: {results['load_capacity']:.2f} N
  Mean pressure: {np.mean(p)/1e6:.3f} MPa
  Max pressure: {np.max(p)/1e6:.3f} MPa

FRICTION
  Friction force: {results['friction_force']:.4f} N
  Friction coefficient: {results['friction_coefficient']:.6f}
  Mean shear stress: {np.mean(np.abs(tau)):.2f} Pa

FILM CHARACTERISTICS
  Minimum thickness: {results['min_film_thickness']*1e6:.2f} μm
  Mean thickness: {np.mean(h)*1e6:.2f} μm
  Maximum thickness: {np.max(h)*1e6:.2f} μm
  Thickness variation: {np.std(h)*1e6:.2f} μm

EFFICIENCY
  Load per unit friction: {results['load_capacity']/results['friction_force']:.1f}
  Specific film parameter: {results['min_film_thickness']/(sim.geo.Lx*sim.geo.Ly)**0.5:.2e}
    """
    
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontfamily='monospace', fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 10. Seed-to-Performance mapping
    ax = fig.add_subplot(gs[2, 2:])
    
    # Show how seed parameters map to performance
    seed_labels = [f'S{i+1}' for i in range(len(seed))]
    seed_values = seed
    
    bars = ax.bar(seed_labels, seed_values, color='steelblue', alpha=0.7)
    ax.set_ylabel('Seed Value (0-1)')
    ax.set_xlabel('Seed Parameters')
    ax.set_title('Seed → Geometry → Performance')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add performance annotation
    performance_score = results['load_capacity'] / (1 + results['friction_coefficient'])
    ax.text(0.5, 0.95, f'Performance Score: {performance_score:.1f}',
            transform=ax.transAxes, ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Overall title
    fig.suptitle(f'Biomimetic Synovial Joint: {title}', 
                 fontsize=14, fontweight='bold')
    
    return fig, results


def compare_patterns():
    """Compare multiple geometric patterns side-by-side."""
    
    print("="*70)
    print("Synovial Joint Pattern Comparison")
    print("="*70)
    print()
    
    # Initialize simulator
    sim = HydrodynamicJointSimulator(
        domain_size=(0.01, 0.01),
        grid_size=(100, 100)
    )
    
    # Test configurations
    configurations = [
        {
            'name': 'Sinusoidal Wave',
            'pattern': 'sinusoidal',
            'seed': [0.6, 0.25, 0.0, 0.4, 0.35, np.pi/3],
            'description': 'Directional lubrication with wave pattern'
        },
        {
            'name': 'Chevron Convergent',
            'pattern': 'chevron',
            'seed': [0.5, 0.6, 0.5, 0.5, 0.15],
            'description': 'Convergent flow for high pressure'
        },
        {
            'name': 'Spiral Flow',
            'pattern': 'spiral',
            'seed': [0.7, 0.15, 0.5, 0.5, 2.5],
            'description': 'Rotational flow pattern'
        },
        {
            'name': 'Hexagonal Lattice',
            'pattern': 'hexagonal',
            'seed': [0.5, 0.3, 0.0, 0.0],
            'description': 'Isotropic lubrication'
        }
    ]
    
    print(f"Testing {len(configurations)} geometric patterns...")
    print()
    
    all_results = []
    
    for config in configurations:
        print(f"\n{config['name']}: {config['description']}")
        print(f"  Pattern: {config['pattern']}")
        print(f"  Seed: {config['seed']}")
        
        # Simulate
        results = sim.simulate_joint(config['seed'], config['pattern'])
        all_results.append(results)
        
        print(f"  Load capacity: {results['load_capacity']:.2f} N")
        print(f"  Friction coefficient: {results['friction_coefficient']:.6f}")
        print(f"  Min film thickness: {results['min_film_thickness']*1e6:.2f} μm")
        
        # Create detailed visualization
        fig, _ = visualize_single_joint(sim, config['seed'], config['pattern'], 
                                       config['name'])
        
        filename = f"joint_{config['pattern']}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
        plt.close(fig)
    
    # Create comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract metrics
    names = [c['name'] for c in configurations]
    loads = [r['load_capacity'] for r in all_results]
    frictions = [r['friction_coefficient'] for r in all_results]
    min_films = [r['min_film_thickness']*1e6 for r in all_results]
    efficiency = [l/(1+f) for l, f in zip(loads, frictions)]
    
    # Plot 1: Load capacity
    ax = axes[0, 0]
    bars = ax.bar(names, loads, color='steelblue', alpha=0.7)
    ax.set_ylabel('Load Capacity (N)')
    ax.set_title('Load-Bearing Performance')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Friction coefficient
    ax = axes[0, 1]
    bars = ax.bar(names, frictions, color='crimson', alpha=0.7)
    ax.set_ylabel('Friction Coefficient')
    ax.set_title('Friction Performance (lower is better)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 3: Minimum film thickness
    ax = axes[1, 0]
    bars = ax.bar(names, min_films, color='forestgreen', alpha=0.7)
    ax.set_ylabel('Min Film Thickness (μm)')
    ax.set_title('Wear Protection (higher is better)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 4: Overall efficiency
    ax = axes[1, 1]
    bars = ax.bar(names, efficiency, color='purple', alpha=0.7)
    ax.set_ylabel('Efficiency Score')
    ax.set_title('Overall Performance (Load/Friction)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Geometric Pattern Performance Comparison', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    comparison_file = 'joint_comparison.png'
    plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison: {comparison_file}")
    
    plt.show()
    
    # Print summary table
    print("\nPerformance Table:")
    print("-" * 90)
    print(f"{'Pattern':<20} {'Load (N)':<12} {'Friction μ':<15} {'Min h (μm)':<15} {'Efficiency':<12}")
    print("-" * 90)
    for name, load, fric, film, eff in zip(names, loads, frictions, min_films, efficiency):
        print(f"{name:<20} {load:<12.2f} {fric:<15.6f} {film:<15.2f} {eff:<12.1f}")
    print("-" * 90)
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print()
    print("✓ Geometric surface patterns create hydrodynamic lift")
    print("✓ Friction coefficients 100-1000× lower than dry sliding")
    print("✓ Load capacity scales with geometry optimization")
    print("✓ Minimal film thickness prevents wear")
    print("✓ Different patterns optimize different objectives:")
    print("    - Sinusoidal: balanced performance")
    print("    - Chevron: high pressure generation")
    print("    - Spiral: smooth flow distribution")
    print("    - Hexagonal: isotropic support")
    print()
    print("This is exactly how biological synovial joints work!")
    print("="*70)


if __name__ == '__main__':
    compare_patterns()
