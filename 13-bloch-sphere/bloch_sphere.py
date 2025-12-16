#!/usr/bin/env python3
"""
Bloch Sphere in Quantum Computing
==================================

This module demonstrates the Bloch sphere representation, including:
- Bloch vector representation of qubit states
- Conversion between state vectors and Bloch coordinates
- Visualization of common quantum states
- Quantum gates as rotations on the Bloch sphere
- Measurement probabilities from Bloch coordinates
- Applications in quantum computing
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

np.random.seed(42)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def bloch_vector(theta, phi):
    """
    Convert spherical coordinates to Bloch vector (Cartesian).
    
    Args:
        theta: Polar angle [0, π]
        phi: Azimuthal angle [0, 2π]
    
    Returns:
        Bloch vector (x, y, z)
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


def state_to_bloch(psi):
    """
    Convert quantum state to Bloch vector.
    
    Args:
        psi: Quantum state [alpha, beta]
    
    Returns:
        Bloch vector (x, y, z)
    """
    alpha, beta = psi[0], psi[1]
    
    if np.abs(alpha) > 1e-10:
        phase = np.angle(alpha)
        alpha = alpha * np.exp(-1j * phase)
        beta = beta * np.exp(-1j * phase)
    
    theta = 2 * np.arccos(np.abs(alpha))
    phi = np.angle(beta) if np.abs(beta) > 1e-10 else 0
    
    return bloch_vector(theta, phi)


def bloch_to_state(theta, phi):
    """
    Convert Bloch coordinates to quantum state.
    
    Args:
        theta: Polar angle [0, π]
        phi: Azimuthal angle [0, 2π]
    
    Returns:
        Quantum state [alpha, beta]
    """
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    return np.array([alpha, beta], dtype=complex)


def bloch_sphere_basics():
    """Demonstrate basic Bloch sphere concepts."""
    print_section("Bloch Sphere Basics")
    
    print("\nThe Bloch sphere represents single-qubit states geometrically.")
    print("Every pure state corresponds to a point on the surface of a unit sphere.")
    
    print("\n" + "-" * 70)
    print("Bloch Sphere Representation")
    print("-" * 70)
    
    print("\nState: |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ) sin(θ/2)|1⟩")
    print("Bloch vector: r⃗ = (sin(θ)cos(φ), sin(θ)sin(φ), cos(θ))")
    
    print("\n" + "-" * 70)
    print("Common States on the Bloch Sphere")
    print("-" * 70)
    
    states = {
        '|0⟩': (0, 0),
        '|1⟩': (np.pi, 0),
        '|+⟩': (np.pi/2, 0),
        '|−⟩': (np.pi/2, np.pi),
        '|+i⟩': (np.pi/2, np.pi/2),
        '|−i⟩': (np.pi/2, 3*np.pi/2)
    }
    
    for name, (theta, phi) in states.items():
        vec = bloch_vector(theta, phi)
        state = bloch_to_state(theta, phi)
        print(f"\n{name}:")
        print(f"  θ = {theta:.4f}, φ = {phi:.4f}")
        print(f"  Bloch vector: ({vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f})")
        print(f"  State: [{state[0]:.3f}, {state[1]:.3f}]")


def visualize_bloch_sphere():
    """Create a 3D visualization of the Bloch sphere."""
    print_section("Bloch Sphere Visualization")
    
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.1, color='cyan', edgecolor='none')
    
    ax.plot_wireframe(x, y, z, alpha=0.2, color='gray', linewidth=0.5)
    
    axis_length = 1.3
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2)
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.1, linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1, linewidth=2)
    
    ax.text(axis_length + 0.1, 0, 0, 'X', fontsize=14, fontweight='bold')
    ax.text(0, axis_length + 0.1, 0, 'Y', fontsize=14, fontweight='bold')
    ax.text(0, 0, axis_length + 0.1, 'Z', fontsize=14, fontweight='bold')
    
    states = {
        '|0⟩': (0, 0, 'blue'),
        '|1⟩': (np.pi, 0, 'red'),
        '|+⟩': (np.pi/2, 0, 'green'),
        '|−⟩': (np.pi/2, np.pi, 'orange'),
        '|+i⟩': (np.pi/2, np.pi/2, 'purple'),
        '|−i⟩': (np.pi/2, 3*np.pi/2, 'brown')
    }
    
    for name, (theta, phi, color) in states.items():
        vec = bloch_vector(theta, phi)
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], 
                 color=color, arrow_length_ratio=0.15, linewidth=2.5, alpha=0.8)
        ax.text(vec[0]*1.15, vec[1]*1.15, vec[2]*1.15, name, 
               fontsize=11, fontweight='bold', color=color)
    
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')
    ax.set_title('Bloch Sphere with Common Quantum States', fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/13-bloch-sphere/bloch_sphere_3d.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: bloch_sphere_3d.png")
    plt.close()


def quantum_gates_as_rotations():
    """Demonstrate quantum gates as rotations on the Bloch sphere."""
    print_section("Quantum Gates as Rotations")
    
    print("\nQuantum gates correspond to rotations on the Bloch sphere.")
    
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    
    ket0 = np.array([1, 0], dtype=complex)
    
    print("\n" + "-" * 70)
    print("Pauli Gates")
    print("-" * 70)
    
    print("\nX Gate: 180° rotation around x-axis")
    state_X = X @ ket0
    bloch_X = state_to_bloch(state_X)
    print(f"  X|0⟩ = {state_X} → Bloch: {bloch_X}")
    print(f"  |0⟩ (north pole) → |1⟩ (south pole)")
    
    print("\nY Gate: 180° rotation around y-axis")
    state_Y = Y @ ket0
    bloch_Y = state_to_bloch(state_Y)
    print(f"  Y|0⟩ = {state_Y} → Bloch: {bloch_Y}")
    
    print("\nZ Gate: 180° rotation around z-axis")
    ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    state_Z = Z @ ket_plus
    bloch_Z = state_to_bloch(state_Z)
    print(f"  Z|+⟩ = {state_Z} → Bloch: {bloch_Z}")
    print(f"  |+⟩ (positive x) → |−⟩ (negative x)")
    
    print("\n" + "-" * 70)
    print("Hadamard Gate")
    print("-" * 70)
    
    print("\nH Gate: 180° rotation around (x+z)/√2 axis")
    state_H = H @ ket0
    bloch_H = state_to_bloch(state_H)
    print(f"  H|0⟩ = {state_H} → Bloch: {bloch_H}")
    print(f"  |0⟩ (north pole) → |+⟩ (positive x-axis)")
    
    fig = plt.figure(figsize=(16, 12))
    
    gates_to_plot = [
        ('X|0⟩', X @ ket0, 'X: 180° around x-axis'),
        ('Y|0⟩', Y @ ket0, 'Y: 180° around y-axis'),
        ('Z|+⟩', Z @ ket_plus, 'Z: 180° around z-axis'),
        ('H|0⟩', H @ ket0, 'H: Creates superposition')
    ]
    
    for idx, (name, state, title) in enumerate(gates_to_plot, 1):
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        
        u = np.linspace(0, 2*np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, alpha=0.1, color='cyan', edgecolor='none')
        
        ax.quiver(0, 0, 0, 1.2, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=1.5, alpha=0.5)
        ax.quiver(0, 0, 0, 0, 1.2, 0, color='green', arrow_length_ratio=0.1, linewidth=1.5, alpha=0.5)
        ax.quiver(0, 0, 0, 0, 0, 1.2, color='blue', arrow_length_ratio=0.1, linewidth=1.5, alpha=0.5)
        
        vec0 = state_to_bloch(ket0)
        ax.quiver(0, 0, 0, vec0[0], vec0[1], vec0[2], 
                 color='gray', arrow_length_ratio=0.15, linewidth=2, alpha=0.5, linestyle='--')
        
        vec_final = state_to_bloch(state)
        ax.quiver(0, 0, 0, vec_final[0], vec_final[1], vec_final[2], 
                 color='red', arrow_length_ratio=0.15, linewidth=3)
        
        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)
        ax.set_zlabel('Z', fontsize=9)
        ax.set_title(f'{name}\n{title}', fontsize=11, fontweight='bold')
        
        ax.set_xlim([-1.3, 1.3])
        ax.set_ylim([-1.3, 1.3])
        ax.set_zlim([-1.3, 1.3])
    
    plt.suptitle('Quantum Gates as Rotations on the Bloch Sphere', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/13-bloch-sphere/gates_as_rotations.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: gates_as_rotations.png")
    plt.close()


def measurement_on_bloch_sphere():
    """Demonstrate measurement probabilities on the Bloch sphere."""
    print_section("Measurement on the Bloch Sphere")
    
    print("\nMeasurement in computational basis projects onto z-axis.")
    print("Probabilities depend on z-coordinate of Bloch vector.")
    
    print("\n" + "-" * 70)
    print("Measurement Probabilities")
    print("-" * 70)
    
    print("\nP(0) = (1 + z)/2")
    print("P(1) = (1 - z)/2")
    
    test_states = [
        ('|0⟩', 0, 0),
        ('|1⟩', np.pi, 0),
        ('|+⟩', np.pi/2, 0),
        ('|−⟩', np.pi/2, np.pi),
        ('Custom', np.pi/3, np.pi/4)
    ]
    
    print("\n" + "-" * 70)
    print("Examples")
    print("-" * 70)
    
    for name, theta, phi in test_states:
        vec = bloch_vector(theta, phi)
        z = vec[2]
        P0 = (1 + z) / 2
        P1 = (1 - z) / 2
        
        print(f"\n{name} (θ={theta:.3f}, φ={phi:.3f}):")
        print(f"  z-coordinate: {z:.3f}")
        print(f"  P(0) = {P0:.3f}")
        print(f"  P(1) = {P1:.3f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    z_values = np.linspace(-1, 1, 100)
    P0_values = (1 + z_values) / 2
    P1_values = (1 - z_values) / 2
    
    ax1.plot(z_values, P0_values, 'b-', linewidth=2.5, label='P(0) = (1+z)/2')
    ax1.plot(z_values, P1_values, 'r-', linewidth=2.5, label='P(1) = (1-z)/2')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    ax1.plot(1, 1, 'bo', markersize=10, label='|0⟩ (z=+1)')
    ax1.plot(-1, 0, 'ro', markersize=10, label='|1⟩ (z=-1)')
    ax1.plot(0, 0.5, 'go', markersize=10, label='|+⟩ (z=0)')
    
    ax1.set_xlabel('z-coordinate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax1.set_title('Measurement Probability vs z-coordinate', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-1.1, 1.1])
    ax1.set_ylim([-0.1, 1.1])
    
    states_names = ['|0⟩\n(z=+1)', '|1⟩\n(z=-1)', '|+⟩\n(z=0)', '|−⟩\n(z=0)', 'Custom\n(z=0.5)']
    z_coords = [1, -1, 0, 0, 0.5]
    
    x = np.arange(len(states_names))
    width = 0.35
    
    P0_bars = [(1 + z) / 2 for z in z_coords]
    P1_bars = [(1 - z) / 2 for z in z_coords]
    
    bars1 = ax2.bar(x - width/2, P0_bars, width, label='P(0)', color='steelblue', alpha=0.7, edgecolor='black')
    bars2 = ax2.bar(x + width/2, P1_bars, width, label='P(1)', color='coral', alpha=0.7, edgecolor='black')
    
    ax2.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax2.set_title('Measurement Probabilities for Different States', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(states_names, fontsize=9)
    ax2.legend(fontsize=11)
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/13-bloch-sphere/measurement_probabilities.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: measurement_probabilities.png")
    plt.close()


def pure_vs_mixed_states():
    """Demonstrate pure vs mixed states on the Bloch sphere."""
    print_section("Pure vs Mixed States")
    
    print("\nPure states: |r⃗| = 1 (on the surface)")
    print("Mixed states: |r⃗| < 1 (inside the sphere)")
    print("Maximally mixed: r⃗ = 0 (at the center)")
    
    print("\n" + "-" * 70)
    print("Examples")
    print("-" * 70)
    
    print("\nPure state: |+⟩")
    vec_pure = state_to_bloch(np.array([1, 1]) / np.sqrt(2))
    norm_pure = np.linalg.norm(vec_pure)
    print(f"  Bloch vector: {vec_pure}")
    print(f"  |r⃗| = {norm_pure:.6f} (on surface)")
    
    print("\nMixed state: ρ = 0.7|0⟩⟨0| + 0.3|1⟩⟨1|")
    vec_mixed = np.array([0, 0, 0.4])  # z = 0.7 - 0.3 = 0.4
    norm_mixed = np.linalg.norm(vec_mixed)
    print(f"  Bloch vector: {vec_mixed}")
    print(f"  |r⃗| = {norm_mixed:.6f} (inside sphere)")
    
    print("\nMaximally mixed: ρ = I/2")
    vec_max_mixed = np.array([0, 0, 0])
    norm_max_mixed = np.linalg.norm(vec_max_mixed)
    print(f"  Bloch vector: {vec_max_mixed}")
    print(f"  |r⃗| = {norm_max_mixed:.6f} (at center)")
    
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.1, color='cyan', edgecolor='none')
    
    ax.quiver(0, 0, 0, 1.3, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2, alpha=0.5)
    ax.quiver(0, 0, 0, 0, 1.3, 0, color='green', arrow_length_ratio=0.1, linewidth=2, alpha=0.5)
    ax.quiver(0, 0, 0, 0, 0, 1.3, color='blue', arrow_length_ratio=0.1, linewidth=2, alpha=0.5)
    
    ax.quiver(0, 0, 0, vec_pure[0], vec_pure[1], vec_pure[2], 
             color='blue', arrow_length_ratio=0.15, linewidth=3, label='Pure: |+⟩')
    
    ax.quiver(0, 0, 0, vec_mixed[0], vec_mixed[1], vec_mixed[2], 
             color='orange', arrow_length_ratio=0.2, linewidth=3, label='Mixed: 0.7|0⟩⟨0|+0.3|1⟩⟨1|')
    
    ax.scatter([0], [0], [0], color='red', s=200, marker='o', label='Max Mixed: I/2')
    
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')
    ax.set_title('Pure vs Mixed States on Bloch Sphere', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left')
    
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/13-bloch-sphere/pure_vs_mixed.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: pure_vs_mixed.png")
    plt.close()


def applications():
    """Demonstrate applications of the Bloch sphere."""
    print_section("Applications in Quantum Computing")
    
    print("\n1. Quantum Algorithm Visualization")
    print("   - Visualize state evolution as trajectories on the sphere")
    print("   - Understand how algorithms manipulate qubit states")
    
    print("\n2. Gate Optimization")
    print("   - Decompose arbitrary gates into rotation sequences")
    print("   - Minimize gate count for specific transformations")
    
    print("\n3. Error Analysis")
    print("   - Visualize how noise affects quantum states")
    print("   - Understand decoherence as movement toward center")
    
    print("\n4. Quantum Control")
    print("   - Design control pulses for desired state trajectories")
    print("   - Optimize pulse sequences for robust operations")
    
    print("\n5. Quantum Tomography")
    print("   - Reconstruct unknown states from measurements")
    print("   - Visualize reconstructed states on Bloch sphere")


def visualize_summary():
    """Create a comprehensive summary visualization."""
    print_section("Summary Visualization")
    
    fig = plt.figure(figsize=(16, 12))
    
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x, y, z, alpha=0.1, color='cyan', edgecolor='none')
    
    states = [
        (np.array([0, 0, 1]), 'blue', '|0⟩'),
        (np.array([0, 0, -1]), 'red', '|1⟩'),
        (np.array([1, 0, 0]), 'green', '|+⟩')
    ]
    for vec, color, label in states:
        ax1.quiver(0, 0, 0, vec[0], vec[1], vec[2], 
                  color=color, arrow_length_ratio=0.15, linewidth=2)
    
    ax1.set_title('Bloch Sphere', fontsize=11, fontweight='bold')
    ax1.set_xlim([-1.2, 1.2])
    ax1.set_ylim([-1.2, 1.2])
    ax1.set_zlim([-1.2, 1.2])
    
    ax2 = plt.subplot(2, 3, 2)
    z_vals = np.linspace(-1, 1, 100)
    P0 = (1 + z_vals) / 2
    P1 = (1 - z_vals) / 2
    ax2.plot(z_vals, P0, 'b-', linewidth=2, label='P(0)')
    ax2.plot(z_vals, P1, 'r-', linewidth=2, label='P(1)')
    ax2.set_xlabel('z-coordinate', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Probability', fontsize=10, fontweight='bold')
    ax2.set_title('Measurement Probabilities', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 3, 3)
    categories = ['Pure\n(surface)', 'Partial\n(inside)', 'Max Mixed\n(center)']
    norms = [1.0, 0.6, 0.0]
    colors_bar = ['green', 'orange', 'red']
    bars = ax3.bar(categories, norms, color=colors_bar, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('|r⃗|', fontsize=10, fontweight='bold')
    ax3.set_title('Purity: Bloch Vector Norm', fontsize=11, fontweight='bold')
    ax3.set_ylim([0, 1.2])
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, norm in zip(bars, norms):
        ax3.text(bar.get_x() + bar.get_width()/2, norm + 0.05, f'{norm:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax4 = plt.subplot(2, 3, 4)
    gates = ['X\n(x-axis)', 'Y\n(y-axis)', 'Z\n(z-axis)', 'H\n((x+z)/√2)']
    angles = [180, 180, 180, 180]
    colors_gates = ['red', 'green', 'blue', 'purple']
    bars = ax4.bar(gates, angles, color=colors_gates, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Rotation Angle (°)', fontsize=10, fontweight='bold')
    ax4.set_title('Quantum Gates as Rotations', fontsize=11, fontweight='bold')
    ax4.set_ylim([0, 200])
    ax4.grid(True, alpha=0.3, axis='y')
    
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    table_data = [
        ['State', 'θ', 'φ', 'z'],
        ['|0⟩', '0', '0', '+1'],
        ['|1⟩', 'π', '0', '-1'],
        ['|+⟩', 'π/2', '0', '0'],
        ['|−⟩', 'π/2', 'π', '0'],
        ['|+i⟩', 'π/2', 'π/2', '0']
    ]
    table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#64ffda')
        table[(0, i)].set_text_props(weight='bold')
    
    ax5.set_title('Common State Coordinates', fontsize=11, fontweight='bold', pad=20)
    
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    properties_text = """
    KEY PROPERTIES
    
    • Pure states: |r⃗| = 1 (surface)
    • Mixed states: |r⃗| < 1 (inside)
    • Orthogonal states: antipodal
    • Gates: rotations on sphere
    • P(0) = (1+z)/2, P(1) = (1-z)/2
    • Single qubit only
    • Global phase invisible
    """
    ax6.text(0.1, 0.5, properties_text, fontsize=10, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Bloch Sphere - Summary', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/13-bloch-sphere/bloch_sphere_summary.png', dpi=150, bbox_inches='tight')
    print("\n✓ Summary visualization saved: bloch_sphere_summary.png")
    plt.close()


def main():
    """Main function to run all demonstrations."""
    print("\n" + "=" * 70)
    print("  BLOCH SPHERE IN QUANTUM COMPUTING")
    print("=" * 70)
    print("\nThis module demonstrates the Bloch sphere representation")
    print("of qubit states in quantum computing.")
    
    try:
        bloch_sphere_basics()
        visualize_bloch_sphere()
        quantum_gates_as_rotations()
        measurement_on_bloch_sphere()
        pure_vs_mixed_states()
        applications()
        visualize_summary()
        
        print("\n" + "=" * 70)
        print("  ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nVisualizations saved in: /home/ubuntu/repos/vectorsforquantum/13-bloch-sphere/")
        print("\nKey Takeaways:")
        print("1. Bloch sphere represents single-qubit states geometrically")
        print("2. Pure states on surface (|r⃗|=1), mixed states inside (|r⃗|<1)")
        print("3. State: |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ) sin(θ/2)|1⟩")
        print("4. Quantum gates are rotations on the sphere")
        print("5. Measurement probabilities: P(0)=(1+z)/2, P(1)=(1-z)/2")
        print("6. Orthogonal states are antipodal points")
        print("7. Only works for single qubits (not multi-qubit systems)")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
