"""
Vectors in Quantum Computing - Concept 1: Introduction
=====================================================

This module provides comprehensive demonstrations of vectors as the foundation
of quantum computing, including classical vs quantum vectors, state representation,
probability amplitudes, and measurement theory.

Author: Devin AI
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.patches as mpatches

np.random.seed(42)


class Arrow3D(FancyArrowPatch):
    """Helper class for drawing 3D arrows in matplotlib."""
    
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_subheader(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---\n")


def classical_vectors_demo():
    """Demonstrate classical vectors with real components."""
    print_header("1. CLASSICAL VECTORS")
    
    print("Classical vectors represent physical quantities with magnitude and direction.")
    print("Examples: force, velocity, displacement, momentum\n")
    
    print_subheader("Example 1: Velocity Vector")
    velocity = np.array([3.0, 4.0])
    magnitude = np.linalg.norm(velocity)
    direction = np.arctan2(velocity[1], velocity[0]) * 180 / np.pi
    
    print(f"Velocity vector: v = {velocity} m/s")
    print(f"Magnitude: |v| = {magnitude:.2f} m/s")
    print(f"Direction: θ = {direction:.2f}° from x-axis")
    
    print_subheader("Example 2: Force Vector")
    force = np.array([5.0, 12.0])
    magnitude = np.linalg.norm(force)
    direction = np.arctan2(force[1], force[0]) * 180 / np.pi
    
    print(f"Force vector: F = {force} N")
    print(f"Magnitude: |F| = {magnitude:.2f} N")
    print(f"Direction: θ = {direction:.2f}° from x-axis")
    
    print_subheader("Example 3: 3D Displacement Vector")
    displacement = np.array([2.0, 3.0, 6.0])
    magnitude = np.linalg.norm(displacement)
    
    print(f"Displacement vector: d = {displacement} m")
    print(f"Magnitude: |d| = {magnitude:.2f} m")
    
    print_subheader("Classical Vector Operations")
    v1 = np.array([3.0, 4.0])
    v2 = np.array([1.0, 2.0])
    
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v1 + v2 = {v1 + v2}")
    print(f"v1 - v2 = {v1 - v2}")
    print(f"2 * v1 = {2 * v1}")
    print(f"v1 · v2 (dot product) = {np.dot(v1, v2):.2f}")


def quantum_vectors_demo():
    """Demonstrate quantum vectors with complex components."""
    print_header("2. QUANTUM VECTORS")
    
    print("Quantum vectors represent quantum states in complex Hilbert space.")
    print("Components are complex probability amplitudes.\n")
    
    print_subheader("Example 1: Equal Superposition State")
    psi = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    
    print(f"Quantum state: |ψ⟩ = {psi}")
    print(f"In Dirac notation: |ψ⟩ = (1/√2)|0⟩ + (1/√2)|1⟩")
    print(f"\nProbability amplitudes:")
    print(f"  α₀ = {psi[0]}")
    print(f"  α₁ = {psi[1]}")
    print(f"\nMeasurement probabilities:")
    print(f"  P(0) = |α₀|² = {np.abs(psi[0])**2:.3f} (50%)")
    print(f"  P(1) = |α₁|² = {np.abs(psi[1])**2:.3f} (50%)")
    print(f"  Total probability = {np.sum(np.abs(psi)**2):.3f}")
    
    print_subheader("Example 2: State with Complex Amplitudes")
    psi2 = np.array([0.6, 0.8j], dtype=complex)
    
    print(f"Quantum state: |ψ⟩ = {psi2}")
    print(f"In Dirac notation: |ψ⟩ = 0.6|0⟩ + 0.8i|1⟩")
    print(f"\nProbability amplitudes:")
    print(f"  α₀ = {psi2[0]}")
    print(f"  α₁ = {psi2[1]}")
    print(f"\nMeasurement probabilities:")
    print(f"  P(0) = |α₀|² = {np.abs(psi2[0])**2:.3f} (36%)")
    print(f"  P(1) = |α₁|² = {np.abs(psi2[1])**2:.3f} (64%)")
    print(f"  Total probability = {np.sum(np.abs(psi2)**2):.3f}")
    
    print_subheader("Example 3: General Qubit State")
    alpha = 0.8
    beta = 0.6
    psi3 = np.array([alpha, beta], dtype=complex)
    
    print(f"Quantum state: |ψ⟩ = {psi3}")
    print(f"In Dirac notation: |ψ⟩ = {alpha}|0⟩ + {beta}|1⟩")
    print(f"\nMeasurement probabilities:")
    print(f"  P(0) = {np.abs(psi3[0])**2:.3f} ({np.abs(psi3[0])**2*100:.1f}%)")
    print(f"  P(1) = {np.abs(psi3[1])**2:.3f} ({np.abs(psi3[1])**2*100:.1f}%)")
    print(f"  Total probability = {np.sum(np.abs(psi3)**2):.3f}")
    
    print_subheader("Example 4: State with Relative Phase")
    psi4 = np.array([1/np.sqrt(2), np.exp(1j * np.pi/4)/np.sqrt(2)], dtype=complex)
    
    print(f"Quantum state: |ψ⟩ = (1/√2)|0⟩ + e^(iπ/4)(1/√2)|1⟩")
    print(f"Numerical: {psi4}")
    print(f"\nProbability amplitudes:")
    print(f"  α₀ = {psi4[0]:.4f}")
    print(f"  α₁ = {psi4[1]:.4f}")
    print(f"\nMeasurement probabilities:")
    print(f"  P(0) = {np.abs(psi4[0])**2:.3f}")
    print(f"  P(1) = {np.abs(psi4[1])**2:.3f}")
    print(f"  Total probability = {np.sum(np.abs(psi4)**2):.3f}")


def hilbert_space_demo():
    """Demonstrate Hilbert space concepts."""
    print_header("3. HILBERT SPACE")
    
    print("Hilbert space is a complete complex inner product space where:")
    print("  • Vectors represent quantum states")
    print("  • Vector addition represents superposition")
    print("  • Inner products give probability amplitudes")
    print("  • Normalization ensures total probability = 1\n")
    
    print_subheader("Single Qubit Hilbert Space (2D)")
    
    print("Basis states:")
    ket0 = np.array([1, 0], dtype=complex)
    ket1 = np.array([0, 1], dtype=complex)
    print(f"|0⟩ = {ket0}")
    print(f"|1⟩ = {ket1}")
    
    print("\nOrthonormality:")
    print(f"⟨0|0⟩ = {np.vdot(ket0, ket0)}")
    print(f"⟨1|1⟩ = {np.vdot(ket1, ket1)}")
    print(f"⟨0|1⟩ = {np.vdot(ket0, ket1)}")
    print(f"⟨1|0⟩ = {np.vdot(ket1, ket0)}")
    
    print("\nGeneral state as linear combination:")
    alpha = 0.6
    beta = 0.8
    psi = alpha * ket0 + beta * ket1
    print(f"|ψ⟩ = {alpha}|0⟩ + {beta}|1⟩ = {psi}")
    
    print_subheader("Two Qubit Hilbert Space (4D)")
    
    print("Basis states (tensor products):")
    ket00 = np.kron(ket0, ket0)
    ket01 = np.kron(ket0, ket1)
    ket10 = np.kron(ket1, ket0)
    ket11 = np.kron(ket1, ket1)
    
    print(f"|00⟩ = {ket00}")
    print(f"|01⟩ = {ket01}")
    print(f"|10⟩ = {ket10}")
    print(f"|11⟩ = {ket11}")
    
    print("\nDimension: 2^n where n is number of qubits")
    print(f"1 qubit: 2^1 = 2 dimensions")
    print(f"2 qubits: 2^2 = 4 dimensions")
    print(f"3 qubits: 2^3 = 8 dimensions")
    print(f"n qubits: 2^n dimensions")


def probability_amplitudes_demo():
    """Demonstrate probability amplitudes and measurement."""
    print_header("4. PROBABILITY AMPLITUDES AND MEASUREMENT")
    
    print("The Born Rule: P(i) = |αᵢ|²")
    print("The probability of measuring state i is the square of the")
    print("magnitude of its amplitude.\n")
    
    print_subheader("Example 1: Different Quantum States")
    
    states = [
        ("Equal superposition", np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)),
        ("Mostly |0⟩", np.array([0.9, 0.436], dtype=complex)),
        ("Mostly |1⟩", np.array([0.3, 0.954], dtype=complex)),
        ("With phase", np.array([0.6, 0.8j], dtype=complex)),
    ]
    
    for name, state in states:
        print(f"\n{name}: |ψ⟩ = {state}")
        probs = np.abs(state)**2
        print(f"P(0) = {probs[0]:.3f} ({probs[0]*100:.1f}%)")
        print(f"P(1) = {probs[1]:.3f} ({probs[1]*100:.1f}%)")
        print(f"Normalized: {np.isclose(np.sum(probs), 1)}")
    
    print_subheader("Example 2: Simulating Quantum Measurement")
    
    psi = np.array([0.6, 0.8], dtype=complex)
    probs = np.abs(psi)**2
    
    print(f"Initial state: |ψ⟩ = {psi}")
    print(f"Probabilities: P(0) = {probs[0]:.3f}, P(1) = {probs[1]:.3f}")
    
    num_measurements = 1000
    measurements = np.random.choice([0, 1], size=num_measurements, p=probs)
    
    count_0 = np.sum(measurements == 0)
    count_1 = np.sum(measurements == 1)
    
    print(f"\nSimulating {num_measurements} measurements:")
    print(f"Measured |0⟩: {count_0} times ({count_0/num_measurements*100:.1f}%)")
    print(f"Measured |1⟩: {count_1} times ({count_1/num_measurements*100:.1f}%)")
    print(f"Expected |0⟩: {probs[0]*100:.1f}%")
    print(f"Expected |1⟩: {probs[1]*100:.1f}%")


def normalization_demo():
    """Demonstrate normalization of quantum states."""
    print_header("5. NORMALIZATION")
    
    print("All valid quantum states must be normalized:")
    print("Σᵢ |αᵢ|² = 1")
    print("This ensures total probability equals 1.\n")
    
    print_subheader("Example 1: Normalizing an Unnormalized State")
    
    unnormalized = np.array([3, 4], dtype=complex)
    print(f"Unnormalized state: |ψ⟩ = {unnormalized}")
    
    norm = np.linalg.norm(unnormalized)
    print(f"Norm: ||ψ|| = {norm:.3f}")
    
    normalized = unnormalized / norm
    print(f"Normalized state: |ψ⟩ = {normalized}")
    print(f"New norm: ||ψ|| = {np.linalg.norm(normalized):.3f}")
    
    probs = np.abs(normalized)**2
    print(f"\nProbabilities:")
    print(f"P(0) = {probs[0]:.3f}")
    print(f"P(1) = {probs[1]:.3f}")
    print(f"Total = {np.sum(probs):.3f}")
    
    print_subheader("Example 2: Normalizing Complex State")
    
    unnormalized2 = np.array([1+2j, 3-1j], dtype=complex)
    print(f"Unnormalized state: |ψ⟩ = {unnormalized2}")
    
    norm2 = np.linalg.norm(unnormalized2)
    print(f"Norm: ||ψ|| = {norm2:.3f}")
    
    normalized2 = unnormalized2 / norm2
    print(f"Normalized state: |ψ⟩ = {normalized2}")
    print(f"New norm: ||ψ|| = {np.linalg.norm(normalized2):.3f}")
    
    probs2 = np.abs(normalized2)**2
    print(f"\nProbabilities:")
    print(f"P(0) = {probs2[0]:.3f}")
    print(f"P(1) = {probs2[1]:.3f}")
    print(f"Total = {np.sum(probs2):.3f}")


def classical_vs_quantum_comparison():
    """Compare classical and quantum vectors."""
    print_header("6. CLASSICAL VS QUANTUM VECTORS")
    
    print("Key Differences:\n")
    
    print("CLASSICAL VECTORS:")
    print("  • Components are real numbers")
    print("  • Represent physical quantities (force, velocity)")
    print("  • Magnitude has direct physical meaning")
    print("  • Direction is in physical space")
    print("  • Operations: addition, scalar multiplication, dot product")
    
    print("\nQUANTUM VECTORS:")
    print("  • Components are complex numbers")
    print("  • Represent quantum states in Hilbert space")
    print("  • Magnitude squared gives probability")
    print("  • 'Direction' is in abstract state space")
    print("  • Operations: superposition, inner product, tensor product")
    
    print("\n" + "-" * 80)
    
    print_subheader("Side-by-Side Example")
    
    print("Classical Vector:")
    classical = np.array([3, 4])
    print(f"  v = {classical}")
    print(f"  |v| = {np.linalg.norm(classical):.2f}")
    print(f"  Direction: {np.arctan2(classical[1], classical[0])*180/np.pi:.2f}°")
    
    print("\nQuantum Vector:")
    quantum = np.array([0.6, 0.8], dtype=complex)
    print(f"  |ψ⟩ = {quantum}")
    print(f"  ||ψ|| = {np.linalg.norm(quantum):.2f}")
    print(f"  P(0) = {np.abs(quantum[0])**2:.2f}")
    print(f"  P(1) = {np.abs(quantum[1])**2:.2f}")


def superposition_demo():
    """Demonstrate quantum superposition using vectors."""
    print_header("7. SUPERPOSITION")
    
    print("Superposition is represented by vector addition.")
    print("A quantum state can be in multiple classical states simultaneously.\n")
    
    ket0 = np.array([1, 0], dtype=complex)
    ket1 = np.array([0, 1], dtype=complex)
    
    print("Basis states:")
    print(f"|0⟩ = {ket0}")
    print(f"|1⟩ = {ket1}")
    
    print_subheader("Example 1: Equal Superposition")
    plus = (ket0 + ket1) / np.sqrt(2)
    print(f"|+⟩ = (|0⟩ + |1⟩)/√2 = {plus}")
    print(f"P(0) = {np.abs(plus[0])**2:.3f}")
    print(f"P(1) = {np.abs(plus[1])**2:.3f}")
    
    print_subheader("Example 2: Minus State")
    minus = (ket0 - ket1) / np.sqrt(2)
    print(f"|−⟩ = (|0⟩ - |1⟩)/√2 = {minus}")
    print(f"P(0) = {np.abs(minus[0])**2:.3f}")
    print(f"P(1) = {np.abs(minus[1])**2:.3f}")
    
    print_subheader("Example 3: Weighted Superposition")
    psi = 0.8 * ket0 + 0.6 * ket1
    print(f"|ψ⟩ = 0.8|0⟩ + 0.6|1⟩ = {psi}")
    print(f"P(0) = {np.abs(psi[0])**2:.3f}")
    print(f"P(1) = {np.abs(psi[1])**2:.3f}")


def visualize():
    """Create comprehensive visualizations."""
    print_header("8. VISUALIZATIONS")
    print("Creating comprehensive visualizations...")
    
    fig = plt.figure(figsize=(16, 12))
    
    ax1 = fig.add_subplot(3, 3, 1)
    vectors = np.array([[3, 4], [1, 2], [-2, 3]])
    colors = ['red', 'blue', 'green']
    labels = ['v₁', 'v₂', 'v₃']
    
    for vec, color, label in zip(vectors, colors, labels):
        ax1.arrow(0, 0, vec[0], vec[1], head_width=0.3, head_length=0.3,
                 fc=color, ec=color, linewidth=2, label=label)
    
    ax1.set_xlim(-3, 5)
    ax1.set_ylim(-1, 5)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Classical Vectors in 2D')
    ax1.legend()
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    
    ax2 = fig.add_subplot(3, 3, 2)
    states = ['|+⟩', '|ψ₁⟩', '|ψ₂⟩', '|ψ₃⟩']
    state_vectors = [
        np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
        np.array([0.8, 0.6]),
        np.array([0.6, 0.8]),
        np.array([0.3, 0.954])
    ]
    
    x = np.arange(len(states))
    p0 = [np.abs(s[0])**2 for s in state_vectors]
    p1 = [np.abs(s[1])**2 for s in state_vectors]
    
    width = 0.35
    ax2.bar(x - width/2, p0, width, label='P(|0⟩)', color='skyblue')
    ax2.bar(x + width/2, p1, width, label='P(|1⟩)', color='lightcoral')
    
    ax2.set_xlabel('Quantum State')
    ax2.set_ylabel('Probability')
    ax2.set_title('Measurement Probabilities')
    ax2.set_xticks(x)
    ax2.set_xticklabels(states)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax3 = fig.add_subplot(3, 3, 3)
    unnormalized = np.array([3, 4])
    normalized = unnormalized / np.linalg.norm(unnormalized)
    
    ax3.arrow(0, 0, unnormalized[0], unnormalized[1], head_width=0.3,
             head_length=0.3, fc='red', ec='red', linewidth=2,
             label='Unnormalized', alpha=0.5)
    ax3.arrow(0, 0, normalized[0], normalized[1], head_width=0.1,
             head_length=0.1, fc='green', ec='green', linewidth=2,
             label='Normalized')
    
    circle = plt.Circle((0, 0), 1, fill=False, color='blue',
                       linestyle='--', label='Unit circle')
    ax3.add_patch(circle)
    
    ax3.set_xlim(-1.5, 4)
    ax3.set_ylim(-1.5, 5)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Component 1')
    ax3.set_ylabel('Component 2')
    ax3.set_title('Normalization')
    ax3.legend()
    ax3.axhline(y=0, color='k', linewidth=0.5)
    ax3.axvline(x=0, color='k', linewidth=0.5)
    
    ax4 = fig.add_subplot(3, 3, 4)
    ket0 = np.array([1, 0])
    ket1 = np.array([0, 1])
    plus = (ket0 + ket1) / np.sqrt(2)
    
    ax4.arrow(0, 0, ket0[0], ket0[1], head_width=0.05, head_length=0.05,
             fc='blue', ec='blue', linewidth=2, label='|0⟩')
    ax4.arrow(0, 0, ket1[0], ket1[1], head_width=0.05, head_length=0.05,
             fc='red', ec='red', linewidth=2, label='|1⟩')
    ax4.arrow(0, 0, plus[0], plus[1], head_width=0.05, head_length=0.05,
             fc='green', ec='green', linewidth=3, label='|+⟩ = (|0⟩+|1⟩)/√2')
    
    ax4.set_xlim(-0.2, 1.2)
    ax4.set_ylim(-0.2, 1.2)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('|0⟩ component')
    ax4.set_ylabel('|1⟩ component')
    ax4.set_title('Superposition: |+⟩ State')
    ax4.legend()
    
    ax5 = fig.add_subplot(3, 3, 5)
    psi = np.array([0.6, 0.8])
    probs = np.abs(psi)**2
    
    num_measurements = 1000
    measurements = np.random.choice([0, 1], size=num_measurements, p=probs)
    
    unique, counts = np.unique(measurements, return_counts=True)
    ax5.bar(['|0⟩', '|1⟩'], counts/num_measurements, color=['skyblue', 'lightcoral'],
           alpha=0.7, label='Measured')
    ax5.axhline(y=probs[0], color='blue', linestyle='--', linewidth=2,
               label=f'Expected P(0)={probs[0]:.2f}')
    ax5.axhline(y=probs[1], color='red', linestyle='--', linewidth=2,
               label=f'Expected P(1)={probs[1]:.2f}')
    
    ax5.set_ylabel('Frequency')
    ax5.set_title(f'Measurement Simulation ({num_measurements} shots)')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    ax6 = fig.add_subplot(3, 3, 6, projection='polar')
    states_complex = [
        np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
        np.array([0.6, 0.8j]),
        np.array([0.5+0.5j, 0.5-0.5j])/np.sqrt(2)
    ]
    colors_complex = ['blue', 'red', 'green']
    labels_complex = ['|+⟩', '|ψ₁⟩', '|ψ₂⟩']
    
    for state, color, label in zip(states_complex, colors_complex, labels_complex):
        for i, amp in enumerate(state):
            r = np.abs(amp)
            theta = np.angle(amp)
            ax6.plot([theta], [r], 'o', markersize=10, color=color,
                    label=f'{label} α_{i}')
    
    ax6.set_title('Complex Amplitudes (Polar)')
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    ax7 = fig.add_subplot(3, 3, 7)
    n_qubits = np.arange(1, 11)
    dimensions = 2**n_qubits
    
    ax7.semilogy(n_qubits, dimensions, 'o-', linewidth=2, markersize=8,
                color='purple')
    ax7.set_xlabel('Number of Qubits')
    ax7.set_ylabel('Hilbert Space Dimension')
    ax7.set_title('Exponential Growth of State Space')
    ax7.grid(True, alpha=0.3)
    
    for i, (n, d) in enumerate(zip(n_qubits[:5], dimensions[:5])):
        ax7.text(n, d*1.5, f'2^{n}={d}', ha='center', fontsize=8)
    
    ax8 = fig.add_subplot(3, 3, 8)
    v1 = np.array([2, 1])
    v2 = np.array([1, 2])
    v_sum = v1 + v2
    
    ax8.arrow(0, 0, v1[0], v1[1], head_width=0.2, head_length=0.2,
             fc='blue', ec='blue', linewidth=2, label='v₁')
    ax8.arrow(0, 0, v2[0], v2[1], head_width=0.2, head_length=0.2,
             fc='red', ec='red', linewidth=2, label='v₂')
    ax8.arrow(0, 0, v_sum[0], v_sum[1], head_width=0.2, head_length=0.2,
             fc='green', ec='green', linewidth=3, label='v₁ + v₂')
    
    ax8.plot([v1[0], v_sum[0]], [v1[1], v_sum[1]], 'k--', alpha=0.3)
    ax8.plot([v2[0], v_sum[0]], [v2[1], v_sum[1]], 'k--', alpha=0.3)
    
    ax8.set_xlim(-0.5, 4)
    ax8.set_ylim(-0.5, 4)
    ax8.set_aspect('equal')
    ax8.grid(True, alpha=0.3)
    ax8.set_xlabel('X')
    ax8.set_ylabel('Y')
    ax8.set_title('Vector Addition')
    ax8.legend()
    ax8.axhline(y=0, color='k', linewidth=0.5)
    ax8.axvline(x=0, color='k', linewidth=0.5)
    
    ax9 = fig.add_subplot(3, 3, 9)
    states_prob = ['|0⟩', '|+⟩', '|ψ⟩', '|1⟩']
    state_vectors_prob = [
        np.array([1, 0]),
        np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
        np.array([0.6, 0.8]),
        np.array([0, 1])
    ]
    
    x_pos = np.arange(len(states_prob))
    for i, (state, vec) in enumerate(zip(states_prob, state_vectors_prob)):
        probs = np.abs(vec)**2
        ax9.bar([i-0.15, i+0.15], probs, width=0.3,
               color=['skyblue', 'lightcoral'])
    
    ax9.set_xlabel('Quantum State')
    ax9.set_ylabel('Probability')
    ax9.set_title('Probability Distributions')
    ax9.set_xticks(x_pos)
    ax9.set_xticklabels(states_prob)
    ax9.grid(True, alpha=0.3, axis='y')
    
    blue_patch = mpatches.Patch(color='skyblue', label='P(|0⟩)')
    red_patch = mpatches.Patch(color='lightcoral', label='P(|1⟩)')
    ax9.legend(handles=[blue_patch, red_patch])
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/01-introduction/vectors_intro_visualization.png',
                dpi=300, bbox_inches='tight')
    print("Saved visualization to: vectors_intro_visualization.png")
    plt.show()


def main():
    """Main function to run all demonstrations."""
    print("\n" + "=" * 80)
    print("  VECTORS IN QUANTUM COMPUTING - CONCEPT 1: INTRODUCTION")
    print("  Comprehensive Demonstrations and Examples")
    print("=" * 80)
    
    classical_vectors_demo()
    quantum_vectors_demo()
    hilbert_space_demo()
    probability_amplitudes_demo()
    normalization_demo()
    classical_vs_quantum_comparison()
    superposition_demo()
    
    visualize()
    
    print_header("SUMMARY")
    print("Key Concepts Covered:")
    print("  1. Classical vectors represent physical quantities with magnitude and direction")
    print("  2. Quantum vectors represent quantum states in complex Hilbert space")
    print("  3. Vector components are complex probability amplitudes")
    print("  4. |amplitude|² gives measurement probability (Born Rule)")
    print("  5. Normalization ensures total probability = 1")
    print("  6. Superposition is represented by vector addition")
    print("  7. Hilbert space dimension grows exponentially with qubits (2^n)")
    print("  8. Vectors are the mathematical foundation of quantum computing")
    
    print("\n" + "=" * 80)
    print("  End of Concept 1: Introduction")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
