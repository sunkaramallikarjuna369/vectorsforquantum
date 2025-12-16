"""
Scalar Multiplication
Comprehensive Python script demonstrating scalar multiplication and quantum phase shifts
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)

def print_subsection(title):
    """Print a formatted subsection header"""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)

def basic_scalar_multiplication():
    """Demonstrate basic scalar multiplication"""
    print_section_header("BASIC SCALAR MULTIPLICATION")
    
    print("\nScalar multiplication: multiply each component by a scalar")
    print("  c · v = [c·v₁, c·v₂, ..., c·vₙ]ᵀ")
    
    print_subsection("Example 1: Positive Scalar")
    v = np.array([[3], [4]], dtype=float)
    c = 2
    scaled = c * v
    
    print(f"\nv = {v.T[0]}")
    print(f"c = {c}")
    print(f"cv = {scaled.T[0]}")
    print(f"\nComponent-wise:")
    print(f"  First component: {c} × {v[0,0]} = {scaled[0,0]}")
    print(f"  Second component: {c} × {v[1,0]} = {scaled[1,0]}")
    
    mag_v = np.linalg.norm(v)
    mag_scaled = np.linalg.norm(scaled)
    print(f"\nMagnitudes:")
    print(f"  |v| = {mag_v:.3f}")
    print(f"  |cv| = {mag_scaled:.3f}")
    print(f"  |cv| = {c} × |v| = {c * mag_v:.3f} ✓")
    
    print_subsection("Example 2: Fractional Scalar")
    c_frac = 0.5
    scaled_frac = c_frac * v
    
    print(f"\nv = {v.T[0]}")
    print(f"c = {c_frac}")
    print(f"cv = {scaled_frac.T[0]}")
    print(f"Vector is shrunk by factor of {c_frac}")
    
    print_subsection("Example 3: Negative Scalar")
    c_neg = -1
    scaled_neg = c_neg * v
    
    print(f"\nv = {v.T[0]}")
    print(f"c = {c_neg}")
    print(f"cv = {scaled_neg.T[0]}")
    print(f"Vector direction is reversed")
    
    print_subsection("Example 4: Zero Scalar")
    c_zero = 0
    scaled_zero = c_zero * v
    
    print(f"\nv = {v.T[0]}")
    print(f"c = {c_zero}")
    print(f"cv = {scaled_zero.T[0]}")
    print(f"Results in zero vector")

def scalar_multiplication_properties():
    """Demonstrate properties of scalar multiplication"""
    print_section_header("PROPERTIES OF SCALAR MULTIPLICATION")
    
    v = np.array([[3], [4]], dtype=float)
    w = np.array([[1], [2]], dtype=float)
    c = 2
    d = 3
    
    print_subsection("1. Distributive over Vector Addition")
    print("c(v + w) = cv + cw")
    
    left = c * (v + w)
    right = c * v + c * w
    print(f"\nc(v + w) = {left.T[0]}")
    print(f"cv + cw = {right.T[0]}")
    print(f"Equal: {np.array_equal(left, right)} ✓")
    
    print_subsection("2. Distributive over Scalar Addition")
    print("(c + d)v = cv + dv")
    
    left = (c + d) * v
    right = c * v + d * v
    print(f"\n(c + d)v = {left.T[0]}")
    print(f"cv + dv = {right.T[0]}")
    print(f"Equal: {np.array_equal(left, right)} ✓")
    
    print_subsection("3. Associative Property")
    print("c(dv) = (cd)v")
    
    left = c * (d * v)
    right = (c * d) * v
    print(f"\nc(dv) = {left.T[0]}")
    print(f"(cd)v = {right.T[0]}")
    print(f"Equal: {np.array_equal(left, right)} ✓")
    
    print_subsection("4. Identity Element")
    print("1 · v = v")
    
    identity = 1 * v
    print(f"\nv = {v.T[0]}")
    print(f"1v = {identity.T[0]}")
    print(f"Equal: {np.array_equal(v, identity)} ✓")

def geometric_interpretation():
    """Demonstrate geometric interpretation"""
    print_section_header("GEOMETRIC INTERPRETATION")
    
    v = np.array([[3], [4]], dtype=float)
    
    print_subsection("Effect of Different Scalars")
    
    scalars = [2, 0.5, -1, -2, 0]
    
    for c in scalars:
        scaled = c * v
        mag_v = np.linalg.norm(v)
        mag_scaled = np.linalg.norm(scaled)
        
        print(f"\nc = {c}:")
        print(f"  v = {v.T[0]}")
        print(f"  cv = {scaled.T[0]}")
        print(f"  |v| = {mag_v:.3f}")
        print(f"  |cv| = {mag_scaled:.3f}")
        
        if c > 1:
            print(f"  Effect: Stretched by factor of {c}")
        elif c == 1:
            print(f"  Effect: Unchanged")
        elif 0 < c < 1:
            print(f"  Effect: Shrunk by factor of {c}")
        elif c == 0:
            print(f"  Effect: Zero vector")
        elif -1 < c < 0:
            print(f"  Effect: Reversed and shrunk by factor of {abs(c)}")
        elif c == -1:
            print(f"  Effect: Reversed")
        else:  # c < -1
            print(f"  Effect: Reversed and stretched by factor of {abs(c)}")

def complex_scalar_multiplication():
    """Demonstrate complex scalar multiplication"""
    print_section_header("COMPLEX SCALAR MULTIPLICATION")
    
    print("\nIn quantum computing, scalars can be complex numbers!")
    print("Complex scalar: c = |c|e^(iφ)")
    print("  |c| = magnitude")
    print("  φ = phase")
    
    print_subsection("Example 1: Pure Imaginary Scalar")
    v = np.array([[1], [0]], dtype=complex)
    c = 1j  # i
    scaled = c * v
    
    print(f"\nv = {v.T[0]}")
    print(f"c = i = e^(iπ/2)")
    print(f"cv = {scaled.T[0]}")
    print(f"\nMagnitude: |cv| = {np.linalg.norm(scaled):.3f}")
    print(f"Phase of first component: {np.angle(scaled[0,0]):.4f} rad = π/2")
    
    print_subsection("Example 2: Complex Scalar with Magnitude and Phase")
    c_complex = 2 * np.exp(1j * np.pi / 4)  # 2e^(iπ/4)
    scaled_complex = c_complex * v
    
    print(f"\nv = {v.T[0]}")
    print(f"c = 2e^(iπ/4) = {c_complex:.4f}")
    print(f"cv = {scaled_complex.T[0]}")
    print(f"\nMagnitude: |cv| = {np.linalg.norm(scaled_complex):.3f}")
    print(f"Phase: {np.angle(scaled_complex[0,0]):.4f} rad = π/4")
    
    print_subsection("Example 3: Euler's Formula")
    print("\ne^(iφ) = cos(φ) + i·sin(φ)")
    
    phases = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    phase_names = ['0', 'π/4', 'π/2', 'π', '3π/2', '2π']
    
    for phi, name in zip(phases, phase_names):
        c_euler = np.exp(1j * phi)
        print(f"\ne^(i{name}) = {c_euler:.4f}")
        print(f"  Real: {c_euler.real:.4f}")
        print(f"  Imaginary: {c_euler.imag:.4f}")
        print(f"  Magnitude: {np.abs(c_euler):.4f}")

def quantum_phase_shifts():
    """Demonstrate quantum phase shifts"""
    print_section_header("QUANTUM PHASE SHIFTS")
    
    zero = np.array([[1], [0]], dtype=complex)
    one = np.array([[0], [1]], dtype=complex)
    
    print_subsection("Global Phase Shift")
    print("\nMultiplying entire state by e^(iφ) is a global phase shift")
    print("Doesn't change measurement probabilities!")
    
    psi = (1/np.sqrt(2)) * zero + (1/np.sqrt(2)) * one
    
    print(f"\n|ψ⟩ = {psi.T[0]}")
    prob_0_orig = np.abs(psi[0, 0])**2
    prob_1_orig = np.abs(psi[1, 0])**2
    print(f"P(0) = {prob_0_orig:.4f}, P(1) = {prob_1_orig:.4f}")
    
    phi = np.pi / 4
    phase_factor = np.exp(1j * phi)
    psi_shifted = phase_factor * psi
    
    print(f"\ne^(iπ/4)|ψ⟩ = {psi_shifted.T[0]}")
    prob_0_shifted = np.abs(psi_shifted[0, 0])**2
    prob_1_shifted = np.abs(psi_shifted[1, 0])**2
    print(f"P(0) = {prob_0_shifted:.4f}, P(1) = {prob_1_shifted:.4f}")
    print(f"\nProbabilities unchanged! ✓")
    
    print_subsection("Relative Phase")
    print("\nBut relative phases between components matter!")
    
    plus = (1/np.sqrt(2)) * zero + (1/np.sqrt(2)) * one
    print(f"\n|+⟩ = {plus.T[0]}")
    
    minus = (1/np.sqrt(2)) * zero - (1/np.sqrt(2)) * one
    print(f"|-⟩ = {minus.T[0]}")
    
    print(f"\nBoth have same probabilities:")
    print(f"  P(0) = 50%, P(1) = 50%")
    print(f"\nBut they are different quantum states!")
    print(f"  Relative phase in |-⟩: π")
    
    print(f"\nInner product ⟨+|-⟩ = {np.vdot(plus, minus)[0]:.4f}")
    print(f"Not equal to 1, so they are different states!")

def phase_gate_examples():
    """Demonstrate phase gate operations"""
    print_section_header("PHASE GATE EXAMPLES")
    
    zero = np.array([[1], [0]], dtype=complex)
    one = np.array([[0], [1]], dtype=complex)
    
    print_subsection("Z Gate (Phase Flip)")
    print("\nZ gate: applies phase of π to |1⟩")
    print("Z|0⟩ = |0⟩")
    print("Z|1⟩ = -|1⟩ = e^(iπ)|1⟩")
    
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    z_zero = Z @ zero
    z_one = Z @ one
    
    print(f"\nZ|0⟩ = {z_zero.T[0]}")
    print(f"Z|1⟩ = {z_one.T[0]}")
    
    plus = (1/np.sqrt(2)) * zero + (1/np.sqrt(2)) * one
    z_plus = Z @ plus
    
    print(f"\n|+⟩ = {plus.T[0]}")
    print(f"Z|+⟩ = {z_plus.T[0]}")
    print(f"This is |-⟩!")
    
    print_subsection("S Gate (Phase Gate)")
    print("\nS gate: applies phase of π/2 to |1⟩")
    print("S|0⟩ = |0⟩")
    print("S|1⟩ = i|1⟩ = e^(iπ/2)|1⟩")
    
    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    
    s_zero = S @ zero
    s_one = S @ one
    
    print(f"\nS|0⟩ = {s_zero.T[0]}")
    print(f"S|1⟩ = {s_one.T[0]}")
    
    s_plus = S @ plus
    
    print(f"\n|+⟩ = {plus.T[0]}")
    print(f"S|+⟩ = {s_plus.T[0]}")
    print(f"This is |+i⟩ state!")
    
    print_subsection("T Gate (π/8 Gate)")
    print("\nT gate: applies phase of π/4 to |1⟩")
    print("T|0⟩ = |0⟩")
    print("T|1⟩ = e^(iπ/4)|1⟩")
    
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    
    t_zero = T @ zero
    t_one = T @ one
    
    print(f"\nT|0⟩ = {t_zero.T[0]}")
    print(f"T|1⟩ = {t_one.T[0]}")
    
    t_plus = T @ plus
    
    print(f"\n|+⟩ = {plus.T[0]}")
    print(f"T|+⟩ = {t_plus.T[0]}")

def normalization_after_scaling():
    """Demonstrate normalization after scalar multiplication"""
    print_section_header("NORMALIZATION AFTER SCALING")
    
    print("\nAfter scalar multiplication, quantum states may need renormalization")
    
    print_subsection("Example 1: Real Scalar")
    psi = np.array([[0.6], [0.8]], dtype=complex)
    c = 2
    
    print(f"\n|ψ⟩ = {psi.T[0]}")
    print(f"Norm: {np.linalg.norm(psi):.4f} ✓")
    
    scaled = c * psi
    norm_scaled = np.linalg.norm(scaled)
    
    print(f"\nc|ψ⟩ = {scaled.T[0]}")
    print(f"Norm: {norm_scaled:.4f} ✗")
    
    normalized = scaled / norm_scaled
    norm_check = np.linalg.norm(normalized)
    
    print(f"\nNormalized: {normalized.T[0]}")
    print(f"Norm: {norm_check:.4f} ✓")
    
    print_subsection("Example 2: Complex Scalar")
    c_complex = 1 + 1j
    
    print(f"\n|ψ⟩ = {psi.T[0]}")
    print(f"c = {c_complex}")
    
    scaled_complex = c_complex * psi
    norm_scaled_complex = np.linalg.norm(scaled_complex)
    
    print(f"\nc|ψ⟩ = {scaled_complex.T[0]}")
    print(f"Norm: {norm_scaled_complex:.4f} ✗")
    
    normalized_complex = scaled_complex / norm_scaled_complex
    norm_check_complex = np.linalg.norm(normalized_complex)
    
    print(f"\nNormalized: {normalized_complex.T[0]}")
    print(f"Norm: {norm_check_complex:.4f} ✓")
    
    print_subsection("Example 3: Phase-Only Scalar")
    c_phase = np.exp(1j * np.pi / 3)
    
    print(f"\n|ψ⟩ = {psi.T[0]}")
    print(f"c = e^(iπ/3) = {c_phase:.4f}")
    print(f"|c| = {np.abs(c_phase):.4f} (unit magnitude)")
    
    scaled_phase = c_phase * psi
    norm_scaled_phase = np.linalg.norm(scaled_phase)
    
    print(f"\nc|ψ⟩ = {scaled_phase.T[0]}")
    print(f"Norm: {norm_scaled_phase:.4f} ✓")
    print(f"\nNo renormalization needed for unit-magnitude scalars!")

def amplitude_amplification_demo():
    """Demonstrate amplitude amplification concept"""
    print_section_header("AMPLITUDE AMPLIFICATION")
    
    print("\nAmplitude amplification: increase probability of target states")
    print("Used in Grover's algorithm for quantum search")
    
    print_subsection("Simple Example")
    
    n_states = 4
    psi = np.ones((n_states, 1), dtype=complex) / np.sqrt(n_states)
    
    print(f"\nInitial state (equal superposition):")
    print(f"|ψ⟩ = {psi.T[0]}")
    print(f"\nProbabilities:")
    for i in range(n_states):
        prob = np.abs(psi[i, 0])**2
        print(f"  P(|{i}⟩) = {prob:.4f} = 25%")
    
    target = 2
    amplification_factor = 2
    
    psi_amplified = psi.copy()
    psi_amplified[target, 0] *= amplification_factor
    
    psi_amplified = psi_amplified / np.linalg.norm(psi_amplified)
    
    print(f"\nAfter amplifying state |{target}⟩:")
    print(f"|ψ'⟩ = {psi_amplified.T[0]}")
    print(f"\nProbabilities:")
    for i in range(n_states):
        prob = np.abs(psi_amplified[i, 0])**2
        print(f"  P(|{i}⟩) = {prob:.4f} = {prob*100:.1f}%")
    
    print(f"\nTarget state probability increased from 25% to {np.abs(psi_amplified[target, 0])**2*100:.1f}%!")

def visualize_scalar_multiplication():
    """Create comprehensive visualizations"""
    print_section_header("CREATING VISUALIZATIONS")
    
    fig = plt.figure(figsize=(16, 12))
    
    ax1 = fig.add_subplot(2, 3, 1)
    v = np.array([3, 4])
    scalars = [-2, -1, 0.5, 1, 2]
    colors = ['red', 'orange', 'yellow', 'green', 'blue']
    
    for c, color in zip(scalars, colors):
        scaled = c * v
        if c != 0:
            ax1.arrow(0, 0, scaled[0], scaled[1], head_width=0.3, head_length=0.3,
                     fc=color, ec=color, linewidth=2, alpha=0.7, label=f'{c}v')
        else:
            ax1.plot(0, 0, 'o', color=color, markersize=10, label=f'{c}v')
    
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Scalar Multiplication Effects', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.set_xlim(-7, 7)
    ax1.set_ylim(-9, 9)
    
    ax2 = fig.add_subplot(2, 3, 2)
    scalars_range = np.linspace(-3, 3, 50)
    magnitudes = [np.abs(c) * np.linalg.norm(v) for c in scalars_range]
    
    ax2.plot(scalars_range, magnitudes, 'b-', linewidth=2)
    ax2.axhline(y=np.linalg.norm(v), color='r', linestyle='--', label='|v|')
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Scalar c', fontsize=12)
    ax2.set_ylabel('Magnitude |cv|', fontsize=12)
    ax2.set_title('Magnitude vs Scalar', fontsize=14, fontweight='bold')
    ax2.legend()
    
    ax3 = fig.add_subplot(2, 3, 3)
    
    v_complex = 1 + 0j
    phases = np.linspace(0, 2*np.pi, 8, endpoint=False)
    
    for i, phi in enumerate(phases):
        c = np.exp(1j * phi)
        scaled = c * v_complex
        ax3.arrow(0, 0, scaled.real, scaled.imag, head_width=0.05, head_length=0.05,
                 fc=plt.cm.hsv(i/8), ec=plt.cm.hsv(i/8), linewidth=2, alpha=0.7)
        ax3.plot(scaled.real, scaled.imag, 'o', color=plt.cm.hsv(i/8), markersize=8)
    
    theta = np.linspace(0, 2*np.pi, 100)
    ax3.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    
    ax3.axhline(y=0, color='k', linewidth=0.5)
    ax3.axvline(x=0, color='k', linewidth=0.5)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Real', fontsize=12)
    ax3.set_ylabel('Imaginary', fontsize=12)
    ax3.set_title('Phase Rotations (e^(iφ))', fontsize=14, fontweight='bold')
    ax3.set_aspect('equal')
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    
    ax4 = fig.add_subplot(2, 3, 4)
    
    psi = np.array([0.6, 0.8])
    
    scalings = {
        'Original': 1,
        '2×': 2,
        '0.5×': 0.5,
        'Phase π/4': np.exp(1j * np.pi / 4)
    }
    
    x = np.arange(len(scalings))
    width = 0.35
    
    probs_0 = []
    probs_1 = []
    
    for name, c in scalings.items():
        scaled = c * psi
        normalized = scaled / np.linalg.norm(scaled)
        probs_0.append(np.abs(normalized[0])**2)
        probs_1.append(np.abs(normalized[1])**2)
    
    ax4.bar(x - width/2, probs_0, width, label='P(0)', alpha=0.7)
    ax4.bar(x + width/2, probs_1, width, label='P(1)', alpha=0.7)
    
    ax4.set_ylabel('Probability', fontsize=12)
    ax4.set_title('Probabilities After Scaling (Normalized)', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(scalings.keys(), rotation=15)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    ax5 = fig.add_subplot(2, 3, 5)
    
    plus = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    
    gates = {
        '|+⟩': np.array([[1, 0], [0, 1]]),  # Identity
        'Z|+⟩': np.array([[1, 0], [0, -1]]),  # Z gate
        'S|+⟩': np.array([[1, 0], [0, 1j]]),  # S gate
        'T|+⟩': np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])  # T gate
    }
    
    x = np.arange(len(gates))
    width = 0.35
    
    probs_0 = []
    probs_1 = []
    
    for name, gate in gates.items():
        result = gate @ plus
        probs_0.append(np.abs(result[0])**2)
        probs_1.append(np.abs(result[1])**2)
    
    ax5.bar(x - width/2, probs_0, width, label='P(0)', alpha=0.7)
    ax5.bar(x + width/2, probs_1, width, label='P(1)', alpha=0.7)
    
    ax5.set_ylabel('Probability', fontsize=12)
    ax5.set_title('Phase Gate Effects on |+⟩', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(gates.keys())
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    ax6 = fig.add_subplot(2, 3, 6)
    
    n_states = 4
    psi_equal = np.ones(n_states) / np.sqrt(n_states)
    
    psi_amp = psi_equal.copy()
    psi_amp[2] *= 2
    psi_amp = psi_amp / np.linalg.norm(psi_amp)
    
    x = np.arange(n_states)
    width = 0.35
    
    probs_equal = np.abs(psi_equal)**2
    probs_amp = np.abs(psi_amp)**2
    
    ax6.bar(x - width/2, probs_equal, width, label='Before', alpha=0.7)
    ax6.bar(x + width/2, probs_amp, width, label='After Amplification', alpha=0.7)
    
    ax6.set_ylabel('Probability', fontsize=12)
    ax6.set_title('Amplitude Amplification (Target: |2⟩)', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels([f'|{i}⟩' for i in range(n_states)])
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('scalar_multiplication_visualizations.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved comprehensive visualizations as 'scalar_multiplication_visualizations.png'")
    plt.close()
    
    fig2 = plt.figure(figsize=(14, 6))
    
    ax1_3d = fig2.add_subplot(1, 2, 1, projection='3d')
    
    v_3d = np.array([2, 3, 1])
    scalars_3d = [-1, 0.5, 1, 2]
    colors_3d = ['red', 'yellow', 'green', 'blue']
    
    for c, color in zip(scalars_3d, colors_3d):
        scaled = c * v_3d
        if c != 0:
            ax1_3d.quiver(0, 0, 0, scaled[0], scaled[1], scaled[2], 
                          color=color, arrow_length_ratio=0.15, linewidth=2, label=f'{c}v')
    
    ax1_3d.set_xlabel('X', fontsize=10)
    ax1_3d.set_ylabel('Y', fontsize=10)
    ax1_3d.set_zlabel('Z', fontsize=10)
    ax1_3d.set_title('3D Scalar Multiplication', fontsize=14, fontweight='bold')
    ax1_3d.legend()
    
    ax2_3d = fig2.add_subplot(1, 2, 2, projection='3d')
    
    u = np.linspace(0, 2 * np.pi, 50)
    v_sphere = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v_sphere))
    y_sphere = np.outer(np.sin(u), np.sin(v_sphere))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v_sphere))
    
    ax2_3d.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
    
    theta = np.pi / 4  # Angle from z-axis
    phases = np.linspace(0, 2*np.pi, 8, endpoint=False)
    
    for i, phi in enumerate(phases):
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        ax2_3d.quiver(0, 0, 0, x, y, z, color=plt.cm.hsv(i/8), arrow_length_ratio=0.15, linewidth=2)
    
    ax2_3d.set_xlabel('X', fontsize=10)
    ax2_3d.set_ylabel('Y', fontsize=10)
    ax2_3d.set_zlabel('Z', fontsize=10)
    ax2_3d.set_title('Phase Rotations on Bloch Sphere', fontsize=14, fontweight='bold')
    ax2_3d.set_xlim(-1, 1)
    ax2_3d.set_ylim(-1, 1)
    ax2_3d.set_zlim(-1, 1)
    
    plt.tight_layout()
    plt.savefig('scalar_multiplication_3d.png', dpi=150, bbox_inches='tight')
    print("✓ Saved 3D visualizations as 'scalar_multiplication_3d.png'")
    plt.close()

def main():
    """Main function to run all demonstrations"""
    print("\n" + "✖️" * 40)
    print("SCALAR MULTIPLICATION")
    print("Comprehensive Python Demonstrations")
    print("✖️" * 40)
    
    basic_scalar_multiplication()
    scalar_multiplication_properties()
    geometric_interpretation()
    complex_scalar_multiplication()
    quantum_phase_shifts()
    phase_gate_examples()
    normalization_after_scaling()
    amplitude_amplification_demo()
    visualize_scalar_multiplication()
    
    print_section_header("KEY TAKEAWAYS")
    print("\n1. Basic Operation:")
    print("   - Multiply each component by scalar: c·v = [c·v₁, c·v₂, ...]ᵀ")
    print("   - Scales magnitude by |c|")
    print("   - Reverses direction if c < 0")
    
    print("\n2. Properties:")
    print("   - Distributive over vector addition: c(v + w) = cv + cw")
    print("   - Distributive over scalar addition: (c + d)v = cv + dv")
    print("   - Associative: c(dv) = (cd)v")
    print("   - Identity: 1·v = v")
    
    print("\n3. Complex Scalars:")
    print("   - c = |c|e^(iφ) has magnitude and phase")
    print("   - Scales magnitude by |c|")
    print("   - Rotates phase by φ")
    
    print("\n4. Quantum Phase Shifts:")
    print("   - Global phase: e^(iφ)|ψ⟩ (doesn't change probabilities)")
    print("   - Relative phase: matters for quantum behavior")
    print("   - Phase gates: Z, S, T gates")
    
    print("\n5. Normalization:")
    print("   - After scaling, may need to renormalize")
    print("   - Ensure ||ψ⟩|| = 1 for valid quantum state")
    print("   - Unit-magnitude scalars preserve normalization")
    
    print("\n6. Applications:")
    print("   - Phase gates in quantum circuits")
    print("   - Amplitude amplification (Grover's algorithm)")
    print("   - State preparation")
    print("   - Quantum interference")
    
    print("\n" + "=" * 80)
    print("All demonstrations completed successfully!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
