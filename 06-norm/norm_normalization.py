"""
Norm and Normalization
Comprehensive Python script demonstrating vector norms and quantum state normalization
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

def basic_norm():
    """Demonstrate basic norm calculation"""
    print_section_header("BASIC NORM CALCULATION")
    
    print("\nNorm (magnitude/length): ||v|| = √(v₁² + v₂² + ... + vₙ²)")
    
    print_subsection("Example 1: 2D Vector")
    v = np.array([[3], [4]], dtype=float)
    norm = np.linalg.norm(v)
    
    print(f"\nv = {v.T[0]}")
    print(f"||v|| = √(3² + 4²)")
    print(f"     = √(9 + 16)")
    print(f"     = √25")
    print(f"     = {norm}")
    
    print_subsection("Example 2: 3D Vector")
    v3d = np.array([[1], [2], [2]], dtype=float)
    norm3d = np.linalg.norm(v3d)
    
    print(f"\nv = {v3d.T[0]}")
    print(f"||v|| = √(1² + 2² + 2²)")
    print(f"     = √(1 + 4 + 4)")
    print(f"     = √9")
    print(f"     = {norm3d}")
    
    print_subsection("Example 3: Higher Dimensions")
    v_high = np.array([[1], [1], [1], [1]], dtype=float)
    norm_high = np.linalg.norm(v_high)
    
    print(f"\nv = {v_high.T[0]}")
    print(f"||v|| = √(1² + 1² + 1² + 1²)")
    print(f"     = √4")
    print(f"     = {norm_high}")

def normalization_demo():
    """Demonstrate normalization"""
    print_section_header("NORMALIZATION")
    
    print("\nNormalization: v̂ = v / ||v||")
    print("Creates a unit vector (length 1) in the same direction")
    
    print_subsection("Example 1: 2D Vector")
    v = np.array([[3], [4]], dtype=float)
    norm = np.linalg.norm(v)
    v_normalized = v / norm
    norm_check = np.linalg.norm(v_normalized)
    
    print(f"\nOriginal vector: v = {v.T[0]}")
    print(f"Norm: ||v|| = {norm}")
    print(f"\nNormalized: v̂ = v / ||v||")
    print(f"           = {v.T[0]} / {norm}")
    print(f"           = {v_normalized.T[0]}")
    print(f"\nVerify: ||v̂|| = {norm_check:.4f} ✓")
    
    print_subsection("Example 2: 3D Vector")
    v3d = np.array([[1], [2], [2]], dtype=float)
    norm3d = np.linalg.norm(v3d)
    v3d_normalized = v3d / norm3d
    norm3d_check = np.linalg.norm(v3d_normalized)
    
    print(f"\nOriginal vector: v = {v3d.T[0]}")
    print(f"Norm: ||v|| = {norm3d}")
    print(f"Normalized: v̂ = {v3d_normalized.T[0]}")
    print(f"Verify: ||v̂|| = {norm3d_check:.4f} ✓")
    
    print_subsection("Example 3: Already Normalized")
    v_unit = np.array([[1], [0]], dtype=float)
    norm_unit = np.linalg.norm(v_unit)
    v_unit_normalized = v_unit / norm_unit
    
    print(f"\nOriginal vector: v = {v_unit.T[0]}")
    print(f"Norm: ||v|| = {norm_unit}")
    print(f"Normalized: v̂ = {v_unit_normalized.T[0]}")
    print(f"Already a unit vector!")

def norm_properties():
    """Demonstrate properties of norm"""
    print_section_header("PROPERTIES OF NORM")
    
    v = np.array([[3], [4]], dtype=float)
    w = np.array([[1], [2]], dtype=float)
    c = 2
    
    print_subsection("1. Non-negativity")
    print("||v|| ≥ 0, and ||v|| = 0 iff v = 0")
    
    norm_v = np.linalg.norm(v)
    print(f"\n||v|| = {norm_v} ≥ 0 ✓")
    
    zero_vec = np.array([[0], [0]], dtype=float)
    norm_zero = np.linalg.norm(zero_vec)
    print(f"||0|| = {norm_zero} = 0 ✓")
    
    print_subsection("2. Scalar Multiplication")
    print("||cv|| = |c| · ||v||")
    
    cv = c * v
    norm_cv = np.linalg.norm(cv)
    expected = np.abs(c) * norm_v
    
    print(f"\nc = {c}, v = {v.T[0]}")
    print(f"||cv|| = {norm_cv}")
    print(f"|c| · ||v|| = {np.abs(c)} × {norm_v} = {expected}")
    print(f"Equal: {np.isclose(norm_cv, expected)} ✓")
    
    print_subsection("3. Triangle Inequality")
    print("||v + w|| ≤ ||v|| + ||w||")
    
    v_plus_w = v + w
    norm_sum = np.linalg.norm(v_plus_w)
    sum_norms = norm_v + np.linalg.norm(w)
    
    print(f"\nv = {v.T[0]}, w = {w.T[0]}")
    print(f"||v + w|| = {norm_sum:.4f}")
    print(f"||v|| + ||w|| = {norm_v} + {np.linalg.norm(w):.4f} = {sum_norms:.4f}")
    print(f"{norm_sum:.4f} ≤ {sum_norms:.4f} ✓")
    
    print_subsection("4. Relationship to Inner Product")
    print("||v|| = √⟨v|v⟩")
    
    inner_vv = np.vdot(v, v)
    norm_from_inner = np.sqrt(inner_vv.real)
    norm_direct = np.linalg.norm(v)
    
    print(f"\n⟨v|v⟩ = {inner_vv.real}")
    print(f"√⟨v|v⟩ = {norm_from_inner:.4f}")
    print(f"||v|| = {norm_direct:.4f}")
    print(f"Equal: {np.isclose(norm_from_inner, norm_direct)} ✓")

def quantum_state_normalization():
    """Demonstrate quantum state normalization"""
    print_section_header("QUANTUM STATE NORMALIZATION")
    
    print("\nIn quantum mechanics, ALL valid states must be normalized!")
    print("||ψ⟩|| = 1 ensures total probability = 1")
    
    print_subsection("Example 1: Real Components")
    psi = np.array([[3], [4]], dtype=complex)
    norm = np.linalg.norm(psi)
    psi_normalized = psi / norm
    
    print(f"\nOriginal: |ψ⟩ = {psi.T[0]}")
    print(f"Norm: ||ψ⟩|| = {norm}")
    print(f"\nNormalized: |ψ̂⟩ = {psi_normalized.T[0]}")
    
    prob_0 = np.abs(psi_normalized[0, 0])**2
    prob_1 = np.abs(psi_normalized[1, 0])**2
    total_prob = prob_0 + prob_1
    
    print(f"\nProbabilities:")
    print(f"  P(0) = |α|² = {prob_0:.4f}")
    print(f"  P(1) = |β|² = {prob_1:.4f}")
    print(f"  Total = {total_prob:.4f} = 1 ✓")
    
    print_subsection("Example 2: Complex Components")
    psi_complex = np.array([[1+1j], [2-1j]], dtype=complex)
    norm_complex = np.linalg.norm(psi_complex)
    psi_complex_normalized = psi_complex / norm_complex
    
    print(f"\nOriginal: |ψ⟩ = {psi_complex.T[0]}")
    print(f"Norm: ||ψ⟩|| = {norm_complex:.4f}")
    print(f"\nNormalized: |ψ̂⟩ = {psi_complex_normalized.T[0]}")
    
    prob_0_c = np.abs(psi_complex_normalized[0, 0])**2
    prob_1_c = np.abs(psi_complex_normalized[1, 0])**2
    total_prob_c = prob_0_c + prob_1_c
    
    print(f"\nProbabilities:")
    print(f"  P(0) = {prob_0_c:.4f}")
    print(f"  P(1) = {prob_1_c:.4f}")
    print(f"  Total = {total_prob_c:.4f} = 1 ✓")
    
    print_subsection("Example 3: Pure Phase")
    psi_phase = np.array([[np.exp(1j*np.pi/4)], [np.exp(1j*np.pi/2)]], dtype=complex)
    norm_phase = np.linalg.norm(psi_phase)
    psi_phase_normalized = psi_phase / norm_phase
    
    print(f"\nOriginal: |ψ⟩ = [e^(iπ/4), e^(iπ/2)]ᵀ")
    print(f"Norm: ||ψ⟩|| = {norm_phase:.4f}")
    print(f"\nNormalized: |ψ̂⟩ = {psi_phase_normalized.T[0]}")
    
    prob_0_p = np.abs(psi_phase_normalized[0, 0])**2
    prob_1_p = np.abs(psi_phase_normalized[1, 0])**2
    total_prob_p = prob_0_p + prob_1_p
    
    print(f"\nProbabilities:")
    print(f"  P(0) = {prob_0_p:.4f}")
    print(f"  P(1) = {prob_1_p:.4f}")
    print(f"  Total = {total_prob_p:.4f} = 1 ✓")

def common_quantum_states():
    """Demonstrate common normalized quantum states"""
    print_section_header("COMMON QUANTUM STATES")
    
    print_subsection("Computational Basis States")
    zero = np.array([[1], [0]], dtype=complex)
    one = np.array([[0], [1]], dtype=complex)
    
    print(f"\n|0⟩ = {zero.T[0]}")
    print(f"||0⟩|| = {np.linalg.norm(zero):.4f} ✓")
    
    print(f"\n|1⟩ = {one.T[0]}")
    print(f"||1⟩|| = {np.linalg.norm(one):.4f} ✓")
    
    print_subsection("Superposition States")
    plus = (1/np.sqrt(2)) * (zero + one)
    minus = (1/np.sqrt(2)) * (zero - one)
    
    print(f"\n|+⟩ = (1/√2)(|0⟩ + |1⟩) = {plus.T[0]}")
    print(f"||+⟩|| = {np.linalg.norm(plus):.4f} ✓")
    
    print(f"\n|-⟩ = (1/√2)(|0⟩ - |1⟩) = {minus.T[0]}")
    print(f"||-⟩|| = {np.linalg.norm(minus):.4f} ✓")
    
    print_subsection("General Qubit State")
    alpha = 0.6
    beta = 0.8
    psi_general = alpha * zero + beta * one
    
    print(f"\n|ψ⟩ = {alpha}|0⟩ + {beta}|1⟩")
    print(f"    = {psi_general.T[0]}")
    print(f"||ψ⟩|| = {np.linalg.norm(psi_general):.4f} ✓")
    print(f"\nNote: {alpha}² + {beta}² = {alpha**2 + beta**2} = 1")
    
    print_subsection("Complex Superposition")
    alpha_c = (1+1j) / 2
    beta_c = np.sqrt(2) / 2
    psi_complex = alpha_c * zero + beta_c * one
    
    print(f"\n|ψ⟩ = {alpha_c:.4f}|0⟩ + {beta_c:.4f}|1⟩")
    print(f"    = {psi_complex.T[0]}")
    print(f"||ψ⟩|| = {np.linalg.norm(psi_complex):.4f} ✓")

def unit_vectors():
    """Demonstrate unit vectors"""
    print_section_header("UNIT VECTORS")
    
    print("\nA unit vector has norm 1: ||v̂|| = 1")
    
    print_subsection("Standard Unit Vectors in 2D")
    e1 = np.array([[1], [0]], dtype=float)
    e2 = np.array([[0], [1]], dtype=float)
    
    print(f"\nê₁ = {e1.T[0]} (x-direction)")
    print(f"||ê₁|| = {np.linalg.norm(e1):.4f} ✓")
    
    print(f"\nê₂ = {e2.T[0]} (y-direction)")
    print(f"||ê₂|| = {np.linalg.norm(e2):.4f} ✓")
    
    print_subsection("Standard Unit Vectors in 3D")
    e1_3d = np.array([[1], [0], [0]], dtype=float)
    e2_3d = np.array([[0], [1], [0]], dtype=float)
    e3_3d = np.array([[0], [0], [1]], dtype=float)
    
    print(f"\nê₁ = {e1_3d.T[0]} (x-direction)")
    print(f"||ê₁|| = {np.linalg.norm(e1_3d):.4f} ✓")
    
    print(f"\nê₂ = {e2_3d.T[0]} (y-direction)")
    print(f"||ê₂|| = {np.linalg.norm(e2_3d):.4f} ✓")
    
    print(f"\nê₃ = {e3_3d.T[0]} (z-direction)")
    print(f"||ê₃|| = {np.linalg.norm(e3_3d):.4f} ✓")
    
    print_subsection("Creating Unit Vector in Any Direction")
    v = np.array([[3], [4]], dtype=float)
    v_hat = v / np.linalg.norm(v)
    
    print(f"\nGiven direction: v = {v.T[0]}")
    print(f"Unit vector: v̂ = v / ||v|| = {v_hat.T[0]}")
    print(f"||v̂|| = {np.linalg.norm(v_hat):.4f} ✓")
    print(f"\nPoints in same direction, but length 1")

def distance_between_vectors():
    """Demonstrate distance between vectors"""
    print_section_header("DISTANCE BETWEEN VECTORS")
    
    print("\nDistance: d(v, w) = ||v - w||")
    
    print_subsection("Example 1: 2D Vectors")
    v = np.array([[3], [4]], dtype=float)
    w = np.array([[1], [2]], dtype=float)
    
    diff = v - w
    distance = np.linalg.norm(diff)
    
    print(f"\nv = {v.T[0]}")
    print(f"w = {w.T[0]}")
    print(f"\nv - w = {diff.T[0]}")
    print(f"d(v, w) = ||v - w|| = {distance:.4f}")
    
    print_subsection("Example 2: Quantum States")
    zero = np.array([[1], [0]], dtype=complex)
    plus = (1/np.sqrt(2)) * np.array([[1], [1]], dtype=complex)
    
    diff_q = zero - plus
    distance_q = np.linalg.norm(diff_q)
    
    print(f"\n|0⟩ = {zero.T[0]}")
    print(f"|+⟩ = {plus.T[0]}")
    print(f"\nd(|0⟩, |+⟩) = {distance_q:.4f}")
    print(f"\nMeasures how different two quantum states are")

def normalization_after_operations():
    """Demonstrate when normalization is needed"""
    print_section_header("NORMALIZATION AFTER OPERATIONS")
    
    print("\nSome operations preserve normalization, others don't")
    
    print_subsection("Operations That Preserve Normalization")
    
    psi = (1/np.sqrt(2)) * np.array([[1], [1]], dtype=complex)
    
    H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    psi_after_H = H @ psi
    
    print(f"\nOriginal: |ψ⟩ = {psi.T[0]}")
    print(f"||ψ⟩|| = {np.linalg.norm(psi):.4f}")
    
    print(f"\nAfter Hadamard gate: H|ψ⟩ = {psi_after_H.T[0]}")
    print(f"||H|ψ⟩|| = {np.linalg.norm(psi_after_H):.4f}")
    print(f"Normalization preserved! ✓")
    
    print_subsection("Operations That Don't Preserve Normalization")
    
    psi_scaled = 2 * psi
    
    print(f"\nOriginal: |ψ⟩ = {psi.T[0]}")
    print(f"||ψ⟩|| = {np.linalg.norm(psi):.4f}")
    
    print(f"\nAfter scaling: 2|ψ⟩ = {psi_scaled.T[0]}")
    print(f"||2|ψ⟩|| = {np.linalg.norm(psi_scaled):.4f}")
    print(f"Normalization lost! Need to renormalize")
    
    psi_renormalized = psi_scaled / np.linalg.norm(psi_scaled)
    print(f"\nRenormalized: {psi_renormalized.T[0]}")
    print(f"||ψ̂|| = {np.linalg.norm(psi_renormalized):.4f} ✓")

def visualize_norm_normalization():
    """Create comprehensive visualizations"""
    print_section_header("CREATING VISUALIZATIONS")
    
    fig = plt.figure(figsize=(16, 12))
    
    ax1 = fig.add_subplot(2, 3, 1)
    v = np.array([3, 4])
    v_norm = v / np.linalg.norm(v)
    
    ax1.arrow(0, 0, v[0], v[1], head_width=0.3, head_length=0.3,
             fc='blue', ec='blue', linewidth=2, label=f'v (||v||={np.linalg.norm(v):.1f})', alpha=0.7)
    ax1.arrow(0, 0, v_norm[0], v_norm[1], head_width=0.1, head_length=0.1,
             fc='green', ec='green', linewidth=2, label='v̂ (||v̂||=1)', alpha=0.7)
    
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'r--', alpha=0.3, label='Unit circle')
    
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Vector Normalization', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.set_xlim(-1.5, 4)
    ax1.set_ylim(-1.5, 5)
    
    ax2 = fig.add_subplot(2, 3, 2)
    
    x_vals = np.linspace(0, 5, 50)
    y_vals = np.linspace(0, 5, 50)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.sqrt(X**2 + Y**2)
    
    contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax2.set_xlabel('x component', fontsize=12)
    ax2.set_ylabel('y component', fontsize=12)
    ax2.set_title('Norm as Function of Components', fontsize=14, fontweight='bold')
    plt.colorbar(contour, ax=ax2, label='||v||')
    
    ax3 = fig.add_subplot(2, 3, 3)
    
    vectors = [
        ([1, 0], 'red'),
        ([3, 4], 'blue'),
        ([2, 3], 'green'),
        ([1, 2], 'orange')
    ]
    
    for v_arr, color in vectors:
        v_np = np.array(v_arr)
        v_norm_np = v_np / np.linalg.norm(v_np)
        
        ax3.arrow(0, 0, v_np[0], v_np[1], head_width=0.15, head_length=0.15,
                 fc=color, ec=color, linewidth=1.5, alpha=0.3)
        ax3.arrow(0, 0, v_norm_np[0], v_norm_np[1], head_width=0.08, head_length=0.08,
                 fc=color, ec=color, linewidth=2, alpha=0.8)
    
    ax3.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, linewidth=2)
    
    ax3.axhline(y=0, color='k', linewidth=0.5)
    ax3.axvline(x=0, color='k', linewidth=0.5)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('y', fontsize=12)
    ax3.set_title('Multiple Vectors → Unit Circle', fontsize=14, fontweight='bold')
    ax3.set_aspect('equal')
    ax3.set_xlim(-1.5, 4)
    ax3.set_ylim(-1.5, 5)
    
    ax4 = fig.add_subplot(2, 3, 4)
    
    psi_unnorm = np.array([3, 4])
    norm_psi = np.linalg.norm(psi_unnorm)
    psi_norm = psi_unnorm / norm_psi
    
    x = np.arange(2)
    width = 0.35
    
    probs_unnorm = (psi_unnorm**2) / np.sum(psi_unnorm**2)
    probs_norm = psi_norm**2
    
    ax4.bar(x - width/2, probs_unnorm, width, label='Before normalization', alpha=0.7)
    ax4.bar(x + width/2, probs_norm, width, label='After normalization', alpha=0.7)
    
    ax4.set_ylabel('Probability', fontsize=12)
    ax4.set_title('Quantum State Probabilities', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['|0⟩', '|1⟩'])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    ax5 = fig.add_subplot(2, 3, 5)
    
    v1 = np.array([3, 4])
    v2 = np.array([1, 2])
    
    ax5.arrow(0, 0, v1[0], v1[1], head_width=0.2, head_length=0.2,
             fc='blue', ec='blue', linewidth=2, label='v', alpha=0.7)
    ax5.arrow(0, 0, v2[0], v2[1], head_width=0.2, head_length=0.2,
             fc='red', ec='red', linewidth=2, label='w', alpha=0.7)
    
    ax5.arrow(v2[0], v2[1], v1[0]-v2[0], v1[1]-v2[1], head_width=0.2, head_length=0.2,
             fc='green', ec='green', linewidth=2, linestyle='--', label='v-w', alpha=0.7)
    
    distance = np.linalg.norm(v1 - v2)
    ax5.text(2, 3.5, f'd(v,w) = {distance:.2f}', fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax5.axhline(y=0, color='k', linewidth=0.5)
    ax5.axvline(x=0, color='k', linewidth=0.5)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlabel('x', fontsize=12)
    ax5.set_ylabel('y', fontsize=12)
    ax5.set_title('Distance Between Vectors', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.set_aspect('equal')
    ax5.set_xlim(-0.5, 4)
    ax5.set_ylim(-0.5, 5)
    
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    
    u = np.linspace(0, 2 * np.pi, 50)
    v_sphere = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v_sphere))
    y_sphere = np.outer(np.sin(u), np.sin(v_sphere))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v_sphere))
    
    ax6.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color='gray')
    
    unit_vectors = [
        ([1, 0, 0], 'red'),
        ([0, 1, 0], 'green'),
        ([0, 0, 1], 'blue'),
        ([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], 'orange')
    ]
    
    for v_arr, color in unit_vectors:
        v_np = np.array(v_arr)
        ax6.quiver(0, 0, 0, v_np[0], v_np[1], v_np[2], 
                   color=color, arrow_length_ratio=0.15, linewidth=2)
    
    ax6.set_xlabel('X', fontsize=10)
    ax6.set_ylabel('Y', fontsize=10)
    ax6.set_zlabel('Z', fontsize=10)
    ax6.set_title('3D Unit Sphere', fontsize=14, fontweight='bold')
    ax6.set_xlim(-1, 1)
    ax6.set_ylim(-1, 1)
    ax6.set_zlim(-1, 1)
    
    plt.tight_layout()
    plt.savefig('norm_normalization_visualizations.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved comprehensive visualizations as 'norm_normalization_visualizations.png'")
    plt.close()

def main():
    """Main function to run all demonstrations"""
    print("\n" + "||·||" * 20)
    print("NORM AND NORMALIZATION")
    print("Comprehensive Python Demonstrations")
    print("||·||" * 20)
    
    basic_norm()
    normalization_demo()
    norm_properties()
    quantum_state_normalization()
    common_quantum_states()
    unit_vectors()
    distance_between_vectors()
    normalization_after_operations()
    visualize_norm_normalization()
    
    print_section_header("KEY TAKEAWAYS")
    print("\n1. Norm Definition:")
    print("   - ||v|| = √(v₁² + v₂² + ... + vₙ²)")
    print("   - Measures length/magnitude of vector")
    print("   - Always non-negative")
    
    print("\n2. Normalization:")
    print("   - v̂ = v / ||v||")
    print("   - Creates unit vector (length 1)")
    print("   - Preserves direction")
    
    print("\n3. Properties:")
    print("   - Non-negativity: ||v|| ≥ 0")
    print("   - Scalar multiplication: ||cv|| = |c| · ||v||")
    print("   - Triangle inequality: ||v + w|| ≤ ||v|| + ||w||")
    print("   - Inner product: ||v|| = √⟨v|v⟩")
    
    print("\n4. Quantum State Normalization:")
    print("   - ALL valid quantum states must be normalized")
    print("   - ||ψ⟩|| = 1 ensures total probability = 1")
    print("   - |α|² + |β|² = 1 for qubit states")
    
    print("\n5. Unit Vectors:")
    print("   - ||v̂|| = 1")
    print("   - Standard basis vectors are unit vectors")
    print("   - Specify direction without magnitude")
    
    print("\n6. Distance:")
    print("   - d(v, w) = ||v - w||")
    print("   - Measures difference between vectors")
    print("   - Used to compare quantum states")
    
    print("\n7. Applications:")
    print("   - Quantum state preparation")
    print("   - Probability calculations")
    print("   - Direction vectors")
    print("   - Orthonormal bases")
    
    print("\n" + "=" * 80)
    print("All demonstrations completed successfully!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
