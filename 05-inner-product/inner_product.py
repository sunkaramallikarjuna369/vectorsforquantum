"""
Inner Product (Dot Product)
Comprehensive Python script demonstrating inner products and quantum measurement
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

def basic_inner_product():
    """Demonstrate basic inner product"""
    print_section_header("BASIC INNER PRODUCT")
    
    print("\nInner product (dot product): sum of products of corresponding components")
    print("  v · w = v₁w₁ + v₂w₂ + ... + vₙwₙ")
    
    print_subsection("Example 1: 2D Real Vectors")
    v = np.array([[3], [4]], dtype=float)
    w = np.array([[1], [2]], dtype=float)
    inner_prod = np.vdot(v, w)
    
    print(f"\nv = {v.T[0]}")
    print(f"w = {w.T[0]}")
    print(f"v · w = {inner_prod}")
    print(f"\nCalculation:")
    print(f"  v · w = ({v[0,0]})({w[0,0]}) + ({v[1,0]})({w[1,0]})")
    print(f"        = {v[0,0]*w[0,0]} + {v[1,0]*w[1,0]}")
    print(f"        = {inner_prod}")
    
    print_subsection("Example 2: 3D Real Vectors")
    v3d = np.array([[1], [2], [3]], dtype=float)
    w3d = np.array([[4], [-1], [2]], dtype=float)
    inner_prod_3d = np.vdot(v3d, w3d)
    
    print(f"\nv = {v3d.T[0]}")
    print(f"w = {w3d.T[0]}")
    print(f"v · w = {inner_prod_3d}")
    print(f"\nCalculation:")
    print(f"  v · w = (1)(4) + (2)(-1) + (3)(2)")
    print(f"        = 4 - 2 + 6")
    print(f"        = {inner_prod_3d}")
    
    print_subsection("Example 3: Complex Vectors (Quantum)")
    psi = np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]], dtype=complex)
    phi = np.array([[1], [0]], dtype=complex)
    inner_prod_complex = np.vdot(psi, phi)
    
    print(f"\n|ψ⟩ = {psi.T[0]}")
    print(f"|φ⟩ = {phi.T[0]}")
    print(f"⟨ψ|φ⟩ = {inner_prod_complex}")
    print(f"\nNote: np.vdot automatically takes complex conjugate of first argument")

def bra_ket_notation():
    """Demonstrate bra-ket notation"""
    print_section_header("BRA-KET NOTATION")
    
    print("\nDirac notation for quantum states:")
    print("  |ψ⟩ = ket (column vector)")
    print("  ⟨ψ| = bra (row vector, conjugate transpose)")
    print("  ⟨ψ|φ⟩ = inner product")
    
    zero = np.array([[1], [0]], dtype=complex)
    one = np.array([[0], [1]], dtype=complex)
    
    print_subsection("Computational Basis States")
    print(f"\n|0⟩ = {zero.T[0]}")
    print(f"|1⟩ = {one.T[0]}")
    
    print(f"\n⟨0|0⟩ = {np.vdot(zero, zero)[0]}")
    print(f"⟨0|1⟩ = {np.vdot(zero, one)[0]}")
    print(f"⟨1|0⟩ = {np.vdot(one, zero)[0]}")
    print(f"⟨1|1⟩ = {np.vdot(one, one)[0]}")
    
    print("\nBasis states are orthonormal:")
    print("  ⟨i|j⟩ = δᵢⱼ (Kronecker delta)")
    
    print_subsection("Superposition States")
    plus = (1/np.sqrt(2)) * (zero + one)
    minus = (1/np.sqrt(2)) * (zero - one)
    
    print(f"\n|+⟩ = (1/√2)(|0⟩ + |1⟩) = {plus.T[0]}")
    print(f"|-⟩ = (1/√2)(|0⟩ - |1⟩) = {minus.T[0]}")
    
    print(f"\n⟨+|+⟩ = {np.vdot(plus, plus)[0]:.4f}")
    print(f"⟨+|-⟩ = {np.vdot(plus, minus)[0]:.4f}")
    print(f"⟨-|+⟩ = {np.vdot(minus, plus)[0]:.4f}")
    print(f"⟨-|-⟩ = {np.vdot(minus, minus)[0]:.4f}")
    
    print("\n|+⟩ and |-⟩ are also orthonormal!")

def geometric_interpretation():
    """Demonstrate geometric interpretation"""
    print_section_header("GEOMETRIC INTERPRETATION")
    
    print("\nThe inner product relates to the angle between vectors:")
    print("  v · w = |v| |w| cos(θ)")
    
    print_subsection("Example: Finding the Angle")
    v = np.array([[3], [4]], dtype=float)
    w = np.array([[1], [2]], dtype=float)
    
    inner_prod = np.vdot(v, w)
    mag_v = np.linalg.norm(v)
    mag_w = np.linalg.norm(w)
    
    cos_theta = inner_prod / (mag_v * mag_w)
    theta_rad = np.arccos(cos_theta)
    theta_deg = theta_rad * 180 / np.pi
    
    print(f"\nv = {v.T[0]}")
    print(f"w = {w.T[0]}")
    print(f"\nv · w = {inner_prod}")
    print(f"|v| = {mag_v:.3f}")
    print(f"|w| = {mag_w:.3f}")
    print(f"\ncos(θ) = (v · w) / (|v| |w|)")
    print(f"       = {inner_prod} / ({mag_v:.3f} × {mag_w:.3f})")
    print(f"       = {cos_theta:.4f}")
    print(f"\nθ = arccos({cos_theta:.4f})")
    print(f"  = {theta_rad:.4f} rad")
    print(f"  = {theta_deg:.2f}°")
    
    print_subsection("Special Cases")
    
    v_parallel = np.array([[1], [2]], dtype=float)
    w_parallel = np.array([[2], [4]], dtype=float)
    inner_parallel = np.vdot(v_parallel, w_parallel)
    mag_v_par = np.linalg.norm(v_parallel)
    mag_w_par = np.linalg.norm(w_parallel)
    
    print(f"\n1. Parallel vectors (θ = 0°):")
    print(f"   v = {v_parallel.T[0]}, w = {w_parallel.T[0]}")
    print(f"   v · w = {inner_parallel} = |v| |w| = {mag_v_par:.3f} × {mag_w_par:.3f} = {mag_v_par * mag_w_par:.3f}")
    
    v_ortho = np.array([[1], [0]], dtype=float)
    w_ortho = np.array([[0], [1]], dtype=float)
    inner_ortho = np.vdot(v_ortho, w_ortho)
    
    print(f"\n2. Orthogonal vectors (θ = 90°):")
    print(f"   v = {v_ortho.T[0]}, w = {w_ortho.T[0]}")
    print(f"   v · w = {inner_ortho} (perpendicular!)")
    
    v_opp = np.array([[1], [2]], dtype=float)
    w_opp = np.array([[-1], [-2]], dtype=float)
    inner_opp = np.vdot(v_opp, w_opp)
    mag_v_opp = np.linalg.norm(v_opp)
    mag_w_opp = np.linalg.norm(w_opp)
    
    print(f"\n3. Opposite vectors (θ = 180°):")
    print(f"   v = {v_opp.T[0]}, w = {w_opp.T[0]}")
    print(f"   v · w = {inner_opp} = -|v| |w| = -{mag_v_opp:.3f} × {mag_w_opp:.3f} = {-mag_v_opp * mag_w_opp:.3f}")

def inner_product_properties():
    """Demonstrate properties of inner product"""
    print_section_header("PROPERTIES OF INNER PRODUCT")
    
    v = np.array([[3], [4]], dtype=complex)
    w = np.array([[1], [2]], dtype=complex)
    u = np.array([[2], [1]], dtype=complex)
    alpha = 2 + 1j
    beta = 1 - 1j
    
    print_subsection("1. Conjugate Symmetry")
    print("⟨v|w⟩ = ⟨w|v⟩*")
    
    vw = np.vdot(v, w)
    wv = np.vdot(w, v)
    
    print(f"\n⟨v|w⟩ = {vw[0]}")
    print(f"⟨w|v⟩ = {wv[0]}")
    print(f"⟨w|v⟩* = {np.conj(wv[0])}")
    print(f"Equal: {np.isclose(vw, np.conj(wv))[0]} ✓")
    
    print_subsection("2. Linearity in Second Argument")
    print("⟨v|αw + βu⟩ = α⟨v|w⟩ + β⟨v|u⟩")
    
    left = np.vdot(v, alpha * w + beta * u)
    right = alpha * np.vdot(v, w) + beta * np.vdot(v, u)
    
    print(f"\n⟨v|αw + βu⟩ = {left[0]}")
    print(f"α⟨v|w⟩ + β⟨v|u⟩ = {right[0]}")
    print(f"Equal: {np.isclose(left, right)[0]} ✓")
    
    print_subsection("3. Positive Definiteness")
    print("⟨v|v⟩ ≥ 0, and ⟨v|v⟩ = 0 iff v = 0")
    
    vv = np.vdot(v, v)
    print(f"\n⟨v|v⟩ = {vv[0].real:.4f}")
    print(f"Is positive: {vv[0].real >= 0} ✓")
    
    zero_vec = np.array([[0], [0]], dtype=complex)
    zero_inner = np.vdot(zero_vec, zero_vec)
    print(f"\n⟨0|0⟩ = {zero_inner[0].real:.4f}")
    
    print_subsection("4. Norm Relationship")
    print("||v|| = √⟨v|v⟩")
    
    norm_from_inner = np.sqrt(np.vdot(v, v)[0].real)
    norm_direct = np.linalg.norm(v)
    
    print(f"\n√⟨v|v⟩ = {norm_from_inner:.4f}")
    print(f"||v|| = {norm_direct:.4f}")
    print(f"Equal: {np.isclose(norm_from_inner, norm_direct)} ✓")

def quantum_measurement():
    """Demonstrate quantum measurement using inner products"""
    print_section_header("QUANTUM MEASUREMENT")
    
    print("\nMeasurement probability: P(φ) = |⟨φ|ψ⟩|²")
    print("The probability of measuring state |ψ⟩ in basis state |φ⟩")
    
    zero = np.array([[1], [0]], dtype=complex)
    one = np.array([[0], [1]], dtype=complex)
    
    print_subsection("Example 1: Measuring |+⟩ in Computational Basis")
    plus = (1/np.sqrt(2)) * (zero + one)
    
    print(f"\n|+⟩ = {plus.T[0]}")
    
    amplitude_0 = np.vdot(zero, plus)[0]
    prob_0 = np.abs(amplitude_0)**2
    
    print(f"\nAmplitude for |0⟩: ⟨0|+⟩ = {amplitude_0:.4f}")
    print(f"Probability: P(0) = |⟨0|+⟩|² = {prob_0:.4f} = 50%")
    
    amplitude_1 = np.vdot(one, plus)[0]
    prob_1 = np.abs(amplitude_1)**2
    
    print(f"\nAmplitude for |1⟩: ⟨1|+⟩ = {amplitude_1:.4f}")
    print(f"Probability: P(1) = |⟨1|+⟩|² = {prob_1:.4f} = 50%")
    
    print(f"\nTotal probability: {prob_0 + prob_1:.4f} = 100% ✓")
    
    print_subsection("Example 2: Measuring |-⟩ in Computational Basis")
    minus = (1/np.sqrt(2)) * (zero - one)
    
    print(f"\n|-⟩ = {minus.T[0]}")
    
    amplitude_0 = np.vdot(zero, minus)[0]
    prob_0 = np.abs(amplitude_0)**2
    amplitude_1 = np.vdot(one, minus)[0]
    prob_1 = np.abs(amplitude_1)**2
    
    print(f"\nP(0) = |⟨0|-⟩|² = {prob_0:.4f} = 50%")
    print(f"P(1) = |⟨1|-⟩|² = {prob_1:.4f} = 50%")
    
    print("\nSame probabilities as |+⟩, but different quantum state!")
    
    print_subsection("Example 3: Measuring |0⟩ in |+⟩/|-⟩ Basis")
    
    amplitude_plus = np.vdot(plus, zero)[0]
    prob_plus = np.abs(amplitude_plus)**2
    amplitude_minus = np.vdot(minus, zero)[0]
    prob_minus = np.abs(amplitude_minus)**2
    
    print(f"\n|0⟩ measured in |+⟩/|-⟩ basis:")
    print(f"P(+) = |⟨+|0⟩|² = {prob_plus:.4f} = 50%")
    print(f"P(-) = |⟨-|0⟩|² = {prob_minus:.4f} = 50%")

def orthogonality_demo():
    """Demonstrate orthogonality"""
    print_section_header("ORTHOGONALITY")
    
    print("\nTwo vectors are orthogonal if ⟨v|w⟩ = 0")
    
    zero = np.array([[1], [0]], dtype=complex)
    one = np.array([[0], [1]], dtype=complex)
    
    print_subsection("Computational Basis")
    print(f"\n|0⟩ = {zero.T[0]}")
    print(f"|1⟩ = {one.T[0]}")
    print(f"⟨0|1⟩ = {np.vdot(zero, one)[0]}")
    print("Orthogonal! ✓")
    
    print_subsection("|+⟩/|-⟩ Basis")
    plus = (1/np.sqrt(2)) * (zero + one)
    minus = (1/np.sqrt(2)) * (zero - one)
    
    print(f"\n|+⟩ = {plus.T[0]}")
    print(f"|-⟩ = {minus.T[0]}")
    print(f"⟨+|-⟩ = {np.vdot(plus, minus)[0]:.4f}")
    print("Orthogonal! ✓")
    
    print_subsection("Creating Orthogonal Vector")
    print("\nGiven v, find w orthogonal to v:")
    
    v = np.array([[3], [4]], dtype=float)
    w_ortho = np.array([[-v[1, 0]], [v[0, 0]]], dtype=float)
    
    print(f"v = {v.T[0]}")
    print(f"w = {w_ortho.T[0]}")
    print(f"v · w = {np.vdot(v, w_ortho):.4f}")
    print("Orthogonal! ✓")

def state_overlap():
    """Demonstrate quantum state overlap"""
    print_section_header("QUANTUM STATE OVERLAP")
    
    print("\n|⟨ψ|φ⟩|² measures how similar two quantum states are")
    print("  = 1: identical states")
    print("  = 0: orthogonal states")
    print("  0 < |⟨ψ|φ⟩|² < 1: partially overlapping")
    
    zero = np.array([[1], [0]], dtype=complex)
    one = np.array([[0], [1]], dtype=complex)
    plus = (1/np.sqrt(2)) * (zero + one)
    minus = (1/np.sqrt(2)) * (zero - one)
    
    print_subsection("Example 1: Identical States")
    overlap = np.abs(np.vdot(plus, plus)[0])**2
    print(f"\n|⟨+|+⟩|² = {overlap:.4f} = 100%")
    print("Completely overlapping!")
    
    print_subsection("Example 2: Orthogonal States")
    overlap = np.abs(np.vdot(zero, one)[0])**2
    print(f"\n|⟨0|1⟩|² = {overlap:.4f} = 0%")
    print("No overlap!")
    
    print_subsection("Example 3: Partial Overlap")
    overlap = np.abs(np.vdot(zero, plus)[0])**2
    print(f"\n|⟨0|+⟩|² = {overlap:.4f} = 50%")
    print("Partial overlap!")
    
    print_subsection("Example 4: Custom States")
    psi = np.array([[0.6], [0.8]], dtype=complex)
    phi = np.array([[0.8], [0.6]], dtype=complex)
    overlap = np.abs(np.vdot(psi, phi)[0])**2
    
    print(f"\n|ψ⟩ = {psi.T[0]}")
    print(f"|φ⟩ = {phi.T[0]}")
    print(f"|⟨ψ|φ⟩|² = {overlap:.4f} = {overlap*100:.1f}%")

def cauchy_schwarz_inequality():
    """Demonstrate Cauchy-Schwarz inequality"""
    print_section_header("CAUCHY-SCHWARZ INEQUALITY")
    
    print("\nFundamental inequality: |⟨v|w⟩| ≤ ||v|| ||w||")
    print("Equality holds when v and w are parallel")
    
    print_subsection("Example 1: General Vectors")
    v = np.array([[3], [4]], dtype=float)
    w = np.array([[1], [2]], dtype=float)
    
    inner_prod = np.abs(np.vdot(v, w))
    product_norms = np.linalg.norm(v) * np.linalg.norm(w)
    
    print(f"\nv = {v.T[0]}")
    print(f"w = {w.T[0]}")
    print(f"\n|⟨v|w⟩| = {inner_prod:.4f}")
    print(f"||v|| ||w|| = {product_norms:.4f}")
    print(f"\n{inner_prod:.4f} ≤ {product_norms:.4f} ✓")
    
    print_subsection("Example 2: Parallel Vectors (Equality)")
    v_par = np.array([[1], [2]], dtype=float)
    w_par = np.array([[2], [4]], dtype=float)
    
    inner_prod_par = np.abs(np.vdot(v_par, w_par))
    product_norms_par = np.linalg.norm(v_par) * np.linalg.norm(w_par)
    
    print(f"\nv = {v_par.T[0]}")
    print(f"w = {w_par.T[0]}")
    print(f"\n|⟨v|w⟩| = {inner_prod_par:.4f}")
    print(f"||v|| ||w|| = {product_norms_par:.4f}")
    print(f"\nEquality holds! Vectors are parallel ✓")

def projection_demo():
    """Demonstrate vector projection"""
    print_section_header("VECTOR PROJECTION")
    
    print("\nProject vector w onto vector v:")
    print("  proj_v(w) = (⟨v|w⟩ / ⟨v|v⟩) v")
    
    print_subsection("Example")
    v = np.array([[1], [0]], dtype=float)
    w = np.array([[3], [4]], dtype=float)
    
    inner_vw = np.vdot(v, w)
    inner_vv = np.vdot(v, v)
    projection = (inner_vw / inner_vv) * v
    
    print(f"\nv = {v.T[0]}")
    print(f"w = {w.T[0]}")
    print(f"\nproj_v(w) = (⟨v|w⟩ / ⟨v|v⟩) v")
    print(f"          = ({inner_vw} / {inner_vv}) v")
    print(f"          = {projection.T[0]}")
    
    perpendicular = w - projection
    print(f"\nPerpendicular component:")
    print(f"w - proj_v(w) = {perpendicular.T[0]}")
    
    check = np.vdot(v, perpendicular)
    print(f"\nVerify orthogonality:")
    print(f"⟨v|w - proj_v(w)⟩ = {check:.4f} ≈ 0 ✓")

def visualize_inner_products():
    """Create comprehensive visualizations"""
    print_section_header("CREATING VISUALIZATIONS")
    
    fig = plt.figure(figsize=(16, 12))
    
    ax1 = fig.add_subplot(2, 3, 1)
    v = np.array([3, 4])
    w = np.array([1, 2])
    
    ax1.arrow(0, 0, v[0], v[1], head_width=0.3, head_length=0.3,
             fc='blue', ec='blue', linewidth=2, label='v', alpha=0.7)
    ax1.arrow(0, 0, w[0], w[1], head_width=0.3, head_length=0.3,
             fc='red', ec='red', linewidth=2, label='w', alpha=0.7)
    
    angle = np.arctan2(w[1], w[0]) - np.arctan2(v[1], v[0])
    if angle < 0:
        angle += 2 * np.pi
    theta_range = np.linspace(np.arctan2(v[1], v[0]), np.arctan2(w[1], w[0]), 50)
    arc_radius = 1.5
    ax1.plot(arc_radius * np.cos(theta_range), arc_radius * np.sin(theta_range), 'g--', linewidth=2)
    
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Inner Product and Angle', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.set_xlim(-1, 5)
    ax1.set_ylim(-1, 5)
    
    ax2 = fig.add_subplot(2, 3, 2)
    
    v_proj = np.array([1, 0])
    w_proj = np.array([3, 4])
    proj = (np.dot(v_proj, w_proj) / np.dot(v_proj, v_proj)) * v_proj
    perp = w_proj - proj
    
    ax2.arrow(0, 0, v_proj[0], v_proj[1], head_width=0.2, head_length=0.2,
             fc='blue', ec='blue', linewidth=2, label='v', alpha=0.7)
    ax2.arrow(0, 0, w_proj[0], w_proj[1], head_width=0.2, head_length=0.2,
             fc='red', ec='red', linewidth=2, label='w', alpha=0.7)
    ax2.arrow(0, 0, proj[0], proj[1], head_width=0.2, head_length=0.2,
             fc='green', ec='green', linewidth=2, label='proj_v(w)', alpha=0.7)
    ax2.arrow(proj[0], proj[1], perp[0], perp[1], head_width=0.2, head_length=0.2,
             fc='orange', ec='orange', linewidth=2, linestyle='--', label='perpendicular', alpha=0.7)
    
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('Vector Projection', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.set_xlim(-1, 4)
    ax2.set_ylim(-1, 5)
    
    ax3 = fig.add_subplot(2, 3, 3)
    
    ortho_pairs = [
        ([1, 0], [0, 1], 'Basis'),
        ([1, 1], [1, -1], '45° rotated'),
        ([3, 4], [-4, 3], 'Custom')
    ]
    
    colors = ['blue', 'red', 'green']
    for i, (v_o, w_o, label) in enumerate(ortho_pairs):
        v_norm = np.array(v_o) / np.linalg.norm(v_o)
        w_norm = np.array(w_o) / np.linalg.norm(w_o)
        ax3.arrow(0, 0, v_norm[0], v_norm[1], head_width=0.05, head_length=0.05,
                 fc=colors[i], ec=colors[i], linewidth=2, alpha=0.5)
        ax3.arrow(0, 0, w_norm[0], w_norm[1], head_width=0.05, head_length=0.05,
                 fc=colors[i], ec=colors[i], linewidth=2, alpha=0.5, linestyle='--')
    
    ax3.axhline(y=0, color='k', linewidth=0.5)
    ax3.axvline(x=0, color='k', linewidth=0.5)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('y', fontsize=12)
    ax3.set_title('Orthogonal Vector Pairs', fontsize=14, fontweight='bold')
    ax3.set_aspect('equal')
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)
    
    ax4 = fig.add_subplot(2, 3, 4)
    
    zero = np.array([1, 0])
    one = np.array([0, 1])
    plus = np.array([1, 1]) / np.sqrt(2)
    minus = np.array([1, -1]) / np.sqrt(2)
    
    states = {
        '|0⟩': zero,
        '|1⟩': one,
        '|+⟩': plus,
        '|-⟩': minus
    }
    
    x = np.arange(len(states))
    width = 0.35
    
    probs_0 = [np.abs(np.dot(zero, state))**2 for state in states.values()]
    probs_1 = [np.abs(np.dot(one, state))**2 for state in states.values()]
    
    ax4.bar(x - width/2, probs_0, width, label='P(|0⟩)', alpha=0.7)
    ax4.bar(x + width/2, probs_1, width, label='P(|1⟩)', alpha=0.7)
    
    ax4.set_ylabel('Probability', fontsize=12)
    ax4.set_title('Measurement in Computational Basis', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(states.keys())
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    ax5 = fig.add_subplot(2, 3, 5)
    
    state_list = [zero, one, plus, minus]
    state_names = ['|0⟩', '|1⟩', '|+⟩', '|-⟩']
    
    overlap_matrix = np.zeros((len(state_list), len(state_list)))
    for i, s1 in enumerate(state_list):
        for j, s2 in enumerate(state_list):
            overlap_matrix[i, j] = np.abs(np.dot(s1, s2))**2
    
    im = ax5.imshow(overlap_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    ax5.set_xticks(range(len(state_names)))
    ax5.set_yticks(range(len(state_names)))
    ax5.set_xticklabels(state_names)
    ax5.set_yticklabels(state_names)
    ax5.set_title('State Overlap Matrix |⟨ψ|φ⟩|²', fontsize=14, fontweight='bold')
    
    for i in range(len(state_names)):
        for j in range(len(state_names)):
            text = ax5.text(j, i, f'{overlap_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax5)
    
    ax6 = fig.add_subplot(2, 3, 6)
    
    n_samples = 100
    inner_prods = []
    norm_prods = []
    
    for _ in range(n_samples):
        v_rand = np.random.randn(2)
        w_rand = np.random.randn(2)
        inner_prods.append(np.abs(np.dot(v_rand, w_rand)))
        norm_prods.append(np.linalg.norm(v_rand) * np.linalg.norm(w_rand))
    
    ax6.scatter(norm_prods, inner_prods, alpha=0.5)
    
    max_val = max(max(norm_prods), max(inner_prods))
    ax6.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='|⟨v|w⟩| = ||v|| ||w||')
    
    ax6.set_xlabel('||v|| ||w||', fontsize=12)
    ax6.set_ylabel('|⟨v|w⟩|', fontsize=12)
    ax6.set_title('Cauchy-Schwarz Inequality', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('inner_product_visualizations.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved comprehensive visualizations as 'inner_product_visualizations.png'")
    plt.close()

def main():
    """Main function to run all demonstrations"""
    print("\n" + "⟨·|·⟩" * 20)
    print("INNER PRODUCT (DOT PRODUCT)")
    print("Comprehensive Python Demonstrations")
    print("⟨·|·⟩" * 20)
    
    basic_inner_product()
    bra_ket_notation()
    geometric_interpretation()
    inner_product_properties()
    quantum_measurement()
    orthogonality_demo()
    state_overlap()
    cauchy_schwarz_inequality()
    projection_demo()
    visualize_inner_products()
    
    print_section_header("KEY TAKEAWAYS")
    print("\n1. Definition:")
    print("   - Real vectors: v · w = v₁w₁ + v₂w₂ + ...")
    print("   - Complex vectors: ⟨v|w⟩ = v₁*w₁ + v₂*w₂ + ...")
    print("   - Measures similarity between vectors")
    
    print("\n2. Bra-Ket Notation:")
    print("   - |ψ⟩ = ket (column vector)")
    print("   - ⟨ψ| = bra (row vector, conjugate transpose)")
    print("   - ⟨ψ|φ⟩ = inner product")
    
    print("\n3. Geometric Interpretation:")
    print("   - v · w = |v| |w| cos(θ)")
    print("   - Relates to angle between vectors")
    print("   - θ = 90° ⟹ v · w = 0 (orthogonal)")
    
    print("\n4. Properties:")
    print("   - Conjugate symmetry: ⟨v|w⟩ = ⟨w|v⟩*")
    print("   - Linearity in second argument")
    print("   - Positive definiteness: ⟨v|v⟩ ≥ 0")
    print("   - Norm relationship: ||v|| = √⟨v|v⟩")
    
    print("\n5. Quantum Measurement:")
    print("   - P(φ) = |⟨φ|ψ⟩|²")
    print("   - Probability of measuring |ψ⟩ in state |φ⟩")
    print("   - Central to quantum mechanics")
    
    print("\n6. Orthogonality:")
    print("   - ⟨v|w⟩ = 0 ⟹ v ⊥ w")
    print("   - Orthonormal basis: ⟨vᵢ|vⱼ⟩ = δᵢⱼ")
    print("   - Computational basis {|0⟩, |1⟩} is orthonormal")
    
    print("\n7. Applications:")
    print("   - Quantum state overlap")
    print("   - Expectation values")
    print("   - Vector projection")
    print("   - Gram-Schmidt orthogonalization")
    
    print("\n" + "=" * 80)
    print("All demonstrations completed successfully!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
