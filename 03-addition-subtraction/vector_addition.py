"""
Vector Addition and Subtraction
Comprehensive Python script demonstrating vector operations and quantum superposition
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

def basic_vector_addition():
    """Demonstrate basic vector addition"""
    print_section_header("BASIC VECTOR ADDITION")
    
    print("\nVector addition: add corresponding components")
    print("  v + w = [v₁ + w₁, v₂ + w₂, ..., vₙ + wₙ]ᵀ")
    
    print_subsection("Example 1: 2D Vectors")
    v = np.array([[3], [4]], dtype=float)
    w = np.array([[1], [-2]], dtype=float)
    sum_vec = v + w
    
    print(f"\nv = {v.T[0]}")
    print(f"w = {w.T[0]}")
    print(f"v + w = {sum_vec.T[0]}")
    print(f"\nComponent-wise:")
    print(f"  First component: {v[0,0]} + {w[0,0]} = {sum_vec[0,0]}")
    print(f"  Second component: {v[1,0]} + {w[1,0]} = {sum_vec[1,0]}")
    
    print_subsection("Example 2: 3D Vectors")
    v3d = np.array([[1], [2], [3]], dtype=float)
    w3d = np.array([[4], [-1], [2]], dtype=float)
    sum3d = v3d + w3d
    
    print(f"\nv = {v3d.T[0]}")
    print(f"w = {w3d.T[0]}")
    print(f"v + w = {sum3d.T[0]}")
    
    print_subsection("Example 3: Complex Vectors (Quantum)")
    v_complex = np.array([[1+2j], [3-1j]], dtype=complex)
    w_complex = np.array([[2-1j], [1+2j]], dtype=complex)
    sum_complex = v_complex + w_complex
    
    print(f"\nv = {v_complex.T[0]}")
    print(f"w = {w_complex.T[0]}")
    print(f"v + w = {sum_complex.T[0]}")
    print(f"\nComponent-wise:")
    print(f"  First: ({v_complex[0,0]}) + ({w_complex[0,0]}) = {sum_complex[0,0]}")
    print(f"  Second: ({v_complex[1,0]}) + ({w_complex[1,0]}) = {sum_complex[1,0]}")

def basic_vector_subtraction():
    """Demonstrate basic vector subtraction"""
    print_section_header("BASIC VECTOR SUBTRACTION")
    
    print("\nVector subtraction: subtract corresponding components")
    print("  v - w = [v₁ - w₁, v₂ - w₂, ..., vₙ - wₙ]ᵀ")
    
    print_subsection("Example 1: 2D Vectors")
    v = np.array([[3], [4]], dtype=float)
    w = np.array([[1], [-2]], dtype=float)
    diff_vec = v - w
    
    print(f"\nv = {v.T[0]}")
    print(f"w = {w.T[0]}")
    print(f"v - w = {diff_vec.T[0]}")
    print(f"\nComponent-wise:")
    print(f"  First component: {v[0,0]} - {w[0,0]} = {diff_vec[0,0]}")
    print(f"  Second component: {v[1,0]} - {w[1,0]} = {diff_vec[1,0]}")
    
    print_subsection("Example 2: Relationship to Addition")
    neg_w = -w
    diff_as_sum = v + neg_w
    
    print(f"\nSubtraction as addition of negative:")
    print(f"v - w = v + (-w)")
    print(f"-w = {neg_w.T[0]}")
    print(f"v + (-w) = {diff_as_sum.T[0]}")
    print(f"v - w = {diff_vec.T[0]}")
    print(f"Equal: {np.array_equal(diff_vec, diff_as_sum)}")

def addition_properties():
    """Demonstrate properties of vector addition"""
    print_section_header("PROPERTIES OF VECTOR ADDITION")
    
    u = np.array([[1], [2]], dtype=float)
    v = np.array([[3], [4]], dtype=float)
    w = np.array([[5], [6]], dtype=float)
    zero = np.array([[0], [0]], dtype=float)
    
    print_subsection("1. Commutative Property")
    print("v + w = w + v")
    
    vw = v + w
    wv = w + v
    print(f"\nv + w = {vw.T[0]}")
    print(f"w + v = {wv.T[0]}")
    print(f"Equal: {np.array_equal(vw, wv)} ✓")
    
    print_subsection("2. Associative Property")
    print("(u + v) + w = u + (v + w)")
    
    left = (u + v) + w
    right = u + (v + w)
    print(f"\n(u + v) + w = {left.T[0]}")
    print(f"u + (v + w) = {right.T[0]}")
    print(f"Equal: {np.array_equal(left, right)} ✓")
    
    print_subsection("3. Identity Element")
    print("v + 0 = v")
    
    v_plus_zero = v + zero
    print(f"\nv = {v.T[0]}")
    print(f"v + 0 = {v_plus_zero.T[0]}")
    print(f"Equal: {np.array_equal(v, v_plus_zero)} ✓")
    
    print_subsection("4. Inverse Element")
    print("v + (-v) = 0")
    
    neg_v = -v
    v_plus_neg_v = v + neg_v
    print(f"\nv = {v.T[0]}")
    print(f"-v = {neg_v.T[0]}")
    print(f"v + (-v) = {v_plus_neg_v.T[0]}")
    print(f"Is zero: {np.allclose(v_plus_neg_v, zero)} ✓")

def geometric_interpretation():
    """Demonstrate geometric interpretation"""
    print_section_header("GEOMETRIC INTERPRETATION")
    
    print_subsection("Tip-to-Tail Method")
    print("\n1. Place tail of w at tip of v")
    print("2. Sum v + w goes from tail of v to tip of w")
    
    v = np.array([[3], [4]], dtype=float)
    w = np.array([[1], [-2]], dtype=float)
    sum_vec = v + w
    
    print(f"\nv = {v.T[0]}")
    print(f"w = {w.T[0]}")
    print(f"v + w = {sum_vec.T[0]}")
    
    print("\nVisualization:")
    print("  Start at origin (0, 0)")
    print(f"  Follow v to reach ({v[0,0]}, {v[1,0]})")
    print(f"  From there, follow w to reach ({sum_vec[0,0]}, {sum_vec[1,0]})")
    print(f"  This is the same as v + w!")
    
    print_subsection("Parallelogram Method")
    print("\n1. Place both vectors at origin")
    print("2. Complete the parallelogram")
    print("3. Diagonal is v + w")
    
    print(f"\nv from origin: (0,0) → ({v[0,0]}, {v[1,0]})")
    print(f"w from origin: (0,0) → ({w[0,0]}, {w[1,0]})")
    print(f"Diagonal (v+w): (0,0) → ({sum_vec[0,0]}, {sum_vec[1,0]})")
    
    print_subsection("Magnitude Relationships")
    mag_v = np.linalg.norm(v)
    mag_w = np.linalg.norm(w)
    mag_sum = np.linalg.norm(sum_vec)
    
    print(f"\n|v| = {mag_v:.3f}")
    print(f"|w| = {mag_w:.3f}")
    print(f"|v + w| = {mag_sum:.3f}")
    print(f"\nTriangle inequality: |v + w| ≤ |v| + |w|")
    print(f"  {mag_sum:.3f} ≤ {mag_v + mag_w:.3f} ✓")

def quantum_superposition():
    """Demonstrate quantum superposition as vector addition"""
    print_section_header("QUANTUM SUPERPOSITION")
    
    print("\nIn quantum computing, vector addition represents superposition!")
    print("If |ψ₁⟩ and |ψ₂⟩ are valid states, then α|ψ₁⟩ + β|ψ₂⟩ is also valid")
    
    zero = np.array([[1], [0]], dtype=complex)
    one = np.array([[0], [1]], dtype=complex)
    
    print_subsection("Example 1: |+⟩ State (Equal Superposition)")
    plus = (1/np.sqrt(2)) * zero + (1/np.sqrt(2)) * one
    
    print(f"\n|0⟩ = {zero.T[0]}")
    print(f"|1⟩ = {one.T[0]}")
    print(f"\n|+⟩ = (1/√2)|0⟩ + (1/√2)|1⟩")
    print(f"    = {plus.T[0]}")
    
    prob_0 = np.abs(plus[0, 0])**2
    prob_1 = np.abs(plus[1, 0])**2
    print(f"\nProbabilities:")
    print(f"  P(0) = |α|² = {prob_0:.4f} = 50%")
    print(f"  P(1) = |β|² = {prob_1:.4f} = 50%")
    
    print_subsection("Example 2: |-⟩ State (With Relative Phase)")
    minus = (1/np.sqrt(2)) * zero - (1/np.sqrt(2)) * one
    
    print(f"\n|-⟩ = (1/√2)|0⟩ - (1/√2)|1⟩")
    print(f"    = {minus.T[0]}")
    
    prob_0 = np.abs(minus[0, 0])**2
    prob_1 = np.abs(minus[1, 0])**2
    print(f"\nProbabilities:")
    print(f"  P(0) = {prob_0:.4f} = 50%")
    print(f"  P(1) = {prob_1:.4f} = 50%")
    print(f"\nNote: Same probabilities as |+⟩, but different relative phase!")
    
    print_subsection("Example 3: Unequal Superposition")
    alpha = 1/2
    beta = np.sqrt(3)/2
    psi = alpha * zero + beta * one
    
    print(f"\n|ψ⟩ = (1/2)|0⟩ + (√3/2)|1⟩")
    print(f"    = {psi.T[0]}")
    
    prob_0 = np.abs(psi[0, 0])**2
    prob_1 = np.abs(psi[1, 0])**2
    print(f"\nProbabilities:")
    print(f"  P(0) = (1/2)² = {prob_0:.4f} = 25%")
    print(f"  P(1) = (√3/2)² = {prob_1:.4f} = 75%")
    
    print_subsection("Example 4: Complex Superposition")
    psi_complex = (1/np.sqrt(2)) * zero + (1j/np.sqrt(2)) * one
    
    print(f"\n|ψ⟩ = (1/√2)|0⟩ + (i/√2)|1⟩")
    print(f"    = {psi_complex.T[0]}")
    
    prob_0 = np.abs(psi_complex[0, 0])**2
    prob_1 = np.abs(psi_complex[1, 0])**2
    print(f"\nProbabilities:")
    print(f"  P(0) = {prob_0:.4f} = 50%")
    print(f"  P(1) = {prob_1:.4f} = 50%")
    
    phase_0 = np.angle(psi_complex[0, 0])
    phase_1 = np.angle(psi_complex[1, 0])
    print(f"\nPhases:")
    print(f"  arg(α) = {phase_0:.4f} rad = 0°")
    print(f"  arg(β) = {phase_1:.4f} rad = 90°")
    print(f"  Relative phase: π/2")

def multi_qubit_superposition():
    """Demonstrate multi-qubit superposition"""
    print_section_header("MULTI-QUBIT SUPERPOSITION")
    
    zero = np.array([[1], [0]], dtype=complex)
    one = np.array([[0], [1]], dtype=complex)
    
    print_subsection("Two-Qubit Bell State")
    print("\nBell state: entangled superposition of two qubits")
    
    state_00 = np.kron(zero, zero)
    state_01 = np.kron(zero, one)
    state_10 = np.kron(one, zero)
    state_11 = np.kron(one, one)
    
    print(f"\n|00⟩ = {state_00.T[0]}")
    print(f"|01⟩ = {state_01.T[0]}")
    print(f"|10⟩ = {state_10.T[0]}")
    print(f"|11⟩ = {state_11.T[0]}")
    
    bell = (1/np.sqrt(2)) * state_00 + (1/np.sqrt(2)) * state_11
    
    print(f"\n|Φ⁺⟩ = (1/√2)|00⟩ + (1/√2)|11⟩")
    print(f"     = {bell.T[0]}")
    
    print("\nProbabilities:")
    for i, basis_name in enumerate(['|00⟩', '|01⟩', '|10⟩', '|11⟩']):
        prob = np.abs(bell[i, 0])**2
        print(f"  P({basis_name}) = {prob:.4f} = {prob*100:.1f}%")
    
    print("\nThis state is entangled!")
    print("  - 50% chance of measuring |00⟩")
    print("  - 50% chance of measuring |11⟩")
    print("  - 0% chance of measuring |01⟩ or |10⟩")
    print("  - Cannot be written as a product of single-qubit states")
    
    print_subsection("Three-Qubit GHZ State")
    print("\nGHZ state: entangled superposition of three qubits")
    
    state_000 = np.kron(np.kron(zero, zero), zero)
    state_111 = np.kron(np.kron(one, one), one)
    
    ghz = (1/np.sqrt(2)) * state_000 + (1/np.sqrt(2)) * state_111
    
    print(f"\n|GHZ⟩ = (1/√2)|000⟩ + (1/√2)|111⟩")
    print(f"      = {ghz.T[0]}")
    
    print("\nThis is a maximally entangled three-qubit state!")
    print("  - 50% chance of measuring |000⟩ (all zeros)")
    print("  - 50% chance of measuring |111⟩ (all ones)")
    print("  - 0% chance of any other outcome")

def superposition_measurement_simulation():
    """Simulate measurement of superposition states"""
    print_section_header("MEASUREMENT SIMULATION")
    
    print("\nSimulating repeated measurements of superposition states")
    
    zero = np.array([[1], [0]], dtype=complex)
    one = np.array([[0], [1]], dtype=complex)
    plus = (1/np.sqrt(2)) * zero + (1/np.sqrt(2)) * one
    
    print_subsection("State: |+⟩ = (1/√2)|0⟩ + (1/√2)|1⟩")
    print(f"\n|+⟩ = {plus.T[0]}")
    
    prob_0 = np.abs(plus[0, 0])**2
    prob_1 = np.abs(plus[1, 0])**2
    
    print(f"\nTheoretical probabilities:")
    print(f"  P(0) = {prob_0:.4f}")
    print(f"  P(1) = {prob_1:.4f}")
    
    n_shots = 1000
    outcomes = np.random.choice([0, 1], size=n_shots, p=[prob_0, prob_1])
    
    count_0 = np.sum(outcomes == 0)
    count_1 = np.sum(outcomes == 1)
    
    print(f"\nSimulation with {n_shots} measurements:")
    print(f"  Measured 0: {count_0} times ({count_0/n_shots:.3f})")
    print(f"  Measured 1: {count_1} times ({count_1/n_shots:.3f})")
    
    print_subsection("State: |ψ⟩ = (1/2)|0⟩ + (√3/2)|1⟩")
    alpha = 1/2
    beta = np.sqrt(3)/2
    psi = alpha * zero + beta * one
    
    print(f"\n|ψ⟩ = {psi.T[0]}")
    
    prob_0 = np.abs(psi[0, 0])**2
    prob_1 = np.abs(psi[1, 0])**2
    
    print(f"\nTheoretical probabilities:")
    print(f"  P(0) = {prob_0:.4f}")
    print(f"  P(1) = {prob_1:.4f}")
    
    outcomes = np.random.choice([0, 1], size=n_shots, p=[prob_0, prob_1])
    
    count_0 = np.sum(outcomes == 0)
    count_1 = np.sum(outcomes == 1)
    
    print(f"\nSimulation with {n_shots} measurements:")
    print(f"  Measured 0: {count_0} times ({count_0/n_shots:.3f})")
    print(f"  Measured 1: {count_1} times ({count_1/n_shots:.3f})")

def normalization_after_addition():
    """Demonstrate normalization after vector addition"""
    print_section_header("NORMALIZATION AFTER ADDITION")
    
    print("\nWhen adding quantum states, the result may need normalization")
    
    psi1 = np.array([[0.6], [0.8]], dtype=complex)
    psi2 = np.array([[0.8], [0.6]], dtype=complex)
    
    print_subsection("Example: Adding Two Normalized States")
    print(f"\n|ψ₁⟩ = {psi1.T[0]}")
    print(f"Norm: {np.linalg.norm(psi1):.4f} ✓")
    
    print(f"\n|ψ₂⟩ = {psi2.T[0]}")
    print(f"Norm: {np.linalg.norm(psi2):.4f} ✓")
    
    psi_sum = psi1 + psi2
    norm_sum = np.linalg.norm(psi_sum)
    
    print(f"\n|ψ₁⟩ + |ψ₂⟩ = {psi_sum.T[0]}")
    print(f"Norm: {norm_sum:.4f} ✗ (not normalized!)")
    
    psi_normalized = psi_sum / norm_sum
    norm_check = np.linalg.norm(psi_normalized)
    
    print(f"\nNormalized:")
    print(f"|ψ⟩ = (|ψ₁⟩ + |ψ₂⟩) / ||ψ₁⟩ + |ψ₂⟩||")
    print(f"    = {psi_normalized.T[0]}")
    print(f"Norm: {norm_check:.4f} ✓")
    
    prob_0 = np.abs(psi_normalized[0, 0])**2
    prob_1 = np.abs(psi_normalized[1, 0])**2
    
    print(f"\nProbabilities:")
    print(f"  P(0) = {prob_0:.4f}")
    print(f"  P(1) = {prob_1:.4f}")
    print(f"  Total: {prob_0 + prob_1:.4f} ✓")

def visualize_vector_operations():
    """Create comprehensive visualizations"""
    print_section_header("CREATING VISUALIZATIONS")
    
    fig = plt.figure(figsize=(16, 12))
    
    ax1 = fig.add_subplot(2, 3, 1)
    v = np.array([3, 4])
    w = np.array([1, -2])
    sum_vec = v + w
    
    ax1.arrow(0, 0, v[0], v[1], head_width=0.3, head_length=0.3,
             fc='blue', ec='blue', linewidth=2, label='v', alpha=0.7)
    ax1.arrow(0, 0, w[0], w[1], head_width=0.3, head_length=0.3,
             fc='red', ec='red', linewidth=2, label='w', alpha=0.7)
    ax1.arrow(0, 0, sum_vec[0], sum_vec[1], head_width=0.3, head_length=0.3,
             fc='green', ec='green', linewidth=3, label='v+w', alpha=0.7)
    
    ax1.arrow(v[0], v[1], w[0], w[1], head_width=0.2, head_length=0.2,
             fc='red', ec='red', linewidth=1, linestyle='--', alpha=0.5)
    
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Vector Addition (Tip-to-Tail)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.set_xlim(-1, 5)
    ax1.set_ylim(-3, 5)
    
    ax2 = fig.add_subplot(2, 3, 2)
    
    ax2.arrow(0, 0, v[0], v[1], head_width=0.3, head_length=0.3,
             fc='blue', ec='blue', linewidth=2, label='v', alpha=0.7)
    ax2.arrow(0, 0, w[0], w[1], head_width=0.3, head_length=0.3,
             fc='red', ec='red', linewidth=2, label='w', alpha=0.7)
    ax2.arrow(0, 0, sum_vec[0], sum_vec[1], head_width=0.3, head_length=0.3,
             fc='green', ec='green', linewidth=3, label='v+w', alpha=0.7)
    
    parallelogram = plt.Polygon([[0, 0], v, sum_vec, w], 
                                fill=False, edgecolor='gray', linewidth=1, linestyle='--')
    ax2.add_patch(parallelogram)
    
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('Vector Addition (Parallelogram)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.set_xlim(-1, 5)
    ax2.set_ylim(-3, 5)
    
    ax3 = fig.add_subplot(2, 3, 3)
    diff_vec = v - w
    
    ax3.arrow(0, 0, v[0], v[1], head_width=0.3, head_length=0.3,
             fc='blue', ec='blue', linewidth=2, label='v', alpha=0.7)
    ax3.arrow(0, 0, w[0], w[1], head_width=0.3, head_length=0.3,
             fc='red', ec='red', linewidth=2, label='w', alpha=0.7)
    ax3.arrow(0, 0, diff_vec[0], diff_vec[1], head_width=0.3, head_length=0.3,
             fc='orange', ec='orange', linewidth=3, label='v-w', alpha=0.7)
    
    neg_w = -w
    ax3.arrow(0, 0, neg_w[0], neg_w[1], head_width=0.2, head_length=0.2,
             fc='pink', ec='pink', linewidth=1, linestyle='--', label='-w', alpha=0.5)
    
    ax3.axhline(y=0, color='k', linewidth=0.5)
    ax3.axvline(x=0, color='k', linewidth=0.5)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('y', fontsize=12)
    ax3.set_title('Vector Subtraction', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.set_aspect('equal')
    ax3.set_xlim(-2, 5)
    ax3.set_ylim(-3, 7)
    
    ax4 = fig.add_subplot(2, 3, 4)
    
    states = {
        '|+⟩': np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
        '|-⟩': np.array([1/np.sqrt(2), -1/np.sqrt(2)]),
        '|ψ₁⟩': np.array([0.6, 0.8]),
        '|ψ₂⟩': np.array([0.8, 0.6])
    }
    
    x = np.arange(len(states))
    width = 0.35
    
    prob_0 = [np.abs(state[0])**2 for state in states.values()]
    prob_1 = [np.abs(state[1])**2 for state in states.values()]
    
    ax4.bar(x - width/2, prob_0, width, label='P(0)', alpha=0.7)
    ax4.bar(x + width/2, prob_1, width, label='P(1)', alpha=0.7)
    
    ax4.set_ylabel('Probability', fontsize=12)
    ax4.set_title('Superposition State Probabilities', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(states.keys())
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    ax5 = fig.add_subplot(2, 3, 5)
    
    n_shots = 1000
    plus_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    prob_0 = np.abs(plus_state[0])**2
    prob_1 = np.abs(plus_state[1])**2
    
    outcomes = np.random.choice([0, 1], size=n_shots, p=[prob_0, prob_1])
    count_0 = np.sum(outcomes == 0)
    count_1 = np.sum(outcomes == 1)
    
    ax5.bar(['Measured 0', 'Measured 1'], [count_0, count_1], alpha=0.7, color=['blue', 'red'])
    ax5.axhline(y=n_shots/2, color='green', linestyle='--', label='Expected (500)')
    ax5.set_ylabel('Count', fontsize=12)
    ax5.set_title(f'Measurement Simulation (|+⟩, {n_shots} shots)', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    ax6 = fig.add_subplot(2, 3, 6)
    
    basis_states = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    bell_probs = [0.5, 0, 0, 0.5]  # Bell state probabilities
    
    colors = ['green' if p > 0 else 'gray' for p in bell_probs]
    ax6.bar(basis_states, bell_probs, alpha=0.7, color=colors)
    ax6.set_ylabel('Probability', fontsize=12)
    ax6.set_title('Bell State |Φ⁺⟩ Probabilities', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_ylim(0, 0.6)
    
    plt.tight_layout()
    plt.savefig('vector_addition_visualizations.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved comprehensive visualizations as 'vector_addition_visualizations.png'")
    plt.close()
    
    fig2 = plt.figure(figsize=(14, 6))
    
    ax1_3d = fig2.add_subplot(1, 2, 1, projection='3d')
    
    v_3d = np.array([2, 3, 1])
    w_3d = np.array([1, -1, 2])
    sum_3d = v_3d + w_3d
    
    ax1_3d.quiver(0, 0, 0, v_3d[0], v_3d[1], v_3d[2], 
                  color='blue', arrow_length_ratio=0.15, linewidth=2, label='v')
    ax1_3d.quiver(0, 0, 0, w_3d[0], w_3d[1], w_3d[2], 
                  color='red', arrow_length_ratio=0.15, linewidth=2, label='w')
    ax1_3d.quiver(0, 0, 0, sum_3d[0], sum_3d[1], sum_3d[2], 
                  color='green', arrow_length_ratio=0.15, linewidth=3, label='v+w')
    
    ax1_3d.set_xlabel('X', fontsize=10)
    ax1_3d.set_ylabel('Y', fontsize=10)
    ax1_3d.set_zlabel('Z', fontsize=10)
    ax1_3d.set_title('3D Vector Addition', fontsize=14, fontweight='bold')
    ax1_3d.legend()
    
    ax2_2d = fig2.add_subplot(1, 2, 2)
    
    complex_states = [
        (1/np.sqrt(2), 'α (|+⟩)', 'blue'),
        (1j/np.sqrt(2), 'β (complex)', 'red'),
        ((1+1j)/2, 'γ (mixed)', 'green')
    ]
    
    for amp, label, color in complex_states:
        ax2_2d.arrow(0, 0, amp.real, amp.imag, head_width=0.05, head_length=0.05,
                    fc=color, ec=color, linewidth=2, alpha=0.7)
        ax2_2d.plot(amp.real, amp.imag, 'o', color=color, markersize=10)
        ax2_2d.text(amp.real + 0.05, amp.imag + 0.05, label, fontsize=10, color=color)
    
    ax2_2d.axhline(y=0, color='k', linewidth=0.5)
    ax2_2d.axvline(x=0, color='k', linewidth=0.5)
    ax2_2d.grid(True, alpha=0.3)
    ax2_2d.set_xlabel('Real Part', fontsize=12)
    ax2_2d.set_ylabel('Imaginary Part', fontsize=12)
    ax2_2d.set_title('Complex Amplitudes in Complex Plane', fontsize=14, fontweight='bold')
    ax2_2d.set_aspect('equal')
    ax2_2d.set_xlim(-0.2, 0.8)
    ax2_2d.set_ylim(-0.2, 0.8)
    
    plt.tight_layout()
    plt.savefig('vector_addition_3d_complex.png', dpi=150, bbox_inches='tight')
    print("✓ Saved 3D and complex visualizations as 'vector_addition_3d_complex.png'")
    plt.close()

def main():
    """Main function to run all demonstrations"""
    print("\n" + "➕" * 40)
    print("VECTOR ADDITION AND SUBTRACTION")
    print("Comprehensive Python Demonstrations")
    print("➕" * 40)
    
    basic_vector_addition()
    basic_vector_subtraction()
    addition_properties()
    geometric_interpretation()
    quantum_superposition()
    multi_qubit_superposition()
    superposition_measurement_simulation()
    normalization_after_addition()
    visualize_vector_operations()
    
    print_section_header("KEY TAKEAWAYS")
    print("\n1. Vector Addition:")
    print("   - Add corresponding components: v + w = [v₁+w₁, v₂+w₂, ...]ᵀ")
    print("   - Commutative: v + w = w + v")
    print("   - Associative: (u + v) + w = u + (v + w)")
    
    print("\n2. Vector Subtraction:")
    print("   - Subtract corresponding components: v - w = [v₁-w₁, v₂-w₂, ...]ᵀ")
    print("   - Equivalent to adding negative: v - w = v + (-w)")
    
    print("\n3. Geometric Interpretation:")
    print("   - Tip-to-tail method: place tail of w at tip of v")
    print("   - Parallelogram method: complete parallelogram, diagonal is sum")
    print("   - Triangle inequality: |v + w| ≤ |v| + |w|")
    
    print("\n4. Quantum Superposition:")
    print("   - Vector addition represents superposition in quantum computing")
    print("   - |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ (linear combination of states)")
    print("   - Enables quantum parallelism and interference")
    
    print("\n5. Multi-Qubit Systems:")
    print("   - Bell states: entangled superpositions")
    print("   - GHZ states: multi-qubit entanglement")
    print("   - Cannot be decomposed into product states")
    
    print("\n6. Measurement:")
    print("   - Superposition collapses to definite state")
    print("   - Probabilities determined by |amplitude|²")
    print("   - Repeated measurements follow probability distribution")
    
    print("\n7. Normalization:")
    print("   - Sum of quantum states may need renormalization")
    print("   - Ensure ||ψ⟩|| = 1 for valid quantum state")
    
    print("\n" + "=" * 80)
    print("All demonstrations completed successfully!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
