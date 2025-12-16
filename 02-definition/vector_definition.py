"""
Mathematical Definition of Vectors
Comprehensive Python script demonstrating vector structure, notation, and properties
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

def basic_vector_definition():
    """Demonstrate basic vector definition and structure"""
    print_section_header("BASIC VECTOR DEFINITION")
    
    print("\nA vector is an ordered collection of numbers:")
    print("  v = [v₁, v₂, ..., vₙ]ᵀ")
    print("\nWhere:")
    print("  - v ∈ ℝⁿ for real vectors")
    print("  - v ∈ ℂⁿ for complex vectors (quantum)")
    print("  - n is the dimension")
    print("  - v₁, v₂, ..., vₙ are the components")
    
    print_subsection("Example 1: 2D Real Vector")
    v1 = np.array([[3], [4]], dtype=float)
    print(f"\nv₁ = {v1.T[0]}")
    print(f"Dimension: {v1.shape[0]}")
    print(f"Components: v₁ = {v1[0, 0]}, v₂ = {v1[1, 0]}")
    print(f"Type: Real vector (v₁ ∈ ℝ²)")
    
    print_subsection("Example 2: 3D Real Vector")
    v2 = np.array([[1], [2], [3]], dtype=float)
    print(f"\nv₂ = {v2.T[0]}")
    print(f"Dimension: {v2.shape[0]}")
    print(f"Components: v₁ = {v2[0, 0]}, v₂ = {v2[1, 0]}, v₃ = {v2[2, 0]}")
    print(f"Type: Real vector (v₂ ∈ ℝ³)")
    
    print_subsection("Example 3: 2D Complex Vector (Quantum)")
    v3 = np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]], dtype=complex)
    print(f"\nv₃ = {v3.T[0]}")
    print(f"Dimension: {v3.shape[0]}")
    print(f"Components: v₁ = {v3[0, 0]:.4f}, v₂ = {v3[1, 0]:.4f}")
    print(f"Type: Complex vector (v₃ ∈ ℂ²)")
    print(f"This represents a quantum state!")
    
    print_subsection("Example 4: Complex Vector with Imaginary Components")
    v4 = np.array([[0.6 + 0.2j], [0.5 - 0.5j]], dtype=complex)
    print(f"\nv₄ = {v4.T[0]}")
    print(f"Dimension: {v4.shape[0]}")
    print(f"Component 1: {v4[0, 0]}")
    print(f"  Real part: {v4[0, 0].real:.2f}")
    print(f"  Imaginary part: {v4[0, 0].imag:.2f}")
    print(f"Component 2: {v4[1, 0]}")
    print(f"  Real part: {v4[1, 0].real:.2f}")
    print(f"  Imaginary part: {v4[1, 0].imag:.2f}")

def column_vector_notation():
    """Demonstrate column vector notation"""
    print_section_header("COLUMN VECTOR NOTATION")
    
    print("\nVectors are typically written as column vectors:")
    print("     ⎡v₁⎤")
    print("  v = ⎢v₂⎥")
    print("     ⎢⋮ ⎥")
    print("     ⎣vₙ⎦")
    
    print_subsection("Creating Column Vectors in Python")
    
    print("\nMethod 1: Direct array creation")
    v1 = np.array([[3], [4]], dtype=float)
    print(f"v₁ = \n{v1}")
    print(f"Shape: {v1.shape} (2 rows, 1 column)")
    
    print("\nMethod 2: Reshape from 1D array")
    v2 = np.array([3, 4]).reshape(-1, 1)
    print(f"v₂ = \n{v2}")
    print(f"Shape: {v2.shape}")
    
    print("\nMethod 3: Using column_stack")
    v3 = np.column_stack([3, 4]).T
    print(f"v₃ = \n{v3}")
    print(f"Shape: {v3.shape}")
    
    print("\n✓ All three methods create the same column vector!")
    print(f"v₁ == v₂: {np.array_equal(v1, v2)}")
    print(f"v₂ == v₃: {np.array_equal(v2, v3)}")

def quantum_ket_notation():
    """Demonstrate quantum ket notation"""
    print_section_header("QUANTUM KET NOTATION (DIRAC NOTATION)")
    
    print("\nIn quantum computing, we use Dirac notation:")
    print("  |ψ⟩ represents a quantum state (ket)")
    print("  ⟨ψ| represents the conjugate transpose (bra)")
    
    print_subsection("Basic Ket States")
    
    print("\nState |0⟩ (computational basis state):")
    zero = np.array([[1], [0]], dtype=complex)
    print(f"|0⟩ = \n{zero}")
    print("Represents the 'zero' state")
    print("100% probability of measuring 0")
    
    print("\nState |1⟩ (computational basis state):")
    one = np.array([[0], [1]], dtype=complex)
    print(f"|1⟩ = \n{one}")
    print("Represents the 'one' state")
    print("100% probability of measuring 1")
    
    print("\nState |+⟩ (superposition state):")
    plus = (1/np.sqrt(2)) * np.array([[1], [1]], dtype=complex)
    print(f"|+⟩ = (1/√2)(|0⟩ + |1⟩) = \n{plus}")
    print("Equal superposition of |0⟩ and |1⟩")
    print("50% probability of measuring 0 or 1")
    
    print("\nState |-⟩ (superposition state):")
    minus = (1/np.sqrt(2)) * np.array([[1], [-1]], dtype=complex)
    print(f"|-⟩ = (1/√2)(|0⟩ - |1⟩) = \n{minus}")
    print("Superposition with relative phase")
    print("50% probability of measuring 0 or 1")
    
    print_subsection("General Qubit State")
    print("\nA general qubit state:")
    print("  |ψ⟩ = α|0⟩ + β|1⟩")
    print("Where α, β ∈ ℂ and |α|² + |β|² = 1")
    
    alpha = 0.6
    beta = 0.8
    psi = alpha * zero + beta * one
    print(f"\nExample: α = {alpha}, β = {beta}")
    print(f"|ψ⟩ = {alpha}|0⟩ + {beta}|1⟩ = \n{psi}")
    print(f"Normalization check: |α|² + |β|² = {alpha**2 + beta**2:.4f}")

def dimension_and_components():
    """Demonstrate dimension and components"""
    print_section_header("DIMENSION AND COMPONENTS")
    
    print("\nThe dimension n is the number of components in the vector")
    
    print_subsection("Single Qubit System")
    print("\nA single qubit exists in a 2-dimensional Hilbert space")
    print("Dimension: n = 2")
    print("Basis states: |0⟩, |1⟩")
    
    psi_1 = np.array([[0.6], [0.8]], dtype=complex)
    print(f"\nExample state: |ψ⟩ = \n{psi_1}")
    print(f"Dimension: {psi_1.shape[0]}")
    print(f"Component 1 (α): {psi_1[0, 0]}")
    print(f"Component 2 (β): {psi_1[1, 0]}")
    
    print_subsection("Two Qubit System")
    print("\nA two-qubit system exists in a 4-dimensional Hilbert space")
    print("Dimension: n = 4 = 2²")
    print("Basis states: |00⟩, |01⟩, |10⟩, |11⟩")
    
    psi_2 = np.array([[0.5], [0.5], [0.5], [0.5]], dtype=complex)
    print(f"\nExample state: |ψ⟩ = \n{psi_2}")
    print(f"Dimension: {psi_2.shape[0]}")
    print(f"Component 1 (α₀₀): {psi_2[0, 0]}")
    print(f"Component 2 (α₀₁): {psi_2[1, 0]}")
    print(f"Component 3 (α₁₀): {psi_2[2, 0]}")
    print(f"Component 4 (α₁₁): {psi_2[3, 0]}")
    
    print_subsection("Three Qubit System")
    print("\nA three-qubit system exists in an 8-dimensional Hilbert space")
    print("Dimension: n = 8 = 2³")
    print("Basis states: |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩")
    
    psi_3 = np.ones((8, 1), dtype=complex) / np.sqrt(8)
    print(f"\nExample state (equal superposition): |ψ⟩ = \n{psi_3}")
    print(f"Dimension: {psi_3.shape[0]}")
    print(f"Each component has amplitude: {psi_3[0, 0]:.4f}")
    
    print_subsection("General n-Qubit System")
    print("\nFor n qubits:")
    print("  Dimension: 2ⁿ")
    print("  Number of basis states: 2ⁿ")
    print("  Number of complex amplitudes: 2ⁿ")
    
    for n in range(1, 6):
        dim = 2**n
        print(f"  {n} qubit(s): {dim} dimensions")

def classical_vs_quantum_vectors():
    """Compare classical and quantum vectors"""
    print_section_header("CLASSICAL VS QUANTUM VECTORS")
    
    print_subsection("Classical Vectors (Real)")
    print("\nClassical vectors have real components:")
    print("  v ∈ ℝⁿ")
    print("  Used in classical physics (velocity, force, etc.)")
    
    print("\nExample 1: Velocity Vector")
    v_velocity = np.array([[3], [4]], dtype=float)
    print(f"v = {v_velocity.T[0]} m/s")
    print(f"Components: vₓ = {v_velocity[0, 0]} m/s, vᵧ = {v_velocity[1, 0]} m/s")
    magnitude = np.linalg.norm(v_velocity)
    print(f"Magnitude: |v| = {magnitude:.2f} m/s")
    angle = np.arctan2(v_velocity[1, 0], v_velocity[0, 0]) * 180 / np.pi
    print(f"Direction: θ = {angle:.2f}°")
    
    print("\nExample 2: Force Vector")
    v_force = np.array([[10], [-5]], dtype=float)
    print(f"F = {v_force.T[0]} N")
    print(f"Components: Fₓ = {v_force[0, 0]} N, Fᵧ = {v_force[1, 0]} N")
    magnitude = np.linalg.norm(v_force)
    print(f"Magnitude: |F| = {magnitude:.2f} N")
    
    print_subsection("Quantum Vectors (Complex)")
    print("\nQuantum vectors have complex components:")
    print("  |ψ⟩ ∈ ℂⁿ")
    print("  Used in quantum mechanics (quantum states)")
    
    print("\nExample 1: Qubit State")
    psi = np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]], dtype=complex)
    print(f"|ψ⟩ = {psi.T[0]}")
    print(f"Components: α = {psi[0, 0]:.4f}, β = {psi[1, 0]:.4f}")
    norm = np.linalg.norm(psi)
    print(f"Norm: ||ψ⟩|| = {norm:.4f}")
    prob_0 = np.abs(psi[0, 0])**2
    prob_1 = np.abs(psi[1, 0])**2
    print(f"Probabilities: P(0) = {prob_0:.4f}, P(1) = {prob_1:.4f}")
    
    print("\nExample 2: Qubit State with Phase")
    psi_phase = np.array([[1/np.sqrt(2)], [1j/np.sqrt(2)]], dtype=complex)
    print(f"|ψ⟩ = {psi_phase.T[0]}")
    print(f"Component 1: α = {psi_phase[0, 0]}")
    print(f"  Magnitude: |α| = {np.abs(psi_phase[0, 0]):.4f}")
    print(f"  Phase: arg(α) = {np.angle(psi_phase[0, 0]):.4f} rad")
    print(f"Component 2: β = {psi_phase[1, 0]}")
    print(f"  Magnitude: |β| = {np.abs(psi_phase[1, 0]):.4f}")
    print(f"  Phase: arg(β) = {np.angle(psi_phase[1, 0]):.4f} rad = π/2")
    
    print_subsection("Key Differences")
    print("\n1. Components:")
    print("   Classical: Real numbers (ℝ)")
    print("   Quantum: Complex numbers (ℂ)")
    print("\n2. Interpretation:")
    print("   Classical: Physical quantities (position, velocity, force)")
    print("   Quantum: Probability amplitudes")
    print("\n3. Measurement:")
    print("   Classical: Direct measurement of components")
    print("   Quantum: Probabilistic measurement (|amplitude|²)")
    print("\n4. Normalization:")
    print("   Classical: Arbitrary magnitude")
    print("   Quantum: Must be normalized (||ψ⟩|| = 1)")

def probability_amplitudes():
    """Demonstrate probability amplitudes and Born rule"""
    print_section_header("PROBABILITY AMPLITUDES AND BORN RULE")
    
    print("\nIn quantum mechanics, vector components are probability amplitudes")
    print("The Born Rule states:")
    print("  P(outcome) = |amplitude|²")
    
    print_subsection("Example 1: Equal Superposition")
    psi1 = np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]], dtype=complex)
    print(f"\n|ψ⟩ = {psi1.T[0]}")
    print(f"α = {psi1[0, 0]:.4f}, β = {psi1[1, 0]:.4f}")
    prob_0 = np.abs(psi1[0, 0])**2
    prob_1 = np.abs(psi1[1, 0])**2
    print(f"\nProbabilities:")
    print(f"  P(0) = |α|² = |{psi1[0, 0]:.4f}|² = {prob_0:.4f} = 50%")
    print(f"  P(1) = |β|² = |{psi1[1, 0]:.4f}|² = {prob_1:.4f} = 50%")
    print(f"  Total: {prob_0 + prob_1:.4f} = 100% ✓")
    
    print_subsection("Example 2: Unequal Superposition")
    psi2 = np.array([[0.6], [0.8]], dtype=complex)
    print(f"\n|ψ⟩ = {psi2.T[0]}")
    print(f"α = {psi2[0, 0]:.4f}, β = {psi2[1, 0]:.4f}")
    prob_0 = np.abs(psi2[0, 0])**2
    prob_1 = np.abs(psi2[1, 0])**2
    print(f"\nProbabilities:")
    print(f"  P(0) = |α|² = |{psi2[0, 0]:.4f}|² = {prob_0:.4f} = 36%")
    print(f"  P(1) = |β|² = |{psi2[1, 0]:.4f}|² = {prob_1:.4f} = 64%")
    print(f"  Total: {prob_0 + prob_1:.4f} = 100% ✓")
    
    print_subsection("Example 3: Complex Amplitudes")
    psi3 = np.array([[0.6 + 0.2j], [0.5 - 0.5j]], dtype=complex)
    print(f"\n|ψ⟩ = {psi3.T[0]}")
    print(f"α = {psi3[0, 0]}, β = {psi3[1, 0]}")
    prob_0 = np.abs(psi3[0, 0])**2
    prob_1 = np.abs(psi3[1, 0])**2
    print(f"\nProbabilities:")
    print(f"  P(0) = |α|² = |{psi3[0, 0]}|² = {prob_0:.4f}")
    print(f"  P(1) = |β|² = |{psi3[1, 0]}|² = {prob_1:.4f}")
    print(f"  Total: {prob_0 + prob_1:.4f}")
    
    if not np.isclose(prob_0 + prob_1, 1.0):
        print(f"\n⚠ State is not normalized! Normalizing...")
        psi3_normalized = psi3 / np.linalg.norm(psi3)
        print(f"|ψ⟩_normalized = {psi3_normalized.T[0]}")
        prob_0_norm = np.abs(psi3_normalized[0, 0])**2
        prob_1_norm = np.abs(psi3_normalized[1, 0])**2
        print(f"  P(0) = {prob_0_norm:.4f}")
        print(f"  P(1) = {prob_1_norm:.4f}")
        print(f"  Total: {prob_0_norm + prob_1_norm:.4f} = 100% ✓")

def normalization_requirement():
    """Demonstrate normalization requirement for quantum states"""
    print_section_header("NORMALIZATION REQUIREMENT")
    
    print("\nQuantum state vectors must be normalized:")
    print("  ||ψ⟩|| = √(|α|² + |β|²) = 1")
    print("\nThis ensures total probability = 1")
    
    print_subsection("Example 1: Already Normalized")
    psi1 = np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]], dtype=complex)
    norm1 = np.linalg.norm(psi1)
    print(f"\n|ψ⟩ = {psi1.T[0]}")
    print(f"Norm: ||ψ⟩|| = {norm1:.6f}")
    print(f"Is normalized: {np.isclose(norm1, 1.0)} ✓")
    
    print_subsection("Example 2: Unnormalized State")
    psi2 = np.array([[3], [4]], dtype=complex)
    norm2 = np.linalg.norm(psi2)
    print(f"\n|ψ⟩ = {psi2.T[0]}")
    print(f"Norm: ||ψ⟩|| = {norm2:.6f}")
    print(f"Is normalized: {np.isclose(norm2, 1.0)} ✗")
    
    print("\nNormalizing the state:")
    psi2_normalized = psi2 / norm2
    norm2_check = np.linalg.norm(psi2_normalized)
    print(f"|ψ⟩_normalized = {psi2_normalized.T[0]}")
    print(f"Norm: ||ψ⟩|| = {norm2_check:.6f}")
    print(f"Is normalized: {np.isclose(norm2_check, 1.0)} ✓")
    
    print_subsection("Example 3: Complex Unnormalized State")
    psi3 = np.array([[1 + 2j], [2 - 1j]], dtype=complex)
    norm3 = np.linalg.norm(psi3)
    print(f"\n|ψ⟩ = {psi3.T[0]}")
    print(f"Norm: ||ψ⟩|| = {norm3:.6f}")
    print(f"Is normalized: {np.isclose(norm3, 1.0)} ✗")
    
    print("\nNormalizing the state:")
    psi3_normalized = psi3 / norm3
    norm3_check = np.linalg.norm(psi3_normalized)
    print(f"|ψ⟩_normalized = {psi3_normalized.T[0]}")
    print(f"Norm: ||ψ⟩|| = {norm3_check:.6f}")
    print(f"Is normalized: {np.isclose(norm3_check, 1.0)} ✓")
    
    print_subsection("Normalization Function")
    print("\nGeneral normalization function:")
    
    def normalize_state(psi):
        """Normalize a quantum state vector"""
        norm = np.linalg.norm(psi)
        if np.isclose(norm, 0):
            raise ValueError("Cannot normalize zero vector")
        return psi / norm
    
    test_states = [
        np.array([[1], [1]], dtype=complex),
        np.array([[3], [4]], dtype=complex),
        np.array([[1+1j], [1-1j]], dtype=complex)
    ]
    
    for i, state in enumerate(test_states, 1):
        print(f"\nTest {i}:")
        print(f"  Original: {state.T[0]}")
        print(f"  Norm: {np.linalg.norm(state):.4f}")
        normalized = normalize_state(state)
        print(f"  Normalized: {normalized.T[0]}")
        print(f"  New norm: {np.linalg.norm(normalized):.6f} ✓")

def basis_states_demo():
    """Demonstrate basis states"""
    print_section_header("BASIS STATES")
    
    print("\nThe computational basis states for a single qubit:")
    
    print_subsection("State |0⟩")
    zero = np.array([[1], [0]], dtype=complex)
    print(f"\n|0⟩ = {zero.T[0]}")
    print("Properties:")
    print(f"  - Norm: {np.linalg.norm(zero):.4f}")
    print(f"  - P(0) = {np.abs(zero[0, 0])**2:.4f} = 100%")
    print(f"  - P(1) = {np.abs(zero[1, 0])**2:.4f} = 0%")
    print("  - Represents definite state '0'")
    
    print_subsection("State |1⟩")
    one = np.array([[0], [1]], dtype=complex)
    print(f"\n|1⟩ = {one.T[0]}")
    print("Properties:")
    print(f"  - Norm: {np.linalg.norm(one):.4f}")
    print(f"  - P(0) = {np.abs(one[0, 0])**2:.4f} = 0%")
    print(f"  - P(1) = {np.abs(one[1, 0])**2:.4f} = 100%")
    print("  - Represents definite state '1'")
    
    print_subsection("Orthogonality")
    inner_product = np.vdot(zero, one)
    print(f"\n⟨0|1⟩ = {inner_product}")
    print("The basis states are orthogonal!")
    
    print_subsection("Completeness")
    print("\nAny qubit state can be written as:")
    print("  |ψ⟩ = α|0⟩ + β|1⟩")
    
    alpha = 0.6
    beta = 0.8
    psi = alpha * zero + beta * one
    print(f"\nExample: α = {alpha}, β = {beta}")
    print(f"|ψ⟩ = {alpha}|0⟩ + {beta}|1⟩")
    print(f"    = {psi.T[0]}")
    
    print("\nVerification:")
    print(f"  α|0⟩ = {(alpha * zero).T[0]}")
    print(f"  β|1⟩ = {(beta * one).T[0]}")
    print(f"  Sum = {psi.T[0]}")

def multi_qubit_dimensions():
    """Demonstrate multi-qubit system dimensions"""
    print_section_header("MULTI-QUBIT SYSTEM DIMENSIONS")
    
    print("\nFor n qubits, the state space has dimension 2ⁿ")
    
    print_subsection("1-Qubit System")
    print("\nDimension: 2¹ = 2")
    print("Basis states: |0⟩, |1⟩")
    zero = np.array([[1], [0]], dtype=complex)
    one = np.array([[0], [1]], dtype=complex)
    print(f"|0⟩ = {zero.T[0]}")
    print(f"|1⟩ = {one.T[0]}")
    
    print_subsection("2-Qubit System")
    print("\nDimension: 2² = 4")
    print("Basis states: |00⟩, |01⟩, |10⟩, |11⟩")
    
    state_00 = np.kron(zero, zero)
    state_01 = np.kron(zero, one)
    state_10 = np.kron(one, zero)
    state_11 = np.kron(one, one)
    
    print(f"|00⟩ = {state_00.T[0]}")
    print(f"|01⟩ = {state_01.T[0]}")
    print(f"|10⟩ = {state_10.T[0]}")
    print(f"|11⟩ = {state_11.T[0]}")
    
    print("\nExample: Bell state")
    bell = (1/np.sqrt(2)) * (state_00 + state_11)
    print(f"|Φ⁺⟩ = (1/√2)(|00⟩ + |11⟩)")
    print(f"     = {bell.T[0]}")
    
    print_subsection("3-Qubit System")
    print("\nDimension: 2³ = 8")
    print("Basis states: |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩")
    
    state_000 = np.kron(np.kron(zero, zero), zero)
    state_111 = np.kron(np.kron(one, one), one)
    ghz = (1/np.sqrt(2)) * (state_000 + state_111)
    print(f"\nExample: GHZ state")
    print(f"|GHZ⟩ = (1/√2)(|000⟩ + |111⟩)")
    print(f"      = {ghz.T[0]}")
    
    print_subsection("Dimension Growth")
    print("\nDimension grows exponentially with number of qubits:")
    for n in range(1, 11):
        dim = 2**n
        print(f"  {n:2d} qubit(s): {dim:4d} dimensions")
    
    print("\nThis exponential growth is the source of quantum advantage!")
    print("Classical simulation becomes intractable for large n")

def visualize_vectors():
    """Create comprehensive visualizations"""
    print_section_header("CREATING VISUALIZATIONS")
    
    fig = plt.figure(figsize=(16, 12))
    
    ax1 = fig.add_subplot(2, 3, 1)
    vectors_2d = [
        ([3, 4], 'blue', 'v₁ = [3, 4]'),
        ([1, -2], 'red', 'v₂ = [1, -2]'),
        ([-2, 3], 'green', 'v₃ = [-2, 3]')
    ]
    
    for vec, color, label in vectors_2d:
        ax1.arrow(0, 0, vec[0], vec[1], head_width=0.3, head_length=0.3,
                 fc=color, ec=color, linewidth=2, alpha=0.7, label=label)
        ax1.plot(vec[0], vec[1], 'o', color=color, markersize=8)
    
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('2D Real Vectors', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.set_xlim(-4, 5)
    ax1.set_ylim(-4, 5)
    
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    vectors_3d = [
        ([1, 2, 3], 'blue'),
        ([2, -1, 2], 'red'),
        ([-1, 2, 1], 'green')
    ]
    
    for vec, color in vectors_3d:
        ax2.quiver(0, 0, 0, vec[0], vec[1], vec[2], 
                  color=color, arrow_length_ratio=0.15, linewidth=2)
    
    ax2.set_xlabel('X', fontsize=10)
    ax2.set_ylabel('Y', fontsize=10)
    ax2.set_zlabel('Z', fontsize=10)
    ax2.set_title('3D Real Vectors', fontsize=14, fontweight='bold')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_zlim(-1, 4)
    
    ax3 = fig.add_subplot(2, 3, 3)
    states = {
        '|0⟩': np.array([1, 0]),
        '|1⟩': np.array([0, 1]),
        '|+⟩': np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
        '|-⟩': np.array([1/np.sqrt(2), -1/np.sqrt(2)])
    }
    
    x = np.arange(len(states))
    width = 0.35
    
    for i, (name, state) in enumerate(states.items()):
        ax3.bar([i*2, i*2+width], [np.abs(state[0])**2, np.abs(state[1])**2],
               width, label=name, alpha=0.7)
    
    ax3.set_ylabel('Probability', fontsize=12)
    ax3.set_title('Quantum State Probabilities', fontsize=14, fontweight='bold')
    ax3.set_xticks([i*2+width/2 for i in range(len(states))])
    ax3.set_xticklabels(states.keys())
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax4 = fig.add_subplot(2, 3, 4)
    complex_states = {
        'α': 0.6 + 0.2j,
        'β': 0.5 - 0.5j,
        'γ': -0.3 + 0.4j
    }
    
    names = list(complex_states.keys())
    real_parts = [c.real for c in complex_states.values()]
    imag_parts = [c.imag for c in complex_states.values()]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax4.bar(x - width/2, real_parts, width, label='Real Part', alpha=0.7)
    ax4.bar(x + width/2, imag_parts, width, label='Imaginary Part', alpha=0.7)
    
    ax4.set_ylabel('Value', fontsize=12)
    ax4.set_title('Complex Amplitude Components', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(names)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='k', linewidth=0.5)
    
    ax5 = fig.add_subplot(2, 3, 5)
    n_qubits = np.arange(1, 11)
    dimensions = 2**n_qubits
    
    ax5.semilogy(n_qubits, dimensions, 'bo-', linewidth=2, markersize=8)
    ax5.set_xlabel('Number of Qubits', fontsize=12)
    ax5.set_ylabel('Dimension (log scale)', fontsize=12)
    ax5.set_title('Hilbert Space Dimension Growth', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    for i, (n, d) in enumerate(zip(n_qubits, dimensions)):
        if i % 2 == 0:
            ax5.text(n, d, f'{d}', ha='center', va='bottom', fontsize=9)
    
    ax6 = fig.add_subplot(2, 3, 6)
    
    unnorm = np.array([3, 4])
    norm_value = np.linalg.norm(unnorm)
    normalized = unnorm / norm_value
    
    categories = ['Component 1', 'Component 2', 'Norm']
    unnorm_values = [unnorm[0], unnorm[1], norm_value]
    norm_values = [normalized[0], normalized[1], np.linalg.norm(normalized)]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax6.bar(x - width/2, unnorm_values, width, label='Unnormalized', alpha=0.7)
    ax6.bar(x + width/2, norm_values, width, label='Normalized', alpha=0.7)
    
    ax6.set_ylabel('Value', fontsize=12)
    ax6.set_title('Normalization Effect', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(categories)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.axhline(y=1, color='r', linewidth=1, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('vector_definition_visualizations.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved comprehensive visualizations as 'vector_definition_visualizations.png'")
    plt.close()
    
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    complex_amps = [
        (0.6 + 0.2j, 'α', 'blue'),
        (0.5 - 0.5j, 'β', 'red'),
        (1/np.sqrt(2) + 1j/np.sqrt(2), 'γ', 'green')
    ]
    
    for amp, label, color in complex_amps:
        ax1.arrow(0, 0, amp.real, amp.imag, head_width=0.05, head_length=0.05,
                 fc=color, ec=color, linewidth=2, alpha=0.7)
        ax1.plot(amp.real, amp.imag, 'o', color=color, markersize=10)
        ax1.text(amp.real + 0.05, amp.imag + 0.05, label, fontsize=12, color=color)
    
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Real Part', fontsize=12)
    ax1.set_ylabel('Imaginary Part', fontsize=12)
    ax1.set_title('Complex Amplitudes in Complex Plane', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.set_xlim(-0.2, 1)
    ax1.set_ylim(-0.8, 1)
    
    amplitudes = [amp for amp, _, _ in complex_amps]
    labels = [label for _, label, _ in complex_amps]
    magnitudes = [np.abs(amp) for amp in amplitudes]
    phases = [np.angle(amp) for amp in amplitudes]
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax2_twin = ax2.twinx()
    
    bars1 = ax2.bar(x - width/2, magnitudes, width, label='Magnitude', alpha=0.7, color='blue')
    bars2 = ax2_twin.bar(x + width/2, phases, width, label='Phase (rad)', alpha=0.7, color='red')
    
    ax2.set_xlabel('Amplitude', fontsize=12)
    ax2.set_ylabel('Magnitude', fontsize=12, color='blue')
    ax2_twin.set_ylabel('Phase (radians)', fontsize=12, color='red')
    ax2.set_title('Magnitude and Phase of Complex Amplitudes', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    ax2.grid(True, alpha=0.3, axis='y')
    
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('complex_amplitudes_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved complex amplitudes analysis as 'complex_amplitudes_analysis.png'")
    plt.close()

def main():
    """Main function to run all demonstrations"""
    print("\n" + "📐" * 40)
    print("MATHEMATICAL DEFINITION OF VECTORS")
    print("Comprehensive Python Demonstrations")
    print("📐" * 40)
    
    basic_vector_definition()
    column_vector_notation()
    quantum_ket_notation()
    dimension_and_components()
    classical_vs_quantum_vectors()
    probability_amplitudes()
    normalization_requirement()
    basis_states_demo()
    multi_qubit_dimensions()
    visualize_vectors()
    
    print_section_header("KEY TAKEAWAYS")
    print("\n1. Vector Definition:")
    print("   - Ordered collection of numbers: v = [v₁, v₂, ..., vₙ]ᵀ")
    print("   - Real vectors: v ∈ ℝⁿ (classical)")
    print("   - Complex vectors: v ∈ ℂⁿ (quantum)")
    
    print("\n2. Column Vector Notation:")
    print("   - Vectors written as columns")
    print("   - Emphasizes vertical structure")
    print("   - Standard in linear algebra and quantum computing")
    
    print("\n3. Quantum Ket Notation:")
    print("   - |ψ⟩ represents a quantum state (ket)")
    print("   - Components are complex probability amplitudes")
    print("   - Dirac notation is standard in quantum mechanics")
    
    print("\n4. Dimension:")
    print("   - n = number of components")
    print("   - Single qubit: n = 2")
    print("   - n qubits: dimension = 2ⁿ")
    
    print("\n5. Classical vs Quantum:")
    print("   - Classical: Real components, physical quantities")
    print("   - Quantum: Complex components, probability amplitudes")
    
    print("\n6. Probability Amplitudes:")
    print("   - Born Rule: P(outcome) = |amplitude|²")
    print("   - Enables quantum superposition and interference")
    
    print("\n7. Normalization:")
    print("   - Quantum states must be normalized: ||ψ⟩|| = 1")
    print("   - Ensures total probability = 1")
    
    print("\n8. Basis States:")
    print("   - |0⟩ and |1⟩ form computational basis")
    print("   - Any qubit state: |ψ⟩ = α|0⟩ + β|1⟩")
    
    print("\n" + "=" * 80)
    print("All demonstrations completed successfully!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
