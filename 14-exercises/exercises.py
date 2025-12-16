#!/usr/bin/env python3
"""
Exercises - Vectors in Quantum Computing
=========================================

This module provides solutions to practice exercises covering:
- Vector normalization
- Inner products and orthogonality
- Quantum gates and transformations
- Multi-qubit systems and tensor products
- Density matrices and mixed states
- Bloch sphere representation
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


def exercise_1_normalization():
    """Exercise 1: Vector Normalization."""
    print_section("Exercise 1: Vector Normalization")
    
    print("\nProblem: Normalize the state |ψ⟩ = 3|0⟩ + 4|1⟩")
    
    psi = np.array([3, 4], dtype=complex)
    
    print(f"\nUnnormalized state: {psi}")
    
    norm = np.linalg.norm(psi)
    print(f"\nStep 1: Calculate norm")
    print(f"  ||ψ|| = √(|3|² + |4|²) = √(9 + 16) = √25 = {norm}")
    
    psi_normalized = psi / norm
    print(f"\nStep 2: Normalize")
    print(f"  |ψ_normalized⟩ = (1/{norm})(3|0⟩ + 4|1⟩)")
    print(f"  = {psi_normalized}")
    
    norm_check = np.linalg.norm(psi_normalized)
    print(f"\nStep 3: Verify normalization")
    print(f"  ||ψ_normalized|| = {norm_check:.6f} ✓")
    
    prob_0 = np.abs(psi_normalized[0])**2
    prob_1 = np.abs(psi_normalized[1])**2
    
    print(f"\nMeasurement probabilities:")
    print(f"  P(0) = |{psi_normalized[0]:.3f}|² = {prob_0:.6f}")
    print(f"  P(1) = |{psi_normalized[1]:.3f}|² = {prob_1:.6f}")
    print(f"  Sum = {prob_0 + prob_1:.6f} ✓")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    labels = ['|0⟩', '|1⟩']
    amplitudes_before = np.abs(psi)
    bars1 = ax1.bar(labels, amplitudes_before, color='coral', alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax1.set_title('Before Normalization\n(Not a valid quantum state)', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 5)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, amp in zip(bars1, amplitudes_before):
        ax1.text(bar.get_x() + bar.get_width()/2, amp + 0.1, f'{amp:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    amplitudes_after = np.abs(psi_normalized)
    bars2 = ax2.bar(labels, amplitudes_after, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax2.set_title('After Normalization\n(Valid quantum state)', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, amp in zip(bars2, amplitudes_after):
        ax2.text(bar.get_x() + bar.get_width()/2, amp + 0.02, f'{amp:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/14-exercises/exercise1_normalization.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: exercise1_normalization.png")
    plt.close()


def exercise_2_inner_product():
    """Exercise 2: Inner Product and Orthogonality."""
    print_section("Exercise 2: Inner Product and Orthogonality")
    
    print("\nProblem: Given |ψ⟩ = (1/√2)(|0⟩ + |1⟩) and |φ⟩ = (1/√2)(|0⟩ − |1⟩)")
    
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)  # |+⟩
    phi = np.array([1, -1], dtype=complex) / np.sqrt(2)  # |−⟩
    
    print(f"\n|ψ⟩ = {psi}")
    print(f"|φ⟩ = {phi}")
    
    print(f"\n(a) Calculate inner product ⟨φ|ψ⟩")
    inner_product = np.vdot(phi, psi)
    print(f"  ⟨φ|ψ⟩ = {inner_product:.6f}")
    
    print(f"\n(b) Are these states orthogonal?")
    is_orthogonal = np.abs(inner_product) < 1e-10
    print(f"  |⟨φ|ψ⟩| = {np.abs(inner_product):.10f}")
    print(f"  Orthogonal: {is_orthogonal} ✓")
    print(f"  These are the |+⟩ and |−⟩ states (Hadamard basis)")
    
    print(f"\n(c) Probability of measuring |φ⟩ when system is in |ψ⟩")
    prob = np.abs(inner_product)**2
    print(f"  P(φ) = |⟨φ|ψ⟩|² = {prob:.6f}")
    print(f"  Zero probability because states are orthogonal")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    labels = ['|0⟩', '|1⟩']
    x = np.arange(len(labels))
    width = 0.35
    
    psi_amps = np.abs(psi)
    phi_amps = np.abs(phi)
    
    bars1 = ax1.bar(x - width/2, psi_amps, width, label='|ψ⟩ = |+⟩', color='steelblue', alpha=0.7, edgecolor='black')
    bars2 = ax1.bar(x + width/2, phi_amps, width, label='|φ⟩ = |−⟩', color='coral', alpha=0.7, edgecolor='black')
    
    ax1.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax1.set_title('State Amplitudes', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(fontsize=11)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    properties = ['⟨φ|ψ⟩', '|⟨φ|ψ⟩|', '|⟨φ|ψ⟩|²']
    values = [inner_product.real, np.abs(inner_product), prob]
    colors = ['blue', 'green', 'red']
    
    bars = ax2.bar(properties, values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax2.set_title('Inner Product Properties', fontsize=13, fontweight='bold')
    ax2.set_ylim(-0.1, 1.1)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/14-exercises/exercise2_inner_product.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: exercise2_inner_product.png")
    plt.close()


def exercise_3_quantum_gates():
    """Exercise 3: Quantum Gates and Transformations."""
    print_section("Exercise 3: Quantum Gates and Transformations")
    
    print("\nProblem: Apply Hadamard gate to |1⟩")
    
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    ket1 = np.array([0, 1], dtype=complex)
    
    print(f"\nH = \n{H}")
    print(f"\n|1⟩ = {ket1}")
    
    print(f"\n(a) Calculate resulting state")
    result = H @ ket1
    print(f"  H|1⟩ = {result}")
    print(f"  = (1/√2)(|0⟩ − |1⟩) = |−⟩")
    
    print(f"\n(b) Measurement probabilities")
    prob_0 = np.abs(result[0])**2
    prob_1 = np.abs(result[1])**2
    print(f"  P(0) = |{result[0]:.6f}|² = {prob_0:.6f}")
    print(f"  P(1) = |{result[1]:.6f}|² = {prob_1:.6f}")
    print(f"  Equal superposition!")
    
    print(f"\n(c) Bloch sphere transformation")
    print(f"  |1⟩ is at south pole (z = -1)")
    print(f"  |−⟩ is on negative x-axis (x = -1, z = 0)")
    print(f"  Hadamard rotates 180° around (x+z)/√2 axis")
    
    fig = plt.figure(figsize=(16, 6))
    
    ax1 = plt.subplot(1, 3, 1)
    im = ax1.imshow(np.abs(H), cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax1.set_title('Hadamard Gate Matrix', fontsize=13, fontweight='bold')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['|0⟩', '|1⟩'])
    ax1.set_yticklabels(['⟨0|', '⟨1|'])
    
    for i in range(2):
        for j in range(2):
            val = H[i, j]
            text = f'{val.real:.2f}'
            ax1.text(j, i, text, ha="center", va="center", 
                    color="white", fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    
    ax2 = plt.subplot(1, 3, 2)
    labels = ['|0⟩', '|1⟩']
    x = np.arange(len(labels))
    width = 0.35
    
    before = np.abs(ket1)
    after = np.abs(result)
    
    bars1 = ax2.bar(x - width/2, before, width, label='Before: |1⟩', color='coral', alpha=0.7, edgecolor='black')
    bars2 = ax2.bar(x + width/2, after, width, label='After: H|1⟩', color='steelblue', alpha=0.7, edgecolor='black')
    
    ax2.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax2.set_title('State Transformation', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend(fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax3 = plt.subplot(1, 3, 3)
    probs = [prob_0, prob_1]
    bars = ax3.bar(['P(0)', 'P(1)'], probs, color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax3.set_title('Measurement Probabilities', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.axhline(y=0.5, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Equal')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, prob in zip(bars, probs):
        ax3.text(bar.get_x() + bar.get_width()/2, prob + 0.02, f'{prob:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/14-exercises/exercise3_quantum_gates.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: exercise3_quantum_gates.png")
    plt.close()


def exercise_4_tensor_products():
    """Exercise 4: Multi-Qubit Systems and Tensor Products."""
    print_section("Exercise 4: Multi-Qubit Systems and Tensor Products")
    
    print("\nProblem: Calculate |+⟩ ⊗ |0⟩ and apply CNOT")
    
    ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    ket0 = np.array([1, 0], dtype=complex)
    
    print(f"\n|+⟩ = {ket_plus}")
    print(f"|0⟩ = {ket0}")
    
    print(f"\n(a) Tensor product")
    initial_state = np.kron(ket_plus, ket0)
    print(f"  |+⟩ ⊗ |0⟩ = {initial_state}")
    print(f"  = (1/√2)(|00⟩ + |10⟩)")
    
    print(f"\n(b) Is this state entangled?")
    print(f"  No! It can be written as |+⟩ ⊗ |0⟩ (separable)")
    
    print(f"\n(c) Apply CNOT gate")
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)
    
    bell_state = CNOT @ initial_state
    print(f"  CNOT(|+⟩ ⊗ |0⟩) = {bell_state}")
    print(f"  = (1/√2)(|00⟩ + |11⟩) = |Φ⁺⟩ (Bell state!)")
    print(f"  Now the state IS entangled!")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
    
    im1 = ax1.imshow(np.abs(CNOT), cmap='viridis', aspect='auto')
    ax1.set_title('CNOT Gate Matrix', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    ax1.set_xticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'], fontsize=9)
    ax1.set_yticklabels(['⟨00|', '⟨01|', '⟨10|', '⟨11|'], fontsize=9)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    x = np.arange(4)
    
    bars2 = ax2.bar(x, np.abs(initial_state), color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax2.set_title('Initial: |+⟩ ⊗ |0⟩\n(Separable)', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, amp in zip(bars2, np.abs(initial_state)):
        if amp > 0.01:
            ax2.text(bar.get_x() + bar.get_width()/2, amp + 0.02, f'{amp:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    bars3 = ax3.bar(x, np.abs(bell_state), color='coral', alpha=0.7, edgecolor='black')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=9)
    ax3.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax3.set_title('After CNOT: |Φ⁺⟩\n(Entangled!)', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, amp in zip(bars3, np.abs(bell_state)):
        if amp > 0.01:
            ax3.text(bar.get_x() + bar.get_width()/2, amp + 0.02, f'{amp:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/14-exercises/exercise4_tensor_products.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: exercise4_tensor_products.png")
    plt.close()


def exercise_5_density_matrices():
    """Exercise 5: Density Matrices and Mixed States."""
    print_section("Exercise 5: Density Matrices and Mixed States")
    
    print("\nProblem: Mixed state with P(|0⟩) = 0.7, P(|1⟩) = 0.3")
    
    ket0 = np.array([[1], [0]], dtype=complex)
    ket1 = np.array([[0], [1]], dtype=complex)
    
    print(f"\n(a) Density matrix")
    rho = 0.7 * (ket0 @ ket0.conj().T) + 0.3 * (ket1 @ ket1.conj().T)
    print(f"  ρ = 0.7|0⟩⟨0| + 0.3|1⟩⟨1| = \n{rho}")
    
    print(f"\n(b) Calculate Tr(ρ) and Tr(ρ²)")
    trace = np.trace(rho)
    rho_squared = rho @ rho
    purity = np.trace(rho_squared)
    
    print(f"  Tr(ρ) = {trace.real:.6f} (normalization)")
    print(f"  Tr(ρ²) = {purity.real:.6f} (purity)")
    
    print(f"\n(c) Pure state or mixed state?")
    print(f"  Tr(ρ²) = {purity.real:.6f} < 1")
    print(f"  This is a MIXED state!")
    print(f"  (Pure states have Tr(ρ²) = 1)")
    
    print(f"\n(d) Measurement probabilities")
    P0 = np.array([[1, 0], [0, 0]])
    P1 = np.array([[0, 0], [0, 1]])
    
    prob_0 = np.trace(P0 @ rho).real
    prob_1 = np.trace(P1 @ rho).real
    
    print(f"  P(0) = Tr(P₀ρ) = {prob_0:.6f}")
    print(f"  P(1) = Tr(P₁ρ) = {prob_1:.6f}")
    print(f"  (Simply the diagonal elements!)")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 14))
    
    im1 = ax1.imshow(np.abs(rho), cmap='plasma', vmin=0, vmax=1, aspect='auto')
    ax1.set_title('Density Matrix ρ', fontsize=13, fontweight='bold')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['|0⟩', '|1⟩'])
    ax1.set_yticklabels(['⟨0|', '⟨1|'])
    
    for i in range(2):
        for j in range(2):
            val = rho[i, j]
            text = f'{val.real:.2f}'
            ax1.text(j, i, text, ha="center", va="center", 
                    color="white", fontsize=14, fontweight='bold')
    
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    ax2.bar(['Tr(ρ)', 'Tr(ρ²)'], [trace.real, purity.real], 
           color=['green', 'orange'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax2.set_title('Trace Properties', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1.2)
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Pure state')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, (label, val) in enumerate(zip(['Tr(ρ)', 'Tr(ρ²)'], [trace.real, purity.real])):
        ax2.text(i, val + 0.02, f'{val:.3f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    bars = ax3.bar(['P(0)', 'P(1)'], [prob_0, prob_1], 
                   color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax3.set_title('Measurement Probabilities', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, prob in zip(bars, [prob_0, prob_1]):
        ax3.text(bar.get_x() + bar.get_width()/2, prob + 0.02, f'{prob:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax4.axis('off')
    comparison_text = f"""
    PURITY ANALYSIS
    
    Tr(ρ²) = {purity.real:.3f}
    
    • Tr(ρ²) = 1.0 → Pure state
    • Tr(ρ²) < 1.0 → Mixed state
    • Tr(ρ²) = 0.5 → Maximally mixed
    
    This state has purity {purity.real:.3f}
    → Partially mixed state
    
    Cannot be written as |ψ⟩⟨ψ|
    for any single state |ψ⟩
    """
    ax4.text(0.1, 0.5, comparison_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/14-exercises/exercise5_density_matrices.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: exercise5_density_matrices.png")
    plt.close()


def main():
    """Main function to run all exercise solutions."""
    print("\n" + "=" * 70)
    print("  EXERCISES - VECTORS IN QUANTUM COMPUTING")
    print("=" * 70)
    print("\nThis module provides detailed solutions to practice exercises")
    print("covering all major topics in vectors for quantum computing.")
    
    try:
        exercise_1_normalization()
        exercise_2_inner_product()
        exercise_3_quantum_gates()
        exercise_4_tensor_products()
        exercise_5_density_matrices()
        
        print("\n" + "=" * 70)
        print("  ALL EXERCISES COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nVisualizations saved in: /home/ubuntu/repos/vectorsforquantum/14-exercises/")
        print("\nKey Concepts Covered:")
        print("1. Vector normalization and probability amplitudes")
        print("2. Inner products and orthogonality")
        print("3. Quantum gates as unitary transformations")
        print("4. Multi-qubit systems and tensor products")
        print("5. Density matrices for pure and mixed states")
        print("6. Measurement probabilities and Born rule")
        print("7. Entanglement creation with CNOT gate")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
