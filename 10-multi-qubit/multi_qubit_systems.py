#!/usr/bin/env python3
"""
Multi-Qubit Systems in Quantum Computing
=========================================

This module demonstrates multi-qubit systems and tensor products, including:
- Tensor product basics
- Two-qubit and three-qubit systems
- Separable vs entangled states
- Bell states and GHZ states
- Scaling to n qubits
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


def tensor_product_basics():
    """Demonstrate basic tensor product operations."""
    print_section("Tensor Product Basics")
    
    print("\nThe tensor product (⊗) combines vector spaces.")
    print("For vectors: v ⊗ w creates a larger vector.")
    
    print("\n" + "-" * 70)
    print("Single Qubit States")
    print("-" * 70)
    
    ket0 = np.array([1, 0], dtype=complex)
    ket1 = np.array([0, 1], dtype=complex)
    
    print(f"\n|0⟩ = {ket0}")
    print(f"|1⟩ = {ket1}")
    
    print("\n" + "-" * 70)
    print("Two-Qubit Basis States (Tensor Products)")
    print("-" * 70)
    
    ket00 = np.kron(ket0, ket0)
    ket01 = np.kron(ket0, ket1)
    ket10 = np.kron(ket1, ket0)
    ket11 = np.kron(ket1, ket1)
    
    print(f"\n|00⟩ = |0⟩ ⊗ |0⟩ = {ket00}")
    print(f"|01⟩ = |0⟩ ⊗ |1⟩ = {ket01}")
    print(f"|10⟩ = |1⟩ ⊗ |0⟩ = {ket10}")
    print(f"|11⟩ = |1⟩ ⊗ |1⟩ = {ket11}")
    
    print("\nNote: These 4 vectors form an orthonormal basis for the")
    print("4-dimensional Hilbert space of a two-qubit system.")
    
    print("\n" + "-" * 70)
    print("Orthonormality Check")
    print("-" * 70)
    
    basis = [ket00, ket01, ket10, ket11]
    labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    
    print("\nInner products:")
    for i, (v1, l1) in enumerate(zip(basis, labels)):
        for j, (v2, l2) in enumerate(zip(basis, labels)):
            inner = np.vdot(v1, v2)
            print(f"⟨{l1[1:-1]}|{l2[1:-1]}⟩ = {inner:.1f}", end="  ")
        print()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    matrix = np.zeros((4, 4))
    for i, v in enumerate(basis):
        matrix[:, i] = v.real
    
    im = ax.imshow(matrix, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(['Component 0', 'Component 1', 'Component 2', 'Component 3'])
    ax.set_title('Two-Qubit Computational Basis States', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Basis State', fontsize=12, fontweight='bold')
    ax.set_ylabel('Vector Component', fontsize=12, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Amplitude', fontsize=11, fontweight='bold')
    
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, f'{matrix[i, j]:.0f}',
                          ha="center", va="center", color="black", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/10-multi-qubit/basis_states.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: basis_states.png")
    plt.close()


def general_two_qubit_states():
    """Demonstrate general two-qubit states."""
    print_section("General Two-Qubit States")
    
    print("\nA general two-qubit state is a superposition of basis states:")
    print("|ψ⟩ = c₀₀|00⟩ + c₀₁|01⟩ + c₁₀|10⟩ + c₁₁|11⟩")
    print("\nwhere Σ|cᵢⱼ|² = 1 (normalization)")
    
    print("\n" + "-" * 70)
    print("Example 1: Product State (Separable)")
    print("-" * 70)
    
    ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    ket0 = np.array([1, 0], dtype=complex)
    
    psi = np.kron(ket_plus, ket0)
    
    print(f"\n|+⟩ = (|0⟩ + |1⟩)/√2 = {ket_plus}")
    print(f"|0⟩ = {ket0}")
    print(f"\n|ψ⟩ = |+⟩ ⊗ |0⟩ = {psi}")
    print(f"\nExpanded: |ψ⟩ = {psi[0]:.3f}|00⟩ + {psi[1]:.3f}|01⟩ + {psi[2]:.3f}|10⟩ + {psi[3]:.3f}|11⟩")
    
    norm = np.sum(np.abs(psi)**2)
    print(f"\nNormalization check: Σ|cᵢⱼ|² = {norm:.6f}")
    
    print("\n" + "-" * 70)
    print("Example 2: Bell State (Entangled)")
    print("-" * 70)
    
    ket00 = np.array([1, 0, 0, 0], dtype=complex)
    ket11 = np.array([0, 0, 0, 1], dtype=complex)
    
    bell = (ket00 + ket11) / np.sqrt(2)
    
    print(f"\n|Φ⁺⟩ = (|00⟩ + |11⟩)/√2 = {bell}")
    print(f"\nExpanded: |Φ⁺⟩ = {bell[0]:.3f}|00⟩ + {bell[1]:.3f}|01⟩ + {bell[2]:.3f}|10⟩ + {bell[3]:.3f}|11⟩")
    
    norm = np.sum(np.abs(bell)**2)
    print(f"\nNormalization check: Σ|cᵢⱼ|² = {norm:.6f}")
    
    print("\nThis state CANNOT be written as |ψ⟩ ⊗ |φ⟩ for any single-qubit states.")
    print("The qubits are entangled!")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    x = np.arange(len(labels))
    
    amplitudes_prod = np.abs(psi)
    phases_prod = np.angle(psi)
    
    bars1 = ax1.bar(x, amplitudes_prod, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Amplitude |cᵢⱼ|', fontsize=12, fontweight='bold')
    ax1.set_title('Product State: |+⟩ ⊗ |0⟩\n(Separable)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1)
    
    for i, (bar, amp) in enumerate(zip(bars1, amplitudes_prod)):
        if amp > 0.01:
            ax1.text(bar.get_x() + bar.get_width()/2, amp + 0.02, f'{amp:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    amplitudes_bell = np.abs(bell)
    phases_bell = np.angle(bell)
    
    bars2 = ax2.bar(x, amplitudes_bell, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Amplitude |cᵢⱼ|', fontsize=12, fontweight='bold')
    ax2.set_title('Bell State: (|00⟩ + |11⟩)/√2\n(Entangled)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1)
    
    for i, (bar, amp) in enumerate(zip(bars2, amplitudes_bell)):
        if amp > 0.01:
            ax2.text(bar.get_x() + bar.get_width()/2, amp + 0.02, f'{amp:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/10-multi-qubit/two_qubit_states.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: two_qubit_states.png")
    plt.close()


def bell_states():
    """Demonstrate the four Bell states."""
    print_section("Bell States")
    
    print("\nThe Bell states are four maximally entangled two-qubit states:")
    
    ket00 = np.array([1, 0, 0, 0], dtype=complex)
    ket01 = np.array([0, 1, 0, 0], dtype=complex)
    ket10 = np.array([0, 0, 1, 0], dtype=complex)
    ket11 = np.array([0, 0, 0, 1], dtype=complex)
    
    bell_00 = (ket00 + ket11) / np.sqrt(2)
    bell_01 = (ket00 - ket11) / np.sqrt(2)
    bell_10 = (ket01 + ket10) / np.sqrt(2)
    bell_11 = (ket01 - ket10) / np.sqrt(2)
    
    bell_states_list = [bell_00, bell_01, bell_10, bell_11]
    bell_names = ['|Φ⁺⟩', '|Φ⁻⟩', '|Ψ⁺⟩', '|Ψ⁻⟩']
    bell_formulas = [
        '(|00⟩ + |11⟩)/√2',
        '(|00⟩ − |11⟩)/√2',
        '(|01⟩ + |10⟩)/√2',
        '(|01⟩ − |10⟩)/√2'
    ]
    
    for name, formula, state in zip(bell_names, bell_formulas, bell_states_list):
        print(f"\n{name} = {formula}")
        print(f"     = {state}")
    
    print("\n" + "-" * 70)
    print("Orthonormality of Bell States")
    print("-" * 70)
    
    print("\nInner products:")
    for i, (s1, n1) in enumerate(zip(bell_states_list, bell_names)):
        for j, (s2, n2) in enumerate(zip(bell_states_list, bell_names)):
            inner = np.vdot(s1, s2)
            print(f"⟨{n1[1:-1]}|{n2[1:-1]}⟩ = {inner.real:.1f}", end="  ")
        print()
    
    print("\nThe Bell states form an orthonormal basis for the two-qubit Hilbert space.")
    
    print("\n" + "-" * 70)
    print("Measurement Properties")
    print("-" * 70)
    
    print("\nFor |Φ⁺⟩ = (|00⟩ + |11⟩)/√2:")
    print("  P(00) = |⟨00|Φ⁺⟩|² = 1/2")
    print("  P(01) = |⟨01|Φ⁺⟩|² = 0")
    print("  P(10) = |⟨10|Φ⁺⟩|² = 0")
    print("  P(11) = |⟨11|Φ⁺⟩|² = 1/2")
    print("\nMeasuring one qubit instantly determines the other!")
    print("If first qubit is 0, second must be 0.")
    print("If first qubit is 1, second must be 1.")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    x = np.arange(len(labels))
    
    for idx, (ax, state, name, formula) in enumerate(zip(axes, bell_states_list, bell_names, bell_formulas)):
        amplitudes = np.abs(state)
        probabilities = amplitudes ** 2
        
        bars = ax.bar(x, amplitudes, color='steelblue', alpha=0.7, edgecolor='black', label='Amplitude')
        
        ax2 = ax.twinx()
        ax2.plot(x, probabilities, 'ro-', linewidth=2, markersize=8, label='Probability')
        ax2.set_ylabel('Probability |cᵢⱼ|²', fontsize=11, fontweight='bold', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 1)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Amplitude |cᵢⱼ|', fontsize=11, fontweight='bold')
        ax.set_title(f'{name} = {formula}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
        for i, (bar, amp, prob) in enumerate(zip(bars, amplitudes, probabilities)):
            if amp > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, amp + 0.02, f'{amp:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('The Four Bell States', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/10-multi-qubit/bell_states.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: bell_states.png")
    plt.close()


def three_qubit_systems():
    """Demonstrate three-qubit systems."""
    print_section("Three-Qubit Systems")
    
    print("\nA three-qubit system has dimension 2³ = 8.")
    print("Basis states: {|000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩}")
    
    ket0 = np.array([1, 0], dtype=complex)
    ket1 = np.array([0, 1], dtype=complex)
    
    basis_3qubit = []
    labels_3qubit = []
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                state = np.kron(np.kron([1, 0] if i == 0 else [0, 1],
                                       [1, 0] if j == 0 else [0, 1]),
                               [1, 0] if k == 0 else [0, 1])
                basis_3qubit.append(state)
                labels_3qubit.append(f'|{i}{j}{k}⟩')
    
    print("\n" + "-" * 70)
    print("Three-Qubit Basis States")
    print("-" * 70)
    
    for label, state in zip(labels_3qubit, basis_3qubit):
        print(f"{label} = {state}")
    
    print("\n" + "-" * 70)
    print("GHZ State (Greenberger-Horne-Zeilinger)")
    print("-" * 70)
    
    ket000 = basis_3qubit[0]
    ket111 = basis_3qubit[7]
    
    ghz = (ket000 + ket111) / np.sqrt(2)
    
    print(f"\n|GHZ⟩ = (|000⟩ + |111⟩)/√2")
    print(f"      = {ghz}")
    
    print("\nThis is a maximally entangled three-qubit state.")
    print("All three qubits are correlated:")
    print("  - Measuring any qubit affects the others")
    print("  - If one is 0, all are 0")
    print("  - If one is 1, all are 1")
    
    print("\n" + "-" * 70)
    print("W State")
    print("-" * 70)
    
    ket001 = basis_3qubit[1]
    ket010 = basis_3qubit[2]
    ket100 = basis_3qubit[4]
    
    w_state = (ket001 + ket010 + ket100) / np.sqrt(3)
    
    print(f"\n|W⟩ = (|001⟩ + |010⟩ + |100⟩)/√3")
    print(f"    = {w_state}")
    
    print("\nThe W state has different entanglement properties than GHZ.")
    print("It's more robust: if one qubit is lost, the remaining two are still entangled.")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(8)
    
    amplitudes_ghz = np.abs(ghz)
    probabilities_ghz = amplitudes_ghz ** 2
    
    bars1 = ax1.bar(x, amplitudes_ghz, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_3qubit, rotation=45)
    ax1.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax1.set_title('GHZ State: (|000⟩ + |111⟩)/√2', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1)
    
    for i, (bar, amp) in enumerate(zip(bars1, amplitudes_ghz)):
        if amp > 0.01:
            ax1.text(bar.get_x() + bar.get_width()/2, amp + 0.02, f'{amp:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    amplitudes_w = np.abs(w_state)
    probabilities_w = amplitudes_w ** 2
    
    bars2 = ax2.bar(x, amplitudes_w, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_3qubit, rotation=45)
    ax2.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax2.set_title('W State: (|001⟩ + |010⟩ + |100⟩)/√3', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1)
    
    for i, (bar, amp) in enumerate(zip(bars2, amplitudes_w)):
        if amp > 0.01:
            ax2.text(bar.get_x() + bar.get_width()/2, amp + 0.02, f'{amp:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/10-multi-qubit/three_qubit_states.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: three_qubit_states.png")
    plt.close()


def scaling_to_n_qubits():
    """Demonstrate scaling properties of n-qubit systems."""
    print_section("Scaling to n Qubits")
    
    print("\nThe Hilbert space dimension grows exponentially with the number of qubits:")
    print("  n qubits → 2ⁿ dimensional space")
    
    print("\n" + "-" * 70)
    print("Dimension vs Number of Qubits")
    print("-" * 70)
    
    n_values = [1, 2, 3, 4, 5, 10, 20, 50, 100, 300]
    
    print(f"\n{'Qubits':>8} | {'Dimension':>15} | {'Classical Bits Equivalent':>25}")
    print("-" * 55)
    
    for n in n_values:
        dim = 2 ** n
        if dim < 1000:
            dim_str = f"{dim}"
        elif dim < 1e6:
            dim_str = f"{dim/1e3:.1f}K"
        elif dim < 1e9:
            dim_str = f"{dim/1e6:.1f}M"
        elif dim < 1e12:
            dim_str = f"{dim/1e9:.1f}B"
        else:
            dim_str = f"~10^{int(np.log10(dim))}"
        
        print(f"{n:>8} | {dim_str:>15} | {n:>25}")
    
    print("\nNote: With 300 qubits, the state space has more dimensions")
    print("than there are atoms in the observable universe (~10⁸⁰)!")
    
    print("\n" + "-" * 70)
    print("Classical Simulation Complexity")
    print("-" * 70)
    
    print("\nTo classically simulate an n-qubit system:")
    print("  - Memory: O(2ⁿ) complex numbers")
    print("  - Gate operations: O(2ⁿ) operations per gate")
    print("\nThis exponential scaling makes classical simulation infeasible")
    print("for large quantum systems (typically n > 50).")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    n_range = np.arange(1, 11)
    dimensions = 2 ** n_range
    
    bars = ax1.bar(n_range, dimensions, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Number of Qubits (n)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Hilbert Space Dimension (2ⁿ)', fontsize=12, fontweight='bold')
    ax1.set_title('Exponential Growth of State Space\n(Linear Scale)', fontsize=13, fontweight='bold')
    ax1.set_xticks(n_range)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, dim in zip(bars, dimensions):
        ax1.text(bar.get_x() + bar.get_width()/2, dim + 20, f'{dim}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    n_range_log = np.arange(1, 21)
    dimensions_log = 2 ** n_range_log
    
    ax2.semilogy(n_range_log, dimensions_log, 'o-', color='coral', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Qubits (n)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Hilbert Space Dimension (2ⁿ)', fontsize=12, fontweight='bold')
    ax2.set_title('Exponential Growth of State Space\n(Log Scale)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xticks(n_range_log[::2])
    
    for n in [1, 5, 10, 15, 20]:
        dim = 2 ** n
        ax2.annotate(f'{dim}', xy=(n, dim), xytext=(n+0.5, dim*1.5),
                    fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1))
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/10-multi-qubit/scaling.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: scaling.png")
    plt.close()


def applications():
    """Demonstrate applications of multi-qubit systems."""
    print_section("Applications in Quantum Computing")
    
    print("\n1. Quantum Algorithms")
    print("   - Shor's algorithm: Factoring large numbers (exponential speedup)")
    print("   - Grover's algorithm: Database search (quadratic speedup)")
    print("   - Quantum simulation: Simulating quantum systems")
    
    print("\n2. Quantum Error Correction")
    print("   - Use multiple physical qubits to encode one logical qubit")
    print("   - Example: 9-qubit Shor code, 7-qubit Steane code")
    print("   - Protects against decoherence and errors")
    
    print("\n3. Quantum Communication")
    print("   - Quantum teleportation: Transfer quantum state using entanglement")
    print("   - Superdense coding: Send 2 classical bits using 1 qubit")
    print("   - Quantum key distribution: Secure communication")
    
    print("\n4. Quantum Machine Learning")
    print("   - Quantum neural networks")
    print("   - Variational quantum eigensolvers (VQE)")
    print("   - Quantum approximate optimization algorithm (QAOA)")
    
    print("\n5. Quantum Chemistry")
    print("   - Molecular simulation")
    print("   - Drug discovery")
    print("   - Materials science")


def visualize_summary():
    """Create a comprehensive summary visualization."""
    print_section("Summary Visualization")
    
    fig = plt.figure(figsize=(16, 12))
    
    ax1 = plt.subplot(2, 3, 1)
    ket0 = np.array([1, 0])
    ket1 = np.array([0, 1])
    basis = [
        np.kron(ket0, ket0),
        np.kron(ket0, ket1),
        np.kron(ket1, ket0),
        np.kron(ket1, ket1)
    ]
    matrix = np.column_stack(basis)
    im1 = ax1.imshow(matrix, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    ax1.set_title('Two-Qubit Basis States', fontsize=11, fontweight='bold')
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'], fontsize=9)
    ax1.set_ylabel('Component', fontsize=10)
    plt.colorbar(im1, ax=ax1)
    
    ax2 = plt.subplot(2, 3, 2)
    ket00 = np.array([1, 0, 0, 0])
    ket11 = np.array([0, 0, 0, 1])
    bell = (ket00 + ket11) / np.sqrt(2)
    x = np.arange(4)
    ax2.bar(x, np.abs(bell), color='coral', alpha=0.7, edgecolor='black')
    ax2.set_title('Bell State |Φ⁺⟩', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'], fontsize=9)
    ax2.set_ylabel('Amplitude', fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax3 = plt.subplot(2, 3, 3)
    n_range = np.arange(1, 16)
    dimensions = 2 ** n_range
    ax3.semilogy(n_range, dimensions, 'o-', color='steelblue', linewidth=2, markersize=6)
    ax3.set_title('Exponential Scaling', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Qubits (n)', fontsize=10)
    ax3.set_ylabel('Dimension (2ⁿ)', fontsize=10)
    ax3.grid(True, alpha=0.3, which='both')
    
    ax4 = plt.subplot(2, 3, 4)
    ghz_3 = np.zeros(8)
    ghz_3[0] = 1/np.sqrt(2)
    ghz_3[7] = 1/np.sqrt(2)
    x = np.arange(8)
    ax4.bar(x, ghz_3, color='green', alpha=0.7, edgecolor='black')
    ax4.set_title('GHZ State (3 qubits)', fontsize=11, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['000', '001', '010', '011', '100', '101', '110', '111'], fontsize=8)
    ax4.set_ylabel('Amplitude', fontsize=10)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3, axis='y')
    
    ax5 = plt.subplot(2, 3, 5)
    categories = ['Separable\n|+⟩⊗|0⟩', 'Bell State\n|Φ⁺⟩', 'GHZ State\n(3 qubits)', 'W State\n(3 qubits)']
    entanglement = [0, 1, 1, 0.8]
    colors = ['lightblue', 'coral', 'green', 'purple']
    bars = ax5.bar(categories, entanglement, color=colors, alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Entanglement Measure', fontsize=10, fontweight='bold')
    ax5.set_title('Entanglement Comparison', fontsize=11, fontweight='bold')
    ax5.set_ylim(0, 1.2)
    ax5.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, entanglement):
        ax5.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    table_data = [
        ['Qubits', 'Dimension', 'States'],
        ['1', '2', '2'],
        ['2', '4', '4'],
        ['3', '8', '8'],
        ['4', '16', '16'],
        ['5', '32', '32'],
        ['10', '1,024', '1,024'],
        ['20', '~1M', '~1M']
    ]
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#64ffda')
        table[(0, i)].set_text_props(weight='bold')
    
    ax6.set_title('Multi-Qubit Dimensions', fontsize=11, fontweight='bold', pad=20)
    
    plt.suptitle('Multi-Qubit Systems - Summary', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/10-multi-qubit/multi_qubit_summary.png', dpi=150, bbox_inches='tight')
    print("\n✓ Summary visualization saved: multi_qubit_summary.png")
    plt.close()


def main():
    """Main function to run all demonstrations."""
    print("\n" + "=" * 70)
    print("  MULTI-QUBIT SYSTEMS IN QUANTUM COMPUTING")
    print("=" * 70)
    print("\nThis module demonstrates multi-qubit systems and tensor products")
    print("in quantum computing.")
    
    try:
        tensor_product_basics()
        general_two_qubit_states()
        bell_states()
        three_qubit_systems()
        scaling_to_n_qubits()
        applications()
        visualize_summary()
        
        print("\n" + "=" * 70)
        print("  ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nVisualizations saved in: /home/ubuntu/repos/vectorsforquantum/10-multi-qubit/")
        print("\nKey Takeaways:")
        print("1. Multi-qubit systems use tensor products (⊗)")
        print("2. n qubits create a 2ⁿ-dimensional Hilbert space")
        print("3. Two-qubit systems have 4 basis states")
        print("4. Bell states are maximally entangled two-qubit states")
        print("5. Entangled states cannot be factored into single-qubit states")
        print("6. Exponential scaling enables quantum advantage")
        print("7. GHZ and W states are important three-qubit entangled states")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
