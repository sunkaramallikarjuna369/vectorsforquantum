#!/usr/bin/env python3
"""
Outer Product in Quantum Computing
===================================

This module demonstrates the outer product and its applications, including:
- Outer product vs inner product
- Projectors and their properties
- Density matrices for pure and mixed states
- Measurement with density matrices
- Partial trace and entanglement
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


def outer_vs_inner_product():
    """Demonstrate the difference between outer and inner products."""
    print_section("Outer Product vs Inner Product")
    
    print("\nInner product: ⟨φ|ψ⟩ → scalar")
    print("Outer product: |ψ⟩⟨φ| → matrix")
    
    ket0 = np.array([[1], [0]], dtype=complex)
    ket1 = np.array([[0], [1]], dtype=complex)
    bra0 = ket0.conj().T
    bra1 = ket1.conj().T
    
    print("\n" + "-" * 70)
    print("Example: Computational Basis States")
    print("-" * 70)
    
    print(f"\n|0⟩ = \n{ket0}")
    print(f"\n⟨0| = {bra0}")
    
    inner = bra0 @ ket0
    print(f"\nInner product: ⟨0|0⟩ = {inner[0, 0]}")
    
    outer = ket0 @ bra0
    print(f"\nOuter product: |0⟩⟨0| = \n{outer}")
    
    print("\n" + "-" * 70)
    print("All Basis Outer Products")
    print("-" * 70)
    
    outer_00 = ket0 @ bra0
    outer_01 = ket0 @ bra1
    outer_10 = ket1 @ bra0
    outer_11 = ket1 @ bra1
    
    print(f"\n|0⟩⟨0| = \n{outer_00}")
    print(f"\n|0⟩⟨1| = \n{outer_01}")
    print(f"\n|1⟩⟨0| = \n{outer_10}")
    print(f"\n|1⟩⟨1| = \n{outer_11}")
    
    print("\n" + "-" * 70)
    print("Completeness Relation")
    print("-" * 70)
    
    identity = outer_00 + outer_11
    print(f"\n|0⟩⟨0| + |1⟩⟨1| = \n{identity}")
    print(f"\nThis equals the identity matrix I")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    outer_products = [
        (outer_00, '|0⟩⟨0|'),
        (outer_01, '|0⟩⟨1|'),
        (outer_10, '|1⟩⟨0|'),
        (outer_11, '|1⟩⟨1|')
    ]
    
    for idx, (ax, (matrix, label)) in enumerate(zip(axes.flatten(), outer_products)):
        im = ax.imshow(np.abs(matrix), cmap='RdBu', vmin=0, vmax=1, aspect='auto')
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['|0⟩', '|1⟩'])
        ax.set_yticklabels(['⟨0|', '⟨1|'])
        
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                text = f'{val.real:.0f}'
                ax.text(j, i, text, ha="center", va="center", 
                       color="black", fontsize=16, fontweight='bold')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Outer Products of Basis States', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/12-outer-product/outer_products.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: outer_products.png")
    plt.close()


def projector_properties():
    """Demonstrate properties of projectors."""
    print_section("Projector Properties")
    
    print("\nA projector is P = |ψ⟩⟨ψ| for a normalized state |ψ⟩")
    
    ket0 = np.array([[1], [0]], dtype=complex)
    ket1 = np.array([[0], [1]], dtype=complex)
    ket_plus = (ket0 + ket1) / np.sqrt(2)
    
    P0 = ket0 @ ket0.conj().T
    P1 = ket1 @ ket1.conj().T
    P_plus = ket_plus @ ket_plus.conj().T
    
    print("\n" + "-" * 70)
    print("Example Projectors")
    print("-" * 70)
    
    print(f"\nP₀ = |0⟩⟨0| = \n{P0}")
    print(f"\nP₁ = |1⟩⟨1| = \n{P1}")
    print(f"\nP₊ = |+⟩⟨+| = \n{P_plus}")
    
    print("\n" + "-" * 70)
    print("Property 1: Hermitian (P† = P)")
    print("-" * 70)
    
    print(f"\nP₀† = P₀: {np.allclose(P0.conj().T, P0)}")
    print(f"P₁† = P₁: {np.allclose(P1.conj().T, P1)}")
    print(f"P₊† = P₊: {np.allclose(P_plus.conj().T, P_plus)}")
    
    print("\n" + "-" * 70)
    print("Property 2: Idempotent (P² = P)")
    print("-" * 70)
    
    print(f"\nP₀² = P₀: {np.allclose(P0 @ P0, P0)}")
    print(f"P₁² = P₁: {np.allclose(P1 @ P1, P1)}")
    print(f"P₊² = P₊: {np.allclose(P_plus @ P_plus, P_plus)}")
    
    print("\n" + "-" * 70)
    print("Property 3: Eigenvalues are 0 or 1")
    print("-" * 70)
    
    eigenvalues_0, _ = np.linalg.eig(P0)
    eigenvalues_plus, _ = np.linalg.eig(P_plus)
    
    print(f"\nEigenvalues of P₀: {eigenvalues_0.real}")
    print(f"Eigenvalues of P₊: {eigenvalues_plus.real}")
    
    print("\n" + "-" * 70)
    print("Property 4: Projects onto Subspace")
    print("-" * 70)
    
    psi = (0.6 * ket0 + 0.8 * ket1)
    psi = psi / np.linalg.norm(psi)
    
    print(f"\nState: |ψ⟩ = {psi.T}")
    print(f"\nP₀|ψ⟩ = {(P0 @ psi).T} (extracts |0⟩ component)")
    print(f"P₁|ψ⟩ = {(P1 @ psi).T} (extracts |1⟩ component)")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    projectors = [
        (P0, 'P₀ = |0⟩⟨0|'),
        (P1, 'P₁ = |1⟩⟨1|'),
        (P_plus, 'P₊ = |+⟩⟨+|')
    ]
    
    for ax, (P, label) in zip(axes, projectors):
        im = ax.imshow(np.abs(P), cmap='viridis', vmin=0, vmax=1, aspect='auto')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['|0⟩', '|1⟩'])
        ax.set_yticklabels(['⟨0|', '⟨1|'])
        
        for i in range(2):
            for j in range(2):
                val = P[i, j]
                text = f'{val.real:.2f}'
                ax.text(j, i, text, ha="center", va="center", 
                       color="white", fontsize=12, fontweight='bold')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Projector Matrices', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/12-outer-product/projectors.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: projectors.png")
    plt.close()


def density_matrices():
    """Demonstrate density matrices for pure states."""
    print_section("Density Matrices")
    
    print("\nDensity matrix for pure state: ρ = |ψ⟩⟨ψ|")
    
    ket0 = np.array([[1], [0]], dtype=complex)
    ket1 = np.array([[0], [1]], dtype=complex)
    ket_plus = (ket0 + ket1) / np.sqrt(2)
    ket_minus = (ket0 - ket1) / np.sqrt(2)
    
    rho_0 = ket0 @ ket0.conj().T
    rho_1 = ket1 @ ket1.conj().T
    rho_plus = ket_plus @ ket_plus.conj().T
    rho_minus = ket_minus @ ket_minus.conj().T
    
    print("\n" + "-" * 70)
    print("Density Matrices for Common States")
    print("-" * 70)
    
    print(f"\nρ₀ = |0⟩⟨0| = \n{rho_0}")
    print(f"\nρ₁ = |1⟩⟨1| = \n{rho_1}")
    print(f"\nρ₊ = |+⟩⟨+| = \n{rho_plus}")
    print(f"\nρ₋ = |−⟩⟨−| = \n{rho_minus}")
    
    print("\n" + "-" * 70)
    print("Density Matrix Properties")
    print("-" * 70)
    
    for name, rho in [('ρ₀', rho_0), ('ρ₊', rho_plus)]:
        trace = np.trace(rho)
        purity = np.trace(rho @ rho)
        hermitian = np.allclose(rho.conj().T, rho)
        
        print(f"\n{name}:")
        print(f"  Tr(ρ) = {trace.real:.6f} (normalization)")
        print(f"  Tr(ρ²) = {purity.real:.6f} (purity)")
        print(f"  Hermitian: {hermitian}")
    
    print("\n" + "-" * 70)
    print("General Pure State")
    print("-" * 70)
    
    alpha = 0.6
    beta = 0.8
    psi = alpha * ket0 + beta * ket1
    rho_psi = psi @ psi.conj().T
    
    print(f"\n|ψ⟩ = {alpha}|0⟩ + {beta}|1⟩")
    print(f"\nρ = |ψ⟩⟨ψ| = \n{rho_psi}")
    print(f"\nDiagonal elements: [{rho_psi[0,0].real:.3f}, {rho_psi[1,1].real:.3f}]")
    print(f"These are the probabilities: |α|² = {alpha**2:.3f}, |β|² = {beta**2:.3f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    density_matrices_list = [
        (rho_0, 'ρ₀ = |0⟩⟨0|'),
        (rho_1, 'ρ₁ = |1⟩⟨1|'),
        (rho_plus, 'ρ₊ = |+⟩⟨+|'),
        (rho_minus, 'ρ₋ = |−⟩⟨−|')
    ]
    
    for ax, (rho, label) in zip(axes, density_matrices_list):
        im = ax.imshow(np.abs(rho), cmap='plasma', vmin=0, vmax=1, aspect='auto')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['|0⟩', '|1⟩'])
        ax.set_yticklabels(['⟨0|', '⟨1|'])
        
        for i in range(2):
            for j in range(2):
                val = rho[i, j]
                if np.abs(val.imag) < 1e-10:
                    text = f'{val.real:.2f}'
                else:
                    text = f'{val.real:.2f}\n+{val.imag:.2f}i'
                ax.text(j, i, text, ha="center", va="center", 
                       color="white", fontsize=11, fontweight='bold')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Density Matrices for Pure States', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/12-outer-product/density_matrices.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: density_matrices.png")
    plt.close()


def mixed_states():
    """Demonstrate mixed states."""
    print_section("Mixed States")
    
    print("\nMixed state: ρ = Σᵢ pᵢ|ψᵢ⟩⟨ψᵢ|")
    print("Cannot be written as |ψ⟩⟨ψ| for any single state |ψ⟩")
    
    ket0 = np.array([[1], [0]], dtype=complex)
    ket1 = np.array([[0], [1]], dtype=complex)
    
    print("\n" + "-" * 70)
    print("Maximally Mixed State")
    print("-" * 70)
    
    rho_mixed = 0.5 * (ket0 @ ket0.conj().T) + 0.5 * (ket1 @ ket1.conj().T)
    
    print(f"\nρ = 0.5|0⟩⟨0| + 0.5|1⟩⟨1| = \n{rho_mixed}")
    print(f"\nThis equals I/2 (identity divided by 2)")
    
    trace = np.trace(rho_mixed)
    purity = np.trace(rho_mixed @ rho_mixed)
    
    print(f"\nTr(ρ) = {trace.real:.6f}")
    print(f"Tr(ρ²) = {purity.real:.6f} < 1 (mixed state!)")
    
    print("\n" + "-" * 70)
    print("Partially Mixed State")
    print("-" * 70)
    
    p = 0.7
    rho_partial = p * (ket0 @ ket0.conj().T) + (1-p) * (ket1 @ ket1.conj().T)
    
    print(f"\nρ = {p}|0⟩⟨0| + {1-p}|1⟩⟨1| = \n{rho_partial}")
    
    purity_partial = np.trace(rho_partial @ rho_partial)
    print(f"\nTr(ρ²) = {purity_partial.real:.6f}")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    rho_pure = ket0 @ ket0.conj().T
    im1 = ax1.imshow(np.abs(rho_pure), cmap='plasma', vmin=0, vmax=1, aspect='auto')
    ax1.set_title('Pure State: ρ = |0⟩⟨0|\nTr(ρ²) = 1.00', fontsize=12, fontweight='bold')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['|0⟩', '|1⟩'])
    ax1.set_yticklabels(['⟨0|', '⟨1|'])
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    im2 = ax2.imshow(np.abs(rho_partial), cmap='plasma', vmin=0, vmax=1, aspect='auto')
    ax2.set_title(f'Partially Mixed: 0.7|0⟩⟨0| + 0.3|1⟩⟨1|\nTr(ρ²) = {purity_partial.real:.2f}', 
                  fontsize=12, fontweight='bold')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['|0⟩', '|1⟩'])
    ax2.set_yticklabels(['⟨0|', '⟨1|'])
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    im3 = ax3.imshow(np.abs(rho_mixed), cmap='plasma', vmin=0, vmax=1, aspect='auto')
    ax3.set_title(f'Maximally Mixed: I/2\nTr(ρ²) = {purity.real:.2f}', fontsize=12, fontweight='bold')
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['|0⟩', '|1⟩'])
    ax3.set_yticklabels(['⟨0|', '⟨1|'])
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    plt.suptitle('Pure vs Mixed States', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/12-outer-product/mixed_states.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: mixed_states.png")
    plt.close()


def measurement_with_density_matrices():
    """Demonstrate measurement using density matrices."""
    print_section("Measurement with Density Matrices")
    
    print("\nBorn rule with density matrices:")
    print("P(outcome i) = Tr(Pᵢρ)")
    
    ket0 = np.array([[1], [0]], dtype=complex)
    ket1 = np.array([[0], [1]], dtype=complex)
    ket_plus = (ket0 + ket1) / np.sqrt(2)
    
    P0 = ket0 @ ket0.conj().T
    P1 = ket1 @ ket1.conj().T
    
    print("\n" + "-" * 70)
    print("Example 1: Measure |+⟩ in Computational Basis")
    print("-" * 70)
    
    rho_plus = ket_plus @ ket_plus.conj().T
    
    print(f"\nState: |+⟩ = (|0⟩ + |1⟩)/√2")
    print(f"ρ = |+⟩⟨+| = \n{rho_plus}")
    
    P_measure_0 = np.trace(P0 @ rho_plus)
    P_measure_1 = np.trace(P1 @ rho_plus)
    
    print(f"\nP(0) = Tr(P₀ρ) = {P_measure_0.real:.6f}")
    print(f"P(1) = Tr(P₁ρ) = {P_measure_1.real:.6f}")
    
    print("\n" + "-" * 70)
    print("Example 2: General State")
    print("-" * 70)
    
    alpha = 0.6
    beta = 0.8
    psi = alpha * ket0 + beta * ket1
    rho_psi = psi @ psi.conj().T
    
    print(f"\nState: |ψ⟩ = {alpha}|0⟩ + {beta}|1⟩")
    
    P_measure_0 = np.trace(P0 @ rho_psi)
    P_measure_1 = np.trace(P1 @ rho_psi)
    
    print(f"\nP(0) = Tr(P₀ρ) = {P_measure_0.real:.6f} = |α|² = {alpha**2:.6f}")
    print(f"P(1) = Tr(P₁ρ) = {P_measure_1.real:.6f} = |β|² = {beta**2:.6f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    labels = ['P(0)', 'P(1)']
    probs_plus = [P_measure_0.real, P_measure_1.real]
    
    bars1 = ax1.bar(labels, probs_plus, color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax1.set_title('Measuring |+⟩ in Computational Basis', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, prob in zip(bars1, probs_plus):
        ax1.text(bar.get_x() + bar.get_width()/2, prob + 0.02, f'{prob:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    probs_psi = [P_measure_0.real, P_measure_1.real]
    
    bars2 = ax2.bar(labels, probs_psi, color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax2.set_title(f'Measuring {alpha}|0⟩ + {beta}|1⟩', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, prob in zip(bars2, probs_psi):
        ax2.text(bar.get_x() + bar.get_width()/2, prob + 0.02, f'{prob:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/12-outer-product/measurement.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: measurement.png")
    plt.close()


def applications():
    """Demonstrate applications of outer products."""
    print_section("Applications in Quantum Computing")
    
    print("\n1. Quantum Measurement")
    print("   - Projectors |ψ⟩⟨ψ| represent measurement operators")
    print("   - Born rule: P(i) = Tr(Pᵢρ)")
    
    print("\n2. Quantum State Tomography")
    print("   - Reconstruct unknown states from measurements")
    print("   - Build density matrix from measurement statistics")
    
    print("\n3. Open Quantum Systems")
    print("   - Density matrices describe systems with decoherence")
    print("   - Mixed states arise from interaction with environment")
    
    print("\n4. Entanglement Detection")
    print("   - Partial trace reveals entanglement")
    print("   - If ρₐ is mixed but ρₐᵦ is pure, systems are entangled")
    
    print("\n5. Quantum Error Correction")
    print("   - Error syndromes detected using projectors")
    print("   - Density matrices track error probabilities")


def visualize_summary():
    """Create a comprehensive summary visualization."""
    print_section("Summary Visualization")
    
    fig = plt.figure(figsize=(16, 12))
    
    ax1 = plt.subplot(2, 3, 1)
    ket0 = np.array([[1], [0]])
    outer = ket0 @ ket0.T
    im1 = ax1.imshow(np.abs(outer), cmap='RdBu', vmin=0, vmax=1, aspect='auto')
    ax1.set_title('Outer Product |0⟩⟨0|', fontsize=11, fontweight='bold')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['|0⟩', '|1⟩'], fontsize=9)
    ax1.set_yticklabels(['⟨0|', '⟨1|'], fontsize=9)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    ax2 = plt.subplot(2, 3, 2)
    properties = ['Hermitian\nP†=P', 'Idempotent\nP²=P', 'Eigenvalues\n0 or 1']
    values = [1, 1, 1]
    bars = ax2.bar(range(len(properties)), values, color='green', alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(properties)))
    ax2.set_xticklabels(properties, fontsize=9)
    ax2.set_ylabel('Satisfied', fontsize=10, fontweight='bold')
    ax2.set_title('Projector Properties', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['No', 'Yes'])
    
    ax3 = plt.subplot(2, 3, 3)
    states = ['Pure\n|0⟩⟨0|', 'Partial\n0.7|0⟩⟨0|+\n0.3|1⟩⟨1|', 'Maximally\nMixed\nI/2']
    purities = [1.0, 0.58, 0.5]
    colors = ['green', 'orange', 'red']
    bars = ax3.bar(range(len(states)), purities, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(states)))
    ax3.set_xticklabels(states, fontsize=8)
    ax3.set_ylabel('Tr(ρ²)', fontsize=10, fontweight='bold')
    ax3.set_title('Purity: Tr(ρ²)', fontsize=11, fontweight='bold')
    ax3.set_ylim(0, 1.2)
    ax3.axhline(y=1, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Pure')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, purity in zip(bars, purities):
        ax3.text(bar.get_x() + bar.get_width()/2, purity + 0.05, f'{purity:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax4 = plt.subplot(2, 3, 4)
    ket_plus = np.array([[1], [1]]) / np.sqrt(2)
    rho_plus = ket_plus @ ket_plus.conj().T
    im4 = ax4.imshow(np.abs(rho_plus), cmap='plasma', vmin=0, vmax=1, aspect='auto')
    ax4.set_title('ρ₊ = |+⟩⟨+|', fontsize=11, fontweight='bold')
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['|0⟩', '|1⟩'], fontsize=9)
    ax4.set_yticklabels(['⟨0|', '⟨1|'], fontsize=9)
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    ax5 = plt.subplot(2, 3, 5)
    P0 = np.array([[1, 0], [0, 0]])
    P1 = np.array([[0, 0], [0, 1]])
    prob_0 = np.trace(P0 @ rho_plus).real
    prob_1 = np.trace(P1 @ rho_plus).real
    
    bars = ax5.bar(['P(0)', 'P(1)'], [prob_0, prob_1], color=['steelblue', 'coral'], 
                   alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Probability', fontsize=10, fontweight='bold')
    ax5.set_title('Measuring |+⟩', fontsize=11, fontweight='bold')
    ax5.set_ylim(0, 1)
    ax5.grid(True, alpha=0.3, axis='y')
    
    for bar, prob in zip(bars, [prob_0, prob_1]):
        ax5.text(bar.get_x() + bar.get_width()/2, prob + 0.02, f'{prob:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    table_data = [
        ['Concept', 'Formula', 'Result'],
        ['Inner Product', '⟨φ|ψ⟩', 'Scalar'],
        ['Outer Product', '|ψ⟩⟨φ|', 'Matrix'],
        ['Projector', '|ψ⟩⟨ψ|', 'Operator'],
        ['Density Matrix', '|ψ⟩⟨ψ|', 'State'],
        ['Measurement', 'Tr(Pρ)', 'Probability']
    ]
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.35, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#64ffda')
        table[(0, i)].set_text_props(weight='bold')
    
    ax6.set_title('Outer Product Summary', fontsize=11, fontweight='bold', pad=20)
    
    plt.suptitle('Outer Product and Density Matrices - Summary', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/12-outer-product/outer_product_summary.png', dpi=150, bbox_inches='tight')
    print("\n✓ Summary visualization saved: outer_product_summary.png")
    plt.close()


def main():
    """Main function to run all demonstrations."""
    print("\n" + "=" * 70)
    print("  OUTER PRODUCT IN QUANTUM COMPUTING")
    print("=" * 70)
    print("\nThis module demonstrates the outer product and its applications")
    print("in quantum computing.")
    
    try:
        outer_vs_inner_product()
        projector_properties()
        density_matrices()
        mixed_states()
        measurement_with_density_matrices()
        applications()
        visualize_summary()
        
        print("\n" + "=" * 70)
        print("  ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nVisualizations saved in: /home/ubuntu/repos/vectorsforquantum/12-outer-product/")
        print("\nKey Takeaways:")
        print("1. Outer product |ψ⟩⟨φ| produces a matrix")
        print("2. Inner product ⟨φ|ψ⟩ produces a scalar")
        print("3. Projectors P = |ψ⟩⟨ψ| are Hermitian and idempotent")
        print("4. Density matrices ρ = |ψ⟩⟨ψ| describe quantum states")
        print("5. Pure states: Tr(ρ²) = 1, Mixed states: Tr(ρ²) < 1")
        print("6. Measurement: P(i) = Tr(Pᵢρ)")
        print("7. Density matrices essential for open quantum systems")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
