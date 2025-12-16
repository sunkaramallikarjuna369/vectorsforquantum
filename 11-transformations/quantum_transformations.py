#!/usr/bin/env python3
"""
Quantum Transformations in Quantum Computing
=============================================

This module demonstrates quantum transformations and gates, including:
- Unitary matrices and their properties
- Single-qubit gates (Pauli, Hadamard, Phase, T)
- Gate composition
- Two-qubit gates (CNOT)
- Universal gate sets
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


I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex)


def is_unitary(U, tol=1e-10):
    """Check if a matrix is unitary."""
    n = U.shape[0]
    identity = np.eye(n)
    product = U @ U.conj().T
    return np.allclose(product, identity, atol=tol)


def unitary_properties():
    """Demonstrate properties of unitary matrices."""
    print_section("Unitary Matrix Properties")
    
    print("\nA matrix U is unitary if U†U = I")
    print("where U† is the conjugate transpose.")
    
    print("\n" + "-" * 70)
    print("Example: Hadamard Gate")
    print("-" * 70)
    
    print(f"\nH = \n{H}")
    print(f"\nH† = \n{H.conj().T}")
    print(f"\nH†H = \n{H.conj().T @ H}")
    print(f"\nIs H unitary? {is_unitary(H)}")
    
    print("\n" + "-" * 70)
    print("Unitary Properties")
    print("-" * 70)
    
    ket0 = np.array([1, 0], dtype=complex)
    ket_transformed = H @ ket0
    
    norm_before = np.linalg.norm(ket0)
    norm_after = np.linalg.norm(ket_transformed)
    
    print(f"\n1. Norm Preservation:")
    print(f"   ||ψ|| = {norm_before:.6f}")
    print(f"   ||Hψ|| = {norm_after:.6f}")
    print(f"   Preserved: {np.isclose(norm_before, norm_after)}")
    
    ket1 = np.array([0, 1], dtype=complex)
    inner_before = np.vdot(ket0, ket1)
    inner_after = np.vdot(H @ ket0, H @ ket1)
    
    print(f"\n2. Inner Product Preservation:")
    print(f"   ⟨0|1⟩ = {inner_before:.6f}")
    print(f"   ⟨H0|H1⟩ = {inner_after:.6f}")
    print(f"   Preserved: {np.isclose(inner_before, inner_after)}")
    
    H_inv = np.linalg.inv(H)
    H_dagger = H.conj().T
    
    print(f"\n3. Inverse Property:")
    print(f"   H⁻¹ = H†: {np.allclose(H_inv, H_dagger)}")
    
    det_H = np.linalg.det(H)
    
    print(f"\n4. Determinant:")
    print(f"   det(H) = {det_H:.6f}")
    print(f"   |det(H)| = {np.abs(det_H):.6f}")
    print(f"   |det(H)| = 1: {np.isclose(np.abs(det_H), 1)}")


def single_qubit_gates():
    """Demonstrate single-qubit quantum gates."""
    print_section("Single-Qubit Gates")
    
    gates = {
        'I': I,
        'X': X,
        'Y': Y,
        'Z': Z,
        'H': H,
        'S': S,
        'T': T
    }
    
    gate_names = {
        'I': 'Identity',
        'X': 'Pauli-X (NOT)',
        'Y': 'Pauli-Y',
        'Z': 'Pauli-Z',
        'H': 'Hadamard',
        'S': 'Phase (S)',
        'T': 'T (π/8)'
    }
    
    for name, gate in gates.items():
        print(f"\n{gate_names[name]} Gate ({name}):")
        print(gate)
        print(f"Unitary: {is_unitary(gate)}")
    
    print("\n" + "-" * 70)
    print("Gate Actions on Basis States")
    print("-" * 70)
    
    ket0 = np.array([1, 0], dtype=complex)
    ket1 = np.array([0, 1], dtype=complex)
    
    for name, gate in gates.items():
        result0 = gate @ ket0
        result1 = gate @ ket1
        
        print(f"\n{name}|0⟩ = {result0}")
        print(f"{name}|1⟩ = {result1}")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, (name, gate) in enumerate(gates.items()):
        ax = axes[idx]
        
        matrix_real = np.real(gate)
        matrix_imag = np.imag(gate)
        matrix_abs = np.abs(gate)
        
        im = ax.imshow(matrix_abs, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        ax.set_title(f'{gate_names[name]} ({name})', fontsize=11, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['|0⟩', '|1⟩'])
        ax.set_yticklabels(['⟨0|', '⟨1|'])
        
        for i in range(2):
            for j in range(2):
                val = gate[i, j]
                if np.abs(val.imag) < 1e-10:
                    text = f'{val.real:.2f}'
                elif np.abs(val.real) < 1e-10:
                    text = f'{val.imag:.2f}i'
                else:
                    text = f'{val.real:.2f}\n+{val.imag:.2f}i'
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=9, fontweight='bold')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    axes[7].axis('off')
    
    plt.suptitle('Single-Qubit Quantum Gates', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/11-transformations/single_qubit_gates.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: single_qubit_gates.png")
    plt.close()


def gate_composition():
    """Demonstrate composition of quantum gates."""
    print_section("Gate Composition")
    
    print("\nGates can be applied sequentially.")
    print("The combined effect is the product of matrices (right to left):")
    print("|ψ''⟩ = U₂U₁|ψ⟩")
    
    print("\n" + "-" * 70)
    print("Example 1: H followed by Z")
    print("-" * 70)
    
    ket0 = np.array([1, 0], dtype=complex)
    
    print(f"\nStarting state: |0⟩ = {ket0}")
    
    state1 = H @ ket0
    print(f"\nAfter H: H|0⟩ = {state1}")
    print(f"         = (1/√2)(|0⟩ + |1⟩) = |+⟩")
    
    state2 = Z @ state1
    print(f"\nAfter Z: Z(H|0⟩) = {state2}")
    print(f"         = (1/√2)(|0⟩ − |1⟩) = |−⟩")
    
    ZH = Z @ H
    state_combined = ZH @ ket0
    
    print(f"\nCombined: ZH|0⟩ = {state_combined}")
    print(f"\nVerification: {np.allclose(state2, state_combined)}")
    
    print("\n" + "-" * 70)
    print("Example 2: Non-Commutativity (HZ vs ZH)")
    print("-" * 70)
    
    HZ = H @ Z
    ZH = Z @ H
    
    print(f"\nHZ = \n{HZ}")
    print(f"\nZH = \n{ZH}")
    print(f"\nHZ = ZH? {np.allclose(HZ, ZH)}")
    print("\nOrder matters! Matrix multiplication is not commutative.")
    
    print("\n" + "-" * 70)
    print("Example 3: Multiple Gate Sequence")
    print("-" * 70)
    
    print("\nSequence: H → S → H")
    
    state = ket0
    print(f"Start: {state}")
    
    state = H @ state
    print(f"After H: {state}")
    
    state = S @ state
    print(f"After S: {state}")
    
    state = H @ state
    print(f"After H: {state}")
    
    combined = H @ S @ H
    result = combined @ ket0
    print(f"\nCombined HSH|0⟩ = {result}")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    sequences = [
        ('HZ', H @ Z),
        ('ZH', Z @ H),
        ('HSH', H @ S @ H),
        ('XYZ', X @ Y @ Z),
        ('HXH', H @ X @ H),
        ('SHS†', S @ H @ S.conj().T)
    ]
    
    for idx, (ax, (name, gate)) in enumerate(zip(axes.flatten(), sequences)):
        matrix_abs = np.abs(gate)
        
        im = ax.imshow(matrix_abs, cmap='viridis', aspect='auto')
        ax.set_title(f'{name} Gate', fontsize=12, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['|0⟩', '|1⟩'])
        ax.set_yticklabels(['⟨0|', '⟨1|'])
        
        for i in range(2):
            for j in range(2):
                val = gate[i, j]
                if np.abs(val.imag) < 1e-10:
                    text = f'{val.real:.2f}'
                elif np.abs(val.real) < 1e-10:
                    text = f'{val.imag:.2f}i'
                else:
                    text = f'{val.real:.1f}\n+{val.imag:.1f}i'
                ax.text(j, i, text, ha="center", va="center", color="white", fontsize=9, fontweight='bold')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Composed Quantum Gates', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/11-transformations/gate_composition.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: gate_composition.png")
    plt.close()


def two_qubit_gates():
    """Demonstrate two-qubit gates."""
    print_section("Two-Qubit Gates")
    
    print("\nTwo-qubit gates are 4×4 unitary matrices.")
    print("The most important is the CNOT (Controlled-NOT) gate.")
    
    print("\n" + "-" * 70)
    print("CNOT Gate (Controlled-NOT)")
    print("-" * 70)
    
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)
    
    print(f"\nCNOT = \n{CNOT}")
    print(f"\nIs CNOT unitary? {is_unitary(CNOT)}")
    
    print("\n" + "-" * 70)
    print("CNOT Action on Basis States")
    print("-" * 70)
    
    ket00 = np.array([1, 0, 0, 0], dtype=complex)
    ket01 = np.array([0, 1, 0, 0], dtype=complex)
    ket10 = np.array([0, 0, 1, 0], dtype=complex)
    ket11 = np.array([0, 0, 0, 1], dtype=complex)
    
    basis_states = [
        ('|00⟩', ket00),
        ('|01⟩', ket01),
        ('|10⟩', ket10),
        ('|11⟩', ket11)
    ]
    
    for name, state in basis_states:
        result = CNOT @ state
        result_idx = np.argmax(np.abs(result))
        result_name = ['|00⟩', '|01⟩', '|10⟩', '|11⟩'][result_idx]
        print(f"CNOT{name} = {result_name}")
    
    print("\nNote: CNOT flips the target (second) qubit if control (first) is |1⟩")
    
    print("\n" + "-" * 70)
    print("Creating Entanglement with CNOT")
    print("-" * 70)
    
    ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    ket0_single = np.array([1, 0], dtype=complex)
    
    initial_state = np.kron(ket_plus, ket0_single)
    
    print(f"\nInitial state: |+⟩ ⊗ |0⟩ = {initial_state}")
    print(f"             = (1/√2)(|00⟩ + |10⟩)")
    
    bell_state = CNOT @ initial_state
    
    print(f"\nAfter CNOT: {bell_state}")
    print(f"          = (1/√2)(|00⟩ + |11⟩)")
    print(f"          = |Φ⁺⟩ (Bell state!)")
    
    print("\nThe qubits are now entangled!")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    im1 = ax1.imshow(np.abs(CNOT), cmap='viridis', aspect='auto')
    ax1.set_title('CNOT Gate Matrix', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    ax1.set_xticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'], fontsize=9)
    ax1.set_yticklabels(['⟨00|', '⟨01|', '⟨10|', '⟨11|'], fontsize=9)
    
    for i in range(4):
        for j in range(4):
            val = CNOT[i, j]
            ax1.text(j, i, f'{int(val.real)}', ha="center", va="center", 
                    color="white", fontsize=11, fontweight='bold')
    
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    x = np.arange(4)
    
    bars2 = ax2.bar(x, np.abs(initial_state), color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
    ax2.set_title('Initial State: |+⟩ ⊗ |0⟩', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, amp in zip(bars2, np.abs(initial_state)):
        if amp > 0.01:
            ax2.text(bar.get_x() + bar.get_width()/2, amp + 0.02, f'{amp:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    bars3 = ax3.bar(x, np.abs(bell_state), color='coral', alpha=0.7, edgecolor='black')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=9)
    ax3.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
    ax3.set_title('After CNOT: Bell State |Φ⁺⟩', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, amp in zip(bars3, np.abs(bell_state)):
        if amp > 0.01:
            ax3.text(bar.get_x() + bar.get_width()/2, amp + 0.02, f'{amp:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/11-transformations/cnot_gate.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: cnot_gate.png")
    plt.close()


def bloch_sphere_rotations():
    """Demonstrate rotations on the Bloch sphere."""
    print_section("Bloch Sphere Rotations")
    
    print("\nQuantum gates can be visualized as rotations on the Bloch sphere.")
    
    def Rx(theta):
        """Rotation around X-axis."""
        return np.array([
            [np.cos(theta/2), -1j*np.sin(theta/2)],
            [-1j*np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)
    
    def Ry(theta):
        """Rotation around Y-axis."""
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)
    
    def Rz(theta):
        """Rotation around Z-axis."""
        return np.array([
            [np.exp(-1j*theta/2), 0],
            [0, np.exp(1j*theta/2)]
        ], dtype=complex)
    
    print("\n" + "-" * 70)
    print("Rotation Gates")
    print("-" * 70)
    
    theta = np.pi / 4
    
    print(f"\nRₓ(π/4) = \n{Rx(theta)}")
    print(f"\nRᵧ(π/4) = \n{Ry(theta)}")
    print(f"\nRᵤ(π/4) = \n{Rz(theta)}")
    
    print("\n" + "-" * 70)
    print("Pauli Gates as π Rotations")
    print("-" * 70)
    
    print(f"\nX = iRₓ(π) (up to global phase)")
    print(f"Y = iRᵧ(π) (up to global phase)")
    print(f"Z = Rᵤ(π)")
    
    print(f"\nVerification:")
    print(f"Rₓ(π) ≈ -iX: {np.allclose(Rx(np.pi), -1j*X)}")
    print(f"Rᵧ(π) ≈ -iY: {np.allclose(Ry(np.pi), -1j*Y)}")
    print(f"Rᵤ(π) ≈ -iZ: {np.allclose(Rz(np.pi), -1j*Z)}")


def applications():
    """Demonstrate applications of quantum transformations."""
    print_section("Applications in Quantum Computing")
    
    print("\n1. Quantum Algorithms")
    print("   - All quantum algorithms are sequences of gates")
    print("   - Examples: Grover's search, Shor's factoring, VQE")
    
    print("\n2. Quantum State Preparation")
    print("   - Prepare arbitrary states from |0⟩")
    print("   - Create superposition and entanglement")
    
    print("\n3. Quantum Error Correction")
    print("   - Detect and correct errors using gate sequences")
    print("   - Stabilizer codes, surface codes")
    
    print("\n4. Quantum Circuit Optimization")
    print("   - Compile high-level algorithms to native gates")
    print("   - Minimize gate count and circuit depth")
    
    print("\n5. Quantum Simulation")
    print("   - Simulate time evolution: U(t) = e^(-iHt)")
    print("   - Trotterization for Hamiltonian simulation")


def visualize_summary():
    """Create a comprehensive summary visualization."""
    print_section("Summary Visualization")
    
    fig = plt.figure(figsize=(16, 12))
    
    ax1 = plt.subplot(2, 3, 1)
    pauli_gates = {'X': X, 'Y': Y, 'Z': Z}
    x_pos = np.arange(len(pauli_gates))
    
    for idx, (name, gate) in enumerate(pauli_gates.items()):
        matrix_abs = np.abs(gate)
        ax1.imshow(matrix_abs, cmap='RdBu', aspect='auto', extent=[idx-0.4, idx+0.4, -0.4, 1.4])
    
    ax1.set_title('Pauli Gates (X, Y, Z)', fontsize=11, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(list(pauli_gates.keys()))
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['⟨0|', '⟨1|'])
    
    ax2 = plt.subplot(2, 3, 2)
    ket0 = np.array([1, 0])
    ket_plus = H @ ket0
    
    states = ['|0⟩', 'H|0⟩ = |+⟩']
    amplitudes = [
        [1, 0],
        [ket_plus[0].real, ket_plus[1].real]
    ]
    
    x = np.arange(2)
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, amplitudes[0], width, label='|0⟩', color='steelblue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, amplitudes[1], width, label='H|0⟩', color='coral', alpha=0.7)
    
    ax2.set_ylabel('Amplitude', fontsize=10, fontweight='bold')
    ax2.set_title('Hadamard Creates Superposition', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['|0⟩', '|1⟩'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax3 = plt.subplot(2, 3, 3)
    gates_seq = ['I', 'H', 'HZ', 'HZH']
    gate_matrices = [I, H, Z@H, H@Z@H]
    
    for idx, (name, gate) in enumerate(zip(gates_seq, gate_matrices)):
        det = np.linalg.det(gate)
        ax3.bar(idx, np.abs(det), color='steelblue', alpha=0.7, edgecolor='black')
        ax3.text(idx, np.abs(det) + 0.05, f'{np.abs(det):.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax3.set_xticks(range(len(gates_seq)))
    ax3.set_xticklabels(gates_seq, fontsize=9)
    ax3.set_ylabel('|det(U)|', fontsize=10, fontweight='bold')
    ax3.set_title('Determinant = 1 (Unitary)', fontsize=11, fontweight='bold')
    ax3.set_ylim(0, 1.5)
    ax3.axhline(y=1, color='red', linestyle='--', linewidth=2, label='|det| = 1')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax4 = plt.subplot(2, 3, 4)
    CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    im4 = ax4.imshow(CNOT, cmap='viridis', aspect='auto')
    ax4.set_title('CNOT Gate (4×4)', fontsize=11, fontweight='bold')
    ax4.set_xticks(range(4))
    ax4.set_yticks(range(4))
    ax4.set_xticklabels(['00', '01', '10', '11'], fontsize=8)
    ax4.set_yticklabels(['00', '01', '10', '11'], fontsize=8)
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    ax5 = plt.subplot(2, 3, 5)
    gates_check = {'I': I, 'X': X, 'H': H, 'S': S, 'T': T}
    gate_names_list = list(gates_check.keys())
    unitary_check = [is_unitary(g) for g in gates_check.values()]
    
    colors = ['green' if u else 'red' for u in unitary_check]
    bars = ax5.bar(range(len(gate_names_list)), [1 if u else 0 for u in unitary_check],
                   color=colors, alpha=0.7, edgecolor='black')
    ax5.set_xticks(range(len(gate_names_list)))
    ax5.set_xticklabels(gate_names_list)
    ax5.set_ylabel('Unitary?', fontsize=10, fontweight='bold')
    ax5.set_title('Unitarity Verification', fontsize=11, fontweight='bold')
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(['No', 'Yes'])
    ax5.set_ylim(0, 1.2)
    
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    table_data = [
        ['Gate', 'Effect', 'Unitary'],
        ['I', 'Identity', '✓'],
        ['X', 'Bit flip', '✓'],
        ['Y', 'Bit+Phase', '✓'],
        ['Z', 'Phase flip', '✓'],
        ['H', 'Superposition', '✓'],
        ['CNOT', 'Entanglement', '✓']
    ]
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.45, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#64ffda')
        table[(0, i)].set_text_props(weight='bold')
    
    ax6.set_title('Quantum Gates Summary', fontsize=11, fontweight='bold', pad=20)
    
    plt.suptitle('Quantum Transformations - Summary', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/11-transformations/transformations_summary.png', dpi=150, bbox_inches='tight')
    print("\n✓ Summary visualization saved: transformations_summary.png")
    plt.close()


def main():
    """Main function to run all demonstrations."""
    print("\n" + "=" * 70)
    print("  QUANTUM TRANSFORMATIONS IN QUANTUM COMPUTING")
    print("=" * 70)
    print("\nThis module demonstrates quantum transformations and gates")
    print("in quantum computing.")
    
    try:
        unitary_properties()
        single_qubit_gates()
        gate_composition()
        two_qubit_gates()
        bloch_sphere_rotations()
        applications()
        visualize_summary()
        
        print("\n" + "=" * 70)
        print("  ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nVisualizations saved in: /home/ubuntu/repos/vectorsforquantum/11-transformations/")
        print("\nKey Takeaways:")
        print("1. Quantum transformations are unitary matrices")
        print("2. Unitary matrices preserve normalization and inner products")
        print("3. Single-qubit gates: I, X, Y, Z, H, S, T")
        print("4. Gates can be composed (order matters!)")
        print("5. CNOT gate creates entanglement")
        print("6. Universal gate sets can approximate any unitary")
        print("7. All quantum operations are reversible (except measurement)")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
