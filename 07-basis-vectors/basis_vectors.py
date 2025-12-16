#!/usr/bin/env python3
"""
Basis Vectors in Quantum Computing
Comprehensive demonstrations of basis vectors, basis representation, and change of basis
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D

class Arrow3D(FancyArrowPatch):
    """3D arrow for matplotlib"""
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        
        return np.min(zs)


def computational_basis():
    """Demonstrate computational basis states"""
    print("=" * 60)
    print("COMPUTATIONAL BASIS STATES")
    print("=" * 60)
    
    zero = np.array([[1], [0]], dtype=complex)
    one = np.array([[0], [1]], dtype=complex)
    
    print("\nComputational basis for single qubit:")
    print(f"|0⟩ = {zero.T[0]}")
    print(f"|1⟩ = {one.T[0]}")
    
    print("\nOrthonormality check:")
    print(f"⟨0|0⟩ = {np.vdot(zero.T[0], zero.T[0]):.4f}")
    print(f"⟨1|1⟩ = {np.vdot(one.T[0], one.T[0]):.4f}")
    print(f"⟨0|1⟩ = {np.vdot(zero.T[0], one.T[0]):.4f}")
    print(f"⟨1|0⟩ = {np.vdot(one.T[0], zero.T[0]):.4f}")
    
    alpha = 0.6
    beta = 0.8
    psi = alpha * zero + beta * one
    
    print(f"\nArbitrary state: |ψ⟩ = {alpha}|0⟩ + {beta}|1⟩")
    print(f"Vector form: {psi.T[0]}")
    print(f"Norm: {np.linalg.norm(psi):.4f}")
    
    prob_0 = np.abs(alpha)**2
    prob_1 = np.abs(beta)**2
    
    print(f"\nMeasurement probabilities:")
    print(f"P(0) = |α|² = {prob_0:.4f}")
    print(f"P(1) = |β|² = {prob_1:.4f}")
    print(f"Total = {prob_0 + prob_1:.4f}")
    
    return zero, one, psi


def classical_basis_2d():
    """Demonstrate classical 2D basis vectors"""
    print("\n" + "=" * 60)
    print("CLASSICAL 2D BASIS VECTORS")
    print("=" * 60)
    
    i_hat = np.array([1, 0])
    j_hat = np.array([0, 1])
    
    print("\nStandard basis:")
    print(f"î = {i_hat}")
    print(f"ĵ = {j_hat}")
    
    v = np.array([3, 4])
    
    print(f"\nVector v = {v}")
    print(f"Representation: v = {v[0]}î + {v[1]}ĵ")
    
    v_reconstructed = v[0] * i_hat + v[1] * j_hat
    print(f"Reconstructed: {v_reconstructed}")
    print(f"Match: {np.allclose(v, v_reconstructed)}")
    
    return i_hat, j_hat, v


def classical_basis_3d():
    """Demonstrate classical 3D basis vectors"""
    print("\n" + "=" * 60)
    print("CLASSICAL 3D BASIS VECTORS")
    print("=" * 60)
    
    i_hat = np.array([1, 0, 0])
    j_hat = np.array([0, 1, 0])
    k_hat = np.array([0, 0, 1])
    
    print("\nStandard basis:")
    print(f"î = {i_hat}")
    print(f"ĵ = {j_hat}")
    print(f"k̂ = {k_hat}")
    
    v = np.array([3, 4, 5])
    
    print(f"\nVector v = {v}")
    print(f"Representation: v = {v[0]}î + {v[1]}ĵ + {v[2]}k̂")
    
    v_reconstructed = v[0] * i_hat + v[1] * j_hat + v[2] * k_hat
    print(f"Reconstructed: {v_reconstructed}")
    print(f"Match: {np.allclose(v, v_reconstructed)}")
    
    return i_hat, j_hat, k_hat, v


def hadamard_basis():
    """Demonstrate Hadamard basis states"""
    print("\n" + "=" * 60)
    print("HADAMARD BASIS STATES")
    print("=" * 60)
    
    plus = np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]], dtype=complex)
    minus = np.array([[1/np.sqrt(2)], [-1/np.sqrt(2)]], dtype=complex)
    
    print("\nHadamard basis:")
    print(f"|+⟩ = {plus.T[0]}")
    print(f"|-⟩ = {minus.T[0]}")
    
    print("\nOrthonormality check:")
    print(f"⟨+|+⟩ = {np.vdot(plus.T[0], plus.T[0]):.4f}")
    print(f"⟨-|-⟩ = {np.vdot(minus.T[0], minus.T[0]):.4f}")
    print(f"⟨+|-⟩ = {np.vdot(plus.T[0], minus.T[0]):.4f}")
    print(f"⟨-|+⟩ = {np.vdot(minus.T[0], plus.T[0]):.4f}")
    
    zero = np.array([[1], [0]], dtype=complex)
    one = np.array([[0], [1]], dtype=complex)
    
    print("\nComputational basis in Hadamard basis:")
    
    coeff_plus_0 = np.vdot(plus.T[0], zero.T[0])
    coeff_minus_0 = np.vdot(minus.T[0], zero.T[0])
    print(f"|0⟩ = {coeff_plus_0:.4f}|+⟩ + {coeff_minus_0:.4f}|-⟩")
    
    coeff_plus_1 = np.vdot(plus.T[0], one.T[0])
    coeff_minus_1 = np.vdot(minus.T[0], one.T[0])
    print(f"|1⟩ = {coeff_plus_1:.4f}|+⟩ + {coeff_minus_1:.4f}|-⟩")
    
    zero_reconstructed = coeff_plus_0 * plus + coeff_minus_0 * minus
    print(f"\nVerify |0⟩ reconstruction: {np.allclose(zero, zero_reconstructed)}")
    
    return plus, minus


def circular_basis():
    """Demonstrate circular basis states"""
    print("\n" + "=" * 60)
    print("CIRCULAR BASIS STATES")
    print("=" * 60)
    
    right = np.array([[1/np.sqrt(2)], [1j/np.sqrt(2)]], dtype=complex)
    left = np.array([[1/np.sqrt(2)], [-1j/np.sqrt(2)]], dtype=complex)
    
    print("\nCircular basis:")
    print(f"|R⟩ = {right.T[0]}")
    print(f"|L⟩ = {left.T[0]}")
    
    print("\nOrthonormality check:")
    print(f"⟨R|R⟩ = {np.vdot(right.T[0], right.T[0]):.4f}")
    print(f"⟨L|L⟩ = {np.vdot(left.T[0], left.T[0]):.4f}")
    print(f"⟨R|L⟩ = {np.vdot(right.T[0], left.T[0]):.4f}")
    print(f"⟨L|R⟩ = {np.vdot(left.T[0], right.T[0]):.4f}")
    
    return right, left


def basis_representation():
    """Demonstrate finding coefficients in a basis"""
    print("\n" + "=" * 60)
    print("BASIS REPRESENTATION")
    print("=" * 60)
    
    zero = np.array([[1], [0]], dtype=complex)
    one = np.array([[0], [1]], dtype=complex)
    
    psi = np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]], dtype=complex)
    
    print(f"\nState |ψ⟩ = {psi.T[0]}")
    
    alpha = np.vdot(zero.T[0], psi.T[0])
    beta = np.vdot(one.T[0], psi.T[0])
    
    print(f"\nCoefficients in computational basis:")
    print(f"α = ⟨0|ψ⟩ = {alpha:.4f}")
    print(f"β = ⟨1|ψ⟩ = {beta:.4f}")
    print(f"|ψ⟩ = {alpha:.4f}|0⟩ + {beta:.4f}|1⟩")
    
    psi_reconstructed = alpha * zero + beta * one
    print(f"\nReconstruction matches: {np.allclose(psi, psi_reconstructed)}")
    
    plus = np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]], dtype=complex)
    minus = np.array([[1/np.sqrt(2)], [-1/np.sqrt(2)]], dtype=complex)
    
    gamma = np.vdot(plus.T[0], psi.T[0])
    delta = np.vdot(minus.T[0], psi.T[0])
    
    print(f"\nCoefficients in Hadamard basis:")
    print(f"γ = ⟨+|ψ⟩ = {gamma:.4f}")
    print(f"δ = ⟨-|ψ⟩ = {delta:.4f}")
    print(f"|ψ⟩ = {gamma:.4f}|+⟩ + {delta:.4f}|-⟩")
    
    psi_reconstructed_h = gamma * plus + delta * minus
    print(f"Reconstruction matches: {np.allclose(psi, psi_reconstructed_h)}")


def change_of_basis():
    """Demonstrate change of basis"""
    print("\n" + "=" * 60)
    print("CHANGE OF BASIS")
    print("=" * 60)
    
    zero = np.array([[1], [0]], dtype=complex)
    one = np.array([[0], [1]], dtype=complex)
    plus = np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]], dtype=complex)
    minus = np.array([[1/np.sqrt(2)], [-1/np.sqrt(2)]], dtype=complex)
    
    alpha = 0.6
    beta = 0.8
    psi = alpha * zero + beta * one
    
    print(f"\nState in computational basis:")
    print(f"|ψ⟩ = {alpha}|0⟩ + {beta}|1⟩")
    print(f"Vector: {psi.T[0]}")
    
    gamma = np.vdot(plus.T[0], psi.T[0])
    delta = np.vdot(minus.T[0], psi.T[0])
    
    print(f"\nState in Hadamard basis:")
    print(f"|ψ⟩ = {gamma:.4f}|+⟩ + {delta:.4f}|-⟩")
    
    psi_h = gamma * plus + delta * minus
    print(f"\nReconstruction matches: {np.allclose(psi, psi_h)}")
    
    print("\nManual calculation:")
    print(f"|0⟩ = (1/√2)|+⟩ + (1/√2)|-⟩")
    print(f"|1⟩ = (1/√2)|+⟩ - (1/√2)|-⟩")
    print(f"\n|ψ⟩ = {alpha}[(1/√2)|+⟩ + (1/√2)|-⟩] + {beta}[(1/√2)|+⟩ - (1/√2)|-⟩]")
    
    gamma_manual = (alpha + beta) / np.sqrt(2)
    delta_manual = (alpha - beta) / np.sqrt(2)
    
    print(f"    = [{alpha} + {beta}]/√2 |+⟩ + [{alpha} - {beta}]/√2 |-⟩")
    print(f"    = {gamma_manual:.4f}|+⟩ + {delta_manual:.4f}|-⟩")
    print(f"\nMatches inner product method: {np.allclose(gamma, gamma_manual) and np.allclose(delta, delta_manual)}")


def multi_qubit_basis():
    """Demonstrate multi-qubit basis states"""
    print("\n" + "=" * 60)
    print("MULTI-QUBIT BASIS STATES")
    print("=" * 60)
    
    print("\nTwo-qubit computational basis:")
    
    basis_00 = np.array([[1], [0], [0], [0]], dtype=complex)
    basis_01 = np.array([[0], [1], [0], [0]], dtype=complex)
    basis_10 = np.array([[0], [0], [1], [0]], dtype=complex)
    basis_11 = np.array([[0], [0], [0], [1]], dtype=complex)
    
    print(f"|00⟩ = {basis_00.T[0]}")
    print(f"|01⟩ = {basis_01.T[0]}")
    print(f"|10⟩ = {basis_10.T[0]}")
    print(f"|11⟩ = {basis_11.T[0]}")
    
    bell = (basis_00 + basis_11) / np.sqrt(2)
    
    print(f"\nBell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2")
    print(f"Vector: {bell.T[0]}")
    
    print("\nCoefficients:")
    for i, (label, basis) in enumerate([('00', basis_00), ('01', basis_01), 
                                         ('10', basis_10), ('11', basis_11)]):
        coeff = np.vdot(basis.T[0], bell.T[0])
        print(f"⟨{label}|Φ⁺⟩ = {coeff:.4f}")
    
    print("\n" + "-" * 60)
    print("Three-qubit system:")
    print(f"Dimension: 2³ = 8")
    print(f"Basis states: |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩")


def basis_properties():
    """Demonstrate properties of basis vectors"""
    print("\n" + "=" * 60)
    print("BASIS PROPERTIES")
    print("=" * 60)
    
    zero = np.array([[1], [0]], dtype=complex)
    one = np.array([[0], [1]], dtype=complex)
    
    print("\n1. Linear Independence:")
    print("   Basis vectors are linearly independent")
    print("   c₁|0⟩ + c₂|1⟩ = 0 only if c₁ = c₂ = 0")
    
    print("\n2. Spanning:")
    print("   Any vector can be written as linear combination")
    print("   |ψ⟩ = α|0⟩ + β|1⟩ for some α, β")
    
    print("\n3. Orthonormality:")
    print(f"   ⟨0|0⟩ = {np.vdot(zero.T[0], zero.T[0]):.4f} (normalized)")
    print(f"   ⟨1|1⟩ = {np.vdot(one.T[0], one.T[0]):.4f} (normalized)")
    print(f"   ⟨0|1⟩ = {np.vdot(zero.T[0], one.T[0]):.4f} (orthogonal)")
    
    print("\n4. Completeness:")
    print("   Σᵢ |bᵢ⟩⟨bᵢ| = I (identity)")
    
    completeness = zero @ zero.conj().T + one @ one.conj().T
    identity = np.eye(2, dtype=complex)
    
    print(f"\n   |0⟩⟨0| + |1⟩⟨1| =")
    print(f"   {completeness}")
    print(f"   Equals identity: {np.allclose(completeness, identity)}")


def visualize_basis_vectors():
    """Create comprehensive visualizations of basis vectors"""
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Basis Vectors in Quantum Computing', fontsize=16, fontweight='bold')
    
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Classical 2D Basis')
    
    ax1.arrow(0, 0, 1, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue', linewidth=2)
    ax1.arrow(0, 0, 0, 1, head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)
    ax1.text(1.1, 0, 'î', fontsize=14, color='blue', fontweight='bold')
    ax1.text(0, 1.1, 'ĵ', fontsize=14, color='red', fontweight='bold')
    
    ax1.arrow(0, 0, 0.6, 0.8, head_width=0.1, head_length=0.1, fc='green', ec='green', linewidth=2, linestyle='--')
    ax1.text(0.7, 0.9, 'v = 0.6î + 0.8ĵ', fontsize=10, color='green', fontweight='bold')
    
    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    ax2.set_xlim([0, 1.5])
    ax2.set_ylim([0, 1.5])
    ax2.set_zlim([0, 1.5])
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title('Classical 3D Basis')
    
    arrow_i = Arrow3D(0, 0, 0, 1, 0, 0, mutation_scale=20, lw=2, arrowstyle='-|>', color='blue')
    arrow_j = Arrow3D(0, 0, 0, 0, 1, 0, mutation_scale=20, lw=2, arrowstyle='-|>', color='red')
    arrow_k = Arrow3D(0, 0, 0, 0, 0, 1, mutation_scale=20, lw=2, arrowstyle='-|>', color='green')
    ax2.add_artist(arrow_i)
    ax2.add_artist(arrow_j)
    ax2.add_artist(arrow_k)
    
    ax2.text(1.2, 0, 0, 'î', fontsize=12, color='blue', fontweight='bold')
    ax2.text(0, 1.2, 0, 'ĵ', fontsize=12, color='red', fontweight='bold')
    ax2.text(0, 0, 1.2, 'k̂', fontsize=12, color='green', fontweight='bold')
    
    ax3 = fig.add_subplot(2, 4, 3)
    zero_state = np.array([1, 0])
    one_state = np.array([0, 1])
    
    width = 0.35
    x = np.arange(2)
    
    ax3.bar(x - width/2, zero_state, width, label='|0⟩', color='blue', alpha=0.7)
    ax3.bar(x + width/2, one_state, width, label='|1⟩', color='red', alpha=0.7)
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Computational Basis States')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Component 0', 'Component 1'])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax4 = fig.add_subplot(2, 4, 4)
    plus_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    minus_state = np.array([1/np.sqrt(2), -1/np.sqrt(2)])
    
    ax4.bar(x - width/2, plus_state, width, label='|+⟩', color='green', alpha=0.7)
    ax4.bar(x + width/2, minus_state, width, label='|-⟩', color='orange', alpha=0.7)
    ax4.set_ylabel('Amplitude')
    ax4.set_title('Hadamard Basis States')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Component 0', 'Component 1'])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    ax5 = fig.add_subplot(2, 4, 5)
    alpha, beta = 0.6, 0.8
    psi_state = np.array([alpha, beta])
    
    ax5.bar(x, psi_state, color=['blue', 'red'], alpha=0.7)
    ax5.set_ylabel('Amplitude')
    ax5.set_title(f'|ψ⟩ = {alpha}|0⟩ + {beta}|1⟩')
    ax5.set_xticks(x)
    ax5.set_xticklabels(['|0⟩', '|1⟩'])
    ax5.grid(True, alpha=0.3, axis='y')
    
    for i, (amp, prob) in enumerate(zip(psi_state, [alpha**2, beta**2])):
        ax5.text(i, amp + 0.05, f'P = {prob:.2f}', ha='center', fontweight='bold')
    
    ax6 = fig.add_subplot(2, 4, 6)
    
    comp_coeffs = np.array([alpha, beta])
    had_gamma = (alpha + beta) / np.sqrt(2)
    had_delta = (alpha - beta) / np.sqrt(2)
    had_coeffs = np.array([had_gamma, had_delta])
    
    x_pos = np.arange(2)
    width = 0.35
    
    ax6.bar(x_pos - width/2, comp_coeffs, width, label='Computational', color='blue', alpha=0.7)
    ax6.bar(x_pos + width/2, had_coeffs, width, label='Hadamard', color='green', alpha=0.7)
    ax6.set_ylabel('Coefficient')
    ax6.set_title('Same State, Different Bases')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(['Basis 1', 'Basis 2'])
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    ax7 = fig.add_subplot(2, 4, 7)
    
    bell_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    x_bell = np.arange(4)
    
    ax7.bar(x_bell, bell_state, color='purple', alpha=0.7)
    ax7.set_ylabel('Amplitude')
    ax7.set_title('Bell State |Φ⁺⟩')
    ax7.set_xticks(x_bell)
    ax7.set_xticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'])
    ax7.grid(True, alpha=0.3, axis='y')
    
    ax8 = fig.add_subplot(2, 4, 8)
    
    zero = np.array([1, 0])
    one = np.array([0, 1])
    
    ortho_matrix = np.array([
        [np.vdot(zero, zero), np.vdot(zero, one)],
        [np.vdot(one, zero), np.vdot(one, one)]
    ])
    
    im = ax8.imshow(ortho_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax8.set_xticks([0, 1])
    ax8.set_yticks([0, 1])
    ax8.set_xticklabels(['⟨0|', '⟨1|'])
    ax8.set_yticklabels(['|0⟩', '|1⟩'])
    ax8.set_title('Orthonormality Matrix')
    
    for i in range(2):
        for j in range(2):
            text = ax8.text(j, i, f'{ortho_matrix[i, j]:.1f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax8, label='Inner Product')
    
    plt.tight_layout()
    return fig


def main():
    """Main function to run all demonstrations"""
    print("\n" + "=" * 60)
    print("BASIS VECTORS IN QUANTUM COMPUTING")
    print("Comprehensive Demonstrations")
    print("=" * 60)
    
    computational_basis()
    classical_basis_2d()
    classical_basis_3d()
    hadamard_basis()
    circular_basis()
    basis_representation()
    change_of_basis()
    multi_qubit_basis()
    basis_properties()
    
    print("\n" + "=" * 60)
    print("Creating visualizations...")
    print("=" * 60)
    
    fig = visualize_basis_vectors()
    plt.savefig('basis_vectors_visualization.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'basis_vectors_visualization.png'")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
