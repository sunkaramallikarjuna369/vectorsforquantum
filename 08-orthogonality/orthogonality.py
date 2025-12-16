#!/usr/bin/env python3
"""
Orthogonality in Quantum Computing
Comprehensive demonstrations of orthogonal vectors, orthonormality, and Gram-Schmidt
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform


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


def basic_orthogonality():
    """Demonstrate basic orthogonality concepts"""
    print("=" * 60)
    print("BASIC ORTHOGONALITY")
    print("=" * 60)
    
    zero = np.array([1, 0], dtype=complex)
    one = np.array([0, 1], dtype=complex)
    
    print("\nComputational basis states:")
    print(f"|0⟩ = {zero}")
    print(f"|1⟩ = {one}")
    
    inner_product = np.vdot(zero, one)
    print(f"\nInner product ⟨0|1⟩ = {inner_product}")
    print(f"Orthogonal: {np.isclose(inner_product, 0)}")
    
    plus = (1/np.sqrt(2)) * np.array([1, 1], dtype=complex)
    minus = (1/np.sqrt(2)) * np.array([1, -1], dtype=complex)
    
    print("\n" + "-" * 60)
    print("Hadamard basis states:")
    print(f"|+⟩ = {plus}")
    print(f"|-⟩ = {minus}")
    
    inner_product_h = np.vdot(plus, minus)
    print(f"\nInner product ⟨+|-⟩ = {inner_product_h:.4f}")
    print(f"Orthogonal: {np.isclose(inner_product_h, 0)}")
    
    print("\n" + "-" * 60)
    print("Non-orthogonal states:")
    print(f"|0⟩ = {zero}")
    print(f"|+⟩ = {plus}")
    
    inner_product_no = np.vdot(zero, plus)
    print(f"\nInner product ⟨0|+⟩ = {inner_product_no:.4f}")
    print(f"Orthogonal: {np.isclose(inner_product_no, 0)}")
    
    angle_rad = np.arccos(np.abs(inner_product_no))
    angle_deg = np.degrees(angle_rad)
    print(f"Angle: {angle_deg:.2f}°")


def orthonormality_check():
    """Check orthonormality of basis sets"""
    print("\n" + "=" * 60)
    print("ORTHONORMALITY CHECK")
    print("=" * 60)
    
    zero = np.array([1, 0], dtype=complex)
    one = np.array([0, 1], dtype=complex)
    
    print("\nComputational basis:")
    print(f"⟨0|0⟩ = {np.vdot(zero, zero):.4f} (should be 1)")
    print(f"⟨1|1⟩ = {np.vdot(one, one):.4f} (should be 1)")
    print(f"⟨0|1⟩ = {np.vdot(zero, one):.4f} (should be 0)")
    print(f"⟨1|0⟩ = {np.vdot(one, zero):.4f} (should be 0)")
    
    basis = [zero, one]
    n = len(basis)
    ortho_matrix = np.zeros((n, n), dtype=complex)
    
    for i in range(n):
        for j in range(n):
            ortho_matrix[i, j] = np.vdot(basis[i], basis[j])
    
    print("\nOrthonormality matrix:")
    print(ortho_matrix)
    print(f"Is identity: {np.allclose(ortho_matrix, np.eye(n))}")
    
    plus = (1/np.sqrt(2)) * np.array([1, 1], dtype=complex)
    minus = (1/np.sqrt(2)) * np.array([1, -1], dtype=complex)
    
    print("\n" + "-" * 60)
    print("Hadamard basis:")
    print(f"⟨+|+⟩ = {np.vdot(plus, plus):.4f} (should be 1)")
    print(f"⟨-|-⟩ = {np.vdot(minus, minus):.4f} (should be 1)")
    print(f"⟨+|-⟩ = {np.vdot(plus, minus):.4f} (should be 0)")
    print(f"⟨-|+⟩ = {np.vdot(minus, plus):.4f} (should be 0)")


def classical_orthogonality():
    """Demonstrate orthogonality in classical vectors"""
    print("\n" + "=" * 60)
    print("CLASSICAL ORTHOGONALITY")
    print("=" * 60)
    
    print("\n2D orthogonal vectors:")
    v1 = np.array([3, 0])
    v2 = np.array([0, 4])
    
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    
    dot_product = np.dot(v1, v2)
    print(f"\nDot product: {dot_product}")
    print(f"Orthogonal: {np.isclose(dot_product, 0)}")
    
    print("\n" + "-" * 60)
    print("3D orthogonal vectors:")
    i_hat = np.array([1, 0, 0])
    j_hat = np.array([0, 1, 0])
    k_hat = np.array([0, 0, 1])
    
    print(f"î = {i_hat}")
    print(f"ĵ = {j_hat}")
    print(f"k̂ = {k_hat}")
    
    print(f"\nî · ĵ = {np.dot(i_hat, j_hat)}")
    print(f"î · k̂ = {np.dot(i_hat, k_hat)}")
    print(f"ĵ · k̂ = {np.dot(j_hat, k_hat)}")
    print("All pairs are orthogonal ✓")


def complex_orthogonality():
    """Demonstrate orthogonality with complex vectors"""
    print("\n" + "=" * 60)
    print("COMPLEX ORTHOGONALITY")
    print("=" * 60)
    
    v = np.array([1, 1j], dtype=complex)
    w = np.array([1j, 1], dtype=complex)
    
    print(f"\nv = {v}")
    print(f"w = {w}")
    
    inner_product = np.vdot(v, w)
    print(f"\n⟨v|w⟩ = {inner_product}")
    print(f"Orthogonal: {np.isclose(inner_product, 0)}")
    
    print("\n" + "-" * 60)
    print("Circular basis:")
    right = (1/np.sqrt(2)) * np.array([1, 1j], dtype=complex)
    left = (1/np.sqrt(2)) * np.array([1, -1j], dtype=complex)
    
    print(f"|R⟩ = {right}")
    print(f"|L⟩ = {left}")
    
    inner_product_circ = np.vdot(right, left)
    print(f"\n⟨R|L⟩ = {inner_product_circ:.4f}")
    print(f"Orthogonal: {np.isclose(inner_product_circ, 0)}")


def pythagorean_theorem():
    """Demonstrate Pythagorean theorem for orthogonal vectors"""
    print("\n" + "=" * 60)
    print("PYTHAGOREAN THEOREM")
    print("=" * 60)
    
    v = np.array([3, 0])
    w = np.array([0, 4])
    
    print(f"\nOrthogonal vectors:")
    print(f"v = {v}")
    print(f"w = {w}")
    
    print(f"\nv · w = {np.dot(v, w)} (orthogonal)")
    
    norm_v = np.linalg.norm(v)
    norm_w = np.linalg.norm(w)
    norm_sum = np.linalg.norm(v + w)
    
    print(f"\n||v|| = {norm_v:.4f}")
    print(f"||w|| = {norm_w:.4f}")
    print(f"||v + w|| = {norm_sum:.4f}")
    
    print(f"\n||v||² + ||w||² = {norm_v**2:.4f} + {norm_w**2:.4f} = {norm_v**2 + norm_w**2:.4f}")
    print(f"||v + w||² = {norm_sum**2:.4f}")
    print(f"Pythagorean theorem holds: {np.isclose(norm_sum**2, norm_v**2 + norm_w**2)}")


def gram_schmidt():
    """Demonstrate Gram-Schmidt orthogonalization"""
    print("\n" + "=" * 60)
    print("GRAM-SCHMIDT ORTHOGONALIZATION")
    print("=" * 60)
    
    v1 = np.array([1, 1], dtype=float)
    v2 = np.array([1, 0], dtype=float)
    
    print("\nOriginal vectors:")
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v1 · v2 = {np.dot(v1, v2):.4f} (not orthogonal)")
    
    u1 = v1 / np.linalg.norm(v1)
    print(f"\nStep 1: Normalize v1")
    print(f"u1 = v1/||v1|| = {u1}")
    
    projection = np.dot(v2, u1) * u1
    v2_orth = v2 - projection
    
    print(f"\nStep 2: Remove projection")
    print(f"proj_u1(v2) = {projection}")
    print(f"v2' = v2 - proj_u1(v2) = {v2_orth}")
    
    u2 = v2_orth / np.linalg.norm(v2_orth)
    print(f"\nStep 3: Normalize")
    print(f"u2 = v2'/||v2'|| = {u2}")
    
    print(f"\n" + "-" * 60)
    print("Verification:")
    print(f"u1 · u2 = {np.dot(u1, u2):.4f} (should be 0)")
    print(f"||u1|| = {np.linalg.norm(u1):.4f} (should be 1)")
    print(f"||u2|| = {np.linalg.norm(u2):.4f} (should be 1)")


def quantum_measurement_orthogonality():
    """Demonstrate role of orthogonality in quantum measurement"""
    print("\n" + "=" * 60)
    print("QUANTUM MEASUREMENT AND ORTHOGONALITY")
    print("=" * 60)
    
    psi = (1/np.sqrt(2)) * np.array([1, 1], dtype=complex)
    
    print(f"\nState: |ψ⟩ = (|0⟩ + |1⟩)/√2 = {psi}")
    
    zero = np.array([1, 0], dtype=complex)
    one = np.array([0, 1], dtype=complex)
    
    print("\nMeasurement in computational basis:")
    prob_0 = np.abs(np.vdot(zero, psi))**2
    prob_1 = np.abs(np.vdot(one, psi))**2
    
    print(f"P(0) = |⟨0|ψ⟩|² = {prob_0:.4f}")
    print(f"P(1) = |⟨1|ψ⟩|² = {prob_1:.4f}")
    print(f"Total = {prob_0 + prob_1:.4f}")
    
    plus = (1/np.sqrt(2)) * np.array([1, 1], dtype=complex)
    minus = (1/np.sqrt(2)) * np.array([1, -1], dtype=complex)
    
    print("\nMeasurement in Hadamard basis:")
    prob_plus = np.abs(np.vdot(plus, psi))**2
    prob_minus = np.abs(np.vdot(minus, psi))**2
    
    print(f"P(+) = |⟨+|ψ⟩|² = {prob_plus:.4f}")
    print(f"P(-) = |⟨-|ψ⟩|² = {prob_minus:.4f}")
    print(f"Total = {prob_plus + prob_minus:.4f}")
    
    print("\nNote: Orthogonal measurement outcomes are mutually exclusive")


def state_distinguishability():
    """Demonstrate distinguishability of orthogonal vs non-orthogonal states"""
    print("\n" + "=" * 60)
    print("STATE DISTINGUISHABILITY")
    print("=" * 60)
    
    print("\nOrthogonal states (perfectly distinguishable):")
    zero = np.array([1, 0], dtype=complex)
    one = np.array([0, 1], dtype=complex)
    
    print(f"|0⟩ = {zero}")
    print(f"|1⟩ = {one}")
    print(f"⟨0|1⟩ = {np.vdot(zero, one):.4f}")
    print("Can be perfectly distinguished by measurement")
    
    print("\n" + "-" * 60)
    print("Non-orthogonal states (cannot be perfectly distinguished):")
    psi1 = np.array([1, 0], dtype=complex)
    psi2 = (1/np.sqrt(2)) * np.array([1, 1], dtype=complex)
    
    print(f"|ψ1⟩ = {psi1}")
    print(f"|ψ2⟩ = {psi2}")
    
    overlap = np.vdot(psi1, psi2)
    print(f"⟨ψ1|ψ2⟩ = {overlap:.4f}")
    print(f"|⟨ψ1|ψ2⟩|² = {np.abs(overlap)**2:.4f}")
    print("Cannot be perfectly distinguished")
    
    success_prob = 0.5 * (1 + np.sqrt(1 - np.abs(overlap)**2))
    print(f"\nOptimal success probability: {success_prob:.4f}")


def visualize_orthogonality():
    """Create comprehensive visualizations of orthogonality"""
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Orthogonality in Quantum Computing', fontsize=16, fontweight='bold')
    
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.set_xlim(-0.5, 4)
    ax1.set_ylim(-0.5, 5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Orthogonal 2D Vectors')
    
    ax1.arrow(0, 0, 3, 0, head_width=0.3, head_length=0.3, fc='blue', ec='blue', linewidth=2)
    ax1.arrow(0, 0, 0, 4, head_width=0.3, head_length=0.3, fc='red', ec='red', linewidth=2)
    ax1.text(3.2, 0, 'v', fontsize=14, color='blue', fontweight='bold')
    ax1.text(0, 4.3, 'w', fontsize=14, color='red', fontweight='bold')
    ax1.text(1.5, 2, 'v · w = 0', fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    square_size = 0.5
    square = plt.Rectangle((0, 0), square_size, square_size, fill=False, edgecolor='green', linewidth=2)
    ax1.add_patch(square)
    
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.set_xlim(-0.5, 4)
    ax2.set_ylim(-0.5, 4)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Non-Orthogonal 2D Vectors')
    
    ax2.arrow(0, 0, 3, 0, head_width=0.2, head_length=0.2, fc='blue', ec='blue', linewidth=2)
    ax2.arrow(0, 0, 2, 2, head_width=0.2, head_length=0.2, fc='red', ec='red', linewidth=2)
    ax2.text(3.2, 0, 'v', fontsize=14, color='blue', fontweight='bold')
    ax2.text(2.2, 2.2, 'w', fontsize=14, color='red', fontweight='bold')
    ax2.text(1.5, 1, 'v · w ≠ 0', fontsize=12, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
    
    from matplotlib.patches import Arc
    arc = Arc((0, 0), 1.5, 1.5, angle=0, theta1=0, theta2=45, color='green', linewidth=2)
    ax2.add_patch(arc)
    ax2.text(0.8, 0.3, '45°', fontsize=10, color='green', fontweight='bold')
    
    ax3 = fig.add_subplot(2, 4, 3)
    zero_state = np.array([1, 0])
    one_state = np.array([0, 1])
    
    width = 0.35
    x = np.arange(2)
    
    ax3.bar(x - width/2, zero_state, width, label='|0⟩', color='blue', alpha=0.7)
    ax3.bar(x + width/2, one_state, width, label='|1⟩', color='red', alpha=0.7)
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Computational Basis (Orthogonal)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Component 0', 'Component 1'])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.text(0.5, 0.5, '⟨0|1⟩ = 0', fontsize=12, ha='center',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax4 = fig.add_subplot(2, 4, 4)
    plus_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    minus_state = np.array([1/np.sqrt(2), -1/np.sqrt(2)])
    
    ax4.bar(x - width/2, plus_state, width, label='|+⟩', color='green', alpha=0.7)
    ax4.bar(x + width/2, minus_state, width, label='|-⟩', color='orange', alpha=0.7)
    ax4.set_ylabel('Amplitude')
    ax4.set_title('Hadamard Basis (Orthogonal)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Component 0', 'Component 1'])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax4.text(0.5, 0.3, '⟨+|-⟩ = 0', fontsize=12, ha='center',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax5 = fig.add_subplot(2, 4, 5)
    
    zero = np.array([1, 0])
    one = np.array([0, 1])
    ortho_matrix = np.array([
        [np.vdot(zero, zero), np.vdot(zero, one)],
        [np.vdot(one, zero), np.vdot(one, one)]
    ])
    
    im = ax5.imshow(ortho_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax5.set_xticks([0, 1])
    ax5.set_yticks([0, 1])
    ax5.set_xticklabels(['⟨0|', '⟨1|'])
    ax5.set_yticklabels(['|0⟩', '|1⟩'])
    ax5.set_title('Orthonormality Matrix')
    
    for i in range(2):
        for j in range(2):
            text = ax5.text(j, i, f'{ortho_matrix[i, j]:.0f}',
                           ha="center", va="center", color="black", fontweight='bold', fontsize=14)
    
    plt.colorbar(im, ax=ax5, label='Inner Product')
    
    ax6 = fig.add_subplot(2, 4, 6)
    ax6.set_xlim(-0.5, 2)
    ax6.set_ylim(-0.5, 2)
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_title('Gram-Schmidt Orthogonalization')
    
    v1 = np.array([1, 1])
    v2 = np.array([1, 0])
    
    u1 = v1 / np.linalg.norm(v1)
    projection = np.dot(v2, u1) * u1
    v2_orth = v2 - projection
    u2 = v2_orth / np.linalg.norm(v2_orth)
    
    ax6.arrow(0, 0, v1[0], v1[1], head_width=0.1, head_length=0.1, 
              fc='gray', ec='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax6.arrow(0, 0, v2[0], v2[1], head_width=0.1, head_length=0.1, 
              fc='gray', ec='gray', linewidth=1, linestyle='--', alpha=0.5)
    
    ax6.arrow(0, 0, u1[0], u1[1], head_width=0.1, head_length=0.1, 
              fc='blue', ec='blue', linewidth=2)
    ax6.arrow(0, 0, u2[0], u2[1], head_width=0.1, head_length=0.1, 
              fc='red', ec='red', linewidth=2)
    
    ax6.text(u1[0]+0.1, u1[1]+0.1, 'u₁', fontsize=12, color='blue', fontweight='bold')
    ax6.text(u2[0]+0.1, u2[1]+0.1, 'u₂', fontsize=12, color='red', fontweight='bold')
    
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.set_xlim(-0.5, 5)
    ax7.set_ylim(-0.5, 5)
    ax7.set_aspect('equal')
    ax7.grid(True, alpha=0.3)
    ax7.set_xlabel('x')
    ax7.set_ylabel('y')
    ax7.set_title('Pythagorean Theorem')
    
    v = np.array([3, 0])
    w = np.array([0, 4])
    v_plus_w = v + w
    
    ax7.arrow(0, 0, v[0], v[1], head_width=0.2, head_length=0.2, 
              fc='blue', ec='blue', linewidth=2, label='v')
    ax7.arrow(v[0], v[1], w[0], w[1], head_width=0.2, head_length=0.2, 
              fc='red', ec='red', linewidth=2, label='w')
    ax7.arrow(0, 0, v_plus_w[0], v_plus_w[1], head_width=0.2, head_length=0.2, 
              fc='green', ec='green', linewidth=2, linestyle='--', label='v+w')
    
    ax7.plot([0, v[0], v_plus_w[0], 0], [0, v[1], v_plus_w[1], 0], 'k--', alpha=0.3)
    
    ax7.text(1.5, -0.5, '||v|| = 3', fontsize=10, color='blue', fontweight='bold')
    ax7.text(3.5, 2, '||w|| = 4', fontsize=10, color='red', fontweight='bold')
    ax7.text(1, 2.5, '||v+w|| = 5', fontsize=10, color='green', fontweight='bold')
    ax7.text(1.5, 1.5, '3² + 4² = 5²', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax8 = fig.add_subplot(2, 4, 8)
    
    plus = (1/np.sqrt(2)) * np.array([1, 1])
    zero = np.array([1, 0])
    one = np.array([0, 1])
    
    prob_0 = np.abs(np.vdot(zero, plus))**2
    prob_1 = np.abs(np.vdot(one, plus))**2
    
    probs = [prob_0, prob_1]
    labels = ['P(0)', 'P(1)']
    colors = ['blue', 'red']
    
    bars = ax8.bar(labels, probs, color=colors, alpha=0.7)
    ax8.set_ylabel('Probability')
    ax8.set_title('Measuring |+⟩ in Computational Basis')
    ax8.set_ylim([0, 1])
    ax8.grid(True, alpha=0.3, axis='y')
    
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax8.text(0.5, 0.8, '|+⟩ = (|0⟩ + |1⟩)/√2', fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    return fig


def main():
    """Main function to run all demonstrations"""
    print("\n" + "=" * 60)
    print("ORTHOGONALITY IN QUANTUM COMPUTING")
    print("Comprehensive Demonstrations")
    print("=" * 60)
    
    basic_orthogonality()
    orthonormality_check()
    classical_orthogonality()
    complex_orthogonality()
    pythagorean_theorem()
    gram_schmidt()
    quantum_measurement_orthogonality()
    state_distinguishability()
    
    print("\n" + "=" * 60)
    print("Creating visualizations...")
    print("=" * 60)
    
    fig = visualize_orthogonality()
    plt.savefig('orthogonality_visualization.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'orthogonality_visualization.png'")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
