#!/usr/bin/env python3
"""
Linear Independence in Quantum Computing
=========================================

This module demonstrates the concept of linear independence in the context
of quantum computing, including:
- Definition of linear independence
- Testing for linear independence
- Linear dependence examples
- Basis and span
- Multi-qubit independence
- Applications in quantum computing
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import sys

np.random.seed(42)


class Arrow3D(FancyArrowPatch):
    """Helper class for drawing 3D arrows in matplotlib."""
    
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def basic_linear_independence():
    """Demonstrate basic concept of linear independence."""
    print_section("Basic Linear Independence")
    
    print("\nDefinition:")
    print("Vectors v₁, v₂, ..., vₙ are linearly independent if:")
    print("  c₁v₁ + c₂v₂ + ... + cₙvₙ = 0")
    print("  implies c₁ = c₂ = ... = cₙ = 0")
    print("\nIn other words, no vector can be written as a linear")
    print("combination of the others.")
    
    print("\n" + "-" * 70)
    print("Example 1: Independent vectors in 2D")
    print("-" * 70)
    
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    
    print(f"\nv₁ = {v1}")
    print(f"v₂ = {v2}")
    
    print("\nCan we write v₂ = c·v₁ for some scalar c?")
    print("  [0, 1] = c·[1, 0]")
    print("  This would require: 0 = c·1 and 1 = c·0")
    print("  No value of c satisfies both equations.")
    print("  Therefore, v₁ and v₂ are linearly independent.")
    
    print("\n" + "-" * 70)
    print("Example 2: Dependent vectors in 2D")
    print("-" * 70)
    
    v1 = np.array([1, 2])
    v2 = np.array([2, 4])
    
    print(f"\nv₁ = {v1}")
    print(f"v₂ = {v2}")
    
    print("\nCan we write v₂ = c·v₁ for some scalar c?")
    print("  [2, 4] = c·[1, 2]")
    print("  This gives: 2 = c·1 and 4 = c·2")
    print("  Both equations are satisfied by c = 2.")
    print("  Therefore, v₂ = 2v₁, so v₁ and v₂ are linearly dependent.")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    v1_ind = np.array([1, 0])
    v2_ind = np.array([0, 1])
    
    ax1.quiver(0, 0, v1_ind[0], v1_ind[1], angles='xy', scale_units='xy', scale=1,
               color='blue', width=0.01, label='v₁')
    ax1.quiver(0, 0, v2_ind[0], v2_ind[1], angles='xy', scale_units='xy', scale=1,
               color='red', width=0.01, label='v₂')
    
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.legend()
    ax1.set_title('Linearly Independent Vectors\n(point in different directions)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    v1_dep = np.array([1, 2])
    v2_dep = np.array([2, 4])
    
    ax2.quiver(0, 0, v1_dep[0], v1_dep[1], angles='xy', scale_units='xy', scale=1,
               color='blue', width=0.008, label='v₁')
    ax2.quiver(0, 0, v2_dep[0], v2_dep[1], angles='xy', scale_units='xy', scale=1,
               color='red', width=0.008, label='v₂ = 2v₁')
    
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(-0.5, 5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.legend()
    ax2.set_title('Linearly Dependent Vectors\n(point in same direction)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/09-linear-independence/basic_independence.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: basic_independence.png")
    plt.close()


def matrix_rank_method():
    """Demonstrate testing linear independence using matrix rank."""
    print_section("Testing Linear Independence: Matrix Rank Method")
    
    print("\nMethod: Form a matrix with vectors as columns, compute rank.")
    print("If rank equals number of vectors, they are independent.")
    
    print("\n" + "-" * 70)
    print("Example 1: Three independent vectors in 3D")
    print("-" * 70)
    
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([0, 0, 1])
    
    A = np.column_stack([v1, v2, v3])
    rank = np.linalg.matrix_rank(A)
    
    print(f"\nv₁ = {v1}")
    print(f"v₂ = {v2}")
    print(f"v₃ = {v3}")
    print(f"\nMatrix A = [v₁ v₂ v₃]:")
    print(A)
    print(f"\nRank(A) = {rank}")
    print(f"Number of vectors = {A.shape[1]}")
    print(f"\nSince rank = number of vectors, they are linearly independent.")
    
    print("\n" + "-" * 70)
    print("Example 2: Three dependent vectors in 3D")
    print("-" * 70)
    
    v1 = np.array([1, 2, 3])
    v2 = np.array([2, 4, 6])
    v3 = np.array([3, 6, 9])
    
    A = np.column_stack([v1, v2, v3])
    rank = np.linalg.matrix_rank(A)
    
    print(f"\nv₁ = {v1}")
    print(f"v₂ = {v2} = 2v₁")
    print(f"v₃ = {v3} = 3v₁")
    print(f"\nMatrix A = [v₁ v₂ v₃]:")
    print(A)
    print(f"\nRank(A) = {rank}")
    print(f"Number of vectors = {A.shape[1]}")
    print(f"\nSince rank < number of vectors, they are linearly dependent.")
    
    print("\n" + "-" * 70)
    print("Example 3: Four vectors in 3D space")
    print("-" * 70)
    
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([0, 0, 1])
    v4 = np.array([1, 1, 1])
    
    A = np.column_stack([v1, v2, v3, v4])
    rank = np.linalg.matrix_rank(A)
    
    print(f"\nv₁ = {v1}")
    print(f"v₂ = {v2}")
    print(f"v₃ = {v3}")
    print(f"v₄ = {v4}")
    print(f"\nMatrix A = [v₁ v₂ v₃ v₄]:")
    print(A)
    print(f"\nRank(A) = {rank}")
    print(f"Number of vectors = {A.shape[1]}")
    print(f"\nSince rank < number of vectors, they are linearly dependent.")
    print("Note: v₄ = v₁ + v₂ + v₃")
    print("\nGeneral principle: In n-dimensional space, you cannot have")
    print("more than n linearly independent vectors.")


def determinant_method():
    """Demonstrate testing linear independence using determinant."""
    print_section("Testing Linear Independence: Determinant Method")
    
    print("\nFor n vectors in n-dimensional space:")
    print("Form a square matrix with vectors as columns.")
    print("If det(A) ≠ 0, vectors are linearly independent.")
    print("If det(A) = 0, vectors are linearly dependent.")
    
    print("\n" + "-" * 70)
    print("Example 1: Independent vectors")
    print("-" * 70)
    
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([0, 0, 1])
    
    A = np.column_stack([v1, v2, v3])
    det = np.linalg.det(A)
    
    print(f"\nv₁ = {v1}")
    print(f"v₂ = {v2}")
    print(f"v₃ = {v3}")
    print(f"\nMatrix A:")
    print(A)
    print(f"\ndet(A) = {det:.6f}")
    print(f"\nSince det(A) ≠ 0, vectors are linearly independent.")
    
    print("\n" + "-" * 70)
    print("Example 2: Dependent vectors")
    print("-" * 70)
    
    v1 = np.array([1, 2, 3])
    v2 = np.array([2, 4, 6])
    v3 = np.array([0, 0, 1])
    
    A = np.column_stack([v1, v2, v3])
    det = np.linalg.det(A)
    
    print(f"\nv₁ = {v1}")
    print(f"v₂ = {v2} = 2v₁")
    print(f"v₃ = {v3}")
    print(f"\nMatrix A:")
    print(A)
    print(f"\ndet(A) = {det:.6f}")
    print(f"\nSince det(A) = 0, vectors are linearly dependent.")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    vectors_ind = [
        ([1, 0, 0], 'blue', 'v₁'),
        ([0, 1, 0], 'red', 'v₂'),
        ([0, 0, 1], 'green', 'v₃')
    ]
    
    ax1 = plt.subplot(121, projection='3d')
    for vec, color, label in vectors_ind:
        arrow = Arrow3D([0, vec[0]], [0, vec[1]], [0, vec[2]],
                       mutation_scale=20, lw=2, arrowstyle="-|>", color=color, label=label)
        ax1.add_artist(arrow)
    
    ax1.set_xlim([0, 1.2])
    ax1.set_ylim([0, 1.2])
    ax1.set_zlim([0, 1.2])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    ax1.set_title('Independent Vectors\ndet(A) ≠ 0', fontsize=12, fontweight='bold')
    
    vectors_dep = [
        ([1, 2, 3], 'blue', 'v₁'),
        ([2, 4, 6], 'red', 'v₂ = 2v₁'),
        ([0, 0, 1], 'green', 'v₃')
    ]
    
    ax2 = plt.subplot(122, projection='3d')
    for vec, color, label in vectors_dep:
        arrow = Arrow3D([0, vec[0]], [0, vec[1]], [0, vec[2]],
                       mutation_scale=20, lw=2, arrowstyle="-|>", color=color, label=label)
        ax2.add_artist(arrow)
    
    ax2.set_xlim([0, 2.5])
    ax2.set_ylim([0, 5])
    ax2.set_zlim([0, 7])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    ax2.set_title('Dependent Vectors\ndet(A) = 0', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/09-linear-independence/determinant_method.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: determinant_method.png")
    plt.close()


def quantum_basis_independence():
    """Demonstrate linear independence in quantum computing bases."""
    print_section("Linear Independence in Quantum Bases")
    
    print("\nQuantum computing relies on linearly independent basis states.")
    
    print("\n" + "-" * 70)
    print("Computational Basis")
    print("-" * 70)
    
    ket0 = np.array([1, 0], dtype=complex)
    ket1 = np.array([0, 1], dtype=complex)
    
    print(f"\n|0⟩ = {ket0}")
    print(f"|1⟩ = {ket1}")
    
    A = np.column_stack([ket0, ket1])
    rank = np.linalg.matrix_rank(A)
    det = np.linalg.det(A)
    
    print(f"\nMatrix A = [|0⟩ |1⟩]:")
    print(A)
    print(f"\nRank(A) = {rank}")
    print(f"det(A) = {det}")
    print(f"\n|0⟩ and |1⟩ are linearly independent.")
    print("They form a basis for the 2D Hilbert space of a qubit.")
    
    print("\n" + "-" * 70)
    print("Hadamard Basis")
    print("-" * 70)
    
    ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    ket_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)
    
    print(f"\n|+⟩ = (|0⟩ + |1⟩)/√2 = {ket_plus}")
    print(f"|−⟩ = (|0⟩ − |1⟩)/√2 = {ket_minus}")
    
    A = np.column_stack([ket_plus, ket_minus])
    rank = np.linalg.matrix_rank(A)
    det = np.linalg.det(A)
    
    print(f"\nMatrix A = [|+⟩ |−⟩]:")
    print(A)
    print(f"\nRank(A) = {rank}")
    print(f"det(A) = {det:.6f}")
    print(f"\n|+⟩ and |−⟩ are linearly independent.")
    print("They also form a basis for the qubit Hilbert space.")
    
    inner_product_pp = np.vdot(ket_plus, ket_plus)
    inner_product_mm = np.vdot(ket_minus, ket_minus)
    inner_product_pm = np.vdot(ket_plus, ket_minus)
    
    print(f"\nOrthonormality check:")
    print(f"⟨+|+⟩ = {inner_product_pp:.6f}")
    print(f"⟨−|−⟩ = {inner_product_mm:.6f}")
    print(f"⟨+|−⟩ = {inner_product_pm:.6f}")
    print(f"\nBoth bases are orthonormal (independent + normalized).")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.quiver(0, 0, ket0[0].real, ket0[1].real, angles='xy', scale_units='xy', scale=1,
               color='blue', width=0.01, label='|0⟩')
    ax1.quiver(0, 0, ket1[0].real, ket1[1].real, angles='xy', scale_units='xy', scale=1,
               color='red', width=0.01, label='|1⟩')
    
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)
    
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.legend()
    ax1.set_title('Computational Basis\n{|0⟩, |1⟩}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('Component 2')
    
    ax2.quiver(0, 0, ket_plus[0].real, ket_plus[1].real, angles='xy', scale_units='xy', scale=1,
               color='green', width=0.01, label='|+⟩')
    ax2.quiver(0, 0, ket_minus[0].real, ket_minus[1].real, angles='xy', scale_units='xy', scale=1,
               color='purple', width=0.01, label='|−⟩')
    
    ax2.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)
    
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.legend()
    ax2.set_title('Hadamard Basis\n{|+⟩, |−⟩}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/09-linear-independence/quantum_bases.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: quantum_bases.png")
    plt.close()


def span_and_basis():
    """Demonstrate the relationship between span and basis."""
    print_section("Span and Basis")
    
    print("\nSpan: The set of all linear combinations of vectors.")
    print("Basis: A linearly independent set that spans the space.")
    
    print("\n" + "-" * 70)
    print("Example: 2D Space")
    print("-" * 70)
    
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    
    print(f"\nv₁ = {v1}")
    print(f"v₂ = {v2}")
    
    print(f"\nSpan(v₁, v₂) = all vectors of the form c₁v₁ + c₂v₂")
    print(f"             = all vectors [c₁, c₂]")
    print(f"             = entire 2D space ℝ²")
    
    print(f"\nSince v₁ and v₂ are:")
    print(f"  1. Linearly independent (rank = 2)")
    print(f"  2. Span the entire 2D space")
    print(f"They form a basis for ℝ².")
    
    print(f"\nSome vectors in Span(v₁, v₂):")
    for i in range(5):
        c1, c2 = np.random.uniform(-2, 2, 2)
        v = c1 * v1 + c2 * v2
        print(f"  {c1:.2f}v₁ + {c2:.2f}v₂ = [{v[0]:.2f}, {v[1]:.2f}]")
    
    print("\n" + "-" * 70)
    print("Example: Insufficient Vectors")
    print("-" * 70)
    
    v1 = np.array([1, 0])
    
    print(f"\nv₁ = {v1}")
    print(f"\nSpan(v₁) = all vectors of the form c₁v₁")
    print(f"         = all vectors [c₁, 0]")
    print(f"         = x-axis only (1D subspace of ℝ²)")
    
    print(f"\nv₁ alone does NOT span ℝ².")
    print(f"It only spans a 1D line.")
    print(f"We need 2 independent vectors to span ℝ².")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    
    ax1.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,
               color='blue', width=0.01, label='v₁', zorder=3)
    ax1.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1,
               color='red', width=0.01, label='v₂', zorder=3)
    
    np.random.seed(42)
    for i in range(20):
        c1, c2 = np.random.uniform(-2, 2, 2)
        v = c1 * v1 + c2 * v2
        ax1.plot(v[0], v[1], 'go', alpha=0.3, markersize=4)
    
    ax1.axhspan(-2.5, 2.5, alpha=0.1, color='green')
    
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.legend()
    ax1.set_title('Basis Spans Entire 2D Space\nSpan(v₁, v₂) = ℝ²', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    v1 = np.array([1, 0])
    
    ax2.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,
               color='blue', width=0.01, label='v₁', zorder=3)
    
    for i in range(20):
        c1 = np.random.uniform(-2, 2)
        v = c1 * v1
        ax2.plot(v[0], v[1], 'ro', alpha=0.5, markersize=4)
    
    ax2.axhline(y=0, color='red', linewidth=2, alpha=0.3, label='Span(v₁)')
    
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.legend()
    ax2.set_title('Single Vector Spans 1D Line\nSpan(v₁) = x-axis', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/09-linear-independence/span_and_basis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: span_and_basis.png")
    plt.close()


def multi_qubit_independence():
    """Demonstrate linear independence in multi-qubit systems."""
    print_section("Linear Independence in Multi-Qubit Systems")
    
    print("\nFor n qubits, the Hilbert space has dimension 2ⁿ.")
    print("We need 2ⁿ linearly independent basis states.")
    
    print("\n" + "-" * 70)
    print("Two-Qubit System")
    print("-" * 70)
    
    print("\nComputational basis states:")
    ket00 = np.array([1, 0, 0, 0], dtype=complex)
    ket01 = np.array([0, 1, 0, 0], dtype=complex)
    ket10 = np.array([0, 0, 1, 0], dtype=complex)
    ket11 = np.array([0, 0, 0, 1], dtype=complex)
    
    print(f"|00⟩ = {ket00}")
    print(f"|01⟩ = {ket01}")
    print(f"|10⟩ = {ket10}")
    print(f"|11⟩ = {ket11}")
    
    A = np.column_stack([ket00, ket01, ket10, ket11])
    rank = np.linalg.matrix_rank(A)
    det = np.linalg.det(A)
    
    print(f"\nMatrix A = [|00⟩ |01⟩ |10⟩ |11⟩]:")
    print(A)
    print(f"\nRank(A) = {rank}")
    print(f"det(A) = {det}")
    print(f"\nAll 4 basis states are linearly independent.")
    print(f"They form a basis for the 4D Hilbert space of 2 qubits.")
    
    print("\n" + "-" * 70)
    print("Bell States (Alternative Basis)")
    print("-" * 70)
    
    bell_00 = (ket00 + ket11) / np.sqrt(2)
    bell_01 = (ket00 - ket11) / np.sqrt(2)
    bell_10 = (ket01 + ket10) / np.sqrt(2)
    bell_11 = (ket01 - ket10) / np.sqrt(2)
    
    print(f"\n|Φ⁺⟩ = (|00⟩ + |11⟩)/√2 = {bell_00}")
    print(f"|Φ⁻⟩ = (|00⟩ − |11⟩)/√2 = {bell_01}")
    print(f"|Ψ⁺⟩ = (|01⟩ + |10⟩)/√2 = {bell_10}")
    print(f"|Ψ⁻⟩ = (|01⟩ − |10⟩)/√2 = {bell_11}")
    
    A = np.column_stack([bell_00, bell_01, bell_10, bell_11])
    rank = np.linalg.matrix_rank(A)
    det = np.linalg.det(A)
    
    print(f"\nMatrix A = [|Φ⁺⟩ |Φ⁻⟩ |Ψ⁺⟩ |Ψ⁻⟩]:")
    print(A)
    print(f"\nRank(A) = {rank}")
    print(f"det(A) = {det:.6f}")
    print(f"\nBell states are also linearly independent.")
    print(f"They form an alternative basis for 2-qubit systems.")
    
    n_qubits = np.arange(1, 6)
    dimensions = 2 ** n_qubits
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(n_qubits, dimensions, color='steelblue', alpha=0.7, edgecolor='black')
    
    for i, (n, d) in enumerate(zip(n_qubits, dimensions)):
        ax.text(n, d + 1, f'{d}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Number of Qubits', fontsize=12, fontweight='bold')
    ax.set_ylabel('Hilbert Space Dimension (2ⁿ)', fontsize=12, fontweight='bold')
    ax.set_title('Hilbert Space Dimension vs Number of Qubits\n(Number of independent basis states needed)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(n_qubits)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/09-linear-independence/multi_qubit_dimension.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: multi_qubit_dimension.png")
    plt.close()


def applications_in_quantum_computing():
    """Demonstrate applications of linear independence in quantum computing."""
    print_section("Applications in Quantum Computing")
    
    print("\n1. Basis Selection")
    print("   - Any set of linearly independent states can serve as a basis")
    print("   - Different bases are useful for different algorithms")
    print("   - Examples: computational, Hadamard, Bell bases")
    
    print("\n2. State Representation")
    print("   - Any quantum state can be uniquely written in a given basis")
    print("   - |ψ⟩ = c₁|b₁⟩ + c₂|b₂⟩ + ... + cₙ|bₙ⟩")
    print("   - Coefficients are unique if basis is linearly independent")
    
    print("\n3. Measurement")
    print("   - Measurement outcomes correspond to basis states")
    print("   - Basis states must be linearly independent")
    print("   - Ensures mutually exclusive outcomes")
    
    print("\n4. Quantum Gates")
    print("   - Quantum gates preserve linear independence")
    print("   - Unitary transformations map independent states to independent states")
    print("   - Essential for maintaining basis structure")
    
    print("\n5. Error Detection")
    print("   - Error syndromes must be linearly independent")
    print("   - Allows unique identification of errors")
    print("   - Critical for quantum error correction")
    
    print("\n" + "-" * 70)
    print("Example: Unique State Representation")
    print("-" * 70)
    
    psi = np.array([0.6, 0.8], dtype=complex)
    psi = psi / np.linalg.norm(psi)  # Normalize
    
    print(f"\nQuantum state: |ψ⟩ = {psi}")
    
    ket0 = np.array([1, 0], dtype=complex)
    ket1 = np.array([0, 1], dtype=complex)
    
    c0 = np.vdot(ket0, psi)
    c1 = np.vdot(ket1, psi)
    
    print(f"\nIn computational basis {|0⟩, |1⟩}:")
    print(f"|ψ⟩ = {c0:.4f}|0⟩ + {c1:.4f}|1⟩")
    
    psi_reconstructed = c0 * ket0 + c1 * ket1
    print(f"\nVerification: {psi_reconstructed}")
    print(f"Match: {np.allclose(psi, psi_reconstructed)}")
    
    ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    ket_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)
    
    c_plus = np.vdot(ket_plus, psi)
    c_minus = np.vdot(ket_minus, psi)
    
    print(f"\nIn Hadamard basis {|+⟩, |−⟩}:")
    print(f"|ψ⟩ = {c_plus:.4f}|+⟩ + {c_minus:.4f}|−⟩")
    
    psi_reconstructed = c_plus * ket_plus + c_minus * ket_minus
    print(f"\nVerification: {psi_reconstructed}")
    print(f"Match: {np.allclose(psi, psi_reconstructed)}")
    
    print(f"\nSame state, different representations!")
    print(f"Both are valid because both bases are linearly independent.")


def visualize_independence_summary():
    """Create a comprehensive summary visualization."""
    print_section("Summary Visualization")
    
    fig = plt.figure(figsize=(16, 10))
    
    ax1 = plt.subplot(2, 3, 1)
    v1_ind = np.array([1, 0])
    v2_ind = np.array([0, 1])
    ax1.quiver(0, 0, v1_ind[0], v1_ind[1], angles='xy', scale_units='xy', scale=1,
               color='blue', width=0.015, label='v₁')
    ax1.quiver(0, 0, v2_ind[0], v2_ind[1], angles='xy', scale_units='xy', scale=1,
               color='red', width=0.015, label='v₂')
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Independent Vectors', fontsize=11, fontweight='bold')
    
    ax2 = plt.subplot(2, 3, 2)
    v1_dep = np.array([1, 2])
    v2_dep = np.array([2, 4])
    ax2.quiver(0, 0, v1_dep[0], v1_dep[1], angles='xy', scale_units='xy', scale=1,
               color='blue', width=0.01, label='v₁')
    ax2.quiver(0, 0, v2_dep[0], v2_dep[1], angles='xy', scale_units='xy', scale=1,
               color='red', width=0.01, label='v₂ = 2v₁')
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(-0.5, 5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title('Dependent Vectors', fontsize=11, fontweight='bold')
    
    ax3 = plt.subplot(2, 3, 3)
    categories = ['2 vectors\n(independent)', '2 vectors\n(dependent)', '3 vectors\nin 2D']
    ranks = [2, 1, 2]
    num_vectors = [2, 2, 3]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, ranks, width, label='Rank', color='steelblue', alpha=0.7)
    bars2 = ax3.bar(x + width/2, num_vectors, width, label='# Vectors', color='coral', alpha=0.7)
    
    ax3.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax3.set_title('Rank vs Number of Vectors', fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=9)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax4 = plt.subplot(2, 3, 4)
    ket0 = np.array([1, 0])
    ket1 = np.array([0, 1])
    ax4.quiver(0, 0, ket0[0], ket0[1], angles='xy', scale_units='xy', scale=1,
               color='blue', width=0.015, label='|0⟩')
    ax4.quiver(0, 0, ket1[0], ket1[1], angles='xy', scale_units='xy', scale=1,
               color='red', width=0.015, label='|1⟩')
    theta = np.linspace(0, 2*np.pi, 100)
    ax4.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)
    ax4.set_xlim(-1.2, 1.2)
    ax4.set_ylim(-1.2, 1.2)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_title('Computational Basis', fontsize=11, fontweight='bold')
    
    ax5 = plt.subplot(2, 3, 5)
    ket_plus = np.array([1, 1]) / np.sqrt(2)
    ket_minus = np.array([1, -1]) / np.sqrt(2)
    ax5.quiver(0, 0, ket_plus[0], ket_plus[1], angles='xy', scale_units='xy', scale=1,
               color='green', width=0.015, label='|+⟩')
    ax5.quiver(0, 0, ket_minus[0], ket_minus[1], angles='xy', scale_units='xy', scale=1,
               color='purple', width=0.015, label='|−⟩')
    ax5.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)
    ax5.set_xlim(-1.2, 1.2)
    ax5.set_ylim(-1.2, 1.2)
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_title('Hadamard Basis', fontsize=11, fontweight='bold')
    
    ax6 = plt.subplot(2, 3, 6)
    n_qubits = np.arange(1, 6)
    dimensions = 2 ** n_qubits
    ax6.bar(n_qubits, dimensions, color='steelblue', alpha=0.7, edgecolor='black')
    for n, d in zip(n_qubits, dimensions):
        ax6.text(n, d + 0.5, f'{d}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax6.set_xlabel('Number of Qubits', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Dimension (2ⁿ)', fontsize=10, fontweight='bold')
    ax6.set_title('Hilbert Space Dimension', fontsize=11, fontweight='bold')
    ax6.set_xticks(n_qubits)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_yscale('log')
    
    plt.suptitle('Linear Independence in Quantum Computing - Summary', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/09-linear-independence/independence_summary.png', dpi=150, bbox_inches='tight')
    print("\n✓ Summary visualization saved: independence_summary.png")
    plt.close()


def main():
    """Main function to run all demonstrations."""
    print("\n" + "=" * 70)
    print("  LINEAR INDEPENDENCE IN QUANTUM COMPUTING")
    print("=" * 70)
    print("\nThis module demonstrates linear independence concepts")
    print("and their applications in quantum computing.")
    
    try:
        basic_linear_independence()
        matrix_rank_method()
        determinant_method()
        quantum_basis_independence()
        span_and_basis()
        multi_qubit_independence()
        applications_in_quantum_computing()
        visualize_independence_summary()
        
        print("\n" + "=" * 70)
        print("  ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nVisualizations saved in: /home/ubuntu/repos/vectorsforquantum/09-linear-independence/")
        print("\nKey Takeaways:")
        print("1. Linear independence means no vector is a combination of others")
        print("2. Test using matrix rank or determinant")
        print("3. Quantum bases must be linearly independent")
        print("4. n-dimensional space needs exactly n independent vectors for a basis")
        print("5. Different independent bases provide different perspectives")
        print("6. Multi-qubit systems need 2ⁿ independent basis states")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
