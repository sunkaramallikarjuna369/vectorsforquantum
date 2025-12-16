#!/usr/bin/env python3
"""
Summary - Vectors in Quantum Computing
======================================

This module provides a comprehensive summary of all concepts covered,
including visualizations and quick reference materials.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import sys

np.random.seed(42)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def create_concept_overview():
    """Create a visual overview of all concepts."""
    print_section("Concept Overview")
    
    concepts = [
        "1. Introduction",
        "2. Definition",
        "3. Addition & Subtraction",
        "4. Scalar Multiplication",
        "5. Inner Product",
        "6. Norm & Normalization",
        "7. Basis Vectors",
        "8. Orthogonality",
        "9. Linear Independence",
        "10. Multi-Qubit Systems",
        "11. Transformations",
        "12. Outer Product",
        "13. Bloch Sphere",
        "14. Exercises",
        "15. Summary"
    ]
    
    print("\nAll 15 Concepts Covered:")
    for concept in concepts:
        print(f"  ✓ {concept}")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    ax.text(0.5, 0.95, 'Vectors in Quantum Computing', 
            ha='center', va='top', fontsize=20, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    categories = {
        'Foundations (1-4)': concepts[0:4],
        'Core Concepts (5-9)': concepts[4:9],
        'Advanced Topics (10-13)': concepts[9:13],
        'Practice & Review (14-15)': concepts[13:15]
    }
    
    y_start = 0.85
    y_step = 0.20
    
    colors = ['#FFE5B4', '#B4E5FF', '#FFB4E5', '#B4FFB4']
    
    for idx, (category, items) in enumerate(categories.items()):
        y_pos = y_start - idx * y_step
        
        rect = FancyBboxPatch((0.05, y_pos - 0.15), 0.9, 0.13,
                              boxstyle="round,pad=0.01", 
                              facecolor=colors[idx], 
                              edgecolor='black', 
                              linewidth=2, 
                              alpha=0.6)
        ax.add_patch(rect)
        
        ax.text(0.5, y_pos - 0.02, category, 
                ha='center', va='top', fontsize=14, fontweight='bold')
        
        concept_text = '\n'.join(items)
        ax.text(0.5, y_pos - 0.05, concept_text, 
                ha='center', va='top', fontsize=10, family='monospace')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/15-summary/concept_overview.png', 
                dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: concept_overview.png")
    plt.close()


def create_formula_reference():
    """Create a visual reference of essential formulas."""
    print_section("Essential Formulas")
    
    formulas = {
        'State Representation': [
            '|ψ⟩ = α|0⟩ + β|1⟩',
            '|α|² + |β|² = 1'
        ],
        'Inner Product': [
            '⟨φ|ψ⟩ = Σᵢ φᵢ* ψᵢ',
            '⟨ψ|ψ⟩ = ||ψ||²'
        ],
        'Normalization': [
            '||ψ|| = √⟨ψ|ψ⟩',
            '|ψ_norm⟩ = |ψ⟩/||ψ||'
        ],
        'Measurement': [
            'P(i) = |⟨i|ψ⟩|²',
            'Σᵢ P(i) = 1'
        ],
        'Tensor Product': [
            '|ψ⟩ ⊗ |φ⟩ = |ψφ⟩',
            'dim = 2ⁿ for n qubits'
        ],
        'Bloch Sphere': [
            '|ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩',
            'P(0) = (1+z)/2'
        ]
    }
    
    print("\nKey Formulas:")
    for category, eqs in formulas.items():
        print(f"\n{category}:")
        for eq in eqs:
            print(f"  • {eq}")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    colors = ['#FFE5B4', '#B4E5FF', '#FFB4E5', '#B4FFB4', '#FFD4B4', '#D4B4FF']
    
    for idx, (category, eqs) in enumerate(formulas.items()):
        ax = axes[idx]
        ax.axis('off')
        
        rect = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                              boxstyle="round,pad=0.02", 
                              facecolor=colors[idx], 
                              edgecolor='black', 
                              linewidth=2, 
                              alpha=0.7)
        ax.add_patch(rect)
        
        ax.text(0.5, 0.85, category, 
                ha='center', va='top', fontsize=13, fontweight='bold',
                transform=ax.transAxes)
        
        formula_text = '\n\n'.join(eqs)
        ax.text(0.5, 0.5, formula_text, 
                ha='center', va='center', fontsize=11, family='monospace',
                transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    plt.suptitle('Essential Formulas - Quick Reference', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/15-summary/formula_reference.png', 
                dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: formula_reference.png")
    plt.close()


def create_gate_reference():
    """Create a visual reference of quantum gates."""
    print_section("Quantum Gates Reference")
    
    gates = {
        'X (NOT)': np.array([[0, 1], [1, 0]]),
        'Y': np.array([[0, -1j], [1j, 0]]),
        'Z': np.array([[1, 0], [0, -1]]),
        'H': np.array([[1, 1], [1, -1]]) / np.sqrt(2),
        'S': np.array([[1, 0], [0, 1j]]),
        'T': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
    }
    
    print("\nSingle-Qubit Gates:")
    for name, gate in gates.items():
        print(f"\n{name}:")
        print(gate)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, (name, gate) in enumerate(gates.items()):
        ax = axes[idx]
        
        im = ax.imshow(np.abs(gate), cmap='viridis', vmin=0, vmax=1, aspect='auto')
        ax.set_title(f'{name} Gate', fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['|0⟩', '|1⟩'], fontsize=10)
        ax.set_yticklabels(['⟨0|', '⟨1|'], fontsize=10)
        
        for i in range(2):
            for j in range(2):
                val = gate[i, j]
                if np.abs(val.imag) < 0.01:
                    text = f'{val.real:.2f}'
                else:
                    text = f'{val.real:.2f}\n{val.imag:+.2f}i'
                ax.text(j, i, text, ha="center", va="center", 
                       color="white", fontsize=10, fontweight='bold')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Quantum Gates - Matrix Representations', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/15-summary/gate_reference.png', 
                dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: gate_reference.png")
    plt.close()


def create_state_reference():
    """Create a visual reference of important quantum states."""
    print_section("Important Quantum States")
    
    states = {
        '|0⟩': np.array([1, 0]),
        '|1⟩': np.array([0, 1]),
        '|+⟩': np.array([1, 1]) / np.sqrt(2),
        '|−⟩': np.array([1, -1]) / np.sqrt(2),
        '|+i⟩': np.array([1, 1j]) / np.sqrt(2),
        '|−i⟩': np.array([1, -1j]) / np.sqrt(2)
    }
    
    print("\nCommon Quantum States:")
    for name, state in states.items():
        print(f"\n{name}: {state}")
        prob_0 = np.abs(state[0])**2
        prob_1 = np.abs(state[1])**2
        print(f"  P(0) = {prob_0:.3f}, P(1) = {prob_1:.3f}")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for idx, (name, state) in enumerate(states.items()):
        ax = axes[idx]
        
        labels = ['|0⟩', '|1⟩']
        real_parts = [state[0].real, state[1].real]
        imag_parts = [state[0].imag, state[1].imag]
        
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, real_parts, width, label='Real', 
                      color=colors[idx], alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, imag_parts, width, label='Imag', 
                      color=colors[idx], alpha=0.4, edgecolor='black')
        
        ax.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
        ax.set_title(f'State: {name}', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.legend(fontsize=9)
        ax.set_ylim(-1, 1)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Important Quantum States', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/15-summary/state_reference.png', 
                dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: state_reference.png")
    plt.close()


def create_comprehensive_summary():
    """Create a comprehensive summary visualization."""
    print_section("Comprehensive Summary")
    
    fig = plt.figure(figsize=(18, 12))
    
    ax1 = plt.subplot(2, 3, 1)
    categories = ['Foundations\n(1-4)', 'Core\n(5-9)', 'Advanced\n(10-13)', 'Practice\n(14-15)']
    counts = [4, 5, 4, 2]
    colors_cat = ['#FFE5B4', '#B4E5FF', '#FFB4E5', '#B4FFB4']
    bars = ax1.bar(categories, counts, color=colors_cat, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Number of Concepts', fontsize=11, fontweight='bold')
    ax1.set_title('Concepts by Category', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 6)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, count + 0.1, str(count),
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2 = plt.subplot(2, 3, 2)
    operations = ['Addition', 'Scalar\nMult', 'Inner\nProduct', 'Tensor\nProduct', 'Gates']
    importance = [3, 3, 5, 5, 5]
    bars = ax2.bar(operations, importance, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Importance (1-5)', fontsize=11, fontweight='bold')
    ax2.set_title('Key Vector Operations', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 6)
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax3 = plt.subplot(2, 3, 3)
    qubits = [1, 2, 3, 4, 5]
    dims = [2**n for n in qubits]
    ax3.plot(qubits, dims, 'o-', linewidth=2.5, markersize=10, color='coral')
    ax3.set_xlabel('Number of Qubits', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Hilbert Space Dimension', fontsize=11, fontweight='bold')
    ax3.set_title('Exponential Growth: 2ⁿ', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log', base=2)
    
    for x, y in zip(qubits, dims):
        ax3.text(x, y * 1.2, f'{y}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax4 = plt.subplot(2, 3, 4)
    gate_types = ['Pauli\n(X,Y,Z)', 'Hadamard', 'Phase\n(S,T)', 'CNOT', 'Custom']
    gate_counts = [3, 1, 2, 1, 10]
    colors_gates = ['red', 'green', 'blue', 'purple', 'orange']
    bars = ax4.bar(gate_types, gate_counts, color=colors_gates, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax4.set_title('Quantum Gate Types', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    checklist_text = """
    KEY PROPERTIES ✓
    
    • Normalization: ||ψ|| = 1
    • Orthogonality: ⟨φ|ψ⟩ = 0
    • Unitarity: U†U = I
    • Probability: Σᵢ P(i) = 1
    • Superposition: α|0⟩ + β|1⟩
    • Entanglement: Non-separable
    • Measurement: Born rule
    • Reversibility: Unitary ops
    """
    ax5.text(0.1, 0.5, checklist_text, fontsize=10, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    apps_text = """
    APPLICATIONS
    
    🔹 Quantum Algorithms
       Shor's, Grover's, VQE
    
    🔹 Quantum Communication
       Teleportation, QKD
    
    🔹 Quantum Simulation
       Chemistry, Physics
    
    🔹 Quantum ML
       QNN, QAOA
    
    🔹 Error Correction
       Stabilizer codes
    """
    ax6.text(0.1, 0.5, apps_text, fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Vectors in Quantum Computing - Complete Summary', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/vectorsforquantum/15-summary/comprehensive_summary.png', 
                dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: comprehensive_summary.png")
    plt.close()


def print_key_takeaways():
    """Print key takeaways from all concepts."""
    print_section("Key Takeaways")
    
    takeaways = [
        "1. Quantum states are vectors in complex Hilbert space",
        "2. Normalization ensures valid probability distributions",
        "3. Inner products give measurement probabilities (Born rule)",
        "4. Orthogonal states represent distinguishable outcomes",
        "5. Tensor products describe multi-qubit systems (2ⁿ dimensions)",
        "6. Quantum gates are unitary transformations",
        "7. Superposition enables quantum parallelism",
        "8. Entanglement creates non-classical correlations",
        "9. Bloch sphere visualizes single-qubit states geometrically",
        "10. Density matrices represent mixed states and ensembles"
    ]
    
    print("\nTop 10 Key Takeaways:")
    for takeaway in takeaways:
        print(f"  ✓ {takeaway}")


def main():
    """Main function to run all summary demonstrations."""
    print("\n" + "=" * 70)
    print("  SUMMARY - VECTORS IN QUANTUM COMPUTING")
    print("=" * 70)
    print("\nThis module provides a comprehensive summary of all 15 concepts")
    print("covered in this guide, with visualizations and quick references.")
    
    try:
        create_concept_overview()
        create_formula_reference()
        create_gate_reference()
        create_state_reference()
        create_comprehensive_summary()
        print_key_takeaways()
        
        print("\n" + "=" * 70)
        print("  ALL SUMMARY VISUALIZATIONS COMPLETED")
        print("=" * 70)
        print("\nVisualizations saved in: /home/ubuntu/repos/vectorsforquantum/15-summary/")
        print("\nCongratulations! You have completed all 15 concepts.")
        print("\nNext Steps:")
        print("  1. Review the exercises (Concept 14)")
        print("  2. Explore quantum programming frameworks (Qiskit, Cirq, Q#)")
        print("  3. Study quantum algorithms (Deutsch-Jozsa, Grover, Shor)")
        print("  4. Learn about quantum error correction")
        print("  5. Understand quantum hardware implementations")
        
        print("\n" + "=" * 70)
        print("  THANK YOU FOR LEARNING WITH US!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
