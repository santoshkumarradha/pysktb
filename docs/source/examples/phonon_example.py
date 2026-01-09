#!/usr/bin/env python3
"""
Phonon Calculation Examples
===========================

This script demonstrates phonon calculations in pysktb for:
1. 1D monatomic chain - compared with analytical solution
2. Graphene - full phonon band structure

The analytical solution for 1D chain is:
    ω(q) = 2 * sqrt(K/M) * |sin(π*q)|

where K is the spring constant and M is the atomic mass.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pysktb import (
    Structure, Atom, Lattice, Hamiltonian,
    Harrison, PowerLaw, BornMayer, Morse
)
from pysktb.phonon import Phonon, ATOMIC_MASSES

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_1d_chain():
    """
    Test 1D monatomic chain phonon dispersion.

    For a 1D chain with nearest-neighbor spring constant K and mass M,
    the phonon dispersion is:
        ω(q) = 2 * sqrt(K/M) * |sin(π*q)|

    We use a simple model with:
    - One atom per unit cell
    - Repulsive potential only (no electronic hopping)
    - BornMayer potential: V(r) = A * exp(-B*r)

    The effective spring constant is K = d²V/dr² at equilibrium.
    """
    print("=" * 60)
    print("Test 1: 1D Monatomic Chain Phonons")
    print("=" * 60)

    # 1D chain parameters
    a = 2.5  # lattice constant in Angstrom
    mass = 12.0  # Carbon mass in amu

    # Create 1D chain structure
    lattice = Lattice([[1, 0, 0], [0, 10, 0], [0, 0, 10]], a)
    atom = Atom("C", [0, 0.5, 0.5], orbitals=["s"])

    # Bond cutoff just beyond first neighbor
    bond_cut = {"CC": {"NN": a * 1.1}}

    structure = Structure(
        lattice, [atom],
        periodicity=[True, False, False],
        bond_cut=bond_cut
    )

    # Use repulsive potential only (simpler for testing)
    # BornMayer: V(r) = A * exp(-B*r)
    # V''(r) = A * B² * exp(-B*r) = K at r=a
    A = 100.0  # eV
    B = 2.0    # 1/Angstrom

    params = {
        "C": {"e_s": 0.0},
        "CC": {
            "V_sss": Harrison(V0=-1.0, d0=a, cutoff=a*1.5),  # Small hopping
            "repulsive": BornMayer(A=A, B=B, cutoff=a*1.5)
        }
    }

    ham = Hamiltonian(structure, params, numba=False)

    # Calculate effective spring constant
    K_eff = A * B**2 * np.exp(-B * a)  # d²V/dr² at r=a
    print(f"Lattice constant: a = {a} Å")
    print(f"Atomic mass: M = {mass} amu")
    print(f"Effective spring constant: K = {K_eff:.4f} eV/Å²")

    # Analytical phonon frequency at zone boundary (q=0.5)
    # ω_max = 2 * sqrt(K/M) in appropriate units
    # Need to convert: K in eV/Å², M in amu → ω in THz
    from pysktb.phonon import EIGENVALUE_TO_THZ
    omega_max_analytical = 2 * np.sqrt(K_eff / mass) * EIGENVALUE_TO_THZ
    print(f"Analytical ω_max (zone boundary): {omega_max_analytical:.2f} THz")

    # Compute phonon dispersion
    phonon = Phonon(
        ham,
        masses={"C": mass},
        method='finite_diff',
        nk_fc=[20, 1, 1],
        n_electrons=2,  # 2 electrons for filled s-orbital
        soc=False
    )

    # Generate q-path along chain direction
    q_path = [[0, 0, 0], [0.5, 0, 0]]
    q_points, q_dist, spl = phonon.get_qpath(q_path, nq=30)
    q_points = np.array(q_points)  # Convert to numpy array

    print("Computing phonon dispersion...")
    frequencies = phonon.get_phonon_bands(q_points, parallel=False)

    # Analytical dispersion
    q_values = q_points[:, 0]  # q along x
    omega_analytical = 2 * np.sqrt(K_eff / mass) * EIGENVALUE_TO_THZ * np.abs(np.sin(np.pi * q_values))

    # Plot comparison
    fig, ax = plt.subplots(figsize=(8, 6))

    # Numerical result
    for i in range(frequencies.shape[0]):
        ax.plot(q_dist, frequencies[i, :], 'b-', lw=2, label='Numerical' if i==0 else None)

    # Analytical result
    ax.plot(q_dist, omega_analytical, 'r--', lw=2, label='Analytical: ω = 2√(K/M)|sin(πq)|')

    ax.set_xlabel('q (reduced units)', fontsize=12)
    ax.set_ylabel('Frequency (THz)', fontsize=12)
    ax.set_title('1D Monatomic Chain Phonon Dispersion', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, q_dist[-1])
    ax.grid(True, alpha=0.3)

    # Mark special points
    for sp in spl:
        ax.axvline(sp, color='gray', linestyle=':', alpha=0.5)

    ax.set_xticks(spl)
    ax.set_xticklabels(['Γ', 'X'])

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, "phonon_1d_chain.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

    # Verify acoustic mode at Gamma
    gamma_freqs = phonon.get_gamma_frequencies()
    print(f"Frequencies at Γ: {gamma_freqs}")

    # Check agreement
    numerical_max = np.max(frequencies)
    error = abs(numerical_max - omega_max_analytical) / omega_max_analytical * 100
    print(f"Numerical ω_max: {numerical_max:.2f} THz")
    print(f"Agreement error: {error:.1f}%")

    if error < 20:  # Allow 20% error due to approximations
        print("✓ Test PASSED: 1D chain phonons match analytical result")
    else:
        print("✗ Test needs investigation: large deviation from analytical")

    print()
    return frequencies


def test_graphene():
    """
    Test graphene phonon dispersion.

    Graphene has 2 atoms per unit cell → 6 phonon branches:
    - 3 acoustic (ZA, TA, LA)
    - 3 optical (ZO, TO, LO)

    At Γ point:
    - 3 acoustic modes have ω ≈ 0
    - 3 optical modes have finite frequency

    Characteristic features:
    - ZA branch: quadratic dispersion near Γ (out-of-plane flexural mode)
    - LA/TA: linear dispersion (in-plane acoustic)
    - K point: Kohn anomaly may appear
    """
    print("=" * 60)
    print("Test 2: Graphene Phonon Dispersion")
    print("=" * 60)

    # Graphene structure
    a_cc = 1.42  # C-C bond length in Angstrom
    a = a_cc * np.sqrt(3)  # lattice constant

    lattice_matrix = [
        [1, 0, 0],
        [0.5, np.sqrt(3)/2, 0],
        [0, 0, 10/a]  # Large vacuum for 2D
    ]
    lattice = Lattice(lattice_matrix, a)

    # Two atoms per unit cell
    atom_A = Atom("C", [0, 0, 0.5], orbitals=["pz"])
    atom_B = Atom("C", [1/3, 1/3, 0.5], orbitals=["pz"])

    bond_cut = {"CC": {"NN": a_cc * 1.2}}

    structure = Structure(
        lattice, [atom_A, atom_B],
        periodicity=[True, True, False],
        bond_cut=bond_cut
    )

    print(f"Graphene lattice constant: a = {a:.3f} Å")
    print(f"C-C bond length: {a_cc} Å")
    print(f"Atoms per cell: {len(structure.atoms)}")

    # Parameters with distance-dependent hopping and repulsive potential
    # Using realistic-ish parameters for carbon
    params = {
        "C": {"e_p": 0.0},
        "CC": {
            "V_ppp": Harrison(V0=-2.7, d0=a_cc, cutoff=a*1.5),
            "repulsive": Morse(D=5.0, a=2.0, d0=a_cc, cutoff=a*1.5)
        }
    }

    ham = Hamiltonian(structure, params, numba=False)

    # Phonon calculation
    phonon = Phonon(
        ham,
        masses={"C": 12.011},
        method='finite_diff',
        nk_fc=[8, 8, 1],
        n_electrons=2,
        soc=False
    )

    # High-symmetry path: Γ → M → K → Γ
    G = [0, 0, 0]
    M = [0.5, 0, 0]
    K = [1/3, 1/3, 0]

    q_path = [G, M, K, G]
    q_points, q_dist, spl = phonon.get_qpath(q_path, nq=25)
    q_points = np.array(q_points)  # Convert to numpy array

    print("Computing graphene phonon dispersion...")
    print("(This may take a minute due to force constant calculation)")
    frequencies = phonon.get_phonon_bands(q_points, parallel=False)

    n_modes = frequencies.shape[0]
    print(f"Number of phonon branches: {n_modes}")

    # Gamma point frequencies
    gamma_freqs = phonon.get_gamma_frequencies()
    print(f"Frequencies at Γ point: {gamma_freqs}")

    # Check acoustic modes near zero
    n_acoustic = 3
    acoustic_at_gamma = np.sort(np.abs(gamma_freqs))[:n_acoustic]
    print(f"Acoustic mode frequencies at Γ: {acoustic_at_gamma}")

    # Plot phonon band structure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Color scheme for branches
    colors = plt.cm.viridis(np.linspace(0, 1, n_modes))

    for i in range(n_modes):
        ax.plot(q_dist, frequencies[i, :], '-', color=colors[i], lw=1.5,
                label=f'Branch {i+1}' if i < 3 else None)

    ax.set_xlabel('Wave vector', fontsize=12)
    ax.set_ylabel('Frequency (THz)', fontsize=12)
    ax.set_title('Graphene Phonon Dispersion', fontsize=14)
    ax.set_xlim(0, q_dist[-1])
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)

    # Mark high-symmetry points
    for sp in spl:
        ax.axvline(sp, color='gray', linestyle=':', alpha=0.5)

    labels = ['Γ', 'M', 'K', 'Γ']
    ax.set_xticks(spl)
    ax.set_xticklabels(labels, fontsize=12)

    # Add legend for first 3 branches
    ax.legend(loc='upper right', fontsize=9, title='Phonon branches')

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, "phonon_graphene.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

    # Test: acoustic modes should be near zero at Gamma
    if np.max(acoustic_at_gamma) < 2.0:  # Allow up to 2 THz numerical error
        print("✓ Test PASSED: Acoustic modes are near zero at Γ")
    else:
        print("✗ Test needs investigation: Acoustic modes not near zero at Γ")

    print()
    return frequencies


def test_phonon_dos():
    """
    Test phonon DOS calculation for 1D chain.
    """
    print("=" * 60)
    print("Test 3: Phonon DOS (1D Chain)")
    print("=" * 60)

    # Simple 1D chain
    a = 2.5
    mass = 12.0

    lattice = Lattice([[1, 0, 0], [0, 10, 0], [0, 0, 10]], a)
    atom = Atom("C", [0, 0.5, 0.5], orbitals=["s"])
    bond_cut = {"CC": {"NN": a * 1.1}}

    structure = Structure(
        lattice, [atom],
        periodicity=[True, False, False],
        bond_cut=bond_cut
    )

    params = {
        "C": {"e_s": 0.0},
        "CC": {
            "V_sss": Harrison(V0=-1.0, d0=a, cutoff=a*1.5),
            "repulsive": BornMayer(A=100.0, B=2.0, cutoff=a*1.5)
        }
    }

    ham = Hamiltonian(structure, params, numba=False)

    phonon = Phonon(
        ham,
        masses={"C": mass},
        method='finite_diff',
        nk_fc=[20, 1, 1],
        n_electrons=2,
        soc=False
    )

    # Compute DOS
    omega_range = np.linspace(0, 20, 100)
    print("Computing phonon DOS...")
    dos = phonon.get_phonon_dos(omega_range, nq=[50, 1, 1], sigma=0.5, parallel=False)

    # Plot DOS
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(omega_range, dos, 'b-', lw=2)
    ax.fill_between(omega_range, dos, alpha=0.3)
    ax.set_xlabel('Frequency (THz)', fontsize=12)
    ax.set_ylabel('DOS (states/THz)', fontsize=12)
    ax.set_title('1D Chain Phonon Density of States', fontsize=14)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, "phonon_dos_1d.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

    # 1D DOS should have Van Hove singularities at band edges
    print("✓ Phonon DOS computed successfully")
    print()


def main():
    """Run all phonon tests."""
    print("\n" + "=" * 60)
    print("PHONON CALCULATION TESTS")
    print("=" * 60 + "\n")

    # Run tests
    test_1d_chain()
    test_graphene()
    test_phonon_dos()

    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print(f"Output images saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
