#!/usr/bin/env python3
"""
Example: Green's Function DOS and LDOS for Graphene.

This example demonstrates the Green's function implementation in pysktb
by computing the density of states and local DOS for graphene.

Features demonstrated:
- Total DOS via Green's function
- LDOS resolved by sublattice (A vs B atoms)
- LDOS resolved by orbital

Author: Santosh Kumar Radha
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pysktb import Structure, Atom, Lattice, Hamiltonian, GreensFunction

# Set up publication-quality plot style
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.dpi'] = 150

# Color palette
COLORS = {
    'primary': '#2563eb',    # Blue
    'secondary': '#dc2626',  # Red
    'accent': '#059669',     # Green
    'gray': '#6b7280',
}


def create_graphene():
    """
    Create graphene structure with pz orbitals.

    Graphene has a honeycomb lattice with two atoms per unit cell (A and B).
    The characteristic linear dispersion at K points creates the famous
    Dirac cone and V-shaped DOS.
    """
    # Lattice constant
    a = 2.46  # Angstroms

    # Honeycomb lattice vectors
    lattice_matrix = [[1, 0, 0],
                      [0.5, np.sqrt(3)/2, 0],
                      [0, 0, 10]]  # Large c for 2D
    lattice = Lattice(lattice_matrix, a)

    # Two atoms: A at origin, B at (1/3, 1/3)
    atom_A = Atom("C", [0, 0, 0], orbitals=["pz"])
    atom_B = Atom("C", [1/3, 1/3, 0], orbitals=["pz"])

    # Nearest neighbor distance
    d_nn = a / np.sqrt(3)
    bond_cut = {"CC": {"NN": d_nn * 1.1}}

    structure = Structure(
        lattice, [atom_A, atom_B],
        periodicity=[True, True, False],
        bond_cut=bond_cut
    )

    return structure


def get_graphene_parameters():
    """
    Return tight-binding parameters for graphene.

    The single pz orbital model with nearest-neighbor hopping t ~ 2.7 eV
    reproduces the essential physics of graphene's band structure.
    """
    params = {
        "C": {
            "e_p": 0.0,  # On-site energy (set to zero as reference)
        },
        "CC": {
            "V_ppp": -2.7,  # pi-bonding between pz orbitals
        }
    }
    return params


def main():
    """Main function demonstrating Green's function calculations."""

    print("=" * 60)
    print("Green's Function DOS Example: Graphene")
    print("=" * 60)

    # Create structure and Hamiltonian
    print("\n1. Setting up graphene structure...")
    structure = create_graphene()
    params = get_graphene_parameters()
    ham = Hamiltonian(structure, params, numba=False)

    print(f"   - Atoms per cell: {len(structure.atoms)}")
    print(f"   - Orbitals: {[a.orbitals for a in structure.atoms]}")
    print(f"   - Hopping t = {params['CC']['V_ppp']} eV")

    # Create Green's function calculator
    print("\n2. Initializing Green's function calculator...")
    gf = GreensFunction(ham)

    # Energy grid
    energies = np.linspace(-8, 8, 400)

    # Compute total DOS
    print("\n3. Computing total DOS (this may take a moment)...")
    dos = gf.dos(energies, nk=[30, 30, 1], eta=0.1, soc=False, parallel=True)
    print(f"   - Energy range: [{energies.min():.1f}, {energies.max():.1f}] eV")
    print(f"   - DOS computed at {len(energies)} energy points")

    # Compute LDOS by sublattice
    print("\n4. Computing LDOS by sublattice...")
    ldos = gf.ldos(energies, atom_indices=[0, 1], nk=[30, 30, 1],
                   eta=0.1, soc=False, parallel=True)
    ldos_A = ldos[0]  # Sublattice A
    ldos_B = ldos[1]  # Sublattice B

    # Create figure with subplots
    print("\n5. Creating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    # Plot 1: Total DOS
    ax1 = axes[0, 0]
    ax1.fill_between(energies, 0, dos, alpha=0.3, color=COLORS['primary'])
    ax1.plot(energies, dos, color=COLORS['primary'], linewidth=1.5)
    ax1.axvline(x=0, color=COLORS['secondary'], linestyle='--', linewidth=1.5, alpha=0.8, label='$E_F$')
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('DOS (states/eV)')
    ax1.set_title('Total DOS — Graphene')
    ax1.set_xlim(energies.min(), energies.max())
    ax1.set_ylim(0, None)
    ax1.legend(frameon=True, fancybox=True, shadow=True)

    # Plot 2: LDOS by sublattice
    ax2 = axes[0, 1]
    ax2.plot(energies, ldos_A, color=COLORS['primary'], linewidth=1.5, label='Sublattice A')
    ax2.plot(energies, ldos_B, color=COLORS['secondary'], linewidth=1.5, linestyle='--', label='Sublattice B')
    ax2.axvline(x=0, color=COLORS['gray'], linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Energy (eV)')
    ax2.set_ylabel('LDOS (states/eV)')
    ax2.set_title('LDOS by Sublattice')
    ax2.set_xlim(energies.min(), energies.max())
    ax2.set_ylim(0, None)
    ax2.legend(frameon=True, fancybox=True, shadow=True)

    # Plot 3: DOS zoomed near Dirac point
    ax3 = axes[1, 0]
    mask = (energies > -3) & (energies < 3)
    ax3.fill_between(energies[mask], 0, dos[mask], alpha=0.3, color=COLORS['accent'])
    ax3.plot(energies[mask], dos[mask], color=COLORS['accent'], linewidth=1.5)
    ax3.axvline(x=0, color=COLORS['secondary'], linestyle='--', linewidth=1.5, alpha=0.8, label='Dirac point')
    ax3.set_xlabel('Energy (eV)')
    ax3.set_ylabel('DOS (states/eV)')
    ax3.set_title('DOS near Dirac Point\n(Linear dispersion → V-shaped DOS)')
    ax3.set_ylim(0, None)
    ax3.legend(frameon=True, fancybox=True, shadow=True)

    # Plot 4: Band structure for reference
    ax4 = axes[1, 1]
    k_path = [
        [0, 0, 0],           # Gamma
        [1/2, 0, 0],         # M
        [1/3, 1/3, 0],       # K
        [0, 0, 0],           # Gamma
    ]
    kpts, kpts_dist, spl_pnts = ham.get_kpts(k_path, nk=50)
    eigen_vals = ham.solve_kpath(kpts, eig_vectors=False, soc=False, parallel=0)

    for i in range(eigen_vals.shape[0]):
        ax4.plot(kpts_dist, eigen_vals[i, :], color=COLORS['primary'], linewidth=2)

    ax4.axhline(y=0, color=COLORS['secondary'], linestyle='--', linewidth=1.5, alpha=0.8)
    labels = ['Γ', 'M', 'K', 'Γ']
    for sp in spl_pnts:
        ax4.axvline(x=sp, color=COLORS['gray'], linestyle='-', linewidth=0.8, alpha=0.5)
    ax4.set_xticks(spl_pnts)
    ax4.set_xticklabels(labels, fontsize=12)
    ax4.set_xlabel('Wave Vector')
    ax4.set_ylabel('Energy (eV)')
    ax4.set_title('Band Structure\n(Dirac cone at K point)')
    ax4.set_xlim(kpts_dist[0], kpts_dist[-1])
    ax4.set_ylim(-8, 8)

    # Save figure
    output_file = os.path.join(os.path.dirname(__file__), "data", "greens_dos_graphene.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n   Plot saved to: {output_file}")

    print("\n" + "=" * 60)
    print("Key Physics Features:")
    print("-" * 40)
    print("1. V-shaped DOS near Fermi level (linear dispersion)")
    print("2. Van Hove singularities at band edges (~±3t)")
    print("3. Identical LDOS on A and B sublattices (symmetry)")
    print("4. Zero DOS at Dirac point (E=0)")
    print("=" * 60)

    plt.show()

    return dos, ldos, ham


if __name__ == "__main__":
    dos, ldos, ham = main()
