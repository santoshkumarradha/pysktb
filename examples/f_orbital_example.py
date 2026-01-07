#!/usr/bin/env python3
"""
Example: f-orbital tight-binding model for a lanthanide-like system.

This example demonstrates the f-orbital implementation in pysktb by
modeling a simple cubic f-electron system similar to Ce (Cerium) or Sm (Samarium).

The model includes:
- s, p, d, and f orbitals
- Spin-orbit coupling for f orbitals
- Typical f-electron band structure features

Author: Santosh Kumar Radha
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from pysktb import Structure, Atom, Lattice, Hamiltonian


def create_simple_cubic_f_system():
    """
    Create a simple cubic system with f orbitals.

    This models a simplified lanthanide-like material with:
    - One atom per unit cell
    - s + p + f orbitals (simplified basis)
    - Typical f-electron parameters
    """

    # Lattice constant (in Angstroms, typical for lanthanides)
    a = 5.0

    # Simple cubic lattice
    lattice_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    lattice = Lattice(lattice_matrix, a)

    # Single atom at origin with s, p, and f orbitals
    # We use a simplified basis: s + 3 p + 7 f = 11 orbitals
    atom = Atom(
        "Ce",  # Cerium-like atom
        [0, 0, 0],
        orbitals=["s", "px", "py", "pz",
                  "fz3", "fxz2", "fyz2", "fz(x2-y2)", "fxyz", "fx(x2-3y2)", "fy(3x2-y2)"]
    )

    # Bond cutoff for nearest neighbors
    bond_cut = {"CeCe": {"NN": a * 1.1}}

    # Create structure
    structure = Structure(
        lattice, [atom],
        periodicity=[True, True, True],
        bond_cut=bond_cut
    )

    return structure


def get_f_electron_parameters():
    """
    Return tight-binding parameters for f-electron system.

    These are approximate parameters inspired by Ce compounds.
    The key physics includes:
    - Localized f electrons with narrow bandwidth
    - Strong spin-orbit coupling
    - f-s and f-p hybridization
    """

    params = {
        # On-site energies (eV)
        "Ce": {
            "e_s": 2.0,      # s orbital - higher energy, broad band
            "e_p": 0.5,      # p orbitals
            "e_f": -1.0,     # f orbitals - lower energy, localized

            # Spin-orbit coupling (eV)
            # Lambda_f is typically large for lanthanides (0.1-0.3 eV)
            "lambda": 0.05,   # p-orbital SOC
            "lambda_f": 0.25,  # f-orbital SOC (strong for 4f)
        },

        # Hopping parameters (eV)
        "CeCe": {
            # s-s hopping
            "V_sss": -0.8,

            # s-p hopping
            "V_sps": 0.6,

            # p-p hopping
            "V_pps": 0.8,
            "V_ppp": -0.2,

            # s-f hopping (small due to f localization)
            "V_sfs": 0.15,

            # p-f hopping
            "V_pfs": 0.2,
            "V_pfp": -0.05,

            # f-f hopping (small - f electrons are localized)
            "V_ffs": -0.1,
            "V_ffp": 0.02,
            "V_ffd": -0.01,
            "V_fff": 0.005,
        }
    }

    return params


def plot_band_structure(kpts, kpts_dist, eigen_vals, spl_pnts, filename="f_orbital_bands.png"):
    """
    Plot the band structure and save to file.

    Parameters
    ----------
    kpts : list
        K-points along the path
    kpts_dist : array
        Cumulative distance along k-path
    eigen_vals : array
        Eigenvalues at each k-point
    spl_pnts : array
        Special k-point positions
    filename : str
        Output filename
    """

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all bands
    for i in range(eigen_vals.shape[0]):
        ax.plot(kpts_dist, eigen_vals[i, :], 'b-', linewidth=0.8, alpha=0.7)

    # Mark the Fermi level (approximate - assume half-filling of f)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1, label='E_F')

    # Add vertical lines at special k-points
    labels = ['Γ', 'X', 'M', 'Γ', 'R', 'X']
    for i, sp in enumerate(spl_pnts):
        ax.axvline(x=sp, color='gray', linestyle='-', linewidth=0.5)

    # Set labels at special k-points
    ax.set_xticks(spl_pnts)
    ax.set_xticklabels(labels[:len(spl_pnts)])

    ax.set_xlabel('k-path', fontsize=12)
    ax.set_ylabel('Energy (eV)', fontsize=12)
    ax.set_title('f-orbital Tight-Binding Band Structure\n(Ce-like simple cubic system with SOC)', fontsize=14)
    ax.set_xlim(kpts_dist[0], kpts_dist[-1])

    # Set energy range to show key features
    ax.set_ylim(-4, 6)

    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Band structure saved to: {filename}")

    return fig, ax


def main():
    """Main function to run the f-orbital example."""

    print("=" * 60)
    print("PySKTB f-orbital Example: Lanthanide-like Band Structure")
    print("=" * 60)

    # Create the system
    print("\n1. Creating simple cubic f-electron structure...")
    structure = create_simple_cubic_f_system()
    print(f"   - Lattice constant: {structure.lattice.a:.2f} Å")
    print(f"   - Atoms: {len(structure.atoms)}")
    print(f"   - Orbitals per atom: {len(structure.atoms[0].orbitals)}")
    print(f"   - Orbital basis: {structure.atoms[0].orbitals}")

    # Get parameters
    print("\n2. Setting up tight-binding parameters...")
    params = get_f_electron_parameters()
    print(f"   - On-site energies: e_s={params['Ce']['e_s']}, e_p={params['Ce']['e_p']}, e_f={params['Ce']['e_f']} eV")
    print(f"   - f-orbital SOC: λ_f={params['Ce']['lambda_f']} eV")

    # Create Hamiltonian
    print("\n3. Building Hamiltonian...")
    ham = Hamiltonian(structure, params, numba=False)
    print(f"   - Total orbitals: {ham.n_orbitals}")
    print(f"   - Hamiltonian size (with spin): {ham.n_orbitals * 2} × {ham.n_orbitals * 2}")

    # Define k-path through high-symmetry points
    # Simple cubic BZ: Γ(0,0,0) - X(0.5,0,0) - M(0.5,0.5,0) - Γ - R(0.5,0.5,0.5) - X
    print("\n4. Generating k-path...")
    k_path = [
        [0.0, 0.0, 0.0],    # Γ
        [0.5, 0.0, 0.0],    # X
        [0.5, 0.5, 0.0],    # M
        [0.0, 0.0, 0.0],    # Γ
        [0.5, 0.5, 0.5],    # R
        [0.5, 0.0, 0.0],    # X
    ]

    kpts, kpts_dist, spl_pnts = ham.get_kpts(k_path, nk=40)
    print(f"   - Number of k-points: {len(kpts)}")
    print(f"   - Path: Γ → X → M → Γ → R → X")

    # Solve for eigenvalues
    print("\n5. Solving tight-binding Hamiltonian...")
    print("   (This includes spin-orbit coupling)")
    eigen_vals = ham.solve_kpath(kpts, eig_vectors=False, soc=True, parallel=0)
    print(f"   - Number of bands: {eigen_vals.shape[0]}")
    print(f"   - Energy range: [{eigen_vals.min():.2f}, {eigen_vals.max():.2f}] eV")

    # Find bands near Fermi level
    fermi_bands = np.sum(eigen_vals[:, 0] < 0)
    print(f"   - Bands below E_F at Γ: {fermi_bands}")

    # Plot and save
    print("\n6. Plotting band structure...")
    output_file = os.path.join(os.path.dirname(__file__), "f_orbital_bands.png")
    fig, ax = plot_band_structure(kpts, kpts_dist, eigen_vals, spl_pnts, output_file)

    # Also create a zoomed plot showing f-band details
    print("\n7. Creating zoomed plot of f-bands...")
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for i in range(eigen_vals.shape[0]):
        ax2.plot(kpts_dist, eigen_vals[i, :], 'b-', linewidth=1.0, alpha=0.8)

    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1, label='E_F')

    labels = ['Γ', 'X', 'M', 'Γ', 'R', 'X']
    for i, sp in enumerate(spl_pnts):
        ax2.axvline(x=sp, color='gray', linestyle='-', linewidth=0.5)

    ax2.set_xticks(spl_pnts)
    ax2.set_xticklabels(labels[:len(spl_pnts)])

    ax2.set_xlabel('k-path', fontsize=12)
    ax2.set_ylabel('Energy (eV)', fontsize=12)
    ax2.set_title('f-bands Detail View\n(showing SOC splitting of f-levels)', fontsize=14)
    ax2.set_xlim(kpts_dist[0], kpts_dist[-1])
    ax2.set_ylim(-3, 2)  # Zoom into f-band region
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    zoom_file = os.path.join(os.path.dirname(__file__), "f_orbital_bands_zoom.png")
    plt.savefig(zoom_file, dpi=150, bbox_inches='tight')
    print(f"   Zoomed plot saved to: {zoom_file}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)

    # Show key physics features
    print("\nKey Physics Features in the Band Structure:")
    print("-" * 45)
    print("1. Narrow f-bands near E_F (localized 4f electrons)")
    print("2. SOC splitting of f-levels (~0.5 eV)")
    print("3. Broader s and p bands at higher energies")
    print("4. f-p hybridization visible at band crossings")

    plt.show()

    return eigen_vals, ham


if __name__ == "__main__":
    eigen_vals, ham = main()
