#!/usr/bin/env python3
"""
Example: Edge States in Graphene Zigzag Ribbon.

This example demonstrates edge state calculations in pysktb by computing
the spectral function for a zigzag graphene nanoribbon. Zigzag ribbons
exhibit topological flat-band edge states at E=0 connecting K and K'.

Features demonstrated:
- Building a graphene zigzag ribbon (quasi-1D structure)
- Bulk vs edge spectral function comparison
- Edge DOS showing localized states
- Edge spectral function A(k,E)

Author: Santosh Kumar Radha
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pysktb import Structure, Atom, Lattice, Hamiltonian, GreensFunction, SurfaceGreensFunction

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
    'primary': '#2563eb',
    'secondary': '#dc2626',
    'accent': '#059669',
    'purple': '#7c3aed',
    'orange': '#ea580c',
    'gray': '#6b7280',
}


def create_zigzag_ribbon(n_chains=8):
    """
    Create a graphene zigzag nanoribbon.

    Constructs an N-ZGNR by making a supercell of graphene along the a2 direction
    and removing periodicity in that direction. The ribbon is periodic along the
    zigzag edge (x-direction).

    Zigzag ribbons have topological flat-band edge states at E=0, appearing
    between projected K and K' points (k in [1/3, 2/3] in units of 2*pi/a).

    Parameters
    ----------
    n_chains : int
        Number of zigzag chains. Total atoms = 2 * n_chains.

    Returns
    -------
    structure : Structure
        Zigzag ribbon structure.
    edge_atoms : list
        Indices of edge atoms (atoms in the first and last chains).

    References
    ----------
    Nakada et al., Phys. Rev. B 54, 17954 (1996) - Edge states in graphene ribbons
    """
    a = 2.46  # Graphene lattice constant (Angstrom)
    d_cc = a / np.sqrt(3)  # C-C bond length = 1.42 Angstrom
    vacuum = 10.0  # Vacuum padding (Angstrom)

    # Supercell height: each chain adds a*sqrt(3)/2 in y-direction
    ribbon_height = n_chains * a * np.sqrt(3) / 2
    total_height = ribbon_height + vacuum

    # Rectangular unit cell for the ribbon
    # a1: along ribbon (zigzag direction), length = a
    # a2: perpendicular to ribbon (width + vacuum)
    # a3: out of plane
    lattice_matrix = [[1, 0, 0],
                      [0, total_height / a, 0],
                      [0, 0, vacuum / a]]
    lattice = Lattice(lattice_matrix, a)

    # Base positions from 2D graphene primitive cell (Cartesian coordinates)
    # A sublattice at origin, B at position derived from graphene geometry
    A_base = np.array([0.0, 0.0])
    B_base = np.array([a / 2, a * np.sqrt(3) / 6])  # = (1.23, 0.71) Angstrom

    # Shift per chain = graphene a2 lattice vector
    # This creates the proper honeycomb bonding pattern
    shift = np.array([a / 2, a * np.sqrt(3) / 2])  # = (1.23, 2.13) Angstrom

    atoms = []
    edge_atoms = []

    for i in range(n_chains):
        # Calculate Cartesian positions for this chain
        A_cart = A_base + i * shift
        A_cart[0] = A_cart[0] % a  # Wrap x into [0, a) for periodicity

        B_cart = B_base + i * shift
        B_cart[0] = B_cart[0] % a  # Wrap x into [0, a)

        # Convert to fractional coordinates
        A_frac = [A_cart[0] / a, A_cart[1] / total_height, 0]
        B_frac = [B_cart[0] / a, B_cart[1] / total_height, 0]

        atoms.append(Atom("C", A_frac, orbitals=["pz"]))
        atoms.append(Atom("C", B_frac, orbitals=["pz"]))

        # Track edge atoms (first and last chains)
        if i == 0 or i == n_chains - 1:
            edge_atoms.extend([len(atoms) - 2, len(atoms) - 1])

    # Bond cutoff for nearest neighbors only
    # NN distance is d_cc = 1.42 Angstrom, use 1.1x for safety margin
    bond_cut = {"CC": {"NN": d_cc * 1.1}}

    structure = Structure(
        lattice, atoms,
        periodicity=[True, False, False],  # Only periodic along ribbon (x)
        bond_cut=bond_cut
    )

    return structure, edge_atoms


def get_graphene_parameters():
    """Return tight-binding parameters for graphene pz orbitals."""
    params = {
        "C": {
            "e_p": 0.0,
        },
        "CC": {
            "V_ppp": -2.7,  # Hopping parameter
        }
    }
    return params


def main():
    """Main function demonstrating edge states in zigzag ribbon."""

    print("=" * 60)
    print("Edge States Example: Graphene Zigzag Ribbon")
    print("=" * 60)

    # Create ribbon structure
    n_width = 12  # Number of zigzag chains
    print(f"\n1. Creating zigzag ribbon (width = {n_width} chains)...")
    structure, edge_atoms = create_zigzag_ribbon(n_chains=n_width)
    params = get_graphene_parameters()
    ham = Hamiltonian(structure, params, numba=False)

    print(f"   - Total atoms: {len(structure.atoms)}")
    print(f"   - Edge atom indices: {edge_atoms}")
    print(f"   - Periodicity: {structure.periodicity}")

    # Compute band structure
    print("\n2. Computing ribbon band structure...")
    k_path = [[0, 0, 0], [0.5, 0, 0], [1, 0, 0]]  # Gamma - X - Gamma (1D BZ)
    kpts, kpts_dist, spl_pnts = ham.get_kpts(k_path, nk=100)
    eigen_vals = ham.solve_kpath(kpts, eig_vectors=False, soc=False, parallel=0)

    # Create Green's function calculators
    print("\n3. Setting up Green's function calculators...")
    gf = GreensFunction(ham)
    sgf = SurfaceGreensFunction(ham, surface_atoms=edge_atoms)

    # Energy grid
    energies = np.linspace(-3, 3, 200)

    # Compute edge DOS
    print("\n4. Computing edge DOS...")
    edge_dos = sgf.edge_dos(energies, nk=50, eta=0.1, soc=False, parallel=True)

    # Compute bulk DOS for comparison
    print("\n5. Computing bulk DOS for comparison...")
    bulk_atoms = [i for i in range(len(structure.atoms)) if i not in edge_atoms][:4]  # Middle atoms
    sgf_bulk = SurfaceGreensFunction(ham, surface_atoms=bulk_atoms)
    bulk_dos = sgf_bulk.edge_dos(energies, nk=50, eta=0.1, soc=False, parallel=True)

    # Compute edge spectral function A(k,E)
    print("\n6. Computing edge spectral function A(k,E)...")
    k_values = np.linspace(0, 1, 100)
    edge_spectral = sgf.edge_spectral_kpath(k_values, energies, eta=0.08, soc=False)

    # Create figure
    print("\n7. Creating plots...")
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)

    # Create grid spec for custom layout
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.2])

    # Plot 1: Ribbon band structure
    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(eigen_vals.shape[0]):
        ax1.plot(kpts_dist, eigen_vals[i, :], color=COLORS['primary'], linewidth=1.5)

    ax1.axhline(y=0, color=COLORS['secondary'], linestyle='--', linewidth=1.5, alpha=0.8)
    ax1.set_xticks([kpts_dist[0], kpts_dist[len(kpts_dist)//2], kpts_dist[-1]])
    ax1.set_xticklabels(['0', 'k (1/a)', '2π/a'])
    ax1.set_xlabel('Wave Vector')
    ax1.set_ylabel('Energy (eV)')
    ax1.set_title('Ribbon Band Structure')
    ax1.set_ylim(-3, 3)

    # Plot 2: Edge vs Bulk DOS
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(energies, 0, edge_dos, alpha=0.3, color=COLORS['secondary'])
    ax2.plot(energies, edge_dos, color=COLORS['secondary'], linewidth=2, label='Edge')
    ax2.fill_between(energies, 0, bulk_dos, alpha=0.3, color=COLORS['primary'])
    ax2.plot(energies, bulk_dos, color=COLORS['primary'], linewidth=2, label='Bulk', linestyle='--')
    ax2.axvline(x=0, color=COLORS['gray'], linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Energy (eV)')
    ax2.set_ylabel('DOS (states/eV)')
    ax2.set_title('Edge vs Bulk DOS')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(0, None)
    ax2.legend(frameon=True, fancybox=True, shadow=True)

    # Plot 3: Edge DOS zoomed at E=0
    ax3 = fig.add_subplot(gs[0, 2])
    mask = (energies > -1) & (energies < 1)
    ax3.fill_between(energies[mask], 0, edge_dos[mask], alpha=0.4, color=COLORS['accent'])
    ax3.plot(energies[mask], edge_dos[mask], color=COLORS['accent'], linewidth=2)
    ax3.axvline(x=0, color=COLORS['secondary'], linestyle='--', linewidth=2, alpha=0.8, label='E = 0')
    ax3.set_xlabel('Energy (eV)')
    ax3.set_ylabel('Edge DOS (states/eV)')
    ax3.set_title('Edge States at E = 0\n(Topological Flat Band)')
    ax3.set_ylim(0, None)
    ax3.legend(frameon=True, fancybox=True, shadow=True)

    # Plot 4: Edge spectral function A(k,E)
    ax4 = fig.add_subplot(gs[1, :])

    # Create colormap plot
    extent = [0, 1, energies.min(), energies.max()]
    im = ax4.imshow(edge_spectral.T, aspect='auto', origin='lower', extent=extent,
                    cmap='magma', vmin=0, vmax=np.percentile(edge_spectral, 98))

    ax4.axhline(y=0, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    ax4.set_xlabel('k (units of 2π/a)')
    ax4.set_ylabel('Energy (eV)')
    ax4.set_title('Edge Spectral Function A(k,E) — Flat Band Edge States Visible at E ≈ 0')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4, shrink=0.6, label='Spectral Weight')

    # Save figure
    output_file = os.path.join(os.path.dirname(__file__), "data", "edge_states_zigzag.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n   Plot saved to: {output_file}")

    print("\n" + "=" * 60)
    print("Key Physics Features:")
    print("-" * 40)
    print("1. Flat bands at E=0 are topological edge states")
    print("2. Edge DOS shows peak at E=0 (localized edge states)")
    print("3. Bulk DOS shows gap at E=0 (no bulk states)")
    print("4. Edge states connect K and K' in projected BZ")
    print("=" * 60)

    plt.show()

    return edge_dos, bulk_dos, edge_spectral, ham


if __name__ == "__main__":
    edge_dos, bulk_dos, edge_spectral, ham = main()
