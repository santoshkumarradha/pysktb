#!/usr/bin/env python3
"""
Phonon Calculations - Professional Visualization
=================================================
1. 1D monatomic chain with analytical comparison
2. Diatomic chain: Peierls/SSH gap opening
3. Graphene phonon dispersion + DOS
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

from pysktb import Structure, Atom, Lattice, Hamiltonian, Harrison, BornMayer
from pysktb.phonon import Phonon, EIGENVALUE_TO_THZ

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Professional color palette
C = {'blue': '#2563EB', 'red': '#DC2626', 'green': '#059669',
     'purple': '#7C3AED', 'amber': '#F59E0B', 'cyan': '#0891B2',
     'dark': '#1F2937', 'gray': '#9CA3AF'}

def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 11,
        'axes.linewidth': 1.2, 'axes.edgecolor': C['dark'],
        'xtick.direction': 'in', 'ytick.direction': 'in',
    })


def test_1d_chain():
    """1D monatomic chain - numerical vs analytical."""
    print("\n" + "="*60 + "\n1D Monatomic Chain\n" + "="*60)
    setup_style()

    a, mass, A, B = 2.5, 12.0, 100.0, 2.0
    K_eff = A * B**2 * np.exp(-B * a)
    omega_analytical = lambda q: 2 * np.sqrt(K_eff / mass) * EIGENVALUE_TO_THZ * np.abs(np.sin(np.pi * q))

    lattice = Lattice([[1, 0, 0], [0, 10, 0], [0, 0, 10]], a)
    atom = Atom("C", [0, 0.5, 0.5], orbitals=["s"])
    structure = Structure(lattice, [atom], periodicity=[True, False, False],
                         bond_cut={"CC": {"NN": a * 1.1}})

    params = {"C": {"e_s": 0.0}, "CC": {
        "V_sss": Harrison(V0=-1.0, d0=a, cutoff=a*1.5),
        "repulsive": BornMayer(A=A, B=B, cutoff=a*1.5)
    }}

    ham = Hamiltonian(structure, params, numba=False)
    phonon = Phonon(ham, masses={"C": mass}, nk_fc=[20, 1, 1], n_electrons=2)

    q_points, q_dist, spl = phonon.get_qpath([[0, 0, 0], [0.5, 0, 0]], nq=50)
    frequencies = phonon.get_phonon_bands(np.array(q_points), parallel=False)

    # Analytical for comparison
    q_frac = np.array([q[0] for q in q_points])
    omega_ana = omega_analytical(q_frac)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(q_dist, frequencies[0], 'o', color=C['blue'], ms=6, mew=0,
            alpha=0.7, label='Numerical')
    ax.plot(q_dist, omega_ana, '-', color=C['red'], lw=2.5, label='Analytical')

    ax.set_xlim(0, q_dist[-1])
    ax.set_ylim(0, None)
    ax.set_xlabel('Wave vector q')
    ax.set_ylabel('Frequency (THz)')
    ax.set_xticks(spl)
    ax.set_xticklabels(['Γ', 'X'])
    ax.legend(frameon=True, loc='lower right')
    ax.set_title('1D Chain: ω = 2√(K/M)|sin(πq)|', fontweight='bold')

    for sp in spl:
        ax.axvline(sp, color=C['gray'], ls='--', lw=0.8, alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "phonon_1d_chain.png"), dpi=200)
    plt.close()

    omega_max_num = np.max(frequencies)
    omega_max_ana = np.max(omega_ana)
    print(f"  ω_max: Numerical = {omega_max_num:.2f} THz, Analytical = {omega_max_ana:.2f} THz")
    print(f"  Agreement: {100*omega_max_num/omega_max_ana:.1f}%")


def test_diatomic_chain():
    """
    Diatomic chain showing Peierls/SSH gap opening.
    Two atoms per cell - only x-direction modes shown (longitudinal).
    """
    print("\n" + "="*60 + "\nDiatomic Chain: SSH/Peierls Gap Opening\n" + "="*60)
    setup_style()

    a, mass = 3.0, 12.0
    A, B = 150.0, 2.0

    results = {}

    for label, d_inner in [('uniform', a/2), ('dimerized', a/2 - 0.2)]:
        print(f"\n  {label}: inner bond = {d_inner:.2f} Å")

        # Two atoms in unit cell
        lattice = Lattice([[1, 0, 0], [0, 10, 0], [0, 0, 10]], a)
        atom1 = Atom("C", [0, 0.5, 0.5], orbitals=["s"])
        atom2 = Atom("C", [d_inner/a, 0.5, 0.5], orbitals=["s"])

        structure = Structure(lattice, [atom1, atom2], periodicity=[True, False, False],
                             bond_cut={"CC": {"NN": a * 0.6}})

        params = {"C": {"e_s": 0.0}, "CC": {
            "V_sss": Harrison(V0=-1.0, d0=a/2, cutoff=a*0.8),
            "repulsive": BornMayer(A=A, B=B, cutoff=a*0.8)
        }}

        ham = Hamiltonian(structure, params, numba=False)
        phonon = Phonon(ham, masses={"C": mass}, nk_fc=[20, 1, 1], n_electrons=4)

        q_points, q_dist, spl = phonon.get_qpath([[0, 0, 0], [0.5, 0, 0]], nq=50)
        frequencies = phonon.get_phonon_bands(np.array(q_points), parallel=False)

        # Extract only x-direction modes (indices 0 and 3 for 2 atoms, x-component)
        # In 3D force constant, x-modes are the first and fourth (atom1-x and atom2-x)
        # Sort by frequency at Gamma to identify acoustic vs optical
        freq_x = frequencies[[0, 3], :]  # x-direction modes only

        # Sort: acoustic (lower at Gamma) first
        idx = np.argsort(np.abs(freq_x[:, 0]))
        freq_x = freq_x[idx]

        # Calculate gap
        if label == 'dimerized':
            # Gap at X point
            gap = np.abs(freq_x[1, -1]) - np.abs(freq_x[0, -1])
        else:
            gap = 0

        results[label] = {'q_dist': q_dist, 'freq': freq_x, 'spl': spl, 'gap': gap}
        print(f"    Gap at X: {gap:.2f} THz" if gap > 0 else "    Bands touch at X")

    # Combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, (label, data) in zip([ax1, ax2], results.items()):
        freq = data['freq']
        # Acoustic (lower) in blue, optical (upper) in red
        ax.plot(data['q_dist'], np.abs(freq[0]), '-', color=C['blue'], lw=2.5, label='Acoustic')
        ax.plot(data['q_dist'], np.abs(freq[1]), '-', color=C['red'], lw=2.5, label='Optical')

        ax.set_xlim(0, data['q_dist'][-1])
        ax.set_ylim(0, None)
        ax.set_xlabel('Wave vector q')
        ax.set_xticks(data['spl'])
        ax.set_xticklabels(['Γ', 'X'])
        for sp in data['spl']:
            ax.axvline(sp, color=C['gray'], ls='--', lw=0.8, alpha=0.5)

        title = 'Uniform: K₁ = K₂\n(Bands touch at X)' if label == 'uniform' else f'Dimerized: Gap = {data["gap"]:.1f} THz'
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper right', frameon=True)

    ax1.set_ylabel('Frequency (THz)')

    # Add schematic insets
    for ax, delta in [(ax1, 0), (ax2, 0.15)]:
        inset = ax.inset_axes([0.05, 0.72, 0.35, 0.22])
        inset.set_xlim(-0.2, 2.2)
        inset.set_ylim(-0.3, 0.3)
        inset.axis('off')

        for i in range(3):
            x = i * 0.8
            inset.add_patch(Circle((x, 0), 0.08, fc=C['blue'], ec='white', lw=1, zorder=10))
            inset.add_patch(Circle((x + 0.4 - delta*0.5, 0), 0.08, fc=C['red'], ec='white', lw=1, zorder=10))
            lw1, lw2 = (3, 1.5) if delta > 0 else (2, 2)
            if i < 2:
                inset.plot([x + 0.12, x + 0.28 - delta*0.5], [0, 0], color=C['dark'], lw=lw1)
                inset.plot([x + 0.52 - delta*0.5, x + 0.68], [0, 0], color=C['dark'], lw=lw2)

    fig.suptitle('SSH/Peierls Model: Gap Opening from Bond Dimerization', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "phonon_diatomic_chain.png"), dpi=200)
    plt.close()


def test_graphene():
    """
    Graphene phonon dispersion + DOS.
    Central-force model: only radial springs, no bond-bending forces.
    Flat bands are artifacts - tangential modes have no restoring force.
    """
    print("\n" + "="*60 + "\nGraphene Phonons (Central-Force Model)\n" + "="*60)
    setup_style()

    a_cc = 1.42
    a = a_cc * np.sqrt(3)

    lattice = Lattice([[1, 0, 0], [0.5, np.sqrt(3)/2, 0], [0, 0, 10]], a)
    atom1 = Atom("C", [0, 0, 0.5], orbitals=["pz"])
    atom2 = Atom("C", [1/3, 1/3, 0.5], orbitals=["pz"])

    structure = Structure(lattice, [atom1, atom2], periodicity=[True, True, False],
                         bond_cut={"CC": {"NN": a_cc * 1.2}})

    params = {"C": {"e_p": 0.0}, "CC": {
        "V_ppp": Harrison(V0=-2.7, d0=a_cc, cutoff=a_cc*1.5),
        "repulsive": BornMayer(A=600.0, B=2.5, cutoff=a_cc*1.5)
    }}

    ham = Hamiltonian(structure, params, numba=False)
    phonon = Phonon(ham, masses={"C": 12.011}, nk_fc=[8, 8, 1], n_electrons=2)

    # High-symmetry path
    G, M, K = [0, 0, 0], [0.5, 0, 0], [1/3, 1/3, 0]
    q_points, q_dist, spl = phonon.get_qpath([G, M, K, G], nq=40)

    print("  Computing phonon bands...")
    frequencies = phonon.get_phonon_bands(np.array(q_points), parallel=False)

    print("  Computing phonon DOS...")
    omega_max = np.max(np.abs(frequencies)) * 1.1
    omega_range = np.linspace(0, omega_max, 150)
    dos = phonon.get_phonon_dos(omega_range, nq=[20, 20, 1], sigma=1.5, parallel=False)

    # Identify dispersive vs flat modes by variance
    variances = np.var(frequencies, axis=1)
    dispersive_idx = np.argsort(variances)[-2:]  # Top 2 most dispersive

    # Plot
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax_bands = fig.add_subplot(gs[0])
    ax_dos = fig.add_subplot(gs[1], sharey=ax_bands)

    # Plot all modes, highlight dispersive ones
    label_used = {'dispersive': False, 'flat': False}
    for i, freq in enumerate(frequencies):
        if i in dispersive_idx:
            color = C['blue']
            lw, alpha = 2.5, 1.0
            label = 'Dispersive (radial)' if not label_used['dispersive'] else None
            label_used['dispersive'] = True
        else:
            color = C['gray']
            lw, alpha = 1.2, 0.5
            label = 'Flat (no angular force)' if not label_used['flat'] else None
            label_used['flat'] = True

        ax_bands.plot(q_dist, freq, '-', color=color, lw=lw, alpha=alpha, label=label)

    ax_bands.set_xlim(0, q_dist[-1])
    ax_bands.set_ylim(-2, omega_max)
    ax_bands.set_xlabel('Wave vector')
    ax_bands.set_ylabel('Frequency (THz)')
    ax_bands.set_xticks(spl)
    ax_bands.set_xticklabels(['Γ', 'M', 'K', 'Γ'])
    ax_bands.legend(loc='upper right', frameon=True, fontsize=9)

    for sp in spl:
        ax_bands.axvline(sp, color=C['gray'], ls='-', lw=0.8, alpha=0.3)
    ax_bands.axhline(0, color=C['gray'], ls='-', lw=0.8, alpha=0.3)

    # DOS
    ax_dos.fill_betweenx(omega_range, 0, dos, alpha=0.4, color=C['blue'])
    ax_dos.plot(dos, omega_range, '-', color=C['blue'], lw=1.5)
    ax_dos.set_xlabel('DOS')
    ax_dos.tick_params(labelleft=False)
    ax_dos.set_xlim(0, None)

    # Honeycomb inset
    ax_inset = fig.add_axes([0.12, 0.62, 0.15, 0.28])
    draw_honeycomb(ax_inset)

    # Note about central-force approximation
    fig.text(0.5, 0.01,
             'Central-force model: flat bands = tangential modes (require angular forces for dispersion)',
             ha='center', fontsize=9, style='italic', color=C['gray'])

    fig.suptitle('Graphene Phonon Dispersion', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "phonon_graphene.png"), dpi=200)
    plt.close()

    print(f"  6 branches (2 dispersive LA/LO, 4 flat)")
    print(f"  Note: Flat bands require angular forces for proper dispersion")


def draw_honeycomb(ax):
    """Draw graphene honeycomb structure."""
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.2, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')

    a1, a2 = np.array([1, 0]), np.array([0.5, np.sqrt(3)/2])
    basis = [np.zeros(2), (a1 + a2) / 3]

    # Draw bonds and atoms
    for i in range(-1, 2):
        for j in range(-1, 2):
            R = i * a1 + j * a2
            for b, base in enumerate(basis):
                pos = base + R
                if -0.2 < pos[0] < 1.2 and -0.1 < pos[1] < 1.0:
                    # Bonds to neighbors
                    deltas = [(a1 + a2)/3, (a1 + a2)/3 - a1, (a1 + a2)/3 - a2] if b == 0 else \
                             [-(a1 + a2)/3, -(a1 + a2)/3 + a1, -(a1 + a2)/3 + a2]
                    for d in deltas:
                        n = pos + d
                        if -0.2 < n[0] < 1.2 and -0.1 < n[1] < 1.0:
                            ax.plot([pos[0], n[0]], [pos[1], n[1]], color=C['dark'], lw=1, zorder=1)

                    color = C['blue'] if b == 0 else C['red']
                    ax.add_patch(Circle(pos, 0.06, fc=color, ec='white', lw=0.8, zorder=10))


def main():
    print("="*60 + "\nPHONON CALCULATIONS\n" + "="*60)
    test_1d_chain()
    test_diatomic_chain()
    test_graphene()
    print("\n" + "="*60 + "\nAll plots saved to:", OUTPUT_DIR, "\n" + "="*60)


if __name__ == "__main__":
    main()
