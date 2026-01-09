#!/usr/bin/env python3
"""
Phonon Calculations - Publication-Quality Visualization
========================================================
1. 1D monatomic chain: analytical verification
2. Diatomic chain: SSH/Peierls gap opening
3. Graphene: 2D phonon dispersion (central-force model)
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

# Color palette
C = {'blue': '#2563EB', 'red': '#DC2626', 'green': '#059669',
     'purple': '#7C3AED', 'amber': '#F59E0B', 'cyan': '#0891B2',
     'dark': '#1F2937', 'gray': '#9CA3AF', 'light': '#E5E7EB'}


def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 11,
        'axes.linewidth': 1.0, 'axes.edgecolor': C['dark'],
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.size': 4, 'ytick.major.size': 4,
    })


def test_1d_chain():
    """1D monatomic chain with analytical comparison."""
    print("\n" + "="*60 + "\n1D Monatomic Chain\n" + "="*60)
    setup_style()

    a, mass, A, B = 2.5, 12.0, 100.0, 2.0
    K_eff = A * B**2 * np.exp(-B * a)
    omega_ana = lambda q: 2 * np.sqrt(K_eff / mass) * EIGENVALUE_TO_THZ * np.abs(np.sin(np.pi * q))

    lattice = Lattice([[1, 0, 0], [0, 10, 0], [0, 0, 10]], a)
    structure = Structure(lattice, [Atom("C", [0, 0.5, 0.5], orbitals=["s"])],
                         periodicity=[True, False, False], bond_cut={"CC": {"NN": a * 1.1}})

    params = {"C": {"e_s": 0.0}, "CC": {
        "V_sss": Harrison(V0=-1.0, d0=a, cutoff=a*1.5),
        "repulsive": BornMayer(A=A, B=B, cutoff=a*1.5)
    }}

    ham = Hamiltonian(structure, params, numba=False)
    phonon = Phonon(ham, masses={"C": mass}, nk_fc=[20, 1, 1], n_electrons=2)

    q_points, q_dist, spl = phonon.get_qpath([[0, 0, 0], [0.5, 0, 0]], nq=50)
    frequencies = phonon.get_phonon_bands(np.array(q_points), parallel=False)

    # Extract longitudinal mode (highest frequency, x-direction)
    freq_long = np.max(frequencies, axis=0)
    q_frac = np.array([q[0] for q in q_points])

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(q_dist, freq_long, 'o', color=C['blue'], ms=5, mew=0, alpha=0.8, label='Numerical')
    ax.plot(q_dist, omega_ana(q_frac), '-', color=C['red'], lw=2, label='Analytical')

    ax.set_xlim(0, q_dist[-1])
    ax.set_ylim(0, np.max(freq_long) * 1.1)
    ax.set_xlabel('Wave vector q', fontsize=12)
    ax.set_ylabel('Frequency (THz)', fontsize=12)
    ax.set_xticks(spl)
    ax.set_xticklabels(['$\\Gamma$', 'X'], fontsize=12)
    ax.legend(frameon=True, loc='lower right', fontsize=10)
    ax.set_title('1D Monatomic Chain: $\\omega = 2\\sqrt{K/M}|\\sin(\\pi q)|$', fontsize=12, pad=10)

    for sp in spl:
        ax.axvline(sp, color=C['light'], ls='-', lw=0.8, zorder=0)
    ax.axhline(0, color=C['light'], ls='-', lw=0.8, zorder=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "phonon_1d_chain.png"), dpi=200, facecolor='white')
    plt.close()

    omega_max_num, omega_max_ana = np.max(freq_long), np.max(omega_ana(q_frac))
    print(f"  Numerical ω_max = {omega_max_num:.2f} THz")
    print(f"  Analytical ω_max = {omega_max_ana:.2f} THz")
    print(f"  Agreement: {100*omega_max_num/omega_max_ana:.1f}%")


def test_diatomic_chain():
    """Diatomic chain: SSH/Peierls gap opening from bond alternation."""
    print("\n" + "="*60 + "\nDiatomic Chain: SSH/Peierls Gap\n" + "="*60)
    setup_style()

    a, mass, A, B = 3.0, 12.0, 150.0, 2.0
    results = {}

    for label, d_inner in [('uniform', a/2), ('dimerized', a/2 - 0.15)]:
        print(f"\n  {label}: d_inner = {d_inner:.2f} Å")

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

        # Extract x-direction modes (longitudinal) - indices 0 and 3
        # These are the modes with dispersion; others are soft (ω≈0)
        freq_all = frequencies
        # Find the two modes with highest max frequency (longitudinal)
        max_freqs = np.max(np.abs(freq_all), axis=1)
        long_idx = np.argsort(max_freqs)[-2:]
        freq_long = freq_all[long_idx]

        # Sort: acoustic (ω→0 at Γ) first
        gamma_freqs = np.abs(freq_long[:, 0])
        sort_idx = np.argsort(gamma_freqs)
        freq_long = freq_long[sort_idx]

        # Gap at X
        freqs_X = np.abs(freq_long[:, -1])
        gap = freqs_X[1] - freqs_X[0] if len(freqs_X) > 1 else 0

        results[label] = {'q_dist': q_dist, 'freq': freq_long, 'spl': spl, 'gap': gap}
        print(f"    Frequencies at X: {freqs_X}")
        print(f"    Gap: {gap:.2f} THz")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    for ax, (label, data) in zip(axes, results.items()):
        freq = np.abs(data['freq'])  # Take absolute value for clean plot
        ax.plot(data['q_dist'], freq[0], '-', color=C['blue'], lw=2.5, label='Acoustic')
        ax.plot(data['q_dist'], freq[1], '-', color=C['red'], lw=2.5, label='Optical')

        ax.set_xlim(0, data['q_dist'][-1])
        ax.set_ylim(0, np.max(freq) * 1.1)
        ax.set_xlabel('Wave vector q', fontsize=12)
        ax.set_xticks(data['spl'])
        ax.set_xticklabels(['$\\Gamma$', 'X'], fontsize=12)

        for sp in data['spl']:
            ax.axvline(sp, color=C['light'], ls='-', lw=0.8, zorder=0)

        if label == 'uniform':
            ax.set_title('Uniform: $K_1 = K_2$\n(Bands touch at X)', fontsize=11, pad=8)
        else:
            ax.set_title(f'Dimerized: $K_1 \\neq K_2$\n(Gap = {data["gap"]:.1f} THz)', fontsize=11, pad=8)
        ax.legend(loc='upper right', frameon=True, fontsize=10)

    axes[0].set_ylabel('Frequency (THz)', fontsize=12)

    # Add inset schematics
    for ax, delta in [(axes[0], 0), (axes[1], 0.12)]:
        ins = ax.inset_axes([0.05, 0.68, 0.35, 0.25])
        ins.set_xlim(-0.1, 2.0); ins.set_ylim(-0.25, 0.25)
        ins.axis('off')
        for i in range(3):
            x = i * 0.7
            ins.add_patch(Circle((x, 0), 0.06, fc=C['blue'], ec='white', lw=0.8, zorder=10))
            ins.add_patch(Circle((x + 0.35 - delta, 0), 0.06, fc=C['red'], ec='white', lw=0.8, zorder=10))
            lw1, lw2 = (2.5, 1.0) if delta > 0 else (1.5, 1.5)
            if i < 2:
                ins.plot([x + 0.08, x + 0.27 - delta], [0, 0], color=C['dark'], lw=lw1)
                ins.plot([x + 0.43 - delta, x + 0.62], [0, 0], color=C['dark'], lw=lw2)

    fig.suptitle('SSH/Peierls Model: Phonon Gap from Bond Dimerization', fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "phonon_diatomic_chain.png"), dpi=200, facecolor='white')
    plt.close()


def test_graphene():
    """Graphene phonon dispersion (central-force model)."""
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

    G, M, K = [0, 0, 0], [0.5, 0, 0], [1/3, 1/3, 0]
    q_points, q_dist, spl = phonon.get_qpath([G, M, K, G], nq=50)

    print("  Computing phonon bands...")
    frequencies = phonon.get_phonon_bands(np.array(q_points), parallel=False)

    print("  Computing phonon DOS...")
    omega_max = np.max(frequencies) * 1.1
    omega_range = np.linspace(0, omega_max, 150)
    dos = phonon.get_phonon_dos(omega_range, nq=[20, 20, 1], sigma=2.0, parallel=False)

    # Classify modes by variance (dispersive vs flat)
    variances = np.var(frequencies, axis=1)
    dispersive_idx = set(np.argsort(variances)[-2:])  # Top 2 most dispersive

    # Plot
    fig = plt.figure(figsize=(11, 6))
    gs = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax_bands = fig.add_subplot(gs[0])
    ax_dos = fig.add_subplot(gs[1], sharey=ax_bands)

    # Plot all modes
    labels_used = set()
    for i, freq in enumerate(frequencies):
        if i in dispersive_idx:
            color, lw, alpha = C['blue'], 2.5, 1.0
            label = 'Dispersive (LA/LO)' if 'disp' not in labels_used else None
            labels_used.add('disp')
        else:
            color, lw, alpha = C['gray'], 1.0, 0.6
            label = 'Soft modes ($\\omega \\approx 0$)' if 'soft' not in labels_used else None
            labels_used.add('soft')
        ax_bands.plot(q_dist, freq, '-', color=color, lw=lw, alpha=alpha, label=label)

    ax_bands.set_xlim(0, q_dist[-1])
    ax_bands.set_ylim(0, omega_max)
    ax_bands.set_xlabel('Wave vector', fontsize=12)
    ax_bands.set_ylabel('Frequency (THz)', fontsize=12)
    ax_bands.set_xticks(spl)
    ax_bands.set_xticklabels(['$\\Gamma$', 'M', 'K', '$\\Gamma$'], fontsize=12)
    ax_bands.legend(loc='upper right', frameon=True, fontsize=10)

    for sp in spl:
        ax_bands.axvline(sp, color=C['light'], ls='-', lw=0.8, zorder=0)

    # DOS
    ax_dos.fill_betweenx(omega_range, 0, dos, alpha=0.4, color=C['blue'])
    ax_dos.plot(dos, omega_range, '-', color=C['blue'], lw=1.5)
    ax_dos.set_xlabel('DOS', fontsize=12)
    ax_dos.tick_params(labelleft=False)
    ax_dos.set_xlim(0, np.max(dos) * 1.2)

    # Honeycomb inset
    ax_ins = fig.add_axes([0.11, 0.62, 0.14, 0.28])
    draw_honeycomb(ax_ins)

    # Note
    fig.text(0.5, 0.01,
             'Central-force model: soft modes require angular forces for dispersion',
             ha='center', fontsize=9, style='italic', color=C['gray'])

    fig.suptitle('Graphene Phonon Dispersion', fontsize=13, y=0.97)
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    plt.savefig(os.path.join(OUTPUT_DIR, "phonon_graphene.png"), dpi=200, facecolor='white')
    plt.close()

    print(f"  6 branches: 2 dispersive (LA/LO), 4 soft (TA/TO/ZA/ZO)")
    print(f"  Max frequency: {np.max(frequencies):.1f} THz")


def draw_honeycomb(ax):
    """Draw graphene honeycomb structure."""
    ax.set_xlim(-0.2, 1.2); ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal'); ax.axis('off')

    a1, a2 = np.array([1, 0]), np.array([0.5, np.sqrt(3)/2])
    basis = [np.zeros(2), (a1 + a2) / 3]

    for i in range(-1, 2):
        for j in range(-1, 2):
            R = i * a1 + j * a2
            for b, base in enumerate(basis):
                pos = base + R
                if -0.15 < pos[0] < 1.15 and -0.05 < pos[1] < 0.95:
                    deltas = [(a1 + a2)/3, (a1 + a2)/3 - a1, (a1 + a2)/3 - a2] if b == 0 else \
                             [-(a1 + a2)/3, -(a1 + a2)/3 + a1, -(a1 + a2)/3 + a2]
                    for d in deltas:
                        n = pos + d
                        if -0.15 < n[0] < 1.15 and -0.05 < n[1] < 0.95:
                            ax.plot([pos[0], n[0]], [pos[1], n[1]], color=C['dark'], lw=0.8, zorder=1)
                    color = C['blue'] if b == 0 else C['red']
                    ax.add_patch(Circle(pos, 0.05, fc=color, ec='white', lw=0.5, zorder=10))


def main():
    print("="*60 + "\nPHONON CALCULATIONS\n" + "="*60)
    test_1d_chain()
    test_diatomic_chain()
    test_graphene()
    print("\n" + "="*60)
    print("All plots saved to:", OUTPUT_DIR)
    print("="*60)


if __name__ == "__main__":
    main()
