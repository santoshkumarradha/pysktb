"""Visualization utilities for orbital and charge density plotting.

This module provides tools for visualizing atomic orbitals and charge
densities from tight-binding calculations in both 2D and 3D.

Example:
    >>> from pysktb import Hamiltonian, OrbitalPlotter
    >>> ham = Hamiltonian(structure, params)
    >>> plotter = OrbitalPlotter(ham)
    >>> plotter.plot_orbital_2d(atom_idx=0, orbital="pz")
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Optional, Tuple, List

from .orbitals import OrbitalBasis, AtomicOrbital


class OrbitalPlotter:
    """Visualize atomic orbitals and charge densities.

    Supports both 2D slice plots and 3D isosurface visualizations.

    Example:
        >>> # From Hamiltonian
        >>> plotter = OrbitalPlotter(ham)
        >>> plotter.plot_orbital_2d(atom_idx=0, orbital="pz")

        >>> # From Structure only (for simple orbital visualization)
        >>> plotter = OrbitalPlotter.from_structure(structure)
        >>> plotter.plot_orbital_2d(atom_idx=0, orbital="pz")
    """

    def __init__(self, hamiltonian_or_structure, orbital_basis: Optional[OrbitalBasis] = None):
        """
        Args:
            hamiltonian_or_structure: pysktb Hamiltonian or Structure object
            orbital_basis: OrbitalBasis instance. If None, creates one with
                          default Slater exponents.
        """
        # Check if it's a Hamiltonian or Structure
        if hasattr(hamiltonian_or_structure, 'structure'):
            self.ham = hamiltonian_or_structure
            self.structure = hamiltonian_or_structure.structure
        else:
            self.ham = None
            self.structure = hamiltonian_or_structure

        if orbital_basis is None:
            self.basis = OrbitalBasis.from_defaults(self.structure)
        else:
            self.basis = orbital_basis

    @classmethod
    def from_structure(cls, structure, orbital_basis: Optional[OrbitalBasis] = None):
        """Create plotter from Structure (no Hamiltonian needed for orbital plots)."""
        return cls(structure, orbital_basis)

    def _create_2d_grid(self, plane: str, slice_value: float,
                        extent: float, resolution: int, center: np.ndarray = None
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create 2D grid for slice plotting.

        Returns:
            x_grid, y_grid: 2D meshgrid for plotting
            X, Y, Z: 3D coordinate arrays for orbital evaluation
        """
        if center is None:
            center = np.zeros(3)

        coords = np.linspace(-extent, extent, resolution)

        if plane == 'xy':
            x_grid, y_grid = np.meshgrid(coords, coords)
            X = x_grid + center[0]
            Y = y_grid + center[1]
            Z = np.full_like(X, slice_value + center[2])
        elif plane == 'xz':
            x_grid, y_grid = np.meshgrid(coords, coords)
            X = x_grid + center[0]
            Y = np.full_like(X, slice_value + center[1])
            Z = y_grid + center[2]
        elif plane == 'yz':
            x_grid, y_grid = np.meshgrid(coords, coords)
            X = np.full_like(x_grid, slice_value + center[0])
            Y = x_grid + center[1]
            Z = y_grid + center[2]
        else:
            raise ValueError(f"Unknown plane: {plane}. Use 'xy', 'xz', or 'yz'.")

        return x_grid, y_grid, X, Y, Z

    def plot_orbital_2d(self, atom_idx: int, orbital: str,
                        plane: str = 'xy', slice_value: float = 0.0,
                        extent: float = 3.0, resolution: int = 100,
                        ax: Optional[plt.Axes] = None,
                        cmap: str = 'RdBu_r', show_colorbar: bool = True,
                        title: Optional[str] = None) -> plt.Axes:
        """Plot 2D slice of atomic orbital.

        Args:
            atom_idx: Index of atom in structure
            orbital: Orbital name ('s', 'px', 'py', 'pz', etc.)
            plane: Slice plane ('xy', 'xz', 'yz')
            slice_value: Position of slice along perpendicular axis
            extent: Half-width of plot in Angstroms
            resolution: Grid resolution
            ax: Matplotlib axes (creates new if None)
            cmap: Colormap name
            show_colorbar: Whether to show colorbar
            title: Plot title (auto-generated if None)

        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))

        # Get atom position and orbital
        atom_pos = self.basis.get_atom_position(atom_idx)
        atomic_orbital = self.basis.get_orbital(atom_idx, orbital)

        # Create grid centered on atom
        x_grid, y_grid, X, Y, Z = self._create_2d_grid(
            plane, slice_value, extent, resolution, center=atom_pos
        )

        # Evaluate orbital (relative to atom position)
        X_rel = X - atom_pos[0]
        Y_rel = Y - atom_pos[1]
        Z_rel = Z - atom_pos[2]
        values = atomic_orbital.evaluate(X_rel, Y_rel, Z_rel)

        # Plot
        vmax = np.abs(values).max()
        im = ax.contourf(x_grid, y_grid, values, levels=50,
                         cmap=cmap, vmin=-vmax, vmax=vmax)

        # Add contour lines
        ax.contour(x_grid, y_grid, values, levels=10,
                   colors='k', linewidths=0.5, alpha=0.3)

        # Mark atom position
        ax.plot(0, 0, 'ko', markersize=8, label='Atom')

        if show_colorbar:
            plt.colorbar(im, ax=ax, label='ψ(r)')

        # Labels
        plane_labels = {'xy': ('x', 'y'), 'xz': ('x', 'z'), 'yz': ('y', 'z')}
        xlabel, ylabel = plane_labels[plane]
        ax.set_xlabel(f'{xlabel} (Å)', fontsize=11)
        ax.set_ylabel(f'{ylabel} (Å)', fontsize=11)

        if title is None:
            element = self.structure.atoms[atom_idx].element
            title = f'{element} {orbital} orbital ({plane} plane)'
        ax.set_title(title, fontsize=12)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        return ax

    def plot_orbital_gallery(self, atom_idx: int, orbitals: List[str] = None,
                             extent: float = 3.0, resolution: int = 80,
                             figsize: Tuple[float, float] = None) -> plt.Figure:
        """Plot gallery of multiple orbitals for an atom.

        Args:
            atom_idx: Index of atom
            orbitals: List of orbital names (uses all if None)
            extent: Plot extent in Angstroms
            resolution: Grid resolution
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        atom = self.structure.atoms[atom_idx]
        if orbitals is None:
            orbitals = atom.orbitals

        n_orb = len(orbitals)
        ncols = min(3, n_orb)
        nrows = (n_orb + ncols - 1) // ncols

        if figsize is None:
            figsize = (5 * ncols, 4.5 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_orb == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, orbital in enumerate(orbitals):
            # Choose appropriate plane for visualization
            if orbital in ['pz', 'dz2']:
                plane = 'xz'
            elif orbital in ['px', 'dxz']:
                plane = 'xy'
            else:
                plane = 'xy'

            self.plot_orbital_2d(atom_idx, orbital, plane=plane,
                                extent=extent, resolution=resolution,
                                ax=axes[i], show_colorbar=True)

        # Hide unused axes
        for i in range(n_orb, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig

    def plot_structure_orbitals_2d(self, plane: str = 'xy', slice_value: float = 0.0,
                                   extent: float = 5.0, resolution: int = 100,
                                   ax: Optional[plt.Axes] = None,
                                   cmap: str = 'RdBu_r') -> plt.Axes:
        """Plot superposition of all orbitals in structure.

        Args:
            plane: Slice plane
            slice_value: Slice position
            extent: Plot half-width
            resolution: Grid resolution
            ax: Matplotlib axes
            cmap: Colormap

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        # Find center of structure
        positions = [self.basis.get_atom_position(i)
                    for i in range(len(self.structure.atoms))]
        center = np.mean(positions, axis=0)

        # Create grid
        x_grid, y_grid, X, Y, Z = self._create_2d_grid(
            plane, slice_value, extent, resolution, center=center
        )

        # Sum contributions from all orbitals
        total = np.zeros_like(X)
        for atom_idx, atom in enumerate(self.structure.atoms):
            atom_pos = self.basis.get_atom_position(atom_idx)
            for orbital in atom.orbitals:
                atomic_orbital = self.basis.get_orbital(atom_idx, orbital)
                X_rel = X - atom_pos[0]
                Y_rel = Y - atom_pos[1]
                Z_rel = Z - atom_pos[2]
                total += atomic_orbital.evaluate(X_rel, Y_rel, Z_rel)

        # Plot
        vmax = np.abs(total).max()
        im = ax.contourf(x_grid + center[0], y_grid + center[1], total,
                         levels=50, cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.contour(x_grid + center[0], y_grid + center[1], total,
                   levels=10, colors='k', linewidths=0.5, alpha=0.3)

        # Mark atom positions
        for atom_idx, atom in enumerate(self.structure.atoms):
            pos = self.basis.get_atom_position(atom_idx)
            if plane == 'xy':
                ax.plot(pos[0], pos[1], 'ko', markersize=10)
            elif plane == 'xz':
                ax.plot(pos[0], pos[2], 'ko', markersize=10)
            elif plane == 'yz':
                ax.plot(pos[1], pos[2], 'ko', markersize=10)

        plt.colorbar(im, ax=ax, label='ψ(r)')

        plane_labels = {'xy': ('x', 'y'), 'xz': ('x', 'z'), 'yz': ('y', 'z')}
        xlabel, ylabel = plane_labels[plane]
        ax.set_xlabel(f'{xlabel} (Å)', fontsize=11)
        ax.set_ylabel(f'{ylabel} (Å)', fontsize=11)
        ax.set_title(f'Structure orbitals ({plane} plane)', fontsize=12)
        ax.set_aspect('equal')

        return ax

    # === 3D Isosurface Plotting ===

    def _create_3d_grid(self, extent: float, resolution: int,
                        center: np.ndarray = None) -> Tuple[np.ndarray, ...]:
        """Create 3D grid for isosurface plotting."""
        if center is None:
            center = np.zeros(3)

        coords = np.linspace(-extent, extent, resolution)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        X = X + center[0]
        Y = Y + center[1]
        Z = Z + center[2]

        return X, Y, Z

    def plot_orbital_3d(self, atom_idx: int, orbital: str,
                        extent: float = 3.0, resolution: int = 40,
                        isosurface: float = 0.1, alpha: float = 0.6,
                        ax=None, colors: Tuple[str, str] = ('#e74c3c', '#3498db')
                        ):
        """Plot 3D isosurface of atomic orbital.

        Shows positive lobe in one color and negative lobe in another.

        Args:
            atom_idx: Index of atom
            orbital: Orbital name
            extent: Plot half-width in Angstroms
            resolution: Grid resolution (lower = faster)
            isosurface: Isosurface value (fraction of max)
            alpha: Surface transparency
            ax: Matplotlib 3D axes (creates new if None)
            colors: (positive_color, negative_color)

        Returns:
            Matplotlib 3D axes
        """
        from mpl_toolkits.mplot3d import Axes3D
        from skimage import measure

        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        # Get atom position and orbital
        atom_pos = self.basis.get_atom_position(atom_idx)
        atomic_orbital = self.basis.get_orbital(atom_idx, orbital)

        # Create grid centered on atom
        X, Y, Z = self._create_3d_grid(extent, resolution, center=atom_pos)

        # Evaluate orbital
        X_rel = X - atom_pos[0]
        Y_rel = Y - atom_pos[1]
        Z_rel = Z - atom_pos[2]
        values = atomic_orbital.evaluate(X_rel, Y_rel, Z_rel)

        # Determine isosurface level
        vmax = np.abs(values).max()
        level = isosurface * vmax

        # Plot positive isosurface
        try:
            verts_pos, faces_pos, _, _ = measure.marching_cubes(
                values, level, spacing=(2*extent/resolution,)*3
            )
            verts_pos = verts_pos - extent + atom_pos
            ax.plot_trisurf(verts_pos[:, 0], verts_pos[:, 1], faces_pos,
                           verts_pos[:, 2], color=colors[0], alpha=alpha,
                           shade=True)
        except ValueError:
            pass  # No isosurface at this level

        # Plot negative isosurface
        try:
            verts_neg, faces_neg, _, _ = measure.marching_cubes(
                values, -level, spacing=(2*extent/resolution,)*3
            )
            verts_neg = verts_neg - extent + atom_pos
            ax.plot_trisurf(verts_neg[:, 0], verts_neg[:, 1], faces_neg,
                           verts_neg[:, 2], color=colors[1], alpha=alpha,
                           shade=True)
        except ValueError:
            pass

        # Mark atom
        ax.scatter([atom_pos[0]], [atom_pos[1]], [atom_pos[2]],
                  color='black', s=100, marker='o')

        # Labels
        ax.set_xlabel('x (Å)')
        ax.set_ylabel('y (Å)')
        ax.set_zlabel('z (Å)')

        element = self.structure.atoms[atom_idx].element
        ax.set_title(f'{element} {orbital} orbital (3D isosurface)')

        return ax

    def plot_orbital_3d_simple(self, atom_idx: int, orbital: str,
                               extent: float = 3.0, resolution: int = 30,
                               n_points: int = 1000, ax=None) -> plt.Axes:
        """Plot 3D orbital using scatter plot (simpler, no skimage needed).

        Args:
            atom_idx: Index of atom
            orbital: Orbital name
            extent: Plot half-width
            resolution: Grid resolution
            n_points: Number of scatter points
            ax: Matplotlib 3D axes

        Returns:
            Matplotlib 3D axes
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        atom_pos = self.basis.get_atom_position(atom_idx)
        atomic_orbital = self.basis.get_orbital(atom_idx, orbital)

        # Create grid
        X, Y, Z = self._create_3d_grid(extent, resolution, center=atom_pos)

        # Evaluate orbital
        X_rel = X - atom_pos[0]
        Y_rel = Y - atom_pos[1]
        Z_rel = Z - atom_pos[2]
        values = atomic_orbital.evaluate(X_rel, Y_rel, Z_rel)

        # Sample points weighted by |ψ|²
        prob = np.abs(values) ** 2
        prob = prob / prob.sum()
        flat_prob = prob.flatten()

        # Select top points by probability
        indices = np.argsort(flat_prob)[-n_points:]
        X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), Z.flatten()
        values_flat = values.flatten()

        x_pts = X_flat[indices]
        y_pts = Y_flat[indices]
        z_pts = Z_flat[indices]
        v_pts = values_flat[indices]

        # Color by sign
        colors = np.where(v_pts > 0, '#e74c3c', '#3498db')
        sizes = 50 * np.abs(v_pts) / np.abs(v_pts).max()

        ax.scatter(x_pts, y_pts, z_pts, c=colors, s=sizes, alpha=0.6)

        # Mark atom
        ax.scatter([atom_pos[0]], [atom_pos[1]], [atom_pos[2]],
                  color='black', s=150, marker='o')

        ax.set_xlabel('x (Å)')
        ax.set_ylabel('y (Å)')
        ax.set_zlabel('z (Å)')

        element = self.structure.atoms[atom_idx].element
        ax.set_title(f'{element} {orbital} orbital')

        return ax


# === Ribbon and Edge State Visualization Helpers ===

def tile_supercell(positions: np.ndarray, lattice_vector: np.ndarray, n_cells: int = 3) -> np.ndarray:
    """Replicate atom positions along a lattice vector for superlattice view.

    Args:
        positions: Array of atom positions, shape (n_atoms, 2) or (n_atoms, 3)
        lattice_vector: Lattice vector to tile along (e.g., [a, 0] for x-direction)
        n_cells: Number of unit cells to show

    Returns:
        Tiled positions array of shape (n_atoms * n_cells, ndim)
    """
    positions = np.asarray(positions)
    lattice_vector = np.asarray(lattice_vector)

    # Center the superlattice around zero
    offsets = np.arange(n_cells) - (n_cells - 1) / 2

    tiled = []
    for offset in offsets:
        shifted = positions + offset * lattice_vector
        tiled.append(shifted)

    return np.vstack(tiled)


def tile_amplitudes(amplitudes: np.ndarray, n_cells: int = 3) -> np.ndarray:
    """Replicate amplitude array for superlattice view.

    Args:
        amplitudes: Array of amplitudes, shape (n_atoms,)
        n_cells: Number of unit cells

    Returns:
        Tiled amplitudes array of shape (n_atoms * n_cells,)
    """
    return np.tile(amplitudes, n_cells)


def get_nearest_neighbor_bonds(positions: np.ndarray, cutoff: float) -> List[Tuple[int, int]]:
    """Find bonds between atoms within a cutoff distance.

    Args:
        positions: Atom positions, shape (n_atoms, ndim)
        cutoff: Maximum bond length

    Returns:
        List of (i, j) index pairs for bonded atoms
    """
    from scipy.spatial.distance import cdist

    n_atoms = len(positions)
    distances = cdist(positions, positions)

    bonds = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if distances[i, j] < cutoff:
                bonds.append((i, j))

    return bonds


def draw_bonds(ax: plt.Axes, positions: np.ndarray, bond_pairs: List[Tuple[int, int]],
               color: str = 'gray', alpha: float = 0.6, linewidth: float = 1.5,
               zorder: int = 1) -> None:
    """Draw bonds between atoms on a matplotlib axes.

    Args:
        ax: Matplotlib axes
        positions: Atom positions, shape (n_atoms, 2)
        bond_pairs: List of (i, j) index pairs
        color: Bond color
        alpha: Transparency
        linewidth: Line width
        zorder: Drawing order
    """
    from matplotlib.collections import LineCollection

    lines = [[positions[i], positions[j]] for i, j in bond_pairs]
    if lines:
        lc = LineCollection(lines, colors=color, alpha=alpha,
                           linewidths=linewidth, zorder=zorder)
        ax.add_collection(lc)


def draw_pz_lobes(ax: plt.Axes, positions: np.ndarray, amplitudes: np.ndarray,
                  lobe_height: float = 0.3, max_width: float = 0.4,
                  colors: Tuple[str, str] = ('#e74c3c', '#3498db'),
                  alpha: float = 0.7, zorder: int = 2) -> None:
    """Draw stylized pz orbital lobes at atom positions weighted by amplitude.

    Args:
        ax: Matplotlib axes
        positions: Atom positions, shape (n_atoms, 2) where y is the ribbon direction
        amplitudes: Wavefunction amplitudes (can be +/-)
        lobe_height: Vertical offset for lobe centers
        max_width: Maximum lobe width at full amplitude
        colors: (positive_color, negative_color) for upper/lower lobes
        alpha: Transparency
        zorder: Drawing order
    """
    from matplotlib.patches import Ellipse

    # Normalize amplitudes for sizing
    amp_abs = np.abs(amplitudes)
    amp_max = amp_abs.max() if amp_abs.max() > 0 else 1.0

    for pos, amp in zip(positions, amplitudes):
        # Size proportional to |amplitude|
        rel_amp = np.abs(amp) / amp_max
        width = max_width * rel_amp
        height = lobe_height * rel_amp * 1.5  # Elongated vertically

        if rel_amp < 0.05:  # Skip very small lobes
            continue

        # Upper lobe (positive z direction - but shown as +y offset in 2D)
        upper = Ellipse((pos[0], pos[1] + lobe_height),
                        width=width, height=height,
                        color=colors[0], alpha=alpha * rel_amp, zorder=zorder)
        ax.add_patch(upper)

        # Lower lobe (negative z direction)
        lower = Ellipse((pos[0], pos[1] - lobe_height),
                        width=width, height=height,
                        color=colors[1], alpha=alpha * rel_amp, zorder=zorder)
        ax.add_patch(lower)


def plot_ribbon_edge_state(positions: np.ndarray, amplitudes: np.ndarray,
                           lattice_vector: np.ndarray = None, n_cells: int = 5,
                           bond_cutoff: float = 1.5, show_bonds: bool = True,
                           show_lobes: bool = False, ax: plt.Axes = None,
                           title: str = None, cmap: str = 'plasma',
                           figsize: Tuple[float, float] = (12, 8)) -> plt.Axes:
    """Create publication-quality edge state visualization for a ribbon.

    Shows honeycomb lattice with bonds and atom sizes proportional to |ψ|².

    Args:
        positions: Atom positions in unit cell, shape (n_atoms, 2)
        amplitudes: Wavefunction amplitudes at each atom
        lattice_vector: Lattice vector for tiling (default: auto-detect from positions)
        n_cells: Number of unit cells to show along periodic direction
        bond_cutoff: Maximum distance for nearest-neighbor bonds
        show_bonds: Whether to draw bonds between atoms
        show_lobes: Whether to draw stylized pz orbital lobes
        ax: Matplotlib axes (creates new if None)
        title: Plot title
        cmap: Colormap for probability density
        figsize: Figure size

    Returns:
        Matplotlib axes
    """
    positions = np.asarray(positions)
    amplitudes = np.asarray(amplitudes)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Auto-detect lattice vector if not provided
    if lattice_vector is None:
        # Assume x is periodic direction, find extent
        x_extent = positions[:, 0].max() - positions[:, 0].min()
        lattice_vector = np.array([x_extent + bond_cutoff * 0.8, 0])

    # Tile for superlattice view
    tiled_positions = tile_supercell(positions, lattice_vector, n_cells)
    tiled_amplitudes = tile_amplitudes(amplitudes, n_cells)

    # Draw bonds first (underneath atoms)
    if show_bonds:
        bonds = get_nearest_neighbor_bonds(tiled_positions, bond_cutoff)
        draw_bonds(ax, tiled_positions, bonds, color='#555555', alpha=0.5,
                  linewidth=1.2, zorder=1)

    # Calculate probability density for coloring
    prob_density = np.abs(tiled_amplitudes) ** 2
    prob_max = prob_density.max() if prob_density.max() > 0 else 1.0

    # Draw atoms with size proportional to probability
    sizes = 50 + 400 * (prob_density / prob_max)  # Min size 50, max 450

    scatter = ax.scatter(tiled_positions[:, 0], tiled_positions[:, 1],
                        c=prob_density, cmap=cmap, s=sizes, edgecolors='black',
                        linewidths=0.5, zorder=3, vmin=0, vmax=prob_max)

    # Draw pz lobes if requested
    if show_lobes:
        draw_pz_lobes(ax, tiled_positions, tiled_amplitudes, zorder=2)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('|ψ|² (probability density)', fontsize=11)

    # Style
    ax.set_xlabel('x (Å)', fontsize=12)
    ax.set_ylabel('y (Å)', fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.tick_params(labelsize=10)

    # Set reasonable limits with padding
    x_range = tiled_positions[:, 0].max() - tiled_positions[:, 0].min()
    y_range = tiled_positions[:, 1].max() - tiled_positions[:, 1].min()
    padding = max(x_range, y_range) * 0.1
    ax.set_xlim(tiled_positions[:, 0].min() - padding,
                tiled_positions[:, 0].max() + padding)
    ax.set_ylim(tiled_positions[:, 1].min() - padding,
                tiled_positions[:, 1].max() + padding)

    return ax


def plot_edge_vs_bulk_comparison(positions: np.ndarray,
                                  edge_amplitudes: np.ndarray,
                                  bulk_amplitudes: np.ndarray,
                                  lattice_vector: np.ndarray = None,
                                  n_cells: int = 5, bond_cutoff: float = 1.5,
                                  figsize: Tuple[float, float] = (14, 6)) -> plt.Figure:
    """Create side-by-side comparison of edge state vs bulk state.

    Args:
        positions: Atom positions in unit cell
        edge_amplitudes: Wavefunction amplitudes for edge state
        bulk_amplitudes: Wavefunction amplitudes for bulk state
        lattice_vector: Lattice vector for tiling
        n_cells: Number of unit cells to show
        bond_cutoff: Maximum bond distance
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    plot_ribbon_edge_state(positions, edge_amplitudes, lattice_vector, n_cells,
                          bond_cutoff, show_bonds=True, ax=ax1,
                          title='Edge State (E ≈ 0)', cmap='Reds')

    plot_ribbon_edge_state(positions, bulk_amplitudes, lattice_vector, n_cells,
                          bond_cutoff, show_bonds=True, ax=ax2,
                          title='Bulk State (E ≫ 0)', cmap='Blues')

    plt.tight_layout()
    return fig
