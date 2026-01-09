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
        >>> plotter = OrbitalPlotter(ham)
        >>> plotter.plot_orbital_2d(atom_idx=0, orbital="pz")
        >>> plotter.plot_charge_density_2d(n_electrons=2, nk=[20, 20, 1])
    """

    def __init__(self, hamiltonian, orbital_basis: Optional[OrbitalBasis] = None):
        """
        Args:
            hamiltonian: pysktb Hamiltonian object
            orbital_basis: OrbitalBasis instance. If None, creates one with
                          default Slater exponents.
        """
        self.ham = hamiltonian
        self.structure = hamiltonian.structure

        if orbital_basis is None:
            self.basis = OrbitalBasis.from_defaults(self.structure)
        else:
            self.basis = orbital_basis

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
