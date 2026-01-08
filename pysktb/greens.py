"""
Green's Function calculations for tight-binding Hamiltonians.

This module provides Green's function based methods for computing:
- Density of States (DOS) with Lorentzian broadening
- Local DOS (LDOS) resolved by atom and orbital
- Spectral functions A(k,E) for ARPES-like visualizations

Author: Santosh Kumar Radha
"""

import numpy as np
from scipy import linalg
from joblib import Parallel, delayed
import multiprocessing


class GreensFunction:
    """
    Green's function calculator for tight-binding Hamiltonians.

    Provides physically-motivated DOS calculations using the retarded
    Green's function G^R(E,k) = (E + iη - H(k))^(-1).

    Parameters
    ----------
    hamiltonian : Hamiltonian
        pysktb Hamiltonian object with structure and parameters defined.

    Examples
    --------
    >>> from pysktb import Structure, Atom, Lattice, Hamiltonian
    >>> from pysktb.greens import GreensFunction
    >>> # ... set up structure and hamiltonian ...
    >>> gf = GreensFunction(ham)
    >>> energies = np.linspace(-5, 5, 500)
    >>> dos = gf.dos(energies, nk=[20, 20, 1])
    """

    def __init__(self, hamiltonian):
        self.ham = hamiltonian
        self.n_orbitals = hamiltonian.n_orbitals
        self.structure = hamiltonian.structure
        self.system = hamiltonian.system

    def retarded(self, E, k, eta=0.01, soc=True):
        """
        Compute the retarded Green's function G^R(E,k).

        G^R(E,k) = (E + iη - H(k))^(-1)

        Parameters
        ----------
        E : float
            Energy value.
        k : array_like
            k-point in fractional coordinates.
        eta : float, optional
            Broadening parameter (default: 0.01 eV).
        soc : bool, optional
            Include spin-orbit coupling (default: True).

        Returns
        -------
        G : ndarray
            Green's function matrix, shape (n, n) where n = n_orbitals * 2 if soc.
        """
        H_k = self.ham.get_ham(k, l_soc=soc)
        n = H_k.shape[0]
        G = linalg.inv((E + 1j * eta) * np.eye(n) - H_k)
        return G

    def spectral(self, E, k, eta=0.01, soc=True):
        """
        Compute the spectral function A(E,k) = -1/π Im[G^R(E,k)].

        The spectral function gives the density of states at a specific
        k-point and energy, useful for ARPES-like visualizations.

        Parameters
        ----------
        E : float
            Energy value.
        k : array_like
            k-point in fractional coordinates.
        eta : float, optional
            Broadening parameter (default: 0.01 eV).
        soc : bool, optional
            Include spin-orbit coupling (default: True).

        Returns
        -------
        A : ndarray
            Spectral function matrix A(E,k).
        """
        G = self.retarded(E, k, eta, soc)
        A = -1.0 / np.pi * G.imag
        return A

    def _generate_kmesh(self, nk):
        """Generate a uniform k-point mesh over the Brillouin zone."""
        nk = np.array(nk)
        k_points = []
        for i in range(nk[0]):
            for j in range(nk[1]):
                for l in range(nk[2]):
                    k = np.array([i / max(nk[0], 1),
                                  j / max(nk[1], 1),
                                  l / max(nk[2], 1)])
                    k_points.append(k)
        return np.array(k_points)

    def dos(self, energies, nk=[20, 20, 20], eta=0.01, soc=True, parallel=True):
        """
        Compute the density of states using Green's functions.

        DOS(E) = -1/π Im[Tr(∫_BZ G^R(E,k) dk)]

        This provides more accurate peak shapes compared to Gaussian
        broadening, with physical Lorentzian lineshapes.

        Parameters
        ----------
        energies : array_like
            Energy values at which to compute DOS.
        nk : list, optional
            k-point mesh density [nkx, nky, nkz] (default: [20, 20, 20]).
        eta : float, optional
            Broadening parameter in eV (default: 0.01).
        soc : bool, optional
            Include spin-orbit coupling (default: True).
        parallel : bool, optional
            Use parallel computation (default: True).

        Returns
        -------
        dos : ndarray
            Density of states at each energy point.
        """
        energies = np.asarray(energies)
        k_mesh = self._generate_kmesh(nk)
        n_kpts = len(k_mesh)

        dos = np.zeros(len(energies))

        if parallel:
            n_cores = multiprocessing.cpu_count()

            def compute_dos_at_energy(E):
                dos_E = 0.0
                for k in k_mesh:
                    A = self.spectral(E, k, eta, soc)
                    dos_E += np.trace(A).real
                return dos_E / n_kpts

            dos = Parallel(n_jobs=n_cores)(
                delayed(compute_dos_at_energy)(E) for E in energies
            )
            dos = np.array(dos)
        else:
            for i, E in enumerate(energies):
                for k in k_mesh:
                    A = self.spectral(E, k, eta, soc)
                    dos[i] += np.trace(A).real
                dos[i] /= n_kpts

        return dos

    def ldos(self, energies, atom_indices=None, orbital_indices=None,
             nk=[20, 20, 20], eta=0.01, soc=True, parallel=True):
        """
        Compute local density of states (LDOS) projected on atoms/orbitals.

        LDOS_i(E) = -1/π Im[G^R_ii(E)]

        Parameters
        ----------
        energies : array_like
            Energy values at which to compute LDOS.
        atom_indices : list, optional
            Atom indices to project onto. If None, computes for all atoms.
        orbital_indices : list, optional
            Orbital indices within each atom. If None, sums over all orbitals.
        nk : list, optional
            k-point mesh density (default: [20, 20, 20]).
        eta : float, optional
            Broadening parameter in eV (default: 0.01).
        soc : bool, optional
            Include spin-orbit coupling (default: True).
        parallel : bool, optional
            Use parallel computation (default: True).

        Returns
        -------
        ldos : ndarray
            Local DOS. Shape depends on atom_indices and orbital_indices.
            If atom_indices is a list, returns shape (n_atoms, n_energies).
        """
        energies = np.asarray(energies)
        k_mesh = self._generate_kmesh(nk)
        n_kpts = len(k_mesh)

        # Determine which orbital indices to use
        if atom_indices is None:
            atom_indices = list(range(len(self.structure.atoms)))

        # Build mapping from atom index to orbital indices in Hamiltonian
        atom_orbital_map = {}
        orbital_idx = 0
        for atom_idx, atom in enumerate(self.structure.atoms):
            n_orb = len(atom.orbitals)
            atom_orbital_map[atom_idx] = list(range(orbital_idx, orbital_idx + n_orb))
            orbital_idx += n_orb

        n_atoms = len(atom_indices)
        ldos = np.zeros((n_atoms, len(energies)))

        if parallel:
            n_cores = multiprocessing.cpu_count()

            def compute_ldos_at_energy(E):
                ldos_E = np.zeros(n_atoms)
                for k in k_mesh:
                    A = self.spectral(E, k, eta, soc)
                    for i, atom_idx in enumerate(atom_indices):
                        orb_indices = atom_orbital_map[atom_idx]
                        if orbital_indices is not None:
                            orb_indices = [orb_indices[j] for j in orbital_indices
                                          if j < len(orb_indices)]
                        # Sum over specified orbitals and both spins
                        for orb_idx in orb_indices:
                            ldos_E[i] += A[orb_idx, orb_idx].real
                            if soc:
                                # Add spin-down contribution
                                ldos_E[i] += A[orb_idx + self.n_orbitals,
                                              orb_idx + self.n_orbitals].real
                return ldos_E / n_kpts

            results = Parallel(n_jobs=n_cores)(
                delayed(compute_ldos_at_energy)(E) for E in energies
            )
            ldos = np.array(results).T
        else:
            for e_idx, E in enumerate(energies):
                for k in k_mesh:
                    A = self.spectral(E, k, eta, soc)
                    for i, atom_idx in enumerate(atom_indices):
                        orb_indices = atom_orbital_map[atom_idx]
                        if orbital_indices is not None:
                            orb_indices = [orb_indices[j] for j in orbital_indices
                                          if j < len(orb_indices)]
                        for orb_idx in orb_indices:
                            ldos[i, e_idx] += A[orb_idx, orb_idx].real
                            if soc:
                                ldos[i, e_idx] += A[orb_idx + self.n_orbitals,
                                                   orb_idx + self.n_orbitals].real
                ldos[:, e_idx] /= n_kpts

        return ldos

    def ldos_by_orbital(self, energies, atom_index=0, nk=[20, 20, 20],
                        eta=0.01, soc=True, parallel=True):
        """
        Compute LDOS resolved by orbital type for a specific atom.

        Parameters
        ----------
        energies : array_like
            Energy values.
        atom_index : int, optional
            Atom index (default: 0).
        nk : list, optional
            k-point mesh density.
        eta : float, optional
            Broadening parameter.
        soc : bool, optional
            Include spin-orbit coupling.
        parallel : bool, optional
            Use parallel computation.

        Returns
        -------
        ldos_orb : dict
            Dictionary mapping orbital names to their LDOS arrays.
        """
        energies = np.asarray(energies)
        k_mesh = self._generate_kmesh(nk)
        n_kpts = len(k_mesh)

        atom = self.structure.atoms[atom_index]
        orbitals = atom.orbitals

        # Find starting index for this atom's orbitals
        start_idx = 0
        for i in range(atom_index):
            start_idx += len(self.structure.atoms[i].orbitals)

        ldos_orb = {orb: np.zeros(len(energies)) for orb in orbitals}

        if parallel:
            n_cores = multiprocessing.cpu_count()

            def compute_at_energy(E):
                result = {orb: 0.0 for orb in orbitals}
                for k in k_mesh:
                    A = self.spectral(E, k, eta, soc)
                    for j, orb in enumerate(orbitals):
                        idx = start_idx + j
                        result[orb] += A[idx, idx].real
                        if soc:
                            result[orb] += A[idx + self.n_orbitals,
                                            idx + self.n_orbitals].real
                for orb in orbitals:
                    result[orb] /= n_kpts
                return result

            results = Parallel(n_jobs=n_cores)(
                delayed(compute_at_energy)(E) for E in energies
            )
            for e_idx, res in enumerate(results):
                for orb in orbitals:
                    ldos_orb[orb][e_idx] = res[orb]
        else:
            for e_idx, E in enumerate(energies):
                for k in k_mesh:
                    A = self.spectral(E, k, eta, soc)
                    for j, orb in enumerate(orbitals):
                        idx = start_idx + j
                        ldos_orb[orb][e_idx] += A[idx, idx].real
                        if soc:
                            ldos_orb[orb][e_idx] += A[idx + self.n_orbitals,
                                                     idx + self.n_orbitals].real
                for orb in orbitals:
                    ldos_orb[orb][e_idx] /= n_kpts

        return ldos_orb

    def spectral_kpath(self, k_path, energies, nk=50, eta=0.01, soc=True):
        """
        Compute spectral function along a k-path for band structure visualization.

        This produces an ARPES-like intensity plot A(k,E).

        Parameters
        ----------
        k_path : list
            List of high-symmetry k-points defining the path.
        energies : array_like
            Energy values.
        nk : int, optional
            Number of k-points between each pair of high-symmetry points.
        eta : float, optional
            Broadening parameter.
        soc : bool, optional
            Include spin-orbit coupling.

        Returns
        -------
        kpts_dist : ndarray
            Cumulative distance along k-path.
        spectral : ndarray
            Spectral function A(k,E), shape (n_kpts, n_energies).
        spl_pnts : ndarray
            Positions of special k-points.
        """
        kpts, kpts_dist, spl_pnts = self.ham.get_kpts(k_path, nk=nk)
        energies = np.asarray(energies)

        spectral = np.zeros((len(kpts), len(energies)))

        for k_idx, k in enumerate(kpts):
            for e_idx, E in enumerate(energies):
                A = self.spectral(E, k, eta, soc)
                spectral[k_idx, e_idx] = np.trace(A).real

        return kpts_dist, spectral, spl_pnts


class SurfaceGreensFunction:
    """
    Surface Green's function for semi-infinite systems.

    Computes surface spectral function using the iterative (decimation)
    method of Lopez-Sancho et al. This reveals topological edge/surface states.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        pysktb Hamiltonian object for a supercell/ribbon structure.
    surface_atoms : list
        Indices of atoms at the surface/edge to project onto.

    References
    ----------
    M.P. Lopez Sancho et al., J. Phys. F: Met. Phys. 15 (1985) 851-858

    Examples
    --------
    >>> # For a graphene zigzag ribbon
    >>> sgf = SurfaceGreensFunction(ham, surface_atoms=[0, 1])
    >>> energies = np.linspace(-3, 3, 300)
    >>> edge_dos = sgf.edge_dos(energies, eta=0.05)
    """

    def __init__(self, hamiltonian, surface_atoms=None):
        self.ham = hamiltonian
        self.n_orbitals = hamiltonian.n_orbitals
        self.structure = hamiltonian.structure

        # Determine surface atoms (default: first few atoms)
        if surface_atoms is None:
            # Default to first atom's orbitals
            surface_atoms = [0]
        self.surface_atoms = surface_atoms

        # Build orbital index mapping for surface atoms
        self._build_surface_indices()

    def _build_surface_indices(self):
        """Build mapping from surface atoms to orbital indices."""
        self.surface_orbital_indices = []
        orbital_idx = 0
        for atom_idx, atom in enumerate(self.structure.atoms):
            n_orb = len(atom.orbitals)
            if atom_idx in self.surface_atoms:
                self.surface_orbital_indices.extend(
                    range(orbital_idx, orbital_idx + n_orb)
                )
            orbital_idx += n_orb

    def surface_spectral(self, E, k, eta=0.01, soc=True):
        """
        Compute surface spectral function at given energy and k-point.

        Parameters
        ----------
        E : float
            Energy value.
        k : array_like
            k-point (for ribbon: 1D k along ribbon direction).
        eta : float
            Broadening parameter.
        soc : bool
            Include spin-orbit coupling.

        Returns
        -------
        A_surf : float
            Surface spectral weight (trace over surface orbitals).
        """
        H_k = self.ham.get_ham(k, l_soc=soc)
        n = H_k.shape[0]

        # Compute full Green's function
        G = linalg.inv((E + 1j * eta) * np.eye(n) - H_k)

        # Extract surface block
        A = -1.0 / np.pi * G.imag

        # Sum over surface orbital indices
        A_surf = 0.0
        for idx in self.surface_orbital_indices:
            A_surf += A[idx, idx]
            if soc and idx + self.n_orbitals < n:
                A_surf += A[idx + self.n_orbitals, idx + self.n_orbitals]

        return A_surf

    def edge_dos(self, energies, k_points=None, nk=50, eta=0.01, soc=True, parallel=True):
        """
        Compute edge/surface density of states.

        For 1D ribbons, integrates over k along the ribbon direction.
        For 2D slabs, integrates over 2D surface BZ.

        Parameters
        ----------
        energies : array_like
            Energy values.
        k_points : array_like, optional
            Custom k-points. If None, generates uniform mesh.
        nk : int
            Number of k-points for integration.
        eta : float
            Broadening parameter.
        soc : bool
            Include spin-orbit coupling.
        parallel : bool
            Use parallel computation.

        Returns
        -------
        edge_dos : ndarray
            Edge density of states at each energy.
        """
        energies = np.asarray(energies)

        # Generate k-points along ribbon direction (1D)
        if k_points is None:
            k_points = np.array([[k, 0, 0] for k in np.linspace(0, 1, nk)])

        n_kpts = len(k_points)
        edge_dos = np.zeros(len(energies))

        if parallel:
            n_cores = multiprocessing.cpu_count()

            def compute_at_energy(E):
                dos_E = 0.0
                for k in k_points:
                    dos_E += self.surface_spectral(E, k, eta, soc)
                return dos_E / n_kpts

            edge_dos = np.array(Parallel(n_jobs=n_cores)(
                delayed(compute_at_energy)(E) for E in energies
            ))
        else:
            for e_idx, E in enumerate(energies):
                for k in k_points:
                    edge_dos[e_idx] += self.surface_spectral(E, k, eta, soc)
                edge_dos[e_idx] /= n_kpts

        return edge_dos

    def edge_spectral_kpath(self, k_values, energies, eta=0.01, soc=True):
        """
        Compute edge spectral function along k-path.

        Creates an intensity plot showing edge states in (k, E) space.

        Parameters
        ----------
        k_values : array_like
            k-values along ribbon direction (0 to 1).
        energies : array_like
            Energy values.
        eta : float
            Broadening parameter.
        soc : bool
            Include spin-orbit coupling.

        Returns
        -------
        spectral : ndarray
            Edge spectral function A(k,E), shape (n_k, n_E).
        """
        k_values = np.asarray(k_values)
        energies = np.asarray(energies)

        spectral = np.zeros((len(k_values), len(energies)))

        for k_idx, k_val in enumerate(k_values):
            k = [k_val, 0, 0]  # 1D k for ribbon
            for e_idx, E in enumerate(energies):
                spectral[k_idx, e_idx] = self.surface_spectral(E, k, eta, soc)

        return spectral
