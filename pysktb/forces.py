"""Force calculations for tight-binding models.

This module computes atomic forces using the Hellmann-Feynman theorem:
    F_α = -∂E_tot/∂R_α = -∂E_band/∂R_α - ∂E_rep/∂R_α

Example:
    >>> from pysktb import Hamiltonian
    >>> from pysktb.forces import Forces
    >>> ham = Hamiltonian(structure, params)
    >>> forces = Forces(ham)
    >>> F = forces.get_forces(n_electrons=4, nk=[10, 10, 1])
"""

import numpy as np
from typing import Optional, Tuple
import multiprocessing
from joblib import Parallel, delayed

from ._params import get_hop_int


class Forces:
    """Compute atomic forces from tight-binding Hamiltonian.

    Forces are computed as:
        F = F_band + F_repulsive

    where F_band comes from the Hellmann-Feynman theorem and F_repulsive
    from the derivative of the repulsive pair potential.
    """

    def __init__(self, hamiltonian):
        """
        Args:
            hamiltonian: Hamiltonian object with distance-dependent parameters
        """
        self.ham = hamiltonian
        self.structure = hamiltonian.structure
        self.system = hamiltonian.system
        self.n_atoms = len(self.structure.atoms)
        self.n_orbitals = hamiltonian.n_orbitals

    def get_forces(self, n_electrons: int, nk: list = [10, 10, 10],
                   temperature: float = 0.0, mu: Optional[float] = None,
                   soc: bool = True, parallel: bool = True) -> np.ndarray:
        """Compute atomic forces.

        Args:
            n_electrons: Number of electrons in the system
            nk: k-point mesh [nx, ny, nz]
            temperature: Electronic temperature in eV (0 = ground state)
            mu: Chemical potential (computed from n_electrons if None)
            soc: Include spin-orbit coupling
            parallel: Use parallel k-point evaluation

        Returns:
            forces: Array of shape (n_atoms, 3) in eV/Å
        """
        # Generate k-point mesh
        kpts = self._get_kpoint_mesh(nk)
        n_kpts = len(kpts)
        weight = 1.0 / n_kpts

        # Compute band structure forces
        forces_band = self._get_band_forces(
            kpts, n_electrons, weight, temperature, mu, soc, parallel
        )

        # Compute repulsive forces
        forces_rep = self._get_repulsive_forces()

        return forces_band + forces_rep

    def _get_kpoint_mesh(self, nk: list) -> list:
        """Generate uniform k-point mesh."""
        kx = np.linspace(0, 1, nk[0], endpoint=False)
        ky = np.linspace(0, 1, nk[1], endpoint=False)
        kz = np.linspace(0, 1, nk[2], endpoint=False)
        return [[x, y, z] for x in kx for y in ky for z in kz]

    def _get_band_forces(self, kpts: list, n_electrons: int, weight: float,
                         temperature: float, mu: Optional[float],
                         soc: bool, parallel: bool) -> np.ndarray:
        """Compute band structure contribution to forces."""
        forces = np.zeros((self.n_atoms, 3))

        # Get occupation function
        n_bands = self.n_orbitals * 2 if soc else self.n_orbitals

        if parallel:
            num_cores = multiprocessing.cpu_count()
            results = Parallel(n_jobs=num_cores)(
                delayed(self._force_contribution_k)(k, n_electrons, n_bands, soc)
                for k in kpts
            )
            for F_k in results:
                forces += weight * F_k
        else:
            for k in kpts:
                F_k = self._force_contribution_k(k, n_electrons, n_bands, soc)
                forces += weight * F_k

        return forces

    def _force_contribution_k(self, k: list, n_electrons: int,
                              n_bands: int, soc: bool) -> np.ndarray:
        """Compute force contribution from a single k-point."""
        # Solve eigenvalue problem
        ham = self.ham.get_ham(k, l_soc=soc)
        eigenvalues, eigenvectors = np.linalg.eigh(ham)

        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Determine occupation (T=0 for now)
        # soc=True: spin explicit in H, fill n_electrons states
        # soc=False: spin-degenerate, fill n_electrons/2 states
        n_occupied = n_electrons if soc else n_electrons // 2
        n_occupied = min(n_occupied, n_bands)

        forces = np.zeros((self.n_atoms, 3))

        # Compute dH/dR for each atom and direction
        for atom_idx in range(self.n_atoms):
            for direction in range(3):
                dH_dR = self._get_dH_dR(k, atom_idx, direction, soc)

                # Hellmann-Feynman: F = -Σ_n f_n ⟨ψ_n|∂H/∂R|ψ_n⟩
                for n in range(n_occupied):
                    psi_n = eigenvectors[:, n]
                    force_contrib = np.real(np.conj(psi_n) @ dH_dR @ psi_n)
                    if soc:
                        forces[atom_idx, direction] -= force_contrib
                    else:
                        forces[atom_idx, direction] -= 2.0 * force_contrib  # Factor 2 for spin

        return forces

    def _get_dH_dR(self, k: list, atom_idx: int, direction: int,
                   soc: bool) -> np.ndarray:
        """Compute derivative of Hamiltonian with respect to atomic position.

        ∂H(k)/∂R_α = Σ_R exp(ik·R) ∂H_R/∂R_α

        The derivative involves:
        1. Change in hopping from distance change: ∂V/∂d × ∂d/∂R_α
        2. Change in direction cosines: ∂(l,m,n)/∂R_α
        """
        n_orb = self.n_orbitals
        dH = np.zeros((n_orb, n_orb), dtype=complex)

        # Get k in cartesian coordinates
        rec_lat = self.system.structure.lattice.get_rec_lattice()
        k_cart = np.dot(k, rec_lat)

        bond_mat = self.structure.bond_mat
        dist_mat_vec = self.structure.dist_mat_vec
        dist_mat = self.structure.dist_mat

        # Loop over atom pairs involving atom_idx
        for atom_1_i, atom_1 in enumerate(self.structure.atoms):
            for atom_2_i, atom_2 in enumerate(self.structure.atoms):
                # Only consider pairs involving atom_idx
                if atom_1_i != atom_idx and atom_2_i != atom_idx:
                    continue

                for image_idx in range(self.structure.max_image):
                    if bond_mat[image_idx, atom_1_i, atom_2_i] == 0:
                        continue
                    if image_idx == self.structure.max_image // 2 and atom_1_i == atom_2_i:
                        continue

                    # Get distance and direction
                    dist_vec = dist_mat_vec[image_idx, atom_1_i, atom_2_i, :]
                    d = dist_mat[image_idx, atom_1_i, atom_2_i]
                    if d < 1e-10:
                        continue

                    l, m, n = dist_vec / d

                    # Compute derivative contributions
                    # ∂d/∂R_α depends on whether α is atom_1 or atom_2
                    if atom_1_i == atom_idx:
                        sign = -1.0  # Moving atom_1 changes d_vec by -delta
                    else:
                        sign = 1.0   # Moving atom_2 changes d_vec by +delta

                    unit_vec = np.array([l, m, n])
                    dd_dR = sign * unit_vec[direction]  # ∂d/∂R_α

                    # Get hopping parameters and derivatives
                    params, dparams_dd, _ = self.system.get_hop_params_with_derivs(
                        atom_1_i, atom_2_i, image_idx
                    )

                    # Compute ∂H_ij/∂R_α = (∂V/∂d)(∂d/∂R_α) × angular_part
                    #                    + V × ∂(angular_part)/∂R_α
                    dH_pair = self._compute_dH_pair(
                        params, dparams_dd, l, m, n, d, direction,
                        atom_1_i == atom_idx
                    )

                    # Phase factor
                    phase = np.exp(2.0 * np.pi * 1j * np.dot(k_cart, dist_vec))

                    # Also need derivative of phase factor if atom_idx is involved
                    if atom_1_i == atom_idx:
                        dphase_dR = phase * 2.0 * np.pi * 1j * (-rec_lat[:, direction][0])
                    else:
                        dphase_dR = phase * 2.0 * np.pi * 1j * rec_lat[:, direction][0]

                    # Fill in matrix elements
                    for orbit_1_i, orbit_1 in enumerate(atom_1.orbitals):
                        for orbit_2_i, orbit_2 in enumerate(atom_2.orbitals):
                            orb_idx_1 = self.ham.get_orb_ind(orbit_1)
                            orb_idx_2 = self.ham.get_orb_ind(orbit_2)

                            # Get hopping integral
                            h_val = self._get_hop_element(params, l, m, n, orb_idx_1, orb_idx_2)
                            dh_val = dH_pair[orb_idx_1, orb_idx_2]

                            ind_1 = self._get_orbital_index(atom_1_i, orbit_1_i, atom_1.element, orbit_1)
                            ind_2 = self._get_orbital_index(atom_2_i, orbit_2_i, atom_2.element, orbit_2)

                            # Total derivative: d(H*exp(ik·R))/dR = dH/dR*exp + H*d(exp)/dR
                            dH[ind_1, ind_2] += dh_val * phase + h_val * dphase_dR

        # Make Hermitian
        dH = 0.5 * (dH + dH.T.conj())

        if soc:
            # Extend to include spin
            dH_soc = np.zeros((2 * n_orb, 2 * n_orb), dtype=complex)
            dH_soc[:n_orb, :n_orb] = dH
            dH_soc[n_orb:, n_orb:] = dH
            return dH_soc

        return dH

    def _compute_dH_pair(self, params: dict, dparams_dd: dict,
                         l: float, m: float, n: float, d: float,
                         direction: int, is_atom1: bool) -> np.ndarray:
        """Compute derivative of hopping matrix for an atom pair.

        This includes both:
        1. dV/dd contribution (distance change)
        2. dl/dR, dm/dR, dn/dR contributions (direction change)
        """
        # Sign for direction derivative
        sign = -1.0 if is_atom1 else 1.0

        # ∂d/∂R components
        unit_vec = np.array([l, m, n])
        dd_dR = sign * unit_vec[direction]

        # ∂l/∂R, ∂m/∂R, ∂n/∂R
        # l = x/d, so ∂l/∂x = (1 - l²)/d, ∂l/∂y = -lm/d, etc.
        dl_dR = np.zeros(3)
        dm_dR = np.zeros(3)
        dn_dR = np.zeros(3)

        dl_dR[0] = (1 - l*l) / d
        dl_dR[1] = -l*m / d
        dl_dR[2] = -l*n / d

        dm_dR[0] = -l*m / d
        dm_dR[1] = (1 - m*m) / d
        dm_dR[2] = -m*n / d

        dn_dR[0] = -l*n / d
        dn_dR[1] = -m*n / d
        dn_dR[2] = (1 - n*n) / d

        # Apply sign for atom1 vs atom2
        dl = sign * dl_dR[direction]
        dm = sign * dm_dR[direction]
        dn = sign * dn_dR[direction]

        # Compute derivative of hopping matrix
        # H_ij = V_type × angular_factor(l, m, n)
        # dH_ij/dR = dV/dd × dd/dR × angular + V × d(angular)/dR

        # Get full hopping matrix and its derivatives
        params_lmn = {"l": l, "m": m, "n": n}
        params_full = {**params, **params_lmn}
        H_base = np.array(get_hop_int(**params_full))

        # Compute numerical derivative with respect to direction cosines
        eps = 1e-6

        params_lmn_dl = {"l": l + eps, "m": m, "n": n}
        params_lmn_dm = {"l": l, "m": m + eps, "n": n}
        params_lmn_dn = {"l": l, "m": m, "n": n + eps}

        H_dl = np.array(get_hop_int(**{**params, **params_lmn_dl}))
        H_dm = np.array(get_hop_int(**{**params, **params_lmn_dm}))
        H_dn = np.array(get_hop_int(**{**params, **params_lmn_dn}))

        dH_dl = (H_dl - H_base) / eps
        dH_dm = (H_dm - H_base) / eps
        dH_dn = (H_dn - H_base) / eps

        # Derivative from hopping scaling: dV/dd × dd/dR
        H_from_dV = np.zeros_like(H_base)
        for key, dV_dd in dparams_dd.items():
            if key.startswith("V_") and abs(dV_dd) > 1e-15:
                # Get hopping matrix with unit hopping for this parameter
                params_unit = {k: (1.0 if k == key else 0.0) for k in params if k.startswith("V_")}
                params_unit.update(params_lmn)
                H_unit = np.array(get_hop_int(**params_unit))
                H_from_dV += dV_dd * dd_dR * H_unit

        # Total derivative
        dH = H_from_dV + dH_dl * dl + dH_dm * dm + dH_dn * dn

        return dH

    def _get_hop_element(self, params: dict, l: float, m: float, n: float,
                         orb_idx_1: int, orb_idx_2: int) -> complex:
        """Get a single hopping matrix element."""
        params_lmn = {"l": l, "m": m, "n": n}
        params_full = {**params, **params_lmn}
        H = get_hop_int(**params_full)
        return H[orb_idx_1][orb_idx_2]

    def _get_orbital_index(self, atom_i: int, orbit_i: int,
                           element: str, orbit: str) -> int:
        """Get the index of an orbital in the full Hamiltonian."""
        return self.system.all_iter.index((atom_i, orbit_i, element, orbit))

    def _get_repulsive_forces(self) -> np.ndarray:
        """Compute forces from repulsive potentials."""
        forces = np.zeros((self.n_atoms, 3))

        bond_mat = self.structure.bond_mat
        dist_mat_vec = self.structure.dist_mat_vec
        dist_mat = self.structure.dist_mat

        for atom_1_i in range(self.n_atoms):
            for atom_2_i in range(self.n_atoms):
                if atom_1_i >= atom_2_i:
                    continue

                for image_idx in range(self.structure.max_image):
                    if bond_mat[image_idx, atom_1_i, atom_2_i] == 0:
                        continue

                    d = dist_mat[image_idx, atom_1_i, atom_2_i]
                    if d < 1e-10:
                        continue

                    dist_vec = dist_mat_vec[image_idx, atom_1_i, atom_2_i, :]
                    unit_vec = dist_vec / d

                    # Get repulsive force magnitude
                    F_rep = self.system.get_repulsive_force(atom_1_i, atom_2_i, image_idx)

                    # Force on atom 1: along -unit_vec (repulsive)
                    # Force on atom 2: along +unit_vec
                    forces[atom_1_i, :] -= F_rep * unit_vec
                    forces[atom_2_i, :] += F_rep * unit_vec

        return forces

    def get_total_energy(self, n_electrons: int, nk: list = [10, 10, 10],
                         soc: bool = True) -> Tuple[float, float, float]:
        """Compute total energy.

        Args:
            n_electrons: Number of electrons
            nk: k-point mesh
            soc: Include spin-orbit coupling

        Returns:
            E_total: Total energy
            E_band: Band structure energy
            E_rep: Repulsive energy
        """
        # Generate k-point mesh
        kpts = self._get_kpoint_mesh(nk)
        n_kpts = len(kpts)
        weight = 1.0 / n_kpts

        # Band energy
        E_band = 0.0
        n_bands = self.n_orbitals * 2 if soc else self.n_orbitals
        # soc=True: spin explicit in H, fill n_electrons states
        # soc=False: spin-degenerate, fill n_electrons/2 states
        n_occupied = n_electrons if soc else n_electrons // 2

        for k in kpts:
            ham = self.ham.get_ham(k, l_soc=soc)
            eigenvalues = np.linalg.eigvalsh(ham)
            eigenvalues = np.sort(eigenvalues)
            if soc:
                E_band += weight * np.sum(eigenvalues[:n_occupied])
            else:
                E_band += weight * 2.0 * np.sum(eigenvalues[:n_occupied])  # Factor 2 for spin

        # Repulsive energy
        E_rep = self._get_repulsive_energy()

        return E_band + E_rep, E_band, E_rep

    def _get_repulsive_energy(self) -> float:
        """Compute total repulsive energy."""
        E_rep = 0.0

        bond_mat = self.structure.bond_mat
        for atom_1_i in range(self.n_atoms):
            for atom_2_i in range(self.n_atoms):
                if atom_1_i >= atom_2_i:
                    continue

                for image_idx in range(self.structure.max_image):
                    if bond_mat[image_idx, atom_1_i, atom_2_i] == 0:
                        continue

                    E_rep += self.system.get_repulsive_energy(
                        atom_1_i, atom_2_i, image_idx
                    )

        return E_rep
