"""
System module for tight-binding calculations.

This module manages atomic structures, orbital assignments, and tight-binding parameters
including on-site energies and spin-orbit coupling (SOC) matrices for s, p, d, and f orbitals.

Author: Santosh Kumar Radha
"""

import itertools
import numpy as np
from scipy import linalg

from .scaling import ScalingLaw, ensure_scaling


class System(object):
    """Atomic structures and tight-binding parameters.

    This class manages the mapping between atoms, orbitals, and tight-binding
    parameters. It handles:
    - Orbital assignment to atoms
    - On-site energy terms with crystal field effects
    - Spin-orbit coupling matrices for p, d, and f orbitals
    - Distance-dependent scaling of hopping parameters
    """

    def __init__(self, structure, orbitals, parameters, scale_params=None):
        self.structure = structure
        self.orbitals = orbitals
        self.set_orbitals()
        self.all_orbitals = self.get_all_orbitals()
        self.all_iter = self.get_all_iter()
        self.params = parameters
        sc = {
            i: None
            for i in [
                "".join(k)
                for k in list(itertools.product(list(list(self.orbitals.keys())), repeat=2))
            ]
        }
        self.scale_params = sc  # scale_params

        assert set(self.get_param_key()).issubset(set(self.params.keys())), (
            "wrong parameter set\n" + f"given: {list(self.params.keys())}\n"
        ) + f"required: {self.get_param_key()}"
        assert (
            self.chk_scale_param_key()
        ), "The hoping parameters and the exponent parameters are not consistent!"

    def get_kpts(self, sp_kpts, kpt_den):
        """ return kpts, kpts_len, spl_pnts"""
        sp_kpts = [sp_kpts]
        kpt_path = self.get_kpt_path(sp_kpts, kpt_den)
        kpts_len = self.get_kpt_len(kpt_path, self.structure.lattice.get_matrix())
        k_all_path = [kpt for kpt_path_seg in kpt_path for kpt in kpt_path_seg]
        spl_pnts = [
            kpts_len[np.all(np.array(k_all_path).reshape(-1, 3) == i, axis=1)] for i in sp_kpts[0]
        ]
        return k_all_path, kpts_len, np.unique(np.concatenate(spl_pnts).ravel())

    def get_kpt_path(self, sp_kpts, kpt_den=30):
        """ return list of kpoints connecting sp_kpts
			args: 
				sp_kpts: list of k-points paths containing special kpoints
						 [n_path, n_sp_kpt, 3]
				kpt_den: number of k-points btw. sp_kpts
		"""
        kpts = []
        for sp_kpt_path in sp_kpts:
            kpts_path = [sp_kpt_path[0]]
            for kpt_ind, kpt in enumerate(sp_kpt_path):
                if kpt_ind == len(sp_kpt_path) - 1:
                    break
                kpt_i = np.array(kpt)
                kpt_f = np.array(sp_kpt_path[kpt_ind + 1])
                for seg_i in range(kpt_den):
                    frac = (seg_i + 1.0) / float(kpt_den)
                    kpt_seg = kpt_f * frac + kpt_i * (1.0 - frac)
                    kpts_path.append(kpt_seg)
            kpts.append(kpts_path)
        return kpts

    def get_kpt_len(self, kpts_path, lat_mat):
        """ return kpts_len"""
        rec_lat_mat = np.linalg.inv(lat_mat).T
        kpts_path_cart = []
        for kpts in kpts_path:
            kpts_cart = [np.dot(rec_lat_mat, kpt) for kpt in kpts]
            kpts_path_cart.append(kpts_cart)

        kpts_path_len = []
        for kpts_cart in kpts_path_cart:
            kpts_len = []
            for kpt_ind, kpt in enumerate(kpts_cart):

                kpt_diff = kpt - kpts_cart[kpt_ind - 1]
                kpts_len.append(np.linalg.norm(kpt_diff))
            kpts_len[0] = 0
            kpts_path_len.append(kpts_len)
        kpts_path_len = [kpt for kpt_path_seg in kpts_path_len for kpt in kpt_path_seg]

        return np.cumsum(kpts_path_len)

    def set_orbitals(self):
        """ set orbitals for each atom"""
        for atom in self.structure.atoms:
            atom.set_orbitals(self.orbitals[atom.element])

    def get_all_orbitals(self):
        """ return all orbitals"""
        all_orbitals = []
        for atom in self.structure.atoms:
            all_orbitals.extend((atom.element, orbit) for orbit in atom.orbitals)
        return all_orbitals

    def get_all_iter(self):
        """ return all orbitals"""
        all_orbitals = []
        for atom_i, atom in enumerate(self.structure.atoms):
            all_orbitals.extend(
                (atom_i, orbit_i, atom.element, orbit)
                for orbit_i, orbit in enumerate(atom.orbitals)
            )
        return all_orbitals

    def get_param_key(self):
        """ return all possible keys for hopping parameters"""
        elements = self.structure.get_elements()
        key_list = []
        key_list += elements
        for key in itertools.combinations_with_replacement(elements, r=2):
            key_list.append("".join(key))
        return key_list

    def chk_scale_param_key(self):
        """ check if the exponent parameters are consistent with the hopping parameters"""
        if self.scale_params is None:
            return True

        elements = self.structure.get_elements()
        key_list = self.get_param_key()
        for ele in elements:
            key_list.remove(ele)

        # compare hopping term and exponent
        l_consist = True
        for pair in key_list:
            scale_params = self.scale_params[pair]
            if scale_params is None:
                continue
            hop_orbit = {hop.replace("V_", "") for hop in self.params[pair] if "V_" in hop}
            exp_orbit = {hop.replace("n_", "") for hop in scale_params if "n_" in hop}

            l_consist = l_consist and exp_orbit == hop_orbit
        return l_consist

    def _get_pair(self, ele_1, ele_2):
        """Get the parameter key for an element pair."""
        key_list = self.get_param_key()
        if f"{ele_1}{ele_2}" in key_list:
            return f"{ele_1}{ele_2}"
        elif f"{ele_2}{ele_1}" in key_list:
            return f"{ele_2}{ele_1}"
        return None

    def _is_hopping_key(self, key):
        """Check if a parameter key is a hopping parameter."""
        # Hopping parameters start with V_ (e.g., V_sss, V_pps)
        # Non-hopping keys include: repulsive, e_s, e_p, etc.
        return key.startswith("V_")

    def get_hop_params(self, atom_1_i, atom_2_i, image_i):
        """Return hopping parameters for an atom pair.

        Handles both constant parameters (floats) and distance-dependent
        scaling laws (ScalingLaw objects). Filters out non-hopping parameters
        like 'repulsive'.
        """
        atoms = self.structure.atoms
        pair = self._get_pair(atoms[atom_1_i].element, atoms[atom_2_i].element)
        all_params = self.params[pair]

        # Filter to only hopping parameters
        hop_params = {k: v for k, v in all_params.items() if self._is_hopping_key(k)}

        # Check if any parameter is a ScalingLaw
        has_scaling = any(isinstance(v, ScalingLaw) for v in hop_params.values())

        if not has_scaling:
            # Legacy path: use scale_params if available
            scale_params = self.scale_params.get(pair)
            if scale_params is None:
                return hop_params
            d_0 = scale_params["d_0"]
            d = self.structure.dist_mat[image_i, atom_1_i, atom_2_i]
            factor = d_0 / float(d)

            params_scaled = {}
            for key, hop in hop_params.items():
                orbit = key.replace("V_", "n_")
                params_scaled[key] = hop * factor ** scale_params[orbit]
            return params_scaled

        # New path: evaluate ScalingLaw objects at actual distance
        d = self.structure.dist_mat[image_i, atom_1_i, atom_2_i]
        params_scaled = {}
        for key, value in hop_params.items():
            if isinstance(value, ScalingLaw):
                params_scaled[key] = value(d)
            else:
                params_scaled[key] = value
        return params_scaled

    def get_hop_params_with_derivs(self, atom_1_i, atom_2_i, image_i):
        """Return hopping parameters and their distance derivatives.

        Returns:
            params: dict of hopping parameter values
            dparams_dd: dict of first derivatives dV/dd
            d2params_dd2: dict of second derivatives d²V/dd²
        """
        atoms = self.structure.atoms
        pair = self._get_pair(atoms[atom_1_i].element, atoms[atom_2_i].element)
        all_params = self.params[pair]
        # Filter to only hopping parameters (exclude 'repulsive', etc.)
        hop_params = {k: v for k, v in all_params.items() if self._is_hopping_key(k)}
        d = self.structure.dist_mat[image_i, atom_1_i, atom_2_i]

        params = {}
        dparams_dd = {}
        d2params_dd2 = {}

        for key, value in hop_params.items():
            if isinstance(value, ScalingLaw):
                params[key] = value(d)
                dparams_dd[key] = value.deriv1(d)
                d2params_dd2[key] = value.deriv2(d)
            else:
                params[key] = value
                dparams_dd[key] = 0.0
                d2params_dd2[key] = 0.0

        return params, dparams_dd, d2params_dd2

    def get_repulsive_energy(self, atom_1_i, atom_2_i, image_i):
        """Get repulsive potential energy for an atom pair.

        Returns:
            energy: Repulsive potential energy (0 if no repulsive defined)
        """
        from .repulsive import RepulsivePotential

        atoms = self.structure.atoms
        pair = self._get_pair(atoms[atom_1_i].element, atoms[atom_2_i].element)
        hop_params = self.params[pair]

        rep = hop_params.get("repulsive")
        if rep is None or not isinstance(rep, RepulsivePotential):
            return 0.0

        d = self.structure.dist_mat[image_i, atom_1_i, atom_2_i]
        return rep(d)

    def get_repulsive_force(self, atom_1_i, atom_2_i, image_i):
        """Get repulsive force magnitude for an atom pair.

        Returns:
            force: -dV_rep/dd (positive = repulsive)
        """
        from .repulsive import RepulsivePotential

        atoms = self.structure.atoms
        pair = self._get_pair(atoms[atom_1_i].element, atoms[atom_2_i].element)
        hop_params = self.params[pair]

        rep = hop_params.get("repulsive")
        if rep is None or not isinstance(rep, RepulsivePotential):
            return 0.0

        d = self.structure.dist_mat[image_i, atom_1_i, atom_2_i]
        return -rep.deriv1(d)

    def has_distance_dependent_params(self):
        """Check if any parameters are distance-dependent ScalingLaws."""
        for key, params in self.params.items():
            if isinstance(params, dict):
                for v in params.values():
                    if isinstance(v, ScalingLaw):
                        return True
        return False

    def calc_volume(self, atom_i):
        """ calc volume of the tetrahedron 
		"""
        struct = self.structure
        dist_mat_vec = struct.dist_mat_vec
        bond_mat = struct.bond_mat
        dist_vec = dist_mat_vec[:, atom_i, :]
        bond = bond_mat[:, atom_i, :]

        d_mat = dist_vec[bond]
        assert len(d_mat) == 4, f"tetrahedron required! # of bond = {len(d_mat)}"
        a, b, c, d = d_mat
        vol = 1 / 6.0 * np.linalg.det([a - d, b - d, c - d])
        print(vol)

    def get_onsite_term(self, atom_i):
        """Calculate on-site term for atom.

        This computes the on-site Hamiltonian matrix including:
        - Orbital energies (e_s, e_p, e_d, e_f, e_S)
        - Crystal field effects via directional cosines
        - Volume-dependent modulation

        Parameters
        ----------
        atom_i : int
            Index of the atom in the structure.

        Returns
        -------
        onsite : ndarray
            On-site Hamiltonian matrix for the atom's orbitals.
        """

        def get_onsite_s(e_s, vol_ratio, alpha):
            """1×1 on-site matrix for s orbital."""
            return (e_s + alpha * vol_ratio) * np.eye(1)

        def get_onsite_p(e_p, vol_ratio, alpha, beta_0, beta_1, delta_d, dir_cos):
            """3×3 on-site matrix for p orbitals with crystal field."""
            b_term_sum = 0
            for d, dc in zip(delta_d, dir_cos):
                beta = beta_0 + beta_1 * d
                l, m, n = dc
                lm = l * m
                mn = m * n
                nl = n * l
                b_term = np.array(
                    [[l ** 2, lm, nl], [lm, m ** 2, mn], [nl, mn, n ** 2]]
                ) - 1 / 3.0 * np.eye(3)
                b_term_sum += beta * b_term
            return (e_p + alpha * vol_ratio) * np.eye(3) + b_term_sum

        def get_onsite_d(e_d, vol_ratio, alpha, beta, gamma, delta_d, dir_cos):
            """5×5 on-site matrix for d orbitals with crystal field."""
            b_term_sum = 0
            g_term_sum = 0
            for d, dc in zip(delta_d, dir_cos):

                l, m, n = dc
                lm = l * m
                mn = m * n
                nl = n * l
                irt3 = 1 / np.sqrt(3)
                u = (l ** 2 - m ** 2) / 2.0
                v = (3 * n ** 2 - 1.0) / 2 * irt3
                b_term = np.array(
                    [
                        [l ** 2, -lm, -nl, mn, -irt3 * mn],
                        [-lm, m ** 2, -mn, -nl, -irt3 * nl],
                        [-nl, -mn, n ** 2, 0, 2 * irt3 * lm],
                        [mn, -nl, 0, n ** 2, 2 * irt3 * u],
                        [-irt3 * mn, -irt3 * nl, 2 * irt3 * lm, 2 * irt3 * u, -(n ** 2) + 2 / 3.0],
                    ]
                ) - 1 / 3.0 * np.eye(5)
                g_term = np.array(
                    [
                        [mn ** 2, nl * mn, lm * mn, mn * u, mn * v],
                        [nl * mn, nl ** 2, nl * lm, nl * u, nl * v],
                        [lm * mn, lm * nl, lm ** 2, lm * u, lm * v],
                        [mn * u, nl * u, lm * u, u ** 2, u * v],
                        [mn * v, nl * v, lm * v, u * v, v ** 2],
                    ]
                )

                b_term_sum += beta * b_term
                g_term_sum += gamma * g_term

            return (e_d + alpha * vol_ratio) * np.eye(5) + beta * b_term + gamma * g_term

        def get_onsite_f(e_f, vol_ratio=0, alpha=0):
            """7×7 on-site matrix for f orbitals.

            For f orbitals, we use a simpler model where crystal field
            splitting is typically small compared to SOC (especially for
            lanthanides). The diagonal on-site energy dominates.

            Parameters
            ----------
            e_f : float
                f orbital on-site energy
            vol_ratio : float
                Volume ratio for strain effects
            alpha : float
                Volume coupling constant

            Returns
            -------
            ndarray
                7×7 diagonal on-site matrix for f orbitals
            """
            return (e_f + alpha * vol_ratio) * np.eye(7)

        def get_onsite_pd(beta_0, beta_1, gamma_0, gamma_1, delta_d, dir_cos):
            b_term_sum = 0
            g_term_sum = 0
            for d, dc in zip(delta_d, dir_cos):
                beta = beta_0 + beta_1 * d
                gamma = gamma_0 + gamma_1 * d

                l, m, n = dc
                lm = l * m
                mn = m * n
                nl = n * l
                lmn = l * m * n
                irt3 = 1 / np.sqrt(3)
                u = (l ** 2 - m ** 2) / 2.0
                v = (3 * n ** 2 - 1.0) / 2 * irt3

                b_term = np.array(
                    [[0, n, m, l, -irt3 * l], [n, 0, l, -m, -irt3 * m], [m, l, 0, 0, 2 * irt3 * n]]
                )
                g_term = np.array(
                    [
                        [lmn, nl * l, lm * l, l * u, l * v],
                        [mn * m, lmn, lm * m, m * u, m * v],
                        [mn * n, nl * n, lmn, n * u, n * v],
                    ]
                )

                b_term_sum += beta * b_term
                g_term_sum += gamma * g_term
            return b_term_sum + g_term_sum

        def get_onsite_sp(beta, dir_cos):
            b_term_sum = 0
            for dc in dir_cos:
                l, m, n = dc
                b_term = np.array([[l, m, n]])
                b_term_sum += beta * b_term
            return b_term_sum

        def get_onsite_sd(beta, dir_cos):
            b_term_sum = 0
            for dc in dir_cos:

                l, m, n = dc
                lm = l * m
                mn = m * n
                nl = n * l
                irt3 = 1 / np.sqrt(3)
                u = (l ** 2 - m ** 2) / 2.0
                v = (3 * n ** 2 - 1.0) / 2 * irt3
                b_term = np.array([[mn, nl, lm, u, v]])

                b_term_sum += beta * b_term
            return b_term_sum

        atoms = self.structure.atoms
        params = self.params[atoms[atom_i].element]

        # f orbital names for checking
        f_orbitals = ["fz3", "fxz2", "fyz2", "fz(x2-y2)", "fxyz", "fx(x2-3y2)", "fy(3x2-y2)"]
        d_orbitals = ["dxy", "dyz", "dxz", "dx2-y2", "dz2"]

        if (
            self.scale_params is None
            or atoms[atom_i].element not in self.scale_params
            or self.scale_params[atoms[atom_i].element] is None
        ):
            # Simple case: no scaling, just diagonal on-site energies
            if "s" in atoms[atom_i].orbitals:
                e_s = params["e_s"]
            if bool({"px", "py", "pz"} & (set(atoms[atom_i].orbitals))):
                e_p = params["e_p"] if isinstance(params["e_p"], list) else [params["e_p"]] * 3
            if bool(set(d_orbitals) & set(atoms[atom_i].orbitals)):
                e_d = params.get("e_d", 0)
            if bool(set(f_orbitals) & set(atoms[atom_i].orbitals)):
                e_f = params.get("e_f", 0)
            if "S" in atoms[atom_i].orbitals:
                e_S = params.get("e_S", 0)

            e_orbit_list = []
            if "s" in atoms[atom_i].orbitals:
                e_orbit_list += [e_s]
            if "px" in atoms[atom_i].orbitals:
                e_orbit_list += [e_p[0]]
            if "py" in atoms[atom_i].orbitals:
                e_orbit_list += [e_p[1]]
            if "pz" in atoms[atom_i].orbitals:
                e_orbit_list += [e_p[2]]
            # d orbitals
            if "dxy" in atoms[atom_i].orbitals:
                e_orbit_list += [e_d]
            if "dyz" in atoms[atom_i].orbitals:
                e_orbit_list += [e_d]
            if "dxz" in atoms[atom_i].orbitals:
                e_orbit_list += [e_d]
            if "dx2-y2" in atoms[atom_i].orbitals:
                e_orbit_list += [e_d]
            if "dz2" in atoms[atom_i].orbitals:
                e_orbit_list += [e_d]
            # f orbitals
            if "fz3" in atoms[atom_i].orbitals:
                e_orbit_list += [e_f]
            if "fxz2" in atoms[atom_i].orbitals:
                e_orbit_list += [e_f]
            if "fyz2" in atoms[atom_i].orbitals:
                e_orbit_list += [e_f]
            if "fz(x2-y2)" in atoms[atom_i].orbitals:
                e_orbit_list += [e_f]
            if "fxyz" in atoms[atom_i].orbitals:
                e_orbit_list += [e_f]
            if "fx(x2-3y2)" in atoms[atom_i].orbitals:
                e_orbit_list += [e_f]
            if "fy(3x2-y2)" in atoms[atom_i].orbitals:
                e_orbit_list += [e_f]
            # S orbital
            if "S" in atoms[atom_i].orbitals:
                e_orbit_list += [e_S]
            return np.diag(e_orbit_list)

        else:
            scale_params = self.scale_params[atoms[atom_i].element]

            d_0 = scale_params["d_0"]

            struct = self.structure
            dist_mat_vec = struct.dist_mat_vec
            bond_mat = struct.bond_mat
            dist_vec = dist_mat_vec[:, atom_i, :]
            bond = bond_mat[:, atom_i, :]

            d_mat = dist_vec[bond]

            atom = struct.atoms[atom_i]
            dir_cos = struct.dir_cos[:, atom_i, :, :][bond]
            delta_d = (np.linalg.norm(d_mat, axis=-1) - d_0) / d_0

            orbitals = atom.orbitals
            n_orbitals = len(orbitals)
            # onsite = np.zeros((n_orbitals, n_orbitals))
            # assume ['s',
            #             'px', 'py', 'pz',
            #             'dxy', 'dyz', 'dxz', 'dx2-y2', 'dz2',
            #             'S']
            # TODO generic

            vol = np.average(np.linalg.norm(d_mat, axis=-1))
            vol = (vol ** 3 - d_0 ** 3) / d_0 ** 3
            vol_ratio = vol

            s_onsite = get_onsite_s(params["e_s"], vol_ratio, scale_params["a_s"])
            S_onsite = get_onsite_s(params["e_S"], vol_ratio, scale_params["a_S"])

            p_onsite = get_onsite_p(
                params["e_p"],
                vol_ratio,
                scale_params["a_p"],
                scale_params["b_p_0"],
                scale_params["b_p_1"],
                delta_d,
                dir_cos,
            )
            d_onsite = get_onsite_d(
                params["e_d"],
                vol_ratio,
                scale_params["a_d"],
                scale_params["b_d_0"],
                0,
                delta_d,
                dir_cos,
            )

            pd_onsite = get_onsite_pd(
                scale_params["b_pd_0"], scale_params["b_pd_1"], 0, 0, delta_d, dir_cos
            )

            sp_onsite = get_onsite_sp(scale_params["b_sp_0"], dir_cos)
            Sp_onsite = get_onsite_sp(scale_params["b_Sp_0"], dir_cos)
            sd_onsite = get_onsite_sd(scale_params["b_sd_0"], dir_cos)
            Sd_onsite = get_onsite_sd(scale_params["b_Sd_0"], dir_cos)
            sS_onsite = np.zeros((1, 1))
            pS_onsite = np.zeros((3, 1))

            onsite_term = np.bmat(
                np.r_[
                    np.c_[s_onsite, sp_onsite, sd_onsite, sS_onsite],
                    np.c_[sp_onsite.T, p_onsite, pd_onsite, pS_onsite],
                    np.c_[sd_onsite.T, pd_onsite.T, d_onsite, Sd_onsite.T],
                    np.c_[sS_onsite.T, Sp_onsite, Sd_onsite, S_onsite],
                ]
            )
            return onsite_term

    def _get_soc_mat_i(self, atom_i):
        """Calculate spin-orbit coupling matrix for a single atom.

        This function computes the SOC Hamiltonian H_SOC = λ L·S for p, d, and f orbitals.
        The matrix is constructed in the basis |orbital, spin⟩ where spin is ↑ or ↓.

        The L·S operator can be written as:
        L·S = Lz·Sz + (L+·S- + L-·S+)/2

        Parameters
        ----------
        atom_i : int
            Index of the atom

        Returns
        -------
        h_soc : ndarray
            SOC matrix of shape (2*n_orbitals, 2*n_orbitals)
        """
        atom = self.structure.atoms[atom_i]
        param = self.params[atom.element]
        orbitals = atom.orbitals

        h_soc = np.zeros((len(orbitals) * 2, len(orbitals) * 2), dtype=complex)

        # Track orbital positions
        p_orbs = ["px", "py", "pz"]
        d_orbs = ["dxy", "dyz", "dxz", "dx2-y2", "dz2"]
        f_orbs = ["fz3", "fxz2", "fyz2", "fz(x2-y2)", "fxyz", "fx(x2-3y2)", "fy(3x2-y2)"]

        # Find starting indices for each orbital type
        p_start = None
        d_start = None
        f_start = None

        for idx, orb in enumerate(orbitals):
            if orb in p_orbs and p_start is None:
                p_start = idx
            if orb in d_orbs and d_start is None:
                d_start = idx
            if orb in f_orbs and f_start is None:
                f_start = idx

        # =====================================================================
        # p-orbital SOC (lambda_p)
        # =====================================================================
        if "lambda" in param or "lambda_p" in param:
            lambda_p = param.get("lambda_p", param.get("lambda", 0))
            if lambda_p != 0 and p_start is not None:
                # Check that all p orbitals are present
                has_all_p = all(p in orbitals for p in p_orbs)
                if has_all_p:
                    # p-orbital SOC matrix in basis [px↑, py↑, pz↑, px↓, py↓, pz↓]
                    # H_SOC = λ_p * L·S
                    # Using |px⟩ = -1/√2(|1,1⟩ - |1,-1⟩), |py⟩ = i/√2(|1,1⟩ + |1,-1⟩), |pz⟩ = |1,0⟩
                    h_soc_p = lambda_p * np.array([
                        [0, -1j, 0, 0, 0, 1],
                        [1j, 0, 0, 0, 0, 1j],
                        [0, 0, 0, -1, -1j, 0],
                        [0, 0, -1, 0, 1j, 0],
                        [0, 0, 1j, -1j, 0, 0],
                        [1, -1j, 0, 0, 0, 0]
                    ], dtype=complex) / 2.0

                    # Place in full matrix
                    # Basis ordering: [orb1↑, orb2↑, ..., orb1↓, orb2↓, ...]
                    n_orb = len(orbitals)
                    for i, pi in enumerate(p_orbs):
                        for j, pj in enumerate(p_orbs):
                            pi_idx = orbitals.index(pi)
                            pj_idx = orbitals.index(pj)
                            # ↑↑ block
                            h_soc[pi_idx, pj_idx] = h_soc_p[i, j]
                            # ↓↓ block
                            h_soc[pi_idx + n_orb, pj_idx + n_orb] = h_soc_p[i + 3, j + 3]
                            # ↑↓ block
                            h_soc[pi_idx, pj_idx + n_orb] = h_soc_p[i, j + 3]
                            # ↓↑ block
                            h_soc[pi_idx + n_orb, pj_idx] = h_soc_p[i + 3, j]

        # =====================================================================
        # d-orbital SOC (lambda_d)
        # =====================================================================
        if "lambda_d" in param:
            lambda_d = param["lambda_d"]
            if lambda_d != 0 and d_start is not None:
                has_all_d = all(d in orbitals for d in d_orbs)
                if has_all_d:
                    # d-orbital SOC matrix in basis [dxy, dyz, dxz, dx2-y2, dz2] × [↑, ↓]
                    # Using real spherical harmonics basis
                    # The L·S matrix for d orbitals (10×10)
                    # Basis: [dxy↑, dyz↑, dxz↑, dx2-y2↑, dz2↑, dxy↓, dyz↓, dxz↓, dx2-y2↓, dz2↓]
                    sqrt3 = np.sqrt(3)

                    h_soc_d = lambda_d * np.array([
                        # dxy↑  dyz↑   dxz↑   dx2-y2↑  dz2↑   dxy↓   dyz↓   dxz↓   dx2-y2↓  dz2↓
                        [0,     1j,    -1,    0,       0,     0,     0,     -1j,   -2,      0],        # dxy↑
                        [-1j,   0,     0,     -1j,     1j*sqrt3, 0,  0,     0,     -1,      -sqrt3],   # dyz↑
                        [1,     0,     0,     1,       sqrt3, 1j,   0,     0,     -1j,     -1j*sqrt3], # dxz↑
                        [0,     1j,    -1,    0,       0,     2,    1,     1j,    0,       0],        # dx2-y2↑
                        [0,     -1j*sqrt3, -sqrt3, 0,  0,     0,    sqrt3, 1j*sqrt3, 0,    0],        # dz2↑
                        [0,     0,     -1j,   2,       0,     0,    -1j,   1,     0,       0],        # dxy↓
                        [0,     0,     0,     1,       sqrt3, 1j,   0,     0,     1j,      -1j*sqrt3], # dyz↓
                        [1j,    0,     0,     -1j,     -1j*sqrt3, -1, 0,   0,     1,       -sqrt3],   # dxz↓
                        [-2,    -1,    1j,    0,       0,     0,    -1j,   1,     0,       0],        # dx2-y2↓
                        [0,     -sqrt3, 1j*sqrt3, 0,   0,     0,    1j*sqrt3, -sqrt3, 0,   0],        # dz2↓
                    ], dtype=complex) / 2.0

                    # Place in full matrix
                    n_orb = len(orbitals)
                    for i, di in enumerate(d_orbs):
                        for j, dj in enumerate(d_orbs):
                            di_idx = orbitals.index(di)
                            dj_idx = orbitals.index(dj)
                            # ↑↑ block
                            h_soc[di_idx, dj_idx] = h_soc_d[i, j]
                            # ↓↓ block
                            h_soc[di_idx + n_orb, dj_idx + n_orb] = h_soc_d[i + 5, j + 5]
                            # ↑↓ block
                            h_soc[di_idx, dj_idx + n_orb] = h_soc_d[i, j + 5]
                            # ↓↑ block
                            h_soc[di_idx + n_orb, dj_idx] = h_soc_d[i + 5, j]

        # =====================================================================
        # f-orbital SOC (lambda_f)
        # =====================================================================
        if "lambda_f" in param:
            lambda_f = param["lambda_f"]
            if lambda_f != 0 and f_start is not None:
                has_all_f = all(f in orbitals for f in f_orbs)
                if has_all_f:
                    # f-orbital SOC matrix in basis [fz3, fxz2, fyz2, fz(x2-y2), fxyz, fx(x2-3y2), fy(3x2-y2)] × [↑, ↓]
                    # Using real spherical harmonics basis
                    # The L·S matrix for f orbitals (14×14)
                    sqrt2 = np.sqrt(2)
                    sqrt3 = np.sqrt(3)
                    sqrt5 = np.sqrt(5)
                    sqrt6 = np.sqrt(6)
                    sqrt10 = np.sqrt(10)
                    sqrt15 = np.sqrt(15)

                    # Build the f-orbital SOC matrix
                    # Basis: [fz3↑, fxz2↑, fyz2↑, fz(x2-y2)↑, fxyz↑, fx(x2-3y2)↑, fy(3x2-y2)↑,
                    #         fz3↓, fxz2↓, fyz2↓, fz(x2-y2)↓, fxyz↓, fx(x2-3y2)↓, fy(3x2-y2)↓]

                    h_soc_f = np.zeros((14, 14), dtype=complex)

                    # Lz eigenvalues for f orbitals in real spherical harmonics basis
                    # fz3 (m=0), fxz2 (m=±1), fyz2 (m=±1), fz(x2-y2) (m=±2), fxyz (m=±2), fx(x2-3y2) (m=±3), fy(3x2-y2) (m=±3)

                    # The L·S = LzSz + (L+S- + L-S+)/2 matrix elements
                    # For real spherical harmonics, we need the matrix representations of Lz, L+, L-

                    # Simplified SOC matrix for f orbitals based on symmetry
                    # This is the standard form from atomic physics literature

                    # ↑↑ block (Lz Sz with Sz = +1/2)
                    h_soc_f[0, 0] = 0  # fz3: m=0
                    h_soc_f[1, 2] = -1j  # fxz2-fyz2 coupling
                    h_soc_f[2, 1] = 1j
                    h_soc_f[3, 4] = -2j  # fz(x2-y2)-fxyz coupling
                    h_soc_f[4, 3] = 2j
                    h_soc_f[5, 6] = -3j  # fx(x2-3y2)-fy(3x2-y2) coupling
                    h_soc_f[6, 5] = 3j

                    # ↓↓ block (Lz Sz with Sz = -1/2)
                    h_soc_f[7, 7] = 0  # fz3
                    h_soc_f[8, 9] = 1j
                    h_soc_f[9, 8] = -1j
                    h_soc_f[10, 11] = 2j
                    h_soc_f[11, 10] = -2j
                    h_soc_f[12, 13] = 3j
                    h_soc_f[13, 12] = -3j

                    # ↑↓ and ↓↑ blocks (L+ S- and L- S+)
                    # L+ raises m by 1, L- lowers m by 1
                    # L±|l,m⟩ = √(l(l+1) - m(m±1)) |l, m±1⟩
                    # For l=3: L+|3,m⟩ = √(12 - m(m+1)) |3,m+1⟩

                    # Coupling between spin-up and spin-down states
                    # fz3 (m=0) couples to fxz2, fyz2 (m=±1)
                    h_soc_f[0, 8] = sqrt6  # fz3↑ - fxz2↓
                    h_soc_f[0, 9] = 1j * sqrt6  # fz3↑ - fyz2↓
                    h_soc_f[8, 0] = sqrt6  # fxz2↓ - fz3↑
                    h_soc_f[9, 0] = -1j * sqrt6

                    # fxz2, fyz2 (m=±1) couple to fz(x2-y2), fxyz (m=±2)
                    h_soc_f[1, 10] = sqrt10  # fxz2↑ - fz(x2-y2)↓
                    h_soc_f[1, 11] = 1j * sqrt10
                    h_soc_f[2, 10] = -1j * sqrt10
                    h_soc_f[2, 11] = sqrt10
                    h_soc_f[10, 1] = sqrt10
                    h_soc_f[11, 1] = -1j * sqrt10
                    h_soc_f[10, 2] = 1j * sqrt10
                    h_soc_f[11, 2] = sqrt10

                    # fz(x2-y2), fxyz (m=±2) couple to fx(x2-3y2), fy(3x2-y2) (m=±3)
                    h_soc_f[3, 12] = sqrt6  # fz(x2-y2)↑ - fx(x2-3y2)↓
                    h_soc_f[3, 13] = 1j * sqrt6
                    h_soc_f[4, 12] = -1j * sqrt6
                    h_soc_f[4, 13] = sqrt6
                    h_soc_f[12, 3] = sqrt6
                    h_soc_f[13, 3] = -1j * sqrt6
                    h_soc_f[12, 4] = 1j * sqrt6
                    h_soc_f[13, 4] = sqrt6

                    # Reverse couplings (↓↑)
                    h_soc_f[7, 1] = sqrt6
                    h_soc_f[7, 2] = -1j * sqrt6
                    h_soc_f[1, 7] = sqrt6
                    h_soc_f[2, 7] = 1j * sqrt6

                    h_soc_f[8, 3] = sqrt10
                    h_soc_f[8, 4] = -1j * sqrt10
                    h_soc_f[9, 3] = 1j * sqrt10
                    h_soc_f[9, 4] = sqrt10
                    h_soc_f[3, 8] = sqrt10
                    h_soc_f[4, 8] = 1j * sqrt10
                    h_soc_f[3, 9] = -1j * sqrt10
                    h_soc_f[4, 9] = sqrt10

                    h_soc_f[10, 5] = sqrt6
                    h_soc_f[10, 6] = -1j * sqrt6
                    h_soc_f[11, 5] = 1j * sqrt6
                    h_soc_f[11, 6] = sqrt6
                    h_soc_f[5, 10] = sqrt6
                    h_soc_f[6, 10] = 1j * sqrt6
                    h_soc_f[5, 11] = -1j * sqrt6
                    h_soc_f[6, 11] = sqrt6

                    # Scale by lambda_f / 2
                    h_soc_f *= lambda_f / 2.0

                    # Place in full matrix
                    n_orb = len(orbitals)
                    for i, fi in enumerate(f_orbs):
                        for j, fj in enumerate(f_orbs):
                            fi_idx = orbitals.index(fi)
                            fj_idx = orbitals.index(fj)
                            # ↑↑ block
                            h_soc[fi_idx, fj_idx] = h_soc_f[i, j]
                            # ↓↓ block
                            h_soc[fi_idx + n_orb, fj_idx + n_orb] = h_soc_f[i + 7, j + 7]
                            # ↑↓ block
                            h_soc[fi_idx, fj_idx + n_orb] = h_soc_f[i, j + 7]
                            # ↓↑ block
                            h_soc[fi_idx + n_orb, fj_idx] = h_soc_f[i + 7, j]

        return h_soc

    def get_soc_mat(self):
        """get the spin-orbit coupling matrix"""

        soc_i_list = []
        for atom_i in range(len(self.structure.atoms)):
            soc_i = self._get_soc_mat_i(atom_i)
            soc_i_list.append(soc_i)

        return linalg.block_diag(*soc_i_list)

