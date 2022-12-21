import itertools
import numpy as np
from scipy import linalg


class System(object):
    """ atomics structures and tight_binding parameters 
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

    def get_hop_params(self, atom_1_i, atom_2_i, image_i):
        """ return parameters dictionary
		"""

        def get_pair(key_list, ele_1, ele_2):
            if "{}{}".format(ele_1, ele_2) in key_list:
                return "{}{}".format(ele_1, ele_2)
            elif "{}{}".format(ele_2, ele_1) in key_list:
                return "{}{}".format(ele_2, ele_1)
            else:
                return None

        atoms = self.structure.atoms
        pair = get_pair(self.get_param_key(), atoms[atom_1_i].element, atoms[atom_2_i].element)
        scale_params = self.scale_params[pair]
        if scale_params is None:
            return self.params[pair]
        d_0 = scale_params["d_0"]
        d = self.structure.dist_mat[image_i, atom_1_i, atom_2_i]
        factor = d_0 / float(d)

        params_scaled = {}
        hop_params = self.params[pair]
        for key, hop in list(hop_params.items()):
            orbit = key.replace("V_", "n_")
            params_scaled[key] = hop * factor ** scale_params[orbit]
        return params_scaled

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
        """ calc onsite term
		"""

        def get_onsite_s(e_s, vol_ratio, alpha):
            return (e_s + alpha * vol_ratio) * np.eye(1)

        def get_onsite_p(e_p, vol_ratio, alpha, beta_0, beta_1, delta_d, dir_cos):
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

        if (
            self.scale_params is None
            or atoms[atom_i].element not in self.scale_params
            or self.scale_params[atoms[atom_i].element] is None
        ):
            if "s" in atoms[atom_i].orbitals:
                e_s = params["e_s"]
            if bool({"px", "py", "pz"} & (set(atoms[atom_i].orbitals))):
                e_p = params["e_p"] if isinstance(params["e_p"], list) else [params["e_p"]] * 3
            #             if 'px' in  atoms[atom_i].orbitals:
            #                 e_p = params['e_p']
            #             if 'px' in  atoms[atom_i].orbitals:
            #                 e_p = params['e_p']
            #             if 'py' in  atoms[atom_i].orbitals:
            #                 e_p = params['e_p']
            #             if 'pz' in  atoms[atom_i].orbitals:
            #                 e_p = params['e_p']
            if "dxy" in atoms[atom_i].orbitals:
                e_d = params["e_d"]
            if "S" in atoms[atom_i].orbitals:
                e_S = params["e_S"]

            e_orbit_list = []
            if "s" in atoms[atom_i].orbitals:
                e_orbit_list += [e_s]
            if "px" in atoms[atom_i].orbitals:
                e_orbit_list += [e_p[0]]
            if "py" in atoms[atom_i].orbitals:
                e_orbit_list += [e_p[1]]
            if "pz" in atoms[atom_i].orbitals:
                e_orbit_list += [e_p[2]]
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

        # only for p_orbitals and need to specify all px py and pz
        # sigh got to improve on that
        atom = self.structure.atoms[atom_i]
        param = self.params[atom.element]
        orbitals = atom.orbitals

        h_soc = np.zeros((len(orbitals) * 2, len(orbitals) * 2), dtype=complex)
        if "lambda" in list(param.keys()):
            assert "".join(map(str, ["px", "py", "pz"])) in "".join(
                map(str, orbitals)
            ), "px, py, and pz should be in orbitals"
            block_diag_list = []

            for orbit_i, orbit in enumerate(orbitals):
                if "p" in orbit:
                    break
            lambda_p = param["lambda"]
            h_soc_p = (
                np.array(
                    [
                        [0, 0, -1j, 0, 0, 1],
                        [0, 0, 0, 1j, 0, 0],
                        [0, 0, 0, 0, 0, -1j],
                        [0, 0, 0, 0, -1j, 0],
                        [0, -1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                )
                * lambda_p
            )
            h_soc_p += h_soc_p.conj().T
            # orbit_i * 2 for spin
            rows = []
            cnt = 0
            if "px" in orbitals:
                rows.append(0)
                rows.append(3)
                cnt += 1
            if "py" in orbitals:
                rows.append(1)
                rows.append(4)
                cnt += 1
            if "pz" in orbitals:
                rows.append(2)
                rows.append(5)
                cnt += 1
            rows = np.sort(rows)
            # h_soc_p=h_soc_p[np.ix_(rows,rows)]
            h_soc[
                orbit_i * 2 : orbit_i * 2 + 2 * cnt, orbit_i * 2 : orbit_i * 2 + 2 * cnt
            ] = h_soc_p
            return h_soc
        else:
            return h_soc

    def get_soc_mat(self):
        """get the spin-orbit coupling matrix"""

        soc_i_list = []
        for atom_i in range(len(self.structure.atoms)):
            soc_i = self._get_soc_mat_i(atom_i)
            soc_i_list.append(soc_i)

        return linalg.block_diag(*soc_i_list)

