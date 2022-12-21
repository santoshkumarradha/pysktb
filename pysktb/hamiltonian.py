#    ____            ____    _  __  _____   ____
#   |  _ \   _   _  / ___|  | |/ / |_   _| | __ )
#   | |_) | | | | | \___ \  | ' /    | |   |  _ \
#   |  __/  | |_| |  ___) | | . \    | |   | |_) |
#   |_|      \__, | |____/  |_|\_\   |_|   |____/
#            |___/
#
#
# by Santosh Kumar Radha,
# srr70@case.edu
# Inspired by various codes
# Used for solving Slater Koster Tightbinding
# hameltonians.
#
# =============================================
__version__ = "0.5"
import numpy as np
from copy import deepcopy
import multiprocessing
from joblib import Parallel, delayed
from .jit_modules import get_gmat_jit
from .utils import energitics

try:
    from scipy.linalg import block_diag, eigh
    from scipy import sparse

    scipy = 1
except ImportError:
    print(
        "scipy is not installed; defaulting to numpy (please install scipy for speed improvements)"
    )

# Initial configuration
# TODO: Move this to a separate comfig file
scipy = False
numba = True


from ._params import get_hop_int
from .system import System
from .atom import Atom
from .lattice import Lattice
from .structure import Structure


def parallel_solove_eigen_val(k, ham1, soc):
    ham = ham1.get_ham(k, l_soc=soc)
    eigen_val = ham1._sol_ham(ham, eig_vectors=False)
    return eigen_val[:]


def parallel_solove_eigen_val_and_evec(k, ham1, soc):
    ham = ham1.get_ham(k, l_soc=soc)
    (eigen_val, evec) = ham1._sol_ham(ham, eig_vectors=True)
    return eigen_val, evec


class Hamiltonian(object):
    """Object to represent hamiltonian of a system """

    E_PREFIX = "e_"

    def __init__(self, structure, inter, numba=1):

        self.structure = structure
        self.inter = inter
        t = {i.element: i.orbitals for i in self.structure.atoms}
        self.system = System(self.structure, t, self.inter)
        self.n_orbitals = len(self.system.all_orbitals)
        self.H_wo_g = np.zeros(
            (self.system.structure.max_image, self.n_orbitals, self.n_orbitals), dtype=complex
        )
        self.calc_ham_wo_k()
        self.soc_mat = self.system.get_soc_mat()
        self.dist_mat_vec = self.system.structure.dist_mat_vec
        self.bond_mat = self.system.structure.bond_mat
        self.numba = numba
        self.orbital_order = self._orbital_order()

    @staticmethod
    def get_orb_ind(orbit):
        """returns the orbital index in the hameltonian"""
        return Atom.ORBITALS_ALL.index(orbit)

    def _orbital_order(self):
        """returns the orbital ordering in the hameltonian
		"""
        orbs = {}
        order = 0
        for i in self.system.all_orbitals:
            orbs[order] = f"{i[0]}-{i[1]}-up"
            order += 1
            orbs[order] = f"{i[0]}-{i[1]}-down"
            order += 1
        return orbs

    def get_ham(self, kpt, l_soc=True):
        """returns the hamiltonian for a given k point"""
        g_mat = self.calc_g(kpt)
        self.g_mat = g_mat
        h = self.H_wo_g * g_mat
        h = np.sum(h, axis=0)
        if l_soc == True:
            h = block_diag(*(2 * [h])) if scipy else np.kron(h, np.eye(2)) + self.soc_mat
        return h

    def _sol_ham(self, ham, eig_vectors=False, spin=False):
        """solves the hamiltonian and returns the eigen values and eigen vectors"""
        # sourcery skip: raise-specific-error
        ham_use = ham
        if np.max(ham_use - ham_use.T.conj()) > 1.0e-9:
            raise Exception("\n\nHamiltonian matrix is not hermitian?!")
        if eig_vectors == False:
            if scipy:
                ham_use_scipy = sparse.csr_matrix(ham_use)
                eigen_val = eigh(ham_use_scipy, eig_vectors=False)
            else:
                eigen_val = np.linalg.eigvalsh(ham_use)
            eigen_val = self.clean_eig(eigen_val)
            return np.array(eigen_val, dtype=float)
        else:
            (eigen_val, eig) = np.linalg.eigh(ham_use)
            eig = eig.T
            (eigen_val, eig) = self.clean_eig(eigen_val, eig)
            return eigen_val, eig

    def solve_kpath(self, k_list=None, eig_vectors=False, soc=True, parallel=1):
        """ solve along a give k path 
			k_list: list of k points (can be generated by get_kpts)
			returns:
			eig:
			eig_vectors: spits out the eigvectors in the format [band*2,kpoint,orbital] (bands*2 for spins)
			parallel: 0 No parallelization (parallelized over k) 
					  1 Parallelized over k
					  2 parallelized using jit optimization (work in progress donot use)
	"""

        if parallel == 2:
            raise Exception("\n\nparallel=2 not ready yet. please use parallel=1")
        # from .jit_modules import solve_ham_jit
        # if not (k_list is None):
        # 	nkp = len(k_list)
        # 	ham_list = []
        # 	if soc == True:
        # 		ret_eigen_val = np.zeros((self.n_orbitals * 2, nkp), dtype=np.float64)
        # 		ret_evec = np.zeros(
        # 			(self.n_orbitals * 2, nkp, self.n_orbitals * 2), dtype=complex
        # 		)
        # 	else:
        # 		print(2)
        # 		ret_eigen_val = np.zeros((self.n_orbitals, nkp), dtype=np.float64)
        # 		ret_evec = np.zeros((self.n_orbitals, nkp, self.n_orbitals), dtype=complex)
        # 	for i, k in enumerate(k_list):
        # 		ham = self.get_ham(k, l_soc=soc)
        # 		ham_list.append(ham)
        # 		ret_eigen_val, ret_evec = solve_ham_jit(
        # 			ham_list, eig_vectors, self.n_orbitals
        # 		)
        # 	if eig_vectors == False:
        # 		# indices of eigen_val are [band,kpoint]
        # 		return ret_eigen_val
        # 	else:
        # 		# indices of eigen_val are [band,kpoint] for evec are [band,kpoint,orbital,(spin)]
        # 		return (ret_eigen_val, ret_evec)
        if parallel == 1 and k_list is not None:
            nkp = len(k_list)
            ret_eigen_val = np.zeros((self.n_orbitals * 2, nkp), dtype=np.float64)
            ret_evec = np.zeros((self.n_orbitals * 2, nkp, self.n_orbitals * 2), dtype=complex)
            num_cores = multiprocessing.cpu_count()
            if eig_vectors == False:
                eigen_val = Parallel(n_jobs=num_cores)(
                    delayed(parallel_solove_eigen_val)(i, self, soc) for i in k_list
                )
                for i, e in enumerate(eigen_val):
                    ret_eigen_val[:, i] = e
                # indices of eigen_val are [band,kpoint]
                return ret_eigen_val
            else:
                (eigen_vals, evecs) = zip(
                    *Parallel(n_jobs=num_cores)(
                        delayed(parallel_solove_eigen_val_and_evec)(i, self, soc) for i in k_list
                    )
                )
                for i in range(len(eigen_vals)):
                    ret_eigen_val[:, i] = eigen_vals[i][:]
                    ret_evec[:, i, :] = evecs[i][:, :]
                # indices of eigen_val are [band,kpoint] for evec are [band,kpoint,orbital,(spin)]
                return (ret_eigen_val, ret_evec)

        if parallel == 0 and k_list is not None:
            nkp = len(k_list)
            if soc == True:
                ret_eigen_val = np.zeros((self.n_orbitals * 2, nkp), dtype=float)
                ret_evec = np.zeros((self.n_orbitals * 2, nkp, self.n_orbitals * 2), dtype=complex)
            else:
                print(2)
                ret_eigen_val = np.zeros((self.n_orbitals, nkp), dtype=float)
                ret_evec = np.zeros((self.n_orbitals, nkp, self.n_orbitals), dtype=complex)
            for i, k in enumerate(k_list):
                ham = self.get_ham(k, l_soc=soc)
                if eig_vectors == False:
                    eigen_val = self._sol_ham(ham, eig_vectors=eig_vectors)
                    ret_eigen_val[:, i] = eigen_val[:]
                else:
                    (eigen_val, evec) = self._sol_ham(ham, eig_vectors=eig_vectors)
                    ret_eigen_val[:, i] = eigen_val[:]
                    ret_evec[:, i, :] = evec[:, :]
            return ret_eigen_val if eig_vectors == False else (ret_eigen_val, ret_evec)

    def solve_k(self, k_point=None, eig_vectors=False):
        """solve the hamiltonian at a single k point"""
        if k_point is not None:
            if eig_vectors == False:
                eigen_val = self.solve_kpath([k_point], eig_vectors=eig_vectors)
                # indices of eigen_val are [band]
                return eigen_val[:, 0]
            else:
                (eigen_val, evec) = self.solve_kpath([k_point], eig_vectors=eig_vectors)
                # indices of eigen_val are [band] for evec are [band,orbital,spin]
                return (eigen_val[:, 0], evec[:, 0, :])

    def clean_eig(self, eigen_val, eig=None):
        """clean the eigen values and eigenvectors"""
        eigen_val = np.array(eigen_val.real, dtype=float)
        args = eigen_val.argsort()
        eigen_val = eigen_val[args]
        if eig is not None:
            eig = eig[args]
            return (eigen_val, eig)
        return eigen_val

    def get_dos(self, energy, eig=None, w=1e-2, nk=[20, 20, 20]):
        """
		energy: energy range to get the DOS 
		eig: could passs the energy eig values (useful if the system is 2D or want to generate your own k mesh)
		nk: k point sampling 1x3 for x,y,z directions
		w: gaussian width
		"""
        if eig is None:
            kx = np.linspace(0, 1, nk[0])
            ky = np.linspace(0, 1, nk[1])
            kz = np.linspace(0, 1, nk[2])
            E = []
            for i in kx:
                for j in ky:
                    E.extend(self.solve_k([i, j, k]) for k in kz)
        else:
            E = eig
        D = 0
        for i in np.array(E).flatten():
            D = D + np.exp(-((energy - i) ** 2) / (2 * w ** 2)) / (np.pi * w * np.sqrt(2))
        return D

    def get_kpts(self, path, nk):
        """get k points along a path"""
        return self.system.get_kpts(path, nk)

    def k_cart2red(self, k):
        """convert k point from cartesian to reduced coordinates"""
        red2cart = np.array(
            [self.structure.get_lattice()[i][: len(k)] for i in range(len(k))]
        ).transpose()
        cart2red = np.linalg.inv(red2cart)
        return cart2red @ np.array(k)

    def k_red2cart(self, k):
        """convert k point from reduced to cartesian coordinates"""
        red2cart = np.array(
            [self.structure.get_lattice()[i][: len(k)] for i in range(len(k))]
        ).transpose()
        cart2red = np.linalg.inv(red2cart)
        return red2cart @ np.array(k)

    def calc_g(self, kpt):
        """ calc g mat as func of bond matrix, dist_mat_vec, and k
			g mat is phase factor
		"""
        rec_lat = self.system.structure.lattice.get_rec_lattice()
        kpt_cart = np.dot(kpt, rec_lat)
        g_mat = np.zeros(
            (self.system.structure.max_image, self.n_orbitals, self.n_orbitals), dtype=complex
        )

        # dist_mat_vec = self.system.structure.dist_mat_vec
        # bond_mat = self.system.structure.bond_mat
        dist_mat_vec = self.dist_mat_vec
        bond_mat = self.bond_mat
        if self.numba:
            g_mat = get_gmat_jit(
                g_mat,
                self.system.all_iter,
                self.system.structure.max_image,
                self.n_orbitals,
                bond_mat,
                dist_mat_vec,
                kpt_cart,
            )
        else:
            for ind_1, (atom_1_i, orbit_1_i, element_1, orbit_1) in enumerate(
                self.system.all_iter
            ):
                for ind_2, (atom_2_i, orbit_2_i, element_2, orbit_2) in enumerate(
                    self.system.all_iter
                ):
                    for image_ind in range(self.system.structure.max_image):
                        if bond_mat[image_ind, atom_1_i, atom_2_i] == 0:
                            continue
                        dist_vec = dist_mat_vec[image_ind, atom_1_i, atom_2_i, :]

                        phase = np.exp(2.0 * np.pi * 1j * np.dot(kpt_cart, dist_vec))
                        g_mat[image_ind, ind_1, ind_2] = phase
            # non-translated image_ind is self.system.structure.max_image/2
            g_mat[int(self.system.structure.max_image / 2), :, :] += np.eye(
                self.n_orbitals, dtype=complex
            )
        return g_mat

    def total_energy(self, filled_band=0, nk=10, dim=3, soc=True):
        """get total energy of the system"""
        return energitics.get_totalenergy(
            deepcopy(self), filled_band=filled_band, nk=nk, dim=dim, soc=soc
        )

    @staticmethod
    def plot_kproj(eigen_vals, vecs, k_dist, index, ax=None, cmap="bwr"):
        """ plots band structure projected on to subbands
		vecs: eigenvecs in format [band*2,kpoint,orbital] (bands*2 for spins)
		eigen_vals: eigen values
		k_dist: distance between k points
		index: orbital index to plot the projection on
		ax: axis object to plot it on
		cmap: colormap value
		
		example :
		eigen_vals,vecs=ham.solve_kpath(k_path, eig_vectors=True)
		fig,ax=plt.subplots()
		ham.plot_kproj(eigen_vals,vecs,k_dist,index=[0,1],ax=ax)
		
		"""
        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
        index_nums = index
        colors = []
        for j in range(vecs.shape[0]):
            col = []
            for i in range(len(k_dist)):
                col.append(np.linalg.norm(vecs[j, i, :][index_nums], ord=2))
            colors.append(col)

        from matplotlib.collections import LineCollection
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm

        def make_segments(x, y):
            """
			Create list of line segments from x and y coordinates, in the correct format for LineCollection:
			an array of the form   numlines x (points per line) x 2 (x and y) array
			"""

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            return segments

        def clear_frame(ax=None):
            # Taken from a post by Tony S Yu
            if ax is None:
                ax = plt.gca()
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            for spine in ax.spines.itervalues():
                spine.set_visible(False)

        def colorline(
            x,
            y,
            z=None,
            cmap=plt.get_cmap(cmap),
            norm=plt.Normalize(0.0, 1.0),
            linewidth=2,
            alpha=1.0,
        ):
            """
			Plot a colored line with coordinates x and y
			Optionally specify colors in the array z
			Optionally specify a colormap, a norm function and a line width
			"""

            # Default colors equally spaced on [0,1]:
            if z is None:
                z = np.linspace(0.0, 1.0, len(x))

            # Special case if a single number:
            if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
                z = np.array([z])

            z = np.asarray(z)

            segments = make_segments(x, y)
            lc = LineCollection(
                segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha
            )

            ax = plt.gca()
            ax.add_collection(lc)

            return lc

        x = k_dist
        for i in range(vecs.shape[0]):

            y = eigen_vals[i]

            colorline(x, y, z=colors[i], alpha=1)

        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(eigen_vals.min(), eigen_vals.max())
        ax.axhline(0, c="k", linestyle=":", linewidth=1)
        # ax.axvline(0.5,c="k",linestyle=":",linewidth=1)
        return ax

    def calc_ham_wo_k(self):
        """ calc hamiltonian with out k
			all g factor is set to 1
		"""

        def get_dir_cos(dist_vec):
            """ return directional cos of distance vector """
            if np.linalg.norm(dist_vec) == 0:
                return 0.0, 0.0, 0.0
            else:
                return dist_vec / np.linalg.norm(dist_vec)

        def get_ind(atom_1_i, orbit_1_i, element_1, orbit_1):
            return self.system.all_iter.index((atom_1_i, orbit_1_i, element_1, orbit_1))

        # params = self.system.params

        # TODO spin interactions

        # off-diagonal
        bond_mat = self.system.structure.bond_mat

        for atom_1_i, atom_1 in enumerate(self.system.structure.atoms):
            for atom_2_i, atom_2 in enumerate(self.system.structure.atoms):
                for image_ind in range(self.system.structure.max_image):
                    if image_ind == self.system.structure.max_image / 2 and atom_1_i == atom_2_i:
                        continue

                    if bond_mat[image_ind, atom_1_i, atom_2_i] == 0:
                        continue
                    param_element = self.system.get_hop_params(atom_1_i, atom_2_i, image_ind)

                    # get direction cosines
                    l, m, n = self.system.structure.get_dir_cos(image_ind, atom_1_i, atom_2_i)
                    param_lmn = dict({"l": l, "m": m, "n": n,})
                    param_element.update(param_lmn)
                    hop_int_pair = get_hop_int(**param_element)

                    for orbit_1_i, orbit_1 in enumerate(atom_1.orbitals):
                        for orbit_2_i, orbit_2 in enumerate(atom_2.orbitals):
                            hop_int_ = hop_int_pair[Hamiltonian.get_orb_ind(orbit_1)][
                                Hamiltonian.get_orb_ind(orbit_2)
                            ]
                            ind_1 = get_ind(atom_1_i, orbit_1_i, atom_1.element, orbit_1)
                            ind_2 = get_ind(atom_2_i, orbit_2_i, atom_2.element, orbit_2)
                            self.H_wo_g[image_ind, ind_1, ind_2] = hop_int_

        # real hermitian -> symmetric
        # self.H_wo_g += np.transpose(self.H_wo_g, [0, 2, 1])#[range(self.H_wo_g.shape[0])[::-1],:,:]

        # diagonal
        H_ind = 0
        for atom_i, atom in enumerate(self.system.structure.atoms):
            len_orbitals = len(atom.orbitals)
            # assert len_orbitals == 10, 'now # of orbitals == {}'.format(len(atom.orbitals))

            onsite_i = self.system.get_onsite_term(atom_i)

            self.H_wo_g[
                int(self.system.structure.max_image / 2),
                H_ind : H_ind + len_orbitals,
                H_ind : H_ind + len_orbitals,
            ] = onsite_i
            H_ind += len_orbitals

