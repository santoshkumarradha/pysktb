import numpy as np
import numba as nb
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
warnings.filterwarnings("ignore")


@nb.jit(nopython=False)
def get_gmat_jit(g_mat, all_iter, max_image, n_orbitals, bond_mat, dist_mat_vec, kpt_cart):
    for ind_1, (atom_1_i, orbit_1_i, element_1, orbit_1) in enumerate(all_iter):
        for ind_2, (atom_2_i, orbit_2_i, element_2, orbit_2) in enumerate(all_iter):
            for image_ind in range(max_image):
                if bond_mat[image_ind, atom_1_i, atom_2_i] == 0:
                    continue
                dist_vec = dist_mat_vec[image_ind, atom_1_i, atom_2_i, :]

                phase = np.exp(2.0 * np.pi * 1j * np.dot(kpt_cart, dist_vec))
                g_mat[image_ind, ind_1, ind_2] = phase
    g_mat[int(max_image / 2), :, :] += np.eye(n_orbitals, dtype=complex)
    return g_mat
