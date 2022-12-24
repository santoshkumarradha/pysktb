from scipy import sparse
import scipy.sparse.linalg as slg
import scipy.linalg as lg
from joblib import Parallel, delayed
import numpy as np
from multiprocessing import cpu_count


def _solve_scipy(ham, k, eig_vectors=False, neig=0, soc=True):
    """only neig implemeneted for now"""
    ham_k = ham.get_ham(k, soc)
    if neig == 0:
        neig = int(ham_k.shape[0] / 2)
    if neig:
        sparce_hk = sparse.csr_matrix(ham_k)
        if not eig_vectors:
            return slg.eigsh(sparce_hk, neig, return_eigenvectors=False, which="SR")
        eigs, evecs = slg.eigsh(sparce_hk, neig, return_eigenvectors=True, which="SR")
        return eigs, evecs
    elif eig_vectors:
        eigs, evecs = slg.eigs(sparce_hk, neig, return_eigenvectors=True)
    else:
        eigs = slg.eigs(sparce_hk, neig, return_eigenvectors=True)


def solve_k_scipy(
    ham, klist=None, method="list", eig_vectors=False, dim=3, nk=100, neig=0, soc=True
):
    if method == "mesh":
        if dim == 3:
            x_p = np.linspace(0, 1, nk)
            k = np.vstack(np.meshgrid(x_p, x_p, x_p)).reshape(3, -1).T
        if dim == 2:
            x_p = np.linspace(0, 1, nk)
            k = np.hstack(
                (
                    np.vstack(np.meshgrid(x_p, x_p)).reshape(2, -1).T,
                    np.zeros(nk ** 2).reshape(-1, 1),
                )
            )
        if dim == 1:
            x_p = np.linspace(0, 1, nk)
            k = np.hstack(
                (np.vstack(np.meshgrid(x_p)).reshape(1, -1).T, np.zeros(2 * nk).reshape(-1, 2))
            )
    if method == "list":
        if klist != None:
            k = klist
        else:
            print("klist cannot be empty in list mode", sys.exc_info()[0])
    num_cores = cpu_count()
    eval = Parallel(n_jobs=num_cores)(
        delayed(_solve_scipy)(ham, i, eig_vectors, neig, soc) for i in k
    )
    return np.real(eval)


def get_totalenergy(ham, filled_band=0, nk=10, dim=3, soc=True):
    """Solve for total energy if filled_band=0 then fermi is set at half filling. 
        Caution for metallic system as fermi level is not calculated !
    """
    evals = solve_k_scipy(ham, method="mesh", dim=dim, nk=nk, soc=soc, neig=filled_band)
    return np.mean(evals)
