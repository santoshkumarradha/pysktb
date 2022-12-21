import numpy as np
import itertools
from .lattice import Lattice
from .atom import Atom


class Structure(object):
    """Object to represent structure of system"""

    def __init__(self, lattice, atoms, periodicity=None, name=None, bond_cut=None, numba=True):
        assert isinstance(lattice, Lattice), "not Lattice object"
        assert isinstance(atoms, list), "atoms is not list"
        assert isinstance(atoms[0], Atom), "atom is not Atom object"
        self.numba = numba
        self.name = name or "system"
        self.lattice = lattice
        self.atoms = atoms
        self.bond_cut = bond_cut
        self.periodicity = periodicity or [True, True, True]
        self.max_image = 3 ** np.sum(self.periodicity)

        self.bond_mat = self.get_bond_mat()
        self.dist_mat_vec = self.get_dist_matrix_vec()
        self.dist_mat = self.get_dist_matrix()
        self.dir_cos = self.get_dir_cos_all()

    def get_supercell(self, sc, vac=[0, 0, 0]):
        """
		 Input-
		 sc:super cell lattice 3x3 
		 vaccume: 1x3
		 returns: pymatgen structure
		 
		 usefull for making use of pymatgen's codes for making finite complex slabs and defects
		"""
        try:
            import pymatgen as p
        except:
            print("Needs pymatgen please install using pip install pymatgen")
        new_s = p.Structure(
            lattice=self.get_lattice(),
            species=[i.element for i in self.atoms],
            coords=[list(i.pos) for i in self.atoms],
        )
        new_s.make_supercell(sc)

        def get_vaccume(s, vac):
            abc = np.add([new_s.lattice.a, new_s.lattice.b, new_s.lattice.c], vac)
            ang = new_s.lattice.angles
            l = p.core.lattice.Lattice.from_parameters(
                a=abc[0], b=abc[1], c=abc[2], alpha=ang[0], beta=ang[1], gamma=ang[2]
            )
            return p.Structure(lattice=l, species=new_s.species, coords=new_s.frac_coords)

        final = get_vaccume(new_s, vac)
        return final

    def get_bond_mat(self):
        """return bond matrix"""

        def get_cutoff(atom_1, atom_2):
            ele_1 = atom_1.element
            ele_2 = atom_2.element
            key_list = list(self.bond_cut.keys())
            if "{}{}".format(ele_1, ele_2) in key_list:
                pair = "{}{}".format(ele_1, ele_2)
            elif "{}{}".format(ele_2, ele_1) in key_list:
                pair = "{}{}".format(ele_2, ele_1)
            else:
                return None
            return self.bond_cut[pair]

        max_image = self.max_image
        n_atom = len(self.atoms)
        bond_mat = np.zeros((max_image, n_atom, n_atom), dtype=bool)
        dist_mat = self.get_dist_matrix()
        atoms = self.atoms
        periodic_image = []
        for period in self.periodicity:
            if period:
                periodic_image.append(np.arange(3) - 1)
            else:
                periodic_image.append([0])

        for image_i, image in enumerate(itertools.product(*periodic_image)):
            for i, atom1 in enumerate(atoms):
                for j, atom2 in enumerate(atoms):
                    cutoff = get_cutoff(atom1, atom2)["NN"]
                    if cutoff is None:
                        continue
                    bond_mat[image_i, i, j] = dist_mat[image_i, i, j] < cutoff
        bond_mat_2 = dist_mat > 0

        return bond_mat * bond_mat_2

    def get_lattice(self):
        """return lattice object"""
        return self.lattice.get_matrix()

    def get_pos(self):
        """return position of atoms"""
        return np.concatenate([i.pos for i in self.atoms]).ravel()

    def get_dist_matrix(self):
        """return distance matrix"""
        dist_mat_vec = self.get_dist_matrix_vec()
        return np.linalg.norm(dist_mat_vec, axis=-1)

    def get_dist_matrix_vec(self):
        """return distance matrix vector"""

        def get_dist_vec(pos1, pos2, lat_vecs, l_min=False):
            """ # p1, p2 direct 
				# return angstrom
				# latConst is included in lat_vecs
			"""
            diff = np.array(pos1) - np.array(pos2)
            if np.linalg.norm(diff) == 0:
                return 0
            if l_min:
                diff = diff - np.round(diff)
            diff = np.dot(lat_vecs.T, diff)
            return diff

        n_atom = len(self.atoms)
        max_image = self.max_image

        lat_vecs = self.lattice.get_matrix()
        atoms = self.atoms
        d_mat = np.zeros((max_image, n_atom, n_atom, 3))
        periodic_image = []
        for period in self.periodicity:
            if period:
                periodic_image.append(np.arange(3) - 1)
            else:
                periodic_image.append([0])

        for image_i, image in enumerate(itertools.product(*periodic_image)):
            for i, atom1 in enumerate(atoms):
                for j, atom2 in enumerate(atoms):
                    diff = get_dist_vec(atom1.pos + image, atom2.pos, lat_vecs)
                    d_mat[image_i, i, j, :] = diff
        return d_mat

    def get_elements(self):
        """return list of elements eg) ['Si', 'O']"""
        from collections import OrderedDict

        return list(OrderedDict.fromkeys([atom.element for atom in self.atoms]))

    @staticmethod
    def read_poscar(file_name="./POSCAR", kwargs={}):
        """read POSCAR file and return Structure object (NOT SUPPORTED YET)"""
        # TODO: add support for POSCAR Files
        raise NotImplementedError
        lat_const, lattice_mat, atom_set_direct, dynamics = readPOSCAR(fileName=file_name)

        atoms = []
        for a in atom_set_direct:
            atoms.append(Atom(a[0], a[1]))

        bravais_lat = np.array(lattice_mat)
        lattice = Lattice(bravais_lat, lat_const)

        structure = Structure(lattice, atoms, **kwargs)
        return structure

    def get_dir_cos(self, image_i, atoms_i, atom_j):
        """ return directional cos of distance vector """
        dist_vec = self.dist_mat_vec[image_i, atoms_i, atom_j, :]
        if np.linalg.norm(dist_vec) == 0:
            return 0, 0, 0
        else:
            return dist_vec / np.linalg.norm(dist_vec)

    def get_dir_cos_all(self):
        """return directional cos of distance vector"""
        dist_vec = self.dist_mat_vec
        dist_norm = np.linalg.norm(dist_vec, axis=-1)
        indx_zero = np.where(dist_norm == 0)
        dist_norm[indx_zero] = 1e-10
        return dist_vec / dist_norm[:, :, :, np.newaxis]

