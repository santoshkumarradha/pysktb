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
#=============================================
__version__='0.5'
import numpy as np
import itertools
from copy import deepcopy
from itertools import permutations
from numpy import sqrt
import multiprocessing
from joblib import Parallel, delayed
import numba as nb
import jit_modules as jit_modules
try:
	from scipy.linalg import block_diag,eigh
	from scipy import sparse
	scipy=1
except ImportError:
	print("scipy is not installed; defaulting to numpy (please install scipy for speed improvements)")
scipy=0
numba=1

class Structure(object):
	def __init__(self, lattice, atoms, periodicity=None, name=None, bond_cut=None,numba=numba):
		assert isinstance(lattice, Lattice), 'not Lattice object'
		assert isinstance(atoms, list), 'atoms is not list'
		assert isinstance(atoms[0], Atom), 'atom is not Atom object'
		self.numba=numba
		self.name = name or 'system'
		self.lattice = lattice
		self.atoms = atoms
		self.bond_cut = bond_cut
		self.periodicity = periodicity or [True, True, True]
		self.max_image = 3 ** np.sum(self.periodicity)

		self.bond_mat = self.get_bond_mat()
		self.dist_mat_vec = self.get_dist_matrix_vec()
		self.dist_mat = self.get_dist_matrix()
		self.dir_cos = self.get_dir_cos_all()

	def get_supercell(self,sc,vac=[0,0,0]):
		'''
		 Input-
		 sc:super cell lattice 3x3 
		 vaccume: 1x3
		 returns: pymatgen structure
		 
		 usefull for making use of pymatgen's codes for making finite complex slabs and defects
		'''
		try:import pymatgen as p
		except: print("Needs pymatgen please install using pip install pymatgen")
		new_s=p.Structure(lattice=self.get_lattice(),species=[i.element for i in self.atoms],
		coords=[list(i.pos) for i in self.atoms])
		new_s.make_supercell(sc)
		def get_vaccume(s,vac):
			abc=np.add([new_s.lattice.a,new_s.lattice.b,new_s.lattice.c],vac)
			ang=new_s.lattice.angles
			l=p.core.lattice.Lattice.from_parameters(a=abc[0],b=abc[1],c=abc[2],alpha=ang[0],beta=ang[1],gamma=ang[2])
			return p.Structure(lattice=l,species=new_s.species,coords=new_s.frac_coords)
		final=get_vaccume(new_s,vac)
		return final

	def get_bond_mat(self):
		def get_cutoff(atom_1, atom_2):
			ele_1 = atom_1.element
			ele_2 = atom_2.element
			key_list = list(self.bond_cut.keys())
			if '{}{}'.format(ele_1, ele_2) in key_list:
				pair = '{}{}'.format(ele_1, ele_2)
			elif '{}{}'.format(ele_2, ele_1) in key_list:
				pair = '{}{}'.format(ele_2, ele_1)
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
					cutoff = get_cutoff(atom1, atom2)['NN']
					if cutoff is None:
						continue
					bond_mat[image_i, i, j] = dist_mat[image_i, i, j] < cutoff
		bond_mat_2 = dist_mat > 0

		return bond_mat * bond_mat_2



	def get_lattice(self):return self.lattice.get_matrix()
	def get_pos(self): return np.concatenate([i.pos for i in self.atoms]).ravel()
	def get_dist_matrix(self):
		dist_mat_vec = self.get_dist_matrix_vec()
		dist_mat = np.linalg.norm(dist_mat_vec, axis=-1)
		return dist_mat

	def get_dist_matrix_vec(self):
		def get_dist_vec(pos1, pos2, lat_vecs, l_min=False):
			""" # p1, p2 direct 
				# return angstrom
				# latConst is included in lat_vecs
			"""
			diff = np.array(pos1) - np.array(pos2)
			if np.linalg.norm(diff) ==  0:
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
	def read_poscar(file_name='./POSCAR', kwargs={}):
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
		dist_vec = self.dist_mat_vec
		dist_norm = np.linalg.norm(dist_vec, axis=-1)
		indx_zero = np.where(dist_norm == 0)
		dist_norm[indx_zero]=1E-10
		dir_cos = dist_vec / dist_norm[:,:,:, np.newaxis]
		return dir_cos

class Lattice:
	"""represent lattice of structure
	"""
	def __init__(self, *args):
		"""
		Args:
			a, b, c, alpha, beta, gamma
		"""
		matrix, lat_const = args
		self.matrix = np.array(matrix) * lat_const
		self.a, self.b, self.c, self.alpha, self.beta , self.gamma = \
		self._to_list(matrix, lat_const)

	def _to_list(self, matrix, lat_const):
		""" see http://en.wikipedia.org/wiki/Fractional_coordinates
		"""
		from numpy.linalg import norm

		a = matrix[0] #* lat_const
		b = matrix[1] #* lat_const
		c = matrix[2] #* lat_const

		alpha = np.arctan2(norm(np.cross(b, c)), np.dot(b, c))
		beta  = np.arctan2(norm(np.cross(c, a)), np.dot(c, a))
		gamma = np.arctan2(norm(np.cross(a, b)), np.dot(a, b))

		return norm(a), norm(b), norm(c), alpha, beta, gamma

	def get_matrix(self):
		matrix = self._to_matrix()
		return matrix

	def _to_matrix(self):
		# see http://en.wikipedia.org/wiki/Fractional_coordinates
		# For the special case of a monoclinic cell (a common case) where alpha = gamma = 90 degree and beta > 90 degree, this gives: <- special care needed
		# so far, alpha, beta, gamma < 90 degree
		a, b, c, alpha, beta, gamma = self.a, self.b, self.c, self.alpha, self.beta, self.gamma

		v = a * b * c * np.sqrt(1. - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma) )

		T = np.zeros((3, 3))
		T = np.array([ \
				  [a, b * np.cos(gamma), c * np.cos(beta)                                                  ] ,\
				  [0, b * np.sin(gamma), c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)] ,\
				  [0, 0                , v / (a * b * np.sin(gamma))                                       ] 
					  ])
		matrix = np.zeros((3, 3))
		matrix[:,0] = np.dot(T, np.array((1, 0, 0)))
		matrix[:,1] = np.dot(T, np.array((0, 1, 0)))
		matrix[:,2] = np.dot(T, np.array((0, 0, 1)))
		# return matrix.T
		return self.matrix

	def get_rec_lattice(self):
		"""
		b_i = (a_j x a_k)/ a_i . (a_j x a_k)
		"""
		lat_mat = self.matrix
		rec_lat_mat = np.linalg.inv(lat_mat).T
		return rec_lat_mat

	def __repr__(self):
		_repr = [self.a, self.b, self.c, self.alpha, self.beta , self.gamma]
		_repr = [str(i) for i in _repr]
		return ' '.join(_repr)


class Atom:
	ORBITALS_ALL = ['s',
					'px', 'py', 'pz',
					'dxy', 'dyz', 'dxz', 'dx2-y2', 'dz2',
					'S']
	def __init__(self, element, pos):
		""" Object to represent atom
			Args:
				element:
					atomic symbol eg) 'Si'
				pos:
					atom position (fractional coordinate) eg) [0.5, 0.5, 0] 
				orbitals:
					subset of ['s',
							   'px', 'py', 'pz',
							   'dxy', 'dyz', 'dxz', 'dx2-y2', 'dz2',
							   'S']
		"""

		self.element = element
		self.pos = np.array(pos)
		self.orbitals = None

	def to_list(self):
		out_list = [self.element, self.pos, self.dyn]

		return out_list

	def set_orbitals(self, orbitals=None):
		assert set(orbitals).issubset(set(Atom.ORBITALS_ALL)), 'wrong orbitals'
		self.orbitals = orbitals
	
	def __repr__(self):
		return '{} {}'.format(self.element, self.pos)

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
		sc=dict()
		for i in [''.join(k) for k in  [j for j in itertools.product([i for i in list(self.orbitals.keys())], repeat=2)]]:
			sc[i]=None
		self.scale_params = sc#scale_params

		assert set(self.get_param_key()).issubset(set(self.params.keys())), \
				'wrong parameter set\n' + \
				'given: {}\n'.format(list(self.params.keys())) + \
				'required: {}'.format(self.get_param_key())
		assert self.chk_scale_param_key(), \
			   'The hoping parameters and the exponent parameters are not consistent!'
	
	def get_kpts(self,sp_kpts,kpt_den):
		sp_kpts=[sp_kpts]
		kpt_path = self.get_kpt_path(sp_kpts, kpt_den)
		kpts_len = self.get_kpt_len(kpt_path, self.structure.lattice.get_matrix())
		k_all_path = [kpt for kpt_path_seg in kpt_path
						  for kpt in kpt_path_seg]
		spl_pnts=[]
		for i in sp_kpts[0]:
			 spl_pnts.append(kpts_len[np.all(np.array(k_all_path).reshape(-1,3)==i,axis=1)])
		return k_all_path, kpts_len, np.unique(np.concatenate(spl_pnts).ravel())
	def get_kpt_path(self,sp_kpts, kpt_den=30):
		""" return list of kpoints connecting sp_kpts
			args: 
				sp_kpts: list of k-points paths containing special kpoints
						 [n_path, n_sp_kpt, 3]
				kpt_den: number of k-points btw. sp_kpts
		"""
		kpts = []
		for sp_kpt_path in sp_kpts:
			kpts_path = []
			kpts_path.append(sp_kpt_path[0])
			for kpt_ind, kpt in enumerate(sp_kpt_path):
				if kpt_ind == len(sp_kpt_path) - 1:
					break
				kpt_i = np.array(kpt)
				kpt_f = np.array(sp_kpt_path[kpt_ind + 1])
				for seg_i in range(kpt_den):
					frac = (seg_i + 1.) / float(kpt_den)
					kpt_seg = kpt_f * frac + kpt_i * (1. - frac)
					kpts_path.append(kpt_seg)
			kpts.append(kpts_path)
		return kpts

	def get_kpt_len(self,kpts_path, lat_mat):
		rec_lat_mat = np.linalg.inv(lat_mat).T
		kpts_path_cart = []
		for kpts in kpts_path:
			kpts_cart = []
			for kpt in kpts:
				kpts_cart.append(np.dot(rec_lat_mat, kpt))
			kpts_path_cart.append(kpts_cart)

		kpts_path_len = []
		for kpts_cart in kpts_path_cart:
			kpts_len = []
			for kpt_ind, kpt in enumerate(kpts_cart):

				kpt_diff = kpt - kpts_cart[kpt_ind - 1]
				kpts_len.append(np.linalg.norm(kpt_diff))
			kpts_len[0] = 0
			kpts_path_len.append(kpts_len)
		kpts_path_len = [kpt for kpt_path_seg in kpts_path_len
								for kpt in kpt_path_seg]

		kpts_path_len = np.cumsum(kpts_path_len)

		return kpts_path_len

	def set_orbitals(self):
		for atom in self.structure.atoms:
			atom.set_orbitals(self.orbitals[atom.element])

	def get_all_orbitals(self):
		all_orbitals = []
		for atom in self.structure.atoms:
			for orbit in atom.orbitals:
				all_orbitals.append((atom.element, orbit))
		return all_orbitals

	def get_all_iter(self):
		all_orbitals = []
		for atom_i, atom in enumerate(self.structure.atoms):
			for orbit_i, orbit in enumerate(atom.orbitals):
				all_orbitals.append((atom_i, orbit_i, atom.element, orbit))
		return all_orbitals

	def get_param_key(self):
		elements = self.structure.get_elements()
		key_list = []
		key_list += elements
		for key in itertools.combinations_with_replacement(elements, r=2):
			key_list.append(''.join(key))
		return key_list

	def chk_scale_param_key(self):
		if self.scale_params is None:
			return True

		elements = self.structure.get_elements()
		key_list = self.get_param_key()
		for ele in elements:
			key_list.remove(ele)
		# for key in itertools.product(elements, repeat=2):
		#     key_list.append(''.join(key))
		
		# compare hopping term and exponent
		l_consist = True
		for pair in key_list:
			scale_params = self.scale_params[pair]
			if scale_params is None:
				continue
			hop_orbit = set([hop.replace('V_', '') for hop in self.params[pair]
							 if 'V_' in hop])
			exp_orbit = set([hop.replace('n_', '') for hop in scale_params
							 if 'n_' in hop])
			
			l_consist = l_consist and exp_orbit == hop_orbit
		return l_consist

	def get_hop_params(self, atom_1_i, atom_2_i, image_i):
		""" return parameters dictionary
		"""
		def get_pair(key_list, ele_1, ele_2):
			# key_list = self.system.get_param_key()
			if '{}{}'.format(ele_1, ele_2) in key_list:
				return '{}{}'.format(ele_1, ele_2)
			elif '{}{}'.format(ele_2, ele_1) in key_list:
				return '{}{}'.format(ele_2, ele_1)
			else:
				return None

		atoms = self.structure.atoms
		pair = get_pair(self.get_param_key(), atoms[atom_1_i].element, atoms[atom_2_i].element)
		scale_params = self.scale_params[pair]
		if scale_params is None:
			return self.params[pair]
		else:
			d_0 = scale_params['d_0']
			d = self.structure.dist_mat[image_i, atom_1_i, atom_2_i]
			factor = (d_0 / float(d))

			params_scaled = dict()
			hop_params = self.params[pair]
			for key, hop in list(hop_params.items()):
				orbit = key.replace('V_', 'n_')
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
		assert len(d_mat) == 4, 'tetrahedron required! # of bond = {}'.format(len(d_mat))
		a, b, c, d = d_mat
		vol = 1/6. * np.linalg.det([a-d, b-d, c-d])
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
				b_term = np.array([[l ** 2, lm, nl],
								   [lm, m ** 2, mn],
								   [nl, mn, n ** 2]]) - 1 / 3. * np.eye(3)
				b_term_sum += beta * b_term
			return (e_p + alpha * vol_ratio) * np.eye(3) + b_term_sum
			#return (alpha * vol_ratio) * np.eye(3) + b_term_sum+e_p
			
		def get_onsite_d(e_d, vol_ratio, alpha, beta, gamma, delta_d, dir_cos):
			b_term_sum = 0
			g_term_sum = 0
			for d, dc in zip(delta_d, dir_cos):

				l, m, n = dc
				lm = l * m
				mn = m * n
				nl = n * l
				irt3 = 1 / np.sqrt(3)
				u = (l ** 2 - m ** 2) / 2.
				v = (3 * n ** 2 - 1.) / 2 * irt3
				b_term = np.array([[l ** 2, -lm, -nl, mn, -irt3*mn],
								   [-lm, m ** 2, -mn, -nl, -irt3*nl],
								   [-nl, -mn, n ** 2, 0, 2*irt3*lm],
								   [mn, -nl, 0, n**2, 2*irt3*u],
								   [-irt3*mn, -irt3*nl, 2*irt3*lm, 2*irt3*u, -n**2 + 2/3.]]) - 1 / 3. * np.eye(5)
				g_term = np.array([[mn**2, nl*mn, lm*mn, mn*u, mn*v],
								   [nl*mn, nl**2, nl*lm, nl*u, nl*v],
								   [lm*mn, lm*nl, lm**2, lm*u, lm*v],
								   [mn*u, nl*u, lm*u, u**2, u*v],
								   [mn*v, nl*v, lm*v, u*v, v**2]])

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
				u = (l ** 2 - m ** 2) / 2.
				v = (3 * n ** 2 - 1.) / 2 * irt3

				b_term = np.array([[0, n, m, l, -irt3*l],
								   [n, 0, l, -m, -irt3*m],
								   [m, l, 0, 0, 2*irt3*n]])
				g_term = np.array([[lmn, nl*l, lm*l, l*u, l*v],
								   [mn*m, lmn, lm*m, m*u, m*v],
								   [mn*n, nl*n, lmn, n*u, n*v]])

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
				u = (l ** 2 - m ** 2) / 2.
				v = (3 * n ** 2 - 1.) / 2 * irt3
				b_term = np.array([[mn, nl, lm, u, v]])

				b_term_sum += beta * b_term
			return b_term_sum


		atoms = self.structure.atoms
		params = self.params[atoms[atom_i].element]

		if self.scale_params is None or \
			(not atoms[atom_i].element in self.scale_params or \
			 self.scale_params[atoms[atom_i].element] is None):
			if "s" in atoms[atom_i].orbitals:
				e_s = params['e_s']
			if bool(set(['px', 'py','pz']) & (set(atoms[atom_i].orbitals))):
				if not isinstance(params['e_p'], list):
					e_p=[params['e_p']]*3
				else:
					e_p=params['e_p']

#             if 'px' in  atoms[atom_i].orbitals:
#                 e_p = params['e_p']
#             if 'px' in  atoms[atom_i].orbitals:
#                 e_p = params['e_p']
#             if 'py' in  atoms[atom_i].orbitals:
#                 e_p = params['e_p']
#             if 'pz' in  atoms[atom_i].orbitals:
#                 e_p = params['e_p']
			if 'dxy' in  atoms[atom_i].orbitals:
				e_d = params['e_d']
			if 'S' in  atoms[atom_i].orbitals:
				e_S = params['e_S']

			e_orbit_list =[]
			if 's' in atoms[atom_i].orbitals:
				e_orbit_list += [e_s]
			if 'px' in atoms[atom_i].orbitals:
				e_orbit_list += [e_p[0]]
			if 'py' in atoms[atom_i].orbitals:
				e_orbit_list += [e_p[1]]
			if 'pz' in atoms[atom_i].orbitals:
				e_orbit_list += [e_p[2]]
			if 'dxy' in atoms[atom_i].orbitals:
				e_orbit_list += [e_d]
			if 'dyz' in atoms[atom_i].orbitals:
				e_orbit_list += [e_d]
			if 'dxz' in atoms[atom_i].orbitals:
				e_orbit_list += [e_d]
			if 'dx2-y2' in atoms[atom_i].orbitals:
				e_orbit_list += [e_d]
			if 'dz2' in atoms[atom_i].orbitals:
				e_orbit_list += [e_d]
			if 'S' in atoms[atom_i].orbitals:
				e_orbit_list += [e_S]
			return np.diag(e_orbit_list)
			
		else:
			scale_params = self.scale_params[atoms[atom_i].element]

			d_0 = scale_params['d_0']

			struct = self.structure
			dist_mat_vec = struct.dist_mat_vec
			bond_mat = struct.bond_mat
			dist_vec = dist_mat_vec[:, atom_i, :]
			bond = bond_mat[:, atom_i, :]

			d_mat = dist_vec[bond]

			atom = struct.atoms[atom_i]
			dir_cos = struct.dir_cos[:, atom_i, :, :][bond]
			delta_d = (np.linalg.norm(d_mat, axis=-1) - d_0)/d_0

			orbitals = atom.orbitals
			n_orbitals = len(orbitals)
			# onsite = np.zeros((n_orbitals, n_orbitals))
			# assume ['s',
			#             'px', 'py', 'pz',
			#             'dxy', 'dyz', 'dxz', 'dx2-y2', 'dz2',
			#             'S']
			# TODO generic

			vol = np.average(np.linalg.norm(d_mat, axis=-1))
			vol = (vol**3 - d_0**3)/d_0**3
			vol_ratio = vol

			s_onsite = get_onsite_s(params['e_s'], vol_ratio, scale_params['a_s'])
			S_onsite = get_onsite_s(params['e_S'], vol_ratio, scale_params['a_S'])

			p_onsite = get_onsite_p(params['e_p'], vol_ratio, scale_params['a_p'], 
									scale_params['b_p_0'], scale_params['b_p_1'], delta_d, dir_cos)
			d_onsite = get_onsite_d(params['e_d'], vol_ratio, scale_params['a_d'], 
									scale_params['b_d_0'], 0, delta_d, dir_cos)
			
			pd_onsite = get_onsite_pd(scale_params['b_pd_0'], scale_params['b_pd_1'], 
									  0, 0, delta_d, dir_cos)

			sp_onsite = get_onsite_sp(scale_params['b_sp_0'], dir_cos)
			Sp_onsite = get_onsite_sp(scale_params['b_Sp_0'], dir_cos)
			sd_onsite = get_onsite_sd(scale_params['b_sd_0'], dir_cos)
			Sd_onsite = get_onsite_sd(scale_params['b_Sd_0'], dir_cos)
			sS_onsite = np.zeros((1,1))
			pS_onsite = np.zeros((3,1))
			
			onsite_term = np.bmat(np.r_[np.c_[s_onsite, sp_onsite, sd_onsite, sS_onsite],
										np.c_[sp_onsite.T, p_onsite, pd_onsite, pS_onsite],
										np.c_[sd_onsite.T, pd_onsite.T, d_onsite, Sd_onsite.T],
										np.c_[sS_onsite.T, Sp_onsite, Sd_onsite, S_onsite]])
			return onsite_term

	def _get_soc_mat_i(self, atom_i):
		# only for p_orbitals and need to specify all px py and pz 
		#sigh got to improve on that
		atom = self.structure.atoms[atom_i]
		param = self.params[atom.element]
		orbitals = atom.orbitals

		h_soc = np.zeros((len(orbitals)*2, len(orbitals)*2), dtype=complex)
		if 'lambda' in list(param.keys()):
			assert ''.join(map(str, ['px', 'py', 'pz'])) in ''.join(map(str, orbitals)), \
				   'px, py, and pz should be in orbitals'
			block_diag_list = []

			for orbit_i, orbit in enumerate(orbitals):
				if 'p' in orbit:
					break
			lambda_p = param['lambda']
			h_soc_p = np.array([[0,   0, -1j,   0,   0,   1],
								[0,   0,   0,  1j,   0,   0],
								[0,   0,   0,   0,   0, -1j],
								[0,   0,   0,   0, -1j,   0],
								[0,  -1,   0,   0,   0,   0],
								[0,   0,   0,   0,   0,   0]]) * lambda_p
			h_soc_p += h_soc_p.conj().T
			# orbit_i * 2 for spin 
			rows=[]
			cnt=0
			if "px" in orbitals:
				rows.append(0);rows.append(3);cnt+=1
			if "py" in orbitals:
				rows.append(1);rows.append(4);cnt+=1
			if "pz" in orbitals:
				rows.append(2);rows.append(5);cnt+=1
			rows=np.sort(rows)
			#h_soc_p=h_soc_p[np.ix_(rows,rows)]
			h_soc[orbit_i*2: orbit_i*2+2*cnt, orbit_i*2: orbit_i*2+2*cnt] = h_soc_p
			return h_soc
		else:
			return h_soc

	def get_soc_mat(self):
		import scipy.linalg
		soc_i_list = []
		for atom_i in range(len(self.structure.atoms)):
			soc_i = self._get_soc_mat_i(atom_i)
			soc_i_list.append(soc_i)

		return scipy.linalg.block_diag(*soc_i_list)
from _params import get_hop_int
def parallel_solove_eval(k,ham1,soc):
		ham=ham1.get_ham(k,l_soc=soc)
		eval=ham1._sol_ham(ham,eig_vectors=False)
		return eval[:]
def parallel_solove_eval_and_evec(k,ham1,soc):
		ham=ham1.get_ham(k,l_soc=soc)
		(eval,evec)=ham1._sol_ham(ham,eig_vectors=True)
		return eval,evec
class Hamiltonian(object):
	E_PREFIX = 'e_'
	def __init__(self, structure,inter,numba=1):
		
		self.structure=structure
		self.inter=inter
		t={}
		for i in self.structure.atoms:
			t[i.element]=i.orbitals

		self.system = System(self.structure,t,self.inter)
		self.n_orbitals = len(self.system.all_orbitals)
		self.H_wo_g = np.zeros((self.system.structure.max_image, 
								self.n_orbitals, self.n_orbitals), dtype=complex)
		self.calc_ham_wo_k()
		self.soc_mat = self.system.get_soc_mat()
		self.dist_mat_vec = self.system.structure.dist_mat_vec
		self.bond_mat = self.system.structure.bond_mat
		self.numba=numba
	@staticmethod
	def get_orb_ind(orbit):
		return Atom.ORBITALS_ALL.index(orbit)

	def get_ham(self, kpt, l_soc=True):
		g_mat = self.calc_g(kpt)
		self.g_mat = g_mat
		h = self.H_wo_g * g_mat
		h = np.sum(h, axis=0)
		if l_soc == True:
			if scipy:
				h=block_diag(*(2 * [h]))
			else:
				h = np.kron(h, np.eye(2)) + self.soc_mat
			
		return h

	def _sol_ham(self,ham,eig_vectors=False,spin=False):
		ham_use=ham
		if np.max(ham_use-ham_use.T.conj())>1.0E-9:
			raise Exception("\n\nHamiltonian matrix is not hermitian?!")
		if eig_vectors==False:
			if scipy:
				ham_use_scipy=sparse.csr_matrix(ham_use)
				eval=eigh(ham_use_scipy,eig_vectors=False)
			else:
				eval=np.linalg.eigvalsh(ham_use)
			eval=self.clean_eig(eval)
			return np.array(eval,dtype=float)
		else:
			(eval,eig)=np.linalg.eigh(ham_use)
			eig=eig.T
			(eval,eig)=self.clean_eig(eval,eig)
			return eval,eig

	def solve_kpath(self,k_list=None,eig_vectors=False,soc=True,parallel=1):
			""" solve along a give k path 
			k_list: list of k points (can be generated by get_kpts)
			returns:
			eig:
			eig_vectors: spits out the eigvectors in the format [band*2,kpoint,orbital] (bands*2 for spins)
			parallel: 0 No parallelization (parallelized over k) 
					  1 Parallelized over k
					  2 parallelized using jit optimization (work in progress donot use)
	"""
			
			if parallel==2:
				raise Exception("\n\nparallel=2 not ready yet. please use parallel=1")
				if not (k_list is None):
					nkp=len(k_list)
					ham_list=[]
					if soc==True:
						ret_eval=np.zeros((self.n_orbitals*2,nkp),dtype=np.float64)
						ret_evec=np.zeros((self.n_orbitals*2,nkp,self.n_orbitals*2),dtype=complex)
					else:
						print(2)
						ret_eval=np.zeros((self.n_orbitals,nkp),dtype=np.float64)
						ret_evec=np.zeros((self.n_orbitals,nkp,self.n_orbitals),dtype=complex)
					for i,k in enumerate(k_list):
						ham=self.get_ham(k,l_soc=soc)
						ham_list.append(ham)
						ret_eval,ret_evec=jit_modules.solve_ham_jit(ham_list,eig_vectors,self.n_orbitals)
					if eig_vectors==False:
						# indices of eval are [band,kpoint]
						return ret_eval
					else:
						# indices of eval are [band,kpoint] for evec are [band,kpoint,orbital,(spin)]
						return (ret_eval,ret_evec)
			if parallel==1:
				if not (k_list is None):
					nkp=len(k_list)
					ret_eval=np.zeros((self.n_orbitals*2,nkp),dtype=np.float64)
					ret_evec=np.zeros((self.n_orbitals*2,nkp,self.n_orbitals*2),dtype=complex)
					num_cores = multiprocessing.cpu_count()
					if eig_vectors==False:
						eval=Parallel(n_jobs=num_cores)(delayed(parallel_solove_eval)(i,self,soc) for i in k_list)
						for i,e in enumerate(eval):
							ret_eval[:,i]=e
						# indices of eval are [band,kpoint]
						return ret_eval
					else:
						(evals,evecs)=zip(*Parallel(n_jobs=num_cores)(delayed(parallel_solove_eval_and_evec)(i,self,soc) for i in k_list))
						for i in range(len(evals)):
							ret_eval[:,i]=evals[i][:]
							ret_evec[:,i,:]=evecs[i][:,:]
						# indices of eval are [band,kpoint] for evec are [band,kpoint,orbital,(spin)]
						return (ret_eval,ret_evec)

			if parallel==0:
				if not (k_list is None):

					nkp=len(k_list)
					if soc==True:
						ret_eval=np.zeros((self.n_orbitals*2,nkp),dtype=float)
						ret_evec=np.zeros((self.n_orbitals*2,nkp,self.n_orbitals*2),dtype=complex)
					else:
						print(2)
						ret_eval=np.zeros((self.n_orbitals,nkp),dtype=float)
						ret_evec=np.zeros((self.n_orbitals,nkp,self.n_orbitals),dtype=complex)
					for i,k in enumerate(k_list):
						ham=self.get_ham(k,l_soc=soc)
						if eig_vectors==False:
							eval=self._sol_ham(ham,eig_vectors=eig_vectors)
							ret_eval[:,i]=eval[:]
						else:
							(eval,evec)=self._sol_ham(ham,eig_vectors=eig_vectors)
							ret_eval[:,i]=eval[:]
							ret_evec[:,i,:]=evec[:,:]
					if eig_vectors==False:
						# indices of eval are [band,kpoint]
						return ret_eval
					else:
						# indices of eval are [band,kpoint] for evec are [band,kpoint,orbital,(spin)]
						return (ret_eval,ret_evec)
				   
			
	def solve_k(self,k_point=None,eig_vectors=False):
		if not (k_point is None):
			if eig_vectors==False:
				eval=self.solve_kpath([k_point],eig_vectors=eig_vectors)
				# indices of eval are [band]
				return eval[:,0]
			else:
				(eval,evec)=self.solve_kpath([k_point],eig_vectors=eig_vectors)
				# indices of eval are [band] for evec are [band,orbital,spin]
				return (eval[:,0],evec[:,0,:]) 

	def clean_eig(self,eval,eig=None):
		eval=np.array(eval.real,dtype=float)
		args=eval.argsort()
		eval=eval[args]
		if not (eig is None):
			eig=eig[args]
			return (eval,eig)
		return eval            
	def get_dos(self,energy,eig=None,w=1e-2,nk=[20,20,20]):
		'''
		energy: energy range to get the DOS 
		eig: could passs the energy eig values (useful if the system is 2D or want to generate your own k mesh)
		nk: k point sampling 1x3 for x,y,z directions
		w: gaussian width
		'''
		if eig!=None:
			E=eig
		else:
			kx=np.linspace(0,1,nk[0])
			ky=np.linspace(0,1,nk[1])
			kz=np.linspace(0,1,nk[2])
			E=[]
			for i in kx:
				for j in ky:
					for k in kz:
						E.append(self.solve_k([i,j,k]))
		D=0
		for i in np.array(E).flatten():
			D=D+np.exp(-(energy - i)**2 / (2 * w**2)) / (np.pi * w * np.sqrt(2))
		return D
	def get_kpts(self,path,nk):
		return self.system.get_kpts(path,nk)
	def k_cart2red(self,k):
		red2cart=np.array([self.structure.get_lattice()[i][:len(k)] for i in range(len(k))]).transpose()
		cart2red = np.linalg.inv(red2cart)
		return cart2red @ np.array(k)
	def k_red2cart(self,k):
		red2cart=np.array([self.structure.get_lattice()[i][:len(k)] for i in range(len(k))]).transpose()
		cart2red = np.linalg.inv(red2cart)
		return red2cart @ np.array(k)
	def calc_g(self, kpt):
		""" calc g mat as func of bond matrix, dist_mat_vec, and k
			g mat is phase factor
		"""
		rec_lat = self.system.structure.lattice.get_rec_lattice()
		kpt_cart = np.dot(kpt, rec_lat)
		g_mat = np.zeros((self.system.structure.max_image, 
						  self.n_orbitals, self.n_orbitals), dtype=complex)

		# dist_mat_vec = self.system.structure.dist_mat_vec
		# bond_mat = self.system.structure.bond_mat
		dist_mat_vec=self.dist_mat_vec
		bond_mat=self.bond_mat
		if self.numba:
			g_mat=jit_modules.get_gmat_jit(g_mat,self.system.all_iter,\
				self.system.structure.max_image,self.n_orbitals,bond_mat,dist_mat_vec,kpt_cart)
		else:
			for ind_1, (atom_1_i, orbit_1_i, element_1, orbit_1) in enumerate(self.system.all_iter):
			    for ind_2, (atom_2_i, orbit_2_i, element_2, orbit_2) in enumerate(self.system.all_iter):
			        for image_ind in range(self.system.structure.max_image):
			            if bond_mat[image_ind, atom_1_i, atom_2_i] == 0:
			                continue
			            dist_vec = dist_mat_vec[image_ind, atom_1_i, atom_2_i, :]

			            phase = np.exp(2.*np.pi*1j * np.dot(kpt_cart, dist_vec))
			            g_mat[image_ind, ind_1, ind_2] = phase 
			# non-translated image_ind is self.system.structure.max_image/2
			g_mat[int(self.system.structure.max_image/2), :, :] += np.eye(self.n_orbitals, dtype=complex)
		return g_mat
	def plot_kproj(self,evals,vecs,k_dist,index,ax=None,cmap="bwr"):
		""" plots band structure projected on to subbands
		vecs: eigenvecs in format [band*2,kpoint,orbital] (bands*2 for spins)
		evals: eigen values
		k_dist: distance between k points
		index: orbital index to plot the projection on
		ax: axis object to plot it on
		cmap: colormap value
		
		example :
		evals,vecs=ham.solve_kpath(k_path, eig_vectors=True)
		fig,ax=plt.subplots()
		ham.plot_kproj(evals,vecs,k_dist,index=[0,1],ax=ax)
		
		"""
		index_nums = index
		colors=[]
		for j in range(vecs.shape[0]):
			col=[]
			for i in range(len(k_dist)):
				col.append(np.linalg.norm(vecs[j,i,:][index_nums], ord=2))
			colors.append(col)

		from matplotlib.collections import LineCollection
		import matplotlib.pyplot as plt
		from matplotlib.colors import ListedColormap, BoundaryNorm
		def make_segments(x, y):
			'''
			Create list of line segments from x and y coordinates, in the correct format for LineCollection:
			an array of the form   numlines x (points per line) x 2 (x and y) array
			'''

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
		def colorline(x, y, z=None, cmap=plt.get_cmap(cmap), norm=plt.Normalize(0.0, 1.0), linewidth=2, alpha=1.0):
			'''
			Plot a colored line with coordinates x and y
			Optionally specify colors in the array z
			Optionally specify a colormap, a norm function and a line width
			'''

			# Default colors equally spaced on [0,1]:
			if z is None:
				z = np.linspace(0.0, 1.0, len(x))

			# Special case if a single number:
			if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
				z = np.array([z])

			z = np.asarray(z)

			segments = make_segments(x, y)
			lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)

			ax = plt.gca()
			ax.add_collection(lc)

			return lc
		x = k_dist
		for i in range(vecs.shape[0]):

			y = evals[i]



			colorline(x, y,z=colors[i],alpha=1)

		ax.set_xlim(x.min(), x.max())
		ax.set_ylim(evals.min(), evals.max())
		ax.axhline(0,c="k",linestyle=":",linewidth=1)
		#ax.axvline(0.5,c="k",linestyle=":",linewidth=1)
		return ax
				
	def calc_ham_wo_k(self):
		""" calc hamiltonian with out k
			all g factor is set to 1
		"""
		def get_dir_cos(dist_vec):
			""" return directional cos of distance vector """
			if np.linalg.norm(dist_vec) == 0:
				return 0., 0., 0.
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
					if image_ind == self.system.structure.max_image/2 and atom_1_i == atom_2_i:
						continue

					if bond_mat[image_ind, atom_1_i, atom_2_i] == 0:
						continue
					param_element = self.system.get_hop_params(atom_1_i, atom_2_i, image_ind)

					# get direction cosines
					l, m, n = self.system.structure.get_dir_cos(image_ind, atom_1_i, atom_2_i)
					param_lmn = dict({'l': l, 'm': m, 'n': n,})
					param_element.update(param_lmn)
					hop_int_pair = get_hop_int(**param_element)

					for orbit_1_i, orbit_1 in enumerate(atom_1.orbitals):
						for orbit_2_i, orbit_2 in enumerate(atom_2.orbitals):
							hop_int_ = hop_int_pair[Hamiltonian.get_orb_ind(orbit_1)][Hamiltonian.get_orb_ind(orbit_2)]                            
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

			self.H_wo_g[int(self.system.structure.max_image/2), 
						H_ind: H_ind+len_orbitals, H_ind: H_ind+len_orbitals] = onsite_i
			H_ind += len_orbitals
			

	
