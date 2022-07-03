import numpy as np


class Atom:
    ORBITALS_ALL = ["s", "px", "py", "pz", "dxy", "dyz", "dxz", "dx2-y2", "dz2", "S"]

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
        assert set(orbitals).issubset(set(Atom.ORBITALS_ALL)), "wrong orbitals"
        self.orbitals = orbitals

    def __repr__(self):
        return "{} {}".format(self.element, self.pos)
