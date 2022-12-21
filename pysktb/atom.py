import numpy as np


class Atom:
    """ Object to represent atom
    
		Args:
  
			element: atomic symbol eg 'Si'
			pos: atom position (fractional coordinate eg- [0.5, 0.5, 0] )
			orbitals: subset of ['s',
					'px', 'py', 'pz',
					'dxy', 'dyz', 'dxz', 'dx2-y2', 'dz2',
					'S']
		"""

    ORBITALS_ALL = ["s", "px", "py", "pz", "dxy", "dyz", "dxz", "dx2-y2", "dz2", "S"]

    def __init__(self, element, pos, orbitals=None):

        self.element = element
        self.pos = np.array(pos)
        if orbitals is not None:
            self.set_orbitals(orbitals)
        else:
            self.orbitals = None

    def to_list(self):
        """return [element, pos, dyn]"""
        return [self.element, self.pos, self.dyn]

    def set_orbitals(self, orbitals=None):
        """set orbitals"""
        assert set(orbitals).issubset(set(Atom.ORBITALS_ALL)), "wrong orbitals"
        self.orbitals = orbitals

    def __repr__(self):
        return f"{self.element} {self.pos}"
