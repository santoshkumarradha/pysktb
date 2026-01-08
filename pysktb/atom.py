import numpy as np


class Atom:
    """ Object to represent atom

		Args:

			element: atomic symbol eg 'Si'
			pos: atom position (fractional coordinate eg- [0.5, 0.5, 0] )
			orbitals: subset of ORBITALS_ALL
		"""

    # Full orbital basis: s(1) + p(3) + d(5) + f(7) + S(1) = 17 orbitals
    # Orbital indices:
    #   0: s
    #   1-3: px, py, pz
    #   4-8: dxy, dyz, dxz, dx2-y2, dz2
    #   9-15: fz3, fxz2, fyz2, fz(x2-y2), fxyz, fx(x2-3y2), fy(3x2-y2)
    #   16: S (excited s orbital)
    ORBITALS_ALL = [
        # s orbital (l=0)
        "s",
        # p orbitals (l=1)
        "px", "py", "pz",
        # d orbitals (l=2)
        "dxy", "dyz", "dxz", "dx2-y2", "dz2",
        # f orbitals (l=3) - real spherical harmonics basis
        # fz3: z(5z^2 - 3r^2) / 2
        # fxz2: sqrt(3/8) * x(5z^2 - r^2)
        # fyz2: sqrt(3/8) * y(5z^2 - r^2)
        # fz(x2-y2): sqrt(15)/2 * z(x^2 - y^2)
        # fxyz: sqrt(15) * xyz
        # fx(x2-3y2): sqrt(5/8) * x(x^2 - 3y^2)
        # fy(3x2-y2): sqrt(5/8) * y(3x^2 - y^2)
        "fz3", "fxz2", "fyz2", "fz(x2-y2)", "fxyz", "fx(x2-3y2)", "fy(3x2-y2)",
        # Excited s orbital
        "S"
    ]

    # Orbital family groupings for convenience
    ORBITAL_FAMILIES = {
        "s": ["s"],
        "p": ["px", "py", "pz"],
        "d": ["dxy", "dyz", "dxz", "dx2-y2", "dz2"],
        "f": ["fz3", "fxz2", "fyz2", "fz(x2-y2)", "fxyz", "fx(x2-3y2)", "fy(3x2-y2)"],
        "S": ["S"]
    }

    # Angular momentum quantum numbers
    ORBITAL_L = {
        "s": 0, "px": 1, "py": 1, "pz": 1,
        "dxy": 2, "dyz": 2, "dxz": 2, "dx2-y2": 2, "dz2": 2,
        "fz3": 3, "fxz2": 3, "fyz2": 3, "fz(x2-y2)": 3,
        "fxyz": 3, "fx(x2-3y2)": 3, "fy(3x2-y2)": 3,
        "S": 0
    }

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
