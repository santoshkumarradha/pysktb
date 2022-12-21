import numpy as np


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
        self.a, self.b, self.c, self.alpha, self.beta, self.gamma = self._to_list(
            matrix, lat_const
        )

    def _to_list(self, matrix, lat_const):
        """ see http://en.wikipedia.org/wiki/Fractional_coordinates
		"""
        from numpy.linalg import norm

        a = matrix[0]  # * lat_const
        b = matrix[1]  # * lat_const
        c = matrix[2]  # * lat_const

        alpha = np.arctan2(norm(np.cross(b, c)), np.dot(b, c))
        beta = np.arctan2(norm(np.cross(c, a)), np.dot(c, a))
        gamma = np.arctan2(norm(np.cross(a, b)), np.dot(a, b))

        return norm(a), norm(b), norm(c), alpha, beta, gamma

    def get_matrix(self):
        """return lattice matrix"""
        return self._to_matrix()

    def _to_matrix(self):
        # see http://en.wikipedia.org/wiki/Fractional_coordinates
        # For the special case of a monoclinic cell (a common case) where alpha = gamma = 90 degree and beta > 90 degree, this gives: <- special care needed
        # so far, alpha, beta, gamma < 90 degree
        a, b, c, alpha, beta, gamma = self.a, self.b, self.c, self.alpha, self.beta, self.gamma

        v = (
            a
            * b
            * c
            * np.sqrt(
                1.0
                - np.cos(alpha) ** 2
                - np.cos(beta) ** 2
                - np.cos(gamma) ** 2
                + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
            )
        )

        T = np.zeros((3, 3))
        T = np.array(
            [
                [a, b * np.cos(gamma), c * np.cos(beta)],
                [
                    0,
                    b * np.sin(gamma),
                    c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
                ],
                [0, 0, v / (a * b * np.sin(gamma))],
            ]
        )
        matrix = np.zeros((3, 3))
        matrix[:, 0] = np.dot(T, np.array((1, 0, 0)))
        matrix[:, 1] = np.dot(T, np.array((0, 1, 0)))
        matrix[:, 2] = np.dot(T, np.array((0, 0, 1)))
        # return matrix.T
        return self.matrix

    def get_rec_lattice(self):
        """
		b_i = (a_j x a_k)/ a_i . (a_j x a_k)
		"""
        lat_mat = self.matrix
        return np.linalg.inv(lat_mat).T

    def __repr__(self):
        _repr = [self.a, self.b, self.c, self.alpha, self.beta, self.gamma]
        _repr = [str(i) for i in _repr]
        return " ".join(_repr)

