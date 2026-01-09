"""Repulsive pair potentials for total energy calculations.

These potentials provide the short-range repulsion needed for accurate
total energies and forces in tight-binding calculations.

Example:
    >>> from pysktb.repulsive import BornMayer
    >>> rep = BornMayer(A=1500, B=3.5, cutoff=4.0)
    >>> rep(1.42)  # Energy at d=1.42 Å
    >>> rep.deriv1(1.42)  # Force (negative of derivative)
"""

from typing import Optional
import numpy as np
from numpy.typing import ArrayLike

from .scaling import ScalingLaw


class RepulsivePotential(ScalingLaw):
    """Base class for repulsive pair potentials.

    Repulsive potentials are ScalingLaws that contribute to total energy
    but not to the electronic Hamiltonian.
    """
    pass


class BornMayer(RepulsivePotential):
    """Born-Mayer repulsive potential: V(d) = A * exp(-B*d).

    Simple exponential repulsion, commonly used in ionic systems.

    Example:
        >>> rep = BornMayer(A=1500, B=3.5, cutoff=4.0)
    """

    def __init__(self, A: float, B: float,
                 cutoff: Optional[float] = None, smooth_width: float = 0.5):
        super().__init__(V0=A, d0=0.0, cutoff=cutoff, smooth_width=smooth_width)
        self.A = A
        self.B = B

    def _raw_value(self, d: ArrayLike) -> ArrayLike:
        return self.A * np.exp(-self.B * d)

    def _raw_deriv1(self, d: ArrayLike) -> ArrayLike:
        return -self.B * self.A * np.exp(-self.B * d)

    def _raw_deriv2(self, d: ArrayLike) -> ArrayLike:
        return self.B**2 * self.A * np.exp(-self.B * d)

    def __repr__(self) -> str:
        params = f"A={self.A}, B={self.B}"
        if self.cutoff is not None:
            params += f", cutoff={self.cutoff}"
        return f"BornMayer({params})"


class Buckingham(RepulsivePotential):
    """Buckingham potential: V(d) = A*exp(-B*d) - C/d⁶.

    Combines exponential repulsion with van der Waals attraction.

    Example:
        >>> rep = Buckingham(A=1500, B=3.5, C=10.0, cutoff=5.0)
    """

    def __init__(self, A: float, B: float, C: float,
                 cutoff: Optional[float] = None, smooth_width: float = 0.5):
        super().__init__(V0=A, d0=0.0, cutoff=cutoff, smooth_width=smooth_width)
        self.A = A
        self.B = B
        self.C = C

    def _raw_value(self, d: ArrayLike) -> ArrayLike:
        return self.A * np.exp(-self.B * d) - self.C / d**6

    def _raw_deriv1(self, d: ArrayLike) -> ArrayLike:
        return -self.B * self.A * np.exp(-self.B * d) + 6 * self.C / d**7

    def _raw_deriv2(self, d: ArrayLike) -> ArrayLike:
        return self.B**2 * self.A * np.exp(-self.B * d) - 42 * self.C / d**8

    def __repr__(self) -> str:
        params = f"A={self.A}, B={self.B}, C={self.C}"
        if self.cutoff is not None:
            params += f", cutoff={self.cutoff}"
        return f"Buckingham({params})"


class Morse(RepulsivePotential):
    """Morse potential: V(d) = D * (1 - exp(-a(d-d0)))² - D.

    Provides realistic description of bond stretching including anharmonicity.
    Minimum at d=d0 with depth D.

    Example:
        >>> rep = Morse(D=5.0, a=2.0, d0=1.42, cutoff=4.0)
    """

    def __init__(self, D: float, a: float, d0: float,
                 cutoff: Optional[float] = None, smooth_width: float = 0.5):
        super().__init__(V0=0.0, d0=d0, cutoff=cutoff, smooth_width=smooth_width)
        self.D = D
        self.a = a

    def _raw_value(self, d: ArrayLike) -> ArrayLike:
        exp_term = np.exp(-self.a * (d - self.d0))
        return self.D * (1 - exp_term)**2 - self.D

    def _raw_deriv1(self, d: ArrayLike) -> ArrayLike:
        exp_term = np.exp(-self.a * (d - self.d0))
        return 2 * self.D * self.a * (1 - exp_term) * exp_term

    def _raw_deriv2(self, d: ArrayLike) -> ArrayLike:
        exp_term = np.exp(-self.a * (d - self.d0))
        return 2 * self.D * self.a**2 * exp_term * (2 * exp_term - 1)

    def __repr__(self) -> str:
        params = f"D={self.D}, a={self.a}, d0={self.d0}"
        if self.cutoff is not None:
            params += f", cutoff={self.cutoff}"
        return f"Morse({params})"


class SplineRepulsive(RepulsivePotential):
    """Spline-based repulsive potential from tabulated data.

    Commonly used with DFTB parameter sets.

    Example:
        >>> distances = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        >>> energies = [10.0, 5.0, 2.0, 0.8, 0.3, 0.1]
        >>> rep = SplineRepulsive(distances, energies)
    """

    def __init__(self, distances: ArrayLike, energies: ArrayLike,
                 cutoff: Optional[float] = None, smooth_width: float = 0.5):
        from scipy.interpolate import CubicSpline

        distances = np.asarray(distances)
        energies = np.asarray(energies)

        super().__init__(V0=energies[0], d0=distances[0],
                        cutoff=cutoff, smooth_width=smooth_width)

        self.distances = distances
        self.energies = energies
        self._spline = CubicSpline(distances, energies, extrapolate=True)
        self._spline_d1 = self._spline.derivative(1)
        self._spline_d2 = self._spline.derivative(2)

    def _raw_value(self, d: ArrayLike) -> ArrayLike:
        return self._spline(d)

    def _raw_deriv1(self, d: ArrayLike) -> ArrayLike:
        return self._spline_d1(d)

    def _raw_deriv2(self, d: ArrayLike) -> ArrayLike:
        return self._spline_d2(d)

    @classmethod
    def from_file(cls, filepath: str, **kwargs) -> 'SplineRepulsive':
        """Load from file (two columns: distance, energy)."""
        data = np.loadtxt(filepath)
        return cls(data[:, 0], data[:, 1], **kwargs)

    def __repr__(self) -> str:
        return f"SplineRepulsive(n_points={len(self.distances)}, d_range=[{self.distances[0]:.2f}, {self.distances[-1]:.2f}])"


class ZeroPotential(RepulsivePotential):
    """Zero repulsive potential (placeholder).

    Use when no repulsive contribution is needed.
    """

    def __init__(self):
        super().__init__(V0=0.0, d0=1.0)

    def _raw_value(self, d: ArrayLike) -> ArrayLike:
        return np.zeros_like(d, dtype=float)

    def _raw_deriv1(self, d: ArrayLike) -> ArrayLike:
        return np.zeros_like(d, dtype=float)

    def _raw_deriv2(self, d: ArrayLike) -> ArrayLike:
        return np.zeros_like(d, dtype=float)

    def __repr__(self) -> str:
        return "ZeroPotential()"
