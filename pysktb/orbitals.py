"""Real-space atomic orbital functions for visualization.

This module provides radial functions and spherical harmonics to enable
real-space visualization of orbitals and charge densities from tight-binding
calculations.

Example:
    >>> from pysktb.orbitals import SlaterOrbital, AtomicOrbital
    >>> radial = SlaterOrbital(n=2, zeta=1.72)
    >>> orbital = AtomicOrbital("pz", radial)
    >>> values = orbital.evaluate(r_grid)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
from scipy.special import factorial, genlaguerre


# Clementi-Raimondi Slater exponents (1963)
DEFAULT_SLATER_ZETA = {
    "H":  {"1s": 1.00},
    "He": {"1s": 1.69},
    "Li": {"1s": 2.69, "2s": 0.65},
    "Be": {"1s": 3.68, "2s": 0.96},
    "B":  {"1s": 4.68, "2s": 1.21, "2p": 1.21},
    "C":  {"1s": 5.67, "2s": 1.57, "2p": 1.57},
    "N":  {"1s": 6.67, "2s": 1.92, "2p": 1.92},
    "O":  {"1s": 7.66, "2s": 2.25, "2p": 2.25},
    "F":  {"1s": 8.65, "2s": 2.56, "2p": 2.56},
    "Ne": {"1s": 9.64, "2s": 2.88, "2p": 2.88},
    "Na": {"1s": 10.63, "2s": 3.31, "2p": 3.31, "3s": 0.88},
    "Si": {"1s": 13.57, "2s": 4.90, "2p": 4.29, "3s": 1.38, "3p": 1.38},
    "Fe": {"3d": 2.14, "4s": 1.05},
    "Cu": {"3d": 2.96, "4s": 1.33},
}


class RadialFunction(ABC):
    """Base class for radial orbital functions R(r)."""

    @abstractmethod
    def __call__(self, r: np.ndarray) -> np.ndarray:
        """Evaluate radial function at distance(s) r."""
        pass

    @abstractmethod
    def get_normalization(self) -> float:
        """Return normalization constant."""
        pass


class SlaterOrbital(RadialFunction):
    """Slater-type orbital (STO).

    R(r) = N * r^(n-1) * exp(-zeta*r)

    where N is the normalization constant ensuring ∫|R|²r²dr = 1

    Args:
        n: Principal quantum number
        zeta: Slater exponent (controls orbital size)
    """

    def __init__(self, n: int, zeta: float):
        self.n = n
        self.zeta = zeta
        self._norm = self.get_normalization()

    def get_normalization(self) -> float:
        """Normalization: ∫₀^∞ |R(r)|² r² dr = 1"""
        n, zeta = self.n, self.zeta
        # ∫ r^(2n) exp(-2ζr) dr = (2n)! / (2ζ)^(2n+1)
        return np.sqrt((2 * zeta) ** (2 * n + 1) / factorial(2 * n))

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """Evaluate R(r) = N * r^(n-1) * exp(-ζr)"""
        r = np.asarray(r)
        return self._norm * (r ** (self.n - 1)) * np.exp(-self.zeta * r)


class HydrogenOrbital(RadialFunction):
    """Hydrogen-like orbital with Laguerre polynomials.

    R_nl(r) = N * (2Zr/n)^l * exp(-Zr/n) * L_{n-l-1}^{2l+1}(2Zr/n)

    Args:
        n: Principal quantum number
        l: Angular momentum quantum number
        Z: Effective nuclear charge (default 1.0)
    """

    def __init__(self, n: int, l: int, Z: float = 1.0):
        if l >= n:
            raise ValueError(f"l must be < n, got l={l}, n={n}")
        self.n = n
        self.l = l
        self.Z = Z
        self._norm = self.get_normalization()

    def get_normalization(self) -> float:
        n, l, Z = self.n, self.l, self.Z
        prefactor = (2 * Z / n) ** 3
        fac_ratio = factorial(n - l - 1) / (2 * n * factorial(n + l))
        return np.sqrt(prefactor * fac_ratio)

    def __call__(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r)
        n, l, Z = self.n, self.l, self.Z
        rho = 2 * Z * r / n
        laguerre = genlaguerre(n - l - 1, 2 * l + 1)(rho)
        return self._norm * (rho ** l) * np.exp(-rho / 2) * laguerre


class GaussianOrbital(RadialFunction):
    """Gaussian-type orbital (GTO).

    R(r) = N * r^l * exp(-alpha*r²)

    Args:
        l: Angular momentum quantum number
        alpha: Gaussian exponent
    """

    def __init__(self, l: int, alpha: float):
        self.l = l
        self.alpha = alpha
        self._norm = self.get_normalization()

    def get_normalization(self) -> float:
        l, alpha = self.l, self.alpha
        # Normalization for r^l * exp(-αr²) with r² dr measure
        from scipy.special import gamma
        return np.sqrt(2 * (2 * alpha) ** (l + 1.5) / gamma(l + 1.5))

    def __call__(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r)
        return self._norm * (r ** self.l) * np.exp(-self.alpha * r * r)


# Mapping from orbital name to (l, m) quantum numbers
ORBITAL_QUANTUM_NUMBERS = {
    "s": (0, 0),
    "px": (1, 1), "py": (1, -1), "pz": (1, 0),
    "dxy": (2, -2), "dyz": (2, -1), "dz2": (2, 0), "dxz": (2, 1), "dx2-y2": (2, 2),
    "fz3": (3, 0), "fxz2": (3, 1), "fyz2": (3, -1),
    "fxyz": (3, -2), "fz(x2-y2)": (3, 2), "fx(x2-3y2)": (3, 3), "fy(3x2-y2)": (3, -3),
}


def real_spherical_harmonic(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Compute real spherical harmonics Y_lm(θ, φ).

    Uses real (tesseral) harmonics that are real-valued combinations of
    complex spherical harmonics.

    Args:
        l: Angular momentum quantum number (0, 1, 2, ...)
        m: Magnetic quantum number (-l, ..., l)
        theta: Polar angle (0 to π)
        phi: Azimuthal angle (0 to 2π)

    Returns:
        Real spherical harmonic values
    """
    from scipy.special import sph_harm

    theta = np.asarray(theta)
    phi = np.asarray(phi)

    if m > 0:
        # Y_l^m = (-1)^m * sqrt(2) * Re[Y_l^m_complex]
        Y_complex = sph_harm(m, l, phi, theta)
        return ((-1) ** m) * np.sqrt(2) * np.real(Y_complex)
    elif m < 0:
        # Y_l^-m = (-1)^m * sqrt(2) * Im[Y_l^|m|_complex]
        Y_complex = sph_harm(-m, l, phi, theta)
        return ((-1) ** m) * np.sqrt(2) * np.imag(Y_complex)
    else:
        # m = 0: Y_l^0 is already real
        return np.real(sph_harm(0, l, phi, theta))


def cartesian_to_spherical(x: np.ndarray, y: np.ndarray, z: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Cartesian to spherical coordinates."""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / np.maximum(r, 1e-10), -1, 1))
    phi = np.arctan2(y, x)
    return r, theta, phi


class AtomicOrbital:
    """Complete atomic orbital: φ(r) = R(r) * Y_lm(θ, φ).

    Args:
        orbital_type: Orbital name ('s', 'px', 'py', 'pz', 'dxy', etc.)
        radial: RadialFunction instance for R(r)
    """

    def __init__(self, orbital_type: str, radial: RadialFunction):
        if orbital_type not in ORBITAL_QUANTUM_NUMBERS:
            raise ValueError(f"Unknown orbital type: {orbital_type}")

        self.orbital_type = orbital_type
        self.radial = radial
        self.l, self.m = ORBITAL_QUANTUM_NUMBERS[orbital_type]

    def evaluate(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Evaluate orbital at Cartesian coordinates.

        Args:
            x, y, z: Cartesian coordinates (can be arrays)

        Returns:
            Orbital values φ(r) = R(r) * Y_lm(θ, φ)
        """
        r, theta, phi = cartesian_to_spherical(x, y, z)
        R = self.radial(r)
        Y = real_spherical_harmonic(self.l, self.m, theta, phi)
        return R * Y

    def evaluate_on_grid(self, grid_points: np.ndarray) -> np.ndarray:
        """Evaluate orbital on grid of shape (N, 3).

        Args:
            grid_points: Array of shape (N, 3) with [x, y, z] coordinates

        Returns:
            Orbital values of shape (N,)
        """
        x, y, z = grid_points[:, 0], grid_points[:, 1], grid_points[:, 2]
        return self.evaluate(x, y, z)
