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


def _get_orbital_shell(orbital_name: str) -> str:
    """Map orbital name to shell (e.g., 'pz' -> '2p')."""
    if orbital_name == "s":
        return "1s"  # Could be 2s, 3s etc - user should override
    elif orbital_name in ["px", "py", "pz"]:
        return "2p"
    elif orbital_name in ["dxy", "dyz", "dz2", "dxz", "dx2-y2"]:
        return "3d"
    elif orbital_name.startswith("f"):
        return "4f"
    return None


def _get_principal_quantum_number(shell: str) -> int:
    """Extract n from shell name like '2p' -> 2."""
    return int(shell[0])


class OrbitalBasis:
    """Collection of atomic orbitals for a structure.

    Manages the radial functions for each orbital type in the structure,
    enabling evaluation of Bloch wavefunctions and charge densities.

    Example:
        >>> basis = OrbitalBasis.from_defaults(structure)
        >>> # Or with custom parameters
        >>> basis = OrbitalBasis(structure, {"C": {"pz": SlaterOrbital(2, 1.5)}})
    """

    def __init__(self, structure, radial_params: Optional[Dict] = None):
        """
        Args:
            structure: pysktb Structure object
            radial_params: Dict mapping element -> orbital -> RadialFunction
                          e.g., {"C": {"pz": SlaterOrbital(2, 1.72)}}
        """
        self.structure = structure
        self.radial_params = radial_params or {}
        self._orbitals = {}
        self._build_orbitals()

    def _build_orbitals(self):
        """Build AtomicOrbital objects for each orbital in structure."""
        for atom_idx, atom in enumerate(self.structure.atoms):
            element = atom.element
            for orbital_name in atom.orbitals:
                key = (atom_idx, orbital_name)

                # Get radial function
                if element in self.radial_params and orbital_name in self.radial_params[element]:
                    radial = self.radial_params[element][orbital_name]
                else:
                    radial = self._get_default_radial(element, orbital_name)

                self._orbitals[key] = AtomicOrbital(orbital_name, radial)

    def _get_default_radial(self, element: str, orbital_name: str) -> RadialFunction:
        """Get default radial function from Clementi-Raimondi table."""
        shell = _get_orbital_shell(orbital_name)
        n = _get_principal_quantum_number(shell)

        if element in DEFAULT_SLATER_ZETA:
            element_params = DEFAULT_SLATER_ZETA[element]
            # Try exact shell match
            if shell in element_params:
                zeta = element_params[shell]
                return SlaterOrbital(n, zeta)
            # Try matching by l (e.g., "2p" for any p orbital)
            for key, zeta in element_params.items():
                if key.endswith(shell[-1]):  # Match 's', 'p', 'd', 'f'
                    n_from_key = int(key[0])
                    return SlaterOrbital(n_from_key, zeta)

        # Fallback: estimate zeta from Slater's rules
        zeta = self._estimate_zeta(element, orbital_name)
        return SlaterOrbital(n, zeta)

    def _estimate_zeta(self, element: str, orbital_name: str) -> float:
        """Estimate Slater exponent using Slater's rules (rough approximation)."""
        # Very rough approximation - user should provide better values
        shell = _get_orbital_shell(orbital_name)
        n = _get_principal_quantum_number(shell)
        # Simple estimate: zeta ~ Z_eff / n, where Z_eff ~ sqrt(ionization energy)
        return 1.0 + 0.3 * n  # Placeholder

    def get_orbital(self, atom_idx: int, orbital_name: str) -> AtomicOrbital:
        """Get AtomicOrbital for given atom and orbital."""
        return self._orbitals[(atom_idx, orbital_name)]

    def get_atom_position(self, atom_idx: int) -> np.ndarray:
        """Get Cartesian position of atom in Angstroms."""
        atom = self.structure.atoms[atom_idx]
        frac_pos = np.array(atom.position)
        return self.structure.lattice.get_cartesian_coords(frac_pos)

    @classmethod
    def from_defaults(cls, structure, overrides: Optional[Dict] = None):
        """Create OrbitalBasis using default Slater exponents.

        Args:
            structure: pysktb Structure
            overrides: Optional dict to override specific orbitals
                      e.g., {"C": {"pz": SlaterOrbital(2, 1.5)}}

        Returns:
            OrbitalBasis with default radial functions
        """
        return cls(structure, overrides)
