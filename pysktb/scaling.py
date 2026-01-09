"""Distance-dependent Slater-Koster scaling laws.

This module provides scaling functions for distance-dependent hopping parameters,
enabling force calculations, phonons, and electron-phonon coupling.

Example:
    >>> from pysktb.scaling import Harrison, Exponential
    >>> V_ppp = Harrison(V0=-2.7, d0=1.42, cutoff=4.0)
    >>> V_ppp(1.5)  # Evaluate at d=1.5 Å
    >>> V_ppp.deriv1(1.5)  # First derivative for forces
"""

from abc import ABC, abstractmethod
from typing import Union, Callable, Optional
import numpy as np
from numpy.typing import ArrayLike


class ScalingLaw(ABC):
    """Base class for distance-dependent SK parameters.

    All implementations must provide:
    - __call__(d): V(d) value
    - deriv1(d): dV/dd first derivative
    - deriv2(d): d²V/dd² second derivative

    All methods support vectorized numpy array inputs.
    """

    def __init__(self, V0: float, d0: float,
                 cutoff: Optional[float] = None,
                 smooth_width: float = 0.5):
        """
        Args:
            V0: Reference hopping value at d0
            d0: Reference distance (typically nearest-neighbor)
            cutoff: Distance beyond which V=0 (None = no cutoff)
            smooth_width: Width of smooth transition region (Å)
        """
        self.V0 = V0
        self.d0 = d0
        self.cutoff = cutoff
        self.smooth_width = smooth_width

    @abstractmethod
    def _raw_value(self, d: ArrayLike) -> ArrayLike:
        """Unscaled value (without cutoff)."""
        pass

    @abstractmethod
    def _raw_deriv1(self, d: ArrayLike) -> ArrayLike:
        """First derivative (without cutoff)."""
        pass

    @abstractmethod
    def _raw_deriv2(self, d: ArrayLike) -> ArrayLike:
        """Second derivative (without cutoff)."""
        pass

    def _cutoff_func(self, d: ArrayLike) -> ArrayLike:
        """Smooth cosine cutoff function f(d)."""
        if self.cutoff is None:
            return np.ones_like(d, dtype=float)

        d = np.asarray(d)
        rc = self.cutoff
        rs = rc - self.smooth_width

        f = np.ones_like(d, dtype=float)
        mask_transition = (d >= rs) & (d <= rc)
        mask_zero = d > rc

        f[mask_transition] = 0.5 + 0.5 * np.cos(np.pi * (d[mask_transition] - rs) / (rc - rs))
        f[mask_zero] = 0.0
        return f

    def _cutoff_deriv1(self, d: ArrayLike) -> ArrayLike:
        """First derivative of cutoff function df/dd."""
        if self.cutoff is None:
            return np.zeros_like(d, dtype=float)

        d = np.asarray(d)
        rc = self.cutoff
        rs = rc - self.smooth_width

        df = np.zeros_like(d, dtype=float)
        mask = (d >= rs) & (d <= rc)
        df[mask] = -0.5 * np.pi / (rc - rs) * np.sin(np.pi * (d[mask] - rs) / (rc - rs))
        return df

    def _cutoff_deriv2(self, d: ArrayLike) -> ArrayLike:
        """Second derivative of cutoff function d²f/dd²."""
        if self.cutoff is None:
            return np.zeros_like(d, dtype=float)

        d = np.asarray(d)
        rc = self.cutoff
        rs = rc - self.smooth_width

        d2f = np.zeros_like(d, dtype=float)
        mask = (d >= rs) & (d <= rc)
        factor = np.pi / (rc - rs)
        d2f[mask] = -0.5 * factor**2 * np.cos(factor * (d[mask] - rs))
        return d2f

    def __call__(self, d: ArrayLike) -> ArrayLike:
        """Evaluate V(d) with smooth cutoff applied."""
        d = np.asarray(d)
        scalar_input = d.ndim == 0
        d = np.atleast_1d(d)

        result = self._raw_value(d) * self._cutoff_func(d)
        return float(result[0]) if scalar_input else result

    def deriv1(self, d: ArrayLike) -> ArrayLike:
        """First derivative with cutoff (for forces): d[V*f]/dd = V'f + Vf'."""
        d = np.asarray(d)
        scalar_input = d.ndim == 0
        d = np.atleast_1d(d)

        V = self._raw_value(d)
        dV = self._raw_deriv1(d)
        f = self._cutoff_func(d)
        df = self._cutoff_deriv1(d)

        result = dV * f + V * df
        return float(result[0]) if scalar_input else result

    def deriv2(self, d: ArrayLike) -> ArrayLike:
        """Second derivative with cutoff (for phonons): V''f + 2V'f' + Vf''."""
        d = np.asarray(d)
        scalar_input = d.ndim == 0
        d = np.atleast_1d(d)

        V = self._raw_value(d)
        dV = self._raw_deriv1(d)
        d2V = self._raw_deriv2(d)
        f = self._cutoff_func(d)
        df = self._cutoff_deriv1(d)
        d2f = self._cutoff_deriv2(d)

        result = d2V * f + 2 * dV * df + V * d2f
        return float(result[0]) if scalar_input else result

    def with_cutoff(self, cutoff: float, smooth_width: float = 0.5) -> 'ScalingLaw':
        """Return copy with different cutoff settings."""
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new.cutoff = cutoff
        new.smooth_width = smooth_width
        return new

    def __repr__(self) -> str:
        params = f"V0={self.V0}, d0={self.d0}"
        if self.cutoff is not None:
            params += f", cutoff={self.cutoff}"
        return f"{self.__class__.__name__}({params})"


class Constant(ScalingLaw):
    """Constant value (no distance dependence).

    Used for backward compatibility with fixed parameters.
    """

    def __init__(self, value: float):
        super().__init__(V0=value, d0=1.0)
        self.value = value

    def _raw_value(self, d: ArrayLike) -> ArrayLike:
        return np.full_like(d, self.value, dtype=float)

    def _raw_deriv1(self, d: ArrayLike) -> ArrayLike:
        return np.zeros_like(d, dtype=float)

    def _raw_deriv2(self, d: ArrayLike) -> ArrayLike:
        return np.zeros_like(d, dtype=float)

    def __repr__(self) -> str:
        return f"Constant({self.value})"


class Harrison(ScalingLaw):
    """Harrison's universal scaling: V(d) = V0 * (d0/d)².

    Theoretical basis for sp-bonded materials.
    Reference: W.A. Harrison, "Electronic Structure" (1980)

    Example:
        >>> V_ppp = Harrison(V0=-2.7, d0=1.42, cutoff=4.0)
    """

    def _raw_value(self, d: ArrayLike) -> ArrayLike:
        return self.V0 * (self.d0 / d)**2

    def _raw_deriv1(self, d: ArrayLike) -> ArrayLike:
        return -2 * self.V0 * self.d0**2 / d**3

    def _raw_deriv2(self, d: ArrayLike) -> ArrayLike:
        return 6 * self.V0 * self.d0**2 / d**4


class PowerLaw(ScalingLaw):
    """Generalized power law: V(d) = V0 * (d0/d)^η.

    - η = 2: Harrison scaling (sp bonds)
    - η = 3-5: Typical for d-orbitals

    Example:
        >>> V_dds = PowerLaw(V0=-1.0, d0=2.5, eta=3.5, cutoff=5.0)
    """

    def __init__(self, V0: float, d0: float, eta: float = 2.0,
                 cutoff: Optional[float] = None, smooth_width: float = 0.5):
        super().__init__(V0, d0, cutoff, smooth_width)
        self.eta = eta

    def _raw_value(self, d: ArrayLike) -> ArrayLike:
        return self.V0 * (self.d0 / d)**self.eta

    def _raw_deriv1(self, d: ArrayLike) -> ArrayLike:
        return -self.eta * self.V0 * self.d0**self.eta / d**(self.eta + 1)

    def _raw_deriv2(self, d: ArrayLike) -> ArrayLike:
        return self.eta * (self.eta + 1) * self.V0 * self.d0**self.eta / d**(self.eta + 2)

    def __repr__(self) -> str:
        params = f"V0={self.V0}, d0={self.d0}, eta={self.eta}"
        if self.cutoff is not None:
            params += f", cutoff={self.cutoff}"
        return f"PowerLaw({params})"


class Exponential(ScalingLaw):
    """Exponential decay: V(d) = V0 * exp(-α(d - d0)).

    Common in DFTB and semi-empirical methods.

    Example:
        >>> V_sss = Exponential(V0=-5.0, d0=1.42, alpha=1.5, cutoff=4.0)
    """

    def __init__(self, V0: float, d0: float, alpha: float = 1.0,
                 cutoff: Optional[float] = None, smooth_width: float = 0.5):
        super().__init__(V0, d0, cutoff, smooth_width)
        self.alpha = alpha

    def _raw_value(self, d: ArrayLike) -> ArrayLike:
        return self.V0 * np.exp(-self.alpha * (d - self.d0))

    def _raw_deriv1(self, d: ArrayLike) -> ArrayLike:
        return -self.alpha * self.V0 * np.exp(-self.alpha * (d - self.d0))

    def _raw_deriv2(self, d: ArrayLike) -> ArrayLike:
        return self.alpha**2 * self.V0 * np.exp(-self.alpha * (d - self.d0))

    def __repr__(self) -> str:
        params = f"V0={self.V0}, d0={self.d0}, alpha={self.alpha}"
        if self.cutoff is not None:
            params += f", cutoff={self.cutoff}"
        return f"Exponential({params})"


class GSP(ScalingLaw):
    """Goodwin-Skinner-Pettifor (NRL tight-binding) scaling.

    V(d) = V0 * (d0/d)^n * exp(n * [-(d/dc)^nc + (d0/dc)^nc])

    Very flexible, can fit most DFT-derived hoppings.
    Reference: Phys. Rev. B 39, 12520 (1989)

    Example:
        >>> V_pps = GSP(V0=6.5, d0=1.42, n=2.0, nc=4.0, dc=3.5, cutoff=5.0)
    """

    def __init__(self, V0: float, d0: float, n: float = 2.0,
                 nc: float = 4.0, dc: float = 3.5,
                 cutoff: Optional[float] = None, smooth_width: float = 0.5):
        super().__init__(V0, d0, cutoff, smooth_width)
        self.n = n
        self.nc = nc
        self.dc = dc
        # Precompute constant term
        self._exp_const = n * (d0 / dc)**nc

    def _raw_value(self, d: ArrayLike) -> ArrayLike:
        ratio = self.d0 / d
        exp_term = self.n * (-(d / self.dc)**self.nc + (self.d0 / self.dc)**self.nc)
        return self.V0 * ratio**self.n * np.exp(exp_term)

    def _raw_deriv1(self, d: ArrayLike) -> ArrayLike:
        V = self._raw_value(d)
        # dV/dd = V * d(ln V)/dd
        # ln V = n*ln(d0/d) + n*[-(d/dc)^nc + const]
        # d(ln V)/dd = -n/d - n*nc/dc * (d/dc)^(nc-1)
        dlnV = -self.n / d - self.n * self.nc / self.dc * (d / self.dc)**(self.nc - 1)
        return V * dlnV

    def _raw_deriv2(self, d: ArrayLike) -> ArrayLike:
        V = self._raw_value(d)
        dlnV = -self.n / d - self.n * self.nc / self.dc * (d / self.dc)**(self.nc - 1)
        d2lnV = self.n / d**2 - self.n * self.nc * (self.nc - 1) / self.dc**2 * (d / self.dc)**(self.nc - 2)
        return V * (dlnV**2 + d2lnV)

    def __repr__(self) -> str:
        params = f"V0={self.V0}, d0={self.d0}, n={self.n}, nc={self.nc}, dc={self.dc}"
        if self.cutoff is not None:
            params += f", cutoff={self.cutoff}"
        return f"GSP({params})"


class Polynomial(ScalingLaw):
    """Polynomial scaling: V(d) = Σ cᵢ(d-d0)^i.

    Useful for local fitting around equilibrium distance.

    Example:
        >>> V_ppp = Polynomial(coeffs=[-2.7, 1.5, -0.3], d0=1.42)
    """

    def __init__(self, coeffs: list, d0: float,
                 cutoff: Optional[float] = None, smooth_width: float = 0.5):
        super().__init__(V0=coeffs[0] if coeffs else 0.0, d0=d0,
                        cutoff=cutoff, smooth_width=smooth_width)
        self.coeffs = np.asarray(coeffs, dtype=float)

    def _raw_value(self, d: ArrayLike) -> ArrayLike:
        x = d - self.d0
        return np.polyval(self.coeffs[::-1], x)

    def _raw_deriv1(self, d: ArrayLike) -> ArrayLike:
        if len(self.coeffs) < 2:
            return np.zeros_like(d, dtype=float)
        deriv_coeffs = self.coeffs[1:] * np.arange(1, len(self.coeffs))
        x = d - self.d0
        return np.polyval(deriv_coeffs[::-1], x)

    def _raw_deriv2(self, d: ArrayLike) -> ArrayLike:
        if len(self.coeffs) < 3:
            return np.zeros_like(d, dtype=float)
        deriv1_coeffs = self.coeffs[1:] * np.arange(1, len(self.coeffs))
        deriv2_coeffs = deriv1_coeffs[1:] * np.arange(1, len(deriv1_coeffs))
        x = d - self.d0
        return np.polyval(deriv2_coeffs[::-1], x)

    def __repr__(self) -> str:
        return f"Polynomial(coeffs={list(self.coeffs)}, d0={self.d0})"


class Tabulated(ScalingLaw):
    """Cubic spline interpolation of tabulated data.

    Useful for DFT-fitted parameters or DFTB files.
    Derivatives computed analytically from spline.

    Example:
        >>> distances = [1.2, 1.4, 1.6, 1.8, 2.0]
        >>> values = [-3.2, -2.7, -2.3, -1.9, -1.6]
        >>> V_ppp = Tabulated(distances, values)
    """

    def __init__(self, distances: ArrayLike, values: ArrayLike,
                 cutoff: Optional[float] = None, smooth_width: float = 0.5):
        from scipy.interpolate import CubicSpline

        distances = np.asarray(distances)
        values = np.asarray(values)

        # Find d0 as the first distance
        d0 = distances[0]
        V0 = values[0]
        super().__init__(V0, d0, cutoff, smooth_width)

        self.distances = distances
        self.values = values
        self._spline = CubicSpline(distances, values, extrapolate=True)
        self._spline_d1 = self._spline.derivative(1)
        self._spline_d2 = self._spline.derivative(2)

    def _raw_value(self, d: ArrayLike) -> ArrayLike:
        return self._spline(d)

    def _raw_deriv1(self, d: ArrayLike) -> ArrayLike:
        return self._spline_d1(d)

    def _raw_deriv2(self, d: ArrayLike) -> ArrayLike:
        return self._spline_d2(d)

    @classmethod
    def from_file(cls, filepath: str, **kwargs) -> 'Tabulated':
        """Load tabulated data from file (two columns: distance, value)."""
        data = np.loadtxt(filepath)
        return cls(data[:, 0], data[:, 1], **kwargs)

    def __repr__(self) -> str:
        return f"Tabulated(n_points={len(self.distances)}, d_range=[{self.distances[0]:.2f}, {self.distances[-1]:.2f}])"


class Custom(ScalingLaw):
    """Wrapper for arbitrary user functions.

    If derivatives not provided, uses finite difference fallback.

    Example:
        >>> def my_V(d):
        ...     return -2.7 * (1.42/d)**2 * np.exp(-0.1*(d-1.42))
        >>> V_ppp = Custom(my_V, d0=1.42)

        >>> # Or provide analytical derivatives
        >>> V_ppp = Custom(my_V, d0=1.42, deriv1=my_dV, deriv2=my_d2V)
    """

    def __init__(self, func: Callable, d0: float,
                 deriv1: Optional[Callable] = None,
                 deriv2: Optional[Callable] = None,
                 cutoff: Optional[float] = None, smooth_width: float = 0.5):
        V0 = func(d0)
        super().__init__(V0, d0, cutoff, smooth_width)

        self._func = func
        self._deriv1_func = deriv1
        self._deriv2_func = deriv2
        self._h = 1e-5  # Finite difference step

    def _raw_value(self, d: ArrayLike) -> ArrayLike:
        return np.vectorize(self._func)(d)

    def _raw_deriv1(self, d: ArrayLike) -> ArrayLike:
        if self._deriv1_func is not None:
            return np.vectorize(self._deriv1_func)(d)
        # Central finite difference
        d = np.asarray(d)
        h = self._h
        return (self._raw_value(d + h) - self._raw_value(d - h)) / (2 * h)

    def _raw_deriv2(self, d: ArrayLike) -> ArrayLike:
        if self._deriv2_func is not None:
            return np.vectorize(self._deriv2_func)(d)
        # Central finite difference
        d = np.asarray(d)
        h = self._h
        return (self._raw_value(d + h) - 2 * self._raw_value(d) + self._raw_value(d - h)) / h**2

    def __repr__(self) -> str:
        return f"Custom(d0={self.d0})"


def ensure_scaling(value: Union[float, ScalingLaw]) -> ScalingLaw:
    """Convert a value to a ScalingLaw if needed.

    Enables backward compatibility: raw floats become Constant instances.
    """
    if isinstance(value, ScalingLaw):
        return value
    return Constant(float(value))
