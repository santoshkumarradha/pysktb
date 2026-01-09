# PRD: Distance-Dependent Hopping & Forces

**Version:** 1.0
**Status:** Draft
**Author:** Santosh Kumar Radha
**Date:** January 2025

---

## Executive Summary

Extend pysktb to support distance-dependent Slater-Koster parameters, enabling:
- **Forces** and geometry optimization
- **Phonon calculations** (dynamical matrix, dispersion)
- **Electron-phonon coupling** (superconductivity, transport)
- **Molecular dynamics** simulations

This transforms pysktb from a fixed-geometry band structure code into a full electronic structure toolkit while maintaining backward compatibility and clean DX.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Requirements](#2-requirements)
3. [Design Overview](#3-design-overview)
4. [Scaling Laws API](#4-scaling-laws-api)
5. [Parameter Specification](#5-parameter-specification)
6. [New Hamiltonian Methods](#6-new-hamiltonian-methods)
7. [Phonon Module](#7-phonon-module)
8. [Electron-Phonon Coupling](#8-electron-phonon-coupling)
9. [Backward Compatibility](#9-backward-compatibility)
10. [Implementation Plan](#10-implementation-plan)
11. [Future Extensions](#11-future-extensions)

---

## 1. Motivation

### Current Limitations

pysktb currently uses fixed Slater-Koster parameters that don't depend on atomic positions:

```python
params = {"CC": {"V_ppp": -2.7}}  # Constant regardless of C-C distance
```

This limits the code to:
- Fixed geometry band structures
- No forces or geometry optimization
- No phonons or lattice dynamics
- No electron-phonon coupling

### Opportunity

By adding distance-dependent hopping:

```python
params = {"CC": {"V_ppp": Harrison(V0=-2.7, d0=1.42)}}  # V(d) = V0 * (d0/d)²
```

We unlock:
| Feature | Physics | Applications |
|---------|---------|--------------|
| Forces | F = -∂E/∂R | Geometry relaxation, defect structures |
| Phonons | ω²(q) from dynamical matrix | Thermal properties, Raman/IR spectra |
| Electron-phonon | g = ⟨ψ|∂H/∂u|ψ⟩ | Superconductivity, resistivity, CDW |
| MD | F = ma integration | Phase transitions, thermal transport |

### Design Philosophy

1. **Backward compatible** - Constant parameters still work unchanged
2. **Minimal boilerplate** - Simple cases should be simple
3. **Full control** - Advanced users can customize everything
4. **Analytical derivatives** - For stability and performance
5. **Physics-informed defaults** - Harrison, GSP, exponential built-in

---

## 2. Requirements

### Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| F1 | Support distance-dependent hopping V(d) | P0 |
| F2 | Analytical first derivatives dV/dd for forces | P0 |
| F3 | Analytical second derivatives d²V/dd² for phonons | P0 |
| F4 | Smooth cutoff functions for MD stability | P0 |
| F5 | Repulsive pair potentials | P0 |
| F6 | Built-in scaling laws (Harrison, exponential, GSP, etc.) | P0 |
| F7 | Custom user-defined functions | P1 |
| F8 | Tabulated/spline interpolation from data | P1 |
| F9 | Load parameters from DFTB .skf files | P2 |
| F10 | Load parameters from NRL-TB format | P2 |

### Non-Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| NF1 | Backward compatible with existing constant params | P0 |
| NF2 | Vectorized evaluation for performance | P0 |
| NF3 | Clear error messages for misconfiguration | P1 |
| NF4 | Comprehensive documentation with examples | P1 |
| NF5 | Optional JAX autodiff for custom functions | P2 |

---

## 3. Design Overview

### Architecture

```
pysktb/
├── scaling.py          # NEW: ScalingLaw base class and implementations
├── repulsive.py        # NEW: Repulsive potential classes
├── parameters.py       # NEW: ParameterSet container
├── forces.py           # NEW: Force calculation
├── phonons.py          # NEW: Dynamical matrix, phonon dispersion
├── electron_phonon.py  # NEW: E-ph coupling, Eliashberg function
├── hamiltonian.py      # MODIFIED: Use ScalingLaw for hopping
└── ...
```

### Class Hierarchy

```
ScalingLaw (ABC)
├── Constant          # Fixed value (backward compat)
├── Harrison          # V(d) = V0 * (d0/d)²
├── PowerLaw          # V(d) = V0 * (d0/d)^η
├── Exponential       # V(d) = V0 * exp(-α(d-d0))
├── GSP               # Goodwin-Skinner-Pettifor
├── Polynomial        # V(d) = Σ cᵢ(d-d0)^i
├── Tabulated         # Spline interpolation
└── Custom            # User function wrapper

RepulsivePotential (ScalingLaw)
├── BornMayer         # A * exp(-B*d)
├── Buckingham        # A*exp(-B*d) - C/d⁶
├── ZBL               # Ziegler-Biersack-Littmark
└── SplineRepulsive   # From DFTB files
```

---

## 4. Scaling Laws API

### Base Class Interface

```python
class ScalingLaw(ABC):
    """
    Base class for distance-dependent SK parameters.

    All implementations must provide:
    - __call__(d): V(d) value
    - deriv1(d): dV/dd first derivative
    - deriv2(d): d²V/dd² second derivative

    All methods support vectorized numpy array inputs.
    """

    def __init__(self, V0: float, d0: float,
                 cutoff: float = None,
                 smooth_width: float = 0.5):
        """
        Args:
            V0: Reference hopping value at d0
            d0: Reference distance (typically nearest-neighbor)
            cutoff: Distance beyond which V=0 (None = no cutoff)
            smooth_width: Width of smooth transition region (Å)
        """

    @abstractmethod
    def _raw_value(self, d: ArrayLike) -> ArrayLike:
        """Unscaled value (without cutoff)"""

    @abstractmethod
    def _raw_deriv1(self, d: ArrayLike) -> ArrayLike:
        """First derivative (without cutoff)"""

    @abstractmethod
    def _raw_deriv2(self, d: ArrayLike) -> ArrayLike:
        """Second derivative (without cutoff)"""

    def __call__(self, d: ArrayLike) -> ArrayLike:
        """Evaluate V(d) with smooth cutoff applied"""

    def deriv1(self, d: ArrayLike) -> ArrayLike:
        """First derivative with cutoff (for forces)"""

    def deriv2(self, d: ArrayLike) -> ArrayLike:
        """Second derivative with cutoff (for phonons)"""

    def with_cutoff(self, cutoff: float, smooth_width: float = 0.5) -> 'ScalingLaw':
        """Return copy with different cutoff settings"""

    def plot(self, dmin: float = None, dmax: float = None,
             show_derivatives: bool = True):
        """Visualize scaling law and derivatives"""
```

### Built-in Scaling Laws

#### Harrison Scaling
```python
class Harrison(ScalingLaw):
    """
    Harrison's universal scaling: V(d) = V0 * (d0/d)²

    Theoretical basis for sp-bonded materials.
    Reference: W.A. Harrison, "Electronic Structure" (1980)

    Example:
        V_ppp = Harrison(V0=-2.7, d0=1.42, cutoff=4.0)
    """
```

#### Power Law
```python
class PowerLaw(ScalingLaw):
    """
    Generalized power law: V(d) = V0 * (d0/d)^η

    - η = 2: Harrison scaling (sp bonds)
    - η = 3-5: Typical for d-orbitals

    Example:
        V_dds = PowerLaw(V0=-1.0, d0=2.5, eta=3.5, cutoff=5.0)
    """
```

#### Exponential
```python
class Exponential(ScalingLaw):
    """
    Exponential decay: V(d) = V0 * exp(-α(d - d0))

    Common in DFTB and semi-empirical methods.

    Example:
        V_sss = Exponential(V0=-5.0, d0=1.42, alpha=1.5, cutoff=4.0)
    """
```

#### GSP (Goodwin-Skinner-Pettifor)
```python
class GSP(ScalingLaw):
    """
    NRL tight-binding scaling:
    V(d) = V0 * (d0/d)^n * exp(n * [-(d/dc)^nc + (d0/dc)^nc])

    Very flexible, can fit most DFT-derived hoppings.
    Reference: Phys. Rev. B 39, 12520 (1989)

    Example:
        V_pps = GSP(V0=6.5, d0=1.42, n=2.0, nc=4.0, dc=3.5, cutoff=5.0)
    """
```

#### Tabulated
```python
class Tabulated(ScalingLaw):
    """
    Cubic spline interpolation of tabulated data.

    Useful for DFT-fitted parameters or DFTB files.
    Derivatives computed analytically from spline.

    Example:
        distances = [1.2, 1.4, 1.6, 1.8, 2.0]
        values = [-3.2, -2.7, -2.3, -1.9, -1.6]
        V_ppp = Tabulated(distances, values)

        # Or from file
        V_ppp = Tabulated.from_file("C-C_ppp.dat")
    """
```

#### Custom
```python
class Custom(ScalingLaw):
    """
    Wrapper for arbitrary user functions.

    If derivatives not provided, uses:
    1. JAX autodiff (if use_jax=True and JAX available)
    2. Finite difference fallback

    Example:
        def my_V(d):
            return -2.7 * (1.42/d)**2 * np.exp(-0.1*(d-1.42))

        # With autodiff
        V_ppp = Custom(my_V, d0=1.42, use_jax=True)

        # Or provide analytical derivatives
        V_ppp = Custom(my_V, d0=1.42, deriv1=my_dV, deriv2=my_d2V)
    """
```

### Cutoff Function

All scaling laws use Tersoff-style smooth cosine cutoff:

```
f(d) = 1                                    for d < rs
f(d) = 0.5 + 0.5*cos(π(d-rs)/(rc-rs))      for rs ≤ d ≤ rc
f(d) = 0                                    for d > rc
```

Where `rc = cutoff` and `rs = cutoff - smooth_width`.

This ensures:
- Continuous value at cutoff
- Continuous first derivative (forces smooth)
- Continuous second derivative (phonons smooth)

---

## 5. Parameter Specification

### Basic Usage (Backward Compatible)

```python
# Constant parameters still work - no changes needed
params = {
    "C": {"e_p": 0.0},
    "CC": {"V_ppp": -2.7}  # Treated as Constant(-2.7)
}
ham = Hamiltonian(structure, params)
```

### Distance-Dependent Parameters

```python
from pysktb.scaling import Harrison, Exponential, BornMayer

params = {
    "C": {
        "e_s": -8.0,      # On-site: always constant
        "e_p": 0.0,
    },
    "CC": {
        # Hopping parameters with scaling
        "V_sss": Harrison(V0=-5.0, d0=1.42, cutoff=4.0),
        "V_sps": Harrison(V0=5.5, d0=1.42, cutoff=4.0),
        "V_pps": Harrison(V0=6.5, d0=1.42, cutoff=4.0),
        "V_ppp": Harrison(V0=-2.7, d0=1.42, cutoff=4.0),

        # Repulsive potential (required for forces)
        "repulsive": BornMayer(A=1500, B=3.5, cutoff=4.0),
    }
}
```

### Convenience Function

```python
from pysktb.parameters import params_with_scaling

# Convert all hoppings to Harrison scaling in one line
base_params = {
    "C": {"e_p": 0.0},
    "CC": {"V_ppp": -2.7, "V_pps": 6.5}
}

params = params_with_scaling(
    base_params,
    scaling='harrison',           # Apply Harrison to all V_* params
    d0={"CC": 1.42},              # Reference distances
    cutoff=4.0,                   # Global cutoff
    repulsive=BornMayer(A=1500, B=3.5)  # Add repulsive
)
```

### Per-Bond-Type Scaling

```python
from pysktb.parameters import ParameterSet
from pysktb.scaling import Harrison, PowerLaw

params = ParameterSet(
    params={
        "C": {"e_p": 0.0},
        "CC": {
            "V_pps": 6.5,   # Will use sigma_scaling
            "V_ppp": -2.7,  # Will use pi_scaling
        }
    },
    sigma_scaling=Harrison(V0=1.0, d0=1.42),  # For V_*s parameters
    pi_scaling=PowerLaw(V0=1.0, d0=1.42, eta=2.5),  # For V_*p parameters
)
```

### Load from Established Parameterizations

```python
from pysktb.parameters import load_dftb_params, load_nrl_params

# DFTB parameter sets
params = load_dftb_params("3ob-3-1", elements=["C", "H", "O", "N"])
params = load_dftb_params("mio-1-1", elements=["C", "H"])

# NRL tight-binding
params = load_nrl_params("Si")
params = load_nrl_params("GaAs")
```

---

## 6. New Hamiltonian Methods

### Forces

```python
class Hamiltonian:

    def get_forces(self, nk: list = [10, 10, 10],
                   parallel: bool = True) -> np.ndarray:
        """
        Compute atomic forces F = -∂E/∂R.

        Includes:
        - Band structure contribution (Hellmann-Feynman)
        - Repulsive potential contribution

        Args:
            nk: k-point mesh for BZ integration
            parallel: Use parallel k-point evaluation

        Returns:
            forces: Array of shape (n_atoms, 3) in eV/Å
        """

    def get_stress(self, nk: list = [10, 10, 10]) -> np.ndarray:
        """
        Compute stress tensor σ = (1/V) ∂E/∂ε.

        Returns:
            stress: 3x3 stress tensor in eV/Å³
        """

    def relax(self, fmax: float = 0.01,
              steps: int = 100,
              optimizer: str = 'BFGS') -> Structure:
        """
        Geometry optimization.

        Args:
            fmax: Force convergence criterion (eV/Å)
            steps: Maximum optimization steps
            optimizer: 'BFGS', 'FIRE', or 'CG'

        Returns:
            relaxed: New Structure with optimized positions
        """
```

### Hamiltonian Derivatives

```python
class Hamiltonian:

    def get_dH_dR(self, k: np.ndarray, atom_idx: int,
                  direction: int) -> np.ndarray:
        """
        Compute ∂H(k)/∂R for electron-phonon coupling.

        Args:
            k: k-point
            atom_idx: Atom index
            direction: Cartesian direction (0=x, 1=y, 2=z)

        Returns:
            dH_dR: Matrix of shape (n_orbitals, n_orbitals)
        """
```

---

## 7. Phonon Module

### Dynamical Matrix

```python
# pysktb/phonons.py

class PhononCalculator:
    """
    Phonon calculations from tight-binding.

    Computes dynamical matrix:
    D_αβ(q) = (1/√MαMβ) Σ_R exp(iq·R) ∂²E/∂uα∂uβ(R)
    """

    def __init__(self, hamiltonian: Hamiltonian):
        self.ham = hamiltonian

    def get_dynamical_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Compute dynamical matrix D(q).

        Args:
            q: q-point in reduced coordinates

        Returns:
            D: Complex array of shape (3*n_atoms, 3*n_atoms)
        """

    def get_phonon_bands(self, qpts: np.ndarray) -> np.ndarray:
        """
        Compute phonon eigenvalues along q-path.

        Args:
            qpts: Array of q-points

        Returns:
            omega: Phonon frequencies in THz, shape (n_modes, n_qpts)
        """

    def get_phonon_dos(self, nq: list = [20, 20, 20],
                       omega_range: tuple = None,
                       n_omega: int = 200) -> tuple:
        """
        Compute phonon density of states.

        Returns:
            omega: Frequency grid
            dos: Phonon DOS
        """
```

### Usage Example

```python
from pysktb import Hamiltonian
from pysktb.phonons import PhononCalculator

ham = Hamiltonian(structure, params)
phonon = PhononCalculator(ham)

# Phonon dispersion
q_path = [[0,0,0], [0.5,0,0], [0.5,0.5,0], [0,0,0]]
qpts, q_dist, q_special = ham.get_kpts(q_path, nk=50)
omega = phonon.get_phonon_bands(qpts)

# Plot
plt.plot(q_dist, omega.T)
```

---

## 8. Electron-Phonon Coupling

### Coupling Matrix Elements

```python
# pysktb/electron_phonon.py

class ElectronPhononCoupling:
    """
    Electron-phonon coupling calculations.

    Computes:
    g_mnν(k,q) = ⟨m,k+q| ∂H/∂u_ν | n,k⟩ / √(2Mω_ν)
    """

    def __init__(self, hamiltonian: Hamiltonian):
        self.ham = hamiltonian
        self.phonon = PhononCalculator(hamiltonian)

    def get_coupling_matrix(self, k: np.ndarray,
                           q: np.ndarray) -> np.ndarray:
        """
        Compute e-ph matrix elements g(k,q).

        Args:
            k: Initial electron k-point
            q: Phonon q-point

        Returns:
            g: Array of shape (n_bands, n_bands, n_phonon_modes)
               Units: eV
        """

    def get_eliashberg_function(self,
                                nk: list = [20, 20, 20],
                                nq: list = [10, 10, 10],
                                omega_range: tuple = (0, 0.2),
                                n_omega: int = 200,
                                eta: float = 0.005) -> tuple:
        """
        Compute Eliashberg spectral function α²F(ω).

        Returns:
            omega: Frequency grid
            alpha2F: Eliashberg function
        """

    def get_lambda(self, nk: list = [20, 20, 20],
                   nq: list = [10, 10, 10]) -> float:
        """
        Compute total e-ph coupling strength λ.

        λ = 2 ∫ α²F(ω)/ω dω

        Returns:
            lambda_eph: Dimensionless coupling strength
        """

    def get_Tc_McMillan(self, omega_log: float = None,
                        mu_star: float = 0.1) -> float:
        """
        Estimate superconducting Tc using McMillan formula.

        Tc = (ω_log/1.2) * exp(-1.04(1+λ) / (λ - μ*(1+0.62λ)))

        Args:
            omega_log: Logarithmic average phonon frequency (computed if None)
            mu_star: Coulomb pseudopotential (typical: 0.1-0.15)

        Returns:
            Tc: Critical temperature in Kelvin
        """
```

### Usage Example

```python
from pysktb.electron_phonon import ElectronPhononCoupling

eph = ElectronPhononCoupling(ham)

# Compute Eliashberg function
omega, alpha2F = eph.get_eliashberg_function(
    nk=[30, 30, 1],
    nq=[15, 15, 1],
    omega_range=(0, 0.1)  # eV
)

# Coupling strength
lambda_eph = eph.get_lambda()
print(f"λ = {lambda_eph:.3f}")

# Superconducting Tc estimate
Tc = eph.get_Tc_McMillan(mu_star=0.1)
print(f"Tc ≈ {Tc:.1f} K")
```

---

## 9. Backward Compatibility

### Guaranteed Behaviors

1. **Constant parameters work unchanged**
   ```python
   # This still works exactly as before
   params = {"CC": {"V_ppp": -2.7}}
   ```

2. **Existing Hamiltonian methods unchanged**
   ```python
   ham.solve_kpath(kpts)  # Works
   ham.solve_k(k)         # Works
   ham.get_dos(...)       # Works
   ```

3. **No performance regression for constant params**
   - Constant detection at initialization
   - Skip derivative calculations if all constant

### Migration Guide

| Old Code | New Code (Optional) |
|----------|---------------------|
| `"V_ppp": -2.7` | `"V_ppp": Harrison(V0=-2.7, d0=1.42)` |
| No forces | `ham.get_forces()` |
| No phonons | `PhononCalculator(ham).get_phonon_bands(qpts)` |

---

## 10. Implementation Plan

### Phase 1: Core Scaling (Week 1-2)

- [ ] Implement `ScalingLaw` base class
- [ ] Implement `Harrison`, `PowerLaw`, `Exponential`
- [ ] Implement smooth cutoff functions
- [ ] Unit tests for all scaling laws
- [ ] Documentation for scaling module

### Phase 2: Integration (Week 3-4)

- [ ] Modify `Hamiltonian` to use `ScalingLaw`
- [ ] Maintain backward compatibility for constants
- [ ] Implement `RepulsivePotential` classes
- [ ] Implement `get_forces()` method
- [ ] Unit tests for forces

### Phase 3: Advanced Scaling (Week 5)

- [ ] Implement `GSP` scaling
- [ ] Implement `Tabulated` with spline
- [ ] Implement `Custom` with autodiff option
- [ ] Parameter file loaders (DFTB, NRL-TB)

### Phase 4: Phonons (Week 6-7)

- [ ] Implement `PhononCalculator`
- [ ] Dynamical matrix computation
- [ ] Phonon band structure
- [ ] Phonon DOS
- [ ] Unit tests and examples

### Phase 5: Electron-Phonon (Week 8-9)

- [ ] Implement `ElectronPhononCoupling`
- [ ] Coupling matrix elements
- [ ] Eliashberg function
- [ ] λ and Tc calculations
- [ ] Documentation and examples

### Phase 6: Polish (Week 10)

- [ ] Performance optimization
- [ ] Comprehensive documentation
- [ ] Tutorial notebooks
- [ ] README updates

---

## 11. Future Extensions

### Enabled by This Feature

| Feature | Dependency | Complexity |
|---------|------------|------------|
| Geometry relaxation | Forces | Low |
| Molecular dynamics | Forces | Medium |
| Thermal conductivity | Phonons | Medium |
| Raman/IR spectra | Phonons + symmetry | Medium |
| Superconductivity (BCS) | E-ph coupling | Done |
| Charge density waves | E-ph + nesting | Medium |
| Polaron formation | E-ph + self-energy | High |

### Not Covered (Future PRDs)

- Berry phase / topological invariants
- Optical properties / conductivity
- Magnetic exchange coupling
- Non-equilibrium transport

---

## Appendix A: Mathematical Details

### Force Calculation

Total energy:
```
E_tot = E_band + E_rep = Σ_n f(εn) εn + Σ_ij E_rep(d_ij)
```

Force on atom α:
```
F_α = -∂E_tot/∂R_α = -∂E_band/∂R_α - ∂E_rep/∂R_α
```

Band structure force (Hellmann-Feynman):
```
∂E_band/∂R_α = Σ_nk f(εnk) ⟨ψnk|∂H/∂R_α|ψnk⟩
```

### Dynamical Matrix

```
D_αβ(q) = (1/√MαMβ) Σ_R exp(iq·R) Φ_αβ(R)
```

Where force constants:
```
Φ_αβ(R) = ∂²E/∂u_α(0)∂u_β(R)
```

### Electron-Phonon Coupling

Coupling matrix element:
```
g_mnν(k,q) = √(ℏ/2Mω_ν) ⟨m,k+q|∂V/∂u_ν|n,k⟩
```

Eliashberg function:
```
α²F(ω) = (1/N_F) Σ_knq |g_mnν(k,q)|² δ(εk-εF) δ(εk+q-εF) δ(ω-ω_ν)
```

---

## Appendix B: Scaling Law Reference

| Scaling | Formula | Parameters | Typical Use |
|---------|---------|------------|-------------|
| Harrison | V₀(d₀/d)² | V₀, d₀ | sp semiconductors |
| PowerLaw | V₀(d₀/d)^η | V₀, d₀, η | d-orbitals (η=3-5) |
| Exponential | V₀exp(-α(d-d₀)) | V₀, d₀, α | DFTB, organics |
| GSP | V₀(d₀/d)^n exp(...) | V₀, d₀, n, nc, dc | NRL-TB, metals |
| Polynomial | Σcᵢ(d-d₀)^i | coeffs, d₀ | Local fitting |
| Tabulated | Spline(data) | distances, values | DFT-derived |

---

## Appendix C: File Formats

### DFTB .skf Format Support

```python
params = load_dftb_params("path/to/C-C.skf")
# Automatically extracts:
# - Hopping integrals (tabulated)
# - Overlap integrals (tabulated)
# - Repulsive spline
```

### NRL-TB Format Support

```python
params = load_nrl_params("path/to/Si.par")
# Automatically extracts:
# - GSP parameters for all integrals
# - On-site energies
```
