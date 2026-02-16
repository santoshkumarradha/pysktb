"""
pysktb.interfaces: Bridge modules for external structure formats.

This package provides interface modules that convert atomic structures from
external libraries (such as ASE - Atomic Simulation Environment) into pysktb
Structure objects. These interfaces allow seamless integration with other
computational tools while keeping pysktb's core dependencies minimal.

Currently provides:
- ase: Convert ASE Atoms objects to pysktb Structures (optional, requires ASE)

Future interfaces may include:
- pymatgen: Convert pymatgen Structure objects
- phonopy: Convert phonopy structures
- VASP/POSCAR: Direct file format support

Each interface module follows a consistent pattern:
1. Optional dependency: imported via try-except to allow graceful degradation
2. Single conversion function: converts external format to pysktb.Structure
3. Comprehensive validation: checks input types and provides clear error messages
4. Full documentation: docstrings with parameters, returns, raises, and examples

This design ensures pysktb remains usable without any external library
dependencies, while providing convenient bridges for users who have them
installed.
"""

try:
    from .ase import ase_atoms_to_pysktb_structure
    __all__ = ["ase_atoms_to_pysktb_structure"]
except ImportError:
    # ASE module not present or ASE not installed;
    # interfaces package is still importable, but this function is unavailable
    __all__ = []
