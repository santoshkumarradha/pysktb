"""
ASE interface module for converting ASE Atoms objects to pysktb Structures.

This module provides conversion functionality from the Atomic Simulation Environment
(ASE) to pysktb's Structure representation. It handles validation, lattice normalization,
and orbital mapping.

The conversion process consists of four stages:
1. Input validation (type, non-empty, cell, volume)
2. Lattice computation (normalization and constant extraction)
3. Atomic position and orbital validation
4. Structure object creation with parameters

ASE should be installed separately via: pip install ase
"""

import warnings
import numpy as np


def ase_atoms_to_pysktb_structure(
    ase_atoms, orbital_dict=None, bond_cutoff_dict=None, periodicity=None, numba=True, name=None
):
    """
    Convert an ASE Atoms object to a pysktb Structure object.

    This function performs a comprehensive conversion from ASE's Atoms representation
    to pysktb's Structure class, with full validation and support for customizable
    orbital definitions and bonding parameters.

    Parameters
    ----------
    ase_atoms : ase.Atoms
        The ASE Atoms object to convert. Must be a valid, non-empty structure with
        a defined unit cell. ASE uses Ångströms as the length unit.
    orbital_dict : dict, optional
        Mapping of element symbols to orbital lists. Each value should be a list of
        valid orbital names (subset of Atom.ORBITALS_ALL). If None, a warning is
        issued and an empty dictionary is used (atoms will have no specified orbitals).
        Example: {'Si': ['s', 'p'], 'O': ['s', 'px', 'py', 'pz']}
    bond_cutoff_dict : dict, optional
        Mapping of element pair combinations to bonding cutoff distances. Keys are
        element pair strings like 'SiO' or 'OC' (order-independent). Values are
        dictionaries with 'NN' key for nearest-neighbor cutoff. If None, no bond
        information is attached to the Structure.
    periodicity : list of bool, optional
        3-element boolean list specifying periodic boundary conditions along x, y, z.
        Defaults to the periodicity from ase_atoms.pbc if not provided.
    numba : bool, optional
        Whether to enable Numba JIT compilation in the resulting Structure.
        Defaults to True.
    name : str, optional
        Name for the resulting Structure. Defaults to 'system' if not provided.

    Returns
    -------
    pysktb.Structure
        A fully initialized Structure object with:
        - Lattice computed from normalized first lattice vector magnitude
        - Fractional coordinates wrapped to [0, 1) range
        - Orbital information attached to atoms (if provided)
        - Bonding information (if bond_cutoff_dict provided)
        - Periodicity information
        - Numba flag for optimization

    Raises
    ------
    ImportError
        If ASE is not installed. Suggest installing with: pip install ase
    TypeError
        If ase_atoms is not an ase.Atoms object.
    ValueError
        If ase_atoms is empty (len < 1), has no cell defined, or has a degenerate
        (zero volume) cell.

    Notes
    -----
    - ASE uses Ångströms for lengths; pysktb's lattice also uses Ångströms
    - Atomic positions are converted from ASE Cartesian to pysktb fractional coordinates
    - Fractional coordinates are wrapped to [0, 1) using modulo arithmetic
    - The lattice is normalized: the magnitude of the first vector becomes the
      lattice constant, and all vectors are scaled accordingly
    - Invalid orbital names trigger a ValueError with the offending name included
    - If orbital_dict is None, a warning is issued but conversion continues

    Examples
    --------
    >>> import ase.build
    >>> import pysktb.interfaces as interfaces
    >>> # Create an ASE structure
    >>> atoms = ase.build.bulk('Si', 'diamond', a=5.431)
    >>> # Define orbital basis for Si
    >>> orbitals = {'Si': ['s', 'px', 'py', 'pz', 'd']}  # or individual d orbitals
    >>> # Define bonding cutoffs
    >>> bonds = {'SiSi': {'NN': 2.5}}
    >>> # Convert to pysktb Structure
    >>> structure = interfaces.ase_atoms_to_pysktb_structure(
    ...     atoms,
    ...     orbital_dict=orbitals,
    ...     bond_cutoff_dict=bonds,
    ...     name='diamond_Si'
    ... )
    >>> print(structure.name)
    diamond_Si
    """
    # Import ASE inside function to maintain optional dependency pattern
    try:
        import ase
    except ImportError:
        raise ImportError(
            "ASE is required for this function. "
            "Please install it with: pip install ase"
        )

    # ========== STAGE 1: INPUT VALIDATION ==========
    # Validate input type
    if not isinstance(ase_atoms, ase.Atoms):
        raise TypeError("Input must be an ase.Atoms object")

    # Validate non-empty structure
    if len(ase_atoms) < 1:
        raise ValueError("Cannot convert empty structure (len < 1)")

    # Validate that cell is defined and has proper rank
    cell = ase_atoms.get_cell()

    # Check if cell has all three lattice vectors (rank 3)
    try:
        rank = np.linalg.matrix_rank(cell)
        if rank < 3:
            # Less than 3 linearly independent vectors = degenerate cell
            raise ValueError("Structure has degenerate or zero volume cell")
    except np.linalg.LinAlgError:
        # If matrix_rank fails due to linalg error (rare), treat as degenerate
        raise ValueError("Structure has degenerate or zero volume cell")

    # Validate non-zero volume
    try:
        cell_volume = ase_atoms.get_volume()
    except ValueError:
        # ASE raises ValueError if cell is degenerate or undefined
        raise ValueError("Structure has degenerate or zero volume cell")

    if np.isclose(cell_volume, 0.0):
        raise ValueError("Structure has degenerate or zero volume cell")

    # ========== STAGE 2: LATTICE COMPUTATION ==========
    # Extract cell vectors from ASE (in Ångströms)
    cell_vectors = ase_atoms.get_cell()[:]  # 3x3 matrix, each row is a lattice vector

    # Normalize lattice: compute magnitude of first vector as lattice constant
    first_vector = cell_vectors[0]
    lattice_constant = np.linalg.norm(first_vector)

    # Create normalized matrix by dividing all vectors by the constant
    normalized_matrix = cell_vectors / lattice_constant

    # Import required classes
    from ..lattice import Lattice
    from ..atom import Atom

    # Create Lattice object with normalized matrix and computed constant
    lattice = Lattice(normalized_matrix, lattice_constant)

    # ========== STAGE 3: ATOMIC POSITIONS & ORBITAL VALIDATION ==========
    # Get fractional (scaled) coordinates from ASE
    fractional_coords = ase_atoms.get_scaled_positions()

    # Wrap coordinates to [0, 1) range using modulo
    fractional_coords = np.mod(fractional_coords, 1.0)

    # Handle orbital_dict: warn if None and use empty dict
    if orbital_dict is None:
        warnings.warn(
            "orbital_dict is None; atoms will be created without specified orbitals",
            UserWarning
        )
        orbital_dict = {}

    # Validate orbital names and create atoms list
    atoms_list = []
    for i, symbol in enumerate(ase_atoms.get_chemical_symbols()):
        pos = fractional_coords[i]

        # Get orbitals for this element if specified
        element_orbitals = None
        if symbol in orbital_dict:
            element_orbitals = orbital_dict[symbol]

            # Validate each orbital name
            for orbital in element_orbitals:
                if orbital not in Atom.ORBITALS_ALL:
                    raise ValueError(
                        f"Invalid orbital name '{orbital}' for element {symbol}. "
                        f"Allowed orbitals: {Atom.ORBITALS_ALL}"
                    )

        # Create Atom with element, position, and orbitals
        atom = Atom(symbol, pos, orbitals=element_orbitals)
        atoms_list.append(atom)

    # ========== STAGE 4: STRUCTURE CREATION ==========
    # Determine periodicity
    if periodicity is None:
        periodicity = ase_atoms.pbc.tolist()  # Convert numpy array to list
    else:
        # Validate periodicity if provided
        if not isinstance(periodicity, (list, tuple)) or len(periodicity) != 3:
            raise ValueError("periodicity must be a 3-element boolean list")
        if not all(isinstance(x, (bool, np.bool_)) for x in periodicity):
            raise ValueError("periodicity elements must be boolean")

    # Determine name
    if name is None:
        name = "system"

    # Import Structure class
    from ..structure import Structure

    # Create and return Structure object with all parameters
    structure = Structure(
        lattice=lattice,
        atoms=atoms_list,
        periodicity=periodicity,
        name=name,
        bond_cut=bond_cutoff_dict,
        numba=numba
    )

    return structure


__all__ = ["ase_atoms_to_pysktb_structure"]
