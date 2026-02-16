"""
ASE test fixtures and helper functions for pysktb interface testing.

This module provides reusable ASE Atoms structures for testing the ASE-to-pysktb
conversion functionality. Each fixture helper function constructs a specific
crystal structure with well-defined properties.

The fixtures include:
- Graphene: 2-atom hexagonal cell
- Silicon diamond: 8-atom cubic cell
- GaAs: 2-atom zinc-blende structure
- Monoclinic: Non-orthogonal cell with non-90° angles
- Simple cubic: Generic cubic structure for default parameter testing

All helpers are importable at module scope for direct reuse in test functions.

This module also contains comprehensive tests for the ase_atoms_to_pysktb_structure()
function, focusing on error cases and invalid inputs. Tests verify that appropriate
exceptions are raised with correct error messages for:
- Invalid orbital names (AC #6)
- Non-ASE input types (AC #7)
- Empty ASE Atoms objects (AC #8)
- Missing or degenerate cells (AC #9)
"""

import pytest
import numpy as np

try:
    import ase
    from ase.build import bulk
    from ase.atoms import Atoms
    HAS_ASE = True
except ImportError:
    HAS_ASE = False

from pysktb.interfaces.ase import ase_atoms_to_pysktb_structure


# ============================================================================
# Fixture Helper Functions
# ============================================================================

def create_graphene_fixture():
    """
    Create a graphene ASE Atoms structure.

    Graphene is a single layer of hexagonal carbon atoms. This fixture
    creates a 2-atom unit cell with a hexagonal cell (a ≠ b, c perpendicular,
    angles = 120°/60°/90°).

    Expected properties:
    - Number of atoms: 2
    - Element: C (carbon)
    - Cell: Hexagonal (60° angle between a and b)
    - Periodicity: 3D periodic

    Returns
    -------
    ase.Atoms
        A graphene structure with 2 carbon atoms in hexagonal cell.
    """
    try:
        import ase
        from ase.build import bulk
    except ImportError:
        raise ImportError("ASE is required for fixtures. Install with: pip install ase")

    # Create graphene using ASE's bulk builder
    # Graphene has a hexagonal cell with a ~2.46 Å
    atoms = bulk('C', 'hcp', a=2.46, c=4.0, cubic=False)

    # The bulk 'hcp' structure gives us hexagonal packing
    # We need only the 2-atom basis for graphene
    atoms = atoms[[0, 1]]

    # Set appropriate periodicity for 2D material (optional, but graphene is typically 2D)
    atoms.pbc = [True, True, True]

    return atoms


def create_si_diamond_fixture():
    """
    Create a silicon diamond ASE Atoms structure.

    Silicon crystallizes in the diamond cubic structure with 8 atoms per
    conventional cubic unit cell. The lattice constant is approximately 5.43 Å.

    Expected properties:
    - Number of atoms: 8
    - Element: Si (silicon)
    - Cell: Cubic, a ≈ 5.43 Å
    - Cell angles: 90°/90°/90°
    - Periodicity: 3D periodic

    Returns
    -------
    ase.Atoms
        A diamond cubic silicon structure with 8 atoms.
    """
    try:
        import ase
        from ase.build import bulk
    except ImportError:
        raise ImportError("ASE is required for fixtures. Install with: pip install ase")

    # Create silicon diamond structure
    # ASE's bulk builder creates the full 8-atom cubic cell for 'diamond'
    atoms = bulk('Si', 'diamond', a=5.43, cubic=True)

    # Verify 8 atoms as expected
    assert len(atoms) == 8, f"Expected 8 Si atoms in diamond cell, got {len(atoms)}"

    # Set periodicity for bulk material
    atoms.pbc = [True, True, True]

    return atoms


def create_gaas_fixture():
    """
    Create a GaAs (gallium arsenide) ASE Atoms structure.

    GaAs crystallizes in the zinc-blende (sphalerite) structure, which is
    similar to diamond but with two different atoms. The primitive cell
    contains 2 atoms (1 Ga + 1 As).

    Expected properties:
    - Number of atoms: 2
    - Elements: Ga and As
    - Cell: Cubic zinc-blende structure
    - Periodicity: 3D periodic

    Returns
    -------
    ase.Atoms
        A zinc-blende GaAs structure with 2 atoms.
    """
    try:
        import ase
        from ase.atoms import Atoms
    except ImportError:
        raise ImportError("ASE is required for fixtures. Install with: pip install ase")

    # Create GaAs zinc-blende structure with 2-atom basis
    # Zinc-blende is FCC lattice with basis [0,0,0] and [1/4,1/4,1/4]
    a = 5.65  # Lattice constant in Angstrom

    # Cubic cell
    cell = np.array([
        [a, 0, 0],
        [0, a, 0],
        [0, 0, a]
    ])

    # Zinc-blende basis: one atom at origin (Ga), one at (1/4, 1/4, 1/4) (As)
    positions = np.array([
        [0.0, 0.0, 0.0],      # Ga at (0,0,0)
        [0.25, 0.25, 0.25]    # As at (1/4,1/4,1/4)
    ])

    # Create ASE Atoms object with GaAs composition
    atoms = Atoms('GaAs', positions=positions, cell=cell, pbc=True)

    # Verify 2 atoms (1 Ga + 1 As) as expected
    assert len(atoms) == 2, f"Expected 2 atoms in GaAs zinc-blende cell, got {len(atoms)}"

    return atoms


def create_monoclinic_fixture():
    """
    Create a monoclinic crystal ASE Atoms structure.

    A monoclinic cell has the following properties:
    - a ≠ b ≠ c (all different)
    - α = γ = 90° (right angles for a-c and b-c)
    - β ≠ 90° (obtuse angle for a-b, typically 90° < β < 180°)

    This fixture uses a simple monoclinic cell with atoms positioned in a
    basic arrangement. The lattice parameters are chosen for stability and
    represent a realistic monoclinic structure.

    Expected properties:
    - Cell angles: α ≈ 90°, β ≈ 110°, γ ≈ 90°
    - Number of atoms: Minimal structure (typically 2-4)
    - Non-orthogonal cell: Non-90° angle between two vectors

    Returns
    -------
    ase.Atoms
        A monoclinic structure with non-90° cell angles.
    """
    try:
        import ase
        from ase.atoms import Atoms
    except ImportError:
        raise ImportError("ASE is required for fixtures. Install with: pip install ase")

    # Create monoclinic cell with non-90° angle
    # Lattice parameters for a simple monoclinic structure
    # a, b, c (in Å) and beta angle (in degrees)
    a = 5.0
    b = 6.0
    c = 4.0
    beta = 110.0  # Non-90° angle

    # Convert beta to radians
    beta_rad = np.deg2rad(beta)

    # Construct lattice vectors for monoclinic cell
    # Standard monoclinic cell: a along x, c along z, b in x-z plane
    cell = np.array([
        [a, 0, 0],
        [0, b, 0],
        [c * np.cos(beta_rad), 0, c * np.sin(beta_rad)]
    ])

    # Create atoms at simple positions
    # Two atoms at different fractional coordinates
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5]
    ])

    # Create ASE Atoms object
    atoms = Atoms('Si2', positions=positions, cell=cell, pbc=True)

    return atoms


def create_simple_cubic_fixture():
    """
    Create a simple cubic crystal structure.

    Simple cubic is the most basic cubic lattice with one atom at the
    conventional cell corners. This fixture provides a minimal, highly
    symmetric structure useful for testing default parameter handling.

    Expected properties:
    - Number of atoms: 1 (per conventional cell)
    - Cell: Cubic with a = 3.0 Å
    - Cell angles: 90°/90°/90°
    - Periodicity: 3D periodic

    Returns
    -------
    ase.Atoms
        A simple cubic structure with 1 atom.
    """
    try:
        import ase
        from ase.atoms import Atoms
    except ImportError:
        raise ImportError("ASE is required for fixtures. Install with: pip install ase")

    # Create simple cubic cell with lattice constant 3.0 Å
    a = 3.0

    # Cubic cell
    cell = np.array([
        [a, 0, 0],
        [0, a, 0],
        [0, 0, a]
    ])

    # Single atom at origin for simple cubic
    positions = np.array([[0.0, 0.0, 0.0]])

    # Create ASE Atoms object
    atoms = Atoms('Fe', positions=positions, cell=cell, pbc=True)

    return atoms


__all__ = [
    "create_graphene_fixture",
    "create_si_diamond_fixture",
    "create_gaas_fixture",
    "create_monoclinic_fixture",
    "create_simple_cubic_fixture",
]


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.skipif(not HAS_ASE, reason="ASE not installed")
class TestErrorHandling:
    """Test error handling and input validation in ASE conversion."""

    def test_invalid_orbital_raises(self):
        """
        Test that invalid orbital names raise ValueError with identifying message.

        Verifies acceptance criterion AC #6: Invalid orbital names should raise
        ValueError mentioning the offending orbital name.
        """
        # Create a valid ASE structure
        atoms = bulk('Si', 'diamond', a=5.43)

        # Use an invalid orbital name
        orbital_dict = {'Si': ['invalid_orb']}

        # Verify ValueError is raised
        with pytest.raises(ValueError) as excinfo:
            ase_atoms_to_pysktb_structure(atoms, orbital_dict=orbital_dict)

        # Verify error message contains the invalid orbital name
        assert 'invalid_orb' in str(excinfo.value), \
            "Error message should mention the invalid orbital name 'invalid_orb'"

    def test_non_ase_input_raises(self):
        """
        Test that non-ASE input raises TypeError with identifying message.

        Verifies acceptance criterion AC #7: Non-ase.Atoms input should raise
        TypeError mentioning 'ase.Atoms'.
        """
        # Try with a string input
        with pytest.raises(TypeError) as excinfo:
            ase_atoms_to_pysktb_structure("not an ase.Atoms object")

        # Verify error message contains reference to ase.Atoms
        assert 'ase.Atoms' in str(excinfo.value), \
            "Error message should mention 'ase.Atoms'"

    def test_empty_atoms_raises(self):
        """
        Test that empty ASE Atoms object raises ValueError with identifying message.

        Verifies acceptance criterion AC #8: Empty structure (len < 1) should raise
        ValueError mentioning 'empty'.
        """
        # Create an empty atoms object
        atoms = ase.Atoms()

        # Verify ValueError is raised
        with pytest.raises(ValueError) as excinfo:
            ase_atoms_to_pysktb_structure(atoms)

        # Verify error message contains the word 'empty'
        error_msg = str(excinfo.value).lower()
        assert 'empty' in error_msg, \
            "Error message should mention 'empty' to identify the issue"

    def test_no_cell_raises(self):
        """
        Test that missing cell raises ValueError with identifying message.

        Verifies acceptance criterion AC #9: Missing cell information should raise
        ValueError mentioning 'cell' or 'missing'.
        """
        # Create atoms with no cell (set_cell(None))
        atoms = ase.Atoms('Si', positions=[[0, 0, 0]])
        atoms.set_cell(None)

        # Verify ValueError is raised
        with pytest.raises(ValueError) as excinfo:
            ase_atoms_to_pysktb_structure(atoms)

        # Verify error message mentions cell or degenerate
        error_msg = str(excinfo.value).lower()
        assert 'cell' in error_msg or 'degenerate' in error_msg, \
            "Error message should mention 'cell' or 'degenerate' to identify missing cell"

    def test_degenerate_cell_raises(self):
        """
        Test that zero-volume (degenerate) cell raises ValueError with identifying message.

        Verifies acceptance criterion AC #9 (extended): Degenerate cells with rank < 3
        should raise ValueError mentioning 'degenerate' or 'zero volume'.
        """
        # Create atoms with degenerate cell (rank < 3, all zeros in one dimension)
        atoms = ase.Atoms('Si', positions=[[0, 0, 0]],
                         cell=[[1, 0, 0], [0, 1, 0], [0, 0, 0]], pbc=True)

        # Verify ValueError is raised
        with pytest.raises(ValueError) as excinfo:
            ase_atoms_to_pysktb_structure(atoms)

        # Verify error message mentions degenerate or zero volume
        error_msg = str(excinfo.value).lower()
        assert 'degenerate' in error_msg or 'zero volume' in error_msg, \
            "Error message should mention 'degenerate' or 'zero volume' to identify the issue"


# ============================================================================
# Periodicity and PBC Tests
# ============================================================================

@pytest.mark.skipif(not HAS_ASE, reason="ASE not installed")
def test_periodicity_conversion():
    """
    Test periodicity conversion from ASE pbc to pysktb Structure.

    This test verifies that ASE's periodic boundary conditions (pbc) are
    correctly mapped to pysktb's periodicity attribute, and that max_image
    is correctly computed based on the periodicity pattern.

    The formula for max_image is: max_image = 3^sum(periodicity)
    - For quasi-2D [T,T,F]: sum=2, max_image = 3^2 = 9
    - For cluster [F,F,F]: sum=0, max_image = 3^0 = 1
    - For 3D [T,T,T]: sum=3, max_image = 3^3 = 27

    Verifies acceptance criteria:
    - AC #4: Periodicity is correctly mapped from ASE pbc to pysktb
    - AC #5: max_image is correctly computed for different periodicity patterns
    """
    # ========== Test 1: Quasi-2D structure (quasi-2D pbc=[T,T,F]) ==========
    # Create a simple cubic structure
    atoms_2d = Atoms('Si', positions=[[0, 0, 0]],
                     cell=[[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]],
                     pbc=[True, True, False])

    # Convert to pysktb Structure with orbital and bond information
    orbital_dict = {'Si': ['s', 'px', 'py', 'pz']}
    bond_dict = {'SiSi': {'NN': 2.5}}
    structure_2d = ase_atoms_to_pysktb_structure(
        atoms_2d,
        orbital_dict=orbital_dict,
        bond_cutoff_dict=bond_dict
    )

    # Verify periodicity is correctly mapped
    assert structure_2d.periodicity == [True, True, False], \
        f"Expected periodicity [True, True, False], got {structure_2d.periodicity}"

    # Verify max_image is correctly computed (3^2 = 9 for quasi-2D)
    assert structure_2d.max_image == 9, \
        f"Expected max_image=9 for quasi-2D, got {structure_2d.max_image}"

    # ========== Test 2: Cluster structure (non-periodic pbc=[F,F,F]) ==========
    # Create a cluster with no periodicity
    atoms_cluster = Atoms('Si', positions=[[0, 0, 0]],
                          cell=[[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]],
                          pbc=[False, False, False])

    # Convert to pysktb Structure with orbital and bond information
    structure_cluster = ase_atoms_to_pysktb_structure(
        atoms_cluster,
        orbital_dict=orbital_dict,
        bond_cutoff_dict=bond_dict
    )

    # Verify periodicity is correctly mapped
    assert structure_cluster.periodicity == [False, False, False], \
        f"Expected periodicity [False, False, False], got {structure_cluster.periodicity}"

    # Verify max_image is correctly computed (3^0 = 1 for non-periodic)
    assert structure_cluster.max_image == 1, \
        f"Expected max_image=1 for cluster, got {structure_cluster.max_image}"

    # ========== Test 3: Explicit periodicity override ==========
    # Create an ASE structure with pbc=[T,T,T]
    atoms_3d = Atoms('Si', positions=[[0, 0, 0]],
                     cell=[[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]],
                     pbc=[True, True, True])

    # Convert with explicit periodicity override to quasi-2D
    structure_override = ase_atoms_to_pysktb_structure(
        atoms_3d,
        orbital_dict=orbital_dict,
        bond_cutoff_dict=bond_dict,
        periodicity=[True, True, False]  # Override default pbc=[T,T,T]
    )

    # Verify the explicit periodicity override was applied
    assert structure_override.periodicity == [True, True, False], \
        f"Expected periodicity [True, True, False] from override, got {structure_override.periodicity}"

    # Verify max_image reflects the overridden periodicity
    assert structure_override.max_image == 9, \
        f"Expected max_image=9 with overridden periodicity, got {structure_override.max_image}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
