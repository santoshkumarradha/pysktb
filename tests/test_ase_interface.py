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
# Lattice Preservation Tests
# ============================================================================

@pytest.mark.skipif(not HAS_ASE, reason="ASE not installed")
class TestLatticePreservation:
    """Test preservation of non-orthogonal cell properties during conversion."""

    def test_monoclinic_cell(self):
        """
        Test monoclinic cell angle preservation during ASE-to-pysktb conversion.

        Creates an ASE Atoms object with a monoclinic cell (non-90° angles),
        converts it to a pysktb Structure, and verifies that all lattice
        parameters (a, b, c, alpha, beta, gamma) are preserved within tolerance.

        This test maps to Acceptance Criterion #3: Non-orthogonal cell preservation.

        Verifies:
        - structure.lattice.alpha ≈ original_alpha within 1e-6 radians
        - structure.lattice.beta ≈ original_beta within 1e-6 radians
        - structure.lattice.gamma ≈ original_gamma within 1e-6 radians
        - structure.lattice.a, .b, .c are positive and reasonable magnitudes
        """
        # Create monoclinic ASE structure with known angles
        atoms = create_monoclinic_fixture()

        # Compute the expected lattice parameters from the original ASE structure
        # Extract cell vectors from ASE
        cell_vectors = atoms.get_cell()[:]

        # Compute original angles using the same formula as Lattice._to_list
        a_vec = cell_vectors[0]
        b_vec = cell_vectors[1]
        c_vec = cell_vectors[2]

        expected_a = np.linalg.norm(a_vec)
        expected_b = np.linalg.norm(b_vec)
        expected_c = np.linalg.norm(c_vec)

        # Expected angles from cell geometry
        expected_alpha = np.arctan2(np.linalg.norm(np.cross(b_vec, c_vec)), np.dot(b_vec, c_vec))
        expected_beta = np.arctan2(np.linalg.norm(np.cross(c_vec, a_vec)), np.dot(c_vec, a_vec))
        expected_gamma = np.arctan2(np.linalg.norm(np.cross(a_vec, b_vec)), np.dot(a_vec, b_vec))

        # Convert to pysktb Structure
        structure = ase_atoms_to_pysktb_structure(atoms)

        # Test 1: Verify all lattice constants are positive and reasonable
        assert structure.lattice.a > 0, "Lattice constant a must be positive"
        assert structure.lattice.b > 0, "Lattice constant b must be positive"
        assert structure.lattice.c > 0, "Lattice constant c must be positive"

        # Verify magnitudes are reasonable (in typical range for atomic structures, 1-20 Å)
        assert 0.5 < structure.lattice.a < 20.0, \
            f"Lattice a={structure.lattice.a} outside reasonable range (0.5-20 Å)"
        assert 0.5 < structure.lattice.b < 20.0, \
            f"Lattice b={structure.lattice.b} outside reasonable range (0.5-20 Å)"
        assert 0.5 < structure.lattice.c < 20.0, \
            f"Lattice c={structure.lattice.c} outside reasonable range (0.5-20 Å)"

        # Test 2: Verify angles are preserved within 1e-6 radian tolerance
        # Note: The conversion process normalizes by the first vector, so angles should be preserved exactly
        tolerance = 1e-6  # radians

        np.testing.assert_allclose(
            structure.lattice.alpha,
            expected_alpha,
            rtol=0,
            atol=tolerance,
            err_msg=f"Alpha angle not preserved: expected {expected_alpha}, got {structure.lattice.alpha}"
        )

        np.testing.assert_allclose(
            structure.lattice.beta,
            expected_beta,
            rtol=0,
            atol=tolerance,
            err_msg=f"Beta angle not preserved: expected {expected_beta}, got {structure.lattice.beta}"
        )

        np.testing.assert_allclose(
            structure.lattice.gamma,
            expected_gamma,
            rtol=0,
            atol=tolerance,
            err_msg=f"Gamma angle not preserved: expected {expected_gamma}, got {structure.lattice.gamma}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
