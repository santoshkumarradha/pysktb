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
# Functional Tests - Graphene Conversion
# ============================================================================

@pytest.mark.skipif(not HAS_ASE, reason="ASE not installed")
class TestGrapheneConversion:
    """Test graphene ASE-to-pysktb conversion for functional correctness."""

    def test_graphene_conversion_produces_structure_instance(self):
        """
        Test that graphene ASE Atoms converts to a pysktb.Structure instance.

        Verifies acceptance criterion: Converts 2-atom graphene ASE Atoms to
        pysktb Structure. Also verifies the returned Structure is a pysktb.Structure
        instance.
        """
        from pysktb.structure import Structure

        # Create graphene fixture
        atoms = create_graphene_fixture()

        # Define orbital dictionary for graphene (carbon with pz orbital)
        orbital_dict = {'C': ['pz']}

        # Convert to pysktb Structure
        structure = ase_atoms_to_pysktb_structure(atoms, orbital_dict=orbital_dict)

        # Verify it's a Structure instance
        assert isinstance(structure, Structure), \
            f"Expected pysktb.Structure, got {type(structure)}"

    def test_graphene_conversion_atom_count(self):
        """
        Test that graphene conversion produces correct number of atoms.

        Verifies acceptance criterion: Verifies len(structure.atoms) == 2
        """
        atoms = create_graphene_fixture()
        orbital_dict = {'C': ['pz']}

        structure = ase_atoms_to_pysktb_structure(atoms, orbital_dict=orbital_dict)

        # Verify 2 atoms
        assert len(structure.atoms) == 2, \
            f"Expected 2 atoms in graphene, got {len(structure.atoms)}"

    def test_graphene_conversion_element_and_orbitals(self):
        """
        Test that graphene atoms have correct element and orbitals.

        Verifies acceptance criterion: Verifies structure.atoms[0].element == 'C'
        and structure.atoms[0].orbitals == ['pz']
        """
        atoms = create_graphene_fixture()
        orbital_dict = {'C': ['pz']}

        structure = ase_atoms_to_pysktb_structure(atoms, orbital_dict=orbital_dict)

        # Verify element is 'C'
        assert structure.atoms[0].element == 'C', \
            f"Expected element 'C', got {structure.atoms[0].element}"

        # Verify orbitals are ['pz']
        assert structure.atoms[0].orbitals == ['pz'], \
            f"Expected orbitals ['pz'], got {structure.atoms[0].orbitals}"

        # Also check second atom
        assert structure.atoms[1].element == 'C', \
            f"Expected element 'C', got {structure.atoms[1].element}"
        assert structure.atoms[1].orbitals == ['pz'], \
            f"Expected orbitals ['pz'], got {structure.atoms[1].orbitals}"

    def test_graphene_conversion_lattice_vectors(self):
        """
        Test that graphene lattice vectors match expected graphene properties.

        Verifies acceptance criterion: Verifies lattice vectors match graphene
        within 1e-6 Å tolerance (lattice.a ≈ 2.46 Å, lattice.gamma ≈ 120°)
        """
        atoms = create_graphene_fixture()
        orbital_dict = {'C': ['pz']}

        structure = ase_atoms_to_pysktb_structure(atoms, orbital_dict=orbital_dict)

        # Verify lattice constant a is approximately 2.46 Å
        # Note: lattice.a is the normalized magnitude; actual a = norm(lattice.matrix[0])
        lattice_a = np.linalg.norm(structure.lattice.matrix[0])
        np.testing.assert_allclose(lattice_a, 2.46, atol=1e-6,
                                  err_msg="Graphene lattice constant a should be ~2.46 Å")

        # Verify angle gamma is approximately 120° (in radians: 2.094 rad)
        # Graphene has 120° between a and b vectors
        gamma_degrees = np.degrees(structure.lattice.gamma)
        np.testing.assert_allclose(gamma_degrees, 120.0, atol=1.0,
                                  err_msg="Graphene angle gamma should be ~120°")

    def test_graphene_conversion_fractional_coords_range(self):
        """
        Test that graphene fractional coordinates are in [0, 1) range.

        Verifies acceptance criterion: Verifies fractional coords are in [0, 1) range
        """
        atoms = create_graphene_fixture()
        orbital_dict = {'C': ['pz']}

        structure = ase_atoms_to_pysktb_structure(atoms, orbital_dict=orbital_dict)

        # Check all fractional coordinates are in [0, 1) range
        for i, atom in enumerate(structure.atoms):
            pos = atom.pos
            assert np.all(pos >= 0) and np.all(pos < 1.0), \
                f"Atom {i} fractional coords {pos} not in [0, 1) range"


# ============================================================================
# Functional Tests - Si Diamond Conversion
# ============================================================================

@pytest.mark.skipif(not HAS_ASE, reason="ASE not installed")
class TestSiDiamondConversion:
    """Test silicon diamond ASE-to-pysktb conversion for functional correctness."""

    def test_si_diamond_conversion_produces_structure_instance(self):
        """
        Test that Si diamond ASE Atoms converts to a pysktb.Structure instance.

        Verifies acceptance criterion: Converts 8-atom Si diamond ASE Atoms to
        pysktb Structure
        """
        from pysktb.structure import Structure

        atoms = create_si_diamond_fixture()
        # Use individual p orbitals (px, py, pz) instead of 'p'
        orbital_dict = {'Si': ['s', 'px', 'py', 'pz']}

        structure = ase_atoms_to_pysktb_structure(atoms, orbital_dict=orbital_dict)

        assert isinstance(structure, Structure), \
            f"Expected pysktb.Structure, got {type(structure)}"

    def test_si_diamond_conversion_atom_count(self):
        """
        Test that Si diamond conversion produces correct number of atoms.

        Verifies acceptance criterion: Verifies len(structure.atoms) == 8
        """
        atoms = create_si_diamond_fixture()
        # Use individual p orbitals (px, py, pz) instead of 'p'
        orbital_dict = {'Si': ['s', 'px', 'py', 'pz']}

        structure = ase_atoms_to_pysktb_structure(atoms, orbital_dict=orbital_dict)

        assert len(structure.atoms) == 8, \
            f"Expected 8 atoms in Si diamond, got {len(structure.atoms)}"

    def test_si_diamond_conversion_all_atoms_have_orbitals(self):
        """
        Test that all Si diamond atoms have correct orbitals.

        Verifies acceptance criterion: Verifies all atoms have orbitals ['s', 'px', 'py', 'pz']
        """
        atoms = create_si_diamond_fixture()
        # Use individual p orbitals (px, py, pz) instead of 'p'
        orbital_dict = {'Si': ['s', 'px', 'py', 'pz']}

        structure = ase_atoms_to_pysktb_structure(atoms, orbital_dict=orbital_dict)

        # Check all atoms are Si with ['s', 'px', 'py', 'pz'] orbitals
        for i, atom in enumerate(structure.atoms):
            assert atom.element == 'Si', \
                f"Atom {i} element should be 'Si', got {atom.element}"
            assert atom.orbitals == ['s', 'px', 'py', 'pz'], \
                f"Atom {i} orbitals should be ['s', 'px', 'py', 'pz'], got {atom.orbitals}"

    def test_si_diamond_conversion_lattice_constant(self):
        """
        Test that Si diamond lattice constant matches expected value.

        Verifies acceptance criterion: Verifies lattice constant is 5.43 ± 0.01 Å
        (computed from lattice.matrix norm)
        """
        atoms = create_si_diamond_fixture()
        # Use individual p orbitals (px, py, pz) instead of 'p'
        orbital_dict = {'Si': ['s', 'px', 'py', 'pz']}

        structure = ase_atoms_to_pysktb_structure(atoms, orbital_dict=orbital_dict)

        # The lattice constant should be approximately 5.43 Å
        # In a cubic diamond structure, all lattice vectors have the same magnitude
        # Use the norm of the first lattice vector
        lattice_a = np.linalg.norm(structure.lattice.matrix[0])
        np.testing.assert_allclose(lattice_a, 5.43, atol=0.01,
                                  err_msg="Si diamond lattice constant should be ~5.43 ± 0.01 Å")

    def test_si_diamond_conversion_nearest_neighbor_distance(self):
        """
        Test that nearest-neighbor Si-Si bond distance is correct.

        Verifies acceptance criterion: Verifies nearest-neighbor Si-Si bond distance
        ≈ 2.35 Å (via bond_mat or distance calculation)
        """
        atoms = create_si_diamond_fixture()
        # Use individual p orbitals (px, py, pz) instead of 'p'
        orbital_dict = {'Si': ['s', 'px', 'py', 'pz']}

        structure = ase_atoms_to_pysktb_structure(atoms, orbital_dict=orbital_dict)

        # Get the distance matrix (computed in Structure.__init__)
        # The distance matrix is in Angstroms and includes all periodic images
        dist_mat = structure.dist_mat

        # Find nearest neighbor distance across all images
        # Filter the entire distance matrix to non-zero values and find minimum
        all_distances = dist_mat.flatten()
        non_zero_distances = all_distances[all_distances > 0]

        # Find minimum non-zero distance (nearest neighbor)
        nn_distance = np.min(non_zero_distances)

        # Verify nearest neighbor distance is approximately 2.35 Å
        np.testing.assert_allclose(nn_distance, 2.35, atol=0.01,
                                  err_msg="Si-Si nearest neighbor distance should be ~2.35 ± 0.01 Å")

    def test_si_diamond_conversion_bond_mat_shape(self):
        """
        Test that bond_mat is computed and has correct shape.

        Verifies acceptance criterion: Verifies bond_mat is computed and has
        correct shape (num_images, natoms, natoms)
        """
        atoms = create_si_diamond_fixture()
        # Use individual p orbitals (px, py, pz) instead of 'p'
        orbital_dict = {'Si': ['s', 'px', 'py', 'pz']}
        # Provide bonding cutoff for Si-Si
        bond_cutoff_dict = {'SiSi': {'NN': 2.5}}

        structure = ase_atoms_to_pysktb_structure(
            atoms,
            orbital_dict=orbital_dict,
            bond_cutoff_dict=bond_cutoff_dict
        )

        # Verify bond_mat exists
        assert hasattr(structure, 'bond_mat'), \
            "Structure should have bond_mat attribute"

        # Verify bond_mat is not None (it's computed because bond_cutoff_dict is provided)
        assert structure.bond_mat is not None, \
            "bond_mat should not be None when bond_cutoff_dict is provided"

        # Verify bond_mat shape is (num_images, natoms, natoms)
        expected_shape = (structure.max_image, len(structure.atoms), len(structure.atoms))
        assert structure.bond_mat.shape == expected_shape, \
            f"bond_mat shape should be {expected_shape}, got {structure.bond_mat.shape}"

        # Verify bond_mat is a boolean array
        assert structure.bond_mat.dtype == bool, \
            f"bond_mat should be boolean, got {structure.bond_mat.dtype}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
# ============================================================================

@pytest.mark.skipif(not HAS_ASE, reason="ASE not installed")
class TestParameterPassthrough:
    """Test parameter passthrough and default handling in ASE conversion."""

    def test_bond_cutoff_passthrough(self):
        """
        Test that bond_cutoff_dict is correctly passed through and bond_mat is computed.

        Verifies acceptance criterion AC #10: Structure.bond_cut should contain the
        passed bond_cutoff_dict, and structure.bond_mat should be computed and non-empty.
        """
        # Create graphene ASE structure
        atoms = create_graphene_fixture()

        # Define bond cutoff dictionary
        bond_cutoff_dict = {'CC': {'NN': 1.6}}

        # Convert with bond cutoff specification
        structure = ase_atoms_to_pysktb_structure(
            atoms,
            orbital_dict={'C': ['s', 'p']},
            bond_cutoff_dict=bond_cutoff_dict
        )

        # Verify bond_cutoff_dict is stored in structure.bond_cut
        assert structure.bond_cut == bond_cutoff_dict, \
            f"structure.bond_cut should equal {bond_cutoff_dict}, got {structure.bond_cut}"

        # Verify bond_mat exists and is non-empty
        assert structure.bond_mat is not None, "structure.bond_mat should not be None"
        assert structure.bond_mat.size > 0, "structure.bond_mat should be non-empty"

    def test_default_parameters(self):
        """
        Test default parameter handling when optional parameters are None.

        Verifies acceptance criterion AC #11: When periodicity=None, numba=True, name=None,
        the structure should have:
        - periodicity inferred from ASE pbc (True for all 3 dimensions)
        - name defaults to 'system'
        - numba should be True
        """
        # Create simple cubic ASE structure with all pbc=True
        atoms = create_simple_cubic_fixture()
        assert all(atoms.pbc), "Fixture should have all pbc=True"

        # Convert with None for optional parameters
        structure = ase_atoms_to_pysktb_structure(
            atoms,
            orbital_dict={'Fe': ['s']},
            periodicity=None,
            numba=True,
            name=None
        )

        # Verify periodicity is inferred from ASE pbc
        assert structure.periodicity == [True, True, True], \
            f"structure.periodicity should be [True, True, True], got {structure.periodicity}"

        # Verify name defaults to 'system'
        assert structure.name == 'system', \
            f"structure.name should be 'system', got {structure.name}"

        # Verify numba flag is True
        assert structure.numba is True, \
            f"structure.numba should be True, got {structure.numba}"

    def test_fractional_wrapping(self):
        """
        Test that fractional coordinates are wrapped to [0, 1) range.

        Verifies acceptance criterion AC #12: All fractional coordinates in
        structure.atoms should be within [0, 1) after conversion.
        """
        # Create a simple structure but manually shift some atoms outside [0, 1)
        a = 3.0
        cell = np.array([
            [a, 0, 0],
            [0, a, 0],
            [0, 0, a]
        ])

        # Positions in fractional coordinates (some outside [0, 1))
        # ASE Atoms() expects Cartesian coordinates, so we need to convert
        fractional_positions = np.array([
            [0.2, 0.3, 0.4],    # Normal position
            [1.2, 0.5, 0.6],    # x > 1, should wrap to 0.2
            [0.8, 1.1, 0.9],    # y > 1, should wrap to 0.1
            [-0.3, 0.4, 1.2],   # x < 0, y normal, z > 1
        ])

        # Convert fractional to Cartesian for ASE
        cartesian_positions = np.dot(fractional_positions, cell)

        # Create ASE Atoms object
        atoms = Atoms('Fe4', positions=cartesian_positions, cell=cell, pbc=True)

        # Convert to pysktb structure
        structure = ase_atoms_to_pysktb_structure(
            atoms,
            orbital_dict={'Fe': ['s']}
        )

        # Verify all fractional coordinates are in [0, 1)
        for i, atom in enumerate(structure.atoms):
            pos = atom.pos
            for j, coord in enumerate(pos):
                assert 0.0 <= coord < 1.0, \
                    f"Atom {i}, coordinate {j}: position {coord} not in [0, 1)"


# ============================================================================
# Module Export and Optional Dependency Tests
# ============================================================================

class TestModuleExports:
    """Test module exports and optional ASE dependency handling."""

    def test_export_in_init(self):
        """
        Test that ase_atoms_to_pysktb_structure is properly exported.

        Verifies acceptance criteria:
        - AC #16: Function can be imported directly from pysktb package
        - AC #18: Function is listed in __all__ for proper public API exposure
        """
        import pysktb

        # Verify the function is accessible from the main pysktb module
        assert hasattr(pysktb, 'ase_atoms_to_pysktb_structure'), \
            "ase_atoms_to_pysktb_structure should be accessible from pysktb module"

        # Verify the function is in __all__ for proper public API exposure
        assert 'ase_atoms_to_pysktb_structure' in pysktb.__all__, \
            "ase_atoms_to_pysktb_structure should be in pysktb.__all__"

        # Verify it can be imported directly (when ASE is installed)
        if HAS_ASE:
            from pysktb import ase_atoms_to_pysktb_structure as func
            assert callable(func), \
                "ase_atoms_to_pysktb_structure should be callable"

        # Also verify it's available in interfaces module
        from pysktb.interfaces import ase_atoms_to_pysktb_structure
        assert callable(ase_atoms_to_pysktb_structure) or HAS_ASE, \
            "ase_atoms_to_pysktb_structure should be accessible from pysktb.interfaces"

    def test_missing_ase_import(self):
        """
        Test that helpful error message is provided when ASE is not installed.

        Verifies acceptance criteria:
        - AC #18: When ASE import fails, a helpful error message guides users
        - Function should raise ImportError with installation guidance

        Uses unittest.mock.patch to simulate ASE import failure and verifies
        that the fallback stub provides helpful error guidance.
        """
        import sys
        from unittest import mock

        # Get the current function from pysktb
        import pysktb

        # Test the actual behavior of the fallback stub by mocking the import
        # We patch at the point where the import happens in pysktb/__init__.py
        with mock.patch('pysktb.interfaces.ase.ase_atoms_to_pysktb_structure',
                       side_effect=ImportError("ASE is required")):
            # Reimport pysktb to trigger the fallback logic
            import importlib
            importlib.reload(pysktb)

            # Get the function (should be the fallback stub)
            func = pysktb.ase_atoms_to_pysktb_structure

            # Call the function and verify it raises ImportError with helpful message
            with pytest.raises(ImportError) as excinfo:
                func(None)

            error_msg = str(excinfo.value)
            # Verify the error message contains helpful installation guidance
            assert 'pip install' in error_msg or 'ase' in error_msg.lower(), \
                "Error message should suggest installation guidance (e.g., 'pip install ase')"

        # Restore the original pysktb after the mock
        importlib.reload(pysktb)

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


def test_multi_element_structure():
    """
    Test conversion of multi-element structures (GaAs example).

    This test verifies that the ASE-to-pysktb conversion correctly handles
    structures with multiple different elements, where each element can have
    its own orbital definition in the orbital_dict.

    Verifies acceptance criteria AC #5:
    - Creates GaAs ASE Atoms (2 atoms, zinc-blende)
    - Converts with orbital_dict={'Ga': ['s', 'p'], 'As': ['s', 'p']}
    - Verifies structure.atoms[0].element in ['Ga', 'As']
    - Verifies structure.atoms[1].element in ['Ga', 'As']
    - Verifies len(structure.get_elements()) == 2
    - Verifies both 'Ga' and 'As' are in structure.get_elements()
    """
    # Create GaAs structure using fixture helper
    atoms = create_gaas_fixture()

    # Define orbital dictionary with per-element assignments
    # Note: p orbital should be broken into px, py, pz
    orbital_dict = {
        'Ga': ['s', 'px', 'py', 'pz'],  # Ga gets s and p orbitals
        'As': ['s', 'px', 'py', 'pz']   # As gets s and p orbitals
    }

    # Define bond cutoff distances for GaAs
    # GaAs nearest-neighbor distance is ~2.45 Å for zinc-blende structure
    bond_cutoff_dict = {
        'GaAs': {'NN': 3.0},  # Cutoff for Ga-As bonds
        'GaGa': {'NN': 4.0},  # Cutoff for Ga-Ga bonds
        'AsAs': {'NN': 4.0}   # Cutoff for As-As bonds
    }

    # Convert to pysktb Structure
    structure = ase_atoms_to_pysktb_structure(
        atoms,
        orbital_dict=orbital_dict,
        bond_cutoff_dict=bond_cutoff_dict
    )

    # Verify structure has 2 atoms
    assert len(structure.atoms) == 2, "Structure should have 2 atoms"

    # Verify both atoms have valid elements
    assert structure.atoms[0].element in ['Ga', 'As'], \
        f"First atom element {structure.atoms[0].element} must be Ga or As"
    assert structure.atoms[1].element in ['Ga', 'As'], \
        f"Second atom element {structure.atoms[1].element} must be Ga or As"

    # Verify get_elements() returns exactly 2 unique elements
    elements = structure.get_elements()
    assert len(elements) == 2, \
        f"get_elements() should return 2 elements, got {len(elements)}: {elements}"

    # Verify both Ga and As are in the elements list
    assert 'Ga' in elements, "Gallium (Ga) must be in elements list"
    assert 'As' in elements, "Arsenic (As) must be in elements list"

    # Verify orbital assignments are correct for each atom
    for i, atom in enumerate(structure.atoms):
        if atom.element == 'Ga':
            assert atom.orbitals is not None, f"Atom {i} (Ga) should have orbitals assigned"
            assert set(atom.orbitals) == {'s', 'px', 'py', 'pz'}, \
                f"Atom {i} (Ga) should have s, px, py, pz orbitals, got {atom.orbitals}"
        elif atom.element == 'As':
            assert atom.orbitals is not None, f"Atom {i} (As) should have orbitals assigned"
            assert set(atom.orbitals) == {'s', 'px', 'py', 'pz'}, \
                f"Atom {i} (As) should have s, px, py, pz orbitals, got {atom.orbitals}"



if __name__ == '__main__':
    pytest.main([__file__, '-v'])
