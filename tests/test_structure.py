import pytest
import numpy as np
from pysktb.lattice import Lattice
from pysktb.atom import Atom
from pysktb.structure import Structure


class TestBondCutoffBackwardCompatibility:
    """Test backward compatibility with 'NN' format"""

    def test_backward_compatibility_nn_format(self):
        """AC1: Existing code using {'XX': {'NN': value}} format continues to work"""
        # Create simple cubic lattice with a=3.0
        lattice_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        lattice = Lattice(lattice_matrix, 3.0)

        # Create two carbon atoms at distance ~1.5 (in fractional coords: 0.5 apart = 1.5 angstrom)
        atom1 = Atom("C", [0.0, 0.0, 0.0])
        atom2 = Atom("C", [0.5, 0.0, 0.0])
        atoms = [atom1, atom2]

        # Use old NN format
        bond_cut = {"CC": {"NN": 3.0}}
        structure = Structure(lattice, atoms, bond_cut=bond_cut)

        # Assert bond_mat is not None and has expected shape
        assert structure.bond_mat is not None
        assert isinstance(structure.bond_mat, np.ndarray)
        # Bonds should exist since distance (1.5) < cutoff (3.0)
        assert np.any(structure.bond_mat == True)


class TestBondCutoffMinMaxFormat:
    """Test new min/max format support"""

    def test_min_max_format_accepted(self):
        """AC2: bond_cut accepts {'XX': {'min': lower, 'max': upper}} format"""
        lattice_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        lattice = Lattice(lattice_matrix, 3.0)

        atom1 = Atom("C", [0.0, 0.0, 0.0])
        atom2 = Atom("C", [0.5, 0.0, 0.0])
        atoms = [atom1, atom2]

        # Use new min/max format
        bond_cut = {"CC": {"min": 1.0, "max": 3.0}}
        structure = Structure(lattice, atoms, bond_cut=bond_cut)

        # Assert bond_mat is not None
        assert structure.bond_mat is not None
        assert isinstance(structure.bond_mat, np.ndarray)


class TestBondCutoffLower:
    """Test lower cutoff exclusion"""

    def test_lower_cutoff_exclusion(self):
        """AC3: Bonds with distance < min_cutoff are excluded from bond_mat"""
        lattice_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        lattice = Lattice(lattice_matrix, 3.0)

        # Two atoms at distance 1.5 angstroms (0.5 in fractional coords * 3.0 lattice constant)
        atom1 = Atom("C", [0.0, 0.0, 0.0])
        atom2 = Atom("C", [0.5, 0.0, 0.0])
        atoms = [atom1, atom2]

        # Set min cutoff to 2.0, so bonds with distance 1.5 should be excluded
        bond_cut = {"CC": {"min": 2.0, "max": 3.0}}
        structure = Structure(lattice, atoms, bond_cut=bond_cut)

        # No bonds should exist (distance 1.5 < min 2.0)
        assert not np.any(structure.bond_mat == True)


class TestBondCutoffUpper:
    """Test upper cutoff still works"""

    def test_upper_cutoff_exclusion(self):
        """AC4: Bonds with distance >= max_cutoff are excluded from bond_mat"""
        lattice_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        lattice = Lattice(lattice_matrix, 3.0)

        # Two atoms at distance 1.5 angstroms
        atom1 = Atom("C", [0.0, 0.0, 0.0])
        atom2 = Atom("C", [0.5, 0.0, 0.0])
        atoms = [atom1, atom2]

        # Set max cutoff to 1.0, so bonds with distance 1.5 should be excluded
        bond_cut = {"CC": {"max": 1.0}}
        structure = Structure(lattice, atoms, bond_cut=bond_cut)

        # No bonds should exist (distance 1.5 >= max 1.0)
        assert not np.any(structure.bond_mat == True)


class TestBondCutoffSelfExclusion:
    """Test self-exclusion preserved"""

    def test_self_exclusion(self):
        """AC5: Atoms exclude themselves (distance = 0) regardless of min value"""
        lattice_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        lattice = Lattice(lattice_matrix, 3.0)

        # Two atoms at same position (distance = 0)
        atom1 = Atom("C", [0.0, 0.0, 0.0])
        atom2 = Atom("C", [0.0, 0.0, 0.0])
        atoms = [atom1, atom2]

        # Set min to 0 (which would include distance 0 if not handled)
        bond_cut = {"CC": {"min": 0, "max": 5.0}}
        structure = Structure(lattice, atoms, bond_cut=bond_cut)

        # Get distance matrix to find where distance is exactly 0
        dist_mat = structure.dist_mat

        # For all positions where distance is 0, bond_mat should be False
        zero_dist_mask = dist_mat == 0
        # All bonds at zero distance should be excluded
        assert not np.any(structure.bond_mat[zero_dist_mask]), (
            "Bonds found at zero distance (self-bonds)"
        )


class TestBondCutoffRangeInclusion:
    """Test range inclusion works"""

    def test_range_inclusion(self):
        """AC6: Bonds within min <= distance < max are included"""
        lattice_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        lattice = Lattice(lattice_matrix, 3.0)

        # Two atoms at distance 1.5 angstroms
        atom1 = Atom("C", [0.0, 0.0, 0.0])
        atom2 = Atom("C", [0.5, 0.0, 0.0])
        atoms = [atom1, atom2]

        # Set range 1.0 to 3.0, so bonds with distance 1.5 should be included
        bond_cut = {"CC": {"min": 1.0, "max": 3.0}}
        structure = Structure(lattice, atoms, bond_cut=bond_cut)

        # Bonds should exist (1.0 <= 1.5 < 3.0)
        assert np.any(structure.bond_mat == True)
