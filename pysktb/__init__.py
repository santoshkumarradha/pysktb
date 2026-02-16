from .structure import Structure
from .atom import Atom
from .lattice import Lattice
from .system import System
from .hamiltonian import Hamiltonian
from .greens import GreensFunction, SurfaceGreensFunction
from .forces import Forces
from .__version__ import __version__

# Scaling laws for distance-dependent hopping
from .scaling import (
    ScalingLaw,
    Constant,
    Harrison,
    PowerLaw,
    Exponential,
    GSP,
    Polynomial,
    Tabulated,
    Custom,
    ensure_scaling,
)

# Repulsive potentials
from .repulsive import (
    RepulsivePotential,
    BornMayer,
    Buckingham,
    Morse,
    SplineRepulsive,
    ZeroPotential,
)

# Orbital visualization
from .orbitals import (
    RadialFunction,
    SlaterOrbital,
    HydrogenOrbital,
    GaussianOrbital,
    AtomicOrbital,
    OrbitalBasis,
    DEFAULT_SLATER_ZETA,
)
from .visualization import (
    OrbitalPlotter,
    tile_supercell,
    tile_amplitudes,
    get_nearest_neighbor_bonds,
    draw_bonds,
    draw_pz_lobes,
    plot_ribbon_edge_state,
    plot_edge_vs_bulk_comparison,
)

# Optional ASE interface
try:
    from .interfaces.ase import ase_atoms_to_pysktb_structure
except ImportError:
    # ASE not installed, provide fallback stub function
    def ase_atoms_to_pysktb_structure(*args, **kwargs):
        raise ImportError(
            "ASE is required for this function. "
            "Please install it with: pip install ase"
        )

__all__ = [
    # Core classes
    "Structure",
    "Atom",
    "Lattice",
    "System",
    "Hamiltonian",
    "GreensFunction",
    "SurfaceGreensFunction",
    "Forces",
    # ASE interface
    "ase_atoms_to_pysktb_structure",
    # Scaling laws
    "ScalingLaw",
    "Constant",
    "Harrison",
    "PowerLaw",
    "Exponential",
    "GSP",
    "Polynomial",
    "Tabulated",
    "Custom",
    "ensure_scaling",
    # Repulsive potentials
    "RepulsivePotential",
    "BornMayer",
    "Buckingham",
    "Morse",
    "SplineRepulsive",
    "ZeroPotential",
    # Orbital visualization
    "RadialFunction",
    "SlaterOrbital",
    "HydrogenOrbital",
    "GaussianOrbital",
    "AtomicOrbital",
    "OrbitalBasis",
    "OrbitalPlotter",
    "DEFAULT_SLATER_ZETA",
    # Ribbon visualization helpers
    "tile_supercell",
    "tile_amplitudes",
    "get_nearest_neighbor_bonds",
    "draw_bonds",
    "draw_pz_lobes",
    "plot_ribbon_edge_state",
    "plot_edge_vs_bulk_comparison",
]

