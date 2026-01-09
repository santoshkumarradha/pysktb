<p align="center">
  <img width="30%" src="./docs/source/_static/logo_full.png">
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.6+-blue.svg" alt="Python"></a>
  <a href="https://pysktb.readthedocs.io"><img src="https://img.shields.io/badge/docs-readthedocs-blue.svg" alt="Documentation"></a>
  <a href="https://github.com/santoshkumarradha/pysktb/issues"><img src="https://img.shields.io/github/issues/santoshkumarradha/pysktb.svg" alt="Issues"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
  <a href="https://zenodo.org/badge/latestdoi/255115236"><img src="https://zenodo.org/badge/255115236.svg" alt="DOI"></a>
</p>

<p align="center">
  <b>Slater-Koster tight-binding Hamiltonians for 1D, 2D, and 3D systems</b><br>
  <sub>From topological insulators to strongly correlated f-electron materials</sub>
</p>

<p align="center">
  <a href="https://pysktb.readthedocs.io">Documentation</a> •
  <a href="#installation">Installation</a> •
  <a href="#examples">Examples</a> •
  <a href="#citation">Citation</a>
</p>

---

## Installation

```bash
pip install pysktb
```

For the latest development version (includes f-orbital support):

```bash
git clone https://github.com/santoshkumarradha/pysktb.git
cd pysktb
pip install -e .
```

## Quick Start

```python
from pysktb import Lattice, Atom, Structure, Hamiltonian

# Define lattice and atoms
lattice = Lattice([[1, 0, 0], [0, 1, 0], [0, 0, 1]], a=3.5)
atom = Atom("X", [0, 0, 0], orbitals=["s", "px", "py", "pz"])
structure = Structure(lattice, [atom], bond_cut={"XX": {"NN": 3.6}})

# Set up Hamiltonian and solve
params = {"X": {"e_s": 0, "e_p": 1.5}, "XX": {"V_sss": -0.5, "V_sps": 0.5, "V_pps": 0.8, "V_ppp": -0.2}}
ham = Hamiltonian(structure, params)
kpts, kpts_dist, spl_pnts = ham.get_kpts([[0,0,0], [0.5,0.5,0.5]], nk=50)
eigenvalues = ham.solve_kpath(kpts)
```

## Features

| Category | Feature | Status |
|----------|---------|:------:|
| **Orbital Basis** | s, p orbitals | ● |
| | d orbitals | ● |
| | f orbitals (lanthanides) | ● |
| | Slater-Koster parametrization | ● |
| **Spin Physics** | Spin-orbit coupling (p, d, f) | ● |
| | Spin-polarized calculations | ● |
| | Magnetic systems | ● |
| **Green's Functions** | DOS with Lorentzian broadening | ● |
| | Local DOS (atom/orbital resolved) | ● |
| | Spectral function A(k,E) | ● |
| | Topological edge states | ● |
| **Forces & Dynamics** | Distance-dependent hopping | ● |
| | Harrison, PowerLaw, Exponential, GSP scaling | ● |
| | Repulsive potentials (BornMayer, Morse) | ● |
| | Atomic forces (Hellmann-Feynman) | ● |
| **Structure** | 1D, 2D, 3D systems | ● |
| | Beyond nearest-neighbor | ● |
| | [pymatgen](https://pymatgen.org) integration | ● |
| **Performance** | [Numba](https://numba.pydata.org) JIT compilation | ● |
| | k-point parallelization | ● |
| | Total energy calculations | ● |

<sub>● complete · ◐ in progress · ○ planned</sub>

## Examples

Full examples in [examples.ipynb](./docs/source/examples/examples.ipynb)

<table>
<tr>
<td align="center" width="33%">
<img src="./docs/source/examples/data/edge_states_zigzag.png" height="180"><br>
<sub><b>Topological Edge States</b><br>Spectral function A(k,E)</sub>
</td>
<td align="center" width="33%">
<img src="./docs/source/examples/data/graphene.png" height="180"><br>
<sub><b>Graphene</b><br>Band structure & BZ colorplot</sub>
</td>
<td align="center" width="33%">
<img src="./docs/source/examples/data/Perovskite_soc.png" height="180"><br>
<sub><b>Halide Perovskites</b><br>Rashba SOC effect</sub>
</td>
</tr>
</table>

### Clean Syntax

```python
# Band structure in 4 lines
ham = Hamiltonian(structure, params)
kpts, dist, pts = ham.get_kpts(path, nk=50)
bands = ham.solve_kpath(kpts)

# Green's function DOS in 3 lines
gf = GreensFunction(ham)
dos = gf.dos(energies, nk=[30, 30, 1], eta=0.1)

# Edge states in 3 lines
sgf = SurfaceGreensFunction(ham, surface_atoms=edge_atoms)
edge_spectral = sgf.edge_spectral_kpath(k_values, energies)
```

<details>
<summary><b>More examples</b></summary>

<br>

**1D sp-chain (SSH model)** — Topological crystalline insulator

<table>
<tr>
<td align="center"><img src="./docs/source/examples/data/sp-chain.png" height="140"><br><sub>Band structure</sub></td>
<td align="center"><img src="./docs/source/examples/data/sp-chain-proj.png" height="140"><br><sub>Orbital projection</sub></td>
<td align="center"><img src="./docs/source/examples/data/sp-chain-dos.png" height="140"><br><sub>Density of states</sub></td>
</tr>
</table>

**Buckled Antimony** — Topological states

<table>
<tr>
<td align="center"><img src="./docs/source/examples/data/Sb-flat.png" height="150"><br><sub>Dirac cone merging<br><a href="https://arxiv.org/abs/1912.03755">arXiv:1912.03755</a></sub></td>
<td align="center"><img src="./docs/source/examples/data/Sb_buckled.png" height="150"><br><sub>Higher-order topology<br><a href="https://arxiv.org/abs/2003.12656">arXiv:2003.12656</a></sub></td>
<td align="center"><img src="./docs/source/examples/data/buckled_sb_SOC.png" height="150"><br><sub>Surface states with SOC</sub></td>
</tr>
</table>

**f-orbital Systems** — See [f_orbital_example.py](./docs/source/examples/f_orbital_example.py) for Cerium-like lanthanide implementation with 4f electrons and spin-orbit coupling.

**Green's Function DOS** — Graphene DOS and LDOS via Green's functions

<img src="./docs/source/examples/data/greens_dos_graphene.png" height="200">

See [greens_dos_example.py](./docs/source/examples/greens_dos_example.py) for computing DOS with physical Lorentzian broadening and sublattice-resolved LDOS.

**Edge States** — Topological edge states in graphene zigzag ribbon

<img src="./docs/source/examples/data/edge_states_zigzag.png" height="220">

See [edge_states_example.py](./docs/source/examples/edge_states_example.py) for computing edge spectral functions and visualizing flat-band edge states at E=0.

**Distance-Dependent Hopping & Forces** — Scaling laws and force calculations

```python
from pysktb import Harrison, BornMayer, Forces

# Distance-dependent hopping: V(d) = V₀(d₀/d)²
params = {
    "C": {"e_s": 0.0},
    "CC": {
        "V_sss": Harrison(V0=-5.0, d0=1.42, cutoff=4.0),
        "repulsive": BornMayer(A=500, B=3.0, cutoff=4.0),
    }
}

ham = Hamiltonian(structure, params)
forces = Forces(ham)

# Compute total energy and forces
E_total, E_band, E_rep = forces.get_total_energy(n_electrons=2, nk=[10, 10, 1])
F = forces.get_forces(n_electrons=2, nk=[10, 10, 1])
```

See [forces_and_scaling.ipynb](./docs/source/examples/forces_and_scaling.ipynb) for visualizing scaling laws, energy curves, and band structure evolution with distance.

</details>

## Performance

<table>
<tr>
<td align="center" width="50%">
<img src="./docs/source/examples/data/pysktb_numba.png" height="180"><br>
<sub>JIT compilation speedup</sub>
</td>
<td align="center" width="50%">
<img src="./docs/source/examples/data/pysktb_parallel.png" height="180"><br>
<sub>k-point parallelization</sub>
</td>
</tr>
</table>

## Roadmap

| Status | Feature |
|:------:|---------|
| ● | Distance-dependent hopping & forces |
| ◐ | Phonon calculations (dynamical matrix) |
| ◐ | Electron-phonon coupling |
| ◐ | Complete pymatgen integration |
| ◐ | Berry phase calculation |
| ○ | Bogoliubov-de-Gennes (BdG) for superconductivity |
| ○ | [ASE](https://wiki.fysik.dtu.dk/ase/) structure interface |
| ● | Green's function DOS |
| ● | Topological edge states |
| ○ | Sympy analytical matrix elements |
| ○ | Low-energy k.p Hamiltonian extraction |

<sub>● complete · ◐ in progress · ○ planned</sub>

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{pysktb,
  doi       = {10.5281/ZENODO.4311595},
  url       = {https://zenodo.org/record/4311595},
  author    = {Radha, Santosh Kumar},
  title     = {pysktb: Tight-binding electronic structure codes},
  publisher = {Zenodo},
  year      = {2020}
}
```

## License

[MIT](LICENSE)
