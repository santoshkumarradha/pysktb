<p align="center"><img width=30.5% src="./logo.png"></p>


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
[![Build Status](https://travis-ci.org/anfederico/Clairvoyant.svg?branch=master)](https://travis-ci.org/anfederico/Clairvoyant)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
[![GitHub Issues](https://img.shields.io/github/issues/anfederico/Clairvoyant.svg)](https://github.com/anfederico/Clairvoyant/issues)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<p align="center">Scientific Python package for solving Slater Koster tight-binding hamiltonian. A python package in development for creating and solving slater koster tight-binding hamiltonians for various 1D 2D and 3D systems from topological insulators to strong correlations.</p>

                        
[![DOI](https://zenodo.org/badge/255115236.svg)](https://zenodo.org/badge/latestdoi/255115236)


## Features

  - Generate s,p,d interactions in any given lattice
  - Total energy *for insulators and semimetals*
  - Specify range of interaction with more then Nearest neibghor
  - Spin Polarized calculations
  - Spin orbit coupling *(only for p orbitals as of now)*
  - Plot orbital weighted colorplots
  - Integration with [pymatgen](https://pymatgen.org) structres 
  - JIT optimized with numba
  - Parallelization on kpoints

## Installation
 1. Copy the files _params.py and pysktb.py to the working directory
 2. Install the modules in requirements.txt.
 ```console
 pip install -U -r requirements.txt
 ```
 3. Import them and use !

 
## Examples

Example usage shown in 	[examples.ipynb](./examples/examples.ipynb)


1. 1D chain of sp (example of 1D topological Crystiline insulator *SSH*)
  
 <img src="./examples/sp-chain.png" style="max-height: 70px; max-width: 70px;" >
  
  - with orbital projection on s
  <img src="./examples/sp-chain-proj.png" style="max-height: 70px; max-width: 70px;" >
  
  - DOS
  
  <img src="./examples/sp-chain-dos.png" height="200" >
  
2. Graphene and band colorplot in BZ

  <img src="./examples/graphene.png" style="max-height: 70px; max-width: 70px;" >
  
2. Intrinsic Spin-Orbit-Coupling Rashba effect in Halide Perovskites

  <img src="./examples/Perovskite_soc.png" style="max-height: 70px; max-width: 70px;" >
  
3. Buckled antimony Sb 

   - preprint of Dirac cones merging in 2D Sb https://arxiv.org/abs/1912.03755
   
   <img src="./examples/Sb-flat.png" style="max-height: 70px; max-width: 70px;" >
   
   - preprint of Higher Order Topological states in 2D Sb https://arxiv.org/abs/2003.12656
   
   <img src="./examples/Sb_buckled.png" style="max-height: 70px; max-width: 70px;" >
   
4. Low buckled Sb Surface states with SOC - Topological Crystalline Insulator

 <img src="./examples/buckled_sb_SOC.png" style="max-height: 70px; max-width: 70px;" >
 


## Optimized 
  - with `jit`
<img src="./examples/pysktb_numba.png" height="200" >
- Parallelized over k
<img src="./examples/pysktb_parallel.png" height="200" >

## Features to be added
   - Complete pymatgen integration (high on priority)
   - Berry phase calculation (high on priority) *already implemented need to interface*
   - ~Parallelization on kpoints~ and orbitals.
   - ~scipy sparse matrix optimized~
   - Spin Orbit Coupling for d,f
   - Bogoliubov-de-Gennes (BdG) solutions for the given system for Superconductivity 
   - Interface with [ASE](https://wiki.fysik.dtu.dk/ase/) structures
   - Create finite structures and slabs for Topological calculations within the code *(requires pymatgen right now)*
   - Greens function DOS
   - Convert all operations to sympy, so that one can output analytical Tightbinding matrix elements for ease of access 
   - Low energy k.p hamiltonian from sympy
   
## Citation
If you are using the code, please consider citing it with the followig bib
[![DOI](https://zenodo.org/badge/255115236.svg)](https://zenodo.org/badge/latestdoi/255115236)
```python
@misc{https://doi.org/10.5281/zenodo.4311595,
  doi = {10.5281/ZENODO.4311595},
  url = {https://zenodo.org/record/4311595},
  author = {Radha,  Santosh Kumar},
  title = {santoshkumarradha/pysktb: Tightbinding Electronic structure codes},
  publisher = {Zenodo},
  year = {2020},
  copyright = {Open Access}
}
```
   
## License

[MIT](LICENSE) 
