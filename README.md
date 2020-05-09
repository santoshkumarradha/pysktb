i
<img src="./logo.png" height="200" >
Scientific Python package for solving Slater Koster tight-binding hamiltonian
                            

A python package in development for creating and solving slater koster tight-binding hamiltonians for various 1D 2D and 3D systems.

### Features

  - Generate s,p,d interactions in any given lattice
  - Total energy *for insulators and semimetals*
  - Specify range of interaction with more then Nearest neibghor
  - Spin Polarized calculations
  - Spin orbit coupling *(only for p orbitals as of now)*
  - Plot orbital weighted colorplots
  - Integration with [pymatgen](https://pymatgen.org) structres 
  - JIT optimized with numba
  - Parallelization on kpoints

### Installation
 1. Copy the files _params.py and pysktb.py to the working directory
 2. Install the modules in requirements.txt.
 ```console
 pip install -U -r requirements.txt
 ```
 3. Import them and use !

 
### Examples

Example usage shown in 	[examples.ipynb](./examples/examples.ipynb)
1. 1D chain of sp (example of 1D topological Crystiline insulator *SSH*)
  <img src="./examples/sp-chain.png" style="max-height: 70px; max-width: 70px;" >
  - with orbital projection on s
  <img src="./examples/sp-chain-proj.png" style="max-height: 70px; max-width: 70px;" >
  - DOS
  <img src="./examples/sp-chain-dos.png" height="200" >
2. Graphene and band colorplot in BZ
  <img src="./examples/graphene.png" style="max-height: 70px; max-width: 70px;" >
3. Buckled antimony Sb 
   - preprint of Dirac cones merging in 2D Sb https://arxiv.org/abs/1912.03755
   <img src="./examples/Sb-flat.png" style="max-height: 70px; max-width: 70px;" >
   - preprint of Higher Order Topological states in 2D Sb https://arxiv.org/abs/2003.12656
   <img src="./examples/Sb_buckled.png" style="max-height: 70px; max-width: 70px;" >
4. Low buckled Sb Surface states with SOC - Topological Crystalline Insulator
 <img src="./examples/buckled_sb_SOC.png" style="max-height: 70px; max-width: 70px;" >


### Optimized 
  - with `jit`
<img src="./examples/pysktb_numba.png" height="200" >
- Parallelized over k
<img src="./examples/pysktb_parallel.png" height="200" >

### Features to be added
   - ~Parallelization on kpoints~ and orbitals.
   - ~scipy sparse matrix optimized~
   - Spin Orbit Coupling for d,f
   - Bogoliubov-de-Gennes (BdG) solutions for the given system for Superconductivity 
   - Interface with [ASE](https://wiki.fysik.dtu.dk/ase/) structures
   - Create finite structures and slabs for Topological calculations within the code *(requires pymatgen right now)*
   - Berry phase calculation (high on priority)
   - Greens function DOS
   - Convert all operations to sympy, so that one can output analytical Tightbinding matrix elements for ease of access 

