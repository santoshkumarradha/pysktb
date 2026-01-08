.. PySKTB documentation master file, created by
   sphinx-quickstart on Tue Dec 20 23:10:50 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _pysktb:

Welcome to PySKTB!
========================

PySKTB (Python Slater Koster Tight Binding), is a python package for Slater-Koster tight binding calculations. pysktb provides tools for constructing and solving tight binding models, as well as for calculating various properties of the resulting electronic structure.

.. code-block:: python

   import pysktb

   # Define structure
   structure = pysktb.Structure( 
      lattice=pysktb.Lattice([[1, 0, 0], [0, 10, 0], [0, 0, 10]], 1), #Lattice
      atoms=[pysktb.Atom("Si", [0, 0, 0], orbitals=["s", "px"])], #s-px orbital
      bond_cut={"SiSi": {"NN": 1.2}}, #NN cuttoff
   )

   # Construct Hamiltonian
   hamiltonian = pysktb.Hamiltonian(
      structure=structure,
      inter={"Si": {"e_p": 0, "e_s": 0},  #SK Interactions
             "SiSi": {"V_sss": -0.2, "V_sps": -0.05, "V_pps": 0.2}})

   #Solve along k-path
   path = [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.5, 0.0, -0.0]]
   k_path, k_dist, k_pts = hamiltonian.get_kpts(path, 40) #Get k-path
   evals, vecs = hamiltonian.solve_kpath(k_path, eig_vectors=True) #Solve for eigenvalues and eigenvectors

   #Plot band structure
   pysktb.Hamiltonian.plot_kproj(evals, vecs, k_dist, index=[0, 1])

*Results in*

.. raw:: html

   <div align='center'>
   <a href="https://en.wikipedia.org/wiki/Graphene_nanoribbon">
   <figure>
   <img width="250" src="./_static/sp-chain-proj.png" alt="sp-chain" style="border-radius: 14px"></a>
   <figcaption><i>Orbital weighted band structure for a chain of s-p orbitals in 1D chain</i></figcaption>
   </figure>
   </div>



Why use PySKTB ❔
------------------------------

- Specifically designed for Slater-Koster tight binding calculations, which means that it includes all of the necessary tools and functions for constructing and solving these types of models.
- Written in Python, which is a high-level programming language that is easy to read and write. This can make it particularly useful for prototyping and developing code quickly.
- Open source, which means that it is freely available to use and modify. This can be useful for researchers who want to modify the code to suit their specific needs or who want to contribute to the development of the package.
- Provides a variety of functions for constructing and solving tight binding models, including functions for generating the tight binding Hamiltonian matrix, diagonalizing the matrix to obtain the energy eigenvalues and eigenvectors, and calculating various properties of the resulting electronic structure.
- Includes tools for visualizing the results of the calculations, such as functions for plotting the band structure, density of states, and wave functions.
- Well-documented, with detailed instructions and examples provided in the package documentation. This can make it easier to get started with the package and to understand how to use its various features.
- Actively maintained and developed, which means that it is likely to receive updates and new features over time.


.. _features:

✨ Features
--------

**Orbital Basis**

- Generate s, p, d, and f orbital interactions
- Slater-Koster parametrization for any lattice
- Arbitrary orbital combinations

**Spin Physics**

- Spin-orbit coupling (p, d, f orbitals)
- Spin-polarized calculations
- Magnetic systems support

**Green's Functions**

- Density of states with Lorentzian broadening
- Local DOS (atom/orbital resolved)
- Spectral function A(k,E)
- Topological edge states via surface Green's functions

**Structure & Performance**

- 1D, 2D, and 3D systems
- Beyond nearest-neighbor interactions
- Integration with `pymatgen <https://pymatgen.org>`_ structures
- JIT optimized with `numba <https://numba.pydata.org>`_
- k-point parallelization

.. _installation:

📦 Installation
---------------------

.. code-block:: bash

   pip install pysktb





:mod:`pysktb`'s primitives
--------------------------

.. automodule:: pysktb
    :no-members:
    :no-inherited-members:

.. currentmodule:: pysktb

.. autosummary::
   :template: class.rst

   pysktb.Atom
   pysktb.Lattice
   pysktb.Structure
   pysktb.Hamiltonian
   pysktb.System
   pysktb.GreensFunction
   pysktb.SurfaceGreensFunction


.. toctree::
   :maxdepth: 1
   :caption: Overview:

   Home <self>
   Installation <install>
   What are Tight Binding models? <tightbinding>


.. toctree::
   :maxdepth: 2
   :caption: Usage Examples:

   examples/examples

.. toctree::
   :maxdepth: 1
   :caption: API reference:

   Atom <api/atom.rst>
   Lattice <api/lattice.rst>
   Structure <api/structure.rst>
   Hamiltonian <api/hamiltonian.rst>
   System <api/system.rst>

