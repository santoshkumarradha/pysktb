.. PySKTB documentation master file, created by
   sphinx-quickstart on Tue Dec 20 23:10:50 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PySKTB!
========================

.. raw:: html

   <div align='center'>
   <a href="https://en.wikipedia.org/wiki/Graphene_nanoribbon">
   <figure>
   <img width="500" src="./_static/graphene-Edge-states.png" alt="Edge states of graphene in both zigzag and armchair directions" style="border-radius: 14px"></a>
   <figcaption><i>Edge states of graphene in both zigzag and armchair directions.</i></figcaption>
   </figure>
   </div>


.. figure::: examples/data/graphene-Edge-states.png
   :target: examples/examples
   :align: center
   :alt: graphene-Edge-states
   :width: 240px

   Edge states of graphene in both zigzag and armchair directions

PySKTB (Python Slater Koster Tight Binding), is a python package for Slater-Koster tight binding calculations. pysktb provides tools for constructing and solving tight binding models, as well as for calculating various properties of the resulting electronic structure.


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

- Generate s,p,d interactions in any given lattice
- Total energy *for insulators and semimetals*
- Specify range of interaction with more then Nearest neibghor
- Spin Polarized calculations
- Spin orbit coupling *(only for p orbitals as of now)*
- Plot orbital weighted colorplots
- Integration with `pymatgen <https://pymatgen.org>`_ structres 
- JIT optimized with numba
- Parallelization on kpoints

.. _installation:

📦 Installation
---------------------

.. code-block:: bash

   pip install pysktb



.. toctree::
   :maxdepth: 2
   :caption: Overview:

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

