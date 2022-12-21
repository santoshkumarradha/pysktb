.. Pysktb documentation master file, created by
   sphinx-quickstart on Tue Dec 20 23:10:50 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pysktb's documentation!
==================================

Features
--------

- Generate s,p,d interactions in any given lattice
- Total energy *for insulators and semimetals*
- Specify range of interaction with more then Nearest neibghor
- Spin Polarized calculations
- Spin orbit coupling *(only for p orbitals as of now)*
- Plot orbital weighted colorplots
- Integration with [pymatgen](https://pymatgen.org) structres 
- JIT optimized with numba
- Parallelization on kpoints

Installation
------------

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

