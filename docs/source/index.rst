msinv documentation
===================

**msinv** is a Python coalescent simulator for chromosomal inversions.

Features:

* Sequential Markov coalescent (SMC) with inversions
* Structured coalescent between karyotype classes (S/I)
* Position-dependent gene flux (Peischl et al. 2013)
* Multiple inversions per chromosome
* Per-population inversion frequency trajectories
* ms-compatible demography
* Tree sequence output (tskit)
* msprime-compatible real-unit API

.. toctree::
   :maxdepth: 2
   :caption: Contents

   ../installation
   ../quickstart
   ../theory
   ../examples
   api_reference

Quick start
-----------

.. code-block:: python

   from msinv import MsinvSimulator

   sim = MsinvSimulator(
       samples=10, population_size=10_000,
       mutation_rate=1e-8, recombination_rate=1e-8,
       sequence_length=100_000,
       n_std=5, n_inv=5, p_inv=0.5,
       t_inv=200_000, bp_left=0.3, bp_right=0.7,
       seed=42,
   )
   positions, haplotypes = sim.simulate_one()

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
