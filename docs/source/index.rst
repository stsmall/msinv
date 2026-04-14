msinv documentation
===================

**msinv** is a Python coalescent simulator for chromosomal inversions
built on the msprime hull algorithm (per-position ancestral material
tracking).

Features:

* ARG-based per-position simulation
* Cross-karyotype barrier (S vs I) with inversion-age (t_inv) cutoff
* Position-dependent gene flux (Peischl et al. 2013)
* Multiple inversions per chromosome (overlapping / nested OK)
* ms-style demography (size changes, growth, migration, merges)
* Selective sweeps (force-coalescence)
* Tree sequence (tskit) output

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

   from msinv import HullSimulator, InversionSpec

   sim = HullSimulator(
       n_std=5, n_inv=5,
       population_size=10_000,
       sequence_length=100_000,
       inversions=[InversionSpec(bp_left=30_000, bp_right=70_000,
                                  p_inv=0.5, t_inv=200_000)],
       seed=42,
   )
   ts = sim.simulate()

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
