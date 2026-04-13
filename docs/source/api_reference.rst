API Reference
=============

Simulator
---------

.. autoclass:: msinv.MsinvSimulator
   :members:
   :undoc-members:
   :show-inheritance:

Inversion specification
-----------------------

.. autoclass:: msinv.InversionSpec
   :members:
   :undoc-members:

.. autoclass:: msinv.GeneFluxModel
   :members:
   :undoc-members:

Frequency trajectories
----------------------

.. autoclass:: msinv.ConstantFrequency
   :members:
   :undoc-members:

.. autoclass:: msinv.DeterministicTrajectory
   :members:
   :undoc-members:

.. autoclass:: msinv.StochasticTrajectory
   :members:
   :undoc-members:

.. autoclass:: msinv.CoupledTrajectory
   :members:
   :undoc-members:

Demography
----------

.. autoclass:: msinv.Demography
   :members:
   :undoc-members:

Tree utilities
--------------

.. autoclass:: msinv.Node
   :members:
   :undoc-members:

.. autoclass:: msinv.EdgeRecorder
   :members:
   :undoc-members:

.. autofunction:: msinv.get_all_nodes

.. autofunction:: msinv.get_branches

.. autofunction:: msinv.find_root

n=2 utilities (exact coalescent)
---------------------------------

.. autofunction:: msinv.build_initial_tree

.. autofunction:: msinv.smc_step

.. autofunction:: msinv.simulate_one_n2

.. autofunction:: msinv.phi
