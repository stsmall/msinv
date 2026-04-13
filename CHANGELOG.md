# Changelog

All notable changes to msinv are documented here.

## [0.1.0] - 2026-04-12

### Initial release

Features:
- Sequential Markov coalescent (SMC) with chromosomal inversions
- Structured coalescent with S/I karyotype classes
- Position-dependent gene flux (Peischl et al. 2013 model)
- Peischl b2 correlated flux tract tracking
- Multiple inversions per chromosome (up to 4)
- ms-compatible demography (eN, en, eG, eg, eM, em, ej, es)
- Per-population inversion frequency trajectories
  - ConstantFrequency
  - DeterministicTrajectory (logistic sweep)
  - StochasticTrajectory (WF diffusion with reflecting boundary)
  - CoupledTrajectory (2D per-population with migration)
- Selective sweep inside inversion (for resistance allele studies)
- Tree sequence output via tskit (single-root, 0/5981 multi-root validated)
- msprime-compatible real-unit API
- 4-walk strategy for symmetric phi(x) shape

Validation:
- 46/46 tests pass (standard, inversion, LD, tree seq, stdpopsim)
- Matches msprime within ~10% on demographic models
- Matches empirical data: Small 2023 (Kir/Fol), Grau-Bové 2020 (RDL),
  2La Fst, MAPT dxy
- Replicates Peischl 2013 T_SI ∝ 1/phi(x) prediction

Package:
- pip/pixi installable via pyproject.toml
- GitHub Actions CI
- Sphinx documentation
- MIT licensed
