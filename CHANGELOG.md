# Changelog

All notable changes to msinv are documented here.

## [0.3.4] - 2026-04-21

### Changed

- Release wheels: Linux-only (x86_64 + aarch64). macOS and Windows
  dropped from the build matrix because the PyO3 0.24.x ↔ Python
  3.14 / maturin toolchain issues weren't worth the CI churn for a
  research simulator whose users all run Linux. Non-Linux users can
  install from sdist with a Rust toolchain. Revisit once PyO3 ≥ 0.25
  lands native 3.14 / 3.14t support.

## [0.3.3] - 2026-04-21

(Reverted — windows fix attempt superseded by dropping non-Linux
wheels in v0.3.4.)

## [0.3.2] - 2026-04-21

### Fixed

- Release workflow aarch64-linux wheel: pin interpreters to CPython
  3.9–3.13 instead of `--find-interpreter`. Manylinux aarch64 docker
  images pull CPython 3.14 free-threaded, which cannot use PyO3's
  limited API (abi3) and therefore can't be rescued by
  `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`. Bumping PyO3 to ≥ 0.25 is
  the long-term fix; interpreter pinning ships now.

## [0.3.1] - 2026-04-21

### Added

- Rust simulator core (`msinv-core` + `msinv-py` PyO3 bridge). Python
  API unchanged; `pip install msinv` ships pre-built wheels so no
  Rust toolchain is needed at install time.
- Thread-local `RateCache` pool in the PyO3 bridge — amortises the
  first-call allocation across every subsequent `simulate()` on the
  same thread.
- `benchmarks/rho_scaling.py`: head-to-head wall-clock against
  msprime across rho 500–16000.
- `tests/hull/test_msprime_validation.py`: segregating-sites (±5 %)
  and branch-mode Fst (±10 %) comparison tests.
- `tests/hull/test_stdpopsim_validation.py`: Africa_1T12 seg-sites,
  OOA_2T12 Fst.

### Changed

- `msinv` at parity or faster than msprime for rho ≥ 8000 single-pop.
  rho=16000 with one inversion: ~4.2 s/rep vs msprime ~5.5 s/rep.
  No-inversion rho=16000: ~1.0 s/rep vs msprime ~5.5 s/rep (~5×
  faster). Peak resident memory at rho=16000: ~1.2 GB.
- README Performance section added with the bench numbers.

### Fixed

- Release workflow wheels: `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`
  for Python 3.14 compat with PyO3 0.24.2 on aarch64-linux and
  windows runners; `--find-links dist msinv` picks wheels by ABI
  tag instead of the alphabetically-first path.

## [0.3.0] - 2026-04-13

### Removed — Legacy SMC engine

The single-tree SMC simulator (``MsinvSimulator``) is removed. The
hull engine supersedes it on every axis (cross-karyotype barriers,
multi-pop demographies, nested / overlapping inversions, sweeps,
TreeSequence output) and the SMC engine had a known multi-pop bug
that produced ~½ the expected cross-pop dxy. Anyone who needs the
old engine can pin to v0.2.0.

Removed:

- ``msinv/simulator.py`` (~3500 lines)
- ``msinv/walk_segment.py``
- All 5 SMC test files (~1500 lines)
- All SMC-only example scripts (``sim_*.py``,
  ``make_kir_fol_figure.py``, ``replicate_peischl.py``,
  ``visualize_trees.py``, ``demo_kir_fol_balanced_Ne.py``,
  ``bakeoff.py``, ``compare_smc_vs_hull.py``,
  ``make_flux_transfer_figure.py``)
- ``slim_validation/`` (validation harness for the SMC engine)

### Changed

- ``msinv.InversionSpec`` and ``msinv.Demography`` now refer to the
  hull versions (the SMC versions are gone). The Hull-prefixed
  back-compat aliases (``HullInversionSpec``, ``HullDemography``)
  introduced in v0.2.0 are also removed.
- ``examples/empirical_kir_fol_hull.py`` → ``empirical_kir_fol.py``
- ``examples/empirical_rdl_sweep_hull.py`` → ``empirical_rdl_sweep.py``
- ``examples/make_figures.py`` rewritten to use the hull simulator;
  fig 8 (feature table) updated accordingly.
- Kir/Fol Da panel now shares the y-axis with the dxy panel so the
  magnitude difference (Da << dxy) is visually obvious — addresses
  the autoscale trap that made Da look "the same magnitude" as dxy.
- Test suite: 98/98 hull tests pass (down from 123 because the SMC
  tests are gone).

## [0.2.0] - 2026-04-13

### Added — Hull simulator (ARG-based, recommended)

A new ``HullSimulator`` engine that tracks per-position ancestral
material in an msprime-style hull. Architecturally correct: the
karyotype-class barrier is preserved by construction (the ARG never
modifies coalescence nodes after they're written), multi-population
demographies match msprime ground truth, and every position has the
correct structured-coalescent marginal.

Phases:

- **Phase 1** Panmictic ARG bookkeeping
- **Phase 2** Karyotype class barrier (S/I, t_inv)
- **Phase 3** Gene-flux events with class flip (Peischl phi(x))
- **Phase 4** Multi-population structure + ms-style demography
- **Phase 5a** Per-segment class (collinear flanks handled correctly)
- **Phase 5b** Multiple non-overlapping inversions per chromosome
- **Phase 5c.1** Independent karyotype assignment per inversion
- **Phase 5c.2** Nested / overlapping inversions
- **Phase 6** Selective sweep events (force-coalescence)

89 new tests in ``tests/hull/``; 114/114 total tests pass on main.

### Added — Bake-off

``examples/bakeoff.py`` runs three-way comparison (msprime ↔ SMC ↔ hull)
on six scenarios. Documents that:

- Panmictic baseline: all three agree.
- Single inversion γ=0: SMC ≈ Hull.
- Two-population split: SMC has a bug (cross-pop dxy ~half of expected);
  hull matches msprime ground truth.

### Fixed (legacy SMC simulator)

- Sample-ordering bug in ``build_structured_tree``: was
  ``sorted(sample_config.items())``, silently re-ordered samples by
  ``(class, pop)`` tuple (e.g. ``'I' < 'S'`` placed I samples before
  S). Now iterates in insertion order.
- Dispatch when ``c=0``: ``_has_inversion()`` previously required
  ``c > 0``, silently routing legitimate gamma=0 inversion runs
  through the no-inversion code path. Now checks only
  ``0 < p_inv < 1``.
- In-inv recombination is now gene-flux only (Option 3 model). The
  prior in-inv SMC prune-reattach could not reliably preserve the
  cross-karyotype T_MRCA constraint under repeated events.

### Removed

- ``c_extension/`` — the C inner-loop experiment is removed in favour
  of a planned Rust port via PyO3 (Phase 7).
- ``archive/`` — obsolete prototypes from the project's early phase.

### Documentation

- New ``docs/hull_algorithm_design.md`` describing the phased build,
  references, and known gaps.
- Updated ``docs/known_issues.md`` with the SMC fixes and the
  remaining model-choice limitations.
- Updated ``README.md`` to feature the hull simulator and document the
  SMC-vs-hull tradeoffs.

## [0.1.0] - 2026-04-12

### Initial release (legacy SMC simulator)

Features:

- Sequential Markov coalescent (SMC) with chromosomal inversions
- Structured coalescent with S/I karyotype classes
- Position-dependent gene flux (Peischl et al. 2013 model)
- Peischl b2 correlated flux tract tracking
- Multiple inversions per chromosome (up to 4)
- ms-compatible demography (eN, en, eG, eg, eM, em, ej, es)
- Per-population inversion frequency trajectories
  (Constant / Deterministic / Stochastic / Coupled)
- Selective sweep inside inversion (for resistance allele studies)
- Tree sequence output via tskit
- msprime-compatible real-unit API

Validation:

- 25/25 tests pass
- Matches msprime within ~10% on simple demographic models
- Replicates Peischl 2013 T_SI ∝ 1/phi(x) prediction
