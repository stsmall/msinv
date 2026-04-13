# Known Issues

## Fixed (2026-04-13): Kir/Fol sample-ordering bug

Earlier versions of `build_structured_tree` called `sorted(sample_config.items())`
when assigning sample IDs to leaves, which reordered leaves by the tuple
`(class, population)`. With `{('S', 0): 10, ('S', 1): 5, ('I', 1): 5}` this
placed Fol-I samples (class 'I') **before** K-S samples (class 'S') because
`'I' < 'S'` alphabetically. Downstream code in the Kir/Fol examples assumed
insertion order, so the analysis was silently comparing Fol-I vs K-S instead
of K-S vs Fol-S, producing swapped dxy patterns.

Fixed by iterating `sample_config.items()` in insertion order (Python dicts
preserve insertion order since 3.7).

## Fixed (2026-04-13): single-vs-multi inversion dispatch and flux consistency

Two linked bugs caused single-inversion and multi-inversion code paths to
silently disagree:

1. `MsinvSimulator._has_inversion()` required `self.c > 0` AND `0 < p_inv < 1`.
   When a user set `c=0` (or `gamma=0`) intending "no gene flux inside the
   inversion", the dispatcher silently fell through to the no-inversion code
   path and ran plain panmictic coalescent. Fixed to test `0 < p_inv < 1` only.

2. `walk_segment.run_walk_segment` used `sim.c * sim.rho / 2` for the flux
   rate during SMC prune-and-reattach events, ignoring `sim.gamma`. So
   `gamma=0` was respected during the *initial* structured-tree build but
   not during subsequent SMC walks: each reattach injected flux at rate
   `c*rho/2`. In the Kir/Fol single-inversion example this hidden flux
   (c=0.01, rho=704 → rate 3.52) mixed karyotype classes, accidentally
   producing a "flatter" dxy pattern; in the multi-inversion example the
   user never set `c` on the simulator so `sim.c=0` and flux was correctly 0,
   giving the **true** structured-coalescent behaviour. The two paths
   therefore disagreed.

   Fixed by threading an explicit `flux_gamma` argument through
   `run_walk_segment` and using it (as `c=gamma, rho=2` so
   `_flux_rate(c, rho) = gamma`) in `smc_prune_and_reattach` and
   `_rebuild_structured_from_leaves`. The `_simulate_one_4walk` path now
   passes `self.gamma`; the `_simulate_one_multi_inv` path passes
   `inv.gamma` per inversion.

Verified: single-inv and multi-inv now produce identical dxy values for the
same inversion when called with matching parameters (ratio = 1.00 across
all comparisons).

## Remaining: dxy depression inside inversions with huge Ne asymmetry

With structured coalescent and extreme Ne asymmetry (e.g., Ne_F=3M vs
Ne_K=70k), inside-inversion S-S pairs coalesce at rate
`k*(k-1)/2 / p_std / Ne_pop`, which is faster than panmictic-after-split.
This gives a shorter T_MRCA for same-class cross-population comparisons
(K-Fs) **inside** the inversion than in collinear regions, producing a dxy
ratio < 1. Under the "recombination modifier" interpretation (inversions
only change recombination, not coalescent rates), this depression would not
occur and K-Fs would be flat across the inversion boundary, matching the
empirical Small et al. 2023 Fig S13 pattern. msinv currently uses the
structured coalescent, so this is a **model-choice** limitation rather than
a bug: the predicted F_ST still shows the correct empirical pattern
(elevated inside for K-F_alt and F_S-F_I, flat for K-F_same).

### What works correctly

- Standard coalescent tests (25/25 pass)
- SLiM validation with proper burn-in: msinv matches forward sim within 5%
- Single- and multi-inversion paths produce identical results (post-fix)
- F_ST patterns match empirical Fig S13 (flat for same-karyotype, elevated
  for cross-karyotype inside inversion)
- Simple inversion scenarios with constant Ne show expected phi(x) pattern
- Basic demographic inference (stdpopsim) within ~10%

### Future work

- Add an alternative "recombination modifier only" coalescent mode that
  does not partition coal rates by karyotype class
