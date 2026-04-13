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

## Fixed (2026-04-13): SMC walk erodes the cross-karyotype barrier

`_reattach` and `_coalesce_above_root` previously used **per-pop** `p_inv`
for the panmictic-early-return check. A target lineage in a population with
`p_inv=0` (e.g. Kiribina in Kir/Fol) would silently drop into the panmictic
branch and coalesce across the class barrier, bypassing `t_inv`. Combined
with the `sim.c*sim.rho/2` flux rate during SMC walks (used regardless of
the user's `gamma=0` setting), these two bugs partially compensated and
produced dxy patterns that looked roughly empirical but were masking a
broken class barrier.

Fixed (architectural):
- Class barrier check now uses the **global** `max` of `p_inv` across pops.
- Candidate branches are filtered by **both** class and population.
- `_coalesce_above_root` clips `t` to `root.time`.

Fixed (the real bug): the single-tree SMC prune-reattach inside an
inversion **cannot** reliably preserve the cross-karyotype T_MRCA
constraint under repeated events — pruning a coalescent node above
`t_inv` and reattaching the floating S subtree to a residual S branch
can fire a coalescence below `t_inv`, eroding the karyotype barrier.
**Replaced** the in-inv SMC prune-reattach with a full structured
rebuild via `build_structured_tree` at each event. Each in-inv site now
has the correct structured-coalescent marginal exactly.

**Trade-off:** in-inv positions are now drawn from independent
structured-coalescent trees, so there is no inversion-internal LD
between adjacent positions. Single-site marginals — which `dxy`, `Da`,
`FST`, and PCA all depend on — are correct. If users need accurate
inversion-internal LD they should use a future per-position
ancestral-material algorithm.

Verification (T_MRCA at end of SMC walk through 3Ra, n=30 reps):

| rho   | K-Fs | K-Fi | Fs-Fi | K-Fi/K-Fs |
|-------|------|------|-------|-----------|
| 0.1   | 0.88 | 5.62 | 5.62  | 6.41      |
| 10    | 0.88 | 5.62 | 5.62  | 6.41      |
| 100   | 0.88 | 5.62 | 5.62  | 6.41      |
| 704   | 0.88 | 5.62 | 5.62  | 6.41      |

Cross-karyotype T_MRCA (~5.6) stays at the expected `t_inv + 1` value
across all `rho`. Empirical dxy ratios (Kir/Fol with constant Ne):

| Scenario       | K-Fs | Fs-Fi | K-Fi |
|----------------|------|-------|------|
| 3Ra+3Rb        | 0.53 | 3.32  | 3.33 |
| 3Ra only       | 0.75 | 2.14  | 2.14 |

Cross-karyotype dxy is now correctly elevated ~2-3× inside the
inversion, matching empirical Fig S13.

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
