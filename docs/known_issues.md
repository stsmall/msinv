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
preserve insertion order since 3.7). Single-inversion Kir/Fol now shows the
expected empirical pattern: cross-karyotype dxy (K-F_alt, F_S-F_I) is higher
inside the inversion (ratio ~1.08) than collinear, while same-karyotype dxy
is slightly lower or flat — matching Small et al. 2023 Fig. S13.

## Remaining: dxy depression inside inversions with 2 inversions + huge Ne_F

With both 3Ra **and** 3Rb plus Folonzo Ne_F=3,000,000, the simulation still
shows dxy that is somewhat **lower** inside inversions than collinear regions
(ratios 0.64-0.90). A single-inversion run with the same parameters produces
the correct empirical pattern (ratios 0.92-1.09), so the issue is specific to
the interaction between two nested inversions and the extreme Ne_F growth.

### Root cause (partially understood)

**Structured coalescent rate scaling with huge N**: Inside an inversion, S-S
pairs coalesce at rate `k*(k-1)/2 / p_std`, which is FASTER than panmictic.
This correctly models the smaller effective "S population", but gives shorter
T_MRCA and lower dxy for same-class comparisons. With Ne_F=3M panmictic
T_MRCA is very long; inside-inversion S-S T_MRCA is bounded by `p_std * Ne`,
so the contrast is extreme. With two inversions this effect accumulates.

### What works correctly

- Standard coalescent tests (25/25 pass)
- SLiM validation with proper burn-in: msinv matches forward sim within 5%
- Single-inversion Kir/Fol with real params: empirical pattern reproduced
- Simple inversion scenarios with constant Ne show expected phi(x) pattern
- Basic demographic inference (stdpopsim) within ~10%

### Workaround

For Kir/Fol-style applications with nested inversions, avoid extreme Ne
asymmetry. Use either:
- Constant Ne across populations
- Smaller Ne_F (e.g., 100,000 instead of 3,000,000)
- A single inversion

### Future work

- Investigate the structured coalescent's behavior under asymmetric Ne
- Consider alternative tree construction at boundaries between nested inversions
