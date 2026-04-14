# Known issues

The hull simulator (the only engine in msinv ≥ 0.3.0) has no
outstanding correctness bugs.

If you find unexpected behaviour, please open an issue at
<https://github.com/stsmall/msinv/issues> with a reproducible example
(seed, parameters, expected vs observed output).

## Limitations / caveats

- **Extreme Ne asymmetry between populations.** With very large Ne
  in one population (e.g. Ne_F = 3M vs Ne_K = 70k), inside-inversion
  same-class cross-population pairs can show a small dxy depression
  inside the inversion vs collinear regions. This is the standard
  structured-coalescent prediction, not a bug. The empirical Kir/Fol
  scenario uses constant Ne (see ``examples/empirical_kir_fol_hull.py``)
  to avoid this artifact.

- **Mutation rate.** `HullSimulator` does not place mutations. Use
  `msprime.sim_mutations(ts, rate=mu)` on the returned TreeSequence.

- **Discrete vs continuous coordinates.** The hull simulator uses
  continuous (float) genomic coordinates. Pass
  `discrete_genome=False` to `msprime.sim_mutations` for compatible
  mutation placement.

## Historical SMC engine (removed in v0.3.0)

The legacy `MsinvSimulator` (single-tree SMC) was deprecated and
removed in v0.3.0 because the hull engine supersedes it on every
axis: cross-karyotype barriers, multi-pop demographies, nested /
overlapping inversions, sweeps, and tree-sequence output. The SMC
engine had a known multi-pop bug that produced ~½ the expected
cross-pop dxy. Anyone needing the old engine can checkout v0.2.0 or
earlier; nothing in v0.3.0+ depends on it.
