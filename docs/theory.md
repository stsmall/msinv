# Theory background

msinv simulates the coalescent for chromosomes carrying inversions.
It implements the structured coalescent of Guerrero et al. (2012) and
the gene flux model of Peischl et al. (2013) on top of the msprime
ARG hull algorithm (Kelleher et al. 2016). Each lineage carries the
genomic intervals it is ancestral to; recombination splits intervals,
coalescence merges them, gene conversion flips the karyotype label
on a tract.

Implementation notes are in
[`hull_algorithm_design.md`](hull_algorithm_design.md).

## The structured coalescent at one site

At a position INSIDE an inversion, chromosomes fall into two karyotype
classes:

- **S** — standard arrangement
- **I** — inverted arrangement

Within a population of total size *Ne*, the S sub-population has size
*Ne · p_std* and the I sub-population has size *Ne · p_inv*, where
*p_inv* is the inversion's frequency and *p_std = 1 − p_inv*.

Per-pair coalescence rate per generation:

- **S–S pair (same pop)**: `1 / (2 · Ne · p_std)`
- **I–I pair (same pop)**: `1 / (2 · Ne · p_inv)`
- **S–I pair**: 0 — heterokaryotypes don't recombine inside the
  inversion, so an S chromosome cannot share a recent parent with an
  I chromosome at this site.

Lineages in the rarer class coalesce faster because they live in a
smaller effective sub-population.

## The class barrier (t_inv)

Inversions originated as a single new mutation at some time *t_inv*
in the past. Before *t_inv*, the inversion didn't exist — every
chromosome was on the same arrangement. Going BACKWARD in time:

- For *t < t_inv*: structured coalescent (S/I separated).
- For *t ≥ t_inv*: panmictic — all lineages can coalesce freely.

The cross-class T_MRCA is bounded below by *t_inv*. At positions
near the inversion's centre, the S and I sub-populations have
effectively been isolated for the full inversion age, producing the
characteristic "inversion-as-barrier-to-introgression" signal seen
in empirical data.

## Gene flux (Peischl 2013 model)

Inside the inversion, gene conversion can transfer a small tract of
DNA between an S chromosome and an I chromosome (going forward), even
though crossing over is suppressed. Going backward, this means a
position can have its karyotype-of-origin flip during a flux event.

The per-position flux rate per lineage per generation:

```
flux_rate(x) = γ · phi(x) · p_other
```

- *γ* = gene conversion rate (per bp per generation)
- *p_other* = frequency of the OTHER karyotype (when gene conversion
  can fire — i.e. heterokaryotype frequency)
- *phi(x)* = position-dependent factor:

```
phi(x) = min(x, 1−x, w) / (1 − w)
```

with *x* the inversion-relative position (0 to 1) and *w* the tract
window width (default 0.05 = 5% of inversion length).

**Shape of phi(x):** triangular roof — zero at the breakpoints, peak
in the middle. So gene flux concentrates in the centre of the
inversion. This is why empirically we see the strongest cross-karyotype
divergence near the breakpoints (less mixing there) and weaker
divergence in the centre (more mixing).

When a flux event fires at position *x*, a tract of length
*w · inv_len* gets its karyotype flipped — implemented exactly via
per-position class flips on the affected segment.

## Multiple populations

The structured coalescent extends naturally to multiple populations:

- A pair (A, B) can coalesce at a site only if they're in the SAME
  population AND the SAME karyotype (inside an inversion).
- Migration moves individual lineages between pops at rate
  *M_ji* per source lineage per generation.
- Population-merge events (`ej`) move all lineages of a source pop
  into a destination pop in one bulk operation.

Combined with the karyotype barrier: cross-pop AND cross-class
T_MRCA is bounded below by `max(t_split, t_inv)`. This produces the
empirically-observed pattern that cross-karyotype divergence between
two populations is much larger than same-karyotype divergence between
the same pops.

## Multiple inversions

Each inversion has its own bounds, frequency, age, and gene flux
rate. Inversions on the same chromosome may overlap or nest. At any
position, the karyotype state is a tuple (one entry per containing
inversion); two lineages can coalesce at that position only if their
states match across ALL containing inversions whose t_inv hasn't
yet been crossed.

In linked-karyotype mode (the default), each sample's `'S'` or `'I'`
applies to every inversion. In independent-karyotype mode, samples
can be S at one inversion and I at another.

## Selective sweeps

A sweep is modeled as a forced-coalescence event at the selected
position *x_sel* and time *t_event*: all lineages carrying the swept
allele at *x_sel* are merged into a single sweep ancestor at
*t_event*. Effects propagate to nearby positions via the existing
recombination/coalescence machinery.

For a within-inversion sweep that started on the S background and was
later transferred to I via gene conversion, you can stack two sweep
events (one targeting class S, one targeting class I, with the
appropriate t_event for each).

## References

- Guerrero, R. F., Rousset, F., & Kirkpatrick, M. (2012). *Coalescent
  patterns for chromosomal inversions in divergent populations.*
  Phil Trans R Soc B 367:430–438.
- Peischl, S., Koch, E., Guerrero, R. F., & Kirkpatrick, M. (2013).
  *A sequential coalescent algorithm for chromosomal inversions.*
  Heredity 111:200–209.
- Kirkpatrick, M., & Barton, N. (2006). *Chromosome inversions, local
  adaptation and speciation.* Genetics 173:419–434.
- Kelleher, J., Etheridge, A. M., & McVean, G. (2016). *Efficient
  coalescent simulation and genealogical analysis for large sample
  sizes.* PLOS Comp Bio 12:e1004842. *(msprime hull algorithm — basis
  for the* HullSimulator *implementation.)*
