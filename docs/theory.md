# Theory background

This page explains what `msinv` is doing under the hood, in plain
language. The math and the citations are at the bottom for anyone
who wants them.

## What "the coalescent" actually means

The coalescent is the standard population-genetics way of simulating
genetic data **backwards in time**. Instead of starting with a
population of individuals and watching mutations accumulate forward,
you start with the chromosomes you sampled today and trace each one's
ancestor back through time until they all meet at a common ancestor.

This is much faster than forward simulation because you only need to
track the ancestors of the chromosomes you actually sampled, not the
whole population.

`msinv` builds on the same engine that powers `msprime`: each
chromosome's family history is tracked **position by position** along
the genome (an "ARG" — ancestral recombination graph). When two
positions on a chromosome have different histories because of a past
recombination event, the simulator handles that exactly.

## What an inversion does to the family tree

An inversion is a chunk of chromosome that got flipped end-to-end at
some point in the past. From then on, individuals carrying the flipped
arrangement (call them **I**) and individuals carrying the standard
arrangement (call them **S**) cannot easily exchange DNA across the
flipped region during meiosis — when an S and an I chromosome try to
pair up, the geometry doesn't work and recombination inside the
inversion gets suppressed.

The consequence: from the perspective of any position **inside** the
inversion, the S chromosomes and the I chromosomes look like two
separate sub-populations. They can only share an ancestor by going far
enough back in time to reach the moment the inversion first occurred.

The age of the inversion is called **`t_inv`** — the time (in
generations ago) when the flip first happened. Going back in time:

- Until you reach `t_inv`, S samples can only meet other S samples,
  and I samples can only meet other I samples.
- At `t_inv` and before, the two arrangements collapse back into one
  (because the inversion didn't exist yet) and everything mixes
  freely.

This is the **karyotype barrier**, and it's why empirical studies see
elevated divergence between S and I haplotypes inside the inversion.

Outside the inversion, none of this applies — recombination works
normally and the coalescent looks like the standard one.

## Inversion frequency matters

If only 10% of chromosomes carry the inverted arrangement, then the
"I sub-population" is small — only 10% of the breeding pool. Small
populations coalesce faster (fewer ancestors to choose from). So the
rarer arrangement tends to have less internal diversity than the
common one. `msinv` handles this automatically — you supply the
inversion frequency `p_inv` (per population if you have multiple
populations) and the simulator scales each sub-population's
coalescence rate accordingly.

### Why pi *within* a karyotype class is depressed inside the inversion

A common surprise when looking at output from `msinv` (or any
structured-coalescent inversion simulator): if you compute pi using
just the S samples, you'll see pi *inside* the inversion is **lower**
than pi in the surrounding collinear region. Same for pi within I.

This is not a bug — it's the direct consequence of the karyotype
barrier. Outside the inversion, your S samples are just random
samples from the full population, so they coalesce with effective
size *Ne* and pi looks normal. Inside the inversion, those same S
samples can only coalesce with other S samples, so they're
effectively living in a sub-population of size *Ne · p_S*. Half the
effective size means twice the coalescence rate, which means roughly
half the diversity. With `p_inv = 0.5`, expect within-class pi inside
the inversion to be ~50% of the collinear-region value; with a rarer
arrangement the depression is proportionally bigger for that class.

The classic Wakeley / Hey / Charlesworth & Charlesworth (1973)
prediction is exactly this: at a structured locus, within-class
diversity scales with the class's frequency. Empirically this is
visible in real inversions when sample sizes are large enough —
within-arrangement pi dips inside the inversion even at neutrally-
evolving loci.

The same effect explains the dxy "dip" sometimes seen for
**same-arrangement** comparisons across populations (e.g. K-vs-Fol-S
in the Kir/Fol example). It's not introgression or admixture — it's
the smaller within-class effective size doing what it always does.

## Gene flux: the leaky barrier

Crossing-over is suppressed inside the inversion when an S and an I
pair up, but **gene conversion** can still transfer short DNA tracts
between the two arrangements. This is the only way new variation can
move from S to I or vice versa within the inversion.

Going forward in time, gene conversion occasionally copies a few
hundred base pairs from an I chromosome onto an S chromosome (or vice
versa). Going backward in time — the way coalescent simulators think
about it — this means a position inside the inversion can occasionally
"flip" its karyotype label, ending up on the other arrangement than
its neighbours.

How often this happens at a given position depends on:

1. The **gene conversion rate** `γ` (a per-bp per-generation rate set
   by the user).
2. The **frequency of the other arrangement** — gene conversion only
   produces a flip when the two chromosomes paired up are different
   karyotypes (an S/I pair); this happens more often when both
   arrangements are common.
3. **Where you are inside the inversion**. Gene conversion tracts are
   short (a few hundred bp) so they almost never reach across the
   inversion's breakpoints. As you move from the centre of the
   inversion outward toward a breakpoint, the chance that a random
   tract covers your position drops to zero.

This last effect is the **`phi(x)` curve** from Peischl et al. (2013)
— a triangular profile that's flat in the middle of the inversion
and tapers down to zero at the two breakpoints. The empirical signal
this predicts: cross-karyotype divergence is **strongest near the
breakpoints** (where gene flux can't reach) and **weakest in the
centre** (where it can). That's exactly what's seen in real
inversions.

## Multiple populations

If you have two or more populations, the karyotype barrier interacts
with the population structure. Two chromosomes can share a recent
ancestor inside an inversion only if they're (a) in the same
population and (b) on the same arrangement. This produces the
characteristic empirical pattern that **cross-population +
cross-karyotype** divergence is much higher than either barrier on its
own.

`msinv` accepts the standard ms-style demography events: population
size changes, exponential growth, migration between populations, and
population merges (going backward, what looks like a merge to the
simulator was a split going forward).

## Multiple inversions

A chromosome can carry several inversions, and they don't have to sit
side-by-side — they can overlap or even nest one inside another. Each
inversion has its own age, frequency, and gene-conversion rate.
`msinv` keeps track of which inversions cover each genomic position
and applies the right karyotype barrier(s) at each one.

By default a sample's S/I label applies to all inversions — useful
when you want to model linked karyotypes (a typical case for tightly-
linked inversions like 3Ra+3Rb in An. funestus). If you want to
model independent inversions you can assign a separate karyotype to
each one per sample.

## Selective sweeps

A selective sweep is a recent positive-selection event that drove a
beneficial allele to high frequency very fast. In coalescent terms,
all chromosomes that carry the swept allele today share a very recent
common ancestor at the selected site.

`msinv` models a sweep as a **forced coalescence**: at a user-
specified position and time, all lineages carrying the swept allele
are merged into a single ancestor. The signal then propagates to
nearby positions via the normal recombination/coalescence machinery
— giving you the classic reduced-diversity valley around the
selected site.

For a sweep that crossed the karyotype barrier (e.g., the RDL
insecticide-resistance allele that arose on the S background and was
later transferred to I via gene conversion in An. gambiae), you can
stack two sweep events with different timings on the two
arrangements.

## References

- Kelleher, J., Etheridge, A. M., & McVean, G. (2016). *Efficient
  coalescent simulation and genealogical analysis for large sample
  sizes.* PLOS Comp Bio 12:e1004842. *(The msprime hull algorithm
  that `HullSimulator` is built on.)*
- Guerrero, R. F., Rousset, F., & Kirkpatrick, M. (2012). *Coalescent
  patterns for chromosomal inversions in divergent populations.*
  Phil Trans R Soc B 367:430–438.
- Peischl, S., Koch, E., Guerrero, R. F., & Kirkpatrick, M. (2013).
  *A sequential coalescent algorithm for chromosomal inversions.*
  Heredity 111:200–209. *(The phi(x) gene-flux model.)*
- Kirkpatrick, M., & Barton, N. (2006). *Chromosome inversions, local
  adaptation and speciation.* Genetics 173:419–434.

## The math, for those who want it

For a position inside an inversion in a population of total effective
size *Ne*, the per-pair coalescence rates per generation are:

- S–S pair (same pop): `1 / (2 · Ne · p_std)`
- I–I pair (same pop): `1 / (2 · Ne · p_inv)`
- S–I pair: 0 (until you reach `t_inv`, after which it's standard)

The Peischl gene-flux rate at inversion-relative position *x* is:

```
flux_rate(x) = γ · phi(x) · p_other
phi(x)       = min(x, 1−x, w) / (1 − w)
```

with *w* the gene-conversion tract width as a fraction of inversion
length (default 5%).

For multi-population scenarios, both the karyotype barrier and the
population structure apply simultaneously: a coalescence requires
same-pop AND same-karyotype, and the cross-pop / cross-karyotype
T_MRCA is bounded below by `max(t_split, t_inv)`.
