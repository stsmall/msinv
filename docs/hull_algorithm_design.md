# Hull algorithm design (feature/hull-algorithm)

Per-position ancestral material tracking for msinv. Replaces the
single-tree SMC representation with an msprime-style "hull" algorithm
that tracks each lineage's ancestral genomic material exactly. SMC' is
correct by construction; cross-karyotype barrier and inversion-internal
LD both fall out of the model.

## Why hull instead of SMC

The current main branch uses Option 3: in-inv events are gene-flux
only. That handles the karyotype barrier and gives a *biological*
in-inv LD signal (LD breaks down via gene conversion, modulated by
phi(x)). But it doesn't model:

- Within-karyotype recombination on the ARG itself (S/S homokaryote
  recombines normally; current code only models its average effect via
  the structured-coalescent rate scaling).
- Multi-locus signatures — selection sweeps inside the inversion
  interact with the LD landscape in ways the gene-flux-only model can't
  capture.
- Future selection scenarios with more complex ARG-dependent statistics
  (e.g., haplotype-based scans, selection-LD coupling).

Hull representation makes each of these natural.

## References

- Kelleher, Etheridge, McVean (2016) — msprime original SMC' algorithm
  with ancestral material tracking.
- Wong et al. 2024 (msprime "discrete recombination" paper) — efficient
  hull representation for very large samples.
- **demestats** (Ragsdale 2026, doi: 10.64898/2026.04.09.717519) —
  rate computation library for arbitrary structured-coalescent
  scenarios. Should be the rate engine for the per-event waiting-time
  computation in the hull loop.
- Schiffels & Wakeley (companion paper, 10.64898/2026.02.18.706396) —
  classification of structured-coalescent regimes; informs validation
  in the rare-pulse regime that arises with small γ > 0.

## Data structures

### `Segment`
A linked-list node representing one ancestral genomic interval:

```python
class Segment:
    __slots__ = ('left', 'right', 'node_id', 'prev', 'next')
    left: float          # genomic interval start (inclusive)
    right: float         # genomic interval end (exclusive)
    node_id: int         # tskit node id this segment refers to
    prev: 'Segment'      # for the lineage's segment list
    next: 'Segment'
```

### `Lineage`
A list of segments + class + population:

```python
class Lineage:
    __slots__ = ('head', 'tail', 'branch_class', 'population', 'uid')
    head: Segment
    tail: Segment
    branch_class: str    # 'S' or 'I'
    population: int
    uid: int
```

A lineage's "hull" = leftmost segment.left to rightmost segment.right.
This is what's used for hull-vs-hull intersection tests in fast
implementations.

### Tables (tskit)
Standard tskit `NodeTable`, `EdgeTable`. Edges accumulated as the
simulation runs; finalized into a `TreeSequence` at the end.

## Event types

For each iteration:

1. Compute per-event total rates (using demestats where possible):
   - **Coalescence** between same-(class, pop) lineage pairs
   - **Recombination** per lineage: rate = ∑ over segments of
     (segment_len × effective_r); effective_r depends on inv vs col
     and on class
   - **Gene flux** per in-inv segment of an S or I lineage:
     rate = γ × ∫ phi(x) dx over the segment × p_other_at_pop
   - **Migration** (cross-pop)
   - **Demographic events** (ej, eg, en, eM, em, es, demestats handles
     these via demes graph)

2. Sample waiting time for next event by sum of rates.

3. Pick which event fires (proportional to its rate).

4. Apply the event:

### Coalescence (same class, same pop)
Pick two random lineages of (class, pop). For each genomic position x
where both lineages have ancestral material, their MRCA is now this
new node. Mark a tskit edge for each interval where exactly one of the
two lineages was the parent.

The two lineages merge into one with the union of their segments.
Where both had material, the new lineage becomes the descendant of the
new node.

### Recombination
Pick a lineage and a position x (uniformly over its ancestral
material, weighted by effective recombination rate at each segment).
Split the lineage:
- Lineage A: segments to the left of x
- Lineage B: segments to the right of x

Both inherit the lineage's class and pop. They are now separate
lineages.

### Gene flux (in-inv only)
Pick an in-inv segment of a lineage (weighted by phi(x) integral and
p_other). Split out a small tract [x, x+w] from the lineage:
- Tract lineage: segments overlapping [x, x+w], **class flipped**
- Original: segments outside [x, x+w], class unchanged

The tract is now a separate lineage in the OTHER class. It can
coalesce with that class's lineages until the next gene flux event
flips it back (or until t_inv when classes merge).

### Migration / demographic events
Standard. demestats provides the rate computations for arbitrary demes
graphs.

## Termination

When all ancestral material has coalesced to a single MRCA at every
position. Equivalently: the sum over all active lineages of segment
length equals the chromosome length.

## Output

Build tskit `TreeSequence` from accumulated edges + nodes. Drop
mutations on top via Poisson on edge spans.

## Validation plan

1. **Single-site marginals**: at every position, the marginal tree must
   match `build_structured_tree` (within sampling noise). Especially:
   cross-karyotype T_MRCA inside inv ≥ t_inv.

2. **In-inv LD**: at γ=0, two in-inv sites should have identical trees
   (perfect LD within karyotype). At γ>0, LD decays gradually with
   distance, with breakpoint regions retaining stronger cross-karyotype
   divergence than the centre (matches Peischl phi(x) prediction).

3. **Cross-validation against SLiM**: forward simulation with
   inversions provides ground truth. Compare marginal tree T_MRCA
   distributions at multiple positions, and LD decay curves.

4. **Cross-validation against current Option 3 implementation**:
   single-site marginals should match exactly (they're both correct);
   LD patterns will differ (hull captures within-karyotype recomb that
   Option 3 ignores).

## File layout

```
msinv/
  hull/
    __init__.py
    segment.py        # Segment class
    lineage.py        # Lineage class with segment ops
    tables.py         # tskit Tables wrapper, edge recording
    rates.py          # rate computation (calls demestats)
    events.py         # apply_coalescence, apply_recombination,
                      # apply_gene_flux, apply_migration
    simulator.py      # main event loop
  __init__.py         # expose hull simulator alongside SMC simulator
tests/
  test_hull_marginals.py   # marginal tree equivalence with build_structured_tree
  test_hull_ld.py          # LD decay + phi(x) gradient
  test_hull_slim.py        # against SLiM forward sim
docs/
  hull_algorithm_design.md  (this file)
```

## Implementation order

1. **Phase 1 ✓ — minimal panmictic hull (no class, no pop, no inv)**
   - Verifies tree-sequence output structure (1 tree, 2n-1 nodes, single
     root) for the simplest case. 6 tests pass.
2. **Phase 2 ✓ — class barrier (S/I, t_inv)**
   - Lineages now carry `branch_class` ('S' or 'I') and `population`.
     Coalescence rates split by class with structured-coalescent
     scaling: S at k(k-1)/2 / (p_std·Ne), I at k(k-1)/2 / (p_inv·Ne).
     Cross-class coalescence forbidden until t_inv; at t_inv all
     lineages flip to a single class and continue panmictically.
   - Validates: (1) cross-class T_MRCA ≥ t_inv at every position, (2)
     within-class T_MRCA can be << t_inv, (3) rare class with
     Ne·p_inv coalesces ~10× faster than panmictic. 9 tests pass.
3. **Phase 3 ✓ — gene flux events with class flip**
   - Per-lineage flux rate = γ × p_other × ∫_inv phi(x)·1_{lineage}(x) dx.
     Closed-form ∫phi via the Peischl triangular-roof construction.
   - Event handler: pick lineage by weight, sample event position by
     phi-weighted rejection over the lineage's in-inv material, draw
     tract via b1-uniform construction, split the tract out of the
     lineage and FLIP its class.
   - Validates: (1) γ=0 ⇒ single in-inv tree (perfect LD), (2) γ>0 ⇒
     multiple trees (LD breaks down), (3) γ=0 strictly preserves the
     class barrier at every position; γ>0 produces sub-t_inv cross-class
     MRCAs ONLY at flux-affected positions (most positions still
     respect the barrier), (4) phi(x) gradient gives more breakpoints
     near the centre than near the edges. 21 tests pass.
4. **Phase 4 ✓ — population structure + demography**
   - Hull-native ``Demography`` class (in generations) with per-pop
     sizes, growth rates, migration matrix, and ms-style events
     (en/eN/eg/eG/em/eM/ej).
   - ``HullSimulator`` accepts ``sample_config={(class, pop): n}``
     and a ``Demography`` instance. Per-(class, pop) coalescence
     rates use ``demography.size_at(pop, t)`` so growth-rate-aware.
     Migration adds per-lineage rates ``M[dst][src]``. Demographic
     events fire by advancing time to the next event boundary
     (analogous to t_inv class-barrier handling).
   - Validates: (1) sample_config dispatches samples to correct pops
     (Demography unit tests), (2) cross-pop T_MRCA >= t_split with no
     migration, (3) M > 0 produces sub-t_split cross-pop MRCAs,
     (4) per-pop size scales coal rate (~10× ratio for Ne_big/Ne_small),
     (5) inversion + multi-pop combined respects both t_split and t_inv
     (Kir/Fol-style). 11 tests pass.
   - **demestats integration:** demestats (jthlab/demestats) is now
     installed and can be used as the analytical rate engine for
     validation. Simple inversion/multi-pop scenarios use the direct
     calc; the demestats hook is wired up in ``msinv/hull/rates.py``
     for future arbitrary-demes-graph support.
5. **Phase 5a ✓ — per-segment class (fix of Phase 4 limitation)**
   - ``Segment`` now carries ``branch_class`` (``'S'``, ``'I'``, or
     ``'P'`` for panmictic). Sample initialisation splits each sample
     into outside-inv (``'P'``) and inside-inv (``'S'``/``'I'``) segments.
   - ``Lineage.branch_class`` is now a derived summary that returns
     the common class if homogeneous or ``'mixed'`` otherwise.
   - Coalescence is now per-pair, per-event-type. Each pair has up to
     three event kinds — ``'outside'`` (P-P, panmictic), ``'inside_S'``
     (S-S, structured), ``'inside_I'`` (I-I, structured). When an event
     fires, the pair coalesces ONLY at the matching-class overlap;
     incompatible-class overlap stays on the original lineages
     (``_coalesce_partial``). After ``t_inv`` the simulator falls back
     to the bucket-by-pop panmictic event for efficiency.
   - Validates: (1) sample segments split correctly at inv boundaries,
     (2) cross-class T_MRCA ≥ t_inv inside-inv but free outside-inv,
     (3) within-S T_MRCA ratio inside/outside ≈ p_std (panmictic
     scaling outside, structured inside), (4) tree-sequence well-
     formed. 8 new tests pass (55/55 hull total).
   - SMC vs hull comparison with collinear flanks now matches on all
     metrics (ratios 0.88-1.05, all near theoretical expectations).

5b. **Phase 5b — Multi-inversion + nested inversions** *(future work)*
6. **Phase 6 — Sweep model integration**
7. **Phase 7 — Performance optimization (Cython/C inner loop)**

## Estimated effort

- Phase 1-2: 1-2 days each
- Phase 3-4: 2-3 days each
- Phase 5-7: 1 week+

All phases: ~3-4 weeks of focused work for a complete, validated
replacement of the SMC simulator.
