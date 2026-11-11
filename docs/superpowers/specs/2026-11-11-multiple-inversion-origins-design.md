# Multiple and recurrent inversion origins — design spec (tier 0 and hull engine)

Status: draft 2026-11-11, for review. Not yet agreed. Builds on the exact
karyotype mode spec (`2026-10-02-exact-karyotype-mode-design.md`, "the
tier spec" below), whose tiers, history description and `origin` argument
it extends.

## 1. Background

### 1.1 Theory

The coalescent theory of inversion polymorphism treats the two arrangements
as two demes in one population: I with size 2N·p(t) and S with size
2N·(1 − p(t)). Gene flux (gene conversion, and double crossovers in
heterokaryotypes) moves short tracts between them and acts like migration
between the demes, with a position-dependent rate that is highest in the
middle of the inversion and vanishes at the breakpoints (Navarro, Betrán,
Barbadilla & Ruiz 1997; Navarro, Barbadilla & Ruiz 2000; Andolfatto,
Depaulis & Navarro 2001; Guerrero, Rousset & Kirkpatrick 2012; Peischl,
Koch, Guerrero & Kirkpatrick 2013 for the φ(x) profile). With a single
origin, the I deme is one chromosome at the inversion's age t_inv, so any
I material still in I at t_inv must coalesce into that founder. Flux lets
material leave I earlier through S.

Some inversions arose more than once at the same breakpoints, usually by
non-allelic homologous recombination between inverted repeats (for
example the recurrent human inversions of Porubsky et al. 2022). This spec
adds such origins to the same framework.

### 1.2 What msinv does now

| Build | Flux | Origin at t_inv | Recurrent origins |
|---|---|---|---|
| Original SMC engine (branch `legacy/smc-engine`, `msinv/simulator.py`) | migration-like class switch, rate c·ρ/2·p_other(t)·φ(x) per lineage | none: at t_inv all lineages become S | reflecting boundary at p = 0 in the stochastic trajectory (`StochasticTrajectory`, "recurrent origins at same breakpoints"); t_inv = first time p reaches 1/(2N) going back. Changes the frequency path only; no lineage moves I → S at an origin. |
| Tier 1, hull engine (current default) | tract flux, `apply_gene_flux` | barrier only: tags cleared at `t_inv_max()` (`cross_barriers_static`) | none |
| Tier 0, `msinv.expect` (branch `design/exact-karyotype-mode`) | site switching γ·C(y)·f_other | `origin="barrier"` or `"single"` (`_chain.py::_inv_image`) | none |
| Tier 2, exact mode (planned) | Wiuf & Hein tract initiation | barrier or single (tier spec §4.6) | none |

## 2. Model

### 2.1 Assumptions

1. Every origin uses the same breakpoints, so all I chromosomes are
   collinear and recombine normally with each other. Origins with
   different breakpoints are different inversions (step 3 of the tier
   spec, multiple inversions), not covered here.
2. I chromosomes are exchangeable within a population regardless of which
   origin they descend from (no fitness differences between origins).
   Because I-I crossing over is normal, origin identity is a mosaic along
   the chromosome, so no per-origin sub-deme is needed.
3. Random mating within populations; the frequency path f_d(I, t) is an
   input, as in the tier spec §3.2.

### 2.2 Discrete origins

The user gives origin times t_1 < t_2 < … < t_K = t_inv and, for each, the
population d_k where it happened. The oldest, t_inv, keeps its current
meaning set by `origin` (`"single"`: founder collapse; `"barrier"`: drop
the inversion's entry). Each younger origin is one S chromosome in d_k
becoming I.

Backward rule at a younger origin t_k. Let c be the number of lineages in
state I in population d_k (in tier 1: lineages carrying I-tagged material
at this inversion). Exactly one of the 2N_{d_k}·f_{d_k}(I, t_k) I
chromosomes is the new one. Distinct lineages are distinct chromosomes,
so with probability c / (2N·f) one lineage, chosen uniformly among the c,
is that chromosome, and its state at this inversion becomes S: all of it,
flanks included, because one generation earlier it was an S chromosome.
Otherwise nothing happens. Requires 2N·f ≥ c, which holds for any path
that can carry c I lineages; violations raise.

### 2.3 Recurrent origins at a rate

New inversions arise at rate μ_inv per S chromosome per generation (per
population, optionally time-dependent). Forward, about 2N·(1 − f)·μ_inv
new I chromosomes appear per generation in a population, so backward each
I lineage moves to S at hazard

    h_origin(t) = μ_inv · (1 − f_d(I, t)) / f_d(I, t),

the whole lineage at once. This has the same form as the migration hazard
of the tier spec (rule 3) and as the flux switching rate. Flux moves a
tract; an origin moves the whole chromosome.

Reversion (I back to S at rate ν, by the same mechanism) is the mirror
case: backward S → I at ν · f / (1 − f). Listed as an extension (§6), not
in v1.

### 2.4 Limits and the meaning of t_inv

- μ_inv = 0 and no younger origins: the current model, unchanged.
- As f_d(I, t) → 0 every hazard out of the I cell grows like 1/f
  (coalescence C(c,2)/(2N f), origin exit c·μ_inv(1 − f)/f, migration),
  so the outcome is decided by the rescaled chain, as at the tier spec's
  zero-support boundaries (§3.5, §4.4). For two I lineages it coalesces
  with probability 1 / (1 + 4N·μ_inv) before either exits. With
  θ_inv = 4N·μ_inv ≪ 1 the single-origin collapse is recovered; with
  θ_inv ≫ 1 the I lineages leave separately, as under `"barrier"`. The
  recurrent-origin model interpolates between the two existing settings.
- A frequency path whose support has an interior zero (p reaches 0 and
  later, going forward, rises again, as with the legacy reflecting
  boundary) has an origin at that zero by construction. With μ_inv > 0 the
  rescaled chain handles it. With μ_inv = 0 it is read as a discrete
  origin: the remaining I lineages collapse by the rescaled coalescence
  chain and the last one moves to S. Today tier 0 raises there
  (impossible history) and tier 1 has no special handling.
- Under recurrent origins, `t_inv` (the oldest origin) is when the
  inversion first existed. The time of the most recent common ancestor of
  the I material can be much younger and is the quantity the data
  constrain.

### 2.5 Trajectories

The inflow 2N·(1 − f)·μ_inv belongs to the frequency dynamics too. For the
built-in trajectory builders (deterministic, stochastic, coupled), add
μ_inv as a mutation term in the forward update (p ← p + μ_inv(1 − p)
before selection and drift), replacing the legacy reflecting boundary.
Precomputed paths (for example the Illex selection models) are the user's
responsibility; msinv does not check them against μ_inv.

## 3. Tier 0 (`msinv.expect`)

### 3.1 Interface

`Model(..., origin="barrier" | "single", younger_origins=[(t, pop), ...],
inversion_mutation_rate=None | float | {pop: float})`. Names to be agreed
(§7). Both default to off; outputs are then bit-identical to the current
code.

### 3.2 Chain changes

- Recurrent origins add a per-lineage I → S switching rate
  μ_inv·(1 − f)/f, the same at every site (inside the inversion and in the
  flanks, since the whole chromosome changes state). Current switching is
  symmetric in form (rate × f_other); this term is one-directional, so the
  chain builder needs a direction-specific switching rate. The
  `switching` flag becomes true when either flux or μ_inv is positive.
- Discrete origins are a new kind of instantaneous map at t_k: a
  stochastic map (each state goes to a distribution over states), unlike
  the existing deterministic maps for joins and t_inv (`_maps`,
  `BT`). In the block-counting chain, a state with c I lineages in d_k
  moves to "one of them is now S" with probability c/(2N f), split over
  the lineage types in proportion to their counts. t_k becomes a forced
  solver break, like t_inv.
- Zero-support boundaries: the rescaled limit chain (tier spec §3.5)
  gains the exit hazard c·μ_inv (rescaled). Interior zeros with μ_inv = 0
  follow §2.4.
- Pairwise times, SFS and f1 need no other change.

## 4. Hull engine

### 4.1 Tier 1 (current default engine, Rust; segment mode)

- Discrete origin: a deterministic boundary at t_k (the mechanism used for
  sweep onset and t_inv in `apply_boundary`). Count the lineages in d_k
  carrying I-tagged material at the inversion; with probability c/(2N f)
  pick one uniformly and retag its I segments of this inversion as S. In
  segment mode flank material is untagged, so only body segments change.
- Recurrent origins: a new event type in the main loop with total rate
  Σ over I-carrying lineages of μ_inv·(1 − f)/f, evaluated at the event
  time and held to the next event, as tier 1 already does for coalescence
  and flux rates. The rate is a count of I-carrying lineages per
  (population, inversion) times one number, kept current in the same
  places as the flux rates (`flux_lineage_rate_arena`). The event retags
  the chosen lineage's I segments as S.
- With μ_inv = 0 and no younger origins, no extra random draws are made,
  so seeded output is unchanged.
- Event log: record each origin switch (lineage, time, population).
- Python hull engine: proposal is Rust first, Python raising
  `NotImplementedError` (as for sweeps), with parity added later (§7).

### 4.2 Tier 2 (exact mode, when built)

The same two rules on the lineage-level state: a discrete origin switches
the chosen lineage's state from I to S at t_k; recurrent origins are an
integrated hazard (tier spec §4.4) of μ_inv·(1 − f)/f per I lineage,
diverging as f → 0 like migration and handled by the same machinery. The
single-origin merge (§4.6 of the tier spec) applies only at the oldest
origin.

## 5. Validation

1. Off by default: μ_inv = 0 and no younger origins reproduce current
   outputs exactly (tier 0 bit-identical; tier 1 seeded output identical).
2. Tier 0, independent checks:
   - two I lineages inside the inversion, constant f, no flux, no
     recombination: P(coalesce in I before an exit) = 1/(1 + 4N·μ_inv·(1 − f));
   - msprime: constant f, no flux, a site inside the inversion: two demes
     of sizes N·f (I) and N·(1 − f) (S), backward migration I → S at
     μ_inv(1 − f)/f, and a mass migration I → S at t_inv (barrier). Means
     and survival functions of pairwise times must match.
3. Discrete origins: tier 0 vs a direct Monte Carlo of the rule in §2.2
   for n = 2–4.
4. Tier 1 vs tier 0: means inside the inversion with equal frequencies in
   all populations and `origin="barrier"` (the tier spec's consistency
   contract), now with μ_inv > 0 and with a younger origin.
5. Limits: θ_inv → 0 approaches `origin="single"`; θ_inv → ∞ approaches
   `origin="barrier"` (tier 0).
6. Forward check (later): SLiM with new inversions introduced on random S
   chromosomes at rate μ_inv, compared on I-S dxy and the I TMRCA.

## 6. Out of scope for v1

- Origins with different breakpoints (separate inversions; step 3).
- Fitness differences between origins (I chromosomes not exchangeable;
  needs each origin's own frequency path).
- Reversion I → S (ν); the backward S → I rule is in §2.3 if wanted.

## 7. Decisions needed

1. Parameter names and placement: on `InversionSpec` (shared by all
   tiers) or on the model/simulator.
2. μ_inv per population, and constant or piecewise in time.
3. Whether the trajectory builders gain the μ_inv term now (§2.5), and
   whether the legacy reflecting boundary should be offered as a
   compatibility option.
4. Python hull engine: Rust first with `NotImplementedError` in Python, or
   both together.
5. Order: tier 0 first (fast checks against msprime and closed forms),
   then tier 1.

## 8. References

- Navarro A, Betrán E, Barbadilla A, Ruiz A (1997) Recombination and gene
  flux caused by gene conversion and crossing over in inversion
  heterokaryotypes. Genetics.
- Navarro A, Barbadilla A, Ruiz A (2000) Effect of inversion polymorphism
  on the neutral nucleotide variability of linked chromosomal regions in
  Drosophila. Genetics.
- Andolfatto P, Depaulis F, Navarro A (2001) Inversion polymorphisms and
  nucleotide variability in Drosophila. Genetical Research.
- Guerrero RF, Rousset F, Kirkpatrick M (2012) Coalescent patterns for
  chromosomal inversions in divergent populations. Phil Trans R Soc B.
  doi:10.1098/rstb.2011.0246 (PDF: `../rstb20110246.pdf`). Analytical
  pairwise (two-sample) structured coalescent for an inversion with gene
  flux.
- Peischl S, Koch E, Guerrero RF, Kirkpatrick M (2013) A sequential
  coalescent algorithm for chromosomal inversions. Heredity 111:200–209
  (PDF: `../hdy201338.pdf`). Extends the sequential Markovian coalescent
  to inversions with gene flux and checks it against the Guerrero et al.
  (2012) pairwise expectations; source of the φ(x) profile.
- Porubsky D et al. (2022) Recurrent inversion polymorphisms in humans
  associate with genetic instability and genomic disorders. Cell.

The two PDF entries were checked against the files; the other journal
details are from memory and should be checked before citation.
