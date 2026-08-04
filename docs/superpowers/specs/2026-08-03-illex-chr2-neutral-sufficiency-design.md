# Illex chr2 inversion — neutral-sufficiency test and age estimate (msinv)

Date: 2026-08-03
Amended: 2026-08-04 after Phase A implementation
Status: **Phase A implemented and reviewed** (plan Tasks 1–6, branch
`feature/illex-chr2-neutral-sufficiency`, HEAD `e1e7c62`, 34 tests green,
never merged). Phases B–D not implemented. **Read the Amendments section
next — Phase A falsified this design's central hypothesis and several
numbers below are superseded.**

---

## Amendments (2026-08-04) — read before anything else

Phase A did what it was built to do: it tested the design's assumptions and
several failed. Every claim below supersedes the corresponding text later in
this document. The full evidence trail, including measurements, is in
`.superpowers/sdd/2026-08-03-illex-chr2-neutral-sufficiency/progress.md`
(gitignored — fold it in before deleting it).

### A1. Gene flux is NOT required. The flux hypothesis is withdrawn.

Two independent lines:

- **No spatial signature.** Stage 2's windowed test (implemented as Task 6)
  found dxy/mean(π_AA,π_BB) flat at edge/core = **0.999** across 38
  differentiated windows — and still flat (0.918) under the original nominal
  windowing. Flux via double crossover must produce a gradient. There is none,
  at any magnitude.
- **A zero-flux fit exists.** On the growth arm, interval-restricted, γ = 1e-15:
  t_inv ≈ **7–8 × 10⁵**, p_start ≈ **0.15–0.20** reproduces both target ratios
  (π_I/π_S = 0.667, dxy/π_I = 1.947 at (8e5, 0.15)). Its **unfitted** held-out
  Fst is **0.358** against the observed 0.3652 — the strongest single result
  Phase A produced, precisely because nothing tuned it.

### A2. The origin model is the second parameter, and it is a continuum.

`InversionSpec` accepts `trajectory={'type':'deterministic', ..., 'p_start':…}`.
`p_start` is the founding frequency (k founders / 2N):

| p_start | model | π_I/π_S |
|---|---|---|
| 1/(2N) | hard sweep, k = 1 | 0.22–0.35 across all t_inv |
| intermediate | soft sweep from standing variation | spans the observed 0.744 |
| → p_inv | constant / multi-background, k → ∞ | ≥ 1.0 by construction |

**Both extremes are excluded by the data.** k = 1 never comes within 2× of
π_I/π_S = 0.744 anywhere in t_inv ∈ [2e5, 1.34e6]; the constant limit cannot go
below 1.0. This is a genuine discrimination, and it reverses — for Illex — the
Anopheles-derived preference for the soft/constant model recorded in
`project_inversion_origin_models.md`.

**Caveat that must travel with any reported result:** p_start ≈ 0.15 is a
**phenomenological** founding frequency, not a mechanistic count of founding
haplotypes. Reaching it required relaxing this design's own premise that an
inversion arises on a single chromosome.

### A3. The dxy/π_I "floor" was correct — the premise moved.

An earlier revision of this document claimed the flux requirement was an
artifact of `illex/theory.py` encoding a model msinv cannot express. **That
framing was wrong.** msinv *can* express single-founder origin (`p_start =
1/(2N)`), and `theory.py`'s constant-`p_i` simplification biases E[T_I]
*upward*, making its floors (2.563 growth / 3.978 constant) **conservative
lower bounds** on the true single-founder floor — which msinv respects (hard
limit gives 4.75–5.33). So the 1.39× shortfall was a **correct exclusion of
strict single-origin monophyly**, not a bookkeeping error. What dissolved the
flux requirement was relaxing the origin premise (A2), which is a modelling
choice with a cost, not a correction.

**Consequence:** `theory.py`'s floors must NOT be used as an msinv acceptance
criterion — msinv's trajectory family is strictly larger than the model
`theory.py` implements. The module now carries that warning in-code.

### A4. Statistics must be restricted to the inversion body.

`illex/model.py` places the inversion at `[0.1L, 0.9L]`, so **20% of each
simulated sequence is collinear flank**, which is panmictic and drags both
ratios toward the null. Measured at one configuration: whole-sequence
0.772 / 1.887 versus interval-restricted 0.701 / 2.281 — **dxy/π_I understated
by 21%, Fst by 16%.** The empirical targets are measured over the inversion
body, so unwindowed simulated statistics are not comparable to them.
`stats.arrangement_stats` now takes a **required** `interval=` keyword.

Any simulated ratio produced before this fix (including the pilot ladder's
`pi_i_over_pi_s` / `dxy_over_pi_i` columns) is whole-sequence and diluted.

### A5. Three normalisations exist; do not conflate them.

- **dxy/π_I** = dxy / π(AA) — the **fitted** target, = 1.846.
- **dxy/mean(π_AA, π_BB)** — the correct baseline for the *windowed spatial*
  analysis, ≈ 1.598. Correct for that purpose; not comparable to 1.846.
- **pooled π** (diversity over the combined AA+BB sample) — **rejected**: it
  contains the between-arrangement differences that constitute dxy, so
  dividing by it partly divides dxy by itself.

Canonical constants now live in `illex/empirical.py`.

### A6. Parameters, per user decision (2026-08-04)

The fit is over **three** parameters — (t_inv, p_start, γ) — with γ retained so
the flux hypothesis stays formally testable rather than set aside on the
indirect evidence in A1. Three parameters against two primary ratios is
**under-determined**: expect a ridge, not a point. Therefore:

- Report **γ as a bound** ("consistent with 0, bounded above by X"), never a
  point estimate off an under-determined ridge.
- The **held-out statistics become load-bearing for identification**, not
  merely validation. State which one breaks the degeneracy.

### A7. Feasibility is much less binding than projected.

The Stage 0 pilot cleared both gates with headroom and never found a wall: the
growth arm ran L = 73 kb (22 s / 0.91 GB) and the constant arm L = 645 kb
(29 s / 1.07 GB) at ρ = 5000, all 20 rungs `ok`. The trajectory path costs
≈1.35–1.4× the legacy path on growth, ≈1.0× on constant. **The true ceiling
remains unknown** — ρ = 5000 was the top rung, and the full 20 Mb inversion
would need ρ ≈ 110,000, i.e. 22× beyond anything benchmarked.

### A8. L-invariance: supported on one arm, unverified on the other.

The rescaling premise (per-site π and dxy depend on Ne, µ, t_inv and not on L)
is **supported on the constant arm** (no trend across a 25× L range) but
**unverified on the growth arm** — the arm required for the fitted statistics.
The pilot's single replicate per rung cannot separate low-L Monte Carlo noise
from a small systematic bias. **Follow-up required before production fitting:**
multiple replicates at 2–3 fixed L values on the growth arm, with CIs.

### A9. Best-fit parameters do not transfer between demographic arms.

At identical (t_inv, p_start) the two arms give materially different ratios. Any
fitted value is arm-specific. Since the fitted statistics must carry the
expansion, **Phase C fits on the growth arm** — and the growth-arm age is
≈ 750–800 ky, **not** the 500 ky obtained on the constant arm.

### A10. The Phase D r² comparison is blocked.

Only **5 of 40** control windows density-match the inversion body, and they are
contiguous within a single ~2.5 Mb span. That span cannot contain marker pairs
at the ~20 Mb separations probed inside the inversion, so it cannot produce an
r²-versus-distance curve at comparable scale. This is the wrong *shape* of
evidence, not merely underpowered — the density distribution is bimodal, so the
count is threshold-insensitive. **A differently located or substantially larger
control region must be chosen before that comparison is attempted.**

### A11. Corrections to specific factual claims in this document

- **SNP density inside the inversion is ≈ 68.6 k/Mb**, not the "95–130 k/Mb"
  asserted later in this document. That figure contradicts this document's own
  SNP count (1,372,654 over 20 Mb) and the committed windowed CSV (range
  33–107 k/Mb, mean 68.6 k). Use the CSV, never the prose.
- **A genuinely neutral single-founder trajectory is not samplable at Illex Ne.**
  `StochasticTrajectory` and `BridgeStochasticTrajectory` are limited to
  N ≲ 10⁴. The deterministic trajectory used instead implies a weak positive
  s ≈ 1.1e-5, so the single-founder arm is **neutral in form, not strictly
  neutral**. This connects to Stage 1: P(a neutral mutation reaching p = 0.626)
  ≈ 1.1e-7, so "neutral single-founder at 0.626" may be intrinsically
  implausible.

### A12. Implementation facts the plan got wrong

Both would have silently corrupted results; recorded here so they are not
reintroduced.

- **msinv emits STANDARD samples first, inverted last** (node IDs
  `0..n_std-1` = S). The plan asserted the opposite. Undetected, this swaps both
  arrangements in every statistic and would have inverted π_I/π_S from 0.744 to
  1.344.
- **Real msinv API:** `Demography(pop_sizes: list[float], migration_matrix=None)`
  plus `add_event((kind, time, pop, val))` as a single positional tuple,
  auto-sorted. There is no `sort_events()`. Also `Demography.size_at()` is a
  live stateful query — replay with `apply_event_at(t, [])` first.
- **tskit 1.0.2 shape asymmetry:** `ts.diversity([one_set], mode="branch")`
  returns a length-1 array; `ts.divergence([a, b], mode="branch")` returns a
  bare scalar.
- **pg_gpu bug:** `windowed_analysis` silently returns π for `populations[0]`
  only when π is requested together with `dxy`/`fst`. Fails silently rather than
  erroring. Use separate per-population calls.

---

## Goal

Three questions about the *Illex illecebrosus* chr2:60–80 Mb inversion:

1. Can the inversion persist at its observed frequency **without selection**?
2. How old is it (t_inv)?
3. Can a neutral model reproduce the observed diversity/divergence pattern?

Scope is deliberately **neutral-only**. No sweep is modelled. This matters: the
msinv sweep stack has unresolved residuals at HEAD (TBL −12.6%/−22.6%/−13.4%
hard/soft/recurrent, num_sites −13.4%/−23.3%/−13.9%, singleton fraction
+41%/−54%/+51% — see `project_sweep_bugs_2026_05_12`). Those are confirmed
sweep-specific; the neutral path agrees with msprime and discoal within 2% TBL.
Adding selection to this project requires fixing them first.

## Empirical facts

Established from the data on 2026-08-03. Where a published methods document
disagrees, the discrepancy is recorded rather than silently resolved.

### Karyotypes

`analysis/steps/03_karyotype/karyotypes.baker.tsv`, 633 samples:

| | count |
|---|---|
| AA | 254 |
| AB | 284 |
| BB | 95 |

Allele frequencies: **A = 0.626, B = 0.374**. `karyotypes.baker.tsv` is a clean
633-sample subset of `karyotypes.tsv` (100% concordant). GMM reclustering on
depth-equalised 1× BAMs (`karyotypes_downsampled.tsv`) agrees at **95.3%**
(603/633) — this validates the *calls*, not the frequency.

Polarization (`polarize_arrangements.py`, est-sfs + coindetii outgroup):
**AA = derived, BB = ancestral**, moderate confidence (~54% of diagnostic
sites). So the inverted arrangement is A, at p_inv = 0.626. Both polarities are
carried as a sensitivity pair.

Diagnostic sites: 18,595 positions spanning **60,040,617–79,995,597**
(~19.95 Mb). chr2 length 119,466,599 bp.

### Demography and rates

From `analysis/steps/08_demography/RESULTS_demography.md`:

| | value |
|---|---|
| µ | 3e-9 /site/gen, generation time 1 yr |
| genome-wide π | 0.00930 |
| θ_W | 0.03161 |
| Tajima's D | −2.07 (100% of 326,470 windows negative) |
| best model | exponential growth, Ne 547,928 → 6,808,096 (×12.4) over 769,519 yr |
| *I. illecebrosus* / *I. argentinus* split | 1.6–3.2 Mya |

**Recombination: there is no chr2 map.** ReLERNN covers chromosomes 1, 3–41,
43–45 — chr2 and chr42 are both absent (chr2 was excluded because of the
inversion, and from the demography's accessible L too). Proxy used:
sex-averaged autosomal mean **r = 2.5e-9 /bp/gen** (male 2.13e-9, female
2.90e-9; per-chromosome means span only 0.198–0.294 cM/Mb, so a proxy is
defensible). Sensitivity: bracket male/female.

### Per-arrangement diversity (computed 2026-08-03)

`variants_filt.vcf.gz`, SNP-only, biallelic, 349 samples (254 AA + 95 BB),
pg_gpu, `missing_data='include'`, no MAF filter. Script:
`docs/superpowers/specs/illex_chr2_derivation/per_arrangement_stats.py`.

| | inversion 60–80 Mb | control 10–30 Mb |
|---|---|---|
| SNPs | 1,372,654 | 3,468,696 |
| π(AA) | 0.001288 | 0.004324 |
| π(BB) | 0.001732 | 0.004374 |
| dxy(AA,BB) | 0.002379 | 0.004364 |
| Fst(AA,BB) | 0.3652 | 0.0035 |
| π_AA/π_BB | 0.744 | 0.989 |
| dxy/π_AA | 1.846 | 1.009 |

Per-variant: π(BB) = 0.0252 in **both** regions; π(AA) falls to 0.0188 inside
the inversion (−25%); dxy rises to 0.0347 (+38%). The ancestral arrangement is
unaffected, the derived one is depleted, between-arrangement divergence is
elevated.

**The control region is the key internal control.** Outside the inversion AA and
BB individuals are indistinguishable (Fst 0.0035, dxy ≈ π, π ratio 0.989),
which rules out differential coverage or missingness between the two sample
sets as a driver of the inversion result. It also confirms Fst = 0.315–0.365 is
karyotype-local, consistent with a geographically panmictic range.

**Absolute per-bp values are lower bounds.** No chr2 accessibility mask exists
(`acc_aut.bed`, `inaccessible.bed`, `genome.bed` all omit chr2), so the
denominator is the full region span. Control π = 0.00434 vs the ANGSD chr2
π = 0.00907 implies ~2.1× deflation. **All primary targets are therefore
ratios**, which are denominator-free.

## Theory: the correct null

An inversion arises on a **single chromosome**, so backward in time every
inverted lineage must coalesce by t_inv. With τ_I = 2·Ne·p_I,
τ_S = 2·Ne·p_S, under constant Ne:

```
E[T_I]       = τ_I·(1 − exp(−t/τ_I))                       # bounded by t_inv
E[T_S]       = τ_S·(1 − exp(−t/τ_S)) + 2Ne·exp(−t/τ_S)
E[T_between] = t + 2Ne
```

(The E[T_S] integral contributes −t·exp(−t/τ_S), which cancels the +t carried in
by survivors entering the ancestral population. An earlier draft double-counted
that t and consequently overestimated t_inv.)

Under the moments exponential-growth model these are evaluated by numerical
integration of the hazard 1/(2·N(t)·p) instead — see
`docs/superpowers/specs/illex_chr2_derivation/demog_null.py`. Results for both demographies:

| | growth (moments) | constant Ne = 775 k |
|---|---|---|
| panmictic E[T] | 1,560,600 (π = 0.00936) | 1,549,900 (π = 0.00930) |
| t_inv from π_I/π_S = 0.744 | **952,984** | 896,340 |
| dxy/π_I at that t_inv | **2.596** | 4.181 |
| dxy/π_I floor | **2.563** @ t_inv 1.14 M | 3.978 @ 1.34 M |

Three consequences:

1. **π_derived < π_ancestral is expected**, not anomalous. An earlier draft used
   the equilibrium subpopulation ratio π_I/π_S ≈ p_I/p_S = 1.674 as the null;
   that ignores the single-origin bottleneck and is wrong. Under the corrected
   null, π_I/π_S = 0.744 implies **t_inv ≈ 0.95 M generations** (growth) or
   0.90 M (constant Ne).

2. **dxy/π_I has a floor** — a genuine minimum of E[T_between]/E[T_I] over
   t_inv, unconditional in both age and Ne, but **conditional on the
   demographic shape**. Young inversions give a *large* ratio, not a small one
   (t_inv = 200 k → 8.07 under growth), because the single-origin bottleneck
   drives π_I → 0 while dxy stays ≈ panmictic. Under growth the floor is
   **2.563**; observed is **1.846**, a shortfall of **1.39×**.

3. ~~Flux is therefore still implied~~ **— SUPERSEDED by amendments A1 and A3.**
   Flux is **not** implied. The 1.39× shortfall was a correct exclusion of
   *strict single-origin monophyly*, not evidence for flux: relaxing the origin
   premise to an intermediate founding frequency (`p_start` ≈ 0.15) reproduces
   both ratios at γ ≈ 0, and Stage 2's windowed test found no spatial flux
   gradient whatsoever (edge/core = 0.999). γ is still fitted, but as a
   co-equal third parameter reported as a **bound** — see A6.
   `project_gene_flux_decoupled` ("γ irrelevant for neutral divergence") turns
   out to hold here after all, for a different reason than it holds in
   Anopheles.

Polarity check: with B as derived, the constant-Ne floor is ≈ 5.4 against an
observed 1.374 — worse, so A-derived fits better. **This check has not been
redone under growth** and must be repeated in stage 3 before being cited.

## Model

### Scaling

Faithful per-bp rates and faithful Ne, **shortened inversion**. This is valid
because per-site π/dxy/heterozygosity depend on Ne, µ, t_inv and not on L, and
r² versus physical distance depends on 4Ne·r·d — so a shorter region yields
correct statistics, merely truncated in range. Background LD decays within
~100 kb, so **L ≥ 300 kb** gives a 3× margin for a collinear control to decay.
Below ~150 kb the contrast is unmeasurable.

**The demography is split by statistic**, because the primary targets and the
LD panel have incompatible requirements.

The neutral null for the *fitted* statistics must carry illex's expansion. π_I
and dxy expectations are demography-dependent, and a neutral inversion under
expansion looks diversity-poor against a constant-N null — which could
masquerade as needing selection or flux. Constant Ne = 775 k is within 0.7% of
the growth model for *panmictic* pairwise coalescence (it was calibrated on π,
which is that mean), but it inflates the dxy/π_I floor from 2.563 to 3.978
because the class-structured statistic samples only the deep phase. That is
exactly the statistic the neutrality claim rests on.

| Statistics | Demography | L |
|---|---|---|
| π_I/π_S, dxy/π_I (**fitted**) | moments exponential growth (N_ANC 547,928 → N0 6,808,096 over T 769,519) | 30–75 kb |
| LD panel (**validation**) | constant Ne = 775,000 | ≥300 kb |

This works because the primary targets are **per-site ratios and need no
length at all**, while growth pushes ρ/bp to 0.0681 and caps L at ~29 kb
(ρ 2000) to ~73 kb (ρ 5000). The LD panel needs ≥300 kb, so it runs under
constant Ne — acceptable because it is validation-only and a ratio, so
demography partially cancels. Recorded limitation: the LD arm is not run under
the growth model, and constant Ne gives Tajima's D ≈ 0 versus the observed
−2.07. No primary target is SFS-based.

**Gene-flux geometry must be held invariant.** `rust/msinv-core/src/phi.rs`
defines φ in inversion-relative coordinates x ∈ [0,1] with
`w = mean_tract_length / inv_length`, so the flux profile is scale-invariant
only if w is fixed. Real w = 2 kb / 20 Mb = **1e-4**. Keeping a biological 2 kb
tract at L = 300 kb would give w = 6.7e-3 and inflate interior flux **67×**.
Pin `mean_tract_length / inv_length = 1e-4` (30 bp at 300 kb), or hold the
interior flux rate γ·w/(1−w) constant and absorb the difference into γ.
(`kir_fol_pilot`'s `MEAN_TRACT_FRAC` exists for exactly this reason — it is a
geometry-preserving parameter, not a biological tract length.)

### Region layout

Two independent sims per parameter point rather than one long one — ρ scales
with total length, so two 300 kb runs cost the same as one 600 kb run while
each stays further from the remnant ratchet.

- **inversion sim**: inversion spanning most of the region with a collinear
  margin either side (also yields the breakpoint-sharpness signal)
- **control sim**: identical Ne, r, µ, L, sample size, no inversion

### Parameters

| Parameter | Value |
|---|---|
| `demography` | **fit arm**: moments exponential growth via `Demography` `eg`, N_ANC 547,928 → N0 6,808,096 over 769,519 gen. **LD arm**: `population_size` = 775,000 constant |
| `recombination_rate` | 2.5e-9 (bracket male 2.13e-9 / female 2.90e-9) |
| mutation rate | 3e-9 |
| `sequence_length` | fit arm 30–75 kb; LD arm ≥300 kb (both pilot-confirmed) |
| `p_inv` | 0.626 and 0.374 (sensitivity pair) |
| `t_inv` | grid spanning 0.2–3.0 M gen; both estimators land near 0.90–0.95 M |
| `gene_conversion_rate` γ | **fitted**, grid |
| `mean_tract_length` | `inv_length × 1e-4` |
| `n_inv` / `n_std` | 100 / 100 sampled lineages (haplotypes) per arrangement |
| trajectory | neutral (`s=0`). Growth arm needs a **time-varying n_e**, which `integer_wf` does not take as a scalar — expected to require the `precomputed` trajectory type with an n_e schedule (the `trajectory_helpers` neutral-walk builders already construct these). **Verify before coding.** |

Sample sizes need not match 254/95 because all primary targets are ratios,
which cancel the ≈1/n bias in r².

### Comparability requirements

1. **Match the empirical SNP filter for LD only.** Empirical LD used biallelic
   + MAF ≥ 0.05 + call rate ≥ 0.5. Simulated SNPs must take the same MAF cut or
   sim r² is diluted by rare variants absent from the data. π and dxy targets
   are computed **without** a MAF filter, matching how the empirical values
   above were produced.
2. **Match the LD estimator.** Empirical is composite genotype r²
   (Rogers–Huff) on unphased diploids. Pair simulated haplotypes into
   pseudo-diploids and use the same pg_gpu path; phased r² runs high.
3. **Statistic mode.** Branch-mode for π and dxy (no mutation noise, cleaner
   t_inv signal); mutation-overlaid for LD.

## Stages

### Stage 0 — ρ-ladder pilot

Two ladders, one per demographic arm, because ρ/bp differs 8.8×:

| Arm | ρ/bp | L at ρ = 200 / 1000 / 2000 / 5000 | Gate |
|---|---|---|---|
| growth (fit) | 0.0681 | 2.9 / 14.7 / 29.4 / 73.4 kb | needs only 30–75 kb |
| constant 775 k (LD) | 7.75e-3 | 25.8 / 129 / 258 / 645 kb | needs ρ ≈ 2,325 for 300 kb |

One rep each; record per-rep wall and peak RSS. Deliverable: largest affordable
L per arm. Precedent: L = 5 Mb + inversion + neutral at Ne ≈ 1e6 cost 67 min/rep
and 25.7 GB; L = 10 Mb was not viable (remnant ratchet, >32 GB at 19 min).

The growth arm is the higher risk: it reaches its required L at a *lower* ρ than
the LD arm, but under a demography whose recent N0 = 6.8 M keeps lineage counts
high, which is the regime that produced the remnant ratchet before.

### Stage 1 — neutral persistence

Neutral (`s=0`) trajectory under the **moments growth n_e schedule**, not
constant Ne, `p_final` ∈ {0.626, 0.374}, over the t_inv grid. Deliverable: the
distribution of p **conditional on still segregating**, and where the observed p
falls within it.

Growth matters here too, and in the direction that helps: drift is faster in the
small ancestral phase, so the expected neutral age at p = 0.626 is below the
constant-Ne Kimura–Ohta value of 2.43 M generations — narrowing the gap with the
~0.95 M diversity-based estimate. Stage 1 produces this properly rather than by
plugging a single Ne into a constant-N formula.

**Do not read the rejection/acceptance rate as the persistence probability.**
Every observed inversion is conditioned on not having been lost, so a low
acceptance rate is expected and is not evidence against neutrality. The valid
test compares observed (p, π_I/π_S, dxy) against the *conditional* neutral
distribution.

### Stage 2 — remaining empirical targets

- **Windowed dxy and per-arrangement π along the inversion.** Decisive flux
  test: φ is zero at breakpoints and flat-maximal in the interior, so flux
  predicts dxy **highest near the breakpoints, lowest mid-inversion**. Absence
  of that dip falsifies the flux interpretation and this design changes again.
  Run early.
- **Density-matched collinear controls.** ~~chr2:10–30 Mb carries ~173 k SNPs/Mb
  against 50–74 k/Mb elsewhere on chr2 and 95–130 k/Mb inside the inversion~~
  **— density figures SUPERSEDED by A11: the inversion body is ≈ 68.6 k/Mb
  (range 33–107 k), measured from the committed windowed CSV, not the ~95–130 k
  asserted here.** The qualitative point stands — chr2:10–30 Mb is the density
  outlier, not the inversion — but see **A10**: matching yields only 5 usable
  control windows spanning 2.5 Mb, which is too short to support the Phase D
  r² comparison at all. A different control region is required.
- ***I. argentinus* presence/absence** (`analysis/steps/08_argentinus`). If
  argentinus lacks the inversion, t_inv < the species split; if shared,
  t_inv > split. Independent hard bracket on the age.
- **chr2 accessibility mask**, or an explicit decision to remain ratio-only.

### Stage 3 — (t_inv, γ) grid fit

Scaled msinv at the pilot's L. Grid over t_inv × γ × both polarization arms.
Fit π_AA/π_BB and dxy/π_AA.

### Stage 4 — validation on held-out statistics

## Acceptance criteria

Two free parameters fitted to two ratios match by construction, so the match
itself is **not** evidence. The test is whether the fitted (t_inv, γ)
reproduces statistics it was not fitted to.

| Role | Statistic | Observed |
|---|---|---|
| Fit | π_AA/π_BB | 0.744 |
| Fit | dxy/π_AA | 1.846 |
| Validate | Fst(AA,BB) | 0.365 |
| Validate | windowed dxy shape | stage-2 deliverable; flux predicts a central dip |
| Validate | control π ratio, Fst | 0.989, 0.0035 |
| Validate | inv:control long-range r² | 3.88 overall; 0.97 within homA |

The LD panel is demoted to validation deliberately. Flat within-inversion LD and
its collapse within a homokaryotype are mechanical consequences of
recombination suppression and are reproduced by any barrier model, neutral or
selected. They confirm the inversion is real; they do not discriminate
neutrality.

Outcome for goal 1/3 is whichever holds: the neutral model reproduces the
held-out set (neutrality sufficient), or no (t_inv, γ) reproduces it
(neutrality insufficient — selection or a more complex history required).

## Harness tests

1. ~~γ = 0 must reproduce the demography-matched floor~~ **— SUPERSEDED by A3.**
   This test was implemented, failed by 154 and 71 SEMs, and the failure was
   correctly diagnosed as a model mismatch rather than a simulator bug:
   `theory.py`'s floors describe a model msinv's trajectory family strictly
   contains, so they are a **lower bound**, not an acceptance criterion.
   Replaced in `tests/illex/test_floor_harness.py` by four tests that *are*
   valid against msinv: monotonicity in `p_start`; bracketing (the hard limit
   gives π_I/π_S < 1 and the soft limit > 1, so the family spans the observed
   0.744); a semantic regression pinning that constant-`p_inv` gives
   E[T_I] > t_inv while the founder limit does not; and an interval-restricted
   regression anchor. All four are green on both demographic arms.
2. **No-inversion run** → π ratio → 1, Fst → 0, reproducing the control null.
3. **msinv ↔ msprime neutral agreement** at matched L and Ne (Track 3 pattern;
   msprime needs `ploidy=1` with 2·N, per repo conventions).

## Risks

| Risk | Response |
|---|---|
| **Flux margin is only 1.39×** | The strongest quantitative claim rests on 1.846 vs a 2.563 floor. N_ANC uncertainty, the r proxy, φ's shape, or polarity could each close that gap. Mandatory robustness arm: recompute the floor across the moments N_ANC CI, the male/female r bracket, and both polarities. If any plausible combination lifts the observation above the floor, the flux conclusion is withdrawn |
| Growth arm hits the remnant ratchet | N0 = 6.8 M keeps lineage counts high. Fall back on the ρ ladder to the largest viable L; per-site targets tolerate small L, so this degrades precision rather than validity |
| Pilot cannot reach 300 kb (LD arm) | Fall back to truncated LD range, or escalate to the `structured-analytic-middle` feature (2–3 sessions, already on the roadmap for exactly this shape) |
| γ needed to fit dxy is implausibly large | Model misspecification — recurrent inversion, or not a single-origin event |
| φ(x) length-dependence mishandled | Harness test 1 catches it |
| Selection later required | Sweep residuals block quantitative work; fix before extending |
| No chr2 recombination map | Autosomal proxy + male/female bracket; absolute ρ uncertain by ~1.4× |

## Canonical call set — resolved

**baker-633 is canonical**: `analysis/steps/03_karyotype/karyotypes.baker.tsv`
with `AA_samples.txt` / `BB_samples.txt` (user decision, 2026-08-03). All
numbers in this document already use it.

The `chr2_inversion_methods.md` heterozygosities (0.083 versus 0.040–0.053) and
cluster sizes (91/325/217) describe a superseded run — those cluster sizes match
none of the three karyotype files (273/284/96, 254/284/95, 224/311/98). This
design does not depend on them, since it uses per-bp π and dxy computed
directly, but **the paper figures must be repointed at baker-633.**
