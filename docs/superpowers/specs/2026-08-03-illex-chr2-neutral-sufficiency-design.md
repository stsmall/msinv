# Illex chr2 inversion — neutral-sufficiency test and age estimate (msinv)

Date: 2026-08-03
Status: design approved, implementation not started

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

3. Flux is therefore still implied, but **the margin no longer supports a
   standalone claim**. At 1.39× it is within reach of N_ANC uncertainty, the
   recombination proxy, φ's shape, or polarity. γ is fitted jointly with t_inv
   and the conclusion carries a robustness arm.
   `project_gene_flux_decoupled` ("γ irrelevant for neutral divergence") is
   treated as scoped to the Anopheles case, not inherited here.

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
- **Density-matched collinear controls.** chr2:10–30 Mb carries ~173 k SNPs/Mb
  against 50–74 k/Mb elsewhere on chr2 and 95–130 k/Mb inside the inversion —
  it is the outlier, not the inversion. Replace with several density-matched
  windows.
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

1. **γ = 0 must reproduce the demography-matched floor**: dxy/π_I ≥ **2.563**
   under growth, ≥ **3.978** under constant Ne = 775 k. Run under *both* arms —
   agreement with two different predicted floors is a far stronger test of
   msinv's per-position class logic than one. If msinv violates either, then
   the derivation in `demog_null.py` or msinv's class handling is wrong.
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
