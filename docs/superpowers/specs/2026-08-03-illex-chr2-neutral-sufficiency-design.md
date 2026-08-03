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
`.tmp/illex_chr2/per_arrangement_stats.py`.

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
τ_S = 2·Ne·p_S:

```
E[T_I]       = τ_I·(1 − exp(−t/τ_I))                       # bounded by t_inv
E[T_S]       = τ_S·(1 − exp(−t/τ_S)) + exp(−t/τ_S)·(t + 2Ne)
E[T_between] = t + 2Ne
```

Two consequences:

1. **π_derived < π_ancestral is expected**, not anomalous. An earlier draft of
   this design used the equilibrium subpopulation ratio π_I/π_S ≈ p_I/p_S =
   1.674 as the null; that ignores the single-origin bottleneck and is wrong.
   Under the corrected null, π_I/π_S = 0.744 implies **t_inv ≈ 1.11 M
   generations** against a neutral expected age of 2.43 M (Kimura–Ohta,
   −4Ne·(p/(1−p))·ln p at Ne = 775 k) — a 2.2× gap, within reach of
   demographic uncertainty.

2. **dxy/π_I has a hard floor of ≈ 4.0** (minimum of (t+2Ne)/E[T_I] over t, at
   t ≈ 0.87·2Ne), for *any* t_inv and *any* Ne. Observed is **1.85** — less
   than half the floor. No age explains this; only exchange between
   arrangements can. **Gene flux is first-order in this system.**

Consequence for parameters: `project_gene_flux_decoupled` ("γ irrelevant for
neutral divergence") does **not** hold here and should be treated as scoped to
the Anopheles case. γ is promoted to a primary fitted parameter.

Polarity check: with B as derived the floor is ≈ 5.4 against an observed 1.374 —
worse. A-derived fits better, independently corroborating the 54% call.

## Model

### Scaling

Faithful per-bp rates and faithful Ne, **shortened inversion**. This is valid
because per-site π/dxy/heterozygosity depend on Ne, µ, t_inv and not on L, and
r² versus physical distance depends on 4Ne·r·d — so a shorter region yields
correct statistics, merely truncated in range. Background LD decays within
~100 kb, so **L ≥ 300 kb** gives a 3× margin for a collinear control to decay.
Below ~150 kb the contrast is unmeasurable.

Constant **Ne = 775,000** = π/(4µ), the size reproducing observed genome-wide π.
The exponential growth is carried as a reduced-L sensitivity run, not the
baseline: ρ/bp = 4·Ne·r, so growth to 6.8 M multiplies ρ by 12.4× and would cap
L at ~30–70 kb. Ratio-based targets are robust to this choice because the
expansion affects both arrangements alike. Accepted cost: constant Ne gives
Tajima's D ≈ 0 rather than the observed −2.07; no primary target is SFS-based.

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
| `population_size` | 775,000 (diploid Ne — msinv convention) |
| `recombination_rate` | 2.5e-9 |
| mutation rate | 3e-9 |
| `sequence_length` | pilot-determined, ≥300 kb |
| `p_inv` | 0.626 and 0.374 (sensitivity pair) |
| `t_inv` | grid, t_inv/2Ne ∈ {0.1, 0.25, 0.5, 1, 2} → 155 k–3.1 M gen |
| `gene_conversion_rate` γ | **fitted**, grid |
| `mean_tract_length` | `inv_length × 1e-4` |
| `n_inv` / `n_std` | 100 / 100 sampled lineages (haplotypes) per arrangement |
| trajectory | `integer_wf`, `s=0` |

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

ρ/bp = 7.75e-3, so L = 26 kb (ρ 200) / 65 kb / 129 kb / 258 kb / 645 kb
(ρ 5000) / 1.29 Mb. One rep each; record per-rep wall and peak RSS.

Deliverable: largest affordable L. Gate: ρ ≈ 2,325 needed for the 300 kb floor.
Precedent: L = 5 Mb + inversion + neutral at Ne ≈ 1e6 cost 67 min/rep and
25.7 GB; L = 10 Mb was not viable (remnant ratchet, >32 GB at 19 min).

### Stage 1 — neutral persistence

`integer_wf`, `s=0`, `p_final` ∈ {0.626, 0.374}, `n_e = 775000`, over the t_inv
grid. Deliverable: the distribution of p **conditional on still segregating**,
and where the observed p falls within it.

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

1. **γ = 0 must reproduce dxy/π_I ≥ 4.0.** If msinv violates the analytic
   floor, either the derivation above or msinv's per-position class logic is
   wrong. Sharp, cheap, and it validates the flux implementation.
2. **No-inversion run** → π ratio → 1, Fst → 0, reproducing the control null.
3. **msinv ↔ msprime neutral agreement** at matched L and Ne (Track 3 pattern;
   msprime needs `ploidy=1` with 2·N, per repo conventions).

## Risks

| Risk | Response |
|---|---|
| Pilot cannot reach 300 kb | Fall back to truncated LD range, or escalate to the `structured-analytic-middle` feature (2–3 sessions, already on the roadmap for exactly this shape) |
| γ needed to fit dxy is implausibly large | Model misspecification — recurrent inversion, or not a single-origin event |
| φ(x) length-dependence mishandled | Harness test 1 catches it |
| Selection later required | Sweep residuals block quantitative work; fix before extending |
| No chr2 recombination map | Autosomal proxy + male/female bracket; absolute ρ uncertain by ~1.4× |

## Unresolved question for the user

Within-region heterozygosity is reported as homA 0.037 / het 0.0704 / homB
0.0605 in `logs/chr2_karyo.log`, but as 0.083 versus 0.040–0.053 in
`chr2_inversion_methods.md`. The methods document's karyotype cluster sizes
(91/325/217) likewise match none of the three karyotype files (273/284/96,
254/284/95, 224/311/98) and appear to describe a superseded clustering run.
**Which run is canonical must be settled before any of this reaches a paper.**
The design above does not depend on these values — it uses per-bp π and dxy
computed directly — but the published figures do.
