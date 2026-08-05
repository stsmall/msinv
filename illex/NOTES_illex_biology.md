# Illex illecebrosus chr2 inversion — consolidated biology and inference notes

Everything established about the system, in one place, with provenance and the
level of confidence attached to each item. Written for the manuscript, so
statements that turned out wrong are recorded as wrong rather than deleted —
several of them were load-bearing for a while.

Companion documents:
- Design/spec with amendments A1–A15:
  `docs/superpowers/specs/2026-08-03-illex-chr2-neutral-sufficiency-design.md`
- Canonical numeric constants (importable): `illex/empirical.py`
- ABC/SLiM pipeline: `illex/slim/README.md`
- **Existing validated SLiM + Talapas harness** (sweep classes; the inversion
  recipe is a sibling of it and copies its conventions):
  `analysis/steps/14_sweep_seqmodel/scripts/harness/` — `slim/{hard,soft}_sweep.slim`,
  `slim/run_slim_sweep.py`, `talapas/config.sh`. Requires **SLiM ≥ 5.0**
  (haplosome/tick API); Talapas account `kernlab`, partition `compute`, conda env
  `illex_slimsim`. Its demography is the same moments growth model, already
  validated to reproduce genome-wide π ≈ 0.0093 and Tajima's D ≈ −2.0.

Status key: **[M]** measured here · **[E]** external/prior analysis ·
**[I]** inferred · **[W]** was believed, now known wrong.

---

## 1. Organismal biology, and why it matters statistically

*Illex illecebrosus* (northern shortfin squid) is an annual, semelparous
broadcast spawner: fecundity ~10⁵ eggs, near-total larval mortality, and
boom-bust recruitment. **[E]**

This is not decoration — it is the profile that motivates a **sweepstakes
reproductive success** model, in which a few individuals contribute
disproportionately to each generation. Sweepstakes reproduction is formalised as
a **multiple-merger (Beta(2−α,α)) coalescent** rather than Kingman's, and it
matters because multiple mergers and population growth are **confounded**: both
inflate rare variants. Had the Illex site-frequency spectrum been driven by
multiple mergers, the inferred ×12.4 expansion would have been an artifact and
every age estimate resting on it would be wrong.

So this was tested rather than assumed. Result in §3.

Life-history consequences that do carry into the modelling:
- **One generation per year**, so generations ≈ years. Ages below are in
  generations and can be read directly as years. **[E]**
- **No overlapping generations**, so a discrete-generation Wright–Fisher forward
  model is a good structural match (relevant to the SLiM design).

---

## 2. Demography

Contemporary Ne agrees across independent data types, which is itself evidence
against multiple mergers (under a Beta coalescent an SFS-derived Ne should be
inflated relative to an LD-derived one):

| Method | Signal | Result | Note |
|---|---|---|---|
| moments | SFS | N_ANC 547,928 → **N0 6,808,096** over 769,519 gen | the adopted history **[E]** |
| GONE2 | LD, recent ~150 gen | ~4.8–5.0 M, flat | **[M]** verified |
| Tajima's D | SFS | −2.07 genome-wide, 100% of windows negative | strong expansion **[E]** |
| momentsLD | LD decay | ranks constant > growth | **do not cite**, see below **[M]** |

**GONE2's flatness is not a contradiction.** Its window is ~150 generations —
about 5,000× shorter than the 769,519-generation growth phase — so it *cannot*
resolve the expansion, and flat is the expected result. Its generation-1 value of
7,132 is the known GONE2 most-recent-generation edge artifact, not a bottleneck.
**[M]**

**momentsLD's constant-Ne preference is a fitting artifact.** Its fitted
parameters sit pinned at ~1e9 against an optimiser bound
(`09_momentsld/momentsld_fit.json`). Do not present it as evidence for constant
Ne. **[M]**

**[W]** I earlier claimed an SFS-vs-LD discordance in contemporary Ne. Withdrawn
— ~5 M (LD) versus 6.8 M (SFS) is agreement, not discordance.

**Mutation rate** µ = 3e-9 per bp per generation. **[E]** This is the weakest
external input in the whole chain: every absolute age scales inversely with it,
so age estimates should be quoted with µ stated.

An alternative history exists at
`analysis/steps/11_relernn/demHist.stairway.txt` (stairway plot) and is worth a
sensitivity arm; not yet run.

---

## 3. The coalescent model is Kingman + growth, not multiple-merger

Test: `illex/scripts/beta_vs_kingman.py`, results
`results/illex/beta_vs_kingman.{json,log}`. msprime only. Compares **normalized
folded SFS shapes** (scale-free, so no θ matching), observed genome-wide spectrum
projected to n = 40, against Kingman-constant, Kingman + moments growth, and
Beta(α) over α ∈ [1.05, 1.99]. **[M]**

| model | L1 shape deviation | singleton fraction (obs 0.4832) |
|---|---|---|
| **Kingman + moments growth** | **0.0356** | 0.4810 (ratio 1.00) |
| Beta, best α = 1.35 | 0.1078 | 0.5155 (ratio 0.94) |
| Kingman constant | 0.5373 | 0.2413 (ratio 2.00) |

Kingman+growth tracks the observed spectrum within 0.93–1.09 across the first ten
bins. **Every α fits worse.** The L1 curve across α is cleanly U-shaped with a
single interior minimum at 1.35 and converges to the Kingman-constant value as
α → 2 (0.5321 vs 0.5373), which validates the machinery rather than just the
result.

The Beta failure mode is **diagnostic**: it overshoots singletons and undershoots
doubletons/tripletons by ~20% — the Λ-coalescent singleton spike with a flattened
tail. The data do not have that shape.

Three things to carry into the write-up:

1. **Do not quote the ΔAIC.** With S = 85 M projected sites the multinomial
   log-likelihoods are ~1.7e8 and differences reach 10⁶. The AIC margin is a
   sample-size artifact. L1 shape deviation is the interpretable statistic.
2. **α̂ = 1.35 is far from Kingman (α = 2), and that strengthens the result.**
   The confounding concern in the literature (e.g. the *P. falciparum* study,
   PMC12871270) bites at α ≈ 1.8, adjacent to Kingman. Illex's Beta optimum is
   nowhere near there and still loses. This is a rejection, not a weak-α
   ambiguity. (It also falsified my own prior prediction of α̂ ≈ 1.85–1.95.)
3. **Branch-mode AFS is required, not a convenience.** The Beta coalescent runs
   on a ~N^(α−1) timescale, so at fixed µ its trees yield ~10 segregating sites
   against Kingman's ~2,500; a mutation-based spectrum is pure noise. Branch mode
   reads the expected spectrum off branch lengths and is insensitive to the
   timescale difference that is not under test.

**Untested and honestly open:** growth **and** mild sweepstakes acting *together*.
α̂ = 1.35 is a genuine multiple-merger signal in absolute terms; this test only
compares the two as alternatives. A joint fit is the correct way to settle it but
the two are confounded in precisely the way that makes it weakly identified.
Stated as a limitation rather than run.

---

## 4. The inversion

| Property | Value | Status |
|---|---|---|
| Location | chr2:60,040,617–79,995,597 (~20 Mb) | **[E]** |
| chr2 length | 119,466,599 bp | **[E]** |
| Discovery | lostruct / local-PCA scan | **[E]** |
| Karyotypes | 633 individuals, GMM-validated ~95.3% | **[E]** |
| Polarization | **AA = derived (inverted), BB = ancestral** | **[E]** moderate confidence, AnchorWave *coindetii*↔*illex* MAF |
| Inverted frequency | **p_inv = 0.626** | **[M]** |
| Karyotype Fst(AA,BB) | **0.3652** | **[M]** |
| Geographic Fst | ~0 | **[E]** |

**The inversion drives the apparent population structure.** Karyotype Fst is
0.365 while geographic Fst is ~0 — the "3-cluster structure" is karyotypic, not
spatial. This is the core of the supergene interpretation and is **untouched** by
anything in §3; it rests on the Fst-with-no-geography contrast, not on the
coalescent.

### 4.1 Corrections to initially-stated parameters **[W]**

Each was traced to its real source. They matter because several would have
changed the modelling:

| Stated | Actual | Status |
|---|---|---|
| Span 2.55–85 Mb | **60.04–80.0 Mb** | **[W]** |
| Ancestral B allele ~54% | **B = 0.374** (so inverted A = 0.626) | **[W]** |
| LD r² = 0.60–0.80 in region | **0.025–0.030** | **[W]** |
| FST = 0.315 | 0.3652 karyotype-based | **[W]** |
| GMM ~95.3% | confirmed | **[E]** |

### 4.2 The differentiated body is narrower than the nominal span **[M]**

The nominal breakpoints sit *inside* the outermost 500 kb windows, and those two
windows have Fst ≈ 0.003–0.006 — indistinguishable from the collinear control
(~0.0035) — while every other window in the region has Fst 0.26–0.51. The
distribution is **sharply bimodal** (0.0055 vs 0.258), so any cutoff in
[0.02, 0.25] gives an identical partition.

Consequence: the outermost windows are collinear flanking sequence, not inversion
body. Use the **empirically differentiated extent** (38 windows), not the nominal
60–80 Mb, for any edge/core or body statistic.

---

## 5. Per-arrangement diversity — the primary evidence

Computed 2026-08-03 from `variants_filt.vcf.gz`, 349 samples (254 AA + 95 BB),
pg_gpu, `missing_data='include'`, no MAF filter. These had never been computed
before. **[M]**

| Statistic | Inversion body | Control (chr2:10–30 Mb) |
|---|---|---|
| π(AA) = π_I | — | 0.004324 |
| π(BB) = π_S | — | 0.004374 |
| **π_I/π_S** | **0.744** | 0.989 |
| **dxy/π_I** | **1.846** | ~1.0 |
| Fst | **0.3652** | 0.0035 |

The control region confirms AA and BB are otherwise exchangeable — Fst ≈ 0,
dxy ≈ π, π ratio ≈ 1 — which rules out a coverage/missingness artifact as the
driver of the inversion-body pattern.

### 5.1 π_derived < π_ancestral is expected, not anomalous **[W→I]**

An early draft used the equilibrium subpopulation ratio π_I/π_S ≈ p_I/p_S = 1.674
as the null. That ignores the single-origin bottleneck and is **wrong**. Under a
single origin the derived arrangement passes through a bottleneck of one (or few)
haplotypes, so π_I < π_S is the *expected direction*, and the ratio is
informative about age: young → small ratio, old → ratio approaching 1.

### 5.2 Three normalisations exist and must not be conflated **[M]**

- **dxy/π_I** = dxy/π(AA) — the fitted target, **1.846**
- **dxy/mean(π_AA, π_BB)** — correct for the windowed spatial analysis, ≈1.598
- **pooled π** over combined AA+BB — **rejected as wrong**: it contains the very
  between-arrangement differences that constitute dxy, so it partly divides dxy
  by itself and damps the signal under test.

### 5.3 Fst is algebraically redundant — it is NOT independent evidence **[M]**

With r = π_I/π_S and d = dxy/π_I:

```
Fst = 1 − ½(π_I + π_S)/dxy = 1 − (r + 1)/(2·d·r)
```

Verified to floating-point identity (max deviation **2.2e-16**) across 600
independent simulations. It holds on the data side too: the published r = 0.744
and d = 1.846 imply Fst = 0.36509 against the measured 0.3652.

**Consequence for the manuscript:** Fst cannot be presented as a held-out
validation of a model fitted to the two ratios, and it cannot break a parameter
degeneracy — any ridge holding r and d fixed holds Fst fixed automatically. An
earlier claim that an "unfitted Fst of 0.358 vs 0.3652" was the strongest result
on the branch is **withdrawn**; nothing tuned it because nothing could, and the
apparent agreement was partly error cancellation (r ran 9.4% low and d 6.5% high,
offsetting in Fst).

---

## 6. Gene flux is absent

Flux via double crossover has a Peischl φ(x) profile that is **zero at the
breakpoints and flat-maximal in the interior**, so it predicts dxy **highest near
breakpoints, lowest mid-inversion**. **[I]**

Measured: dxy/mean(π_AA,π_BB) is **flat**, edge/core = **0.999** across the 38
differentiated windows; raw dxy edge/core 0.847. Under the original nominal
windowing it is still flat and in fact *more* strongly falsifying (0.918 /
0.716). **[M]**

**There is no flux gradient at any magnitude.** Combined with the existence of a
zero-flux parameter point (§7), flux is not required to explain the data and the
flux hypothesis is withdrawn.

Two provenance notes worth keeping: the verdict is threshold-insensitive (the
Fst cut sits in a clean bimodal gap), and a **pg_gpu bug** was found in the
process — `windowed_analysis` silently returns π for `populations[0]` only when π
is requested together with dxy/fst. It fails silently rather than erroring; use
separate per-population calls.

---

## 7. What the coalescent modelling established, and what it did not

Work done with `msinv` (ARG-based, `illex/` package, 34 tests green).

### 7.1 The origin model is a continuum, and both extremes are excluded **[M]**

The founding frequency `p_start` (= k founders / 2N) interpolates between two
models the field usually treats as a binary:

| p_start | model | π_I/π_S |
|---|---|---|
| 1/(2N) | hard sweep, single origin, k = 1 | 0.22–0.35 across all t_inv |
| intermediate | soft sweep from standing variation | spans the observed 0.744 |
| → p_inv | constant / multi-background, k → ∞ | ≥ 1.0 by construction |

**Both extremes are excluded by the data.** k = 1 never comes within 2× of 0.744
anywhere in t_inv ∈ [2e5, 1.34e6]; the constant limit cannot go below 1.0. For
Illex this *reverses* the Anopheles-derived preference for the soft/constant
model.

**Caveat that must travel with the result:** the fitted p_start ≈ 0.15 is a
**phenomenological** founding frequency, not a mechanistic count of founding
haplotypes. Reaching it required relaxing the single-chromosome-origin premise.

### 7.2 Current best point, and it is not yet a fit **[M]**

Growth arm, interval-restricted, γ ≈ 0, t_inv = 8e5, p_start = 0.15:

| | simulated | empirical | miss |
|---|---|---|---|
| π_I/π_S | 0.6743 ± 0.0008 | 0.744 | **−9.4%** |
| dxy/π_I | 1.9664 ± 0.0022 | 1.846 | **+6.5%** |
| Fst | 0.3685 | 0.3652 | +0.9% (redundant, §5.3) |

Close enough to remove flux's *necessity* — the residual is far smaller than the
1.39× that originally motivated flux — but **this is not a fit**, and the two
ratios miss in **opposite directions**, which is the signature of a genuine
model-shape problem rather than a scaling error. Closing it is the ABC's job.

**Age: ~750–800 ky (generations ≈ years).** Arm-specific — the constant-Ne arm
gives ~350–500 ky at the same p_start and does **not** transfer. Since the null
must carry the expansion, the growth-arm value is the one to report. **[M]**

### 7.3 Rescaling is licensed **[M]**

The whole approach simulates a 20 Mb inversion at 30–75 kb. Verified on the
growth arm with 600 sims (5 L × 120 reps): means flat to 0.6–0.8% across a 25× L
range, slopes against log10(L) bracket zero under three estimators that agree,
and **worst-case extrapolated bias to L = 20 Mb is 2.1% (π ratio) and 1.8% (dxy
ratio)** — well inside the ~10% residual being closed.

The earlier apparent trend (|r| ≈ 0.65–0.74) was Monte Carlo noise: replicate SD
falls 4.8× across a 25× L range, i.e. SD ∝ 1/√L to within rounding. Fixed mean,
shrinking variance.

Mechanistic check: the only process that would bias small L is breakpoint leakage
within a recombination escape length 1/(2·Ne·r) = **29.4 bp**, against a 2,350 bp
inversion body at the smallest rung — 80×, hence negligible.

**Caveat:** L-invariance holds for per-site π and dxy. It does **not** hold for
r²-versus-distance, which is explicitly length-dependent.

### 7.4 The identification problem **[M]**

Three parameters (t_inv, p_start, γ) against two ratios is **under-determined** —
expect a ridge, not a point. γ must be reported as a **bound**, never a point
estimate. And per §5.3, Fst cannot break the degeneracy.

This is the single biggest methodological obstacle, and it is what the ABC is
designed to address by adding genuinely independent statistics (§9).

---

## 8. Recombination map and accessibility mask — what actually exists

**[M]** Both exist. An earlier conclusion that neither did was wrong.

**In progress (as of 2026-08-05):** a chr2-specific accessibility mask and
ReLERNN map are being built. Until they land, §8.1/§8.2 describe the proxy and
the genome-wide mask actually in use. Hooks for the chr2 versions are
`illex/slim/config.py::CHR2_RMAP` / `CHR2_MASK_BED` and
`rec_rate_for_inversion()`.

### 8.1 Recombination: ReLERNN, genome-wide, chr2 deliberately excluded

`analysis/steps/11_relernn/run_{male,female}_auto/proj/*.PREDICT.BSCORRECTED.txt`

| | windows | modal window | genome-wide length-weighted mean r |
|---|---|---|---|
| male | 114,328 | 9 kb | **2.148e-9** |
| female | 116,081 | 18 kb | **2.892e-9** |

- **Sex-averaged r = 2.52e-9** — confirms the 2.5e-9 the modelling already used.
- Between-chromosome variation is tight: male per-chromosome IQR
  [2.095, 2.186]e-9, female [2.871, 2.914]e-9 (43 chromosomes).
- **chr2 is absent by design.** `build_persex_vcf.sh` line 3: *"autosomes (excl
  chr2 inv, chr42, chrZ)"* — the inversion's LD block would corrupt a ReLERNN
  fit. chr42 and chrZ likewise excluded.
- **Proxy for chr2:** the six length-matched autosomes (1, 3, 6, 9, 4, 13; 93–105
  Mb) give sex-averaged 2.467–2.594e-9. So r for chr2 is well constrained and is
  effectively **known, not a free parameter** — a tight prior is justified.
- Within-chromosome heterogeneity is real (male window-level 5–95%:
  1.25–2.87e-9) and is available if a heterogeneous map is wanted.

### 8.2 Accessibility: `degenotate_illex/accessible_sites.bed`

**[M]** Includes chr2 (168,250 intervals).

| region | accessible bp | fraction |
|---|---|---|
| chr2 whole | 57,996,480 | 48.55% |
| inversion 60–80 Mb | 9,582,174 | **47.91%** |
| control 10–30 Mb | 12,138,077 | **60.69%** |

Note `analysis/steps/11_relernn/acc_aut.bed` is **not** an Illex mask (different
taxa) and has zero chr2 intervals — do not use it.

### 8.3 The absolute-diversity gap, half explained

Simulated absolute levels exceeded empirical by 4.42–4.84×. Applying the mask
(dividing empirical by 0.4791 in the inversion body) removes about half of it:

| | empirical | per accessible bp | simulated | residual |
|---|---|---|---|---|
| π_AA | 0.001308 | 0.002730 | 0.005785 | 2.12× |
| π_BB | 0.001774 | 0.003703 | 0.008579 | 2.32× |
| dxy | 0.002455 | 0.005124 | 0.011374 | 2.22× |

A residual offset remains **in the collinear control too**: control π per
accessible bp is 0.00713 against the model's panmictic 4·Ne·µ = 0.00930, a factor
**1.31×**. Because it is present where there is no inversion, it is a
**calibration** offset — µ, the Ne derivation, or residual site filtering — not a
failure of the inversion model.

**Consequence for inference:** absolute levels are *nearly* usable, and the right
way to use them is with a **nuisance scale parameter** absorbing the 1.31×, not
by assuming the mask fixes everything. Ratios and normalized shapes remain
calibration-free and stay primary.

---

## 9. Where identification has to come from

Given §5.3 (Fst redundant) and §8.3 (absolute levels need a nuisance parameter),
the genuinely independent constraints are:

| Statistic | Status | Calibration-free? |
|---|---|---|
| π_I/π_S, dxy/π_I | fitted targets | yes |
| Windowed spatial dxy profile | **spent** — used to kill flux (§6) | yes |
| **Within-arrangement folded SFS shape** | **untapped, best candidate** | **yes** |
| Absolute π_I, π_S, dxy | usable with a scale nuisance | no |
| r²-vs-distance decay | **blocked**, see below | yes but length-dependent |
| *I. argentinus* presence/absence | not done — a hard age bracket | n/a |

**The within-arrangement SFS shape is the recommended addition.** It is
normalized, so it needs no accessibility mask, and it responds to t_inv and
p_start *differently* from mean π: a young inversion founded by few haplotypes
leaves a different within-I spectrum than an old one founded by many, even at
matched π_I/π_S. The machinery already exists from §3.

**The r² comparison is blocked.** Only 5 of 40 control windows density-match the
inversion body, and all 5 sit in a single ~2.5 Mb span, which cannot contain
marker pairs at the ~20 Mb separations probed inside the inversion. This is the
wrong *shape* of evidence, not merely underpowered, and the density distribution
is bimodal so the count is threshold-insensitive. **A differently located or
substantially larger control region is required.**

***I. argentinus* is the cheapest big win available.** If argentinus lacks the
inversion, t_inv < the species split; if it shares it, t_inv > split. That is an
independent hard bracket on the age from data that already exists
(`analysis/steps/08_argentinus`).

---

## 10. What is excluded, and what remains open

**Excluded by evidence:**
- Multiple-merger (Beta) coalescent — every α fits worse than Kingman+growth (§3)
- Constant Ne — 2× singleton deficit (§3)
- Gene flux as an explanation of the divergence pattern — no spatial gradient (§6)
- Strict single-founder monophyly (k = 1) — cannot reach π_I/π_S = 0.744 (§7.1)
- Constant-p_inv / multi-background origin (k → ∞) — cannot go below 1.0 (§7.1)

**Open:**
- The −9.4% / +6.5% residual in opposite directions (§7.2) — a model-shape issue
- Selection versus neutrality. Not yet properly tested: a genuinely neutral
  single-founder trajectory is **not samplable** at Illex Ne (msinv's stochastic
  samplers cap at N ≲ 10⁴), and P(a neutral mutation reaching p = 0.626) ≈ 1.1e-7,
  so a forward rejection sampler yields zero survivors. This is exactly what the
  forward SLiM model is for.
- Why p_inv sits at an intermediate 0.626. Overdominance would explain a stable
  intermediate frequency and is directly testable in a forward model.
- Whether growth and mild sweepstakes act together (§3)
- Absolute calibration: the 1.31× control-region offset (§8.3)
- A usable control region for LD (§9)

---

## 11. Operational gotchas worth not rediscovering

- **msinv emits STANDARD samples first, inverted last** (node IDs
  `0..n_std-1` = S). Getting this backwards inverts π_I/π_S from 0.744 to 1.344
  and reverses the conclusion, silently.
- **Statistics must be interval-restricted** to the inversion body. `illex/model.py`
  puts 20% of the sequence in collinear flank; integrating the whole sequence
  understates dxy/π_I by 21% and Fst by 16%.
- **pg_gpu:** `windowed_analysis` silently returns π for `populations[0]` only
  when π is requested together with dxy/fst.
- **Normalise divergence by mean(π_AA, π_BB), never pooled π.**
- **Run illex scripts as modules** (`.venv/bin/python -m illex.scripts.NAME`) —
  the venv's `msinv.pth` still points at the pre-move
  `/home/ssmall/inversion_sims/files`.
- **Generations ≈ years** for this species, so ages need no conversion.
- Every absolute age is inversely proportional to µ = 3e-9; state µ with any age.
