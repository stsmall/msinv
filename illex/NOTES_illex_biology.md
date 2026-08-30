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

### 5.4 The targets now have error bars, and the interval matters **[M]**

`illex/scripts/empirical_jackknife.py`, results in
`results/illex/empirical_jackknife.{txt,csv}`. Delete-one-block jackknife on
100 kb base windows grouped into blocks, 1,372,654 variants × 698 haplotypes
(AA 508, BB 190 chromosomes).

Both targets are **ratios of region-wide sums**, so blocks are deleted from
numerator and denominator together and the ratio recomputed — averaging
per-window ratios is a different and biased estimator. Accessibility cancels
(π_AA, π_BB and dxy share a denominator), so no mask is needed.

The estimator reproduces the historical targets to 4 dp — 0.7439 / 1.8464
against the published 0.744 / 1.846 — which validates it before anything is
read off it.

| block | n | π_I/π_S | dxy/π_I |
|---|---|---|---|
| 250 kb | 80 | 0.7438 ± 0.0181 | 1.8459 ± 0.0389 |
| 500 kb | 40 | 0.7437 ± 0.0219 | 1.8457 ± 0.0455 |
| **1 Mb** | **20** | **0.7437 ± 0.0262** | **1.8455 ± 0.0534** |
| 2 Mb | 10 | 0.7437 ± 0.0297 | 1.8463 ± 0.0478 |
| 4 Mb | 5 | 0.7440 ± 0.0382 | 1.8443 ± 0.0569 |

**1 Mb is the block size to quote.** dxy/π_I's SE plateaus from 1 Mb up
(0.048–0.057); π_I/π_S's keeps climbing, which is expected rather than
alarming — π declines toward the breakpoints, so a real spatial gradient makes
larger blocks differ systematically and the jackknife correctly absorbs that
into the SE. Beyond 1 Mb there are too few blocks (n ≤ 10) for the SE itself to
be stable.

So the targets are known to **≈3%**, not to the four figures they were quoted
to.

#### 5.4.1 The like-for-like target is the differentiated body **[M]**

Restricting to the empirically differentiated extent (Fst > 0.15: 189 of 200
windows, 60.5–79.5 Mb) shifts both targets:

| | nominal 60–80 Mb | differentiated body | shift |
|---|---|---|---|
| π_I/π_S | 0.7439 ± 0.0262 | **0.7368 ± 0.0263** | −0.95% |
| dxy/π_I | 1.8464 ± 0.0534 | **1.8794 ± 0.0503** | **+1.8%** |

The two excluded windows are collinear flanking sequence, so including them
dilutes dxy/π_I downward — the same flank-dilution effect §11 warns about on
the *simulated* side, appearing here on the empirical side. **The simulations
are interval-restricted to the inversion body, so the body values are the
correct targets** and the nominal ones were a mild apples-to-oranges
comparison. Both are now in `illex.empirical`.

### 5.5 A µ-free scale for the age **[M]**

`illex/scripts/mu_free_ratio.py`, results in
`results/illex/mu_free_ratio.{txt,json}`.

The age is ~735 ky ± 19 ky statistical, but it scales as 1/µ and µ = 3e-9 is the
weakest input in the chain (§2, §7.5.4). **No amount of simulation can improve
it.** Both of these quantities are 2µT for some T, so their ratio removes µ,
the accessibility mask and the generation time at once:

```
R = dxy(AA,BB) / div(illecebrosus, coindetii) = (t_inv + T_anc_ill)/(T_split + T_anc_spp)
```

| interval | R (1 Mb blocks) | Jukes-Cantor corrected |
|---|---|---|
| nominal span 60–80 Mb | 0.5019 ± 0.0171 | 0.4985 |
| **differentiated body** (the like-for-like one) | **0.5137 ± 0.0146** | 0.5102 |

**The arrangements' divergence is 51.4% ± 1.5% of the illecebrosus–coindetii
divergence.** The SE plateaus from 250 kb blocks upward, so this is stable.
Per shared bp: dxy(AA,BB) = 0.005146, div(ill,coin) = 0.010019.

Read in the useful direction: given any independent calibration `T_cal` for the
illecebrosus–coindetii split — a fossil, a vicariance date, a published
cephalopod substitution rate — the inversion's age follows with **µ nowhere in
it**:

```
t_inv = R · (T_cal + T_anc_spp) − 2·N_ANC
```

Going the other way, and this direction *is* model-dependent because it needs
the fitted t_inv and the model's ancestral coalescent depth: R implies
T_split(coindetii) ≈ **2.47 M generations**.

#### 5.5.1 Two denominator traps, both of which I fell into **[W]**

Worth recording because either one silently turns a ratio of *times* into a
ratio of *denominators*.

1. **`2.callable.bed` does not mean "aligned".** It means aligned **and
   identical to the reference** — it is perfectly disjoint from `2.snps.vcf.gz`
   (0 of 216,739 SNP positions inside it), which is the layout est-sfs wants.
   Treating it as the aligned span excludes every substitution and drives the
   numerator to exactly zero. The comparable span is
   `callable ∪ SNP positions` = 18,307,768 bp (91.75% of the region). The script
   now asserts the disjointness so this cannot come back quietly.
2. **A first pass got R = 0.227, which was wrong by 2.3×.** dxy came from pg_gpu
   normalised by *nominal window span* while its real denominator is the
   illecebrosus-accessible subset (47.9%); the coindetii rate's real denominator
   is the comparable span (91.8%). Dividing both by nominal span compares two
   different base sets. Fixed by intersecting to
   `accessible ∩ comparable` = 8,849,184 bp and forming the ratio **as a ratio
   of counts**, so no per-base rate is ever built and no accessibility fraction
   can leak in.

Both known biases now run the same way and are small: the illecebrosus callset's
MAF/quality filtering removes some real low-frequency variants (dxy low), and
multiple hits at ~1% divergence cost another 0.7% (JC column). So R is a mild
**under**estimate.

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

### 7.2 The rising-logistic point — superseded by §7.5 **[M]**

**Read §7.5 first.** This section records the best *rising-logistic* fit and its
residual. That residual is now closed by the balancing-selection trajectory, and
the diagnosis below ("a genuine model-shape problem") turned out to be correct.

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

### 7.5 The balancing-selection fit — the residual closes, and the age holds **[M]**

`illex/balancing.py`, `illex/scripts/fit_balancing.py`, results in
`results/illex/fit_balancing.{csv,json}`. 1,728 sims, 96 reps/cell, growth arm,
interval-restricted, L = 37 kb (ρ = 2000), r = 1.977e-9, γ ≈ 0, µ = 3e-9.

**The model.** The inversion rises under overdominance to an *equilibrium* and
is then held there, instead of still rising at the present moment. Fitness
w_II = 1−s_I, w_IS = 1, w_SS = 1−s_het, with p* fixed at the observed 0.626,
which forces s_I = s_het(1−p*)/p*. To first order the dynamics reduce to
dp/dt = (s_het/p*)·p(1−p)(p*−p), integrated in closed form.

**Result.** Both targets are met simultaneously:

| plateau | t_inv | p_start | s_het | residual π_I/π_S | residual dxy/π_I |
|---|---|---|---|---|---|
| 0 | **727,301** | 0.0271 | 3.58e-5 | −0.00% | −0.00% |
| 100,000 | **718,872** | 0.0222 | 4.24e-5 | +0.00% | +0.00% |

against the rising logistic's **−9.4% / +6.5%**. The prediction stated in
advance — that a plateau raises π_I, pushing the first ratio up and the second
down, so both misses close *together* — held.

**The age is identified; the other two parameters are not.** Across the one
dimension that stays degenerate (plateau length 0 → 100,000 generations) the
fitted age moves **1.2%** while p_start moves 22% and s_het 18%. dxy/π_I carries
the age and is steep in it, π_I/π_S carries the founding frequency, and the two
are close enough to orthogonal that the age survives. Local sensitivity:
∂(dxy/π_I)/∂t_inv = 0.283 per 10⁵ generations, so a 1% error in the empirical
target moves the age 6,500 generations and a 5% error moves it 33,000.

**Age ≈ 730,000–740,000 generations ≈ 730–740 ky** (generations = years;
inversely proportional to µ). **[Superseded in scope by §8.6: a family with an
explicit arrival time, fitted to all three statistics, gives ~900 ky. Quote
730–950 ky across families.]** Implied selection s_het ≈ 3.6–4.2e-5,
s_I ≈ 2.1–2.5e-5 — minute per generation but N_e·s ≈ 240 at N₀, i.e. firmly
deterministic.

Refit against the like-for-like body targets of §5.4.1 (0.7368 / 1.8794)
rather than the nominal-span ones, on the same grid:

| plateau | targets | t_inv | p_start |
|---|---|---|---|
| 0 | nominal | 727,396 | 0.0271 |
| 100,000 | nominal | 718,972 | 0.0222 |
| 0 | **body** | **737,003** | 0.0268 |
| 100,000 | **body** | **728,861** | 0.0220 |

#### 7.5.4 Uncertainty budget for the age **[M]**

Ordered by size. The last term dominates and no amount of simulation touches it.

| source | effect on t_inv | note |
|---|---|---|
| **µ = 3e-9** | **∝ 1/µ** | age scales inversely; ±30% on µ is ±220,000 gen. The weakest external input in the chain (§2) |
| measurement (1 Mb jackknife on dxy/π_I, ±0.0534) | **±18,900 gen** | §5.4; the 1σ statistical error |
| interval definition | +9,600 gen | nominal → body; now **resolved** by using the body |
| plateau degeneracy | ±8,100 gen | §7.5, the dimension that stays degenerate |
| model process variance | negligible | per-replicate SD 0.0371 at L = 37 kb, but §7.3 verified SD ∝ 1/√L over a 25× range, so at 20 Mb this is sub-1,000 gen |

Direct check by refitting at d ± 1 SE: 1.8260 → t_inv 703,309;
1.9328 → beyond the grid's 750,000 ceiling, consistent with the ±18,900
linear extrapolation.

**Quote the age as ≈ 730–740 ky, ±19 ky (1σ, statistical), conditional on
µ = 3e-9.** Any statement tighter than that is false precision, and the
µ-conditionality must travel with the number.



**The most important thing here is what did *not* change.** §7.2's
rising-logistic age was 750–800 ky. Correcting a trajectory misspecification
large enough to close a 9.4% residual moved the age by ~4%. The age estimate is
therefore **not an artifact of the trajectory shape**, which is a far stronger
claim than the fit itself.

#### 7.5.1 Two of my predictions failed, and both are informative **[W]**

1. **A fast rise does NOT rescue the single origin.** I predicted that because
   selection shortens the rare phase, `p_start = 1/(2N(t_inv))` — one
   chromosome — would become viable and the phenomenological founding frequency
   could be retired. It cannot. At p_start = 9.1e-7, t_inv = 8e5 gives
   (0.60, 2.71); reaching π_I/π_S = 0.744 needs t_inv ≈ 1.05e6, where dxy/π_I
   ≈ 2.8, over 50% high. Strict monophyly caps π_I at 2µ·t_inv and that cap
   binds. The fitted p_start ≈ 0.025 is 6× smaller than the old 0.15 but still
   ~34,000 founding haplotypes at N(7.2e5) ≈ 6.4e5. **§7.1's caveat stands: the
   origin is soft, and `p_start` remains phenomenological.**
2. **s_het is not identifiable from its magnitude.** Over [1e-4, 1e-2] at fixed
   t_inv the statistics are flat above ~1e-3 — 1e-3 and 1e-2 are
   indistinguishable. Once the rise is fast relative to t_inv only its *timing*
   matters. Quoting s_het as a point estimate without the plateau assumption
   attached would be spurious precision.

#### 7.5.2 π_I/π_S < 1 bounds how long the inversion has been at 0.626 **[M]**

An unanticipated third consequence, which binds harder than either of the two
the model was built around. A **long** plateau also suppresses π_S, because the
standard arrangement is confined to 1−p* = 0.374 of the population for that
whole period — and π_S is the denominator. So a long plateau drives π_I/π_S
*above* 1: the model reaches 1.35 at t_inv = 3e6, and the equilibrium limit is
p*/(1−p*) = 1.674.

**Observed π_I/π_S = 0.744 < 1 is therefore direct evidence that the inversion
has not been at 0.626 for long on a coalescent timescale.** The common
arrangement carrying *less* diversity than the rare one is the signature of a
recent rise, and it is what dates the event.

#### 7.5.3 Neutrality is quantitatively excluded, without µ or a mask **[M]**

> **Superseded by §8.17** (rerun at the corrected frequency), and **one claim
> below is an overstatement**: "excluded by an order of magnitude" quotes the
> largest-Ne case. The *binding* case — the smallest Ne on the arm — gave a
> margin of only **1.26×**, then and now.

`illex.balancing.neutral_hitting_time`. Exact diffusion result, verified against
Wright-Fisher simulation (N = 200/500/1000, 1–2 M replicates: probabilities
matched to 3%, conditional times to 0.7–3.0%, converging as N grows).

Solving (p(1−p)/4N)·T″ = −p/x for the probability-weighted absorption time and
dividing by the hitting probability p/x, as p → 0:

```
E[generations to reach x | it gets there] = (4N/x)·[x + (1−x)·ln(1−x)]
                                          = 1.650 · N     at x = 0.626
```

| N_e | E[t | reaches 0.626] | P(ever reaching 0.626) |
|---|---|---|
| N_ANC = 547,928 | 903,893 | 1.46e-6 |
| N(t = 7.2e5) = 644,381 | 1,063,007 | 1.24e-6 |
| N₀ = 6,808,096 | 11,231,018 | 1.17e-7 |

Against a fitted age of ~7.2e5 generations, **neutral drift cannot deliver the
observed frequency in the time the divergence allows** — at the smallest N_e on
the arm the rise alone would consume more than the whole age. The comparison is
generous to neutrality three times over: it conditions on a ~10⁻⁶ event, uses
the mean rather than a lower quantile, and evaluating at the smallest N_e
understates the timescale since the rise mostly occurs at larger N.

This is the cleanest non-neutrality statement available because it needs no
mutation rate, no accessibility mask and no absolute diversity — only the
observed frequency and the demography.

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

**Partly resolved by §7.5.** Two things collapsed the ridge: γ was falsified
independently by the spatial dxy profile (§6), and the balancing trajectory
makes dxy/π_I steep in t_inv while π_I/π_S carries p_start, so the two are close
to orthogonal. The **age** comes out identified to ~1% across the remaining
degenerate dimension. What is *not* identified is the pair (p_start, s_het) —
they trade off along a ridge, so both must be quoted as a joint range with the
plateau assumption attached, never as point estimates.

---

## 8. Recombination map and accessibility mask — what actually exists

**[M]** Both exist. An earlier conclusion that neither did was wrong.

**Both landed 2026-08-07** and are wired in (`config.CHR2_RMAP`,
`config.CHR2_MASK_BED`). The recombination rate changed as a result; the
accessibility numbers did not. §8.0 is the reading of the maps, §8.1 the adopted
rate, §8.2 the mask.

### 8.0 The chr2 maps landed, and they detect no barrier **[M]**

Four chr2 ReLERNN maps exist (2026-08-07, `analysis/steps/11_relernn`).
Reproduce with `.venv/bin/python -m illex.slim.chr2_rmap_report`.

| map | interior | collinear | **interior/collinear** |
|---|---|---|---|
| autonet **male** — *the simulation's r* | 1.967e-9 | **1.977e-9** | **0.995** |
| autonet female | 2.265e-9 | 2.248e-9 | 1.008 |
| all 633 pooled (own network) — **`forceDiploid` artifact, do not quote** | 5.34e-10 | 4.78e-10 | 1.117 |
| AA homokaryotypes only (own network) — **`forceDiploid` artifact, do not quote** | 6.38e-10 | 7.19e-10 | 0.888 |

Absolute levels are not comparable across maps — the AA and pooled runs trained
their own networks on chr2 subsets, so their calibration differs 3–4× from the
autonet. **The statistic to read is each map's own interior/collinear ratio.**
The spatial profiles are flat in all four; there is no U-shape anywhere.

**[W] My prediction that the interior would read biased-LOW is falsified.** Both
autonet maps — the only two on a valid absolute calibration — show no
suppression (0.995, 1.008) and flat interior profiles.

**[W] And my first reading of *why* leaned on two maps I should not have used.**
I argued that the ordering was incoherent for a barrier because the AA-only map
showed the largest deficit (0.888) while the pooled map showed an excess (1.117).
**The AA and all-samples runs are `--forceDiploid` artifacts** — the project
manuscript §3 records them as ~3× too low in absolute terms and says they must
not be quoted, and a karyotype-stratified rerun without `forceDiploid` is
deferred. Their interior/collinear *ratios* may or may not survive that rerun;
either way they cannot carry an argument. The conclusion is unchanged and rests
on the autonet maps alone, but the supporting argument was built on sand and is
withdrawn.

The remaining explanation is the one that does not need those maps, and it is
sufficient on its own: **scale.**

The barrier suppresses crossovers *between* arrangements across ~20 Mb. Within a
19 kb ReLERNN window, LD decay is governed by *within*-arrangement
recombination, which the barrier does not touch. ReLERNN is measuring the right
quantity — the meiotic rate — and is simply blind to the inversion. Detecting the
barrier requires a long-range LD statistic.

Two consequences:

1. **Good news for the simulation input.** The collinear-vs-interior distinction
   was a precaution against double-counting the barrier; it turns out not to
   matter (0.995), so `r` is robust to the choice of region. `REC_RATE` is still
   taken from the collinear regions on principle.
2. **The interior rate is withdrawn as a validation target** (it was listed in
   §9 as one of the few remaining independent constraints). It carries no
   barrier signal, so a fitted model cannot be checked against it. This *removes*
   an item from the evidence list. `rec_rate_inversion_interior()` is retained as
   a documented diagnostic only.

A third, weaker inference points the same way as everything else: if the two
arrangements were deeply diverged, the pooled sample would carry enough
long-range LD to drag even a 19 kb window's inference down. It does not. That is
consistent with a **young** inversion (d = dxy/π_I = 1.846, i.e. modest
divergence), though it is corroboration, not a measurement.

### 8.1 Recombination: chr2 is measured, and it is lower than the proxy

**Adopted: r = 1.977e-9** — chr2 collinear, **male** map (`config.REC_RATE`).
Male rather than sex-averaged so this pipeline matches the `14_sweep_seqmodel`
campaign, which also runs on the male map. Sex-averaged collinear is 2.113e-9
and the female/male pair (1.977e-9, 2.248e-9) is the sensitivity bracket.

**This supersedes the 2.52e-9 length-matched-autosome proxy, which ran 27% high.**
chr2 recombines *less* than the genome-wide average in both sexes:

| | genome-wide (43 chr) | chr2 collinear | chr2/genome |
|---|---|---|---|
| male | 2.148e-9 | **1.977e-9** | 0.92 |
| female | 2.892e-9 | 2.248e-9 | 0.78 |
| sex-averaged | 2.52e-9 | 2.113e-9 | 0.84 |

The old proxy reasoning ("six length-matched autosomes give 2.467–2.594e-9, so
chr2 is effectively known") was sound in method but landed 27% off, because chr2
is not an average chromosome. Direct measurement was worth having. Effect on the
model: ρ falls 27%, and the recombination escape length 1/(2·Ne·r) grows from
29.4 bp to 37.4 bp — still ~63× smaller than the smallest simulated inversion
body, so §7.3's L-invariance argument is unaffected.

Genome-wide maps (`run_{male,female}_auto/proj/*.PREDICT.BSCORRECTED.txt`):
male 114,328 windows at modal 9 kb, female 116,081 at modal 18 kb;
between-chromosome variation tight (male per-chromosome IQR [2.095, 2.186]e-9,
female [2.871, 2.914]e-9). Within-chromosome heterogeneity is real (male
window-level 5–95%: 1.25–2.87e-9) and a heterogeneous map is available via the
existing `harness/slim/relernn_to_slim_map.py`.

### 8.2 Accessibility: `degenotate_illex/accessible_sites.bed`

**[M]** Includes chr2 (168,250 intervals).

| region | accessible bp | fraction |
|---|---|---|
| chr2 whole | 57,996,480 | 48.55% |
| inversion 60–80 Mb | 9,582,174 | **47.91%** |
| control 10–30 Mb | 12,138,077 | **60.69%** |

Note `analysis/steps/11_relernn/acc_aut.bed` is **not** an Illex mask (different
taxa) and has zero chr2 intervals — do not use it.

**chr2 3-state mask** (`analysis/steps/03_karyotype/chr2_mask/chr2.mask.3state.bed`,
`config.CHR2_MASK_BED`): 19.8 M intervals labelled `accessible_invariant` /
`accessible_variant` / `inaccessible`. It reproduces the fractions above exactly
(chr2 48.55%, inversion 47.908%), so it revises no number. Two things it adds:

- **The accessibility deficit is in the control region, not the inversion.**
  All chr2 collinear (±2 Mb) is **48.79%** accessible against 47.91% inside the
  inversion — a ratio of 0.982. So the diversity deficit inside the inversion is
  **not** a masking artifact. The chr2:10–30 Mb control's 60.69% is the outlier,
  which is why per-accessible-bp normalisation is mandatory when comparing to it.
- **The invariant/variant split gives the SFS zero class**, i.e. the denominator
  needed to put the within-arrangement spectra of §9 on an absolute footing.

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

## 8.4 The within-arrangement SFS shape — the called-genotype version fails **[M]**

> **Superseded in part by §8.5.** The diagnosis below is correct and the
> called-genotype estimator really is unusable, but the recommended replacement
> (a per-karyotype ANGSD/GL spectrum) has now been run and it **works**. Read
> §8.5 for the result; this section is the post-mortem on why the VCF route
> failed.

`illex/scripts/sfs_shape.py`, results in `results/illex/sfs_shape.txt`.
§9 named this "untapped, best candidate" for breaking the identification
problem. It has now been tapped. **It fails, for three independent reasons, and
none of them is about the inversion model.**

**1. The empirical estimate is not identified.** Per-site called chromosomes run
from 20 to 508 (AA) and 20 to 190 (BB), and at low called-n a rare variant is
often not sampled at all, so low-n sites are ascertainment-depleted of rare
variants. The projected spectrum therefore depends on the floor imposed on
called-n far more strongly than on anything biological — and the
inverted-vs-standard contrast, the quantity of interest, **changes sign**:

| floor (AA, BB) | sites | f₁(AA) | f₁(BB) | ratio |
|---|---|---|---|---|
| 200, 100 | 931,997 | 0.639 | 0.542 | **1.18** |
| 300, 150 | 479,196 | 0.759 | 0.785 | **0.97** |
| 400, 170 | 25,066 | 0.907 | 0.879 | 1.03 |

Both floors are defensible; they disagree about whether the inverted arrangement
is *more* or *less* singleton-skewed than the standard one. There is no estimate
here to compare a model against.

**2. The neutral baseline is off by more than the effect.** This is the decisive
one. In the *collinear* control region there is no inversion, so the panmictic
model should be right. It is not: model f₁ = 0.525 against empirical 0.829 (AA)
and 0.819 (BB), an L1 of **0.61 and 0.59** — *larger* than the inversion-body
gaps (0.39 AA, 0.71 BB). Whatever the neutral coalescent is missing dominates
any inversion signal. This is almost certainly the project manuscript's §8b
gap — real per-window diversity variation (mutation-rate heterogeneity, BGS/DFE)
that flat uniform-θ sims do not reproduce — surfacing in the SFS.

**3. It lacks the resolution anyway.** Model-internally, at 96 replicates the two
ridge points differ by at most **1.1–1.4 SE** in every bin (L1 0.0020 inverted,
0.0027 standard). Even with a perfect empirical target it could not separate the
(p_start, plateau) ridge it was recruited to break.

**The fix is a different estimator, not more simulation.** An SFS from called
genotypes on a variants-only, variable-depth callset is measuring the depth
structure. The project already avoids this everywhere else — diversity comes from
**ANGSD/GL off the BAMs** precisely because the VCF is variants-only. A
per-karyotype ANGSD SAF (AA-only and BB-only sample lists, same 349 individuals)
would give a properly depth-aware within-arrangement spectrum, and is the only
version of this statistic worth attempting.

**One model-side result is worth keeping** even though it cannot be tested here:
the single-origin point is grossly different from the ridge points in spectrum
shape (I/S singleton ratio 1.54 versus 1.31, and it is the only point whose
spectrum collapses in the high bins). So the statistic *would* have power against
a hard bottleneck if the data supported it — the failure above is a data
limitation, not a lack of sensitivity to the parameter that matters.

## 8.5 The ANGSD/GL redo works, and it adds a new constraint **[M]**

`illex/scripts/sfs_shape_angsd.py`, results in
`results/illex/sfs_shape_angsd.txt`. **No new ANGSD run was needed** — 
`steps/04_angsd_chr2` already built per-karyotype SAFs for all of chr2
(2026-07-04; AA 254, AB 284, BB 95 individuals, `-GL 1`, `-minQ 20 -minMapQ 20`,
`-remove_bads -only_proper_pairs`, restricted to `chr2.accessible.sites`). Only
`realSFS -r` over the intervals was required: 6 minutes, four runs.

`-fold 0` deliberately: the SAFs use `-anc $REF`, so the unfolded spectrum is the
reference-polarised ALT-count spectrum. Projecting that exactly and folding only
at n = 20 is the identical transform applied to the model side, so
mis-polarisation cancels.

### 8.5.1 The gate passes, which is why anything below it can be read

| check | result | called-genotype version |
|---|---|---|
| AA vs BB in the **collinear control** (must be ~0; they are exchangeable there) | **L1 = 0.019** | — |
| collinear AA vs model no-inversion baseline | **L1 = 0.137** | 0.61 |
| collinear BB vs model no-inversion baseline | **L1 = 0.147** | 0.59 |

The internal consistency check is essentially perfect, and the baseline gap
shrinks **4×**. It is not zero — the empirical collinear spectrum is more
singleton-skewed than the neutral model (f₁ 0.588 vs 0.525), the direction
expected if purifying selection at linked sites is missing (§8b's gap). But the
residual is now smaller than the inversion effect, which is the condition the
called-genotype version failed.

### 8.5.2 The result: the STANDARD arrangement is the one that was reshaped

Comparing each class inside the inversion against *itself* in the collinear
control — so region-level confounders (gene density, mutation rate, BGS) cancel:

| class | L1(body, control) | singleton fraction |
|---|---|---|
| **AA** (inverted, p = 0.626) | **0.053** | 0.5885 → 0.6024 |
| **BB** (standard, 1−p = 0.374) | **0.201** | 0.5980 → **0.4973** |

**The inverted arrangement's spectrum is barely changed from the collinear
background; the standard arrangement's is strongly changed — a 4× larger
effect.** Both classes sit in the same 19 Mb, so anything about the region
affects them equally; an arrangement-specific difference of this size can only
come from the arrangement's own frequency history.

**This is direct confirmation of §7.5.2's mechanism, by a statistic that was
never fitted.** §7.5.2 inferred from π_I/π_S < 1 that confining the standard
class to 0.374 suppresses *its* diversity. The spectrum now shows exactly that:
the smaller class coalesces faster, which erodes the expansion's singleton excess
and shifts weight to intermediate frequencies. The common arrangement, being
close to the whole population, is nearly untouched.

### 8.5.3 Model comparison: the soft origin is confirmed, and a tension appears

Against the inverted class (where the baseline residual is smallest):

| model point | L1 vs AA (inverted) |
|---|---|
| ridge, plateau 0 | **0.0768** |
| ridge, plateau 100 ky | **0.0752** |
| rising logistic (superseded) | 0.1236 |
| **single origin** | **0.1854** |

The fitted balancing points win, and the **single origin is worst by 2.5×** —
an independent confirmation of §7.5.1's conclusion by a statistic not used to
reach it. Worth having: that was previously supported only by the two ratios.

**But the demography-cancelling contrast disagrees with the fit.** Taking the
inverted-minus-standard profile, which cancels any misfit that moves both
classes together:

| point | f₁(I) | f₁(S) | ratio | L1(I−S profile) |
|---|---|---|---|---|
| **ANGSD empirical** | 0.6024 | 0.4973 | **1.211** | — |
| ridge, plateau 0 | 0.5640 | 0.4299 | 1.312 | 0.101 |
| ridge, plateau 100 ky | 0.5648 | 0.4287 | 1.318 | 0.106 |
| rising logistic | 0.5406 | 0.4465 | **1.211** | **0.022** |
| single origin | 0.6617 | 0.4311 | 1.535 | 0.316 |

The ridge points **over-predict** the inverted-vs-standard skew difference by
~8%; the superseded rising-logistic trajectory reproduces it. So the two fitted
ratios and the SFS contrast prefer different trajectories, and **no point in this
family satisfies both.**

The direction is interpretable rather than merely awkward: the model
over-restricts the standard class, i.e. it keeps BB confined to 0.374 for too
long. That points the same way as §7.5.2 — **the inversion reached 0.626 more
recently than the fitted trajectory implies.** The `plateau` parameterisation
could not express this well, because of the trap documented in
`illex/balancing.py`: the overdominance curve decelerates into p*, so it has been
within a few percent of 0.626 for ~10⁵ generations even at "plateau = 0".

**So the third statistic finally does something the first two could not** — it
constrains the arrival time, and it says "later". Testing that properly needs a
trajectory family with an explicit, adjustable arrival time rather than one
inherited from the overdominance ODE. That is the next modelling step.

### 8.5.4 What it still does not do

It does **not** break the (p_start, plateau) ridge. The two ridge points differ
by 0.0016 in L1 against the inverted class and 0.004 in the contrast — at the
Monte Carlo noise level (model-side per-bin separation 1.1–1.4 SE, §8.4). The
ridge stands; both parameters remain a joint range.

## 8.6 A trajectory family with an explicit arrival time — all three statistics fit **[M]**

`illex.balancing.arrival_curve` / `build_arrival_sim`,
`illex/scripts/scan_arrival.py`, results in
`results/illex/scan_arrival{,_tinv,_refine}.{csv,txt}`.

### 8.6.1 Why the old family could not be rescued by re-tuning

Its rise obeys dp/dt = (s_het/p*)·p(1−p)(p*−p), whose approach to p* is
exponential at a rate set by the *same* s_het that sets the take-off. Measured on
the curve: **70.3% of the rise is spent between 0.90 p* and arrival, and that
fraction is invariant in s_het** (checked at 3.58e-5, 1e-4, 1e-3). At the fitted
s_het the frequency is above 0.563 for 511,600 of 727,600 generations. So
"plateau = 0" was never "no plateau", and no value of s_het makes the arrival
recent. That is exactly the shape §8.5.3 rejects.

### 8.6.2 The family

Three phases, arrival time explicit:

```
[t_inv, t_arrive+t_rise]    dormant   p = p_start
[t_arrive+t_rise, t_arrive] rise      p_start -> p*   (overdominance ODE)
[t_arrive, 0]               plateau   p = p*
```

`t_rise` is now *given* and s_het *derived* from it (`s_het_for_rise`) — the
inverse of the old parameterisation, which is the point: the rise can be made
fast and late instead of slow and early. Mechanistically a soft sweep from
standing variation: the inversion segregated at low frequency, became
advantageous, swept to a balanced equilibrium. It **nests** the old family at
t_arrive = 0, t_rise = t_inv (verified: curves agree to 1.1e-3).

Validation that the new axis does what it exists to do — fraction of history
above 0.90 p*, at t_inv = 730 ky:

| t_arrive | t_rise | above 0.9 p* |
|---|---|---|
| 0 | 727,600 (= old family) | **70.1%** |
| 0 | 50,000 | ~5% |
| 100,000 | 50,000 | **18.5%** |
| 400,000 | 50,000 | 59.6% |

### 8.6.3 All three statistics fit simultaneously — the first time

2,736 sims over three scans, 48 reps/cell, t_rise = 50 ky.

| | π_I/π_S | dxy/π_I | ANGSD f₁(I)/f₁(S) |
|---|---|---|---|
| target | 0.7368 | 1.8794 | 1.211 |
| **t_inv 900 ky, t_arrive 200 ky, p_start 0.32** | **0.7647 (+3.8%)** | **1.9276 (+2.6%)** | **1.200 (−0.9%)** |
| two-phase family at its fitted point | 0.737 (fitted) | 1.879 (fitted) | 1.312 (**+8.3%**) |

`p_start = 0.32` is an **interior** optimum (0.38 and 0.44 are worse in every
cell), and the solution region is t_inv ≈ 900–950 ky, t_arrive ≈ 150–250 ky.
The new axis is powerful: across the first scan the f₁ ratio spans **0.93–1.78**,
against a two-phase family effectively pinned at 1.31.

**The story it tells is qualitatively different.** Not "rose slowly to
equilibrium over 730 ky" but: **arose ~900 ky ago, sat at ~32% for ~650 ky, swept
to 0.626 about 200 ky ago, and has been held there since.**

**Why p_start must be large:** dormancy at a low standing frequency squeezes the
inverted class hard. At p_start = 0.027 with t_arrive = 0 the model gives
π_I/π_S = 0.151 and dxy/π_I = 7.84 — π_I collapses. The standing frequency is
the parameter the dormancy phase pushes hardest.

### 8.6.4 **[W] This corrects my "the age is robust to trajectory shape" claim**

§7.5 reported that correcting the trajectory moved the age only ~4% (750–800 →
719–737 ky) and concluded the age "is not an artifact of the trajectory shape".
That was **scoped too broadly**. It is robust *within* a family — across the
two-phase family's degenerate plateau dimension the age moves 1.2%. It is **not**
robust *across* families: adding a third phase and fitting the SFS contrast moves
it to ~900 ky, a **+23%** shift.

So the age should now be quoted as a range across trajectory families:

**t_inv ≈ 730–950 ky**, with the upper end preferred because it is the only value
that fits all three statistics, and all of it still conditional on µ = 3e-9
(which remains the dominant term: ±30% on µ is ±220–290 ky). **[Narrowed by
§8.7.4: with a drifting dormancy — the better-fitting and more defensible
family — the age is ~800 ky and the spread across families is 730–900 ky.]**

### 8.6.5 Caveats, strongest first

1. **Dormancy is held at a constant frequency, not drifting.** A real standing
   variant would wander; holding it fixed gives the inverted class a constant
   coalescent size 2·N(t)·p_start throughout dormancy. This is the strongest
   assumption in the family and it is doing real work, since it is what forces
   p_start up to 0.32.
2. **t_rise is fixed at 50 ky, not fitted.** Chosen so arrival is sharp. The
   (t_rise, t_arrive) pair is likely partly degenerate and has not been explored.
3. **The contrast target inherits the baseline residual.** The ANGSD collinear
   spectra still miss the neutral model by L1 0.137/0.147 (§8.5.1), so the
   target ratio 1.211 carries a systematic uncertainty that is not quantified
   here. The contrast cancels a *shared* shift, not a differential one.
4. p* is still asserted at 0.626, not inferred.
5. No error bars on (t_inv, t_arrive, p_start) yet — only a grid.

## 8.7 Drifting dormancy — the strongest caveat, removed **[M]**

§8.6.5 flagged the constant-frequency dormancy as the family's strongest
assumption. It is now replaced by a drifting one
(`illex.balancing.dormancy_bridge`, `arrival_curve_drift`,
`build_arrival_drift_sim`; scans `results/illex/scan_arrival_drift_*.{csv,txt}`).

**It was not a small assumption.** Over a 650,000-generation dormancy on the
growth arm the neutral drift standard deviation at p = 0.32 is **0.275** —
comparable to the frequency itself.

**The implementation** is a guided (Durham–Gallant modified) diffusion bridge:
WF volatility √(p(1−p)/2N(t)) with the Brownian bridge's linear guiding drift,
floored at one copy (below that the arrangement is lost, and it is not). Each
replicate draws its own path, so the ensemble carries the drift variance instead
of averaging it into a single mean trajectory. It is **not** an exact WF bridge —
the true conditioned drift is not used — and that is stated in the code.

### 8.7.1 Drift squeezes the inverted class harder, exactly as predicted

The prediction was recorded before running: the coalescent rate inside the
inverted class is 1/(2N(t)p(t)), so π_I integrates ∫dt/(2Np); because the
integrand goes as 1/p it is dominated by time spent at LOW frequency, and by
Jensen a wandering path accumulates strictly more coalescence than a constant
path at the same mean. At matched parameters (t_inv 900 ky, t_arrive 200 ky,
p = 0.32):

| statistic | constant dormancy | drifting | change |
|---|---|---|---|
| π_I/π_S | 0.7647 | 0.7052 | **−7.8%** |
| dxy/π_I | 1.9276 | 2.1678 | **+12.5%** |
| f₁(I)/f₁(S) | 1.200 | 1.247 | +3.9% |

All three moved in the predicted direction and by a material amount.

### 8.7.2 The drifting model fits BEST, and it moves the age down

3,000+ additional sims, 48–64 reps/cell.

| | π_I/π_S | dxy/π_I | f₁ ratio | score |
|---|---|---|---|---|
| target | 0.7368 | 1.8794 | 1.211 | — |
| **drifting: t_inv 800 ky, t_arrive 200 ky, p_hand 0.28** | **0.7277 (−1.2%)** | **1.9208 (+2.2%)** | **1.222 (+0.9%)** | **0.00072** |
| constant: t_inv 900 ky, t_arrive 200 ky, p 0.32 | 0.7647 (+3.8%) | 1.9276 (+2.6%) | 1.200 (−0.9%) | 0.00218 |

**All three parameters are interior optima**, not boundary artifacts: t_inv 800 ky
sits between 750 ky (0.0015) and 850 ky (0.0083); t_arrive 200 ky between 150 ky
(0.0039) and 250 ky (0.0010); p_hand 0.28 between 0.24 (0.0362) and 0.32
(0.0128).

The drifting version needs a **shorter dormancy and a lower handoff frequency**
than the constant one — 550 ky at p ≈ 0.28 rather than 650 ky at 0.32 — which is
the expected compensation for drift supplying extra squeeze on its own.

### 8.7.3 A genuine single origin is now decisively excluded

The drifting family can express what the constant one could not: start at one
chromosome at t_inv and climb. So §7.5.1's soft-origin conclusion can be
re-tested rather than assumed, and it survives emphatically. Best single-origin
cell over t_inv ∈ [900 ky, 1.4 My]: **score 0.67**, against 0.00072 — roughly
**900× worse**, with π_I/π_S 0.34–0.54 (target 0.737) and dxy/π_I 3.1–5.0
(target 1.879) everywhere. The 1/p integrand near the origin enforces monophyly
at t_inv and π_I collapses.

This matters because the earlier exclusion could be dismissed as an artifact of
holding p constant at a low value. It cannot: under a proper drifting model from
a single chromosome the fit is far worse still. **The origin is soft.**

### 8.7.4 Where the age now stands

| model | targets fitted | age |
|---|---|---|
| two-phase, rise to equilibrium | 2 (π ratio, dxy ratio) | 730–740 ky |
| three-phase, constant dormancy | 3 (+ SFS contrast) | ~900 ky |
| **three-phase, drifting dormancy** | 3 | **~800 ky** |
| three-phase, drifting, single origin | 3 | excluded |

**Quote ~800 ky, with 730–900 ky as the spread across defensible trajectory
families** — narrower than §8.6.4's 730–950 ky, because the best-fitting and
most defensible family lands in the middle. Still conditional on µ = 3e-9, which
remains the dominant term (±30% on µ is ±240 ky, an order of magnitude larger
than the trajectory-family spread).

**Current best picture:** the inversion arose ~800 ky ago, drifted as a standing
polymorphism around ~28% for ~550 ky, swept to 0.626 about 200 ky ago, and has
been held there since.

### 8.7.5 What is still assumed

1. The bridge is guided, not an exact WF bridge (§8.7 above).
2. Dormancy is **neutral**. If the inversion was under weak selection while
   standing, the path law changes. Untested.
3. `t_rise` fixed at 50 ky; its degeneracy with `t_arrive` unexplored.
4. The contrast target still inherits the §8.5.1 baseline residual (L1
   0.137/0.147), so the 1.211 target carries unquantified systematic error.
5. p* asserted at 0.626. No error bars on the three parameters — grid only.

## 8.8 Non-neutral dormancy — it fits best, and the inversion was ALREADY balanced **[M]**

`illex.balancing.dormancy_balanced` / `arrival_curve_balanced` /
`build_arrival_balanced_sim`; scan `results/illex/scan_arrival_sdorm.{csv,txt}`.
§8.7.5 listed "dormancy is neutral" as the last live assumption. Removed.

**The model.** During dormancy the inversion is a balanced polymorphism at
equilibrium p_eq = p_hand, with the same overdominance form as the rise:
dp/dt = (s_dorm/p_eq)·p(1−p)(p_eq−p) plus WF noise. It is a **one-parameter
interpolation between the two arms already run** — s_dorm → 0 recovers free
drift (§8.7), s_dorm → large recovers the constant frequency (§8.6) — so both
limits were known before running and the only question was which strength the
data prefer.

**It needs no bridge, which makes it the most rigorous arm.** With a restoring
force the diffusion has a stationary measure, and every 1-D diffusion is
reversible with respect to its stationary measure, so the backward path obeys
the same SDE and can be simulated directly from the handoff frequency. The
Durham–Gallant approximation that §8.7 required does not arise. (The s_dorm = 0
row in the scan is the exception — neutral WF has absorbing boundaries and no
stationary measure, so that row is approximate; use `dormancy_bridge` for the
properly conditioned neutral case.)

Validated against the analytic stationary SD √(p_eq/4N·s_dorm): observed/predicted
0.97–1.32 over s_dorm ∈ [3e-6, 1e-4], with the 1e-6 case running low (0.66)
because its SD is large enough to be truncated by the boundaries.

### 8.8.1 The fit, and an interior optimum in s_dorm

| dormancy model | π_I/π_S | dxy/π_I | f₁ ratio | score |
|---|---|---|---|---|
| target | 0.7368 | 1.8794 | 1.211 | — |
| **balanced, s_dorm = 3e-5** | **0.7289 (−1.1%)** | **1.8603 (−1.0%)** | **1.213 (+0.1%)** | **0.00022** |
| free drift (§8.7) | 0.7277 (−1.2%) | 1.9208 (+2.2%) | 1.222 (+0.9%) | 0.00072 |
| constant frequency (§8.6) | 0.7647 (+3.8%) | 1.9276 (+2.6%) | 1.200 (−0.9%) | 0.00218 |

All three residuals are ~1% or better — the best fit obtained for this system.
And s_dorm is **interior**, at t_inv 800 ky / t_arrive 200 ky / p_hand 0.28:

| s_dorm | 0 | 1e-6 | 3e-6 | 1e-5 | **3e-5** | 1e-4 |
|---|---|---|---|---|---|---|
| score | 0.0122 | 0.0127 | 0.0123 | 0.0074 | **0.0002** | 0.0016 |

Both limits are worse. **The data prefer a specific dormancy selection strength,
not merely "more" or "less" than neutral.** Identification is real but soft: the
optimum is sharp at the grid point yet the neighbouring cells differ by
residuals of only 2–3%, so read it as **s_dorm ≈ 1e-5 to 1e-4 preferred over
both limits**, not as a point estimate.

### 8.8.2 What it means: the sweep was an equilibrium SHIFT, not the onset of selection

At the optimum, N·s_dorm ≈ **40** — selection ~40× stronger than drift, so the
standing phase was unambiguously non-neutral, with the frequency held within
about ±0.04 of 0.28 (stationary SD 0.042).

**So the inversion was already a balanced polymorphism before it swept.** It was
not a neutral standing variant that happened to become useful; it was an adaptive
polymorphism whose *optimum moved*. Selection strengthened ~**15×** across the
transition, from s_dorm ≈ 3e-5 maintaining p ≈ 0.28 to s_het ≈ 4.6e-4 driving the
50 ky rise to 0.626.

This also matters for reading §7.5.3's non-neutrality result: that argument
excluded neutral drift as a route to the *present* frequency. This one goes
further and says the arrangement was under selection for most of its history.

### 8.8.3 The age is now stable across dormancy models

t_inv = **800 ky** at the optimum — identical to §8.7's drifting fit, and the
improvement from 0.00072 to 0.00022 came entirely from s_dorm, t_arrive and
p_hand being better resolved. So the age stopped moving as the dormancy model
improved, having gone 730 ky (2 targets) → 900 ky (3 targets, constant dormancy)
→ 800 ky (drifting) → **800 ky (balanced)**.

**Quote ~800 ky, spread 730–900 ky across families**, unchanged from §8.7.4 and
now supported by the two best-fitting models agreeing. µ still dominates
(±30% = ±240 ky).

**Best current picture:** the inversion arose ~800 ka as the population began its
12.4× expansion, was maintained as a balanced polymorphism near 28% for ~550 ky
(N·s ≈ 40), its equilibrium shifted ~250 ka, it swept to 0.626 over ~50 ky with
selection ~15× stronger, and has been held there for ~200 ky.

### 8.8.4 Still assumed

1. p_eq during dormancy is set equal to p_hand — the standing equilibrium and the
   handoff frequency are the same number by construction, not independently
   fitted.
2. The equilibrium shift is instantaneous at the start of the rise.
3. `t_rise` fixed at 50 ky; its degeneracy with `t_arrive` still unexplored.
4. The contrast target inherits the §8.5.1 baseline residual (L1 0.137/0.147).
5. p* asserted at 0.626; overdominance is one of several mechanisms that would
   give the same p(t) shape (§7.5, caveat 2). No error bars — grid only.

## 8.9 The GO content of the inversion — the published enrichment does not hold **[M][W]**

`illex/scripts/go_inversion.py`, results in `results/illex/go_inversion.{tsv,txt}`.
§8.8/§3i named the "102 metabolism-enriched genes" as the mechanistic handle and
the cheapest way to turn the environmental scenario from story into result. It
was checked. **The enrichment does not survive a null that respects linkage, and
the headline number is a miscount.**

### 8.9.1 Two errors in how the region's gene content is described

1. **"102 metabolism-enriched genes" conflates two things.** 102 is the *total*
   gene count in the nominal span. Only **36** of those carry any GO annotation
   at all, and only **30** lie in the Fst-defined differentiated body — many of
   the rest are literally "function unknown". No reading of the data supports
   102 metabolism genes.
2. **The gene set uses the nominal span and spills outside it.** It runs
   60,013,280–79,998,501, so it starts *before* the 60,040,617 breakpoint, and
   the outermost ~500 kb at each end is collinear flanking sequence with
   control-like Fst (§4.2).

And a third fact worth having: the inversion is **gene-poor**, not gene-rich —
30 annotated genes against a median of 50 in random 19 Mb windows (5–95%: 18–90).

### 8.9.2 Why the published test was the wrong test

`steps/06_gene_content/go_enrichment.py` uses a hypergeometric test against a
genome-wide gene background. That null assumes the genes are a random *sample of
genes*. They are one contiguous block inherited as a unit, so two things break:
tandem duplicates count as independent observations, and gene families cluster,
which makes *any* contiguous window look enriched for whatever sits in it.

Replacing it with a **window null** — the same count in 2,000 random contiguous
19 Mb windows elsewhere in the genome, plus collapsing genes within 200 kb that
share a term — dissolves the result:

| term | published p | published FDR | window p (clustered) | window FDR |
|---|---|---|---|---|
| fructose-bisphosphate aldolase activity | 3.4e-5 | 0.019 | **not testable** | — |
| fatty acid binding | 3.1e-4 | 0.019 | **not testable** | — |
| aldehyde-lyase activity | 3.2e-4 | 0.019 | **not testable** | — |
| carboxylic acid biosynthetic process | 1.7e-4 | 0.019 | 0.024 | 0.38 |
| monocarboxylic acid biosynthetic process | 2.4e-4 | 0.019 | 0.032 | 0.43 |
| carboxylic acid metabolic process | 2.8e-4 | 0.019 | 0.184 | 0.99 |

"Not testable" means **fewer than two of those genes are in the differentiated
body at all**. The single most significant published term, FBPA, is
LOC_00005292 and LOC_00005293 — 33 kb apart, so one duplication event scored
twice, and *both sit in the collinear flank* at 60.12 and 60.17 Mb, outside the
Fst-defined body. The headline result came from a tandem pair in sequence that
is not part of the inversion.

Across all 505 testable terms, 4 reach FDR < 0.10 and all four sit at 0.063 —
and two of them ("bone cell development", "megakaryocyte development") are
vertebrate-specific terms in a squid, i.e. GO transferred from human/mouse
orthologs rather than biology.

### 8.9.3 What is actually there: a real but statistically unsupported lipid thread

Five genes in the differentiated body carry peroxisomal / fatty-acid /
lipid-catabolism annotation, and unlike the FBPA pair they are **spread across
the region at ~4 independent locations**:

| gene | position | annotation thread |
|---|---|---|
| LOC_00005301 | 61.39 Mb | peroxisomal protein import |
| LOC_00005339 | 69.07 Mb | fatty acid β-oxidation / metabolism |
| LOC_00005375 | 78.39 Mb | regulation of lipid & phospholipid catabolism |
| LOC_00005376 | 78.64 Mb | "bile acid" biosynthesis/conjugation, fatty acid metabolism |
| LOC_00005384 | 79.31 Mb | "bile acid", peroxisomal import, fatty acid β-oxidation |

The peroxisome terms recur through the top-30 (protein targeting to peroxisome,
peroxisomal matrix, microbody lumen, peroxisomal transport, peroxisome
organisation) — but they are the *same two genes* under nested GO terms, not
independent evidence. **Read "bile acid" with caution**: cephalopods do not make
bile acids, so those terms are almost certainly bile-acid-CoA-ligase homologs
that are really acyl-CoA synthetases — fatty-acid activation. That *strengthens*
the lipid reading while making the literal annotation meaningless.

### 8.9.4 The honest verdict, and what it does to §3i

**A 19 Mb region with 30 annotated genes cannot support a GO enrichment
analysis.** With terms requiring k ≥ 2 the test has almost no power, so the
absence of significance here is weak evidence of absence — but the presence of
significance in the published version was an artifact of the null, not a
discovery.

So the correct output is a **gene list, not an enrichment claim**: the inversion
carries a handful of peroxisomal / fatty-acid-oxidation genes at several
independent positions. That is *consistent with* the metabolic reading in §3i
and worth stating as such, but it does **not** turn that scenario from story into
result, which is what §3i hoped it would do. The scenario remains interpretation.

Both suggested follow-ups were run; see §8.10. Neither survives, and the
five-gene "lipid thread" shrinks further under proper annotation.

## 8.10 What those five genes actually are — the lipid thread shrinks to two **[M][W]**

§8.9.4 proposed two follow-ups. Both were run. One is impossible and the other
weakens the story further.

### 8.10.1 The InterProScan run does not cover the proteome

`gene_family/InterProScan/interpro_results.tsv` has **101 protein rows total**
(90 `LOC_`, 10 `novel_model_`, 1 `temp_model_`) — it is a small targeted run, not
a genome-wide annotation. Only LOC_00005301 of the five is in it. My §8.9.4
recommendation assumed a proteome-wide InterProScan existed; it does not.

The information is available elsewhere, and is better: `entap_results.tsv`
carries the best similarity hit, EggNOG orthology and KEGG KO for every gene.

### 8.10.2 The five genes, properly identified

| gene | pos (Mb) | best hit | id / cov | KEGG | what it actually is |
|---|---|---|---|---|---|
| LOC_00005301 | 61.39 | Lon protease homolog 2, peroxisomal (*O. bimaculoides*) | 80.5% / 100% | K01338 | peroxisomal matrix **protease** — organelle quality control, not β-oxidation |
| LOC_00005339 | 69.07 | malonyl-CoA-ACP transacylase, mitochondrial (*O. vulgaris*) | 65.4% / 75% | K00645 | mitochondrial fatty-acid **synthesis** (mtFAS) — not catabolism |
| LOC_00005375 | 78.39 | **PRKCD**, protein kinase C delta (*S. pharaonis*) | 98.1% / 100% | K06068 | **signalling kinase — not a metabolic enzyme at all** |
| LOC_00005376 | 78.64 | acyl-CoA amino acid N-acyltransferase 1-like (*O. vulgaris*) | 43.5% / 100% | K00659, K01068 | acyl-CoA **thioesterase / N-acyltransferase** (BAAT/ACNAT family) |
| LOC_00005384 | 79.31 | *S. pharaonis*; EggNOG "acyl-CoA oxidase activity" | 76.5% / 100% | **K00232 (ACOX1)** | **ACOX1 — the rate-limiting enzyme of peroxisomal β-oxidation** |

**[W] Three corrections to §8.9.3, all of them mine:**

1. **My "bile acid = acyl-CoA synthetase" call was directionally right, specifically
   wrong.** LOC_00005376 is BAAT/ACNAT family — an N-acyltransferase/thioesterase,
   not a ligase. Both handle acyl-CoA, but they are different reactions. The
   broader point stands: cephalopods make no bile acids, so the literal term is
   meaningless and the gene is fatty-acyl-CoA machinery.
2. **Two of the five are not lipid-catabolism genes at all.** PRKCD is a
   signalling kinase whose "regulation of lipid catabolic process" GO is a
   regulatory annotation transferred from human PKCδ's role in lipolysis
   signalling; LONP2 is a protease that is peroxisomal but not metabolic. And
   MCAT is fatty-acid **synthesis**, so the "fatty acid β-oxidation" GO attached
   to it looks mis-transferred. The strongest-annotated gene in the whole set
   (PRKCD, 98.1% identity, E = 0) is the one with no metabolic role.
3. **They are not well spread.** §8.9.3 said "~4 independent positions". At the
   200 kb clustering threshold that is true, but biologically three of the five
   (78.39, 78.64, 79.31 Mb) sit within a **~900 kb window** at one end of the
   inversion. The independence was overstated.

**So the coherent core is two genes, not five:** ACOX1 (79.31 Mb, genuine
peroxisomal β-oxidation, rate-limiting) and ACNAT (78.64 Mb, acyl-CoA handling),
670 kb apart. LONP2 at 61.39 Mb adds a second *peroxisomal* gene at the opposite
end, which is the only reason "peroxisome" recurs at all.

### 8.10.3 The sweep-candidate check is unanswerable, not negative

**chr2 was never scanned.** `results/empirical_scan_fullsfs/genome.preds` contains
**zero chr2 windows**. The not-scanned set is exactly **{chr2, chr42, chrZ}**,
matching `build_persex_vcf.sh`'s "autosomes (excl chr2 inv, chr42, chrZ)"; those
three are also the three missing from `11_relernn/genome.sizes` (43 of 46
sequences), so they were dropped upstream in the shared prep rather than by the
scan. **[M] Confirmed by the user 2026-08-24: not run. Re-run completed 2026-08-25 —
results in §8.11.**

**[W] Two errors in an earlier version of this paragraph.**

1. I wrote the missing set as "no 2, 40, 42 or Z". **chr40 was scanned** — 240
   windows in `genome.preds`, and it simply produced no outlier regions. Lumping
   it with the unscanned chromosomes conflated "scanned, no hits" with "never
   looked at", which is precisely the distinction the rest of this section is
   about.
2. Listing chr42 next to chrZ implied it was excluded for the same reason.
   **chr42 is an autosome** — 47,554,665 bp, larger than chr40 or chr45 — and
   §1 of the project manuscript records that as **[ESTABLISHED]** (chrZ is the
   sex chromosome; chr42 is not). Its exclusion is a legacy of an earlier
   misidentification, not a property of the chromosome, which means chr42 is
   missing from both the recombination map and the sweep scan for **no valid
   reason** and should be added back.

So "no sweep candidates in the inversion" is an **absence of data**, not
evidence, and must not be reported as a negative result.

### 8.10.4 Bottom line: the gene content does not support the scenario

With **30 annotated genes** in the body, finding one or two in any given pathway
is exactly what chance produces. There is one specific, interesting candidate —
ACOX1, the rate-limiting step of peroxisomal β-oxidation, which in marine
invertebrates handles the very-long-chain and branched fatty acids abundant in a
zooplankton-based diet — and a second peroxisomal gene at the far end. That is a
**hypothesis worth naming, not evidence**.

**§3i's environmental scenario therefore stands or falls on other grounds.** The
gene-content route was proposed (§3i), tested (§8.9), refined (§8.10), and does
not deliver at any stage. It should be reported as a named candidate gene plus an
explicit statement that the region's gene content carries no statistical signal —
not as support.

## 8.11 chr2 sweep scan, and a genome-wide test for polygenic architecture **[M]**

### 8.11.1 The inversion is not a sweep outlier — and the composition flips

chr2 and chr42 were re-run 2026-08-25 (1,050 and 126 windows). Comparing within
chr2, so the classifier's known genome-wide over-call (§8b of the project
manuscript) cancels:

| | n | soft | hard | mean S |
|---|---|---|---|---|
| inversion body | 184 | 5.4% | **0.5%** | 0.072 |
| chr2 collinear | 832 | 3.4% | **5.5%** | 0.094 |

**Fewer sweep calls inside the inversion than outside it**, and the composition
inverts: hard calls are depleted **11-fold** inside while soft calls are mildly
enriched. That is what a maintained two-class polymorphism should look like to
this classifier — two haplotype backgrounds read as "soft", while the
single-haplotype, low-diversity signature of a hard sweep is absent. It is an
independent line of support for balancing selection over a classic sweep, from a
method that shares nothing with the coalescent modelling.

None of the 11 sweep-called windows in the body contains any of the five
annotated lipid-pathway genes; the nearest miss is 68.9–69.0 Mb against MCAT at
69.07 Mb. **ACOX1 (79.31 Mb) is not in a sweep-called window.**

chr42, the autosome excluded for no valid reason (§8.10.3), behaves like an
ordinary chromosome: 4.0% soft, 3.2% hard, mean S 0.074.

### 8.11.2 Is the supergene part of a larger co-adapted complex? No.

`illex/scripts/karyotype_fst_scan.py`, results in
`results/illex/karyotype_fst_scan.{tsv,txt}`. Hudson Fst(AA, BB) per 100 kb
window, genome-wide, as a ratio of sums; 34,398 windows, 149 AA and 53 BB
individuals (the overlap with the clean-350 callset).

Inside the inversion every locus is linked to karyotype by construction — that
*is* the supergene. The testable question is whether anything **outside** it
covaries with the arrangement, which would indicate epistatic partners or
co-adapted alleles the inversion does not physically contain.

| set | n | median Fst | p99 | max |
|---|---|---|---|---|
| **inversion body** | 199 | **0.3962** | 0.5823 | 0.6030 |
| chr2 collinear | 955 | 0.0137 | 0.2323 | 0.3549 |
| all other chromosomes | 33,204 | 0.0132 | 0.2372 | 0.6387 |

**The tail is exactly what chance produces.** Above the background's own 99.9th
percentile: 35 windows observed against 34.2 expected by construction (ratio
1.02). Above the 99.99th: 4 against 3.4 (ratio 1.17). **There is no excess at
all.**

**And every apparent outlier is a callability artifact.** Fst rises sharply as
window information falls:

| sites/window | n windows | median Fst | max |
|---|---|---|---|
| <500 | 456 | **0.1909** | 0.6387 |
| 500–2k | 921 | 0.1029 | 0.5017 |
| 2k–5k | 2,019 | 0.0156 | 0.3719 |
| 5k–20k | 30,747 | **0.0129** | 0.2249 |
| >20k | 16 | 0.0097 | 0.0226 |

corr(log₁₀ n_sites, Fst) = **−0.648**; the top-20 background windows have a
median of 202 sites against a background median of 9,858. The apparent
"region" on chr36 at 64.5–65.1 Mb is a dropout, not a locus: Fst is 0.01–0.02
where the window holds ~10,000 sites and jumps to 0.36–0.62 exactly where the
count collapses to 57–642.

### 8.11.2a What is in the chr36 dropout: nothing **[M]**

Asked directly, since a callability dropout can itself be meaningful (a
segmental duplication, a paralog array). It is not. chr36:63.9–65.2 Mb contains:

* **1.1% accessible sequence** — 14,135 bp of 1.3 Mb, against 26.5% chr36-wide.
  A **24-fold depletion**.
* **Two annotated features, both ~100 bp**: LOC_00002817 (98 bp) and
  LOC_00002818 (119 bp), each a single exon, each `Name= function unknown`,
  encoding ~32 and ~39 residues. These are annotation noise, not genes.
* Gene density **1.5/Mb against 5.4/Mb** chromosome-wide.

So the region is inaccessible, gene-poor, near-certainly repetitive or
heterochromatic sequence, and its elevated Fst is computed from 57–642 SNPs in
windows where ~1% of the sequence is callable at all. There is no locus there.

**And the same holds for every remaining candidate.** Among the well-covered
windows (n ≥ 5,000), the ten highest-Fst windows genome-wide contain **zero
annotated genes between them**, and all ten sit at the low end of the
well-covered range (5,180–8,241 sites against a median of 9,858) — the
coverage-noise relationship persists even inside the "well-covered" set.

*Method note:* `MIN_SITES = 50` in the scan was too permissive; an
accessible-fraction filter would be the better gate. The headline conclusion is
unaffected, because §8.11.3's bound was already computed on the well-covered
subset — but the "top outliers" table in the first pass was polluted by dropouts
and should not be read as a candidate list.

### 8.11.3 The negative is bounded, not merely underpowered

The scan recovers the inversion at median Fst **0.3962** against a published
0.3652 (different sample subset), so it plainly detects real differentiation of
that magnitude. In the **90% of the genome that is well covered** (≥5,000
sites/window) the maximum Fst anywhere is **0.2249** and the median is 0.0129.

**So co-adapted partners differentiated above ~56% of the inversion's own value
are excluded across 90% of the genome.** The supergene is self-contained. This
also corroborates the project manuscript's §2 windowed-PCA result ("structure
only inside the inversion", η² 0.748 vs 0.003) at per-window resolution, by a
different statistic.

### 8.11.4 The temporal method is not available here

The polygenic framework that motivated the question — decomposing the variance
in allele-frequency change into drift versus linked selection, using LD among
loci — requires **allele frequencies across consecutive generations** plus a
fitness proxy. Illex has a single time point, so it cannot be run. The
genome-wide Fst scan above is the single-timepoint analogue of the same
question, and it answers it negatively.

## 8.12 The argentinus equidistance test — dead, and the reference is why **[M][W]**

`illex/scripts/argentinus_equidistance.py`, results in
`results/illex/argentinus_equidistance.txt`.

### 8.12.1 The design, which was sound

If the inversion **predates** the illecebrosus/argentinus split, the arrangement
classes were already separate lineages when argentinus diverged and argentinus
should sit closer to one of them. If it **postdates** the split, AA and BB
lineages are exchangeable before the split and E[dxy(AA,arg)] = E[dxy(BB,arg)]
exactly. So the statistic is

    dxy(AA,arg) − dxy(BB,arg) = Σ (p_AA − p_BB)(1 − 2q)

with the collinear region as a built-in control where it must be zero.

Two features made this look ideal: sites where AA and BB agree cancel exactly,
so the variants-only illecebrosus callset suffices; and no argentinus karyotypes
are needed, only allele frequencies, which genotype likelihoods estimate well at
0.6× — which is why it replaced the (wrong) resequencing recommendation of
§8.10.

### 8.12.2 It fails its own control, twice

| run | allele handling | control z (must be ~0) | body z |
|---|---|---|---|
| 1 | ANGSD-inferred minor, kept if it matched the VCF ALT (36%) | **+4.44** | −1.97 |
| 2 | VCF REF/ALT forced via `-doMajorMinor 3` + sites file | **+3.13** | +7.21 |

Run 1 also carried a selection artifact of its own — keeping only sites where
ANGSD's inferred minor equalled the VCF ALT selects on argentinus carrying that
allele — which is why the sign of the body statistic flips between runs. Run 2
removes it, and the body statistic still flips sign *across frequency classes*
(−0.34 at MAF 0.02–0.05, **+0.19** at MAF > 0.10), so the aggregate is not
estimating a single quantity.

### 8.12.3 **[W] The cause: the reference genome carries BB, and I claimed it could not matter**

I asserted in the design that reference mapping bias could not fake a signal,
because it depresses q identically in both terms and a common shift cancels.
**That is wrong**, and it is wrong for a reason specific to inversions: inside
the inversion the reference is not neutral between the arrangements — it *is*
one of them. Measured, with p = frequency of the NON-reference allele at
arrangement-diagnostic sites (MAF > 0.10):

| region | p_AA | p_BB | difference |
|---|---|---|---|
| **inversion body** | 0.5149 | 0.2165 | **+0.2984** |
| collinear control | 0.3029 | 0.2874 | +0.0156 |

BB matches the reference inside the inversion; outside it the two arrangements
are equivalent. Argentinus reads are mapped to that reference, and reads
matching the reference map and call more readily, so **argentinus is pulled
toward BB precisely at the diagnostic sites inside the inversion and nowhere
else** — the exact direction, magnitude and location of the apparent signal. The
collinear control is clean for the same reason it must be: no arrangement
structure, no asymmetric reference.

### 8.12.4 What this means for the argentinus question

**Deeper argentinus sequencing would not fix it.** The bias is in the reference,
not the coverage — which corrects the framing in §8.10 as well as the original
resequencing recommendation. Both were wrong, for different reasons.

The fix is to map both species to a **third genome** so neither arrangement is
privileged. *I. coindetii* is assembled (GCA_977009265.1) and already aligned to
illecebrosus (§5.5), so the ingredients exist, but re-mapping and re-calling both
species against it is a project, not a re-run.

**A general lesson worth carrying:** any between-species comparison *inside* an
inversion inherits the reference's arrangement. Every statistic in §5–§8 that
compares AA against BB is safe, because the bias is common to both when they are
compared to each other. It is only comparisons to an **outgroup mapped on the
same reference** that break — which is exactly the µ-free calibration of §5.5,
and that number should now be treated as suspect inside the inversion.

## 8.13 Direct chr2-to-coindetii alignment: R withdrawn, inversion confirmed, polarization questioned **[M][W]**

`.tmp/mmcoin/run.sh` (minimap2 `-x asm10`, three chr2 regions against the
coindetii assembly, 8 min); check script `illex/scripts/mu_free_check.py`,
results in `results/illex/mu_free_check.txt`.

### 8.13.1 R is withdrawn — the denominator is not a stable quantity

R = dxy(AA,BB)/div(illecebrosus,coindetii) was computed with both terms inside
the inversion. The numerator is safe (a within-illecebrosus contrast). The
denominator is not, and not for the reason §8.12 suggested. Measured three ways:

| source | inversion body | collinear 10–30 Mb | collinear 85–115 Mb |
|---|---|---|---|
| AnchorWave substitutions/comparable bp | 0.0100 | **0.0340** | **0.0201** |
| direct minimap2, gap-compressed `de` | 0.0128 | **0.0361** | **0.0242** |
| aligned fraction of the query | **95.4%** | 69.6% | 87.9% |

**The two collinear controls disagree with each other by 70%.** Local divergence
to coindetii varies 2–3× along a single chromosome by either method, so it
cannot serve as a precise denominator anywhere.

| R computed with… | value |
|---|---|
| AnchorWave, inversion-internal (**published**) | 0.5137 |
| direct alignment, inversion region | 0.4032 |
| direct alignment, collinear | **0.1709** |

A **3× spread**. **R = 0.514 ± 0.015 is withdrawn.** The idea is still sound —
a ratio of two 2µT quantities removes µ — but it needs a divergence estimate
stable enough to divide by, and this one is not.

**[W] And my ascertainment explanation for it was wrong.** I proposed that the
inversion's low divergence came from alignment ascertainment (substitutions only
counted where alignment succeeds, alignment succeeding where divergence is low).
The direct alignment refutes it: the inversion aligns **better** than either
control (95.4% against 69.6% and 87.9%) and is *still* the least diverged. The
low divergence inside the inversion is real, not an alignment artifact. Why it
is low is unexplained.

### 8.13.2 The inversion is confirmed structurally, on a shared scaffold

All three query regions align overwhelmingly to the **same** coindetii scaffold
OZ346549.1 (99.7%, 99.2%, 99.9% of aligned bp), so scaffold orientation is
shared and strand is interpretable:

| region | aligned to OZ346549.1 | orientation |
|---|---|---|
| **inversion body** | 18,084,220 bp | **100.0% minus** |
| collinear 10–30 Mb | 13,835,158 bp | 94.2% plus |
| collinear 85–115 Mb | 26,376,254 bp | 99.8% plus |

**18.1 Mb aligning in the opposite orientation to its own flanks, on one
scaffold.** This is the cleanest structural confirmation of the inversion in the
project — direct assembly-to-assembly evidence, independent of PCA, LD, karyotype
clustering and Fst.

### 8.13.3 **[W] It appeared to put the polarization in question — RESOLVED in §8.14, the polarization STANDS**

The manuscript records **AA = derived (inverted), BB = ancestral** as
**[ESTABLISHED]** (AnchorWave coindetii↔illex MAF). Two observations now point
the other way:

1. The illecebrosus **reference** is inverted relative to coindetii across this
   region (§8.13.2), so the reference carries the *derived* orientation if
   coindetii is ancestral.
2. §8.12 found the reference leans **BB**-like inside the inversion
   (non-reference allele frequency at diagnostic sites: p_AA 0.515 vs
   p_BB 0.217).

Together those imply **BB is derived**, not AA.

**This is deliberately flagged rather than acted on**, because point 2 is weak:
a mean allele-frequency lean is not a clean arrangement assignment. If the
reference were cleanly AA we would expect mean p_AA ≈ 0 and p_BB ≈ 1 at
fixed-different sites; the observed 0.515/0.217 is a lean, not an assignment.

**Why it matters enormously if true.** §11 warns that getting the arrangement
labels backwards flips π_I/π_S from 0.744 to 1.344 "and reverses the conclusion,
silently". Every result in §7 and §8.6–8.8 assumes the derived arrangement is
the one at frequency 0.626 with *lower* diversity. If instead the derived
arrangement is at 0.374 with *higher* diversity, the entire model — the diversity
deficit, the recent-rise argument, the age — must be rebuilt. **No downstream
result should be revised until this is settled.**

**The clean test exists and is cheap:** `polarize/` holds est-sfs output with
inferred ancestral alleles. Ask which arrangement carries more ancestral states
at the arrangement-diagnostic sites. That is a direct answer and does not depend
on inferring what the reference individual's karyotype was.

## 8.14 **[W][RETRACTED]** Polarization "confirmed" — the test was CIRCULAR

> **This entire section is wrong. See §8.15.** The parsimony polarization it
> relies on used the illecebrosus REFERENCE base as the ancestral state wherever
> coindetii had no SNP — which is 69.2% of diagnostic sites. Since the reference
> haplotype IS one of the two arrangements, that scores "the reference's own
> arrangement is ancestral" by construction. The conclusion below is an artifact
> of that circularity, and so is the "two inversion events" result built on it.
> Retained only as a record of the error.

## 8.14 (retracted) Polarization confirmed, and the region carries TWO inversion events **[W]**

`illex/scripts/polarization_check.py`, results in
`results/illex/polarization_check.txt`. §8.13.3 raised a doubt about the
recorded **AA = derived / BB = ancestral**. The doubt is resolved: **the
recorded polarization is correct.** My suspicion was wrong.

### 8.14.1 Which arrangement carries ancestral alleles

At arrangement-diagnostic sites, the share where the **AA** majority allele
equals the ancestral base:

| \|p_AA − p_BB\| | est-sfs | coindetii parsimony |
|---|---|---|
| ≥ 0.3 | 31.7% (n=27,715) | 38.9% (n=44,089) |
| ≥ 0.5 | 23.0% (n=17,005) | 30.5% (n=23,779) |
| **≥ 0.7** | **19.0%** (n=9,840) | **25.0%** (n=13,582) |

**BB carries the ancestral allele 75–81% of the time**, and the signal
strengthens as the sites become more diagnostic — the right direction. So
**BB is ancestral and AA is derived, as recorded.**

Two independent polarizations agree. I had flagged est-sfs as circular for this
question, because its ingroup-frequency term should push it toward calling the
*commoner* arrangement's allele ancestral, and AA is commoner (0.626). That
would have inflated the "AA ancestral" column. It did not: est-sfs is *more*
extreme than parsimony (19.0% vs 25.0%), i.e. biased the other way. Either way
the **non-circular** method confirms the result on its own.

### 8.14.2 The reference genome carries BB, cleanly

§8.12 inferred this from a mean allele-frequency lean and I called it weak. It
is not weak when measured properly. At sites fixed-different between arrangements
(|p_AA − p_BB| ≥ 0.9, n = 3,504):

* the reference carries the **BB allele 86.5%** of the time
* mean p_AA = **0.829**, mean p_BB = **0.139** (p = non-reference allele frequency)

So the reference haplotype is BB — the **ancestral** arrangement.

### 8.14.3 Therefore the region holds two separate inversion events

Three solid facts now sit together:

1. The reference carries **BB** (§8.14.2).
2. **BB is ancestral** within illecebrosus (§8.14.1).
3. The reference's 60.5–79.5 Mb is **inverted relative to coindetii** — 100.0%
   minus strand over 18.1 Mb on a shared scaffold, against 94–100% plus for both
   flanking collinear controls (§8.13.2).

The *ancestral* illecebrosus arrangement is inverted relative to coindetii. That
cannot be explained by the polymorphic inversion, so **the orientation difference
between the species is a separate event** from the polymorphism segregating
within illecebrosus. Either an inversion fixed on one lineage after the split, or
the polymorphic inversion is nested inside an older interspecific rearrangement.

Which lineage it happened on cannot be settled here — that needs a third
outgroup. But the robust statement is that **chr2:60–80 Mb has inverted at least
twice**, which makes it an inversion hotspot rather than a single event, and is
consistent with the breakpoint reuse commonly reported for inversions.

### 8.14.4 What this changes

**Nothing downstream.** §8.13.3 warned that reversed labels would flip π_I/π_S
from 0.744 to 1.344 and silently invert every conclusion in §7 and §8.6–8.8. The
polarization is confirmed, so all of that stands unchanged: the derived
arrangement is the one at 0.626 carrying *less* diversity, and the recent-rise
argument, the age and the balancing-selection fit are unaffected.

What it adds is a new structural result (§8.14.3), and it removes a live risk
from the record.

## 8.15 **THE ARRANGEMENT LABELS ARE BACKWARDS** — BB is the inverted, derived one **[M][W]**

Raised by the user 2026-08-27 as the parsimonious alternative to §8.14's two
inversions, and confirmed on testing. **The recorded polarization is wrong, my
"confirmation" of it was circular, and every fit in §7 and §8.6–8.8 is built on
inverted targets.**

### 8.15.1 The circularity

§8.14 called a "coindetii parsimony" polarization non-circular. It was not.
Ancestral was taken as coindetii's base, and coindetii's base was taken as the
illecebrosus **REF** wherever coindetii had no SNP record. At diagnostic sites:

| | n | share |
|---|---|---|
| coindetii has a SNP (differs from the illex REF) | 4,318 | 30.8% |
| **coindetii matches the illex REF** | **9,680** | **69.2%** |

For that 69.2%, "ancestral = REF" is identical to "ancestral = whatever the
reference haplotype carries". The reference carries BB (§8.14.2, 86.5% at
fixed-different sites — that measurement stands), so the test scored BB as
ancestral **by construction**. est-sfs inherits the same outgroup alignment and
the same problem.

### 8.15.2 The non-circular test reverses the answer

Restricting to sites where coindetii carries the illecebrosus **ALT** allele —
so the ancestral state is defined without reference to the reference's own base
(n = 4,175, |p_AA − p_BB| ≥ 0.7):

| arrangement | carries the ancestral allele |
|---|---|
| **AA** | **63.0%** (2,631) |
| BB | 37.0% (1,544) |

**AA is ancestral; BB is derived.**

### 8.15.3 One inversion, and everything is consistent

The user's reading, now supported: the reference was assembled from an
**inverted** individual, so the reference itself is in the inverted orientation.
Then

* the reference carries BB (§8.14.2) ⇒ **BB is the inverted arrangement**
* AA carries ancestral alleles (§8.15.2) ⇒ **AA is the standard arrangement**
* the reference aligns 100% minus to coindetii (§8.13.2) ⇒ **because the
  reference is the inverted one**
* coindetii is non-inverted ancestral

**One inversion event.** §8.14.3's "chr2:60–80 Mb has inverted at least twice /
inversion hotspot" is **withdrawn** — it was an artifact of the circular
polarization. The 18.1 Mb orientation result itself stands and is still the
cleanest structural confirmation of the inversion; only its interpretation
changes.

### 8.15.4 The corrected targets, and what they do to the story

Ratios are mask-free, so these follow directly from the measured π and dxy with
the labels swapped (I = BB inverted, S = AA standard):

| | as fitted | **corrected** |
|---|---|---|
| p_inverted | 0.626 | **0.374** |
| π_I/π_S | 0.7368 ± 0.0263 | **1.357 ± ~0.048** |
| dxy/π_I | 1.8794 ± 0.0503 | **1.385 ± ~0.037** |
| equilibrium p/(1−p) | 1.674 | **0.597** |

**The central observation flips sign.** It was "the common arrangement carries
2.25× LESS diversity than equilibrium warrants" — a deficit, read as a recent
rise. It is now "**the rare derived arrangement carries 2.3× MORE diversity than
its frequency warrants**" — an excess. A rare, derived arrangement that is
*more* diverse than its frequency allows is the signature of an **old**
polymorphism, possibly one that was formerly commoner, not a recent sweep.

### 8.15.5 What is invalidated

**Invalid as they stand** — all fitted to (0.744, 1.846) at p = 0.626:
§7.2, §7.5 (age ~730 ky), §8.6 (explicit arrival, ~900 ky), §8.7 (drifting
dormancy, ~800 ky), §8.8 (balanced dormancy, s_dorm ≈ 3e-5). The age, the
trajectory family, the arrival time and the dormancy selection all have to be
refitted against the corrected targets.

**Probably survives, but must be redone rather than assumed:** §7.5.3's
neutrality exclusion. At x = 0.374 the conditional hitting time is
(4N/x)[x + (1−x)ln(1−x)] = **0.864·N** rather than 1.650·N — about half, but
still 0.5–5.9 M generations, so drift remains excluded on any plausible age.

**Unaffected:** the ANGSD/GL machinery (§8.5), the sweep-scan comparison
(§8.11.1, which is symmetric in labels), the genome-wide Fst scan (§8.11.2), the
gene content (§8.9–8.10), and the coalescent-model test (§3).

### 8.15.6 Two decisions needed before any rebuild

1. **Does the project's own polarization carry the same circularity?** The
   recorded "AA = derived" came from an AnchorWave coindetii↔illex MAF pipeline.
   If that pipeline also used the reference base as the ancestral state where the
   outgroup matched, it has the identical flaw, and §4's [ESTABLISHED] tag should
   come off.
2. **Swap the cluster labels, or only their interpretation?** Every downstream
   file keys off `AA_samples.txt` / `BB_samples.txt`. Renaming is invasive;
   keeping the names and inverting the interpretation is safer but is exactly
   the setup §11 warns silently reverses conclusions.

## 8.16 REFIT under the corrected polarization — a formerly common arrangement in decline **[M]**

`illex.balancing.decline_curve` / `build_decline_sim`,
`illex/scripts/refit_decline.py`, results in
`results/illex/refit_decline_*.{csv,txt,json}`. Decision of 2026-08-27: keep the
cluster labels, swap the interpretation, refit everything.

### 8.16.1 Corrected targets

Re-derived with I = BB (differentiated body, 1 Mb block jackknife):

| | old (wrong) | **corrected** |
|---|---|---|
| p_inverted | 0.626 | **0.374** |
| π_I/π_S | 0.7368 ± 0.0263 | **1.3556 ± 0.0481** |
| dxy/π_I | 1.8794 ± 0.0503 | **1.3848 ± 0.0214** |
| ANGSD f₁(I)/f₁(S) | 1.211 | **0.8256** |

### 8.16.2 Why every earlier trajectory family is structurally incapable

Two analytic facts, established before running anything:

1. A long-standing balanced polymorphism at p = 0.374 gives π_I/π_S → p/(1−p)
   = 0.597. Observed is **1.356 — 2.27× higher.** The inverted class carries far
   more diversity than its *current* frequency can sustain, so it must have been
   **commoner in the past**. A constant frequency reproducing 1.356 is p = 0.576.
2. Every family in §8.6–§8.8 rises to an equilibrium and stays. None can place
   the inverted class *above* its equilibrium diversity, because in none of them
   was it ever commoner than now.

Hence a **decline** family: held at `p_hist`, then falling to 0.374. The fall
reuses the same overdominance ODE run toward a *lower* equilibrium, so it is the
mirror of the rise rather than a new mechanism — an environmental shift moving
the optimum down instead of up.

**[W] One analytic step of mine was wrong and the simulation caught it.** I
argued dxy/π_I ≥ 1 + T_anc/t_inv, hence t_inv ≥ 2.85 My. That treats π_I as
growing like 2µ·t_inv, but π_I *saturates* at the inverted class's own
equilibrium, after which extra t_inv only inflates dxy. The pilot showed
dxy/π_I *rising* with t_inv (3.84 at 3 My, 6.67 at 6 My) — the opposite of the
predicted direction. The bound was inverted; the informative range is
t_inv ≈ 10⁵–10⁶, not ≥ 2.85 My.

### 8.16.3 The fit

3,000+ sims, 48–64 reps/cell, t_fall fixed at 100 ky.

| | π_I/π_S | dxy/π_I | f₁ ratio | score |
|---|---|---|---|---|
| target | 1.3556 | 1.3848 | 0.8256 | — |
| t_inv 850 ky, p_hist 0.70, t_decline 175 ky (t_fall fixed 100 ky) | 1.3160 (−2.9%) | 1.4087 (+1.7%) | 0.8269 (+0.2%) | 0.00115 |
| **+ t_fall fitted at 50 ky (§8.18)** | **1.3396 (−1.2%)** | **1.3992 (+1.0%)** | **0.8465 (+2.5%)** | **0.00089** |
| t_inv 800 ky, p_hist 0.74, t_decline 175 ky | 1.3868 (+2.3%) | 1.3435 (−3.0%) | 0.8165 (−1.1%) | 0.00150 |

All three statistics inside 3%. `p_hist` is an **interior** optimum (0.66 → 0.0101,
0.70 → 0.0012, 0.74 → 0.0039); `t_inv` sits at the upper edge of the final grid,
though the wider scan shows 1 My is clearly worse (0.0177), so the optimum lies
between.

**Current best picture: the inversion arose ~850 ka, was maintained near
p ≈ 0.70 for most of its history, and has been declining since ~175 ka, reaching
0.374 today.**

### 8.16.4 The age survived the reversal, and there is a reason

t_inv ≈ 800–850 ky — **essentially identical to every pre-correction fit**
(§8.7, §8.8 both gave 800 ky), despite the story inverting completely.

That is not luck. **dxy(AA,BB) is label-symmetric** — it is the divergence
*between* the two classes and does not care which is called inverted — and dxy
is what carries t_inv. The reversal changed only its normalisation (dividing by
π_BB instead of π_AA). So the age was always the best-determined quantity here,
and it is the one result that survives the polarization error intact.

### 8.16.5 What the story is now, versus what it was

| | before (wrong polarization) | **after** |
|---|---|---|
| derived arrangement | AA, common (0.626) | **BB, rare (0.374)** |
| its diversity | 2.25× *deficit* vs equilibrium | **2.27× *excess*** |
| reading | young, recently swept up, bottlenecked | **formerly common (~0.70), now declining** |
| t_decline / t_arrive | arrived ~200 ka | **decline began ~175 ka** |
| age | ~800 ky | ~800–850 ky (unchanged) |

The balancing-selection interpretation is if anything **strengthened**: an
850 ky polymorphism sitting at intermediate frequency for most of its history is
far too old and too stable to be drifting. What has changed is the direction of
the recent dynamics — the arrangement is on its way down, not up.

### 8.16.6 Caveats

1. `t_fall` is fixed at 100 ky, not fitted, and is likely degenerate with
   `t_decline`.
2. Three parameters against three targets — exactly identified, so no residual
   degrees of freedom test the model, and there are no error bars yet.
3. §7.5.3's neutrality argument must be redone at x = 0.374, where the
   conditional hitting time is **0.864·N** rather than 1.650·N. Still 0.5–5.9 M
   generations, so drift is very likely still excluded, but it has not been
   rerun.
4. The decline is a phenomenological trajectory shape; no mechanism is fitted,
   and overdominance-with-a-moving-optimum is one of several processes that
   would produce it.
5. µ still scales every age inversely and remains the dominant uncertainty.

## 8.17 Neutrality rerun at the corrected frequency **[M][W]**

`illex/scripts/neutrality_check.py`, results in
`results/illex/neutrality_check.txt`. §7.5.3 excluded drift using p = 0.626. The
polarization is reversed and the refit (§8.16) says the arrangement did not rise
to its present 0.374 — it **declined to it from p_hist ≈ 0.70 over ~100 ky
ending ~175 ka**. That changes the argument's structure, and splits it in two.

### 8.17.1 Attainment — could drift reach the frequency it actually attained?

The target is the **highest** frequency inferred, 0.70, not the present 0.374: a
neutral allele has to get up there before it can come back down.

| Ne | E[t \| reach 0.374] | E[t \| reach 0.70] |
|---|---|---|
| N_ANC = 547,928 | 473,378 | **1,060,814** |
| N(275 ka) = 2.77 M | 2,390,251 | 5,356,424 |
| N₀ = 6,808,096 | 5,881,796 | 13,180,791 |

Coefficients: **0.864·N** at x = 0.374, **1.936·N** at x = 0.70 (1.650·N at the
old, wrong 0.626). P(a new neutral inversion ever reaching 0.70) = 1.3e-6 to
1.1e-7.

Against a fitted age of 850 ky, the binding case — smallest Ne — needs
**1.25× the entire lifetime of the inversion**, widening to 16× at N₀.

### 8.17.2 The decline — a test that did not exist before

Under the old reading the arrangement was rising, so its current frequency was
the only thing to test. Now there is a second, independent constraint: a fall of
**0.326** in ~100 ky, against the neutral drift SD over that window
(N = 2.77–3.84 M):

| assumed t_fall | drift SD | observed fall |
|---|---|---|
| **100 ky** (the fitted value) | 0.0569 | **5.7 SD** |
| 200 ky | 0.0879 | 3.7 SD |
| 500 ky | 0.1860 | **1.8 SD** |

~~**This is where the argument is soft.**~~ **RESOLVED in §8.18** — `t_fall` has
now been fitted, and the data exclude the slow fall that would have let drift
work. Across the well-fitting range the decline is **5.7–12.2 SD**.

### 8.17.3 **[W] §7.5.3 overstated the margin, and the polarization is not why**

§7.5.3 said drift was excluded "by an order of magnitude". That figure came from
the **largest**-Ne case. The binding case is the smallest Ne, and there:

| | hitting time at N_ANC | fitted age | margin |
|---|---|---|---|
| old reading (x = 0.626) | 903,893 | 720,000 | **1.26×** |
| corrected (x = 0.70) | 1,060,814 | 850,000 | **1.25×** |

**The margin is unchanged.** Per unit Ne the requirement rose (1.936·N vs
1.650·N, since the arrangement reached a higher frequency), but the fitted age
rose too, and the two nearly cancel. So the attainment test is neither stronger
nor weaker than it was — it was simply always narrower than §7.5.3 implied.

### 8.17.4 Verdict

**Neutrality is still excluded, on two independent and calibration-free grounds
— but not by an order of magnitude, and that should be stated plainly.**

* attainment: 1.25× the inversion's lifetime at the most favourable Ne, up to
  16× at the least favourable
* decline: 5.7 SD at the fitted t_fall, but only 1.8 SD if the fall took 500 ky

Neither uses µ, the accessibility mask, or a simulation, which is their value.
Both are diffusion results for an unlinked neutral allele: they bound **drift**,
not selection at linked sites, and they take the fitted p_hist and t_fall as
given.

**The cheapest way to firm this up is to fit t_fall** instead of fixing it. If
the data prefer a fast fall the decline test becomes the strongest argument in
the analysis; if they tolerate a slow one, the case rests on attainment alone at
a 1.25× margin.

## 8.18 t_fall fitted, not fixed — the decline test survives **[M]**

`illex/scripts/refit_decline.py --t-fall ...`, results in
`results/illex/refit_decline_tfall.{csv,txt,json}`. §8.17.2 flagged the decline
test as resting on a `t_fall` that had been **fixed** at 100 ky rather than
estimated: the fall is 5.7 SD of drift at 100 ky but only 1.8 SD at 500 ky,
which drift produces ~4% of the time. So `t_fall` was scanned.

### 8.18.1 It can only be profiled, and that is stated up front

Four parameters (t_inv, p_hist, t_decline, t_fall) against three targets, so
`t_fall` is not point-identified. What the scan delivers is a **profile**: the
best achievable fit at each `t_fall` with the other three re-optimised.
4,320 sims, 48 reps/cell.

Scores are converted to a χ² using the jackknife SEs on the two ratios (the
ANGSD f₁ ratio has no SE and is reported separately):

| t_fall | χ² (2 df) | Δχ² | f₁ miss | drift SD of the fall | |
|---|---|---|---|---|---|
| 25,000 | 0.20 | 0.00 | +4.2% | **12.2 SD** | best χ² |
| **50,000** | 0.56 | 0.36 | +2.5% | **8.5 SD** | **best combined** |
| 100,000 | 1.88 | 1.68 | +0.1% | **5.7 SD** | ok |
| 200,000 | 3.14 | 2.93 | −1.1% | 4.9 SD | disfavoured |
| 400,000 | 10.84 | 10.64 | −5.2% | 2.9 SD | **excluded** |

The two ratios pull toward a fast fall and the SFS shape toward ~100 ky; 50 ky
is the compromise and wins on the combined score.

### 8.18.2 The answer: the data require a fast fall

**Well-fitting range (Δχ² < 2): t_fall = 25–100 ky**, over which the decline is
**5.7–12.2 SD** of pure drift. The 400 ky case — the one where the decline test
collapsed to 1.8 SD and drift became viable — is **excluded at Δχ² = 10.6**
(~2.5σ+), and 200 ky is already disfavoured.

**So §8.17.2's soft spot is closed.** The decline test does not rest on an
assumption: the data themselves rule out a fall slow enough for drift to
produce. It is now the **strongest** of the two neutrality arguments, since the
attainment test sits at only a 1.25× margin at the binding Ne.

### 8.18.3 Updated best fit and the implied selection

**t_inv = 850 ky, p_hist = 0.70, t_decline = 175 ky, t_fall ≈ 50 ky**
→ π_I/π_S 1.3396 (−1.2%), dxy/π_I 1.3992 (+1.0%), f₁ ratio 0.8465 (+2.5%).

The fall implies a selection coefficient against the inverted arrangement of

| t_fall | s_fall | Ne·s at mid-fall |
|---|---|---|
| 25 ky | 5.7e-4 | 2,109 |
| **50 ky** | **2.9e-4** | **1,012** |
| 100 ky | 1.4e-4 | 466 |

Ne·s of order 10³ — strong, unambiguous selection, and 6–20× the s_het ≈ 3–4e-5
that the (superseded) pre-correction fits inferred for the rise. The arrangement
is not drifting down; it is being removed.

### 8.18.4 The picture, and what is still assumed

**The inversion arose ~850 ka, was maintained near p ≈ 0.70 for ~600 ky, and
then fell sharply to 0.374 in roughly 50 ky ending ~175 ka, under selection of
order Ne·s ~ 10³.** A long-balanced polymorphism whose equilibrium moved, and
which is still on its way down.

Still assumed: `t_decline` and `t_fall` are partly degenerate (the 200 ky row
re-optimises to a different t_decline and p_hist); the decline shape is
phenomenological, with no mechanism fitted; there are still no joint error bars;
and µ scales every age inversely.

## 9. Where identification has to come from

Given §5.3 (Fst redundant) and §8.3 (absolute levels need a nuisance parameter),
the genuinely independent constraints are:

| Statistic | Status | Calibration-free? |
|---|---|---|
| π_I/π_S, dxy/π_I | fitted targets | yes |
| Windowed spatial dxy profile | **spent** — used to kill flux (§6) | yes |
| ~~Within-arrangement SFS from called genotypes~~ | **fails** — depth-driven (§8.4) | unusable |
| **Per-karyotype ANGSD/GL SFS** | **WORKS** (§8.5): confirms the soft origin, confirms the §7.5.2 mechanism, and constrains the arrival time | **yes** |
| Absolute π_I, π_S, dxy | usable with a scale nuisance | no |
| r²-vs-distance decay | **blocked**, see below | yes but length-dependent |
| ~~ReLERNN interior rate~~ | **withdrawn** — no barrier signal (§8.0) | — |
| *I. argentinus* presence/absence | **blocked by coverage, not by biology** — see below | n/a |

**[W] The within-arrangement SFS shape was the recommended addition, and it does
not work.** The reasoning was sound — normalized, mask-free, and sensitive to
p_start differently from mean π — but the estimator is defeated by the callset's
variable depth, and the neutral baseline misses the collinear spectrum by more
than the inversion signal. Full post-mortem in §8.4. The replacement is a
per-karyotype ANGSD/GL spectrum off the BAMs, **which has now been run — see
§8.5.** It works, and it independently confirms the soft origin and the §7.5.2
mechanism, but it still does not break the (p_start, plateau) ridge, so both
must be quoted as a joint range.

**The r² comparison is blocked.** Only 5 of 40 control windows density-match the
inversion body, and all 5 sit in a single ~2.5 Mb span, which cannot contain
marker pairs at the ~20 Mb separations probed inside the inversion. This is the
wrong *shape* of evidence, not merely underpowered, and the density distribution
is bimodal so the count is threshold-insensitive. **A differently located or
substantially larger control region is required.**

***I. argentinus*: potentially the most informative comparison available, and
blocked only by sequencing depth.** If argentinus lacks the inversion,
t_inv < the species split; if it shares it, t_inv > split. **[W]** Two earlier
statements about it were wrong and are corrected here:

- `analysis/steps/08_argentinus` is an **empty scaffold** (a bare `logs/`). The
  data is in `polarize/argentinus_input/` and `mkado_illex/inputs/`, 10 samples.
- I claimed the argentinus split was ≳2 My, ~2.7× the inversion's age, hence
  the bracket could not bite. That rested on a coindetii split of ~7 My, which
  was itself wrong by 2.7× because of the denominator error in §5.5.1. Corrected,
  the coindetii split is ~2.47 M generations, and scaling by CDS divergence
  (argentinus ≈0.38× coindetii) puts the argentinus split **plausibly at or
  below the inversion's age** — which is exactly the regime where the bracket is
  most informative. The timescale objection is **withdrawn**.

What actually blocks it is coverage. Across the 20 Mb region the whole-genome
argentinus callset has 169,043 records but each sample calls only ~30,000 of
them (~18%), and the overlap is negligible: **102 sites have ≥8 of 10 samples
called**, and 43 have ≥5 in the CDS subset. That is not enough to karyotype
argentinus, run a regional PCA, or measure arrangement-specific divergence.

So this is a **data-acquisition recommendation, not a dead end**: moderate-depth
resequencing of a modest argentinus panel would deliver a hard, µ-free bracket
on the inversion's age, and it is the only identified route to one besides a
better µ.

---

## 10. What is excluded, and what remains open

**Excluded by evidence:**
- Neutrality of the inversion — the frequency cannot be reached by drift in the
  time the divergence allows (§7.5.3)
- A long-established balanced polymorphism — a long plateau forces
  π_I/π_S > 1 against the observed 0.744 (§7.5.2)
- Multiple-merger (Beta) coalescent — every α fits worse than Kingman+growth (§3)
- Constant Ne — 2× singleton deficit (§3)
- Gene flux as an explanation of the divergence pattern — no spatial gradient (§6)
- Strict single-founder monophyly (k = 1) — cannot reach π_I/π_S = 0.744 (§7.1)
- Constant-p_inv / multi-background origin (k → ∞) — cannot go below 1.0 (§7.1)

**Open:**
- ~~The −9.4% / +6.5% residual~~ — **closed** (§7.5). It was a model-shape
  issue, as diagnosed: the trajectory was still rising when it should have been
  settling into an equilibrium.
- ~~Selection versus neutrality~~ — **settled analytically** (§7.5.3). Neutral
  drift needs 1.650·N_e generations to reach 0.626 conditional on getting there,
  i.e. 0.9–11 M generations, against a fitted age of ~7.2e5. No forward
  simulation was required, and the argument is free of µ and of the mask. The
  original obstacle (a neutral trajectory is not samplable at Illex N_e) was
  real but was the wrong tool for the question.
- Why p_inv sits at an intermediate 0.626. §7.5 *assumes* balancing selection
  and shows it fits; it does not discriminate overdominance from associative
  overdominance, frequency-dependent selection or antagonistic pleiotropy. What
  the data constrain is the *shape* of p(t) — a rise settling into a plateau —
  not the mechanism.
- **(p_start, s_het) are jointly degenerate** (§7.5.1). Breaking this needs a
  statistic sensitive to the founding haplotype count independently of mean π —
  the within-arrangement SFS shape (§9).
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

## 8.19 An error bar on f1 -- and the control fails its own check (2026-08-28)

**Why.** The decline fit is scored against three statistics. Two carry 1 Mb
block-jackknife SEs (3.5% on pi_I/pi_S, 1.5% on dxy/pi_I). The third, the ANGSD
singleton ratio f1(I)/f1(S) = 0.8256, had none. That gap became binding because
the question "is the decline still ongoing?" is answered almost entirely by f1:
across the grid, moving t_decline 175 -> 100 ka moves pi_I/pi_S by +2.5% but f1
by +4.6%. A delta-chi-square on t_decline is uninterpretable without a variance
on f1.

**Method.** `realSFS -r` takes one contiguous region, so a true leave-one-out
(span minus a hole) cannot be requested. Instead each 1 Mb block was run
separately off the existing chr2 SAFs (78 runs, ~25 min) and delete-one-block
spectra formed by summing the others. realSFS optimises by EM over its whole
region, so per-block estimates need not sum to the global one -- this is GATED
in `sfs_f1_jackknife.py` and the script refuses to report an SE if they
disagree by >2%. **Gate passed, worst 0.06%.**

**Result.**

| arm | f1(BB) | f1(AA) | ratio f1(I)/f1(S) |
|---|---|---|---|
| body 60.5-79.5 Mb | 0.4973 +- 0.0030 | 0.6022 +- 0.0070 | **0.8256 +- 0.0076** (0.9%) |
| collinear control | 0.5983 +- 0.0034 | 0.5888 +- 0.0035 | **1.0161 +- 0.0011** |

**THE CONTROL FAILS.** AA and BB are exchangeable outside the inversion, so the
control ratio must be 1. It is 1.6% off -- small, but the jackknife SE is
smaller still, so it sits **15 SE from 1**. The offset is near-identical in
every block, which is exactly why a spatial jackknife cannot see it: it is a
systematic property of the two SAMPLE SETS, not of the genome. Prime candidate
is n = 254 vs 95, so realSFS's EM estimates the two source spectra with
different bias before either is projected to n = 20; coverage differences
between the bamlists would do the same. **Not diagnosed** -- confirming it needs
an AA SAF rebuilt at n = 95 over the control region, a new ANGSD run.

**The fix does not require the diagnosis.** The model side has n_i = n_s = 100
and carries no class-size asymmetry; the empirical ratio carries one. Dividing
body by control cancels any class-level systematic shared across regions,
whatever its cause -- the same logic already used for the per-class
body-vs-control comparison in sec 8.5.

    calibrated target = 0.8256 / 1.0161 = **0.8125 +- 0.0076**

The assumption doing the work is that the offset is the same in both regions
(same individuals, same bamlists, same n). Plausible, but an assumption.
`empirical.SFS_F1_RATIO_BODY` updated 0.8256 -> 0.8125. Per-class
body-vs-control L1 values are unaffected -- they compare each class to itself,
so the systematic already cancels there.

**Consequence for the fit.** Rescoring the ~130 existing decline cells against
the calibrated target (no new sims -- the statistics are stored) moves the best
fit from `t_inv 850 ka / p_hist 0.70 / t_decline 175 ka / t_fall 50 ky`
(score 0.00089) to `800 ka / 0.74 / 100 ka / 200 ky` (score 0.00084). **The
improvement is negligible and the top five cells span t_inv 800-850 ka, p_hist
0.70-0.74, t_decline 100-175 ka, t_fall 100-200 ky.** So the calibration moves
the point estimate *within an already-flat ridge* rather than relocating the
fit. The age is stable at ~800-850 ka; (t_decline, t_fall) remain unidentified.

## 8.20 Can we test whether the decline is still ongoing? (2026-08-28)

Three routes, only one of which is a new computation.

**1. Temporal sampling -- dead.** At the fitted s ~ 2.9e-4 the frequency drops
0.0034 per 50 generations (Illex is annual/semelparous, so generations =
years), against a sampling SE of 0.0136 at n = 633. A two-sample test needs
**~570 years** of separation. Decadal archives and fishery time series have no
power. Do not pursue.

**2. The transit window -- free, and most of the answer.** At the fitted s the
arrangement crosses p in [0.30, 0.45] in **2,230 generations = 0.26% of its
850-ky history**, and runs 0.70 -> 0.05 in **13 ky**. Catching it mid-fall is
~1-in-400. For the observation to be unremarkable (>=5-10% of the history in
that band) the decline would need s <= 8e-6 - 1.5e-5, i.e. **Ne*s ~ 50-100
rather than 10^3, a 20-100x weaker process**. Put differently: an ongoing
decline at the FITTED rate implies p ~ 0.999 only 25 ky ago, irreconcilable
with a 0.70 plateau. So the strong selection that produced the fall is excluded
as an ongoing process. A slow decline is not excluded -- and one that slow is
close to indistinguishable from drift.

**3. The t_decline profile -- queued 2026-08-28 19:00, 72 cells, ~4 h.**
t_decline over {0, 25, 50, 100, 175, 250} ka with t_inv, p_hist and t_fall all
free to re-optimise at each value (`.tmp/queue_tdecline.sh`).

**WHAT IT DOES AND DOES NOT TEST -- a correction to what I first proposed.** I
claimed a t_decline -> 0 profile would bound the ongoing case from the steep
side. It does not. `decline_curve` drives p toward p* = p_now, so it ASYMPTOTES
onto 0.374: even at t_decline = 0 the frequency is 0.377 at 50 ka and flat
across the last 25 ky. The family cannot express "p passing THROUGH 0.374 on
the way to a lower target". So the profile answers **"how recently could the
fall have ended"**, not "is p changing now". A profile that localises t_decline
away from 0 says the fall is over; a flat one says we cannot date its end.
Whether p is changing today is answered by (2), not by this run.

## 8.21 The sweep scan is karyotype-agnostic -- a withdrawal (2026-08-28)

Asked whether selection was ever examined without reference to karyotype. It
was, in one place, and I had read too much into it.

**What is stratified:** pg_gpu diversity/dxy/FST (sec 8.3), the ANGSD SAFs and
within-arrangement SFS (sec 8.5), the genome-wide FST(AA,BB) scan, the f1
jackknife (sec 8.19). All per-arrangement by construction.

**What is not:** the diploSHIC/RAiSD sweep scan -- pooled n = 350, no
arrangement stratification. The manuscript's sec 8 already flagged "chr2 ...
its calls are inversion-confounded", but sec 3l never said what the confound
was.

**The confound.** Inside the inversion the pooled sample is a MIXTURE of two
classes at FST = 0.365. That mechanically inflates pi, pushes Tajima's D up,
and presents two haplotype backgrounds. So sec 3l(a)'s "hard depleted 11-fold,
soft mildly enriched" is largely what pooling two divergent arrangements looks
like -- the same structure FST already reports -- **not** the "independent
support for balancing selection from a method sharing nothing with the
coalescent modelling" I wrote. **Withdrawn.** What survives: no hard-sweep
signature in the pooled data, one-sided, since a sweep confined to one
arrangement would be diluted by the other.

**The gap this exposes.** Selection WITHIN each arrangement has never been
scanned. That is where a supergene's adaptive content would appear, and it is
the only thing a sweep scan could add beyond the FST scan. Cost: diploSHIC's
theta_W/D/dist* features are sample-size dependent (the reason chrZ needed its
own n = 330 model), so this needs models retrained at n = 254 (AA) and n = 95
(BB); whether n = 95 suffices for the CNN is untested. chrZ is the precedent.

## 8.22 The t_decline profile: the fall is OVER (2026-08-28, 72 cells)

Ran `.tmp/queue_tdecline.sh` -- t_decline over {0, 25, 50, 100, 175, 250} ka
with t_inv, p_hist and t_fall all free to re-optimise at each value.

**Scored properly for the first time.** All three targets now have SEs (sec
8.19 gave f1 its first), so the fit is a weighted chi-square rather than the
unweighted sum-of-squared-relative-misses used until now:

    pi_I/pi_S  1.3556 +- 0.0481  (3.55%)
    dxy/pi_I   1.3848 +- 0.0214  (1.55%)
    f1(I)/f1(S) 0.8125 +- 0.0076 (0.94%)   <- much the tightest, so it dominates

Weighting changes the answer: the unweighted score picked
`800 ka / 0.74 / 250 ka / 50 ky`, the weighted chi-square picks
**`t_inv 850 ka / p_hist 0.74 / t_decline 175 ka / t_fall 100 ky`, chi2 = 2.62**
(pi +1.4 SE, dxy +0.5 SE, f1 -0.6 SE). Note p_hist = 0.74 is now preferred over
0.70 -- it appears in 7 of the 8 cells within chi2_min + 2.

**THE PROFILE.**

| t_decline | min chi2 | delta chi2 | at |
|---|---|---|---|
| **0 (fall ends now)** | 38.07 | **+35.45** | 850 ka / 0.74 / 200 ky |
| 25 ka | 19.70 | +17.08 | 850 ka / 0.74 / 200 ky |
| 50 ka | 5.79 | +3.17 | 850 ka / 0.74 / 200 ky |
| 100 ka | 3.40 | +0.78 | 800 ka / 0.74 / 200 ky |
| **175 ka** | **2.62** | **0.00** | 850 ka / 0.74 / 100 ky |
| 250 ka | 3.47 | +0.85 | 800 ka / 0.74 / 50 ky |

**The fall ended >= ~50-100 ka. "It ended now" is excluded at delta chi2 = 35.**
The profile is flat from 100 to 250 ka -- we cannot date the end more precisely
than that -- but it rises steeply toward 0. This is the outcome sec 8.20 called
the informative one: t_decline is localised well away from 0.

**Robustness, because f1 does most of the work.** The f1 SE is a MEASUREMENT
error and a lower bound (blocks inside a non-recombining inversion share one
origin), so the profile was recomputed with it inflated:

| f1 SE x | 0 | 25 ka | 50 ka | 100 ka | 175 ka | 250 ka |
|---|---|---|---|---|---|---|
| 1 (0.0076) | 35.4 | 17.1 | 3.2 | 0.8 | 0 | 0.9 |
| 2 (0.0152) | 10.8 | 6.8 | 3.0 | 0.5 | 0 | 0.5 |
| 3 (0.0228) | 4.1 | 2.5 | 1.0 | 0.6 | 0 | 0.4 |
| 5 (0.0380) | 0.7 | 0.4 | 0.0 | 0.6 | 0 | 0.4 |

**The conclusion survives a 2x inflation, weakens at 3x, and dies at 5x.** So it
holds unless the true uncertainty on f1 is >= 3x its measurement SE. It is also
not purely an f1 result: at the best t_decline = 0 cell the pi ratio is +2.8 SE
off on its own (vs +1.4 SE at the optimum), so ~6 units of the delta chi2 come
from pi alone.

**WHAT THIS DOES AND DOES NOT SAY -- carried forward from sec 8.20.** It says
the fall finished long ago, not that the frequency is static today.
`decline_curve` asymptotes onto p_now, so even t_decline = 0 means "just
arrived", not "still falling"; the family cannot express p passing THROUGH
0.374. Whether p is changing now is answered by the transit-window bound in sec
8.20 (an ongoing decline must be 20-100x weaker than the fitted one), not by
this profile. The two agree: the strong fall is over.

## 8.23 Per-site FST off the called VCF FAILS its control (2026-08-28)

Asked for high-FST SNPs between karyotypes, the genes, and codon positions.
Built it from the callset (`fst_snp_coding.py`) and it does not work.

**The control kills it.** In the collinear region AA and BB are exchangeable,
so there should be essentially no differentiated sites. Instead: **267 apparent
FIXED differences and 0.25% of SNPs at |dp| >= 0.8**, against 171 and 0.75% in
the body. A 3x body/control ratio where FST differs 100-fold (0.365 vs 0.0035)
means the tail is noise-dominated in both.

**Diagnosis: structured missingness.** NONE of the apparent fixed differences,
in either region, survives AN >= 75% of maximum -- 0 of 205 (body), 0 of 339
(control). A site reads as fixed exactly when few genotypes are called.

**Fourth occurrence of one lesson.** Called-genotype SFS (sec 8.4), argentinus
sufficiency (sec 8.12), the polarization subset (sec 8.15), now per-site FST.
**Standing rule: do not read per-site allele frequencies off this callset.**

**What survives, because it does not need individual sites to be right:** the
differentiated tail is overwhelmingly NON-CODING -- 43 of 11,381 sites at
dp >= 0.8 are in CDS (0.4%) -- and shows no 0-fold enrichment (OR 0.89-1.30,
p >= 0.57, n = 20-43). Zero fixed differences fall in CDS in either region.
Consistent with a gene-poor block; underpowered rather than a strong negative.

## 8.24 The unfolded AA x BB 2D SFS -- the same question, answered (2026-08-29)

`realSFS -fold 0` on fresh n=40 SAFs for body and control (2 h and 2.5 h EM;
the full-n 509x191 version failed back in July). `sfs2d_karyotype.py`.

| as % of variable sites | body | control | ratio |
|---|---|---|---|
| shared polymorphism | 21.40% | 47.16% | **0.45** |
| AA-private | 41.40% | 25.96% | 1.59 |
| BB-private | 37.20% | 26.88% | 1.38 |
| near-fixed AA-ALT/BB-REF | 0.196% (1,950) | **0.000% (0)** | inf |
| near-fixed AA-REF/BB-ALT | 0.027% (269) | **0.000% (0)** | inf |

1. **The control is spotless** -- every near-fixed cell exactly zero, 3,229 of
   6,561 cells empty. Best control pass in the project, and what licenses
   reading the body at all. Contrast sec 8.23, where the same question off the
   called VCF failed this test outright.
2. **No exact fixed differences anywhere -- and that is EXPECTED.** dxy/pi_I =
   1.385 puts between-class divergence at ~1.4x within-BB diversity, nowhere
   near reciprocal monophyly, so complete sorting across 160 chromosomes should
   be rare. Not a null result about the barrier.
3. **Near-fixed differentiation is real: 2,219 sites in the body, ZERO in the
   control.** This is the "are there high-FST SNPs" answer, and only the GL
   route could give it.
4. **Clean barrier signature:** shared polymorphism collapses 47% -> 21% while
   private variation rises in both classes. Symmetric in AA/BB, so the least
   reference-sensitive statistic here.
5. **The 7.2:1 directional lean is NOT biology.** The reference IS a BB
   haplotype inside the inversion, so BB-derived variants shared with the
   reference are invisible by construction. The control cannot remove this --
   outside the inversion the reference carries no arrangement identity. Counts
   are LOWER BOUNDS with a known lean. (This is the sec 8.13 equidistance trap
   in a new place.)
6. **AA is not lost to mapping bias:** AA-private (41.4%) exceeds BB-private
   (37.2%) in the body; severe reference-mapping loss would deplete AA.

**LIMITATION:** a 2D SFS carries counts, not positions -- it cannot say which
genes the near-fixed sites are in. That needs per-site output
(`realSFS fst print`) intersected with degenotate. Natural follow-on, and the
only route to the gene/codon half of the original question that has a control
behind it.

## 8.25 Per-site GL FST x degenotate: the GL route works, the FST STATISTIC does not (2026-08-29)

Ran `realSFS fst index/print` off the n=40 SAFs and the unfolded 2D SFS, then
intersected with degenotate. `fst_persite_degen.py`.

**The data are sound this time.** Region FST body **0.3228** vs control
**0.0053** -- the control passes outright, unlike the called-VCF attempt in sec
8.23. 9,128,070 body sites and 12,027,890 control sites with FST defined.

**But the statistic is not.** Hudson FST = sum(A)/sum(B), and at a
near-monomorphic site both go to ~0 while the numerator carries the negative
sampling corrections -p(1-p)/(n-1). The ratio is therefore dragged toward 0
wherever diversity is low -- and CDS is mostly constrained.

**How it was caught.** Transcripts reading FST ~ 0.012 sit inside 100 kb windows
at FST 0.37-0.46. That cannot happen in a non-recombining block. corr(transcript
FST, window FST) is only +0.20 (Spearman +0.16).

**Confirmed:**

    Spearman corr(per-transcript FST, polymorphism per site) = +0.694
    Q1 least polymorphic  median FST 0.0199
    Q2                                0.1534
    Q3                                0.2741
    Q4 most polymorphic               0.2942

**Both answers are therefore withdrawn:**
* **The gene ranking is invalid** -- it orders transcripts by how polymorphic
  they are. (For the record, the ranking it produced was cytochrome P450 4X1,
  cysteine-rich venom protein, PNPLA-domain phospholipases, malonyl-CoA-ACP
  transacylase -- a lipid/metabolism flavour that echoes the sec 8.10 GO story,
  which was ALSO a null artifact. Do not cite either.)
* **The 0-fold depletion in the high-FST tail is the same artifact from the
  other side** (OR 0.18-0.41, p ~ 1e-34): 0-fold sites are constrained, hence
  less polymorphic, hence pushed out of the tail. **The CONTROL shows the same
  direction** (OR 0.44-0.81, p 4e-05), which settles it as generic.

**Conditioning does not rescue it.** At B >= 0.25 the bias is gone but only 220
coding sites remain in the body and 5 in the control -- no test, and zero
transcripts with enough sites. The block is gene-poor (95 scored transcripts),
CDS is mostly constrained, and n = 40/class is a small subsample.

**One positional worry ruled out:** the top transcripts are NOT a linkage
cluster. They span 60.91-79.31 Mb of the 19 Mb block and corr(FST, position) is
-0.03. The consecutive LOC_000053xx IDs just reflect gene numbering along the
chromosome.

**RECOMMENDED FOLLOW-ON: per-transcript dxy, not FST.** dxy is an absolute
divergence rather than a ratio, so it has no small-denominator pathology. It
needs no new ANGSD run. Until then, **the gene/codon half of the question is
unanswered.**

## 8.26 Per-transcript dxy: no gene stands out (2026-08-29)

The sec 8.25 fix, run. `dxy_per_transcript.py`. B (the Hudson denominator from
`fst print`) is proportional to between-class dxy on ANGSD's own scale -- 3.9x
(body) and 3.1x (control) the pg_gpu value, and NOT a constant factor between
regions -- so it is used only in WITHIN-region ratios and no absolute dxy is
quoted from it.

**Normalisation.** Raw per-transcript dxy confounds barrier age (wanted) with
constraint (not wanted). Each transcript is divided by the mean B of its own
100 kb window, which shares the local mutation rate and the same barrier age.
The collinear control -- no barrier, so only constraint plus noise -- sets the
bar for what "unremarkable" looks like.

**The statistic behaves far better than FST.**

| | corr(transcript, window) | Spearman |
|---|---|---|
| dxy, control | +0.620 | **+0.851** |
| dxy, body | +0.310 | +0.408 |
| FST (sec 8.25) | +0.20 | +0.16 |

**THE RESULT IS NULL.**

| ratio transcript/window | n | median | p90 | p95 | max |
|---|---|---|---|---|---|
| control | 63 | 0.444 | 2.042 | 2.548 | 4.554 |
| body | 95 | 0.351 | 1.633 | 2.687 | 6.102 |

Body transcripts above the control's 95th percentile: **5 of 95 = 5.3%, against
5% expected by chance.** The body's spread is no wider than the control's. **No
gene in the inversion is unusually diverged between arrangements relative to its
own neighbourhood.**

And the top of the ranking is uninformative anyway: **10 of the top 12 have no
EnTAP hit at all** -- unannotated predicted genes.

**Two honest weaknesses.**
1. The constraint check came out BACKWARDS: corr(ratio, 0-fold fraction) =
   +0.29 (body), +0.63 (control), where more constraint should mean LESS
   divergence. So the window normalisation is not fully clean, and the ranking
   should not be read even as a weak ordering.
2. Power. 95 transcripts at n = 40/class detects only a large excess at a single
   gene. This is "no strong outlier", NOT "no functional divergence".

**Where this leaves the gene question.** Combined with sec 8.10 (the GO
enrichment is a null artifact), sec 8.25 (the FST gene ranking is a polymorphism
artifact) and the FST scan showing no co-adapted partners OUTSIDE the inversion:
nothing in the gene content explains this inversion, inside or out. The block is
differentiated as a UNIT, which is what suppressed recombination does, and no
individual locus carries excess signal. That is a real and consistent finding,
not an absence of analysis.

## 8.27 The karyotype-stratified sweep scan -- done, and negative (2026-08-29)

Closes the gap sec 8.21 identified. `sweep_within_karyotype.py`.

**No diploSHIC retrain was needed, and retraining would have been the wrong
move.** I first said this required models at n = 254 and n = 95 because
diploSHIC's features are sample-size dependent. True, but it would also have to
be trained on a genome-wide neutral demography, whereas the derived class INSIDE
the inversion has its own coalescent (class size ~2Np plus a single-origin cap).
Training on the wrong null is how the pooled scan went wrong in the first place.
The per-karyotype ANGSD thetas from 2026-07-05 (AA/AB/BB pi, theta_W, Tajima's D
in 50 kb windows across chr2) avoid both problems: each class is compared to ITS
OWN chr2 background, so no external null is needed. Unphased data forces SFS
statistics anyway.

**Structural background (median over 50 kb windows, >= 5,000 callable sites):**

| | pi(AA) | pi(BB) | D(AA) | D(BB) | n win |
|---|---|---|---|---|---|
| inversion body | 0.005966 | 0.007720 | -2.372 | -2.028 | 1994 |
| collinear control | 0.013991 | 0.013785 | -2.097 | -2.034 | 2000 |

Control passes (the classes are indistinguishable there). Both classes lose ~55%
of their diversity inside the inversion, so a SHARED dip carries no information
and the scan keys on class ASYMMETRY.

**The first pass looked positive and was wrong.** Standardising each class
against the collinear control flagged **44 AA-specific windows (2.2%) and 0
BB-specific**, with 0 in the control. But pi_AA sits ~23% below pi_BB throughout
the block -- that IS pi_I/pi_S = 1.356 with I = BB, the project's central
observation -- so any control-standardised test flags AA across the whole
inversion.

**Retested against the block's own distribution, it is a clean null:**

    pi_AA/pi_BB over 1994 body windows: median 0.787, sd 0.195
    windows > 2 SD below that median: 3 of 1994 (0.2%) -- 2.3% expected if Normal

The block is **light**-tailed. Only **2 of the 44** candidates survive, and they
spread over 17 distinct 0.5 Mb bins across 64.1-79.3 Mb -- not one focal region.
**No within-arrangement sweep signal in either class.**

**The direction matters.** ZERO BB-specific windows by any criterion, and BB
carries HIGHER pi than AA throughout. The derived arrangement -- where a
supergene's adaptive variant would sit -- shows no sweep signature at all.

**POWER CAVEAT.** 50 kb windows, SFS statistics only, BB n = 95. A weak or old
sweep would be missed, and one predating the inversion would be shared by both
classes and invisible to an asymmetry test. This is "no detectable sweep", not
"no selection".

**Fifth consecutive null on gene-level/locus-level selection** (sec 8.10 GO,
8.21 pooled scan, 8.25 FST ranking, 8.26 per-transcript dxy, now this). The
consistent picture: the block behaves as a UNIT under frequency-level selection,
with nothing localised inside it.

## 8.28 Protein-level differences between arrangements -- few, and purifying (2026-08-29)

`protein_diff_karyotype.py`. Per-site allele frequencies per karyotype from
ANGSD genotype likelihoods (`-doMajorMinor 4` pins the major allele to the
reference for both groups so |dp| compares the same allele site by site), body
and collinear control, AA n=254 / BB n=95.

**THE CONTROL IS PERFECT -- the first time this question has had a working
instrument.**

| dp >= | body | control |
|---|---|---|
| 0.5 | -- | **0 of 8,009,993** |
| 0.8 | 0.1349% | **0.0000%** |
| 0.9 | 0.0450% | **0.0000%** |
| 0.95 | 0.0090% | **0.0000%** |

Zero sites anywhere in 8 M collinear sites reach dp = 0.5. Compare sec 8.23,
where the called VCF gave 267 apparent FIXED differences in the same region, and
a body/control ratio of only ~3x where FST differs 100-fold.

**THE ANSWER: protein-level differences exist but are strikingly FEW, and their
composition says purifying selection, not adaptation.**

* **13 unique near-fixed nonsynonymous positions** (dp >= 0.8, 0-fold) across
  the whole 19 Mb block, in ~12 transcripts. (The raw count of 16 double-counts
  positions covered by overlapping transcripts in different frames -- e.g.
  69.0906 Mb appears 3x.)
* **The differentiated tail is ~4x DEPLETED for nonsynonymous sites:**

      dp >= 0.5:  (0-fold/4-fold) tail 0.644  vs baseline 4.892  -> ratio 0.132
      dp >= 0.8:  (0-fold/4-fold) tail 1.143  vs baseline 4.892  -> ratio 0.234

  A dN/dS-like value of **0.13-0.23**. Between-arrangement differentiation is
  overwhelmingly SYNONYMOUS. That is what neutral accumulation behind a
  recombination barrier looks like while purifying selection keeps acting within
  both arrangements -- **no signature of adaptive protein divergence**.

**Two apparent stop-gains (S->*, E->*) in the AA arrangement at 68.089 and
71.895 Mb. TREAT AS ARTIFACTS UNTIL CHECKED.** The gene models were built on the
reference, which is a BB haplotype (sec 8.15), so codon frames are defined by BB
sequence. Applying that frame to a divergent AA haplotype inside the inversion
is exactly where spurious premature stops appear. Also, LOC_00005340
("acidic repeat-containing protein") contributes 5 of the 16 raw hits, and
repeat-containing genes are alignment-error prone.

**CAVEAT THAT SURVIVES A CLEAN RESULT.** Recombination is suppressed across the
whole block, so a near-fixed nonsynonymous site is a PASSENGER on the
arrangement's haplotype unless something independent implicates it. This counts
protein-level differences; it cannot identify a causal one -- and the five
preceding analyses (sec 8.10, 8.21, 8.25, 8.26, 8.27) found nothing localised.

**Sixth null on locus-level causation, but the FIRST positive characterisation
of the coding divergence itself:** it is sparse and constrained.

## 8.29 The two stop-gains were MY BUG, and an annotation-coverage limit (2026-08-29)

**RETRACTION.** Sec 8.28 reported two apparent premature stops in the AA
arrangement (S->* at 2:68,088,901, E->* at 2:71,894,582) and guessed they were
reference-is-BB frame artifacts. Wrong diagnosis, and the finding itself is
withdrawn: **there are ZERO stop-gains.**

**The actual cause is a strand mismatch in my own join.** degenotate reports
`ref` and the alternative-codon table on the **transcript** strand; ANGSD
reports major/minor on the **genomic** strand. For minus-strand genes these are
complements. Verified directly against the reference FASTA:

    2:68,088,901  genome = G   degenotate ref = C   (complement)
    2:71,894,582  genome = C   degenotate ref = G   (complement)

Both genes (LOC_00005332, LOC_00005351) are on the minus strand. Complementing
the ANGSD minor allele before the lookup turns both into ordinary missense:
**S->L** and **E->K**. Roughly half the annotation is minus-strand (18,562 of
36,708 mRNAs), so about half of sec 8.28's residue identities were wrong.

**Corrected: all 13 near-fixed nonsynonymous changes, none a stop.**

| position | gene | strand | dp | change |
|---|---|---|---|---|
| 60,908,622 | LOC_00005300 | + | 0.92 | E->G |
| 61,552,548 | LOC_00005304 | - | 0.95 | H->D |
| 64,103,547 | LOC_00005316 | - | 0.88 | T->S |
| 68,088,901 | LOC_00005332 | - | 0.94 | S->L |
| 68,920,207 | LOC_00005336 | - | 0.96 | R->C |
| 69,090,644/649/650/665, 69,092,152 | LOC_00005340 | - | 0.80-0.86 | F->L, S->L, S->P, A->T, L->P |
| 71,561,097 | LOC_00005350 | + | 0.86 | G->R |
| 71,894,582 | LOC_00005351 | - | 0.86 | E->K |
| 78,295,550 | LOC_00005374 | + | 0.84 | S->T |

**The dN/dS-like result of 0.13-0.23 is UNAFFECTED** -- degeneracy class is
strand-invariant. Only residue identities changed.

**ANNOTATION COVERAGE -- a limit on every gene-level null in sec 8.10-8.28.**
Raised by stsmall: high-differentiation regions with no annotated genes might
just be missing annotation.

    annotated gene models cover      14.3% of the 19.95 Mb block
    assayed sites inside a gene      16.24%
    assayed sites inside CDS          0.95%
    of 9,740 sites at dp >= 0.8, 8,299 (85.2%) fall OUTSIDE any annotated gene

**So every gene-level analysis in this project speaks to ~14% of the block**,
and 85% of the near-fixed differentiation sits in sequence with no gene model.
The six nulls are nulls *within annotated genes*, which must be stated that way.

What can be said against the cryptic-gene reading: high-dp sites are **not**
enriched outside genes either -- the in-gene fraction runs 0.80-1.06x the
background across dp thresholds, i.e. differentiation is distributed roughly in
proportion to sequence. That is what a linked block does, and it gives no
positive evidence that the extragenic signal is concentrated in unannotated
genes. But it does not exclude it, and the ISOSEQ/sq3 novel transcript set has
not been checked against these regions.

## 8.30 ISOSEQ/transcript evidence vs the high-differentiation regions (2026-08-29)

stsmall asked whether the high-differentiation regions with no annotated genes
(sec 8.29: 85.2% of near-fixed sites fall outside any gene model) might just be
missing annotation. Tested against the independent transcript evidence.

**First: the ISOSEQ transcripts are ALREADY in the annotation.** The curated GFF
is `...func.fix.sq3.FINAL.v2...` -- sq3 = SQANTI3, already merged. chr2 carries
751 standard LOC genes + 27 SQ (isoseq/novel); in the inversion body it is
**99 standard + 2 SQ = 101**, exactly the count in `Illex.genes.bed`. So the
14.3% coverage figure already includes them, and SQ adds 2 genes. lncRNA adds
nothing either -- 3 in the body, 197 kb, already inside the same 14.3%.

**Second: the independent StringTie assembly (RNA-seq + ISOSEQ) covers LESS, not
more.**

| set | union bp in the body | % of block |
|---|---|---|
| curated genes + lncRNA | 2,850,534 | **14.3%** |
| StringTie transcript spans | 1,217,583 | 6.1% |
| StringTie exons | 103,774 | 0.5% |

**Third, and decisive -- the direct intersection:**

| | all sites | dp>=0.8 | dp>=0.9 |
|---|---|---|---|
| in curated gene model | 16.24% | 14.79% | 17.20% |
| in StringTie/ISOSEQ transcript | 7.09% | 3.64% | 3.47% |
| in EITHER | 16.65% | 14.83% | 17.28% |

Of the **8,299** near-fixed (dp >= 0.8) sites outside any curated gene, **3 fall
in a StringTie/ISOSEQ transcript. Three.** That is **0.07x** the background rate
of StringTie coverage in extragenic sequence -- **depleted, not enriched.**
StringTie adds only 0.4 percentage points beyond the curated models (16.65% vs
16.24%), so the annotation is effectively saturated against this evidence.

**VERDICT: the missing-annotation hypothesis gets no support.** The
high-differentiation extragenic sequence is not merely unannotated -- it has no
transcript evidence from either source, and less than random extragenic
sequence does.

**What this does NOT rule out.**
1. **Tissue/stage coverage.** StringTie reflects the tissues and stages actually
   sequenced. A gene expressed only in an unsampled context is invisible to both
   the curated models and this test. This bounds "transcribed in what was
   sampled", not "functional".
2. **Regulatory sequence.** Enhancers and other cis-regulatory elements are not
   transcripts and are invisible to every analysis in this project. The
   differentiation could be regulatory and we would not see it.
3. Note these sites are inside the ACCESSIBILITY MASK, so they are mappable
   sequence -- not simply collapsed repeats.

So sec 8.29's framing stands but sharpens: the gene-level nulls apply to ~14% of
the block, and the other 86% is not hiding genes -- it is sequence with no
transcriptional evidence at all, in a gene-poor block.

## 8.31 Raw ISOSEQ alignments vs the high-diff regions -- better test, same answer (2026-08-29)

stsmall: "there are already mapped Isoseq to the genome, you can check for gene
models there." Correct, and sec 8.30 used the wrong instrument. The StringTie
merge had LOST most of the ISOSEQ signal: it covered 6.1% of the block by
transcript span, while the raw collapsed ISOSEQ alignments cover **12.1%** --
twice as much, and comparable to the curated models' 14.3%.

`isoseq.collapse.bam` in the body: **2,482 collapsed transcripts, 7,677 exon
blocks** (the raw `isoseq.mm2.bam` holds 2,248,258 alignments there).

| | union bp in body | % of block |
|---|---|---|
| curated genes + lncRNA | 2,850,534 | 14.3% |
| **ISOSEQ collapse, spans** | **2,418,650** | **12.1%** |
| ISOSEQ collapse, exons | 553,117 | 2.8% |
| StringTie spans (sec 8.30) | 1,217,583 | 6.1% |

**So the test now has real power -- and the answer does not change.**

| | all sites | dp>=0.8 | ratio |
|---|---|---|---|
| curated gene model | 16.24% | 14.79% | 0.91x |
| ISOSEQ exon | 3.92% | 2.65% | **0.68x** |
| ISOSEQ span (incl introns) | 13.55% | 11.92% | 0.88x |

Of the **8,299** near-fixed sites outside any curated gene:

    in an ISOSEQ EXON:  125 (1.5%)  vs 1.5% background  -> enrichment 0.98x
    in an ISOSEQ SPAN:  399 (4.8%)  vs 4.2% background  -> enrichment 1.14x
    NEITHER gene nor ISOSEQ span:  7,900 = 95.2%

**0.98x is background exactly.** Adding ISOSEQ raises total coverage from 16.2%
to 19.8% of assayed sites, but it does not preferentially cover the
differentiated sites. The missing-annotation hypothesis is now tested with an
instrument that has ~13% genomic coverage rather than 6%, and it still finds
nothing.

**What IS there:** 125 near-fixed sites fall in ISOSEQ exons with no curated gene
model -- genuine candidate unannotated exons carrying arrangement
differentiation. At 0.98x they are exactly the proportional expectation, so they
are not evidence of a hidden functional class, but they exist and are the only
concrete leads if anyone wants to chase them.

**The two exemptions from sec 8.30 still stand and are now the whole residual
argument:** ISOSEQ reflects the tissues and stages actually sequenced, and
**cis-regulatory sequence is not a transcript and is invisible to every analysis
in this project.** Regulatory divergence remains the main untested functional
hypothesis for this inversion.

## 8.32 Breakpoints PINNED, and the clustering question (2026-08-29)

**Two independent lines, and they agree on the right breakpoint.**

**(a) Arrangement differentiation, 10 kb windows** (per-site GL allele-frequency
difference, sec 8.28). The transition is a STEP, not a ramp:

    left  flank 60.04-60.53 Mb:  18% of the interior plateau, rate 0.0
    ---- breakpoint ----
    left  inside 60.55-61.40 Mb: 72% of plateau, median rate 25.0
    core        63.00-77.00 Mb: 104%, median rate 37.3
    right inside 79.20-79.50 Mb: 111%, median rate 36.7
    ---- breakpoint ----
    right flank 79.52-79.99 Mb:  29% of plateau, rate 0.0

    LEFT  breakpoint  ~60.540 Mb  (between 60.530 and 60.550)
    RIGHT breakpoint  ~79.500 Mb  (between 79.490 and 79.520)
    SPAN  ~18.96 Mb

**(b) AnchorWave anchors vs I. coindetii** (strand runs, whole chr2):

| strand | span | size |
|---|---|---|
| - | 0.323 - 2.032 Mb | 1.71 Mb |
| + | 2.050 - 44.376 Mb | 42.33 Mb |
| **-** | **44.439 - 79.312 Mb** | **34.87 Mb** |
| + | 79.569 - 115.834 Mb | 36.26 Mb |

The strand flip at **79.312-79.569 Mb** brackets the differentiation-based right
breakpoint at 79.50 Mb. **Independent confirmation.**

**But the LEFT breakpoint has NO structural counterpart.** 60.54 Mb sits in the
middle of a 34.87 Mb minus-oriented block, with 48 anchors in 60.2-60.9 Mb, all
minus -- so this is not sparse-anchor blindness.

**Reading:** relative to coindetii, illecebrosus chr2 carries a large (34.9 Mb)
fixed orientation difference spanning 44.4-79.3 Mb. The SEGREGATING inversion
occupies only the right ~19 Mb of it and **shares its right breakpoint**. So
either the segregating inversion reused a pre-existing fragile site, or it is
nested inside an older fixed rearrangement. Its left breakpoint at 60.54 Mb is
novel. (Distinct from the retracted sec 8.15 "two inversions" claim, which was
about AA/BB polarity; this is about illecebrosus-vs-coindetii orientation.)
*Caveat: AnchorWave anchors are CDS-based, and a lineage assignment cannot be
made without a third genome -- the 34.9 Mb block could be a coindetii-lineage
rearrangement.*

**CONSEQUENCE: the nominal span 60,040,617-79,995,597 overshoots by ~0.5 Mb on
each side.** The "differentiated body" 60.5-79.5 Mb used for the SFS, dxy, f1
and 2D-SFS work was essentially correct -- those results are unaffected. The
nominal-span figures in sec 8.3 include ~1 Mb of collinear sequence.

**CLUSTERING (stsmall's question): clustered, but DIFFUSELY.**

    Poisson dispersion 23.6 within the pinned block (1 = uniform)
    rate per 10k assayed sites: median 10.1, sd 8.7, range 0-49.7
    top 10% of windows hold 24.8% of near-fixed sites
    top 25% hold 50.6%;  top 50% hold 79.7%

~2.5x enrichment in the top decile -- patchy on a ~100 kb scale, not focal. The
20 hottest windows form **16 separate runs**, mostly single windows, scattered
61.7-77.8 Mb. Consistent with everything else: no locus stands out.

**FLUX, and a caveat on sec 8.4's "flux excluded".** With breakpoints pinned:

    outer 1 Mb   mean rate  7.91
    1-2 Mb in               10.65
    core                    12.91
    corr(rate, distance from breakpoint) = +0.138
      (nominal span +0.238; Fst-defined body +0.151)

Pinning the breakpoints REDUCES the edge effect but does not remove it. A ~40%
depletion in the outer 1 Mb is what gene flux near breakpoints produces. sec 8.4
excluded flux on FLAT MEAN dxy (edge/core 0.999); the near-fixed tail is more
sensitive than the mean, so these are not in direct contradiction -- but
**"flux excluded" is too strong and should read "no flux detectable in mean
dxy; a residual edge gradient is present in the near-fixed tail."**

## 8.33 Regulatory divergence: not detectable in promoters or UTRs (2026-08-29)

The last untested functional hypothesis. `regulatory_divergence.py`.

**What could be tested.** There is no regulatory annotation for this genome --
no ATAC, no ChIP, no conserved-element track. The GFF supports a COMPARTMENT
test: promoters (2 kb upstream of each TSS, strand-aware), 5'/3' UTRs, introns,
CDS, intergenic. Body uses the PINNED breakpoints 60.54-79.50 Mb (sec 8.32).

**Near-fixed rate (dp >= 0.8) per 10k sites, body:**

| compartment | n sites | near-fixed | per 10k | vs intergenic |
|---|---|---|---|---|
| promoter | 92,348 | 75 | 8.12 | 0.66x |
| 5' UTR | 10,860 | 7 | 6.45 | **0.52x** |
| 3' UTR | 13,268 | 12 | 9.04 | 0.73x |
| CDS | 65,735 | 45 | 6.85 | 0.55x |
| intron | 1,102,915 | 1,377 | 12.49 | 1.01x |
| intergenic | 6,645,265 | 8,235 | 12.39 | 1.00x |

Every functional compartment is DEPLETED; introns sit at background. That looks
like a result -- but it is not calibrated, and **the control has ZERO near-fixed
sites (0 of 8M, sec 8.28)**, so this rate cannot be calibrated against it at all.

**Mean dp IS calibratable, and it kills the reading:**

| compartment | body /interg | control /interg | ratio |
|---|---|---|---|
| **promoter** | 0.757 | 0.776 | **0.975** |
| 5' UTR | 0.427 | 0.398 | 1.074 |
| 3' UTR | 0.522 | 0.373 | 1.398 |
| CDS | 0.392 | 0.372 | 1.053 |
| intron | 0.854 | 0.734 | 1.164 |

**Promoters sit at 0.975 -- exactly their control baseline.** The depletion of
functional compartments in the body is present in the CONTROL at the same
magnitude, so it is a generic property of those compartments (constraint,
mappability, GC), **not arrangement-specific**. No promoter-proximal or UTR
regulatory divergence is detectable.

The 3' UTR at 1.398 is the only value notably above 1. It rests on 13,268 body
sites, no significance test was done, and 3' UTRs are the least canonical
regulatory compartment. **Unverified -- do not build on it.** (This project has
now produced three plausible-looking signals that dissolved on calibration:
the GO enrichment, the FST gene list, and the control-standardised sweep
candidates. Treat this the same way until tested.)

**WHAT THIS DOES AND DOES NOT BOUND.** It bounds PROMOTER-PROXIMAL (2 kb) and
UTR regulatory divergence. **Distal enhancers remain entirely untested** and
could sit anywhere in the 85% of the block with no gene model -- there is no
instrument in this project that would see them. A regulatory explanation for the
inversion is not excluded; it is only excluded near genes.

**Seventh consecutive null on localised causation.** The picture is now
consistent across coding, regulatory-proximal, gene-level, window-level and
site-level resolution: **near-fixed divergence accumulates in intergenic and
intronic sequence, is depleted everywhere purifying selection acts, and is
concentrated nowhere.** That is neutral accumulation behind a recombination
barrier. Whatever drove the frequency changes did not leave a localised mark.

## 8.34 The annotation IS incomplete -- sec 8.30/8.31 retracted (2026-08-30)

**RETRACTION.** Sec 8.31 concluded "the missing-annotation hypothesis gets no
support" because near-fixed sites fall in ISOSEQ exons at 0.98x background.
**That conclusion is wrong, and the reasoning behind it was wrong too.**

**The error was framing this as an ENRICHMENT question.** It never was. The
question stsmall asked is whether our GFF is complete, and enrichment of
near-fixed sites in transcript evidence has nothing to do with that. ISOSEQ from
a single tissue and life stage cannot establish absence of a gene; a proportional
overlap rate says nothing either way. The right instrument is cross-species
protein homology, which does not depend on our expression data at all.

**diamond blastx settles it.** 488 kb of unannotated, non-repeat sequence from
the 20 highest-divergence 100 kb windows, against 282,815 proteins from seven
cephalopods (Sthenoteuthis -- same family as Illex -- Architeuthis, Doryteuthis,
Sepioteuthis, Euprymna, Sepia, Octopus): **11 intervals with real protein
homology**, tens to hundreds of kb from the nearest gene model. The strongest,
2:62,180,079-62,181,727, is a 375 aa alignment at 42.7% identity, **E = 1e-80**,
89 kb from any annotated gene. Several carry near-fixed arrangement differences
(2:67,219,944 has 13; 2:61,746,024 has 8 at dp up to 0.94).

**Consequences.**
1. The "85% of near-fixed sites have no transcript evidence" framing (sec 8.30,
   8.31) is retracted. It measured our ISOSEQ coverage, not the sequence.
2. The gene-level nulls (sec 8.10, 8.25-8.27, 8.33) rest on a gene set that is
   incomplete inside this region. They stand as statements about ANNOTATED
   genes and must be written that way -- the true denominator is unknown.
3. The 14.3% annotated-coverage figure is a lower bound on genic sequence.

**Also worth noting: repeats are not the story.** The block is 50.8%
repeat-masked, but only 0.2% of assayed sites fall in repeats, because ANGSD was
restricted to the accessibility mask. The unannotated sequence carrying the
divergence is accessible, non-repetitive, and now demonstrably part-genic.

**NO CONTROL RUN, DELIBERATELY.** A matched blastx in the collinear region would
answer "are genes enriched in divergent windows", which is not the question and
was never claimed. Annotation completeness is a property of the annotation, not
a contrast between regions.
