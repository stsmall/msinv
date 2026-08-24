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
