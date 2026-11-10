# Illex chr2 inversion: ABC-SMC results (Talapas, 2026-11-07 to 2026-11-08)

Hand-off for integrating the trajectory ABC-SMC into the paper. All numbers below come from the pyABC History
databases; the machine-readable versions are the CSVs in this directory.

## Headline

- **Model choice: balancing selection (B) is favoured.** P(B) = 0.95-0.999 across both gene-flux settings and
  both priors on s. N (neutral drift) is eliminated by generation 1-3 in every run. D1/D2 (directional selection)
  keep 0-4.6% between them, carried by 1-22 particles. About 5x of B's lead comes from matching the present
  frequency without fine-tuning, which depends on the prior. The diversity data add a further 3-210x against
  directional selection: strong against a recent or strong incomplete sweep, weak against a slow one when gene flux
  is allowed. See "How B beat an incomplete (still-sweeping) history".
- **Posterior predictive checks:** B reproduces the data adequately overall (joint ppp 0.13-0.18) and matches the
  key diversity ratios. The surviving incomplete-sweep histories do not (joint ppp 0.004-0.03). Every model, B
  included, misses an excess of low-frequency variants shared by both arrangements and a faster LD decay than
  modelled, which is a process outside the trajectory models to flag as a limitation. The held-out Tajima's D fits
  all models and does not discriminate.
- **Collinear baseline:** against the representative collinear control (chr2 40–55 + 85–115 Mb) the model's
  neutral SFS fits; LD is 1.6–1.8× too high at short range, pointing to the recombination rate. The old control's
  rare-variant excess came from a high-diversity block at 24–36 Mb. The control JSFS fits the baseline in the
  shared low-frequency cells, so the inversion's 3–3.6x excess there is specific to the inversion, not a 2D realSFS
  or baseline artefact. See "Collinear control check".
- **B parameters are stable across flux settings and priors:** t_inv about 4.4-4.7e5 generations, t_sel about
  3.5-3.8e5 generations, founding frequency p0 about 0.04, equilibrium frequency p* = 0.618 [0.606, 0.64].
- **s (B) has a lower bound only:** s >~ 1.3e-4 (2.5% quantile). Above that the posterior is close to flat in log s
  up to whatever ceiling the prior has (1e-2 or 1e-1). The statistics see the equilibrium at p*, not how strongly
  it is held. Report s as a lower bound, not as a point estimate.
- **Gene flux (flux 1):** gamma_eff = 4.3e-8 [2-3e-9, 2.7-3.4e-7] under both priors.
- **Hard (single-copy) starts are rejected:** hard-start posterior mass in B is 0.4-3.4%. Soft starts dominate.

## Procedure

**Question and models.** Four trajectory models for the frequency of the derived 2a arrangement (present
frequency 0.626): N (neutral drift), D1 (directional/genic selection s p(1-p) from t_sel to the present), D2 (D1
with selection switched off at t_stop) and B (overdominance, delta p = s p(1-p)(p*-p)/p*, from t_sel). Each model
runs without gene flux (flux 0; gamma fixed at 1e-15 because msinv rejects 0) or with gene flux (flux 1, gamma
free), and with a hard start (one copy, importance-weighted by the exact establishment probability) or a soft start
(founding frequency p0). Times are in generations before present.

**Simulator.** Per proposal: draw a frequency path, simulate the 40 kb inversion body with msinv 0.4.1 (crossover
suppression in heterokaryotypes: an arrangement-k lineage recombines at r p_k(t); includes the flank-merge fix) and a
40 kb neutral control with msprime 1.4.1, then compute the summary statistics against
`results/illex/abc_observed.json` (unchanged). Boundary statistics (`bnd_*`) were disabled, as on the branch.

**Inference.** pyABC 0.13.1 ABC-SMC, joint model choice over N, D1, D2 and B with the hard/soft class as a
discrete parameter (mixed, P(hard) = 0.5 prior). Population 1000 particles per generation, at most 15 generations,
redis sampler over 4 nodes x 32 worker processes. Distance scales are refitted every generation. Stop rules: max_gen
15; valid-path acceptance < 0.01; the driver walltime cutoff. The "relative epsilon change < 5%" rule was **turned
off** (`--min-eps-rel 0`) for the reported runs, because epsilon is not comparable across generations once the
scales are refitted, and the rule stopped the first attempt at generations 2 and 6.

**Priors** (`illex/abc/constants.py`): t_inv U(1e5, 1.5e6); p0 logU(1e-3, 0.10); f_sel = t_sel/t_inv U(0, 1);
f_stop U(0, 1); p* U(0.3, 0.95); gamma logU(1e-9, 1e-5); s logU(1e-6, 1e-2) in run set A and logU(1e-6, 1e-1) in
run set B. p0's upper bound of 0.10 is kept deliberately as the biologically reasonable limit on a founding
frequency. For run set B the hard-start establishment grid (`illex/abc/establishment.py`) was extended to s = 0.1
(250 nodes) and rebuilt; it matches the exact first-passage solver to a relative error of 1e-8 or less up to s = 0.1.

**Code.** GitHub `stsmall/msinv`, branch `illex/trajectory-abc` at 45681c5, plus local, uncommitted changes:
(1) run set B's prior and grid change, in a separate checkout; (2) a fix to `illex/abc/smc.py` (see Caveats). The
environment is conda `illex_abc`: Python 3.12.16, numpy 2.4.6, scipy 1.17.1, pandas 3.0.6, msprime 1.4.1, tskit
1.0.2, pyabc 0.13.1, redis-py 8.1.0, msinv 0.4.1.

## Runs

All runs: pop 1000, mixed hard/soft, models N D1 D2 B, Slurm account kernlab, partition kern, 4 nodes x 32 CPUs.

| run set | s prior | flux | job | generations | stop rule | wall | CPU-h |
|---|---|---|---|---|---|---|---|
| A | (1e-6, 1e-2) | 0 | 48374624 | 0-11 | valid-path acceptance 0.0063 < 0.01 | 2:43 | 333 |
| A | (1e-6, 1e-2) | 1 | 48374625 | 0-12 | valid-path acceptance 0.0098 < 0.01 | 2:34 | 307 |
| **B** | **(1e-6, 1e-1)** | 0 | 48375102 | 0-11 | valid-path acceptance 0.0064 < 0.01 | 2:40 | 326 |
| **B** | **(1e-6, 1e-1)** | 1 | 48375103 | 0-14 | max_gen (15) | 1:17 | 145 |

Total simulations per run: 0.74-1.26 M. Superseded, for the record only: a first attempt with the s prior of run set
A and the 5% epsilon rule on (jobs 48364669/48364670) stopped at generations 6 and 2; its model probabilities agree
(P(B) 0.984 / 0.952). Two 2-node smoke tests passed (48364637, 48364638).

## Results

### Posterior model probabilities (final generation)

| run set | flux | N | D1 | D2 | B |
|---|---|---|---|---|---|
| A, s <= 1e-2 | 0 | 0 | 0.0010 | 0 | **0.9990** |
| A, s <= 1e-2 | 1 | 0 | 0.0039 | 0.0088 | **0.9872** |
| B, s <= 1e-1 | 0 | 0 | 0 | 0.0176 | **0.9824** |
| B, s <= 1e-1 | 1 | 0 | 0.0460 | 0 | **0.9540** |

P(B) rises nearly monotonically from about 0.79-0.83 at generation 0 (`generation_trace.csv`). Hard-start mass
within B at the final generation is 0.004 (A0), 0.005 (A1), 0.013 (B0) and 0.034 (B1).

### Model B posteriors: median [95% interval]

p0 and t_sel are over soft-start particles (they are undefined for hard starts); the rest are over all particles.
Particle ESS 708-817.

| parameter | A, flux 0 | A, flux 1 | B, flux 0 | B, flux 1 |
|---|---|---|---|---|
| t_inv (gen) | 4.41e5 [2.96e5, 6.00e5] | 4.74e5 [3.03e5, 6.98e5] | 4.36e5 [2.95e5, 6.05e5] | 4.67e5 [2.89e5, 8.27e5] |
| t_sel (gen) | 3.58e5 [2.64e5, 5.05e5] | 3.77e5 [2.48e5, 6.20e5] | 3.48e5 [2.57e5, 4.85e5] | 3.63e5 [2.25e5, 7.30e5] |
| p0 | 0.037 [0.0089, 0.091] | 0.039 [0.0094, 0.093] | 0.037 [0.0082, 0.091] | 0.048 [0.016, 0.092] |
| p* | 0.618 [0.606, 0.635] | 0.618 [0.606, 0.635] | 0.618 [0.607, 0.635] | 0.619 [0.607, 0.639] |
| s | 1.7e-3 [1.4e-4, 8.0e-3]* | 1.4e-3 [1.3e-4, 8.2e-3]* | 5.7e-3 [1.2e-4, 7.6e-2]* | 5.0e-3 [1.4e-4, 7.8e-2]* |
| gamma_eff | n/a | 4.2e-8 [2.0e-9, 2.7e-7] | n/a | 4.3e-8 [3.3e-9, 3.4e-7] |

\* s is prior-bounded above: the upper quantile follows the prior ceiling. In run set B, log10 s has quantiles
-3.9 / -2.25 / -1.12 (2.5/50/97.5%), against -3.93 / -2.5 / -1.08 for a flat distribution on [-4, -1]. In both run
sets s's median rose from generation 3 to the end only as far as the prior allowed. **Supported statement:** s >~ 1e-4
(2.5% quantile 1.2-1.4e-4 in all four runs); no upper bound from these data.

The p0 upper interval sits near the prior bound of 0.10, so p0 is only partly identified at the top. That bound is
the intended biological constraint.

### Interpreting s (what the wider prior shows)

Run set B widened s's prior ceiling from 1e-2 to 1e-1 to test whether the run-set-A posterior, which sat against
1e-2, had a peak just beyond it. It does not.

- **Posterior = prior truncated below about 1e-4.** Under logU(1e-6, 1e-1) cut off at 1e-4, P(s > 1e-2) = 1/3; the
  run-set-B final posteriors give 0.358 (flux 0) and 0.362 (flux 1). Above about 1e-4 the data do not reweight s at all. The 2.5% quantile is the
  same in all four runs (1.2-1.4e-4); the median (1.4-1.7e-3 in A, 5.0-5.7e-3 in B) and the upper quantile follow the
  prior ceiling.
- **Mechanism.** The statistics record an inversion held near p* = 0.62 for t_sel of about 3.5e5 generations. Below
  s of about 1e-4, overdominance is too weak to pull the frequency to p* and hold it against drift over that span, so
  those values fail. Above it, the frequency reaches p* in a small fraction of t_sel and then stays there; stronger
  selection only shortens the approach phase, which present-day summary statistics cannot see. s = 1e-3 and
  s = 5e-2 produce practically indistinguishable data.
- **Other parameters are unaffected.** t_inv, t_sel, p0, p* and gamma are essentially identical under both
  priors, so the old 1e-2 ceiling was not distorting the rest of the fit.
- **Model choice.** P(B) falls slightly under the wider prior (0.999 to 0.982 without flux, 0.987 to 0.954 with
  flux). The D1/D2 share behind that rests on 7-22 particles and is within run-to-run Monte Carlo noise. Posterior
  model probabilities depend in general on the widths of each model's priors, which is worth a sentence in the
  methods.
- **What to report:** balancing selection with s >~ 1e-4 (95% lower bound), and the data do not bound s from above.
  Do not report a median or credible interval for s; either would reflect the prior choice. An upper bound would
  need data informative about the approach to equilibrium (the rise of 2a from p0 to p*), which these summary
  statistics are not.

### How B beat an incomplete (still-sweeping) history

D1 is the "incomplete sweep" hypothesis: directional selection on 2a from t_sel to the present, with 2a not yet
fixed. D2 is a sweep that stalled (selection off at t_stop, drift since). Both competed in every run. "B wins" means
more posterior mass within this four-model set and these priors, and the evidence has two parts.

**1. Landing on the present frequency (generation 0; prior-dependent, not evidence about the history).** Every
simulation must end with 2a at 0.626 +/- 0.02. Many parameter combinations do this in every model: in D1 a ridge of
(s, t_sel, p0) trades stronger selection against a later start; in B any p* near 0.62 with s >~ 1e-4 works. ABC
integrates over all of them. At generation 0 (epsilon = inf, every valid path accepted) P(m) is proportional to the
prior mass of each model's "ends at 0.626" region, so B's generation-0 lead is a fine-tuning (Occam) penalty on
directional selection, which must be caught mid-sweep, while B sits at p*. The penalty depends entirely on the prior
ranges (p* U(0.3, 0.95); D1's s, t_sel and p0 ranges): a wider p* prior lowers it, and narrower D1 ranges change it.

**2. The sequence statistics (generations 1 onward).** The final odds divided by the generation-0 odds isolate the
evidence from the diversity data, given the present frequency.

B against the directional models combined (D1 + D2), the robust comparison:

| run | gen 0 odds (frequency only) | final odds | factor from sequence data |
|---|---|---|---|
| A, s <= 1e-2, flux 0 | 5.0 | 1,032 | ~210x |
| A, s <= 1e-2, flux 1 | 4.5 | 77 | ~17x |
| B, s <= 1e-1, flux 0 | 5.5 | 56 | ~10x |
| B, s <= 1e-1, flux 1 | 6.7 | 21 | ~3x |

B against D1 alone has generation-0 odds of 9.6-15. Final odds are about 1,000 (A0, 1 D1 particle), 250 (A1, 2
particles) and 21 (B1, 22 particles; sequence factor about 1.4). In B0, D1 died out, so the odds are not finite. With
1-22 particles these are noisy; the combined comparison above is the one to quote.

**Which incomplete sweeps were rejected.** At generation 0, 5-9% of the accepted D1 particles had strong selection
(s > 1e-4) and 8-13% a recent start (t_sel < 5e4 generations). By generation 1-2 none remained, in all four runs. The
surviving D1 particles are slow, weak pushes: s about 5e-6 acting over 4-5e5 generations.

**What can be claimed:**
- Given the present frequency, the diversity data reject a recent or strong incomplete sweep in all runs.
- They favour balancing over directional selection in general by about 10-210x without gene flux, and only 3-17x
  with gene flux. A slow, weak ongoing sweep is disfavoured, not excluded, especially with gene flux.
- These are statements within the model set {N, D1, D2, B}. Fluctuating selection, other forms of frequency
  dependence and structured demography were not modelled. Winning the comparison is not the same as fitting the
  data; see the posterior predictive checks below.

**Observed signals bearing on "has 2a always been at 0.62?"** One present-day frequency cannot say whether the
frequency has been stable; the diversity data can, partly.
- pi_I/pi_S = 0.75, although 2a is the commoner arrangement. If both arrangements had held their present
  frequencies for a very long time, within-arrangement diversity would scale with frequency, giving a ratio near
  0.626/0.374 = 1.7. So 2a's diversity has not caught up: 2a rose to its present frequency recently on the
  coalescent timescale. The B fit agrees: 2a starts rare (p0 about 0.04) and selection begins at t_sel about 3.5e5.
- dxy/pi = 1.86: high divergence between arrangements, consistent with an old inversion (t_inv about 4.5e5).
- f1_ratio = 1.23 (singleton share, 2a over 2+): a slight lean toward 2a being the younger arrangement.
- Tajima's D is positive in both arrangements (+1.03 in 2a, +0.76 in 2+). **Tajima's D was not fitted;** it is one
  of the held-out PPC statistics. The PPC shows that every surviving model, including the slow D1, reproduces it, so
  it does not discriminate between them (see below). It is consistent with no recent sweep, not evidence for B.

The best-supported history: 2a arose long ago, rose from rare to about 0.62 a few hundred thousand generations ago,
and has had no recent sweep. The ABC treats selection on the arrangement as a whole. It simulates a representative
40 kb of the inversion body and says nothing about which genes are targets.

### Posterior predictive checks

**Method** (`ppc/`). For every model alive at the final generation of each run, 1,000 particles were drawn by
posterior weight and re-simulated with the SMC's own simulator (frequency path, retried until it lands in the p_now
window; msinv body; msprime control; statistics). Run set B used the checkout with the wider s prior. Each statistic
gets the observed value's predictive quantile and a two-sided predictive p-value (ppp). The joint check is the
observed vector's MAD-scaled L2 distance from the predictive median over the 39 fitted statistics (the ABC's own
metric), ranked against the replicates' distances. The full Mahalanobis form is unusable because the JSFS cells sum
to 1, which makes the covariance singular. Three statistics were never fitted and serve as out-of-sample checks:
tajd_I, tajd_S and edge_centre. Jobs 48381037-48381040 (partition compute), 13-29 min each.

**Caveat on D1/D2:** their predictives rest on 1-22 distinct particles, so they carry less parameter uncertainty
than B's (978-999 particles) and misfit is easier to detect. Their joint ppp values lean conservative against them.

**Joint fit (ppp; small = the model does not reproduce the data):**

| run | B | D1 | D2 |
|---|---|---|---|
| A, flux 0 | 0.16 | 0.030 (1 particle) | n/a |
| A, flux 1 | 0.14 | 0.004 (2) | 0.000 (2) |
| B, flux 0 | 0.18 | n/a | 0.085 (7) |
| B, flux 1 | 0.13 | 0.006 (22) | n/a |

**B fits the data adequately overall (joint ppp 0.13-0.18); the surviving incomplete-sweep histories (D1) do not
(0.004-0.03).** D2 is poor or borderline on very few particles.

**B reproduces the key diversity ratios** (observed; B predictive median [95%], run set B flux 0):
pi_I/pi_S 0.746 vs 0.72 [0.24, 0.95]; dxy/pi 1.86 vs 1.57 [1.21, 4.74]; f1_ratio 1.23 vs 1.23 [1.04, 1.87].
The other runs are similar (ppp 0.59-0.97).

**Held-out statistics fit in every model** (ppp >= 0.18 for B, >= 0.32 for the others). Tajima's D: observed
1.03 / 0.76 (2a / 2+), B predicts about 0.98-1.05 / 0.65-0.67. edge_centre: 0.996 vs about 1.00. The surviving D1
predicts Tajima's D in 2+ slightly better than B (observed quantile about 0.4 in D1 vs 0.9 in B). So Tajima's D does
not support B over a slow sweep.

**Shared misfit: statistics no model reproduces.** B misses 7-8 of the 39 fitted statistics at ppp < 0.05; D1/D2
miss 8-13. The core misses are common to every model, prior and flux setting, so they reflect something none of the
models contains, not the trajectory:
- **Excess variants shared at low frequency in both arrangements.** In jsfs_11 (projected count 1-4 of 20 in both
  2a and 2+), observed 0.073 vs B 0.016 [0, 0.047]. jsfs_13 (rare in one arrangement, 16-19 of 20 in the other) is
  also high: 0.019 vs 0.005 [0, 0.014]. This persists with gene flux on, so more allele sharing between arrangements
  is observed than msinv's gene flux plus the genotype error model produce.
- **LD decays faster than modelled.** Short-range LD (100-580 bp: ld_I_1 is 100-240 bp, ld_S_1 and ld_S_2 cover 100-580 bp) is higher than predicted
  (ld_S_1 1.62 vs 1.13 [0.95, 1.37]). Long-range LD in 2+ (3.4-20 kb: ld_S_5, ld_S_6) is lower (ld_S_6 0.965 vs 1.09
  [1.04, 1.14]). LD bins are 6 log-spaced bins from 100 bp to 20 kb.

Candidates, none tested: recombination-rate or gene-conversion heterogeneity along the body, a mismatch in the
genotype error, phasing or paralog model, or demographic structure. **Worth flagging in the paper:** model choice
partly runs through statistics that no model fits.

**Which statistics separate B from the directional models.** Ranked by how well each statistic's predictive
distribution separates B from D1 or D2. Mostly 2+ (S) statistics: LD in 2+ at intermediate distances (0.6-3.4 kb: ld_S_3,
ld_S_4), Tajima's D in 2+, the 2+ singleton share (sfs_S_1) and f1_ratio. The direction is mixed: the observed ld_S_3
is central in B (quantile 0.65-0.72) and at D1's edge (0.99-1.00), but ld_S_4 is at B's edge (0.03-0.06) and central
in D1 (0.64-0.76). So B's advantage comes from the overall pattern, partly in LD bins where the shared misfit sits,
not from one clean diagnostic. With the wider s prior and no gene flux, B and D2 barely separate (best separation
0.37, against 0.8-0.9 for B vs D1).

**Bottom line from the PPC:**
- B is an adequate description of the data overall. Slow incomplete sweeps (D1) are not.
- No model captures the excess of variants shared between arrangements or the steep LD decay. Those point to a
  missing process outside the trajectory, which the paper should name as a limitation.
- The held-out Tajima's D does not discriminate between the models.

### Collinear control check

**Question.** Is the shared misfit in the posterior predictive checks specific to the inversion, or is it already
in the collinear background?

**Method** (`collinear/collinear_check.py`). The observed chr2 10–30 Mb collinear statistics (`meta["control"]` in
`abc_observed.json`; these are the divisors of every fitted SFS, LD and ratio statistic) are compared with 2,000
replicates of the ABC's own neutral control simulation:
- msprime, no inversion, 40 kb
- growth from N_anc 5.5e5 to N0 6.8e6 over 7.7e5 generations
- r = 2.158e-9, mu = 3e-9
- the data's genotype error model applied to the LD diploids

There is no observed control JSFS (realSFS 2D was run on the body only), so the JSFS cells cannot be checked here.
Raw data for other chromosomes is not on Talapas.

**Result.** The neutral baseline does not fit the collinear data:

| statistic | observed / simulated median | two-sided p |
|---|---|---|
| singletons-class SFS bin (sfs_*_1) | 1.12–1.14 | 0.001 |
| common-variant bin (sfs_*_6p) | 0.65–0.66 | ≤ 0.003 |
| Tajima's D | −2.02 / −2.08 vs −1.71 | 0.001 |
| LD, 100–580 bp | 1.2–1.3× | ≤ 0.002 |
| LD, 0.6–8.3 kb | 1.4–1.6× | < 0.001 |
| LD, 8.3–20 kb | 1.2–1.3× | < 0.001 |
| pi_ratio, dxy_pi, f1_ratio (I vs S labels) | ≈ 1 | 0.06–0.19 |

The two karyotype groups behave identically in the collinear region, as expected.

**Interpretation.**
- The collinear data carry more rare variation, fewer common variants and a more negative Tajima's D than the
  growth model gives. This points to stronger or more recent expansion than the demography in `illex.theory`, or
  to background selection or genome-wide linked selection. It is not a property of the inversion.
- Collinear LD is elevated at every distance. This could come from a lower effective recombination rate than the
  2.158e-9 used here (a male-map value), from population structure, or from an error model that misses some LD.
- The fitted SFS and LD statistics are divided by the control, so a background misfit cancels to first order. It
  does not cancel exactly when the distortion differs between body and control.
- The JSFS is not divided by the control. A genome-wide excess of rare variants therefore carries straight into the
  shared low-frequency cells (jsfs_11, jsfs_13), which is the main shared misfit in the predictive checks.
- The LD excess peaks at intermediate distances (1.4–1.6× at 0.6–8.3 kb). Dividing body LD by this inflated
  control can produce the pattern the predictive checks show in 2+ (ratio high at short range, low at 3.4–20 kb).
  This is consistent with the misfit but not shown.
- **Bottom line:** the shared misfit most likely comes from the neutral baseline (demography and/or recombination),
  not from the trajectory models. The comparisons between models share that baseline. A fix would refit the
  baseline demography and recombination rate to collinear windows before rerunning the ABC.
- Sweeps: diploS/HIC found no sweeping genes in either fixed karyotype. Sweeps in the collinear background would
  have to be pervasive to shift the whole 20 Mb control this much, which is closer to genome-wide linked selection
  than to discrete sweeps.

#### Rerun on the representative control (chr2 40–55 + 85–115 Mb)

The old control (10–30 Mb) overlaps a block with doubled diversity and low alignability (24.1–35.9 Mb). Repeating
the check against `results/illex/abc_observed_ctl2.json` (same baseline simulation, that control's genotype error
model; `collinear/ctl2/`) gives observed ÷ simulated median:

| statistic | old control | representative control | p (representative) |
|---|---|---|---|
| rarest SFS bin (I, S) | 1.12, 1.14 | 1.04, 1.08 | 0.19, 0.02 |
| commonest SFS bin (I, S) | 0.66, 0.65 | 0.86, 0.81 | 0.23, 0.09 |
| Tajima's D (I, S) | −2.02, −2.08 | −1.85, −1.90 (sim −1.71) | 0.13, 0.04 |
| LD, 100–580 bp | 1.2–1.3× | 1.6–1.8× | < 0.001 |
| LD, 0.6–3.4 kb | 1.45–1.64× | 1.44–1.75× | < 0.001 |
| LD, 8.3–20 kb | 1.20–1.31× | 1.04–1.18× | < 0.001 |
| pi_ratio, dxy_pi, f1_ratio | 0.99, 1.01, 0.98 | 0.94, 1.03, 0.97 | < 0.001, < 0.001, 0.005 |

- **SFS:** the current growth demography fits the representative control within its range or close to it. Most of
  the rare-variant excess in the old control came from the 24–36 Mb block, not from the demography.
- **LD:** still too high, now most at short range (1.6–1.8×) and close to the model by 8–20 kb. This steeper decay
  points to the recombination side of the baseline (a lower population-scaled recombination rate than the model's
  4·N·2.158e-9) more than to the growth history.
- **Karyotype groups in collinear sequence:** 2a homokaryotypes have 6% lower diversity than 2+ ones, and dxy
  exceeds within-group diversity by 3.5%. Far from the inversion the groups should be exchangeable, so this
  suggests population structure correlated with karyotype, or a difference in data quality (call rates 0.62 vs
  0.67). The fitted ratios are divided by this control, which removes it to first order.
- **The shared JSFS misfit is not explained by the baseline SFS any more:** the representative collinear SFS fits,
  so the excess of low-frequency variants shared by both arrangements is more likely specific to the inversion or
  to the 2D realSFS. There is still no control JSFS to test this.

#### Control JSFS (AA40 x BB40, chr2 40–55 + 85–115 Mb)

`meta["jsfs_control"]` in `abc_observed_ctl2.json` against the 12 JSFS cells of the same 2,000 baseline replicates
(`collinear/ctl2/collinear_jsfs_check.csv`). Cell `jsfs_ab` = 2a count bin a, 2+ count bin b, with bins 0, 1–4,
5–15, 16–19 and 20 of 20 projected chromosomes. The spectrum is folded, so each cell also counts its mirror image
(bin 4−a, bin 4−b):

| cell | observed control | baseline median [95%] | p | observed inversion | B predicted inversion | p |
|---|---|---|---|---|---|---|
| jsfs_11 (low in both, or high in both) | 0.233 | 0.247 [0.229, 0.264] | 0.15 | 0.073 | 0.020 | 0.008 |
| jsfs_13 (low in one arrangement, high in the other) | ~0 | ~0 | 0.59 | 0.0185 | 0.0062 | 0.05 |
| jsfs_01 (absent in 2a, low in 2+; or mirror) | 0.352 | 0.293 [0.276, 0.311] | < 0.001 | 0.376 | 0.418 | 0.31 |
| jsfs_10 (low in 2a, absent in 2+; or mirror) | 0.272 | 0.293 [0.277, 0.311] | 0.011 | 0.331 | 0.330 | 0.97 |

(B predictions: run set B, flux 1.)

- **The shared low-frequency excess is specific to the inversion.** In collinear sequence the baseline reproduces
  the cells for variants at low frequency in both groups (jsfs_11, jsfs_13), with the same individuals, the same
  2D realSFS pipeline and the same projection. So the inversion's excess (3.6x in jsfs_11, 3x in jsfs_13) is not
  an artefact of the 2D realSFS, of the baseline demography or of collinear population structure.
- **What it could be** (untested): more exchange of low-frequency variants between arrangements than the
  gene-conversion model (flux 1) allows, or a few misassigned chromosomes in the AA/BB groups; each misassigned
  chromosome carries the other arrangement's variants into a group at low frequency.
- **Collinear asymmetry:** 2+ carries more private rare variants than 2a (jsfs_01 above the baseline, jsfs_10
  below). This matches 2a's 6% lower collinear diversity: structure correlated with karyotype, or call-rate
  differences between the groups.

### Baseline refit (representative control)

`illex/abc/smc_results/baseline/baseline_refit.py` (job 48429847): 20,000 log-uniform prior draws of N_anc, the
growth fold, T_grow and r (mu fixed at 3e-9), 200 kb each, the control's genotype error model; rejection at 1% (200
draws) with local-linear regression adjustment, against the representative control
(`results/illex/abc_observed_ctl2.json`). Outputs are in `baseline_refit/`.

| parameter | current baseline | refit, no JSFS (`fit.json`) | refit, + 8 JSFS cells (`fit_jsfs8.json`) |
|---|---|---|---|
| N_anc | 5.48e5 | 5.15e5 [3.87e5, 6.52e5] | 5.07e5 [3.78e5, 6.36e5] |
| N0 | 6.81e6 | 5.79e6 [4.90e6, 7.96e6] | 9.11e6 [7.63e6, 1.17e7] |
| growth fold | 12.4 | 11.3 [9.3, 16] | 17.8 [14.2, 26.1] |
| T_grow (generations) | 7.70e5 | 4.66e5 [3.63e5, 5.82e5] | 3.83e5 [3.15e5, 4.72e5] |
| r (per bp per generation) | 2.158e-9 | 4.94e-10 [4.04e-10, 5.83e-10] | 5.51e-10 [4.61e-10, 6.49e-10] |

- **The main change is recombination:** r drops about 4x, to about 5e-10 (r/mu about 0.17 against 0.72 now). With
  mu fixed only r/mu is identified. The growth history moves less: the same ancestral size, a similar or somewhat
  larger fold, and growth starting about 1.7–2x more recently.
- **Fit of the refitted baseline:** the accepted draws reproduce the SFS, Tajima's D, pi, LD from 240 bp to 20 kb and
  the symmetric JSFS cells (observed quantiles 0.2–0.7). Two exceptions remain: LD at 100–240 bp (observed 0.167,
  above every accepted draw), and the private-rare asymmetry (jsfs_01 above, jsfs_10 below every accepted draw), which
  a single panmictic population cannot produce.
- **The 12-cell JSFS fit (`fit_jsfs.json`) is not usable.** The four near-fixed-in-one-group cells (jsfs_03, 04, 13,
  14) are ~1e-9 to 1e-14 in collinear sequence; the observed jsfs_04 is 830 MADs from the draws' median and swamps
  the distance, giving a 860-fold expansion that misses the SFS and Tajima's D. `fit_jsfs8.py` drops those four cells
  (it overrides the cell list; `baseline_refit.py` is unchanged).

## Minor models (for completeness; too few particles to interpret)

D1 and D2 survive with 1-22 particles, with s about 4-20e-6 and t_inv about 4.9-5.9e5. Per-parameter values are in
`posterior_summary.csv`.

### Robustness

- Gene flux on or off changes no conclusion; flux 1 widens the t_inv and t_sel intervals slightly.
- Widening s's prior by a decade leaves t_inv, t_sel, p0, p* and gamma essentially unchanged and moves only s (see
  "Interpreting s").
- Stopping on the epsilon rule (first attempt) or running to the acceptance floor gives the same model choice.

## Caveats

1. **prob_* bug, fixed after the runs.** `illex/abc/smc.py` (`write_outputs`) wrote every `prob_*` trace column and
   every `abc_smc_modelchoice.csv` probability as 0.0. With t given, pyABC 0.13.1's
   `History.get_model_probabilities(t)` returns an (m x "p") frame, and the code took `.iloc[0]` of it, which is one
   model's row. The fix is a helper `_model_probs(h, t)` that reads the all-generations (t x m) frame, plus regression
   assertions in `tests/illex/test_abc_smc.py::test_modelchoice_outputs_distinct`. The inference itself was never
   affected. The `prob_*` columns and `abc_smc_modelchoice.csv` in every output directory were rewritten from the
   databases; the originals are kept as `*.bak`. All numbers here come straight from the databases.
2. **Commit status.** The smc.py fix is committed on `illex/trajectory-abc` as b39b98f and pushed to GitHub (patch copy in this directory). Run set B's prior and grid change
   exists only in the copy on Talapas, uncommitted. To integrate: pull b39b98f, and decide whether s's prior
   ceiling becomes 1e-1 on the branch. If it does, update `tests/illex/test_abc_constants.py`, which asserts the old
   prior.
3. **Pre-existing test failures**, unrelated to this work (they fail identically on the untouched branch):
   `test_stats.py::test_no_barrier_gives_no_differentiation` (0.0676 vs a 0.05 threshold) and four in
   `test_theory.py` (`test_old_inversion_pi_ratio_approaches_frequency_ratio`, `test_dxy_floor_values_match_spec`,
   `test_solve_t_inv_matches_spec`, `test_observed_below_single_founder_bound`).
4. **Wasted hard proposals** (known): in mixed runs hard proposals keep costing simulations after the hard class
   loses posterior mass. This affects cost only.
5. `abc_rf_vs_smc.csv` compares against the stale Task 8 ABC-RF results. Ignore it.
6. **Environment note:** on Talapas, `~/.local` user site-packages (scipy 1.14.1) shadow the conda env. Every run
   used `PYTHONNOUSERSITE=1`.

## Files

In the repo, `illex/abc/smc_results/` holds this README, the predictive-check scripts (`ppc/ppc_run.py`,
`ppc/ppc.sbatch`, `ppc/ppc_analyze.py`; `ppc_analyze.py` reads the observed statistics from
`results/illex/abc_observed.json`), the collinear checks (`collinear/collinear_check.py`,
`collinear/collinear_jsfs_check.py`) and the baseline refit (`baseline/`, including `fit_jsfs8.py`). The CSVs and
JSON fits listed below are on Talapas only, in `/gpfs/projects/kernlab/ssmall/illex_traj_abc_summary/`:
- `README.md`: this document
- `model_probabilities.csv`: final P(m), particle counts and hard-start mass per run set, flux and model
- `posterior_summary.csv`: median and 2.5/97.5% quantiles for every parameter of every surviving model
- `generation_trace.csv`: per-generation epsilon, simulations and P(m)
- `baseline_refit/`: baseline refit. `draws.csv` (20,000 draws), `fit.json`, `fit_jsfs8.json` (+ `fit_jsfs8.py`),
  `fit_jsfs.json` (12 cells, not usable).
- `collinear/`: collinear control check. `collinear_check.py`; `collinear_check.csv` (per statistic),
  `collinear_sims.csv` (replicates).
- `ppc/`: posterior predictive checks. `ppc_run.py` and `ppc.sbatch` simulate; `ppc_analyze.py` summarises.
  Replicates are in `ppc_{A,B}_f{0,1}.csv`. Summaries are in `ppc_per_stat.csv` (per statistic), `ppc_joint.csv`
  (joint check) and `ppc_separation.csv` (B vs D1/D2 separation per statistic).

Raw outputs (pyABC History DB `abc_smc_mc_{flux}.db`, `abc_smc_trace_mc_{flux}.csv`, posterior draws
`abc_smc_post_mc_{M}_{flux}.csv`, `abc_smc_posteriors.csv`, `abc_smc_modelchoice.csv`):
- run set A: `/gpfs/projects/kernlab/ssmall/illex_traj_abc_noeps/smc/`
- run set B: `/gpfs/projects/kernlab/ssmall/illex_traj_abc_s1e-1/smc/`
- superseded first attempt: `/gpfs/projects/kernlab/ssmall/illex_traj_abc/smc/`

Code: `/gpfs/projects/kernlab/ssmall/msinv_illex_abc` (branch checkout plus the smc.py fix) and
`/gpfs/projects/kernlab/ssmall/msinv_illex_abc_s1e-1` (the same plus the s prior and grid change). The run log is
`/gpfs/projects/kernlab/ssmall/msinv_illex_abc/logs/RUN_NOTES_2026-11-07.md`.
