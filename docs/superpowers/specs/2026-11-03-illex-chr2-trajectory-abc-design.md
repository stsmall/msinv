# Illex chr2 inversion: trajectory ABC for the rise of 2a

Status: draft for approval, 2026-11-03. Replaces the grid fits (`refit_decline.py`, `refit_rise.py`) as the
inference for the manuscript. The grid and the pilots stay as the source of prior bounds.

## Intent

Find the history of the chr2 inversion (60.54–79.50 Mb) that fits the data, using an approach that holds up
better than a grid search. That means rejection ABC with model choice and posterior parameter estimates. The
question is posed the way a sweep is with discoal and diploSHIC:

1. **Stage 1, the rise.** Was 2a's rise hard (one copy under selection from its origin) or soft (standing
   variation at p0 when selection began)? Give the posterior share of each and the posteriors of the start
   frequency, barrier age, selection onset and strength.
2. **Stage 2, holding at 0.626.** Which kind of selection fits, and what would each model force us to accept?
   For example: selection strong enough to overshoot unless it is switched off, too little diversity after a
   fast rise, or a start so soft that 2a carries too much diversity.

The deliverable is a full description of which models fit and under which parameter ranges. The Discussion
then says how to test or discount the survivors. Selection is expected. The neutral model is a check that
selection is needed, not the hypothesis under test.

## Polarization and naming (decided)

- **2+** is the reference arrangement, the former BB cluster, present frequency 0.374. It is ancestral by
  assumption, supported by parsimony against *I. coindetii*, which carries its own larger inversion
  (reference ~44.4–79.5 Mb).
- **2a** is the arrangement inverted relative to the reference, the former AA cluster, at 0.626. It is
  derived. Karyotypes are 2++ (95), 2a+ (284) and 2aa (254).
- In msinv, the inverted class I is 2a and the standard class S is 2+.
- The scenario where 2+ is derived (rise and fall) is not refitted. It stays a Discussion alternative,
  described from the existing fits (manuscript commit 887a47b, msinv a9f92bb).

## Fixed inputs (decided)

- **Demography is the fitted moments growth model and is not varied.** N_ANC 547,928 to N0 6,808,096,
  exponential over 769,519 generations, constant before that. This is `illex/theory.py` and
  `illex/demography.py:growth_demography`, the same as manuscript Table `tab:demog`. μ = 3e-9.
  Generations are years.
- **Recombination:** the chr2 collinear rate from the current male ReLERNN map, the one the manuscript
  uses since commit 7d69fd2, with the 200-rep bootstrap correction. `illex/slim/config.py:REC_RATE` is
  1.977e-9, from the earlier map, and it is re-derived before any production run. The pilots used the old
  value.
- **Present frequency of 2a: 0.626.** Simulated paths are accepted only if they end within ±0.02. The
  binomial SE for 633 individuals is 0.014.

## Models (decided)

Every model draws its own stochastic frequency path forward in time, with Wright-Fisher drift on the growth
demography. Paths that are lost or that end away from 0.626 are rejected. The founding is a parameter inside
each model, not a separate model.

- **Hard start:** one copy, 1/(2N(t_inv)). Under selection from its origin (t_sel = t_inv). Exact WF until
  it is established, so the loss of single copies is modelled.
- **Soft start:** frequency p0 at t_inv. Neutral drift until t_sel, then selection.

| Model | Selection after t_sel | Extra parameters |
|---|---|---|
| N, neutral | none (s = 0), drift from the founding to the present | none |
| D1, directional, ongoing | genic, dp = s·p(1−p), to the present (2a may still be rising) | s |
| D2, directional, stopped | genic s from t_sel to t_stop, then neutral drift | s, t_stop |
| B, balancing | overdominance toward an equilibrium p*, to the present | s_het, p* (free in the prior) |

Each of the four models is run without and with gene flux, giving 8 models. Directional selection can end
at 0.626 in two ways, as decided:

- **D1, an incomplete sweep caught at the time of sampling.** Projected forward, 2a would keep rising and
  fix.
- **D2, a change in selection pressure in the past,** followed by drift.

**B is balancing throughout.** Overdominance carries a rare 2a up from its origin, because a rare
arrangement sits mostly in heterozygotes, and then holds it near p*. A directional rise followed by
balancing is not a model (decided).

**Discounting neutrality.** The neutral model is discounted by its own improbability as well as by the
statistics. Report:

- the probability that a new neutral copy establishes and reaches 0.626 by the present on this demography,
  which is about 1/(2N·0.626) ≈ 1.5e-6 for reaching it at all, before timing is considered
- the neutral soft-start equivalent: the chance of drifting from p0 to 0.626 by the present

Both are the p_now acceptance rates of model N (call 1 below) and go into the neutral model's posterior.

## Priors (bounds from the pilots, refined by the stage-1 pilot now running)

| Parameter | Prior | Basis |
|---|---|---|
| t_inv | U(0.1, 1.5 Ma) | grid fit ~850 ka; widened from 0.3 Ma after the ABC-RF pilot piled up at the 0.3 Ma edge in D1, D2 and N |
| founding | P(hard) = 0.5. Soft p0 ~ logU(1e-3, 0.10) | upper bound 0.10 fixed by the author on biological grounds (a soft start is standing variation at low frequency). A pile-up at this bound is read as misfit in another parameter or statistic, not as a reason to widen it |
| t_sel (soft) | U(0, t_inv) | |
| s, s_het | logU(1e-6, 1e-2) | up to strong selection, so the "super-strong" case is reachable |
| t_stop (D2) | U(0, t_sel) | |
| p* (B) | U(0.3, 0.95) | free, as decided |
| γ_eff (flux models) | logU(1e-9, γ_max) | effective per-site exchange rate (call 3), γ_max from calibration |

**Flux calibration.** The pilots found no visible effect of flux up to 1e-5 per bp per generation, and the
γ scale in msinv is uncalibrated. Before production, run a short sweep to find the γ at which flux
measurably changes dxy/π_2a and the breakpoint-to-centre profile. Set γ_max one to two orders above that, so
the flux models can express strong exchange.

## Summary statistics (decided)

Every observed statistic is computed from the data exactly as it is from the simulations. That means the
same sample sizes, the same projection, and genotype-based measures where the data are unphased.

- **The three ratios:** π_2a/π_2+, dxy/π_2a, and the singleton ratio f1(2a)/f1(2+). Current targets: 0.7377
  ± 0.0263, 1.8763 ± 0.0503 (corr −0.91), 1.2307 ± 0.0115. Sources: `illex/scripts/empirical_jackknife.py`
  and `sfs_f1_jackknife.py`. These are re-expressed with 2a as I, so they are not the old BB-as-I values.
- **The per-arrangement SFS for 2aa and 2++ homokaryotypes,** projected as in `sfs_shape_angsd.py`. Source:
  ANGSD SAFs in `analysis/steps/04_angsd_chr2/saf/{AA,BB}.2.saf.gz`.
- **The joint SFS of 2a × 2+,** binned as in `sfs2d_karyotype.py`.
- **Windowed π_2a, π_2+ and dxy along the inversion,** and the breakpoint-to-centre profile. Source:
  `empirical_windowed.py`.
- **LD decay within each arrangement,** from genotypes: r² between genotypes, binned by distance, in 2aa and
  2++ individuals separately. Simulated diploids are formed by pairing simulated chromosomes within each class.

The observed sample sizes are the homokaryotypes (2aa 254 and 2++ 95 individuals, 508 and 190 chromosomes),
and simulations sample to match. The singleton ratio keeps its ANGSD calibration (1/0.8256, control
1/1.0161).

## Inference (decided: hybrid, 2026-11-03)

The inference is a hybrid. Model choice uses a rejection reference table drawn from the prior. Parameters are
estimated by ABC-SMC within each model that survives.

- **Model choice: rejection ABC plus ABC random forest** (abcrf; Pudlo et al. 2016). The reference table is
  drawn from the prior, and ABC-RF gives the posterior model probabilities. Those are combined with the
  p_now term as call 1 below describes. ABC-RF also gives first-pass parameter posteriors (regAbcrf), and
  rejection-plus-regression posteriors are reported as a cross-check.
- **Parameter estimation: ABC-SMC with pyABC** (Klinger, Rickert & Hasenauer 2018,
  doi:10.1093/bioinformatics/bty361), run once for each model × flux with prob_total > 0.05.
  - The simulator is the same trajectory sampler and msinv pipeline. A path that misses p_now, or a hard
    start that fails to establish, returns an infinite distance. The p_now and establishment terms
    therefore sit inside the SMC posterior through the importance weights, not as a separate factor.
  - The distance is an adaptive weighted p-norm over MODEL_KEYS, re-weighted each generation by the spread
    of each statistic (Prangle 2017), against the observed `stats`.
  - SMC concentrates the simulations where the posterior is. That matters for the flux models, where a
    single run costs about 90 s and 2 GB at c = 1e-5.
- **Cross-checks between the two arms:**
  - One pyABC run with all 8 models together, giving its own posterior model probabilities, set beside the
    ABC-RF ones. Both are vulnerable to insufficient statistics (Robert et al. 2011), so agreement is the
    claim and disagreement is reported.
  - The ABC-SMC and ABC-RF posteriors for each surviving model are compared. Each median should fall inside
    the other's 50% interval, and any that does not is reported.
- **The reported parameter posteriors** are the ABC-SMC ones. The PPCs and the SLiM validation draw from
  them.
- **Validation:**
  - a confusion matrix on pseudo-observed data sets drawn from each model, to show the models can be told
    apart with these statistics
  - posterior coverage checks for the main parameters
- **Validation of the SMC arm:** coverage of the 95% intervals on 10 pseudo-observed data sets per surviving
  model, drawn from the ABC-RF posterior, since a full SMC per pod is costly.
- **Stage-1 answer:** the posterior share of hard against soft, and the p0 posterior, within each model and
  averaged over models. Both come from ABC-SMC, with the ABC-RF values as a cross-check.
- **Posterior predictive checks:** draw from the posterior of each model that fits, re-simulate, and compare
  every statistic to the data. That includes statistics left out of the fit, such as Tajima's D per
  arrangement and the edge-to-centre ratio of dxy/π.

## Independent simulators (decided)

- **msprime check.** msinv with no inversion (or t_inv = 0) against msprime on the growth demography and
  the same recombination rate, sample sizes and statistics. This covers the collinear flanks and the
  neutral limit. Extend `slim_validation/run_comparison.py` rather than writing new code.
- **SLiM validation of msinv.** Use the forward recipe `illex/slim/inversion_abc.slim` (SLiM ≥ 5.0),
  updated to the 2a-as-I polarization, the four selection models and gene conversion in heterokaryotypes.
  - Take 4 to 6 parameter sets from the ABC posterior, covering at least soft D1, hard B and D2 with flux.
  - Simulate each in SLiM (Q-scaled, recapitated in msprime) and in msinv, using paths drawn under the
    same parameters.
  - Compare the full distribution of every statistic. A disagreement is reported. It is not tuned away.
- **SLiM posterior predictive checks.** The final PPCs for the chosen model or models are generated in SLiM
  from posterior draws. msinv PPCs are kept alongside as a comparison.
- **Scaling limit.** SLiM's Q-scaling is valid only while s·Q < 0.1. Posterior draws with larger s use a
  smaller Q for those runs. The s prior is never truncated to fit SLiM.

## Deliverables

1. Scripts in msinv `illex/scripts/` for the trajectory simulator (generalising `abc_traj_pilot.py` to the 8
   models), the statistic pipeline, observed-statistic extraction, the ABC and validation, and the PPCs.
2. A results table for each model:
   - posterior model probability
   - PPC fit
   - parameter posteriors: t_inv, hard/soft share, p0, t_sel, s, t_stop, p*, γ, and the highest frequency
     reached
   - what the model requires us to accept
3. Figures: posterior trajectories of 2a frequency through time for each model that fits, the confusion
   matrix, and the ABC-SMC ε and acceptance-rate traces per generation.
5. ABC-SMC databases (pyABC History, one per model × flux) and a table comparing the ABC-RF and ABC-SMC
   model probabilities and posteriors.
4. **Flux side experiment (for the Discussion):** show how much flux erases the dxy signal of an older
   single-copy origin, so dxy cannot trace the origin past the point where flux has homogenised it.

## Compute

- Pilots run on the shared box within the load rule: user load under 150, and under 100 when the total load
  is over 200. Never touch `/sietch_colab/data_share/balanus` processes.
- Production runs on Talapas in two parts:
  - The reference table for model choice: about 10⁴ accepted rows per model × flux, about 8×10⁴ msinv runs.
    It is cut from the earlier 2×10⁴, because parameter estimation has moved to SMC.
  - ABC-SMC: population size about 1,000 per generation, run until the acceptance rate falls below 1% or ε
    changes by less than 5% between generations. pyABC's Redis sampler runs inside one multi-node Slurm
    job: one node hosts redis-server and the orchestrator, and the others run `abc-redis-worker`. Box runs
    use the multicore sampler within the load rule.
- Use msinv `main` (5199374 or later, SMC′ available) with Illex code from branch `illex/pboot-recovery`.

## Non-goals

- Refitting the demography.
- Refitting the 2+-derived rise-and-fall scenario.
- Manuscript edits. Those come after results and need approval.
- Modelling selection on individual sites inside the arrangements.

## Anti-goals

- Reporting a grid D as the inference.
- Statistics computed differently on the data and on the simulations.
- Truncated priors where the posterior piles up against an edge. Widen and rerun.

## Rules

- No msinv commits without approval.
- Never kill or stop a job unless the user says so.

## Calls made after the builder check (author's calls, open to change)

1. **The present frequency counts as data.**
   - Each model and founding class (hard or soft) gets its own simulation run. Its acceptance rate at p_now
     estimates P(p_now | model, class).
   - The posterior is prior × P(p_now | ·) × P(statistics | ·, p_now). The last term comes from ABC-RF on
     a reference table balanced across models and classes after the p_now filter.
   - So rare classes, such as a neutral hard start, still get enough rows, and the evidence in p_now is
     kept rather than discarded.
   - The stage-1 hard/soft share is reported both ways, with and without the p_now term.
2. **Artefacts are cancelled with a simulated collinear control.**
   - Every simulation carries collinear flanks, and each statistic is divided by its flank value in the
     data and in the simulations alike. f1 is already handled this way.
   - The joint SFS is folded, because the reference is a 2+ haplotype inside the inversion.
   - LD is computed from genotypes on the same accessible, depth-filtered site set in data and simulations,
     and also expressed relative to the flank.
   - If the LD PPCs show that the data artefacts dominate, LD is dropped from the fit and kept only as a
     PPC, and the user is told.
3. **Flux uses physical units.**
   - In msinv the per-site flux rate is γ·φ(x), with φ(x) = min(x, 1−x, w)/(1−w)
     (`msinv/hull/simulator.py:_phi`) and w = tract length / inversion length. Inside the inversion φ ≈ w,
     so the effective exchange rate is γ_eff = γ·w/(1−w).
   - The prior is placed on γ_eff, per bp per generation in heterokaryotypes. That is the quantity Korunes
     & Noor estimate, at 1e-5 to 2.5e-5. γ is derived from γ_eff for the simulated segment's own w.
   - Because φ is flat except within one tract of each breakpoint, gene flux in msinv produces no
     breakpoint-to-centre gradient. The edge-to-centre statistic is therefore a PPC check, not a fitted
     statistic. A double-crossover gradient is not modelled.
   - Tract length is fixed in base pairs, about 1 kb. The value is to be confirmed from the gene-conversion
     literature (Korunes & Noor) before the calibration.
   - The Peischl position profile uses the real breakpoint geometry, not the length of the simulated
     segment.
   - The pilots' TRACT_FRACTION rule on a ~37 kb segment gave tracts of a few base pairs. That is the likely
     reason flux showed no effect up to 1e-5, so the γ_max calibration is redone under this rule.
4. **Breakpoints are 60.55 and 79.52 Mb.** These are the assembly-alignment breakpoints, from the F24 hap1
   and reference unaligned gaps at 60,541,825–60,550,215 and 79,514,987–79,523,190. They match the
   differentiated extent of 60.54–79.50 Mb.
5. **Karyotype sampling is simulated.**
   - Each accepted path is sampled as 633 binomial karyotypes, and the observed p_now is a statistic.
   - The ±0.02 cut is a cheap pre-filter that sits wider than that sampling noise. The binomial SE is 0.014,
     so the cut is about 1.4 SE wide. Widen it to ±0.03 if the D1 and B posteriors press against it.

## Left to the builder

- How to get the windowed profile and LD at realistic cost: one long simulation of the whole inversion, or
  shorter segments placed at the edge, quarter and centre positions. The constraint is that the flux profile
  depends on position relative to the breakpoints.
- The SMC or SMC′ setting, the step size of the diffusion, the ABC tolerance, the number of random-forest
  trees, and the exact bins for the SFS and LD.
- R or Python for the ABC.
