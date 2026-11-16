# Full-length PPC: LD misfit, manuscript corrections, model changes, simulated length

2026-11-15. Working note only. The manuscript has not been changed.

## 1. Result (block-pooled PPC, 474 x 40 kb body = 18.96 Mb, 1125 x 40 kb control = 45 Mb)

Joint predictive p, MAD-scaled L2 distance (ppc_joint.csv, joint_subsets.tsv):

| statistics in joint test | 40 kb, flux | 40 kb, no flux | full length, flux | full length, no flux |
|---|---|---|---|---|
| all 39 fitted | 0.51 | 0.089 | 0.044 | 0.112 |
| without LD bins 1-2 | 0.58 | 0.089 | 0.127 | 0.121 |
| without LD bins 1-3 | 0.60 | 0.089 | 0.169 | 0.121 |
| without any LD | 0.51 | 0.089 | 0.295 | 0.123 |
| LD only | 0.32 | 0.62 | 0.007 | 0.005 |

These statistics fail at full length with flux (two-sided p < 0.05):
- ld_S_1-3: observed about 1.17-1.18, predicted 0.89-1.08
- ld_S_6: observed 1.064, predicted 1.10-1.23
- ld_I_1-2: observed about 1.65
- jsfs_11: observed 0.064, predicted 0.011-0.058
- jsfs_14

Across the posterior, gamma (gene flux) correlates negatively with ld_S_1/ld_S_2 (r = -0.73 and -0.86) but positively with jsfs_11 (+0.35). No draw fits both, so this is a structural conflict and not a matter of the posterior being in the wrong place.

Shape: in the model, the body/control LD ratio rises with distance. For 2+ it goes from about 1.0 in bin 1 to 1.2-1.3 in bins 4-5. The data are flat at about 1.18 over 100 bp-8 kb and then fall to 1.06 in bin 6. So relative to the model, the data have too much LD at short range and too little at the longest range.

## 2. Checks already done (scratchpad)
- **Call-rate filter** (ld_callrate.tsv). Keeping only sites called in at least 70% or 80% of each subsample lowers raw r2 in both body and control. The body/control ratio does not fall. It rises:
  - 2+ bin 1: 1.18, then 1.24 at 70%, then 1.30 at 80%
  - 2a bin 1: 1.66, then 1.74, then 1.88
  - 2a bin 6: 1.11, then 1.27, then 1.56

  The excess is therefore not driven by poorly called sites.
- **Per-Mb pattern** (ld_blocks_obs.tsv, callrate_blocks.tsv). The excess is spread evenly along the inversion. Per-Mb bin-1 r2 correlates with call rate at -0.57 (2+) and -0.73 (2a).
- **Control choice.** Bin-1 r2 is 0.12 in the old 10-30 Mb control and 0.165 in the fitted control (ctl2). The ratio depends on the control.
- **Exchange tract length** (tract_test.tsv). 300 bp, 2 kb and 8 kb tracts at the posterior gene-flux rate barely move the fit: ld_S_1 0.99-1.07, jsfs_11 0.025-0.030.

## 3. Manuscript statements contradicted by the full-length check
1. main.tex about l.160: "reproduce the data (joint posterior predictive p=0.51)". This is the 40-kb value. At full length it is 0.044.
2. main.tex about l.162: "No statistic falls below a two-sided predictive p of 0.05". False at full length (see section 1).
3. main.tex about l.162: "Without gene flux the joint fit is poorer (p=0.09)". At full length the order reverses: 0.11 without flux, 0.044 with it.
4. main.tex about l.164 and supplement l.150: the shared low-frequency cell, "p=0.07". At full length it falls outside the predictive range.
5. supplement l.144, "Simulated length": "widens the posterior rather than shifting it". True for the means. Not true for the predictive check, because the narrower full-length spread exposes the LD and sharing misfit.
6. supplement l.150, "Posterior predictive checks": the same p-values and the same "no statistic below 0.05" claim.
7. supplement Fig. S11 caption, l.361: 40-kb only. It needs the full-length check alongside, or in its place.
8. supplement l.138: genotype error is modelled through iid h and call rate. A reviewer will ask why the LD excess survives call-rate filtering.
9. main.tex l.52, orange list: "Analyses still open: M1-M3". M3 has now been run.

## 4. What the model would need in order to fit
These are listed by how well each one explains both the excess short-range LD and the excess shared low-frequency variants (jsfs_11), and by how cheap it is to test.

**A. Genotype-error model with per-individual heterogeneity.** No msinv change is needed. The simulation applies one iid dropout factor h and one call rate to every individual. In the data, depth varies by individual (about 2.5x mean). Low-depth individuals call homozygotes at many sites together, which adds correlated, LD-like signal across distances.
- If the body and the control differ in calling quality, this does not cancel in the ratio. The per-Mb correlation between r2 and call rate points to a calling-quality link.
- Read-level correlated errors would add to bins 1-2 only.
- Test: draw per-individual h and call rate from the observed per-individual Ho/He and missingness, separately for body and control, then rerun the block-pooled LD for about 50 posterior draws. This is cheap: LD only, a few CPU-hours.
- Weakness: call-rate filtering made the ratio larger, not smaller, so error is probably not the whole story.

**B. Long, centre-biased exchange between arrangements (double crossovers).** This needs an msinv change. Gene flux in msinv is currently gene conversion with 300 bp tracts, and tracts up to 8 kb were tested. Double crossovers in a 19-Mb inversion would move tracts of about 100 kb to Mb, mostly away from the breakpoints.
- Imported tracts at low frequency produce both shared low-frequency variants (jsfs_11) and admixture-type LD within the receiving arrangement over the 1-20 kb bins.
- This is the one mechanism that can raise both together, which removes the gamma conflict.
- The msinv change: a second exchange process with long tracts (rate and mean length as parameters) and a rate that rises with distance from the breakpoints. edge_centre then constrains the profile.

**C. Structure or selection within an arrangement.** This needs an msinv change, or the sweep-in-arrangement option with the caveats in msinv issue #22. The 2a excess (1.66-1.88) is much larger than the 2+ excess (1.18-1.30). That fits sub-haplotypes within the derived arrangement, for example a later sweep or hitchhiking inside 2a, or more than one 2a lineage. The model treats each arrangement as one panmictic class.
- Changes needed: either selected sweeps within the arrangement after the barrier forms, or two 2a sub-classes.
- Expected signature: raised LD at all distances in 2a, plus a skewed 2a SFS.

**D. A different within-arrangement recombination rate in the body** (the model uses the flank rho). This is low priority. A lower rate raises LD more at long range than at short range, which is the opposite of the observed shape.

Suggested order: A first (cheap, no new code in msinv), then B if A leaves the excess, with C as the alternative. Inference would be rerun only after you decide.

## 5. How long does the simulated segment need to be?
Source: msinv_pboot_wt/results/illex/npe/length_check/length_check.jsonl. 47 NPE draws (flux), each with the same path simulated at 40 kb, 160 kb, 640 kb, 2.56 Mb and 18.96 Mb. This ran on the pre-karyofix tree, but the scaling should carry over. Script: scratchpad/lenvar.py.

The predictive spread across draws is var(L) = a + b/L:
- a is variance from parameters and paths. It does not depend on length, and it makes up 97-100% of the full-length variance for every statistic.
- b/L is coalescent and sampling noise.

The table gives the SD at length L divided by the SD at 18.96 Mb (how much too wide the predictive distribution is):

| L | median over fitted stats | range | wall per sim | memory |
|---|---|---|---|---|
| 40 kb | 1.39 | 1.04-3.0 | 0.75 s | 0.3 GB |
| 160 kb | 1.13 | 0.96-1.76 | 3.6 s | 0.4 GB |
| 640 kb | 1.10 | 0.96-1.30 | 23 s | 1.1 GB |
| 2.56 Mb | 1.01 | 0.88-1.10 | 145 s | 4.3 GB |
| 18.96 Mb | 1 | - | 2380 s | 38 GB |
| block pool (474+1125 x 40 kb) | about 1 | - | about 600 s | small |

The 40-kb stats with the widest SD (2.0-3.0) are the long-range LD bins (ld_S_6, ld_I_6), jsfs_12, jsfs_14, sfs_S_1 and tajd_S.

From the fit, the length that brings the SD within 10% of full length is:
- median statistic: about 160 kb
- upper quartile: about 330 kb
- worst fitted statistic (ld_I_6): about 2.5 Mb

Each SD ratio rests on 47 draws, so it carries about ±0.1 of sampling error. edge_centre is an exception: at 2.56 Mb its SD is still 3.2 times the full-length value. It depends on the real distance between the breakpoints and the centre, so it needs the full geometry or a dedicated design.

Answer:
- **PPC:** a single 2.56-Mb segment matches the full-length spread for every fitted statistic. At about 145 s per draw, 1000 draws cost about 40 CPU-h, against about 180 CPU-h for the block pool.
- **640 kb** is within 10-30% and costs 23 s.
- **NPE training at full-length precision:** 60k sims would cost about 380 CPU-h at 640 kb and about 2,400 CPU-h at 2.56 Mb.
- Training at 40 kb gave correct posterior means (length check) but more spread, so the posterior is wider than it needs to be. The misfit in section 1 is not a training-length problem: it is the model.

## 6. Test A result: per-individual genotype error (2026-11-15)
Scripts and output are in the scratchpad: err_prep.py, err_profiles.npz, err_test.py, err_test.tsv.

**Observed per-individual error** on the LD SNP set:
- Ho/He ranges 0.10-0.60 across individuals (5-95%); call rate ranges 0.41-0.86.
- Ho/He and call rate are tightly correlated within an individual (r = 0.91-0.94).
- Each individual has the same error in the body and in the control (r 0.99-1.00).

**Design.** 48 NPE posterior draws (flux), each with 120 x 40 kb body blocks and 120 x 40 kb control blocks. Within a draw, the trees and the diploid pairing are shared, and three error models are compared:
- **V0:** iid, as in production.
- **V1:** per-individual call rate and h = 1 - Ho/He, taken from the actual LD-subsample individuals, with the same individuals used in the body and the control.
- **V2:** the V1 h plus the observed missingness pattern, mapped by position from a random stretch of the region.

Median body/control ratio, with the two-sided p of the observed value:

| class, bin | observed | V0 | V1 | V2 |
|---|---|---|---|---|
| 2+ bin 1 | 1.177 | 0.98 (p<0.02) | 0.98 (p<0.02) | 1.02 (p<0.02) |
| 2+ bin 2 | 1.193 | 0.96 | 0.97 | 1.02 |
| 2+ bin 3 | 1.191 | 1.01 | 1.02 | 1.11 |
| 2+ bin 6 | 1.064 | 1.17 (p<0.02) | 1.19 (p<0.02) | 1.44 (p<0.02) |
| 2a bin 1 | 1.655 | 1.11 (p<0.02) | 1.11 (p<0.02) | 1.12 (p=0.04) |
| 2a bin 2 | 1.677 | 1.09 | 1.10 | 1.12 |
| 2a bin 6 | 1.112 | 1.16 | 1.16 | 1.39 (p<0.02) |

**Conclusions:**
- V0 reproduces the block PPC.
- Per-individual error (V1) changes nothing. It is the same in the body and the control, so it cancels in the ratio.
- The observed missingness pattern (V2) raises only the long-range ratios, which makes the shape mismatch worse. It also moves simulated control raw r2 away from the observed value.
- No error model raises the short-range ratio. Together with the call-rate filter, this rules out genotype error as the cause of the excess short-range LD in the body.
- What remains are biological model omissions: long exchange tracts or double crossovers (B), or structure or selection within an arrangement (C).

## 7. Robustness of the main inference (suggested, not run)
Whether the misfit matters for the main conclusions can be checked with the existing simulations alone, without any new simulation:
- retrain NPE on the 27 non-LD statistics and compare t_inv, p0, s and p*;
- rerun ABC-RF model choice without the LD statistics.

Without LD, the full-length joint PPC already passes (p 0.30 with flux, 0.12 without).

## 8. Long-tract test, 2.56-Mb body (2026-11-15)
Scripts and output are in the scratchpad: longtract.py, longtract.tsv.

**Design.** 24 NPE posterior draws (flux). Each draw keeps one frequency path and is simulated at mean tract lengths of 300 bp, 20 kb, 100 kb and 500 kb. The per-site flux c is held at the draw's value, so the exchanged fraction stays the same and the tracts become longer and rarer. LD is divided by the simulated control means from the block PPC.

Medians (with the 5-95% range for the key statistics):

| tract | ld_S_1 | ld_S_3 | ld_S_5 | ld_S_6 | ld_I_1 | ld_I_6 | jsfs_11 | dxy/pi_I |
|---|---|---|---|---|---|---|---|---|
| 300 bp | 0.98 (0.92-1.02) | 1.02 | 1.23 | 1.17 | 1.11 | 1.16 | 0.030 (0.014-0.044) | 1.92 |
| 20 kb | 1.05 | 1.15 | 1.28 | 1.18 | 1.22 | 1.24 | 0.026 | 2.09 |
| 100 kb | 1.05 | 1.15 | 1.30 | 1.20 | 1.19 | 1.27 | 0.018 | 2.21 |
| 500 kb | 1.05 (0.99-1.14) | 1.15 | 1.31 | 1.20 | 1.25 | 1.24 | 0.010 (0.004-0.041) | 2.42 |
| observed | 1.165 | 1.18 | 1.15 | 1.06 | 1.65 | 1.11 | 0.064 | 1.95 |

The 300-bp row reproduces the block PPC (ld_S_1 0.98), so the 2.56-Mb body and the 474 pooled blocks agree.

**At the fitted flux rate, longer tracts do not fix the misfit:**
- Short-range LD rises a little: 2+ bin 1 goes from 0.98 to 1.05 and 2a bin 1 from 1.11 to 1.2. The gain saturates by about 20 kb and stays well short of the data (1.17 and 1.65).
- Long-range LD rises too (2+ bin 5 from 1.23 to 1.31 against 1.15 observed), so the shape mismatch gets worse.
- Shared low-frequency variants (jsfs_11) fall from 0.030 to 0.010, against 0.064 observed. The same exchanged fraction delivered in fewer, longer events carries fewer independent shared variants.
- dxy/pi_I rises from 1.92 to 2.42, against 1.95 observed.

**Not tested:** fitting tract length and rate jointly. A higher rate with long tracts might restore the sharing, but at 300 bp a higher rate lowers short-range LD. Testing this properly needs tract length as a free parameter and a new training table at 2.56 Mb.

## 9. ABC-RF model choice without the 12 LD statistics (2026-11-15)

Run: modelchoice_karyofix/abc_rf_noLD/ (mc_only.R copy, spec_noLD.json, 27 statistics, flux units only, same reference table and SMC acceptance rates).

| model | prob_stats noLD (orig) | vote noLD (orig) | prob_total noLD (orig) |
|---|---|---|---|
| B  | 0.142 (0.289) | 0.153 (0.431) | 0.664 (0.828) |
| D2 | 0.266 (0.309) | 0.320 (0.318) | 0.109 (0.078) |
| N  | 0.425 (0.235) | 0.417 (0.154) | 0.166 (0.057) |
| D1 | 0.167 (0.167) | 0.110 (0.097) | 0.061 (0.038) |

Accuracy CV 0.496 OOB 0.460 (orig 0.51 / 0.47). prob_total = combine_total(uniform prior, SMC acceptance rate, prob_stats); the acceptance term (B 6.4e-3 vs 5e-4 for the others) is unchanged, so B stays the best model at 0.66. The forest's own (statistics-only) support moves from B to N once LD is removed: the LD statistics carried much of the B-vs-N discrimination in the classifier.
Caveat: the acceptance rates come from the original ABC-SMC runs, whose distance included the LD statistics, so prob_total here is not fully LD-free. The LD-free part is prob_stats/vote, which favours N (0.42) over D2 (0.27) and B (0.14) with ~50% classifier accuracy.

## 10. NPE (B, flux) without the 12 LD statistics (2026-11-15)

Run: npe_karyofix/f1_noLD_v2/ (msinv_pboot_wt train_npe.py via scratchpad wrapper; first attempt f1_noLD used the stale karyofix copy and crashed in calibrate on a cuda/cpu mismatch). Calibration fine (cov95 0.94-0.96, SBC KS p 0.09-0.91).

| parameter | no LD median (95%) | original median (95%) |
|---|---|---|
| t_inv | 916 ka (492-1373) | 390 ka (262-718) |
| t_sel | 393 ka (103-1038) | 307 ka (193-523) |
| p0 | 0.048 (0.003-0.097) | 0.051 (0.004-0.097) |
| s | 2.5e-4 (2.7e-6-0.072) | 1.6e-3 (1.2e-5-0.079) |
| p* | 0.601 (0.544-0.869) | 0.603 (0.575-0.707) |
| gamma | 6.3e-7 (5.0-7.6e-7) | 3.9e-7 (3.7e-9-6.1e-7) |

t_inv prior is 100 ka-1.5 Ma, so the no-LD upper bound approaches the prior edge. LD statistics were what pulled the age young and constrained p* and t_sel. gamma is higher and bounded away from zero without LD.

## 11. Draft paragraph (not in manuscript)

See chat of 2026-11-15; text kept below.

The full-length posterior predictive check shows that the balancing model with gene flux reproduces diversity, divergence and the joint frequency spectrum of the inversion but not its linkage disequilibrium within arrangements. Observed short-range r² within arrangements exceeds the simulated value by a factor of 1.18 to 1.30 in 2+ and 1.66 to 1.88 in 2a, long-range r² is lower than simulated, and polymorphisms shared between arrangements are more common than predicted. Genotyping error does not explain this pattern. Simulations that apply each individual's observed call rate and heterozygote deficit, with or without the observed missingness, give short-range r² in 2a of about 1.1 times the collinear value against an observed 1.66. Two processes could produce the excess. The first is selection within arrangements. Linked selection on variants that arose after the inversion, or several selected haplotypes maintained inside 2a, would add short-range associations and substructure within the derived arrangement in particular, consistent with the larger excess in 2a. The second is gene flux that exchanges longer blocks than our model allows. Our simulations used conversion tracts of a few hundred base pairs. Lengthening tracts to 20 to 500 kb at the fitted per-site rate did not raise short-range r² in 2a to the observed level (1.19 to 1.25 against 1.65) and reduced shared polymorphism, so longer tracts at the fitted rate do not account for the data. A model that estimates tract length and rate jointly remains to be tested. Removing the LD statistics from inference leaves the equilibrium frequency unchanged (p* 0.60, 95% interval 0.54 to 0.87) and the balancing model the most probable (0.66 against 0.83 with LD), although the random forest alone no longer separates balancing selection from drift on the remaining statistics. The inversion age becomes older and less precise (916 ka, 492 to 1373 ka, against 390 ka, 262 to 718 ka), so the LD statistics drive the younger age estimate.

Added to main.tex Discussion in c8cfb59 (after the "selective mechanism" paragraph) without the last two sentences on the no-LD fits. No-LD table stays in this note only.

Section 3 corrections are done in the manuscript: 858fc33 (Results, Methods and supplement report the 40-kb check, p 0.51, and the full-length check, p 0.04 with LD and 0.30 without; Fig. S11a/b), 73513c5 (why the statistics transfer across lengths, wider posteriors, cost of full length), 0321d5b (orange list).

## 12. Full-length rerun plan (decided 2026-11-15, running on Talapas)

### Decisions (user)
- Only the balancing model with gene flux. No ABC-RF rerun and no no-flux NPE: the 40-kb runs already establish flux and balancing.
- Train the NPE on simulations of the whole 18.96-Mb inversion, run on Talapas.
- Tract length is free, through a second flux process (option "two processes"). Short gene conversion stays at c (logU 1e-9..1e-5, mean tract 300 bp). The new long-tract exchange process approximates double crossovers in heterokaryotypes:
  - c_long: logU 1e-11..1e-6 per site per generation. The lower end effectively switches the process off.
  - l_long: logU 10 kb..5 Mb mean tract.
  - Both processes are geometric and independent.
- Save everything we may need, so these simulations never have to be run again.
- SAVE_TS_EVERY = 1000.

### Why the old fits need rerunning beyond the length
The 40-kb fits used msinv 0.4.1 (36a8b77). The LD-relevant fix #21 is in them: crossover suppression in heterokaryotypes, which had made rho_S 2.7x too high. The flux fix #28/#29 (0.4.2, merged 2026-11-12) is not. Under 0.4.1, a lineage carrying only small pieces of the inversion got far too little flux, down to 0.4% of the forward rate. Realized flux was therefore below nominal, which may bias the fitted gamma and the sharing/LD balance. The long-tract test (section 8) also ran on 0.4.1.

### Code
- **msinv PR #31** (squash-merged to main as 7cf1746; feature branch feat/long-flux-30, 191f961). It adds InversionSpec.long_flux_rate, long_mean_tract_length and long_tract_distribution to both engines. Rates add over the processes, and each event is drawn per process.
  - Tests: the Rust two-process rate equals the sum, and per-site cover matches the forward process; the Python reference engine; the end-to-end hit rate.
  - Results: cargo tests pass; pytest 355 passed and 9 failed, the same 9 that fail on main; CI green on Python 3.10, 3.11 and 3.12.
  - Issue #30 stays open for item 2 (a window inside a longer inversion) and item 4 (a forward/SLiM check of long tracts).
- **Branch illex/fulllength** (msinv_dir/msinv_full worktree; 05501bd + main + #31; latest 2aa1548):
  - illex/abc/fulllen.py: the row simulator. The 42 statistics use the statvec functions and streams, and a test pins them to region_stats. Rows are divided by a 45-Mb control from a bank.
  - illex/abc/control_bank.py and run_full.py: the drivers.
  - statvec.ld_block_sums gains an edges argument (default unchanged).
  - talapas/full.sbatch, npe/submit_full.sh and talapas/TALAPAS_FULL.md: the brief for the Talapas agent.
  - tests/illex/test_abc_fulllen.py.

### Saved per row (beyond the 42 statistics, params and seeds)
- raw body statistics before control division, and the control's bank index and seed
- LD sums and counts per 1-Mb block (6 fitted bins)
- 12 fine LD bins (100 bp to 20 kb), with and without the error model
- LD 20 kb to 2 Mb on thinned sites (~1 per 2 kb)
- unfolded spectra per class, and the exact sparse joint count spectrum
- site-mode pi, dxy and segregating sites
- per-Mb branch pi_I, pi_S and dxy
- edge profile (20 windows)
- frequency path (up to 200 points)
- the bank keeps the same material for each control

### Measured on the box
- **0.4.2 benchmark, 4 karyofix posterior draws, body only (finished 2026-11-16):** 18.96 Mb took 49, 62, 82 and 466 min with peaks of 61, 72, 91 and 152 GB (0.4.1 had been ~40 min and 38 GB). 40 kb took 1 s.
  - Cost rises steeply with selection strength. The 466-min, 152-GB draw had s = 0.094 (t_inv 370 ka); the others had s from 5.5e-5 to 0.012.
  - The first two draws alone gave the earlier 49-62 min, 60-72 GB estimate, which was too low.
  - Prior draws on Talapas may hold more strong-selection rows than these posterior draws.
- **End-to-end smoke run (run_full + control_bank, real lengths):**
  - Control: 12 min, 13 GB.
  - Rows: 33 min/36 GB (t_inv 346 ka) and 5 min/6.5 GB (t_inv 1.29 Ma).
  - All 42 statistics finite, and all extras present.
  - Sizes: ~35 kB per row gz, 220 MB per tree sequence.

### Talapas run
- Locations:
  - REPO /gpfs/projects/kernlab/ssmall/msinv_illex_full (branch illex/fulllength), env illex_full (separate from illex_abc).
  - FULL_OUTROOT /gpfs/projects/kernlab/ssmall/illex_full/{bank,table}.
- Bank: 100 tasks x 8 controls (4 CPUs, 80G).
- Rows (resized 2026-11-16, 9be20ad, after the slow benchmark draw): 40,000 tasks x 2 rows = 80,000 rows, 1 CPU, 200G, 24 h per task.
  - The earlier default (2 workers in 180G, 8 rows, 12 h) would have run out of memory or time on a draw like the 466-min one.
  - Four arrays of 10,000: ROW_OFFSET 0, 10000, 20000, 30000 (STAGE=rows after the first). Row task id and seeds = ROW_OFFSET + array index.
  - Total ~100-220k CPU-h (median vs mean of the four benchmark rows), memory-bound at 200G per row.
  - A row that runs out of memory or time is reported, not resized, by the Talapas session.
  - Tree sequences kept every 1000th row (~80 x 220 MB).
- Sequence: install, tests, DRYRUN, pilot (2+2 tasks), then STOP for sizing before production. Started 2026-11-15.

### After the run (box)
1. Collect to npz with 8 parameters: t_inv, u_sel, log10_p0, log10_s, p_star, log10_c, log10_c_long, log10_l_long.
2. train_npe: msinv_pboot_wt copy (device-safe), sbi ensemble of 5 NSF, calibration on held-out rows.
3. Full-length PPC from the new posterior.
4. Check whether the long-tract process absorbs the within-arrangement LD and shared-variant misfit (sections 1, 8). If it does not, the selection-within-arrangement explanation stands (Discussion).
5. Update the manuscript only when the user asks.
