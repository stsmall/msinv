# SLiM + ABC pipeline for the Illex chr2 inversion

Forward-in-time simulation of the chr2:60–80 Mb inversion, with rejection ABC, to
answer three manuscript questions the coalescent work could not:

1. **Can the inversion persist without selection?** → `P(s = 0 | data)` and a
   Bayes factor.
2. **How old is it?** → marginal posterior for `t_inv`, in generations = years.
3. **Why is it at an intermediate frequency (0.626)?** → joint posterior for
   `(s, h)`; `h > 1` is overdominance.

Background, provenance, and every fixed constant: `../NOTES_illex_biology.md`.

## Why forward, when msinv already exists

msinv (backward/ARG) is the right tool for the diversity ratios and produced the
current estimate. It structurally cannot do two things:

- **Neutral persistence.** A genuinely neutral trajectory is not samplable
  backwards at Illex Ne — msinv's stochastic trajectory samplers cap at N ≲ 10⁴,
  and P(a neutral mutation reaching p = 0.626) ≈ 1.1e-7, so forward rejection on
  a coalescent model yields zero survivors. A scaled forward WF model samples it
  directly, and the restart counter becomes data.
- **Explicit fitness.** Overdominance maintaining a stable intermediate frequency
  needs a forward fitness model.

Illex is annual and semelparous with non-overlapping generations, so discrete-
generation WF is a good structural match and generations are years.

## Files

| File | Role |
|---|---|
| `inversion_abc.slim` | the model (SLiM 4.x) |
| `config.py` | fixed constants, priors, statistic contract — **single source of truth** |
| `summarize.py` | tree sequence → statistic vector (recapitate, mutate, split, measure) |
| `run_one.py` | one array task: draw priors → SLiM → summarize → TSV |
| `observed_targets.py` | the empirical statistic vector to match |
| `abc_fit.py` | rejection ABC + regression adjustment + the three answers |
| `submit_talapas.sbatch` | SLURM array |
| `collect.sh` | concatenate task TSVs, report failure breakdown |
| `smoke_slim.sh` | **run this first**, on a machine with SLiM |
| `selftest.py` | validates everything except the `.slim` file |

## Order of operations

```bash
# 0. Locally: everything except SLiM itself.
.venv/bin/python -m illex.slim.selftest

# 1. On Talapas (or any box with SLiM): smoke-test the model. Minutes.
bash illex/slim/smoke_slim.sh

# 2. Observed targets. Needs pg_gpu, so a different interpreter.
/home/ssmall/miniforge3/envs/varbuddy-pggpu/bin/python \
    -m illex.slim.observed_targets --out results/illex/abc_observed.json

# 3. Benchmark ONE production-Q simulation before committing the array.
REPS=1 QSCALE=200 .venv/bin/python -m illex.slim.run_one \
    --task-id 0 --reps 1 --Q 200 --out-dir ./.tmp/bench --slim $(which slim)

# 4. Launch. --account is mandatory on Talapas.
sbatch --account=<your_pirg> --partition=compute illex/slim/submit_talapas.sbatch

# 5. Collect and fit.
bash illex/slim/collect.sh results/abc results/abc/sims_all.tsv
.venv/bin/python -m illex.slim.abc_fit \
    --sims results/abc/sims_all.tsv \
    --observed results/illex/abc_observed.json \
    --tol 0.005 --out results/illex/abc_posterior.json
```

**Do not skip step 3.** The cost is dominated by the recent high-N phase and I
have not been able to benchmark it (no SLiM on the analysis box), so the
`--time=12:00:00` in the sbatch file is a guess, not a measurement.

## The scaling factor Q — the one knob that matters

Illex N0 = 6,808,096 cannot be simulated directly. Standard Q-scaling divides N
by Q and multiplies µ, r, s by Q, preserving ρ = 4Nr, θ = 4Nµ and Ns.

It has a hard validity limit: **the scaled selection coefficient must stay
small.** `inversion_abc.slim` aborts if `s·Q ≥ 0.1`. With the prior's
`s_max = 3e-4`, that caps Q at 333.

| Q | N0 scaled | N_ANC scaled | forward ticks (t_inv=8e5) | validity |
|---|---|---|---|---|
| 100 | 68,081 | 5,479 | 8,000 | safest, most expensive |
| **200 (default)** | **34,040** | **2,740** | **4,000** | `s·Q ≤ 0.06` ✓ |
| 500 | 13,616 | 1,096 | 1,600 | `s·Q ≤ 0.15` ✗ for large s |
| 2000 | 3,404 | 274 | 400 | smoke tests only |

If step 3 shows Q=200 is unaffordable, **reduce the `s` prior upper bound rather
than raising Q past 333** — otherwise the strongly-selected simulations abort and
the posterior on `s` is silently truncated from above, which would bias question 1
toward "neutral".

Validity check worth running once: at s = 0 the simulated `pi_s` should match
4·N_e·µ for the growth history. If it does not, the scaling or the recapitation
is wrong.

## Design decisions you should know about before trusting the output

**The inversion is simulated at 100 kb, not 20 Mb.** Licensed by the verified
L-invariance of per-site π and dxy — worst-case 2.1%/1.8% bias extrapolating to
20 Mb (NOTES §7.3). Flux tract length is held at `w = tract/inv_len = 1e-4` so
the flux geometry is scale-invariant too. **This licence does not extend to
r²-vs-distance**, which is why no LD statistic is in the vector.

**Fst is deliberately excluded from the statistic vector.** `Fst = 1 − (r+1)/(2dr)`
exactly, where r and d are the two fitted ratios — verified to 2.2e-16 over 600
simulations. Including it would add a column that is a deterministic function of
two others, which inflates apparent agreement without adding information
(NOTES §5.3). `selftest.py` asserts it stays out.

**The within-arrangement folded SFS shapes are the identifying statistics.** Two
ratios cannot pin three parameters. The SFS shapes are normalized (so no
accessibility mask needed) and respond to `t_inv` and `p_start` differently from
mean π: a young inversion from few founders leaves a different within-I spectrum
than an old one from many, at matched π_I/π_S. This is the piece that breaks the
ridge.

**Absolute π is off by default.** A ~1.31× calibration offset survives the
accessibility mask and is present in the *collinear control* too, so it is a µ/Ne/
filtering issue, not the inversion model (NOTES §8.3). `--use-absolute` exists but
needs a nuisance scale parameter; do not use it naively.

**Conditioning is on segregation, not on frequency.** Simulations restart if the
inversion is lost or fixes — legitimate, since we observe a segregating
inversion. `p_final` is recorded as a *statistic* and left for ABC to match.
Conditioning on 0.626 in the simulation would use the same datum twice.

**Failures are recorded, not dropped.** `collect.sh` prints the status breakdown
because it is itself evidence: many `lost_too_often` rows at low `p_start` is a
direct measurement of how hard it is for a single-founder inversion to reach
p = 0.626 — which is question 1 from a different angle.

## Fixed inputs and their provenance

| Input | Value | Source |
|---|---|---|
| Demography | N_ANC 547,928 → N0 6,808,096 over 769,519 gen | moments SFS fit |
| Coalescent | Kingman (Beta rejected at every α) | `illex/scripts/beta_vs_kingman.py` |
| µ | 3e-9 | external; **every age scales inversely with it** |
| r | 2.52e-9 sex-averaged | ReLERNN, `11_relernn/` |
| Accessibility | 47.91% in the inversion body | `degenotate_illex/accessible_sites.bed` |
| p_inv | 0.626 | karyotypes, baker-633 |

**chr2 has no recombination map of its own, by design** — `build_persex_vcf.sh`
excludes it ("autosomes (excl chr2 inv, chr42, chrZ)") because the inversion's LD
block would corrupt a ReLERNN fit. The six length-matched autosomes give
2.467–2.594e-9, so r is effectively known rather than free, and is held fixed.
`config.REC_RATE_BRACKET` has the male/female bracket for a sensitivity arm.

## Known limitations

- **`inversion_abc.slim` has not been executed.** No SLiM on the analysis box.
  `selftest.py` covers everything downstream; the `.slim` file is carefully
  written but unrun, so step 1 is not optional.
- Written for **SLiM 4.x**. SLiM 5 renamed `Genome` → `Haplosome`, which breaks
  the `recombination()` callback and `p1.genomes`.
- Single panmictic population. Justified by geographic Fst ≈ 0, but it means the
  model cannot express local adaptation as the balancing mechanism.
- The `--time` and `--mem` in the sbatch file are guesses pending step 3.
- Rejection ABC with 23 statistics is at the edge of where curse-of-
  dimensionality starts to bite. If the posterior looks prior-like, the first
  thing to try is reducing the SFS to fewer bins (`config.SFS_BINS`) rather than
  loosening `--tol`.
