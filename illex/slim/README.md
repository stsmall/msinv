# SLiM + ABC pipeline for the Illex chr2 inversion

Forward-in-time simulation of the chr2:60–80 Mb inversion, with rejection ABC, to
answer three manuscript questions the coalescent work could not:

1. **Can the inversion persist without selection?** → `P(s = 0 | data)` and a
   Bayes factor.
2. **How old is it?** → marginal posterior for `t_inv`, in generations = years.
3. **Why is it at an intermediate frequency (0.626)?** → joint posterior for
   `(s, h)`; `h > 1` is overdominance.

Background, provenance, and every fixed constant: `../NOTES_illex_biology.md`.

## Relationship to the existing sweep harness — read this first

This is a **sibling of an existing, validated harness**, not a new stack:

```
analysis/steps/14_sweep_seqmodel/scripts/harness/
    slim/hard_sweep.slim, soft_sweep.slim     <- the recipes this one copies
    slim/run_slim_sweep.py                    <- the driver pattern
    talapas/config.sh, gen_array.sbatch, submit.sh
```

`inversion_abc.slim` deliberately follows those recipes' conventions:
`!exists()` defaults for every parameter, `initializeTreeSeq()`,
`community.rescheduleScriptBlock()` from a `1 early()` block, seed via
`slim -s <seed>`, restart-against-a-snapshot with the same prime seed stride
(7919), and one machine-readable result line (`[slim] INVERSION_RESULT status=…`)
for the driver to parse. `config.sh` copies `ACCOUNT=kernlab`,
`PARTITION=compute`, `ENV=illex_slimsim` and the `/gpfs/projects/$ACCOUNT/$USER`
scratch convention from `14_.../talapas/config.sh` rather than inventing them.
The seed base is `4e8`, distinct from the sweep campaign's `1e8`/`2e8`/`3e8`, so
seeds cannot collide across campaigns.

**SLiM ≥ 5.0 is required**, exactly as for the sweep recipes — the haplosome/tick
API. SLiM 4.x will not run this.

**Why a separate recipe at all**, rather than reusing `soft_sweep.slim`: the
selected element is an inversion, so recombination must be suppressed inside it
in heterokaryotypes. That barrier is the entire mechanism producing the observed
`dxy/π_I = 1.846` and `π_I/π_S = 0.744`, and no sweep recipe has it. Conditioning
also differs — on **segregation**, not fixation.

Two differences from the sweep recipes that are intentional, not drift:
- `R` defaults to **2.52e-9** (sex-averaged ReLERNN) where the sweep recipes use
  2.1e-9 (**male map only**). ~20% in ρ, and it sets the barrier's leakage scale.
- The forward window extends back to `T_INV` when the inversion is older than the
  growth epoch, where the sweep recipes always cover exactly the growth epoch.
  Either way the pre-forward history is constant `NREF`, so recapitation is
  unchanged.

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
| `config.sh` | Talapas settings, copied from the sweep harness |
| `submit.sh` | sizes and submits the array from `config.sh` |
| `submit_talapas.sbatch` | one array task |
| `collect.sh` | concatenate task TSVs, report failure breakdown |
| `smoke_slim.sh` | **run this first**, on a machine with SLiM ≥ 5 |
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
python -m illex.slim.run_one --task-id 0 --reps 1 --Q 200 \
    --out-dir ./.tmp/bench --slim $(which slim)

# 4. Launch. Account/partition come from config.sh (kernlab/compute).
DRYRUN=1 bash illex/slim/submit.sh      # check the sbatch line first
bash illex/slim/submit.sh

# 5. Collect and fit.
bash illex/slim/collect.sh "$OUTROOT/sims" "$OUTROOT/sims_all.tsv"
python -m illex.slim.abc_fit \
    --sims "$OUTROOT/sims_all.tsv" \
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

**chr2 is excluded from the existing ReLERNN run by design** —
`build_persex_vcf.sh` reads "autosomes (excl chr2 inv, chr42, chrZ)", because the
inversion's LD block would corrupt the fit. The six length-matched autosomes give
2.467–2.594e-9, so r is effectively known rather than free, and is held fixed.
`config.REC_RATE_BRACKET` has the male/female bracket for a sensitivity arm.

**A chr2-specific mask and ReLERNN map are being built.** When they land, set
`config.CHR2_RMAP` and `config.CHR2_MASK_BED` rather than editing `REC_RATE` by
hand, so the provenance stays visible. `config.rec_rate_for_inversion()` will
then return the length-weighted mean across the inversion body instead of the
autosomal proxy. Note a positional map cannot be applied to a 100 kb stand-in for
a 20 Mb inversion, so what the chr2 map buys is the correct **scalar** mean (and,
if wanted later, a heterogeneity sensitivity arm) — not per-window rates.
Likewise a chr2 mask improves the **observed** absolute levels
(`observed_targets.py`) and the accessible-fraction correction; it cannot be
applied positionally to the rescaled simulation, so the shape statistics continue
to assume masking is not diversity-biased. That assumption is worth checking once
the mask exists.

## Known limitations

- **`inversion_abc.slim` has not been executed.** No SLiM on the analysis box.
  `selftest.py` covers everything downstream; the `.slim` file follows the sweep
  recipes' SLiM 5.2 idiom closely but is unrun, so step 1 is not optional.
- Single panmictic population. Justified by geographic Fst ≈ 0, but it means the
  model cannot express local adaptation as the balancing mechanism.
- `WALLTIME` and `MEM` in `config.sh` are guesses pending step 3. The cost is
  dominated by the recent high-N phase, and unlike the sweep recipes this one may
  run a longer forward window (back to `T_INV`, not just the growth epoch) when
  the inversion is old — so per-sim cost rises with `t_inv`, and the prior goes to
  3e6. Budget from the benchmark, not from the sweep campaign's numbers.
- The restart loop can be the dominant cost for parameter combinations where the
  inversion rarely survives. `MAX_RESTARTS` (default 20,000) bounds it, and those
  runs are recorded as `abort_restarts` rather than silently retried forever.
- Rejection ABC with 23 statistics is at the edge of where curse-of-
  dimensionality starts to bite. If the posterior looks prior-like, the first
  thing to try is reducing the SFS to fewer bins (`config.SFS_BINS`) rather than
  loosening `--tol`.
