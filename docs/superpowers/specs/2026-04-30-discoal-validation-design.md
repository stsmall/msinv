# discoal validation harness — design

**Date:** 2026-04-30
**Status:** spec, awaiting user approval
**Sibling track:** `2026-04-30-msprime-validation-extension-design.md`
(neutral demography). This is the second of three planned cross-simulator
validation tracks (msprime → **discoal** → SLiM).

## Goal

Validate msinv's sweep model (`Sweep` operator, joint forward WF
trajectory) against discoal's selective-sweep simulator across five
scenarios: neutral baseline, hard sweep, soft sweep from standing
variation, partial sweep, and focal-site recurrent sweeps. discoal is
the canonical sweep simulator from the Kern-Schrider lab and the
natural cross-engine reference for the sweep code path that has no
analytical anchor at the distributional level.

This harness is panmictic, single-pop, no-inversion. discoal cannot
model inversions or joint kary×allele trajectories; SLiM (the third
track) covers the inversion+selection cross-engine validation.

## Scope

Five scenarios, all single-population, all without inversions. discoal
v2.0.0-beta provides native flags for all five.

| Scenario | What it tests | discoal flags | msinv side |
|---|---|---|---|
| **D1 — neutral baseline** | convention parity (θ/ρ/Ne, no sweep) | (no `-ws`) | `Sweep=None` |
| **D2 — hard complete sweep** | hitchhiking footprint, fixation | `-ws τ -a α -x 0.5 -N Ne` | `Sweep` with `mode=Stochastic`, `target_freq=1.0` |
| **D3 — soft sweep from SV** | multi-origin trajectory at startup | `-ws τ -a α -x 0.5 -f 0.05 -N Ne` | `Sweep` with `initial_freq_a=0.05` |
| **D4 — partial sweep** | sweep stops at intermediate freq | `-ws τ -a α -x 0.5 -c 0.5 -N Ne` | `Sweep` with `target_freq=0.5` |
| **D5 — focal-site recurrent** | repeated sweeps at same locus | `-ws τ -a α -x 0.5 -uA λ -N Ne` | `Sweep` with `mode=Recurrent`, `recurrent_rate=λ` |

D5 maps to discoal `-uA` (rate of recurrent adaptive mutation during
the sweep phase) — the focal-site flavor of recurrent sweeps. The
genome-wide flavor (`-R`) is **out of scope** for this spec because
msinv v1's simulator hard-asserts at most one `Sweep` operator
(`rust/msinv-core/src/simulator.rs:236`) — multi-position recurrent
needs core simulator work first.

Out of scope for this spec: multiple sweeps at distinct positions
(needs simulator change), demographic events under sweep (needs joint
trajectory `pop_size_at(t)` from the msprime extension's deferred
items — already deferred in the sweep-rewrite spec), inversions
(SLiM track), multi-pop sweeps with migration (SLiM track).

## Convention bridge (load-bearing)

discoal v2.0.0-beta and msinv differ in three units:

| Quantity | discoal CLI | msinv | Conversion |
|---|---|---|---|
| Time | 4·Ne generations | generations | `tau_disc = gens / (4·Ne)` |
| Selection coeff | `α = 2·Ne·s` (`-a α`) | `s` | `s = α / (2·Ne)` |
| Population mut rate | `θ = 4·Ne·μ·L` (`-t θ`) | `μ` per gen per bp | `μ = θ / (4·Ne·L)` |
| Population recomb rate | `ρ = 4·Ne·r·L` (`-r ρ`) | `r` per gen per bp | `r = ρ / (4·Ne·L)` |
| Sample size | haploid n (`sampleSize`) | haploid samples | identical |
| Effective Ne for sweep | `-N Ne` (diploid) | `population_size=Ne` (diploid) | identical |

Both interpret `Ne` as **diploid**. This matches the msprime track's
`population_size` doubling convention from the other direction:
- msprime CLI takes haploid Ne (`ploidy=1`); we double on the msprime
  side from msinv's diploid Ne.
- discoal CLI takes diploid Ne (`-N`); we pass it directly.

The `-N Ne` flag is **required** for sweep models — it tells discoal
the population size for the sweep window, distinct from the implied
Ne from `-t θ` and `-r ρ`. We pass the same Ne to both
`-N` and the implied `θ/ρ` denominator, so they agree.

## Output format

Both engines produce **tskit tree sequences** that we read with
`tskit.load`. Critical flags:

- discoal: `-ts FILENAME -F` — write tree sequence with full ARG
  (records all recombination edges including non-ancestral, matching
  msinv's convention). `-F` is the discoal analogue of msprime's
  `record_full_arg=True`.
- msinv: existing `HullSimulator(...).simulate()` output with no extra
  flags — msinv records full ARG by default.

This collapses the previous spec's "Class A site-mode" decision: with
`.trees` output on both sides we can compute branch-mode stats
directly, no `sim_mutations` step needed. **Class B (branch-mode
moments) + Class C (windowed branch-mode π) is the stat plan.**

The msprime track's spec at `2026-04-29-msprime-validation-design.md`
established `record_full_arg=True` is load-bearing for `n_trees`
agreement (msprime simplifies non-ancestral recombs by default,
producing ~78% of msinv's tree count). discoal `-F` is the equivalent
flag and serves the same purpose here.

## Per-scenario parameters

All scenarios use the canonical baseline:

- `Ne = 10000` (diploid)
- `n = 10` haploid samples
- `L = 100_000` bp
- `r = 1e-8` per gen per bp (`ρ = 40`)
- `μ` is **not used** — both engines run via tskit branch-mode stats
  with no mutations. discoal still requires `θ` to be passed (it
  governs the post-coalescent infinite-sites mutation placement which
  we ignore); we pass `θ = 40` for parity-debugging if the user ever
  enables site-mode stats by hand. Mutations are placed on a discoal
  branch-mode-equivalent path that doesn't affect tree topology.
- `seed_base = 0` (rep `i` uses seed `i` on each engine; engines have
  different RNGs so the comparison is distributional, not per-rep)

### D1 — neutral baseline

Both engines run with no sweep, no demography events. Confirms
convention parity (Ne, θ, ρ, n, L, time scaling) before any sweep
machinery is exercised.

**discoal:**
```
discoal 10 200 100000 -t 40 -r 40 -ts $tmp.trees -F -d $seed1 $seed2
```

**msinv:**
```python
HullSimulator(
    samples=10,
    population_size=10000.0,
    sequence_length=100_000.0,
    recombination_rate=1e-8,
    inversions=[],
    sweeps=None,
    seed=seed,
).simulate()
```

Pass criteria identical to msprime track N1: `pi_branch`, `n_trees`,
`mean_tmrca` agree within `3·SE`. K=10 windowed-π bins agree within
`3.61·SE` (Bonferroni z for K=10).

### D2 — hard complete sweep

Selective sweep at locus center, `s=0.05` (`α=1000`), completed
`τ=1000` generations ago.

**discoal CLI flags:**
- `-ws 0.025`  (= 1000 gen / (4·10000) = 0.025 in 4N coalescent units)
- `-a 1000` (`α = 2·Ne·s = 2·10000·0.05 = 1000`)
- `-x 0.5` (sweep at locus midpoint)
- `-N 10000`

**msinv:**
```python
from msinv.hull.sweep import Sweep

sweep = Sweep(
    x_sel=50_000.0,
    tau=1000.0,                       # generations: when sweep ENDED (backward)
    origin_pop=0,
    origin_kary='S',                  # placeholder; no inversions
    target_inv=0,                     # placeholder; no inversions
    mode='Stochastic',
    s=0.05,
    t_origin=1500.0,                  # when sweep STARTED in forward time;
                                      # > tau + sweep_duration. For s=0.05, Ne=1e4
                                      # the deterministic duration is ~ln(2N)/s ≈ 200 g,
                                      # so 500 g of buffer is plenty.
    f0=1.0/(2*10000),                 # one founding adaptive copy (hard sweep)
    partial_sweep_final_freq=1.0,     # complete fixation
)
HullSimulator(
    samples=10, population_size=10000.0,
    sequence_length=100_000.0, recombination_rate=1e-8,
    inversions=[], sweeps=[sweep], seed=seed,
).simulate()
```

Expected result: `pi_branch` reduction near `x_sel`, recovering with
distance. Pass = both engines agree on the depth and width of the
reduction.

### D3 — soft sweep from standing variation

Same as D2 but the beneficial allele starts at frequency `f₀=0.05`
(soft sweep — multiple haplotype origins).

**discoal:** add `-f 0.05` to D2 flags.

**msinv:** `Sweep(..., f0=0.05, ...)`, all other params identical
to D2.

### D4 — partial sweep

Sweep stops at frequency `0.5` instead of fixing.

**discoal:** add `-c 0.5` to D2 flags. (`-c` works alongside `-ws`;
the sweep stops at the given frequency rather than at fixation.)

**msinv:** `Sweep(..., partial_sweep_final_freq=0.5, ...)`.

Hitchhiking depth should be ~half of D2's (rough Kim-Stephan scaling).

### D5 — focal-site recurrent sweep

Recurrent adaptive mutations during the sweep window. discoal calls
this rate `uA`; msinv calls it `recurrent_mutation_rate` and uses
the same per-2N-per-generation convention.

**discoal:** add `-uA 1e-3` to D2 flags. (Rate per 2N individuals per
generation. Tunable; `1e-3` gives ~1 new origin per 100 gens at
Ne=10000.)

**msinv:** `Sweep(..., mode='Stochastic', recurrent_mutation_rate=1e-3, ...)`.

⚠️ **Empirical verification needed during plan execution**: J9
(`test_phase6b_sweep_joint.py:206`) tests recurrence counting under
`mode='Neutral'` (no selection) — it does NOT validate the
combination `Stochastic` + `recurrent_mutation_rate > 0` end-to-end.
The plan should add a quick smoke test that this combination
produces a non-empty trajectory with multiple adaptive origins
before attempting the full discoal cross-comparison. If the joint
trajectory builder has a latent bug in this combination, the
implementer surfaces it during scenario execution and STOPS — does
not adjust bounds. Same triage discipline as the C1 finding in the
msprime track.

This is the same path J9 already tests internally
(`test_j9_recurrent_de_novo_count`); D5 adds the cross-simulator MC
distributional check.

## Statistics

### Class B — moments (genome-wide)

Same set as the msprime track:

| Stat | tskit call |
|---|---|
| `pi_branch` | `ts.diversity(mode="branch")` |
| `n_trees` | `ts.num_trees` |
| `mean_tmrca` | weighted mean of `tree.time(tree.root)` |

Pass criterion: `|Δ| ≤ 3·sqrt(SE_msinv² + SE_disc²)` (3·SE bound,
matches msprime track).

### Class C — hitchhiking footprint (windowed π)

Compute `pi_branch` in **K=10 distance-from-sweep bins** of equal
width. Sweep at `x_sel=50_000`; for `L=100_000` the maximum distance
is 50_000 each side. Folded around the sweep center: bin `i` covers
distance window `[i·5000, (i+1)·5000]` in absolute distance, summing
both sides of the sweep.

Implementation:
```python
windows = [0.0]
for i in range(K):
    windows.append((i + 1) * 5000.0)
# windows = [0, 5000, 10000, ..., 50000] — K+1 edges, K bins
# Apply twice: [x_sel - dist_high, x_sel - dist_low] AND
#              [x_sel + dist_low, x_sel + dist_high]
# Sum the two per bin (folded).
```

Folded gives a single 10-bin profile per replicate. Mean and SE per
bin across replicates feed into the per-bin Bonferroni-corrected
bound.

For D1 (no sweep) the expected profile is flat — all 10 bins ≈ same
π. The check still applies; if any bin diverges between engines, it
indicates a windowing-code bug not specific to sweeps.

**Pass criterion (per bin):** Bonferroni-corrected two-sided z-bound
at family-wise α = 0.003 across the 10 bins. Per-bin α = 0.0003 →
z ≈ 3.61. Bound: `z · sqrt(SE_msinv² + SE_disc²)`.

For D2-D5 the expected profile slopes upward from low π near the
sweep to baseline π at the windows farthest from the sweep. msinv
and discoal should both produce that shape; agreement on each bin is
what the check validates.

### Why these stats

- **Moments alone** (Class B) catch convention/rate bugs but not
  spatial-shape bugs in the sweep.
- **Footprint alone** (Class C) catches spatial shape but is
  windowed and noisier per-bin than a genome-wide stat.
- **Both together** = spatial-aware regression net with diagnostic
  power: a moments-only failure points at θ/Ne/time conversion, a
  footprint-only failure points at the sweep semantics
  (recombination during sweep, hitchhiking model, partial vs full).

K=10 is chosen so the per-bin bound (`3.61·SE`) stays close to the
moment bound (`3·SE`). Doubling K to 20 would push the bound to
`~3.78·SE` and reduce sensitivity to small per-bin differences without
proportional gain.

## File layout

Three new files plus a refactor:

- **New**: `tests/hull/_discoal_bench_runner.py` — child-process
  runner. Spawns the discoal binary at `/home/adkern/discoal/discoal`,
  reads its `-ts` output via `tskit.load`, computes the same per-rep
  stat dict the msprime runner produces, prints JSON to stdout. Each
  invocation runs the full N-rep batch (`numReplicates=N`) in a single
  discoal call. The msinv runner stays separate and produces the same
  stat shape.

- **New**: `tests/hull/_validation_common.py` — extracted from
  `test_validation_msprime.py`: `_run_validation`, `_run_one_engine`,
  `_mean_se`, `_bonferroni_z`, `_agg_engine_vals`,
  `_print_benchmark_block`, `_git_short_sha`. Both validation tracks
  import from here. msprime helpers test file moves to
  `test_validation_common_helpers.py` (renamed) so the helper unit
  tests cover both tracks.

- **New**: `tests/hull/test_validation_discoal.py` — five test
  functions (`test_discoal_validation_d1_neutral` through
  `test_discoal_validation_d5_focal_recurrent`), each calling
  `_run_validation("d1")` etc. Re-uses the same `_run_validation`
  helper, parameterized by which engines to compare ("msinv vs
  msprime" for the msprime track, "msinv vs discoal" for this one).

- **Refactor (no behavior change)**: `tests/hull/test_validation_msprime.py`
  imports from `_validation_common.py` instead of defining helpers
  inline. The existing 6 tests still pass.

The Class C window list is per-scenario (sweep at center for D2-D5,
flat for D1), so it lives in the runner's SCENARIOS dict for each
track. The parent `_run_validation` doesn't need to know about
windows — it just sees additional stat keys (`pi_window_0`,
`pi_window_1`, ..., `pi_window_9`) in the per-rep stat dict, treats
them as a non-AFS family with a Bonferroni z, and applies the bound.

The naming convention `pi_window_K` distinguishes them from the AFS
keys (`afs_*`) so the parent's family-routing logic doesn't mix
them. The Bonferroni z applies separately per family — moments at
3·SE, AFS at z(K_afs)·SE, windows at z(K_win)·SE — though for the
discoal track AFS isn't computed (single-pop, no demographic-event
shape sensitivity warranted).

## Test budget

Estimated wall-clock at N=200 reps, single-pop scenarios:

| Scenario | per-rep msinv | per-rep discoal | 200 reps × 2 |
|---|---|---|---|
| D1 (neutral) | 1 ms | 5 ms | 1.2 s |
| D2 (hard sweep) | 5 ms | 30 ms | 7 s |
| D3 (soft sweep) | 5 ms | 30 ms | 7 s |
| D4 (partial sweep) | 5 ms | 25 ms | 6 s |
| D5 (recurrent) | 8 ms | 50 ms | 12 s |

Subprocess overhead: ~0.7 s × 5 scenarios × 2 engines = ~7 s.
Estimated total ~40 s. Comfortably within the 180 s budget
established by the msprime track. discoal is the slower side
(simulating the forward sweep is harder than discoal's neutral
default); D5 is the slowest because of recurrent-mutation event
density.

`N_REPS=200` matches msprime track. To sharpen drift detection,
bump it; budget allows N up to ~1000.

## Benchmarks

Same pattern as the msprime track: per-engine wall-clock + peak RSS
via subprocess isolation + `os.wait4` rusage. Persisted to
`.tmp/discoal_validation_bench.jsonl`. Same JSON record shape as
the msprime side (with engine fields renamed `msinv` / `discoal`).

The benchmark print block compares msinv-vs-discoal directly:

```
[d2 hard sweep] benchmarks
  msinv:   per-rep   5.4 ms ± 0.1, total  1.1 s, peak RSS  60.7 MB
  discoal: per-rep  31.2 ms ± 0.4, total  6.2 s, peak RSS  18.4 MB
  ratio:   per-rep msinv/discoal = 0.17x;  RAM msinv/discoal = 3.30x
```

(msinv typically faster but uses more RAM than discoal's compact
in-memory ARG. Ratios are illustrative; real numbers come from
implementation.)

## Triage table

| Failing pattern | Most likely cause |
|---|---|
| D1 fails moments only | `Ne`/`θ`/`ρ`/time convention bridge bug |
| D1 footprint windows fail uniformly | windowing helper bug (msinv side, since both engines should be flat) |
| D2 moments OK, footprint slope wrong | sweep `s` or `tau` conversion bug |
| D2 moments fail (low π far from sweep) | `Ne` or `θ` mismatch unrelated to sweep |
| D3 fails near sweep but D2 passes | soft-sweep `f₀` convention; verify `initial_freq_a` semantics on msinv side |
| D4 footprint shallower than D2 in BOTH engines but disagreement | partial-sweep `target_freq` semantics |
| D5 fails distributionally on D5 only | recurrent-rate `uA` ↔ `recurrent_rate` units (per-2N-per-gen vs per-gen) |

Same triage discipline as msprime track: a failure flags a regression
or convention mismatch; investigate before adjusting bounds.

## Out of scope / explicitly deferred

- **Genome-wide recurrent sweeps (`-R`)**: requires lifting the
  `simulator.rs:236` single-sweep assertion + multi-position event
  loop work. Separate spec.
- **Sweeps under demographic events**: requires `pop_size_at(t)`
  finalization in the joint trajectory builder (deferred from the
  sweep-rewrite spec).
- **Multi-pop sweeps with migration**: requires multi-pop
  `pop_size_at` + per-pop sweep state. Deferred to SLiM track or a
  later discoal extension.
- **Inversions**: SLiM track (third planned track).
- **Site-mode summary stats** (Class A): branch-mode is sufficient
  for distributional comparison; site-mode adds a `sim_mutations`
  step and a `μ` tuning question without changing the regression
  signal. Skip.
- **AFS bin checks for D1-D5**: the msprime track exercises AFS for
  demographic-shape sensitivity; the sweep test isn't sensitive to
  AFS in a way that adds value beyond the windowed-π Class C check.
  Skip.

## References

- `docs/superpowers/specs/2026-04-30-msprime-validation-extension-design.md`
  — sibling track; conventions and test framework reused.
- `docs/superpowers/specs/2026-04-28-sweep-rewrite-design.md` — msinv
  joint forward WF sweep design (the engine being validated).
- `rust/msinv-core/src/sweep.rs`, `sweep_trajectory.rs` — msinv sweep
  implementation.
- `tests/hull/test_phase6_sweep.py`, `test_phase6b_sweep_joint.py`
  — existing internal sweep tests (T1-T5, J1-J9; analytical anchors).
- `rust/msinv-core/tests/sweep_kim_stephan_anchors.rs` — Kim-Stephan
  closed-form anchors (single-locus expectations).
- `/home/adkern/discoal/discoal` — discoal v2.0.0-beta binary.
- `/home/adkern/discoal/docs/discoal_msprime_parameter_guide.md` —
  discoal↔msprime convention bridge (which we extend to msinv).
- `/home/adkern/discoal/docs/yaml_configuration.md` — confirms time
  in 4N coalescent units, selection per-generation.
