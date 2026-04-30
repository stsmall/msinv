# msprime validation extension — design

**Date:** 2026-04-30
**Status:** spec, awaiting user approval
**Predecessor:** `docs/superpowers/specs/2026-04-29-msprime-validation-design.md`
(N1 panmictic + N2 two-pop migration; both shipped 2026-04-29)

## Goal

Extend the existing msprime validation harness from its current 2 scenarios
(constant-size hot path) to cover the four msinv demographic-event paths
that have no analytical anchor today: backward merge (`ej`), instantaneous
size change (`en`), exponential growth (`eg`), and the multi-population
event-interaction case (`ej` + ongoing migration). Adds a per-scenario
benchmark layer (wall-clock + peak RSS per engine) so cost is visible
alongside correctness.

This is the first of three planned cross-simulator validation tracks
(msprime → discoal → SLiM); the discoal and SLiM tracks get their own
specs once this lands.

## Scope

Four new scenarios, all neutral, all no-inversion, all within msprime's
native capability:

| Scenario | Tests | Headline event |
|---|---|---|
| **N3 — population merge** | `ej` path; `add_population_split` helper | two-pop merge backward |
| **N4 — bottleneck** | `en` rate-update path | step-down then step-up Ne window |
| **N5 — exponential growth** | `eg` time-varying rate path | constant growth backward |
| **N6 — three-pop with split** | event interaction (`ej` + migration) | three pops merge to one with ongoing migration |

Out of scope (deferred to future specs): inversions, sweeps, class
migration (`cmig`), per-pop growth differences (`eg` vs `eG`), admixture
(`add_admixture`), `inversion_freq_change`, asymmetric migration matrix
beyond N6's symmetric case.

## Per-scenario parameters

All scenarios use `n=10` total samples (5+5 or 4+3+3 for multi-pop),
`L=100_000`, `r=1e-8` (`ρ=40` per pop where applicable), N=200 reps per
engine, seed `i` on rep `i`. Engine equivalence follows the established
convention: msprime side `ploidy=1`, `record_full_arg=True`, and
`population_size` doubled (msinv N is diploid Ne).

### N3 — two-pop merge backward (`ej`)

Two extant populations with symmetric migration `M=1e-4` until time
`T_split=2000` generations, where they merge into a single ancestral
population. Sample 5+5 at present.

**msinv:**
```python
demo = Demography(
    pop_sizes=[10000.0, 10000.0],
    migration_matrix=[[0.0, 1e-4], [1e-4, 0.0]],
)
demo.add_population_split(time=2000.0, derived=[1], ancestral=0)
HullSimulator(
    sample_config={(None, 0): 5, (None, 1): 5},
    demography=demo,
    sequence_length=100_000.0,
    recombination_rate=1e-8,
    inversions=[],
    seed=seed,
).simulate()
```

**msprime:**
```python
demo = msprime.Demography()
demo.add_population(name="A", initial_size=20000.0)
demo.add_population(name="B", initial_size=20000.0)
demo.set_migration_rate(source="A", dest="B", rate=1e-4)
demo.set_migration_rate(source="B", dest="A", rate=1e-4)
# add_mass_migration with proportion=1.0 (per feedback_msprime_api.md;
# we do NOT use add_population_split because it auto-derives size /
# growth resets that don't match msinv's ej semantics).
demo.add_mass_migration(time=2000.0, source="B", dest="A", proportion=1.0)
msprime.sim_ancestry(
    samples={"A": 5, "B": 5},
    demography=demo,
    sequence_length=100_000,
    recombination_rate=1e-8,
    ploidy=1,
    record_full_arg=True,
    random_seed=seed + 1,
)
```

The migration after the merge (i.e. backward of T_split, when only pop A
exists) is auto-zero in msinv (`ej` puts everyone in one pop) and remains
zero in msprime (mass migration empties B; B's outgoing rates apply to no
lineages). No explicit `em` event is needed.

### N4 — bottleneck (`en`)

Single pop, present-day Ne=10000, drops to Ne=1000 at t=1000 (going
backward), recovers to Ne=10000 at t=2000. Two `en` events.

**msinv:**
```python
demo = Demography(pop_sizes=[10000.0])
demo.add_population_size_change(time=1000.0, population=0, new_size=1000.0)
demo.add_population_size_change(time=2000.0, population=0, new_size=10000.0)
HullSimulator(
    samples=10,
    demography=demo,
    sequence_length=100_000.0,
    recombination_rate=1e-8,
    inversions=[],
    seed=seed,
).simulate()
```

**msprime:**
```python
demo = msprime.Demography()
demo.add_population(name="A", initial_size=20000.0)
demo.add_population_parameters_change(
    time=1000.0, initial_size=2000.0, population="A")
demo.add_population_parameters_change(
    time=2000.0, initial_size=20000.0, population="A")
msprime.sim_ancestry(
    samples=10,
    demography=demo,
    sequence_length=100_000,
    recombination_rate=1e-8,
    ploidy=1,
    record_full_arg=True,
    random_seed=seed + 1,
)
```

The bottleneck depth and duration are deliberately sharp (10× drop, 1000
gens wide) to make the deviation from the constant-Ne SFS large enough
to detect at N=200 reps.

### N5 — exponential growth (`eg`)

Single pop, present-day Ne=10000, exponential growth (forward) at rate
`α=0.0005` per generation, applied at all times (no end). Equivalent to
`N(t backward) = 10000 · exp(−0.0005 · t)`.

**msinv:** msinv supports growth via `add_growth_rate_change`. Its
docstring says `N(t') = N(t) * exp(-growth_rate * (t' - t))` going
backward, matching msprime convention exactly.

```python
demo = Demography(pop_sizes=[10000.0])
demo.add_growth_rate_change(time=0.0, population=0, growth_rate=0.0005)
HullSimulator(
    samples=10,
    demography=demo,
    sequence_length=100_000.0,
    recombination_rate=1e-8,
    inversions=[],
    seed=seed,
).simulate()
```

**msprime:**
```python
demo = msprime.Demography()
demo.add_population(name="A", initial_size=20000.0, growth_rate=0.0005)
msprime.sim_ancestry(
    samples=10,
    demography=demo,
    sequence_length=100_000,
    recombination_rate=1e-8,
    ploidy=1,
    record_full_arg=True,
    random_seed=seed + 1,
)
```

Growth rate `α=0.0005` is small enough that Ne doesn't blow up over the
typical coal window (~4N=40000 gens → Ne_anc ≈ 10000·exp(−20) ≈ 2×10⁻⁵,
which is fine: the coalescent finishes long before the growth rate
becomes degenerate). It's large enough to produce a strongly skewed SFS
(excess of singletons) that the bin-wise check will catch if the
time-varying rate path is broken.

### N6 — three-pop with split (`ej` + migration interaction)

Three extant populations (sample sizes 4+3+3), symmetric migration
`M=5e-5` between all pairs, populations 1 and 2 merge backward into pop 0
at time `T_split=3000`. After the merge (backward) only pop 0 exists.

**msinv:**
```python
demo = Demography(
    pop_sizes=[10000.0, 10000.0, 10000.0],
    migration_matrix=[
        [0.0,  5e-5, 5e-5],
        [5e-5, 0.0,  5e-5],
        [5e-5, 5e-5, 0.0 ],
    ],
)
demo.add_population_split(time=3000.0, derived=[1, 2], ancestral=0)
HullSimulator(
    sample_config={(None, 0): 4, (None, 1): 3, (None, 2): 3},
    demography=demo,
    sequence_length=100_000.0,
    recombination_rate=1e-8,
    inversions=[],
    seed=seed,
).simulate()
```

**msprime:**
```python
demo = msprime.Demography()
demo.add_population(name="A", initial_size=20000.0)
demo.add_population(name="B", initial_size=20000.0)
demo.add_population(name="C", initial_size=20000.0)
for src, dst in [("A","B"),("B","A"),("A","C"),("C","A"),("B","C"),("C","B")]:
    demo.set_migration_rate(source=src, dest=dst, rate=5e-5)
demo.add_mass_migration(time=3000.0, source="B", dest="A", proportion=1.0)
demo.add_mass_migration(time=3000.0, source="C", dest="A", proportion=1.0)
msprime.sim_ancestry(
    samples={"A": 4, "B": 3, "C": 3},
    demography=demo,
    sequence_length=100_000,
    recombination_rate=1e-8,
    ploidy=1,
    record_full_arg=True,
    random_seed=seed + 1,
)
```

The mid-strength `M=5e-5` (vs N2's `1e-4`) keeps wall-clock manageable on
three pops and still leaves an `Fst`-style signal in the per-pop SFS for
the SFS check to bite on.

## Statistics

Per scenario, two stat families are computed.

### Family A — moments (existing)

Same set as N1/N2:

| Stat | tskit call | Single- vs multi-pop |
|---|---|---|
| `pi_branch` | `ts.diversity(mode="branch")` | both |
| `n_trees` | `ts.num_trees` | both |
| `mean_tmrca` | weighted mean of `tree.time(tree.root)` | both |
| `dxy_branch` | `ts.divergence(sample_sets=…, mode="branch")` | multi-pop only (N3, N6) |

For N6 (three pops), `dxy_branch` is computed **as a single number**:
mean of the three pairwise divergences `(A,B), (A,C), (B,C)`, to keep
the per-stat pass/fail message readable. (Pairwise breakdown can be
added later if a single-mean failure needs triage.)

**Pass criterion (per stat):** `|mean_msinv − mean_msprime| ≤ 3·sqrt(SE_msinv² + SE_msprime²)` (the existing N1/N2 bound).

### Family B — branch-mode AFS (new)

Stat: `ts.allele_frequency_spectrum(mode="branch", polarised=True)`. The
output array has length `n_set + 1` where `n_set` is the size of the
sample set. Bins `[0]` (no derived) and `[n_set]` (all-derived /
above-root) are zero by construction in branch mode and are excluded.
The informative bin count is `K_set = n_set - 1`.

| Scenario | sample set(s) | AFS calls | bins per call | total bins K |
|---|---|---|---|---|
| N4 (single-pop, n=10) | all 10 | 1 | 9 | 9 |
| N5 (single-pop, n=10) | all 10 | 1 | 9 | 9 |
| N3 (5+5)  | per-pop | 2 | 4 each | 8 |
| N6 (4+3+3) | per-pop | 3 | 3, 2, 2 | 7 |

For multi-pop scenarios (N3, N6) the AFS is computed **marginally per
population** with `ts.allele_frequency_spectrum(sample_sets=[pop_i],
polarised=True, mode="branch")`. Joint multi-pop SFS is out of scope
(the bin count blows up and the per-cell SE degrades below detection).

**Pass criterion (per bin):** Bonferroni-corrected two-sided z-bound.
For a scenario with `K` informative AFS bins, per-bin α = 0.003 / K
(family-wise α=0.003 to match the moments family). Per-bin bound is
`z * sqrt(SE_msinv² + SE_msprime²)` with `z = qnorm(1 − α/2)`.

| K | per-bin two-sided α | z | bound |
|---|---|---|---|
| 7 (N6) | 4.3×10⁻⁴ | 3.52 | `3.52·SE` |
| 8 (N3) | 3.8×10⁻⁴ | 3.55 | `3.55·SE` |
| 9 (N4, N5) | 3.3×10⁻⁴ | 3.59 | `3.59·SE` |

The harness computes z from K via `scipy.stats.norm.ppf(1 - α/2)` (or
the equivalent `statistics.NormalDist().inv_cdf` from the stdlib if
scipy is to be avoided) rather than hard-coding values. Single failing
bin = scenario fails; the failure message names the bin plus its
`(msinv, msprime, delta, bound)` so the SFS shape direction is visible
(e.g. "afs_p0_bin_1 high → too many singletons in pop 0").

### Why this combination

- **Moments alone** are shape-blind. A growth-rate sign flip can leave
  `pi_branch` unchanged while distorting the SFS (the singleton bin
  scales with `α/N`, not just `α·E[T_total]`). The user explicitly
  picked the SFS option after seeing this trade-off.
- **AFS alone** would lose the rate/scale sensitivity — `pi_branch` and
  `n_trees` directly probe the coal+recomb hot path with much sharper
  per-rep variance than any single AFS bin. Keeping moments preserves
  that diagnostic.
- **Bin-wise Bonferroni vs chi-square aggregate.** Branch-mode AFS bins
  are correlated (tree branches constrain each other), so a naive
  chi-square aggregate has the wrong null. Bin-wise Bonferroni is
  conservative but readable: a failure message points at a specific
  bin (e.g. "bin 1 high → too many singletons → growth rate too high").

## File layout

Two test files:

- `tests/hull/test_validation_msprime.py` — existing N1/N2 plus four
  new test functions (N3–N6). Each test function specifies the
  scenario inputs and calls `_run_validation`, which now spawns the
  per-engine child subprocesses, collects per-rep stats, applies the
  pass criteria, and prints the benchmark block.
- `tests/hull/_msprime_bench_runner.py` — child-process entry point.
  Accepts `--scenario {n1,n2,n3,n4,n5,n6} --engine {msinv,msprime}
  --n-reps N --seed-base K`, runs the rep batch, writes a JSON
  document to stdout (`{"per_stat_values": {...}, "per_rep_seconds":
  [...]}`), exits. Importing this file does NOT execute any sim;
  it's a pytest-collectable but pytest-skipped module (sentinel
  `pytestmark = pytest.mark.skip("child runner — invoked via subprocess")`).

Four new test functions in `test_validation_msprime.py`:

```python
def test_msprime_validation_n3_two_pop_split(): ...
def test_msprime_validation_n4_bottleneck(): ...
def test_msprime_validation_n5_exponential_growth(): ...
def test_msprime_validation_n6_three_pop_with_split(): ...
```

Helper changes:

- `_stats_from_ts(ts, sample_sets=None)` → also returns AFS bins under
  keys `afs_bin_1 … afs_bin_K` (single-pop) or `afs_p{p}_bin_{k}`
  (multi-pop); zero-edge bins (0 and n) are skipped.
- `_run_validation(scenario_name, n_reps=200)` → looks up the
  scenario by name, spawns two children (msinv then msprime), reads
  per-rep stats from each child's stdout, applies the `3.0·SE` bound
  to `_branch`/`_trees`/`_tmrca`/`dxy` stats and the Bonferroni
  bound (`z = qnorm(1 − α / K_afs / 2)`, α=0.003) to all `afs_…`
  stats. The split is purely on the stat-key prefix. Then prints
  the benchmark block from per-child timing + `os.wait4` rusage.

The N1/N2 tests are migrated to the same dispatch pattern (their
factories move into the runner module so the runner can build them
by scenario name). N1/N2 then automatically benefit from the
benchmark layer too — no behavior change otherwise. AFS-bin checks
are NOT added retroactively to N1/N2; the bound is the existing 3·SE
on the four moments.

No changes to `msinv/`.

## Test budget

Budget ceiling: **180 s** (3 min) for the full extended harness
(`tests/hull/test_validation_msprime.py`, all six scenarios).
This is a deliberate change from the predecessor spec's tight ~5 s
budget — short fast sims like these finish in seconds each, and the
real value of the harness is sharp drift detection. The budget gives
headroom to scale `N_REPS` upward (per-bin SE shrinks as `1/√N`) when
small-magnitude drift becomes the regression class to catch.

Estimated wall-clock at the current `N_REPS=200` (extrapolating from
N1/N2's measured ~5 ms/rep; multi-pop and split scenarios run ~2×
slower because of event handling):

| Scenario | per-rep avg | 200 reps × 2 engines | + subprocess overhead |
|---|---|---|---|
| N1 (panmictic) | 5 ms | 2.0 s | +0.7 s |
| N2 (two-pop migration) | 11 ms | 4.4 s | +0.7 s |
| N3 (two-pop split) | 12 ms | 4.8 s | +0.7 s |
| N4 (bottleneck) | 8 ms | 3.2 s | +0.7 s |
| N5 (growth) | 8 ms | 3.2 s | +0.7 s |
| N6 (three-pop split) | 18 ms | 7.2 s | +0.7 s |

Subprocess overhead is two `python -m` boots per scenario (one per
engine), each ≈ 300–400 ms cold-start (msprime + numpy + tskit + msinv
imports dominate). Across 6 scenarios that's ~30 s of
compute + ~5 s of overhead → harness total ~35 s at `N_REPS=200`.
Comfortably inside the 180 s ceiling, with ~5× headroom.

`N_REPS` is exposed as a top-of-file constant (`N_REPS = 200`) in
`tests/hull/test_validation_msprime.py`. To sharpen drift detection
without changing scenarios, raise it (e.g. `N_REPS=1000` ≈ 175 s,
SE shrinks 2.24×). Larger sequence lengths or larger sample sizes —
intended for finding rare-corner regressions — get their own future
spec rather than overflowing this one's budget.

## Benchmarks

Each scenario also reports wall-clock and peak RAM for both engines.
This is **report-only** — no benchmark assertions; absolute numbers
drift with hardware and the harness should not be the regression
gate for performance. The point is to surface relative cost (msinv
vs msprime per rep) and absolute footprint (peak RSS) when the
test runs, so cost-of-feature trade-offs across the three planned
validation tracks (msprime → discoal → SLiM) are visible at a
glance instead of buried in `bench_rho` runs.

### What we measure

Per scenario, per engine:

| Metric | How |
|---|---|
| wall-clock per rep (mean, SE) | `time.perf_counter()` around the engine call, inside the rep loop in the child process |
| peak RSS for the rep batch | child-process `ru_maxrss` from `os.wait4` after the subprocess exits — clean per-engine attribution |
| total wall-clock for the rep batch | sum of per-rep wall-clocks |

Subprocess-per-engine, not per-rep:

- `resource.getrusage(RUSAGE_SELF).ru_maxrss` is monotone within a
  process, so running both engines in the same Python process makes
  per-engine peak attribution impossible. We solve this by spawning
  **one subprocess per engine per scenario** (8 subprocesses total).
  Each subprocess does the full 200-rep batch for one engine, prints
  the per-stat values to stdout as JSON, and exits. The parent
  collects values, calls `os.wait4(pid, 0)`, and reads `rusage.ru_maxrss`
  for that specific child. That number is the engine's actual peak,
  with no cross-contamination.
- 200 reps of msinv runs in ~1–2 seconds in-process; subprocess fork
  overhead (≈30 ms once) is amortized.
- First-rep cost (engine import, `.so` warm-up) is excluded by
  running one untimed warm-up rep inside the child before the timed
  batch starts.
- Linux `ru_maxrss` is in **kilobytes**; the report converts to MB
  with one decimal.

Subprocessing the rep batch decouples the benchmark layer from the
distributional comparison — the latter still reads the JSON values
from stdout and applies the same 3·SE / Bonferroni bounds described
in the Statistics section. Test pass/fail does not depend on
benchmark numbers.

### Reported output

At the end of `_run_validation`, after the per-stat OK/FAIL lines, the
harness prints a per-scenario benchmark block (visible only with `-s`,
matching the existing per-stat summary):

```
[N3 two-pop split] benchmarks
  msinv:   per-rep 11.8 ms ± 0.3, total  2.4 s, peak RSS  47.2 MB
  msprime: per-rep  5.9 ms ± 0.2, total  1.2 s, peak RSS 142.7 MB
  ratio:   per-rep msinv/msprime = 2.00x;  RAM msinv/msprime = 0.33x
```

The ratio line is the headline: a single number that says "msinv is
2× slower per rep but uses 1/3 the RAM" without scrolling. When
discoal and SLiM tracks land, they'll print the same block, so all
three sims are directly comparable on the same scenarios.

### Persistent record

After the test completes (pass or fail), the per-scenario benchmark
dict is appended to `.tmp/msprime_validation_bench.jsonl` (one JSON
line per scenario per run, with timestamp + git HEAD short SHA + the
six numbers above). This gives a low-friction record of runtime drift
across commits without setting up a real benchmark CI. The file lives
in `.tmp/` (per `feedback_local_tmp.md`); not committed, not asserted
on.

## Triage table (failure mode → suspect path)

| Failing stat pattern | Most likely subsystem |
|---|---|
| N3 `dxy_branch` only | mass-migration / `ej` event handler |
| N3 `dxy_branch` + AFS bin 1 high | merge time T or migration rate path |
| N4 AFS only (bin 1 high, bin 9 low) | `en` rate-update at event boundary |
| N4 `pi_branch` only | `en` semantics (Ne value applied to wrong window) |
| N5 AFS skewed (singletons high) | `eg` time-varying integration (under-integrating window) |
| N5 `pi_branch` low + AFS singletons high | `eg` sign error (growth applied as decline) |
| N6 `pi_branch`/`mean_tmrca` only | three-pop coal-rate aggregation |
| N6 AFS marginal-per-pop bin 1 high | migration / split interaction; check `em`+`ej` joint |

## Out of scope / explicitly deferred

- **Class migration** (`cmig`) — Tier 3 cmig anchors already cover this
  in `test_phase4b_class_migration.py`.
- **Inversions** — analytical anchors (Andolfatto cross-class T_MRCA,
  Tier 3 cmig + Tier 3-cheap Q) cover the no-msprime path; SLiM
  validation track will cover the with-msprime path.
- **Sweeps** — Kim–Stephan anchors cover this; discoal validation track
  will extend cross-simulator coverage.
- **Asymmetric migration** beyond the symmetric N6 case — would
  need its own scenario; small extension if user wants it later.
- **Joint multi-pop AFS** — bin count blows up; per-pop marginals
  already catch the regression class of interest.
- **Per-rep paired comparison** (same seed → same output) — engines use
  different RNGs, this was the right choice for N1/N2 and stays.
- **Performance assertions.** Benchmarks are report-only. We don't
  fail the test on a wall-clock or RAM regression — drift on a shared
  device is too noisy, and `bench_rho` is the right vehicle if/when
  perf gating becomes worthwhile. The `.tmp/` JSONL log is for ad-hoc
  inspection.

## References

- `docs/superpowers/specs/2026-04-29-msprime-validation-design.md` —
  predecessor; population_size doubling and `record_full_arg`
  conventions are inherited verbatim.
- `tests/hull/test_validation_msprime.py` — N1/N2 implementation; the
  extension appends to this file.
- `feedback_msprime_api.md` — `mass_migration` not `population_split`
  on the msprime side; `ploidy=1` for stats.
- `msinv/hull/demography.py` — `add_population_split`,
  `add_population_size_change`, `add_growth_rate_change`,
  `add_mass_migration`, `add_migration_rate_change`.
