# msprime validation harness — design

**Date:** 2026-04-29
**Status:** spec, awaiting user approval
**Predecessor / supersedes:** the "cross-engine bit-equivalence parity harness"
deferred in `2026-04-28-event-hook-t3-cmig-tier3q-design.md` and
`2026-04-28-tier3-full-andolfatto-design.md`. That framing is dropped: the
Python fallback (`HullSimulator(...).simulate(use_rust=False)`) is ~2 weeks
behind the Rust core (no sweeps, no joint trajectory, no A-tag inheritance,
last meaningful update Apr 17–22), so it is not a reliable oracle. The Rust
core is the engine; this harness validates the Rust core against an
independent oracle (msprime).

## Goal

Catch silent regressions in the Rust core's neutral hot path
(`coal + recomb + class-aware migration`) by comparing distributions of
branch-length statistics against `msprime.sim_ancestry` on equivalent
demographic models.

Inversion-specific and sweep semantics are **out of scope** — those have
analytical anchors already (Wakeley `E[T_total]`, Andolfatto cross-class
`T_MRCA`, Kim–Stephan `T_MRCA` reduction, `sweep_kim_stephan_anchors.rs`,
Tier 3 cmig + Tier 3-cheap Q).

## Scope

Two scenarios, both neutral, both within msprime's native capability:

| Scenario | Purpose | Parameters |
|---|---|---|
| **N1 — single-pop panmictic** | base coal + recomb hot path | `n=10`, `Ne=10000`, `L=100_000`, `r=1e-8` (ρ=40) |
| **N2 — two-pop symmetric migration** | demography + migration paths | sample_config `{(None,0):5, (None,1):5}`, `pop_sizes=[10000, 10000]`, symmetric `M=1e-4`, `L=100_000`, `r=1e-8`, no demographic events |

No inversions, no sweeps, no demographic events beyond the constant-size
two-population island model.

## Comparison

For each scenario, draw N=200 independent reps on each engine. Rep `i` uses
`seed=i` on both engines (so the same rep across reruns is stable; per-rep
RNG streams are independent across `i`).

For each rep, compute three branch-length statistics from the resulting
`tskit.TreeSequence` (no mutation placement needed):

| Stat | tskit call | What it tests |
|---|---|---|
| `pi_branch` | `ts.diversity(mode="branch")` | mean pairwise coal time × 2 — direct probe of the coal rate |
| `n_trees` | `ts.num_trees` | recomb breakpoint count; `E ≈ ρ · H_{n-1}` |
| `mean_tmrca` | average of `tree.tmrca(*samples)` weighted by `tree.span`, over all trees in the TS | tree-level coal time, sensitive to overlap-vs-k-based coal rates (the past gotcha) |

For N2 multi-pop, `pi_branch` uses default sample set (all samples); a
between-pop variant `dxy_branch = ts.divergence(sample_sets=[pop0_ids, pop1_ids], mode="branch")`
is added as a fourth stat for that scenario only.

### Test criterion

For each stat, compare engine means with the bound:

```
|mean_msinv − mean_msprime| ≤ 3 · sqrt(SE_msinv² + SE_msprime²)
```

where `SE_engine = stdev_engine / sqrt(N)`. The 3·SE bound corresponds to a
two-sided Welch-t at α ≈ 0.003, which catches drift bigger than ~3% of the
mean (well below any regression class we'd care about) while keeping false
positives below ~1 per 100 CI runs across all stats.

The pytest assertion uses a single-line message of the form
`"pi_branch: msinv=12345.6 ± 67.8, msprime=12300.0 ± 65.0, |Δ|=45.6, 3·SE=283.4 → OK"`
(or `"... → FAIL"`), so a failure log shows direction, magnitude, and the
budget without re-running.

## msprime equivalence

### N1 — panmictic

```python
msprime.sim_ancestry(
    samples=10,
    population_size=10000.0,
    sequence_length=100_000,
    recombination_rate=1e-8,
    ploidy=1,
    random_seed=seed + 1,  # msprime requires seed >= 1
)
```

Equivalent to the msinv call:

```python
HullSimulator(
    samples=10,
    population_size=10000.0,
    sequence_length=100_000.0,
    recombination_rate=1e-8,
    inversions=[],
    seed=seed,
).simulate()
```

### N2 — two-pop migration

msprime side:

```python
demo = msprime.Demography()
demo.add_population(name="A", initial_size=10000.0)
demo.add_population(name="B", initial_size=10000.0)
demo.set_migration_rate(source="A", dest="B", rate=1e-4)
demo.set_migration_rate(source="B", dest="A", rate=1e-4)
msprime.sim_ancestry(
    samples={"A": 5, "B": 5},
    demography=demo,
    sequence_length=100_000,
    recombination_rate=1e-8,
    ploidy=1,
    random_seed=seed + 1,
)
```

msinv side:

```python
demo = Demography(
    pop_sizes=[10000.0, 10000.0],
    migration_matrix=[[0.0, 1e-4], [1e-4, 0.0]],
)
HullSimulator(
    sample_config={(None, 0): 5, (None, 1): 5},
    demography=demo,
    sequence_length=100_000.0,
    recombination_rate=1e-8,
    inversions=[],
    seed=seed,
).simulate()
```

### Migration convention check

msprime: `migration_matrix[i][j]` is the per-generation rate at which a
lineage in pop *i* (backward in time) moves to pop *j*. Forward-flow is
*j* → *i*.

msinv: `migration_matrix[dst][src]` is the fraction of pop *dst* absorbing
from *src* (per CLAUDE.md and `feedback_msprime_api.md`). Forward-flow is
*src* → *dst*.

These are the same convention: `msprime[i][j] ≡ msinv[i][j]` with both
indices interpreted as `[backward-current-pop][backward-destination]`. For
the symmetric `M=1e-4` test case the matrices are identical
(`[[0, 1e-4], [1e-4, 0]]`), so the convention question doesn't bite this
spec, but the equivalence is asserted here for future asymmetric extensions.

### Ploidy

`ploidy=1` on the msprime side because msinv samples are haploid lineages
(no diploid pairing). `ts.diversity(mode="branch")` on a haploid tskit TS
gives `2 · E[T_2]` directly. (This matches the `feedback_msprime_api.md`
guidance: `ploidy=1` for stats, `ploidy=2` only for tree-count comparisons,
which we are not making here.)

## File layout

Single new file: `tests/hull/test_validation_msprime.py`.

Two test functions:

```python
def test_msprime_validation_n1_panmictic():
    """Rust msinv vs msprime — single-pop panmictic, n=10, ρ=40."""

def test_msprime_validation_n2_two_pop_migration():
    """Rust msinv vs msprime — two-pop symmetric migration."""
```

Each calls a shared helper `_run_validation(scenario_name, msinv_factory, msprime_factory, n_reps=200)` that returns the per-stat (mean, SE) for both engines and asserts the 3·SE bound for each stat.

The helper is local to the test file. No new modules under `msinv/`.

## Test budget and CI integration

Wall-clock measured on a 5-rep smoke run, extrapolated to N=200:

| Scenario | per-rep msinv | per-rep msprime | total (200 reps × 2 engines) |
|---|---|---|---|
| N1 | 3.3 ms | 6.0 ms | 1.9 s |
| N2 | 11.5 ms | 5.7 ms | 3.4 s |

Total ≈ 5.3 s wall-clock. Comfortably inside the default
`pytest tests/hull/` run; no `slow` mark needed.

The test file is included in the default pytest target. No pytest plugin
or fixture is added.

## RNG independence

Each rep is an independent run with a distinct seed on each engine.
The 200-rep batch is *not* a paired comparison — we compare distributions,
not per-rep outputs. So the fact that msinv (Xoshiro256++) and msprime
(its internal MT-based RNG) draw different sequences for the same `seed=i`
is irrelevant: the only thing that has to match is the population
distribution of the three stats.

## Failure mode and triage

If the 3·SE bound fails for a stat, the message is interpretable enough
to start triage immediately:

- `pi_branch` mismatch → coal rate or pair-count formula
- `n_trees` mismatch → recombination-rate code or rho-clamp
- `mean_tmrca` mismatch (with `pi_branch` OK) → distribution shape change,
  likely a coal-pair selection bug (overlap-based vs k-based)
- `dxy_branch` mismatch (N2 only, with single-pop stats OK) → migration
  rate or migration-matrix convention bug

The harness does not attempt automatic root-cause identification — it
flags the regression and points the developer at the relevant subsystem.

## Out of scope / explicitly deferred

- **Inversion validation against msprime.** msprime cannot natively
  simulate inversion-aware sampling; existing analytical anchors
  (Andolfatto cross-class `T_MRCA`, Tier 3 cmig + Tier 3-cheap Q) cover
  this. Not added here.
- **Sweep validation.** Kim–Stephan anchors already cover this in
  `sweep_kim_stephan_anchors.rs`; msprime cannot natively simulate the
  joint-WF sweep model.
- **Demographic-event paths** (`split`, `en`, `ej`, growth). The Phase-4b
  Tier 3-cheap Q tests cover the migration / class-mig path; broader
  demographic-event validation is a separate harness item once the
  current resource-cap roadmap (`project_panic_kirfol_en.md`) settles.
- **Golden-snapshot regression layer** (option B in brainstorming).
  Add later if drift in practice motivates it.
- **N-convergence to analytical limits** (option C). Useful but
  separate; existing Tier 3 anchors already do this for `E[T_total]`.
- **CI flake budget** beyond the 3·SE bound. If real-world flake rate
  exceeds 1/100 we revisit the bound; not pre-engineered.

## References

- Existing anchor tests:
  - `rust/msinv-core/tests/python_parity.rs` (25 tests; misnamed — Rust-side regressions, not parity)
  - `rust/msinv-core/tests/sweep_kim_stephan_anchors.rs`
  - `rust/msinv-core/tests/sweep_trajectory_built_from_demography.rs`
  - `tests/hull/test_phase4b_class_migration.py` (Tier 3 cmig + Tier 3-cheap Q)
- Memory pointers:
  - `feedback_parity_misnomer.md` — `python_parity.rs` is not cross-engine
  - `feedback_msprime_api.md` — `ploidy=1` for stats, `ploidy=2` for tree counts; `mass_migration` not `population_split`
  - `feedback_coal_rates.md` — overlap-based coal rates, not k-based
- CLAUDE.md — migration-matrix convention, Rust build instructions
