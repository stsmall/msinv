# msinv / CLAUDE.md

## Build + install Rust extension
cd rust && cargo build --release -p msinv-py
/bin/cp -f target/release/lib_msinv_core.so ../msinv/_msinv_core.cpython-312-x86_64-linux-gnu.so
- `/bin/cp` explicit: shell alias adds `-i` and prompts.

## Tests
- Rust: `cd rust && cargo test --release` (132 lib + 17 integration + 4 sweep-anchor + 2 sweep-trajectory as of 2026-04-29).
  `--lib` skips `tests/` and `examples/`; use plain `cargo test --release` to catch missed struct-field updates in those.
- Python: `.venv/bin/python -m pytest tests/hull/ --ignore=tests/hull/test_stress_corners.py`
  (183 passed, 3 skipped as of 2026-04-29; the 12 sweep-rewrite follow-up skips are now active
  after Phases A-D of `docs/superpowers/plans/2026-04-29-sweep-followups.md`).
- msprime validation: `tests/hull/test_validation_msprime.py` (N1 panmictic + N2 two-pop migration
  vs `msprime.sim_ancestry`; ~7 s; spec `docs/superpowers/specs/2026-04-29-msprime-validation-design.md`).
  Use `-s` to surface the per-stat OK/FAIL summary on failure.
- ALWAYS `.venv/bin/python`, not system.
- Targeted Rust subset: `cargo test --release --lib <substring>` (e.g. `class_mig`, `trajectory`).
- Single Python file: `.venv/bin/python -m pytest tests/hull/test_phase8_trajectory_selection.py -v`.
- New test files follow `test_phase{N}_*`: 1=panmictic, 2=class barrier, 3=gene flux,
  4=demography, 4b=class migration, 5=per-segment/multi-inv, 6=sweep, 8=trajectory selection.
- Sweep test files: `tests/hull/test_phase6_sweep.py` (T1-T5 against joint-WF Sweep API),
  `tests/hull/test_phase6b_sweep_joint.py` (J1-J9 trajectory integration),
  `rust/msinv-core/tests/sweep_kim_stephan_anchors.rs` (Tier-1 closed-form anchors),
  `rust/msinv-core/tests/sweep_trajectory_built_from_demography.rs` (live demography wiring).

## Pytest progress visibility
- `pytest ... 2>&1 | tail -N` BUFFERS — output appears only at exit.  For long runs:
  `pytest -v --tb=no > .tmp/pytest.log 2>&1 &` then `tail -f .tmp/pytest.log`.
- `-x` to fail-fast, `--timeout=30` per-test cap (pytest-timeout plugin).

## Pre-existing test failures (NOT regressions)
- `test_stress_corners.py::test_flux_in_nested_inv_only_flips_one_inv_class` hangs (>15 min,
  ~35 GB RAM) at the remnant-ratchet path. `--ignore=tests/hull/test_stress_corners.py` for full-suite runs.

(Sweep tests: as of 2026-04-29 all 12 sweep tests (T1-T5 + J1-J9) are active. The
sweep rewrite (spec `docs/superpowers/specs/2026-04-28-sweep-rewrite-design.md`)
plus the follow-ups (spec `docs/superpowers/plans/2026-04-29-sweep-followups.md`)
shipped joint-WF trajectory + simulator-side `apply_sweep` + `apply_sweep_finalize`
+ trajectory-driven `ne_cell` inside the sweep window. The 17 prior panmictic
`target_class='P'` failures are gone.)

Confirm pre-existing via `git stash`+rerun before chasing.

## Perf + debugging
- Bench: `cd rust && cargo run --release --example bench_rho -- <rho> <reps> [n_pops]`
- Event-loop trace: `MSINV_TRACE=1 <python cmd>` (requires instrumentation commit).
- Flamegraph: `debug = 1` already set in `rust/Cargo.toml` release profile.
- Timebox long sims with OS-level `timeout <s>`; Python `signal.alarm` does NOT interrupt a PyO3 call holding the GIL.

## Layout
- Rust core: `rust/msinv-core/src/` (simulator.rs, rate_index.rs, events.rs, lineage.rs).
- PyO3 bridge: `rust/msinv-py/src/lib.rs`.
- Python reference (legacy): `msinv/hull/simulator.py` — canonical semantics for Rust to mirror.
- Python wrapper / tskit conversion: `msinv/hull/_rust_bridge.py`.

## Event log hook (opt-in)
Off by default; enable to capture cmig + flux events for analysis.
```python
sim = HullSimulator(..., record_events=True)
sim.simulate()
sim.event_log  # list[dict] with "kind"="cmig"|"flux", or None when off
```
Helpers: `msinv.hull._event_log` (`filter_cmig`, `filter_flux`,
`tract_lengths`, `survival_curve`, `coverage_count`).
Off-path is zero-overhead; production sims should leave the flag off.

## Persistent context (not in repo)
Project memory: `/home/ssmall/.claude/projects/-home-ssmall-inversion-sims-files/memory/`
Read `MEMORY.md` there first — index of what's known about the code + biology.

## Known scale limits
- Realistic anopheles Ne_anc ≥ 1e6 + old inversions (≥100k gen) hits the remnant-ratchet:
  partial-coal on class-mismatched pairs grows active-n unboundedly.
  Minutes + multi-GB RAM per rep. Blocked on Path 2 rewrite.
  Tracked in `project_panic_kirfol_en.md`, `project_feature_branches_roadmap.md`.
- Path 1 (analytic-middle, feature branch) rejected: slower at realistic scale + rate-bug.

## Shared device — resource cap
- Parallel msinv runs: ≤100 worker procs, ≤400 GB RAM total.  Profile per-worker RSS
  before scaling: a 0.5 Mb / 100-rep Kir/Fol worker uses ~5 GB RSS → 50 workers ≈ 250 GB.
- Run scenarios serially when chained, not all at once.

## External tools
- pg_gpu at `~/programs/pg_gpu` — NOT in project venv. GPU summary stats via CuPy.
- msprime in venv, used for recapitation (`sim_ancestry(initial_state=ts)`).

## Conventions
- New features → feature branch. Do not break `main` event loop.
- Bug fixes → regression test in cargo before commit.
- Rust ↔ Python divergence is a common bug vector — diff both when in doubt.
- Adding a field to a public Rust struct: audit `Self { ... }` literals in `src/`, `tests/`, `examples/`, `benches/` and the PyO3 bridge — `cargo build -p <crate>` catches `src/` only.
- `Karyotype` enum (`rust/msinv-core/src/class_tag.rs`) has variants `S` and `I` only.
  No `Colinear`/`Inverted`/`Pan` — panmictic is a `BranchClass` state, not a `Karyotype`.
- Rust RNG: this crate is on rand 0.9 — use `rng.random::<f64>()`, NOT `rng.gen::<f64>()`.
  See existing usage in `rust/msinv-core/src/trajectory.rs` (`sample_binomial_2n`).
- Migration matrix convention: `m_ij` = "fraction of pop i ABSORBING FROM pop j",
  matching `Demography::migration_matrix[dst][src]`. Forward-flow A→B needs
  `m(B, A) > 0`, NOT `m(A, B) > 0`.
- `population_size` convention: msinv treats `population_size=N` as **diploid Ne**
  (per-pair coal rate = `1/(2N)`; verified at `rust/msinv-core/src/simulator.rs:1338`
  and `msinv/hull/simulator.py:635`). To compare against msprime, double on the
  msprime side: `msprime.sim_ancestry(population_size=2*N, ploidy=1, record_full_arg=True)`.
  `ploidy=1` reads N as haploid; `record_full_arg=True` is needed because msinv
  records all recombs as tree boundaries while msprime default simplifies
  non-ancestral ones. Migration rate is per-lineage-per-gen on both sides — do NOT
  rescale. Spec: `docs/superpowers/specs/2026-04-29-msprime-validation-design.md`.
- Sweep + inversion trajectories use the **discrete-time** WF logistic update
  `p_{t+1} = p_t·(1+s)/(1+s·p_t)` with closed form `f0·(1+s)^t / (1 - f0 + f0·(1+s)^t)`.
  Don't write tests against the continuous `exp(s·t)` form — they diverge at O(s²·t).
- A PostToolUse hook runs `cargo check` after each Rust Edit/Write — every individual
  edit must leave the workspace compiling. For API-rewrite tasks, bridge through a
  transitional struct that carries both old and new fields, migrate callers, then
  strip the compat shim. Don't try a single big-bang Edit.
