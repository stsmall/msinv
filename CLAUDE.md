# msinv / CLAUDE.md

## Build + install Rust extension
cd rust && cargo build --release -p msinv-py
/bin/cp -f target/release/lib_msinv_core.so ../msinv/_msinv_core.cpython-312-x86_64-linux-gnu.so
- `/bin/cp` explicit: shell alias adds `-i` and prompts.

## Tests
- Rust: `cd rust && cargo test --release` (96 lib + 23 parity as of 2026-04-27)
- Python: `.venv/bin/python -m pytest tests/hull/` — expect 17 pre-existing sweep failures (see below)
- ALWAYS `.venv/bin/python`, not system.
- Targeted Rust subset: `cargo test --release --lib <substring>` (e.g. `class_mig`, `trajectory`).
- Single Python file: `.venv/bin/python -m pytest tests/hull/test_phase8_trajectory_selection.py -v`.
- New test files follow `test_phase{N}_*`: 1=panmictic, 2=class barrier, 3=gene flux,
  4=demography, 4b=class migration, 5=per-segment/multi-inv, 6=sweep, 8=trajectory selection.

## Pytest progress visibility
- `pytest ... 2>&1 | tail -N` BUFFERS — output appears only at exit.  For long runs:
  `pytest -v --tb=no > .tmp/pytest.log 2>&1 &` then `tail -f .tmp/pytest.log`.
- `-x` to fail-fast, `--timeout=30` per-test cap (pytest-timeout plugin).

## Pre-existing test failures (NOT regressions)
17 sweep tests use `target_class='P'` which `msinv/hull/_rust_bridge.py:89` rejects with
`ValueError: target_class='P' (panmictic-only sweep) is not supported by the Rust backend`:
`test_phase6_sweep.py::{test_sweep_forces_coalescence_at_x_sel,test_sweep_does_not_affect_distant_positions,test_sweep_hitchhiking_produces_valid_ts_at_moderate_rho,test_sweep_window_mode_no_disjoint_corruption[*],test_soft_sweep_diversity_signature,test_two_simultaneous_window_sweeps[*-True],test_two_simultaneous_hitchhiking_sweeps[*-True]}`.
Confirm pre-existing via `git stash`+rerun before chasing.  Deselect with
`--deselect tests/hull/test_phase6_sweep.py::<name>`.

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
