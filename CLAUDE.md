# msinv / CLAUDE.md

## Build + install Rust extension
cd rust && cargo build --release -p msinv-py
/bin/cp -f target/release/lib_msinv_core.so ../msinv/_msinv_core.cpython-312-x86_64-linux-gnu.so
- `/bin/cp` explicit: shell alias adds `-i` and prompts.

## Tests
- Rust: `cd rust && cargo test --release` (63 lib + 23 parity)
- Python: `.venv/bin/python -m pytest tests/hull/` (154 + 3 skipped, ~4 min)
- ALWAYS `.venv/bin/python`, not system.

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

## External tools
- pg_gpu at `~/programs/pg_gpu` — NOT in project venv. GPU summary stats via CuPy.
- msprime in venv, used for recapitation (`sim_ancestry(initial_state=ts)`).

## Conventions
- New features → feature branch. Do not break `main` event loop.
- Bug fixes → regression test in cargo before commit.
- Rust ↔ Python divergence is a common bug vector — diff both when in doubt.
