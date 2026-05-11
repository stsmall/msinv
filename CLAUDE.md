# CLAUDE.md — msinv

Operational guidance for Claude sessions in this repo. Full project
documentation, theory notes, specs, and plans live in
`/home/ssmall/inversion_sims/msinv_paper/` (sibling directory, not
under git).

## Build + install Rust extension
```
cd rust && cargo build --release -p msinv-py
/bin/cp -f target/release/lib_msinv_core.so ../msinv/_msinv_core.cpython-312-x86_64-linux-gnu.so
```
- `/bin/cp` explicit: shell alias adds `-i` and prompts.
- The PreToolUse hook blocks the cp if any `python.*(pytest|msinv)` process is running — wait or kill deliberately.

## Tests
- Rust: `cd rust && cargo test --release`
- Python: `.venv/bin/python -m pytest tests/hull/ --ignore=tests/hull/test_stress_corners.py`
- Validation: `.venv/bin/python -m pytest tests/validation/ -v`
- Always `.venv/bin/python`, not system.

## Pre-commit hooks (in `.claude/hooks/`)
- `gitleaks-pre-commit.sh` — blocks commits with secrets
- `ruff-pre-commit.sh` — runs `ruff check` on staged `.py` files
- `cargo-check-on-rs-edit.sh` — PostToolUse cargo check after Rust edits
- Bypass with `--no-verify` only when justified.

## Ruff
- Config in `pyproject.toml [tool.ruff]`. Line length 88, py39 target, E/F/W/B/UP rules.
- Auto-fix: `.venv/bin/python -m ruff check ... --fix` (add `--unsafe-fixes` for aggressive transforms).
- Formatter: `.venv/bin/python -m ruff format <path>` (Black-style; splits E702 semicolons).

## Resource discipline (shared device)
- Production runs (≥1 h wall, ≥10 cores): **default to overnight**; confirm before launching during business hours.
- Kill pattern for Python+subprocess workloads: `pkill -f <workload-string>` — `pkill -P parent` misses grandchildren (e.g. discoal spawned by Python workers).
- Watchdog pattern: poll RSS in background bash, SIGKILL if `>threshold_kb`.
- Resource cap: ≤100 worker procs, ≤400 GB total RAM.

## Convention checks (cross-engine)
- **msinv `population_size=N` is diploid Ne** (per-pair coal rate `1/(2N)`, verified at `rust/msinv-core/src/simulator.rs:1586,2104`).
- **msprime ploidy=1**: pass `2·N` to match msinv coal rates. Use `record_full_arg=True` to match msinv's full recomb-boundary tree count.
- **discoal**: same diploid convention; tskit output has `time_units = "coalescent units (2N generations)"` — multiply node times by `2·N` on load.
- `num_trees` cross-engine differs by convention: msinv records all recombination boundaries; msprime default simplifies non-ancestral ones.

## Validation suite (post-2026-05-10)
- v12 demography in `validation/_lib/demography.py`: `v12_msinv()`, `v12_msprime()`, `v12_discoal_events()`. v12 = v11 minus Ghost+Moz, no K↔F migrations.
- Engine runners: `validation/_lib/engines.py` (`msinv_run`, `msprime_run`, `discoal_run`).
- Aggregator: `validation/_lib/aggregator.py::track_equivalence_table(dir_a, dir_b)` — KS + Cohen's D verdict per stat (alpha=0.01, |D|<0.2).
- Track drivers in `validation/track{3,4}_*/run.py`. **Must use parallel** (`ProcessPoolExecutor(max_workers=50)`) — Track 4 first pass shipped serial, cost 3 h wasted.

## Pilot-before-production
- Before launching n=100 reps at production L, run **1 rep at the actual production L** to bench per-rep wall + RSS.
- Smoke tests at scaled-down L are not predictive: L=100 kb smoke at 1.7 s gave zero signal about L=5 Mb at 73 min/rep.

## Resource budgets (msinv at HEAD, observed)
| Scenario | Per-rep wall | Peak RSS |
|---|---|---|
| L=1 Mb + v12 + neutral, n=100 from F | ~5 s | 0.25 GB |
| L=5 Mb + v12 + neutral, n=100 | untested, ~1-2 min est | unknown |
| L=5 Mb + v12 + 3Ra + neutral, n=100 | 67 min | 25.7 GB |
| L=5 Mb + v12 + sweep (no inv), n=100 | 73 min | 0.95 GB |
| L=10 Mb + v12 + 3Ra | **NOT VIABLE** — remnant-ratchet, >32 GB at 19 min | — |

## discoal CLI gotchas (v2.0.0-beta)
- `-wd` (deterministic sweep) is rejected when any `-en` events are present — error: "you chose 1 or more population size changes with a deterministic sweep". Use `-ws` (stochastic) with large α (≥10⁴) — statistically indistinguishable at that scale.
- discoal supports the full ms event set: `-p`, `-en`, `-eg`, `-ed`/`-ej`, `-ea`, `-em`/`-eM`, `-A`, `-D` (demes YAML).

## msprime gotchas
- `msprime.PopulationSplit` not exported at top level — use `msprime.demography.PopulationSplit` for `isinstance` / type-check.
- `add_population_split` + manually-added stair-step `add_population_parameters_change`: call `d.sort_events()` before validate — split internally appends sub-events out of time-order.

## Sweep mechanism notes (for sweep-stack edits)
- Sweep stack gates on `Sweep::has_sv_phase()` (true ⟺ `t_de_novo > t_origin`): per-segment partition fires for SV only; `still_a` force-coalesce for hard only; any-pair picker for SV only; tau-leap dt cap for hard only.
- `target_inv=0` with no inversions configured is a sentinel (works in panmictic sweep tests like D2-D5).

## msinv API conventions
- `Karyotype` enum (`rust/msinv-core/src/class_tag.rs`): variants `S` and `I` only.
- Rust RNG: rand 0.9 — use `rng.random::<f64>()`, NOT `rng.gen::<f64>()`.
- Migration matrix: `m_ij` = "fraction of pop i absorbing from pop j", matching `Demography::migration_matrix[dst][src]`. Forward-flow A→B needs `m(B, A) > 0`.
- `recombination_rate > 0` required; for non-recombining smoke tests use `1e-12`, not zero.
- Sweep + inversion trajectories use discrete-time WF logistic: `p_{t+1} = p_t·(1+s)/(1+s·p_t)` — don't test against continuous `exp(s·t)` form.

## Layout
- Rust core: `rust/msinv-core/src/` (simulator.rs, rate_index.rs, events.rs, lineage.rs, hull_index.rs).
- PyO3 bridge: `rust/msinv-py/src/lib.rs`.
- Python wrapper / tskit conversion: `msinv/hull/_rust_bridge.py`.
- Examples: `examples/kir_fol_pilot.py` is a working reference for v12-shape sims at smaller scales.

## Persistent context (not in repo)
- Project memory: `/home/ssmall/.claude/projects/-home-ssmall-inversion-sims-files/memory/` — read `MEMORY.md` index first.
- Paper materials + docs: `/home/ssmall/inversion_sims/msinv_paper/` — README, CHANGELOG, theory, specs, plans.

## Git
- Default to creating new commits; don't `--amend`.
- HEREDOC for commit messages; `-c commit.gpgsign=false` to avoid signing prompts.
- `git mv` doesn't cross repo boundaries — use plain `mv` + `git add -A` to record deletion + new path.

## When wrong, do this
1. Stop running anything that depends on the wrong assumption.
2. Find ground truth (source, git history, tests).
3. State it to the user, including how prior output was affected.
4. Wait for direction. Don't propose 5 paths and don't guess.
