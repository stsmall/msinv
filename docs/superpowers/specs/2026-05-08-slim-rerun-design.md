# SLiM cross-engine rerun against HEAD msinv (Track B option 1a)

Date: 2026-05-08
Status: brainstorm-approved, pending writing-plans

## Goal

Verify the existing `slim_validation/` harness (committed April 21,
2026) still matches SLiM within MC noise at HEAD, and learn whether
scenario 3 (sweep in inversion) diverges from its April baseline.
Major scenario-3 divergence would be an informative signal about the
post-Apr-30 sweep stack (joint trajectory, SV phase, per-segment
hitchhiking, tag-aware recombination), not a failure.

This is a verification pass, not a redesign. Out of scope: updating
the SLiM `.slim` scripts, designing new scenarios, or extending to
multi-pop / Kir-Fol shapes (those are separate tracks).

## What exists at HEAD

- SLiM 4.2.2 binary at `/home/ssmall/miniforge3/envs/popgen/bin/slim`
- `slim_validation/run_comparison.py` — SLiM + msinv side-by-side
  runner
- `slim_validation/scenarios/scenario{1,2,3}*.slim` — three scenarios:
    1. single inversion, neutral + gene flux
    2. two inversions, neutral + gene flux
    3. single inversion + hard sweep on S karyotype at `x_sel=50_000`
- `slim_validation/plot_comparison.py` — figure generator
- `slim_validation/output/scenario{1,2,3}_results.npz` — April baseline
  (per-window pi_S, pi_I, dxy, Fst aggregates over 3 reps each)
- `figures/slim_validation_scenario{1,2,3}.pdf` — April figures

## API drift since April

`InversionSpec` and `HullSimulator.__init__` are still compatible
with the April call sites in `run_comparison.py`. Scenarios 1 + 2
should run unchanged.

`Sweep` (msinv/hull/sweep.py) was rewritten as part of the Apr-28
sweep stack rewrite. The April call site at `run_comparison.py:119-
124` uses field names that no longer exist:

| April field | HEAD field | Notes |
|---|---|---|
| `x_sel` | `x_sel` | unchanged |
| `t_event` | `tau` | sweep onset (forward time) |
| `target_class="S"` | `origin_kary="S"` | renamed |
| `selection_coefficient` | `s` | renamed |
| `starting_frequency` | `f0` | renamed |
| — | `origin_pop=0` | new required |
| — | `target_inv=0` | new required, anchored to first inversion |
| — | `mode='Stochastic'` | new required choice |

Mode rationale: the SLiM `scenario3` script introduces 20 sweep
copies onto random S-karyotype genomes, giving `f0=20/(2·Ne)=0.01`
at `Ne=1000`. Since `f0 > 1/(2N)=0.0005`, this triggers msinv's
SV phase (standing-variation soft origin) at HEAD. `mode='Stochastic'`
mirrors the discoal D3 soft-sweep calibration — closest semantic
match to SLiM's multi-genome introduction. `Deterministic` would
suppress SV-phase drift; `Neutral` and `StochasticConditioned` are
calibration tools for D-tests, not the right choice here.

## Plan

1. Snapshot `slim_validation/output/scenario{1,2,3}_results.npz` to
   `slim_validation/output/baseline_april/` so the rerun has a
   committed reference to diff against.

2. Port the `Sweep(...)` call in `run_comparison.py:119-124` per the
   field mapping above. Add an inline comment recording the mapping
   so the next reader doesn't have to re-derive it.

3. Run `run_comparison.py --scenario N --reps 3` for N ∈ {1, 2, 3}.
   Three reps matches the April baseline rep count, so the diff is
   apples-to-apples. Single proc; well under the 50–100 worker /
   400 GB resource cap. Compute estimate ≤ 30 min total, dominated
   by SLiM (forward-time at Ne=1000, 8·Ne burn-in).

4. Diff new `.npz` vs `baseline_april/` `.npz`:
   - Per-window % deviation for `pi_S`, `pi_I`, `dxy`, `Fst`
   - Aggregated mean of each over the 40 windows
   - Max single-window deviation
   Write to `.tmp/slim_rerun_diff.md`.

5. Regenerate `figures/slim_validation_scenario{1,2,3}.pdf` via
   `plot_comparison.py --all`.

6. Report findings to user.

## Pass / fail criteria

**Scenarios 1 + 2 (neutral, no sweep):** window-mean stats within
~5% of the April baseline. This matches the historical "msinv
matches SLiM within 5%" tolerance from `a93cae8` and surrounding
commits. Larger deviation is a real regression — stop and
investigate before proceeding to other tracks.

**Scenario 3 (sweep):** no fixed pass criterion. The post-Apr-30
sweep stack rewrites are independently validated by D2/D3 against
discoal at 3·SE. Any divergence here vs the April baseline is the
question being asked, not a failure. Report the new numbers + figure
and let the user decide whether to update the April baseline or to
treat it as a known semantic shift.

## Caveats (pre-existing, documented in slim_validation/README.md)

1. SLiM gene-flux semantics differ from msinv γ. SLiM models short
   non-crossover tracts; msinv models per-bp allele transfer. Effective
   rates are close but not identical.
2. SLiM uses balancing selection (s_bal=0.01) to keep the inversion
   polymorphic; msinv conditions on the specified `p_inv`. Adds a
   small amount of extra coalescent time at the marker.
3. Scenario 3 SLiM run retries on sweep loss; can be slow.

## Risks

- **Scenario 3 mode choice unverified.** `mode='Stochastic'` is the
  best a-priori match for the SLiM setup but has not been bench-
  matched against this specific scenario's sample shape (10 S +
  10 I haploids, x_sel=50_000 inside an inversion). If the diff is
  large, may need to try `Deterministic` or revisit. Documented as
  a follow-up rather than a blocker.

- **April baseline may not be byte-stable.** SLiM run + msinv run
  are seeded (seed = 1000 + rep*100 + attempt for SLiM, 2000 + rep
  for msinv). Given identical seeds + binaries, scenarios 1 + 2 should
  reproduce the April numbers nearly exactly modulo any RNG-order
  changes from the post-Apr-21 perf work (notably the hull-index
  `cdbae9a` introduces BST-order peer iteration). Treat ~1-2% rep-
  level deviation as acceptable; > 5% is a real change.

## Deliverables

- Edit: `slim_validation/run_comparison.py` (Sweep port, ~6 lines)
- New dir: `slim_validation/output/baseline_april/` (3 .npz files)
- New one-shot script: `.tmp/slim_diff_baseline.py` that reads the
  baseline + new `.npz` files and produces the per-window deviation
  report. One-shot (not committed) since this is a single
  verification pass, not a recurring tool.
- Refreshed: `slim_validation/output/scenario{1,2,3}_results.npz`,
  `figures/slim_validation_scenario{1,2,3}.pdf`
- Report: `.tmp/slim_rerun_diff.md`
- No changes to .slim scripts, no changes to msinv source

## Out of scope

- Updating .slim scripts to mirror the new sweep stack semantics
  (option 2 work)
- New scenarios (multi-pop, Kir-Fol shape, partial sweep) (option 3 work)
- ABC harness or production-scale runs (Track A)
- Any perf work on msinv (rejected per resume memory)
