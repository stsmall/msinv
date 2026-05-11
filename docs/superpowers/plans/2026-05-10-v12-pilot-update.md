# v12 Pilot Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the wrong constant-Ne pilot harness with one driven by the v12 demography (K + F + KF + Anc, no Ghost/Moz, no K↔F migrations) and rerun the phase-0 pilot at v12 scale to gate Plan 2.

**Architecture:** Add `validation/_lib/demography.py` with a single `v12_msinv()` builder that returns an `msinv.Demography` object encoding v12 events. Update `validation/pilot/bench_msinv.py` to consume v12, drop the old scalar-Ne parameters, and configure the 3Ra inversion + F-only sampling. Other engine builders (msprime, discoal, SLiM) are deferred to Plan 2.

**Tech Stack:** Python 3.12, numpy, msinv, tskit. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-09-msinv-validation-suite-design.md` (v12 revision committed at `5dedbee` + discoal fix at `cc38ef7`)

**Supersedes:** Tasks 6 + 7 of `docs/superpowers/plans/2026-05-09-validation-infra-and-pilot.md` (the constant-Ne pilot harness that blew up at 180 GB RSS). Tasks 1–5 of the previous plan (seeds, stats, equivalence, IO) are demography-agnostic and remain valid; do NOT re-implement them.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `validation/_lib/demography.py` | Create | `v12_msinv() -> Demography` builder + constants |
| `tests/validation/test_demography.py` | Create | Tests for v12 event count, ordering, key Ne values |
| `validation/pilot/bench_msinv.py` | Modify (rewrite signature) | Drop scalar-Ne params; consume v12 + InversionSpec + F-only sampling |
| `tests/validation/test_pilot_bench.py` | Modify | Update smoke test to call the new signature at small-L |
| `results/validation/pilot/rep_{000..002}/` | Create | Pilot output |
| `.tmp/pilot_v12_report.md` | Create | Pass/fail report (not committed) |

No changes to T1–T5 modules (`seeds.py`, `stats.py`, `equivalence.py`, `io.py`).
No msinv source changes.

---

### Task 1: v12 demography builder for msinv

**Files:**
- Create: `validation/_lib/demography.py`
- Create: `tests/validation/test_demography.py`

**Why:** v12 is the single canonical demographic model used across all 5 tracks of the validation suite. A pure-Python builder ensures every track and the pilot use bit-identical event lists.

- [ ] **Step 1: Write failing tests**

Create `tests/validation/test_demography.py`:

```python
"""Tests for v12 demography builder."""
import pytest

from msinv import Demography
from validation._lib.demography import (
    v12_msinv,
    NE_K_PRESENT, NE_F_PRESENT, NE_F_AT_SPLIT,
    NE_KF_AT_MERGE, NE_ANC_DEEP,
    T_KF_SPLIT, T_ANC_RENAME,
    T_INV_3RA, P_INV_F_3RA, P_INV_K_3RA, P_INV_ANC_3RA,
    GAMMA_3RA,
)


def test_v12_returns_demography():
    d = v12_msinv()
    assert isinstance(d, Demography)


def test_v12_two_populations():
    """v12 has exactly K (pop 0) and F (pop 1) as named pops."""
    d = v12_msinv()
    assert len(d.pop_sizes) == 2
    assert d.pop_sizes[0] == NE_K_PRESENT
    assert d.pop_sizes[1] == NE_F_PRESENT


def test_v12_constants():
    """Sanity-check the v12 constants against Small 2023 / v11 file."""
    assert NE_K_PRESENT == 126_772
    assert NE_F_PRESENT == 2_496_632
    assert NE_F_AT_SPLIT == 158_711
    assert NE_KF_AT_MERGE == 86_000
    assert NE_ANC_DEEP == 450_000
    assert T_KF_SPLIT == 9_194
    assert T_ANC_RENAME == 87_163
    assert T_INV_3RA == 330_000
    assert P_INV_F_3RA == 0.73
    assert P_INV_K_3RA == 0.0


def test_v12_has_kf_split_event():
    """An 'ej' event at T_KF_SPLIT joining F (pop 1) into K (pop 0)."""
    d = v12_msinv()
    events = list(d.events)
    ej_events = [e for e in events if e[0] == "ej"]
    assert any(e[1] == T_KF_SPLIT and e[2] == 1 and e[3] == 0
               for e in ej_events), (
        f"expected ('ej', {T_KF_SPLIT}, 1, 0) — got ej events: {ej_events}")


def test_v12_has_anc_deep_size_change():
    """An 'en' event at T_ANC_RENAME setting pop 0 to NE_ANC_DEEP."""
    d = v12_msinv()
    events = list(d.events)
    en_events = [e for e in events if e[0] == "en"]
    assert any(e[1] == T_ANC_RENAME and e[2] == 0 and e[3] == NE_ANC_DEEP
               for e in en_events), (
        f"expected ('en', {T_ANC_RENAME}, 0, {NE_ANC_DEEP}) — got: {en_events}")


def test_v12_no_migration_events():
    """v12 has zero migration events (the agreed K↔F simplification)."""
    d = v12_msinv()
    events = list(d.events)
    mig_events = [e for e in events if e[0] in ("em", "eM")]
    assert mig_events == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/validation/test_demography.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'validation._lib.demography'`

- [ ] **Step 3: Implement v12_msinv builder**

Create `validation/_lib/demography.py`:

```python
"""v12 demography for the msinv validation suite.

v12 = v11 from examples/kir_fol_demography.py minus Ghost and Moz pops,
minus all K↔F migrations. Two populations (K=0, F=1) at present,
merged into ancestral pop at the K-F split, with a deep ancestral size
change. All Ne(t) stair-steps from v11 are preserved.

Spec: docs/superpowers/specs/2026-05-09-msinv-validation-suite-design.md
"""
from __future__ import annotations

from msinv import Demography

# --- v12 parameters (Small 2023 + v11 lock) -------------------------

# Present-day Ne (t=0 ancestrally backward)
NE_K_PRESENT = 126_772
NE_F_PRESENT = 2_496_632

# F at the K-F split
NE_F_AT_SPLIT = 158_711

# Merged KF pop at the moment of split
NE_KF_AT_MERGE = 86_000

# Deep ancestral Ne (after the KF -> Anc rename)
NE_ANC_DEEP = 450_000

# Times (generations backward, present = 0)
T_KF_SPLIT = 9_194
T_ANC_RENAME = 87_163

# Inversion (3Ra)
T_INV_3RA = 330_000
P_INV_F_3RA = 0.73
P_INV_K_3RA = 0.0
P_INV_ANC_3RA = 0.30
GAMMA_3RA = 1.0e-7


def v12_msinv() -> Demography:
    """Build the v12 msinv.Demography object.

    Backward-time events ordered from present to deep past. K = pop 0,
    F = pop 1. After K-F join at T_KF_SPLIT, the merged pop continues
    as pop 0 (the ms convention used by msinv).
    """
    d = Demography(pop_sizes=[NE_K_PRESENT, NE_F_PRESENT])

    # ---- K Ne(t): ABC stair-step (t=200 dip dropped per v11 lock) --
    d.add_event(("en", 400.0,   0, 161_546))
    d.add_event(("en", 600.0,   0, 152_453))
    d.add_event(("en", 1_400.0, 0, 174_800))
    d.add_event(("en", 3_000.0, 0, 182_180))
    d.add_event(("en", 6_200.0, 0, 159_861))

    # ---- F Ne(t): ABC stair-step ----------------------------------
    d.add_event(("en", 400.0,   1, 1_157_768))
    d.add_event(("en", 600.0,   1, 205_260))
    d.add_event(("en", 1_000.0, 1, 1_374_810))
    d.add_event(("en", 1_400.0, 1, 674_766))
    d.add_event(("en", 3_000.0, 1, 340_074))
    d.add_event(("en", 6_200.0, 1, NE_F_AT_SPLIT))

    # ---- K-F split: F (pop 1) joins K (pop 0) at T_KF_SPLIT -------
    d.add_event(("ej", float(T_KF_SPLIT), 1, 0))
    d.add_event(("en", float(T_KF_SPLIT), 0, NE_KF_AT_MERGE))

    # ---- KF Ne(t) trajectory (50k bottleneck floor per v11 lock) --
    d.add_event(("en", 13_000.0, 0, 81_072))
    d.add_event(("en", 20_000.0, 0, 95_546))
    d.add_event(("en", 30_000.0, 0, 73_250))
    d.add_event(("en", 40_000.0, 0, 50_000))
    d.add_event(("en", 50_000.0, 0, 50_000))
    d.add_event(("en", 60_000.0, 0, 50_000))
    d.add_event(("en", 70_000.0, 0, 50_000))

    # ---- KF -> Anc rename: deep ancestral Ne change ---------------
    d.add_event(("en", float(T_ANC_RENAME), 0, NE_ANC_DEEP))

    return d
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/validation/test_demography.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

Run:
```bash
git add validation/_lib/demography.py tests/validation/test_demography.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(validation): v12 demography builder for msinv

v12 = v11 minus Ghost and Moz pops, minus K↔F migrations. Two pops
(K, F) with stair-step Ne(t), join at t=9,194, KF stair-step, deep
Anc rename at t=87,163. Parameters locked from Small 2023 / v11 file.

Spec: docs/superpowers/specs/2026-05-09-msinv-validation-suite-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Update pilot harness to use v12 + F-only sampling + 3Ra inversion

**Files:**
- Modify: `validation/pilot/bench_msinv.py` (rewrite `run_pilot_rep` signature; update `_cli_main`)
- Modify: `tests/validation/test_pilot_bench.py` (update smoke test to new signature)

**Why:** The current pilot harness takes scalar `Ne` and a generic inversion config, which can't carry the v12 stair-step Ne(t). The new signature consumes the v12 Demography directly plus a single `L` knob; everything else (3Ra params, F-only sampling) is fixed by the spec.

- [ ] **Step 1: Update the smoke test**

Replace the entire contents of `tests/validation/test_pilot_bench.py` with:

```python
"""Smoke tests for the pilot bench harness at SCALED-DOWN L.

The full bench (L=10 Mb on v12) is too slow for unit tests; we test
the harness mechanics here and run the real bench manually in Task 3.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from validation.pilot.bench_msinv import run_pilot_rep


def test_smoke_run_creates_outputs(tmp_path):
    """Run the bench at L=50 kb and verify it produces stats + timing."""
    out_dir = tmp_path / "rep_000"
    result = run_pilot_rep(
        out_dir=out_dir,
        rep=0,
        L=50_000,
        seed=12345,
    )
    assert (out_dir / "stats.npz").exists()
    assert (out_dir / "timing.json").exists()
    timing = json.loads((out_dir / "timing.json").read_text())
    assert "wall_seconds" in timing
    assert "peak_rss_bytes" in timing
    assert "iters_consumed" in timing
    assert "num_trees" in timing
    assert "num_sites" in timing
    assert timing["wall_seconds"] > 0
    assert result["wall_seconds"] == timing["wall_seconds"]


def test_smoke_stats_has_expected_keys(tmp_path):
    out_dir = tmp_path / "rep_000"
    run_pilot_rep(
        out_dir=out_dir, rep=0, L=50_000, seed=12345,
    )
    z = np.load(out_dir / "stats.npz", allow_pickle=False)
    keys = set(z.files)
    # Spot-check a few stats from each module are present.
    # F-only sampling: subgroups are F_S and F_I (from p_inv_F=0.73).
    assert "pi__F_S" in keys
    assert "pi__F_I" in keys
    assert "dxy__F_I_F_S" in keys or "dxy__F_S_F_I" in keys
    assert "fst__F_I_F_S" in keys or "fst__F_S_F_I" in keys
    assert any(k.startswith("tajimas_d__") for k in keys)
    assert any(k.startswith("tree_") for k in keys)
    assert any(k.startswith("ld_") for k in keys)
```

- [ ] **Step 2: Run smoke test to verify it fails**

Run: `.venv/bin/python -m pytest tests/validation/test_pilot_bench.py -v --timeout=120`
Expected: FAIL — the old `run_pilot_rep` signature doesn't match these args (it expected `Ne`, `n_samples`, `inv_bp_left`, etc.).

- [ ] **Step 3: Rewrite the pilot harness**

Replace the entire contents of `validation/pilot/bench_msinv.py` with:

```python
"""Phase-0 pilot bench: msinv on v12 demography.

Runs a single rep at L=10 Mb (or smaller for smoke tests) on the v12
Kir/Fol demography with a 3Ra inversion in F. Measures wall + peak RSS
+ iters consumed, computes the full validation-suite stats panel, and
persists everything to `out_dir / {stats.npz, timing.json}`.

Used to gate the full n=100 launch: per-rep wall < 4h AND peak RSS < 8GB
must both hold.
"""
from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import numpy as np
import msprime

from msinv import HullSimulator, InversionSpec
from validation._lib import io, stats
from validation._lib.demography import (
    v12_msinv,
    T_INV_3RA, P_INV_F_3RA, P_INV_K_3RA, GAMMA_3RA,
)

# 3Ra geometry: position 0.18·L start, width 0.20·L.
INV_LEFT_FRAC = 0.18
INV_WIDTH_FRAC = 0.20
MEAN_TRACT_FRAC = 0.05  # fraction of inv_width

# Rates per spec
MU = 1.0e-8
R = 1.0e-8

# F-only sampling at n=100 with p_inv_F = 0.73 → 27 F_S + 73 F_I.
N_F_S = 27
N_F_I = 73
N_TOTAL = N_F_S + N_F_I


def run_pilot_rep(
    *,
    out_dir: str | Path,
    rep: int,
    L: float,
    seed: int,
    iters_max: int = 1_000_000_000,
) -> dict[str, float]:
    """Run one msinv pilot rep on v12 + 3Ra at the given L and persist outputs.

    Returns a small dict with the timing info that's also written to disk.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inv_bp_left = INV_LEFT_FRAC * L
    inv_bp_right = inv_bp_left + INV_WIDTH_FRAC * L
    inv_width = inv_bp_right - inv_bp_left

    inv_3ra = InversionSpec(
        bp_left=int(inv_bp_left),
        bp_right=int(inv_bp_right),
        p_inv={0: P_INV_K_3RA, 1: P_INV_F_3RA},
        t_inv=float(T_INV_3RA),
        gene_conversion_rate=GAMMA_3RA,
        mean_tract_length=MEAN_TRACT_FRAC * inv_width,
        tract_distribution="fixed",
        inv_id=0,
    )

    sim = HullSimulator(
        sample_config={("S", 0): 0, ("S", 1): N_F_S, ("I", 1): N_F_I},
        demography=v12_msinv(),
        sequence_length=float(L),
        recombination_rate=R,
        inversions=[inv_3ra],
        sweeps=[],
        seed=int(seed),
        iters_max=iters_max,
    )

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.time()
    ts_raw = sim.simulate()
    wall = time.time() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in KB on Linux; convert to bytes
    peak_rss = max(rss_before, rss_after) * 1024

    # Overlay neutral mutations
    ts = msprime.sim_mutations(
        ts_raw, rate=MU, random_seed=seed + 1, keep=True,
    )

    # Sample-set partition: F_S = first N_F_S, F_I = next N_F_I.
    # (sample_config above produces samples in order (S,1)x27, (I,1)x73.)
    samples = list(ts.samples())
    sset = {"F_S": samples[:N_F_S], "F_I": samples[N_F_S:]}

    win = stats.window_stats(ts, sample_sets=sset, n_windows=40)
    sfs_S = stats.sfs(ts, sample_set=sset["F_S"], folded=True)
    sfs_I = stats.sfs(ts, sample_set=sset["F_I"], folded=True)
    tree_d = stats.tree_shape_stats(ts, n_samples=200, seed=seed + 2)
    bins = np.logspace(2, np.log10(L), 11)
    ld_d = stats.ld_decay(
        ts, distance_bins=bins, max_pairs=2000, seed=seed + 3,
    )

    flat: dict[str, np.ndarray] = {}
    for sname, arr in win["pi"].items():
        flat[f"pi__{sname}"] = arr
    for pname, arr in win["dxy"].items():
        flat[f"dxy__{pname}"] = arr
    for pname, arr in win["fst"].items():
        flat[f"fst__{pname}"] = arr
    for sname, arr in win["tajimas_d"].items():
        flat[f"tajimas_d__{sname}"] = arr
    flat["sfs__F_S"] = sfs_S
    flat["sfs__F_I"] = sfs_I
    flat["tree_tmrca"] = tree_d["tmrca"]
    flat["tree_total_branch"] = tree_d["total_branch"]
    flat["tree_colless"] = tree_d["colless"]
    flat["ld_bin_edges"] = ld_d["bin_edges"]
    flat["ld_mean_r2"] = ld_d["mean_r2"]
    flat["ld_count"] = ld_d["count"]
    flat["num_trees"] = np.asarray(ts.num_trees)
    flat["num_sites"] = np.asarray(ts.num_sites)
    io.save_rep_stats(out_dir / "stats.npz", **flat)

    timing = {
        "wall_seconds": float(wall),
        "peak_rss_bytes": int(peak_rss),
        "iters_consumed": int(getattr(sim, "iters_used", -1)),
        "num_trees": int(ts.num_trees),
        "num_sites": int(ts.num_sites),
        "rep": int(rep),
        "L": float(L),
        "seed": int(seed),
        "n_samples": int(N_TOTAL),
    }
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    return timing


def _cli_main():
    """Run the production-scale pilot: 3 reps at L=10 Mb on v12."""
    import sys
    from validation._lib.seeds import seed_for

    out_root = Path("results/validation/pilot")
    n_reps = 3
    L = 10_000_000
    timings = []
    for rep in range(n_reps):
        out_dir = out_root / f"rep_{rep:03d}"
        seed = seed_for(
            track="pilot", scenario="v12", engine="msinv", rep=rep,
        )
        print(
            f"Pilot rep {rep}: seed={seed}, L={L}, out={out_dir}",
            flush=True,
        )
        t = run_pilot_rep(out_dir=out_dir, rep=rep, L=L, seed=seed)
        timings.append(t)
        print(
            f"  wall={t['wall_seconds']:.1f}s, "
            f"peak_rss={t['peak_rss_bytes'] / 1e9:.2f} GB, "
            f"trees={t['num_trees']}, sites={t['num_sites']}",
            flush=True,
        )

    walls = [t["wall_seconds"] for t in timings]
    rsses = [t["peak_rss_bytes"] for t in timings]
    print(f"\nPilot summary over {n_reps} reps:")
    print(
        f"  wall: median={np.median(walls):.1f}s, "
        f"min={min(walls):.1f}s, max={max(walls):.1f}s"
    )
    print(
        f"  rss : median={np.median(rsses) / 1e9:.2f}GB, "
        f"max={max(rsses) / 1e9:.2f}GB"
    )
    if max(walls) > 4 * 3600:
        print("  GATE: ❌ per-rep wall > 4h — escalate before full launch")
        sys.exit(2)
    if max(rsses) > 8 * 1e9:
        print("  GATE: ⚠️ per-rep RSS > 8GB — discuss before full launch")
        sys.exit(1)
    print("  GATE: ✅ within pilot pass criteria")


if __name__ == "__main__":
    _cli_main()
```

- [ ] **Step 4: Run smoke test to verify it passes**

Run: `.venv/bin/python -m pytest tests/validation/test_pilot_bench.py -v --timeout=120`
Expected: 2 passed.

- [ ] **Step 5: Commit**

Run:
```bash
git add validation/pilot/bench_msinv.py tests/validation/test_pilot_bench.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(validation): pilot harness now consumes v12 demography

run_pilot_rep dropped scalar-Ne signature; new signature takes
(out_dir, rep, L, seed). v12 demography + 3Ra inversion + F-only
n=100 sampling (27 F_S + 73 F_I per p_inv=0.73) are spec-fixed.
CLI driver runs 3 reps at L=10 Mb and applies the spec gate
(wall < 4h, RSS < 8 GB).

Spec: docs/superpowers/specs/2026-05-09-msinv-validation-suite-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Run the v12 pilot + report

**Files:**
- Create: `results/validation/pilot/rep_{000,001,002}/{stats.npz, timing.json}`
- Create: `.tmp/pilot_v12_report.md`

**Why:** This is the gating measurement. Determines Plan 2 scope.

**Resource note:** msinv at v12 + L=10 Mb has not been benched. The CLAUDE.md remnant-ratchet warning targets `Ne ≥ 1e6 + old inversions ≥ 100k gen`. v12's peak Ne is 2.5M (F current) AND t_inv=330k, both in the regime. The previous attempt at Ne=1e6 constant + t_inv=4M consumed 180 GB. The v12 demography has small Ne for the bulk of the timeline (K=44k–180k, F shrinks to 158k at the split, KF=50k–95k for the long ancestral era), which SHOULD avoid the worst remnant-ratchet behavior, but this needs to be verified empirically. **Monitor RAM during the run.**

- [ ] **Step 1: Run the pilot — with RAM monitoring**

Run the pilot in the background and monitor RAM every 60 s. Kill if RSS exceeds 32 GB (4× the spec gate; protects the shared device).

```bash
mkdir -p .tmp results/validation/pilot
.venv/bin/python -m validation.pilot.bench_msinv 2>&1 \
  | tee .tmp/pilot_v12_run.log &
PILOT_PID=$!
echo "pilot PID=$PILOT_PID"

# RAM watchdog: kill if RSS > 32 GB
while kill -0 $PILOT_PID 2>/dev/null; do
    sleep 60
    RSS_KB=$(ps -o rss= -p $PILOT_PID 2>/dev/null | tr -d ' ')
    if [ -z "$RSS_KB" ]; then break; fi
    RSS_GB=$(awk "BEGIN { printf \"%.1f\", $RSS_KB / 1024 / 1024 }")
    echo "  watchdog: pilot RSS = ${RSS_GB} GB"
    if [ "$RSS_KB" -gt 33554432 ]; then  # 32 GB in KB
        echo "  watchdog: RSS > 32 GB, KILLING pilot"
        kill -9 $PILOT_PID
        break
    fi
done
wait $PILOT_PID 2>/dev/null
echo "Pilot exit status: $?"
```

Expected outcome (one of three):
- ✅ All 3 reps complete with wall < 4 h and RSS < 8 GB per rep
- ⚠️ Reps complete but exceed wall or RSS gate (still report numbers)
- ❌ Watchdog kills the run (RAM exceeded 32 GB) — record the partial state and escalate

- [ ] **Step 2: Verify outputs**

```bash
ls -la results/validation/pilot/rep_*/ 2>&1
cat .tmp/pilot_v12_run.log | tail -30
```
Expected: 3 dirs `rep_000`, `rep_001`, `rep_002`, each with `stats.npz` and `timing.json`. (Or fewer if the run was killed mid-pilot.)

- [ ] **Step 3: Write the pilot report**

Read each `results/validation/pilot/rep_NNN/timing.json` and produce `.tmp/pilot_v12_report.md` with the following structure, filling in actual numbers from disk:

```markdown
# v12 Pilot Bench Report

Date: <fill from `date -I`>
Spec: docs/superpowers/specs/2026-05-09-msinv-validation-suite-design.md
Plan: docs/superpowers/plans/2026-05-10-v12-pilot-update.md
Demography: v12 (Small 2023 / v11-derived, no Ghost no Moz no K↔F migrations)

## Parameters

L = 10 Mb, n = 100 from F (27 F_S + 73 F_I per p_inv=0.73).
3Ra inversion at 0.18·L start, 0.20·L wide (1.8–3.8 Mb).
t_inv = 330,000 gen. μ = r = 1e-8, γ_3Ra = 1e-7.

## Per-rep results

| rep | seed | wall (s) | peak RSS (GB) | iters consumed | num_trees | num_sites |
|---|---|---|---|---|---|---|
| 0 | <fill> | <fill> | <fill> | <fill> | <fill> | <fill> |
| 1 | <fill> | <fill> | <fill> | <fill> | <fill> | <fill> |
| 2 | <fill> | <fill> | <fill> | <fill> | <fill> | <fill> |

Median wall: <fill> s.
Median RSS: <fill> GB.
Max wall: <fill> s.
Max RSS: <fill> GB.

## Pass / fail vs spec gates

- Wall < 4 h per rep: <✅|❌|⚠️>
- RSS < 8 GB per rep: <✅|❌|⚠️>
- No "barrier era INCOMPLETE" warnings in pilot log: <✅|❌>

## Recommendation for Plan 2

One of:
- "Proceed to full n=100 across Tracks 3 + 4 + Q-bias on local 50-cpu, then Tracks 1 + 2 + 5 on HPC."
- "Scale-down required for inversion-bearing tracks; tracks 1/2/5 need L reduction or sample-size reduction before launch."
- "Escalate — v12 pilot still exceeds 32 GB / 8 hour envelope; the realistic-scale validation claim needs rethinking."

## Total compute estimate (extrapolation)

At median per-rep wall of <fill> s, n=100 = <fill> CPU-hours per track.
Tracks 3 + 4 + Q-bias on local 50 cpu: <fill> hours wall.
Tracks 1 + 2 + 5 on HPC SLURM at 100 cpu: <fill> hours wall.
```

- [ ] **Step 4: Commit pilot results**

Run:
```bash
git add results/validation/pilot/
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
test(validation): v12 phase-0 pilot bench results

3 reps msinv at v12 demography, L=10 Mb, n=100 from F, single 3Ra.
Per-rep timing + RSS captured. Verdict vs spec gates in
.tmp/pilot_v12_report.md (scratch artifact, not committed).

Spec: docs/superpowers/specs/2026-05-09-msinv-validation-suite-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(If the run was killed mid-pilot, commit the partial state and the report still — the empirical RAM curve is the headline finding.)

- [ ] **Step 5: Report findings to controller**

Output the contents of `.tmp/pilot_v12_report.md` so the controller can decide whether to proceed to Plan 2 or escalate. Keep the summary tight.

---

## Self-Review

**Spec coverage:**
- Spec § "v12 demography (used by ALL tracks)" → Task 1 (`v12_msinv()` builder) ✓
- Spec § "Inversion (Tracks 1, 2, 5)" — 3Ra params → Task 2 (`InversionSpec` in pilot) ✓
- Spec § "Pilot phase 0 — bench msinv on v12" → Tasks 2 + 3 ✓
- Spec § pass/fail gates (wall < 4h, RSS < 8GB) → Task 2 CLI driver + Task 3 report ✓

Spec items deferred to later plans (out of scope for this plan):
- Track runners for 1/2/3/4/5 (Plan 2 + Plan 3)
- HPC SLURM scripts (Plan 3)
- v12 builders for msprime / discoal / SLiM (Plan 2 — only msinv builder here)
- Q-bias side-track (Plan 2)
- Plot module + equivalence-table generation (Plan 2 once data exists)

**Placeholder scan:** No "TBD" / "implement later" / "add appropriate". `<fill>` markers in the Task 3 report template are filled at runtime from disk; the agent reads `timing.json` and writes actual numbers. All test code shown in full.

**Type / name consistency:**
- `v12_msinv()` and its constants are imported into both `test_demography.py` and `bench_msinv.py` under the same names.
- `run_pilot_rep(out_dir=, rep=, L=, seed=)` signature is consistent across smoke test, CLI, and report.
- `sample_config={("S", 0): 0, ("S", 1): N_F_S, ("I", 1): N_F_I}` matches `HullSimulator.__init__` API from CLAUDE.md.
- Stat keys (`pi__F_S`, `pi__F_I`, `dxy__F_S_F_I`, etc.) follow the existing `validation/_lib/io.py` `__`-separator convention.

**Known soft spot:** The pilot is uncharted territory for msinv at v12 + L=10 Mb. The RAM watchdog at 32 GB protects the shared device; the spec gate (8 GB) is the success threshold. Task 3 explicitly handles the "kill mid-run" case by committing the partial state.
