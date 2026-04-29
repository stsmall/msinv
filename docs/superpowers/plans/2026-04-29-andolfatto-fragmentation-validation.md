# Andolfatto Fragmentation Correction — Validation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate (or refute) the fragmentation-corrected closed form `f_corrected(t) = (1 - exp(-r_A·t)) / (2 - exp(-r_A·t))` derived in `docs/theory/2026-04-29-andolfatto-fragmentation-correction.md` against an independent Monte Carlo of the simulator's flux mechanics, then either tighten the existing Tier-3 test or document the residual gap as the next theory step.

**Architecture:** Build a single-lineage Monte Carlo (Python) that simulates the backward-time flux process exactly as the simulator does — same `phi(x, w)` placement, same `Exp(1/λ)` tract length, same tract clipping. No coalescence (we want to isolate the flux-fragmentation mechanism from the coalescence correction). Compare empirical f̂(t) to both Andolfatto and corrected formulas. Stop at a research-decision point and report findings; do not pre-commit to a test change without empirical evidence.

**Tech Stack:** Python 3.12 (`.venv/bin/python`), NumPy, matplotlib (optional for the residual plot). No new project dependencies.

---

## Pre-flight (read once before starting)

- Spec: `docs/theory/2026-04-29-andolfatto-fragmentation-correction.md`. Read §1 (notation), §2 (mechanism), §3 (derivation), §4 (open questions). The closed-form formula being validated is eq 9: `f_corrected(t) = (1 - exp(-r_A·t)) / (2 - exp(-r_A·t))` with `r_A = γ·p_other·λ²/L`.
- Existing reference test: `tests/hull/test_phase3b_b2_flux.py::test_andolfatto_sample_fraction_matches_closed_form` uses parameters `γ=1.5e-5, λ=300, L=6000, p_other=0.5, t_inv ∈ {1000, 4000, 10_000, 25_000}`. The validation script must use these exact parameters at minimum (plus optionally a few more for shape checking).
- Existing helper for reference behavior: `msinv/hull/_event_log.py::samples_converted_at`. The helper counts samples whose ancestral path at `x_c` was hit by ≥1 flux event. **Our MC tracks one lineage's path directly — no marginal-tree lookup needed.**
- Project conventions in `/home/ssmall/inversion_sims/files/CLAUDE.md`:
  - `.tmp/` not `/tmp/` for scratch output (the script's plot, if generated, goes to `.tmp/`).
  - `.venv/bin/python` not system Python.
- Working directory: `/home/ssmall/inversion_sims/files`.
- Branch: `main` (already merged sweep-followups). Stay on main; this is theory work, not a feature.

---

## File structure

Files this plan creates or modifies:

- **Create**: `scripts/validate_andolfatto_fragmentation.py` — the MC validation tool. Self-contained Python script (no module imports from `msinv/`), parameterizable via CLI args. Runs in seconds for default `N_reps=10_000`.
- **Modify (conditional, after research-decision point)**: `tests/hull/test_phase3b_b2_flux.py` — add `test_andolfatto_sample_fraction_matches_corrected_form` only if MC validation supports the formula at the parameters of the test ladder.
- **Modify (always)**: `docs/theory/2026-04-29-andolfatto-fragmentation-correction.md` — append a final "§8. Validation results" section with concrete numbers from the MC.

The script is intentionally outside `msinv/` because (a) it doesn't reuse simulator code (different oracle), (b) it's a one-shot validation tool, not a long-lived library function.

---

## Task 1: Build the single-lineage MC simulator

**Files:**
- Create: `scripts/validate_andolfatto_fragmentation.py`

**Goal:** A self-contained Python script that, given `(γ, λ, L, p_other, t_max, N_reps, seed)`, returns empirical f̂(t) for a grid of `t` values, sampled by simulating the sample's ancestral lineage at `x_c` going backward in time through flux events.

### What the MC must do

1. **State**: each rep tracks one lineage = a list of segments `[(l_i, r_i), ...]` covering positions in `[bp_left, bp_right]`. The focal position is `x_c = (bp_left + bp_right) / 2`. Initial state at `t = 0`: one segment `[(bp_left, bp_right)]` (full coverage).
2. **Per-event waiting time**: exponential with rate `γ·p_other·λ·(C/L)` where `C = sum(r_i - l_i for i)` is total in-inversion coverage. (Per-lineage flux rate; matches `flux_lineage_rate_arena` in the simulator's hot path.)
3. **Per-event mechanics** (matches `sample_flux_position` + `draw_tract` in `rust/msinv-core/src/simulator.rs`):
   - Sample `x_event` from segments weighted by the simulator's `phi(x, w)` profile, where `w = λ/L` is the dimensionless tract-length-to-inversion ratio. The phi profile is `phi(x, w) = (sinh(w*x) * sinh(w*(1-x))) / sinh(w)` for `x ∈ [0, 1]` (relative position in the inversion). For `w ≪ 1`, `phi ≈ x*(1-x)*w` — but for the test parameters `w = 300/6000 = 0.05`, so the boundary correction is small.
   - Sample tract length `T ~ Exp(1/λ)`, then clip to `min(T, 0.99 * L)`.
   - Sample `b1 ~ Uniform[max(0, x_event - T), min(L - T, x_event)]`. The tract is `[b1, b1 + T]`.
4. **Update state**:
   - If `x_c ∈ [b1, b1 + T]` AND `x_c` is in the current segment list: **flip event**. Set `flipped[rep] = True`. Lineage's new segments = the intersection of `[b1, b1+T]` with the current segment list (the tract piece). Continue from the tract piece.
   - Else: **outside event**. Lineage's new segments = current segments minus `[b1, b1+T]` (the outside piece, with a hole at the tract location). Continue.
5. **Termination**: when cumulative time `s` exceeds `t_max`, stop. Empirical f̂(t) at any `t ≤ t_max` is the fraction of reps that have `flipped` set to True by their first flip-event time `≤ t`.

### Approximations that are OK for v1

- We use exponential tract length (the simulator's continuous-time analog of geometric).
- The phi profile is simulator-faithful but not used to weight `b1` placement (b1 is uniform within its constraint window, matching the simulator).

### Step-by-step

- [ ] **Step 1: Write the script skeleton**

Create `scripts/validate_andolfatto_fragmentation.py`:

```python
"""Monte Carlo validation of the Andolfatto fragmentation-corrected formula.

Simulates the backward-time flux process on a single sample's ancestral
lineage at x_c, exactly matching the simulator's per-event mechanics
(phi-weighted x_event placement, Exp(1/λ) tract length, b1 uniform
within constraint window). Reports empirical f̂(t) at a grid of t values
and compares to:

  - f_A(t) = 1 - exp(-r_A·t)                                  (Andolfatto)
  - f_corrected(t) = (1 - exp(-r_A·t)) / (2 - exp(-r_A·t))    (eq 9 of spec)

with r_A = γ · p_other · λ² / L.

Spec: docs/theory/2026-04-29-andolfatto-fragmentation-correction.md
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Params:
    gamma: float
    lam: float
    L: float
    p_other: float
    bp_left: float = 2000.0
    bp_right: float = 8000.0  # so x_c = 5000 and L = bp_right - bp_left = 6000

    @property
    def x_c(self) -> float:
        return 0.5 * (self.bp_left + self.bp_right)

    @property
    def r_A(self) -> float:
        return self.gamma * self.p_other * (self.lam ** 2) / self.L


def f_andolfatto(t: float, params: Params) -> float:
    return 1.0 - math.exp(-params.r_A * t)


def f_corrected(t: float, params: Params) -> float:
    e = math.exp(-params.r_A * t)
    return (1.0 - e) / (2.0 - e)
```

Run: `.venv/bin/python -c "from scripts.validate_andolfatto_fragmentation import Params; p = Params(1.5e-5, 300, 6000, 0.5); print(p.r_A)"`
Expected output (no error): `0.0001125`

- [ ] **Step 2: Verify the formulas**

Add a `__main__` smoke check:

```python
def _smoke() -> None:
    p = Params(gamma=1.5e-5, lam=300.0, L=6000.0, p_other=0.5)
    assert abs(p.r_A - 1.125e-4) < 1e-12
    assert f_andolfatto(0.0, p) == 0.0
    assert f_corrected(0.0, p) == 0.0
    # At r_A·t = ln(2), Andolfatto = 0.5; corrected has e^{-ln 2} = 0.5,
    # so corrected = 0.5/1.5 = 1/3.
    t_half = math.log(2.0) / p.r_A
    assert abs(f_andolfatto(t_half, p) - 0.5) < 1e-9
    assert abs(f_corrected(t_half, p) - 1.0/3.0) < 1e-9
    print("formulas: OK")


if __name__ == "__main__":
    _smoke()
```

Run: `.venv/bin/python scripts/validate_andolfatto_fragmentation.py`
Expected output: `formulas: OK`

- [ ] **Step 3: Implement the segment data structure**

Add segment helpers to the script. Segments are `list[tuple[float, float]]` (sorted, non-overlapping):

```python
def total_coverage(segments: list[tuple[float, float]]) -> float:
    return sum(r - l for (l, r) in segments)


def position_in_segments(x: float, segments: list[tuple[float, float]]) -> bool:
    for (l, r) in segments:
        if l <= x < r:
            return True
    return False


def subtract_interval(
    segments: list[tuple[float, float]],
    a: float, b: float,
) -> list[tuple[float, float]]:
    """Return segments minus the interval [a, b)."""
    out = []
    for (l, r) in segments:
        if r <= a or l >= b:
            out.append((l, r))
            continue
        if l < a:
            out.append((l, a))
        if r > b:
            out.append((b, r))
    return out


def intersect_interval(
    segments: list[tuple[float, float]],
    a: float, b: float,
) -> list[tuple[float, float]]:
    """Return segments intersected with [a, b)."""
    out = []
    for (l, r) in segments:
        ll = max(l, a)
        rr = min(r, b)
        if ll < rr:
            out.append((ll, rr))
    return out
```

Add a unit test in the same file:

```python
def _test_segments() -> None:
    s = [(0.0, 100.0)]
    assert total_coverage(s) == 100.0
    assert position_in_segments(50.0, s)
    assert not position_in_segments(150.0, s)
    s2 = subtract_interval(s, 30.0, 70.0)
    assert s2 == [(0.0, 30.0), (70.0, 100.0)]
    assert total_coverage(s2) == 60.0
    s3 = intersect_interval(s, 30.0, 70.0)
    assert s3 == [(30.0, 70.0)]
    print("segments: OK")
```

Update `_smoke()` to call `_test_segments()` first. Run: `.venv/bin/python scripts/validate_andolfatto_fragmentation.py`
Expected: `segments: OK\nformulas: OK`

- [ ] **Step 4: Implement x_event sampling under the phi profile**

The simulator's `sample_flux_position` weights segments by `phi_integral(a, b, w) * inv_len`, with `w = λ/L` and `phi(x, w) = sinh(w*x) * sinh(w*(1-x)) / sinh(w)`. For our purposes the phi-integral over a sub-segment is well-approximated by the trapezoidal rule, but for clarity we'll implement a simple analytic form valid for `w < 0.5` (which holds for the test ladder where `w = 0.05`).

Add to the script:

```python
def phi(x: float, w: float) -> float:
    """Tract-placement density (simulator's phi). x in [0, 1] is the
    relative position in the inversion."""
    if w <= 0:
        return 1.0
    return math.sinh(w * x) * math.sinh(w * (1.0 - x)) / math.sinh(w)


def sample_x_event(
    segments: list[tuple[float, float]],
    bp_left: float, bp_right: float, w: float,
    rng: np.random.Generator,
) -> float:
    """Sample x_event from segments weighted by phi(x, w)."""
    inv_len = bp_right - bp_left
    # Numeric integration: split each segment into 64 bins, compute phi at
    # midpoint, weight by bin length. Accurate to ~1% for w ≤ 0.5.
    candidates: list[tuple[float, float]] = []  # (x, weight)
    for (l, r) in segments:
        l_clip = max(l, bp_left)
        r_clip = min(r, bp_right)
        if r_clip <= l_clip:
            continue
        n_bins = 64
        bin_len = (r_clip - l_clip) / n_bins
        for i in range(n_bins):
            x_mid = l_clip + (i + 0.5) * bin_len
            x_rel = (x_mid - bp_left) / inv_len
            wgt = phi(x_rel, w) * bin_len
            candidates.append((x_mid, wgt))
    if not candidates:
        return bp_left
    weights = np.array([w for (_, w) in candidates])
    weights /= weights.sum()
    idx = rng.choice(len(candidates), p=weights)
    return candidates[idx][0]
```

No test of this directly (it's tested indirectly via Step 6 below).

- [ ] **Step 5: Implement the per-rep simulation**

```python
def simulate_one_rep(
    params: Params, t_max: float, rng: np.random.Generator,
) -> tuple[float, bool]:
    """Simulate one rep of the backward-time flux process on the sample's
    ancestral lineage at x_c. Returns (time_of_first_flip, ever_flipped).
    If never flipped, time_of_first_flip = t_max and ever_flipped = False.
    """
    segments: list[tuple[float, float]] = [(params.bp_left, params.bp_right)]
    s = 0.0
    w = params.lam / params.L
    while True:
        coverage = total_coverage(segments)
        if coverage <= 0.0:
            return (t_max, False)
        rate = params.gamma * params.p_other * params.lam * (coverage / params.L)
        if rate <= 0.0:
            return (t_max, False)
        dt = rng.exponential(1.0 / rate)
        s += dt
        if s > t_max:
            return (t_max, False)
        # Sample x_event and tract.
        x_event = sample_x_event(segments, params.bp_left, params.bp_right, w, rng)
        T = rng.exponential(params.lam)
        T = min(T, 0.99 * params.L)
        x_event_rel = x_event - params.bp_left
        b1_lo = max(0.0, x_event_rel - T)
        b1_hi = min(params.L - T, x_event_rel)
        if b1_hi <= b1_lo:
            b1_rel = max(b1_lo, min(b1_hi, x_event_rel - T / 2.0))
        else:
            b1_rel = rng.uniform(b1_lo, b1_hi)
        tract_l = params.bp_left + b1_rel
        tract_r = tract_l + T
        # Determine if x_c is in tract AND in segments.
        if tract_l <= params.x_c < tract_r and position_in_segments(params.x_c, segments):
            return (s, True)  # flip event — sample's class flipped at x_c
        # Outside event: remove the tract region from segments.
        segments = subtract_interval(segments, tract_l, tract_r)


def run_mc(
    params: Params, t_max: float, n_reps: int, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (flip_times, ever_flipped) arrays of shape (n_reps,)."""
    rng = np.random.default_rng(seed)
    flip_times = np.zeros(n_reps)
    ever = np.zeros(n_reps, dtype=bool)
    for i in range(n_reps):
        ft, fl = simulate_one_rep(params, t_max, rng)
        flip_times[i] = ft
        ever[i] = fl
    return flip_times, ever


def empirical_f_at_grid(
    flip_times: np.ndarray, ever: np.ndarray, t_grid: np.ndarray,
) -> np.ndarray:
    """f̂(t_k) = fraction of reps that flipped before t_k."""
    out = np.zeros_like(t_grid)
    for k, t in enumerate(t_grid):
        out[k] = float(np.mean(ever & (flip_times <= t)))
    return out
```

- [ ] **Step 6: Add a sanity-check vs Andolfatto in the small-t regime**

At very small `t` (where `r_A · t ≪ 1`), both formulas agree (corrected → Andolfatto to leading order). Empirical f̂ should match both within MC error.

```python
def _sanity_check_small_t() -> None:
    """At small t, f̂ ≈ r_A · t ≈ f_A ≈ f_corrected within MC error."""
    p = Params(gamma=1.5e-5, lam=300.0, L=6000.0, p_other=0.5)
    t_max = 100.0  # r_A · t = 0.01 — very small
    n_reps = 100_000
    _, ever = run_mc(p, t_max, n_reps, seed=42)
    f_emp = float(np.mean(ever))
    f_pred = f_andolfatto(t_max, p)
    se = math.sqrt(f_pred * (1 - f_pred) / n_reps)
    assert abs(f_emp - f_pred) < 4 * se, (
        f"f̂={f_emp:.5f} vs predicted {f_pred:.5f} (4σ_MC = {4*se:.5f})"
    )
    print(f"sanity: f̂(t={t_max})={f_emp:.5f} vs Andolfatto {f_pred:.5f} ✓")
```

Update `_smoke()` to call this. Run: `.venv/bin/python scripts/validate_andolfatto_fragmentation.py`
Expected: previous OKs plus `sanity: f̂(t=100)=0.001XX vs Andolfatto 0.001125 ✓`. Should take ~5–15 seconds.

- [ ] **Step 7: Commit**

```bash
cd /home/ssmall/inversion_sims/files
git add scripts/validate_andolfatto_fragmentation.py
git commit -m "$(cat <<'EOF'
theory: MC validation tool for Andolfatto fragmentation correction

Self-contained Python script that simulates the backward-time flux
process on a single sample's ancestral lineage at x_c, exactly matching
the simulator's per-event mechanics. Reports empirical f̂(t) at a grid
of t values for comparison to f_A and f_corrected (eq 9 of
docs/theory/2026-04-29-andolfatto-fragmentation-correction.md).

Sanity check: at small t (r_A·t ≪ 1) the MC matches Andolfatto within
4σ MC error. Full validation against the corrected formula across the
test ladder is run in the next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Run the validation against the test ladder

**Files:**
- Modify: `scripts/validate_andolfatto_fragmentation.py` — add a `main()` entrypoint with CLI args.
- Output (not committed): `.tmp/andolfatto_validation.txt` — the residuals report.

### Step-by-step

- [ ] **Step 1: Add CLI entrypoint**

Append to `scripts/validate_andolfatto_fragmentation.py`:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--gamma", type=float, default=1.5e-5)
    parser.add_argument("--lam", type=float, default=300.0)
    parser.add_argument("--L", type=float, default=6000.0)
    parser.add_argument("--p-other", type=float, default=0.5)
    parser.add_argument("--n-reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ladder", type=str, default="1000,4000,10000,25000",
                        help="Comma-separated t values to evaluate.")
    parser.add_argument("--smoke", action="store_true",
                        help="Run only the smoke + sanity checks and exit.")
    args = parser.parse_args()

    if args.smoke:
        _test_segments()
        _smoke()
        _sanity_check_small_t()
        return

    p = Params(gamma=args.gamma, lam=args.lam, L=args.L, p_other=args.p_other)
    ladder = sorted(float(t) for t in args.ladder.split(","))
    t_max = max(ladder)
    flip_times, ever = run_mc(p, t_max, args.n_reps, seed=args.seed)

    print(f"# Validation report")
    print(f"#   γ={p.gamma}, λ={p.lam}, L={p.L}, p_other={p.p_other}")
    print(f"#   r_A={p.r_A:.4e} per gen, n_reps={args.n_reps}")
    print(f"# {'t':>8} {'f_emp':>8} {'f_A':>8} {'f_corr':>8} "
          f"{'|f_emp-f_A|':>12} {'|f_emp-f_corr|':>14} {'4σ_MC':>8}")
    grid = np.array(ladder)
    f_emp = empirical_f_at_grid(flip_times, ever, grid)
    for t, fe in zip(grid, f_emp):
        fa = f_andolfatto(float(t), p)
        fc = f_corrected(float(t), p)
        se = math.sqrt(max(fe, 1e-9) * (1 - fe) / args.n_reps)
        print(f"  {t:8.0f} {fe:8.4f} {fa:8.4f} {fc:8.4f} "
              f"{abs(fe-fa):12.4f} {abs(fe-fc):14.4f} {4*se:8.4f}")


if __name__ == "__main__":
    if "--smoke" in __import__("sys").argv or len(__import__("sys").argv) == 1:
        # Bare invocation runs smoke checks; explicit --smoke also.
        if len(__import__("sys").argv) == 1:
            _test_segments()
            _smoke()
            _sanity_check_small_t()
        else:
            main()
    else:
        main()
```

(Replace the bare `if __name__ == "__main__": _smoke()` from Task 1 with this dispatcher.)

- [ ] **Step 2: Run the validation**

```bash
cd /home/ssmall/inversion_sims/files
mkdir -p .tmp
.venv/bin/python scripts/validate_andolfatto_fragmentation.py \
    --gamma 1.5e-5 --lam 300 --L 6000 --p-other 0.5 \
    --n-reps 50000 --seed 0 \
    --ladder 1000,4000,10000,25000 \
    > .tmp/andolfatto_validation.txt
cat .tmp/andolfatto_validation.txt
```

Expected runtime: ~30–90 seconds. Output is a table comparing f̂ to both formulas.

- [ ] **Step 3: Read the residuals**

Read `.tmp/andolfatto_validation.txt`. For each `t` in the ladder, you should see four numbers:
- `f_emp`: the MC estimate, with 4σ error bars roughly `4·sqrt(f_emp·(1-f_emp)/50000) ≈ 0.009` for `f_emp = 0.5`.
- `f_A`: the Andolfatto prediction.
- `f_corr`: the corrected prediction (eq 9).
- `|f_emp - f_A|` and `|f_emp - f_corr|`: absolute deviations.

**Three possible outcomes:**

(a) **`|f_emp - f_corr| < 4σ_MC` at every rung**: the corrected formula is validated. Proceed to Task 3.

(b) **`|f_emp - f_corr|` exceeds `4σ_MC` at one or more rungs but `f_corr` is closer than `f_A`**: the corrected formula is an improvement but not exact. Document the residual; do NOT add a tightened test (the formula isn't ready). Proceed to Task 4 (document findings).

(c) **`|f_emp - f_A|` is smaller than `|f_emp - f_corr|`**: the corrected formula is *worse* than Andolfatto. The mean-field derivation has a sign error or wrong dependency. Stop and escalate to user — derivation needs rework.

- [ ] **Step 4: Commit the validation tool extension**

```bash
git add scripts/validate_andolfatto_fragmentation.py
git commit -m "$(cat <<'EOF'
theory: CLI entrypoint for the Andolfatto MC validator

Adds a main() with arg parsing for γ, λ, L, p_other, ladder, n_reps,
seed. Default ladder matches the existing Tier-3 test
(t_inv ∈ {1000, 4000, 10000, 25000}). Outputs a residuals table
suitable for human review.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

> **Note**: do NOT commit `.tmp/andolfatto_validation.txt` — it's scratch output per CLAUDE.md.

---

## Task 3: Research-decision point — report findings to user

This is NOT a coding task. The implementer reads the validation output and reports back to the user, who decides what to do next.

- [ ] **Step 1: Summarize findings**

Write a brief summary (2–3 sentences) in your report-back to the user:

```
Validation results (50,000 reps, parameters γ=1.5e-5, λ=300, L=6000, p_other=0.5):

  t      f_emp     f_A      f_corr    |f_emp-f_A|   |f_emp-f_corr|
  1000   <fill>    0.106    <fill>    <fill>        <fill>
  4000   <fill>    0.362    <fill>    <fill>        <fill>
  10000  <fill>    0.675    <fill>    <fill>        <fill>
  25000  <fill>    0.940    <fill>    <fill>        <fill>

Outcome: (a) corrected formula validated within MC error
       OR (b) corrected closer than Andolfatto but not exact
       OR (c) corrected worse than Andolfatto

Recommended next step:
  (a) → Task 4 (tighten test).
  (b) → Task 5 (document residual; queue coalescence/second-moment refinement).
  (c) → escalate; redo derivation.
```

- [ ] **Step 2: Wait for user direction**

Do NOT proceed to Task 4 or Task 5 without explicit user choice. The decision affects whether the test gets tightened (which would lock in the formula prematurely if it's wrong).

---

## Task 4: Tighten the existing Tier-3 test (run only if validation outcome (a))

**Files:**
- Modify: `tests/hull/test_phase3b_b2_flux.py`

### Step-by-step

- [ ] **Step 1: Add the new test alongside the existing loose one**

Open `tests/hull/test_phase3b_b2_flux.py`. After the existing
`test_andolfatto_sample_fraction_matches_closed_form` (around line 455),
add:

```python
def test_andolfatto_sample_fraction_matches_corrected_form():
    """Tighter Tier-3 anchor: empirical f̂(t) tracks the fragmentation-
    corrected closed form (1 - exp(-r_A·t)) / (2 - exp(-r_A·t)) within
    ±5% relative tolerance per point.

    The corrected form is derived in
    docs/theory/2026-04-29-andolfatto-fragmentation-correction.md
    (eq 9), validated by scripts/validate_andolfatto_fragmentation.py
    against an independent Monte Carlo of the simulator's flux
    mechanics.
    """
    from msinv.hull._event_log import filter_flux, samples_converted_at

    gamma = 1.5e-5
    lam = 300.0
    bp_left, bp_right = 2000.0, 8000.0
    L = bp_right - bp_left
    Ne, p_inv = 1000, 0.5
    p_other = 1.0 - p_inv
    inv_center = 0.5 * (bp_left + bp_right)
    n_seeds = 30
    t_inv_ladder = [1000.0, 4000.0, 10_000.0, 25_000.0]
    r_A = gamma * p_other * (lam ** 2) / L

    for t_inv in t_inv_ladder:
        f_emp = []
        for seed in range(n_seeds):
            ts, log = _run_tier3_sim(t_inv=t_inv, gamma=gamma, seed=seed,
                                     bp_left=bp_left, bp_right=bp_right,
                                     lam=lam, p_inv=p_inv, Ne=Ne)
            flux = filter_flux(log, inv_id=0)
            f_emp.append(samples_converted_at(flux, ts, inv_center))
        f_hat = float(np.mean(f_emp))
        e = math.exp(-r_A * t_inv)
        f_pred = (1.0 - e) / (2.0 - e)
        tol = max(0.05, 0.05 * f_pred)
        assert abs(f_hat - f_pred) < tol, (
            f"t_inv={t_inv}: f̂={f_hat:.3f} vs corrected predicted "
            f"{f_pred:.3f} (tol={tol:.3f}, n_seeds={n_seeds}). "
            f"Spec: docs/theory/2026-04-29-andolfatto-fragmentation-"
            f"correction.md.")
```

- [ ] **Step 2: Run the new test**

```bash
.venv/bin/python -m pytest tests/hull/test_phase3b_b2_flux.py::test_andolfatto_sample_fraction_matches_corrected_form -v --timeout=600
```

Expected: PASS.

- [ ] **Step 3: Run the full Tier-3 suite (verify no regression in the existing loose test)**

```bash
.venv/bin/python -m pytest tests/hull/test_phase3b_b2_flux.py -v --timeout=600
```

Expected: all existing tests still pass; new test passes.

- [ ] **Step 4: Commit**

```bash
git add tests/hull/test_phase3b_b2_flux.py
git commit -m "$(cat <<'EOF'
theory: add test_andolfatto_sample_fraction_matches_corrected_form

Tighter Tier-3 anchor using the fragmentation-corrected closed form
(1 - exp(-r_A·t)) / (2 - exp(-r_A·t)) at ±5% relative tolerance per
point. Validated by scripts/validate_andolfatto_fragmentation.py
against independent MC; spec at
docs/theory/2026-04-29-andolfatto-fragmentation-correction.md.

The existing loose-tolerance test is kept as a smoke check; this
test catches small regressions that the loose tolerance can't.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Document validation findings in the spec (run regardless of outcome)

**Files:**
- Modify: `docs/theory/2026-04-29-andolfatto-fragmentation-correction.md`

### Step-by-step

- [ ] **Step 1: Replace the "Status" note at the top**

Find the "**Status:** Draft 2026-04-29..." paragraph and replace with one of three forms based on outcome:

For outcome (a):
```
**Status:** Validated 2026-04-29. Formula in §3.4 matches MC within
4σ_MC across the test ladder; tightened test at
tests/hull/test_phase3b_b2_flux.py::test_andolfatto_sample_fraction_matches_corrected_form
asserts ±5% relative tolerance.
```

For outcome (b):
```
**Status:** Partial validation 2026-04-29. Formula in §3.4 is closer
to MC than Andolfatto across the test ladder, but exceeds 4σ_MC at the
larger t_inv rungs (residual roughly +0.X at t_inv=25000). The
flux-only mean-field misses coalescence-driven coverage restoration
(see §4.2). No tightened test added pending further refinement.
```

For outcome (c):
```
**Status:** Refuted 2026-04-29. Formula in §3.4 is worse than
Andolfatto in the partial-tract regime. The mean-field ODE in §3.2 has
a sign error or wrong dependency that needs rework. See §8 for MC
numbers.
```

- [ ] **Step 2: Add §8 with concrete numbers**

Append to the end of the spec:

```markdown
## §8. Validation results (2026-04-29)

MC: `scripts/validate_andolfatto_fragmentation.py --gamma 1.5e-5
--lam 300 --L 6000 --p-other 0.5 --n-reps 50000 --seed 0 --ladder
1000,4000,10000,25000`. Output (formatted):

| t     | f̂      | f_A    | f_corr | residual_A | residual_corr | 4σ_MC |
|-------|--------|--------|--------|------------|---------------|-------|
| 1000  | <fill> | 0.1064 | <fill> | <fill>     | <fill>        | <fill>|
| 4000  | <fill> | 0.362  | <fill> | <fill>     | <fill>        | <fill>|
| 10000 | <fill> | 0.675  | <fill> | <fill>     | <fill>        | <fill>|
| 25000 | <fill> | 0.940  | <fill> | <fill>     | <fill>        | <fill>|

Conclusion: <one sentence per outcome>.
```

(Fill in the table from `.tmp/andolfatto_validation.txt`.)

- [ ] **Step 3: Commit**

```bash
git add docs/theory/2026-04-29-andolfatto-fragmentation-correction.md
git commit -m "$(cat <<'EOF'
theory: record Andolfatto fragmentation MC validation results

Records the MC numbers from
scripts/validate_andolfatto_fragmentation.py against the test ladder
(γ=1.5e-5, λ=300, L=6000, p_other=0.5, n_reps=50000). Updates the
status flag at the top of the spec to reflect the validation outcome.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:** §1 (notation) + §3 (derivation) → Task 1 implements the model the formula describes. §5 (MC validation plan) → Task 1 + 2. §6 (tightened test plan) → Task 4 (conditional). §4 (open questions on coalescence + second-moment) → handled by the research-decision point in Task 3 — the plan does NOT silently force-fit the formula; if MC says it's wrong, Task 4 doesn't run.

**Placeholder scan:** "<fill>" markers in Task 5 are intentional — they're filled in by reading the actual MC output, not pre-committed in the plan. No "TBD" / "implement later" references.

**Type consistency:** `Params`, `f_andolfatto`, `f_corrected`, `total_coverage`, `position_in_segments`, `subtract_interval`, `intersect_interval`, `phi`, `sample_x_event`, `simulate_one_rep`, `run_mc`, `empirical_f_at_grid`, `main` — all defined in Task 1, used consistently in Task 2.

**Risk callout:** the MC's per-event mechanics (Step 5 of Task 1) match the simulator's `sample_flux_position` + `draw_tract` faithfully, BUT only for the single-lineage case. The simulator handles many lineages with coalescence and recombination interleaved; this MC strips that out to isolate the fragmentation mechanism. If the formula matches the MC but disagrees with the simulator (Tier-3 test), the residual is from coalescence (§4.2 of spec), not from the §3 derivation. Task 3's research-decision point treats this case as outcome (b).

**Scope check:** focused on a single goal (validate a single formula). Single implementation plan is appropriate.
