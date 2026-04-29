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


# ---------------------------------------------------------------------------
# Segment helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# phi profile (simulator's Peischl triangular-roof form)
# phi(x, w) = min(x, 1-x, w) / (1 - w)  for x in [0, 1]
# ---------------------------------------------------------------------------

def phi(x: float, w: float) -> float:
    """Tract-placement density (simulator's phi, Peischl triangular-roof).

    x in [0, 1] is the relative position in the inversion.
    w = λ/L is the dimensionless tract-length ratio.
    """
    if x <= 0.0 or x >= 1.0:
        return 0.0
    if w >= 1.0:
        return 1.0
    val = min(x, 1.0 - x, w) / (1.0 - w)
    return max(0.0, min(1.0, val))


def sample_x_event(
    segments: list[tuple[float, float]],
    bp_left: float, bp_right: float, w: float,
    rng: np.random.Generator,
) -> float:
    """Sample x_event from segments weighted by phi(x, w).

    Uses numeric integration with 64 bins per segment — accurate to ~1% for
    w ≤ 0.5 (test ladder: w = 300/6000 = 0.05).
    """
    inv_len = bp_right - bp_left
    # candidates: list of (x_abs, weight)
    candidates: list[tuple[float, float]] = []
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
    weights = np.array([wt for (_, wt) in candidates])
    weights /= weights.sum()
    idx = rng.choice(len(candidates), p=weights)
    return candidates[idx][0]


# ---------------------------------------------------------------------------
# Per-rep simulation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Smoke + sanity checks
# ---------------------------------------------------------------------------

def _smoke() -> None:
    p = Params(gamma=1.5e-5, lam=300.0, L=6000.0, p_other=0.5)
    assert abs(p.r_A - 1.125e-4) < 1e-12
    assert f_andolfatto(0.0, p) == 0.0
    assert f_corrected(0.0, p) == 0.0
    # At r_A·t = ln(2), Andolfatto = 0.5; corrected has e^{-ln 2} = 0.5,
    # so corrected = 0.5/1.5 = 1/3.
    t_half = math.log(2.0) / p.r_A
    assert abs(f_andolfatto(t_half, p) - 0.5) < 1e-9
    assert abs(f_corrected(t_half, p) - 1.0 / 3.0) < 1e-9
    print("formulas: OK")


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


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

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
    import sys
    # Bare invocation (no args) → smoke checks; --smoke also → smoke checks;
    # any other args → full CLI.
    if len(sys.argv) == 1:
        _test_segments()
        _smoke()
        _sanity_check_small_t()
    else:
        main()
