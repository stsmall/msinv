"""Helpers for building precomputed inversion-frequency trajectories.

Workflow target: the Kir/Fol "K-fixed-S vs F-polymorphic" study.  We
want a single InversionSpec whose per-population trajectory has:

  - Pre-split (t > t_split, deep history): one shared ancestral curve.
  - At t_split: bifurcation — K and F take different post-split paths.
  - K's tail: fixation (→ p_K_today, typically 0).
  - F's tail: drift / mild selection (→ p_F_today, typically polymorphic).

Build the three pieces and stitch into one PrecomputedTrajectory.

Functions
---------
- ``deep_neutral_curve(p_split, n_e, t_inv, t_split, seed)``: one-pop
  WF backward from ``p_split`` at ``t_split`` toward ``1/(2 n_e)`` at
  ``t_inv``.  Returns ``(times, freqs)`` covering ``[t_split, t_inv]``.
- ``post_split_logistic(p_today, p_split, t_split, n_e)``: deterministic
  logistic curve ``p_today → p_split`` over ``[0, t_split]`` with the
  selection coefficient implied by the two endpoints.
- ``post_split_neutral_walk(p_today, p_split, t_split, n_e, seed)``:
  stochastic neutral WF curve ``p_today → p_split`` over
  ``[0, t_split]`` (rejection-resampled to land near ``p_split``).
- ``bifurcate(deep, k_tail, f_tail, n_e_pops, t_inv)``: stitch the deep
  shared curve and per-ecotype tails into one trajectory dict ready
  for ``InversionSpec(trajectory=...)``.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def deep_neutral_curve(
    p_split: float,
    n_e: float,
    t_inv: float,
    t_split: float,
    seed: int = 0,
    max_attempts: int = 100,
) -> Tuple[List[float], List[float]]:
    """Backward neutral WF from p_split at t=t_split toward 1/(2N) at
    t=t_inv.  Returns ``(times, freqs)`` with times in
    ``[t_split, t_inv]`` and freqs ending at the founding-allele
    floor.  Reseeds and retries up to ``max_attempts`` if the path
    doesn't reach the floor by t=t_inv.

    Caveat: at large N (≳ 10^5), neutral drift backward from
    moderate p_split rarely reaches 1/(2N) inside any tractable
    window.  See project memory ``project_trajectory_port.md`` for
    the discussion of stochastic-bridge limitations.  In that regime
    this function returns a path whose terminal value is whatever it
    reached at t_inv (not 1/(2N)).  Caller can detect this via the
    returned final freq.
    """
    rng = np.random.default_rng(seed)
    p0 = 1.0 / (2.0 * n_e)
    duration = float(t_inv - t_split)
    if duration <= 0:
        raise ValueError("t_inv must be > t_split")

    p = float(p_split)
    times = [float(t_split)]
    freqs = [p]
    t_back = float(t_split)
    while t_back < t_inv and p > p0:
        # Per-gen WF SD (continuous diffusion approx).
        sd = math.sqrt(max(0.0, p * (1.0 - p) / (2.0 * n_e)))
        dp = float(rng.normal(0.0, sd)) if sd > 0 else 0.0
        p_new = p + dp
        # Reflect off boundaries.
        if p_new <= 0.0:
            p_new = abs(p_new) + p0
        elif p_new >= 1.0:
            p_new = 2.0 - p_new
        p = max(p0, min(1.0 - p0, p_new))
        t_back += 1.0
        times.append(t_back)
        freqs.append(p)
    return times, freqs


def post_split_logistic(
    p_today: float,
    p_split: float,
    t_split: float,
    n_e: float,
    n_samples: int = 200,
) -> Tuple[List[float], List[float]]:
    """Deterministic logistic from p_today (at t=0) to p_split
    (at t=t_split) over [0, t_split].  Returns ``(times, freqs)``.

    The selection coefficient is implied by the two endpoints:
        s = ln((p_today/(1-p_today)) / (p_split/(1-p_split))) / t_split
    (positive s ⇒ p rose forward ⇒ S being selected against in K
     forward = I being lost forward = p_today < p_split ... or vice
     versa depending on the direction).

    Edge cases:
    - If ``p_today`` ≈ 0 (K fixed-S): logistic blows up.  We clamp
      to ``1/(2 n_e)`` so the curve has a tractable shape.
    - If ``p_today`` ≈ ``p_split``: no selection needed; flat curve.
    """
    p0 = 1.0 / (2.0 * n_e)
    p_t = max(p0, min(1.0 - p0, p_today))
    p_s = max(p0, min(1.0 - p0, p_split))
    if abs(p_t - p_s) < 1e-9 or t_split <= 0:
        # Flat curve.
        return ([0.0, t_split], [p_t, p_t])
    s = (math.log(p_t / (1 - p_t)) - math.log(p_s / (1 - p_s))) / t_split
    # Logistic: p(t_back) = p_t * exp(-s*t_back) / (1 - p_t + p_t*exp(-s*t_back))
    # Then sample uniformly on [0, t_split].
    times = np.linspace(0.0, t_split, n_samples).tolist()
    freqs = []
    for t in times:
        e = math.exp(-s * t)
        p = p_t * e / (1 - p_t + p_t * e)
        freqs.append(max(p0, min(1.0 - p0, p)))
    return times, freqs


def post_split_neutral_walk(
    p_today: float,
    p_split: float,
    t_split: float,
    n_e: float,
    seed: int = 0,
    tolerance: float = 0.05,
    max_attempts: int = 1000,
) -> Tuple[List[float], List[float]]:
    """Stochastic neutral WF from ``p_today`` (at t=0) to ``p_split``
    (at t=t_split) over [0, t_split].  Rejection-sampled — runs
    backward WF and accepts paths whose terminal freq is within
    ``tolerance`` of ``p_split``.  Falls back to logistic if no
    attempt succeeds (logged via the returned tuple).

    Suitable for moderate N (≲ 10^5) and moderate divergence between
    p_today and p_split.  At Anopheles scale acceptance is poor;
    use ``post_split_logistic`` instead.
    """
    rng = np.random.default_rng(seed)
    p0 = 1.0 / (2.0 * n_e)
    for _attempt in range(max_attempts):
        local = np.random.default_rng(rng.integers(0, 2**63))
        p = float(p_today)
        times = [0.0]
        freqs = [p]
        t = 0.0
        while t < t_split:
            sd = math.sqrt(max(0.0, p * (1.0 - p) / (2.0 * n_e)))
            dp = float(local.normal(0.0, sd)) if sd > 0 else 0.0
            p_new = p + dp
            if p_new <= 0.0:
                p_new = abs(p_new) + p0
            if p_new >= 1.0:
                p_new = 2.0 - p_new
            p = max(p0, min(1.0 - p0, p_new))
            t += 1.0
            times.append(t)
            freqs.append(p)
        if abs(p - p_split) < tolerance:
            return times, freqs
    # Fallback to deterministic logistic.
    return post_split_logistic(p_today, p_split, t_split, n_e)


def bifurcate(
    deep_curve: Tuple[List[float], List[float]],
    k_tail: Tuple[List[float], List[float]],
    f_tail: Tuple[List[float], List[float]],
    n_e_pops: List[float],
    t_inv: float,
    t_split: float,
) -> dict:
    """Stitch (k_tail, f_tail, deep_curve) into one PrecomputedTrajectory
    dict ready for ``InversionSpec(trajectory=...)``.

    Inputs
    ------
    deep_curve : (times, freqs) over [t_split, t_inv].  freqs is a
        single shared trajectory for both populations.
    k_tail : (times, freqs) over [0, t_split].  K-pop's post-split
        curve ending at K's present-day freq.
    f_tail : (times, freqs) over [0, t_split].  F-pop's post-split
        curve ending at F's present-day freq.
    n_e_pops : [N_K, N_F] effective sizes.  Used by the simulator's
        founder-floor t_inv detection.
    t_inv : barrier dissolution time (where deep_curve ends).
    t_split : ecotype split time (where the tails meet the deep curve).

    Output
    ------
    A dict with shape:
        {'type': 'precomputed', 'times': [...], 'freqs': [k, f],
         'n_e': [N_K, N_F], 't_inv': [t_inv, t_inv]}
    Both per-pop arrays share the same ``times`` axis.  In the deep
    region (t > t_split) both curves equal the shared deep_curve.
    """
    k_t, k_f = k_tail
    f_t, f_f = f_tail
    deep_t, deep_f = deep_curve
    if k_t[0] != 0.0 or f_t[0] != 0.0:
        raise ValueError("k_tail and f_tail must start at t=0")
    if abs(k_t[-1] - t_split) > 1e-3 or abs(f_t[-1] - t_split) > 1e-3:
        raise ValueError("k_tail and f_tail must end at t_split")
    if abs(deep_t[0] - t_split) > 1e-3:
        raise ValueError("deep_curve must start at t_split")
    if abs(deep_t[-1] - t_inv) > 1e-3:
        raise ValueError("deep_curve must end at t_inv")
    # Build a single combined times axis.  Use the union of all times,
    # sorted.  Then resample each per-pop curve onto that axis.
    all_times = sorted(set(k_t) | set(f_t) | set(deep_t))
    k_arr = np.interp(all_times, k_t + deep_t[1:], list(k_f) + list(deep_f[1:]))
    f_arr = np.interp(all_times, f_t + deep_t[1:], list(f_f) + list(deep_f[1:]))
    return {
        'type': 'precomputed',
        'times': list(map(float, all_times)),
        'freqs': [list(map(float, k_arr)), list(map(float, f_arr))],
        'n_e':   list(map(float, n_e_pops)),
        't_inv': [float(t_inv), float(t_inv)],
    }


def deterministic_logistic_curve(
    p_start: float,
    p_final: float,
    s: float,
    t_inv: float,
    n_samples: int = 200,
) -> Tuple[List[float], List[float]]:
    """Deterministic logistic going backward from ``p_final`` (at t=0)
    to ``p_start`` (at t=t_inv).  The selection coefficient ``s`` should
    satisfy s · t_inv ≈ logit(p_final) - logit(p_start) (so the
    trajectory cleanly hits both endpoints).

    Returns ``(times, freqs)`` covering [0, t_inv].
    """
    times = np.linspace(0.0, float(t_inv), n_samples).tolist()
    freqs = []
    for t in times:
        t_fwd = float(t_inv) - t
        if s <= 0.0 or t_fwd <= 0.0:
            freqs.append(p_start if t_fwd <= 0.0 else p_final)
            continue
        e = math.exp(s * t_fwd)
        p = p_start * e / (1.0 - p_start + p_start * e)
        freqs.append(min(p_final, max(p_start, p)))
    return times, freqs


def kir_fol_softsweep_trajectory(
    n_e_anc: float = 450_000.0,
    n_e_K: float   = 126_772.0,
    n_e_F: float   = 2_496_632.0,
    p_F_today: float = 0.734,
    p_start: float = 0.05,
    t_split: float = 9_194.0,
    t_inv: float   = 330_000.0,
) -> dict:
    """Per-pop bifurcated soft-sweep trajectory.

    K's curve has two regimes:
    - [0, t_split]: p_inv[K] = 0 (no class structure in K — K-S coal
      is panmictic, F-I lineages still in F can coalesce among
      themselves with F's own p_inv).
    - [t_split, t_inv]: p_inv[K] = F's curve.  After ej(F→K) at
      t_split, the merged pop (which the simulator calls pop 0 = K)
      needs to see the deep-ancestor inversion frequency, otherwise
      F-I lineages migrating into K get stranded with p_class=0.

    F's curve: deterministic logistic from ``p_F_today`` (today) back
    to ``p_start`` at t_inv.

    The selection coefficient is implied by the endpoints:
        s = (logit(p_F_today) - logit(p_start)) / t_inv

    With p_start=0.05 and standard Anopheles params, s ≈ 1.2e-5 — mild
    positive selection rolling the inversion from a 5% standing-variation
    soft-sweep founder to ~73% present-day frequency.

    Output: PrecomputedTrajectory dict with per-pop curves on a shared
    times axis.
    """
    s = (math.log(p_F_today / (1 - p_F_today))
         - math.log(p_start / (1 - p_start))) / float(t_inv)
    f_curve = deterministic_logistic_curve(
        p_start=p_start, p_final=p_F_today, s=s, t_inv=t_inv,
        n_samples=200)
    times = list(f_curve[0])
    f_freqs = list(f_curve[1])
    # K curve: 0 below t_split, then follows F above t_split.
    k_freqs = [
        0.0 if t < float(t_split) else f_freqs[i]
        for i, t in enumerate(times)
    ]
    # Make sure the t_split boundary point is sampled (anchors the
    # discontinuity from 0 to f(t_split) cleanly).
    if all(abs(t - float(t_split)) > 1.0 for t in times):
        # Insert a discrete jump at t_split.
        idx = next(i for i, t in enumerate(times) if t > float(t_split))
        times.insert(idx, float(t_split))
        # K jumps from 0 to F's value at t_split.
        f_at_split = float(np.interp(float(t_split), times, f_freqs + [f_freqs[-1]])
                           if False else
                           f_freqs[idx - 1] + (f_freqs[idx] - f_freqs[idx - 1])
                           * (float(t_split) - times[idx - 1])
                           / (times[idx + 1] - times[idx - 1]))
        f_freqs.insert(idx, f_at_split)
        k_freqs.insert(idx, f_at_split)  # K joins F's curve at t_split
    return {
        'type': 'precomputed',
        'times': list(map(float, times)),
        'freqs': [list(map(float, k_freqs)),
                  list(map(float, f_freqs))],
        'n_e':   [float(n_e_K), float(n_e_F)],
        't_inv': [float(t_inv), float(t_inv)],
    }


def kir_fol_drift_filter_trajectory(
    n_e_anc: float = 450_000.0,
    n_e_K: float   = 126_772.0,
    n_e_F: float   = 2_496_632.0,
    p_F_today: float = 0.734,
    p_K_today: float = 0.0,
    p_split: float = 0.5,
    t_split: float = 9_194.0,
    t_inv: float   = 330_000.0,
    seed: int = 0,
) -> dict:
    """Build a Kir/Fol bifurcated trajectory for the **drift / filter**
    scenario:

    - K's tail: flat at ``p_inv = 0`` over [0, t_split].  Drift-filter
      assumption: K is sampled S-only today, and the I lineages that
      may have existed in K's gene pool during the drift period have
      gone extinct in the genealogy of the sampled lineages.  Setting
      ``p_inv[K] = 0`` keeps K panmictic during the K-tail (rate =
      1/(2·Ne_K) for K-S pairs).  A nonzero K-tail would impose a
      structured-coalescent rate boost (rate ∝ 1/(p_std · Ne_K)) that
      double-counts the absent I lineages and inflates K's intra-S
      coalescence — observed in v0 of this helper as Fst K-FI ≈ 0.47.
    - F's tail: stochastic neutral walk from ``p_F_today`` (today) back
      to ``p_split`` (at t_split).
    - Deep ancestor: stochastic neutral walk from ``p_split`` back to
      ``1/(2 n_e_anc)`` at t_inv.

    At t = t_split the K-tail value (0) is replaced by the deep curve
    value (p_split).  This is biologically the moment K and F coalesce
    into the shared ancestor — K's "no-I" state lifts because we're
    now in F's polymorphic gene pool.

    Defaults match v11 frozen Kir/Fol parameters: K-F split 9194 g,
    3Ra age 330k g, F freq 0.734.
    """
    rng = np.random.default_rng(seed)
    deep = deep_neutral_curve(
        p_split=p_split, n_e=n_e_anc,
        t_inv=t_inv, t_split=t_split,
        seed=int(rng.integers(0, 2**32)))
    # K-tail flat at 0 — drift-filter, no class structure in K.
    k_tail = ([0.0, t_split], [0.0, 0.0])
    f_tail = post_split_neutral_walk(
        p_today=p_F_today, p_split=p_split,
        t_split=t_split, n_e=n_e_F,
        seed=int(rng.integers(0, 2**32)))
    return bifurcate(
        deep_curve=deep, k_tail=k_tail, f_tail=f_tail,
        n_e_pops=[n_e_K, n_e_F],
        t_inv=t_inv, t_split=t_split)
