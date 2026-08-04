"""HullSimulator builders for the Illex arms.

Scaling rule: per-bp rates and Ne stay faithful; only the inversion is
shortened. That preserves per-site pi/dxy and r^2-vs-distance.

Flux geometry: phi() in rust/msinv-core/src/phi.rs works in
inversion-relative coordinates with w = mean_tract_length / inv_length, so the
flux profile is scale-invariant only if w is held fixed. Real w = 2 kb / 20 Mb
= 1e-4. Keeping a biological 2 kb tract at L = 30 kb would inflate interior
flux ~670x.

Inversion-origin model (``p_start``)
-------------------------------------
Task 4's harness cross-validation found that the plain ``p_inv=``/``t_inv=``
constant-frequency path (msinv's ``ConstantTrajectory``) does *not* enforce
single-origin monophyly at ``t_inv`` -- it relabels I-class lineages to S at
the barrier and lets them keep coalescing under ordinary demography, so
E[T_I] can exceed t_inv. ``theory.py``'s forced-coalescence formula for
within_i models a different, stricter process (see
task-4-report.md). Neither is a bug; they are different models of what a
"single origin" inversion's early history looked like.

``build_inversion_sim``'s ``p_start`` parameter makes that choice a
continuum instead of a binary. It selects msinv's ``deterministic``
trajectory (closed-form logistic from a founding frequency ``p_start`` at
t_inv up to ``p_inv`` today):

- ``p_start=None`` (default): unchanged legacy behaviour -- constant p_inv
  from 0 to t_inv (the "soft"/no-monophyly limit). Do not change this path.
- ``p_start=1/(2*Ne)``: the single-founder / hard-sweep limit (one copy at
  t_inv, matching the monophyly assumption theory.py encodes for within_i).
- Any intermediate ``p_start`` interpolates between the two, and is a
  legitimate fitted parameter (see task-4-report.md's fix report) rather
  than an ad hoc knob.

The selection coefficient ``s`` implied by a target ``(p_start, p_inv,
t_inv)`` triple is solved analytically from msinv's own logistic trajectory
equation (``rust/msinv-core/src/trajectory.rs::DeterministicTrajectory``):
``s = [logit(p_inv) - logit(p_start)] / t_inv``. This exactly reproduces the
requested ``t_inv`` (msinv derives its own t_inv from the same equation, so
the two are consistent by construction as long as ``p_start`` isn't clamped
against the ``1/(2*Ne) <= p_start <= 1 - 1/(2*Ne)`` boundary).

On the deterministic path, ``n_e`` is used by msinv (see
``DeterministicTrajectory::new_with_p_start`` in
``rust/msinv-core/src/trajectory.rs``) for exactly one thing: clamping
``p_start`` to ``[1/(2*n_e), 1 - 1/(2*n_e)]`` (plus an ``s <= 0`` fallback
this module never reaches, since ``_derive_s`` only produces ``s <= 0`` if
``p_start >= p_final``, which callers should not request). For an
intermediate ``p_start`` (e.g. 0.15) no clamping occurs, so the exact value
of ``n_e`` is **inert** -- it doesn't matter which Ne is passed. But it is
*not* inert at the hard-sweep limit, where callers set
``p_start = 1/(2*n_e)`` themselves: there, ``n_e`` directly *defines* the
founding frequency (``1/(2*N_ANC) = 9.1e-7`` vs. ``1/(2*N0) = 7.3e-8`` differ
12x), so which Ne the growth arm uses is a real modeling choice, not a
free parameter.

The growth arm passes ``theory.N_ANC`` here. This is *not* because N_ANC is
"the size during the barrier era" -- that claim is false whenever
``t_inv < T_GROW`` (=769,519): at the current best-fit ``t_inv ~ 5e5``, the
entire barrier era falls inside the growth phase, where N(t) runs from
~1.32e6 up to N0=6.81e6, nowhere near N_ANC=547,928. N_ANC is simply the
one available fixed reference point on the growth arm's N(t) curve (the
pre-growth asymptote); using it is a documented, deliberately-visible
approximation of "some large-N scale", not a claim about the trajectory's
actual local Ne. Because n_e is inert away from the hard-sweep limit (see
above), this approximation only bites when someone actually fits
``p_start = 1/(2*Ne)`` for the growth arm -- flagged here so that choice
isn't silently baked in.

One further consequence: solving for ``s`` this way generally yields a
small positive selection coefficient (s on the order of 1e-5 for the anchor
scenario), so a fitted ``p_start != None`` arm is neutral *in form* (it's
still the neutral-sufficiency null being tested) but not perfectly neutral
in the literal sense of s=0 -- the ``s`` is a bookkeeping device for hitting
a target founding frequency by a target time, not a claim of real selection
on the inversion.
"""

from __future__ import annotations

import numpy as np
from msinv import HullSimulator, InversionSpec

from .demography import (PRESENT_NE_CONST, PRESENT_NE_GROWTH,
                         constant_demography, growth_demography)
from .theory import N_ANC

TRACT_FRACTION = 1e-4
MARGIN_FRACTION = 0.1        # collinear flank on each side of the inversion


def _arm_parts(arm: str):
    if arm == "growth":
        return growth_demography(), PRESENT_NE_GROWTH
    if arm == "constant":
        return constant_demography(), PRESENT_NE_CONST
    raise ValueError(f"arm must be 'growth' or 'constant', got {arm!r}")


def trajectory_ne(arm: str) -> float:
    """Scalar Ne fed to msinv's deterministic trajectory for ``arm``.

    Only matters at the hard-sweep limit (``p_start = 1/(2*n_e)``); for
    intermediate ``p_start`` this value only clamps the bound and is inert
    in practice. Growth arm: uses ``theory.N_ANC`` as a fixed reference
    point on the growth curve -- NOT because it's "the barrier-era size"
    (false whenever t_inv < T_GROW, which includes the current best-fit
    t_inv ~ 5e5). See module docstring for the full explanation. Constant
    arm: the (single, true) Ne throughout.
    """
    if arm == "growth":
        return N_ANC
    if arm == "constant":
        return PRESENT_NE_CONST
    raise ValueError(f"arm must be 'growth' or 'constant', got {arm!r}")


def _derive_s(p_final: float, p_start: float, t_inv: float) -> float:
    """Selection coefficient reproducing (p_start, p_final, t_inv).

    Solves msinv's own logistic trajectory equation for s (see
    ``DeterministicTrajectory::new_with_p_start`` in
    ``rust/msinv-core/src/trajectory.rs``):
    ``s * t_inv = logit(p_final) - logit(p_start)``.
    """
    logit_final = np.log(p_final / (1.0 - p_final))
    logit_start = np.log(p_start / (1.0 - p_start))
    return float((logit_final - logit_start) / t_inv)


def build_inversion_sim(*, arm, seq_length, t_inv, gamma, p_inv=0.626,
                        p_start=None,
                        n_i=100, n_s=100, seed=None, recomb_rate=2.5e-9):
    demog, present_ne = _arm_parts(arm)
    margin = seq_length * MARGIN_FRACTION
    bp_left, bp_right = margin, seq_length - margin
    inv_len = bp_right - bp_left
    tract_length = max(1.0, inv_len * TRACT_FRACTION)

    if p_start is None:
        # Legacy / soft limit: constant p_inv, no forced monophyly at t_inv.
        spec = InversionSpec(
            bp_left=bp_left,
            bp_right=bp_right,
            p_inv=p_inv,
            t_inv=t_inv,
            gene_conversion_rate=gamma,
            mean_tract_length=tract_length,
            tract_distribution="geometric",
        )
    else:
        # Deterministic-logistic trajectory from a founding frequency
        # p_start at t_inv up to p_inv today. p_start=1/(2*Ne) is the
        # single-founder limit; see trajectory_ne() and module docstring.
        ne_traj = trajectory_ne(arm)
        s = _derive_s(p_final=p_inv, p_start=p_start, t_inv=t_inv)
        spec = InversionSpec(
            bp_left=bp_left,
            bp_right=bp_right,
            gene_conversion_rate=gamma,
            mean_tract_length=tract_length,
            tract_distribution="geometric",
            trajectory={
                "type": "deterministic",
                "p_final": p_inv,
                "n_e": ne_traj,
                "s": s,
                "p_start": p_start,
            },
        )
    return HullSimulator(
        n_std=n_s, n_inv=n_i,
        population_size=present_ne,
        demography=demog,
        sequence_length=seq_length,
        recombination_rate=recomb_rate,
        inversions=[spec],
        seed=seed,
    )


def build_control_sim(*, arm, seq_length, n_i=100, n_s=100, seed=None,
                      recomb_rate=2.5e-9):
    """Collinear control: same rates, no inversion barrier.

    A degenerate 1 bp inversion keeps the karyotype labels (so the same
    statistics code applies) while imposing no meaningful barrier.
    """
    demog, present_ne = _arm_parts(arm)
    spec = InversionSpec(
        bp_left=1.0, bp_right=2.0,
        p_inv=0.626, t_inv=1.0e6,
        gene_conversion_rate=1e-15,
        mean_tract_length=1.0,
    )
    return HullSimulator(
        n_std=n_s, n_inv=n_i,
        population_size=present_ne,
        demography=demog,
        sequence_length=seq_length,
        recombination_rate=recomb_rate,
        inversions=[spec],
        seed=seed,
    )
