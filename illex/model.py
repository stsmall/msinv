"""HullSimulator builders for the Illex arms.

Scaling rule: per-bp rates and Ne stay faithful; only the inversion is
shortened. That preserves per-site pi/dxy and r^2-vs-distance.

Flux geometry: phi() in rust/msinv-core/src/phi.rs works in
inversion-relative coordinates with w = mean_tract_length / inv_length, so the
flux profile is scale-invariant only if w is held fixed. Real w = 2 kb / 20 Mb
= 1e-4. Keeping a biological 2 kb tract at L = 30 kb would inflate interior
flux ~670x.
"""

from __future__ import annotations

from msinv import HullSimulator, InversionSpec

from .demography import (PRESENT_NE_CONST, PRESENT_NE_GROWTH,
                         constant_demography, growth_demography)

TRACT_FRACTION = 1e-4
MARGIN_FRACTION = 0.1        # collinear flank on each side of the inversion


def _arm_parts(arm: str):
    if arm == "growth":
        return growth_demography(), PRESENT_NE_GROWTH
    if arm == "constant":
        return constant_demography(), PRESENT_NE_CONST
    raise ValueError(f"arm must be 'growth' or 'constant', got {arm!r}")


def build_inversion_sim(*, arm, seq_length, t_inv, gamma, p_inv=0.626,
                        n_i=100, n_s=100, seed=None, recomb_rate=2.5e-9):
    demog, present_ne = _arm_parts(arm)
    margin = seq_length * MARGIN_FRACTION
    bp_left, bp_right = margin, seq_length - margin
    inv_len = bp_right - bp_left

    spec = InversionSpec(
        bp_left=bp_left,
        bp_right=bp_right,
        p_inv=p_inv,
        t_inv=t_inv,
        gene_conversion_rate=gamma,
        mean_tract_length=max(1.0, inv_len * TRACT_FRACTION),
        tract_distribution="geometric",
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
