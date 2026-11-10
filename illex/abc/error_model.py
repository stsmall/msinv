"""The data's genotype error model, reproduced on the simulated side for the LD statistics.

The chr2 callset is low depth: on the LD SNP set (MAF >= LD_MAF within a 50-individual subsample) roughly a
third of genotypes are missing, and heterozygotes are under-called (Ho/He ~ 0.31-0.36). Finite-sample r^2 bias is
additive in 1 / n_pair and depends on the genotype distribution, so dividing by the control does not cancel
either effect. Instead the simulation's random-pairing diploids are passed through the same error process before
r^2 is computed with the same pairwise-complete kernel (``statvec.ld_block_sums``):

  * a heterozygote is called homozygous with probability h, REF or ALT with equal probability
  * each genotype is independently missing with probability 1 - call_rate

``h = 1 - Ho/He`` is the miscall rate that makes the simulated Ho/He (HWE pairing, so 1 in expectation before
errors) equal the observed one. Both parameters are estimated per class (I = 2a = AA, S = 2+ = BB) and region
(body = the inversion interior, control = the chr2 10-30 Mb collinear control) by ``illex/abc/observed.py``
with ``genotype_error_stats`` over the same 20 LD subsamples, and recorded in abc_observed.json
meta["ld_error_model"]. The constants below are those values (a test checks they agree).
"""
from __future__ import annotations

import numpy as np

# region -> class -> (call_rate, h). From results/illex/abc_observed.json meta["ld_error_model"].
LD_ERROR = {
    "body": {"I": (0.6155, 0.6904), "S": (0.6741, 0.6757)},
    "control": {"I": (0.6240, 0.6656), "S": (0.6720, 0.6843)},   # abc_observed_ctl2.json (chr2 40-55 + 85-115 Mb)
}


def apply_genotype_errors(dosage, call_rate, h, rng):
    """Dosage 0/1/2 (any shape) -> float copy with het->hom miscalls (prob h, 50/50 REF/ALT) and NaN missing.

    Three uniform arrays are always drawn (miscall, direction, missing), so the rng stream does not depend on the
    parameters. With call_rate = 1 and h = 0 the values are returned unchanged.
    """
    d = np.array(dosage, dtype=np.float64)
    u_mis = rng.random(d.shape)
    u_dir = rng.random(d.shape)
    u_miss = rng.random(d.shape)
    flip = (d == 1.0) & (u_mis < h)
    d[flip] = np.where(u_dir[flip] < 0.5, 0.0, 2.0)
    d[u_miss >= call_rate] = np.nan
    return d


def genotype_error_stats(dosage):
    """(call_rate, Ho/He) of a (sites, individuals) dosage with NaN or -1 missing, pooled over sites.

    He per site is the unbiased 2pq * 2n / (2n - 1) from the called genotypes, so Ho/He is ~1 for HWE genotypes
    whatever n. Pooled as sum(Ho) / sum(He) over sites with >= 2 called genotypes.
    """
    d = np.asarray(dosage, dtype=np.float64)
    called = np.isfinite(d) & (d >= 0)
    nc = called.sum(1).astype(np.float64)
    x = np.where(called, d, 0.0)
    ok = nc >= 2
    p = x.sum(1)[ok] / (2.0 * nc[ok])
    n2 = 2.0 * nc[ok]
    he = (2.0 * p * (1.0 - p) * n2 / (n2 - 1.0) * nc[ok]).sum()
    ho = (called & (d == 1.0)).sum(1)[ok].sum()
    return float(called.mean()), float(ho / he) if he > 0 else float("nan")


def h_from_ho_he(ho_he):
    """The het->hom miscall probability that turns HWE genotypes into this Ho/He."""
    return float(min(max(1.0 - ho_he, 0.0), 1.0))
