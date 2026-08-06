"""Fixed constants, ABC priors, and the summary-statistic contract.

Single source of truth for the pipeline: the SLiM driver, the observed-target
script, and the ABC fitter all import from here so a prior can never drift
between the simulations and the inference.

Provenance for every fixed value is in illex/NOTES_illex_biology.md.
"""
from __future__ import annotations

import numpy as np

# --- fixed demography (moments SFS fit; NOTES sec 2) ------------------------
N_ANC = 547_928.0
N_NOW = 6_808_096.0
T_GROW = 769_519.0
MU = 3e-9

# --- fixed recombination (ReLERNN; NOTES sec 8.1) ---------------------------
# Sex-averaged genome-wide length-weighted mean, male 2.148e-9 / female 2.892e-9.
# chr2 is deliberately absent from the existing ReLERNN run (the inversion's LD
# block would corrupt the fit), but the six length-matched autosomes span
# 2.467-2.594e-9, so r is effectively KNOWN for chr2 rather than a free
# parameter. Kept fixed; REC_RATE_BRACKET is the male/female sensitivity arm.
#
# NOTE the sweep recipes in 14_sweep_seqmodel default to 2.1e-9, which is the
# MALE map only. This pipeline uses the sex-averaged value; the difference is
# ~20% in rho and matters for the barrier's leakage scale.
REC_RATE = 2.52e-9
REC_RATE_BRACKET = (2.148e-9, 2.892e-9)

# A chr2-specific mask and ReLERNN map are being built. When they land, set these
# and rerun -- do NOT edit REC_RATE by hand, so the provenance stays visible.
# CHR2_RMAP is expected in ReLERNN PREDICT format (chrom/start/end/nSites/
# recombRate/CI95LO/CI95HI, with chrom possibly a b'2' bytes-repr).
CHR2_RMAP = None            # e.g. ".../chr2.kept.PREDICT.BSCORRECTED.txt"
CHR2_MASK_BED = None        # e.g. ".../chr2_accessible.bed"


def _load_chr2_rmap(rmap_path: str | None = None):
    import pandas as pd
    path = rmap_path or CHR2_RMAP
    if path is None:
        return None
    d = pd.read_csv(path, sep="\t")
    d["chrom"] = d["chrom"].astype(str).str.replace(r"^b'|'$", "", regex=True)
    d = d[d.chrom == "2"].copy()
    if d.empty:
        raise ValueError(f"no chr2 rows in {path}")
    return d


def _weighted_mean(d) -> float:
    w = (d.end - d.start).to_numpy(dtype=float)
    return float((d.recombRate.to_numpy(dtype=float) * w).sum() / w.sum())


def rec_rate_for_inversion(rmap_path: str | None = None,
                           margin: int = 2_000_000) -> float:
    """Meiotic recombination rate to give the SIMULATION.

    Taken from chr2's COLLINEAR regions, NOT from inside the inversion.

    This is the important subtlety in using a chr2 ReLERNN map. ReLERNN infers
    recombination from LD decay, and the inversion elevates LD across 60-80 Mb in
    heterokaryotypes, so the interior windows report a **downward-biased** rate:
    they measure the realized barrier, not the underlying meiotic rate. The SLiM
    model already imposes the barrier structurally, so feeding it the interior
    estimate would suppress recombination TWICE.

    (This is also why chr2 was excluded from the original genome-wide run --
    ``build_persex_vcf.sh``: "autosomes (excl chr2 inv, chr42, chrZ)".)

    ``margin`` excludes a buffer either side of the breakpoints, since LD spills
    beyond them and the differentiated extent is itself narrower than the nominal
    span (NOTES sec 4.2).
    """
    d = _load_chr2_rmap(rmap_path)
    if d is None:
        return REC_RATE
    coll = d[(d.end < INV_START_REAL - margin)
             | (d.start > INV_STOP_REAL + margin)]
    if coll.empty:
        raise ValueError(
            f"no collinear chr2 windows outside the inversion +/-{margin:,} bp; "
            "cannot derive an unbiased meiotic rate from this map")
    return _weighted_mean(coll)


def rec_rate_inversion_interior(rmap_path: str | None = None) -> float | None:
    """ReLERNN's rate INSIDE the inversion -- a validation target, not an input.

    Deliberately a separate function so it can never be mistaken for the
    simulation's ``r``. Its value is that the simulation PREDICTS it: a fitted
    (t_inv, p_start, s) with the barrier produces some realized LD inside the
    inversion, which maps to an apparent recombination rate. Comparing that to
    this number is an independent check on the barrier's strength -- and it is one
    of the few genuinely independent constraints left (NOTES sec 9), since Fst is
    algebraically redundant and absolute levels need a nuisance scale.

    Caveat that must travel with it: within an arrangement the local effective
    population size is also altered (pi_I/pi_S = 0.744), and ReLERNN's training
    assumes a genome-wide demography, so the interior estimate is confounded by
    Ne as well as by LD. Treat it as an order-of-magnitude check, not a precise
    target.
    """
    d = _load_chr2_rmap(rmap_path)
    if d is None:
        return None
    body = d[(d.start >= INV_START_REAL) & (d.end <= INV_STOP_REAL)]
    if body.empty:
        raise ValueError("no chr2 windows inside the inversion body")
    return _weighted_mean(body)

# --- observed inversion ------------------------------------------------------
P_INV_OBS = 0.626
INV_START_REAL = 60_040_617
INV_STOP_REAL = 79_995_597
INV_LEN_REAL = INV_STOP_REAL - INV_START_REAL

# --- accessibility (NOTES sec 8.2) ------------------------------------------
ACCESSIBLE_BED = ("/sietch_colab/data_share/illex/popgen_data/"
                  "degenotate_illex/accessible_sites.bed")
ACC_FRAC_INV = 0.4791
ACC_FRAC_CONTROL = 0.6069

# --- simulation geometry -----------------------------------------------------
# The inversion is simulated far shorter than 20 Mb. Licensed by the verified
# L-invariance of per-site pi and dxy (worst-case 2.1%/1.8% bias extrapolating
# to 20 Mb; NOTES sec 7.3). NOT licensed for r^2-vs-distance, which is why LD is
# excluded from the statistic vector below.
INV_LEN_SIM = 100_000
FLANK_LEN_SIM = 25_000
# w = tract/inv_len held fixed at the real 2kb/20Mb so flux geometry is
# scale-invariant.
TRACT_FRAC = 1e-4

N_HAP_I = 200                      # inverted haplotypes sampled
N_HAP_S = 200                      # standard haplotypes sampled
SFS_PROJ = 20                      # haploid size for the SFS projection
SFS_BINS = SFS_PROJ // 2           # folded bins retained

# Scaling factor. Cost knob; see README. Q-scaling requires s*Q << 1, enforced
# in the .slim script (aborts at s*Q >= 0.1).
Q_DEFAULT = 200

# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------
# t_inv: log-uniform. Lower bound below any plausible estimate, upper bound
#   above the 1.34e6 the coalescent work explored, so the posterior is not
#   truncated by the prior at either end.
# p_start: log-uniform from the single-founder limit 1/(2*N_ANC) up to 0.4.
#   Spans the whole origin continuum -- both extremes are already excluded by
#   the data (NOTES sec 7.1), so the prior must contain them to show that.
# s: log-uniform on [1e-7, 3e-4] with an atom at exactly 0. At Ne = 6.8e6,
#   Ne*s = 1 at s = 1.5e-7, so this spans "effectively neutral" to "strongly
#   selected". The atom at 0 makes P(neutral | data) directly estimable, which
#   is manuscript question 1.
# h: dominance. 0.5 = additive; h > 1 = OVERDOMINANCE, the classic mechanism
#   for a stable intermediate frequency, which is manuscript question 3.
# p_flux: per-gamete probability of a flux tract. Atom at 0 plus log-uniform,
#   so flux stays formally testable even though the spatial test already
#   falsified it (NOTES sec 6).
PRIOR_NEUTRAL_WEIGHT = 0.5         # P(s == 0) under the full model
PRIOR_NOFLUX_WEIGHT = 0.5          # P(p_flux == 0)

PRIORS = {
    "t_inv":   ("loguniform", 5.0e4, 3.0e6),
    "p_start": ("loguniform", 1.0 / (2.0 * N_ANC), 0.4),
    "s":       ("loguniform_atom0", 1.0e-7, 3.0e-4),
    "h":       ("uniform", 0.5, 3.0),
    "p_flux":  ("loguniform_atom0", 1.0e-6, 1.0e-2),
}

# Parameters written to the results table, in order.
PARAM_NAMES = ["t_inv", "p_start", "s", "h", "p_flux"]

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
# Chosen for identification, not completeness (NOTES sec 9):
#   - the two ratios are the primary targets and are calibration-free
#   - p_final carries the frequency information; it is NOT conditioned on in the
#     simulation, so ABC is free to use it
#   - the within-arrangement folded SFS shapes are the key ADDITION: normalized
#     (so no accessibility mask needed) and sensitive to t_inv and p_start
#     differently from mean pi, which is what breaks the ridge
#   - Fst is deliberately EXCLUDED: Fst = 1-(r+1)/(2dr) exactly, so it is
#     algebraically redundant with the two ratios and adds nothing (NOTES 5.3)
#   - absolute pi levels are excluded from the default vector because of the
#     1.31x calibration offset (NOTES 8.3); use --use-absolute to include them
#     with a fitted nuisance scale
STAT_NAMES = (
    ["pi_i_over_pi_s", "dxy_over_pi_i", "p_final"]
    + [f"sfs_i_{k}" for k in range(1, SFS_BINS + 1)]
    + [f"sfs_s_{k}" for k in range(1, SFS_BINS + 1)]
)
STAT_NAMES_ABSOLUTE = ["pi_i_abs", "pi_s_abs", "dxy_abs"]


def draw_params(rng: np.random.Generator, neutral_only: bool = False,
                noflux_only: bool = False) -> dict:
    """One draw from the joint prior.

    ``neutral_only`` forces s = 0 and ``noflux_only`` forces p_flux = 0, for the
    model-comparison arms. Under the full model each has an atom at zero, so
    P(neutral | data) can also be read off a single combined run; the dedicated
    arms exist for a cleaner Bayes factor.
    """
    out = {}
    for name, spec in PRIORS.items():
        kind, lo, hi = spec
        if kind == "uniform":
            out[name] = float(rng.uniform(lo, hi))
        elif kind == "loguniform":
            out[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        elif kind == "loguniform_atom0":
            atom = (PRIOR_NEUTRAL_WEIGHT if name == "s"
                    else PRIOR_NOFLUX_WEIGHT)
            if rng.random() < atom:
                out[name] = 0.0
            else:
                out[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            raise ValueError(f"unknown prior kind {kind!r} for {name!r}")
    if neutral_only:
        out["s"] = 0.0
    if noflux_only:
        out["p_flux"] = 0.0
    # h is meaningless when s == 0; pin it so the ABC does not waste a dimension
    # inferring an unidentifiable parameter on the neutral atom.
    if out["s"] == 0.0:
        out["h"] = 0.5
    return out
