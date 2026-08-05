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


def rec_rate_for_inversion(rmap_path: str | None = None) -> float:
    """Mean recombination rate inside the inversion body.

    Falls back to the autosomal proxy REC_RATE when no chr2 map is available.
    Once a chr2 map exists, this returns the length-weighted mean across the
    inversion body, which is the right scalar for the rescaled proxy inversion
    (a positional map cannot be applied to a 100 kb stand-in for 20 Mb).
    """
    path = rmap_path or CHR2_RMAP
    if path is None:
        return REC_RATE
    import pandas as pd
    d = pd.read_csv(path, sep="\t")
    d["chrom"] = d["chrom"].astype(str).str.replace(r"^b'|'$", "", regex=True)
    d = d[d.chrom == "2"]
    body = d[(d.start >= INV_START_REAL) & (d.end <= INV_STOP_REAL)]
    if body.empty:
        raise ValueError(f"no chr2 windows inside the inversion body in {path}")
    w = (body.end - body.start).to_numpy(dtype=float)
    return float((body.recombRate.to_numpy(dtype=float) * w).sum() / w.sum())

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
