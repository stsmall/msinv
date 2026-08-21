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

# --- fixed recombination (ReLERNN chr2, MEASURED 2026-08-07; NOTES sec 8.1) ---
# chr2 now has its own ReLERNN maps. Four of them (sec 8.0):
#   run_chr2_AA    -- AA homokaryotypes only, own network
#   run_chr2_all   -- all 633 samples, own network
#   chr2_{male,female}.autonet.PREDICT.txt -- the GENOME-WIDE autosomal network
#                     applied to chr2; the only maps on the same absolute
#                     calibration as the genome-wide numbers below.
#
# REC_RATE is the chr2 COLLINEAR rate from the MALE autonet map (+/-2 Mb buffer
# outside the breakpoints). Male, not sex-averaged, so this pipeline matches the
# 14_sweep_seqmodel campaign, which also runs on the male map.
#   male   collinear 1.977e-9   interior 1.967e-9   (ratio 0.995)
#   female collinear 2.248e-9   interior 2.265e-9   (ratio 1.008)
#   sex-averaged collinear 2.113e-9 (recorded, not used)
#
# This SUPERSEDES the 2.52e-9 length-matched-autosome proxy, which ran 27% high
# against the male map (2.52 vs 1.977). chr2 recombines less than the
# genome-wide average: male chr2 collinear / male genome-wide = 1.977/2.148 =
# 0.92, female 2.248/2.892 = 0.78.
REC_RATE = 1.977e-9
REC_RATE_BRACKET = (1.977e-9, 2.248e-9)   # chr2 collinear male, female
REC_RATE_PROXY_OLD = 2.52e-9              # superseded; kept for provenance

# Measured chr2 maps. CHR2_RMAP is used by the report script and by
# rec_rate_for_inversion(); REC_RATE above is already the number it returns, so
# nothing silently changes if the path is unset.
_RELERNN = ("/sietch_colab/data_share/illex/popgen_data/analysis/steps/"
            "11_relernn")
CHR2_RMAP = f"{_RELERNN}/chr2_autosomal_predict/chr2_male.autonet.PREDICT.txt"
CHR2_RMAP_FEMALE = f"{_RELERNN}/chr2_autosomal_predict/chr2_female.autonet.PREDICT.txt"
CHR2_RMAP_AA = f"{_RELERNN}/run_chr2_AA/proj/chr2_AA.kept.PREDICT.BSCORRECTED.txt"
CHR2_RMAP_ALL = f"{_RELERNN}/run_chr2_all/proj/chr2_all.kept.PREDICT.BSCORRECTED.txt"

# chr2-specific 3-state mask (accessible_invariant / accessible_variant /
# inaccessible). Confirms ACC_FRAC_INV below to 4 dp, so it changes no number --
# its new content is the invariant/variant split, which gives the SFS zero class.
CHR2_MASK_BED = ("/sietch_colab/data_share/illex/popgen_data/analysis/steps/"
                 "03_karyotype/chr2_mask/chr2.mask.3state.bed")

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

    Taken from chr2's COLLINEAR regions, ``margin`` bp outside the breakpoints.

    Using the collinear regions was originally a precaution: ReLERNN infers
    recombination from LD decay, so the interior windows might have measured the
    realized barrier rather than the meiotic rate, and feeding that to a model
    that already imposes the barrier would suppress twice. The chr2 maps landed
    on 2026-08-07 and the precaution turned out to be unnecessary -- interior and
    collinear agree to 0.5% (NOTES sec 8.0). It is kept anyway because it is the
    correct region on principle and costs nothing.
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
    """ReLERNN's rate INSIDE the inversion. Diagnostic only -- NOT a target.

    **Withdrawn as a validation statistic (2026-08-07).** It was proposed as one
    of the few remaining independent constraints: a fitted (t_inv, p_start, s)
    with the barrier implies a realized LD level inside the inversion, which
    would map to an apparent recombination rate. The chr2 maps falsify the
    premise the idea rested on -- ReLERNN does not respond to the barrier at its
    ~19 kb window scale:

        map              interior/collinear
        autonet male              0.995
        autonet female            1.008
        all 633 pooled            1.117
        AA homokaryotypes only    0.888

    Three maps show no suppression, the pooled map (the one containing every
    heterokaryotype) shows a slight EXCESS, and the largest deficit is in the
    AA-only map, where recombination is not suppressed at all and the ratio
    should be 1. The ordering is inconsistent with a barrier signal. It is not a
    clean readout of local Ne either: AA diversity inside the inversion is 38%
    of AA diversity in the collinear control, against an 11% ReLERNN deficit.

    The reason is scale. The barrier suppresses crossovers BETWEEN arrangements
    across ~20 Mb; within a 19 kb window, LD decay is governed by
    within-arrangement recombination, which is unsuppressed. Detecting the
    barrier needs a long-range LD statistic, not ReLERNN.

    Kept as a diagnostic so the numbers stay reproducible from code.
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
ACC_FRAC_CONTROL = 0.6069     # chr2:10-30 Mb control region
ACC_FRAC_COLLINEAR = 0.4879   # all chr2 collinear (+/-2 Mb); the
#   control region is the outlier, not the inversion: 0.4791 inside vs
#   0.4879 collinear is a ratio of 0.982, so the diversity deficit inside
#   the inversion is NOT a masking artifact. Verified on the chr2 3-state
#   mask 2026-08-07.

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
