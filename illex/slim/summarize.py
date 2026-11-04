"""Turn a SLiM tree sequence into the ABC summary-statistic vector.

Three steps, in order:

1. **Recapitate.** SLiM's forward phase starts at max(t_inv, T_GROW) generations
   ago, by which point the population is at constant N_ANC, so the deeper history
   is recapitated with msprime at constant N_ANC/Q. Without this the trees have
   uncoalesced roots and every diversity statistic is truncated.
2. **Overlay neutral mutations** at the SCALED rate mu*Q. SLiM simulated none;
   doing it here is far cheaper and statistically identical.
3. **Split by karyotype and measure**, restricted to the inversion body.

Karyotype is read from the SLiM marker mutation at the inversion's left
breakpoint: a sample node carrying it is an inverted (I) haplotype. This is the
arrangement-level definition, matching msinv. (The empirical per-arrangement pi
was computed from AA/BB *homokaryotypic individuals* for calling convenience;
under a barrier model an I haplotype from a heterokaryote belongs to the same
arrangement class, so the definitions agree.)

Statistics are **interval-restricted to the inversion body**: the simulated
sequence has collinear flanks, which are panmictic and drag both ratios toward
the null (~21% understatement of dxy/pi_I if ignored).
"""
from __future__ import annotations

import numpy as np

from . import config as C


def load_and_prepare(trees_path: str, q: float, seed: int):
    """Recapitate + overlay mutations. Returns (ts, metadata)."""
    import msprime
    import pyslim
    import tskit

    ts = tskit.load(trees_path)
    meta = dict(ts.metadata.get("SLiM", {}).get("user_metadata", {}))
    # SLiM stores user metadata values as length-1 lists.
    meta = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
            for k, v in meta.items()}

    # Prefer the rates the recipe actually ran with over the config defaults, so
    # a sensitivity arm that overrides R or MU on the command line is recapitated
    # and mutated consistently instead of silently reverting to the default.
    n_anc = float(meta.get("n_anc", C.N_ANC))
    rec = float(meta.get("rec_rate", C.REC_RATE))
    mu = float(meta.get("mu", C.MU))

    n_anc_q = max(2, int(round(n_anc / q)))
    # The forward sim's time unit is rescaled generations (SLiM 5 "ticks"), so
    # ancestral_Ne and the rates are all in those same rescaled units. msprime's
    # ticks-vs-generations TimeUnitsMismatchWarning is therefore expected.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = pyslim.recapitate(ts, ancestral_Ne=n_anc_q,
                               recombination_rate=rec * q, random_seed=seed)
        ts = msprime.sim_mutations(ts, rate=mu * q, random_seed=seed + 1,
                                   keep=True)
    return ts, meta


def karyotype_sample_nodes(ts, inv_start: float):
    """(inverted_nodes, standard_nodes) from the marker mutation.

    The marker is the SLiM mutation at ``inv_start``. Any msprime-overlaid
    neutral mutation that happens to land on the same site would corrupt the
    read, so the site is identified by position AND the genotype is taken from
    the SLiM-origin allele.
    """
    samples = np.asarray(ts.samples(), dtype=np.int32)
    target = None
    for site in ts.sites():
        if abs(site.position - inv_start) < 0.5:
            target = site
            break
    if target is None:
        raise RuntimeError(f"no site at inversion start {inv_start}; the marker "
                           "mutation is missing from the tree sequence")

    var = next(ts.variants(samples=samples, left=target.position,
                           right=target.position + 1))
    # Allele 0 is ancestral (no inversion). Any non-zero state at the marker site
    # means the inverted background.
    g = np.asarray(var.genotypes)
    inv_mask = g != 0
    return samples[inv_mask], samples[~inv_mask]


def folded_sfs_shape(ts, nodes, interval, n_proj: int) -> np.ndarray:
    """Normalized folded SFS over ``interval``, projected to ``n_proj``.

    Branch mode, for the same reason the coalescent test used it: it reads the
    expected spectrum off branch lengths, so it is free of mutation noise and
    insensitive to overall timescale -- and the SHAPE is what carries the
    identifying information, not the scale.

    Subsampling to n_proj rather than hypergeometric projection: with simulated
    data we can just draw the sample we want, which is exact.
    """
    rng = np.random.default_rng(abs(hash((len(nodes), n_proj))) % (2 ** 31))
    take = rng.choice(np.asarray(nodes), size=min(n_proj, len(nodes)),
                      replace=False)
    sub = ts.simplify(samples=take, filter_sites=False)
    left, right = interval
    af = sub.allele_frequency_spectrum(
        polarised=False, span_normalise=True, mode="branch",
        windows=[0.0, left, right, sub.sequence_length])[1]
    nb = min(C.SFS_BINS, len(af) - 1)
    v = np.asarray(af[1:nb + 1], dtype=float)
    tot = v.sum()
    out = np.full(C.SFS_BINS, np.nan)
    if tot > 0:
        out[:nb] = v / tot
    return out


def arrangement_stats(ts, i_nodes, s_nodes, interval, mu_scaled: float) -> dict:
    """pi within each arrangement and dxy between, over ``interval``.

    Branch mode: tskit branch-mode diversity is the branch length separating a
    pair, i.e. 2*T_coal, so pi = mu * branch_diversity.
    """
    left, right = interval
    windows = [0.0, left, right, ts.sequence_length]
    i_nodes = list(map(int, i_nodes))
    s_nodes = list(map(int, s_nodes))
    pi_i = mu_scaled * ts.diversity([i_nodes], mode="branch",
                                    windows=windows)[1, 0]
    pi_s = mu_scaled * ts.diversity([s_nodes], mode="branch",
                                    windows=windows)[1, 0]
    dxy = mu_scaled * ts.divergence([i_nodes, s_nodes], mode="branch",
                                    windows=windows)[1]
    return {
        "pi_i_abs": float(pi_i), "pi_s_abs": float(pi_s), "dxy_abs": float(dxy),
        "pi_i_over_pi_s": float(pi_i / pi_s) if pi_s > 0 else np.nan,
        "dxy_over_pi_i": float(dxy / pi_i) if pi_i > 0 else np.nan,
    }


def summarize(trees_path: str, q: float, seed: int) -> dict:
    """Full statistic vector for one simulation."""
    ts, meta = load_and_prepare(trees_path, q, seed)
    inv_start = float(meta.get("inv_start", C.FLANK_LEN_SIM))
    inv_end = float(meta.get("inv_end", C.FLANK_LEN_SIM + C.INV_LEN_SIM - 1))
    interval = (inv_start, inv_end + 1.0)

    i_all, s_all = karyotype_sample_nodes(ts, inv_start)
    if len(i_all) < C.SFS_PROJ or len(s_all) < C.SFS_PROJ:
        raise RuntimeError(f"too few haplotypes: I={len(i_all)} S={len(s_all)}")

    rng = np.random.default_rng(seed + 2)
    i_nodes = rng.choice(i_all, size=min(C.N_HAP_I, len(i_all)), replace=False)
    s_nodes = rng.choice(s_all, size=min(C.N_HAP_S, len(s_all)), replace=False)

    # mu is scaled because the tree sequence is in scaled generations; the RATIOS
    # are unaffected, and the absolute levels come out on the real per-site scale
    # because mu*Q against times/Q cancels.
    mu = float(meta.get("mu", C.MU))
    out = arrangement_stats(ts, i_nodes, s_nodes, interval, mu * q)

    sfs_i = folded_sfs_shape(ts, i_all, interval, C.SFS_PROJ)
    sfs_s = folded_sfs_shape(ts, s_all, interval, C.SFS_PROJ)
    for k in range(C.SFS_BINS):
        out[f"sfs_i_{k + 1}"] = float(sfs_i[k])
        out[f"sfs_s_{k + 1}"] = float(sfs_s[k])

    # p_final is the realized inverted-haplotype frequency in the whole final
    # population, taken from SLiM (not from the subsample).
    out["p_final"] = float(meta.get("p_final", len(i_all)
                                    / max(1, len(i_all) + len(s_all))))
    out["n_restarts"] = int(meta.get("n_restarts", -1))
    out["n_trees"] = int(ts.num_trees)
    return out


# ------------------------------------------------------------------ trajectory-ABC statistics (Task 9) ----
#
# The SLiM side of the simulator validation (``illex.abc.slim_vs_msinv``): the same ``STAT_KEYS`` as an ABC row
# (``illex.abc.simulate.simulate_stats``) and the boundary keys (``illex.abc.boundary.BND_KEYS``, as
# ``simulate.boundary_ppc``), from one SLiM run of ``inversion_abc.slim`` with a BODY_BP body between two
# BND_MARGIN_BP collinear margins. Classes: I = the marker-carrying arrangement (2a), S = the other (2+).
#
#   1. recapitate at constant N_ANC / Q (the forward phase starts at or before T_GROW);
#   2. split the final generation's haplosomes by the marker and record p_now = I / (I + S);
#   3. sample N_INV I and N_STD S chromosomes at random (haploid, as the ABC), simplify to them and drop the
#      marker site;
#   4. overlay binary infinite-sites mutations at MU * Q (the tree sequence is in ticks of Q generations);
#   5. body statistics over [inv_start, inv_end + 1) with the body genotype errors (``statvec.region_stats``),
#      edge_centre on the branch tree sequence, the msprime neutral control of the ABC
#      (``simulate.control_stats``) and ``statvec.assemble``;
#   6. boundary keys over both margins with the control genotype errors and the data's flank mask.
#
# Seed words (``abc_seed_words``) are named like simulate's: slim (the SLiM seed), ctl, mi, mc, pi, pc, obs, ei,
# ec, plus recap and take (the class subsample).

ABC_WORDS = ("slim", "ctl", "mi", "mc", "pi", "pc", "obs", "ei", "ec", "recap", "take")


def abc_seed_words(seed: int) -> dict:
    """The 11 named SeedSequence words of one SLiM validation replicate (all in [1, 2^31 - 1))."""
    ss = np.random.SeedSequence(int(seed))
    w = ss.generate_state(len(ABC_WORDS)) % (2**31 - 1) + 1
    return {k: int(v) for k, v in zip(ABC_WORDS, w)}


def _slim_meta(ts) -> dict:
    meta = dict(ts.metadata.get("SLiM", {}).get("user_metadata", {}))
    return {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in meta.items()}


def recapitate_abc(trees_path: str, seed: int):
    """(ts, meta): the SLiM tree sequence recapitated at N_ANC / Q, no neutral mutations yet."""
    import warnings

    import pyslim
    import tskit

    ts = tskit.load(trees_path)
    meta = _slim_meta(ts)
    q = float(meta["Q"])
    n_anc_q = max(2, int(round(float(meta["n_anc"]) / q)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # ticks vs generations time units: expected, see load_and_prepare
        ts = pyslim.recapitate(ts, ancestral_Ne=n_anc_q, recombination_rate=float(meta["rec_rate"]) * q,
                               random_seed=abc_seed_words(seed)["recap"])
    return ts, meta


def abc_stats_from_ts(ts, inv_start: float, inv_end: float, q: float, seed: int, mu: float | None = None,
                      ld_error=None, mask="data", n_inv: int | None = None, n_std: int | None = None) -> dict:
    """``STAT_KEYS`` + ``BND_KEYS`` + p_now / p_obs / n_nan / n_i_pop / n_s_pop of a complete (coalesced) tree
    sequence whose samples are the final population's chromosomes and whose marker site sits at ``inv_start``.

    The body is [inv_start, inv_end + 1). ``q`` scales the mutation rate (``mu`` per generation, default
    ``illex.abc.constants.MU``). ``ld_error`` defaults to ``error_model.LD_ERROR``; ``mask`` is passed to
    ``boundary.boundary_stats`` ("data" loads the chr2 flank mask, None keeps every flank site). Raises
    RuntimeError when either class has fewer chromosomes than its sample size.
    """
    import math

    import msprime

    from illex.abc import statvec as SV
    from illex.abc.boundary import boundary_stats, load_flank_mask
    from illex.abc.constants import MU, N_INV, N_KARYO, N_STD
    from illex.abc.error_model import LD_ERROR
    from illex.abc.simulate import control_stats

    mu = MU if mu is None else float(mu)
    ld_error = LD_ERROR if ld_error is None else ld_error
    n_inv = N_INV if n_inv is None else int(n_inv)
    n_std = N_STD if n_std is None else int(n_std)
    w = abc_seed_words(seed)

    i_all, s_all = karyotype_sample_nodes(ts, inv_start)
    p_now = len(i_all) / max(1, len(i_all) + len(s_all))
    if len(i_all) < n_inv or len(s_all) < n_std:
        raise RuntimeError(f"too few chromosomes per class: I={len(i_all)} S={len(s_all)} "
                           f"(need {n_inv} / {n_std})")
    take = np.random.default_rng(w["take"])
    i_take = take.choice(i_all, size=n_inv, replace=False)
    s_take = take.choice(s_all, size=n_std, replace=False)
    sub = ts.simplify(samples=np.concatenate([i_take, s_take]).astype(np.int32), keep_unary=False)
    tables = sub.dump_tables()
    tables.mutations.clear()
    tables.sites.clear()
    bts = tables.tree_sequence()                       # the branch tree sequence, marker dropped
    i_nodes = np.arange(n_inv, dtype=np.int32)
    s_nodes = np.arange(n_inv, n_inv + n_std, dtype=np.int32)
    mts = msprime.sim_mutations(bts, rate=mu * q, random_seed=w["mi"], model=msprime.BinaryMutationModel(),
                                discrete_genome=False)

    win = (float(inv_start), float(inv_end) + 1.0)
    body = SV.region_stats(mts, i_nodes, s_nodes, win, np.random.default_rng(w["pi"]),
                           ld_err=ld_error["body"], err_rng=np.random.default_rng(w["ei"]))
    edge = SV.edge_centre(bts, i_nodes, s_nodes, win)
    control = control_stats(w["ctl"], w["mc"], w["pc"], w["ec"], ld_error["control"])
    out = SV.assemble(body, control, edge)

    if isinstance(mask, str):
        if mask != "data":
            raise ValueError(f"mask must be 'data', None or a dict, got {mask!r}")
        mask = load_flank_mask()
    out.update(boundary_stats(mts, i_nodes, s_nodes, win[0], win[1], rng=np.random.default_rng(w["pi"]),
                              err=ld_error["control"], err_rng=np.random.default_rng(w["ei"]), mask=mask))
    two_n = 2 * N_KARYO
    out["p_now"] = float(p_now)
    out["p_obs"] = float(np.random.default_rng(w["obs"]).binomial(two_n, p_now) / two_n)
    out["n_nan"] = int(sum(math.isnan(out[k]) for k in SV.MODEL_KEYS))
    out["n_i_pop"] = int(len(i_all))
    out["n_s_pop"] = int(len(s_all))
    return out


def summarize_abc(trees_path: str, seed: int, mask="data", ld_error=None) -> dict:
    """The trajectory-ABC statistic row of one SLiM run (``abc_stats_from_ts`` after ``recapitate_abc``)."""
    ts, meta = recapitate_abc(trees_path, seed)
    out = abc_stats_from_ts(ts, float(meta["inv_start"]), float(meta["inv_end"]), float(meta["Q"]), seed,
                            mu=float(meta["mu"]), ld_error=ld_error, mask=mask)
    out["n_restarts"] = int(meta.get("n_restarts", -1))
    out["p_final_slim"] = float(meta.get("p_final", float("nan")))
    return out
