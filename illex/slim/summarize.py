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

    n_anc_q = max(2, int(round(C.N_ANC / q)))
    ts = pyslim.recapitate(ts, ancestral_Ne=n_anc_q,
                           recombination_rate=C.REC_RATE * q,
                           random_seed=seed)
    ts = msprime.sim_mutations(ts, rate=C.MU * q, random_seed=seed + 1,
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
    out = arrangement_stats(ts, i_nodes, s_nodes, interval, C.MU * q)

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
