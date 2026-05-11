"""Per-rep stats for the validation suite.

All implementations are tskit-native where possible. The H-stats and LD
decay are hand-rolled in later tasks since tskit does not provide them.
"""

from __future__ import annotations

import numpy as np
import tskit


def window_stats(
    ts: tskit.TreeSequence,
    *,
    sample_sets: dict[str, list[int]],
    n_windows: int = 40,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-window pi, dxy, Fst, Tajima's D for each set / set-pair.

    Returns
    -------
    {"pi":         {name: (n_windows,) array, ...},
     "dxy":        {f"{a}_{b}": (n_windows,) array, ...},
     "fst":        {f"{a}_{b}": (n_windows,) array, ...},
     "tajimas_d":  {name: (n_windows,) array, ...}}

    `dxy` and `fst` are computed for every ordered pair of sets (a, b)
    with a < b lexicographically.
    """
    wins = np.linspace(0, ts.sequence_length, n_windows + 1)
    names = sorted(sample_sets)

    pi = {}
    tajd = {}
    for name in names:
        pi[name] = ts.diversity([sample_sets[name]], windows=wins, mode="site").reshape(
            -1
        )
        tajd[name] = ts.Tajimas_D(
            [sample_sets[name]], windows=wins, mode="site"
        ).reshape(-1)

    dxy = {}
    fst = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            key = f"{a}_{b}"
            d = ts.divergence(
                [sample_sets[a], sample_sets[b]], windows=wins, mode="site"
            ).reshape(-1)
            dxy[key] = d
            pi_w = (pi[a] + pi[b]) / 2
            with np.errstate(divide="ignore", invalid="ignore"):
                fst[key] = np.where(d > 0, 1.0 - pi_w / d, np.nan)

    return {"pi": pi, "dxy": dxy, "fst": fst, "tajimas_d": tajd}


def sfs(
    ts: tskit.TreeSequence,
    *,
    sample_set: list[int],
    folded: bool = True,
) -> np.ndarray:
    """Site frequency spectrum (folded or unfolded).

    Folded shape: (len(sample_set) // 2 + 1,).
    Unfolded shape: (len(sample_set) + 1,).
    """
    result = ts.allele_frequency_spectrum(
        [sample_set], polarised=not folded, span_normalise=False
    )
    if folded:
        n = len(sample_set)
        result = result[: n // 2 + 1]
    return result


def tree_shape_stats(
    ts: tskit.TreeSequence,
    *,
    n_samples: int = 1000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Distributions of TMRCA, total branch length, Colless imbalance.

    Sample n_samples random positions across the tree sequence; for each,
    extract the local tree and compute the three statistics.
    """
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, ts.sequence_length, size=n_samples)

    tmrca = np.empty(n_samples)
    total_branch = np.empty(n_samples)
    colless = np.empty(n_samples)
    for i, pos in enumerate(positions):
        tree = ts.at(float(pos))
        tmrca[i] = tree.time(tree.root)
        total_branch[i] = tree.total_branch_length
        colless[i] = _colless_imbalance(tree)
    return {"tmrca": tmrca, "total_branch": total_branch, "colless": colless}


def _colless_imbalance(tree) -> int:
    """Sum over internal nodes of |#leaves(left) - #leaves(right)|.

    Defined for binary trees. For multifurcating internal nodes, treat
    children pairwise: for k>=2 children, sum |L_i - L_j| over i<j.
    Multifurcations are rare in coalescent trees; this generalisation
    keeps the statistic finite.
    """
    leaves_below = {}
    total = 0
    for u in tree.nodes(order="postorder"):
        children = tree.children(u)
        if not children:
            leaves_below[u] = 1
        else:
            cnt = sum(leaves_below[c] for c in children)
            leaves_below[u] = cnt
            counts = [leaves_below[c] for c in children]
            for i in range(len(counts)):
                for j in range(i + 1, len(counts)):
                    total += abs(counts[i] - counts[j])
    return total


def ld_decay(
    ts: tskit.TreeSequence,
    *,
    distance_bins: np.ndarray,
    max_pairs: int = 5000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Mean r² in distance bins, computed over up to `max_pairs` random
    site pairs.

    Parameters
    ----------
    distance_bins : np.ndarray
        Bin edges (length n_bins + 1). Pair distance |pos_a - pos_b| is
        placed in the bin where bin_edges[i] <= d < bin_edges[i+1].

    Returns
    -------
    {"bin_edges":  distance_bins,
     "mean_r2":    (n_bins,) array, NaN if a bin had no pairs,
     "count":      (n_bins,) int array}
    """
    rng = np.random.default_rng(seed)
    n_bins = len(distance_bins) - 1
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)

    # Build per-site genotype matrix at biallelic sites only
    geno = []
    positions = []
    for var in ts.variants():
        if len(var.alleles) != 2:
            continue
        geno.append(var.genotypes.astype(np.int8))
        positions.append(var.site.position)
    if not geno:
        return {
            "bin_edges": distance_bins,
            "mean_r2": np.full(n_bins, np.nan),
            "count": counts,
        }
    geno = np.array(geno)  # shape (S, n_samples)
    positions = np.array(positions)  # shape (S,)
    n_sites = len(positions)

    # Sample random pairs (i, j) with i != j
    n_pairs = min(max_pairs, n_sites * (n_sites - 1) // 2)
    pairs_seen = 0
    while pairs_seen < n_pairs:
        i = rng.integers(0, n_sites, size=n_pairs * 2)
        j = rng.integers(0, n_sites, size=n_pairs * 2)
        ok = i != j
        i = i[ok]
        j = j[ok]
        for a, b in zip(i.tolist(), j.tolist()):
            if pairs_seen >= n_pairs:
                break
            d = abs(positions[a] - positions[b])
            bin_idx = np.searchsorted(distance_bins, d, side="right") - 1
            if 0 <= bin_idx < n_bins:
                ga = geno[a]
                gb = geno[b]
                pa = ga.mean()
                pb = gb.mean()
                pab = (ga * gb).mean()
                num = (pab - pa * pb) ** 2
                den = pa * (1 - pa) * pb * (1 - pb)
                if den > 0:
                    sums[bin_idx] += num / den
                    counts[bin_idx] += 1
            pairs_seen += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, sums / counts, np.nan)
    return {"bin_edges": distance_bins, "mean_r2": mean, "count": counts}


def hstats_from_haps(haps: np.ndarray) -> dict[str, float]:
    """Garud et al. 2015 H1, H12, H2, H2/H1 from a haplotype matrix.

    Parameters
    ----------
    haps : np.ndarray, shape (n_haplotypes, n_sites)
        Genotype matrix; each row is a haplotype, each column a SNP.

    Returns
    -------
    {"H1": ..., "H12": ..., "H2": ..., "H2_over_H1": ...}
    """
    n = haps.shape[0]
    if n == 0:
        return {
            "H1": float("nan"),
            "H12": float("nan"),
            "H2": float("nan"),
            "H2_over_H1": float("nan"),
        }
    # Pack each row to a tuple so we can count duplicates
    keys = [tuple(row.tolist()) for row in haps]
    counts: dict[tuple, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    freqs = sorted((c / n for c in counts.values()), reverse=True)
    H1 = float(sum(f * f for f in freqs))
    if len(freqs) >= 2:
        H12 = float((freqs[0] + freqs[1]) ** 2 + sum(f * f for f in freqs[2:]))
        H2 = float(H1 - freqs[0] * freqs[0])
    else:
        H12 = float(freqs[0] ** 2)
        H2 = 0.0
    H2_over_H1 = H2 / H1 if H1 > 0 else float("nan")
    return {"H1": H1, "H12": H12, "H2": H2, "H2_over_H1": H2_over_H1}


def hstats(
    ts: tskit.TreeSequence,
    *,
    sample_set: list[int],
    x_sel: float | None = None,
    window_bp: float | None = None,
) -> dict[str, float]:
    """Garud H-stats from a tskit ts, optionally restricted to a window
    around `x_sel` of width `window_bp`.

    If `x_sel` is None, computes H-stats genome-wide.
    """
    if x_sel is not None and window_bp is None:
        raise ValueError("window_bp required when x_sel is set")
    haps_rows = []
    for var in ts.variants(samples=sample_set):
        if len(var.alleles) != 2:
            continue
        if x_sel is not None:
            if abs(var.site.position - x_sel) > window_bp / 2:
                continue
        haps_rows.append(var.genotypes.astype(np.int8))
    if not haps_rows:
        return hstats_from_haps(np.empty((0, 0), dtype=np.int8))
    haps = np.array(haps_rows).T  # shape (n_samples, n_sites)
    return hstats_from_haps(haps)
