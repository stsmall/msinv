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
        pi[name] = ts.diversity(
            [sample_sets[name]], windows=wins, mode="site"
        ).reshape(-1)
        tajd[name] = ts.Tajimas_D(
            [sample_sets[name]], windows=wins, mode="site"
        ).reshape(-1)

    dxy = {}
    fst = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
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
