"""Pre-registered equivalence criteria for the validation suite.

Equivalence is declared when KS p > alpha AND Cohen's D < d_threshold.
Equivalence is rejected when KS p < alpha AND Cohen's D > d_threshold.
The asymmetric cases (one but not both) yield a "investigate" verdict.

Defaults match the spec: alpha=0.01, d_threshold=0.2.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import stats


def ks_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test. Returns (statistic, p_value)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    res = stats.ks_2samp(a, b)
    return float(res.statistic), float(res.pvalue)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's D effect size with pooled SD.

    d = (mean(a) - mean(b)) / s_pooled
    where s_pooled = sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) /
                          (n_a + n_b - 2))
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled == 0:
        return 0.0 if float(np.mean(a)) == float(np.mean(b)) else float("inf")
    return float((np.mean(a) - np.mean(b)) / pooled)


def equivalence_verdict(
    a: np.ndarray,
    b: np.ndarray,
    *,
    alpha: float = 0.01,
    d_threshold: float = 0.2,
) -> dict[str, float | Literal["equivalent", "not_equivalent", "investigate"]]:
    """Run KS + Cohen's D and return verdict per the pre-registered rule."""
    stat, p = ks_test(a, b)
    d = cohens_d(a, b)
    if np.isnan(p) or np.isnan(d):
        verdict = "investigate"
    elif p > alpha and abs(d) < d_threshold:
        verdict = "equivalent"
    elif p < alpha and abs(d) > d_threshold:
        verdict = "not_equivalent"
    else:
        verdict = "investigate"
    return {"ks_stat": stat, "ks_p": p, "cohens_d": d, "verdict": verdict}
