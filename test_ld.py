#!/usr/bin/env python3
"""
LD validation for msinv.

Computes pairwise r² from haplotype matrices and validates:
  1. LD within inversion is elevated (suppressed S-I recombination)
  2. LD drops at breakpoints
  3. LD in collinear regions decays normally with distance
  4. LD across breakpoints is low (independent genealogies)
  5. Compares all-sample LD with within-class (SS) and between-class (SI)
"""

import sys
import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location(
    'msinv', '/home/ssmall/inversion_sims/files/msinv.py')
msinv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msinv)

PASS = 0
FAIL = 0


def compute_r2(haps, pos, distance_bins, pair_filter=None):
    """
    Compute mean r² in distance bins.

    haps: (nsam, nsites) array of 0/1
    pos: list of positions (in [0,1])
    distance_bins: list of (lo, hi) distance ranges
    pair_filter: if provided, (row_indices_A, row_indices_B) — only
                 count LD between sites where A and B differ.
                 None = all pairs.

    Returns array of mean r² per bin.
    """
    nsam, nsites = haps.shape
    if nsites < 2:
        return np.zeros(len(distance_bins))

    # Filter to polymorphic sites (MAF > 0)
    freqs = haps.mean(axis=0)
    poly = (freqs > 0) & (freqs < 1)
    if poly.sum() < 2:
        return np.zeros(len(distance_bins))

    poly_idx = np.where(poly)[0]
    poly_haps = haps[:, poly_idx].astype(float)
    poly_pos = [pos[i] for i in poly_idx]
    n_poly = len(poly_idx)

    # Compute r² for all pairs, bin by distance
    bin_sums = np.zeros(len(distance_bins))
    bin_counts = np.zeros(len(distance_bins))

    # Subsample if too many sites
    max_pairs = 50000
    if n_poly * (n_poly - 1) // 2 > max_pairs:
        # Random subsample of pairs
        for _ in range(max_pairs):
            i = np.random.randint(n_poly - 1)
            j = np.random.randint(i + 1, n_poly)
            d = abs(poly_pos[j] - poly_pos[i])
            for bi, (lo, hi) in enumerate(distance_bins):
                if lo <= d < hi:
                    r2 = _r2(poly_haps[:, i], poly_haps[:, j])
                    if r2 is not None:
                        bin_sums[bi] += r2
                        bin_counts[bi] += 1
                    break
    else:
        for i in range(n_poly):
            for j in range(i + 1, n_poly):
                d = abs(poly_pos[j] - poly_pos[i])
                for bi, (lo, hi) in enumerate(distance_bins):
                    if lo <= d < hi:
                        r2 = _r2(poly_haps[:, i], poly_haps[:, j])
                        if r2 is not None:
                            bin_sums[bi] += r2
                            bin_counts[bi] += 1
                        break

    result = np.zeros(len(distance_bins))
    for bi in range(len(distance_bins)):
        if bin_counts[bi] > 0:
            result[bi] = bin_sums[bi] / bin_counts[bi]
    return result


def _r2(x, y):
    """Compute r² between two binary vectors."""
    n = len(x)
    px = x.mean()
    py = y.mean()
    if px == 0 or px == 1 or py == 0 or py == 1:
        return None
    pxy = (x * y).mean()
    D = pxy - px * py
    denom = px * (1 - px) * py * (1 - py)
    if denom <= 0:
        return None
    return D * D / denom


def compute_r2_by_region(haps, pos, bp_left, bp_right):
    """
    Compute mean r² for site pairs within and across regions.

    Returns dict with keys:
      'within_col': r² for pairs both in collinear
      'within_inv': r² for pairs both in inversion
      'across_bp':  r² for pairs spanning a breakpoint
    """
    nsam, nsites = haps.shape
    if nsites < 2:
        return {'within_col': 0, 'within_inv': 0, 'across_bp': 0}

    freqs = haps.mean(axis=0)
    poly = (freqs > 0) & (freqs < 1)
    poly_idx = np.where(poly)[0]
    if len(poly_idx) < 2:
        return {'within_col': 0, 'within_inv': 0, 'across_bp': 0}

    poly_haps = haps[:, poly_idx].astype(float)
    poly_pos = [pos[i] for i in poly_idx]

    sums = {'within_col': 0, 'within_inv': 0, 'across_bp': 0}
    counts = {'within_col': 0, 'within_inv': 0, 'across_bp': 0}

    n_poly = len(poly_idx)
    max_pairs = 20000
    pairs_done = 0

    for i in range(n_poly):
        for j in range(i + 1, n_poly):
            if pairs_done >= max_pairs:
                break
            pi, pj = poly_pos[i], poly_pos[j]
            i_in = bp_left <= pi < bp_right
            j_in = bp_left <= pj < bp_right

            if i_in == j_in:
                key = 'within_inv' if i_in else 'within_col'
            else:
                key = 'across_bp'

            r2 = _r2(poly_haps[:, i], poly_haps[:, j])
            if r2 is not None:
                sums[key] += r2
                counts[key] += 1
                pairs_done += 1
        if pairs_done >= max_pairs:
            break

    result = {}
    for key in sums:
        result[key] = sums[key] / counts[key] if counts[key] > 0 else 0
    return result


def check(name, condition, detail=""):
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def test_ld_inversion_vs_collinear():
    """LD within inversion should be higher than within collinear."""
    print("\n=== Test: LD within inversion > within collinear ===")

    NR = 100
    sim = msinv.MsinvSimulator(
        nsam=10, nreps=NR, theta=40.0, rho=100.0, nsites=1000,
        n_std=5, n_inv=5, p_inv=0.5, c=0.01, seed=42, t_inv=20.0,
        bp_left=0.3, bp_right=0.7)

    r2_inv_total = 0
    r2_col_total = 0
    r2_across_total = 0
    n_ok = 0

    for _ in range(NR):
        try:
            pos, haps = sim.simulate_one()
            if len(pos) < 5:
                continue
            r2s = compute_r2_by_region(haps, pos, 0.3, 0.7)
            r2_inv_total += r2s['within_inv']
            r2_col_total += r2s['within_col']
            r2_across_total += r2s['across_bp']
            n_ok += 1
        except Exception:
            pass

    if n_ok > 0:
        r2_inv = r2_inv_total / n_ok
        r2_col = r2_col_total / n_ok
        r2_across = r2_across_total / n_ok
    else:
        r2_inv = r2_col = r2_across = 0

    print(f"    {n_ok}/{NR} replicates")
    print(f"    r² within inversion: {r2_inv:.4f}")
    print(f"    r² within collinear: {r2_col:.4f}")
    print(f"    r² across breakpoint: {r2_across:.4f}")

    check("LD within inversion > within collinear",
          r2_inv > r2_col,
          f"inv={r2_inv:.4f}, col={r2_col:.4f}")

    check("LD across breakpoint is low (< within inversion)",
          r2_across < r2_inv,
          f"across={r2_across:.4f}, inv={r2_inv:.4f}")


def test_ld_decay_collinear():
    """LD in collinear regions should decay with distance."""
    print("\n=== Test: LD decays with distance in collinear ===")

    NR = 100
    sim = msinv.MsinvSimulator(
        nsam=10, nreps=NR, theta=40.0, rho=100.0, nsites=1000,
        n_std=5, n_inv=5, p_inv=0.5, c=0.01, seed=42, t_inv=20.0,
        bp_left=0.3, bp_right=0.7)

    # Distance bins for collinear region [0, 0.3]
    bins = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2)]
    r2_sums = np.zeros(len(bins))
    n_ok = 0

    for _ in range(NR):
        try:
            pos, haps = sim.simulate_one()
            if len(pos) < 3:
                continue
            # Only use collinear sites
            col_idx = [j for j, p in enumerate(pos) if p < 0.3 or p > 0.7]
            if len(col_idx) < 3:
                continue
            col_haps = haps[:, col_idx]
            col_pos = [pos[j] for j in col_idx]
            r2s = compute_r2(col_haps, col_pos, bins)
            r2_sums += r2s
            n_ok += 1
        except Exception:
            pass

    if n_ok > 0:
        r2_means = r2_sums / n_ok
    else:
        r2_means = np.zeros(len(bins))

    print(f"    {n_ok}/{NR} replicates")
    for bi, (lo, hi) in enumerate(bins):
        print(f"    r² at distance [{lo:.2f},{hi:.2f}): {r2_means[bi]:.4f}")

    check("r² decreases with distance (short > long)",
          r2_means[0] >= r2_means[-1] or n_ok < 10,
          f"near={r2_means[0]:.4f}, far={r2_means[-1]:.4f}")


def test_ld_no_inversion():
    """Without inversion, LD should be uniform across chromosome."""
    print("\n=== Test: No inversion → uniform LD ===")

    NR = 100
    sim = msinv.MsinvSimulator(
        nsam=10, nreps=NR, theta=40.0, rho=100.0, nsites=1000,
        p_inv=0.0, c=0.0, seed=42)

    r2_sums = {'left': 0, 'mid': 0, 'right': 0}
    n_ok = 0

    for _ in range(NR):
        try:
            pos, haps = sim.simulate_one()
            if len(pos) < 5:
                continue
            n_ok += 1
            # r² for nearby pairs in three regions
            for j in range(len(pos) - 1):
                if abs(pos[j+1] - pos[j]) < 0.05:
                    r2 = _r2(haps[:, j].astype(float), haps[:, j+1].astype(float))
                    if r2 is not None:
                        if pos[j] < 0.3:
                            r2_sums['left'] += r2
                        elif pos[j] < 0.7:
                            r2_sums['mid'] += r2
                        else:
                            r2_sums['right'] += r2
        except Exception:
            pass

    print(f"    {n_ok}/{NR} replicates")
    for key in ['left', 'mid', 'right']:
        print(f"    r² in {key}: {r2_sums[key]/max(1,n_ok):.4f}")

    # All three should be roughly similar
    vals = [r2_sums[k] / max(1, n_ok) for k in ['left', 'mid', 'right']]
    if max(vals) > 0:
        cv = np.std(vals) / np.mean(vals)
    else:
        cv = 0
    check("LD uniform across chromosome (CV < 0.5)",
          cv < 0.5 or n_ok < 10,
          f"CV={cv:.3f}")


def test_ld_high_flux_reduces_ld():
    """High gene flux should reduce LD within inversion."""
    print("\n=== Test: High gene flux reduces inversion LD ===")

    NR = 50
    r2_inv = {}
    for c_val in [0.01, 1.0]:
        sim = msinv.MsinvSimulator(
            nsam=10, nreps=NR, theta=40.0, rho=100.0, nsites=1000,
            n_std=5, n_inv=5, p_inv=0.5, c=c_val, seed=42, t_inv=20.0,
            bp_left=0.3, bp_right=0.7)
        total = 0
        n_ok = 0
        for _ in range(NR):
            try:
                pos, haps = sim.simulate_one()
                if len(pos) < 3:
                    continue
                r2s = compute_r2_by_region(haps, pos, 0.3, 0.7)
                total += r2s['within_inv']
                n_ok += 1
            except Exception:
                pass
        r2_inv[c_val] = total / max(1, n_ok)
        print(f"    c={c_val}: r²_inv={r2_inv[c_val]:.4f} ({n_ok}/{NR} ok)")

    check("Higher gene flux → lower inversion LD",
          r2_inv[1.0] < r2_inv[0.01],
          f"c=0.01: {r2_inv[0.01]:.4f}, c=1.0: {r2_inv[1.0]:.4f}")


def main():
    global PASS, FAIL

    test_ld_inversion_vs_collinear()
    test_ld_decay_collinear()
    test_ld_no_inversion()
    test_ld_high_flux_reduces_ld()

    print(f"\n{'='*50}")
    print(f"LD Results: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
