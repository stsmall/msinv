#!/usr/bin/env python3
"""
msinv_stats.py

Summary statistics for msinv output, following Guerrero et al. (2012).

Computes windowed statistics from ms-format output:
  - pi:  nucleotide diversity (within a group of samples)
  - dxy: absolute divergence (between two groups)
  - FST: fixation index between populations
  - FAT: fixation index between karyotype arrangements
         (Guerrero et al. eq. 22: FAT = 1 - T_A/T_T)

Usage:
  python msinv_stats.py <ms_output> --n_std N1 --n_inv N2 [--windows W]
"""

import numpy as np
import sys
import argparse


def parse_ms_output(filename):
    """
    Parse ms-format output file.

    Returns list of replicates, each a dict with:
      'segsites': int
      'positions': array of floats
      'haplotypes': (nsam x segsites) binary array
    """
    reps = []
    current = None

    if hasattr(filename, 'read'):
        fh = filename
        should_close = False
    elif filename == '-':
        fh = sys.stdin
        should_close = False
    else:
        fh = open(filename)
        should_close = True

    for line in fh:
        line = line.strip()
        if line == '//':
            if current is not None:
                reps.append(current)
            current = {'segsites': 0, 'positions': np.array([]),
                       'haplotypes': []}
        elif line.startswith('segsites:'):
            if current is not None:
                current['segsites'] = int(line.split(':')[1].strip())
        elif line.startswith('positions:'):
            if current is not None:
                parts = line.split(':')[1].strip().split()
                current['positions'] = np.array([float(p) for p in parts])
        elif current is not None and set(line) <= {'0', '1'} and len(line) > 0:
            current['haplotypes'].append([int(c) for c in line])

    if current is not None:
        reps.append(current)

    if should_close:
        fh.close()

    # Convert haplotypes to numpy arrays
    for rep in reps:
        if rep['haplotypes']:
            rep['haplotypes'] = np.array(rep['haplotypes'], dtype=int)
        else:
            rep['haplotypes'] = np.zeros((0, 0), dtype=int)

    return reps


def pairwise_diffs(haps):
    """
    Mean pairwise differences among a set of haplotypes.
    haps: (n x sites) binary array

    Returns: mean number of pairwise differences per site
    """
    n = haps.shape[0]
    if n < 2:
        return 0.0
    # Allele frequencies
    freqs = haps.mean(axis=0)
    # pi = sum over sites of 2*p*(1-p) * n/(n-1)
    pi = np.sum(2 * freqs * (1 - freqs) * n / (n - 1))
    return pi


def pairwise_diffs_between(haps1, haps2):
    """
    Mean pairwise differences between two groups.
    dxy = mean over all (i,j) pairs where i in group1, j in group2
          of the number of sites where they differ.
    """
    n1 = haps1.shape[0]
    n2 = haps2.shape[0]
    if n1 == 0 or n2 == 0:
        return 0.0
    f1 = haps1.mean(axis=0)
    f2 = haps2.mean(axis=0)
    # dxy = sum over sites of (f1 * (1-f2) + f2 * (1-f1))
    dxy = np.sum(f1 * (1 - f2) + f2 * (1 - f1))
    return dxy


def compute_windowed_stats(rep, n_std, n_inv, n_windows=20):
    """
    Compute windowed statistics for a single replicate.

    Args:
        rep: parsed replicate dict
        n_std: number of standard chromosomes (first n_std samples)
        n_inv: number of inverted chromosomes (remaining samples)
        n_windows: number of equal-width windows across [0,1]

    Returns:
        dict with arrays indexed by window:
          'midpoints': window midpoints
          'pi_S': diversity within standard
          'pi_I': diversity within inverted
          'dxy_SI': divergence between S and I
          'FAT': arrangement fixation index
    """
    nsam = n_std + n_inv
    haps = rep['haplotypes']
    positions = rep['positions']

    if haps.shape[0] < nsam or len(positions) == 0:
        return None

    # Split by class
    haps_S = haps[:n_std]
    haps_I = haps[n_std:n_std + n_inv]

    edges = np.linspace(0, 1, n_windows + 1)
    midpoints = (edges[:-1] + edges[1:]) / 2
    win_width = 1.0 / n_windows

    pi_S = np.zeros(n_windows)
    pi_I = np.zeros(n_windows)
    dxy_SI = np.zeros(n_windows)
    pi_total = np.zeros(n_windows)
    FAT = np.zeros(n_windows)

    for w in range(n_windows):
        mask = (positions >= edges[w]) & (positions < edges[w + 1])
        if not np.any(mask):
            continue

        h_S = haps_S[:, mask]
        h_I = haps_I[:, mask]
        h_all = haps[:, mask]

        pi_s = pairwise_diffs(h_S)
        pi_i = pairwise_diffs(h_I)
        dxy = pairwise_diffs_between(h_S, h_I)
        pi_t = pairwise_diffs(h_all)

        pi_S[w] = pi_s / win_width if win_width > 0 else 0
        pi_I[w] = pi_i / win_width if win_width > 0 else 0
        dxy_SI[w] = dxy / win_width if win_width > 0 else 0
        pi_total[w] = pi_t / win_width if win_width > 0 else 0

        # FAT = 1 - T_A / T_T (Guerrero et al.)
        # T_A ~ pi within arrangements (weighted average of pi_S and pi_I)
        # T_T ~ pi total
        if pi_t > 0:
            pi_within = (n_std * pi_s + n_inv * pi_i) / nsam
            FAT[w] = 1.0 - pi_within / pi_t
        else:
            FAT[w] = 0.0

    return {
        'midpoints': midpoints,
        'pi_S': pi_S,
        'pi_I': pi_I,
        'dxy_SI': dxy_SI,
        'pi_total': pi_total,
        'FAT': FAT,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compute summary statistics from msinv output')
    parser.add_argument('input', help='ms-format input file (- for stdin)')
    parser.add_argument('--n_std', type=int, required=True,
                        help='Number of standard chromosomes')
    parser.add_argument('--n_inv', type=int, required=True,
                        help='Number of inverted chromosomes')
    parser.add_argument('--windows', type=int, default=20,
                        help='Number of windows (default: 20)')
    args = parser.parse_args()

    reps = parse_ms_output(args.input)
    if not reps:
        print("No replicates found", file=sys.stderr)
        sys.exit(1)

    # Accumulate stats across replicates
    all_stats = []
    for rep in reps:
        s = compute_windowed_stats(rep, args.n_std, args.n_inv, args.windows)
        if s is not None:
            all_stats.append(s)

    if not all_stats:
        print("No valid replicates", file=sys.stderr)
        sys.exit(1)

    midpoints = all_stats[0]['midpoints']
    n_reps = len(all_stats)

    # Average across replicates
    mean_pi_S = np.mean([s['pi_S'] for s in all_stats], axis=0)
    mean_pi_I = np.mean([s['pi_I'] for s in all_stats], axis=0)
    mean_dxy = np.mean([s['dxy_SI'] for s in all_stats], axis=0)
    mean_FAT = np.mean([s['FAT'] for s in all_stats], axis=0)

    # Print results
    print(f"# msinv_stats: {n_reps} replicates, {args.windows} windows")
    print(f"# n_std={args.n_std}, n_inv={args.n_inv}")
    print(f"{'position':>10} {'pi_S':>12} {'pi_I':>12} {'dxy_SI':>12} {'FAT':>10}")
    for i in range(len(midpoints)):
        print(f"{midpoints[i]:10.4f} {mean_pi_S[i]:12.4f} "
              f"{mean_pi_I[i]:12.4f} {mean_dxy[i]:12.4f} {mean_FAT[i]:10.4f}")


if __name__ == '__main__':
    main()
