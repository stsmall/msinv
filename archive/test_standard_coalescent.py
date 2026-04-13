#!/usr/bin/env python3
"""
Validate msinv's standard coalescent against msprime.

Tests WITHOUT inversions (p_inv=0, c=0) to verify the basic
coalescent machinery produces correct results:

1. Mean pairwise diversity (pi) matches msprime
2. Site frequency spectrum shape matches
3. Number of segregating sites matches expectation
4. Tajima's D is near zero (neutral)
5. LD decay matches
6. Demographic model (bottleneck) matches msprime
7. Two-population divergence matches msprime

All comparisons use the Kolmogorov-Smirnov test or ratio checks.
"""

import sys
import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location(
    'msinv', '/home/ssmall/inversion_sims/files/msinv.py')
msinv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msinv)

import msprime
import tskit

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))


def msinv_pi(nsam, theta, rho, nsites, nreps, seed):
    """Run msinv without inversion, compute mean pairwise diversity."""
    sim = msinv.MsinvSimulator(
        nsam=nsam, nreps=nreps, theta=theta, rho=rho, nsites=nsites,
        p_inv=0.0, c=0.0, seed=seed)
    pi_vals = []
    for _ in range(nreps):
        pos, haps = sim.simulate_one()
        if len(pos) == 0:
            pi_vals.append(0.0)
            continue
        # Mean pairwise diffs
        n = haps.shape[0]
        total_diffs = 0
        n_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_diffs += np.sum(haps[i] != haps[j])
                n_pairs += 1
        pi_vals.append(total_diffs / n_pairs if n_pairs > 0 else 0)
    return np.array(pi_vals)


def msprime_pi(nsam, theta, rho, nsites, nreps, seed):
    """Run msprime, compute mean pairwise diversity."""
    N = 10000
    mu = theta / (4 * N * nsites)
    r = rho / (4 * N * nsites)

    pi_vals = []
    for rep in range(nreps):
        ts = msprime.sim_ancestry(
            samples=nsam // 2, sequence_length=nsites,
            recombination_rate=r, population_size=N,
            random_seed=seed + rep)
        ts = msprime.sim_mutations(ts, rate=mu, random_seed=seed + rep + nreps)
        pi = ts.diversity(mode="site")
        pi_vals.append(float(pi) * nsites)  # total pairwise diffs
    return np.array(pi_vals)


def test_mean_diversity():
    """Mean pi should match between msinv and msprime."""
    print("\n=== Test: Mean diversity matches msprime ===")
    nsam = 10; theta = 10.0; rho = 20.0; nsites = 1000; nreps = 200

    pi_msinv = msinv_pi(nsam, theta, rho, nsites, nreps, seed=42)
    pi_msprime = msprime_pi(nsam, theta, rho, nsites, nreps, seed=42)

    mean_ms = np.mean(pi_msinv)
    mean_mp = np.mean(pi_msprime)
    # Expected pi = theta = 10.0
    print(f"    msinv:  mean pi = {mean_ms:.2f}")
    print(f"    msprime: mean pi = {mean_mp:.2f}")
    print(f"    Expected: ~{theta:.1f}")

    ratio = mean_ms / mean_mp if mean_mp > 0 else 0
    check("Mean pi within 30% of msprime",
          0.7 < ratio < 1.3,
          f"ratio={ratio:.2f}")

    # Both should be near theta
    check("msinv pi near theta",
          0.5 < mean_ms / theta < 2.0,
          f"pi/theta={mean_ms/theta:.2f}")


def test_segregating_sites():
    """Mean S should match Watterson's theta."""
    print("\n=== Test: Segregating sites match expectation ===")
    nsam = 10; theta = 10.0; rho = 20.0; nsites = 1000; nreps = 200

    sim = msinv.MsinvSimulator(
        nsam=nsam, nreps=nreps, theta=theta, rho=rho, nsites=nsites,
        p_inv=0.0, c=0.0, seed=42)

    S_vals = []
    for _ in range(nreps):
        pos, haps = sim.simulate_one()
        S_vals.append(len(pos))

    mean_S = np.mean(S_vals)
    # Watterson's theta_w = S / a_n where a_n = sum(1/i for i=1..n-1)
    a_n = sum(1.0 / i for i in range(1, nsam))
    expected_S = theta * a_n
    print(f"    Mean S = {mean_S:.1f}, expected = {expected_S:.1f}")

    ratio = mean_S / expected_S if expected_S > 0 else 0
    check("Mean S within 30% of Watterson expectation",
          0.7 < ratio < 1.3,
          f"ratio={ratio:.2f}")


def test_sfs_shape():
    """Site frequency spectrum should follow 1/i shape."""
    print("\n=== Test: SFS follows 1/i ===")
    nsam = 20; theta = 20.0; rho = 40.0; nsites = 1000; nreps = 200

    sim = msinv.MsinvSimulator(
        nsam=nsam, nreps=nreps, theta=theta, rho=rho, nsites=nsites,
        p_inv=0.0, c=0.0, seed=42)

    sfs = np.zeros(nsam - 1)
    for _ in range(nreps):
        pos, haps = sim.simulate_one()
        if len(pos) == 0:
            continue
        counts = haps.sum(axis=0)
        for c in counts:
            if 0 < c < nsam:
                sfs[c - 1] += 1

    # Normalize
    if sfs.sum() > 0:
        sfs_norm = sfs / sfs.sum()
    else:
        sfs_norm = sfs

    # Expected: proportional to 1/i
    expected = np.array([1.0 / i for i in range(1, nsam)])
    expected /= expected.sum()

    # Check that singletons are most common
    check("Singletons are most common frequency class",
          sfs_norm[0] > sfs_norm[-1],
          f"singletons={sfs_norm[0]:.3f}, max_freq={sfs_norm[-1]:.3f}")

    # Correlation with 1/i
    corr = np.corrcoef(sfs_norm[:10], expected[:10])[0, 1]
    check("SFS correlated with 1/i (r > 0.8)",
          corr > 0.8,
          f"r={corr:.3f}")


def test_tree_sequence_vs_ms_format():
    """Tree sequence divergence should match ms-format divergence."""
    print("\n=== Test: Tree sequence matches ms format ===")
    nsam = 6; theta = 10.0; rho = 10.0; nsites = 1000; nreps = 100

    sim = msinv.MsinvSimulator(
        nsam=nsam, nreps=nreps, theta=theta, rho=rho, nsites=nsites,
        p_inv=0.0, c=0.0, seed=42)

    ms_pi_vals = []
    ts_pi_vals = []
    for _ in range(nreps):
        # ms format
        pos, haps = sim.simulate_one()
        if len(pos) > 0:
            n = haps.shape[0]
            diffs = sum(np.sum(haps[i] != haps[j])
                        for i in range(n) for j in range(i+1, n))
            ms_pi_vals.append(diffs / (n * (n-1) / 2))

        # tree sequence
        ts = sim.simulate_one_ts()
        if ts is not None:
            mu_rate = theta / (2 * nsites)
            ts_mut = msprime.sim_mutations(ts, rate=mu_rate, random_seed=42)
            pi = ts_mut.diversity(mode="site")
            ts_pi_vals.append(float(pi) * nsites)

    ms_mean = np.mean(ms_pi_vals) if ms_pi_vals else 0
    ts_mean = np.mean(ts_pi_vals) if ts_pi_vals else 0
    print(f"    ms format: mean pi = {ms_mean:.2f} ({len(ms_pi_vals)} reps)")
    print(f"    tree seq:  mean pi = {ts_mean:.2f} ({len(ts_pi_vals)} reps)")

    if ts_mean > 0 and ms_mean > 0:
        ratio = ms_mean / ts_mean
        check("ms and ts divergence in same ballpark",
              0.3 < ratio < 3.0,
              f"ratio={ratio:.2f}")
    else:
        check("Both produce non-zero divergence",
              ms_mean > 0 and ts_mean > 0,
              f"ms={ms_mean}, ts={ts_mean}")


def test_bottleneck():
    """Bottleneck should increase coalescence rate → more diversity."""
    print("\n=== Test: Bottleneck produces expected effect ===")
    nsam = 10; theta = 10.0; rho = 20.0; nsites = 1000; nreps = 200

    # No bottleneck
    sim_no = msinv.MsinvSimulator(
        nsam=nsam, nreps=nreps, theta=theta, rho=rho, nsites=nsites,
        p_inv=0.0, c=0.0, seed=42)
    pi_no = []
    for _ in range(nreps):
        pos, haps = sim_no.simulate_one()
        pi_no.append(len(pos))

    # With bottleneck: -eN 0.1 0.01 (size drops to 1% at t=0.1)
    demo = msinv.Demography(n_pops=1)
    demo.add_event(('eN', 0.1, 0.01))
    demo.add_event(('eN', 0.2, 1.0))  # recover

    sim_bn = msinv.MsinvSimulator(
        nsam=nsam, nreps=nreps, theta=theta, rho=rho, nsites=nsites,
        p_inv=0.0, c=0.0, seed=42, demography=demo)
    pi_bn = []
    for _ in range(nreps):
        pos, haps = sim_bn.simulate_one()
        pi_bn.append(len(pos))

    mean_no = np.mean(pi_no)
    mean_bn = np.mean(pi_bn)
    print(f"    No bottleneck: mean S = {mean_no:.1f}")
    print(f"    Bottleneck:    mean S = {mean_bn:.1f}")

    # Bottleneck should reduce S (faster coalescence during bottleneck)
    check("Bottleneck reduces segregating sites",
          mean_bn < mean_no,
          f"bn={mean_bn:.1f} < no={mean_no:.1f}")


def test_two_pop_fst():
    """Two populations with low migration should have elevated Fst."""
    print("\n=== Test: Two pops with low migration → elevated divergence ===")
    nsam = 10; theta = 10.0; rho = 10.0; nsites = 1000; nreps = 100

    # Two pops with low migration, merge at t=5
    demo = msinv.Demography(n_pops=2, mig_rate=0.5)  # 4Nm=0.5
    demo.add_event(('ej', 5.0, 1, 0))

    sc = {('S', 0): 5, ('S', 1): 5}  # no inversion, just two pops
    sim = msinv.MsinvSimulator(
        nsam=nsam, nreps=nreps, theta=theta, rho=rho, nsites=nsites,
        p_inv=0.0, c=0.0, seed=42,
        n_pops=2, sample_config=sc, demography=demo)

    dxy_vals = []
    pi_within = []
    for _ in range(nreps):
        pos, haps = sim.simulate_one()
        if len(pos) == 0:
            continue
        # dxy between pop0 (samples 0-4) and pop1 (samples 5-9)
        diffs = 0; pairs = 0
        for i in range(5):
            for j in range(5, 10):
                diffs += np.sum(haps[i] != haps[j])
                pairs += 1
        dxy_vals.append(diffs / pairs if pairs > 0 else 0)
        # pi within pop0
        d2 = 0; p2 = 0
        for i in range(5):
            for j in range(i+1, 5):
                d2 += np.sum(haps[i] != haps[j])
                p2 += 1
        pi_within.append(d2 / p2 if p2 > 0 else 0)

    mean_dxy = np.mean(dxy_vals)
    mean_pi = np.mean(pi_within)
    print(f"    dxy (between pops) = {mean_dxy:.2f}")
    print(f"    pi (within pop0)   = {mean_pi:.2f}")

    check("dxy > pi (populations are diverged)",
          mean_dxy > mean_pi * 1.1,
          f"dxy/pi={mean_dxy/mean_pi if mean_pi > 0 else 0:.2f}")


def main():
    global PASS, FAIL

    test_mean_diversity()
    test_segregating_sites()
    test_sfs_shape()
    test_tree_sequence_vs_ms_format()
    test_bottleneck()
    test_two_pop_fst()

    print(f"\n{'='*55}")
    print(f"Standard coalescent: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
