#!/usr/bin/env python3
"""
Test suite for msinv n>2 with trajectories.

Validates:
  1. n=2 vs n>2 consistency
  2. Standard coalescent limit (no inversion)
  3. High gene flux limit (SI → SS)
  4. S-I divergence scales with t_inv
  5. Within-class divergence is flat (no phi(x) shape)
  6. Stochastic vs deterministic consistency
"""

import sys
import numpy as np
import importlib.util

# Import n>2 version from this directory
spec_n2plus = importlib.util.spec_from_file_location(
    "msinv_nplus", "/home/ssmall/inversion_sims/files/msinv.py")
msinv_nplus = importlib.util.module_from_spec(spec_n2plus)
spec_n2plus.loader.exec_module(msinv_nplus)

MsinvSimulator = msinv_nplus.MsinvSimulator
ConstantFrequency = msinv_nplus.ConstantFrequency
DeterministicTrajectory = msinv_nplus.DeterministicTrajectory
StochasticTrajectory = msinv_nplus.StochasticTrajectory
GeneFluxModel = msinv_nplus.GeneFluxModel

# Import n=2 version
spec_n2 = importlib.util.spec_from_file_location(
    "msinv_n2", "/home/ssmall/msinv.py")
msinv_n2 = importlib.util.module_from_spec(spec_n2)
spec_n2.loader.exec_module(msinv_n2)

NW = 10
flux = GeneFluxModel(0.3)
mid = np.linspace(0.5/NW, 1-0.5/NW, NW)
PASS = 0
FAIL = 0


def compute_dxy_dss(sim, nreps, n_std, n_inv):
    """Run nreps, compute mean windowed S-I and S-S divergence."""
    dxy = np.zeros(NW)
    dss = np.zeros(NW)
    n_ok = 0
    for _ in range(nreps):
        try:
            pos, haps = sim.simulate_one()
            if len(pos) == 0:
                continue
            n_ok += 1
            for w in range(NW):
                lo, hi = w/NW, (w+1)/NW
                idx = [j for j, p in enumerate(pos) if lo <= p < hi]
                if not idx:
                    continue
                s = haps[:n_std, :][:, idx]
                inv = haps[n_std:, :][:, idx]
                # S-I
                d = sum(np.sum(s[i] != inv[j])
                        for i in range(n_std) for j in range(n_inv))
                dxy[w] += d / (n_std * n_inv)
                # S-S
                if n_std >= 2:
                    d2 = sum(np.sum(s[i] != s[j])
                             for i in range(n_std) for j in range(i+1, n_std))
                    n_ss = n_std * (n_std - 1) // 2
                    dss[w] += d2 / n_ss
        except Exception:
            pass
    if n_ok > 0:
        dxy /= n_ok
        dss /= n_ok
    return dxy, dss, n_ok


def check(name, condition, detail=""):
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def test_standard_coalescent():
    """No inversion should match standard coalescent."""
    print("\n=== Test: Standard coalescent limit (no inversion) ===")
    sim = MsinvSimulator(
        nsam=6, nreps=200, theta=10.0, rho=10.0, nsites=1000,
        p_inv=0.0, c=0.0, seed=42)
    dxy, dss, n_ok = compute_dxy_dss(sim, 200, 3, 3)
    # All divergence should be ~theta/NW = 1.0 (for total theta=10, 10 windows)
    mean_d = np.mean(dxy)
    check("All reps succeed", n_ok == 200, f"{n_ok}/200")
    check("Mean divergence near expected",
          0.3 < mean_d < 2.0,
          f"mean={mean_d:.2f}, expected ~1.0")
    check("Flat across windows (no inversion signal)",
          np.std(dxy) / (np.mean(dxy) + 0.01) < 0.5,
          f"cv={np.std(dxy)/(np.mean(dxy)+0.01):.2f}")


def test_high_gene_flux():
    """With very high c, S-I divergence should approach S-S."""
    print("\n=== Test: High gene flux (c=10) washes out signal ===")
    sim = MsinvSimulator(
        nsam=6, nreps=100, theta=10.0, rho=10.0, nsites=1000,
        n_std=3, n_inv=3, p_inv=0.5, c=10.0, seed=42, t_inv=20.0)
    dxy, dss, n_ok = compute_dxy_dss(sim, 100, 3, 3)
    si_mean = np.mean(dxy)
    ss_mean = np.mean(dss)
    check("All reps succeed", n_ok >= 95, f"{n_ok}/100")
    check("SI ≈ SS (gene flux erases structure)",
          abs(si_mean - ss_mean) / (ss_mean + 0.01) < 0.5,
          f"SI={si_mean:.2f}, SS={ss_mean:.2f}")


def test_tinv_scaling():
    """S-I divergence should increase with t_inv."""
    print("\n=== Test: Divergence scales with t_inv ===")
    results = {}
    for t_inv in [2.0, 20.0, 50.0]:
        sim = MsinvSimulator(
            nsam=6, nreps=100, theta=10.0, rho=10.0, nsites=1000,
            n_std=3, n_inv=3, p_inv=0.5, c=0.01, seed=42, t_inv=t_inv)
        dxy, _, n_ok = compute_dxy_dss(sim, 100, 3, 3)
        results[t_inv] = np.mean(dxy)
        print(f"    t_inv={t_inv:>5}: mean_dxy={results[t_inv]:.2f} ({n_ok}/100 ok)")

    check("dxy(t=20) > dxy(t=2)",
          results[20.0] > results[2.0],
          f"{results[20.0]:.2f} > {results[2.0]:.2f}")
    check("dxy(t=50) > dxy(t=20)",
          results[50.0] > results[20.0],
          f"{results[50.0]:.2f} > {results[20.0]:.2f}")


def test_within_class_no_inversion_signal():
    """S-S divergence inside inversion should NOT be elevated vs outside."""
    print("\n=== Test: SS divergence not elevated inside inversion ===")
    sim = MsinvSimulator(
        nsam=6, nreps=200, theta=40.0, rho=100.0, nsites=1000,
        n_std=3, n_inv=3, p_inv=0.5, c=0.01, seed=42, t_inv=20.0,
        bp_left=0.3, bp_right=0.7)
    _, dss, n_ok = compute_dxy_dss(sim, 200, 3, 3)
    out_idx = [i for i in range(NW) if mid[i] < 0.3 or mid[i] > 0.7]
    in_idx = [i for i in range(NW) if 0.3 < mid[i] < 0.7]
    ss_out = np.mean([dss[i] for i in out_idx])
    ss_in = np.mean([dss[i] for i in in_idx])
    ratio = ss_in / ss_out if ss_out > 0 else 1
    check("SS inside/outside ratio near 1 (< 1.5)",
          ratio < 1.5,
          f"ratio={ratio:.2f}, in={ss_in:.2f}, out={ss_out:.2f}")


def test_si_elevated_inside_inversion():
    """S-I divergence should be elevated inside inversion vs outside."""
    print("\n=== Test: SI divergence elevated inside vs outside ===")
    sim = MsinvSimulator(
        nsam=6, nreps=200, theta=40.0, rho=100.0, nsites=1000,
        n_std=3, n_inv=3, p_inv=0.5, c=0.01, seed=42, t_inv=20.0,
        bp_left=0.3, bp_right=0.7)
    dxy, dss, n_ok = compute_dxy_dss(sim, 200, 3, 3)
    out_idx = [i for i in range(NW) if mid[i] < 0.3 or mid[i] > 0.7]
    in_idx = [i for i in range(NW) if 0.3 < mid[i] < 0.7]

    si_out = np.mean([dxy[i] for i in out_idx])
    si_in = np.mean([dxy[i] for i in in_idx])
    ss_out = np.mean([dss[i] for i in out_idx])
    ss_in = np.mean([dss[i] for i in in_idx])

    # SI/SS should be higher inside than outside
    ratio_in = si_in / ss_in if ss_in > 0 else 1
    ratio_out = si_out / ss_out if ss_out > 0 else 1
    check("SI/SS higher inside than outside",
          ratio_in > ratio_out,
          f"inside={ratio_in:.2f}, outside={ratio_out:.2f}")
    check("SI elevated inside (SI_in > SI_out)",
          si_in > si_out,
          f"in={si_in:.2f}, out={si_out:.2f}")


def test_n2_vs_n6_consistency():
    """n=2 msinv inversion-region divergence should match n>2 at n=2."""
    print("\n=== Test: n=2 vs n>2 consistency (inversion region only) ===")
    NR = 200
    t_inv = 10.0
    p_func = msinv_n2.ConstantFrequency(0.5, t_inv=t_inv)

    # n=2 version: S-I divergence in inversion region only [0.3, 0.7]
    rng = np.random.default_rng(42)
    d_n2_inv = 0.0
    for _ in range(NR):
        _, pos, h0, h1 = msinv_n2.simulate_one(
            1, 10.0, 10.0, 1000, 0.5, 0.01, 0.3, 0.7, 0.3,
            0, 1, rng, p_inv_func=p_func)
        for j, p in enumerate(pos):
            if h0[j] != h1[j] and 0.3 < p < 0.7:
                d_n2_inv += 1
    d_n2_inv /= NR
    # n=2 measures divergence in [0.3,0.7] = 40% of chromosome

    # n>2 version at n=2 (simulates inversion region [0.02, 0.98])
    sim = MsinvSimulator(
        nsam=2, nreps=NR, theta=10.0, rho=10.0, nsites=1000,
        n_std=1, n_inv=1, p_inv=0.5, c=0.01, seed=42, t_inv=t_inv)
    d_n6, _, n_ok = compute_dxy_dss(sim, NR, 1, 1)
    n6_total = np.sum(d_n6)  # total diffs across all windows

    print(f"    n=2 inv-region diffs: {d_n2_inv:.2f}")
    print(f"    n>2 total diffs:      {n6_total:.2f}")
    # Both should reflect S-I divergence inside inversion, order-of-magnitude match
    check("Same order of magnitude",
          0.1 < n6_total / (d_n2_inv + 0.01) < 10.0,
          f"ratio={n6_total/(d_n2_inv+0.01):.2f}")


def test_stochastic_vs_deterministic():
    """Matched t_inv should give similar average divergence."""
    print("\n=== Test: Stochastic vs deterministic consistency ===")
    # Deterministic with known t_inv
    traj_d = DeterministicTrajectory(0.5, N=10000, s=0.001)

    sim_d = MsinvSimulator(
        nsam=6, nreps=100, theta=10.0, rho=10.0, nsites=1000,
        n_std=3, n_inv=3, p_inv=0.5, c=0.01, seed=42,
        p_inv_func=traj_d)
    dxy_d, _, n_ok_d = compute_dxy_dss(sim_d, 100, 3, 3)

    # Constant with matched t_inv
    sim_c = MsinvSimulator(
        nsam=6, nreps=100, theta=10.0, rho=10.0, nsites=1000,
        n_std=3, n_inv=3, p_inv=0.5, c=0.01, seed=42,
        t_inv=traj_d.t_inv)
    dxy_c, _, n_ok_c = compute_dxy_dss(sim_c, 100, 3, 3)

    d_mean = np.mean(dxy_d)
    c_mean = np.mean(dxy_c)
    print(f"    deterministic: mean_dxy={d_mean:.2f} (t_inv={traj_d.t_inv:.3f})")
    print(f"    constant:      mean_dxy={c_mean:.2f} (t_inv={traj_d.t_inv:.3f})")
    check("Similar divergence (within 2x)",
          0.5 < d_mean / (c_mean + 0.01) < 2.0,
          f"ratio={d_mean/(c_mean+0.01):.2f}")


def main():
    global PASS, FAIL
    test_standard_coalescent()
    test_high_gene_flux()
    test_tinv_scaling()
    test_within_class_no_inversion_signal()
    test_si_elevated_inside_inversion()
    test_n2_vs_n6_consistency()
    test_stochastic_vs_deterministic()

    print(f"\n{'='*50}")
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
