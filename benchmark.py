#!/usr/bin/env python3
"""
Benchmark: baseline Python vs optimized Python vs C-accelerated.

Runs identical simulations with each version and compares:
  - Speed (ms per replicate)
  - Correctness (mean segregating sites)
"""

import sys
import time as tm
import importlib.util
import numpy as np

base = '/home/ssmall/inversion_sims/files'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load versions
baseline = load('baseline', f'{base}/msinv_baseline.py')
python = load('python', f'{base}/msinv_python.py')
fast = load('fast', f'{base}/msinv_fast.py')

NR = 10

# Parameters: n=10, rho=100 (many recombination events)
params_baseline = dict(nsam=10, nreps=1, theta=20.0, rho=100.0, nsites=1000,
                       n_std=5, n_inv=5, p_inv=0.5, c=0.01, seed=0)

params_full = dict(nsam=10, nreps=1, theta=20.0, rho=100.0, nsites=1000,
                   n_std=5, n_inv=5, p_inv=0.5, c=0.01, t_inv=10.0,
                   bp_left=0.3, bp_right=0.7)

# Warmup fast (Numba + C)
print("Warming up...")
s = fast.MsinvSimulatorFast(**{**params_full, 'seed': 0})
s.simulate_one()

print(f"\nBenchmark: {NR} replicates, n=10 (5S+5I), rho=50\n")

results = {}

# 1. Baseline (inversion-only, pre-optimization)
print("Running baseline (inversion-only, pre-optimization)...")
try:
    times_b = []
    sites_b = []
    for i in range(NR):
        sim = baseline.MsinvSimulator(**{**params_baseline, 'seed': 42+i})
        t0 = tm.time()
        pos, haps = sim.simulate_one()
        times_b.append(tm.time() - t0)
        sites_b.append(len(pos))
    results['baseline'] = (np.mean(times_b)*1000, np.median(times_b)*1000,
                           np.mean(sites_b))
    print(f"  {np.mean(times_b)*1000:.1f} ms/rep (median {np.median(times_b)*1000:.1f}), "
          f"mean sites={np.mean(sites_b):.0f}")
except Exception as e:
    print(f"  FAILED: {e}")
    results['baseline'] = (999, 999, 0)

# 2. Optimized Python (full chromosome)
print("Running optimized Python (full chromosome)...")
times_p = []
sites_p = []
for i in range(NR):
    sim = python.MsinvSimulator(**{**params_full, 'seed': 42+i})
    t0 = tm.time()
    pos, haps = sim.simulate_one()
    times_p.append(tm.time() - t0)
    sites_p.append(len(pos))
results['python'] = (np.mean(times_p)*1000, np.median(times_p)*1000,
                     np.mean(sites_p))
print(f"  {np.mean(times_p)*1000:.1f} ms/rep (median {np.median(times_p)*1000:.1f}), "
      f"mean sites={np.mean(sites_p):.0f}")

# 3. C-accelerated (full chromosome)
print("Running C-accelerated (full chromosome)...")
times_f = []
sites_f = []
for i in range(NR):
    sim = fast.MsinvSimulatorFast(**{**params_full, 'seed': 42+i})
    t0 = tm.time()
    pos, haps = sim.simulate_one()
    times_f.append(tm.time() - t0)
    sites_f.append(len(pos))
results['fast_c'] = (np.mean(times_f)*1000, np.median(times_f)*1000,
                     np.mean(sites_f))
print(f"  {np.mean(times_f)*1000:.1f} ms/rep (median {np.median(times_f)*1000:.1f}), "
      f"mean sites={np.mean(sites_f):.0f}")

# Summary
print(f"\n{'='*55}")
print(f"{'Version':>20} {'Mean ms':>10} {'Median ms':>10} {'Speedup':>10}")
print(f"{'-'*55}")
ref = results['python'][0]
for name, (mean, median, sites) in results.items():
    speedup = ref / mean if mean > 0 else 0
    print(f"{name:>20} {mean:>10.1f} {median:>10.1f} {speedup:>9.1f}x")

# Check C availability
try:
    import smc_bridge
    print(f"\nC library: {'loaded' if smc_bridge.is_available() else 'not found'}")
except ImportError:
    print("\nC library: not available")
