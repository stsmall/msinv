#!/bin/bash
# SLiM vs msinv validation with proper burn-in.
# Runs 8*Ne = 80000 generations per rep to ensure coalescent equilibrium.
#
# This validates msinv's standard coalescent engine against forward
# simulation. Inversion-specific features (phi(x), gene flux) are
# validated separately via test_ld.py and test_msinv.py.
#
# Usage: nohup ./run_validation.sh > validation.log 2>&1 &

set -e
cd "$(dirname "$0")"

# Parameters
Ne=10000
L=100000
mu=1e-8
r=1e-8
p_init=0.5
burn_in=80000  # 8*Ne for safe equilibrium
bp_mark=30000
n_samp=10
NREPS=100
N_PARALLEL=6  # concurrent SLiM processes

echo "=== SLiM vs msinv validation (proper burn-in) ==="
echo "Ne=$Ne, L=$L, mu=$mu, r=$r"
echo "Burn-in: $burn_in gen ($(echo "scale=1; $burn_in / $Ne" | bc)*Ne)"
echo "n_samp=$n_samp diploid, NREPS=$NREPS, parallel=$N_PARALLEL"
echo "Started: $(date)"
echo ""

mkdir -p output_slim output_msinv

# ==============================================================
# Part 1: Run SLiM replicates in parallel
# ==============================================================
echo "[1/3] Running SLiM simulations..."
t_start=$(date +%s)

run_slim() {
    local seed=$1
    local outfile="output_slim/rep${seed}.txt"
    if [ -f "$outfile" ]; then
        return
    fi
    slim -d "Ne=$Ne" -d "L=$L" -d "mu=$mu" -d "r=$r" \
         -d "p_init=$p_init" -d "burn_in=$burn_in" \
         -d "bp_mark=$bp_mark" -d "n_samp=$n_samp" \
         -d "seed=$seed" -d "outfile='$outfile'" \
         inversion_sim.slim > "output_slim/rep${seed}.log" 2>&1
}

export -f run_slim
export Ne L mu r p_init burn_in bp_mark n_samp

for rep in $(seq 1 $NREPS); do
    run_slim $rep &
    if (( rep % N_PARALLEL == 0 )); then
        wait
        elapsed=$(($(date +%s) - t_start))
        echo "  Done rep $rep/$NREPS (${elapsed}s elapsed, $(date +%H:%M:%S))"
    fi
done
wait
elapsed=$(($(date +%s) - t_start))
echo "  SLiM done in ${elapsed}s"

# ==============================================================
# Part 2: Run matching msinv simulations
# ==============================================================
echo ""
echo "[2/3] Running msinv simulations..."
t_start=$(date +%s)
python3 << EOF
import sys, os
sys.path.insert(0, '..')
from msinv import MsinvSimulator
import numpy as np

Ne = $Ne
L = $L
mu = $mu
r = $r
n_samp = $n_samp
NREPS = $NREPS

for rep in range(1, NREPS + 1):
    outfile = f'output_msinv/rep{rep}.txt'
    if os.path.exists(outfile):
        continue

    # Standard coalescent, no inversion
    sim = MsinvSimulator(
        samples=2 * n_samp,
        population_size=Ne,
        mutation_rate=mu,
        recombination_rate=r,
        sequence_length=L,
        p_inv=0, c=0,
        seed=rep,
    )
    pos, haps = sim.simulate_one()

    with open(outfile, 'w') as f:
        f.write(f"# msinv output, n={haps.shape[0]}, L={L}\n")
        f.write(f"# seed={rep}\n")
        f.write("positions:")
        for p in pos:
            f.write(f" {int(p * L)}")
        f.write("\n")
        for h in haps:
            f.write("".join(str(int(x)) for x in h) + "\n")

    if rep % 20 == 0:
        print(f"  Done rep {rep}/{NREPS}")

print("  msinv done")
EOF
elapsed=$(($(date +%s) - t_start))
echo "  msinv done in ${elapsed}s"

# ==============================================================
# Part 3: Compare summary statistics
# ==============================================================
echo ""
echo "[3/3] Computing summary statistics..."
python3 << 'EOF' > validation_results.txt
import os
import numpy as np
import glob

def load_output(path):
    with open(path) as f:
        lines = f.readlines()
    positions = None
    haps = []
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        if line.startswith('positions:'):
            parts = line.split()[1:]
            positions = np.array([int(p) for p in parts]) if parts else np.array([])
            continue
        if positions is not None:
            haps.append([int(c) for c in line])
    if positions is None or len(haps) == 0:
        return None, None
    return positions, np.array(haps, dtype=np.int8)


def compute_stats(pos, haps, L):
    """Compute S, pi, Tajima's D, SFS."""
    if haps is None or len(pos) == 0:
        return dict(S=0, pi=0.0, tajD=0.0, sfs=np.zeros(haps.shape[0] if haps is not None else 1),
                    singletons=0, doubletons=0)
    n = haps.shape[0]
    S = haps.shape[1]

    # pi (total pairwise differences)
    pi = 0.0
    for j in range(S):
        k = int(haps[:, j].sum())
        # Contribution to pi: k*(n-k)*2 (unordered pairs that differ)
        pi += 2 * k * (n - k) / (n * (n - 1))

    # SFS (folded not needed for validation)
    sfs = np.zeros(n + 1, dtype=int)
    for j in range(S):
        k = int(haps[:, j].sum())
        sfs[k] += 1

    # Tajima's D
    a1 = sum(1.0/i for i in range(1, n))
    a2 = sum(1.0/(i*i) for i in range(1, n))
    theta_w = S / a1 if a1 > 0 else 0
    b1 = (n + 1) / (3 * (n - 1)) if n > 2 else 0
    b2 = 2 * (n*n + n + 3) / (9 * n * (n - 1)) if n > 2 else 0
    c1 = b1 - 1/a1 if a1 > 0 else 0
    c2 = b2 - (n + 2) / (a1 * n) + a2 / (a1 * a1) if a1 > 0 else 0
    e1 = c1 / a1 if a1 > 0 else 0
    e2 = c2 / (a1*a1 + a2) if (a1*a1 + a2) > 0 else 0
    var_D = e1 * S + e2 * S * (S - 1) if S > 0 else 0
    tajD = (pi - theta_w) / np.sqrt(var_D) if var_D > 0 else 0

    return dict(S=S, pi=pi, tajD=tajD, sfs=sfs,
                singletons=sfs[1] + sfs[n-1] if n > 1 else 0,
                doubletons=sfs[2] + sfs[n-2] if n > 2 else 0)


slim_files = sorted(glob.glob('output_slim/rep*.txt'))
msinv_files = sorted(glob.glob('output_msinv/rep*.txt'))

print(f"SLiM files:  {len(slim_files)}")
print(f"msinv files: {len(msinv_files)}")
print()

L = 100000

def collect(files):
    stats = dict(S=[], pi=[], tajD=[], singletons=[], doubletons=[])
    n_sfs = None
    sfs_sum = None
    for f in files:
        pos, haps = load_output(f)
        if pos is None or haps is None:
            continue
        s = compute_stats(pos, haps, L)
        stats['S'].append(s['S'])
        stats['pi'].append(s['pi'])
        stats['tajD'].append(s['tajD'])
        stats['singletons'].append(s['singletons'])
        stats['doubletons'].append(s['doubletons'])
        if sfs_sum is None:
            sfs_sum = np.zeros_like(s['sfs'], dtype=float)
        sfs_sum += s['sfs']
    return stats, sfs_sum


slim_stats, slim_sfs = collect(slim_files)
msinv_stats, msinv_sfs = collect(msinv_files)

# Expected values
Ne = 10000
mu = 1e-8
theta_expected = 4 * Ne * mu * L  # = 40
n = 20  # 10 diploid = 20 haplosomes
H_n1 = sum(1.0/i for i in range(1, n))
E_S = theta_expected * H_n1
E_pi = theta_expected

print(f"Expected: theta = 4*Ne*mu*L = {theta_expected:.1f}")
print(f"          E[S] = theta * H_{n-1} = {E_S:.1f}")
print(f"          E[pi] = theta = {E_pi:.1f}")
print()

print(f"{'Metric':<15} {'SLiM':>12} {'msinv':>12} {'ratio':>8} {'expected':>12}")
print("-" * 70)
for key, expected in [('S', E_S), ('pi', E_pi), ('tajD', 0.0),
                       ('singletons', None), ('doubletons', None)]:
    s = np.mean(slim_stats[key])
    m = np.mean(msinv_stats[key])
    ratio = m / s if s != 0 else 0
    s_se = np.std(slim_stats[key]) / np.sqrt(len(slim_stats[key]))
    m_se = np.std(msinv_stats[key]) / np.sqrt(len(msinv_stats[key]))
    exp_str = f"{expected:.2f}" if expected is not None else "—"
    print(f"{key:<15} {s:>8.2f}±{s_se:.2f} {m:>8.2f}±{m_se:.2f} {ratio:>8.3f} {exp_str:>12}")

print()
print("Site Frequency Spectrum (counts, folded-unfolded):")
print(f"{'freq':>5} {'SLiM':>10} {'msinv':>10} {'expected':>12}")
# Expected SFS: E[xi_k] = theta/k (unfolded)
for k in range(1, min(len(slim_sfs), len(msinv_sfs))):
    exp = theta_expected / k if k > 0 else 0
    # Normalize by number of reps
    s = slim_sfs[k] / len(slim_files)
    m = msinv_sfs[k] / len(msinv_files)
    print(f"{k:>5} {s:>10.2f} {m:>10.2f} {exp:>12.2f}")

print()
import datetime
print(f"Completed: {datetime.datetime.now()}")
EOF

echo ""
echo "=== Final Results ==="
cat validation_results.txt
echo ""
echo "Files in: $(pwd)"
