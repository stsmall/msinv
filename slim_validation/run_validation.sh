#!/bin/bash
# SLiM vs msinv validation - runs autonomously in background
# All output written to files; no interactive required.
#
# Usage: nohup ./run_validation.sh > validation.log 2>&1 &

set -e
cd "$(dirname "$0")"

# Parameters (coalescent equivalent for msinv)
Ne=10000
L=100000
mu=1e-8
r=1e-8
p_init=0.5
t_inv_gen=20000
bp_left=30000
bp_right=70000
n_samp=10
NREPS=50  # number of replicate pairs

echo "=== SLiM vs msinv validation ==="
echo "Ne=$Ne, L=$L, mu=$mu, r=$r"
echo "Inversion: bp_left=$bp_left, bp_right=$bp_right, t_inv=$t_inv_gen gen"
echo "n_samp=$n_samp diploid, NREPS=$NREPS"
echo "Started: $(date)"
echo ""

mkdir -p output_slim output_msinv

# ==============================================================
# Part 1: Run SLiM replicates in parallel (4 at a time)
# ==============================================================
echo "[1/3] Running SLiM simulations..."
run_slim() {
    local seed=$1
    local outfile="output_slim/rep${seed}.txt"
    if [ -f "$outfile" ]; then
        echo "  skipping rep $seed (exists)"
        return
    fi
    slim -d "Ne=$Ne" -d "L=$L" -d "mu=$mu" -d "r=$r" \
         -d "p_init=$p_init" -d "t_inv=$t_inv_gen" \
         -d "bp_left=$bp_left" -d "bp_right=$bp_right" \
         -d "n_samp=$n_samp" -d "seed=$seed" \
         -d "outfile='$outfile'" \
         inversion_sim.slim > "output_slim/rep${seed}.log" 2>&1
}

export -f run_slim
export Ne L mu r p_init t_inv_gen bp_left bp_right n_samp

# Run in parallel batches of 4
for rep in $(seq 1 $NREPS); do
    run_slim $rep &
    if (( rep % 4 == 0 )); then
        wait
        echo "  Completed rep $rep/$NREPS at $(date +%H:%M:%S)"
    fi
done
wait
echo "  SLiM done"

# ==============================================================
# Part 2: Run matching msinv simulations
# ==============================================================
echo ""
echo "[2/3] Running msinv simulations..."
python3 << EOF
import sys, os
sys.path.insert(0, '..')
from msinv import MsinvSimulator, ConstantFrequency
import numpy as np

Ne = $Ne
L = $L
mu = $mu
r = $r
p_init = $p_init
t_inv_gen = $t_inv_gen
bp_left = $bp_left
bp_right = $bp_right
n_samp = $n_samp
NREPS = $NREPS

# Convert to coalescent units
t_inv = t_inv_gen / (2 * Ne)
bp_l_frac = bp_left / L
bp_r_frac = bp_right / L

for rep in range(1, NREPS + 1):
    outfile = f'output_msinv/rep{rep}.txt'
    if os.path.exists(outfile):
        continue

    sim = MsinvSimulator(
        samples=2 * n_samp,  # diploid → haplosomes
        population_size=Ne,
        mutation_rate=mu,
        recombination_rate=r,
        sequence_length=L,
        n_std=n_samp,  # rough split — actual freq will differ
        n_inv=n_samp,
        p_inv=p_init,
        c=0.01,
        gamma=0.0,  # neutral flux for this test
        t_inv=t_inv_gen,  # will be converted to coal units
        bp_left=bp_l_frac,
        bp_right=bp_r_frac,
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

    if rep % 10 == 0:
        print(f"  Completed rep {rep}/{NREPS}")

print("  msinv done")
EOF

# ==============================================================
# Part 3: Compare statistics
# ==============================================================
echo ""
echo "[3/3] Computing summary statistics..."
python3 << 'EOF' > validation_results.txt
import os
import numpy as np
import glob

def load_output(path):
    """Parse SLiM or msinv output → (positions, haplotype_matrix)."""
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
            positions = np.array([int(p) for p in parts])
            continue
        if positions is not None:
            haps.append([int(c) for c in line])
    if positions is None or len(haps) == 0:
        return None, None
    return positions, np.array(haps, dtype=np.int8)

def pi_in_window(pos, haps, left, right):
    mask = (pos >= left) & (pos < right)
    if mask.sum() == 0:
        return 0.0
    n = haps.shape[0]
    pi = 0.0
    for j in np.where(mask)[0]:
        for a in range(n):
            for b in range(a+1, n):
                if haps[a, j] != haps[b, j]:
                    pi += 1
    return pi / (n * (n - 1) / 2)

slim_files = sorted(glob.glob('output_slim/rep*.txt'))
msinv_files = sorted(glob.glob('output_msinv/rep*.txt'))

L = 100000
bp_left = 30000
bp_right = 70000

print(f"SLiM files: {len(slim_files)}")
print(f"msinv files: {len(msinv_files)}")
print()
print(f"{'Metric':<30} {'SLiM':>10} {'msinv':>10} {'ratio':>8}")
print("-" * 60)

# Collect stats
for label, files in [('SLiM', slim_files), ('msinv', msinv_files)]:
    pi_total = []
    pi_inv = []
    pi_col = []
    S_total = []
    for f in files:
        pos, haps = load_output(f)
        if pos is None:
            continue
        pi_total.append(pi_in_window(pos, haps, 0, L))
        pi_inv.append(pi_in_window(pos, haps, bp_left, bp_right))
        pi_col.append(pi_in_window(pos, haps, 0, bp_left) +
                      pi_in_window(pos, haps, bp_right, L))
        S_total.append(len(pos))

    if label == 'SLiM':
        slim_stats = dict(pi_total=np.mean(pi_total),
                          pi_inv=np.mean(pi_inv),
                          pi_col=np.mean(pi_col),
                          S=np.mean(S_total))
    else:
        msinv_stats = dict(pi_total=np.mean(pi_total),
                           pi_inv=np.mean(pi_inv),
                           pi_col=np.mean(pi_col),
                           S=np.mean(S_total))

for metric in ['S', 'pi_total', 'pi_inv', 'pi_col']:
    s = slim_stats[metric]
    m = msinv_stats[metric]
    r = m / s if s > 0 else 0
    print(f"{metric:<30} {s:>10.2f} {m:>10.2f} {r:>8.2f}")

print()
print(f"Completed at: {os.popen('date').read().strip()}")
EOF

echo ""
echo "=== Done ==="
cat validation_results.txt
echo ""
echo "All files in: $(pwd)"
