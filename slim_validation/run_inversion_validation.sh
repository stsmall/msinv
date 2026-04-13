#!/bin/bash
# SLiM inversion simulation vs msinv validation.
# This is the key test for msinv's inversion dynamics.
#
# Kills previous orchestrators, cleans old outputs, runs in background.
#
# Usage: nohup ./run_inversion_validation.sh > inversion_validation.log 2>&1 &

set -e
cd "$(dirname "$0")"

# Parameters (chosen so inversion persists in Ne=10k without selection)
Ne=10000
L=100000
mu=1e-8
r=1e-8
INV_TICK=40000   # burn-in of 4*Ne
END_TICK=80000   # inversion age = 4*Ne
bp_left=30000
bp_right=70000
p_init=0.5
n_samp=10
NREPS=10         # reduced from 30 — SLiM with restarts is too expensive
N_PARALLEL=4
SLIM_TIMEOUT=10800   # 3h per SLiM rep (was 1800s → all reps timed out)

echo "=== SLiM inversion + msinv validation ==="
echo "Ne=$Ne, L=$L, mu=$mu, r=$r"
echo "INV_TICK=$INV_TICK, END_TICK=$END_TICK (inv age = $((END_TICK - INV_TICK)) gen)"
echo "NREPS=$NREPS, parallel=$N_PARALLEL"
echo "Started: $(date)"
echo ""

mkdir -p inversion/output inversion/msinv_output

# ====================================================
# Part 1: SLiM with inversions (with restart-on-extinction)
# ====================================================
echo "[1/3] Running SLiM inversion simulations..."

run_slim_inv() {
    local seed=$1
    local out="inversion/output/rep${seed}.txt"
    if [ -f "$out" ]; then return; fi

    timeout $SLIM_TIMEOUT slim -d "Ne=$Ne" -d "L=$L" -d "mu=$mu" -d "r=$r" \
        -d "INV_TICK=$INV_TICK" -d "END_TICK=$END_TICK" \
        -d "bp_left=$bp_left" -d "bp_right=$bp_right" \
        -d "p_init=$p_init" -d "n_samp=$n_samp" \
        -d "seed=$seed" -d "outfile='$out'" \
        inversion/inversion_full.slim \
        > "inversion/output/rep${seed}.log" 2>&1
}

export -f run_slim_inv
export Ne L mu r INV_TICK END_TICK bp_left bp_right p_init n_samp SLIM_TIMEOUT

t_start=$(date +%s)
for rep in $(seq 1 $NREPS); do
    run_slim_inv $rep &
    if (( rep % N_PARALLEL == 0 )); then
        wait
        echo "  SLiM rep $rep/$NREPS at $(date +%H:%M:%S)"
    fi
done
wait
elapsed=$(($(date +%s) - t_start))
echo "  SLiM done in ${elapsed}s"

# ====================================================
# Part 2: Matching msinv simulations
# ====================================================
echo ""
echo "[2/3] Running msinv simulations..."

python3 << EOF
import sys, os
sys.path.insert(0, '..')
from msinv import MsinvSimulator, ConstantFrequency
import numpy as np

Ne = $Ne; L = $L; mu = $mu; r = $r
bp_left = $bp_left; bp_right = $bp_right
p_init = $p_init
inv_age_gen = $END_TICK - $INV_TICK
n_samp = $n_samp
NREPS = $NREPS

for rep in range(1, NREPS + 1):
    out = f'inversion/msinv_output/rep{rep}.txt'
    if os.path.exists(out):
        continue

    # msinv: n_samp SS + n_samp II = 2*n_samp * 2 = 4*n_samp haplosomes
    # n_std = 2*n_samp (SS haplotypes)
    # n_inv = 2*n_samp (II haplotypes)
    sim = MsinvSimulator(
        samples=4*n_samp,
        population_size=Ne,
        mutation_rate=mu,
        recombination_rate=r,
        sequence_length=L,
        n_std=2*n_samp, n_inv=2*n_samp,
        p_inv=p_init, c=0, gamma=0,
        bp_left=bp_left, bp_right=bp_right,
        t_inv=inv_age_gen,
        seed=rep,
    )
    pos, haps = sim.simulate_one()

    with open(out, 'w') as f:
        f.write(f"# msinv inversion, n_SS_haps={2*n_samp}, n_II_haps={2*n_samp}, L={L}\n")
        f.write(f"# seed={rep}\n")
        f.write("positions:")
        for p in pos:
            f.write(f" {int(p * L)}")
        f.write("\n")
        for h in haps:
            f.write("".join(str(int(x)) for x in h) + "\n")

    if rep % 5 == 0:
        print(f"  msinv rep {rep}/{NREPS}")
EOF
echo "  msinv done"

# ====================================================
# Part 3: Compare
# ====================================================
echo ""
echo "[3/3] Computing comparison..."

python3 << 'EOF' > inversion_results.txt
import os, glob, numpy as np

def load_output(path):
    with open(path) as f:
        lines = f.readlines()
    pos = None; haps = []
    n_ss_haps = None; n_ii_haps = None
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith('#'):
            # Extract haplosome counts
            if 'n_SS_haps' in line:
                for part in line.split(','):
                    if 'n_SS_haps' in part:
                        n_ss_haps = int(part.split('=')[1].strip())
                    if 'n_II_haps' in part:
                        n_ii_haps = int(part.split('=')[1].strip())
            continue
        if line.startswith('positions:'):
            parts = line.split()[1:]
            pos = np.array([int(p) for p in parts]) if parts else np.array([])
            continue
        if pos is not None:
            haps.append([int(c) for c in line])
    if pos is None or not haps: return None, None, None, None
    return pos, np.array(haps, dtype=np.int8), n_ss_haps, n_ii_haps

def region_dxy(haps, pos, gA, gB, lo, hi):
    mask = (pos >= lo) & (pos < hi)
    if mask.sum() == 0: return 0.0
    d = 0
    for j in np.where(mask)[0]:
        for a in gA:
            for b in gB:
                if haps[a, j] != haps[b, j]: d += 1
    return d / (len(gA) * len(gB) * mask.sum())

def region_pi(haps, pos, grp, lo, hi):
    mask = (pos >= lo) & (pos < hi)
    if mask.sum() == 0: return 0.0
    n = len(grp)
    if n < 2: return 0.0
    d = 0
    for j in np.where(mask)[0]:
        for a in range(n):
            for b in range(a+1, n):
                if haps[grp[a], j] != haps[grp[b], j]: d += 1
    return d / (n*(n-1)/2) / mask.sum()

L = 100000
bp_l, bp_r = 30000, 70000

print("=" * 75)
print("SLiM inversion vs msinv — comparison")
print("=" * 75)

for source_dir, label in [('inversion/output', 'SLiM'),
                            ('inversion/msinv_output', 'msinv')]:
    files = sorted(glob.glob(f'{source_dir}/rep*.txt'))
    if not files:
        print(f"\n{label}: no data")
        continue

    pi_SS_inv = []; pi_SS_col = []
    pi_II_inv = []; pi_II_col = []
    dxy_SI_inv = []; dxy_SI_col = []

    for f in files:
        pos, haps, n_ss, n_ii = load_output(f)
        if pos is None or haps is None: continue
        if n_ss is None: continue
        if n_ss < 4 or n_ii < 4: continue  # need enough samples

        ss = list(range(n_ss))
        ii = list(range(n_ss, n_ss + n_ii))

        pi_SS_inv.append(region_pi(haps, pos, ss, bp_l, bp_r))
        pi_SS_col.append(region_pi(haps, pos, ss, 0, bp_l) +
                          region_pi(haps, pos, ss, bp_r, L))
        pi_II_inv.append(region_pi(haps, pos, ii, bp_l, bp_r))
        pi_II_col.append(region_pi(haps, pos, ii, 0, bp_l) +
                          region_pi(haps, pos, ii, bp_r, L))
        dxy_SI_inv.append(region_dxy(haps, pos, ss, ii, bp_l, bp_r))
        dxy_SI_col.append(region_dxy(haps, pos, ss, ii, 0, bp_l) +
                           region_dxy(haps, pos, ss, ii, bp_r, L))

    print(f"\n{label} (n_reps = {len(pi_SS_inv)})")
    if len(pi_SS_inv) > 0:
        print(f"  pi_SS  inv={np.mean(pi_SS_inv):.4f} col={np.mean(pi_SS_col):.4f}  "
              f"ratio={np.mean(pi_SS_inv)/np.mean(pi_SS_col) if np.mean(pi_SS_col) else 0:.2f}")
        print(f"  pi_II  inv={np.mean(pi_II_inv):.4f} col={np.mean(pi_II_col):.4f}  "
              f"ratio={np.mean(pi_II_inv)/np.mean(pi_II_col) if np.mean(pi_II_col) else 0:.2f}")
        print(f"  dxy_SI inv={np.mean(dxy_SI_inv):.4f} col={np.mean(dxy_SI_col):.4f}  "
              f"ratio={np.mean(dxy_SI_inv)/np.mean(dxy_SI_col) if np.mean(dxy_SI_col) else 0:.2f}")

print("\n" + "=" * 75)
print("Expected pattern:")
print("  pi_SS within inversion similar to collinear (SS recombines normally)")
print("  pi_II within inversion similar to collinear (II recombines normally)")
print("  dxy_SI inside inversion >> collinear (recombination blocked)")
print("=" * 75)

import datetime
print(f"\nCompleted: {datetime.datetime.now()}")
EOF

echo ""
echo "=== RESULTS ==="
cat inversion_results.txt
echo ""
echo "Done: $(date)"
