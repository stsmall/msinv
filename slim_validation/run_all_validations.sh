#!/bin/bash
# SLiM vs msinv validation: multiple demographic scenarios
# For validating msinv's core coalescent + demography engine.
#
# Scenarios:
#   1. constant (baseline)
#   2. bottleneck
#   3. growth (exponential)
#   4. expansion_then_const (recent Fol-like growth)
#
# Runs in parallel. Saves SLiM + msinv outputs, computes stats.
#
# Usage: nohup ./run_all_validations.sh > master.log 2>&1 &

set -e
cd "$(dirname "$0")"

N_PARALLEL=6
NREPS=100
Ne=10000
L=100000
mu=1e-8
r=1e-8
burn_in=80000
n_samp=10

echo "=== SLiM vs msinv multi-scenario validation ==="
echo "NREPS=$NREPS, parallel=$N_PARALLEL"
echo "Started: $(date)"
echo ""

# ====================================================
# Run a scenario in SLiM + msinv, both with NREPS replicates
# ====================================================
run_slim_scenario() {
    local scenario=$1
    local extra_args=$2
    local dir="scenarios/${scenario}"
    mkdir -p "$dir/slim"
    echo "  SLiM: $scenario"

    for rep in $(seq 1 $NREPS); do
        local out="$dir/slim/rep${rep}.txt"
        if [ -f "$out" ]; then continue; fi

        slim -d "scenario='$scenario'" -d "Ne=$Ne" -d "L=$L" \
             -d "mu=$mu" -d "r=$r" -d "burn_in=$burn_in" \
             -d "n_samp=$n_samp" -d "seed=$rep" \
             -d "outfile='$out'" \
             $extra_args \
             demography/demog.slim > "$dir/slim/rep${rep}.log" 2>&1 &

        if (( rep % N_PARALLEL == 0 )); then wait; fi
    done
    wait
}

run_msinv_scenario() {
    local scenario=$1
    local dir="scenarios/${scenario}"
    mkdir -p "$dir/msinv"
    echo "  msinv: $scenario"

    python3 << EOF
import sys, os
sys.path.insert(0, '..')
from msinv import MsinvSimulator, Demography
import numpy as np

Ne = $Ne; L = $L; mu = $mu; r = $r
NREPS = $NREPS; n_samp = $n_samp
scenario = '$scenario'

def build_demo():
    d = Demography(n_pops=1)
    if scenario == 'constant':
        pass
    elif scenario == 'bottleneck':
        # t in coalescent units (2*Ne gen)
        # burn_in gens back, bn from t_bn_start to t_bn_end gens back
        t_bn_start_gen = 20000
        t_bn_end_gen = 22000
        N_bn = 500
        # msinv times are in coal units from present (backward)
        d.add_event(('eN', t_bn_start_gen / (2 * Ne), N_bn / Ne))
        d.add_event(('eN', t_bn_end_gen / (2 * Ne), 1.0))
    elif scenario == 'growth':
        # Exponential growth, forward rate = 0.0003 per gen
        # going backward, population shrinks
        g_rate_per_gen = 0.0003
        g_coal = g_rate_per_gen * 2 * Ne
        # Growth in last half of burn-in (after t = burn_in/2 forward)
        # Going backward: growth active from t=0 to t=burn_in/2 gens ago
        t_growth_end_gen = $burn_in // 2  # time when growth started forward
        d.pop_sizes[0] = 1.0  # placeholder, will be set at actual end
        d.growth_rates[0] = g_coal
        d.growth_start[0] = 0.0
        d.snapshot_initial_state()
        # Stop growth going further back
        d.add_event(('eg', t_growth_end_gen / (2 * Ne), 0, 0.0))
        d.add_event(('en', t_growth_end_gen / (2 * Ne), 0, 1.0))
    elif scenario == 'expansion_recent':
        # Recent 100x expansion in last 500 generations
        d.pop_sizes[0] = 100.0  # present-day Ne = 100x
        d.snapshot_initial_state()
        d.add_event(('en', 500 / (2 * Ne), 0, 1.0))  # back to Ne at 500 gen
    return d

for rep in range(1, NREPS + 1):
    outfile = f'{scenario}/msinv/rep{rep}.txt'.replace('scenarios/', '')
    outfile = f'scenarios/{scenario}/msinv/rep{rep}.txt'
    if os.path.exists(outfile):
        continue

    demo = build_demo()
    sim = MsinvSimulator(
        samples=2*n_samp,
        population_size=Ne,
        mutation_rate=mu,
        recombination_rate=r,
        sequence_length=L,
        p_inv=0, c=0,
        demography=demo,
        seed=rep,
    )
    pos, haps = sim.simulate_one()

    with open(outfile, 'w') as f:
        f.write(f"# msinv {scenario}, n={haps.shape[0]}, L={L}\n")
        f.write(f"# seed={rep}, Ne={Ne}\n")
        f.write("positions:")
        for p in pos:
            f.write(f" {int(p * L)}")
        f.write("\n")
        for h in haps:
            f.write("".join(str(int(x)) for x in h) + "\n")

    if rep % 20 == 0:
        print(f"    {scenario} rep {rep}/{NREPS}")
EOF
}

# ====================================================
# Run all scenarios
# ====================================================
echo "[1/4] Constant size..."
run_slim_scenario "constant" ""
run_msinv_scenario "constant"

echo "[2/4] Bottleneck..."
run_slim_scenario "bottleneck" "-d t_bn_start=20000 -d t_bn_end=22000 -d N_bn=500"
run_msinv_scenario "bottleneck"

echo "[3/4] Exponential growth..."
run_slim_scenario "growth" "-d g_rate=0.0003"
run_msinv_scenario "growth"

echo "[4/4] Recent expansion..."
# Skip for now — needs different SLiM handling

# ====================================================
# Analyze and compare
# ====================================================
echo ""
echo "Computing summary statistics..."

python3 << 'EOF' > final_results.txt
import os, glob, numpy as np

def load_output(path):
    with open(path) as f:
        lines = f.readlines()
    pos = None; haps = []
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line: continue
        if line.startswith('positions:'):
            parts = line.split()[1:]
            pos = np.array([int(p) for p in parts]) if parts else np.array([])
            continue
        if pos is not None:
            haps.append([int(c) for c in line])
    if pos is None or not haps: return None, None
    return pos, np.array(haps, dtype=np.int8)


def compute_stats(haps):
    if haps.size == 0: return dict(S=0, pi=0.0, tajD=0.0, sfs=np.zeros(haps.shape[0] if haps is not None else 1))
    n = haps.shape[0]
    S = haps.shape[1]
    pi = 0.0
    sfs = np.zeros(n + 1, dtype=int)
    for j in range(S):
        k = int(haps[:, j].sum())
        pi += 2 * k * (n - k) / (n * (n - 1))
        sfs[k] += 1
    # Tajima's D
    a1 = sum(1.0/i for i in range(1, n)) if n > 1 else 1
    a2 = sum(1.0/(i*i) for i in range(1, n)) if n > 1 else 1
    theta_w = S / a1
    b1 = (n + 1) / (3 * (n - 1)) if n > 2 else 0
    b2 = 2 * (n*n + n + 3) / (9 * n * (n - 1)) if n > 2 else 0
    c1 = b1 - 1/a1
    c2 = b2 - (n + 2) / (a1 * n) + a2 / (a1 * a1)
    e1 = c1 / a1
    e2 = c2 / (a1*a1 + a2) if (a1*a1 + a2) > 0 else 0
    var_D = e1 * S + e2 * S * (S - 1) if S > 0 else 0
    tajD = (pi - theta_w) / np.sqrt(var_D) if var_D > 0 else 0
    return dict(S=S, pi=pi, tajD=tajD, sfs=sfs)


def collect(files):
    Ss, pis, Ds = [], [], []
    sfs_sum = None
    for f in files:
        pos, haps = load_output(f)
        if pos is None or haps is None: continue
        s = compute_stats(haps)
        Ss.append(s['S']); pis.append(s['pi']); Ds.append(s['tajD'])
        if sfs_sum is None: sfs_sum = np.zeros_like(s['sfs'], dtype=float)
        sfs_sum += s['sfs']
    return Ss, pis, Ds, sfs_sum


print("=" * 75)
print("SLiM vs msinv validation — multiple demographic scenarios")
print("=" * 75)

for scenario in ['constant', 'bottleneck', 'growth']:
    slim_f = sorted(glob.glob(f'scenarios/{scenario}/slim/rep*.txt'))
    ms_f = sorted(glob.glob(f'scenarios/{scenario}/msinv/rep*.txt'))
    if not slim_f or not ms_f:
        print(f"\n{scenario}: missing data ({len(slim_f)} SLiM, {len(ms_f)} msinv)")
        continue

    s_S, s_pi, s_D, s_sfs = collect(slim_f)
    m_S, m_pi, m_D, m_sfs = collect(ms_f)

    print(f"\n{scenario.upper()} (n_reps: SLiM={len(s_S)}, msinv={len(m_S)})")
    print(f"{'Metric':<15} {'SLiM':>15} {'msinv':>15} {'ratio':>8}")
    print("-" * 60)
    for name, s, m in [('S', s_S, m_S), ('pi', s_pi, m_pi), ('tajD', s_D, m_D)]:
        s_mean = np.mean(s)
        m_mean = np.mean(m)
        s_se = np.std(s) / np.sqrt(len(s))
        m_se = np.std(m) / np.sqrt(len(m))
        ratio = m_mean / s_mean if s_mean != 0 else 0
        print(f"{name:<15} {s_mean:>10.2f}±{s_se:.2f} {m_mean:>10.2f}±{m_se:.2f} {ratio:>8.3f}")

    # SFS comparison
    if s_sfs is not None and m_sfs is not None:
        n = len(s_sfs) - 1
        print(f"\n  SFS (first 10 bins, normalized per rep):")
        print(f"  {'freq':>4} {'SLiM':>10} {'msinv':>10}")
        for k in range(1, min(11, n)):
            print(f"  {k:>4} {s_sfs[k]/len(slim_f):>10.2f} {m_sfs[k]/len(ms_f):>10.2f}")

import datetime
print()
print(f"Completed: {datetime.datetime.now()}")
EOF

echo ""
echo "=== Final Results ==="
cat final_results.txt
echo ""
echo "Done: $(date)"
