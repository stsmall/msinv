"""Concatenate the ppc_blocks.py slices of each flux into ppc_blocks_f{0,1}.csv and score them with ppc_analyze.py.

    PYTHONNOUSERSITE=1 PYTHONPATH=<karyofix tree> python illex/abc/npe/collect_ppc_blocks.py OUTROOT

OUTROOT holds the array outputs (blocks_f<flux>_t<task>.csv). Reports missing draws, then writes ppc_blocks_f*.csv and
ppc_per_stat.csv / ppc_joint.csv in OUTROOT (ppc_analyze writes into the working directory).
"""
import glob
import os
import sys

import pandas as pd

from illex.abc.smc_results.ppc import ppc_analyze

root = sys.argv[1]
os.chdir(root)
done = []
for flux in (0, 1):
    parts = sorted(glob.glob(f"blocks_f{flux}_t*.csv"))
    if not parts:
        continue
    d = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).sort_values("particle")
    n = int(d.n_distinct.iloc[0])
    miss = sorted(set(range(n)) - set(d.particle))
    print(f"flux {flux}: {len(parts)} slices, {len(d)} draws, {int(d.ok.sum())} simulated, missing {len(miss)}"
          + (f" (first {miss[:10]})" if miss else "")
          + f"; wall per draw median {d.wall_s.median() / 60:.1f} min, max {d.wall_s.max() / 60:.1f} min")
    d.to_csv(f"ppc_blocks_f{flux}.csv", index=False)
    done.append(f"ppc_blocks_f{flux}.csv")
ppc_analyze.main(done)
print(pd.read_csv("ppc_joint.csv").to_string(index=False))
