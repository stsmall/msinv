"""Compare SLiM and msinv inversion statistics per scenario.

Pass criterion per statistic (as in the msprime neutral validation, Track 3):
two-sample Kolmogorov-Smirnov p > 0.01 and |Cohen's d| < 0.2.
  python -m illex.slim_bench.compare_bench results/slim_bench
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

STATS = ["T_II", "T_SS", "T_IS", "T_flank", "pi_I_over_pi_S", "dxy_over_pi_I"]


def main(d):
    recs = defaultdict(list)
    for f in sorted(Path(d).glob("*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r.get("status") != "FAILED":
                recs[r["scenario"]].append(r)
    out = {}
    for sc, rs in sorted(recs.items()):
        print(f"\n== {sc}: {len(rs)} replicates, mean p_final {np.mean([r['p_final'] for r in rs]):.3f}")
        print(f"{'stat':>15} {'SLiM':>10} {'msinv':>10} {'ratio':>7} {'KS p':>8} {'d':>7}  verdict")
        out[sc] = {}
        for s in STATS:
            a = np.array([r["slim"][s] for r in rs])
            b = np.array([r["msinv"][s] for r in rs])
            ks = stats.ks_2samp(a, b).pvalue
            sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            dd = (b.mean() - a.mean()) / sd if sd > 0 else 0.0
            ok = ks > 0.01 and abs(dd) < 0.2
            out[sc][s] = dict(slim=float(a.mean()), msinv=float(b.mean()), ks_p=float(ks),
                              d=float(dd), passed=bool(ok))
            print(f"{s:>15} {a.mean():10.4g} {b.mean():10.4g} {b.mean() / a.mean():7.3f} "
                  f"{ks:8.3g} {dd:+7.3f}  {'PASS' if ok else 'FAIL'}")
    json.dump(out, open(Path(d) / "comparison.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/slim_bench")
