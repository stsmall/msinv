"""Observed collinear control statistics on a REPRESENTATIVE chr2 background: [40, 55) + [85, 115) Mb.

The ABC control (observed.py, [10, 30) Mb) overlaps a pi-elevated block (chr2:24.1-35.9 Mb by the 1 Mb rolling
pi > 2x median rule, pi 0.013 in every karyotype group against 0.006-0.007 elsewhere on chr2, coindetii
alignability 0.57 against 0.79-0.91). This recomputes meta["control"] on 40-55 + 85-115 Mb (pi 0.006-0.007,
alignability 0.79-0.85) with observed.py's own functions, unchanged: called genotypes (pi_ratio, dxy_pi, ld_*,
the LD genotype error model) from a callset extracted exactly as ctl.vcf.gz was, and ANGSD realSFS within-class
spectra (f1_ratio, sfs_*, tajd_*) from the same SAFs, the two intervals' spectra summed.

    bcftools view -r 2:40000001-55000000,2:85000001-115000000 -S samples.txt --force-samples -m2 -M2 -v snps \
        variants_filt.vcf.gz | bcftools annotate -x INFO,^FORMAT/GT -Oz -o .tmp/abc_control2/ctl2.vcf.gz
    realSFS {AA,BB}.2.saf.idx -r <interval> -fold 0  ->  .tmp/abc_control2/{AA,BB}.{r4055,r85115}.sfs
    PYTHONPATH=$PWD <pg_gpu python> illex/abc/smc_results/baseline/observed_control2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import illex.abc.observed as O
from illex.abc import statvec as SV
from illex.abc.error_model import genotype_error_stats, h_from_ho_he
from illex.abc.statvec import ld_block_sums

W = Path(".tmp/abc_control2")
INTERVALS = ((40_000_000, 55_000_000), (85_000_000, 115_000_000))
OUT = Path("results/illex/smc_summary/baseline/control_representative.json")


def main():
    O.WORK = W
    O.VCF["control"] = W / "ctl2.vcf.gz"
    O.CTL_LO, O.CTL_HI = INTERVALS[0][0], INTERVALS[-1][1]   # the callset holds only the two intervals
    names = {c: O._read_samples(f) for c, f in O.KARYO.items()}
    pos, dos, samples = O.load_region("control")
    inside = np.zeros(pos.size, bool)
    for lo, hi in INTERVALS:
        inside |= (pos - 1 >= lo) & (pos - 1 < hi)
    col = {s: i for i, s in enumerate(samples)}
    idx = {c: np.array([col[s] for s in names[c]]) for c in ("I", "S")}
    acc = O.accessible(pos) & inside
    keep = acc & ((dos[idx["I"]] >= 0).any(0) | (dos[idx["S"]] >= 0).any(0))
    ctl = {}
    st = O.site_stats(dos, idx["I"], idx["S"])
    ctl.update(O._ratios({k: v[keep].sum() for k, v in st.items()}))

    xp = O._xp()
    subs = O.ld_subsamples()
    err = {}
    for c in ("I", "S"):
        cs, cc, cr, hh = [], [], [], []
        for sub in subs[c]:
            d = dos[idx[c][sub]][:, keep].T
            s, n = ld_block_sums(d, pos[keep], np.zeros(int(keep.sum()), int), 1, xp=xp)
            cs.append(s[0]), cc.append(n[0])
            k, _, _ = SV.ld_site_keep(d)
            a, b = genotype_error_stats(d[k])
            cr.append(a), hh.append(b)
        with np.errstate(invalid="ignore", divide="ignore"):
            cont = np.nanmean(np.array(cs) / np.array(cc), axis=0)
        for k in range(len(SV.LD_EDGES) - 1):
            ctl[f"ld_{c}_{k + 1}"] = float(cont[k])
        ho_he = float(np.mean(hh))
        err[c] = {"call_rate": float(np.mean(cr)), "ho_he": ho_he, "h": h_from_ho_he(ho_he)}

    unf = {c: sum(np.array((W / f"{O.CLS_FILE[c]}.{lab}.sfs").read_text().split(), float)
                  for lab in ("r4055", "r85115")) for c in ("I", "S")}
    a = O._angsd_stats(unf["I"], unf["S"])
    for k in ("f1_ratio",) + SV.SFS_KEYS + ("tajd_I", "tajd_S"):
        ctl[k] = a[k]
    old = json.load(open("results/illex/abc_observed.json"))["meta"]
    res = {"intervals": INTERVALS, "control": {k: ctl[k] for k in old["control"]},
           "ld_error_model": err, "old_control": old["control"], "old_ld_error_model": old["ld_error_model"]["control"],
           "variants": {"vcf": int(pos.size), "accessible_used": int(keep.sum())},
           "angsd_sites": {c: float(unf[c].sum()) for c in unf}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"{'stat':10s} {'10-30 Mb':>10s} {'40-55+85-115':>13s}")
    for k in old["control"]:
        print(f"{k:10s} {old['control'][k]:10.4f} {ctl[k]:13.4f}")
    print("LD error model old", old["ld_error_model"]["control"], "\nnew", err)


if __name__ == "__main__":
    main()
