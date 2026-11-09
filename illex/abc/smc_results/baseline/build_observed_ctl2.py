"""results/illex/abc_observed_ctl2.json: the observed ABC vector re-divided by the representative control.

Every control-divided statistic (observed.DIVIDED: the ratio, SFS, LD and Tajima's D statistics) becomes the
unchanged inversion value (raw) over the 40-55 + 85-115 Mb control (control_representative.json, written by
observed_control2.py); the JSFS and edge_centre are not divided and are unchanged. The collinear pi per site of the
new control, from the summed within-class ANGSD spectra, is stored as meta["control_pi"] (the refit's pi target).

    PYTHONPATH=$PWD python illex/abc/smc_results/baseline/build_observed_ctl2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from illex.abc import statvec as SV
from illex.abc.observed import CLS_FILE, DIVIDED, N_DIP

BASE = Path("results/illex")
CTRL = BASE / "smc_summary" / "baseline" / "control_representative.json"
W = Path(".tmp/abc_control2")


def pi_per_site(unf, n):
    k = np.arange(n + 1)
    return float((unf * k * (n - k)).sum() / (n * (n - 1) / 2.0) / unf.sum())


def main():
    obs = json.load(open(BASE / "abc_observed.json"))
    ctrl = json.load(open(CTRL))
    new = ctrl["control"]
    pis = {}
    for c in ("I", "S"):
        unf = sum(np.array((W / f"{CLS_FILE[c]}.{lab}.sfs").read_text().split(), float) for lab in ("r4055", "r85115"))
        pis[c] = pi_per_site(unf, 2 * N_DIP[c])
    out = json.loads(json.dumps(obs))
    for k in SV.STAT_KEYS:
        if k in DIVIDED:
            out["stats"][k] = SV._div(obs["raw"][k], new[k])
            out["se"][k] = obs["se"][k] * abs(obs["meta"]["control"][k] / new[k])   # body SE, control held fixed
    out["meta"]["control"] = new
    out["meta"]["control_pi"] = {**pis, "mean": 0.5 * (pis["I"] + pis["S"])}
    out["meta"]["regions"]["control"] = {"span": ctrl["intervals"], "note": "representative collinear chr2"}
    out["meta"]["ld_error_model"]["control"] = ctrl["ld_error_model"]
    (BASE / "abc_observed_ctl2.json").write_text(json.dumps(out, indent=2) + "\n")
    print("control pi per site", out["meta"]["control_pi"])
    for k in SV.STAT_KEYS:
        if k in DIVIDED:
            print(f"  {k:10s} old {obs['stats'][k]:9.4f}  new {out['stats'][k]:9.4f}")


if __name__ == "__main__":
    main()
