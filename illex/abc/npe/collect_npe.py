"""Collect the NPE training table into one npz per flux setting (run on the box, msinv venv).

    PYTHONNOUSERSITE=1 PYTHONPATH=$PWD python illex/abc/npe/collect_npe.py \
        --table results/illex/npe/table --observed results/illex/abc_observed_ctl2.json --out results/illex/npe

Reads B_{flux}_soft_*.jsonl.gz from run_table.py. Parameters go to the coordinates in which the prior is uniform,
so the NPE prior is a box:

  t_inv          U(1e5, 1.5e6)
  u_sel          t_sel / t_inv, U(0, 1)
  log10_p0       U(-3, -1)
  log10_s        U(-6, -1)
  p_star         U(0.3, 0.95)
  log10_gamma    U(-9, -5)           (flux 1 only)

x = the 39 MODEL_KEYS the ABC-SMC fitted. Rows with any non-finite x are dropped (counted). Writes
npe_B_f{flux}.npz with theta, theta_names, lo, hi, x, x_names, x_obs, and the PPC keys for later checks. The npz
holds plain arrays so train_npe.py does not import illex.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path

import numpy as np

from illex.abc.constants import GAMMA_MAX, GAMMA_MIN, PRIOR
from illex.abc.statvec import MODEL_KEYS, ROW_PPC_KEYS


def bounds(flux):
    b = [("t_inv", *PRIOR["t_inv"]), ("u_sel", 0.0, 1.0),
         ("log10_p0", *np.log10(PRIOR["p0"])), ("log10_s", *np.log10(PRIOR["s"])),
         ("p_star", *PRIOR["p_star"])]
    if flux:
        b.append(("log10_gamma", np.log10(GAMMA_MIN), np.log10(GAMMA_MAX)))
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--observed", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st = json.load(open(a.observed))["stats"]
    x_obs = np.array([float(st[k]) for k in MODEL_KEYS])
    assert np.all(np.isfinite(x_obs)), "observed statistics must be finite"
    for flux in (0, 1):
        files = sorted(glob.glob(str(Path(a.table) / f"B_{flux}_soft_*.jsonl.gz")))
        rows = [json.loads(line) for f in files for line in gzip.open(f, "rt")]
        if not rows:
            print(f"flux {flux}: no rows")
            continue
        b = bounds(flux)
        g = lambda k: np.array([float(r[k]) if r[k] is not None else np.nan for r in rows])  # noqa: E731
        cols = {"t_inv": g("t_inv"), "u_sel": g("t_sel") / g("t_inv"), "log10_p0": np.log10(g("p0")),
                "log10_s": np.log10(g("s")), "p_star": g("p_star")}
        if flux:
            cols["log10_gamma"] = np.log10(g("gamma_eff"))
        theta = np.column_stack([cols[n] for n, *_ in b])
        x = np.column_stack([g(k) for k in MODEL_KEYS])
        ppc = np.column_stack([g(k) for k in ROW_PPC_KEYS])
        ok = np.all(np.isfinite(x), axis=1) & np.all(np.isfinite(theta), axis=1)
        lo = np.array([v[1] for v in b])
        hi = np.array([v[2] for v in b])
        inside = np.all((theta >= lo - 1e-9) & (theta <= hi + 1e-9), axis=1)
        assert inside[ok].all(), "a parameter falls outside its prior box"
        np.savez_compressed(out / f"npe_B_f{flux}.npz", theta=theta[ok], theta_names=np.array([n for n, *_ in b]),
                            lo=lo, hi=hi, x=x[ok], x_names=np.array(MODEL_KEYS), x_obs=x_obs,
                            ppc=ppc[ok], ppc_names=np.array(ROW_PPC_KEYS),
                            task_id=g("task_id")[ok], seed=g("seed")[ok])
        print(f"flux {flux}: {len(files)} files, {len(rows)} rows, {int((~ok).sum())} dropped (non-finite), "
              f"{int(ok.sum())} kept -> {out / f'npe_B_f{flux}.npz'}")


if __name__ == "__main__":
    main()
