#!/usr/bin/env python
"""Choose the six Task 9 SLiM-validation parameter sets and write results/illex/abc_slim_sets.json.

Sets: soft D1, hard D1, hard B, soft B, D2 with flux, N soft. Provenance rules (each set records its own):

  soft_D1   Task 8b ABC-SMC pilot (soft D1, no flux) posterior medians, abc_smc_pilot/abc_smc_posteriors.csv.
  soft_B, N_soft, D2_flux
            Task 8 ABC-RF v3 joint posterior draws (abc_rf_v3/abc_post_{model}_0.csv, soft rows), unique source
            rows ranked by their resampling weight (the number of draws of that row): the highest-weight row
            whose per-path window acceptance P(win) = P(|p_now - P_NOW| <= P_TOL) is >= PWIN_MIN, else the row
            with the largest P(win). P(win) from N_ACC simulate_paths draws (unfiltered, fixed seed). The window
            is applied to the SLiM side after the run, so a point that rarely lands in it costs ~1 / P(win) SLiM
            replicates per accepted one; the rule keeps the full run affordable without leaving the posterior.
  D2_flux   the D2 no-flux point above with gamma_eff = GAMMA_PLACEHOLDER (1e-7, the log-midpoint of the
            [GAMMA_MIN, GAMMA_MAX] prior, and the Task 5 c* of the hard B / soft N paths). PROVISIONAL: no flux
            posterior exists yet.
  hard_D1, hard_B
            the hard class has ~0 posterior mass (abc_posteriors.csv post_hard_mass <= 4e-5, no hard joint draws),
            so the candidates are the accepted HARD rows of the v3 table (abc_pilot_noflux_v3/abc_table.parquet),
            ranked by the observed-SE-scaled RMS distance to abc_observed.json over the MODEL_KEYS with a finite
            SE. The nearest row with Q >= Q_MIN (cost) whose P(win | survival) >= PWIN_MIN with >= N_SURV_MIN
            survivors (simulate_paths, max_attempts = 20000, i.e. conditioned on establishment, as SLiM's
            restart-on-loss conditions on survival).

Q = min(200, 10 * floor(SQ_TARGET / (10 s))) so s Q <= SQ_TARGET = 0.09 < 0.1 (N: 200).

  PYTHONPATH=$PWD .venv/bin/python illex/scripts/choose_slim_sets.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from illex.abc import statvec as SV
from illex.abc.constants import GAMMA_MAX, GAMMA_MIN, P_NOW, P_TOL
from illex.abc.slim_vs_msinv import path_params
from illex.abc.trajectory import simulate_paths

RES = Path("results/illex")
PWIN_MIN = 0.05
N_ACC = 4000
N_HARD = 2000
N_SURV_MIN = 20
Q_MIN = 50.0
Q_MAX = 200.0
SQ_TARGET = 0.09
GAMMA_PLACEHOLDER = 1e-7
TOP_K = 12
KEYS = ("model", "hard", "t_inv", "p0", "t_sel", "s", "t_stop", "p_star", "gamma_eff")


def q_for(model, s):
    if model == "N" or not math.isfinite(s):
        return Q_MAX
    return float(min(Q_MAX, 10.0 * math.floor(SQ_TARGET / (10.0 * s))))


def _clean(row: dict) -> dict:
    out = {}
    for k in KEYS:
        v = row.get(k)
        if k == "model":
            out[k] = str(v)
        elif k == "hard":
            out[k] = bool(v) if not isinstance(v, str) else v.lower() == "true"
        else:
            out[k] = None if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v) else float(v)
    if out["hard"]:
        out["p0"] = None
        if out["model"] != "N":
            out["t_sel"] = out["t_inv"]
    if out["model"] != "D2":
        out["t_stop"] = None
    if out["model"] != "B":
        out["p_star"] = None
    if out["model"] == "N":
        out["t_sel"] = out["s"] = None
    return out


def window_acceptance(pset: dict, n: int, seed: int = 1, max_attempts: int = 1) -> dict:
    paths, counts = simulate_paths(path_params(pset, n), np.random.default_rng(seed), apply_filter=False,
                                   max_attempts=max_attempts)
    pn = np.array([p.p_now for p in paths])
    w = int((np.abs(pn - P_NOW) <= P_TOL).sum())
    return dict(n_draws=n, n_survived=int(pn.size), n_window=w, p_win=w / n,
                p_win_given_surv=(w / pn.size) if pn.size else float("nan"))


def soft_from_draws(model: str) -> tuple[dict, dict]:
    f = RES / f"abc_rf_v3/abc_post_{model}_0.csv"
    d = pd.read_csv(f, comment="#")
    d = d[~d["hard"].astype(str).str.lower().eq("true")]
    g = d.groupby("src").agg(weight=("t_inv", "size")).sort_values("weight", ascending=False, kind="stable")
    first = d.drop_duplicates("src").set_index("src")
    tried = []
    for src, w in g["weight"].items():
        r = _clean(first.loc[src].to_dict())
        acc = window_acceptance(r, N_ACC)
        tried.append(dict(src=int(src), weight=int(w), **acc))
        if acc["p_win"] >= PWIN_MIN:
            return r, dict(source=str(f), rule=f"highest-weight soft draw with P(win) >= {PWIN_MIN}",
                           src_row=int(src), weight=int(w), n_draws_total=int(len(d)), candidates=tried, **acc)
    best = max(tried, key=lambda t: t["p_win"])
    r = _clean(first.loc[best["src"]].to_dict())
    return r, dict(source=str(f), rule=f"no draw reaches P(win) >= {PWIN_MIN}: the draw with the largest P(win)",
                   src_row=best["src"], weight=best["weight"], n_draws_total=int(len(d)),
                   candidates=sorted(tried, key=lambda t: -t["p_win"])[:TOP_K],
                   **{k: best[k] for k in ("n_draws", "n_survived", "n_window", "p_win", "p_win_given_surv")})


def hard_from_table(model: str) -> tuple[dict, dict]:
    obs = json.loads((RES / "abc_observed.json").read_text())
    t = pd.read_parquet(RES / "abc_pilot_noflux_v3/abc_table.parquet")
    keys = [k for k in SV.MODEL_KEYS if np.isfinite(obs["se"].get(k) or np.nan) and obs["se"][k] > 0]
    z = ((t[keys] - pd.Series({k: obs["stats"][k] for k in keys})) / pd.Series({k: obs["se"][k] for k in keys})) ** 2
    t = t.assign(dist=np.sqrt(z.mean(axis=1, skipna=True)))
    cand = t[(t["model"] == model) & t["hard"]].sort_values("dist")
    tried = []
    for idx, row in cand.iterrows():
        r = _clean(row.to_dict())
        q = q_for(model, r["s"] if r["s"] is not None else float("nan"))
        if q < Q_MIN:
            tried.append(dict(row=int(idx), dist=float(row["dist"]), s=r["s"], Q=q, skipped=f"Q < {Q_MIN}"))
            continue
        acc = window_acceptance(r, N_HARD, max_attempts=20_000)
        tried.append(dict(row=int(idx), dist=float(row["dist"]), s=r["s"], Q=q, **acc))
        if acc["n_survived"] >= N_SURV_MIN and acc["p_win_given_surv"] >= PWIN_MIN:
            return r, dict(source=str(RES / "abc_pilot_noflux_v3/abc_table.parquet"),
                           rule=(f"nearest accepted hard row (observed-SE RMS distance over {len(keys)} MODEL_KEYS) "
                                 f"with Q >= {Q_MIN} and P(win | survival) >= {PWIN_MIN} "
                                 f"(>= {N_SURV_MIN} survivors)"),
                           table_row=int(idx), dist=float(row["dist"]), dist_rank=len(tried),
                           candidates=tried, **acc)
        if len(tried) >= 3 * TOP_K:
            break
    raise RuntimeError(f"no hard {model} row passes: {tried}")


def main() -> None:
    sets, prov = {}, {}
    smc = pd.read_csv(RES / "abc_smc_pilot/abc_smc_posteriors.csv")
    med = {r.parameter: float(r.median) for r in smc.itertuples() if r.model == "D1"}
    sets["soft_D1"] = _clean(dict(model="D1", hard=False, gamma_eff=None, **med))
    prov["soft_D1"] = dict(source=str(RES / "abc_smc_pilot/abc_smc_posteriors.csv"),
                           rule="Task 8b ABC-SMC soft D1 no-flux pilot: marginal posterior medians",
                           **window_acceptance(sets["soft_D1"], N_ACC))
    for name, model in (("soft_B", "B"), ("N_soft", "N"), ("D2_flux", "D2")):
        sets[name], prov[name] = soft_from_draws(model)
        print(name, prov[name]["rule"], prov[name]["p_win"], flush=True)
    sets["D2_flux"]["gamma_eff"] = GAMMA_PLACEHOLDER
    prov["D2_flux"]["gamma_eff"] = (f"PLACEHOLDER {GAMMA_PLACEHOLDER:g}: log-midpoint of the prior "
                                    f"[{GAMMA_MIN:g}, {GAMMA_MAX:g}] (= the Task 5 c* of hard B / soft N); "
                                    "no flux posterior yet. P(win) above is the no-flux path acceptance (paths "
                                    "carry no flux).")
    for name, model in (("hard_D1", "D1"), ("hard_B", "B")):
        sets[name], prov[name] = hard_from_table(model)
        print(name, prov[name]["table_row"], prov[name]["p_win_given_surv"], flush=True)
    order = ("soft_D1", "hard_D1", "hard_B", "soft_B", "D2_flux", "N_soft")
    out = {"note": ("Task 9 SLiM-validation parameter sets (illex/scripts/choose_slim_sets.py). Q keeps s Q < 0.1, "
                    "Q <= 200. p_win = P(|p_now - 0.626| <= 0.02) per drawn path; p_win_given_surv is the SLiM "
                    "acceptance (SLiM restarts on loss/fixation, then the window is applied)."),
           "sets": {}}
    for name in order:
        s = sets[name]
        s["Q"] = q_for(s["model"], s["s"] if s["s"] is not None else float("nan"))
        s["provisional"] = name == "D2_flux"
        s["provenance"] = prov[name]
        out["sets"][name] = s
    (RES / "abc_slim_sets.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", RES / "abc_slim_sets.json")


if __name__ == "__main__":
    main()
