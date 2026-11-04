"""Task 9: SLiM validation of the trajectory-ABC simulator.

For each fixed parameter set (``results/illex/abc_slim_sets.json``) the forward model ``illex/slim/inversion_abc.slim``
(``python -m illex.slim.run_one --params ... --set NAME``) and the ABC simulator (paths from
``trajectory.simulate_paths`` with the same parameters, statistics from ``simulate.simulate_stats`` on the engine
``simulate.engine_for`` picks: msprime for a no-flux set, msinv for a flux set) are compared statistic by
statistic.

  * Both sides are filtered on the same window |p_now - P_NOW| <= P_TOL. SLiM conditions only on segregation,
    so the window is applied to its final inversion frequency here; the simulator's paths are drawn with the
    same window (``simulate_paths(apply_filter=True)``), so on that side the filter is a no-op, applied anyway.
  * Per statistic (``STAT_KEYS`` + ``BND_KEYS``): a two-sample KS test (``scipy.stats.ks_2samp``) and the mean
    difference in units of the pooled SD, delta = (mean_slim - mean_sim) / sd_pooled. PASS when
    |delta| < DELTA_MAX = 0.25, FAIL otherwise, NA when either side has fewer than 2 finite values or the pooled
    SD is 0. The KS p-value is reported, not used in the verdict. Disagreements are reported, not tuned.
  * The boundary keys need ``simulate.boundary_ppc`` (msinv) on the simulator side, which is gated by
    ``simulate.BND_PPC_READY`` until the msinv barrier coalescence fix (PR #19) and the I x S flank-merge fix are
    in this branch. While it is closed the simulator's boundary keys are NaN and their verdict is NA.

Usage:
  python -m illex.abc.slim_vs_msinv simulate --sets results/illex/abc_slim_sets.json --set soft_D1 \\
      --reps 100 --out results/illex/abc_slim_validation_runs/sim_soft_D1.tsv --workers 8
  python -m illex.abc.slim_vs_msinv compare --sets results/illex/abc_slim_sets.json \\
      --slim 'results/illex/abc_slim_validation_runs/slim_*.tsv' \\
      --sim 'results/illex/abc_slim_validation_runs/sim_*.tsv' --out results/illex/abc_slim_validation
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from illex.abc import statvec as SV
from illex.abc.constants import P_NOW, P_TOL

DELTA_MAX = 0.25
HARD_MAX_ATTEMPTS = 20_000   # establishment attempts per hard draw (iid for a fixed theta, as SLiM's restarts)
SET_SEED_STRIDE = 10_000_000  # seed offset per set index (both sides), so sets never share controls or pairings
COMPARE_KEYS = tuple(SV.STAT_KEYS) + tuple(SV.BND_KEYS)
PATH_KEYS = ("t_inv", "p0", "t_sel", "s", "t_stop", "p_star", "gamma_eff")


# ------------------------------------------------------------------ the comparison ----
def in_window(p_now, p_target: float = P_NOW, tol: float = P_TOL) -> np.ndarray:
    """|p_now - p_target| <= tol (NaN is outside)."""
    p = np.asarray(p_now, float)
    with np.errstate(invalid="ignore"):
        return np.abs(p - p_target) <= tol


def _ok_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that finished (status ok, when a status column exists) and fall in the p_now window."""
    keep = np.ones(len(df), bool)
    if "status" in df.columns:
        keep &= (df["status"].astype(str) == "ok").to_numpy()
    keep &= in_window(df["p_now"].to_numpy(float))
    return df.loc[keep]


def null_fail_prob(n1: int, n2: int, delta_max: float = DELTA_MAX) -> float:
    """P(|delta| >= delta_max) when both sides come from one normal distribution: delta / sqrt(1/n1 + 1/n2) is
    Student t with n1 + n2 - 2 df. The expected FAIL count of a table with no real difference is the sum of this
    over its non-NA statistics (correlated statistics: an expectation, not a binomial count)."""
    from scipy.stats import t as student_t

    if n1 < 2 or n2 < 2:
        return float("nan")
    return float(2.0 * student_t.sf(delta_max / math.sqrt(1.0 / n1 + 1.0 / n2), n1 + n2 - 2))


def compare_stat(a, b, delta_max: float = DELTA_MAX) -> dict:
    """KS and pooled-SD mean difference of SLiM values ``a`` against simulator values ``b`` (NaN dropped)."""
    from scipy.stats import ks_2samp

    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    out = dict(n_slim=int(a.size), n_sim=int(b.size), mean_slim=np.nan, mean_sim=np.nan, sd_slim=np.nan,
               sd_sim=np.nan, sd_pooled=np.nan, delta_sd=np.nan, ks_D=np.nan, ks_p=np.nan, p_fail_null=np.nan,
               verdict="NA")
    if a.size:
        out["mean_slim"] = float(a.mean())
    if b.size:
        out["mean_sim"] = float(b.mean())
    if a.size < 2 or b.size < 2:
        return out
    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = math.sqrt(((a.size - 1) * va + (b.size - 1) * vb) / (a.size + b.size - 2))
    ks = ks_2samp(a, b)
    out.update(sd_slim=float(math.sqrt(va)), sd_sim=float(math.sqrt(vb)), sd_pooled=float(sp),
               ks_D=float(ks.statistic), ks_p=float(ks.pvalue))
    out["p_fail_null"] = null_fail_prob(a.size, b.size, delta_max)
    if sp > 0:
        d = (out["mean_slim"] - out["mean_sim"]) / sp
        out["delta_sd"] = float(d)
        out["verdict"] = "PASS" if abs(d) < delta_max else "FAIL"
    return out


def compare(slim: pd.DataFrame, sim: pd.DataFrame, keys=COMPARE_KEYS, delta_max: float = DELTA_MAX) -> pd.DataFrame:
    """Per-statistic table (``compare_stat``) after the same p_now window filter on both sides."""
    a, b = _ok_rows(slim), _ok_rows(sim)
    rows = []
    for k in keys:
        av = a[k].to_numpy(float) if k in a.columns else np.array([])
        bv = b[k].to_numpy(float) if k in b.columns else np.array([])
        rows.append(dict(stat=k, **compare_stat(av, bv, delta_max)))
    df = pd.DataFrame(rows)
    df.insert(1, "n_slim_rows", len(a))
    df.insert(2, "n_sim_rows", len(b))
    return df


def compare_sets(slim: pd.DataFrame, sim: pd.DataFrame, sets: dict | None = None, keys=COMPARE_KEYS,
                 delta_max: float = DELTA_MAX) -> pd.DataFrame:
    """``compare`` per parameter set (column ``set`` on both tables); sets missing on one side are NA rows."""
    names = list(sets) if sets else sorted(set(slim["set"]) | set(sim["set"]))
    out = []
    for name in names:
        t = compare(slim[slim["set"] == name], sim[sim["set"] == name], keys, delta_max)
        t.insert(0, "set", name)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def report_text(table: pd.DataFrame, slim: pd.DataFrame, sim: pd.DataFrame, sets: dict | None = None) -> str:
    """Plain-text summary: per set, the rows before / after the window, PASS / FAIL / NA counts and the FAILs."""
    lines = [f"SLiM vs ABC simulator (Task 9). PASS = |mean_slim - mean_sim| / sd_pooled < {DELTA_MAX}; "
             f"both sides filtered on |p_now - {P_NOW}| <= {P_TOL}. KS p reported, not used in the verdict.",
             f"The rule has a null FAIL rate: P(|delta| >= {DELTA_MAX}) with no real difference is 2 t_sf("
             f"{DELTA_MAX} / sqrt(1/n1 + 1/n2)), ~7.7% per statistic at n = 100 per side, so a set's expected FAIL "
             "count under no difference is printed with its counts (a sum over correlated statistics).", ""]
    for name, t in table.groupby("set", sort=False):
        a, b = slim[slim["set"] == name], sim[sim["set"] == name]
        a_ok = int((a["status"].astype(str) == "ok").sum()) if "status" in a.columns else len(a)
        prov = (sets or {}).get(name, {})
        engine = sorted(set(b["engine"].astype(str))) if "engine" in b.columns and len(b) else []
        lines.append(f"[{name}] model={prov.get('model', '?')} hard={prov.get('hard', '?')} Q={prov.get('Q', '?')} "
                     f"provisional={prov.get('provisional', False)} engine={','.join(engine) or '?'}")
        lines.append(f"  SLiM rows {len(a)} (ok {a_ok}, in window {int(t['n_slim_rows'].iloc[0])}); "
                     f"simulator rows {len(b)} (in window {int(t['n_sim_rows'].iloc[0])})")
        vc = t["verdict"].value_counts()
        scored = t[t["verdict"] != "NA"]
        exp_fail = float(scored["p_fail_null"].sum()) if len(scored) else 0.0
        lines.append(f"  PASS {int(vc.get('PASS', 0))}  FAIL {int(vc.get('FAIL', 0))}  NA {int(vc.get('NA', 0))}  "
                     f"(expected FAIL under no difference at these n: {exp_fail:.1f} of {len(scored)})")
        for _, r in t[t["verdict"] == "FAIL"].iterrows():
            lines.append(f"    FAIL {r['stat']:<16} delta={r['delta_sd']:+.3f} SD  slim={r['mean_slim']:.4g} "
                         f"sim={r['mean_sim']:.4g}  KS D={r['ks_D']:.3f} p={r['ks_p']:.2g}")
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ the simulator side ----
def load_sets(path) -> dict:
    """{name: parameter set} of a sets file ({"sets": {...}})."""
    return json.loads(Path(path).read_text())["sets"]


def path_params(pset: dict, n: int) -> dict:
    """``simulate_paths`` input of ``n`` copies of one fixed parameter set (hard: t_sel = t_inv, p0 NaN)."""
    prm = {}
    for k in PATH_KEYS:
        v = pset.get(k)
        prm[k] = np.full(n, np.nan if v is None else float(v))
    hard = bool(pset["hard"])
    prm["hard"] = np.full(n, hard)
    if hard:
        prm["p0"][:] = np.nan
        if pset["model"] != "N":
            prm["t_sel"] = prm["t_inv"].copy()
    prm["model"] = pset["model"]
    prm["flux"] = bool(np.isfinite(prm["gamma_eff"][0]))
    return prm


def draw_paths(pset: dict, n: int, rng, batch: int = 2000, max_draws: int = 5_000_000):
    """``n`` accepted paths (the p_now window applied) of one parameter set, and the draw counts.

    A hard set gets up to HARD_MAX_ATTEMPTS establishment attempts per draw (``simulate_paths(max_attempts=...)``).
    For a fixed theta the attempts are iid, so the accepted paths are exactly the paths conditioned on
    establishment, the conditioning of SLiM's restart-on-loss (and of ``choose_slim_sets.py``'s acceptance).
    """
    from illex.abc.trajectory import simulate_paths

    paths, drawn, surv = [], 0, 0
    while len(paths) < n:
        if drawn >= max_draws:
            raise RuntimeError(f"{len(paths)} of {n} paths after {drawn} draws (window too rare for this set)")
        got, counts = simulate_paths(path_params(pset, batch), rng, apply_filter=True,
                                     max_attempts=HARD_MAX_ATTEMPTS if pset["hard"] else 1)
        drawn += batch
        c = counts["hard" if pset["hard"] else "soft"]
        surv += c.drawn - c.lost - c.fixed - c.lost_establishing - c.est_failed
        paths.extend(got)
    return paths[:n], dict(drawn=drawn, survived=surv)


def _sim_one(args):
    from illex.abc.simulate import BND_PPC_READY, boundary_ppc, engine_for, simulate_stats

    path, seed, name = args
    eng = engine_for(path)
    t0 = time.time()
    row = dict(set=name, seed=seed, engine=eng, p_now=float(path.p_now), status="ok")
    try:
        row.update(simulate_stats(path, seed, engine=eng))
        if BND_PPC_READY:
            row.update(boundary_ppc(path, seed, engine="msinv"))
        else:
            row.update(dict.fromkeys(SV.BND_KEYS, float("nan")))
    except Exception as exc:                          # noqa: BLE001
        row["status"] = f"error:{type(exc).__name__}:{str(exc)[:200]}"
    row["wall_s"] = round(time.time() - t0, 2)
    return row


def set_index(sets: dict, name: str) -> int:
    """Position of ``name`` in the sets file (the seed offset unit, ``SET_SEED_STRIDE``)."""
    return list(sets).index(name)


def simulate_side(pset: dict, name: str, n_reps: int, seed: int, workers: int = 1,
                  index: int = 0) -> tuple[pd.DataFrame, dict]:
    """(rows, draw counts) of the ABC simulator for one parameter set: ``n_reps`` window-accepted paths, each
    through ``simulate_stats`` on ``engine_for(path)`` (plus ``boundary_ppc`` once BND_PPC_READY).

    Seeds: base = seed + index * SET_SEED_STRIDE draws the paths, and replicate j uses base * 1000 + j.
    """
    base = int(seed) + int(index) * SET_SEED_STRIDE
    rng = np.random.default_rng(base)
    paths, counts = draw_paths(pset, n_reps, rng)
    jobs = [(p, base * 1000 + j, name) for j, p in enumerate(paths)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_sim_one, jobs))
    else:
        rows = [_sim_one(j) for j in jobs]
    df = pd.DataFrame(rows)
    for k in PATH_KEYS + ("hard", "model"):
        df[k] = pset.get(k)
    return df, counts


# ------------------------------------------------------------------ CLI ----
def _read_tables(patterns) -> pd.DataFrame:
    files = sorted({f for p in patterns for f in glob.glob(p) if Path(f).stat().st_size > 0})   # skip in-progress
    if not files:
        raise FileNotFoundError(f"no files match {patterns}")
    return pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate", help="the ABC simulator side of one parameter set")
    s.add_argument("--sets", required=True)
    s.add_argument("--set", required=True)
    s.add_argument("--reps", type=int, default=100)
    s.add_argument("--seed", type=int, default=9_000)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare", help="the per-statistic table")
    c.add_argument("--sets", required=True)
    c.add_argument("--slim", nargs="+", required=True, help="SLiM run_one TSVs (globs)")
    c.add_argument("--sim", nargs="+", required=True, help="simulator TSVs (globs)")
    c.add_argument("--out", type=Path, required=True, help="output prefix (.csv and .txt)")
    args = ap.parse_args(argv)

    sets = load_sets(args.sets)
    if args.cmd == "simulate":
        if args.workers > 16:
            raise SystemExit("--workers is capped at 16 on the shared box")
        t0 = time.time()
        df, counts = simulate_side(sets[args.set], args.set, args.reps, args.seed, args.workers,
                                   index=set_index(sets, args.set))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, sep="\t", index=False)
        print(f"wrote {args.out}: {len(df)} rows, {counts['drawn']} paths drawn, {counts['survived']} survived, "
              f"engine {sorted(set(df['engine']))}, {time.time() - t0:.0f} s, "
              f"{df['wall_s'].mean():.1f} s/rep")
        return
    slim, sim = _read_tables(args.slim), _read_tables(args.sim)
    table = compare_sets(slim, sim, sets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(f"{args.out}.csv", index=False)
    Path(f"{args.out}.txt").write_text(report_text(table, slim, sim, sets))
    print(f"wrote {args.out}.csv and {args.out}.txt")


if __name__ == "__main__":
    main()
