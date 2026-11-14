"""Block-pooled posterior predictive check at the length of the data (review M3a).

The NPE and the 40-kb PPC (ppc_npe.py) simulate one BODY_BP = 40 kb inversion body and one CONTROL_BP = 40 kb
control per draw, so a simulated statistic scatters far more than the same statistic from the 18.96-Mb inversion
and the 45-Mb chr2 collinear control of the data. Here each posterior draw keeps its one frequency path (the same
draw, seed and path as ppc_npe.py with the same --seed and --n) and simulates N_BODY = 474 independent 40-kb bodies
(474 x 40 kb = 18.96 Mb) and N_CTL = 1125 independent 40-kb controls (45 Mb), each with its own seed words, the
error model and the same engines as simulate_stats. Raw quantities are pooled over the blocks before any ratio is
taken, as the data pool sites over the region:

  pi_I, pi_S, dxy             branch mode, mean over blocks (equal lengths, so the pooled per-bp value)
  f1                          branch-mode AFS summed over blocks, then projected (as _branch_ratios)
  sfs_*, tajd_*               unfolded per-site count spectra summed over blocks; SFS bins as _sfs_bins_unfolded,
                              Tajima's D from the pooled spectrum
  jsfs_*                      binned 5 x 5 joint spectra summed over blocks, then folded (as _jsfs)
  ld_*                        r^2 sums and pair counts of ld_block_sums summed over blocks
  edge_centre                 per-window branch pi and dxy averaged over the body blocks, then edge / core

Blocks are unlinked, so the pooled statistics scatter slightly less than linked 19-Mb sequence would; the check is
conservative in that direction (it can only reject more easily). The output CSV has the columns of ppc_npe.py, so
illex/abc/smc_results/ppc/ppc_analyze.py scores it unchanged (run label "blocks").

    PYTHONNOUSERSITE=1 PYTHONPATH=<karyofix tree> OMP_NUM_THREADS=1 python illex/abc/npe/ppc_blocks.py \
        --draws results/illex/npe_karyofix/f1/posterior_samples.csv --flux 1 --out blocks_f1_t000.csv \
        --n 1000 --first 0 --count 16 --procs 8

--first/--count select a slice of the n sampled draws (array tasks); collect the slices with
illex/abc/npe/collect_ppc_blocks.py. --self-test checks that one pooled block equals region_stats.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

N_BODY = 474       # 18.96 Mb / 40 kb
N_CTL = 1125       # 45 Mb (chr2 40-55 + 85-115 Mb) / 40 kb
MAX_TRIES = 200


# ------------------------------------------------------------------ raw, poolable quantities of one block ----
def region_raw(ts, i_nodes, s_nodes, win, rng, ld_err=None, err_rng=None, edge=False):
    """Additive raw quantities of one block over ``win`` (see the module docstring). Mirrors statvec.region_stats
    step for step (same random draws in the same order) so one block reproduces it exactly."""
    from illex.abc import statvec as SV

    i_nodes, s_nodes = np.asarray(i_nodes), np.asarray(s_nodes)
    windows = SV._windows(ts, win)
    k = 1 if win[0] > 0 else 0
    pi = ts.diversity([i_nodes, s_nodes], mode="branch", windows=windows)[k]
    dxy = ts.divergence([i_nodes, s_nodes], mode="branch", windows=windows)[k]
    bafs = []
    for nodes in (i_nodes, s_nodes):
        bafs.append(np.asarray(ts.allele_frequency_spectrum(sample_sets=[list(nodes)], windows=windows, mode="branch",
                                                            polarised=True, span_normalise=True)[k], float))
    pos = ts.tables.sites.position
    sel = (pos >= win[0]) & (pos < win[1])
    if ts.num_sites:
        g = np.minimum(ts.genotype_matrix()[sel], 1).astype(np.int8)
    else:
        g = np.zeros((0, ts.num_samples), np.int8)
    p = pos[sel]
    col = {int(u): c for c, u in enumerate(ts.samples())}
    ci_cols = np.array([col[int(u)] for u in i_nodes])
    cs_cols = np.array([col[int(u)] for u in s_nodes])
    ci = g[:, ci_cols].sum(1).astype(np.int64)
    cs = g[:, cs_cols].sum(1).astype(np.int64)
    ni, ns = ci_cols.size, cs_cols.size
    out = dict(pi_I=float(pi[0]), pi_S=float(pi[1]), dxy=float(dxy), baf_I=bafs[0], baf_S=bafs[1],
               unf_I=np.bincount(ci, minlength=ni + 1).astype(float),
               unf_S=np.bincount(cs, minlength=ns + 1).astype(float),
               jsfs=(SV._hyper_binned(ci, ni).T @ SV._hyper_binned(cs, ns)) if ci.size else np.zeros((5, 5)))
    for c, cols in (("I", ci_cols), ("S", cs_cols)):
        err = None if ld_err is None else ld_err[c]
        out[f"ld_s_{c}"], out[f"ld_c_{c}"] = _ld_sums(g, p, cols, rng, err, err_rng)
    if edge:
        left, right = win
        w = np.linspace(left, right, SV.N_WIN + 1)
        wins = list(w)
        if left > 0:
            wins = [0.0] + wins
        if right < ts.sequence_length:
            wins = wins + [ts.sequence_length]
        k0 = 1 if left > 0 else 0
        out["win_pi"] = ts.diversity([list(i_nodes), list(s_nodes)], mode="branch", windows=wins)[k0:k0 + SV.N_WIN]
        out["win_dxy"] = ts.divergence([list(i_nodes), list(s_nodes)], mode="branch", windows=wins)[k0:k0 + SV.N_WIN]
        out["win_rel"] = np.abs(0.5 * (w[:-1] + w[1:]) - 0.5 * (left + right)) / (0.5 * (right - left))
    return out


def _ld_sums(geno, pos, cols, rng, err, err_rng):
    """(sums, counts) of statvec._ld for one class: the same pairing and error draws, not yet divided."""
    from illex.abc import statvec as SV
    from illex.abc.error_model import apply_genotype_errors

    nbin = len(SV.LD_EDGES) - 1
    z = (np.zeros(nbin), np.zeros(nbin))
    cols = np.asarray(cols)
    perm = rng.permutation(cols.size)
    m = cols.size // 2
    a, b = cols[perm[:m]], cols[perm[m:2 * m]]
    if m < 2 or geno.shape[0] < 2:
        return z
    d = geno[:, a].astype(np.float64) + geno[:, b]
    if err is not None:
        d = apply_genotype_errors(d, err[0], err[1], err_rng)
    s, c = SV.ld_block_sums(d, pos, np.zeros(d.shape[0], int), 1)
    return s[0], c[0]


def pool(blocks):
    """Sum (or, for the per-bp branch values, average) the raw quantities of equal-length blocks."""
    out = {}
    for k in blocks[0]:
        if k == "win_rel":
            out[k] = blocks[0][k]
            continue
        v = sum(np.asarray(b[k], float) for b in blocks)
        out[k] = v / len(blocks) if k in ("pi_I", "pi_S", "dxy", "baf_I", "baf_S", "win_pi", "win_dxy") else v
    return out


def _tajd_unf(unf):
    """Tajima's D of an unfolded per-site count spectrum unf[0..n] (tskit's site-mode formula)."""
    n = len(unf) - 1
    k = np.arange(n + 1)
    S = float(unf[1:n].sum())
    if S == 0 or n < 4:
        return float("nan")
    pi = float((unf * k * (n - k)).sum()) / (n * (n - 1) / 2)
    a1 = float((1.0 / np.arange(1, n)).sum())
    a2 = float((1.0 / np.arange(1, n) ** 2).sum())
    b1 = (n + 1) / (3 * (n - 1))
    b2 = 2 * (n * n + n + 3) / (9 * n * (n - 1))
    c1 = b1 - 1 / a1
    c2 = b2 - (n + 2) / (a1 * n) + a2 / a1 ** 2
    e1, e2 = c1 / a1, c2 / (a1 * a1 + a2)
    return float((pi - S / a1) / math.sqrt(e1 * S + e2 * S * (S - 1)))


def finish(raw):
    """statvec.region_stats-style statistics from pooled raw quantities."""
    from illex.abc import statvec as SV
    from illex.scripts.sfs_shape import PROJ, project_unfolded

    out = dict(pi_ratio=SV._div(raw["pi_I"], raw["pi_S"]), dxy_pi=SV._div(raw["dxy"], raw["pi_I"]))
    f1 = []
    for c in ("I", "S"):
        af = raw[f"baf_{c}"]
        n = len(af) - 1
        tot = af[1:n].sum()
        if tot <= 0:
            f1.append(float("nan"))
            continue
        v = project_unfolded(af / tot, n, PROJ)
        f1.append(SV._div(v[0], v.sum()))
    out["f1_ratio"] = SV._div(f1[0], f1[1])
    for c in ("I", "S"):
        for b, v in SV._sfs_bins_unfolded(raw[f"unf_{c}"], len(raw[f"unf_{c}"]) - 1).items():
            out[f"sfs_{c}_{b}"] = float(v)
    out.update(SV._jsfs_fold(raw["jsfs"]))
    for c in ("I", "S"):
        s, n = raw[f"ld_s_{c}"], raw[f"ld_c_{c}"]
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        for k, x in enumerate(v):
            out[f"ld_{c}_{k + 1}"] = float(x)
    out["tajd_I"] = _tajd_unf(raw["unf_I"])
    out["tajd_S"] = _tajd_unf(raw["unf_S"])
    return out


def edge_from(raw):
    from illex.abc import statvec as SV

    if "win_pi" not in raw:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        r = raw["win_dxy"] / raw["win_pi"].mean(1)
    rel = raw["win_rel"]
    core, edge = r[rel < SV.CORE_REL], r[rel > SV.EDGE_REL]
    if not (np.isfinite(core).all() and np.isfinite(edge).all()) or core.size == 0 or edge.size == 0:
        return float("nan")
    return SV._div(edge.mean(), core.mean())


# ------------------------------------------------------------------ one block of each kind ----
def body_block(path, seed):
    import illex.abc.simulate as SIM
    from illex import stats as S
    from illex.abc.constants import BODY_BP, MARGIN_BP

    s_inv, _, s_mi, _, s_pi, _, _, s_ei, _ = SIM._seed_words(seed)
    sim, ts = SIM._inversion_ts(path, s_inv, crossover_suppression=True)
    i_nodes, s_nodes = S.sample_nodes_by_karyotype(sim, ts)
    win = (MARGIN_BP, MARGIN_BP + BODY_BP)
    return region_raw(SIM._mutate(ts, s_mi), i_nodes, s_nodes, win, np.random.default_rng(s_pi),
                      ld_err=SIM.LD_ERROR["body"], err_rng=np.random.default_rng(s_ei), edge=True)


def control_block(seed):
    import illex.abc.simulate as SIM
    from illex.abc.constants import CONTROL_BP, N_INV, N_STD

    _, s_ctl, _, s_mc, _, s_pc, _, _, s_ec = SIM._seed_words(seed)
    cts = SIM._mutate(SIM._control_ts(s_ctl), s_mc)
    smp = cts.samples()
    return region_raw(cts, smp[N_STD:N_STD + N_INV], smp[:N_STD], (0.0, CONTROL_BP), np.random.default_rng(s_pc),
                      ld_err=SIM.LD_ERROR["control"], err_rng=np.random.default_rng(s_ec))


# ------------------------------------------------------------------ one posterior draw ----
def path_for(par, flux, rng):
    """The draw's frequency path, drawn exactly as ppc_npe._one draws it (same rng stream)."""
    from illex.abc.trajectory import KEYS, simulate_paths

    p = dict(t_inv=par["t_inv"], hard=False, p0=par["p0"], t_sel=par["t_sel"], s=par["s"], t_stop=np.nan,
             p_star=par["p_star"], gamma_eff=par["gamma"] if flux else np.nan)
    arr = {k: np.array([p[k]], dtype=bool if k == "hard" else float) for k in KEYS}
    arr.update(model="B", flux=bool(flux))
    for tries in range(1, MAX_TRIES + 1):
        paths, _ = simulate_paths(arr, rng, max_attempts=1)
        if paths:
            return paths[0], tries
    return None, MAX_TRIES


def one_draw(task):
    from illex.abc import statvec as SV

    par, flux, seed, n_body, n_ctl = task
    t0 = time.time()
    rng = np.random.default_rng(seed)
    path, tries = path_for(par, flux, rng)
    if path is None:
        return dict(ok=0, tries=tries)
    root = int(rng.integers(1, 2**31 - 1))          # ppc_npe's single simulation seed; here the root of the blocks
    ss = np.random.SeedSequence(root)
    bseeds = [int(x) for x in ss.spawn(1)[0].generate_state(n_body) % (2**31 - 1) + 1]
    cseeds = [int(x) for x in ss.spawn(2)[1].generate_state(n_ctl) % (2**31 - 1) + 1]
    body = pool([body_block(path, s) for s in bseeds])
    ctl = pool([control_block(s) for s in cseeds])
    out = SV.assemble(finish(body), finish(ctl), edge_from(body))
    res = {k: float(out[k]) for k in SV.STAT_KEYS}
    res.update(ok=1, tries=tries, p_now=float(path.p_now), n_nan=int(sum(math.isnan(res[k]) for k in SV.MODEL_KEYS)),
               n_body=n_body, n_ctl=n_ctl, wall_s=time.time() - t0)
    return res


def preflight():
    import illex.theory as T
    import illex.abc.constants as C
    from illex.abc.smc import OBSERVED
    from illex.abc.error_model import LD_ERROR
    ok = (round(T.N_ANC) == 507288 and round(T.N0) == 9112696 and round(T.T_GROW) == 382957
          and float(f"{C.REC_RATE:.4g}") == 5.507e-10 and str(OBSERVED).endswith("abc_observed_ctl2_karyofix.json")
          and LD_ERROR["control"]["I"] == (0.6768, 0.6645) and C.P_NOW == 0.602 and C.N_KARYO == 619
          and C.BODY_BP == 40_000 and C.CONTROL_BP == 40_000)
    if not ok:
        raise SystemExit("FATAL: PYTHONPATH is not a base2-karyofix (illex/baseline-refit-karyofix) tree")


def self_test():
    """One pooled block must reproduce statvec.region_stats on the same tree sequence and rng streams."""
    import illex.abc.simulate as SIM
    from illex.abc import statvec as SV
    from illex.abc.constants import CONTROL_BP, N_INV, N_STD

    _, s_ctl, _, s_mc, _, s_pc, _, _, s_ec = SIM._seed_words(12345)
    cts = SIM._mutate(SIM._control_ts(s_ctl), s_mc)
    smp = cts.samples()
    args = (cts, smp[N_STD:N_STD + N_INV], smp[:N_STD], (0.0, CONTROL_BP))
    ref = SV.region_stats(*args, np.random.default_rng(s_pc), ld_err=SIM.LD_ERROR["control"],
                          err_rng=np.random.default_rng(s_ec))
    got = finish(pool([region_raw(*args, np.random.default_rng(s_pc), ld_err=SIM.LD_ERROR["control"],
                                  err_rng=np.random.default_rng(s_ec))]))
    bad = [k for k in ref if not (np.isclose(ref[k], got[k], rtol=1e-9, atol=1e-12, equal_nan=True))]
    for k in bad:
        print(f"MISMATCH {k}: region_stats {ref[k]!r} pooled {got[k]!r}")
    print("self-test", "FAILED" if bad else f"passed ({len(ref)} statistics)")
    raise SystemExit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws")
    ap.add_argument("--flux", type=int)
    ap.add_argument("--out")
    ap.add_argument("--label", default="blocks")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--n-body", type=int, default=N_BODY)
    ap.add_argument("--n-ctl", type=int, default=N_CTL)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    preflight()
    if a.self_test:
        self_test()
    # the same n draws and per-draw seeds as ppc_npe.py with this --seed and --n
    d = pd.read_csv(a.draws).sample(n=a.n, replace=False, random_state=a.seed).reset_index(drop=True)
    rng = np.random.default_rng(a.seed)
    seeds = [int(rng.integers(1, 2**31 - 1)) for _ in range(len(d))]
    last = len(d) if a.count is None else min(len(d), a.first + a.count)
    idx = list(range(a.first, last))
    tasks = [(d.iloc[i].to_dict(), a.flux, seeds[i], a.n_body, a.n_ctl) for i in idx]
    with mp.get_context("fork").Pool(a.procs, maxtasksperchild=1) as pool_:
        res = pool_.map(one_draw, tasks, chunksize=1)
    meta = pd.DataFrame({"run": a.label, "flux": a.flux, "model": "B", "particle": idx, "n_distinct": a.n})
    meta = pd.concat([meta, d.iloc[idx].reset_index(drop=True).add_prefix("par_")], axis=1)
    out = pd.concat([meta, pd.DataFrame(res)], axis=1)
    out.to_csv(a.out, index=False)
    print(f"flux {a.flux} draws {a.first}-{last - 1}: {int(out.ok.sum())} of {len(out)} simulated", flush=True)


if __name__ == "__main__":
    main()
