#!/usr/bin/env python
"""Within-arrangement folded SFS shape: the last independent statistic.

    CUDA_VISIBLE_DEVICES=0 /home/ssmall/miniforge3/envs/varbuddy-pggpu/bin/python \
        illex/scripts/sfs_shape.py --stage empirical
    .venv/bin/python -m illex.scripts.sfs_shape --stage model --reps 96

Two stages because they need different interpreters: the empirical side needs
pg_gpu to read the VCF, the model side needs msinv. The empirical stage writes
its spectra to JSON and the model stage reads them back.

WHY THIS STATISTIC
------------------
After Fst turned out algebraically redundant (NOTES sec 5.3), the spatial dxy
profile was spent falsifying flux (sec 6), the ReLERNN interior carried no
barrier signal (sec 8.0), and absolute levels needed a nuisance scale (sec 8.3),
the within-arrangement SFS shape is the one genuinely independent constraint
left (sec 9). It is normalised, so it needs no accessibility mask and no theta;
and it is a *shape*, so the 1.31x calibration offset cannot touch it.

WHAT IT IS BEING ASKED TO DO
----------------------------
Not to re-confirm the fit. The balancing-selection fit already matches its two
targets exactly, and a third statistic that merely agreed would add little.
The job is to break the **degeneracy that survived** (sec 7.5.1): plateau length
trades against founding frequency along a ridge on which pi_I/pi_S and dxy/pi_I
are both constant by construction. Two points on that ridge --

    plateau        0 gen:  t_inv 727,301   p_start 0.0271   s_het 3.58e-5
    plateau  100,000 gen:  t_inv 718,872   p_start 0.0222   s_het 4.24e-5

-- are indistinguishable to every statistic used so far. They should NOT be
indistinguishable here, because founding frequency and plateau length act on the
*shape* of the within-inverted genealogy in different ways: a smaller founding
frequency is a harder bottleneck, which collapses the deep branches and shifts
weight toward rare variants, while a longer plateau lets the inverted class
re-accumulate intermediate-frequency variation at its equilibrium size.

If the two ridge points give the same spectrum to within Monte Carlo error, the
degeneracy is real and (p_start, plateau) must be reported as a joint range
forever. That is a publishable negative and is stated here in advance.

CONTROLS, so a null result cannot be mistaken for a broken statistic
--------------------------------------------------------------------
The old rising-logistic point (t_inv 8e5, p_start 0.15) is run alongside. It
misses both fitted ratios by ~9% and ~7%, so it MUST be separable. If the
statistic cannot even distinguish that, it has no power and nothing else it says
means anything.

TWO THINGS THAT COULD INVALIDATE THE EMPIRICAL COMPARISON
---------------------------------------------------------
1. **Rare-variant ascertainment.** The callset is quality-filtered, so
   singletons are the bin most likely to be depleted relative to truth, while
   the model's branch-mode spectrum has no such filter. Every L1 is therefore
   reported twice, with and without bin 1. If the ranking flips between them,
   the result is an artifact of the filter and not a fact about the inversion.
2. **Projection must be exact, and a first pass here was not.** The obvious
   route -- reuse ``beta_vs_kingman.py``'s ``project_folded``, which unfolds a
   folded spectrum by splitting each bin between classes i and n-i on neutral
   1/i weights, projects, then refolds -- carries an approximation whose size
   depends on ``n_from``. That script had no choice, because its input was
   already a folded spectrum file. Here it produced a spurious ~0.07-0.10
   singleton deficit in BOTH arrangements, because the empirical side projected
   from n = 150-376 while the model side projected from n = 100, so the
   approximation did not cancel between them.

   It is avoidable. Per-site allele counts are available, so the UNFOLDED
   spectrum can be projected exactly -- hypergeometric subsampling of
   chromosomes is exact for unfolded counts -- and folded only at the target
   size. The model side asks tskit for ``polarised=True`` and does the same.
   Folding last makes the two sides comparable without either needing a
   polarisation assumption. ``project_unfolded`` below is exact; the old
   approximate path is deliberately not kept.

The data are UNPHASED, which does not matter: AA individuals are homozygous for
the inverted arrangement and BB for the standard, so per-arrangement allele
counts are read straight off genotypes. The 284 AB heterozygotes are unusable
and excluded, which is why n = 349 rather than 633.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path("results/illex")
EMP_JSON = OUT / "sfs_shape_empirical.json"
MODEL_JSON = OUT / "sfs_shape_model.json"

PROJ = 20  # haploid projection size -> 10 folded bins
FST_DIFF_CUTOFF = 0.15
AXIS_START, AXIS_STOP = 60_000_000, 80_000_000
BASE_WINDOW = 100_000

# Ridge points that match BOTH fitted ratios exactly (NOTES sec 7.5), plus the
# superseded rising-logistic point as a power control.
POINTS = [
    (
        "ridge_plateau_0",
        dict(kind="balancing", t_inv=727_301.0, p_start=0.0271, plateau=0.0),
    ),
    (
        "ridge_plateau_100k",
        dict(kind="balancing", t_inv=718_872.0, p_start=0.0222, plateau=1.0e5),
    ),
    ("control_rising_logistic", dict(kind="logistic", t_inv=8.0e5, p_start=0.15)),
    (
        "control_single_origin",
        dict(kind="balancing", t_inv=1.05e6, p_start=None, plateau=0.0),
    ),
    # Baseline: no inversion, same demography, so its spectrum is what the
    # COLLINEAR region should look like. If the model misses the collinear
    # empirical spectrum by as much as it misses the inversion-body one, the
    # gap lives in the neutral baseline (mutation-rate variation, BGS/DFE --
    # cf. the project manuscript's sec 8b spatial-heterogeneity gap) and only
    # the differential contrast is interpretable.
    ("baseline_panmictic", dict(kind="panmictic")),
]

# Floors on called chromosomes per site. Required, not optional: at low
# called-n a rare variant is often not sampled at all, so low-n sites are
# ascertainment-depleted of rare variants and including them drags the
# singleton fraction down. Measured on this callset, BB's singleton fraction
# runs 0.527 (no floor) to 0.782 (n >= 150) -- the shape is a stronger function
# of the floor than of anything biological, so an unfloored spectrum is not an
# estimate of anything.
FLOOR_AA, FLOOR_BB = 300, 150
FLOOR_ARMS = [(300, 150), (200, 100), (400, 170)]


# ---------------------------------------------------------------- shared ----
def project_unfolded(unf, n_from_hap, n_to_hap):
    """EXACT hypergeometric projection of an UNFOLDED spectrum, then fold.

    ``unf[j]`` is the (expected) number of sites with j copies of the tracked
    allele among ``n_from_hap`` chromosomes, j = 0..n. Drawing m of n
    chromosomes without replacement makes the count in the subsample
    Hypergeometric(n, j, m), so this projection involves no approximation --
    unlike unfolding a folded spectrum, which needs an assumed shape.

    Folding is applied only at the target size, so neither side of the
    comparison needs its alleles polarised correctly; it just needs to be the
    SAME transform, which folding-last guarantees.
    """
    from scipy.stats import hypergeom

    n, m = int(n_from_hap), int(n_to_hap)
    unf = np.asarray(unf, dtype=float)
    j = np.arange(len(unf))
    nz = unf > 0
    if not nz.any():
        return np.zeros(m // 2, dtype=float)
    kk = np.arange(m + 1)
    # (n_j, m+1) matrix in one call.
    pmf = hypergeom.pmf(kk[None, :], n, j[nz][:, None], m)
    out = (unf[nz][:, None] * pmf).sum(axis=0)
    nb = m // 2
    fol = np.zeros(nb, dtype=float)
    for k in range(1, m):
        fol[min(k, m - k) - 1] += out[k]
    return fol


def l1_shape(a, b) -> float:
    """Total absolute deviation between two normalised spectra."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.abs(a / a.sum() - b / b.sum()).sum())


# ------------------------------------------------------------- empirical ----
def stage_empirical() -> None:
    import pandas as pd
    from pg_gpu import HaplotypeMatrix

    jk = OUT / "empirical_jackknife_windows.csv"
    if not jk.exists():
        raise SystemExit(
            f"{jk} not found -- run illex/scripts/empirical_jackknife.py first; "
            "it defines the Fst>0.15 differentiated body used here so the two "
            "analyses cover the same interval."
        )
    w = pd.read_csv(jk)
    body_starts = set(w.loc[w.fst > FST_DIFF_CUTOFF, "window_start"].astype(int))
    print(f"differentiated body: {len(body_starts)} of {len(w)} windows")

    h = HaplotypeMatrix.from_vcf(
        ".tmp/illex_chr2/inv.vcf.gz", region=f"2:{AXIS_START}-{AXIS_STOP}"
    )
    h.load_pop_file(".tmp/illex_chr2/pops.tsv")
    H = np.asarray(h.haplotypes)
    pos = np.asarray(h.positions)
    print(f"loaded {H.shape[1]:,} variants x {H.shape[0]:,} haplotypes")

    win_of = ((pos - AXIS_START) // BASE_WINDOW) * BASE_WINDOW + AXIS_START
    in_body = np.isin(win_of, list(body_starts))
    print(f"  {in_body.sum():,} variants inside the differentiated body")

    out = {"proj": PROJ, "n_variants_body": int(in_body.sum()),
           "floors": {"AA": FLOOR_AA, "BB": FLOOR_BB}}

    def spectra(mat, cols, sets, tag, floors):
        """Per-arrangement spectra on a COMMON, well-called site set.

        The site set must be shared between the two arrangements, or the
        inverted-vs-standard contrast compares different genealogies. So a site
        enters if it is (a) called on at least ``floors[pop]`` chromosomes in
        BOTH arrangements and (b) polymorphic in the POOLED AA+BB sample --
        never "polymorphic in this arrangement", which would give each
        arrangement its own set and its own ascertainment.

        Sites monomorphic within one arrangement contribute to its bin 0, which
        the fold in ``project_unfolded`` discards, so each spectrum is still a
        spectrum of that arrangement's own variation.
        """
        n_c, a_c = {}, {}
        for pop in ("AA", "BB"):
            idx = np.asarray(sets[pop])
            sub = mat[np.ix_(idx, cols)] if cols is not None else mat[idx]
            ok = sub >= 0
            n_c[pop] = ok.sum(axis=0).astype(int)
            a_c[pop] = np.where(ok, sub, 0).sum(axis=0).astype(int)
        pooled_alt = a_c["AA"] + a_c["BB"]
        pooled_n = n_c["AA"] + n_c["BB"]
        keep = ((n_c["AA"] >= floors["AA"]) & (n_c["BB"] >= floors["BB"])
                & (pooled_alt > 0) & (pooled_alt < pooled_n))
        res = {"n_sites": int(keep.sum())}
        for pop in ("AA", "BB"):
            acc = np.zeros(PROJ // 2, dtype=float)
            for n in np.unique(n_c[pop][keep]):
                sel = keep & (n_c[pop] == n)
                unf = np.bincount(a_c[pop][sel], minlength=int(n) + 1)
                acc += project_unfolded(unf.astype(float), int(n), PROJ)
            res[pop] = acc.tolist()
        print(f"  {tag}: {keep.sum():,} common sites "
              f"(floors AA>={floors['AA']}, BB>={floors['BB']})")
        for pop in ("AA", "BB"):
            v = np.array(res[pop])
            print(f"    {pop} " + " ".join(f"{x:.4f}" for x in v / v.sum()))
        return res

    cols = np.flatnonzero(in_body)
    for fa, fb in FLOOR_ARMS:
        key = "body" if (fa, fb) == (FLOOR_AA, FLOOR_BB) else f"body_{fa}_{fb}"
        out[key] = spectra(H, cols, h.sample_sets, key, {"AA": fa, "BB": fb})
    for pop in ("AA", "BB"):
        out[f"{pop}_n_chrom"] = int(len(h.sample_sets[pop]))

    # --- collinear control region, chr2:10-30 Mb -------------------------
    # Same samples, same pipeline, same floors, no inversion. A model-data gap
    # that appears HERE belongs to the neutral baseline, not the inversion.
    hc = HaplotypeMatrix.from_vcf(
        ".tmp/illex_chr2/ctl.vcf.gz", region="2:10000000-30000000")
    hc.load_pop_file(".tmp/illex_chr2/pops.tsv")
    Hc = np.asarray(hc.haplotypes)
    print(f"control region: {Hc.shape[1]:,} variants")
    out["control"] = spectra(Hc, None, hc.sample_sets, "control",
                             {"AA": FLOOR_AA, "BB": FLOOR_BB})

    OUT.mkdir(parents=True, exist_ok=True)
    EMP_JSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {EMP_JSON}")


# ----------------------------------------------------------------- model ----
def _one(job):
    import time

    from illex import balancing as B
    from illex import model as M
    from illex import stats as S
    from illex.demography import PRESENT_NE_GROWTH
    from illex.slim.config import REC_RATE

    name, spec, rep = job
    L = int(round(2000.0 / (4.0 * PRESENT_NE_GROWTH * REC_RATE)))
    seed = 5_000_000 + 1000 * abs(hash(name)) % 100000 + rep
    t0 = time.time()
    if spec["kind"] == "panmictic":
        sim = M.build_control_sim(arm="growth", seq_length=L, seed=seed,
                                  recomb_rate=REC_RATE)
    elif spec["kind"] == "balancing":
        from scipy import optimize

        p0 = spec["p_start"]
        if p0 is None:
            p0 = B.founding_frequency(spec["t_inv"])
        t_rise = spec["t_inv"] - spec["plateau"]
        s_het = float(
            optimize.brentq(
                lambda s: B.rise_time(p0, s) - t_rise, 1e-9, 1.0, rtol=1e-12
            )
        )
        sim = B.build_balancing_sim(
            seq_length=L,
            t_inv=spec["t_inv"],
            s_het=s_het,
            p_start=p0,
            gamma=1e-15,
            seed=seed,
        )
    else:
        sim = M.build_inversion_sim(
            arm="growth",
            seq_length=L,
            t_inv=spec["t_inv"],
            gamma=1e-15,
            p_start=spec["p_start"],
            seed=seed,
            recomb_rate=REC_RATE,
        )
    ts = sim.simulate()
    i_nodes, s_nodes = S.sample_nodes_by_karyotype(sim, ts)
    if spec["kind"] == "panmictic":
        # build_control_sim's inversion is a degenerate 1 bp placeholder, so
        # there is no body to restrict to -- the whole sequence is the region.
        # windows must be STRICTLY increasing, so [0, 0, L, L] is rejected;
        # take the single whole-sequence window instead.
        windows, widx, interval = [0.0, ts.sequence_length], 0, None
    else:
        left, right = M.inversion_interval(sim)
        windows, widx, interval = (
            [0.0, left, right, ts.sequence_length], 1, (left, right))

    res = {"name": name, "rep": rep, "wall_s": round(time.time() - t0, 2)}
    for tag, nodes in (("I", i_nodes), ("S", s_nodes)):
        n = len(nodes)
        # Branch mode: expected spectrum off branch lengths, no mutation noise.
        # windows= restricts to the inversion body (NOTES sec 11: integrating
        # the collinear flank dilutes every arrangement statistic).
        af = ts.allele_frequency_spectrum(
            sample_sets=[list(nodes)],
            windows=windows,
            mode="branch",
            polarised=True,
            span_normalise=True,
        )[widx]
        # Unfolded (derived) expected counts, j = 0..n; folded only at the
        # target size by project_unfolded, matching the empirical side exactly.
        res[tag] = np.asarray(af, dtype=float).tolist()
        res[f"{tag}_n"] = n
    # Ratios too, so the ridge points can be confirmed to still sit on the ridge.
    st = S.arrangement_stats(ts, i_nodes, s_nodes, interval=interval)
    res["pi_i_over_pi_s"] = st["pi_i_over_pi_s"]
    res["dxy_over_pi_i"] = st["dxy_over_pi_i"]
    return res


def stage_model(reps: int, workers: int) -> None:
    import time
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(name, spec, r) for name, spec in POINTS for r in range(reps)]
    print(f"{len(jobs):,} simulations, {reps} reps x {len(POINTS)} points")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_one, jobs, chunksize=1))
    print(f"done in {time.time() - t0:.0f} s")

    out = {"proj": PROJ, "reps": reps, "points": {}}
    for name, _ in POINTS:
        sub = [r for r in rows if r["name"] == name]
        entry = {
            "pi_i_over_pi_s": float(np.mean([r["pi_i_over_pi_s"] for r in sub])),
            "dxy_over_pi_i": float(np.mean([r["dxy_over_pi_i"] for r in sub])),
        }
        for tag in ("I", "S"):
            n = sub[0][f"{tag}_n"]
            # Average the per-replicate NORMALISED spectra, then project once:
            # normalising first stops replicates with more total branch length
            # from dominating the mean shape.
            acc = np.zeros(n + 1, dtype=float)
            for r in sub:
                v = np.asarray(r[tag], dtype=float)
                acc += v / max(v[1:n].sum(), 1e-300)
            acc /= len(sub)
            entry[tag] = project_unfolded(acc, n, PROJ).tolist()
            # Jackknife over replicates for a Monte Carlo error on the shape.
            partial = []
            for drop in range(len(sub)):
                a = np.zeros(n + 1, dtype=float)
                for k, r in enumerate(sub):
                    if k == drop:
                        continue
                    v = np.asarray(r[tag], dtype=float)
                    a += v / max(v[1:n].sum(), 1e-300)
                a /= len(sub) - 1
                p = project_unfolded(a, n, PROJ)
                partial.append(p / p.sum())
            partial = np.array(partial)
            b = len(partial)
            entry[f"{tag}_sem"] = np.sqrt(
                (b - 1) / b * ((partial - partial.mean(0)) ** 2).sum(0)
            ).tolist()
        out["points"][name] = entry

    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_JSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {MODEL_JSON}")
    report()


# ---------------------------------------------------------------- report ----
def report() -> None:
    emp = json.loads(EMP_JSON.read_text())
    mod = json.loads(MODEL_JSON.read_text())
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    def norm(v):
        v = np.asarray(v, dtype=float)
        return v / v.sum()

    emit("Within-arrangement folded SFS shape, projected to "
         f"n = {emp['proj']} ({emp['proj'] // 2} bins)")
    emit(f"empirical: differentiated body, common site set, floors "
         f"AA>={emp['floors']['AA']} BB>={emp['floors']['BB']}; "
         f"{emp['body']['n_sites']:,} sites")
    emit()

    # -- 1. is the empirical estimate stable? ------------------------------
    emit(f"{'=' * 78}\n1. STABILITY OF THE EMPIRICAL ESTIMATE\n{'=' * 78}")
    emit("  An unfloored spectrum is not an estimate: at low called-n a rare")
    emit("  variant is often not sampled, so low-n sites are depleted of rare")
    emit("  variants. Floors must be imposed and the answer must not depend on")
    emit("  which floor. READ THIS TABLE BEFORE ANYTHING BELOW IT: if the arms")
    emit("  disagree, sections 2 and 5 are comparing models to a target that")
    emit("  does not exist, and no ranking in them means anything.")
    emit(f"  {'arm':<16s} {'n_sites':>9s}  " + " ".join(
        f"{i + 1:>6d}" for i in range(emp['proj'] // 2)))
    for key in [k for k in emp if k == "body" or k.startswith("body_")]:
        for pop in ("AA", "BB"):
            emit(f"  {key + '/' + pop:<16s} {emp[key]['n_sites']:>9,}  "
                 + " ".join(f"{x:6.4f}" for x in norm(emp[key][pop])))
    emit()

    # -- 2. model vs data, per arrangement --------------------------------
    for arr, pop, label in (("I", "AA", "INVERTED (AA)"),
                            ("S", "BB", "STANDARD (BB)")):
        o = norm(emp["body"][pop])
        emit(f"{'=' * 78}\n2. {label}\n{'=' * 78}")
        emit("  bin        " + " ".join(f"{i + 1:>7d}" for i in range(len(o))))
        emit("  empirical  " + " ".join(f"{x:7.4f}" for x in o))
        for name in mod["points"]:
            if name == "baseline_panmictic":
                continue
            emit(f"  {name[:10]:<10s} "
                 + " ".join(f"{x:7.4f}" for x in norm(mod["points"][name][arr])))
        emit()
        emit(f"  {'point':<26s} {'L1':>8s} {'L1 no-singleton':>17s}")
        for name in mod["points"]:
            if name == "baseline_panmictic":
                continue
            v = norm(mod["points"][name][arr])
            emit(f"  {name:<26s} {l1_shape(o, v):8.4f} "
                 f"{l1_shape(o[1:], v[1:]):17.4f}")
        emit()

    # -- 3. does it break the degeneracy? ---------------------------------
    emit(f"{'=' * 78}\n3. DOES THE SFS BREAK THE (p_start, plateau) "
         f"DEGENERACY?\n{'=' * 78}")
    a, b = "ridge_plateau_0", "ridge_plateau_100k"
    for arr, label in (("I", "inverted"), ("S", "standard")):
        va, vb = norm(mod["points"][a][arr]), norm(mod["points"][b][arr])
        sem = np.sqrt(np.array(mod["points"][a][f"{arr}_sem"], float) ** 2
                      + np.array(mod["points"][b][f"{arr}_sem"], float) ** 2)
        z = np.abs(va - vb) / np.maximum(sem, 1e-12)
        emit(f"  {label}: L1 between the two ridge points = "
             f"{np.abs(va - vb).sum():.4f}")
        emit("    per-bin |diff|/SE: " + " ".join(f"{x:.1f}" for x in z))
        emit(f"    max {z.max():.1f} SE   "
             + ("SEPARABLE" if z.max() > 3 else "NOT separable at 3 SE"))
    emit()

    # -- 4. baseline: is the gap about the inversion at all? --------------
    if "baseline_panmictic" in mod["points"] and "control" in emp:
        emit(f"{'=' * 78}\n4. IS THE MODEL-DATA GAP ABOUT THE INVERSION, OR "
             f"THE BASELINE?\n{'=' * 78}")
        base = norm(mod["points"]["baseline_panmictic"]["I"])
        emit("  bin              " + " ".join(
            f"{i + 1:>7d}" for i in range(len(base))))
        emit("  MODEL no-inv     " + " ".join(f"{x:7.4f}" for x in base))
        for pop in ("AA", "BB"):
            emit(f"  EMP collinear {pop} " + " ".join(
                f"{x:7.4f}" for x in norm(emp["control"][pop])))
        emit()
        emit(f"  {'comparison':<40s} {'L1':>8s}")
        for pop in ("AA", "BB"):
            emit(f"  {'collinear ' + pop + ' vs model no-inversion':<40s} "
                 f"{l1_shape(norm(emp['control'][pop]), base):8.4f}")
        for arr, pop in (("I", "AA"), ("S", "BB")):
            emit(f"  {'inversion body ' + pop + ' vs fitted model':<40s} "
                 f"{l1_shape(norm(emp['body'][pop]), norm(mod['points'][a][arr])):8.4f}")
        emit("  If the collinear gap is as large as the inversion-body gap, the")
        emit("  deficit belongs to the NEUTRAL BASELINE (mutation-rate")
        emit("  variation, BGS/DFE -- the project manuscript's sec 8b")
        emit("  spatial-heterogeneity gap), and only the contrast below is")
        emit("  interpretable for the inversion itself.")
        emit()

    # -- 5. the demography-cancelling contrast ----------------------------
    emit(f"{'=' * 78}\n5. DEMOGRAPHY-CANCELLING CONTRAST: inverted vs "
         f"standard\n{'=' * 78}")
    oi, os_ = norm(emp["body"]["AA"]), norm(emp["body"]["BB"])
    emit(f"  {'point':<26s} {'f1(I)':>7s} {'f1(S)':>7s} {'ratio':>7s} "
         f"{'L1(I-S profile)':>16s}")
    emit(f"  {'EMPIRICAL':<26s} {oi[0]:7.4f} {os_[0]:7.4f} "
         f"{oi[0] / os_[0]:7.3f} {'--':>16s}")
    for name in mod["points"]:
        if name == "baseline_panmictic":
            continue
        vi, vs = norm(mod["points"][name]["I"]), norm(mod["points"][name]["S"])
        emit(f"  {name:<26s} {vi[0]:7.4f} {vs[0]:7.4f} {vi[0] / vs[0]:7.3f} "
             f"{np.abs((vi - vs) - (oi - os_)).sum():16.4f}")
    emit("  The ratio is the robust column: any shift moving both arrangements")
    emit("  together leaves it unchanged.")

    (OUT / "sfs_shape.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}/sfs_shape.txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["empirical", "model", "report"], required=True)
    ap.add_argument("--reps", type=int, default=96)
    ap.add_argument("--workers", type=int, default=26)
    a = ap.parse_args()
    if a.stage == "empirical":
        stage_empirical()
    elif a.stage == "model":
        stage_model(a.reps, a.workers)
    else:
        report()


if __name__ == "__main__":
    main()
