"""Track 4: msinv ↔ discoal, v_simple 2-pop + migration, sweep in pop 0 only.

Demography: v_simple (pop 0 = 1e6, pop 1 = 1e5, split 15k gen ago,
N_anc=1e6, symmetric migration m=1e-5). Sweep originates in pop 0;
pop 1 is the neutral outgroup that receives migrants at rate m.

Three sub-scenarios:
- hard: f0 = 1/(2*N_eff), msinv mode='Deterministic', discoal -ws
  (discoal rejects -wd when -en events are present; -ws at large
  alpha is statistically indistinguishable from deterministic).
- soft: f0 = 0.05, msinv mode='Stochastic', discoal -ws -f 0.05.
- recurrent: discoal -uA at the calibrated rate, msinv
  mode='StochasticConditioned'; matches the D5 calibration
  (msinv recurrent_mutation_rate = discoal_uA / (2N), per CLAUDE.md).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from msinv import Sweep
from validation._lib import io, stats
from validation._lib.aggregator import track_equivalence_table
from validation._lib.demography import (
    v_simple_discoal_events,
    v_simple_msinv,
    V_SIMPLE_DISCOAL_N0,
)
from validation._lib.engines import discoal_run, msinv_run
from validation._lib.seeds import seed_for


S_SEL = 0.1
TAU = 0.0
T_ORIGIN_GENS = 4_000.0  # post-split (split=15,000 gen); within pop0/pop1 era
F0_SOFT = 0.05
U_RECURRENT_DISCOAL = 5.0e-4  # discoal -uA rate (4N*uA units)
MIGRATION = 1.0e-5
# discoal `-ws tau` is in 4*N0 coalescent units (parsed as `tau*2` in 2N
# units inside discoal_multipop.c:582). Convert msinv's t_origin (gens)
# to match. Without this, `-ws 0.0` puts discoal's sweep at the present
# while msinv runs it 4000 gens ago — different scenarios, breaking all
# sweep-window stats (verified 2026-05-13 against rep_000).
DISCOAL_WS_TAU = T_ORIGIN_GENS / (4.0 * V_SIMPLE_DISCOAL_N0)


def _msinv_sweep_for(subscenario: str, seed: int) -> Sweep:
    if subscenario == "hard":
        mode = "Deterministic"
        # Explicit de-novo f0. The prior comment "f0=0 as de-novo" was
        # wrong: f0=0 produces a flat-zero trajectory and a no-op sweep
        # (verified 2026-05-12). Use 1/(2N_eff) so the trajectory rises
        # from 1/(2N) to 1 deterministically.
        f0 = 1.0 / (2.0 * V_SIMPLE_DISCOAL_N0)
        recurrent_rate = 0.0
    elif subscenario == "soft":
        mode = "Stochastic"
        f0 = F0_SOFT
        recurrent_rate = 0.0
    elif subscenario == "recurrent":
        mode = "StochasticConditioned"
        f0 = 0.0
        recurrent_rate = U_RECURRENT_DISCOAL / (2.0 * V_SIMPLE_DISCOAL_N0)
    else:
        raise ValueError(f"unknown subscenario: {subscenario}")
    return Sweep(
        x_sel=2_500_000.0,  # midpoint of L=5 Mb
        tau=TAU,
        origin_pop=0,  # sweep in pop 0 (the big pop), to match discoal
        origin_kary="S",
        target_inv=0,  # no inversion; target_inv=0 is sentinel
        mode=mode,
        s=S_SEL,
        t_origin=T_ORIGIN_GENS,
        f0=f0,
        recurrent_mutation_rate=recurrent_rate,
        seed=int(seed),
    )


def _discoal_sweep_args_for(subscenario: str, L: float) -> list[str]:
    """discoal sweep CLI args. Sweep is on pop 0 (default for discoal)."""
    if subscenario == "hard":
        return [
            "-ws", f"{DISCOAL_WS_TAU:.10g}",
            "-a", str(2.0 * V_SIMPLE_DISCOAL_N0 * S_SEL),
            "-x", str(2_500_000.0 / L),
        ]
    elif subscenario == "soft":
        return [
            "-ws", f"{DISCOAL_WS_TAU:.10g}",
            "-a", str(2.0 * V_SIMPLE_DISCOAL_N0 * S_SEL),
            "-x", str(2_500_000.0 / L),
            "-f", str(F0_SOFT),
        ]
    elif subscenario == "recurrent":
        return [
            "-ws", f"{DISCOAL_WS_TAU:.10g}",
            "-a", str(2.0 * V_SIMPLE_DISCOAL_N0 * S_SEL),
            "-x", str(2_500_000.0 / L),
            "-uA", str(U_RECURRENT_DISCOAL),
        ]
    raise ValueError(f"unknown subscenario: {subscenario}")


def _split_samples_by_pop(ts, n_pop0: int, n_pop1: int) -> dict[str, list[int]]:
    samples = list(ts.samples())
    assert len(samples) == n_pop0 + n_pop1
    return {"pop0": samples[:n_pop0], "pop1": samples[n_pop0:]}


def _compute_and_save_stats(ts, out_dir, seed, x_sel, n_pop0, n_pop1):
    """Stats panel: per-pop π/TajD/SFS + dxy/Fst + tree-shape + LD +
    H-stats (local + global), at L=5 Mb."""
    sset = _split_samples_by_pop(ts, n_pop0, n_pop1)
    win = stats.window_stats(ts, sample_sets=sset, n_windows=40)
    sfs_pop0 = stats.sfs(ts, sample_set=sset["pop0"], folded=True)
    sfs_pop1 = stats.sfs(ts, sample_set=sset["pop1"], folded=True)
    tree_d = stats.tree_shape_stats(ts, n_samples=200, seed=seed + 2)
    bins = np.logspace(2, np.log10(ts.sequence_length), 11)
    ld_d = stats.ld_decay(
        ts, distance_bins=bins, max_pairs=2000, seed=seed + 3,
    )
    # H-stats: sweep is in pop 0, so the local/global signal lives there.
    h_local_pop0 = stats.hstats(
        ts, sample_set=sset["pop0"], x_sel=x_sel, window_bp=100_000.0,
    )
    h_global_pop0 = stats.hstats(ts, sample_set=sset["pop0"])
    h_global_pop1 = stats.hstats(ts, sample_set=sset["pop1"])
    flat: dict[str, np.ndarray] = {}
    for name, arr in win["pi"].items():
        flat[f"pi__{name}"] = arr
    for name, arr in win["tajimas_d"].items():
        flat[f"tajimas_d__{name}"] = arr
    for key, arr in win["dxy"].items():
        flat[f"dxy__{key}"] = arr
    for key, arr in win["fst"].items():
        flat[f"fst__{key}"] = arr
    flat["sfs__pop0"] = sfs_pop0
    flat["sfs__pop1"] = sfs_pop1
    flat["tree_tmrca"] = tree_d["tmrca"]
    flat["tree_total_branch"] = tree_d["total_branch"]
    flat["tree_colless"] = tree_d["colless"]
    flat["ld_mean_r2"] = ld_d["mean_r2"]
    flat["h1_local_pop0"] = np.asarray(h_local_pop0["H1"])
    flat["h12_local_pop0"] = np.asarray(h_local_pop0["H12"])
    flat["h2_over_h1_local_pop0"] = np.asarray(h_local_pop0["H2_over_H1"])
    flat["h1_global_pop0"] = np.asarray(h_global_pop0["H1"])
    flat["h1_global_pop1"] = np.asarray(h_global_pop1["H1"])
    flat["num_trees"] = np.asarray(ts.num_trees)
    flat["num_sites"] = np.asarray(ts.num_sites)
    io.save_rep_stats(out_dir / "stats.npz", **flat)


def _run_one_rep(rep, subscenario, msinv_dir, discoal_dir, L, n_pop0,
                 n_pop1, mu, r, x_sel):
    seed_a = seed_for(
        track="track4", scenario=subscenario, engine="msinv", rep=rep,
    )
    seed_b1 = seed_for(
        track="track4", scenario=subscenario, engine="discoal_s1", rep=rep,
    )
    seed_b2 = seed_for(
        track="track4", scenario=subscenario, engine="discoal_s2", rep=rep,
    )
    # msinv side
    sweep = _msinv_sweep_for(subscenario, seed=seed_a)
    sweep.x_sel = float(x_sel)
    ts_a = msinv_run(
        demography=v_simple_msinv(two_pop=True, migration=MIGRATION),
        sample_config={("S", 0): n_pop0, ("S", 1): n_pop1},
        L=L, r=r, mu=mu, seed=seed_a,
        inversions=None, sweeps=[sweep],
    )
    _compute_and_save_stats(
        ts_a, msinv_dir / f"rep_{rep:03d}", seed=seed_a, x_sel=x_sel,
        n_pop0=n_pop0, n_pop1=n_pop1,
    )
    # discoal side
    discoal_demog = v_simple_discoal_events(
        two_pop=True, migration=MIGRATION,
    )
    discoal_sweep = _discoal_sweep_args_for(subscenario, L=L)
    # discoal -x is in normalized [0,1] units; engine helper expects raw bp
    # and translates, but here we already wrote the position as 0.5 in
    # discoal_sweep. discoal -x is normalized so 2.5Mb/L=0.5 is correct.
    all_args = (
        ["-p", "2", str(n_pop0), str(n_pop1)]
        + discoal_demog
        + discoal_sweep
    )
    tmp_dir = discoal_dir / f"rep_{rep:03d}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ts_b = discoal_run(
        n_samples=n_pop0 + n_pop1,
        L=L, r=r, mu=mu,
        seed=(seed_b1, seed_b2),
        ne_diploid=V_SIMPLE_DISCOAL_N0,
        demography_args=all_args,
        sweep_args=None,
        tmp_dir=tmp_dir,
    )
    _compute_and_save_stats(
        ts_b, tmp_dir, seed=seed_b1, x_sel=x_sel,
        n_pop0=n_pop0, n_pop1=n_pop1,
    )
    return rep


def _rep_is_done(msinv_dir: Path, discoal_dir: Path, rep: int) -> bool:
    """A rep counts as done when both engines' stats.npz exist."""
    return (
        (msinv_dir / f"rep_{rep:03d}" / "stats.npz").exists()
        and (discoal_dir / f"rep_{rep:03d}" / "stats.npz").exists()
    )


def run_track4_subscenario(
    *,
    out_root: str | Path,
    subscenario: str,
    n_reps: int,
    L: float = 5_000_000,
    n_pop0: int = 50,
    n_pop1: int = 50,
    mu: float = 1.0e-8,
    r: float = 1.0e-8,
    max_workers: int = 50,
    batch_size: int = 10,
    resume: bool = True,
) -> dict:
    """Batched parallel run with resume-from-existing.

    Reps are processed in batches of ``batch_size``. Each batch uses a
    fresh ProcessPoolExecutor (clean parent RAM, clean worker state).
    If ``resume`` is True, reps whose ``stats.npz`` already exists for
    both engines are skipped — so a crashed/interrupted run can be
    resumed by re-invoking with the same arguments.
    """
    if subscenario not in {"hard", "soft", "recurrent"}:
        raise ValueError(subscenario)
    out_root = Path(out_root)
    msinv_dir = out_root / "msinv"
    discoal_dir = out_root / "discoal"
    msinv_dir.mkdir(parents=True, exist_ok=True)
    discoal_dir.mkdir(parents=True, exist_ok=True)
    x_sel = L * 0.5

    todo = [
        r for r in range(n_reps)
        if not (resume and _rep_is_done(msinv_dir, discoal_dir, r))
    ]
    n_skipped = n_reps - len(todo)
    if n_skipped:
        print(
            f"  {subscenario}: resuming, {n_skipped}/{n_reps} reps already done",
            flush=True,
        )

    n_done = n_skipped
    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start : batch_start + batch_size]
        if max_workers <= 1:
            for rep in batch:
                try:
                    _run_one_rep(rep, subscenario, msinv_dir, discoal_dir,
                                 L, n_pop0, n_pop1, mu, r, x_sel)
                    n_done += 1
                except Exception as exc:
                    print(f"  rep {rep} FAILED: {exc!r}", flush=True)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = {
                    ex.submit(
                        _run_one_rep, rep, subscenario, msinv_dir, discoal_dir,
                        L, n_pop0, n_pop1, mu, r, x_sel,
                    ): rep
                    for rep in batch
                }
                for fut in as_completed(futures):
                    rep_idx = futures[fut]
                    try:
                        fut.result()
                        n_done += 1
                    except Exception as exc:
                        print(f"  rep {rep_idx} FAILED: {exc!r}", flush=True)
        print(
            f"  {subscenario}: batch {batch_start // batch_size + 1} done, "
            f"{n_done}/{n_reps} total reps complete",
            flush=True,
        )

    table = track_equivalence_table(msinv_dir, discoal_dir)
    return {"equivalence_table": table}


def run_track4(
    *,
    out_root: str | Path,
    n_reps: int = 100,
    L: float = 5_000_000,
    max_workers: int = 50,
    batch_size: int = 10,
    resume: bool = True,
) -> dict[str, dict]:
    out_root = Path(out_root)
    results = {}
    for sub in ("hard", "soft", "recurrent"):
        results[sub] = run_track4_subscenario(
            out_root=out_root / sub, subscenario=sub, n_reps=n_reps, L=L,
            max_workers=max_workers, batch_size=batch_size, resume=resume,
        )
    return results


def _cli_main():
    import json
    out_root = Path("results/validation/track4")
    results = run_track4(out_root=out_root, n_reps=100)
    for sub, res in results.items():
        print(f"\n=== {sub} ===")
        print(json.dumps(res["equivalence_table"], indent=2, default=float))


if __name__ == "__main__":
    _cli_main()
