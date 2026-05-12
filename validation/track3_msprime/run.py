"""Track 3: msinv ↔ msprime, v_simple demography, no inv, no sweep.

Two sub-scenarios:
- 3_a: no migration (m=0). Pure 2-pop split.
- 3_b: with migration (m=1e-5). Tests the migration mechanism.

L = 5 Mb, n = 50 + 50 (pop0 + pop1). 100 reps per sub-scenario.

Per-rep: run both engines with deterministically-paired seeds, compute
the validation-suite stats panel (per-pop pi/TajD/SFS + dxy/Fst +
tree-shape + LD), save to .npz. After all reps, compute the per-stat
equivalence verdict via the aggregator.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from validation._lib import io, stats
from validation._lib.aggregator import track_equivalence_table
from validation._lib.demography import v_simple_msinv, v_simple_msprime
from validation._lib.engines import msinv_run, msprime_run
from validation._lib.seeds import seed_for


SUBSCENARIOS: dict[str, float] = {"no_mig": 0.0, "mig": 1.0e-5}


def _split_samples_by_pop(ts, n_pop0: int, n_pop1: int) -> dict[str, list[int]]:
    samples = list(ts.samples())
    assert len(samples) == n_pop0 + n_pop1
    return {"pop0": samples[:n_pop0], "pop1": samples[n_pop0:]}


def _compute_and_save_stats(ts, out_dir, seed, n_pop0, n_pop1):
    sset = _split_samples_by_pop(ts, n_pop0, n_pop1)
    win = stats.window_stats(ts, sample_sets=sset, n_windows=40)
    sfs_pop0 = stats.sfs(ts, sample_set=sset["pop0"], folded=True)
    sfs_pop1 = stats.sfs(ts, sample_set=sset["pop1"], folded=True)
    tree_d = stats.tree_shape_stats(ts, n_samples=200, seed=seed + 2)
    bins = np.logspace(2, np.log10(ts.sequence_length), 11)
    ld_d = stats.ld_decay(
        ts, distance_bins=bins, max_pairs=2000, seed=seed + 3,
    )
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
    flat["num_trees"] = np.asarray(ts.num_trees)
    flat["num_sites"] = np.asarray(ts.num_sites)
    io.save_rep_stats(out_dir / "stats.npz", **flat)


def _run_one_rep(rep, subscenario, msinv_dir, msprime_dir, L, n_pop0,
                 n_pop1, mu, r, migration):
    """One full (msinv + msprime) rep. Picklable for ProcessPoolExecutor."""
    seed_a = seed_for(
        track="track3", scenario=subscenario, engine="msinv", rep=rep,
    )
    seed_b = seed_for(
        track="track3", scenario=subscenario, engine="msprime", rep=rep,
    )
    # msinv side
    ts_a = msinv_run(
        demography=v_simple_msinv(two_pop=True, migration=migration),
        sample_config={("S", 0): n_pop0, ("S", 1): n_pop1},
        L=L, r=r, mu=mu, seed=seed_a,
    )
    _compute_and_save_stats(
        ts_a, msinv_dir / f"rep_{rep:03d}", seed=seed_a,
        n_pop0=n_pop0, n_pop1=n_pop1,
    )
    # msprime side
    ts_b = msprime_run(
        demography=v_simple_msprime(two_pop=True, migration=migration),
        samples_by_pop={"pop0": n_pop0, "pop1": n_pop1},
        L=L, r=r, mu=mu, seed=seed_b,
    )
    _compute_and_save_stats(
        ts_b, msprime_dir / f"rep_{rep:03d}", seed=seed_b,
        n_pop0=n_pop0, n_pop1=n_pop1,
    )
    return rep


def _rep_is_done(msinv_dir: Path, msprime_dir: Path, rep: int) -> bool:
    """A rep counts as done when both engines' stats.npz exist."""
    return (
        (msinv_dir / f"rep_{rep:03d}" / "stats.npz").exists()
        and (msprime_dir / f"rep_{rep:03d}" / "stats.npz").exists()
    )


def run_track3_subscenario(
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
    """Run one Track 3 sub-scenario end-to-end with batched parallel reps.

    Reps are processed in batches of ``batch_size``. Each batch uses a
    fresh ProcessPoolExecutor (clean parent RAM, clean worker state).
    If ``resume`` is True, reps whose ``stats.npz`` already exists for
    both engines are skipped — so a crashed/interrupted run can be
    resumed by re-invoking with the same arguments.
    """
    if subscenario not in SUBSCENARIOS:
        raise ValueError(f"unknown sub-scenario: {subscenario}")
    migration = SUBSCENARIOS[subscenario]
    out_root = Path(out_root)
    msinv_dir = out_root / "msinv"
    msprime_dir = out_root / "msprime"
    msinv_dir.mkdir(parents=True, exist_ok=True)
    msprime_dir.mkdir(parents=True, exist_ok=True)

    todo = [
        r for r in range(n_reps)
        if not (resume and _rep_is_done(msinv_dir, msprime_dir, r))
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
                    _run_one_rep(rep, subscenario, msinv_dir, msprime_dir,
                                 L, n_pop0, n_pop1, mu, r, migration)
                    n_done += 1
                except Exception as exc:
                    print(f"  rep {rep} FAILED: {exc!r}", flush=True)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = {
                    ex.submit(
                        _run_one_rep, rep, subscenario, msinv_dir, msprime_dir,
                        L, n_pop0, n_pop1, mu, r, migration,
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

    table = track_equivalence_table(msinv_dir, msprime_dir)
    return {"equivalence_table": table}


def run_track3(
    *,
    out_root: str | Path,
    n_reps: int = 100,
    L: float = 5_000_000,
) -> dict[str, dict]:
    """Run both Track 3 sub-scenarios sequentially."""
    out_root = Path(out_root)
    results = {}
    for sub in SUBSCENARIOS:
        results[sub] = run_track3_subscenario(
            out_root=out_root / sub, subscenario=sub, n_reps=n_reps, L=L,
        )
    return results


def _cli_main():
    import json
    out_root = Path("results/validation/track3")
    results = run_track3(out_root=out_root, n_reps=100)
    for sub, res in results.items():
        print(f"\n=== {sub} ===")
        print(json.dumps(res["equivalence_table"], indent=2, default=float))


if __name__ == "__main__":
    _cli_main()
