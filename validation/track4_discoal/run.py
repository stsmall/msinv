"""Track 4: msinv ↔ discoal, v12 demography, no inv, sweep, L=5 Mb, s=0.1.

Three subscenarios:
- hard: f0 = 1/(2*N_eff), msinv mode='Deterministic', discoal -ws
  (discoal rejects -wd when -en events are present; -ws at alpha=25354
  is statistically indistinguishable from deterministic)
- soft: f0 = 0.05, msinv mode='Stochastic', discoal -ws -f 0.05
- recurrent: discoal -uA at the calibrated rate, msinv
  mode='StochasticConditioned'; matches the D5 calibration framework
  (msinv recurrent_mutation_rate = discoal_uA / (2N), per CLAUDE.md)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from msinv import Sweep
from validation._lib import io, stats
from validation._lib.aggregator import track_equivalence_table
from validation._lib.demography import (
    v12_discoal_events, v12_msinv, V12_DISCOAL_N0,
)
from validation._lib.engines import discoal_run, msinv_run
from validation._lib.seeds import seed_for


S_SEL = 0.1
TAU = 0.0
T_ORIGIN_GENS = 4_000.0  # within F-only era (< T_KF_SPLIT=9194)
F0_SOFT = 0.05
U_RECURRENT_DISCOAL = 5.0e-4  # discoal -uA rate (4N*uA units)


def _msinv_sweep_for(subscenario: str, seed: int) -> Sweep:
    """Build the msinv Sweep object for the given subscenario."""
    if subscenario == "hard":
        mode = "Deterministic"
        f0 = 0.0  # msinv treats f0=0 as de-novo from 1/(2N)
        recurrent_rate = 0.0
    elif subscenario == "soft":
        mode = "Stochastic"
        f0 = F0_SOFT
        recurrent_rate = 0.0
    elif subscenario == "recurrent":
        mode = "StochasticConditioned"
        f0 = 0.0
        # msinv recurrent_mutation_rate = discoal_uA / (2N), per CLAUDE.md
        recurrent_rate = U_RECURRENT_DISCOAL / (2.0 * V12_DISCOAL_N0)
    else:
        raise ValueError(f"unknown subscenario: {subscenario}")
    return Sweep(
        x_sel=2_500_000.0,  # midpoint of L=5 Mb
        tau=TAU,
        origin_pop=1,  # F
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
    """Build the discoal sweep-related CLI args for the subscenario.

    NOTE: discoal rejects -wd (deterministic sweep) when the command
    includes any -en population-size-change events.  The v12 demography
    has many -en events, so we always use -ws (stochastic sweep).
    At alpha=2*N0*s=25354 the stochastic and deterministic trajectories
    are statistically indistinguishable.
    """
    if subscenario == "hard":
        # Use -ws: discoal disallows -wd with -en size-change events.
        # alpha large (25354) → stochastic ≈ deterministic.
        return [
            "-ws", "0.0",
            "-a", str(2.0 * V12_DISCOAL_N0 * S_SEL),
            "-x", str(2_500_000.0 / L),
        ]
    elif subscenario == "soft":
        return [
            "-ws", "0.0",
            "-a", str(2.0 * V12_DISCOAL_N0 * S_SEL),
            "-x", str(2_500_000.0 / L),
            "-f", str(F0_SOFT),
        ]
    elif subscenario == "recurrent":
        return [
            "-ws", "0.0",
            "-a", str(2.0 * V12_DISCOAL_N0 * S_SEL),
            "-x", str(2_500_000.0 / L),
            "-uA", str(U_RECURRENT_DISCOAL),
        ]
    raise ValueError(f"unknown subscenario: {subscenario}")


def _compute_and_save_stats(ts, out_dir, seed, x_sel):
    """Stats panel for Track 4: window stats + tree-shape + LD + H-stats
    + spatial pi profile around x_sel."""
    samples = list(ts.samples())
    sset = {"F": samples}
    win = stats.window_stats(ts, sample_sets=sset, n_windows=40)
    sfs_F = stats.sfs(ts, sample_set=samples, folded=True)
    tree_d = stats.tree_shape_stats(ts, n_samples=200, seed=seed + 2)
    bins = np.logspace(2, np.log10(ts.sequence_length), 11)
    ld_d = stats.ld_decay(
        ts, distance_bins=bins, max_pairs=2000, seed=seed + 3,
    )
    h_local = stats.hstats(
        ts, sample_set=samples, x_sel=x_sel, window_bp=100_000.0,
    )
    h_global = stats.hstats(ts, sample_set=samples)
    flat: dict[str, np.ndarray] = {}
    for name, arr in win["pi"].items():
        flat[f"pi__{name}"] = arr
    for name, arr in win["tajimas_d"].items():
        flat[f"tajimas_d__{name}"] = arr
    flat["sfs__F"] = sfs_F
    flat["tree_tmrca"] = tree_d["tmrca"]
    flat["tree_total_branch"] = tree_d["total_branch"]
    flat["tree_colless"] = tree_d["colless"]
    flat["ld_mean_r2"] = ld_d["mean_r2"]
    flat["h1_local"] = np.asarray(h_local["H1"])
    flat["h12_local"] = np.asarray(h_local["H12"])
    flat["h2_over_h1_local"] = np.asarray(h_local["H2_over_H1"])
    flat["h1_global"] = np.asarray(h_global["H1"])
    flat["num_trees"] = np.asarray(ts.num_trees)
    flat["num_sites"] = np.asarray(ts.num_sites)
    io.save_rep_stats(out_dir / "stats.npz", **flat)


def run_track4_subscenario(
    *,
    out_root: str | Path,
    subscenario: str,
    n_reps: int,
    L: float = 5_000_000,
    n_samples: int = 100,
    mu: float = 1.0e-8,
    r: float = 1.0e-8,
) -> dict:
    """Run one Track 4 subscenario end-to-end."""
    if subscenario not in {"hard", "soft", "recurrent"}:
        raise ValueError(subscenario)
    out_root = Path(out_root)
    msinv_dir = out_root / "msinv"
    discoal_dir = out_root / "discoal"
    msinv_dir.mkdir(parents=True, exist_ok=True)
    discoal_dir.mkdir(parents=True, exist_ok=True)
    x_sel = L * 0.5  # midpoint at any L (smoke + production)

    for rep in range(n_reps):
        seed_a = seed_for(
            track="track4", scenario=subscenario, engine="msinv", rep=rep,
        )
        seed_b1 = seed_for(
            track="track4", scenario=subscenario, engine="discoal_s1", rep=rep,
        )
        seed_b2 = seed_for(
            track="track4", scenario=subscenario, engine="discoal_s2", rep=rep,
        )
        # msinv: build sweep w/ L-adjusted x_sel
        sweep = _msinv_sweep_for(subscenario, seed=seed_a)
        sweep.x_sel = float(x_sel)  # override for any L (smoke uses smaller L)
        ts_a = msinv_run(
            demography=v12_msinv(),
            sample_config={("S", 0): 0, ("S", 1): n_samples},
            L=L, r=r, mu=mu, seed=seed_a,
            inversions=None, sweeps=[sweep],
        )
        _compute_and_save_stats(
            ts_a, msinv_dir / f"rep_{rep:03d}", seed=seed_a, x_sel=x_sel,
        )
        # discoal: build CLI args; sampleSize spec via `-p 2 0 n_samples`
        # to mirror msinv's K=0 + F=n_samples sample config.
        discoal_demog = v12_discoal_events()
        discoal_sweep = _discoal_sweep_args_for(subscenario, L=L)
        # Override -x for any L (smoke uses smaller L)
        discoal_sweep_fixed = []
        i = 0
        while i < len(discoal_sweep):
            if discoal_sweep[i] == "-x":
                discoal_sweep_fixed.extend(["-x", str(0.5)])
                i += 2
            else:
                discoal_sweep_fixed.append(discoal_sweep[i])
                i += 1
        all_args = (
            ["-p", "2", "0", str(n_samples)]
            + discoal_demog
            + discoal_sweep_fixed
        )
        tmp_dir = discoal_dir / f"rep_{rep:03d}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        ts_b = discoal_run(
            n_samples=n_samples,
            L=L, r=r, mu=mu,
            seed=(seed_b1, seed_b2),
            ne_diploid=V12_DISCOAL_N0,
            demography_args=all_args,
            sweep_args=None,  # already in all_args
            tmp_dir=tmp_dir,
        )
        _compute_and_save_stats(
            ts_b, tmp_dir, seed=seed_b1, x_sel=x_sel,
        )

    table = track_equivalence_table(msinv_dir, discoal_dir)
    return {"equivalence_table": table}


def run_track4(
    *,
    out_root: str | Path,
    n_reps: int = 100,
    L: float = 5_000_000,
) -> dict[str, dict]:
    """Run all three Track 4 subscenarios sequentially."""
    out_root = Path(out_root)
    results = {}
    for sub in ("hard", "soft", "recurrent"):
        results[sub] = run_track4_subscenario(
            out_root=out_root / sub, subscenario=sub, n_reps=n_reps, L=L,
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
