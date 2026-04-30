"""Child-process runner for the discoal validation harness.

Invoked as ``python -m tests.hull._discoal_bench_runner --scenario NAME
--engine {msinv,discoal} --n-reps N --seed-base K``.  Runs the rep
batch and writes a JSON document to stdout::

    {"per_rep_stats": [{stat_name: value, ...}, ...], "per_rep_seconds": [float, ...]}

Importing this module does NOT execute any sim. It is pytest-collectable
but pytest-skipped.
"""

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import time

import pytest
import tskit

from msinv.hull.simulator import HullSimulator

pytestmark = pytest.mark.skip("child runner — invoked via subprocess")


DISCOAL_BIN = "/home/adkern/discoal/discoal"


# Scenario registry filled in by Task B2 (D1) and Phase C tasks (D2-D5).
# Each entry: name -> {
#   "n_pops": int (always 1 for discoal track v1),
#   "ne_diploid": float,  # diploid Ne; used to rescale discoal's 2N-unit stats
#   "compute_pi_windows": bool,
#   "windows_left_to_right": list[float] | None,  # bin edges within [0, L]
#   "make_msinv": Callable[[int], tskit.TreeSequence],   # kwarg: seed -> ts
#   "make_discoal_args": Callable[..., list[str]],
#       # kwargs: out_prefix: str, seed1: int, seed2: int, n_reps: int -> argv
#   "L": float,  # sequence length, needed for window math
#   "x_sel": float | None,  # sweep position, for footprint folding
# }
SCENARIOS: dict[str, dict] = {}


def _gens_to_discoal_time(gens: float, ne_diploid: float) -> float:
    """Convert generations (msinv side) to discoal CLI time (4N units).

    Empirically pinned via B0 task probe: deterministic sweep at tau=0.05
    with Ne=10000 produces tree heights at the sweep position clustered
    around 2343 gens (consistent with 4N convention: 0.05*4*10000=2000
    gens + ~17% forward-sweep overhead). Matches
    docs/discoal_msprime_parameter_guide.md.
    """
    return gens / (4.0 * ne_diploid)


def _s_to_alpha(s: float, ne_diploid: float) -> float:
    """Convert per-generation selection coefficient s to discoal -a alpha."""
    return 2.0 * ne_diploid * s


def _stats_from_ts(ts, scenario_spec):
    """Branch-length stats from a tskit TS, optionally with windowed pi.

    Returns dict suitable for JSON serialization.  Single-pop only.

    Cross-engine convention quirk surfaced by D1 first run: discoal v2
    emits tree sequences with ``ts.time_units == "coalescent units (2N
    generations)"`` while msinv emits them in generations.  Branch-mode
    stats need to be rescaled by ``2 * ne_diploid`` on the discoal side
    so both engines report in generations.

    Stats included: ``pi_branch`` (diploid-Ne-equivalent expected pairwise
    coal time x 2) and ``n_trees`` (recombination-driven local tree count).
    Local-tree root time stats (``mean_tmrca``, ``tmrca_grand``) are NOT
    cross-engine comparable: discoal's full-ARG ``-F`` output collapses
    all local-tree roots to the grand MRCA, while msinv's full-ARG
    records per-local-tree roots plus above-root non-ancestral edges.
    These produce different distributional quantities under
    ``tree.time(tree.root)`` and were dropped after D1 confirmed
    pi_branch + n_trees alone validate the convention bridge cleanly.
    """
    ne_diploid = scenario_spec["ne_diploid"]
    time_scale = (
        2.0 * ne_diploid
        if str(ts.time_units).startswith("coalescent units")
        else 1.0
    )
    out: dict[str, float] = {
        "pi_branch": ts.diversity(mode="branch") * time_scale,
        "n_trees": float(ts.num_trees),
    }

    if scenario_spec.get("compute_pi_windows", False):
        # Folded windowed pi around x_sel.  K bins at distance
        # [0, w], [w, 2w], ..., [(K-1)w, K*w] from the sweep site.
        # Each bin sums the left and right halves of the genome.
        L = scenario_spec["L"]
        x_sel = scenario_spec["x_sel"]
        edges = scenario_spec["windows_left_to_right"]
        K = len(edges) - 1
        # Build windows on the [0, L] coordinate.  Each folded bin k
        # corresponds to two windows: [x_sel - edges[k+1], x_sel - edges[k]]
        # and [x_sel + edges[k], x_sel + edges[k+1]], clamped to [0, L].
        windowed = [0.0] * K
        for k in range(K):
            lo, hi = edges[k], edges[k + 1]
            left_lo = max(0.0, x_sel - hi)
            left_hi = max(0.0, x_sel - lo)
            right_lo = min(L, x_sel + lo)
            right_hi = min(L, x_sel + hi)
            wins = []
            if left_hi > left_lo:
                wins.extend([left_lo, left_hi])
            if right_hi > right_lo:
                wins.extend([right_lo, right_hi])
            if not wins:
                continue
            # tskit needs a sorted, contiguous-edge list starting at 0
            # and ending at L for diversity(windows=...).  Build a full
            # list with masking.
            full = [0.0]
            for v in wins:
                if v > full[-1]:
                    full.append(v)
            if full[-1] < L:
                full.append(L)
            # ts.diversity returns one value per [edges[i], edges[i+1]];
            # we need to identify which of those are inside our sub-windows.
            divs = ts.diversity(windows=full, mode="branch")
            total_span = 0.0
            total_pi_span = 0.0
            for i in range(len(full) - 1):
                seg_lo, seg_hi = full[i], full[i + 1]
                # Inside left window?
                in_left = (seg_lo >= left_lo and seg_hi <= left_hi)
                in_right = (seg_lo >= right_lo and seg_hi <= right_hi)
                if in_left or in_right:
                    span = seg_hi - seg_lo
                    total_span += span
                    total_pi_span += divs[i] * span
            windowed[k] = (
                (total_pi_span / total_span) * time_scale
                if total_span > 0
                else 0.0
            )
        for k, val in enumerate(windowed):
            out[f"pi_window_{k}"] = float(val)
    return out


def _run_one_engine_batch(scenario_name, engine, n_reps, seed_base):
    """Returns (per_rep_stats, per_rep_seconds)."""
    spec = SCENARIOS[scenario_name]
    if engine == "msinv":
        return _run_msinv_batch(spec, n_reps, seed_base)
    elif engine == "discoal":
        return _run_discoal_batch(spec, n_reps, seed_base)
    raise ValueError(f"unknown engine: {engine}")


def _run_msinv_batch(spec, n_reps, seed_base):
    factory = spec["make_msinv"]
    # warm-up rep, untimed
    factory(seed=seed_base)
    per_rep_stats = []
    per_rep_seconds = []
    for i in range(n_reps):
        t0 = time.perf_counter()
        ts = factory(seed=seed_base + i)
        stats = _stats_from_ts(ts, spec)
        per_rep_seconds.append(time.perf_counter() - t0)
        per_rep_stats.append(stats)
    return per_rep_stats, per_rep_seconds


def _run_discoal_batch(spec, n_reps, seed_base):
    """Run discoal once with numReplicates=n_reps; load each .trees file."""
    make_args = spec["make_discoal_args"]
    # discoal seeds are two integers; derive both from seed_base
    seed1 = seed_base + 1
    seed2 = seed_base + 100001
    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = pathlib.Path(tmpdir) / "scenario.trees"
        argv = make_args(out_prefix=str(out_prefix), seed1=seed1, seed2=seed2,
                         n_reps=n_reps)
        # discoal writes DEBUG to stdout; redirect to /dev/null so it
        # doesn't pollute our subprocess pipe budget.
        t0 = time.perf_counter()
        result = subprocess.run(
            argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False)
        wall = time.perf_counter() - t0
        if result.returncode != 0:
            raise RuntimeError(
                f"discoal failed (rc={result.returncode}); argv={argv}")
        # discoal writes <prefix>_rep1.trees ... _rep{n_reps}.trees
        # Strip trailing ".trees" so the actual files are
        # <prefix-without-extension>_rep{N}.trees
        prefix_base = str(out_prefix).removesuffix(".trees")
        per_rep_stats = []
        per_rep_seconds = []
        for i in range(n_reps):
            t0 = time.perf_counter()
            tree_path = f"{prefix_base}_rep{i + 1}.trees"
            ts = tskit.load(tree_path)
            stats = _stats_from_ts(ts, spec)
            per_rep_seconds.append(time.perf_counter() - t0)
            per_rep_stats.append(stats)
    # Total wall time was for the full discoal call; we report the
    # parsing time per rep.  The combined "total wall" for a rep is
    # the parse + (run / n_reps) — but for benchmark fidelity we
    # attribute the discoal subprocess wall as a single chunk added
    # to rep 0's time.  Per-rep timing for the engine comparison is
    # parse-only; the total in the benchmark block reflects subprocess
    # wall via os.wait4 rusage at the parent level.
    if per_rep_seconds:
        per_rep_seconds[0] += wall
    return per_rep_stats, per_rep_seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    ap.add_argument("--engine", required=True, choices=["msinv", "discoal"])
    ap.add_argument("--n-reps", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=0)
    args = ap.parse_args()
    per_rep_stats, per_rep_seconds = _run_one_engine_batch(
        args.scenario, args.engine, args.n_reps, args.seed_base)
    json.dump(
        {"per_rep_stats": per_rep_stats,
         "per_rep_seconds": per_rep_seconds},
        sys.stdout)
    sys.stdout.write("\n")


def _make_d1_msinv(seed: int):
    return HullSimulator(
        samples=10,
        population_size=10000.0,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        inversions=[],
        sweeps=None,
        seed=seed,
    ).simulate()


def _make_d1_discoal_args(out_prefix: str, seed1: int, seed2: int,
                          n_reps: int):
    return [
        DISCOAL_BIN,
        "10",            # sampleSize (haploid)
        str(n_reps),     # numReplicates
        "100000",        # nSites
        "-t", "40",      # theta = 4*Ne*mu*L = 4*10000*1e-8*1e5 = 40
        "-r", "40",      # rho   = 4*Ne*r *L = 4*10000*1e-8*1e5 = 40
        "-N", "10000",   # diploid Ne
        "-ts", out_prefix,
        "-F",            # full ARG mode (matches msinv record_full_arg=True)
        "-d", str(seed1), str(seed2),
    ]


SCENARIOS["d1"] = {
    "n_pops": 1,
    "ne_diploid": 10000.0,
    "compute_pi_windows": False,
    "windows_left_to_right": None,
    "make_msinv": _make_d1_msinv,
    "make_discoal_args": _make_d1_discoal_args,
    "L": 100_000.0,
    "x_sel": None,
}


# ---------------------------------------------------------------------------
# D2 — hard sweep
# ---------------------------------------------------------------------------
# Deviation from the plan: msinv side uses mode='Deterministic' and discoal
# uses '-wd' instead of '-ws'. PS2 (per-segment hitchhiking spatial profile
# study) established that Stochastic sweeps at f0=1/(2N) are extinction-prone
# and produce high-variance / biased output for hard-fixation parameter
# regimes. Both engines now run the deterministic logistic trajectory; this
# matches what the moments-based comparison expects and removes the
# extinction-rejection sampling bias.
def _make_d2_msinv(seed: int):
    from msinv.hull.sweep import Sweep
    sweep = Sweep(
        x_sel=50_000.0,
        tau=1000.0,
        origin_pop=0,
        origin_kary='S',
        target_inv=0,
        mode='Deterministic',
        s=0.05,
        t_origin=1500.0,
        f0=1.0 / (2 * 10_000),
        partial_sweep_final_freq=1.0,
        seed=seed,
    )
    return HullSimulator(
        samples=10,
        population_size=10000.0,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        inversions=[],
        sweeps=[sweep],
        seed=seed,
    ).simulate()


def _make_d2_discoal_args(out_prefix: str, seed1: int, seed2: int,
                          n_reps: int):
    # tau in 4N units: 1000 gens / (4 * 10000) = 0.025
    # alpha = 2 * Ne * s = 2 * 10000 * 0.05 = 1000
    return [
        DISCOAL_BIN,
        "10", str(n_reps), "100000",
        "-t", "40", "-r", "40", "-N", "10000",
        "-wd", "0.025",
        "-a", "1000",
        "-x", "0.5",
        "-ts", out_prefix, "-F",
        "-d", str(seed1), str(seed2),
    ]


SCENARIOS["d2"] = {
    "n_pops": 1,
    "ne_diploid": 10000.0,
    "compute_pi_windows": False,
    "windows_left_to_right": None,
    "make_msinv": _make_d2_msinv,
    "make_discoal_args": _make_d2_discoal_args,
    "L": 100_000.0,
    "x_sel": 50_000.0,
}


# ---------------------------------------------------------------------------
# D3 — soft sweep from standing variation
# ---------------------------------------------------------------------------
# f0=0.05 means the beneficial allele starts at 5% frequency across the
# population (~1000 founder copies in a 2N=20000 pool), so the stochastic
# trajectory is no longer extinction-prone. Both engines run stochastic:
# msinv mode='Stochastic', discoal -ws -f 0.05.
def _make_d3_msinv(seed: int):
    from msinv.hull.sweep import Sweep
    sweep = Sweep(
        x_sel=50_000.0,
        tau=1000.0,
        origin_pop=0,
        origin_kary='S',
        target_inv=0,
        mode='Stochastic',
        s=0.05,
        t_origin=1500.0,
        f0=0.05,
        partial_sweep_final_freq=1.0,
        seed=seed,
    )
    return HullSimulator(
        samples=10,
        population_size=10000.0,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        inversions=[],
        sweeps=[sweep],
        seed=seed,
    ).simulate()


def _make_d3_discoal_args(out_prefix: str, seed1: int, seed2: int,
                          n_reps: int):
    return [
        DISCOAL_BIN,
        "10", str(n_reps), "100000",
        "-t", "40", "-r", "40", "-N", "10000",
        "-ws", "0.025",
        "-a", "1000",
        "-x", "0.5",
        "-f", "0.05",
        "-ts", out_prefix, "-F",
        "-d", str(seed1), str(seed2),
    ]


SCENARIOS["d3"] = {
    "n_pops": 1,
    "ne_diploid": 10000.0,
    "compute_pi_windows": False,
    "windows_left_to_right": None,
    "make_msinv": _make_d3_msinv,
    "make_discoal_args": _make_d3_discoal_args,
    "L": 100_000.0,
    "x_sel": 50_000.0,
}


# ---------------------------------------------------------------------------
# D4 — partial sweep (final freq 0.5)
# ---------------------------------------------------------------------------
# Like D2 but the trajectory plateaus at p=0.5 instead of fixing. f0=1/(2N)
# again uses Deterministic to dodge the extinction sampling bias.
def _make_d4_msinv(seed: int):
    from msinv.hull.sweep import Sweep
    sweep = Sweep(
        x_sel=50_000.0,
        tau=1000.0,
        origin_pop=0,
        origin_kary='S',
        target_inv=0,
        mode='Deterministic',
        s=0.05,
        t_origin=1500.0,
        f0=1.0 / (2 * 10_000),
        partial_sweep_final_freq=0.5,
        seed=seed,
    )
    return HullSimulator(
        samples=10,
        population_size=10000.0,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        inversions=[],
        sweeps=[sweep],
        seed=seed,
    ).simulate()


def _make_d4_discoal_args(out_prefix: str, seed1: int, seed2: int,
                          n_reps: int):
    return [
        DISCOAL_BIN,
        "10", str(n_reps), "100000",
        "-t", "40", "-r", "40", "-N", "10000",
        "-wd", "0.025",
        "-a", "1000",
        "-x", "0.5",
        "-c", "0.5",
        "-ts", out_prefix, "-F",
        "-d", str(seed1), str(seed2),
    ]


SCENARIOS["d4"] = {
    "n_pops": 1,
    "ne_diploid": 10000.0,
    "compute_pi_windows": False,
    "windows_left_to_right": None,
    "make_msinv": _make_d4_msinv,
    "make_discoal_args": _make_d4_discoal_args,
    "L": 100_000.0,
    "x_sel": 50_000.0,
}


if __name__ == "__main__":
    main()
