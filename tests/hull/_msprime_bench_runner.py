"""Child-process runner for the msprime validation harness.

Invoked as ``python -m tests.hull._msprime_bench_runner --scenario NAME
--engine {msinv,msprime} --n-reps N --seed-base K``. Runs the rep batch
and writes a JSON document to stdout::

    {"per_rep_stats": [{stat_name: value, ...}, ...], "per_rep_seconds": [float, ...]}

Importing this module does NOT execute any sim. It is pytest-collectable
but pytest-skipped — pytest sees ``pytestmark = pytest.mark.skip(...)``.
"""

import argparse
import json
import sys
import time

import pytest

pytestmark = pytest.mark.skip("child runner — invoked via subprocess")


# Scenario registry filled in by Phase A2/A3 and Phase C tasks.
# Each entry: name -> {
#   "compute_afs": bool,
#   "n_pops": int,
#   "make_msinv": Callable[[int], (ts, sample_sets_or_None)],
#   "make_msprime": Callable[[int], (ts, sample_sets_or_None)],
# }
SCENARIOS: dict[str, dict] = {}


def _stats_from_ts(ts, sample_sets, compute_afs: bool):
    """Branch-length stats + (optional) AFS bins from a tskit TS.

    Returns a dict suitable for JSON serialization. Single-pop AFS bins
    are keyed ``afs_bin_{k}``; multi-pop marginals are keyed
    ``afs_p{p}_bin_{k}``. Edge bins (0 and n_set) are excluded.
    """
    out: dict[str, float] = {
        "pi_branch": ts.diversity(mode="branch"),
        "n_trees": float(ts.num_trees),
    }
    weighted = sum(tree.time(tree.root) * tree.span for tree in ts.trees())
    out["mean_tmrca"] = weighted / ts.sequence_length
    if sample_sets is not None:
        out["dxy_branch"] = _mean_pairwise_divergence(ts, sample_sets)
    if compute_afs:
        if sample_sets is None:
            afs = ts.allele_frequency_spectrum(
                mode="branch", polarised=True, span_normalise=True)
            for k in range(1, len(afs) - 1):
                out[f"afs_bin_{k}"] = float(afs[k])
        else:
            for p, samples in enumerate(sample_sets):
                afs = ts.allele_frequency_spectrum(
                    sample_sets=[samples], mode="branch",
                    polarised=True, span_normalise=True)
                for k in range(1, len(afs) - 1):
                    out[f"afs_p{p}_bin_{k}"] = float(afs[k])
    return out


def _mean_pairwise_divergence(ts, sample_sets):
    """For 2-pop scenarios, return the single A-B branch divergence.
    For 3-pop, return the mean of the three pairwise divergences.
    """
    n = len(sample_sets)
    if n == 2:
        return float(ts.divergence(sample_sets=sample_sets, mode="branch"))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    vals = [
        float(ts.divergence(
            sample_sets=[sample_sets[i], sample_sets[j]], mode="branch"))
        for i, j in pairs
    ]
    return sum(vals) / len(vals)


def _run_one_engine(scenario_name: str, engine: str,
                    n_reps: int, seed_base: int):
    spec = SCENARIOS[scenario_name]
    factory = (spec["make_msinv"] if engine == "msinv"
               else spec["make_msprime"])
    compute_afs = spec["compute_afs"]
    # warm-up rep, untimed (engine import / .so warm-up)
    factory(seed=seed_base)
    per_rep_stats = []
    per_rep_seconds = []
    for i in range(n_reps):
        t0 = time.perf_counter()
        ts, sample_sets = factory(seed=seed_base + i)
        stats = _stats_from_ts(ts, sample_sets, compute_afs)
        per_rep_seconds.append(time.perf_counter() - t0)
        per_rep_stats.append(stats)
    return per_rep_stats, per_rep_seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    ap.add_argument("--engine", required=True, choices=["msinv", "msprime"])
    ap.add_argument("--n-reps", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=0)
    args = ap.parse_args()
    per_rep_stats, per_rep_seconds = _run_one_engine(
        args.scenario, args.engine, args.n_reps, args.seed_base)
    json.dump(
        {"per_rep_stats": per_rep_stats,
         "per_rep_seconds": per_rep_seconds},
        sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
