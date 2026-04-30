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

import msprime
import pytest
from msinv.hull.demography import Demography
from msinv.hull.simulator import HullSimulator

pytestmark = pytest.mark.skip("child runner — invoked via subprocess")


# Scenario registry filled in by Phase A2/A3 and Phase C tasks.
# Each entry: name -> {
#   "compute_afs": bool,
#   "n_pops": int,
#   "make_msinv": Callable[[int], (ts, sample_sets_or_None)],
#   "make_msprime": Callable[[int], (ts, sample_sets_or_None)],
# }
SCENARIOS: dict[str, dict] = {}


def _make_n1_msinv(seed: int):
    ts = HullSimulator(
        samples=10,
        population_size=10000.0,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        inversions=[],
        seed=seed,
    ).simulate()
    return ts, None


def _make_n1_msprime(seed: int):
    # population_size doubled vs msinv: msinv N = diploid Ne (2N chrom);
    # msprime ploidy=1 reads N as haploid Ne. record_full_arg=True so
    # non-ancestral recombs survive into the TS (msinv's convention).
    ts = msprime.sim_ancestry(
        samples=10,
        population_size=20000.0,
        sequence_length=100_000,
        recombination_rate=1e-8,
        ploidy=1,
        record_full_arg=True,
        random_seed=seed + 1,
    )
    return ts, None


SCENARIOS["n1"] = {
    "compute_afs": False,
    "n_pops": 1,
    "make_msinv": _make_n1_msinv,
    "make_msprime": _make_n1_msprime,
}


def _samples_by_pop(ts, n_pops: int):
    populations = ts.tables.nodes.population[ts.samples()]
    return [
        ts.samples()[populations == p].tolist() for p in range(n_pops)]


def _make_n2_msinv(seed: int):
    demo = Demography(
        pop_sizes=[10000.0, 10000.0],
        migration_matrix=[[0.0, 1e-4], [1e-4, 0.0]],
    )
    ts = HullSimulator(
        sample_config={(None, 0): 5, (None, 1): 5},
        demography=demo,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        inversions=[],
        seed=seed,
    ).simulate()
    return ts, _samples_by_pop(ts, n_pops=2)


def _make_n2_msprime(seed: int):
    # population sizes doubled; migration rate NOT rescaled
    # (per-lineage per-gen on both sides).
    demo = msprime.Demography()
    demo.add_population(name="A", initial_size=20000.0)
    demo.add_population(name="B", initial_size=20000.0)
    demo.set_migration_rate(source="A", dest="B", rate=1e-4)
    demo.set_migration_rate(source="B", dest="A", rate=1e-4)
    ts = msprime.sim_ancestry(
        samples={"A": 5, "B": 5},
        demography=demo,
        sequence_length=100_000,
        recombination_rate=1e-8,
        ploidy=1,
        record_full_arg=True,
        random_seed=seed + 1,
    )
    return ts, _samples_by_pop(ts, n_pops=2)


SCENARIOS["n2"] = {
    "compute_afs": False,
    "n_pops": 2,
    "make_msinv": _make_n2_msinv,
    "make_msprime": _make_n2_msprime,
}


def _make_n3_msinv(seed: int):
    demo = Demography(
        pop_sizes=[10000.0, 10000.0],
        migration_matrix=[[0.0, 1e-4], [1e-4, 0.0]],
    )
    demo.add_population_split(time=2000.0, derived=[1], ancestral=0)
    ts = HullSimulator(
        sample_config={(None, 0): 5, (None, 1): 5},
        demography=demo,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        inversions=[],
        seed=seed,
    ).simulate()
    return ts, _samples_by_pop(ts, n_pops=2)


def _make_n3_msprime(seed: int):
    demo = msprime.Demography()
    demo.add_population(name="A", initial_size=20000.0)
    demo.add_population(name="B", initial_size=20000.0)
    demo.set_migration_rate(source="A", dest="B", rate=1e-4)
    demo.set_migration_rate(source="B", dest="A", rate=1e-4)
    # mass_migration with proportion=1.0, NOT add_population_split (which
    # auto-derives growth/size resets that don't match msinv ej semantics).
    demo.add_mass_migration(
        time=2000.0, source="B", dest="A", proportion=1.0)
    # Zero all migration at T=2000 to match msinv's ej semantics: the ej
    # event zeros migration to/from the source population.  Without this,
    # msprime keeps M[B][A]=1e-4 active above T=2000, sending lineages to
    # the now-empty B and inflating effective Ne ~2x relative to msinv.
    # Order matters: msprime applies simultaneous events in insertion
    # order, so this must come AFTER the add_mass_migration above.
    # Global form (no source/dest) zeros all off-diagonal rates.
    demo.add_migration_rate_change(time=2000.0, rate=0.0)
    ts = msprime.sim_ancestry(
        samples={"A": 5, "B": 5},
        demography=demo,
        sequence_length=100_000,
        recombination_rate=1e-8,
        ploidy=1,
        record_full_arg=True,
        random_seed=seed + 1,
    )
    return ts, _samples_by_pop(ts, n_pops=2)


SCENARIOS["n3"] = {
    "compute_afs": True,
    "n_pops": 2,
    "make_msinv": _make_n3_msinv,
    "make_msprime": _make_n3_msprime,
}


def _make_n4_msinv(seed: int):
    demo = Demography(pop_sizes=[10000.0])
    demo.add_population_size_change(
        time=1000.0, population=0, new_size=1000.0)
    demo.add_population_size_change(
        time=2000.0, population=0, new_size=10000.0)
    ts = HullSimulator(
        samples=10,
        demography=demo,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        inversions=[],
        seed=seed,
    ).simulate()
    return ts, None


def _make_n4_msprime(seed: int):
    demo = msprime.Demography()
    demo.add_population(name="A", initial_size=20000.0)
    demo.add_population_parameters_change(
        time=1000.0, initial_size=2000.0, population="A")
    demo.add_population_parameters_change(
        time=2000.0, initial_size=20000.0, population="A")
    ts = msprime.sim_ancestry(
        samples=10,
        demography=demo,
        sequence_length=100_000,
        recombination_rate=1e-8,
        ploidy=1,
        record_full_arg=True,
        random_seed=seed + 1,
    )
    return ts, None


SCENARIOS["n4"] = {
    "compute_afs": True,
    "n_pops": 1,
    "make_msinv": _make_n4_msinv,
    "make_msprime": _make_n4_msprime,
}


def _make_n5_msinv(seed: int):
    demo = Demography(pop_sizes=[10000.0])
    demo.add_growth_rate_change(
        time=0.0, population=0, growth_rate=0.0005)
    ts = HullSimulator(
        samples=10,
        demography=demo,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        inversions=[],
        seed=seed,
    ).simulate()
    return ts, None


def _make_n5_msprime(seed: int):
    demo = msprime.Demography()
    demo.add_population(
        name="A", initial_size=20000.0, growth_rate=0.0005)
    ts = msprime.sim_ancestry(
        samples=10,
        demography=demo,
        sequence_length=100_000,
        recombination_rate=1e-8,
        ploidy=1,
        record_full_arg=True,
        random_seed=seed + 1,
    )
    return ts, None


SCENARIOS["n5"] = {
    "compute_afs": True,
    "n_pops": 1,
    "make_msinv": _make_n5_msinv,
    "make_msprime": _make_n5_msprime,
}


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
