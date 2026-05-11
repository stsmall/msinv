"""Generic engine runners for the msinv validation suite.

Each runner takes a unified set of parameters (demography, samples, L,
r, mu, seed) and engine-specific extras (inversions / sweeps for msinv;
sweep CLI args for discoal). Returns a mutated tskit.TreeSequence ready
for the validation/_lib/stats panel.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

import msprime
import tskit

from msinv import HullSimulator, InversionSpec, Sweep
from msinv import Demography as MsinvDemography


def msinv_run(
    *,
    demography: MsinvDemography,
    sample_config: dict,
    L: float,
    r: float,
    mu: float,
    seed: int,
    inversions: list[InversionSpec] | None = None,
    sweeps: list[Sweep] | None = None,
    iters_max: int = 1_000_000_000,
) -> tskit.TreeSequence:
    """Run msinv at the given config and overlay neutral mutations."""
    sim = HullSimulator(
        sample_config=sample_config,
        demography=demography,
        sequence_length=float(L),
        recombination_rate=float(r),
        inversions=inversions or [],
        sweeps=sweeps or [],
        seed=int(seed),
        iters_max=iters_max,
    )
    ts_raw = sim.simulate()
    ts = msprime.sim_mutations(
        ts_raw, rate=float(mu), random_seed=int(seed) + 1, keep=True,
    )
    return ts


def msprime_run(
    *,
    demography: msprime.Demography,
    samples_by_pop: dict[str, int],
    L: float,
    r: float,
    mu: float,
    seed: int,
) -> tskit.TreeSequence:
    """Run msprime at the given config and overlay neutral mutations.

    `samples_by_pop` keys must match the v12 population names ("K", "F").
    msprime is called with ploidy=1 (haploid) to match msinv's sample
    convention (per CLAUDE.md `population_size` / sample convention).
    """
    ts_raw = msprime.sim_ancestry(
        samples=samples_by_pop,
        demography=demography,
        sequence_length=float(L),
        recombination_rate=float(r),
        random_seed=int(seed),
        ploidy=1,
    )
    ts = msprime.sim_mutations(
        ts_raw, rate=float(mu), random_seed=int(seed) + 1, keep=True,
    )
    return ts


DISCOAL_BIN = "/home/adkern/discoal/build/discoal"


def discoal_run(
    *,
    n_samples: int,
    L: float,
    r: float,
    mu: float,
    seed: tuple[int, int],
    ne_diploid: float,
    demography_args: list[str],
    sweep_args: list[str] | None = None,
    tmp_dir: Path,
    n_reps: int = 1,
) -> tskit.TreeSequence:
    """Run discoal once and return the resulting tskit.TreeSequence.

    discoal CLI shape:
        discoal <sampleSize> <nReps> <nSites> -ts <out.trees> ...

    Time scaling: discoal emits in 4*N units; mutation/recomb rates and
    times in the demography args were pre-scaled by v12_discoal_events.
    Mutation rate is passed via discoal's -t (theta = 4*N*mu*L) and -r
    (rho = 4*N*r*L) conventions; we provide them per-locus (per-site
    rates are auto-multiplied by L by discoal). discoal's tree-seq
    output uses coalescent time units (2N generations); on load we
    rescale to generations to match msinv.
    """
    out_trees = Path(tmp_dir) / "discoal_out.trees"
    theta = 4.0 * ne_diploid * mu * L
    rho = 4.0 * ne_diploid * r * L
    cmd = [
        DISCOAL_BIN,
        str(int(n_samples)),
        str(int(n_reps)),
        str(int(L)),
        "-t", f"{theta:.10g}",
        "-r", f"{rho:.10g}",
        "-d", str(int(seed[0])), str(int(seed[1])),
        "-ts", str(out_trees),
    ]
    cmd.extend(demography_args)
    if sweep_args:
        cmd.extend(sweep_args)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"discoal failed (exit {proc.returncode})\n"
            f"  cmd:    {' '.join(cmd)}\n"
            f"  stderr: {proc.stderr[-500:]}"
        )
    if not out_trees.exists():
        raise RuntimeError(
            f"discoal exit 0 but {out_trees} missing\n"
            f"  stdout: {proc.stdout[-500:]}"
        )
    ts = tskit.load(str(out_trees))
    # Rescale time units if discoal emitted coalescent units.
    if str(ts.time_units).startswith("coalescent units"):
        tables = ts.dump_tables()
        tables.nodes.time *= 2.0 * ne_diploid
        tables.time_units = "generations"
        ts = tables.tree_sequence()
    return ts
