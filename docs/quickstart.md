# Quick start

This guide walks through the recommended ARG-based **hull simulator**.
For the legacy SMC simulator (kept for backwards compatibility), see
the bottom of this page.

## Install

```bash
pip install msinv                    # core
pip install "msinv[test,plots]"      # with msprime + matplotlib
```

Or with [pixi](https://pixi.sh):

```bash
pixi install
pixi run test
```

## Minimal example: one inversion, one population

```python
from msinv import HullSimulator

sim = HullSimulator(
    n_std=5, n_inv=5,                # 5 standard + 5 inverted haplotypes
    population_size=10_000,           # diploid Ne
    sequence_length=100_000,          # chromosome length in bp
    p_inv=0.5,                        # inversion frequency in the pop
    t_inv=200_000,                    # inversion age in generations
    bp_left=30_000, bp_right=70_000,  # breakpoints in bp
    seed=42,
)
ts = sim.simulate()                   # returns a tskit TreeSequence
print(ts.num_trees, "trees")
```

The output is a [tskit](https://tskit.dev) ``TreeSequence`` — directly
usable for diversity stats, mutation dropping, and most downstream
tools.

## Add gene flux

Inside an inversion, gene conversion can transfer alleles between the
two karyotypes. Set ``gene_conversion_rate`` (per bp per generation):

```python
sim = HullSimulator(
    n_std=5, n_inv=5, population_size=10_000, sequence_length=100_000,
    p_inv=0.5, t_inv=200_000,
    bp_left=30_000, bp_right=70_000,
    gene_conversion_rate=1e-9,        # γ — Peischl 2013 phi(x) model
    seed=42,
)
ts = sim.simulate()
```

## Multiple inversions

Pass a list of ``HullInversionSpec`` objects:

```python
from msinv import HullSimulator, HullInversionSpec

sim = HullSimulator(
    n_std=5, n_inv=5,
    population_size=10_000,
    sequence_length=100_000,
    inversions=[
        HullInversionSpec(bp_left=10_000, bp_right=40_000,
                          p_inv=0.5, t_inv=200_000),
        HullInversionSpec(bp_left=60_000, bp_right=90_000,
                          p_inv=0.3, t_inv=300_000,
                          gene_conversion_rate=1e-9),
    ],
    seed=42,
)
ts = sim.simulate()
```

Inversions may overlap or be nested — each contributes its own class
barrier independently.

## Independent karyotype per inversion

By default, a sample's ``'S'`` or ``'I'`` applies to every inversion
("linked" karyotype). To assign karyotypes independently per inversion,
pass a per-inv tuple:

```python
sim = HullSimulator(
    sample_config={
        # 5 samples that are S at inv 0 and S at inv 1 ('SS' linked)
        ('SS', 0): 5,
        # 3 samples that are S at inv 0 but I at inv 1 (recombinant)
        (('S', 'I'), 0): 3,
    },
    population_size=10_000, sequence_length=100_000,
    inversions=[...],   # as above
    seed=42,
)
```

## Two populations with a split

```python
from msinv import HullSimulator, HullInversionSpec, HullDemography

demo = HullDemography(pop_sizes=[10_000, 10_000])
demo.add_event(('ej', 14_000, 1, 0))   # at t=14k gen, pop 1 → pop 0

sim = HullSimulator(
    sample_config={('S', 0): 5, ('S', 1): 3, ('I', 1): 3},
    demography=demo,
    sequence_length=100_000,
    inversions=[HullInversionSpec(bp_left=30_000, bp_right=70_000,
                                    p_inv=0.3, t_inv=385_000)],
    seed=42,
)
ts = sim.simulate()
```

This is the Kir/Fol scenario from Small et al. 2023 — see
[``examples/empirical_kir_fol_hull.py``](../examples/empirical_kir_fol_hull.py).

## Selective sweep

A forced-coalescence sweep at a specific position and time:

```python
from msinv import HullSimulator, Sweep

sweep = Sweep(
    x_sel=50_000,            # genomic position of the selected site
    t_event=300,             # sweep MRCA at 300 gen ago
    target_class='S',        # carriers are on the S background
    sweep_window=500.0,      # ±500 bp around x_sel
)
sim = HullSimulator(
    n_std=5, n_inv=5, population_size=100_000, sequence_length=100_000,
    p_inv=0.5, t_inv=80_000,
    bp_left=30_000, bp_right=70_000,
    sweeps=[sweep], seed=42,
)
ts = sim.simulate()
```

## Drop mutations + compute summary stats

```python
import msprime
mts = msprime.sim_mutations(ts, rate=1e-8, random_seed=1,
                              discrete_genome=False)
G = mts.genotype_matrix()      # (n_sites, n_samples) — 0/1 matrix
print(mts.diversity())         # tskit's pi
print(mts.divergence([[0,1,2,3,4], [5,6,7,8,9]]))
```

## Legacy SMC simulator (back-compat)

The original SMC engine remains available. New code should prefer
``HullSimulator``; the SMC simulator has a known multi-pop bug
(see [`docs/known_issues.md`](known_issues.md)).

```python
from msinv import MsinvSimulator   # legacy

sim = MsinvSimulator(
    samples=10, population_size=10_000,
    mutation_rate=1e-8, recombination_rate=1e-8,
    sequence_length=100_000,
    n_std=5, n_inv=5, p_inv=0.5, t_inv=200_000,
    bp_left=30_000, bp_right=70_000, seed=42,
)
positions, haplotypes = sim.simulate_one()
```

## Next steps

- [Theory background](theory.md) — what the simulator is doing under the hood
- [Hull algorithm design](hull_algorithm_design.md) — implementation phases
- [Known issues](known_issues.md) — the SMC bug history and current limitations
- [Examples](examples.md) — Kir/Fol, RDL, 2La, MAPT validated applications
