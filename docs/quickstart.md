# Quick start

## Minimal example

Standard coalescent simulation with an inversion:

```python
from msinv import MsinvSimulator

sim = MsinvSimulator(
    samples=10,                  # 10 haplotypes
    population_size=10_000,      # Ne
    mutation_rate=1e-8,          # per bp per generation
    recombination_rate=1e-8,     # per bp per generation
    sequence_length=100_000,     # chromosome length in bp
    n_std=5, n_inv=5,            # karyotype composition
    p_inv=0.5,                   # inversion frequency
    t_inv=200_000,               # inversion age (generations)
    bp_left=0.3, bp_right=0.7,   # breakpoints (fraction of chromosome)
    seed=42,
)

# ms-format output: positions in [0,1], haplotype matrix (nsam × nsites)
positions, haplotypes = sim.simulate_one()

print(f"{len(positions)} segregating sites")
print(f"Haplotype matrix: {haplotypes.shape}")
```

## Tree sequence output

For downstream analysis with tskit:

```python
ts = sim.simulate_one_ts()
print(f"Trees: {ts.num_trees}")
print(f"pi: {ts.diversity(mode='site'):.4f}")

# Save to disk
ts.dump("my_simulation.trees")
```

## Per-population selection (local adaptation)

The inversion is favored in one population but neutral in another:

```python
from msinv import MsinvSimulator, CoupledTrajectory, Demography
import numpy as np

rng = np.random.default_rng(42)

# Trajectory: 2 pops, inversion favored in pop 0, neutral in pop 1
traj = CoupledTrajectory(
    p_final=[0.7, 0.1],    # present-day frequencies
    N=[10_000, 10_000],    # population sizes
    s=[0.01, 0.0],         # selection coefficients
    m=0.001,               # migration rate
    rng=rng,
)

# Demography: 2 pops that merge 1000 generations ago
demo = Demography(n_pops=2, mig_rate=0.001 * 4 * 10000)  # 4Nm
demo.add_event(('ej', 1000 / (2 * 10000), 0, 1))  # merge pop 0 → pop 1

sim = MsinvSimulator(
    samples=20, population_size=10_000,
    mutation_rate=1e-8, recombination_rate=1e-8,
    sequence_length=100_000,
    n_std=10, n_inv=10, p_inv=0.5,
    p_inv_func=traj,
    demography=demo,
    n_pops=2,
    sample_config={('S', 0): 5, ('I', 0): 5, ('S', 1): 5, ('I', 1): 5},
    seed=42,
)
pos, haps = sim.simulate_one()
```

## Multiple inversions

```python
from msinv import MsinvSimulator, InversionSpec

inv1 = InversionSpec(bp_left=0.1, bp_right=0.3,
                     p_inv=0.5, c=0.01, t_inv=10.0)
inv2 = InversionSpec(bp_left=0.6, bp_right=0.9,
                     p_inv=0.3, c=0.02, t_inv=20.0)

sim = MsinvSimulator(
    samples=10, population_size=10_000,
    mutation_rate=1e-8, recombination_rate=1e-8,
    sequence_length=100_000,
    n_std=5, n_inv=5,
    inversions=[inv1, inv2],
    seed=42,
)
pos, haps = sim.simulate_one()
```

## Selective sweep inside inversion (RDL scenario)

```python
sim = MsinvSimulator(
    samples=10, population_size=100_000,
    mutation_rate=1e-8, recombination_rate=1e-8,
    sequence_length=100_000,
    n_std=5, n_inv=5, p_inv=0.5,
    t_inv=10.0,
    sweep=(0.5, 0.1, 'S'),  # selected site at x=0.5, s=0.1, on S background
    seed=42,
)
pos, haps = sim.simulate_one()
```

## Next steps

- Read the [theory background](theory.md)
- See [examples](examples.md) for validated biological applications
- Browse the [API reference](api.md)
