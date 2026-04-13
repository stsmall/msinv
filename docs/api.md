# API reference

Full API documentation is auto-generated via Sphinx in `docs/source/`. See the rendered version at [readthedocs](https://msinv.readthedocs.io) (when published).

## Core classes

### `MsinvSimulator`

Main simulator class.

**Constructors (two APIs):**

**msprime-style (real units):**
```python
MsinvSimulator(
    samples=10,                    # number of haploid samples (int)
    population_size=10000,         # Ne
    mutation_rate=1e-8,            # per bp per generation
    recombination_rate=1e-8,       # per bp per generation
    sequence_length=100000,        # in bp
    gene_conversion_rate=None,     # per bp per generation (optional)
    # Inversion parameters
    n_std=5, n_inv=5,
    p_inv=0.5,
    bp_left=0.3, bp_right=0.7,     # as fractions [0, 1]
    t_inv=200000,                  # in generations
    # Advanced
    p_inv_func=None,               # custom trajectory
    inversions=None,               # list of InversionSpec
    demography=None,               # Demography object
    sweep=None,                    # (x_sel, s, origin_class)
    seed=42,
)
```

**ms-style (coalescent units):**
```python
MsinvSimulator(
    nsam=10, theta=10, rho=50, nsites=1000,
    n_std=5, n_inv=5, p_inv=0.5, gamma=0.05,
    t_inv=10.0,
    ...
)
```

**Methods:**
- `simulate_one()` → `(positions, haplotypes)` in ms format
- `simulate_one_ts()` → tskit TreeSequence

### `InversionSpec`

Specification for a single inversion (used with multiple inversions).

```python
InversionSpec(
    bp_left=0.1, bp_right=0.3,
    p_inv=0.5,
    c=0.01,                # Peischl flux coefficient
    gamma=None,            # OR absolute flux rate (coal units)
    t_inv=10.0,
    flux_w=0.3,
    trajectory=None,       # optional p_inv_func
    label='inv1',
)
```

### `Demography`

ms-compatible demography with events.

```python
demo = Demography(n_pops=2, mig_rate=0.001)

# Events (time in coalescent units, 2N generations):
demo.add_event(('eN', t, x))              # set all pop sizes to x*N0
demo.add_event(('en', t, pop_i, x))       # set one pop size
demo.add_event(('eG', t, alpha))          # set all growth rates
demo.add_event(('eg', t, pop_i, alpha))   # set one pop growth
demo.add_event(('eM', t, M))              # set symmetric migration
demo.add_event(('em', t, i, j, M))        # set pairwise migration
demo.add_event(('ej', t, src, dst))       # merge populations
demo.add_event(('es', t, pop_i, p))       # admixture split

# Query
demo.get_size(pop, t)                # N(t) for a population
demo.coal_rate_factor(pop, t)        # 1/N(t)
demo.copy()                          # fresh copy for replicate

# After modifying pop_sizes directly (not via add_event):
demo.snapshot_initial_state()
```

## Frequency trajectories

All have `__call__(t, pop=0)` returning p_inv(t).

### `ConstantFrequency`
```python
ConstantFrequency(p_inv=0.5, t_inv=10.0)
```

### `DeterministicTrajectory`
Logistic sweep from 1/(2N) to p_final under selection s.
```python
DeterministicTrajectory(p_final=0.5, N=10000, s=0.01)
```

### `StochasticTrajectory`
WF diffusion backward with reflecting boundary (models recurrent origins).
```python
StochasticTrajectory(p_final=0.5, N=10000, s=0.0, rng=rng)
```

### `CoupledTrajectory`
Per-population 2D diffusion with local selection and migration.
```python
CoupledTrajectory(
    p_final=[0.7, 0.1],    # per-pop present-day freq
    N=[10000, 10000],      # per-pop Ne
    s=[0.01, 0.0],         # per-pop selection
    m=0.001,               # migration rate
    rng=rng,
)
```

## Output formats

### ms format
```python
positions, haplotypes = sim.simulate_one()
# positions: list of floats in [0, 1]
# haplotypes: numpy array (nsam × n_sites) of 0/1
```

### Tree sequence
```python
ts = sim.simulate_one_ts()
# Returns tskit.TreeSequence
# Can be saved: ts.dump("out.trees")
# Can be analyzed: ts.diversity(), ts.divergence(), etc.
```

## Utility functions

- `phi(x, w)` — compute phi(x) at position x within inversion
- `GeneFluxModel(w)` — gene flux model with window w
- `get_all_nodes(root)`, `find_root(node)` — tree helpers
- `build_initial_tree(...)` — n=2 utility for exact validation
- `simulate_one_n2(...)` — n=2 exact simulation (for tests)
