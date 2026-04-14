# API reference

Full API documentation is auto-generated via Sphinx in `docs/source/`.
The brief reference below covers the classes you'll use day-to-day.

## `HullSimulator`

ARG-based per-position ancestral material tracking simulator. Returns
a [tskit](https://tskit.dev) `TreeSequence`.

### Constructor

```python
HullSimulator(
    # ----- samples -----
    n_std=5, n_inv=5,                # back-compat: linked-karyotype, single pop
    sample_config=None,              # OR {(class, pop): n} dict for full control
    # ----- demography -----
    population_size=10_000,          # used when no `demography` passed
    demography=None,                 # Demography object for multi-pop
    # ----- chromosome -----
    sequence_length=100_000,
    recombination_rate=0.0,
    # ----- inversion(s) -----
    bp_left=None, bp_right=None,     # back-compat: single inversion
    p_inv=None, t_inv=None,
    gene_conversion_rate=0.0,
    flux_window=0.05,
    inversions=None,                 # OR list of InversionSpec
    # ----- selection -----
    sweeps=None,                     # list of Sweep
    # ----- misc -----
    seed=None,
)
```

### Methods

- `simulate()` → `tskit.TreeSequence`

## `InversionSpec`

```python
InversionSpec(
    bp_left=30_000, bp_right=70_000,
    p_inv=0.5,
    t_inv=200_000,                   # generations
    gene_conversion_rate=0.0,        # γ per bp per gen (Peischl 2013 model)
    flux_window=0.05,                # tract width as fraction of inv length
)
```

Inversions may overlap or nest. Each contributes its own `t_inv`
barrier independently. The simulator tags class labels with the
inversion's id (e.g. `'S0'`, `'I1'`).

## `Sweep`

Forced-coalescence selective sweep at a single position and time.

```python
Sweep(
    x_sel=50_000,                    # selected site (bp)
    t_event=300,                     # sweep MRCA at 300 gen ago
    target_class='S',                # 'S'/'I' (single inv) or 'S0'/'I0' etc.
    sweep_window=500.0,              # ±bp around x_sel that get force-merged
)
```

Stack sweeps (e.g., one S, one I) to model an introgressed allele
that swept through both arrangements (RDL pattern).

## `Demography`

ms-style demography with size changes, growth, migration, and
population merges.

```python
demo = Demography(pop_sizes=[10_000, 10_000])

# Events (time in generations):
demo.add_event(('en', t, pop_i, x))      # set one pop's size to x*N0
demo.add_event(('eg', t, pop_i, alpha))  # exponential growth
demo.add_event(('em', t, i, j, M))       # migration rate from j into i
demo.add_event(('ej', t, src, dst))      # merge src → dst

# Query
demo.get_size(pop, t)
demo.copy()
```

## Output

`sim.simulate()` returns a `tskit.TreeSequence`. From there you can:

- Drop mutations: `mts = msprime.sim_mutations(ts, rate=mu, random_seed=s)`
- Compute statistics: `mts.diversity()`, `mts.divergence([[0,1,2],[3,4,5]])`,
  `mts.Fst(...)`, `mts.genotype_matrix()`, etc.
- Save / load: `ts.dump('out.trees')`, `tskit.load('out.trees')`
- Inspect trees: `ts.at(position)`, `for tree in ts.trees(): ...`
