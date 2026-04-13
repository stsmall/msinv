# msinv

**Coalescent simulator with chromosomal inversions**

[![Tests](https://img.shields.io/badge/tests-114%2F114%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Two coalescent engines for modeling chromosomal inversions:

- **`HullSimulator`** *(recommended)* — ARG-based per-position ancestral
  material tracking. Architecturally correct: cross-karyotype barriers
  preserved, multi-population demographies match msprime ground truth,
  multiple inversions including nested/overlapping supported.
- **`MsinvSimulator`** *(legacy)* — single-tree SMC implementation. Faster
  for inversion-only single-pop scenarios but has a known multi-pop bug
  (cross-pop dxy is ~half of expected; see [`examples/bakeoff.py`](examples/bakeoff.py)).
  Kept for backwards compatibility.

Both engines support:

- Structured coalescent between karyotype classes (S/I) with t_inv barrier
- Position-dependent gene flux (Peischl et al. 2013, *phi(x)* model)
- Multiple inversions per chromosome (hull also supports nested/overlapping)
- ms-style demography (size changes, growth, migration, merges)
- Per-population inversion frequency trajectories
- Selective sweep events (force-coalescence)
- tskit `TreeSequence` output

## Quick start

### Hull simulator (recommended)

```python
from msinv import HullSimulator, InversionSpec

sim = HullSimulator(
    n_std=5, n_inv=5,
    population_size=10_000,
    sequence_length=100_000,
    p_inv=0.5,
    t_inv=200_000,                  # generations
    bp_left=30_000, bp_right=70_000,
    gene_conversion_rate=1e-9,
    seed=42,
)
ts = sim.simulate()                 # returns a tskit TreeSequence

# Multiple inversions
sim = HullSimulator(
    n_std=5, n_inv=5,
    population_size=10_000,
    sequence_length=100_000,
    inversions=[
        InversionSpec(bp_left=10_000, bp_right=40_000,
                      p_inv=0.5, t_inv=200_000),
        InversionSpec(bp_left=60_000, bp_right=90_000,
                      p_inv=0.3, t_inv=300_000,
                      gene_conversion_rate=1e-9),
    ],
    seed=42,
)
ts = sim.simulate()
```

### Legacy SMC simulator

```python
from msinv import MsinvSimulator
sim = MsinvSimulator(
    samples=10, population_size=10_000,
    mutation_rate=1e-8, recombination_rate=1e-8,
    sequence_length=100_000,
    n_std=5, n_inv=5, p_inv=0.5,
    t_inv=200_000,
    bp_left=30_000, bp_right=70_000,
    seed=42,
)
positions, haplotypes = sim.simulate_one()
```

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.9, numpy, tskit. Optional: msprime for the bake-off
comparison, matplotlib for figure-generating examples.

## What msinv uniquely provides

Both engines model **position-dependent karyotype-class structure** along
the chromosome. Inside an inversion, recombination is suppressed in
heterokaryotypes; gene conversion occasionally transfers tracts between
karyotypes (modulated by the *phi(x)* function from Peischl et al. 2013).
The hull engine implements this exactly via per-position ancestral
material tracking; the SMC engine approximates it via a single-tree SMC
walk.

## Cross-validation

`examples/bakeoff.py` runs three-way comparisons (msprime ↔ SMC ↔ hull)
on six scenarios. Headline results:

| Scenario              | msprime | SMC      | Hull     | Notes                          |
|-----------------------|---------|----------|----------|--------------------------------|
| Panmictic baseline    | 0.000392 | 0.000377 | 0.000440 | All agree                      |
| Single inv, γ=0       | —       | 0.00197  | 0.00195  | SMC ≈ Hull                     |
| Two-pop split         | 0.00119 | 0.00051  | 0.00114  | **SMC bug**; Hull matches msprime |

See [`docs/hull_algorithm_design.md`](docs/hull_algorithm_design.md)
and [`docs/known_issues.md`](docs/known_issues.md) for details.

## Documentation

- [Hull algorithm design](docs/hull_algorithm_design.md) — phased build of the ARG simulator
- [Known issues](docs/known_issues.md) — the SMC bug-history and their fixes
- [Installation guide](docs/installation.md)
- [Quick start tutorial](docs/quickstart.md)
- [Theory background](docs/theory.md) — SMC, structured coalescent, gene flux
- [Example simulations](docs/examples.md) — Kir/Fol, RDL, 2La, MAPT

## Validated applications

- **An. funestus Kiribina/Folonzo** — matches Fst patterns from Small et al. (2023) PNAS
- **An. gambiae RDL introgression** — matches haplotype asymmetry from Grau-Bové et al. (2020) MBE
- **An. gambiae 2La**, **Human MAPT H1/H2**, **Peischl et al. (2013) replication**

## Running the tests

```bash
pytest tests/
```

All 114 tests should pass:

- `tests/test_standard_coalescent.py`, `test_msinv.py`, `test_ld.py`,
  `test_treeseq.py`, `test_stdpopsim.py` — 25 SMC tests
- `tests/hull/test_phase{1..6,5{a,b,c1,c2}}_*.py` — 89 hull tests

## Citation

If you use msinv, please cite:

```bibtex
@software{msinv,
  author = {Small, Scott T.},
  title  = {msinv: Coalescent simulator with chromosomal inversions},
  year   = {2026},
  url    = {https://github.com/stsmall/msinv}
}
```

## References

1. Kelleher, J., Etheridge, A. M., & McVean, G. (2016). *Efficient coalescent simulation and genealogical analysis for large sample sizes.* PLOS Comp Bio 12:e1004842. *(msprime hull algorithm)*
2. Guerrero, R. F., Rousset, F., & Kirkpatrick, M. (2012). *Coalescent patterns for chromosomal inversions in divergent populations.* Phil Trans R Soc B 367:430–438.
3. Peischl, S., Koch, E., Guerrero, R. F., & Kirkpatrick, M. (2013). *A sequential coalescent algorithm for chromosomal inversions.* Heredity 111:200–209.
4. Small, S. T. et al. (2023). *Standing genetic variation and chromosome differences drove rapid ecotype formation in a major malaria mosquito.* PNAS 120:e2219835120.
5. Grau-Bové, X. et al. (2020). *Evolution of the insecticide target Rdl in African Anopheles is driven by interspecific and interkaryotypic introgression.* MBE 37:2900–2917.

## License

MIT — see [LICENSE](LICENSE)
