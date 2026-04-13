# msinv

**Coalescent simulator with chromosomal inversions**

[![Tests](https://img.shields.io/badge/tests-46%2F46%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A sequential Markov coalescent (SMC) simulator for modeling chromosomal inversions with:

- Structured coalescent between karyotype classes (S/I)
- Position-dependent gene flux (Peischl et al. 2013)
- Multiple inversions per chromosome
- ms-compatible demography (size changes, growth, migration, merges)
- Per-population inversion frequency trajectories (for local adaptation)
- Selective sweep within inversion (for insecticide resistance studies)
- Tree sequence output (tskit compatible)
- msprime-compatible real-unit API

## Quick start

```python
from msinv import MsinvSimulator

sim = MsinvSimulator(
    samples=10,
    population_size=10_000,
    mutation_rate=1e-8,
    recombination_rate=1e-8,
    sequence_length=100_000,
    n_std=5, n_inv=5,           # 5 standard + 5 inverted haplotypes
    p_inv=0.5,                  # inversion frequency
    t_inv=200_000,              # inversion age in generations
    bp_left=0.3, bp_right=0.7,  # breakpoints (fraction of chromosome)
    seed=42,
)

# ms-format output
positions, haplotypes = sim.simulate_one()

# Tree sequence output
ts = sim.simulate_one_ts()
```

## Installation

```bash
# Via pip
pip install -e .

# Via pixi
pixi install
pixi run test
```

Requires Python ≥ 3.9 and numpy. Optional: msprime, tskit, stdpopsim, matplotlib for tests and examples.

## What msinv uniquely provides

Unlike msprime or standard coalescent simulators, msinv models **position-dependent recombination suppression** along the chromosome:

- Inside the inversion, recombination is weighted by karyotype frequency
- Gene flux (phi(x)) is position-dependent: peaks near breakpoints, minimal at center
- Tracks correlated flux tracts via Peischl et al.'s b2 model
- Three-way comparison (from our validation):

| Model                       | dxy inv | dxy col | ratio |
|-----------------------------|---------|---------|-------|
| msinv (with flux)           | 1.82    | 0.49    | 3.68  |
| msinv (no flux)             | 1.65    | 0.48    | 3.41  |
| msprime + migration matrix  | 0.51    | 0.50    | 1.02  |

Only msinv captures the spatial variation in divergence that empirical data show.

## Documentation

- [Installation guide](docs/installation.md)
- [Quick start tutorial](docs/quickstart.md)
- [Theory background](docs/theory.md) (SMC, structured coalescent, gene flux)
- [API reference](docs/api.md)
- [Example simulations](docs/examples.md) (Kir/Fol, RDL, 2La, MAPT)

## Validated applications

- **An. funestus Kiribina/Folonzo** — matches empirical Fst/dxy patterns from Small et al. (2023) PNAS
- **An. gambiae RDL introgression** — matches haplotype asymmetry from Grau-Bové et al. (2020) MBE
- **An. gambiae 2La** — Fst = 0.53 (empirical 0.57)
- **Human MAPT H1/H2** — dxy/site = 0.0031 (empirical 0.0026)
- **Peischl et al. (2013)** — T_SI ∝ 1/phi(x) replication

## Running the tests

```bash
make test
```

All 46 tests should pass:
- `test_standard_coalescent.py` — 8 tests
- `test_msinv.py` — 12 tests
- `test_ld.py` — 5 tests
- `test_treeseq.py` — 17 tests
- `test_stdpopsim.py` — 4 tests

## Citation

If you use msinv, please cite:

```bibtex
@software{msinv,
  author = {Small, Scott T.},
  title = {msinv: Coalescent simulator with chromosomal inversions},
  year = {2026},
  url = {https://github.com/stsmall/msinv}
}
```

## References

1. Guerrero, R. F., Rousset, F., & Kirkpatrick, M. (2012). *Coalescent patterns for chromosomal inversions in divergent populations.* Phil Trans R Soc B 367:430–438.
2. Peischl, S., Koch, E., Guerrero, R. F., & Kirkpatrick, M. (2013). *A sequential coalescent algorithm for chromosomal inversions.* Heredity 111:200–209.
3. Small, S. T. et al. (2023). *Standing genetic variation and chromosome differences drove rapid ecotype formation in a major malaria mosquito.* PNAS 120:e2219835120.
4. Grau-Bové, X. et al. (2020). *Evolution of the insecticide target Rdl in African Anopheles is driven by interspecific and interkaryotypic introgression.* MBE 37:2900–2917.

## License

MIT — see [LICENSE](LICENSE)
