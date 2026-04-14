# Installation

## Requirements

- Python ≥ 3.9
- numpy ≥ 1.20
- tskit ≥ 0.5

Optional for tests and figure-generating examples:
- msprime ≥ 1.2  (for mutation dropping + ground-truth comparison)
- matplotlib ≥ 3.5
- pytest, pytest-timeout

## Install via pip

```bash
git clone https://github.com/stsmall/msinv.git
cd msinv
pip install -e .
```

For development (tests + plots):
```bash
pip install -e ".[test,plots]"
```

## Install via pixi

```bash
git clone https://github.com/stsmall/msinv.git
cd msinv
pixi install
pixi run test    # runs the full test suite
```

## Verify installation

```python
from msinv import HullSimulator, InversionSpec

sim = HullSimulator(
    n_std=5, n_inv=5,
    population_size=10_000,
    sequence_length=100_000,
    inversions=[InversionSpec(bp_left=30_000, bp_right=70_000,
                               p_inv=0.5, t_inv=200_000)],
    seed=42,
)
ts = sim.simulate()
print(f"Simulated TreeSequence with {ts.num_trees} trees")
```

## Running tests

```bash
pytest tests/hull/
```

98 tests should pass in a few seconds.
