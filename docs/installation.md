# Installation

## Requirements

- Python ≥ 3.9
- numpy ≥ 1.20

Optional for tests and examples:
- msprime ≥ 1.2
- tskit ≥ 0.5
- stdpopsim ≥ 0.2
- matplotlib ≥ 3.5

## Install via pip

```bash
git clone https://github.com/stsmall/msinv.git
cd msinv
pip install -e .
```

For development (tests, examples):
```bash
pip install -e ".[test]"
pip install -e ".[all]"  # includes matplotlib
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
from msinv import MsinvSimulator
sim = MsinvSimulator(
    nsam=6, theta=10, rho=10, nsites=1000,
    n_std=3, n_inv=3, p_inv=0.5, c=0.01, t_inv=10.0,
    seed=42)
pos, haps = sim.simulate_one()
print(f"Got {len(pos)} segregating sites")
```

## Running tests

```bash
make test
```

All 46 tests should pass in ~2 minutes.
