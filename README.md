# msinv

**Coalescent simulator with chromosomal inversions**

[![Tests](https://img.shields.io/badge/tests-98%2F98%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Explain like I'm 5

Imagine every chromosome in a population is like a long string of
beads. Most chromosomes have the beads in the **standard order**
(call it **S**). But sometimes, a piece of the chromosome flips upside
down — like grabbing a chunk of the bead-string, flipping it, and
sewing it back. We call that flipped chromosome **I** (for inverted).

Now here's the trick: when an S chromosome and an I chromosome live
in the same person, that person can't easily mix the two during egg
or sperm production at the flipped section. **The flipped piece is
"locked"** — S beads stay with S beads going forward, I beads stay
with I beads. (Outside the flipped section, life goes on normally.)

If you wait long enough, the S and I chunks become quite different
from each other — like two villages that stopped talking centuries
ago. That's an **inversion polymorphism**.

`msinv` is a program that says: **"Pretend you are running time
backwards. Trace each chromosome's ancestors all the way back."** It
honours the rules above:

- Inside the flipped piece, an S chromosome can only have an S parent
  (and same for I).
- Outside the flipped piece, anyone can be anyone's parent.
- Very rarely, a tiny piece of DNA *does* leak between S and I
  through "gene conversion". `msinv` knows about that too.
- Eventually you go far enough back in time that the inversion didn't
  exist yet (the **t_inv age**). At that point, S and I were the same
  thing — they merge.

The output is a **tree of every chromosome's family history at every
position along the genome**. From that tree you can compute things
like "how different are S samples from I samples" — exactly what
biologists measure when they study real inversions in mosquitoes,
flies, or sticklebacks.

`msinv` does this with a **hull algorithm** (Kelleher et al. 2016):
each chromosome's family history is tracked *position by position*.
Cross-karyotype barriers, multi-population demographies, and nested
or overlapping inversions all fall out of the model.

That's it. The rest of this README is the technical version.

---

ARG-based coalescent simulator for chromosomes with inversions:

- **Per-position ancestral material tracking.** Each lineage carries
  the genomic intervals it's ancestral to; recombination splits
  intervals, coalescence merges them.
- **Cross-karyotype barriers** preserved exactly (no SMC-style
  prune-and-reattach approximations).
- **Multi-population demographies** match msprime ground truth.
- **Multiple inversions** (including nested / overlapping) supported.

Features:

- Structured coalescent between karyotype classes (S/I) with t_inv barrier
- Position-dependent gene flux (Peischl et al. 2013, *phi(x)* model)
- Multiple inversions per chromosome (hull also supports nested/overlapping)
- ms-style demography (size changes, growth, migration, merges)
- Per-population inversion frequency trajectories
- Selective sweep events (force-coalescence)
- tskit `TreeSequence` output

## Quick start

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

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.9, numpy, tskit. Optional: msprime for the bake-off
comparison, matplotlib for figure-generating examples.

## What msinv uniquely provides

`msinv` models **position-dependent karyotype-class structure** along
the chromosome. Inside an inversion, recombination is suppressed in
heterokaryotypes; gene conversion occasionally transfers tracts between
karyotypes (modulated by the *phi(x)* function from Peischl et al. 2013).
The hull algorithm implements this exactly via per-position ancestral
material tracking — no single-tree SMC approximations.

In the no-inversion limit, `msinv` reproduces msprime ground truth.
With one or more inversions it adds the karyotype barrier (`t_inv`)
and gene flux (`gene_conversion_rate`, `flux_window`) that msprime
and msprime-style simulators cannot model directly.

## Documentation

- [Hull algorithm design](docs/hull_algorithm_design.md) — implementation notes
- [Known issues](docs/known_issues.md) — current limitations
- [Installation guide](docs/installation.md)
- [Quick start tutorial](docs/quickstart.md)
- [Theory background](docs/theory.md) — structured coalescent, gene flux
- [Example simulations](docs/examples.md) — Kir/Fol, RDL, presentation figures
- [API reference](docs/api.md) — short reference for the public API

## Validated applications

- **An. funestus Kiribina/Folonzo** — matches Fst patterns from Small et al. (2023) PNAS
- **An. gambiae RDL introgression** — matches haplotype asymmetry from Grau-Bové et al. (2020) MBE
- **An. gambiae 2La**, **Human MAPT H1/H2**, **Peischl et al. (2013) replication**

## Running the tests

```bash
pytest tests/
```

All 98 hull tests should pass in a few seconds.

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
