
<img width="2932" height="800" alt="Untitled" src="https://github.com/user-attachments/assets/afae242b-49ab-43f8-baaf-3eb4b292c277" />


[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)
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

### Coalescent approximation

By default `msinv` simulates under the sequentially Markov coalescent
(SMC; McVean & Cardin 2005): two lineages can coalesce only where they
share ancestral material. Single-site expectations (mean coalescence
times, π, dxy and the site frequency spectrum at each position) are
exact, but the correlation of genealogies along the sequence is weaker
than under the full coalescent, so linkage-dependent statistics vary
less among replicates. `HullSimulator(..., smc_prime=True)` switches
lineages without inversion tags to SMC′ (msprime's hull algorithm with
k = 0), which matches the linkage of the full coalescent much more
closely at a cost of roughly 1.8x run time. Lineages carrying inversion
tags use the per-class rule in both modes.

---

ARG-based coalescent simulator for chromosomes with inversions:

- **Per-position ancestral material tracking.** Each lineage carries
  the genomic intervals it's ancestral to; recombination splits
  intervals, coalescence merges them.
- **Cross-karyotype barriers** preserved exactly.
- **Multi-population demographies** match msprime ground truth.
- **Multiple inversions** (including nested / overlapping) supported.

Features:

- Structured coalescent between karyotype classes (S/I) with t_inv barrier
- Position-dependent gene flux (Peischl et al. 2013, *phi(x)* model)
- Multiple inversions per chromosome (supports nested/overlapping)
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

### Via pip

```bash
git clone https://github.com/stsmall/msinv.git
cd msinv
pip install -e .                    # core
pip install -e ".[test,plots]"      # with msprime + matplotlib + pytest
```

### Via [pixi](https://pixi.sh)

```bash
git clone https://github.com/stsmall/msinv.git
cd msinv
pixi install
pixi run test       # runs the hull test suite
pixi run figures    # regenerates the presentation PDFs in figures/
```

Requires Python ≥ 3.10, numpy, tskit. Optional: msprime (for mutation
dropping in the figure examples) and matplotlib.

To rebuild the Rust core in a development checkout:

```bash
cd rust
PYO3_PYTHON=$(which python) cargo build --release -p msinv-py
cp target/release/lib_msinv_core.so ../msinv/_msinv_core.cpython-312-x86_64-linux-gnu.so
```

(Use the extension filename that matches your Python version.)

## What msinv uniquely provides

`msinv` models **position-dependent karyotype-class structure** along
the chromosome. Inside an inversion, recombination is suppressed in
heterokaryotypes; gene conversion occasionally transfers tracts between
karyotypes (modulated by the *phi(x)* function from Peischl et al. 2013).
The hull algorithm implements this through per-position ancestral
material tracking under the SMC approximation described above.

**Crossover suppression in heterokaryotypes** (default since 0.4.0). A
crossover inside an inversion needs a homokaryotypic partner, so while
the barrier is active a lineage carrying arrangement *k* at a position
inside the inversion recombines there at `r · p_k(t)`, where `p_k(t)` is
the frequency of its own arrangement at that time. For nested or
overlapping inversions the weight is the product of the per-inversion
frequencies, which assumes the arrangements are independent (the same
convention as the coalescence rate). Collinear flanks, material without
an arrangement tag (samples with a `None` karyotype; their material is
treated as panmictic, as in coalescence) and times older than `t_inv`
keep the full rate `r`. The standard SLiM inversion recipe does the same:
it drops crossovers inside the inversion in a heterokaryotype. With a
constant `p`, an inverted chromosome then behaves exactly like one
population of size `p·N` with recombination rate `r·p`. For a
time-varying trajectory the treatment of recombination is exact:
breakpoints are proposed at rate `r` per bp and kept with probability
`p_k(t)` at the proposal time. Coalescence, as before, evaluates `p(t)`
at each event and holds it fixed until the next one. Up to 0.3.5 msinv used
`r` everywhere. That over-estimates within-arrangement ρ by `1/p_k`
(about 2.7x in the minority and 1.6x in the majority arrangement at
`p = 0.626`) and under-estimates within-arrangement LD. Pass
`HullSimulator(..., crossover_suppression=False)` to reproduce results
from earlier versions. Both engines (Rust and Python) implement it.

**Flank linkage** (since 0.4.1). A lineage that carries arrangement
material is one chromosome of that arrangement, flanks included, so an
I-carrying and an S-carrying lineage cannot coalesce while the barrier is
active, also where they overlap in the collinear flank. Flank material
leaves its arrangement only through a recombination in the flank, which
gives an untagged (panmictic) lineage. Such an escaped flank lineage
coalesces with lineages of either arrangement, or with other untagged
ones, at the panmictic rate `1/(2N)`: its chromosome carries arrangement `k`
with probability `p_k` and then meets a given `k` chromosome at
`1/(2N p_k)`, so the product is `1/(2N)`. When it coalesces with a
tagged lineage it rejoins that lineage's arrangement.
Flank dxy therefore decays from the body level at the breakpoint toward
the panmictic value with distance, on the scale `1 / (r · t)`.

In the no-inversion limit, `msinv` reproduces msprime's SMC (default)
or SMC′ (`smc_prime=True`) models, and single-site statistics match
msprime's full (Hudson) coalescent.
With one or more inversions it adds the karyotype barrier (`t_inv`)
and gene flux (`gene_conversion_rate`, `mean_tract_length`) that msprime
and msprime-style simulators cannot model directly.

## Performance

Since v0.3.0 the simulator core is implemented in Rust (PyO3 bridge).
The Python API is unchanged — `pip install msinv` ships pre-built
wheels, so no Rust toolchain is needed at install time.

Head-to-head against msprime at n=20 haploid samples, Ne=1000,
L=100 kb (`benchmarks/rho_scaling.py`):

| rho   | msprime  | msinv (no inv) | msinv (+1 inv) |
|------:|---------:|---------------:|---------------:|
|   500 |   24 ms  |    28 ms       |    31 ms       |
|  2000 |  145 ms  |   165 ms       |   220 ms       |
|  8000 | 1555 ms  |   765 ms       |  1577 ms       |
| 16000 | 5448 ms  |  1041 ms       |  4224 ms       |

Without an inversion, `msinv` matches msprime at low rho and pulls
ahead (~5×) at rho ≥ 8000. With one inversion, `msinv` reaches parity
at rho=8000 and stays within msprime's wall-clock at rho=16000. ABC
inference driving thousands of sims is feasible on a laptop: peak
resident memory at rho=16000 is ~1.2 GB.

Re-run locally:

```bash
python benchmarks/rho_scaling.py --rho 500,1000,2000,4000,8000,16000 --reps 5
```

## Validated applications

- **An. funestus Kiribina/Folonzo** — matches Fst patterns from Small et al. (2023) PNAS
- **An. gambiae RDL introgression** — matches haplotype asymmetry from Grau-Bové et al. (2020) MBE
- **An. gambiae 2La**, **Human MAPT H1/H2**, **Peischl et al. (2013) replication**

## Known limitations

- **Sweeps with more than one active inversion barrier.** While a
  barrier is active, karyotype-compatible pairs (I x I, S x S, or with an
  untagged lineage) coalesce by a full Hudson merge, which keeps
  within-arrangement LD right. Inside a sweep window, AA / aa
  coalescences pick any two lineages of the allele class, and the rule
  that counts each compatible pair once is derived only for a single
  active barrier. With two or more barriers active at the same time,
  those sweep-window events use the earlier partial merge, restricted
  to the cell's class. Marginal trees inside the inversion bodies are
  unaffected, but within-arrangement short-range LD inside the sweep
  window keeps the old barrier-era deficit, and a partial merge can
  leave a remnant that holds only margin material, so flank
  differentiation inside the sweep window is slightly understated.
  I x S pairs are never merged there. Coalescences outside sweep
  windows are not affected.

- **A sweep allele carried by both arrangements.** With gene flux
  between arrangements (`gamma_flux > 0`) the A allele can reach the
  arrangement it did not arise on. Those A lineages are not merged into
  the founder when a hard sweep finalizes: their path back to it runs
  through a conversion event the coalescent does not track.

- **Standing-variation sweep escape probability ignores the frequency
  path.** It uses the length of the selection phase, not the A-allele
  trajectory, so deterministic and stochastic trajectories of equal
  length give the same escape probability (issue #24).

## Changes in 0.4.2

- **Sweeps inside an active inversion** (issue #22). The hitchhiking
  escape probability of standing-variation sweeps uses the suppressed
  crossover rate: inside the inversion a linked lineage of arrangement k
  escapes at `r · p_k(t)`, integrated over the selection phase, and at
  `r` in the flanks. Recombination during the sweep window draws the
  non-carrier's allele from the A frequency of its own arrangement
  instead of the founder's, so A no longer spreads to the arrangement it
  did not arise on. The forced merges at the sweep origin (the de novo
  merge and the hard-sweep collapse) only join karyotype-compatible
  lineages; the founder carries `origin_kary`. Sweeps without an
  inversion are unchanged. The warning for sweeps combined with an
  inversion is removed.

## Changes in 0.4.1

- **Flank linkage while a barrier is active** (see "What msinv uniquely
  provides"). Earlier versions merged the shared flank of I x S pairs at
  1/(2N), so dxy fell about 3-fold at the breakpoints even with the
  flank fully linked to the body. Both engines.
- **Python high-rho path** (4 N r L > 100): an untagged (margin-only)
  lineage now coalesces with tagged lineages at 1/(2N), as in the Rust
  engine; before, only lineages of the same karyotype bucket coalesced.

## Running the tests

```bash
pytest tests/
```

The hull suite (`pytest tests/hull`) runs in about 4 minutes. Four
sweep tests in `tests/hull/test_stress_corners.py` currently fail
because they pass a `t_event` keyword that `Sweep()` no longer accepts.
`cargo test --release` in `rust/` runs the Rust unit tests.

## Citation

If you use msinv, please cite:

<img width="90" height="30" alt="image2" src="https://github.com/user-attachments/assets/e83e710c-7c94-4c00-943e-01451564a902" />

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
