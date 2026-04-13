# msinv: Coalescent simulator with chromosomal inversions

## Overview

`msinv` is a Python coalescent simulator that extends Hudson's `ms` framework
with chromosomal inversion polymorphism. It uses the Sequential Markovian
Coalescent (SMC) for recombination along the chromosome and a structured
coalescent for the two karyotype classes (Standard and Inverted).

## References

- Hudson RR (2002) Generating samples under a Wright-Fisher neutral model.
  *Bioinformatics* 18:337-338.
- Guerrero RF, Rousset F, Kirkpatrick M (2012) Coalescent patterns for
  chromosomal inversions in divergent populations.
  *Phil Trans R Soc B* 367:430-438.
- Peischl S, Koch E, Guerrero RF, Kirkpatrick M (2013) A sequential coalescent
  algorithm for chromosomal inversions. *Heredity* 111:200-209.

## Dependencies

- Python >= 3.8
- NumPy

## Usage

```bash
python msinv.py <nsam> <nreps> -t <theta> -r <rho> <nsites> [options]
```

### Parameters

| Flag | Description |
|------|-------------|
| `nsam` | Total sample size |
| `nreps` | Number of replicates |
| `-t theta` | Population-scaled mutation rate: 4Nμ × L |
| `-r rho nsites` | Population-scaled recombination rate 4Nr and number of discrete sites |
| `-inv p_inv c` | Inversion frequency `p_inv` and gene flux coefficient `c` |
| `-I n_std n_inv` | Number of standard and inverted chromosomes (must sum to nsam) |
| `-flux_window w` | Gene flux interval width in [0,1] inversion coordinates (default: 0.3) |
| `-seed s` | Random seed |

### Examples

```bash
# Standard coalescent (no inversion) - equivalent to ms
python msinv.py 10 100 -t 10 -r 50 1000

# With inversion: p_inv=0.5, gene flux coefficient c=0.01
# Sample 5 standard + 5 inverted chromosomes
python msinv.py 10 100 -t 10 -r 50 1000 -inv 0.5 0.01 -I 5 5

# Higher gene flux (less divergence between arrangements)
python msinv.py 10 100 -t 10 -r 50 1000 -inv 0.5 0.1 -I 5 5

# Asymmetric frequency: inversion at 20%
python msinv.py 10 100 -t 10 -r 50 1000 -inv 0.2 0.01 -I 8 2
```

### Output format

ms-compatible: header, then per replicate:
```
//
segsites: K
positions: p1 p2 ... pK
010010...
110100...
```

## Model details

### Population structure

The population contains two karyotype classes: Standard (S) and Inverted (I)
at frequencies `1 - p_inv` and `p_inv`. Going backward in time:

- **Coalescence** occurs only between lineages of the same class, at rate
  `C(k,2) / p` where `k` is the number of lineages and `p` is the class
  frequency (Peischl et al. eq. 2).

- **Gene flux** moves lineages between classes at rate
  `c × (ρ/2) × p_other × φ(x)`, where `φ(x)` is the probability that
  site `x` is affected by a random gene flux event.

### Gene flux spatial model

Following Peischl et al. (2013), gene flux is modeled as double crossover
with fixed window width `w`. The probability that a site at position `x`
(in [0,1] inversion coordinates, where 0 and 1 are breakpoints) is affected:

```
φ(x) = min(x, 1-x, w) / (1-w)
```

This produces maximum flux at the center and zero flux at breakpoints.

### SMC algorithm

Homokaryotypic recombination is simulated sequentially along the chromosome
(Peischl et al. 2013, extending McVean & Cardin 2005). At each step:

1. Draw distance to next recombination event
2. Pick a branch to cut (weighted by class-specific branch lengths)
3. Prune the subtree above the cut
4. Reattach using the structured coalescent (class-aware)

### Breakpoint handling

Coalescence times between arrangements diverge to infinity at inversion
breakpoints (Guerrero et al. 2012, figure 1). Following Peischl et al.:
"pick two points x0 and x1 arbitrarily close to 0 or 1". The simulator
uses x0=0.02 and x1=0.98, with constant-tree approximation for the
edge regions [0, x0) and (x1, 1].

## What's implemented

- [x] ms-compatible output format
- [x] Structured coalescent with two karyotype classes (S and I)
- [x] SMC with inversion-aware recombination and gene flux
- [x] Position-dependent gene flux rate (φ(x) model)
- [x] Arbitrary sample sizes and compositions
- [x] Infinite sites mutation model
- [x] Standard coalescent mode (no inversion)

## Current approximations

1. **No pending flux tracking**: Each position's gene flux is treated
   independently. This gives correct marginal distributions but slightly
   approximate LD patterns. The full Peischl et al. algorithm tracks
   gene flux interval boundaries (b2) across positions — this is the main
   extension needed for fully correct LD.

2. **Constant inversion frequency**: The inversion frequency `p_inv` is
   assumed constant (equilibrium). The algorithm can be extended to
   time-varying frequencies for young inversions or sweeps (Guerrero et al.
   2012, section 2c-d).

3. **Single population**: No spatial structure or migration. Guerrero et al.
   model two populations with migration; extending the structured coalescent
   to include both population AND karyotype structure is straightforward.

4. **Neutral sites only**: Selected sites within the inversion (Guerrero et al.
   "locally adapted alleles" model) are not implemented.

## Performance

Pure Python. Typical timing per replicate:

| Parameters | Time |
|-----------|------|
| n=6, ρ=20, with inversion | ~0.1-0.5s |
| n=10, ρ=50, with inversion | ~0.3s |
| n=20, ρ=100, with inversion | ~0.5s |
| n=20, ρ=100, no inversion | ~0.1s |

For production use (ABC, large parameter sweeps), porting the inner loop
(SMC step + reattach) to C/Cython would give ~50-100× speedup.

## Planned extensions

1. **Pending gene flux tracking** (Peischl et al. algorithm steps 2-4):
   correct LD within inverted regions
2. **Two-population model** (Guerrero et al.): migration-selection balance
3. **Time-varying inversion frequency**: young inversions, selective sweeps
4. **C core**: performance-critical inner loop in C with Python wrapper
5. **Tree sequence output**: compatible with tskit for downstream analysis
