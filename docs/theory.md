# Theory background

msinv implements the sequential Markov coalescent (SMC) with chromosomal inversions, following Guerrero et al. (2012) and Peischl et al. (2013).

## Standard SMC

Going backward in time along the chromosome, the coalescent tree changes as recombination events are encountered. The SMC approximates this by doing prune-and-reattach operations:

1. Pick a branch proportional to its length
2. Prune the branch at some time `t_cut`
3. Reattach it via coalescent-based walking backward in time

msinv uses **coalescent-based reattachment** (not uniform-on-branch), which matches msprime within 2% for panmictic simulations.

## Structured coalescent

Inside an inversion, chromosomes fall into two karyotype classes:
- **S** (standard orientation)
- **I** (inverted orientation)

Recombination is suppressed between heterokaryotypes. msinv models this as:

- **S-S coalescence rate**: `k_S(k_S - 1) / (2 × p_std)` per unit time
- **I-I coalescence rate**: `k_I(k_I - 1) / (2 × p_inv)` per unit time
- **S ↔ I exchange**: only via gene flux (see below)

Lineages in the rarer class coalesce faster because they see a smaller effective population.

## Gene flux (Peischl model)

Gene flux models double crossovers (gene conversion tracts) that can transfer short segments between opposing karyotype orientations. The rate per lineage is:

```
flux_rate = k × gamma × p_other × phi(x)
```

where:
- `gamma = 4·N·g` is the scaled gene conversion rate (decoupled from recombination rate in msinv)
- `p_other` is the frequency of the opposite arrangement (heterokaryotype frequency)
- `phi(x) = min(x, 1-x, w) / (1 - w)` is the position-dependent factor
  - `x` = position within inversion (fraction, 0 to 1)
  - `w` = flux window width parameter (default 0.3)

**phi(x) shape**: trapezoid peaking at center, zero at breakpoints. This creates the characteristic divergence gradient observed empirically: breakpoint regions have deeper S-I divergence than the center.

## Correlated flux tracts (Peischl b2)

Each gene flux event transfers a tract of DNA, not a single site. When a flux event fires at position x (drawn as `b1`), the right boundary `b2 = b1 + w` is also computed. Going backward in time along the chromosome:
- At `x = b1`: class switches
- At `x = b2`: class reverts

This models the Peischl b2 tract-length dependence and produces realistic LD decay.

## Recombination suppression

Inside the inversion, effective recombination rate for a single lineage is:

```
effective_rate = rho × p_same
```

where `p_same = p_std` for S lineages and `p_inv` for I lineages. This models the fact that crossovers inside the inversion only occur in homokaryotype matings (SS or II); heterokaryotype matings produce crossover products that are inviable (paracentric) or unbalanced (pericentric).

In homokaryotypes, recombination proceeds normally. In heterokaryotypes:
- Crossovers inside the inversion → inviable gametes (no gene flow)
- Gene conversion / double crossover → small tract transferred (gene flux)

## Inversion age (t_inv)

Inversions are finite-age events. msinv bounds the maximum S-I divergence by `t_inv`:
- At t < t_inv: full structured coalescent with flux
- At t ≥ t_inv: inversion didn't exist, all lineages panmictic

Without a finite t_inv, S-I coalescence time diverges to infinity at breakpoints (where flux rate → 0).

## Frequency trajectories

Four trajectory types are supported:

1. **ConstantFrequency**: fixed `p_inv`, with optional `t_inv` cutoff
2. **DeterministicTrajectory**: logistic sweep from 1/(2N) to p_final under selection s
3. **StochasticTrajectory**: WF diffusion backward in time with reflecting boundary at p=0 (models recurrent origins at the same breakpoints)
4. **CoupledTrajectory**: per-population 2D diffusion with local selection and migration (for local adaptation scenarios)

## 4-walk strategy

To eliminate left-to-right asymmetry from the SMC walk, msinv uses a 4-walk strategy per inversion:

- Walk 1: `bp_left → center` (rightward, structured)
- Walk 2: `bp_right → center` (leftward via mirror, structured)
- Walk 3: `bp_left → 0` (leftward via mirror, collinear)
- Walk 4: `bp_right → 1` (rightward, collinear)

Each walk builds a fresh tree at the boundary and accumulates recombination events. Mutations are pooled from all four walks.

## References

- Guerrero, R. F., Rousset, F., & Kirkpatrick, M. (2012). *Coalescent patterns for chromosomal inversions in divergent populations.* Phil Trans R Soc B 367:430–438.
- Peischl, S., Koch, E., Guerrero, R. F., & Kirkpatrick, M. (2013). *A sequential coalescent algorithm for chromosomal inversions.* Heredity 111:200–209.
- Kirkpatrick, M., & Barton, N. (2006). *Chromosome inversions, local adaptation and speciation.* Genetics 173:419–434.
- McVean, G. A., & Cardin, N. J. (2005). *Approximating the coalescent with recombination.* Phil Trans R Soc B 360:1387–1393.
