# Structured Coalescent with Chromosomal Inversions: Mathematical Framework

**msinv** — a position-tracking (hull) coalescent simulator for chromosomal inversions

---

## 1. Model Overview

We model the coalescent for a chromosomal region containing one or more inversions. Each inversion defines two karyotype classes — Standard (**S**) and Inverted (**I**) — that act as partially isolated subpopulations. The key features:

- **Karyotype class barrier**: recombination between S and I chromosomes is suppressed inside inversions in heterokaryotypes, creating divergence.
- **Gene flux**: ectopic gene conversion allows limited exchange between classes, with a position-dependent rate that is highest at the inversion centre and vanishes at breakpoints.
- **Multiple inversions**: independent or overlapping inversions on the same chromosome, each with its own age, frequency, and class structure.
- **Selective sweeps**: hitchhiking model for class-specific sweeps.
- **Multi-population demography**: migration, population splits, size changes.

The simulator tracks ancestral material per base pair (the "hull" representation), allowing exact treatment of recombination, gene flux tracts, and position-dependent class membership.

---

## 2. State Space

### 2.1 Lineages and Segments

A **lineage** is a set of contiguous or disjoint **segments** $[l_i, r_i)$ of ancestral material, each carrying:

- A **node ID** in the tree sequence
- A **branch class** $c \in \{S_k, I_k, P\}$ indicating karyotype at inversion $k$

For $K$ inversions, a segment inside the overlap of inversions $k$ and $j$ carries a class tag $c \in \{S_k, I_k\} \times \{S_j, I_j\}$ (implemented as a `frozenset`).

### 2.2 Populations and Classes

Let there be $D$ populations (demes) and $K$ inversions. Each lineage is in population $d \in \{1, \ldots, D\}$. At a position $x$ inside inversion $k$, the lineage belongs to class $S_k$ or $I_k$. Outside all inversions, every lineage has class $P$ (panmictic).

Define:
- $p_k$ = frequency of the inverted arrangement at inversion $k$
- $q_k = 1 - p_k$ = frequency of the standard arrangement
- $N_d(t)$ = effective size of population $d$ at time $t$ (backward)
- $t_k$ = age of inversion $k$ (the time at which the class barrier is lifted going backward)

---

## 3. Coalescence Rates

### 3.1 Within a Single Inversion

For two lineages in the same population $d$, both carrying ancestral material at position $x$ inside inversion $k$ with the same class $c \in \{S_k, I_k\}$:

$$\lambda_{\text{coal}}(c, d) = \frac{1}{2 \, p_c \, N_d(t)}$$

where $p_c = p_k$ for class $I_k$ and $p_c = q_k$ for class $S_k$. This is the standard structured coalescent rate for a deme of effective size $p_c \cdot N_d$.

For lineages in *different* classes at position $x$ (one $S_k$, one $I_k$): coalescence rate is **zero** while $t < t_k$. At $t = t_k$ the class barrier is lifted and both lineages become $P$, after which they coalesce at rate $1/(2 N_d(t))$.

### 3.2 Multiple Inversions

For $K$ inversions, a segment at position $x$ may lie inside multiple inversions simultaneously. Its class is a tuple $(c_1, \ldots, c_m)$ for the $m$ inversions covering $x$. The effective population fraction for this compound class is:

$$p_{\text{compound}} = \prod_{j=1}^{m} p_{c_j}$$

Two lineages at position $x$ can coalesce only if they share the same compound class at $x$. The rate is:

$$\lambda_{\text{coal}} = \frac{1}{2 \, p_{\text{compound}} \, N_d(t)}$$

As each inversion reaches its age $t_k$ going backward, that inversion's class tag is removed and lineages that differed only at inversion $k$ can now coalesce.

### 3.3 Per-Pair Overlap Computation

Because recombination fragments lineages, two lineages may overlap at only a subset of positions. The simulator computes exact pairwise overlap by class:

For lineages $a$ and $b$, define the overlap in class $c$:

$$O_c(a, b) = \sum_{\substack{s_a \in a, \, s_b \in b \\ c(s_a) = c(s_b) = c}} |[l_{s_a}, r_{s_a}) \cap [l_{s_b}, r_{s_b})|$$

Each $(a, b, c)$ triple with $O_c > 0$ generates a coalescence event at rate $1/(2 \, p_c \, N_d)$.

At high recombination rates ($\rho > 100$), a Hudson-style bucket approximation groups lineages by class and uses the rate $\binom{k}{2}/(2 \, p_c \, N_d)$ with rejection sampling for non-overlapping pairs.

---

## 4. Recombination

Recombination splits a lineage at a uniformly random position within its ancestral material. The per-lineage recombination rate is:

$$\lambda_{\text{recomb}}(\ell) = r \cdot L_{\text{anc}}(\ell)$$

where $r$ is the per-bp per-generation recombination rate and $L_{\text{anc}}(\ell)$ is the total length of ancestral material carried by lineage $\ell$.

Inside an inversion in a heterokaryotype, crossing over between S and I produces unbalanced gametes and is effectively suppressed. In homokaryotypes (S/S or I/I), recombination proceeds normally. The effective recombination rate inside inversion $k$ for class $c$ is:

$$r_{\text{eff}}(x) = r \cdot p_c \quad \text{(for } x \text{ inside inversion } k, \text{ lineage of class } c\text{)}$$

This reflects the probability that the recombination partner is a homokaryotype.

---

## 5. Gene Flux

### 5.1 The $\phi(x)$ Function

Gene flux (ectopic gene conversion) allows transfer of short tracts between karyotype classes. We use the Peischl et al. (2013) model where the probability that position $x$ inside the inversion is covered by a random conversion tract of relative width $w$ is:

$$\phi(x; w) = \frac{\min(x, \, 1-x, \, w)}{1 - w}$$

where $x \in (0, 1)$ is the position in inversion-relative coordinates ($x = 0$ at the left breakpoint, $x = 1$ at the right breakpoint), and $w$ is the gene conversion tract length relative to inversion length.

This is a triangular "roof" function:
- $\phi(x) = 0$ at both breakpoints ($x = 0$ and $x = 1$)
- $\phi(x)$ rises linearly from 0 to $w/(1-w)$ over $[0, w]$
- $\phi(x) = w/(1-w)$ on the plateau $[w, 1-w]$
- $\phi(x)$ falls linearly back to 0 over $[1-w, 1]$

### 5.2 Gene Flux Rate

For a lineage of class $c$ at inversion $k$ in population $d$, the per-generation gene-flux rate is:

$$\lambda_{\text{flux}}(\ell, k) = \gamma_k \cdot p_{\bar{c}} \cdot \int_{\text{inv}} \phi\!\left(\frac{x - b_L}{b_R - b_L}\right) \mathbf{1}[x \in \ell] \, dx$$

where:
- $\gamma_k$ is the per-bp gene conversion initiation rate for inversion $k$
- $p_{\bar{c}}$ is the frequency of the *other* class ($p_{\bar{c}} = p_k$ if $c = S_k$, or $q_k$ if $c = I_k$)
- $b_L, b_R$ are the inversion breakpoints
- $\mathbf{1}[x \in \ell]$ indicates that position $x$ is carried as ancestral material by lineage $\ell$

The integral is computed exactly via the closed-form antiderivative of $\phi$.

### 5.3 Gene Flux Event

When a gene-flux event fires on lineage $\ell$ at inversion $k$, a conversion tract $[x_0, x_0 + w \cdot L_k)$ is placed at a random position weighted by $\phi(x)$ within the lineage's in-inversion material. The tract is excised from $\ell$ and placed on a new lineage with the *opposite* class at inversion $k$:

$$c_{\text{new}} = \begin{cases} I_k & \text{if } c = S_k \\ S_k & \text{if } c = I_k \end{cases}$$

For nested inversions, only inversion $k$'s tag is flipped; tags for other inversions are preserved.

### 5.4 Gene Flux Only Within the Same Population

Gene flux can only occur between chromosomes in the same population. Cross-population gene flux is not modelled — populations exchange genetic material only through migration.

---

## 6. Demographic Events

The simulator supports ms-style demographic events applied at discrete times going backward:

| Event | Effect |
|-------|--------|
| `en(t, d, N)` | Set $N_d = N$ at time $t$ |
| `eg(t, d, \alpha)` | Set exponential growth: $N_d(t') = N_d(t) \cdot e^{-\alpha(t'-t)}$ |
| `em(t, i, j, M)` | Set migration rate $M_{ij}$ |
| `ej(t, i, j)` | Merge population $i$ into $j$; zero all migration to/from $i$ |

Migration moves individual lineages between populations at rate $M_{ij}$ per source lineage per generation (backward-time).

---

## 7. Selective Sweeps

### 7.1 Hitchhiking Mode

A selective sweep at position $x_{\text{sel}}$ with selection coefficient $s$ on class $c$ is modelled by probabilistic inclusion of segments. Each segment with midpoint $x$ is included in the sweep with probability:

$$P(\text{linked}) = \exp\!\left(-r \cdot |x - x_{\text{sel}}| \cdot t_{\text{dur}}\right)$$

where $t_{\text{dur}} = \ln(2 N_e s) / s$ is the sweep duration (Maynard Smith & Haigh 1974). Included segments are excised from their lineages and force-coalesced to a single ancestor at $t_{\text{event}}$. Non-included segments remain on their original lineages.

This produces the classic hitchhiking valley: $T_{\text{MRCA}} = t_{\text{event}}$ at $x_{\text{sel}}$, with the effect decaying exponentially with recombination distance.

### 7.2 Window Mode

An alternative deterministic mode force-coalesces all qualifying lineages with material in $[x_{\text{sel}} - w, x_{\text{sel}} + w]$ at time $t_{\text{event}}$. This is the Hudson & Kaplan (1995) approximation.

### 7.3 Class-Specific Sweeps

Sweeps target a specific class ($S$, $I$, or any). For the RDL-style sweep-through-inversion scenario, two sweeps are used: first on $S$ at $t_S$, then on $I$ at $t_I < t_S$ (more recent), representing a gene-conversion transfer of the selected allele from S to I background.

---

## 8. Theoretical Predictions

### 8.1 Expected Coalescence Times

For a neutral locus at position $x$ inside inversion $k$ with no gene flux ($\gamma_k = 0$), frequency $p_k$, and population size $N_e$:

**Within class:**
$$E[T_{\text{coal}} \mid \text{same class } c] = 2 \, p_c \, N_e$$

**Between classes:**
$$E[T_{\text{coal}} \mid S_k \text{ vs } I_k] = t_k + 2 \, N_e$$

**With gene flux** (Guerrero, Rousset & Kirkpatrick 2012):
$$E[T_{\text{coal}} \mid S_k \text{ vs } I_k] \approx \frac{1}{2 \gamma_k \phi(x)} + 2 N_e$$

where $\phi(x)$ is the position-dependent flux rate.

### 8.2 Expected Divergence

$$E[d_{XY}(x)] = 2 \mu \cdot E[T_{\text{coal}}(x)]$$

(Factor of 2: mutations accumulate on **both** lineages leading to the MRCA.)

Inside the inversion (no flux):
$$E[d_{XY}] = 2 \mu \left(t_k + 2 N_e\right)$$

Outside the inversion (or after $t_k$):
$$E[d_{XY}] = 2 \mu \cdot 2 N_e = \theta$$

The ratio of inside-to-outside divergence:
$$\frac{d_{XY}^{\text{inv}}}{d_{XY}^{\text{col}}} = 1 + \frac{t_k}{2 N_e}$$

(Navarro & Barton 2003)

### 8.3 Expected Diversity

Within-class nucleotide diversity at a neutral locus:
$$E[\pi_c] = 4 \, p_c \, N_e \, \mu = p_c \cdot \theta$$

This is reduced relative to the panmictic $\theta$ by the class frequency $p_c$, because each class is effectively a subpopulation of size $p_c N_e$.

**Note:** Total diversity (pooling both classes) is *elevated* inside the inversion:
$$E[\pi_{\text{total}}] = \theta + 2 \, p_k \, q_k \, \mu \, t_k$$

The excess is driven by the between-class component (Charlesworth & Charlesworth 1973).

### 8.4 Expected $F_{ST}$

Hudson's $F_{ST}$ between karyotype classes at a locus inside the inversion:

$$F_{ST} = 1 - \frac{\pi_W}{d_{XY}} = 1 - \frac{2 \, p_k \, q_k \, N_e}{t_k + 2 N_e}$$

For balanced frequencies ($p_k = q_k = 0.5$):
$$F_{ST} = 1 - \frac{N_e}{t_k + 2 N_e}$$

$F_{ST} \to 0$ for young inversions ($t_k \ll 2 N_e$) and $F_{ST} \to 1$ for old inversions ($t_k \gg 2 N_e$).

At the breakpoints, gene flux erases differentiation: $F_{ST}(x \to 0) \to 0$.

### 8.5 Gene Flux Regime

The balance between class isolation and gene-flux mixing is governed by the compound parameter:

$$4 \, N_e \, \gamma_k \, \phi(x)$$

| Regime | Condition | Behaviour |
|--------|-----------|-----------|
| Isolated | $4 N_e \gamma \phi(x) \ll 1$ | Full divergence: $d_{XY} \approx \mu(t_k + 2N_e)$ |
| Transition | $4 N_e \gamma \phi(x) \sim 1$ | Partial erosion of barrier |
| Panmictic | $4 N_e \gamma \phi(x) \gg 1$ | No structure: $d_{XY} \approx 2 N_e \mu$ |

Because $\phi(x) = 0$ at breakpoints, breakpoints always remain in the isolated regime regardless of $\gamma$.

---

## 9. Multiple and Overlapping Inversions

### 9.1 Independent Inversions

For $K$ non-overlapping inversions on the same chromosome, each inversion $k$ independently creates its own class barrier with parameters $(p_k, t_k, \gamma_k)$. Segments carry a single class tag per inversion. The collinear gap between inversions is panmictic.

A sample can carry different karyotypes at different inversions (e.g., $S$ at inversion 0 and $I$ at inversion 1), representing recombinant chromosomes. Sample configurations are specified as tuples: `('S', 'I')` denotes Standard at inv 0, Inverted at inv 1.

### 9.2 Overlapping (Nested) Inversions

When inversions overlap, a segment in the overlap region carries a compound class — a frozenset of tags, e.g., $\{S_0, I_1\}$. Two lineages at such a position can coalesce only if their compound classes match exactly.

The effective population fraction for compound class $\{c_0, c_1\}$ is $p_{c_0} \cdot p_{c_1}$ (assuming independence of the two inversions). Class barriers are lifted independently: when $t = t_0$, the $S_0/I_0$ tags are removed but $S_1/I_1$ tags persist until $t = t_1$.

---

## 10. Algorithm Summary

The simulator implements a continuous-time Markov chain on the space of active lineages. At each step:

1. Compute all event rates: coalescence (per pair or per bucket), recombination, gene flux, migration, demographic events, sweeps.
2. Sample the next event time $\Delta t \sim \text{Exp}(\Lambda_{\text{total}})$.
3. If a demographic event or class-barrier lifting occurs before $t + \Delta t$, advance to that time and process the event.
4. Otherwise, advance by $\Delta t$ and execute the sampled event (coalescence, recombination, gene flux, or migration).
5. Record edges and nodes in the tree-sequence table.
6. Repeat until one lineage remains.
7. Call `simplify()` on the final table collection to remove unary nodes and non-ancestral edges.

The output is a `tskit.TreeSequence` compatible with all downstream tskit analyses.

---

## References

1. Charlesworth B, Charlesworth D (1973). Selection of new inversions in multi-locus genetic systems. *Genet Res* 21:167–183.

2. Charlesworth B, Nordborg M, Charlesworth D (1997). The effects of local selection, balanced polymorphism and background selection on equilibrium patterns of genetic diversity in subdivided populations. *Genet Res* 70:155–174.

3. Guerrero RF, Rousset F, Kirkpatrick M (2012). Coalescent patterns for chromosomal inversions in divergent populations. *Phil Trans R Soc B* 367:430–438.

4. Hudson RR, Kaplan NL (1988). The coalescent process in models with selection and recombination. *Genetics* 120:831–840.

5. Kirkpatrick M, Barton NH (2006). Chromosome inversions, local adaptation, and speciation. *Genetics* 173:419–434.

6. Maynard Smith J, Haigh J (1974). The hitch-hiking effect of a favourable gene. *Genet Res* 23:23–35.

7. Navarro A, Betran E, Barbadilla A, Ruiz A (1997). Recombination and gene flux caused by gene conversion and crossing over in inversion heterokaryotypes. *Genetics* 146:695–709.

8. Navarro A, Barton NH (2003). Accumulating postzygotic isolation genes in parapatry: a new twist on chromosomal speciation. *Evolution* 57:447–459.

9. Nordborg M (1997). Structured coalescent processes on different time scales. *Genetics* 146:1501–1514.

10. Peischl S, Kirkpatrick M, Burger R (2013). The evolution of recombination rates and chromosome structure under balancing selection. *Genetics* 195:1385–1406.

11. Wakeley J (2009). *Coalescent Theory: An Introduction.* Roberts & Company.
