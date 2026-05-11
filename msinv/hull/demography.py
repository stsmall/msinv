"""Demography for the hull simulator (in generations).

Multi-population demographic models with migration and ms-style
discrete events (population size changes, growth, migration changes,
mergers, splits). Time in generations (NOT coalescent units), to match
the hull simulator's main loop.

Event types
-----------
``('en', t, pop, N)``         Set pop's size to N at time t (going backward).
``('eN', t, N)``              Set ALL pops' sizes to N.
``('eg', t, pop, alpha)``     Set pop's exp-growth rate to alpha.
                              N(t') = N(t) * exp(-alpha * (t' - t))
``('eG', t, alpha)``          Set ALL pops' growth rate to alpha.
``('em', t, dst, src, M)``    Set migration rate from src → dst (per
                              generation, per src lineage) to M.
``('eM', t, M)``              Set ALL off-diagonal migration rates to
                              M / (n_pops - 1).
``('ej', t, src, dst)``       Going backward: at time t every lineage
                              in pop ``src`` moves to pop ``dst``;
                              both populations' migration is zeroed
                              afterward (standard ms convention).

Future work: demestats (Ragsdale 2026) integration as the analytical
rate engine for arbitrary demes graphs.
"""

import math
from typing import Optional


class Demography:
    """Multi-population demographic model in generations.

    Parameters
    ----------
    pop_sizes : list[float]
        Per-population effective sizes at t=0 (present-day).
    migration_matrix : list[list[float]], optional
        ``M[i][j]`` is the per-generation per-src-lineage migration rate
        from population j → i (going backward in time). Diagonals are
        ignored. Defaults to no migration.
    """

    def __init__(
        self,
        pop_sizes: list[float],
        migration_matrix: Optional[list[list[float]]] = None,
    ):
        self.pop_sizes = list(map(float, pop_sizes))
        self.n_pops = len(self.pop_sizes)
        self.growth_rates = [0.0] * self.n_pops
        self.growth_start = [0.0] * self.n_pops
        if migration_matrix is None:
            self.migration_matrix = [[0.0] * self.n_pops for _ in range(self.n_pops)]
        else:
            if len(migration_matrix) != self.n_pops or any(
                len(row) != self.n_pops for row in migration_matrix
            ):
                raise ValueError(
                    f"migration_matrix shape must be {self.n_pops}x{self.n_pops}."
                )
            self.migration_matrix = [list(map(float, row)) for row in migration_matrix]
        # Original events for replaying after a copy() / reset.
        self.events: list[tuple] = []

    def add_event(self, event: tuple):
        """Add a raw ms-style demographic event. Events are kept sorted by time.

        Prefer the named methods below (``add_population_split``,
        ``add_population_size_change``, etc.) for readability.
        """
        self.events.append(event)
        self.events.sort(key=lambda e: e[1])

    def check_connectivity(self, warn: bool = True) -> bool:
        """Verify cross-population coalescence is possible. Returns True
        if connected; False if disjoint components remain after all
        events. When disjoint and ``warn`` is True, emits a warning.

        Lineages in disjoint components never reach a common ancestor
        — downstream Hudson recap via msprime will hang with "infinite
        waiting time until next simulation event".

        Edges considered:
          - migration_matrix nonzero (either direction)
          - 'ej' events (src → dst merge)
          - 'em' events changing migration_matrix at some time
        """
        if self.n_pops <= 1:
            return True
        parent = list(range(self.n_pops))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(self.n_pops):
            for j in range(self.n_pops):
                if i == j:
                    continue
                if self.migration_matrix[i][j] != 0.0:
                    union(i, j)
        for ev in self.events:
            if ev[0] == "ej" and len(ev) >= 4:
                union(int(ev[2]), int(ev[3]))
            elif ev[0] == "em" and len(ev) >= 5 and ev[4] != 0.0:
                union(int(ev[2]), int(ev[3]))
            elif ev[0] == "cmig" and len(ev) >= 7 and ev[6] > 0.0:
                # ('cmig', t, src, dst, kary, inv_id, proportion)
                # Class-conditional migration is also a connectivity edge.
                union(int(ev[2]), int(ev[3]))
            elif ev[0] == "ema" and len(ev) >= 3:
                mat = ev[2]
                for i in range(self.n_pops):
                    for j in range(self.n_pops):
                        if i != j and mat[i][j] != 0.0:
                            union(i, j)
        roots = {find(i) for i in range(self.n_pops)}
        if len(roots) == 1:
            return True
        if warn:
            import warnings as _warnings

            _warnings.warn(
                f"Demography has {len(roots)} disjoint population "
                f"components: {roots}. Lineages across components will "
                f"never reach a common ancestor — msprime recap will "
                f"hang with 'infinite waiting time'. Add migration or "
                f"an 'ej' event.",
                RuntimeWarning,
                stacklevel=2,
            )
        return False

    # -- msprime-compatible convenience methods ----------------------------

    def add_population_split(self, time: float, derived: list[int], ancestral: int):
        """Going backward, merge *derived* populations into *ancestral*.

        Equivalent to msprime ``Demography.add_population_split``
        (and to one ``ej`` event per derived pop in ms).
        """
        for src in derived:
            self.add_event(("ej", time, src, ancestral))

    def add_mass_migration(
        self, time: float, source: int, dest: int, proportion: float = 1.0
    ):
        """Going backward, move *proportion* of lineages from *source*
        to *dest*.

        With proportion=1.0 this is identical to ``ej``.  For fractional
        proportions, use ``add_class_migration`` (or call
        ``add_admixture``) which can also condition on karyotype.
        """
        if proportion != 1.0:
            raise NotImplementedError(
                "Unconditional fractional mass migration (es) not yet "
                "implemented; only the class-conditional version is. "
                "Use add_class_migration(time, src, dst, kary, inv_id, "
                "proportion) or add_admixture(time, src, dst, proportion, "
                "kary=...) — both support proportion < 1."
            )
        self.add_event(("ej", time, source, dest))

    def add_class_migration(
        self,
        time: float,
        source: int,
        dest: int,
        karyotype: str,
        inv_id: int = 0,
        proportion: float = 1.0,
    ):
        """Class-conditional migration / admixture / class-mass-merge.

        Going backward at ``time``, for each lineage currently in
        ``source`` whose karyotype at inversion ``inv_id`` matches
        ``karyotype`` ('S' or 'I'), move it to ``dest`` with probability
        ``proportion``.

        Uses
        ----
        - ``proportion=1.0`` → unconditional class merge.  Models e.g.
          "K's founders at the K-F split were all S karyotype" (forward
          view) by setting ``add_class_migration(t_split, src=K, dst=F,
          karyotype='S')`` — going backward all of K's S lineages join
          F at the split.
        - ``proportion<1.0`` → stochastic admixture / class-conditional
          migration pulse.  Each matching lineage migrates independently
          with the given probability.
        """
        if karyotype not in ("S", "I"):
            raise ValueError(f"karyotype must be 'S' or 'I', got {karyotype!r}")
        if not (0.0 < proportion <= 1.0):
            raise ValueError(f"proportion must be in (0, 1], got {proportion}")
        self.add_event(("cmig", time, source, dest, karyotype, inv_id, proportion))

    def add_admixture(
        self,
        time: float,
        source: int,
        dest: int,
        proportion: float,
        karyotype: Optional[str] = None,
        inv_id: int = 0,
    ):
        """Admixture pulse: at ``time``, fraction ``proportion`` of
        ``source`` migrates into ``dest`` (going backward).

        If ``karyotype`` is given, only lineages of that class at
        ``inv_id`` are eligible.  Otherwise this is currently a synonym
        for ``add_class_migration`` with karyotype='S' (placeholder
        until unconditional fractional migration lands).

        Caveat: with extensive gene flux, some lineages may have all
        of their inv-region segments converted to PAN class.  These
        lineages are NOT caught by cmig (kary check returns None).
        For proportion=1.0 / class-merge use cases, prefer
        ``add_class_split`` which adds a residual ej to catch the
        PAN stragglers.
        """
        if karyotype is None:
            raise NotImplementedError(
                "Class-unconditional admixture (proportion < 1, no "
                "karyotype filter) not yet implemented; supply "
                "karyotype='S' or 'I' to use the class-conditional "
                "operator instead."
            )
        self.add_class_migration(time, source, dest, karyotype, inv_id, proportion)

    def add_class_split(self, time: float, source: int, dest: int, inv_id: int = 0):
        """Class-routed full split: at ``time`` going backward, every
        lineage in ``source`` moves to ``dest``, but the simulator
        routes S- and I-lineages through their class buckets first
        (cmig kary='S', then cmig kary='I'), then mops up any PAN-only
        lineages with a final ej.

        Functionally equivalent to ``add_population_split`` for
        connectivity purposes, but exercises the class-routing code
        path.  Use this if you want to know whether class-routed
        coalescence at the split changes anything compared with
        plain ej (in our K=S-only, F=S+I sampling, it does not).

        Caveats:
        - Final ej is required because gene flux can convert a
          lineage's full inv span to PAN; cmig misses those.
        - With proportion=1.0, the cmig events are unconditional —
          they don't consume RNG (no Bernoulli draw needed).
        """
        self.add_class_migration(time, source, dest, "S", inv_id, 1.0)
        self.add_class_migration(time, source, dest, "I", inv_id, 1.0)
        self.add_event(("ej", float(time), int(source), int(dest)))

    def add_population_size_change(
        self,
        time: float,
        population: Optional[int] = None,
        new_size: Optional[float] = None,
    ):
        """Change effective population size going backward.

        If *population* is None, change all populations (``eN``).
        Otherwise change only that population (``en``).
        """
        if population is None:
            self.add_event(("eN", time, new_size))
        else:
            self.add_event(("en", time, population, new_size))

    def add_growth_rate_change(
        self, time: float, population: Optional[int] = None, growth_rate: float = 0.0
    ):
        """Set exponential growth rate going backward.

        N(t') = N(t) * exp(-growth_rate * (t' - t)).

        If *population* is None, change all populations (``eG``).
        """
        if population is None:
            self.add_event(("eG", time, growth_rate))
        else:
            self.add_event(("eg", time, population, growth_rate))

    def add_migration_rate_change(
        self,
        time: float,
        source: Optional[int] = None,
        dest: Optional[int] = None,
        rate: float = 0.0,
    ):
        """Change migration rate going backward.

        If *source* and *dest* are both None, set all off-diagonal
        migration rates (``eM``).  Otherwise set the single rate
        from *source* to *dest* (``em``; note dest receives migrants
        from source, matching msprime convention).
        """
        if source is None and dest is None:
            self.add_event(("eM", time, rate))
        else:
            self.add_event(("em", time, dest, source, rate))

    def add_inversion_freq_change(
        self, time: float, population: int, inv_id: int, p_inv: float
    ):
        """Change inversion frequency for a specific population at *time*.

        This is useful at population merge events where the ancestral
        population has a different inversion frequency than the derived
        populations (e.g. K has p_inv=0, F has p_inv=0.73, and the
        ancestral pop had p_inv=0.3).

        Parameters
        ----------
        time : float
            Time in generations (going backward).
        population : int
            Population index to modify.
        inv_id : int
            Inversion index (0-based, matching the inversions list order).
        p_inv : float
            New inverted-arrangement frequency for this population.
        """
        self.add_event(("eig", time, population, inv_id, p_inv))

    def copy(self) -> "Demography":
        """Fresh copy with replayable events."""
        d = Demography(list(self.pop_sizes), [list(r) for r in self.migration_matrix])
        d.growth_rates = list(self.growth_rates)
        d.growth_start = list(self.growth_start)
        d.events = list(self.events)
        return d

    # -- size accessors ---------------------------------------------------

    def size_at(self, pop: int, t: float) -> float:
        """Effective size of population ``pop`` at time ``t`` (going
        backward), accounting for current growth rate and the most recent
        size-set event. Does NOT replay events past ``t``.
        """
        if pop >= self.n_pops:
            return 1.0
        N = self.pop_sizes[pop]
        g = self.growth_rates[pop]
        if g == 0.0:
            return N
        return N * math.exp(-g * (t - self.growth_start[pop]))

    # -- event application -------------------------------------------------

    def next_event_time(self, t_now: float) -> float:
        """Time of the next event at or after t_now (or +inf).

        Uses ``>= t_now`` (with a tiny epsilon tolerance) so that
        events scheduled at exactly the current time are still
        returned — important when other event types (class barrier,
        sweep) fire at the same time and advance ``t`` to that value
        without consuming the demographic event.
        """
        for ev in self.events:
            if ev[1] >= t_now - 1e-9:
                return ev[1]
        return float("inf")

    def apply_event_at(self, t: float, active_lineages):
        """Apply all events scheduled for time t in chronological order,
        mutating ``active_lineages`` in place where required.

        Returns a list of (inv_id, pop, p_inv) tuples for any ``eig``
        events that fired — the caller must apply these to its
        InversionSpec list.
        """
        new_events = []
        inv_changes = []
        for ev in self.events:
            if abs(ev[1] - t) > 1e-9:
                new_events.append(ev)
                continue
            etype = ev[0]
            if etype == "eN":
                _, _, N = ev
                for p in range(self.n_pops):
                    # Lock in current size with growth, then reset.
                    self.pop_sizes[p] = N
                    self.growth_rates[p] = 0.0
                    self.growth_start[p] = t
            elif etype == "en":
                _, _, p, N = ev
                if p < self.n_pops:
                    self.pop_sizes[p] = N
                    self.growth_rates[p] = 0.0
                    self.growth_start[p] = t
            elif etype == "eG":
                _, _, alpha = ev
                for p in range(self.n_pops):
                    self.pop_sizes[p] = self.size_at(p, t)
                    self.growth_rates[p] = alpha
                    self.growth_start[p] = t
            elif etype == "eg":
                _, _, p, alpha = ev
                if p < self.n_pops:
                    self.pop_sizes[p] = self.size_at(p, t)
                    self.growth_rates[p] = alpha
                    self.growth_start[p] = t
            elif etype == "eM":
                _, _, M = ev
                if self.n_pops > 1:
                    per = M / (self.n_pops - 1)
                else:
                    per = 0.0
                for i in range(self.n_pops):
                    for j in range(self.n_pops):
                        if i != j:
                            self.migration_matrix[i][j] = per
            elif etype == "em":
                _, _, dst, src, M = ev
                if dst < self.n_pops and src < self.n_pops and dst != src:
                    self.migration_matrix[dst][src] = M
            elif etype == "ej":
                _, _, src, dst = ev
                # Move every active lineage in src to dst.
                for lin in active_lineages:
                    if lin.population == src:
                        lin.population = dst
                # Zero migration to/from src (it no longer exists).
                for k in range(self.n_pops):
                    self.migration_matrix[src][k] = 0.0
                    self.migration_matrix[k][src] = 0.0
            elif etype == "eig":
                _, _, pop, inv_id, p_inv = ev
                inv_changes.append((inv_id, pop, p_inv))
            else:
                raise ValueError(f"Unknown demographic event type: {etype}")
        # Drop processed events.
        self.events = new_events
        return inv_changes
