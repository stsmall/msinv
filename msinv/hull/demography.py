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
from typing import List, Optional, Tuple


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

    def __init__(self,
                 pop_sizes: List[float],
                 migration_matrix: Optional[List[List[float]]] = None):
        self.pop_sizes = list(map(float, pop_sizes))
        self.n_pops = len(self.pop_sizes)
        self.growth_rates = [0.0] * self.n_pops
        self.growth_start = [0.0] * self.n_pops
        if migration_matrix is None:
            self.migration_matrix = [[0.0] * self.n_pops
                                      for _ in range(self.n_pops)]
        else:
            if len(migration_matrix) != self.n_pops or any(
                    len(row) != self.n_pops for row in migration_matrix):
                raise ValueError(
                    f"migration_matrix shape must be "
                    f"{self.n_pops}x{self.n_pops}.")
            self.migration_matrix = [list(map(float, row))
                                       for row in migration_matrix]
        # Original events for replaying after a copy() / reset.
        self.events: List[Tuple] = []

    def add_event(self, event: Tuple):
        """Add a raw ms-style demographic event. Events are kept sorted by time.

        Prefer the named methods below (``add_population_split``,
        ``add_population_size_change``, etc.) for readability.
        """
        self.events.append(event)
        self.events.sort(key=lambda e: e[1])

    # -- msprime-compatible convenience methods ----------------------------

    def add_population_split(self, time: float, derived: List[int],
                             ancestral: int):
        """Going backward, merge *derived* populations into *ancestral*.

        Equivalent to msprime ``Demography.add_population_split``
        (and to one ``ej`` event per derived pop in ms).
        """
        for src in derived:
            self.add_event(('ej', time, src, ancestral))

    def add_mass_migration(self, time: float, source: int, dest: int,
                           proportion: float = 1.0):
        """Going backward, move *proportion* of lineages from *source*
        to *dest*.

        With proportion=1.0 this is identical to ``ej``.  Fractional
        proportions (``es`` in ms) are not yet implemented; use
        proportion=1.0 for now.
        """
        if proportion != 1.0:
            raise NotImplementedError(
                "Fractional mass migration (es) not yet implemented. "
                "Use proportion=1.0 for a full population merge.")
        self.add_event(('ej', time, source, dest))

    def add_population_size_change(self, time: float,
                                   population: Optional[int] = None,
                                   new_size: Optional[float] = None):
        """Change effective population size going backward.

        If *population* is None, change all populations (``eN``).
        Otherwise change only that population (``en``).
        """
        if population is None:
            self.add_event(('eN', time, new_size))
        else:
            self.add_event(('en', time, population, new_size))

    def add_growth_rate_change(self, time: float,
                               population: Optional[int] = None,
                               growth_rate: float = 0.0):
        """Set exponential growth rate going backward.

        N(t') = N(t) * exp(-growth_rate * (t' - t)).

        If *population* is None, change all populations (``eG``).
        """
        if population is None:
            self.add_event(('eG', time, growth_rate))
        else:
            self.add_event(('eg', time, population, growth_rate))

    def add_migration_rate_change(self, time: float,
                                  source: Optional[int] = None,
                                  dest: Optional[int] = None,
                                  rate: float = 0.0):
        """Change migration rate going backward.

        If *source* and *dest* are both None, set all off-diagonal
        migration rates (``eM``).  Otherwise set the single rate
        from *source* to *dest* (``em``; note dest receives migrants
        from source, matching msprime convention).
        """
        if source is None and dest is None:
            self.add_event(('eM', time, rate))
        else:
            self.add_event(('em', time, dest, source, rate))

    def copy(self) -> 'Demography':
        """Fresh copy with replayable events."""
        d = Demography(list(self.pop_sizes),
                        [list(r) for r in self.migration_matrix])
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
        return float('inf')

    def apply_event_at(self, t: float, active_lineages):
        """Apply all events scheduled for time t in chronological order,
        mutating ``active_lineages`` in place where required.
        """
        new_events = []
        for ev in self.events:
            if abs(ev[1] - t) > 1e-9:
                new_events.append(ev)
                continue
            etype = ev[0]
            if etype == 'eN':
                _, _, N = ev
                for p in range(self.n_pops):
                    # Lock in current size with growth, then reset.
                    self.pop_sizes[p] = N
                    self.growth_rates[p] = 0.0
                    self.growth_start[p] = t
            elif etype == 'en':
                _, _, p, N = ev
                if p < self.n_pops:
                    self.pop_sizes[p] = N
                    self.growth_rates[p] = 0.0
                    self.growth_start[p] = t
            elif etype == 'eG':
                _, _, alpha = ev
                for p in range(self.n_pops):
                    self.pop_sizes[p] = self.size_at(p, t)
                    self.growth_rates[p] = alpha
                    self.growth_start[p] = t
            elif etype == 'eg':
                _, _, p, alpha = ev
                if p < self.n_pops:
                    self.pop_sizes[p] = self.size_at(p, t)
                    self.growth_rates[p] = alpha
                    self.growth_start[p] = t
            elif etype == 'eM':
                _, _, M = ev
                if self.n_pops > 1:
                    per = M / (self.n_pops - 1)
                else:
                    per = 0.0
                for i in range(self.n_pops):
                    for j in range(self.n_pops):
                        if i != j:
                            self.migration_matrix[i][j] = per
            elif etype == 'em':
                _, _, dst, src, M = ev
                if dst < self.n_pops and src < self.n_pops and dst != src:
                    self.migration_matrix[dst][src] = M
            elif etype == 'ej':
                _, _, src, dst = ev
                # Move every active lineage in src to dst.
                for lin in active_lineages:
                    if lin.population == src:
                        lin.population = dst
                # Zero migration to/from src (it no longer exists).
                for k in range(self.n_pops):
                    self.migration_matrix[src][k] = 0.0
                    self.migration_matrix[k][src] = 0.0
            else:
                raise ValueError(f"Unknown demographic event type: {etype}")
        # Drop processed events.
        self.events = new_events
