"""Block-counting chain over lineages labelled by (population, state).

A state is a sorted tuple of (cell, block) pairs, one per lineage, where
cell = 3 * population + k (k: 0 = S, 1 = I, 2 = P, panmictic after t_inv)
and block is the number of sampled chromosomes the lineage subtends
(Hobolth, Siri-Jegousse & Bladt 2019). Every state with a single lineage is
merged into one absorbing state, index 0.
"""

from __future__ import annotations

from collections import Counter, deque

import numpy as np
import scipy.sparse as sp

S, I, P = 0, 1, 2
K_OF = {"S": S, "I": I}
ABSORBED = 0


def cell(pop: int, k: int) -> int:
    return 3 * pop + k


def pop_of(c: int) -> int:
    return c // 3


def k_of(c: int) -> int:
    return c % 3


def canon(lineages) -> tuple:
    return tuple(sorted(lineages))


class Chain:
    """Enumerated chain for one sample.

    ``sample``: list of (pop, 'S'|'I'), one per sampled chromosome.
    ``mig_pairs``: (src, dst) with positive migration in some epoch; a lineage
    moves src -> dst backward in time.
    ``joins``: (src, dst) population joins (ej) at any time.
    ``switching``: whether the site's switching coefficient is positive.
    """

    def __init__(self, sample, n_pops, mig_pairs, joins, switching, origin,
                 origin_pop, max_states=None):
        self.n = len(sample)
        if self.n < 2:
            raise ValueError("need at least two sampled chromosomes")
        if origin not in ("barrier", "single"):
            raise ValueError(f"origin must be 'barrier' or 'single', got {origin!r}")
        if origin == "single" and origin_pop is None:
            raise ValueError("origin='single' requires origin_pop")
        self.n_pops = n_pops
        self._mig_pairs = sorted(mig_pairs)
        self._joins = sorted(set(joins))
        self._switching = switching
        self._origin = origin
        self._origin_pop = origin_pop
        self._max_states = max_states
        self.states = [None]
        self.index = {}
        self._edges = {}
        self._maps = {}
        start = canon((cell(p, K_OF[s]), 1) for p, s in sample)
        queue = deque()
        self.init = self._target(start, queue)
        while queue:
            s = queue.popleft()
            i = self.index[s]
            for key, t, mult in self._moves(s):
                self._edges.setdefault(key, []).append(
                    (i, self._target(t, queue), mult)
                )
            for name, t in self._boundary_images(s):
                self._maps.setdefault(name, []).append((i, self._target(t, queue)))
        self.size = len(self.states)
        self._build_arrays()

    def _target(self, state: tuple, queue) -> int:
        if len(state) == 1:
            return ABSORBED
        if state not in self.index:
            self.index[state] = len(self.states)
            self.states.append(state)
            queue.append(state)
            if self._max_states is not None and len(self.states) > self._max_states:
                raise ValueError(
                    f"tier 0 state space exceeds max_states={self._max_states}; "
                    "use the simulator (tier 1/2) for this sample"
                )
        return self.index[state]

    def _moves(self, s: tuple):
        cnt = Counter(s)
        types = sorted(cnt)
        lin = list(s)
        for ai, a in enumerate(types):
            for b in types[ai:]:
                if a[0] != b[0]:
                    continue
                mult = cnt[a] * (cnt[a] - 1) // 2 if a == b else cnt[a] * cnt[b]
                if mult == 0:
                    continue
                rest = list(lin)
                rest.remove(a)
                rest.remove(b)
                rest.append((a[0], a[1] + b[1]))
                yield ("coal", pop_of(a[0]), k_of(a[0])), canon(rest), mult
        for a in types:
            c, blk = a
            d, k = pop_of(c), k_of(c)
            for src, dst in self._mig_pairs:
                if src != d:
                    continue
                rest = list(lin)
                rest.remove(a)
                rest.append((cell(dst, k), blk))
                yield ("mig", d, dst, k), canon(rest), cnt[a]
            if self._switching and k != P:
                rest = list(lin)
                rest.remove(a)
                rest.append((cell(d, 1 - k), blk))
                yield ("sw", d, k), canon(rest), cnt[a]

    def _boundary_images(self, s: tuple):
        for src, dst in self._joins:
            if any(pop_of(c) == src for c, _ in s):
                yield ("join", src, dst), canon(
                    (cell(dst, k_of(c)), b) if pop_of(c) == src else (c, b)
                    for c, b in s
                )
        if any(k_of(c) != P for c, _ in s):
            yield "t_inv", self._inv_image(s)

    def _inv_image(self, s: tuple) -> tuple:
        if self._origin == "single":
            merged = [b for c, b in s if k_of(c) == I]
            rest = [(cell(pop_of(c), P), b) for c, b in s if k_of(c) != I]
            if merged:
                rest.append((cell(self._origin_pop, P), sum(merged)))
            return canon(rest)
        return canon((cell(pop_of(c), P), b) for c, b in s)

    def _build_arrays(self) -> None:
        n = self.size
        self.AT = {}
        for key, edges in self._edges.items():
            e = np.asarray(edges, dtype=float)
            a = sp.coo_matrix(
                (e[:, 2], (e[:, 0].astype(int), e[:, 1].astype(int))), shape=(n, n)
            ).tocsr()
            a = a - sp.diags(np.asarray(a.sum(axis=1)).ravel())
            self.AT[key] = a.T.tocsr()
        self.BT = {}
        for name, pairs in self._maps.items():
            to = np.arange(n)
            for i, j in pairs:
                to[i] = j
            b = sp.csr_matrix((np.ones(n), (np.arange(n), to)), shape=(n, n))
            self.BT[name] = b.T.tocsr()
        self.rewards = np.zeros((self.n - 1, n))
        self.cell_mask = {}
        for idx in range(1, n):
            for c, b in self.states[idx]:
                self.rewards[b - 1, idx] += 1.0
                if c not in self.cell_mask:
                    self.cell_mask[c] = np.zeros(n, dtype=bool)
                self.cell_mask[c][idx] = True
        self.transient = np.ones(n, dtype=bool)
        self.transient[ABSORBED] = False
        # States after t_inv (all lineages panmictic): the only ones that can
        # hold mass in the final constant-rate phase.
        self.post = np.array(
            [s is not None and all(k_of(c) == P for c, _ in s) for s in self.states]
        )
