"""
Array-based coalescent tree with Numba-accelerated operations.

Tree stored as parallel arrays:
  parent[i]     = parent node index (-1 for root)
  left_child[i] = first child index (-1 for leaf)
  right_sib[i]  = next sibling index (-1 for last child)
  time[i]       = node time
  klass[i]      = karyotype class (0=S, 1=I)
  population[i] = population index
  sample_id[i]  = sample id for leaves (-1 for internal)

All hot-path functions are Numba JIT compiled.
"""

import numpy as np
from numba import njit, int32, float64, int8, boolean
from numba.types import Tuple

NULL = -1
CLASS_S = 0
CLASS_I = 1


# ===================================================================
# Numba-compiled tree operations
# ===================================================================

@njit(cache=True)
def branch_lengths_by_class(parent, time, klass, n):
    """Return (L_S, L_I, t_max) in one pass."""
    L_S = 0.0
    L_I = 0.0
    t_max = 0.0
    for i in range(n):
        if time[i] > t_max:
            t_max = time[i]
        p = parent[i]
        if p >= 0:
            bl = time[p] - time[i]
            if bl > 0:
                if klass[i] == CLASS_S:
                    L_S += bl
                else:
                    L_I += bl
    return L_S, L_I, t_max


@njit(cache=True)
def total_branch_length(parent, time, n):
    """Total branch length of the tree."""
    total = 0.0
    for i in range(n):
        p = parent[i]
        if p >= 0:
            bl = time[p] - time[i]
            if bl > 0:
                total += bl
    return total


@njit(cache=True)
def get_branches_arr(parent, time, klass, n, filter_class):
    """
    Get branch indices and lengths.
    filter_class: -1 for all, 0 for S, 1 for I.
    Returns (indices, lengths, count).
    """
    indices = np.empty(n, dtype=int32)
    lengths = np.empty(n, dtype=float64)
    count = 0
    for i in range(n):
        p = parent[i]
        if p >= 0:
            bl = time[p] - time[i]
            if bl > 0 and (filter_class < 0 or klass[i] == filter_class):
                indices[count] = i
                lengths[count] = bl
                count += 1
    return indices[:count], lengths[:count], count


@njit(cache=True)
def find_root(parent, n):
    """Find the root node (parent == -1)."""
    for i in range(n):
        if parent[i] == NULL:
            # Check this node has children (is actually the root)
            for j in range(n):
                if parent[j] == i:
                    return i
    # Fallback: node with no parent and max time
    best = 0
    for i in range(n):
        if parent[i] == NULL and time[i] > time[best]:
            best = i
    return best


@njit(cache=True)
def get_sibling(parent, left_child, right_sib, node):
    """Get sibling of a node (other child of same parent)."""
    p = parent[node]
    if p == NULL:
        return NULL
    c = left_child[p]
    while c != NULL:
        if c != node:
            return c
        c = right_sib[c]
    return NULL


@njit(cache=True)
def num_children(left_child, right_sib, node):
    """Count children of a node."""
    count = 0
    c = left_child[node]
    while c != NULL:
        count += 1
        c = right_sib[c]
    return count


@njit(cache=True)
def add_child(parent, left_child, right_sib, p, child):
    """Add child to parent node."""
    parent[child] = p
    right_sib[child] = left_child[p]
    left_child[p] = child


@njit(cache=True)
def remove_child(parent, left_child, right_sib, p, child):
    """Remove child from parent node."""
    prev = NULL
    c = left_child[p]
    while c != NULL:
        if c == child:
            if prev == NULL:
                left_child[p] = right_sib[c]
            else:
                right_sib[prev] = right_sib[c]
            parent[child] = NULL
            right_sib[child] = NULL
            return
        prev = c
        c = right_sib[c]


@njit(cache=True)
def get_leaves_below(left_child, right_sib, sample_id, node, n):
    """Get sample IDs of leaves below a node. Uses stack."""
    result = np.empty(n, dtype=int32)
    count = 0
    stack = np.empty(n, dtype=int32)
    stack[0] = node
    sp = 1
    while sp > 0:
        sp -= 1
        cur = stack[sp]
        if sample_id[cur] >= 0:
            result[count] = sample_id[cur]
            count += 1
        c = left_child[cur]
        while c != NULL:
            stack[sp] = c
            sp += 1
            c = right_sib[c]
    return result[:count]


@njit(cache=True)
def find_branches_above_time(parent, time, n, t_cut, filter_class, klass):
    """
    Find branches alive at time t_cut.
    Returns (indices, lengths, count) of branches where
    node.time <= t_cut < parent.time.
    """
    indices = np.empty(n, dtype=int32)
    lengths = np.empty(n, dtype=float64)
    count = 0
    for i in range(n):
        p = parent[i]
        if p >= 0:
            if time[i] <= t_cut < time[p]:
                if filter_class < 0 or klass[i] == filter_class:
                    indices[count] = i
                    lengths[count] = time[p] - max(time[i], t_cut)
                    count += 1
    return indices[:count], lengths[:count], count


# ===================================================================
# Tree class (Python wrapper around arrays)
# ===================================================================

class ArrayTree:
    """Array-based coalescent tree with Numba-accelerated operations."""

    def __init__(self, max_nodes=512):
        self.max_nodes = max_nodes
        self.time = np.zeros(max_nodes, dtype=np.float64)
        self.parent = np.full(max_nodes, NULL, dtype=np.int32)
        self.left_child = np.full(max_nodes, NULL, dtype=np.int32)
        self.right_sib = np.full(max_nodes, NULL, dtype=np.int32)
        self.klass = np.zeros(max_nodes, dtype=np.int8)
        self.population = np.zeros(max_nodes, dtype=np.int8)
        self.sample_id = np.full(max_nodes, NULL, dtype=np.int32)
        self.n = 0
        self.root = NULL

    def _ensure_space(self, needed=1):
        while self.n + needed >= self.max_nodes:
            new_max = self.max_nodes * 2
            self.time = np.concatenate([self.time, np.zeros(self.max_nodes)])
            self.parent = np.concatenate([self.parent, np.full(self.max_nodes, NULL, dtype=np.int32)])
            self.left_child = np.concatenate([self.left_child, np.full(self.max_nodes, NULL, dtype=np.int32)])
            self.right_sib = np.concatenate([self.right_sib, np.full(self.max_nodes, NULL, dtype=np.int32)])
            self.klass = np.concatenate([self.klass, np.zeros(self.max_nodes, dtype=np.int8)])
            self.population = np.concatenate([self.population, np.zeros(self.max_nodes, dtype=np.int8)])
            self.sample_id = np.concatenate([self.sample_id, np.full(self.max_nodes, NULL, dtype=np.int32)])
            self.max_nodes = new_max

    def add_node(self, t, kl=CLASS_S, pop=0, sid=NULL):
        self._ensure_space()
        i = self.n
        self.time[i] = t
        self.parent[i] = NULL
        self.left_child[i] = NULL
        self.right_sib[i] = NULL
        self.klass[i] = kl
        self.population[i] = pop
        self.sample_id[i] = sid
        self.n += 1
        return i

    def add_child_to(self, p, child):
        add_child(self.parent, self.left_child, self.right_sib, p, child)

    def remove_child_from(self, p, child):
        remove_child(self.parent, self.left_child, self.right_sib, p, child)

    def get_branch_lengths(self):
        return branch_lengths_by_class(
            self.parent, self.time, self.klass, self.n)

    def get_total_bl(self):
        return total_branch_length(self.parent, self.time, self.n)

    def get_branches(self, filter_class=-1):
        return get_branches_arr(
            self.parent, self.time, self.klass, self.n, filter_class)

    def get_leaves(self, node):
        return get_leaves_below(
            self.left_child, self.right_sib, self.sample_id, node, self.n)

    def get_sibling(self, node):
        return get_sibling(
            self.parent, self.left_child, self.right_sib, node)

    def num_children(self, node):
        return num_children(self.left_child, self.right_sib, node)

    def find_root_node(self):
        """Find root by walking up from node 0."""
        node = 0
        while self.parent[node] != NULL:
            node = self.parent[node]
        self.root = node
        return node

    def branches_above(self, t_cut, filter_class=-1):
        return find_branches_above_time(
            self.parent, self.time, self.n, t_cut, filter_class, self.klass)
