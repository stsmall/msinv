/// Fenwick tree (binary indexed tree) for O(log n) prefix sums.
///
/// Used to maintain per-position lineage counts for efficient
/// coalescence rate computation. Each "position" in the tree
/// corresponds to a discretized genomic coordinate.

pub struct Fenwick {
    tree: Vec<f64>,
    n: usize,
}

impl Fenwick {
    pub fn new(n: usize) -> Self {
        Self {
            tree: vec![0.0; n + 1],
            n,
        }
    }

    /// Add `delta` to position `i` (0-indexed).
    pub fn update(&mut self, mut i: usize, delta: f64) {
        i += 1; // 1-indexed internally
        while i <= self.n {
            self.tree[i] += delta;
            i += i & i.wrapping_neg();
        }
    }

    /// Prefix sum [0, i] (0-indexed, inclusive).
    pub fn prefix_sum(&self, mut i: usize) -> f64 {
        i += 1;
        let mut s = 0.0;
        while i > 0 {
            s += self.tree[i];
            i -= i & i.wrapping_neg();
        }
        s
    }

    /// Range sum [l, r) (0-indexed, r exclusive).
    pub fn range_sum(&self, l: usize, r: usize) -> f64 {
        if r == 0 || r <= l {
            return 0.0;
        }
        let right = self.prefix_sum(r - 1);
        if l == 0 {
            right
        } else {
            right - self.prefix_sum(l - 1)
        }
    }

    /// Total sum of all elements. O(log n).
    pub fn total(&self) -> f64 {
        if self.n == 0 { return 0.0; }
        self.prefix_sum(self.n - 1)
    }

    /// Find the smallest index i such that prefix_sum(i) > target.
    /// O(log n) binary descent on the tree structure.
    /// Returns n if target >= total (shouldn't happen with valid draws).
    pub fn find(&self, target: f64) -> usize {
        let mut pos = 0usize;
        let mut remaining = target;
        let mut bit_mask = 1usize;
        while bit_mask <= self.n {
            bit_mask <<= 1;
        }
        bit_mask >>= 1;
        while bit_mask > 0 {
            let next = pos + bit_mask;
            if next <= self.n && self.tree[next] <= remaining {
                remaining -= self.tree[next];
                pos = next;
            }
            bit_mask >>= 1;
        }
        pos // 0-indexed result
    }

    /// Add `delta` to all positions in [l, r) using two point updates.
    /// Combined with prefix_sum this gives a "range add, point query"
    /// BIT, but here we use it with range_sum for "range add, range
    /// query" via two Fenwick trees. For simplicity, this single-tree
    /// version just loops — adequate when intervals are few.
    pub fn range_add(&mut self, l: usize, r: usize, delta: f64) {
        for i in l..r {
            self.update(i, delta);
        }
    }

    /// Reset size to `n` and zero all entries, reusing the underlying
    /// allocation whenever possible. Avoids reallocation on hot rebuild
    /// paths where the tree is recreated each iteration.
    pub fn reset(&mut self, n: usize) {
        let need = n + 1;
        if self.tree.len() < need {
            self.tree.resize(need, 0.0);
        }
        for x in self.tree[..need].iter_mut() {
            *x = 0.0;
        }
        self.n = n;
    }

    /// Batch build from a slice of rates in O(n) instead of
    /// n * O(log n) point updates. Uses the standard in-place Fenwick
    /// construction: copy rates into the tree, then propagate each
    /// cell into its parent in a single linear pass.
    pub fn build_from(&mut self, rates: &[f64]) {
        let n = rates.len();
        let need = n + 1;
        if self.tree.len() < need {
            self.tree.resize(need, 0.0);
        }
        // Copy rates to 1-indexed tree slots.
        self.tree[0] = 0.0;
        for i in 0..n {
            self.tree[i + 1] = rates[i];
        }
        // Propagate each node up to its parent in one pass.
        for i in 1..=n {
            let parent = i + (i & i.wrapping_neg());
            if parent <= n {
                let v = self.tree[i];
                self.tree[parent] += v;
            }
        }
        self.n = n;
    }

    /// Current tree size (number of leaves).
    pub fn len(&self) -> usize { self.n }

    /// Grow the tree to at least `new_n` leaves, zero-initialising any
    /// new positions. Existing internal nodes remain valid because
    /// Fenwick aggregating nodes at index k aggregate the range
    /// [k - (k & -k) + 1, k], which does not depend on n — n only
    /// bounds the walk in `update`, `prefix_sum`, and `find`.
    pub fn grow(&mut self, new_n: usize) {
        if new_n <= self.n { return; }
        let need = new_n + 1;
        if self.tree.len() < need {
            self.tree.resize(need, 0.0);
        } else {
            for x in self.tree[self.n + 1..need].iter_mut() {
                *x = 0.0;
            }
        }
        self.n = new_n;
    }

    /// Set the value at position `i` to `new_rate`. Uses prefix-sum
    /// diff to compute the delta and apply via `update`.
    pub fn set(&mut self, i: usize, new_rate: f64) {
        let prev = self.prefix_sum(i) - if i == 0 { 0.0 } else { self.prefix_sum(i - 1) };
        let delta = new_rate - prev;
        if delta != 0.0 {
            self.update(i, delta);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_update_and_query() {
        let mut f = Fenwick::new(10);
        f.update(3, 1.0);
        f.update(7, 2.0);
        assert_eq!(f.prefix_sum(3), 1.0);
        assert_eq!(f.prefix_sum(7), 3.0);
        assert_eq!(f.range_sum(4, 8), 2.0);
    }

    #[test]
    fn find_selects_correct_leaf() {
        let mut f = Fenwick::new(4);
        f.update(0, 1.0);  // leaf 0: rate 1
        f.update(1, 3.0);  // leaf 1: rate 3
        f.update(2, 1.0);  // leaf 2: rate 1
        f.update(3, 5.0);  // leaf 3: rate 5
        // total = 10
        assert_eq!(f.total(), 10.0);
        // find(0.5) → leaf 0 (prefix[0]=1 > 0.5)
        assert_eq!(f.find(0.5), 0);
        // find(1.5) → leaf 1 (prefix[0]=1 <= 1.5, prefix[1]=4 > 1.5)
        assert_eq!(f.find(1.5), 1);
        // find(4.5) → leaf 2 (prefix[1]=4 <= 4.5, prefix[2]=5 > 4.5)
        assert_eq!(f.find(4.5), 2);
        // find(5.5) → leaf 3
        assert_eq!(f.find(5.5), 3);
    }

    #[test]
    fn range_add() {
        let mut f = Fenwick::new(10);
        f.range_add(2, 5, 1.0); // positions 2,3,4 each get +1
        assert_eq!(f.range_sum(0, 2), 0.0);
        assert_eq!(f.range_sum(2, 5), 3.0);
        assert_eq!(f.range_sum(5, 10), 0.0);
    }
}
