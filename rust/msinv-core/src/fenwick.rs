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
