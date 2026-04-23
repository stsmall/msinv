/// Karyotype class tag for a genomic segment.
///
/// Encodes which inversion arrangement(s) a segment belongs to.
/// Uses a compact bitmask representation: 2 bits per inversion
/// (supports up to 32 inversions in a single u64).
///
///   00 = not inside this inversion (or panmictic after t_inv)
///   01 = Standard arrangement (S)
///   10 = Inverted arrangement (I)
///   11 = reserved
///
/// Panmictic (no inversions, or all t_inv barriers crossed) is
/// represented as all-zero bits.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BranchClass(u64);

/// Per-inversion karyotype.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Karyotype {
    S,
    I,
}

impl BranchClass {
    /// Panmictic — no active inversion barriers.
    pub const PANMICTIC: Self = Self(0);

    /// Maximum number of inversions supported by the bitmask.
    pub const MAX_INVERSIONS: usize = 32;

    /// Create a class tag with a single inversion's karyotype set.
    #[inline]
    pub fn single(inv_id: u16, kary: Karyotype) -> Self {
        debug_assert!((inv_id as usize) < Self::MAX_INVERSIONS);
        let bits = match kary {
            Karyotype::S => 0b01u64,
            Karyotype::I => 0b10u64,
        };
        Self(bits << (inv_id as u64 * 2))
    }

    /// Set the karyotype for one inversion, preserving other inversions'
    /// tags. Returns a new BranchClass.
    #[inline]
    pub fn with_inv(self, inv_id: u16, kary: Karyotype) -> Self {
        let shift = inv_id as u64 * 2;
        let mask = !(0b11u64 << shift);
        let bits = match kary {
            Karyotype::S => 0b01u64,
            Karyotype::I => 0b10u64,
        };
        Self((self.0 & mask) | (bits << shift))
    }

    /// Get the karyotype at a specific inversion, or None if the
    /// inversion is panmictic / not present at this position.
    #[inline]
    pub fn get_inv(self, inv_id: u16) -> Option<Karyotype> {
        let bits = (self.0 >> (inv_id as u64 * 2)) & 0b11;
        match bits {
            0b01 => Some(Karyotype::S),
            0b10 => Some(Karyotype::I),
            _ => None,
        }
    }

    /// Flip S<->I for one inversion, leaving other inversions unchanged.
    #[inline]
    pub fn flip_inv(self, inv_id: u16) -> Self {
        let shift = inv_id as u64 * 2;
        let bits = (self.0 >> shift) & 0b11;
        let flipped = match bits {
            0b01 => 0b10u64, // S -> I
            0b10 => 0b01u64, // I -> S
            _ => bits,       // panmictic or reserved — no-op
        };
        let mask = !(0b11u64 << shift);
        Self((self.0 & mask) | (flipped << shift))
    }

    /// Clear the karyotype for one inversion (set to panmictic for
    /// that inversion). Used when crossing t_inv.
    #[inline]
    pub fn clear_inv(self, inv_id: u16) -> Self {
        let mask = !(0b11u64 << (inv_id as u64 * 2));
        Self(self.0 & mask)
    }

    /// True if the two tags agree on every inversion that BOTH have
    /// an active (non-panmictic) karyotype. This is the coalescence
    /// criterion: two segments can coalesce at a position only if
    /// their classes match at every active inversion.
    #[inline]
    pub fn can_coalesce(self, other: Self) -> bool {
        // XOR gives bits that differ. But we only care about positions
        // where BOTH have an active tag (not 00). The trick: a
        // position matters only if at least one of its 2 bits is set
        // in BOTH self and other.
        let diff = self.0 ^ other.0;
        // Mask of positions where both have active tags:
        let a = self.0;
        let b = other.0;
        // For each 2-bit position: active if (hi | lo) != 0.
        let a_active = (a | (a >> 1)) & 0x5555_5555_5555_5555;
        let b_active = (b | (b >> 1)) & 0x5555_5555_5555_5555;
        let both_active = a_active & b_active;
        // Expand back to 2-bit mask:
        let both_mask = both_active | (both_active << 1);
        // If any differing bit is in a position where both are active,
        // they can't coalesce.
        (diff & both_mask) == 0
    }

    /// True if this tag is fully panmictic (no active inversions).
    #[inline]
    pub fn is_panmictic(self) -> bool {
        self.0 == 0
    }

    /// Raw bitmask (for debugging / serialization).
    #[inline]
    pub fn bits(self) -> u64 {
        self.0
    }

    /// Construct from raw bits. Caller must ensure the bitmask
    /// encodes a valid class (no reserved 0b11 per-inv bit pair).
    /// Used by `apply_coalescence_compound` to union two can-coalesce-
    /// compatible classes.
    #[inline]
    pub fn from_bits_unchecked(bits: u64) -> Self {
        Self(bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panmictic_is_zero() {
        assert!(BranchClass::PANMICTIC.is_panmictic());
        assert_eq!(BranchClass::PANMICTIC.bits(), 0);
    }

    #[test]
    fn single_inversion_s_and_i() {
        let s0 = BranchClass::single(0, Karyotype::S);
        let i0 = BranchClass::single(0, Karyotype::I);
        assert_eq!(s0.get_inv(0), Some(Karyotype::S));
        assert_eq!(i0.get_inv(0), Some(Karyotype::I));
        assert_ne!(s0, i0);
        assert!(!s0.can_coalesce(i0));
        assert!(s0.can_coalesce(s0));
    }

    #[test]
    fn flip_inv() {
        let s0 = BranchClass::single(0, Karyotype::S);
        let flipped = s0.flip_inv(0);
        assert_eq!(flipped.get_inv(0), Some(Karyotype::I));
        assert_eq!(flipped.flip_inv(0), s0);
    }

    #[test]
    fn clear_inv_makes_panmictic() {
        let s0 = BranchClass::single(0, Karyotype::S);
        assert_eq!(s0.clear_inv(0), BranchClass::PANMICTIC);
    }

    #[test]
    fn multi_inv_tags() {
        let tag = BranchClass::PANMICTIC
            .with_inv(0, Karyotype::S)
            .with_inv(1, Karyotype::I)
            .with_inv(2, Karyotype::S);
        assert_eq!(tag.get_inv(0), Some(Karyotype::S));
        assert_eq!(tag.get_inv(1), Some(Karyotype::I));
        assert_eq!(tag.get_inv(2), Some(Karyotype::S));
        assert_eq!(tag.get_inv(3), None); // not set
    }

    #[test]
    fn can_coalesce_multi_inv() {
        let a = BranchClass::PANMICTIC
            .with_inv(0, Karyotype::S)
            .with_inv(1, Karyotype::I);
        let b = BranchClass::PANMICTIC
            .with_inv(0, Karyotype::S)
            .with_inv(1, Karyotype::I);
        let c = BranchClass::PANMICTIC
            .with_inv(0, Karyotype::S)
            .with_inv(1, Karyotype::S); // differs at inv 1
        assert!(a.can_coalesce(b));
        assert!(!a.can_coalesce(c));
    }

    #[test]
    fn panmictic_can_coalesce_with_anything() {
        let p = BranchClass::PANMICTIC;
        let s0 = BranchClass::single(0, Karyotype::S);
        // Panmictic has no active inversions, so no position where
        // both have active tags → can coalesce.
        assert!(p.can_coalesce(s0));
        assert!(p.can_coalesce(p));
    }

    #[test]
    fn high_inv_id() {
        let tag = BranchClass::single(31, Karyotype::I);
        assert_eq!(tag.get_inv(31), Some(Karyotype::I));
        assert_eq!(tag.get_inv(0), None);
    }
}
