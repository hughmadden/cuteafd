//! Resident expert sets shared by placement and coordinator route marking.
use crate::CuteafdError;
fn rejected(reason: String) -> CuteafdError { CuteafdError::ExpertRoutePlanRejected { reason } }
fn bit(words: &[u64], id: usize) -> bool { words[id / 64] & (1u64 << (id % 64)) != 0 }

/// One V4.1 layer's resident experts: six words, one bit per expert.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CoordinatorExpertResidency([u64; 6]);
impl CoordinatorExpertResidency {
    pub const fn empty() -> Self { Self([0; 6]) }
    pub fn from_experts(ids: &[u16]) -> Result<Self, CuteafdError> {
        let mut set = Self::empty();
        for &id in ids {
            let id = usize::from(id);
            if id >= 384 || bit(&set.0, id) {
                return Err(rejected(format!("invalid or repeated coordinator expert {id}")));
            }
            set.0[id / 64] |= 1u64 << (id % 64);
        }
        Ok(set)
    }
    pub fn contains(&self, id: u16) -> bool { id < 384 && bit(&self.0, usize::from(id)) }
}

/// A fixed set of resident routed experts of one layer, as a bitset over expert
/// ids: built once at startup from placement.
#[derive(Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct ExpertResidencySet {
    experts: usize,
    words: Vec<u64>,
}

impl ExpertResidencySet {
    /// No resident expert among `experts` routed experts.
    pub fn empty(experts: usize) -> Self {
        Self {
            experts,
            words: vec![0; experts.div_ceil(64)],
        }
    }

    /// The set of `ids` among `experts` routed experts; out-of-range and
    /// repeated ids are rejected.
    pub fn from_ids(
        experts: usize,
        ids: impl IntoIterator<Item = usize>,
    ) -> Result<Self, CuteafdError> {
        let mut set = Self::empty(experts);
        for id in ids {
            if !set.insert(id)? {
                return Err(rejected(format!("expert {id} named twice in a resident set")));
            }
        }
        Ok(set)
    }

    /// Adds `id`; false when it was already resident.
    pub fn insert(&mut self, id: usize) -> Result<bool, CuteafdError> {
        if id >= self.experts {
            return Err(rejected(format!(
                "resident expert {id} is outside 0..{}",
                self.experts
            )));
        }
        let (word, mask) = (id / 64, 1u64 << (id % 64));
        let fresh = self.words[word] & mask == 0;
        self.words[word] |= mask;
        Ok(fresh)
    }

    pub fn contains(&self, id: usize) -> bool {
        id < self.experts && bit(&self.words, id)
    }

    /// Resident experts.
    pub fn len(&self) -> usize {
        self.words.iter().map(|word| word.count_ones() as usize).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|word| *word == 0)
    }

    /// Routed experts of the layer (the bitset's universe).
    pub fn experts(&self) -> usize {
        self.experts
    }

    /// Bitset words in ascending expert-id order.
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Resident expert ids, ascending.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.experts).filter(|&id| bit(&self.words, id))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn coordinator_residency_bounds_and_word_edges() {
        let set = CoordinatorExpertResidency::from_experts(&[0, 63, 64, 383]).unwrap();
        for id in 0..=384 { assert_eq!(set.contains(id), [0, 63, 64, 383].contains(&id)); }
        assert!(CoordinatorExpertResidency::from_experts(&[384]).is_err());
        assert!(CoordinatorExpertResidency::from_experts(&[1, 1]).is_err());
        assert_eq!(std::mem::size_of::<CoordinatorExpertResidency>(), 48);
    }
    #[test]
    fn general_residency_preserves_tier_masks_without_scheduler() {
        let set = ExpertResidencySet::from_ids(384, [0, 63, 64, 383]).unwrap();
        assert_eq!(set.iter().collect::<Vec<_>>(), [0, 63, 64, 383]);
        assert_eq!(set.len(), 4);
        assert_eq!(set.words().len(), 6);
        assert!(!set.contains(384));
        assert!(ExpertResidencySet::from_ids(384, [384]).is_err());
        assert!(ExpertResidencySet::from_ids(384, [0, 0]).is_err());
    }

}
