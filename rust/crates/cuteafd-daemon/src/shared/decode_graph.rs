//! Canonical serving decode shapes; diagnostic scoring stays ungraphed.

pub(crate) const ROW_BUCKETS: [usize; 4] = [1, 4, 16, 64];

pub(crate) fn row_bucket(rows: usize) -> usize {
    ROW_BUCKETS.into_iter().find(|&bucket| bucket >= rows).unwrap_or(rows)
}

/// No KV/index/KDA storage writes, no MLA keys, and an independent conv sequence.
/// Families append these sentinels to their own tables and ignore padded logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MaskedRow {
    pub position: i64,
    pub kv_slot: i64,
    pub pool_slot: i64,
    pub state_slot: i32,
    pub seq_first: i32,
    pub cache_length: i32,
}

pub(crate) fn masked_row(row: usize) -> MaskedRow {
    MaskedRow { position: -1, kv_slot: -1, pool_slot: -1, state_slot: -1,
        seq_first: row as i32, cache_length: 0 }
}

#[cfg(test)]
mod tests {
    #[test]
    fn canonical_rows_and_mask() {
        use super::*;
        for (rows, bucket) in [(1, 1), (3, 4), (4, 4), (10, 16), (16, 16), (17, 64), (64, 64)] {
            assert_eq!(row_bucket(rows), bucket);
        }
        let row = masked_row(10);
        assert_eq!((row.position, row.kv_slot, row.pool_slot, row.state_slot, row.cache_length), (-1, -1, -1, -1, 0));
        assert_eq!(row.seq_first, 10);
    }
}
