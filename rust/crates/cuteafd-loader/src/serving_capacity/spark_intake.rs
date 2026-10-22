//! BF16 coordinator receive planes shared by Spark-backed families.
/// One full-capacity BF16 partial plane per rank and independently queued lane.
pub fn spark_intake_bytes(lanes: u64, ranks: u64, rows: u64, hidden: u64) -> u64 {
    lanes * ranks * rows * hidden * 2
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn glm_flash_intake_charge_is_unchanged() {
        for lanes in [1, 2, 4] { for ranks in [0, 4, 6] { for rows in [64, 4096] {
            assert_eq!(spark_intake_bytes(lanes, ranks, rows, 6144), lanes * ranks * rows * 6144 * 2);
        } } }
        assert_eq!(spark_intake_bytes(1, 6, 4096, 3584), 176_160_768);
    }
}
