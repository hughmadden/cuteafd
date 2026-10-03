//! Device-sized launch policy for the grid-stride FP8 expert-input quantizer.
use std::num::NonZeroUsize;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum SmCountError {
    #[error("device reported zero multiprocessors")]
    Unavailable,
    #[error("--sms must be between 1 and the device's {actual} multiprocessors, got {requested}")]
    InvalidLimit { requested: usize, actual: usize },
}

/// Cached at engine creation, so captured and ordinary launches use the same
/// device geometry without querying CUDA on the request path.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Fp8QuantizeGrid(NonZeroUsize);

impl Fp8QuantizeGrid {
    pub fn new(actual_sms: usize, limit: Option<usize>) -> Result<Self, SmCountError> {
        let actual = NonZeroUsize::new(actual_sms).ok_or(SmCountError::Unavailable)?;
        let selected = match limit {
            Some(requested) => NonZeroUsize::new(requested).filter(|sms| sms.get() <= actual.get())
                .ok_or(SmCountError::InvalidLimit { requested, actual: actual.get() })?,
            None => actual,
        };
        Ok(Self(selected))
    }

    /// Eight 256-column tiles per block, at most four blocks per SM. The
    /// exported kernel strides over any work beyond the chosen grid.
    pub fn blocks(self, rows: usize, hidden: usize) -> usize {
        rows.saturating_mul(hidden.div_ceil(256)).div_ceil(8)
            .clamp(1, self.0.get().saturating_mul(4))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workload_rounds_up_tiles_and_blocks() {
        let grid = Fp8QuantizeGrid::new(188, None).unwrap();
        for (rows, hidden, blocks) in [
            (1, 1, 1), (1, 2048, 1), (1, 2049, 2), (8, 257, 2), (9, 257, 3),
            (64, 7168, 224), (256, 4096, 512), (512, 4096, 752),
        ] {
            assert_eq!(grid.blocks(rows, hidden), blocks, "{rows} rows x {hidden} columns");
        }
    }

    #[test]
    fn grid_follows_the_device_and_covers_grid_stride_work() {
        for sms in [1, 17, 128, 170, 188, 256] {
            let grid = Fp8QuantizeGrid::new(sms, None).unwrap();
            assert_eq!(grid.blocks(1, 256), 1);
            assert_eq!(grid.blocks(8192, 7168), 4 * sms);
            assert_eq!(grid.blocks(usize::MAX, usize::MAX), 4 * sms);
        }
    }

    #[test]
    fn tuning_can_reduce_but_never_exceed_device_capacity() {
        let grid = Fp8QuantizeGrid::new(170, Some(73)).unwrap();
        assert_eq!(grid.blocks(8192, 7168), 292);
        assert_eq!(Fp8QuantizeGrid::new(170, Some(170)).unwrap().blocks(8192, 7168), 680);
        for requested in [0, 171, 188] {
            assert_eq!(Fp8QuantizeGrid::new(170, Some(requested)).unwrap_err(),
                SmCountError::InvalidLimit { requested, actual: 170 });
        }
        assert_eq!(Fp8QuantizeGrid::new(0, None).unwrap_err(), SmCountError::Unavailable);
    }
}
