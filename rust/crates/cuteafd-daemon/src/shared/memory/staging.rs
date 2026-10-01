//! A thread's reusable host buffer for weight loads: one positioned read
//! fills it and one copy uploads it. Its pages stay resident from tensor to
//! tensor, so a load faults them in once instead of once per tensor (a fresh
//! `vec![0; n]` per tensor faults every page of every tensor).
use std::cell::RefCell;

thread_local! {
    static STAGING: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static STAGING_SOURCE: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Runs `body` with this thread's staging buffer at `bytes` bytes (contents
/// unspecified until written). Not reentrant.
pub(crate) fn with_staging<T>(bytes: usize, body: impl FnOnce(&mut [u8]) -> T) -> T {
    STAGING.with(|cell| {
        let mut buffer = cell.borrow_mut();
        if buffer.len() < bytes {
            buffer.resize(bytes, 0);
        }
        body(&mut buffer[..bytes])
    })
}

/// [`with_staging`] plus a second buffer of `source_bytes` (a relayout's source).
pub(crate) fn with_staging_pair<T>(bytes: usize, source_bytes: usize, body: impl FnOnce(&mut [u8], &mut [u8]) -> T)
    -> T {
    STAGING_SOURCE.with(|cell| {
        let mut source = cell.borrow_mut();
        if source.len() < source_bytes {
            source.resize(source_bytes, 0);
        }
        with_staging(bytes, |buffer| body(buffer, &mut source[..source_bytes]))
    })
}

/// Frees this thread's staging buffers (after a model load).
pub(crate) fn release_staging() {
    STAGING.with(|cell| *cell.borrow_mut() = Vec::new());
    STAGING_SOURCE.with(|cell| *cell.borrow_mut() = Vec::new());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staging_grows_keeps_contents_and_releases() {
        with_staging(16, |b| b.copy_from_slice(&[7; 16]));
        assert_eq!(with_staging(8, |b| b.to_vec()), vec![7; 8]);
        assert_eq!(with_staging(32, |b| b.len()), 32);
        release_staging();
        assert_eq!(with_staging(4, |b| b.to_vec()), vec![0; 4]);
    }
}
