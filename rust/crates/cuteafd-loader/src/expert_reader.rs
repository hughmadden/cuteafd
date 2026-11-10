//! Loader-owned shard handles and bounded, positional I/O into admitted buffers.
use rayon::prelude::*;
use std::{collections::HashMap, fs::File, io, os::{fd::{AsRawFd, RawFd}, unix::fs::FileExt}, path::{Path, PathBuf}, sync::{Arc, Mutex}};

// A 4 ms fabric round trip at 12 GB/s needs at least 48 MB outstanding.
// Sixteen expert lanes with 8 MiB windows exceed that without touching pinned
// staging. Two 4 MiB requests per lane bound the separate I/O pool at 32.
pub const EXPERT_READ_WINDOW_BYTES: usize = 8 << 20;
const READ_CHUNK_BYTES: usize = 4 << 20;
const READ_WORKERS: usize = 32;

#[derive(Debug, Default)]
pub(crate) struct ExpertReader {
    state: Mutex<ReaderState>,
}

#[derive(Debug, Default)]
struct ReaderState {
    files: HashMap<PathBuf, Arc<File>>,
    pool: Option<Arc<rayon::ThreadPool>>,
}

#[derive(Clone)]
pub(crate) struct ShardFile {
    file: Arc<File>,
    pool: Arc<rayon::ThreadPool>,
}

impl ExpertReader {
    pub(crate) fn open(&self, path: &Path) -> io::Result<ShardFile> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let file = match state.files.get(path) {
            Some(file) => file.clone(),
            None => {
                let file = Arc::new(File::open(path)?);
                state.files.insert(path.to_owned(), file.clone());
                file
            }
        };
        if state.pool.is_none() {
            state.pool = Some(Arc::new(rayon::ThreadPoolBuilder::new().num_threads(READ_WORKERS)
                .thread_name(|i| format!("expert-io-{i}"))
                .build().map_err(io::Error::other)?));
        }
        Ok(ShardFile { file, pool: state.pool.as_ref().unwrap().clone() })
    }

    pub(crate) fn clear(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.files.clear();
        state.pool = None;
    }
}

pub(crate) trait PositionedRead {
    fn read_exact_at(&self, destination: &mut [u8], offset: u64) -> io::Result<()>;
}

impl PositionedRead for File {
    fn read_exact_at(&self, destination: &mut [u8], offset: u64) -> io::Result<()> {
        FileExt::read_exact_at(self, destination, offset)
    }
}

impl ShardFile {
    pub(crate) fn read_exact_at(&self, destination: &mut [u8], offset: u64) -> io::Result<()> {
        PositionedRead::read_exact_at(self, destination, offset)
    }
}

impl PositionedRead for ShardFile {
    fn read_exact_at(&self, destination: &mut [u8], offset: u64) -> io::Result<()> {
        offset.checked_add(u64::try_from(destination.len()).map_err(io::Error::other)?)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "expert read offset overflow"))?;
        if destination.len() <= READ_CHUNK_BYTES {
            return self.pool.install(|| FileExt::read_exact_at(&*self.file, destination, offset));
        }
        // All jobs borrow disjoint ranges; install joins even on an I/O error,
        // before the caller may reuse its staging bank or publish an upload.
        self.pool.install(|| destination.par_chunks_mut(READ_CHUNK_BYTES).enumerate()
            .try_for_each(|(index, chunk)| FileExt::read_exact_at(&*self.file, chunk,
                offset + (index * READ_CHUNK_BYTES) as u64)))
    }
}

impl AsRawFd for ShardFile {
    fn as_raw_fd(&self) -> RawFd { self.file.as_raw_fd() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn shared_handles_parallel_reads_and_release_are_exact() -> io::Result<()> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("shard");
        let payload: Vec<u8> = (0..(17 << 20) + 93).map(|i| (i * 37 + i / 257) as u8).collect();
        File::create(&path)?.write_all(&payload)?;
        let reader = ExpertReader::default();
        let a = reader.open(&path)?;
        let b = reader.open(&path)?;
        assert!(Arc::ptr_eq(&a.file, &b.file));
        std::thread::scope(|scope| {
            let left = scope.spawn(|| {
                let mut bytes = vec![0; (9 << 20) + 13];
                a.read_exact_at(&mut bytes, 71).unwrap();
                assert_eq!(bytes, payload[71..71 + bytes.len()]);
            });
            let right = scope.spawn(|| {
                let mut bytes = vec![0; (8 << 20) + 3];
                b.read_exact_at(&mut bytes, 97).unwrap();
                assert_eq!(bytes, payload[97..97 + bytes.len()]);
            });
            left.join().unwrap();
            right.join().unwrap();
        });
        let mut truncated = vec![0; 18 << 20];
        assert_eq!(a.read_exact_at(&mut truncated, 0).unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
        assert_eq!(a.read_exact_at(&mut [0; 1], u64::MAX).unwrap_err().kind(), io::ErrorKind::InvalidInput);
        drop((a, b));
        reader.clear();
        let state = reader.state.lock().unwrap();
        assert!(state.files.is_empty() && state.pool.is_none());
        Ok(())
    }
}
