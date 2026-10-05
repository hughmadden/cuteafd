//! Row-at-a-time safetensors output; memory is bounded by one vocabulary row.
use super::LogSoftmax;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::PathBuf;

#[derive(Debug)]
pub(super) struct RowDump {
    path: PathBuf,
    files: Option<DumpFiles>,
    failed: bool,
}

#[derive(Debug)]
struct DumpFiles {
    manifest: File,
    vocab: usize,
    rows: usize,
}

impl RowDump {
    pub(super) fn new(path: PathBuf) -> Self {
        Self { path, files: None, failed: false }
    }

    pub(super) fn write(&mut self, position: usize, logits: &[f32], softmax: &LogSoftmax) -> io::Result<()> {
        if self.failed {
            return Err(io::Error::other("row dump stopped after an earlier error"));
        }
        let result = self.write_inner(position, logits, softmax);
        self.failed = result.is_err();
        result
    }

    fn write_inner(&mut self, position: usize, logits: &[f32], softmax: &LogSoftmax) -> io::Result<()> {
        if logits.is_empty() || !softmax.0.is_finite() || logits.iter().any(|v| v.is_nan() || *v == f32::INFINITY) {
            return Err(io::Error::other("row dump requires a nonempty normalizable vocabulary"));
        }
        if self.files.is_none() {
            // Parents may be new on the server's shared mount. Only the leaf is
            // exclusive: concurrent probes must never mix rows or overwrite arms.
            if let Some(parent) = self.path.parent().filter(|p| !p.as_os_str().is_empty()) {
                fs::create_dir_all(parent)?;
            }
            fs::create_dir(&self.path)?;
            let manifest = match OpenOptions::new().write(true).create_new(true).open(self.path.join("manifest.jsonl")) {
                Ok(file) => file,
                Err(error) => {
                    fs::remove_dir(&self.path)?;
                    return Err(error);
                }
            };
            self.files = Some(DumpFiles { manifest, vocab: logits.len(), rows: 0 });
        }
        let files = self.files.as_mut().expect("dump initialized");
        if logits.len() != files.vocab {
            return Err(io::Error::other(format!("row dump vocabulary changed: expected {}, got {}", files.vocab, logits.len())));
        }
        let name = format!("row-{:08}.safetensors", files.rows);
        let temporary = self.path.join(format!("{name}.partial"));
        let destination = self.path.join(&name);
        let extent = logits.len().checked_mul(4).ok_or_else(|| io::Error::other("vocabulary byte extent overflow"))?;
        let mut header = serde_json::to_vec(&serde_json::json!({
            "log_probs": { "dtype": "F32", "shape": [logits.len()], "data_offsets": [0, extent] }
        }))?;
        header.resize(header.len().next_multiple_of(8), b' ');
        let mut line = serde_json::to_vec(&serde_json::json!({
            "position": position, "vocab_size": logits.len(), "file": name,
            "tensor": "log_probs", "dtype": "F32", "byte_order": "little"
        }))?;
        line.push(b'\n');
        let manifest_end = files.manifest.stream_position()?;
        let result = (|| {
            let mut file = OpenOptions::new().write(true).create_new(true).open(&temporary)?;
            file.write_all(&(header.len() as u64).to_le_bytes())?;
            file.write_all(&header)?;
            let mut bytes = Vec::with_capacity(extent);
            for &value in logits {
                bytes.extend_from_slice(&softmax.at(value).to_le_bytes());
            }
            file.write_all(&bytes)?;
            file.flush()?;
            drop(file);
            fs::rename(&temporary, &destination)?;
            files.manifest.write_all(&line)?;
            files.manifest.flush()
        })();
        if let Err(error) = result {
            // Roll back both sides of a failed row publication. Earlier complete
            // rows remain usable; no partial JSON line or unindexed tensor stays.
            let mut cleanup_error = None;
            for path in [&temporary, &destination] {
                if let Err(e) = fs::remove_file(path) {
                    if e.kind() != io::ErrorKind::NotFound {
                        cleanup_error.get_or_insert(e);
                    }
                }
            }
            if let Err(e) = files.manifest.set_len(manifest_end)
                .and_then(|()| files.manifest.seek(SeekFrom::Start(manifest_end)).map(|_| ())) {
                cleanup_error.get_or_insert(e);
            }
            return Err(cleanup_error.unwrap_or(error));
        }
        files.rows += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_manifest_write_removes_the_unindexed_tensor() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("rows");
        let mut dump = RowDump::new(path.clone());
        let logits = [1.0, 2.0];
        let softmax = LogSoftmax::new(&logits);
        dump.write(1, &logits, &softmax).unwrap();
        // A read-only descriptor deterministically injects manifest IO failure.
        dump.files.as_mut().unwrap().manifest = File::open(path.join("manifest.jsonl")).unwrap();
        assert!(dump.write(2, &logits, &softmax).is_err());
        assert!(!path.join("row-00000001.safetensors").exists());
        assert!(!path.join("row-00000001.safetensors.partial").exists());
        assert_eq!(fs::read_dir(&path).unwrap().count(), 2);
        assert_eq!(fs::read_to_string(path.join("manifest.jsonl")).unwrap().lines().count(), 1);
    }

    #[test]
    fn nested_dump_parents_are_created_without_overwriting_the_leaf() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("arm/decode/window-000");
        let logits = [1.0, 2.0];
        let softmax = LogSoftmax::new(&logits);
        let mut dump = RowDump::new(path.clone());
        dump.write(1, &logits, &softmax).unwrap();
        let original = fs::read(path.join("manifest.jsonl")).unwrap();
        let mut duplicate = RowDump::new(path.clone());
        assert_eq!(duplicate.write(2, &logits, &softmax).unwrap_err().kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(fs::read(path.join("manifest.jsonl")).unwrap(), original);
        assert!(!path.join("row-00000001.safetensors").exists());
    }

    #[test]
    fn invalid_rows_do_not_create_a_dump() {
        for logits in [vec![], vec![f32::NAN], vec![f32::INFINITY, 0.0], vec![f32::NEG_INFINITY]] {
            let temporary = tempfile::tempdir().unwrap();
            let path = temporary.path().join("rows");
            let mut dump = RowDump::new(path.clone());
            assert!(dump.write(1, &logits, &LogSoftmax::new(&logits)).is_err());
            assert!(!path.exists());
        }
    }
}
