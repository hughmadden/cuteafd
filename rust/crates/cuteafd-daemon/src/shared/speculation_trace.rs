//! One append owner per speculation trace file, shared by every producer.
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock, Weak};

#[derive(Clone)]
pub(crate) struct Trace(Arc<Mutex<File>>);

type FileId = (u64, u64);
type Writers = HashMap<FileId, Weak<Mutex<File>>>;

impl Trace {
    pub(crate) fn open() -> Result<Option<Self>> {
        std::env::var("CUTEAFD_SPECULATION_TRACE").ok().filter(|path| !path.is_empty())
            .map(|path| Self::open_path(Path::new(&path)).with_context(|| format!("CUTEAFD_SPECULATION_TRACE {path}")))
            .transpose()
    }

    fn open_path(path: &Path) -> io::Result<Self> {
        static WRITERS: OnceLock<Mutex<Writers>> = OnceLock::new();
        let mut writers = WRITERS.get_or_init(Mutex::default).lock()
            .map_err(|_| io::Error::other("speculation trace registry poisoned"))?;
        // Identity, rather than path spelling, also unifies symlinks and hard links.
        let file = OpenOptions::new().create(true).append(true).mode(0o600).open(path)?;
        let metadata = file.metadata()?;
        let id = (metadata.dev(), metadata.ino());
        if let Some(writer) = writers.get(&id).and_then(Weak::upgrade) {
            return Ok(Self(writer));
        }
        writers.retain(|_, writer| writer.strong_count() != 0);
        let writer = Arc::new(Mutex::new(file));
        writers.insert(id, Arc::downgrade(&writer));
        Ok(Self(writer))
    }

    pub(crate) fn write(&self, value: &serde_json::Value) -> io::Result<()> {
        let line = record_line(value)?;
        let mut writer = self.0.lock()
            .map_err(|_| io::Error::other("speculation trace writer poisoned"))?;
        // Keep partial-write retries under the same lock. File has no userspace
        // buffer: completed records survive process exit without flush or fsync.
        write_line(&mut *writer, &line)
    }
}

fn record_line(value: &serde_json::Value) -> io::Result<Vec<u8>> {
    let mut line = serde_json::to_vec(value)?;
    line.push(b'\n');
    Ok(line)
}

fn write_line(writer: &mut impl Write, line: &[u8]) -> io::Result<()> {
    writer.write_all(line)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    #[test]
    fn concurrent_producers_keep_every_complete_record() {
        const PRODUCERS: usize = 8;
        const RECORDS: usize = 128;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let owner = Trace::open_path(&path).unwrap();
        let symlink = directory.path().join("symlink.jsonl");
        std::os::unix::fs::symlink(&path, &symlink).unwrap();
        let hardlink = directory.path().join("hardlink.jsonl");
        std::fs::hard_link(&path, &hardlink).unwrap();
        let barrier = Barrier::new(PRODUCERS);
        std::thread::scope(|scope| {
            for producer in 0..PRODUCERS {
                let path = match producer % 3 { 0 => &path, 1 => &symlink, _ => &hardlink };
                let barrier = &barrier;
                let owner = &owner;
                scope.spawn(move || {
                    let writer = Trace::open_path(path).unwrap();
                    assert!(Arc::ptr_eq(&owner.0, &writer.0));
                    barrier.wait();
                    for record in 0..RECORDS {
                        // Large route arrays exceed BufWriter's default 8 KiB capacity;
                        // short confidence/cycle records exercise competing producers.
                        let routes = vec![producer; if producer % 2 == 0 { 12_000 } else { 4 }];
                        writer.write(&serde_json::json!({
                            "type": if producer % 2 == 0 { "draft_round" } else { "confidence" },
                            "producer": producer, "record": record, "routes": routes,
                            "text": "escaped\nnewline\"quote"
                        })).unwrap();
                    }
                });
            }
        });
        // Read before dropping the owner: no shutdown/flush is needed for visibility.
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.last(), Some(&b'\n'));
        let text = std::str::from_utf8(&bytes).unwrap();
        let mut seen = vec![vec![false; RECORDS]; PRODUCERS];
        let mut count = 0;
        for line in text.lines() {
            let value: serde_json::Value = serde_json::from_str(line).unwrap();
            let producer = value["producer"].as_u64().unwrap() as usize;
            let record = value["record"].as_u64().unwrap() as usize;
            assert!(!seen[producer][record], "duplicate producer {producer} record {record}");
            seen[producer][record] = true;
            let routes = value["routes"].as_array().unwrap();
            assert_eq!(routes.len(), if producer % 2 == 0 { 12_000 } else { 4 });
            assert!(routes.iter().all(|route| route.as_u64() == Some(producer as u64)));
            assert_eq!(value["text"], "escaped\nnewline\"quote");
            count += 1;
        }
        assert_eq!(count, PRODUCERS * RECORDS);
        assert!(seen.iter().flatten().all(|&record| record));
    }

    #[test]
    fn complete_lines_prevent_old_buffer_flush_interleaving() {
        // Model a second append handle writing between two write syscalls.
        // The former writeln!(BufWriter<File>, "{value}") splits large JSON
        // records at 8 KiB; no concurrent thread is needed to force the race.
        #[derive(Default)]
        struct CompetingAppend { bytes: Vec<u8>, inserted: bool }
        impl Write for CompetingAppend {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.bytes.extend_from_slice(bytes);
                if !self.inserted {
                    self.bytes.extend_from_slice(b"{\"type\":\"confidence\"}\n");
                    self.inserted = true;
                }
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> { Ok(()) }
        }
        fn valid_lines(bytes: &[u8]) -> usize {
            bytes.split(|&byte| byte == b'\n').filter(|line| !line.is_empty())
                .filter(|line| serde_json::from_slice::<serde_json::Value>(line).is_ok()).count()
        }
        let value = serde_json::json!({"type": "draft_round", "routes": vec![42; 12_000]});
        let mut old = std::io::BufWriter::new(CompetingAppend::default());
        writeln!(old, "{value}").unwrap();
        old.flush().unwrap();
        assert_eq!(valid_lines(&old.get_ref().bytes), 0, "old fragmented writes must reproduce corrupt JSONL");
        let mut fixed = CompetingAppend::default();
        write_line(&mut fixed, &record_line(&value).unwrap()).unwrap();
        assert_eq!(valid_lines(&fixed.bytes), 2);
    }

    #[test]
    fn records_are_one_serialized_write_and_errors_propagate() {
        struct Writer { calls: usize, bytes: Vec<u8> }
        impl Write for Writer {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.calls += 1;
                self.bytes.extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> { panic!("trace must not flush") }
        }
        let value = serde_json::json!({"type": "draft_round", "routes": vec![42; 12_000]});
        let line = record_line(&value).unwrap();
        let mut writer = Writer { calls: 0, bytes: Vec::new() };
        write_line(&mut writer, &line).unwrap();
        assert_eq!(writer.calls, 1);
        assert_eq!(writer.bytes.last(), Some(&b'\n'));
        assert_eq!(serde_json::from_slice::<serde_json::Value>(&writer.bytes).unwrap(), value);

        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::new(io::ErrorKind::BrokenPipe, "trace sink closed"))
            }
            fn flush(&mut self) -> io::Result<()> { panic!("trace must not flush") }
        }
        assert_eq!(write_line(&mut Broken, &line).unwrap_err().kind(), io::ErrorKind::BrokenPipe);
    }
}
