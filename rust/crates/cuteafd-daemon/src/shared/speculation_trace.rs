//! One bounded, buffered append owner per file, shared by every producer.
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::Path;
use std::sync::{mpsc, Arc, Mutex, OnceLock, Weak};
use std::thread::JoinHandle;

// Field expressions only capture owned data. JSON trees are built by the writer.
macro_rules! record {
    ($($field:ident: $value:expr),* $(,)?) => {{
        #[allow(non_camel_case_types)]
        #[derive(serde::Serialize)]
        struct Record<$($field),*> { $($field: $field),* }
        Record { $($field: $value),* }
    }};
}
pub(crate) use record;

/// Small family-specific metadata whose JSON formatting also belongs to the writer.
pub(crate) struct Deferred(Box<dyn Fn() -> serde_json::Value + Send>);
impl Deferred {
    pub(crate) fn new(build: impl Fn() -> serde_json::Value + Send + 'static) -> Self { Self(Box::new(build)) }
}
impl serde::Serialize for Deferred {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        (self.0)().serialize(serializer)
    }
}

type Job = Box<dyn FnOnce() -> io::Result<serde_json::Value> + Send>;
// At five records per round this buffers >800 C16 rounds, without unbounded growth.
const CAPACITY: usize = 4096;
type FileId = (u64, u64);
type Writers = HashMap<FileId, Weak<Writer>>;
static WRITERS: OnceLock<Mutex<Writers>> = OnceLock::new();

#[derive(Clone)]
pub(crate) struct Trace(Arc<Writer>);
struct Writer {
    sender: Mutex<Option<mpsc::SyncSender<Job>>>,
    thread: Mutex<Option<JoinHandle<io::Result<()>>>>,
}

impl Writer {
    fn shutdown(&self) -> io::Result<()> {
        // Serialize shutdowns as well as sends; never join while holding sender.
        let mut thread = self.thread.lock().unwrap_or_else(|p| p.into_inner());
        self.sender.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let Some(thread) = thread.take() {
            thread.join().map_err(|_| io::Error::other("speculation trace writer panicked"))??;
        }
        Ok(())
    }
}
impl Drop for Writer {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown() { tracing::warn!(%error, "speculation trace drain failed"); }
    }
}

pub(crate) fn shutdown() {
    let writers: Vec<_> = WRITERS.get_or_init(Mutex::default).lock().unwrap_or_else(|p| p.into_inner())
        .values().filter_map(Weak::upgrade).collect();
    for writer in writers {
        if let Err(error) = writer.shutdown() { tracing::warn!(%error, "speculation trace drain failed"); }
    }
}

impl Trace {
    pub(crate) fn open() -> Result<Option<Self>> {
        std::env::var("CUTEAFD_SPECULATION_TRACE").ok().filter(|path| !path.is_empty())
            .map(|path| Self::open_path(Path::new(&path)).with_context(|| format!("CUTEAFD_SPECULATION_TRACE {path}")))
            .transpose()
    }

    fn open_path(path: &Path) -> io::Result<Self> {
        let mut writers = WRITERS.get_or_init(Mutex::default).lock()
            .map_err(|_| io::Error::other("speculation trace registry poisoned"))?;
        // Identity also unifies symlinks and hard links. Only this worker writes.
        let file = OpenOptions::new().create(true).append(true).mode(0o600).open(path)?;
        let metadata = file.metadata()?;
        let id = (metadata.dev(), metadata.ino());
        if let Some(writer) = writers.get(&id).and_then(Weak::upgrade) { return Ok(Self(writer)); }
        writers.retain(|_, writer| writer.strong_count() != 0);
        let (sender, receiver) = mpsc::sync_channel::<Job>(CAPACITY);
        let thread = std::thread::Builder::new().name("speculation-trace".into()).spawn(move || {
            write_records(file, receiver)
        })?;
        let writer = Arc::new(Writer { sender: Mutex::new(Some(sender)), thread: Mutex::new(Some(thread)) });
        writers.insert(id, Arc::downgrade(&writer));
        Ok(Self(writer))
    }

    pub(crate) fn write<T: serde::Serialize + Send + 'static>(&self, record: T) -> io::Result<()> {
        self.defer(move || serde_json::to_value(record).map_err(io::Error::other))
    }

    pub(crate) fn defer(&self, record: impl FnOnce() -> io::Result<serde_json::Value> + Send + 'static) -> io::Result<()> {
        self.0.sender.lock().map_err(|_| io::Error::other("speculation trace sender poisoned"))?
            .as_ref().ok_or_else(|| io::Error::other("speculation trace closed"))?
            .send(Box::new(record)).map_err(|_| io::Error::other("speculation trace writer stopped"))
    }
}

fn write_records(file: File, receiver: mpsc::Receiver<Job>) -> io::Result<()> {
    let mut writer = BufWriter::with_capacity(256 << 10, file);
    let mut error = None;
    // Continue draining after an error, releasing producers and reporting failure on join.
    for record in receiver {
        if error.is_some() { continue; }
        let result = record().and_then(|value| {
            serde_json::to_writer(&mut writer, &value).map_err(io::Error::other)?;
            writer.write_all(b"\n")
        });
        if let Err(failed) = result { tracing::warn!(%failed, "speculation trace write failed"); error = Some(failed); }
    }
    writer.flush()?;
    error.map_or(Ok(()), Err)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "CPU speculation trace round microbenchmark; run explicitly with --nocapture"]
    fn round_capture_microseconds() {
        // Five JSON records matching MiMo's measured 57/16/2.5/2.2/0.3 KB mix.
        let payloads: Vec<_> = [57_000, 16_000, 2_500, 2_200, 300].into_iter()
            .map(|bytes| vec![12345u32; bytes / 6]).collect();
        let repeats = 200;
        let start = std::time::Instant::now();
        for _ in 0..repeats {
            for (kind, tokens) in payloads.iter().enumerate() {
                let value = serde_json::json!({"kind": kind, "tokens": tokens});
                std::hint::black_box(serde_json::to_vec(&value).unwrap());
            }
        }
        let before = start.elapsed().as_secs_f64() * 1e6 / repeats as f64;
        let directory = tempfile::tempdir().unwrap();
        let trace = Trace::open_path(&directory.path().join("trace.jsonl")).unwrap();
        let mut cost = std::time::Duration::ZERO;
        for _ in 0..repeats {
            let start = std::time::Instant::now();
            for (kind, tokens) in payloads.iter().enumerate() {
                trace.write(record! { kind: kind, tokens: tokens.clone() }).unwrap();
            }
            cost += start.elapsed();
            // Outside timing: do not include artificial writer saturation in caller cost.
            let (done, wait) = mpsc::channel();
            trace.defer(move || { done.send(()).unwrap(); Ok(serde_json::Value::Null) }).unwrap();
            wait.recv().unwrap();
        }
        drop(trace);
        println!("speculation trace 78 KB / 5 records: legacy {:.1} us -> typed copy+send {:.1} us ({} iterations, no legacy filesystem)",
            before, cost.as_secs_f64() * 1e6 / repeats as f64, repeats);
    }

    #[test]
    fn full_channel_blocks_instead_of_losing_records() {
        let directory = tempfile::tempdir().unwrap();
        let trace = Trace::open_path(&directory.path().join("trace.jsonl")).unwrap();
        let (entered, started) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        trace.defer(move || { entered.send(()).unwrap(); wait.recv().unwrap(); Ok(serde_json::Value::Null) }).unwrap();
        started.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        for n in 0..CAPACITY { trace.write(record! { record: n }).unwrap(); }
        let (done, completed) = mpsc::channel();
        let other = trace.clone();
        let producer = std::thread::spawn(move || { other.write(record! { last: true }).unwrap(); done.send(()).unwrap(); });
        assert!(completed.recv_timeout(std::time::Duration::from_millis(10)).is_err());
        release.send(()).unwrap();
        completed.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        producer.join().unwrap();
        drop(trace);
        let text = std::fs::read_to_string(directory.path().join("trace.jsonl")).unwrap();
        assert_eq!(text.lines().count(), CAPACITY + 2);
        assert_eq!(text.lines().last(), Some("{\"last\":true}"));
    }

    #[test]
    fn concurrent_producers_keep_order_and_every_complete_record_on_drop() {
        const PRODUCERS: usize = 8;
        const RECORDS: usize = 128;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let owner = Trace::open_path(&path).unwrap();
        let symlink = directory.path().join("symlink.jsonl");
        std::os::unix::fs::symlink(&path, &symlink).unwrap();
        let hardlink = directory.path().join("hardlink.jsonl");
        std::fs::hard_link(&path, &hardlink).unwrap();
        std::thread::scope(|scope| {
            for producer in 0..PRODUCERS {
                let path = match producer % 3 { 0 => &path, 1 => &symlink, _ => &hardlink };
                let owner = &owner;
                scope.spawn(move || {
                    let writer = Trace::open_path(path).unwrap();
                    assert!(Arc::ptr_eq(&owner.0, &writer.0));
                    for record in 0..RECORDS {
                        writer.write(record! { producer: producer, record: record,
                            routes: vec![producer; if producer % 2 == 0 { 12_000 } else { 4 }],
                            text: "escaped\nnewline\"quote" }).unwrap();
                    }
                });
            }
        });
        drop(owner);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.last(), Some(&b'\n'));
        let mut next = [0; PRODUCERS];
        for line in std::str::from_utf8(&bytes).unwrap().lines() {
            let value: serde_json::Value = serde_json::from_str(line).unwrap();
            let producer = value["producer"].as_u64().unwrap() as usize;
            assert_eq!(value["record"], next[producer]);
            next[producer] += 1;
            assert_eq!(value["routes"].as_array().unwrap().len(), if producer % 2 == 0 { 12_000 } else { 4 });
            assert_eq!(value["text"], "escaped\nnewline\"quote");
        }
        assert_eq!(next, [RECORDS; PRODUCERS]);
    }

    #[test]
    fn writer_builds_json_off_producer_and_explicit_shutdown_joins() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let trace = Trace::open_path(&path).unwrap();
        let producer = std::thread::current().id();
        trace.defer(move || {
            assert_ne!(std::thread::current().id(), producer);
            Ok(serde_json::json!({"last": true}))
        }).unwrap();
        trace.0.shutdown().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"{\"last\":true}\n");
        assert!(trace.write(record! { last: false }).is_err());
    }
}
