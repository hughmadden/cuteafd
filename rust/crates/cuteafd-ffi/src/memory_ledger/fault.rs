//! Failure-only streaming evidence, captured before native constructors clean up.
use super::{state, Context, Space, CONTEXT};
use serde::{ser::{SerializeMap, SerializeSeq}, Serialize, Serializer};
use std::{cell::{Cell, RefCell}, ffi::{c_char, CStr}, fs::{File, OpenOptions}, io::{self, Write}, path::{Path, PathBuf}, sync::{Arc, Mutex, OnceLock}};
use cuteafd_core::memory_layout::Category;

const LIMITS: &[&str] = &[
    "plan: unavailable: accepted-plan registration not captured in this version",
    "arenas: partial: shared coordinator weight arenas only; native/private arenas not registered",
    "inputs: partial: daemon values; launcher provenance and unresolved family defaults not captured in this version",
    "kv: partial: resolved planner capacity/page geometry in inputs where available; live pool occupancy not captured in this version",
    "memory: partial: CUDA physical free/total and current-pool telemetry; opaque library allocations cannot be enumerated",
    "capture: partial: eight preopened reports per process; concurrent capture may retain only a stderr trigger; interrupted/disk-full files may be incomplete",
    "identity: partial: host/role/thread/PID recorded; build/checkpoint/UUID provenance not captured in this version",
    "trigger: partial: raw native API code recorded; translated ABI status remains in typed error chain, not the pre-cleanup JSON",
    "memory: unavailable: NVML/graph/host extensions not captured in this version",
    "workers: unavailable: Spark worker delivery not captured in this version",
    "ledger: partial: instrumented Rust/native backings only; third-party private allocations are opaque",
];

#[repr(C)]
pub(crate) struct NativeFailure {
    pub api: *const c_char, pub domain: *const c_char, pub file: *const c_char,
    pub line: i32, pub code: i32, pub bytes: usize, pub bytes_known: i32, pub handle: usize,
}

#[derive(Debug)]
pub struct FailureReport { pub path: PathBuf, pub summary: String, pub complete: bool, pub raw_code: i32, pub domain: String, pub api: String }

#[derive(Clone, Debug, Serialize)]
pub struct ArenaView { pub name: String, pub offset: usize, pub bytes: usize }
#[derive(Debug, Serialize)]
pub struct Arena {
    pub id: String, pub kind: Category, pub device: i32, pub bytes: usize,
    pub padding_bytes: usize, pub tail_bytes: Option<usize>, pub views: Vec<ArenaView>,
}
#[derive(Default)]
struct Attachments { arenas: std::collections::BTreeMap<u64, Arc<Arena>>, inputs: serde_json::Value }
fn attachments() -> &'static Mutex<Attachments> {
    static DATA: OnceLock<Mutex<Attachments>> = OnceLock::new(); DATA.get_or_init(Default::default)
}
/// Keep on the physical backing, not its short-lived loader: tensor views may
/// outlive DeviceWeightArena and must keep its diagnostic registration alive.
#[derive(Debug)]
pub struct ArenaRegistration(u64);
pub fn register_arena(arena: Arena) -> ArenaRegistration {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    attachments().lock().unwrap_or_else(|e| e.into_inner()).arenas.insert(id, Arc::new(arena));
    ArenaRegistration(id)
}
impl Drop for ArenaRegistration {
    fn drop(&mut self) { attachments().lock().unwrap_or_else(|e| e.into_inner()).arenas.remove(&self.0); }
}
pub fn set_inputs(inputs: serde_json::Value) {
    let mut data = attachments().lock().unwrap_or_else(|e| e.into_inner());
    if !data.inputs.is_object() { data.inputs = serde_json::json!({}); }
    if let Some(inputs) = inputs.as_object() {
        for (name, value) in inputs {
            data.inputs.as_object_mut().unwrap().insert(name.clone(),
                serde_json::json!({"value": value, "source": "daemon"}));
        }
    }
}

struct Service { spools: Vec<(File, PathBuf)>, directory: PathBuf, host: String, role: String, scratch: Box<[u8; 65536]> }
static SERVICE: OnceLock<Mutex<Service>> = OnceLock::new();
thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static PENDING: RefCell<Option<Arc<FailureReport>>> = const { RefCell::new(None) };
}

/// Preopen private storage and fixed host scratch before any native allocation.
/// No report file is published on successful startup unless census was asked for.
pub fn initialize(directory: &Path, host: &str, role: &str) -> io::Result<()> {
    if SERVICE.get().is_some() { return Ok(()); }
    std::fs::create_dir_all(directory)?;
    let mut spools = Vec::new();
    for index in 0..8 {
        let path = directory.join(format!(".memory-spool-{}-{index}", std::process::id()));
        let file = OpenOptions::new().write(true).create_new(true).mode(0o600).open(&path)?;
        spools.push((file, path));
    }
    let service = Service { spools, directory: directory.into(),
        host: sanitize(host), role: sanitize(role), scratch: Box::new([0; 65536]) };
    let _ = SERVICE.set(Mutex::new(service));
    Ok(())
}
use std::os::unix::fs::OpenOptionsExt;
fn sanitize(text: &str) -> String { text.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_') { c } else { '_' }).collect() }

/// Install after loading each native DSO; old native binaries remain usable but
/// explicitly lack the pre-cleanup hook.
pub(crate) fn install(library: &libloading::Library) {
    type Install = unsafe extern "C" fn(extern "C" fn(*const NativeFailure));
    // SAFETY: optional symbol implements the bundled POD callback ABI. The
    // callback is process-static and never unwinds into the native caller.
    let _ = cuda_queries();
    type Events = unsafe extern "C" fn(extern "C" fn(i32, usize, usize, i32));
    // SAFETY: bundled callback takes scalar values only, and stays process-live.
    if let Ok(events) = unsafe { library.get::<Events>(b"cuteafd_set_memory_event_callback") } {
        unsafe { events(event_callback); }
    }
    match unsafe { library.get::<Install>(b"cuteafd_set_memory_failure_callback") } {
        Ok(install) => unsafe { install(callback) },
        Err(_) => tracing::warn!(target: "cuteafd::memory", "failure capture unavailable: native pre-cleanup callback unsupported"),
    }
}
extern "C" fn event_callback(space: i32, address: usize, bytes: usize, released: i32) {
    let _ = std::panic::catch_unwind(|| {
        if released != 0 { super::record_free(address); return; }
        let space = match space { 1 => Space::Pinned, 2 => Space::Managed, 3 => Space::Host, _ => Space::Device };
        let mut device = -1;
        if matches!(space, Space::Device | Space::Managed) { if let Some(q) = cuda_queries() {
            // SAFETY: valid stack output, query does not select a device.
            unsafe { (q.get)(&mut device); }
        } }
        super::record_alloc(space, device, address, bytes);
    });
}
extern "C" fn callback(trigger: *const NativeFailure) {
    if trigger.is_null() || ACTIVE.with(|active| active.replace(true)) { return; }
    let result = std::panic::catch_unwind(|| {
        // SAFETY: native wrapper retains the POD and its literal strings for
        // this synchronous callback; no pointer escapes the callback.
        let trigger = unsafe { &*trigger };
        if let Some(report) = capture(trigger, true) { PENDING.with(|pending| *pending.borrow_mut() = Some(report)); }
    });
    ACTIVE.with(|active| active.set(false));
    if result.is_err() { eprintln!("memory failure capture unavailable: callback panicked; original native error retained"); }
}
pub(crate) fn fallback() -> Option<Arc<FailureReport>> {
    capture(&NativeFailure { api: c"Rust error boundary".as_ptr(), domain: c"unavailable: no native callback".as_ptr(),
        file: c"unavailable".as_ptr(), line: 0, code: 0, bytes: 0, bytes_known: 0, handle: 0 }, false)
}
pub(crate) fn clear_failure() { PENDING.with(|pending| { pending.borrow_mut().take(); }); }
pub(crate) fn take_failure() -> Option<Arc<FailureReport>> { PENDING.with(|pending| pending.borrow_mut().take()) }

struct Buffered<'a> { file: &'a mut File, buffer: &'a mut [u8], used: usize }
impl Write for Buffered<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.used == self.buffer.len() { self.flush()?; }
        let n = bytes.len().min(self.buffer.len() - self.used);
        self.buffer[self.used..self.used+n].copy_from_slice(&bytes[..n]); self.used += n; Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> { self.file.write_all(&self.buffer[..self.used])?; self.used = 0; Ok(()) }
}

struct DisplayValue<T>(T);
impl<T: std::fmt::Display> Serialize for DisplayValue<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> { serializer.collect_str(&self.0) }
}
struct Hex(usize);
impl std::fmt::Display for Hex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "0x{:x}", self.0) }
}
impl Serialize for Hex {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> { serializer.collect_str(self) }
}
#[derive(Serialize)]
struct Trigger<'a> { api: &'a str, domain: &'a str, code: i32, file: &'a str, line: i32,
    requested_bytes: Option<usize>, handle: Hex, scope: &'a str, owner: Option<&'a str>,
    tensor: Option<&'a str>, tensor_stem: Option<&'a str>, format: Option<&'a str> }
fn native_text<'a>(ptr: *const c_char) -> &'a str {
    if ptr.is_null() { return "unavailable"; }
    // SAFETY: used only within the synchronous callback for literal native
    // strings supplied by the instrumented wrappers.
    unsafe { CStr::from_ptr(ptr) }.to_str().unwrap_or("unavailable: non-UTF8 native label")
}
struct LiveRows<'a>(&'a super::State);
impl Serialize for LiveRows<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.0.live.len()))?;
        for (address, live) in &self.0.live {
            #[derive(Serialize)]
            struct Row<'a> { address: Hex, bytes: usize, device: i32, space: &'a str, scope: &'a str,
                category: Category, owner: Option<&'a str>, tensor: Option<&'a str>, tensor_stem: Option<&'a str>,
                format: Option<&'a str>, site: DisplayValue<&'a std::panic::Location<'static>> }
            seq.serialize_element(&Row { address: Hex(*address), bytes: live.bytes,
                device: live.key.device, space: live.key.space.label(), scope: live.key.scope,
                category: Category::of_scope(live.key.scope), owner: live.key.owner, tensor: live.name,
                tensor_stem: live.key.tensor, format: live.key.format, site: DisplayValue(live.site) })?;
        }
        seq.end()
    }
}
struct Growth<'a>(&'a super::State);
impl Serialize for Growth<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("ready", &self.0.ready)?;
        map.serialize_entry("allocated_bytes_since_ready", &self.0.post_ready_bytes)?;
        struct Sites<'a>(&'a super::State);
        impl Serialize for Sites<'_> {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut map = serializer.serialize_map(None)?;
                for (site, bytes) in &self.0.post_ready_sites { map.serialize_entry(&DisplayValue(*site), bytes)?; }
                map.end()
            }
        }
        map.serialize_entry("sites", &Sites(self.0))?; map.end()
    }
}

type GetDevice = unsafe extern "C" fn(*mut i32) -> i32;
type SetDevice = unsafe extern "C" fn(i32) -> i32;
type MemInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> i32;
type GetPool = unsafe extern "C" fn(*mut *mut std::ffi::c_void, i32) -> i32;
type PoolAttribute = unsafe extern "C" fn(*mut std::ffi::c_void, i32, *mut std::ffi::c_void) -> i32;
struct CudaQueries { get: GetDevice, set: SetDevice, info: MemInfo, pool: Option<GetPool>, attribute: Option<PoolAttribute> }
fn cuda_queries() -> Option<&'static CudaQueries> {
    static QUERIES: OnceLock<CudaQueries> = OnceLock::new();
    if let Some(queries) = QUERIES.get() { return Some(queries); }
    let runtime = super::loaded_cuda_runtime()?;
    // SAFETY: symbols are the documented CUDA runtime query signatures and the
    // already-loaded DSO is retained for the lifetime of the process.
    let queries = unsafe { CudaQueries { get: *runtime.get::<GetDevice>(b"cudaGetDevice").ok()?,
        set: *runtime.get::<SetDevice>(b"cudaSetDevice").ok()?, info: *runtime.get::<MemInfo>(b"cudaMemGetInfo").ok()?,
        pool: runtime.get::<GetPool>(b"cudaDeviceGetMemPool").ok().map(|symbol| *symbol),
        attribute: runtime.get::<PoolAttribute>(b"cudaMemPoolGetAttribute").ok().map(|symbol| *symbol) } };
    Some(QUERIES.get_or_init(|| queries))
}
#[derive(Serialize)]
struct PoolObservation { bytes: [Option<u64>; 4], raw_status: [Option<i32>; 4], status: &'static str }
impl PoolObservation {
    fn unavailable() -> Self { Self { bytes: [None; 4], raw_status: [None; 4], status: "unavailable: CUDA current-pool API unsupported, skipped, or failed" } }
}
#[derive(Serialize)]
struct Observation { device: i32, free_bytes: Option<usize>, total_bytes: Option<usize>, status: &'static str, raw_status: Option<i32>,
    cuda_current_pool: PoolObservation }
fn observe(device: i32, sticky: bool) -> Observation {
    let unavailable = |reason| Observation { device, free_bytes: None, total_bytes: None, status: reason, raw_status: None, cuda_current_pool: PoolObservation::unavailable() };
    if sticky { return unavailable("unavailable: context lost or sticky CUDA error; follow-up skipped"); }
    let Some(q) = cuda_queries() else { return unavailable("unavailable: CUDA runtime query API unsupported or not loaded"); };
    let (mut previous, mut free, mut total) = (-1, 0, 0);
    let mut pool = PoolObservation::unavailable();
    // SAFETY: stack outputs are valid; only current/already-ledger-tracked
    // devices are queried. Restore selection even when the query fails.
    unsafe {
        let status = (q.get)(&mut previous);
        if status != 0 { return Observation { raw_status: Some(status), ..unavailable("unavailable: cudaGetDevice failed") }; }
        let status = if device == previous { 0 } else { (q.set)(device) };
        let status = if status == 0 { (q.info)(&mut free, &mut total) } else { status };
        if status == 0 { if let (Some(get_pool), Some(attribute)) = (q.pool, q.attribute) {
            let mut handle = std::ptr::null_mut();
            if get_pool(&mut handle, device) == 0 && !handle.is_null() {
                for (index, id) in [5, 6, 7, 8].into_iter().enumerate() {
                    let mut bytes = 0u64;
                    let status = attribute(handle, id, (&mut bytes as *mut u64).cast());
                    pool.raw_status[index] = Some(status);
                    pool.bytes[index] = (status == 0).then_some(bytes);
                }
                pool.status = if pool.bytes.iter().all(Option::is_some) { "available: reserved_current,reserved_high,used_current,used_high bytes; physical telemetry, not additional ownership" }
                    else { "partial: CUDA pool attribute query failed; null means unknown" };
            }
        } }
        let restored = if device != previous { (q.set)(previous) } else { 0 };
        if status != 0 || restored != 0 { return Observation { raw_status: Some(if status != 0 { status } else { restored }), ..unavailable("unavailable: CUDA memory query or selection restore failed") }; }
    }
    Observation { device, free_bytes: Some(free), total_bytes: Some(total), status: "available: raw cudaMemGetInfo; device-wide, includes other processes; GB10 reclaimable page cache is not allocatable headroom here", raw_status: Some(0), cuda_current_pool: pool }
}

fn capture(raw: &NativeFailure, before_cleanup: bool) -> Option<Arc<FailureReport>> {
    let Some(service) = SERVICE.get() else { eprintln!("memory failure dump unavailable: spool initialization not completed"); return None; };
    let Ok(mut service) = service.try_lock() else { eprintln!("memory failure dump partial: concurrent capture busy; trigger domain={} code={} api={}", native_text(raw.domain), raw.code, native_text(raw.api)); return None; };
    let timestamp = utc_timestamp();
    let path = service.directory.join(format!("memory-failure-{timestamp}-{}-{}-{}-{}.json", service.host, service.role, std::process::id(), sequence()));
    let Some((mut file, spool_path)) = service.spools.pop() else { eprintln!("memory failure dump unavailable: preopened spool exhausted; original error retained"); return None; };
    let sticky = native_text(raw.domain) == "cuda" && raw.code >= 700;
    let mut devices = [-1i32; 32];
    let mut device_count = 0;
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let Service { scratch, host, role, .. } = &mut *service;
        let mut out = Buffered { file: &mut file, buffer: scratch.as_mut_slice(), used: 0 };
        // Core records go first. No CUDA query runs while the ledger is locked.
        {
            let ledger = state().lock().map_err(|_| io::Error::other("ledger lock poisoned"))?;
            let data = attachments().lock().map_err(|_| io::Error::other("attachments lock poisoned"))?;
            out.write_all(b"{\"schema_version\":1,\"identity\":")?;
            #[derive(Serialize)] struct ThreadIdentity<'a> { utc: &'a str, pid: u32, host: &'a str, role: &'a str, thread: u64, pre_native_cleanup: bool }
            // SAFETY: gettid reads the calling Linux thread's identity without allocation.
            let thread = unsafe { libc::syscall(libc::SYS_gettid) } as u64;
            serde_json::to_writer(&mut out, &ThreadIdentity { utc: &timestamp, pid: std::process::id(), host, role, thread, pre_native_cleanup: before_cleanup })?;
            out.write_all(b",\"trigger\":")?;
            CONTEXT.with(|context| -> Result<(), serde_json::Error> {
                let context = context.borrow();
                serde_json::to_writer(&mut out, &trigger(raw, &context))
            })?;
            out.write_all(b",\"live_allocations\":")?; serde_json::to_writer(&mut out, &LiveRows(&ledger))?;
            out.write_all(b",\"post_ready\":")?; serde_json::to_writer(&mut out, &Growth(&ledger))?;
            struct Arenas<'a>(&'a Attachments);
            impl Serialize for Arenas<'_> {
                fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                    let mut seq = serializer.serialize_seq(Some(self.0.arenas.len()))?;
                    for arena in self.0.arenas.values() { seq.serialize_element(arena.as_ref())?; }
                    seq.end()
                }
            }
            out.write_all(b",\"arenas\":")?; serde_json::to_writer(&mut out, &Arenas(&data))?;
            for (_, device) in ledger.current.keys() {
                if *device >= 0 && !devices[..device_count].contains(device) && device_count < devices.len() {
                    devices[device_count] = *device; device_count += 1;
                }
            }
            out.write_all(b",\"inputs\":")?; serde_json::to_writer(&mut out, &data.inputs)?;
            out.write_all(b",\"limits\":")?; serde_json::to_writer(&mut out, LIMITS)?;
        }
        out.write_all(b",\"memory\":")?;
        // Iterate only the tiny device-counter table without holding the
        // allocation lock across runtime APIs.
        let mut serializer = serde_json::Serializer::new(&mut out);
        let mut seq = serializer.serialize_seq(None)?;
        let mut current = -1;
        if !sticky { if let Some(q) = cuda_queries() {
            // SAFETY: runtime writes one valid stack output.
            unsafe { (q.get)(&mut current); }
        } }
        if current >= 0 { seq.serialize_element(&observe(current, sticky))?; }
        for &device in &devices[..device_count] { if device != current { seq.serialize_element(&observe(device, sticky))?; } }
        if current < 0 { seq.serialize_element(&observe(-1, true))?; }
        SerializeSeq::end(seq)?;
        out.write_all(b",\"capture\":{\"complete\":true,\"pre_rust_unwind\":true}}\n")?; out.flush()?;
        Ok(())
    })();
    let complete = result.is_ok();
    if let Err(error) = result { eprintln!("memory failure dump partial: serialization/I/O failure: {error}"); }
    if let Err(error) = std::fs::rename(&spool_path, &path) {
        eprintln!("memory failure dump unavailable: cannot publish spool {}: {error}", spool_path.display()); return None;
    }
    drop(service);
    let summary = summary("failure", sticky);
    tracing::error!(target: "cuteafd::memory", path = %path.display(), api = native_text(raw.api), domain = native_text(raw.domain), code = raw.code, complete, "memory failure dump\n{summary}");
    Some(Arc::new(FailureReport { path, summary, complete, raw_code: raw.code, domain: native_text(raw.domain).into(), api: native_text(raw.api).into() }))
}
fn trigger<'a>(raw: &'a NativeFailure, context: &'a Context) -> Trigger<'a> {
    Trigger { api: native_text(raw.api), domain: native_text(raw.domain), code: raw.code,
        file: native_text(raw.file), line: raw.line, requested_bytes: (raw.bytes_known != 0).then_some(raw.bytes),
        handle: Hex(raw.handle), scope: context.scopes.last().copied().unwrap_or("other"),
        owner: context.owner, tensor: context.name, tensor_stem: context.tensor, format: context.format }
}
fn sequence() -> u64 { static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0); NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) }
fn utc_timestamp() -> String {
    let mut time = 0;
    let mut calendar = std::mem::MaybeUninit::<libc::tm>::uninit();
    let mut text = [0u8; 32];
    // SAFETY: all outputs are valid stack storage; UTC conversion is reentrant.
    unsafe {
        libc::time(&mut time); libc::gmtime_r(&time, calendar.as_mut_ptr());
        libc::strftime(text.as_mut_ptr().cast(), text.len(), c"%Y%m%dT%H%M%SZ".as_ptr(), calendar.as_ptr());
    }
    String::from_utf8_lossy(&text[..text.iter().position(|&b| b == 0).unwrap_or(text.len())]).into_owned()
}

pub fn ready() {
    state().lock().unwrap_or_else(|e| e.into_inner()).ready = true;
    if std::env::var_os("CUTEAFD_STARTUP_MEMORY_PROBE").is_some_and(|value| value == "1") {
        let _ = capture(&NativeFailure { api: c"Ready census".as_ptr(), domain: c"requested census".as_ptr(),
            file: c"Rust Ready boundary".as_ptr(), line: 0, code: 0, bytes: 0, bytes_known: 0, handle: 0 }, false);
    }
}
/// Shared, bounded human-readable view for Ready and failure. Raw planned and
/// observed values remain separate; the signed residual is not a diagnosis.
pub fn compact_summary(stage: &str) -> String { summary(stage, false) }
/// Available byte inputs are numbers; absent optional inputs stay explicitly
/// unresolved rather than being silently represented as zero.

fn summary(stage: &str, sticky: bool) -> String {
    use std::fmt::Write as _;
    let snapshot = super::snapshot();
    let mut out = format!("memory summary stage={stage}\n");
    for device in snapshot.devices() {
        let tracked = snapshot.total(Space::Device, device) + snapshot.total(Space::Managed, device);
        let observed = observe(device, sticky);
        let _ = writeln!(out, "device={device} free={:?} total={:?} tracked={tracked} residual={:?} status={}",
            observed.free_bytes, observed.total_bytes, observed.total_bytes.zip(observed.free_bytes).map(|(t,f)| t as i128-f as i128-tracked as i128), observed.status);
        let mut categories = std::collections::BTreeMap::new();
        for row in &snapshot.rows { if row.key.device == device {
            *categories.entry(Category::of_scope(row.key.scope).label()).or_insert(0usize) += row.bytes;
        } }
        let _ = writeln!(out, "device={device} categories={categories:?}");
    }
    let data = attachments().lock().unwrap_or_else(|e| e.into_inner());
    for arena in data.arenas.values() { let _ = writeln!(out, "arena={} kind={} device={} bytes={} padding={} tail={:?} views={}",
        arena.id, arena.kind.label(), arena.device, arena.bytes, arena.padding_bytes, arena.tail_bytes, arena.views.len()); }
    if let Some(inputs) = data.inputs.as_object() { for (name, value) in inputs { let _ = writeln!(out, "{name} = {} ({})", value["value"], value["source"]); } }
    drop(data);
    let ledger = state().lock().unwrap_or_else(|e| e.into_inner());
    let _ = writeln!(out, "post_ready ready={} allocated_bytes={}", ledger.ready, ledger.post_ready_bytes);
    for reason in LIMITS { let _ = writeln!(out, "{reason}"); }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(api: &'static CStr, domain: &'static CStr, code: i32, bytes: Option<usize>, handle: usize) -> NativeFailure {
        NativeFailure { api: api.as_ptr(), domain: domain.as_ptr(), file: c"fake-native.cc".as_ptr(),
            line: 42, code, bytes: bytes.unwrap_or(0), bytes_known: i32::from(bytes.is_some()), handle }
    }
    fn setup() {
        let root = std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from).unwrap_or_else(std::env::temp_dir)
            .join(format!("memory-failure-tests-{}", std::process::id()));
        initialize(&root, "fixture-host", "rank3").unwrap();
    }
    #[test]
    fn native_failures_preserve_live_backings_raw_status_and_timestamp() {
        let _guard = super::super::test_guard();
        setup();
        let _scope = super::super::scope("workspace/fault-fixture");
        let address = 0x7800_0000_0000;
        event_callback(0, address, 4096, 0);
        callback(&fixture(c"cudaMalloc", c"cuda", 2, Some(8192), 0));
        event_callback(0, address, 0, 1);
        let report = take_failure().unwrap();
        let data: serde_json::Value = serde_json::from_slice(&std::fs::read(&report.path).unwrap()).unwrap();
        assert_eq!(data["trigger"]["code"], 2);
        assert_eq!(data["trigger"]["requested_bytes"], 8192);
        assert!(data["live_allocations"].as_array().unwrap().iter().any(|row| row["address"] == "0x780000000000" && row["bytes"] == 4096));
        assert!(data["capture"]["complete"].as_bool().unwrap());
        let name = report.path.file_name().unwrap().to_str().unwrap();
        let timestamp = name.strip_prefix("memory-failure-").unwrap();
        assert_eq!(&timestamp[8..9], "T"); assert_eq!(&timestamp[15..16], "Z");
        assert!(name.contains("fixture-host-rank3"));
        assert!(data["limits"].as_array().unwrap().iter().any(|v| v.as_str().unwrap().contains("unavailable:")));
        callback(&fixture(c"cublasCreate", c"cublas", 3, None, 0));
        let error = super::super::capture_failure(anyhow::anyhow!("ABI status -3")).context("outer wrapper");
        let saved = error.chain().find_map(|cause| cause.downcast_ref::<super::super::FailureSnapshot>()).unwrap();
        let report = saved.report.as_ref().unwrap();
        assert_eq!(report.raw_code, 3); assert_eq!(report.domain, "cublas");
        let data: serde_json::Value = serde_json::from_slice(&std::fs::read(&report.path).unwrap()).unwrap();
        assert!(data["trigger"]["requested_bytes"].is_null());
        assert!(format!("{error:#}").contains("ABI status -3"));
        std::fs::remove_file(&report.path).unwrap();
    }
    #[test]
    fn graph_failure_is_captured_before_descriptor_cleanup() {
        let _guard = super::super::test_guard(); setup();
        let mut graph_alive = true;
        callback(&fixture(c"cudaGraphInstantiate", c"cuda", 2, None, 0x1234));
        let report = take_failure().unwrap();
        assert!(graph_alive);
        graph_alive = false;
        let data: serde_json::Value = serde_json::from_slice(&std::fs::read(&report.path).unwrap()).unwrap();
        assert_eq!(data["trigger"]["handle"], "0x1234");
        assert_eq!(data["identity"]["pre_native_cleanup"], true);
        assert!(!graph_alive); std::fs::remove_file(&report.path).unwrap();
    }
    #[test]
    fn ready_summary_with_ten_arenas_is_compact_and_registrations_retire() {
        let _guard = super::super::test_guard();
        let arenas: Vec<_> = (0..10).map(|i| register_arena(Arena { id: format!("arena-{i}"),
            kind: Category::Weights, device: 0, bytes: 512, padding_bytes: 0, tail_bytes: None,
            views: vec![ArenaView { name: "weight".into(), offset: 0, bytes: 512 }] })).collect();
        let text = compact_summary("ready");
        assert!(text.lines().count() < 60, "{text}");
        assert_eq!(text.lines().filter(|line| line.starts_with("arena=")).count(), 10);
        drop(arenas);
        assert!(!compact_summary("failure").contains("arena=arena-"));
    }
    #[test]
    fn unsupported_api_is_unknown_not_zero() {
        let value = observe(0, true);
        assert!(value.free_bytes.is_none() && value.total_bytes.is_none());
        assert!(value.status.starts_with("unavailable:"));
    }
}
