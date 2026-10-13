//! Read-only NVML sensors for the throttle report: temperature, T.Limit
//! headroom, power, SM clock and the clock-event (throttle) counters.
//!
//! `libnvidia-ml.so.1` is opened with `dlopen` at runtime (Docker `--gpus`
//! mounts it into the container), so there is no link-time dependency and a
//! host without the driver simply reports no GPUs. Every reading is optional:
//! a field this board does not expose (memory temperature on the RTX PRO 6000,
//! the power limit on GB10) reads `None`, never an error.
use anyhow::{bail, Context, Result};
use cuteafd_core::thermal::{GpuInfo as GpuIdentity, Reading as GpuReading};
use libloading::os::unix::{Library, Symbol};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_uint, c_ulonglong, c_void};

type Device = *mut c_void;
type Ret = c_int;
const SUCCESS: Ret = 0;

/// `nvmlFieldValue_t` (nvml.h): 40 bytes on every 64-bit target.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FieldValue {
    field_id: c_uint,
    scope_id: c_uint,
    timestamp: i64,
    latency_usec: i64,
    value_type: c_int,
    nvml_return: Ret,
    value: u64,
}

/// `nvmlPciInfo_t` (`nvmlDeviceGetPciInfo_v3`).
#[repr(C)]
struct PciInfo {
    bus_id_legacy: [c_char; 16],
    domain: c_uint,
    bus: c_uint,
    device: c_uint,
    pci_device_id: c_uint,
    pci_sub_system_id: c_uint,
    bus_id: [c_char; 32],
}

/// `nvmlUtilization_t`.
#[repr(C)]
#[derive(Default)]
struct Utilization {
    gpu: c_uint,
    memory: c_uint,
}

/// `nvmlMarginTemperature_v1_t`.
#[repr(C)]
struct Margin {
    version: c_uint,
    margin: c_int,
}
const MARGIN_V1: c_uint = std::mem::size_of::<Margin>() as c_uint | (1 << 24);

// Field ids (nvml.h, `NVML_FI_DEV_*`).
const FI_SW_POWER_CAP_NS: c_uint = 74;
const FI_MEMORY_TEMP: c_uint = 82;
const FI_SHUTDOWN_TLIMIT: c_uint = 193;
const FI_SLOWDOWN_TLIMIT: c_uint = 194;
const FI_GPU_MAX_TLIMIT: c_uint = 196;
const FI_SW_THERMAL_NS: c_uint = 269;
const FI_HW_THERMAL_NS: c_uint = 270;
const FI_HW_POWER_BRAKE_NS: c_uint = 271;
/// The slow tier's batch (about 0.8 ms on an RTX PRO 6000: the HW counters are GSP RPCs).
const SLOW_FIELDS: [c_uint; 5] = [FI_SW_POWER_CAP_NS, FI_SW_THERMAL_NS, FI_HW_THERMAL_NS, FI_HW_POWER_BRAKE_NS,
    FI_MEMORY_TEMP];
/// Read once per device.
const LIMIT_FIELDS: [c_uint; 3] = [FI_SLOWDOWN_TLIMIT, FI_SHUTDOWN_TLIMIT, FI_GPU_MAX_TLIMIT];

// `nvmlValueType_t`.
const VALUE_DOUBLE: c_int = 0;
const VALUE_UINT: c_int = 1;
const VALUE_ULONG: c_int = 2;
const VALUE_ULONGLONG: c_int = 3;
const VALUE_SLONGLONG: c_int = 4;
const VALUE_SINT: c_int = 5;

impl FieldValue {
    fn get(&self) -> Option<i64> {
        if self.nvml_return != SUCCESS { return None; }
        let raw = self.value;
        Some(match self.value_type {
            VALUE_UINT => raw as u32 as i64,
            VALUE_SINT => raw as u32 as i32 as i64,
            VALUE_ULONG | VALUE_ULONGLONG | VALUE_SLONGLONG => raw as i64,
            VALUE_DOUBLE => f64::from_bits(raw) as i64,
            _ => return None,
        })
    }
}

type InitFn = unsafe extern "C" fn() -> Ret;
type ErrorStringFn = unsafe extern "C" fn(Ret) -> *const c_char;
type CountFn = unsafe extern "C" fn(*mut c_uint) -> Ret;
type HandleFn = unsafe extern "C" fn(c_uint, *mut Device) -> Ret;
type StringFn = unsafe extern "C" fn(Device, *mut c_char, c_uint) -> Ret;
type PciFn = unsafe extern "C" fn(Device, *mut PciInfo) -> Ret;
type FieldsFn = unsafe extern "C" fn(Device, c_int, *mut FieldValue) -> Ret;
type TemperatureFn = unsafe extern "C" fn(Device, c_int, *mut c_uint) -> Ret;
type MarginFn = unsafe extern "C" fn(Device, *mut Margin) -> Ret;
type ClockFn = unsafe extern "C" fn(Device, c_int, *mut c_uint) -> Ret;
type ReasonsFn = unsafe extern "C" fn(Device, *mut c_ulonglong) -> Ret;
type UtilizationFn = unsafe extern "C" fn(Device, *mut Utilization) -> Ret;
type PowerFn = unsafe extern "C" fn(Device, *mut c_uint) -> Ret;

/// An initialized NVML session. Dropping it calls `nvmlShutdown`.
pub struct Nvml {
    error_string: ErrorStringFn,
    shutdown: InitFn,
    count: CountFn,
    handle: HandleFn,
    uuid: StringFn,
    name: StringFn,
    pci: Option<PciFn>,
    fields: Option<FieldsFn>,
    temperature: Option<TemperatureFn>,
    margin: Option<MarginFn>,
    clock: Option<ClockFn>,
    reasons: Option<ReasonsFn>,
    utilization: Option<UtilizationFn>,
    power: Option<PowerFn>,
    power_limit: Option<PowerFn>,
    // Keeps every function pointer above valid; dropped last.
    _library: Library,
}

// SAFETY: NVML is thread-safe (nvml.h: "all NVML functions are thread-safe");
// the struct holds only function pointers into the library it keeps loaded.
unsafe impl Send for Nvml {}
// SAFETY: as for `Send`: shared calls into NVML from several threads are allowed.
unsafe impl Sync for Nvml {}

/// A device handle bound to its session.
pub struct NvmlDevice<'a> {
    nvml: &'a Nvml,
    handle: Device,
    pub identity: GpuIdentity,
}

fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Option<T> {
    // SAFETY: each call site names the symbol with its exact nvml.h signature as `T`.
    unsafe { library.get::<T>(name).ok().map(|s: Symbol<T>| *s) }
}

fn c_string(buffer: &[c_char]) -> String {
    // SAFETY: NVML writes a NUL-terminated string within the buffer; the buffer is
    // zero-initialized, so it is terminated even after a failed call.
    unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_string_lossy().into_owned()
}

impl Nvml {
    /// Load and initialize NVML; an error means this host has no usable NVIDIA driver.
    pub fn open() -> Result<Self> {
        // SAFETY: loading the driver's NVML library runs only its own initializers.
        let library = unsafe { Library::new("libnvidia-ml.so.1") }.context("dlopen libnvidia-ml.so.1")?;
        let init: InitFn = symbol(&library, b"nvmlInit_v2").context("nvmlInit_v2")?;
        let nvml = Self {
            error_string: symbol(&library, b"nvmlErrorString").context("nvmlErrorString")?,
            shutdown: symbol(&library, b"nvmlShutdown").context("nvmlShutdown")?,
            count: symbol(&library, b"nvmlDeviceGetCount_v2").context("nvmlDeviceGetCount_v2")?,
            handle: symbol(&library, b"nvmlDeviceGetHandleByIndex_v2").context("nvmlDeviceGetHandleByIndex_v2")?,
            uuid: symbol(&library, b"nvmlDeviceGetUUID").context("nvmlDeviceGetUUID")?,
            name: symbol(&library, b"nvmlDeviceGetName").context("nvmlDeviceGetName")?,
            pci: symbol(&library, b"nvmlDeviceGetPciInfo_v3"),
            fields: symbol(&library, b"nvmlDeviceGetFieldValues"),
            temperature: symbol(&library, b"nvmlDeviceGetTemperature"),
            margin: symbol(&library, b"nvmlDeviceGetMarginTemperature"),
            clock: symbol(&library, b"nvmlDeviceGetClockInfo"),
            reasons: symbol(&library, b"nvmlDeviceGetCurrentClocksEventReasons")
                .or_else(|| symbol(&library, b"nvmlDeviceGetCurrentClocksThrottleReasons")),
            utilization: symbol(&library, b"nvmlDeviceGetUtilizationRates"),
            power: symbol(&library, b"nvmlDeviceGetPowerUsage"),
            power_limit: symbol(&library, b"nvmlDeviceGetEnforcedPowerLimit"),
            _library: library,
        };
        // SAFETY: nvmlInit_v2 takes no arguments; a matching nvmlShutdown runs on drop.
        let status = unsafe { init() };
        if status != SUCCESS {
            // Without a successful init there is nothing to shut down.
            let message = nvml.message(status);
            std::mem::forget(nvml);
            bail!("nvmlInit_v2: {message}");
        }
        Ok(nvml)
    }

    fn message(&self, status: Ret) -> String {
        // SAFETY: nvmlErrorString returns a static string for any code (or null).
        let raw = unsafe { (self.error_string)(status) };
        if raw.is_null() { return format!("NVML error {status}"); }
        // SAFETY: a non-null result is a static NUL-terminated string.
        format!("{} ({status})", unsafe { CStr::from_ptr(raw) }.to_string_lossy())
    }

    fn check(&self, what: &str, status: Ret) -> Result<()> {
        if status == SUCCESS { Ok(()) } else { bail!("{what}: {}", self.message(status)) }
    }

    /// Every device this process can see, in NVML index order.
    pub fn devices(&self) -> Result<Vec<NvmlDevice<'_>>> {
        let mut count: c_uint = 0;
        // SAFETY: `count` is a valid out-pointer for the call.
        self.check("nvmlDeviceGetCount_v2", unsafe { (self.count)(&mut count) })?;
        let mut devices = Vec::new();
        for index in 0..count {
            let mut handle: Device = std::ptr::null_mut();
            // SAFETY: `handle` is a valid out-pointer; the index is below the count.
            if unsafe { (self.handle)(index, &mut handle) } != SUCCESS { continue; }
            let mut device = NvmlDevice { nvml: self, handle, identity: GpuIdentity { index, ..GpuIdentity::default() } };
            device.identify();
            devices.push(device);
        }
        Ok(devices)
    }
}

impl Drop for Nvml {
    fn drop(&mut self) {
        // SAFETY: paired with the successful nvmlInit_v2 in `open`.
        unsafe { (self.shutdown)() };
    }
}

impl NvmlDevice<'_> {
    fn string(&self, call: StringFn) -> String {
        let mut buffer = [0 as c_char; 96];
        // SAFETY: the buffer is 96 bytes (the largest NVML UUID/name size) and zeroed.
        unsafe { call(self.handle, buffer.as_mut_ptr(), buffer.len() as c_uint) };
        c_string(&buffer)
    }

    fn field_values<const N: usize>(&self, ids: [c_uint; N]) -> [FieldValue; N] {
        let mut values = ids.map(|field_id| FieldValue { field_id, nvml_return: -1, ..FieldValue::default() });
        if let Some(call) = self.nvml.fields {
            // SAFETY: `values` holds N initialized entries; NVML writes each in place.
            unsafe { call(self.handle, N as c_int, values.as_mut_ptr()) };
        }
        values
    }

    fn identify(&mut self) {
        self.identity.uuid = self.string(self.nvml.uuid);
        self.identity.name = self.string(self.nvml.name);
        if let Some(call) = self.nvml.pci {
            // SAFETY: PciInfo is plain data; an all-zero value is valid.
            let mut pci: PciInfo = unsafe { std::mem::zeroed() };
            // SAFETY: `pci` is a valid out-pointer of the v3 layout.
            if unsafe { call(self.handle, &mut pci) } == SUCCESS { self.identity.pci_bus_id = c_string(&pci.bus_id); }
        }
        let [slowdown, shutdown, gpu_max] = self.field_values(LIMIT_FIELDS);
        self.identity.slowdown_tlimit_c = slowdown.get().map(|v| v as i32);
        self.identity.shutdown_tlimit_c = shutdown.get().map(|v| v as i32);
        self.identity.gpu_max_tlimit_c = gpu_max.get().map(|v| v as i32);
    }

    fn uint(&self, call: Option<PowerFn>) -> Option<u32> {
        let call = call?;
        let mut value: c_uint = 0;
        // SAFETY: `value` is a valid out-pointer.
        (unsafe { call(self.handle, &mut value) } == SUCCESS).then_some(value)
    }

    /// One reading. The fast tier (temperature, SM clock, power, limit, reasons,
    /// utilization) costs about 0.1 ms per device; `slow` adds the cumulative throttle
    /// counters, memory temperature and T.Limit margin (about 2 ms, GSP round trips), which
    /// read `None` when `slow` is false.
    pub fn read(&self, slow: bool) -> GpuReading {
        let n = self.nvml;
        let temperature_c = n.temperature.and_then(|call| {
            let mut value: c_uint = 0;
            // SAFETY: sensor 0 is NVML_TEMPERATURE_GPU; `value` is a valid out-pointer.
            (unsafe { call(self.handle, 0, &mut value) } == SUCCESS).then_some(value as i32)
        });
        let sm_clock_mhz = n.clock.and_then(|call| {
            let mut value: c_uint = 0;
            // SAFETY: clock type 1 is NVML_CLOCK_SM; `value` is a valid out-pointer.
            (unsafe { call(self.handle, 1, &mut value) } == SUCCESS).then_some(value)
        });
        let reasons = n.reasons.and_then(|call| {
            let mut value: c_ulonglong = 0;
            // SAFETY: `value` is a valid out-pointer.
            (unsafe { call(self.handle, &mut value) } == SUCCESS).then_some(value)
        });
        let utilization_pct = n.utilization.and_then(|call| {
            let mut value = Utilization::default();
            // SAFETY: `value` is a valid out-pointer of nvmlUtilization_t's layout.
            (unsafe { call(self.handle, &mut value) } == SUCCESS).then_some(value.gpu)
        });
        let mut reading = GpuReading {
            temperature_c, sm_clock_mhz, reasons, utilization_pct,
            power_mw: self.uint(n.power),
            power_limit_mw: self.uint(n.power_limit),
            ..GpuReading::default()
        };
        if slow {
            let [power_cap, sw_thermal, hw_thermal, brake, memory] = self.field_values(SLOW_FIELDS);
            let unsigned = |v: &FieldValue| v.get().and_then(|v| u64::try_from(v).ok());
            reading.sw_power_cap_ns = unsigned(&power_cap);
            reading.sw_thermal_ns = unsigned(&sw_thermal);
            reading.hw_thermal_ns = unsigned(&hw_thermal);
            reading.hw_power_brake_ns = unsigned(&brake);
            reading.memory_temperature_c = memory.get().map(|v| v as i32);
            reading.tlimit_margin_c = n.margin.and_then(|call| {
                let mut margin = Margin { version: MARGIN_V1, margin: 0 };
                // SAFETY: `margin` carries its struct version; NVML fills `margin`.
                (unsafe { call(self.handle, &mut margin) } == SUCCESS).then_some(margin.margin)
            });
        }
        reading
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_layouts_match_nvml_h() {
        assert_eq!(std::mem::size_of::<FieldValue>(), 40);
        assert_eq!(std::mem::size_of::<PciInfo>(), 68);
        assert_eq!(MARGIN_V1, 8 | (1 << 24));
    }

    #[test]
    fn field_values_decode_by_type_and_status() {
        let field = |value_type, value, nvml_return| FieldValue { value_type, value, nvml_return, ..FieldValue::default() };
        assert_eq!(field(VALUE_SINT, (-2i32) as u32 as u64, 0).get(), Some(-2));
        assert_eq!(field(VALUE_UINT, 325_000, 0).get(), Some(325_000));
        assert_eq!(field(VALUE_ULONGLONG, 234_435_635_904, 0).get(), Some(234_435_635_904));
        assert_eq!(field(VALUE_DOUBLE, 3.0f64.to_bits(), 0).get(), Some(3));
        assert_eq!(field(VALUE_UINT, 7, 3).get(), None, "NOT_SUPPORTED reads as absent");
    }
}

#[cfg(test)]
mod live {
    /// Reads this host's GPUs (`cargo test -p cuteafd-ffi nvml -- --ignored --nocapture`).
    #[test]
    #[ignore = "needs an NVIDIA driver"]
    fn reads_every_visible_gpu() {
        let nvml = super::Nvml::open().expect("NVML");
        let devices = nvml.devices().expect("devices");
        assert!(!devices.is_empty());
        for device in &devices {
            let reading = device.read(true);
            println!("{:?}\n{:?}", device.identity, reading);
            assert!(reading.temperature_c.is_some_and(|t| (1..125).contains(&t)));
            assert!(reading.sm_clock_mhz.is_some() && reading.reasons.is_some());
            assert!(reading.sw_power_cap_ns.is_some());
        }
    }
}
