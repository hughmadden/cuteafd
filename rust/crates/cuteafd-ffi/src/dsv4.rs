//! DeepSeek V4 coordinator programs: the exported b12x programs behind the
//! generic launcher in `native/src/dsv4_programs.cc`.
//!
//! Callers name each program and pass its pointers in the documented order;
//! [`Dsv4Programs::with_manifest`] checks that order against the exporter's
//! `dsv4_programs.json` so a stale image cannot silently shift arguments.
use crate::NativeLibrary;
use anyhow::{bail, ensure, Context, Result};
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;

#[repr(C)]
struct ProgramInfo {
    name: [u8; 64],
    pointers: u32,
    scalars: u32,
    scalar_kinds: [u8; 16],
}

impl Default for ProgramInfo {
    fn default() -> Self {
        Self { name: [0; 64], pointers: 0, scalars: 0, scalar_kinds: [0; 16] }
    }
}

type CountFn = unsafe extern "C" fn() -> u32;
type InfoFn = unsafe extern "C" fn(u32, *mut ProgramInfo) -> i32;
type LoadFn = unsafe extern "C" fn(u32) -> i32;
type LaunchFn = unsafe extern "C" fn(u32, *const *mut c_void, *const u64, *mut c_void) -> i32;

/// One scalar launch argument.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dsv4Scalar {
    I32(i32),
    I64(i64),
    F32(f32),
}

impl Dsv4Scalar {
    fn kind(self) -> u8 {
        match self {
            Self::I32(_) => b'i',
            Self::I64(_) => b'l',
            Self::F32(_) => b'f',
        }
    }
    fn slot(self) -> u64 {
        match self {
            Self::I32(v) => v as i64 as u64,
            Self::I64(v) => v as u64,
            Self::F32(v) => u64::from(v.to_bits()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Dsv4ProgramSpec {
    pub name: String,
    pub pointers: Vec<String>,
    pub scalar_kinds: Vec<u8>,
    /// Scratch pointer name -> bytes at the program's capacity (manifest only).
    pub scratch: HashMap<String, u64>,
    pub capacity_rows: u32,
}

pub struct Dsv4Programs<'a> {
    library: &'a NativeLibrary,
    load: LoadFn,
    launch: LaunchFn,
    programs: HashMap<String, (u32, Dsv4ProgramSpec)>,
}

impl NativeLibrary {
    /// The DeepSeek V4 programs this library was built with.
    pub fn dsv4_programs(&self) -> Result<Dsv4Programs<'_>> {
        let count = *unsafe { self.lib.get::<CountFn>(b"cuteafd_dsv4_program_count") }
            .context("native library was built without the DeepSeek V4 programs (CUTEAFD_ENABLE_DSV4_AOT)")?;
        let info = unsafe { *self.lib.get::<InfoFn>(b"cuteafd_dsv4_program_info")? };
        let load = unsafe { *self.lib.get::<LoadFn>(b"cuteafd_dsv4_program_load")? };
        let launch = unsafe { *self.lib.get::<LaunchFn>(b"cuteafd_dsv4_program_launch")? };
        let mut programs = HashMap::new();
        // SAFETY: the count and info entry points only read the static table.
        for index in 0..unsafe { count() } {
            let mut raw = ProgramInfo::default();
            let status = unsafe { info(index, &mut raw) };
            ensure!(status == 0, "DeepSeek V4 program {index} info failed with {status}");
            let name = c_string(&raw.name);
            let kinds = c_string(&raw.scalar_kinds).into_bytes();
            ensure!(kinds.len() == raw.scalars as usize, "program {name} scalar table is inconsistent");
            let spec = Dsv4ProgramSpec {
                name: name.clone(),
                // Positional until a manifest names them.
                pointers: (0..raw.pointers).map(|i| format!("#{i}")).collect(),
                scalar_kinds: kinds,
                scratch: HashMap::new(),
                capacity_rows: 0,
            };
            programs.insert(name, (index, spec));
        }
        Ok(Dsv4Programs { library: self, load, launch, programs })
    }
}

impl<'a> Dsv4Programs<'a> {
    /// Attaches pointer names, scratch sizes and capacities from the
    /// exporter manifest, rejecting any disagreement with the native table.
    pub fn with_manifest(mut self, path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let manifest: serde_json::Value = serde_json::from_str(&text)?;
        let entries = manifest["programs"].as_array().context("manifest has no programs")?;
        ensure!(entries.len() == self.programs.len(), "manifest lists {} programs, the library {}",
            entries.len(), self.programs.len());
        for entry in entries {
            let name = entry["name"].as_str().context("program without a name")?;
            let (_, spec) = self.programs.get_mut(name)
                .with_context(|| format!("manifest program {name} is not in the library"))?;
            let pointers: Vec<String> = entry["pointers"].as_array().context("pointers")?
                .iter().map(|p| p["name"].as_str().unwrap_or_default().to_string()).collect();
            ensure!(pointers.len() == spec.pointers.len(), "{name}: manifest and library disagree on pointers");
            spec.pointers = pointers;
            spec.capacity_rows = entry["capacity_rows"].as_u64().unwrap_or(0) as u32;
            if let Some(scratch) = entry["scratch_bytes_at_capacity"].as_object() {
                spec.scratch = scratch.iter().filter_map(|(k, v)| Some((k.clone(), v.as_u64()?))).collect();
            }
        }
        Ok(self)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.programs.keys().map(String::as_str)
    }

    pub fn spec(&self, name: &str) -> Result<&Dsv4ProgramSpec> {
        Ok(&self.programs.get(name).with_context(|| format!("no DeepSeek V4 program {name}"))?.1)
    }

    /// Resolves a program and loads its kernels on the current device.
    pub fn program(&self, name: &str, pointer_names: &[&str]) -> Result<Dsv4Program<'_>> {
        let (index, spec) = self.programs.get(name).with_context(|| format!("no DeepSeek V4 program {name}"))?;
        ensure!(pointer_names.len() == spec.pointers.len(), "{name} takes {} pointers, caller passes {}",
            spec.pointers.len(), pointer_names.len());
        if !spec.pointers[0].starts_with('#') {
            for (expected, given) in spec.pointers.iter().zip(pointer_names) {
                ensure!(expected == given, "{name}: pointer order {:?} != {:?}", spec.pointers, pointer_names);
            }
        }
        // SAFETY: loading reads the program table and loads a CUDA library on
        // the caller's current device.
        let status = unsafe { (self.load)(*index) };
        ensure!(status == 0, "loading {name} failed with CUDA status {status}");
        Ok(Dsv4Program { programs: self, index: *index, spec })
    }

    pub fn library(&self) -> &'a NativeLibrary {
        self.library
    }
}

pub struct Dsv4Program<'a> {
    programs: &'a Dsv4Programs<'a>,
    index: u32,
    spec: &'a Dsv4ProgramSpec,
}

impl Dsv4Program<'_> {
    pub fn spec(&self) -> &Dsv4ProgramSpec {
        self.spec
    }

    /// # Safety
    /// Every pointer must reference live device memory of the documented
    /// shape for `rows`, on the device the program was loaded on, and stay
    /// valid until the stream reaches this launch.
    pub unsafe fn launch(&self, pointers: &[*mut c_void], scalars: &[Dsv4Scalar], stream: *mut c_void) -> Result<()> {
        ensure!(pointers.len() == self.spec.pointers.len(), "{}: pointer count", self.spec.name);
        ensure!(scalars.len() == self.spec.scalar_kinds.len(), "{}: scalar count", self.spec.name);
        for (scalar, kind) in scalars.iter().zip(&self.spec.scalar_kinds) {
            if scalar.kind() != *kind {
                bail!("{}: scalar kinds {:?} do not match {:?}", self.spec.name,
                    String::from_utf8_lossy(&self.spec.scalar_kinds), scalars);
            }
        }
        let slots: Vec<u64> = scalars.iter().map(|s| s.slot()).collect();
        // SAFETY: counts and kinds were checked against the native table; the
        // caller guarantees pointer validity (see the function contract).
        let status = unsafe { (self.programs.launch)(self.index, pointers.as_ptr(), slots.as_ptr(), stream) };
        ensure!(status == 0, "{} launch failed with status {status}", self.spec.name);
        Ok(())
    }
}

fn c_string(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_slots() {
        assert_eq!(Dsv4Scalar::I32(-1).slot(), u64::MAX);
        assert_eq!(Dsv4Scalar::F32(1.0).slot(), 0x3f80_0000);
        assert_eq!(c_string(b"abc\0def"), "abc");
    }
}
