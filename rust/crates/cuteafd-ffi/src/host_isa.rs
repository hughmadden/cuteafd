//! Host CPU code variants of the daemon and the native library.
//!
//! Coordinator images carry a portable x86-64-v2 build as `bin/cuteafd` and
//! `lib/libcuteafd_native.so`, plus an x86-64-v4 (AVX-512) build of each in an
//! `x86-64-v4/` directory beside it. Spark images carry one GB10-tuned build and no
//! variant directories. [`startup`] runs first in `main`, before any thread, tracing
//! or CUDA state exists: it picks the variant from `/proc/cpuinfo` (or the
//! `CUTEAFD_HOST_ISA` override) and execs the matching daemon when this binary is
//! not it. [`native_library`] maps every native-library path the daemon dlopens to
//! the same variant, so the daemon and its library always agree. A missing variant
//! falls back to the base build; only the startup line says so.

use anyhow::{bail, Result};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{Once, OnceLock};

/// Override: `auto` (or unset), `x86-64-v2` or `x86-64-v4`.
pub const OVERRIDE_ENV: &str = "CUTEAFD_HOST_ISA";
/// Set by [`startup`] for this process and inherited by children: the selection
/// already made, which also stops a second exec.
pub const SELECTED_ENV: &str = "CUTEAFD_HOST_ISA_SELECTED";

pub const X86_64_V2: &str = "x86-64-v2";
pub const X86_64_V4: &str = "x86-64-v4";
/// The AVX-512 flags `x86-64-v4` code needs (the psABI v4 level; every such CPU has v3).
pub const V4_FLAGS: [&str; 4] = ["avx512f", "avx512bw", "avx512cd", "avx512dq"];
const V4_EXTRA_FLAGS: [&str; 1] = ["avx512vl"];

/// The variant this process runs, and why it was chosen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    /// Variant directory name (`x86-64-v2`, `x86-64-v4`), or `None` where the
    /// architecture ships a single build.
    pub isa: Option<&'static str>,
    pub reason: String,
}

impl Selection {
    fn encode(&self) -> String {
        format!("{}|{}", self.isa.unwrap_or(""), self.reason)
    }

    fn decode(value: &str) -> Option<Self> {
        let (isa, reason) = value.split_once('|')?;
        let isa = match isa {
            "" => None,
            X86_64_V2 => Some(X86_64_V2),
            X86_64_V4 => Some(X86_64_V4),
            _ => return None,
        };
        Some(Self { isa, reason: reason.to_string() })
    }
}

/// The instruction-set level this binary was compiled for.
pub fn compiled_isa() -> &'static str {
    if cfg!(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512cd",
        target_feature = "avx512dq",
        target_feature = "avx512vl"
    )) {
        X86_64_V4
    } else if cfg!(all(
        target_arch = "x86_64",
        target_feature = "sse4.2",
        target_feature = "popcnt",
        target_feature = "cmpxchg16b"
    )) {
        X86_64_V2
    } else if cfg!(target_arch = "x86_64") {
        "x86-64"
    } else if cfg!(all(target_arch = "aarch64", target_feature = "sve2", target_feature = "i8mm")) {
        "aarch64-sve2"
    } else {
        std::env::consts::ARCH
    }
}

fn cpu_flags(cpuinfo: &str) -> Vec<&str> {
    cpuinfo
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            (key.trim() == "flags").then_some(value)
        })
        .map(|value| value.split_whitespace().collect())
        .unwrap_or_default()
}

/// Choose a variant from `/proc/cpuinfo` text and the override. Pure, so tests
/// can feed any CPU. `x86` is false on architectures with a single build.
pub fn select(x86: bool, cpuinfo: Option<&str>, override_value: Option<&str>) -> Result<Selection> {
    let requested = match override_value.map(str::trim) {
        None | Some("") | Some("auto") => None,
        Some(X86_64_V2) => Some(X86_64_V2),
        Some(X86_64_V4) => Some(X86_64_V4),
        Some(other) => bail!("{OVERRIDE_ENV}={other} is not auto, {X86_64_V2} or {X86_64_V4}"),
    };
    if !x86 {
        let reason = match requested {
            None => "single tuned build for this architecture".to_string(),
            Some(isa) => format!("{OVERRIDE_ENV}={isa} ignored: single tuned build for this architecture"),
        };
        return Ok(Selection { isa: None, reason });
    }
    let flags = cpuinfo.map(cpu_flags).unwrap_or_default();
    let missing: Vec<&str> = V4_FLAGS
        .iter()
        .chain(V4_EXTRA_FLAGS.iter())
        .copied()
        .filter(|flag| !flags.contains(flag))
        .collect();
    match requested {
        Some(X86_64_V4) if !missing.is_empty() => bail!(
            "{OVERRIDE_ENV}={X86_64_V4} needs AVX-512 ({}); this CPU lacks {}",
            V4_FLAGS.iter().chain(V4_EXTRA_FLAGS.iter()).copied().collect::<Vec<_>>().join(", "),
            missing.join(", ")
        ),
        Some(isa) => Ok(Selection { isa: Some(isa), reason: format!("{OVERRIDE_ENV}={isa}") }),
        None if cpuinfo.is_none() => {
            Ok(Selection { isa: Some(X86_64_V2), reason: "/proc/cpuinfo unreadable".to_string() })
        }
        None if missing.is_empty() => Ok(Selection {
            isa: Some(X86_64_V4),
            reason: "CPU has avx512f, avx512bw, avx512cd, avx512dq and avx512vl".to_string(),
        }),
        None => Ok(Selection { isa: Some(X86_64_V2), reason: format!("CPU lacks {}", missing.join(", ")) }),
    }
}

fn detect() -> Result<Selection> {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").ok();
    let override_value = std::env::var(OVERRIDE_ENV).ok();
    select(cfg!(target_arch = "x86_64"), cpuinfo.as_deref(), override_value.as_deref())
}

static SELECTION: OnceLock<Selection> = OnceLock::new();

/// This process's selection: the one [`startup`] made, the one a parent passed
/// down, or a fresh detection (a library user that never called `startup`).
pub fn selection() -> &'static Selection {
    SELECTION.get_or_init(|| {
        std::env::var(SELECTED_ENV)
            .ok()
            .and_then(|value| Selection::decode(&value))
            .or_else(|| detect().ok())
            .unwrap_or(Selection { isa: None, reason: "no valid selection".to_string() })
    })
}

/// Where variant `isa` of the file at `path` lives: `<dir>/<isa>/<file>`.
pub fn variant_path(path: &Path, isa: &str) -> Option<PathBuf> {
    Some(path.parent()?.join(isa).join(path.file_name()?))
}

/// What [`startup`] decided, for the startup log line.
#[derive(Debug, Clone)]
pub struct Startup {
    pub selection: Selection,
    /// Set when the selected variant's daemon is not installed and this build
    /// runs instead.
    pub fallback: Option<String>,
}

impl std::fmt::Display for Startup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.selection.isa {
            Some(isa) => write!(f, "host ISA {isa} ({}); daemon built for {}", self.selection.reason, compiled_isa())?,
            None => write!(f, "host ISA {} ({})", compiled_isa(), self.selection.reason)?,
        }
        if let Some(fallback) = &self.fallback {
            write!(f, "; {fallback}")?;
        }
        Ok(())
    }
}

/// Decide whether `exe` should exec a variant for `selection`. Returns the
/// executable to exec, or the fallback note when the variant is absent.
pub fn plan_exec(exe: &Path, selection: &Selection, built_for: &str) -> std::result::Result<Option<PathBuf>, String> {
    let Some(isa) = selection.isa else { return Ok(None) };
    if isa == built_for {
        return Ok(None);
    }
    match variant_path(exe, isa) {
        Some(candidate) if candidate.is_file() => Ok(Some(candidate)),
        // A development build (not one of the image variants) has no siblings to expect.
        _ if built_for != X86_64_V2 && built_for != X86_64_V4 => Ok(None),
        _ => Err(format!("no {isa} daemon beside {}, running the {built_for} build", exe.display())),
    }
}

/// Select the host variant and exec its daemon if this binary is not it. Call
/// first in `main`, before threads, tracing or CUDA; on a successful exec it
/// does not return. An invalid or unsatisfiable override is an error.
pub fn startup() -> Result<Startup> {
    if let Some(previous) = std::env::var(SELECTED_ENV).ok().and_then(|value| Selection::decode(&value)) {
        // Already selected by a parent (or by this binary before an exec): never exec again.
        let fallback = previous
            .isa
            .filter(|isa| *isa != compiled_isa())
            .map(|isa| format!("selected {isa} but running the {} build", compiled_isa()));
        let _ = SELECTION.set(previous.clone());
        return Ok(Startup { selection: previous, fallback });
    }
    let selection = detect()?;
    // Every process from here on (an exec'd variant, spawned helpers) inherits the choice.
    // `startup` runs before any other thread exists, so nothing reads the environment concurrently.
    std::env::set_var(SELECTED_ENV, selection.encode());
    let _ = SELECTION.set(selection.clone());
    let mut fallback = None;
    if let Ok(exe) = std::env::current_exe() {
        match plan_exec(&exe, &selection, compiled_isa()) {
            Ok(Some(target)) => {
                use std::os::unix::process::CommandExt;
                let mut args = std::env::args_os();
                let argv0 = args.next().unwrap_or_else(|| OsString::from("cuteafd"));
                let error = std::process::Command::new(&target).arg0(argv0).args(args).exec();
                fallback = Some(format!("exec {} failed ({error}), running the {} build", target.display(), compiled_isa()));
            }
            Ok(None) => {}
            Err(note) => fallback = Some(note),
        }
    }
    Ok(Startup { selection, fallback })
}

/// The native library to dlopen for `path`: its variant for this process's
/// selection when installed (`<dir>/<isa>/<file>`), else `path`. Every load of
/// `libcuteafd_native.so` goes through here so one process maps one copy.
pub fn native_library(path: &Path) -> PathBuf {
    let resolved = selection()
        .isa
        .and_then(|isa| variant_path(path, isa))
        .filter(|candidate| candidate.is_file())
        .unwrap_or_else(|| path.to_path_buf());
    static LOGGED: Once = Once::new();
    LOGGED.call_once(|| {
        tracing::info!(native_library = %resolved.display(), requested = %path.display(), "native library variant");
    });
    resolved
}

#[cfg(test)]
mod tests {
    use super::*;

    const AVX512: &str = "processor\t: 0\nvendor_id\t: AuthenticAMD\nflags\t\t: fpu sse sse2 avx avx2 fma bmi2 avx512f avx512dq avx512cd avx512bw avx512vl avx512_bf16\nbugs\t\t: none\n";
    const AVX2: &str = "processor\t: 0\nflags\t\t: fpu sse sse2 avx avx2 fma bmi2 movbe\n";
    // AVX-512 foundation without BW/DQ/VL (Knights Landing): not a v4 part.
    const PARTIAL: &str = "flags\t\t: avx2 fma bmi2 avx512f avx512cd avx512er avx512pf\n";

    #[test]
    fn avx512_cpu_selects_v4() {
        let selection = select(true, Some(AVX512), None).unwrap();
        assert_eq!(selection.isa, Some(X86_64_V4));
        assert!(selection.reason.contains("avx512bw"), "{}", selection.reason);
    }

    #[test]
    fn avx2_cpu_selects_the_portable_base_and_names_what_is_missing() {
        let selection = select(true, Some(AVX2), Some("auto")).unwrap();
        assert_eq!(selection.isa, Some(X86_64_V2));
        assert!(selection.reason.contains("avx512f"), "{}", selection.reason);
        let partial = select(true, Some(PARTIAL), None).unwrap();
        assert_eq!(partial.isa, Some(X86_64_V2));
        assert!(partial.reason.contains("avx512bw") && !partial.reason.contains("avx512f,"), "{}", partial.reason);
    }

    #[test]
    fn override_forces_the_base_on_avx512_and_refuses_v4_without_it() {
        let forced = select(true, Some(AVX512), Some(X86_64_V2)).unwrap();
        assert_eq!(forced.isa, Some(X86_64_V2));
        assert_eq!(forced.reason, format!("{OVERRIDE_ENV}={X86_64_V2}"));
        assert_eq!(select(true, Some(AVX512), Some(X86_64_V4)).unwrap().isa, Some(X86_64_V4));
        let error = select(true, Some(AVX2), Some(X86_64_V4)).unwrap_err().to_string();
        assert!(error.contains("lacks avx512f"), "{error}");
        assert!(select(true, Some(AVX512), Some("znver5")).is_err());
    }

    #[test]
    fn unreadable_cpuinfo_is_conservative_and_aarch64_has_one_build() {
        assert_eq!(select(true, None, None).unwrap().isa, Some(X86_64_V2));
        let spark = select(false, Some("Features\t: fp asimd sve2"), None).unwrap();
        assert_eq!(spark.isa, None);
        let ignored = select(false, None, Some(X86_64_V4)).unwrap();
        assert_eq!(ignored.isa, None);
        assert!(ignored.reason.contains("ignored"));
    }

    #[test]
    fn selection_round_trips_through_the_environment_value() {
        for selection in [
            select(true, Some(AVX512), None).unwrap(),
            select(true, Some(AVX2), None).unwrap(),
            select(false, None, None).unwrap(),
        ] {
            assert_eq!(Selection::decode(&selection.encode()), Some(selection));
        }
        assert_eq!(Selection::decode("x86-64-v9|nope"), None);
        assert_eq!(Selection::decode("no separator"), None);
    }

    #[test]
    fn exec_plan_uses_the_variant_directory_and_falls_back_when_absent() {
        struct Dir(PathBuf);
        impl Dir { fn path(&self) -> &Path { &self.0 } }
        impl Drop for Dir { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
        let dir = Dir(std::env::temp_dir().join(format!("cuteafd-host-isa-{}", std::process::id())));
        std::fs::create_dir_all(dir.path()).unwrap();
        let exe = dir.path().join("cuteafd");
        std::fs::write(&exe, b"").unwrap();
        let v4 = select(true, Some(AVX512), None).unwrap();
        let v2 = select(true, Some(AVX2), None).unwrap();
        // Running the v2 base on an AVX-512 host without the variant: note, no exec.
        let note = plan_exec(&exe, &v4, X86_64_V2).unwrap_err();
        assert!(note.contains("running the x86-64-v2 build"), "{note}");
        std::fs::create_dir(dir.path().join(X86_64_V4)).unwrap();
        std::fs::write(dir.path().join(X86_64_V4).join("cuteafd"), b"").unwrap();
        assert_eq!(plan_exec(&exe, &v4, X86_64_V2).unwrap(), Some(dir.path().join("x86-64-v4/cuteafd")));
        // The selected build is already running, or a single-build architecture.
        assert_eq!(plan_exec(&exe, &v4, X86_64_V4).unwrap(), None);
        assert_eq!(plan_exec(&exe, &v2, X86_64_V2).unwrap(), None);
        assert_eq!(plan_exec(&exe, &select(false, None, None).unwrap(), "aarch64-sve2").unwrap(), None);
        // A generic development build execs a variant when one is installed, else runs quietly.
        assert_eq!(plan_exec(&exe, &v4, "x86-64").unwrap(), Some(dir.path().join("x86-64-v4/cuteafd")));
        std::fs::remove_file(dir.path().join(X86_64_V4).join("cuteafd")).unwrap();
        assert_eq!(plan_exec(&exe, &v4, "x86-64").unwrap(), None);
    }

    #[test]
    fn native_variant_path_is_beside_the_library() {
        assert_eq!(
            variant_path(Path::new("/opt/cuteafd/lib/libcuteafd_native.so"), X86_64_V4),
            Some(PathBuf::from("/opt/cuteafd/lib/x86-64-v4/libcuteafd_native.so"))
        );
    }
}
