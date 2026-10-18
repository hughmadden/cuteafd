//! `cuteafd programs-check`: load a built library's program table with its
//! PROGRAMS.json through the engine's fail-closed loader, as a slot seal or
//! image inventory step. Nothing is launched and no weights are read.
use anyhow::{Context, Result, ensure};
use clap::Args;
use cuteafd_ffi::NativeLibrary;
use std::collections::BTreeSet;
use std::path::PathBuf;

#[derive(Debug, Args)]
pub(crate) struct ProgramsCheckArgs {
    /// The native library whose program table is checked.
    #[arg(long)]
    pub(crate) native_lib: PathBuf,
    /// The exporter manifest installed beside it.
    #[arg(long)]
    pub(crate) manifest: PathBuf,
    /// Require the EXLR8 programs the manifest records for this role
    /// (`families.exlr8`), each present in the library and validated.
    #[arg(long, value_parser = ["coordinator", "spark"])]
    pub(crate) exlr8_role: Option<String>,
}

pub(crate) fn run(args: ProgramsCheckArgs) -> Result<()> {
    // SAFETY: loading a native library runs its initializers; the build output is ours.
    let library = unsafe { NativeLibrary::load(&args.native_lib)? };
    let programs = library.programs()?.with_manifest(&args.manifest)?;
    let names: BTreeSet<&str> = programs.names().collect();
    let exlr8: BTreeSet<&str> = names.iter().copied().filter(|n| n.starts_with("exlr8_")).collect();
    if let Some(role) = &args.exlr8_role {
        let text = std::fs::read_to_string(&args.manifest)?;
        let manifest: serde_json::Value = serde_json::from_str(&text)?;
        let family = &manifest["families"]["exlr8"];
        ensure!(family["role"].as_str() == Some(role.as_str()),
            "{}: EXLR8 programs were exported for role {}, not {role}", args.manifest.display(), family["role"]);
        let recorded: BTreeSet<&str> = family["programs"].as_array().context("manifest records no EXLR8 programs")?
            .iter().map(|p| p.as_str().context("EXLR8 program name")).collect::<Result<_>>()?;
        ensure!(!recorded.is_empty() && recorded == exlr8,
            "EXLR8 programs in the library {exlr8:?} differ from the manifest's {role} set {recorded:?}");
    }
    println!("programs-check: {} programs ({} EXLR8: {}) validated against {}", names.len(), exlr8.len(),
        exlr8.iter().copied().collect::<Vec<_>>().join(","), args.manifest.display());
    Ok(())
}
