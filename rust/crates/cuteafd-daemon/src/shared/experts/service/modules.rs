//! Initialize only installed code; keep owners alive through execution teardown.
use super::*;
use crate::shared::experts::{
    exl3::{execution::{Exl3InputFormat, Exl3Modules}, worker::Exl3Worker},
    fp8::{self, Fp8Modules},
};

pub(super) enum WarmModules<'a> {
    Native { _kernels: Vec<cuteafd_ffi::V41ExpertKernel<'a>> },
    Exl3(Option<Vec<Exl3Modules<'a>>>),
    Fp8(Option<Fp8Modules>),
}
impl<'a> WarmModules<'a> {
    pub(super) fn load(library: &'a NativeLibrary, catalog: &OfficialV41Catalog,
        config: &NativeExpertServiceConfig) -> Result<Self> {
        if let Some(tensors) = catalog.fp8() {
            let directory = config.fp8_package.clone().unwrap_or_else(||
                fp8::package_directory(&config.library, config.world, tensors.format()));
            let (directory, _) = fp8::exact_layout(&directory, tensors, config.world, config.rank);
            let bf16 = fp8::bf16_sibling(&directory).filter(|path| path.is_dir());
            return Ok(Self::Fp8(Some(Fp8Modules::load(&directory, bf16.as_deref())?)));
        }
        if let Some(manifest) = catalog.exl3() {
            let directory = config.exl3_directory_for(manifest.decoder_tiers());
            Exl3Worker::validate_exports(&directory, config.capacity, config.rank, config.exl3_schedule)?;
            let modules = cuteafd_core::Exl3RowPolicy::active().capacities(config.capacity as usize)?
                .into_iter().map(|capacity| {
                    // SAFETY: exports were validated above; no arenas or weights are allocated.
                    unsafe { Exl3Modules::load(library,
                        &config.exl3_schedule.directory(&directory, capacity), Exl3InputFormat::Fp8K32) }
                }).collect::<Result<Vec<_>>>()?;
            return Ok(Self::Exl3(Some(modules)));
        }
        let layer = config.selection(0)?;
        let nvfp4 = catalog.nvfp4().is_some();
        let compiled = config.compiled_capacity()?;
        let info = layer.select_info(library, compiled, nvfp4)?;
        let mut capacities = vec![compiled];
        if matches!(info.role, 1 | 5 | 6 | 7) {
            if compiled > 1 { capacities.push(1); }
            if compiled > 80 { capacities.push(80); }
        }
        let kernels = capacities.into_iter().map(|rows| layer.select_kernel(library, rows, nvfp4))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::Native { _kernels: kernels })
    }

    pub(super) fn take_fp8(&mut self) -> Result<Fp8Modules> {
        match self {
            Self::Fp8(modules) => modules.take().context("FP8 modules already consumed"),
            _ => anyhow::bail!("FP8 module owner/backend mismatch"),
        }
    }
    pub(super) fn take_exl3(&mut self) -> Result<Vec<Exl3Modules<'a>>> {
        match self {
            Self::Exl3(modules) => modules.take().context("EXL3 modules already consumed"),
            _ => anyhow::bail!("EXL3 module owner/backend mismatch"),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn owner_binds_primary_context_without_census_before_all_backends() {
        let worker = include_str!("local.rs");
        let context = worker.split("census.sample(\"before_context\", None)?;").nth(1).unwrap();
        let first_statement = context.lines().map(str::trim)
            .find(|line| !line.is_empty() && !line.starts_with("//")).unwrap();
        assert_eq!(first_statement, "library.cuda_set_device(library.cuda_get_device()?)?;");
        let modules = context.find("let mut modules = match super::modules::WarmModules::load").unwrap();
        let execution = context.find("weights.execution(&library").unwrap();
        // Warm-up and execution stay on the synchronous GPU owner, not a
        // separate census thread with a different thread-local context.
        let owner = &context[..execution];
        assert!(modules < execution);
        for handoff in [".spawn(", "spawn_blocking(", ".await"] {
            assert!(!owner.contains(handoff), "CUDA owner handed off through {handoff}");
        }
    }

    #[test]
    fn module_phase_has_no_weight_or_arena_allocation() {
        let source = include_str!("modules.rs").split("#[cfg(test)]").next().unwrap();
        for forbidden in ["DeviceAllocation", "HostAllocation", "load_weights", "cuda_stream_create",
            "Exl3Workspace::new", "Fp8Experts::load", "weights.execution"] {
            assert!(!source.contains(forbidden), "module phase contains {forbidden}");
        }
        let worker = include_str!("local.rs");
        let modules = worker.find("let mut modules = match super::modules::WarmModules::load").unwrap();
        let weights = worker.find("load_weights(&library").unwrap();
        let execution = worker.find("weights.execution(&library").unwrap();
        assert!(modules < weights && weights < execution);
    }
}
