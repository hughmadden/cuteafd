//! Same-process evidence. No MemAvailable/system residual is called CUDA code.
use anyhow::{Context, Result};
use cuteafd_core::expert_geometry::{SparkAllocationTerm, SparkProcessSample, SparkRuntimeDetail};
use cuteafd_ffi::{memory_ledger::{Snapshot, Space}, NativeLibrary};
use std::time::Instant;

pub(super) struct Census {
    start: Instant,
    samples: Vec<SparkProcessSample>,
    enabled: bool,
}
impl Census {
    pub(super) fn new() -> Self {
        Self { start: Instant::now(), samples: vec![],
            enabled: std::env::var("CUTEAFD_GRAPH_CENSUS").is_ok_and(|v| v == "1") }
    }
    pub(super) fn sample(&mut self, phase: &str, library: Option<&NativeLibrary>) -> Result<()> {
        if !self.enabled { return Ok(()); }
        let text = std::fs::read_to_string("/proc/self/smaps_rollup")
            .context("reading process-attributable Spark census")?;
        let (rss_bytes, pss_bytes, private_bytes) = parse_smaps(&text)?;
        let ledger = cuteafd_ffi::memory_ledger::snapshot();
        let device = library.map(|l| l.cuda_get_device()).transpose()?.unwrap_or(0);
        self.samples.push(SparkProcessSample { phase: phase.into(), pid: std::process::id(),
            monotonic_ns: self.start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            rss_bytes, pss_bytes, private_bytes,
            tracked_device_bytes: ledger.total(Space::Device, device) as u64,
            tracked_pinned_bytes: ledger.total(Space::Pinned, -1) as u64,
            raw_cuda_free_bytes: library.and_then(|l| l.cuda_physical_memory_info().ok()).map(|(free, _)| free as u64),
            method: "same_pid_smaps_rollup; RSS/PSS host mappings; phase delta is measured residual, not isolated CUDA context/modules; quiet-host qualification external".into() });
        tracing::info!(target: "cuteafd::spark_memory", record = %serde_json::to_string(self.samples.last().unwrap())?,
            "Spark process census");
        Ok(())
    }
    pub(super) fn detail(&self, before: &Snapshot, after: &Snapshot, device: i32) -> Result<SparkRuntimeDetail> {
        let mut allocations = Vec::new();
        for row in &after.rows {
            if !((row.key.space == Space::Device && row.key.device == device)
                || row.key.space == Space::Pinned) { continue; }
            let old = before.rows.iter().find(|r| r.key == row.key);
            // Other baseline owners may retire independently; this report names
            // positive retained owners, while aggregate admission reconciles totals.
            let bytes = row.bytes.saturating_sub(old.map_or(0, |r| r.bytes));
            if bytes == 0 { continue; }
            allocations.push(SparkAllocationTerm { space: row.key.space.label().into(),
                scope: row.key.scope.into(), tensor: row.key.tensor.map(str::to_owned),
                format: row.key.format.map(str::to_owned), bytes: bytes as u64,
                allocations: row.allocations.saturating_sub(old.map_or(0, |r| r.allocations)) as u64,
                method: "native allocation ledger owner delta; payload bytes, excludes driver granularity".into() });
        }
        Ok(SparkRuntimeDetail { load_transient: vec![], allocations, process_samples: self.samples.clone(),
            runtime_sample_method: "PostLoad allocatable-free minus device ledger; GB10 system/cache residual; not CUDA context/modules calibration".into(),
            context_modules_qualified: false })
    }
}

fn parse_smaps(text: &str) -> Result<(u64, u64, u64)> {
    let field = |key: &str| -> Result<u64> {
        let line = text.lines().find_map(|line| line.strip_prefix(key))
            .with_context(|| format!("missing smaps {key}"))?;
        let mut tokens = line.split_whitespace();
        let kib: u64 = tokens.next().context("missing smaps number")?.parse()?;
        anyhow::ensure!(tokens.next() == Some("kB"), "unexpected smaps unit");
        kib.checked_mul(1024).context("smaps extent overflow")
    };
    Ok((field("Rss:")?, field("Pss:")?, field("Private_Clean:")?
        .checked_add(field("Private_Dirty:")?).context("private memory overflow")?))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn process_snapshot_requires_explicit_fields_and_units() {
        assert_eq!(parse_smaps("Rss: 12 kB\nPss: 8 kB\nPrivate_Clean: 2 kB\nPrivate_Dirty: 3 kB\n").unwrap(),
            (12288, 8192, 5120));
        assert!(parse_smaps("Rss: 12 MB").is_err());
        assert!(parse_smaps("Rss: 18446744073709551615 kB").is_err());
    }
    #[test]
    fn retained_scope_details_are_not_context_qualification() {
        let census = Census::new();
        let detail = census.detail(&Snapshot::default(), &Snapshot::default(), 0).unwrap();
        assert!(!detail.context_modules_qualified);
        assert!(detail.runtime_sample_method.contains("system/cache residual"));
    }
}

#[cfg(test)]
mod source_order_tests {
    #[test]
    fn census_phases_bracket_owners_without_claiming_module_isolation() {
        let source = include_str!("local.rs");
        let phases = ["census.sample(\"before_media_or_native_library\"",
            "crate::shared::vision::worker::start_encoders(",
            "census.sample(\"after_media_owners_before_expert_bootstrap\"",
            "let library = unsafe { NativeLibrary::load(&config.library) }?;",
            "census.sample(\"after_native_library_before_explicit_context\"",
            "census.sample(\"after_context_existing_media_may_share_primary_context\"",
            "load_weights(&library, &catalog, &config)",
            "census.sample(\"after_expert_load_includes_weights_and_loader_code\"",
            "census.sample(\"before_execution_modules_and_retained_arenas\"",
            "weights.execution(&library, &config, remaining)",
            "census.sample(\"after_execution_modules_and_retained_arenas\"",
            "census.sample(\"serve_owners_before_endpoint_rings\""];
        let mut next = 0;
        for phase in phases {
            next += source[next..].find(phase).unwrap_or_else(|| panic!("missing ordered {phase}")) + phase.len();
        }
    }
}
