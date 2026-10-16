//! Independent same-process maps and system headroom; neither alone is CUDA attribution.
use anyhow::{ensure, Context, Result};
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
    pub(super) fn enabled(&self) -> bool { self.enabled }

    pub(super) fn sample(&mut self, phase: &str, library: Option<&NativeLibrary>) -> Result<()> {
        if !self.enabled { return Ok(()); }
        let text = std::fs::read_to_string("/proc/self/smaps")
            .context("reading process-attributable Spark census")?;
        let maps = parse_smaps(&text)?;
        let meminfo = std::fs::read_to_string("/proc/meminfo")?;
        let ledger = cuteafd_ffi::memory_ledger::snapshot();
        let device = library.map(|l| l.cuda_get_device()).transpose()?.unwrap_or(0);
        self.samples.push(SparkProcessSample { phase: phase.into(), pid: std::process::id(),
            monotonic_ns: self.start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            rss_bytes: maps.rss, pss_bytes: maps.pss, private_bytes: maps.private,
            cuda_mapping_pss_bytes: maps.cuda_pss, anonymous_pss_bytes: maps.anon_pss,
            file_pss_bytes: maps.file_pss,
            system_available_bytes: kib_field(&meminfo, "MemAvailable:")?,
            system_total_bytes: kib_field(&meminfo, "MemTotal:")?,
            allocatable_bytes: library.map(|l| l.cuda_allocatable_memory_info()
                .map(|(free, _)| free as u64)).transpose()?,
            tracked_device_bytes: ledger.total(Space::Device, device) as u64,
            tracked_pinned_bytes: ledger.total(Space::Pinned, -1) as u64,
            raw_cuda_free_bytes: library.and_then(|l| l.cuda_physical_memory_info().ok()).map(|(free, _)| free as u64),
            method: "same_pid_smaps (CUDA device mapping PSS, anonymous PSS, file PSS); independent MemAvailable/system and cgroup-bounded admission headroom; raw CUDA diagnostic only; quiet-host qualification external".into() });
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
            let bytes = row.bytes.saturating_sub(old.map_or(0, |r| r.bytes));
            if bytes == 0 { continue; }
            allocations.push(SparkAllocationTerm { space: row.key.space.label().into(),
                scope: row.key.scope.into(), tensor: row.key.tensor.map(str::to_owned),
                format: row.key.format.map(str::to_owned), bytes: bytes as u64,
                allocations: row.allocations.saturating_sub(old.map_or(0, |r| r.allocations)) as u64,
                method: "native allocation ledger owner delta; payload bytes, excludes driver granularity".into() });
        }
        let verdict = witness_verdict(&self.samples);
        tracing::info!(target: "cuteafd::spark_memory", %verdict,
            "Spark dual-witness census qualification (quiet-host evidence required externally)");
        Ok(SparkRuntimeDetail { load_transient: vec![], allocations, process_samples: self.samples.clone(),
            runtime_sample_method: format!("{verdict}; PostLoad system/cache remainder is measured_residual, not CUDA context/modules calibration"),
            // A process cannot certify that the entire host was quiet. The offline
            // qualified proof constructor also requires that external evidence.
            context_modules_qualified: false })
    }
}

fn kib_field(text: &str, key: &str) -> Result<u64> {
    let mut values = text.lines().filter_map(|line| line.strip_prefix(key));
    let line = values.next().with_context(|| format!("missing {key}"))?;
    ensure!(values.next().is_none(), "duplicate {key}");
    let mut tokens = line.split_whitespace();
    let number = tokens.next().context("missing memory number")?;
    ensure!(!number.is_empty() && number.bytes().all(|b| b.is_ascii_digit()), "invalid memory number");
    let kib: u64 = number.parse()?;
    ensure!(tokens.next() == Some("kB") && tokens.next().is_none(), "unexpected memory unit");
    kib.checked_mul(1024).context("memory extent overflow")
}

#[derive(Default, Debug, PartialEq, Eq)]
struct Maps { rss: u64, pss: u64, private: u64, cuda_pss: u64, anon_pss: u64, file_pss: u64 }
fn parse_smaps(text: &str) -> Result<Maps> {
    let mut maps = Maps::default();
    let mut block = String::new();
    let mut header = String::new();
    let mut add = |header: &str, block: &str| -> Result<()> {
        if header.is_empty() { return Ok(()); }
        let rss = kib_field(block, "Rss:")?;
        let pss = kib_field(block, "Pss:")?;
        let private = kib_field(block, "Private_Clean:")?.checked_add(kib_field(block, "Private_Dirty:")?)
            .context("private memory overflow")?;
        let path = header.split_whitespace().skip(5).collect::<Vec<_>>().join(" ");
        let target = if path.starts_with("/dev/nvidia") || path.starts_with("/dev/nvmap") {
            &mut maps.cuda_pss
        } else if path.starts_with('/') { &mut maps.file_pss } else { &mut maps.anon_pss };
        *target = target.checked_add(pss).context("mapping PSS overflow")?;
        maps.rss = maps.rss.checked_add(rss).context("RSS overflow")?;
        maps.pss = maps.pss.checked_add(pss).context("PSS overflow")?;
        maps.private = maps.private.checked_add(private).context("private memory overflow")?;
        Ok(())
    };
    for line in text.lines() {
        let range = line.split_whitespace().next().unwrap_or("");
        let is_header = range.split_once('-').is_some_and(|(a,b)| !a.is_empty() && !b.is_empty()
            && a.bytes().chain(b.bytes()).all(|c| c.is_ascii_hexdigit()));
        if is_header {
            add(&header, &block)?;
            header = line.into(); block.clear();
        } else { block.push_str(line); block.push('\n'); }
    }
    add(&header, &block)?;
    ensure!(!header.is_empty(), "smaps has no mappings");
    Ok(maps)
}

fn witness_verdict(samples: &[SparkProcessSample]) -> &'static str {
    let phase = |name: &str| {
        let mut found = samples.iter().filter(|sample| sample.phase == name);
        let first = found.next()?;
        found.next().is_none().then_some(first)
    };
    let Some(context_before) = phase("before_context") else { return "unqualified: missing/duplicate isolated context phase"; };
    let Some(context_after) = phase("after_context_before_media") else { return "unqualified: missing/duplicate isolated context phase"; };
    let Some(before) = phase("before_expert_modules") else { return "unqualified: missing/duplicate isolated module phase"; };
    let Some(after) = phase("after_expert_modules_before_weights") else { return "unqualified: missing/duplicate isolated module phase"; };
    let Some(loaded) = phase("serve_owners_before_endpoint_rings") else { return "unqualified: missing/duplicate retained owner phase"; };
    let ordered = [context_before, context_after, before, after, loaded];
    if context_before.pid == 0 || context_before.system_total_bytes == 0
        || ordered.iter().any(|s| s.pid != context_before.pid
            || s.system_total_bytes != context_before.system_total_bytes
            || s.system_available_bytes > s.system_total_bytes)
        || ordered.windows(2).any(|pair| pair[0].monotonic_ns >= pair[1].monotonic_ns) {
        return "unqualified: phase/process/system mismatch";
    }
    // A large known retained allocation is the visibility control, not another
    // witness. Zero CUDA PSS must never be replaced with anonymous RSS or CUDA free.
    let retained = loaded.tracked_device_bytes.checked_sub(after.tracked_device_bytes).unwrap_or(0);
    let visible = loaded.cuda_mapping_pss_bytes.checked_sub(after.cuda_mapping_pss_bytes).unwrap_or(0);
    if retained == 0 || visible == 0 || visible.abs_diff(retained) > retained / 20 {
        return "unqualified: CUDA UMA retained allocations invisible/incomplete in process maps; stop";
    }
    for (before, after) in [(context_before, context_after), (before, after)] {
        if before.tracked_device_bytes != after.tracked_device_bytes
            || before.tracked_pinned_bytes != after.tracked_pinned_bytes {
            return "unqualified: isolated context/module phase allocated payload storage";
        }
        let Some(process_delta) = after.cuda_mapping_pss_bytes.checked_sub(before.cuda_mapping_pss_bytes) else {
            return "unqualified: CUDA mapping PSS decreased during isolated phase";
        };
        let Some(system_delta) = before.system_available_bytes.checked_sub(after.system_available_bytes) else {
            return "unqualified: system memory increased during isolated phase";
        };
        if process_delta == 0 || process_delta.abs_diff(system_delta) > (16 << 20) {
            return "unqualified: process-map/system-memory phase witnesses disagree or are invisible; stop";
        }
    }
    "witnesses consistent; external quiet-host context/module qualification still required"
}

#[cfg(test)]
mod tests {
    use super::*;
    fn samples() -> Vec<SparkProcessSample> {
        let phases = ["before_context", "after_context_before_media", "before_expert_modules",
            "after_expert_modules_before_weights", "serve_owners_before_endpoint_rings"];
        phases.into_iter().enumerate().map(|(i, phase)| {
            let mib = 1 << 20;
            let cuda = [0, 32, 32, 96, 1120][i] * mib;
            SparkProcessSample { phase: phase.into(), pid: 42, monotonic_ns: i as u64 + 1,
                rss_bytes: cuda, pss_bytes: cuda, private_bytes: 0,
                tracked_device_bytes: if i == 4 { 1024 * mib } else { 0 }, tracked_pinned_bytes: 0,
                cuda_mapping_pss_bytes: cuda, anonymous_pss_bytes: 0, file_pss_bytes: 0,
                system_available_bytes: 8192 * mib - cuda, system_total_bytes: 16384 * mib,
                allocatable_bytes: None, raw_cuda_free_bytes: None, method: "test maps/system".into() }
        }).collect()
    }
    #[test]
    fn witnesses_require_visible_cuda_in_both_isolated_phases() {
        assert!(witness_verdict(&samples()).starts_with("witnesses consistent"));
        for bad in 0..10 {
            let mut s = samples();
            match bad {
                0 => s[1].pid = 43,
                1 => s[2].monotonic_ns = s[1].monotonic_ns,
                2 => s[1].tracked_device_bytes = 1,
                3 => s[3].tracked_pinned_bytes = 1,
                4 => s[4].cuda_mapping_pss_bytes = s[3].cuda_mapping_pss_bytes,
                5 => s[1].cuda_mapping_pss_bytes = 0,
                6 => s[3].system_available_bytes -= 32 << 20,
                7 => s[1].system_total_bytes = 0,
                8 => s.push(s[0].clone()),
                _ => s[4].tracked_device_bytes = 0,
            }
            assert!(witness_verdict(&s).starts_with("unqualified:"), "case {bad}");
        }
    }
    #[test]
    fn process_snapshot_requires_explicit_fields_and_units() {
        let block = "Rss: 12 kB\nPss: 8 kB\nPrivate_Clean: 2 kB\nPrivate_Dirty: 3 kB\n";
        let maps = parse_smaps(&format!("1000-2000 rw-s 00000000 00:00 0 /dev/nvidia0\n{block}")).unwrap();
        assert_eq!((maps.rss, maps.pss, maps.private, maps.cuda_pss), (12288,8192,5120,8192));
        assert!(kib_field("Rss: 12 MB", "Rss:").is_err());
        assert!(kib_field("Rss: 18446744073709551615 kB", "Rss:").is_err());
        assert!(kib_field("Rss: 1 kB\nRss: 2 kB", "Rss:").is_err());
        assert!(parse_smaps("Rss: 1 kB").is_err());
    }
    #[test]
    fn mapping_domains_are_disjoint_and_anonymous_is_not_cuda() {
        let fields = "Rss: 10 kB\nPss: 8 kB\nPrivate_Clean: 0 kB\nPrivate_Dirty: 2 kB\n";
        let maps = parse_smaps(&format!("1000-2000 rw-p 0 00:00 0\n{fields}2000-3000 r-xp 0 00:00 1 /lib/code.so\n{fields}3000-4000 rw-s 0 00:00 0 /dev/nvmap\n{fields}")).unwrap();
        assert_eq!(maps.pss, maps.cuda_pss + maps.anon_pss + maps.file_pss);
        assert_eq!((maps.cuda_pss, maps.anon_pss, maps.file_pss), (8192,8192,8192));
    }
    #[test]
    fn retained_scope_details_are_not_context_qualification() {
        let census = Census::new();
        let detail = census.detail(&Snapshot::default(), &Snapshot::default(), 0).unwrap();
        assert!(!detail.context_modules_qualified);
        assert!(detail.runtime_sample_method.contains("measured_residual"));
    }
    #[test]
    fn census_phases_isolate_context_and_retained_modules() {
        let source = include_str!("local.rs");
        let phases = ["census.sample(\"before_native_library\"", "let library = unsafe { NativeLibrary::load",
            "census.sample(\"before_context\"", "library.cuda_device_synchronize()?",
            "census.sample(\"after_context_before_media\"", "crate::shared::vision::worker::start_encoders(",
            "census.sample(\"before_expert_modules\"", "super::modules::WarmModules::load",
            "census.sample(\"after_expert_modules_before_weights\"", "load_weights(&library",
            "census.sample(\"before_retained_arenas\"", "weights.execution(&library",
            "census.sample(\"after_retained_arenas\"", "census.sample(\"serve_owners_before_endpoint_rings\""];
        let mut next = 0;
        for phase in phases {
            next += source[next..].find(phase).unwrap_or_else(|| panic!("missing ordered {phase}")) + phase.len();
        }
    }
}
