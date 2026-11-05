//! Provenance-bound startup observations, not allocation receipts or admission policy.
use super::planner::{
    CoordinatorDevices, EmbeddingHome, ExpertDestination, Operation, Phase, PhaseOrder,
    PrefillShape,
};
use super::{AttentionPlacement, LayerMode};
use crate::plan::inputs::PublishedPlacementInputs;
use cuteafd_core::spark_reply::SparkReplyFormats;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Component, Path},
};
use thiserror::Error;

pub const RUNTIME_MEMORY_SCHEMA: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeMemoryInputs {
    pub schema: u32,
    pub records: Vec<RuntimeMemoryRecord>,
}

/// Relevant build identities must be supplied independently by the consumer.
/// An engine commit label alone does not establish runtime-source compatibility.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeBuildIdentity {
    pub engine_commit: String,
    pub fork_revision: String,
    pub fork_tree_sha256: String,
    pub runtime_sources_sha256: BTreeMap<String, String>,
    pub programs_sha256: String,
    pub compiler_image_sha256: String,
    pub native_artifacts_sha256: BTreeMap<String, String>,
    pub coordinator_binary_sha256: String,
    pub compiler_toolchain: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeMemoryProvenance {
    pub build: RuntimeBuildIdentity,
    pub evidence: Vec<InputFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputFile {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointIdentity {
    pub model: String,
    pub revision: String,
    pub format: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoordinatorIdentity {
    pub rank: u8,
    /// CUDA ordinal in the process's namespace, not an assumed host ordinal.
    pub physical_device: i32,
    /// Stable hardware identity/class, independent of CUDA visibility remapping.
    pub hardware: String,
    pub architecture: String,
    pub sms: u32,
    pub driver: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeLayoutKey {
    pub checkpoint: CheckpointIdentity,
    pub draft: Option<CheckpointIdentity>,
    pub draft_kv_format: Option<String>,
    pub coordinators: Vec<CoordinatorIdentity>,
    pub attention: AttentionPlacement,
    pub layers: Vec<LayerMode>,
    pub expert_homes: BTreeMap<usize, ExpertDestination>,
    pub spark_groups: Vec<Vec<u8>>,
    pub selected_context: u64,
    pub compiled_context: u64,
    pub kv_format: String,
    pub page_tokens: u64,
    pub pool_pages: u64,
    pub concurrency: u64,
    pub prefill: PrefillShape,
    pub lanes: u64,
    pub operation: Operation,
    pub embedding: EmbeddingHome,
    pub selector: String,
    pub l2_policy: String,
    pub host_prefix_budget_bytes: u64,
    /// None means no physical Spark link; present selections are fully resolved.
    pub spark_reply_formats: Option<SparkReplyFormats>,
    pub instrumentation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StartupStage {
    pub id: String,
    pub phase: Phase,
    pub native_event_position: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuMemorySample {
    pub stage: String,
    pub raw_total_bytes: u64,
    pub raw_free_bytes: u64,
    pub raw_used_bytes: u64,
    pub exact_backing_bytes: u64,
    pub measured_unattributed_bytes: i64,
    /// Difference of consecutive unattributed observations, including graphs.
    /// Graph instantiate bytes remain separate, never automatically added again.
    pub delta_bytes: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphCounts {
    pub captures: u64,
    pub nodes: u64,
}

/// Stable shape/segment keys, not addresses or a universal price per executable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphInventory {
    pub keys: BTreeMap<String, GraphCounts>,
    pub fingerprint: String,
}
impl GraphInventory {
    pub fn fingerprint(keys: &BTreeMap<String, GraphCounts>) -> Result<String, RuntimeMemoryError> {
        let bytes = serde_json::to_vec(keys).map_err(RuntimeMemoryError::Json)?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphObservation {
    pub before_stage: String,
    pub after_stage: String,
    /// Signed immediate cudaMemGetInfo change across instantiate calls.
    /// Allocator granularity/reuse means this is not per-executable pricing.
    pub signed_instantiate_bytes: i64,
    pub bytes_valid: bool,
    pub inventory: GraphInventory,
    pub evidence: InputFile,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuMemoryObservation {
    pub rank: u8,
    pub context_baseline_bytes: i64,
    pub samples: Vec<GpuMemorySample>,
    pub graphs: Option<GraphObservation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostMemorySample {
    pub stage: String,
    pub smaps_rss_bytes: u64,
    pub file_resident_bytes: u64,
    pub pinned_in_ordinary_rss_bytes: u64,
    pub ordinary_rss_bytes: Option<u64>,
    pub ordinary_rss_complete: bool,
    pub reconstruction_residual_bytes: Option<i64>,
    pub pinned_requested_bytes: u64,
    pub pinned_page_covered_bytes: u64,
    pub pinned_present_bytes: Option<u64>,
    pub page_bytes: u64,
    pub complete: bool,
    pub findings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeMemoryRecord {
    pub id: String,
    pub key: RuntimeLayoutKey,
    pub provenance: RuntimeMemoryProvenance,
    pub phase_order: PhaseOrder,
    pub stages: Vec<StartupStage>,
    pub gpus: Vec<GpuMemoryObservation>,
    /// One process-wide sequence, never one copy per coordinator rank.
    pub host: Vec<HostMemorySample>,
    pub findings: Vec<String>,
}

#[derive(Debug, Error)]
pub enum RuntimeMemoryError {
    #[error("invalid runtime memory input: {0}")]
    Invalid(String),
    #[error("runtime memory JSON: {0}")]
    Json(#[source] serde_json::Error),
    #[error("published runtime memory input: {0}")]
    Publication(String),
}

type Result<T, E = RuntimeMemoryError> = std::result::Result<T, E>;
fn require(condition: bool, reason: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(RuntimeMemoryError::Invalid(reason.into()))
    }
}
fn text(value: &str) -> bool {
    !value.trim().is_empty()
}
fn digest(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
fn relative(value: &str) -> bool {
    text(value)
        && Path::new(value)
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
}
fn hashes(values: &BTreeMap<String, String>) -> bool {
    !values.is_empty()
        && values
            .iter()
            .all(|(path, sha)| relative(path) && digest(sha, 64))
}
fn signed(value: i128) -> Result<i64> {
    value
        .try_into()
        .map_err(|_| RuntimeMemoryError::Invalid("signed byte observation overflow".into()))
}

/// Validation only: this is neither an admission authorization nor a calibrated
/// allowance. There is deliberately no Deserialize implementation for this type.
#[derive(Debug)]
pub struct PublishedRuntimeMemory {
    inputs: RuntimeMemoryInputs,
}
impl PublishedRuntimeMemory {
    pub fn read(
        publication: &PublishedPlacementInputs,
        name: &str,
        expected: &BTreeMap<String, RuntimeBuildIdentity>,
    ) -> Result<Self> {
        let bytes = publication
            .read(name)
            .map_err(|error| RuntimeMemoryError::Publication(format!("{error:#}")))?;
        let inputs: RuntimeMemoryInputs =
            serde_json::from_slice(&bytes).map_err(RuntimeMemoryError::Json)?;
        inputs.validate(expected, publication.checksums())?;
        publication
            .read("PROGRAMS.json")
            .map_err(|error| RuntimeMemoryError::Publication(format!("{error:#}")))?;
        let mut evidence = BTreeSet::new();
        for record in &inputs.records {
            require(
                record.provenance.build.fork_revision == publication.fork_revision()
                    && record.provenance.build.fork_tree_sha256 == publication.fork_tree_sha256(),
                "runtime/placement fork identity mismatch",
            )?;
            for reference in record.provenance.evidence.iter().chain(
                record
                    .gpus
                    .iter()
                    .filter_map(|gpu| gpu.graphs.as_ref().map(|graphs| &graphs.evidence)),
            ) {
                if evidence.insert(&reference.path) {
                    publication
                        .read(&reference.path)
                        .map_err(|error| RuntimeMemoryError::Publication(format!("{error:#}")))?;
                }
            }
        }
        Ok(Self { inputs })
    }
    pub fn records(&self) -> &[RuntimeMemoryRecord] {
        &self.inputs.records
    }
}

impl RuntimeBuildIdentity {
    fn validate(&self) -> Result<()> {
        require(
            digest(&self.engine_commit, 40)
                && digest(&self.fork_revision, 40)
                && digest(&self.fork_tree_sha256, 64)
                && digest(&self.programs_sha256, 64)
                && digest(&self.compiler_image_sha256, 64)
                && digest(&self.coordinator_binary_sha256, 64)
                && text(&self.compiler_toolchain)
                && hashes(&self.runtime_sources_sha256)
                && hashes(&self.native_artifacts_sha256),
            "incomplete runtime build identity",
        )
    }
}

impl RuntimeMemoryInputs {
    /// `expected` and the publication checksums are independent inputs, not
    /// values copied from this document to make its provenance self-validating.
    pub fn validate(
        &self,
        expected: &BTreeMap<String, RuntimeBuildIdentity>,
        files: &BTreeMap<String, String>,
    ) -> Result<()> {
        require(
            self.schema == RUNTIME_MEMORY_SCHEMA && !self.records.is_empty(),
            "schema or records",
        )?;
        require(
            expected.len() == self.records.len(),
            "independent record identity coverage",
        )?;
        let mut ids = BTreeSet::new();
        for record in &self.records {
            require(
                text(&record.id) && ids.insert(&record.id),
                "empty or duplicate record id",
            )?;
            let build = expected.get(&record.id).ok_or_else(|| {
                RuntimeMemoryError::Invalid(format!("missing expected identity: {}", record.id))
            })?;
            build.validate()?;
            require(
                files.get("PROGRAMS.json") == Some(&build.programs_sha256),
                "program manifest identity",
            )?;
            record.validate(build, files)?;
        }
        Ok(())
    }
}

fn file(reference: &InputFile, files: &BTreeMap<String, String>) -> Result<()> {
    require(
        relative(&reference.path)
            && digest(&reference.sha256, 64)
            && files.get(&reference.path) == Some(&reference.sha256),
        "evidence absent from checksummed publication",
    )
}

impl RuntimeLayoutKey {
    fn validate(&self) -> Result<()> {
        let checkpoint = |identity: &CheckpointIdentity| {
            text(&identity.model) && digest(&identity.revision, 40) && text(&identity.format)
        };
        require(
            checkpoint(&self.checkpoint)
                && self.draft.as_ref().is_none_or(checkpoint)
                && self.draft.is_some() == self.draft_kv_format.is_some()
                && self.draft_kv_format.as_deref().is_none_or(text),
            "checkpoint/draft identity",
        )?;
        let physical: Vec<_> = self
            .coordinators
            .iter()
            .map(|gpu| gpu.physical_device)
            .collect();
        CoordinatorDevices::new(&physical)
            .map_err(|error| RuntimeMemoryError::Invalid(error.to_string()))?;
        require(
            self.coordinators.iter().enumerate().all(|(rank, gpu)| {
                usize::from(gpu.rank) == rank
                    && text(&gpu.hardware)
                    && text(&gpu.architecture)
                    && gpu.sms > 0
                    && text(&gpu.driver)
            }),
            "coordinator identity",
        )?;
        let gpus = self.coordinators.len();
        require(
            !self.layers.is_empty()
                && self.layers.iter().all(|mode| match mode {
                    LayerMode::Whole { gpu, .. } => usize::from(*gpu) < gpus,
                    LayerMode::HeadSplit | LayerMode::ContextSplit => gpus == 2,
                }),
            "layer homes",
        )?;
        require(
            self.selected_context > 0
                && self.compiled_context >= self.selected_context
                && text(&self.kv_format)
                && self.page_tokens > 0
                && self.pool_pages > 0
                && self.concurrency > 0
                && self.prefill.rows > 0
                && self.prefill.waves > 0
                && self.lanes > 0
                && self
                    .pool_pages
                    .checked_mul(self.page_tokens)
                    .is_some_and(|tokens| tokens >= self.selected_context)
                && self.prefill.rows.checked_mul(self.prefill.waves).is_some(),
            "context/pool/execution geometry",
        )?;
        require(
            match self.embedding {
                EmbeddingHome::Device(rank) => usize::from(rank) < gpus,
                EmbeddingHome::HostPinned => true,
            },
            "embedding home",
        )?;
        require(
            text(&self.selector) && text(&self.l2_policy) && text(&self.instrumentation),
            "execution policy identity",
        )?;
        let mut ranks = BTreeSet::new();
        require(
            self.spark_groups
                .iter()
                .all(|group| !group.is_empty() && group.iter().all(|rank| ranks.insert(*rank))),
            "Spark rank groups",
        )?;
        require(
            self.spark_reply_formats.is_some() == !ranks.is_empty()
                && self
                    .spark_reply_formats
                    .is_none_or(|formats| formats.prefill.is_some() && formats.decode.is_some()),
            "resolved phase-pair replies",
        )?;
        require(
            self.expert_homes.iter().all(|(layer, destination)| {
                *layer < self.layers.len()
                    && match destination {
                        ExpertDestination::Coordinator { gpus: homes } => {
                            !homes.is_empty()
                                && homes.iter().all(|rank| usize::from(*rank) < gpus)
                                && homes.iter().collect::<BTreeSet<_>>().len() == homes.len()
                        }
                        ExpertDestination::Sparks { ranks: homes } => {
                            !homes.is_empty()
                                && homes.iter().all(|rank| ranks.contains(rank))
                                && homes.iter().collect::<BTreeSet<_>>().len() == homes.len()
                        }
                    }
            }),
            "expert homes",
        )
    }
}

impl RuntimeMemoryRecord {
    fn validate(
        &self,
        expected: &RuntimeBuildIdentity,
        files: &BTreeMap<String, String>,
    ) -> Result<()> {
        self.key.validate()?;
        require(
            self.findings.iter().all(|finding| text(finding)),
            "empty runtime finding",
        )?;
        let provenance = &self.provenance;
        provenance.build.validate()?;
        require(
            provenance.build == *expected,
            format!("stale runtime build identity: {}", self.id),
        )?;
        require(!provenance.evidence.is_empty(), "runtime provenance")?;
        let mut evidence = BTreeSet::new();
        for reference in &provenance.evidence {
            file(reference, files)?;
            require(evidence.insert(&reference.path), "duplicate evidence path")?;
        }
        require(
            self.stages.len() >= 2
                && self.stages[0].phase == Phase::Load
                && self
                    .stages
                    .last()
                    .is_some_and(|stage| stage.phase == Phase::Ready),
            "startup endpoints",
        )?;
        let mut names = BTreeSet::new();
        for stage in &self.stages {
            require(
                text(&stage.id) && names.insert(&stage.id),
                "empty or duplicate startup stage",
            )?;
        }
        require(
            self.stages.windows(2).all(|pair| {
                pair[0].native_event_position <= pair[1].native_event_position
                    && self.phase_order.at_or_before(pair[0].phase, pair[1].phase)
            }),
            "actual startup chronology",
        )?;
        require(
            self.gpus.len() == self.key.coordinators.len(),
            "GPU observation coverage",
        )?;
        for (rank, gpu) in self.gpus.iter().enumerate() {
            require(
                usize::from(gpu.rank) == rank && gpu.samples.len() == self.stages.len(),
                "GPU rank/stage coverage",
            )?;
            let mut previous = None;
            for (stage, sample) in self.stages.iter().zip(&gpu.samples) {
                require(
                    sample.raw_total_bytes.checked_sub(sample.raw_free_bytes)
                        == Some(sample.raw_used_bytes),
                    "raw CUDA total/free/used arithmetic",
                )?;
                let unclassified = signed(
                    i128::from(sample.raw_used_bytes) - i128::from(sample.exact_backing_bytes),
                )?;
                let delta = previous.map_or(Ok(0), |bytes| {
                    signed(i128::from(unclassified) - i128::from(bytes))
                })?;
                require(
                    sample.stage == stage.id
                        && sample.measured_unattributed_bytes == unclassified
                        && sample.delta_bytes == delta,
                    "signed GPU observation arithmetic",
                )?;
                previous = Some(unclassified);
            }
            require(
                gpu.context_baseline_bytes == gpu.samples[0].measured_unattributed_bytes,
                "context baseline",
            )?;
            if let Some(graphs) = &gpu.graphs {
                file(&graphs.evidence, files)?;
                let before = self
                    .stages
                    .iter()
                    .position(|stage| stage.id == graphs.before_stage);
                let after = self
                    .stages
                    .iter()
                    .position(|stage| stage.id == graphs.after_stage);
                require(
                    matches!((before, after), (Some(a), Some(b)) if a < b),
                    "graph stage bracket",
                )?;
                require(
                    !graphs.inventory.keys.is_empty()
                        && graphs
                            .inventory
                            .keys
                            .iter()
                            .all(|(key, count)| text(key) && count.captures > 0 && count.nodes > 0),
                    "whole graph inventory",
                )?;
                require(
                    graphs.inventory.fingerprint
                        == GraphInventory::fingerprint(&graphs.inventory.keys)?,
                    "graph inventory fingerprint",
                )?;
                let _totals = graphs
                    .inventory
                    .keys
                    .values()
                    .try_fold((0u64, 0u64), |(captures, nodes), count| {
                        Some((
                            captures.checked_add(count.captures)?,
                            nodes.checked_add(count.nodes)?,
                        ))
                    })
                    .ok_or_else(|| {
                        RuntimeMemoryError::Invalid("graph inventory totals overflow".into())
                    })?;
                require(
                    graphs.bytes_valid || !self.findings.is_empty(),
                    "invalid graph sample without finding",
                )?;
            } else {
                require(
                    !self.findings.is_empty(),
                    "missing graph observation without finding",
                )?;
            }
        }
        require(
            self.host.len() == self.stages.len(),
            "process-wide host stage coverage",
        )?;
        for (stage, sample) in self.stages.iter().zip(&self.host) {
            sample.validate(&stage.id)?;
        }
        Ok(())
    }
}

impl HostMemorySample {
    fn validate(&self, stage: &str) -> Result<()> {
        require(
            self.stage == stage && self.page_bytes.is_power_of_two(),
            "host stage/page geometry",
        )?;
        require(
            self.findings.iter().all(|finding| text(finding)),
            "empty host finding",
        )?;
        require(
            self.pinned_page_covered_bytes >= self.pinned_requested_bytes
                && self.pinned_page_covered_bytes % self.page_bytes == 0
                && self.pinned_present_bytes.is_none_or(|present| {
                    present <= self.pinned_page_covered_bytes && present % self.page_bytes == 0
                }),
            "pinned requested/covered/present extents",
        )?;
        require(
            self.ordinary_rss_complete == self.ordinary_rss_bytes.is_some()
                && self.ordinary_rss_complete == self.reconstruction_residual_bytes.is_some(),
            "ordinary RSS completeness",
        )?;
        if let Some(ordinary) = self.ordinary_rss_bytes {
            let residual = signed(
                i128::from(self.smaps_rss_bytes)
                    - i128::from(self.file_resident_bytes)
                    - i128::from(self.pinned_in_ordinary_rss_bytes)
                    - i128::from(ordinary),
            )?;
            require(
                self.reconstruction_residual_bytes == Some(residual),
                "smaps reconstruction arithmetic",
            )?;
            require(
                residual.unsigned_abs() <= self.page_bytes || !self.findings.is_empty(),
                "smaps residual without finding",
            )?;
        }
        let complete = self.ordinary_rss_complete
            && self.pinned_present_bytes == Some(self.pinned_page_covered_bytes)
            && self
                .reconstruction_residual_bytes
                .is_some_and(|residual| residual.unsigned_abs() <= self.page_bytes);
        require(
            self.complete == (complete && self.findings.is_empty())
                && (complete || !self.findings.is_empty()),
            "host completeness/finding disagreement",
        )
    }
}

#[cfg(test)]
mod tests;
