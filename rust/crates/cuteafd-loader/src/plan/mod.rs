//! `cuteafd plan`: what a checkpoint is, where each part would live, and what
//! this build can or cannot run - with hints a code agent can act on.
pub mod checkpoint;
pub mod families;
pub mod family;
pub mod format;
pub mod names;
pub mod spec;

use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub use checkpoint::Checkpoint;
pub use family::{Family, Hint, RuntimeStatus};
pub use format::WeightFormat;
pub use spec::{AttentionKind, Component, FfnKind, ModelSpec, TensorRole};

const GIB: f64 = (1u64 << 30) as f64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Owner {
    Rtx,
    SparkSliced,
    HostMapped,
}

impl Owner {
    fn label(self, sparks: usize) -> String {
        match self {
            Owner::Rtx => "rtx".into(),
            Owner::SparkSliced => format!("spark x{sparks}"),
            Owner::HostMapped => "host-mapped".into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Executes in this build.
    Ready,
    /// The family runs, but this format for this component does not.
    MissingKernel,
    /// The family's execution path is not written yet.
    Planned,
    /// Optional for serving and not executed by this build (e.g. a speculator).
    Unused,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComponentPlan {
    pub component: Component,
    pub owner: Owner,
    pub tensors: usize,
    pub bytes: u64,
    /// Detected storage formats with the number of logical weights in each.
    pub formats: BTreeMap<String, usize>,
    pub status: Status,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlanReport {
    pub snapshot: String,
    pub family: Option<String>,
    pub architectures: Vec<String>,
    pub runtime: Option<RuntimeStatus>,
    pub spec: Option<ModelSpec>,
    pub components: Vec<ComponentPlan>,
    pub unclassified: Vec<String>,
    pub missing_shards: Vec<String>,
    pub bytes_by_owner: BTreeMap<String, u64>,
    pub spark_ranks: usize,
    /// Fewest Spark ranks whose budget holds every routed expert.
    pub min_spark_ranks: usize,
    /// Share of the routed-expert bytes the widest Spark rank holds: ranks
    /// own whole 128-blocks of the intermediate (TP6 of 2048: 3 of 16).
    pub spark_rank_share: f64,
    pub fits: bool,
    pub hints: Vec<Hint>,
}

impl PlanReport {
    pub fn executable(&self) -> bool {
        self.family.is_some()
            && self.missing_shards.is_empty()
            && self.unclassified.is_empty()
            && self.components.iter().all(|c| matches!(c.status, Status::Ready | Status::Unused))
            && self.fits
    }
}

pub struct PlanOptions {
    pub spark_ranks: usize,
    /// Routed-expert bytes one Spark rank may hold (weights only).
    pub spark_budget_bytes: u64,
}

impl Default for PlanOptions {
    fn default() -> Self {
        Self { spark_ranks: 4, spark_budget_bytes: 100 << 30 }
    }
}

fn owner_for(component: Component) -> Owner {
    match component {
        Component::RoutedExpert => Owner::SparkSliced,
        Component::MappedTable => Owner::HostMapped,
        _ => Owner::Rtx,
    }
}

pub fn plan(snapshot: &Path, options: &PlanOptions) -> Result<PlanReport> {
    let checkpoint = Checkpoint::open(snapshot)
        .with_context(|| format!("reading checkpoint at {}", snapshot.display()))?;
    let mut report = PlanReport {
        snapshot: snapshot.display().to_string(),
        family: None,
        architectures: checkpoint.architectures(),
        runtime: None,
        spec: None,
        components: Vec::new(),
        unclassified: Vec::new(),
        missing_shards: checkpoint.missing_shards.clone(),
        bytes_by_owner: BTreeMap::new(),
        spark_ranks: options.spark_ranks,
        min_spark_ranks: 0,
        spark_rank_share: 1.0 / options.spark_ranks.max(1) as f64,
        fits: true,
        hints: Vec::new(),
    };
    let Some(family) = family::detect(&checkpoint) else {
        report.hints.push(Hint {
            what: format!(
                "no family recognizes architectures {:?} (model_type {:?})",
                report.architectures,
                checkpoint.model_type()
            ),
            how: "Add a family under rust/crates/cuteafd-loader/src/plan/families/ implementing \
                  Family (detect, spec, classify, component_hint) and register it in \
                  plan/family.rs::registry. Start from the closest existing family."
                .into(),
        });
        return Ok(report);
    };
    let spec = family.spec(&checkpoint)?;
    report.family = Some(family.id().into());
    report.runtime = Some(family.runtime());

    // Classify every tensor, then group each component's tensors by stem to
    // detect one storage format per logical weight.
    let mut by_component: BTreeMap<Component, Vec<checkpoint::CheckpointTensor>> = BTreeMap::new();
    for tensor in &checkpoint.tensors {
        match family.classify(&spec, &tensor.meta.name) {
            Some(role) => by_component.entry(role.component).or_default().push(tensor.clone()),
            None => report.unclassified.push(tensor.meta.name.clone()),
        }
    }
    let mut hinted = BTreeSet::new();
    for (component, tensors) in &by_component {
        let groups = format::group_by_stem(tensors);
        let mut formats: BTreeMap<String, usize> = BTreeMap::new();
        let mut all_ready = true;
        for members in groups.values() {
            let detected = format::detect(members);
            if !family.executes(*component, &detected) {
                all_ready = false;
            }
            *formats.entry(detected.label()).or_default() += 1;
        }
        let status = match family.runtime() {
            RuntimeStatus::Planned => Status::Planned,
            RuntimeStatus::Serving if all_ready => Status::Ready,
            RuntimeStatus::Serving if family.optional(*component) => Status::Unused,
            RuntimeStatus::Serving => Status::MissingKernel,
        };
        if !matches!(status, Status::Ready | Status::Unused) && hinted.insert(*component) {
            let labels: Vec<String> = formats.keys().cloned().collect();
            if let Some(hint) = family.component_hint_for(&spec, *component, &labels) {
                if !report.hints.iter().any(|h| h.what == hint.what) {
                    report.hints.push(hint);
                }
            } else if status == Status::MissingKernel {
                report.hints.push(Hint {
                    what: format!("{} in formats {:?}", component.label(), formats.keys().collect::<Vec<_>>()),
                    how: "Add an execution path for this format (b12x kernel export + loader staging) \
                          and teach the family's `executes` to accept it."
                        .into(),
                });
            }
        }
        let bytes: u64 = tensors.iter().map(|t| t.meta.byte_length).sum();
        let owner = owner_for(*component);
        *report.bytes_by_owner.entry(owner.label(options.spark_ranks)).or_default() += bytes;
        report.components.push(ComponentPlan {
            component: *component,
            owner,
            tensors: tensors.len(),
            bytes,
            formats,
            status,
        });
    }
    let routed: u64 = report
        .components
        .iter()
        .filter(|c| c.owner == Owner::SparkSliced)
        .map(|c| c.bytes)
        .sum();
    // Expert tensor parallelism supports these group sizes; ranks own whole
    // 128-blocks of the intermediate, so the widest rank bounds the budget.
    let intermediate = spec.moe.as_ref().map(|moe| moe.intermediate);
    let share = |ranks: usize| match intermediate {
        Some(i) if i % 128 == 0 && i / 128 >= ranks => (i / 128).div_ceil(ranks) as f64 / (i / 128) as f64,
        _ => 1.0 / ranks.max(1) as f64,
    };
    report.spark_rank_share = share(options.spark_ranks);
    let fits_on = |ranks: usize| routed as f64 * share(ranks) <= options.spark_budget_bytes as f64;
    let needed = routed.div_ceil(options.spark_budget_bytes.max(1)) as usize;
    report.min_spark_ranks = [1usize, 2, 3, 4, 6]
        .into_iter()
        .find(|&ranks| fits_on(ranks))
        .unwrap_or(needed);
    if !fits_on(options.spark_ranks) {
        report.fits = false;
        report.hints.push(Hint {
            what: format!(
                "routed experts need {:.1} GiB on the widest of {} Spark ranks, over the {:.0} GiB budget",
                routed as f64 * report.spark_rank_share / GIB,
                options.spark_ranks,
                options.spark_budget_bytes as f64 / GIB
            ),
            how: format!(
                "Use at least {} Spark ranks (--spark-ranks), keep bottom layers' experts resident \
                 on the RTX cards, or quantize the experts further (EXL3 K2-K3).",
                report.min_spark_ranks
            ),
        });
    }
    if !report.unclassified.is_empty() {
        report.hints.push(Hint {
            what: format!("{} tensors match no {} rule", report.unclassified.len(), family.id()),
            how: format!(
                "Extend {}::classify in cuteafd-loader/src/plan/families/ (first: {}).",
                family.id(),
                report.unclassified[0]
            ),
        });
    }
    if !report.missing_shards.is_empty() {
        report.hints.push(Hint {
            what: format!("{} checkpoint shards are missing or unreadable", report.missing_shards.len()),
            how: "Finish the download (hf download) or replicate it (nest replicate hf:ORG/NAME).".into(),
        });
    }
    report.spec = Some(spec);
    Ok(report)
}

/// Human-readable report.
pub fn render(report: &PlanReport) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let _ = writeln!(out, "snapshot   {}", report.snapshot);
    let _ = writeln!(out, "arch       {}", report.architectures.join(", "));
    match (&report.family, &report.spec) {
        (Some(family), Some(spec)) => {
            let runtime = match report.runtime {
                Some(RuntimeStatus::Serving) => "serving",
                _ => "planned (execution path not written yet)",
            };
            let _ = writeln!(out, "family     {family}: {runtime}");
            let moe_layers = spec.moe_layers();
            let _ = write!(
                out,
                "spec       hidden {}, vocab {}, {} layers ({} MoE)",
                spec.hidden,
                spec.vocab,
                spec.layers.len(),
                moe_layers
            );
            if let Some(moe) = &spec.moe {
                let _ = write!(
                    out,
                    ", {} experts top-{} (intermediate {}, {} shared, {})",
                    moe.experts, moe.top_k, moe.intermediate, moe.shared_experts, moe.scoring
                );
            }
            let _ = writeln!(out);
            let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
            for layer in &spec.layers {
                let label = match &layer.attention {
                    AttentionKind::CompressedMla { ratio, indexer } => {
                        format!("compressed-mla r{ratio}{}", if *indexer { "+idx" } else { "" })
                    }
                    AttentionKind::MlaDsa { indexer } => format!("mla-dsa{}", if *indexer { "+idx" } else { "(shared idx)" }),
                    AttentionKind::Gqa { heads, kv_heads, head_dim } => format!("gqa {heads}/{kv_heads}x{head_dim}"),
                    AttentionKind::SlidingGqa { window, sinks, .. } => {
                        format!("swa-gqa w{window}{}", if *sinks { "+sink" } else { "" })
                    }
                    AttentionKind::Kda => "kda".into(),
                    AttentionKind::GatedDeltaNet => "gated-deltanet".into(),
                };
                *kinds.entry(label).or_default() += 1;
            }
            let _ = writeln!(
                out,
                "attention  {}",
                kinds.iter().map(|(k, n)| format!("{k} x{n}")).collect::<Vec<_>>().join(", ")
            );
            if let Some(speculator) = &spec.speculator {
                let _ = writeln!(out, "speculator {}", serde_json::to_string(speculator).unwrap_or_default());
            }
            for table in &spec.tables {
                let _ = writeln!(out, "table      {} on layers {:?}", table.name, table.layers);
            }
            for note in &spec.notes {
                let _ = writeln!(out, "note       {note}");
            }
        }
        _ => {
            let _ = writeln!(out, "family     not recognized");
        }
    }
    if !report.components.is_empty() {
        let _ = writeln!(out);
        let _ = writeln!(out, "{:<18} {:>8} {:>9}  {:<13} {:<9} formats", "component", "tensors", "GiB", "placement", "status");
        for c in &report.components {
            let formats = c.formats.iter().map(|(f, n)| format!("{f} x{n}")).collect::<Vec<_>>().join(", ");
            let status = match c.status {
                Status::Ready => "ready",
                Status::MissingKernel => "MISSING",
                Status::Planned => "planned",
                Status::Unused => "unused",
            };
            let _ = writeln!(
                out,
                "{:<18} {:>8} {:>9.2}  {:<13} {:<9} {}",
                c.component.label(),
                c.tensors,
                c.bytes as f64 / GIB,
                c.owner.label(report.spark_ranks),
                status,
                formats
            );
        }
        let _ = writeln!(out);
        for (owner, bytes) in &report.bytes_by_owner {
            let per = if owner.starts_with("spark") {
                format!(" ({:.1} GiB on the widest rank)", *bytes as f64 * report.spark_rank_share / GIB)
            } else {
                String::new()
            };
            let _ = writeln!(out, "total {:<12} {:>9.2} GiB{per}", owner, *bytes as f64 / GIB);
        }
    }
    if !report.unclassified.is_empty() {
        let _ = writeln!(out, "\nunclassified tensors: {}", report.unclassified.len());
        for name in report.unclassified.iter().take(10) {
            let _ = writeln!(out, "  {name}");
        }
    }
    if !report.missing_shards.is_empty() {
        let _ = writeln!(out, "\nmissing shards: {}", report.missing_shards.len());
    }
    if report.family.is_some() {
        let _ = writeln!(
            out,
            "capacity   routed experts need >= {} Spark ranks at {} ranks: {}",
            report.min_spark_ranks,
            report.spark_ranks,
            if report.fits { "fits" } else { "DOES NOT FIT" }
        );
    }
    let _ = writeln!(out, "\nverdict    {}", if report.executable() { "READY to serve" } else { "NOT servable by this build" });
    if !report.hints.is_empty() {
        let _ = writeln!(out, "\n--- hints for a code agent ---");
        for hint in &report.hints {
            let _ = writeln!(out, "* {}\n  {}", hint.what, hint.how);
        }
    }
    out
}

#[cfg(test)]
mod tests;
