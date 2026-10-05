//! `cuteafd plan`: what a checkpoint is, where each part would live, and what
//! this build can or cannot run - with hints a code agent can act on.
pub mod checkpoint;
pub mod experts;
pub mod families;
pub mod family;
pub mod format;
pub mod launch;
pub mod layout;
pub mod names;
pub mod spec;
#[doc(hidden)]
pub mod testing;

use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub use checkpoint::Checkpoint;
pub use family::{ConfigError, ExpertContract, Family, FamilyModel, Hint, RuntimeStatus};
pub use format::{Encoding, Malformed, QuantOperand, RowTiling, ScaleEncoding};
pub use spec::{AttentionKind, Component, FfnKind, ModelSpec, TensorRole};

const GIB: f64 = (1u64 << 30) as f64;
/// Rejected tensors kept per component (the count covers the rest).
const REJECTIONS_KEPT: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Owner {
    Rtx,
    SparkSliced,
    HostMapped,
}

impl Owner {
    fn label(self, placement: ExpertPlacement) -> String {
        match (self, placement) {
            (Owner::Rtx, _) => "rtx".into(),
            (Owner::SparkSliced, ExpertPlacement::Sparks { ranks }) => format!("spark x{ranks}"),
            (Owner::SparkSliced, ExpertPlacement::Local) => "rtx (local)".into(),
            (Owner::HostMapped, _) => "host-mapped".into(),
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

/// A tensor (group) the family's loaders would not take, and why.
#[derive(Debug, Clone, Serialize)]
pub struct Rejection {
    pub tensor: String,
    pub reason: String,
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
    /// Logical weights the contract rejects (malformed or not executed).
    pub rejected: usize,
    /// The first of them, with reasons.
    pub rejections: Vec<Rejection>,
}

/// Where the routed experts live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ExpertPlacement {
    /// Every routed-expert layer resident on the coordinator GPU.
    Local,
    /// Tensor-parallel slices over `ranks` Spark ranks.
    Sparks { ranks: usize },
}

impl ExpertPlacement {
    /// The `--spark-ranks` spelling: 0 is the local-only placement.
    pub fn from_spark_ranks(ranks: usize) -> Self {
        if ranks == 0 { Self::Local } else { Self::Sparks { ranks } }
    }

    pub fn spark_ranks(self) -> usize {
        match self {
            Self::Local => 0,
            Self::Sparks { ranks } => ranks,
        }
    }
}

/// Planning options that cannot describe a deployment.
#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error("invalid {option}: {reason}")]
    InvalidOption { option: &'static str, reason: String },
    #[error(transparent)]
    Checkpoint(#[from] anyhow::Error),
}

#[derive(Debug, Clone, Serialize)]
pub struct PlanReport {
    pub snapshot: String,
    pub family: Option<String>,
    pub architectures: Vec<String>,
    /// The checkpoint's ModelOpt quantization metadata (hf_quant_config.json /
    /// config.json), summarized; every weight was checked against it.
    pub quantization: Option<String>,
    pub runtime: Option<RuntimeStatus>,
    /// Why the family's runtime refuses this configuration, when it does.
    pub config_error: Option<String>,
    pub spec: Option<ModelSpec>,
    /// Canonical target cache storage, separate from the weight-only verdict.
    /// Complete serving admission also needs actual loaded representations,
    /// modules/workspaces, prefix marks and optional drafter reservations.
    pub cache_requirements: Option<crate::serving_capacity::CacheRequirements>,
    pub components: Vec<ComponentPlan>,
    pub unclassified: Vec<String>,
    pub missing_shards: Vec<String>,
    pub bytes_by_owner: BTreeMap<String, u64>,
    pub placement: ExpertPlacement,
    /// Spark ranks of the placement (0: local-only).
    pub spark_ranks: usize,
    /// The package and layouts that serve this checkpoint's routed experts.
    pub experts: Option<ExpertContract>,
    /// Whether this build has a layout for the requested placement.
    pub placement_supported: bool,
    /// Fewest Spark ranks with a package layout whose budget holds every
    /// routed expert, if any does.
    pub min_spark_ranks: Option<usize>,
    /// Share of the routed-expert bytes the widest Spark rank holds: ranks
    /// own whole blocks of the intermediate (TP6 of 2048: 3 of 16).
    pub spark_rank_share: f64,
    pub fits: bool,
    /// Family and routed-expert layers for the launch scripts.
    pub launch: Option<launch::LaunchDescription>,
    /// Where the expert service read the routed EXL3 storage layout from.
    pub expert_storage: Option<crate::formats::exl3_storage::Exl3StorageSource>,
    pub hints: Vec<Hint>,
    /// Per-device memory layout, when requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_layout: Option<cuteafd_core::memory_layout::MemoryLayout>,
}

impl PlanReport {
    pub fn executable(&self) -> bool {
        self.family.is_some()
            && self.config_error.is_none()
            && self.missing_shards.is_empty()
            && self.unclassified.is_empty()
            && self.components.iter().all(|c| matches!(c.status, Status::Ready | Status::Unused))
            && self.placement_supported
            && self.fits
    }
}

#[derive(Debug, Clone)]
pub struct PlanOptions {
    pub placement: ExpertPlacement,
    /// Routed-expert bytes one Spark rank may hold (weights only).
    pub spark_budget_bytes: u64,
    /// Weight bytes the coordinator GPU may hold (its own tensors, plus every
    /// routed expert in the local-only placement).
    pub coordinator_budget_bytes: u64,
    /// Also lay out every device's memory (`plan --layout`).
    pub layout: Option<layout::LayoutOptions>,
}

impl Default for PlanOptions {
    fn default() -> Self {
        Self {
            placement: ExpertPlacement::Sparks { ranks: 4 },
            spark_budget_bytes: 100 << 30,
            coordinator_budget_bytes: 80 << 30,
            layout: None,
        }
    }
}

impl PlanOptions {
    /// Rejects options that describe no deployment: Spark worlds the
    /// transport does not run (it runs 1, 2, 3, 4 or 6; 0 is the local-only
    /// placement) and empty budgets.
    pub fn validate(&self) -> Result<(), PlanError> {
        if let ExpertPlacement::Sparks { ranks } = self.placement {
            if !experts::TRANSPORT_WORLDS.contains(&ranks) {
                return Err(PlanError::InvalidOption {
                    option: "spark ranks",
                    reason: format!("{ranks}: the expert transport runs 1, 2, 3, 4 or 6 Spark ranks \
                        (0 places every routed expert on the coordinator)"),
                });
            }
        }
        for (option, bytes) in [("Spark budget", self.spark_budget_bytes), ("coordinator budget", self.coordinator_budget_bytes)] {
            if bytes == 0 {
                return Err(PlanError::InvalidOption { option, reason: "must be positive".into() });
            }
        }
        Ok(())
    }
}

/// `gib` as bytes: a finite, positive number of GiB that fits in u64.
pub fn budget_bytes(option: &'static str, gib: f64) -> Result<u64, PlanError> {
    let bytes = gib * GIB;
    if !gib.is_finite() || gib <= 0.0 || bytes < 1.0 || bytes >= u64::MAX as f64 {
        return Err(PlanError::InvalidOption { option, reason: format!("{gib} GiB is not a finite positive size") });
    }
    Ok(bytes as u64)
}

fn owner_for(component: Component) -> Owner {
    match component {
        Component::RoutedExpert => Owner::SparkSliced,
        Component::MappedTable => Owner::HostMapped,
        _ => Owner::Rtx,
    }
}

pub fn plan(snapshot: &Path, options: &PlanOptions) -> Result<PlanReport, PlanError> {
    options.validate()?;
    let checkpoint = Checkpoint::open(snapshot)
        .map_err(|error| error.context(format!("reading checkpoint at {}", snapshot.display())))?;
    let mut report = PlanReport {
        snapshot: snapshot.display().to_string(),
        family: None,
        architectures: checkpoint.architectures(),
        quantization: None,
        runtime: None,
        config_error: None,
        spec: None,
        cache_requirements: None,
        components: Vec::new(),
        unclassified: Vec::new(),
        missing_shards: checkpoint.missing_shards.clone(),
        bytes_by_owner: BTreeMap::new(),
        placement: options.placement,
        spark_ranks: options.placement.spark_ranks(),
        experts: None,
        placement_supported: true,
        min_spark_ranks: None,
        spark_rank_share: 0.0,
        fits: true,
        launch: launch::describe(&checkpoint.config).ok(),
        expert_storage: None,
        hints: Vec::new(),
        memory_layout: None,
    };
    // ModelOpt exports describe what they quantized; each weight's tensors
    // must agree with that description.
    let modelopt = match crate::formats::modelopt::ModelOpt::read(&checkpoint.snapshot, &checkpoint.config) {
        Ok(modelopt) => modelopt,
        Err(error) => {
            report.hints.push(Hint {
                what: format!("unreadable quantization metadata: {error}"),
                how: "ModelOpt checkpoints are read from their own hf_quant_config.json and config.json \
                      quantization_config (cuteafd-loader/src/formats/modelopt.rs); extend the reader for a new \
                      algorithm or spelling.".into(),
            });
            report.config_error = Some(error.to_string());
            None
        }
    };
    report.quantization = modelopt.as_ref().map(|m| m.summary());
    let Some(family) = family::detect(&checkpoint) else {
        report.hints.push(Hint {
            what: format!(
                "no family recognizes architectures {:?} (model_type {:?})",
                report.architectures,
                checkpoint.model_type()
            ),
            how: "Add a family under rust/crates/cuteafd-loader/src/plan/families/ implementing \
                  Family (detect, open, classify, component_hint) and register it in \
                  plan/family.rs::registry. Start from the closest existing family."
                .into(),
        });
        return Ok(report);
    };
    report.family = Some(family.id().into());
    report.runtime = Some(family.runtime());
    let model = match family.open(&checkpoint) {
        Ok(model) => model,
        Err(error) => {
            report.hints.push(Hint {
                what: format!("{} configuration is not servable: {error}", family.id()),
                how: format!("The {} runtime parses config.json with the same reader \
                    (cuteafd-loader/src/families/{}/config.rs); extend it and the engine together.",
                    family.id(), family.id()),
            });
            report.config_error.get_or_insert(error.0);
            return Ok(report);
        }
    };
    let spec = model.spec();
    match crate::serving_capacity::cache_requirements(model.as_ref(), &checkpoint.config) {
        Ok(requirements) => report.cache_requirements = requirements,
        Err(error) => report.hints.push(Hint {
            what: format!("cache capacity cannot be described: {error}"),
            how: "Use a positive checkpoint max_position_embeddings and an implemented family cache geometry; \
                serving admission must resolve actual GPU and workspace reservations before allocating.".into(),
        }),
    }

    // Classify every tensor, then group each component's tensors by stem: one
    // operand per logical weight, checked against the family's contract.
    let mut by_component: BTreeMap<Component, Vec<checkpoint::CheckpointTensor>> = BTreeMap::new();
    let mut roles: BTreeMap<String, TensorRole> = BTreeMap::new();
    for tensor in &checkpoint.tensors {
        match family.classify(spec, &tensor.meta.name) {
            Some(role) => {
                roles.insert(format::split_stem(&tensor.meta.name).0.to_owned(), role.clone());
                by_component.entry(role.component).or_default().push(tensor.clone());
            }
            None => report.unclassified.push(tensor.meta.name.clone()),
        }
    }
    // The expert staging's own verdict on the routed experts.
    let catalog = (family.expert_catalog() && by_component.contains_key(&Component::RoutedExpert))
        .then(|| crate::read_expert_catalog(&checkpoint.snapshot));
    report.expert_storage = catalog.as_ref().and_then(|catalog| catalog.as_ref().ok()?.exl3().map(|m| m.storage));
    let catalog_error = catalog.and_then(Result::err).map(|error| format!("{error:#}"));
    let mut hinted = BTreeSet::new();
    // Routed-expert operands by label, for the placement contract.
    let mut routed_operands: BTreeMap<String, QuantOperand> = BTreeMap::new();
    for (component, tensors) in &by_component {
        let groups = format::group_by_stem(tensors);
        let mut formats: BTreeMap<String, usize> = BTreeMap::new();
        let mut rejected = 0usize;
        let mut rejections = Vec::new();
        for (stem, members) in &groups {
            let role = roles.get(stem).cloned().unwrap_or_else(|| TensorRole::new(*component));
            let verdict = match format::detect(members) {
                Ok(mut operand) => {
                    let declared = modelopt.as_ref().map_or(Ok(()), |m| m.check(stem, &operand));
                    let accepted = declared.and_then(|()| model.accepts(&role, stem, &mut operand));
                    *formats.entry(operand.label()).or_default() += 1;
                    if accepted.is_ok() && *component == Component::RoutedExpert {
                        routed_operands.entry(operand.label()).or_insert(operand);
                    }
                    accepted.map_err(|reason| Rejection { tensor: stem.clone(), reason })
                }
                Err(malformed) => {
                    *formats.entry("malformed".into()).or_default() += 1;
                    Err(Rejection { tensor: malformed.tensor, reason: malformed.reason })
                }
            };
            if let Err(rejection) = verdict {
                rejected += 1;
                if rejections.len() < REJECTIONS_KEPT {
                    rejections.push(rejection);
                }
            }
        }
        if let (Component::RoutedExpert, Some(error)) = (component, &catalog_error) {
            rejected += 1;
            rejections.insert(0, Rejection { tensor: "routed experts (read_expert_catalog)".into(), reason: error.clone() });
            rejections.truncate(REJECTIONS_KEPT);
        }
        let status = match family.runtime() {
            RuntimeStatus::Planned => Status::Planned,
            RuntimeStatus::Serving if rejected == 0 => Status::Ready,
            RuntimeStatus::Serving if family.optional(*component) => Status::Unused,
            RuntimeStatus::Serving => Status::MissingKernel,
        };
        if !matches!(status, Status::Ready | Status::Unused) && hinted.insert(*component) {
            if let Some(first) = rejections.first() {
                report.hints.push(Hint {
                    what: format!("{}: {rejected} of {} weights not executable, e.g. {}: {}", component.label(),
                        groups.len(), first.tensor, first.reason),
                    how: "The family's tensor contract (plan/families/) mirrors its loaders; a new format needs \
                        the loader staging and kernels first, then the contract."
                        .into(),
                });
            }
            let labels: Vec<String> = formats.keys().cloned().collect();
            if let Some(hint) = family.component_hint_for(spec, *component, &labels) {
                if !report.hints.iter().any(|h| h.what == hint.what) {
                    report.hints.push(hint);
                }
            }
        }
        let bytes: u64 = tensors.iter().map(|t| t.meta.byte_length).sum();
        let owner = owner_for(*component);
        report.components.push(ComponentPlan {
            component: *component,
            owner,
            tensors: tensors.len(),
            bytes,
            formats,
            status,
            rejected,
            rejections,
        });
    }
    place(&mut report, options, spec, model.as_ref(), &routed_operands);
    if let Some(layout_options) = &options.layout {
        report.memory_layout = Some(layout::layout(&report, model.as_ref(), &checkpoint, layout_options));
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
    report.spec = Some(spec.clone());
    Ok(report)
}

/// Places the routed experts (Spark slices or the coordinator) and checks
/// memory: the widest Spark rank against the Spark budget, the coordinator's
/// own tensors (plus every expert, local-only) against its budget.
fn place(report: &mut PlanReport, options: &PlanOptions, spec: &ModelSpec, model: &dyn FamilyModel,
    routed_operands: &BTreeMap<String, QuantOperand>) {
    let bytes_of = |owner: Owner| -> u64 {
        report.components.iter().filter(|c| c.owner == owner).map(|c| c.bytes).sum()
    };
    let (routed, rtx, mapped) = (bytes_of(Owner::SparkSliced), bytes_of(Owner::Rtx), bytes_of(Owner::HostMapped));
    for (owner, bytes) in [(Owner::Rtx, rtx), (Owner::SparkSliced, routed), (Owner::HostMapped, mapped)] {
        if bytes > 0 {
            *report.bytes_by_owner.entry(owner.label(options.placement)).or_default() += bytes;
        }
    }
    // One contract for every routed format present (EXL3 K3 and K4 tiers
    // share theirs); mixed packages keep the layouts they all have.
    let contracts: Vec<ExpertContract> = routed_operands.values().filter_map(|op| model.experts(op)).collect();
    let contract = contracts.first().cloned().map(|mut first| {
        for other in &contracts[1..] {
            first.spark_worlds.retain(|w| other.spark_worlds.contains(w));
            if first.block != other.block {
                first.spark_worlds.clear();
            }
            if other.local.is_err() {
                first.local = other.local.clone();
            }
            if !first.package.split(" + ").any(|p| p == other.package) {
                first.package = format!("{} + {}", first.package, other.package);
            }
        }
        first
    });
    let intermediate = spec.moe.as_ref().map(|moe| moe.intermediate);
    let gib = |bytes: f64| bytes / GIB;
    let coordinator = options.coordinator_budget_bytes as f64;
    if routed > 0 {
        let share = |ranks: usize| -> Option<f64> {
            let contract = contract.as_ref()?;
            let i = intermediate?;
            experts::stored_slice(i, contract.block, ranks).map(|slice| slice as f64 / i as f64)
        };
        let fits_on = |ranks: usize| share(ranks).is_some_and(|s| routed as f64 * s <= options.spark_budget_bytes as f64);
        report.min_spark_ranks = contract.as_ref().and_then(|c| c.spark_worlds.iter().copied().find(|&r| fits_on(r)));
        let advice = match report.min_spark_ranks {
            Some(ranks) => format!("Use {ranks} Spark ranks (--spark-ranks {ranks})"),
            None => "No Spark layout of this package holds them at this budget".into(),
        };
        match (options.placement, &contract) {
            (_, None) => {} // the routed-expert component already reports its missing kernel
            (ExpertPlacement::Sparks { ranks }, Some(contract)) => {
                report.spark_rank_share = share(ranks).unwrap_or(0.0);
                if !contract.spark_worlds.contains(&ranks) || share(ranks).is_none() {
                    report.placement_supported = false;
                    report.hints.push(Hint {
                        what: format!("no {} layout for {ranks} Spark ranks (this build packages {:?}{})",
                            contract.package, contract.spark_worlds,
                            intermediate.map_or(String::new(), |i| format!(", intermediate {i} in {}-row blocks", contract.block))),
                        how: match share(ranks) {
                            Some(_) => format!("{advice}, or build a tp{ranks} layout (python/tools/aot/\
                                package_fp8_moe_aot.py --layouts / package_exl3_aot.py)."),
                            None => format!("{advice}: {ranks} ranks cannot each own whole {}-row blocks.",
                                contract.block),
                        },
                    });
                } else if !fits_on(ranks) {
                    report.fits = false;
                    report.hints.push(Hint {
                        what: format!("routed experts need {:.1} GiB on the widest of {ranks} Spark ranks, over the \
                            {:.0} GiB budget", gib(routed as f64 * report.spark_rank_share),
                            gib(options.spark_budget_bytes as f64)),
                        how: format!("{advice}, keep bottom layers' experts resident on the RTX cards, or quantize \
                            the experts further (EXL3 K2-K3)."),
                    });
                }
            }
            (ExpertPlacement::Local, Some(contract)) => match &contract.local {
                Err(why) => {
                    report.placement_supported = false;
                    report.hints.push(Hint {
                        what: format!("unsupported: no local expert package for {}: {why}", spec.family),
                        how: format!("{advice}."),
                    });
                }
                Ok(_) => {
                    let need = (rtx + routed) as f64;
                    if need > coordinator {
                        report.fits = false;
                        report.hints.push(Hint {
                            what: format!("local-only placement needs {:.1} GiB on the coordinator GPU ({:.1} GiB \
                                routed experts + {:.1} GiB other weights), {:.1} GiB over the {:.0} GiB budget",
                                gib(need), gib(routed as f64), gib(rtx as f64), gib(need - coordinator), gib(coordinator)),
                            how: format!("{advice}."),
                        });
                    }
                }
            },
        }
    }
    if options.placement != ExpertPlacement::Local && rtx as f64 > coordinator {
        report.fits = false;
        report.hints.push(Hint {
            what: format!("coordinator weights need {:.1} GiB, over the {:.0} GiB coordinator budget",
                gib(rtx as f64), gib(coordinator)),
            how: "Split the backbone over two coordinator GPUs or move components to Sparks.".into(),
        });
    }
    report.experts = contract;
}

/// Select a measured faster local layout when it fits a serving reservation.
/// Explicit placements use `plan` instead. Qualification is intentionally narrow:
/// Qwen EXL3 K4/K5 has resident TP1 experts; other formats retain their placement.
pub fn plan_preferred(snapshot: &Path, options: &PlanOptions) -> Result<PlanReport, PlanError> {
    let fallback = plan(snapshot, options)?;
    if fallback.family.as_deref() != Some("qwen4")
        || fallback.experts.as_ref().is_none_or(|e| e.package != "qwen4:exl3-k45" || e.local.is_err()) {
        return Ok(fallback);
    }
    // The runtime reserves 12 GiB before loading resident experts (4096-row
    // logits, two lane workspaces, recurrent/prefix state and graph growth).
    // Reserve the optional native MTP weights too, even when drafting is off.
    let mtp: u64 = fallback.components.iter()
        .filter(|c| matches!(c.component, Component::Speculator | Component::SpeculatorExpert))
        .map(|c| c.bytes).sum();
    let mut layout = options.layout.clone().unwrap_or_else(|| layout::LayoutOptions {
        // The coordinator budget is weight-only; the default inventory is
        // the same 95.5 GiB RTX used by `plan --layout`.
        pool_tokens: Some(32_768),
        ..Default::default()
    });
    layout.head_split = false;
    // The layout already charges native MTP weights and expert arenas.
    // Keep the weight-only admission reserve without duplicating them as
    // an external drafter in the per-device layout.
    let capacity = layout.rtx_bytes.first().copied().unwrap_or(options.coordinator_budget_bytes);
    let local_options = PlanOptions {
        placement: ExpertPlacement::Local,
        coordinator_budget_bytes: options.coordinator_budget_bytes.min(capacity.saturating_sub(12 << 30).saturating_sub(mtp)),
        layout: Some(layout),
        ..options.clone()
    };
    if local_options.coordinator_budget_bytes == 0 { return Ok(fallback); }
    let mut local = plan(snapshot, &local_options)?;
    if !local.executable() || local.memory_layout.as_ref().is_none_or(|l| l.pool_tokens == 0
        || l.devices.iter().any(|d| d.free_bytes() < 0)) {
        return Ok(fallback);
    }
    local.hints.push(Hint {
        what: "Qwen EXL3: prefer resident local experts (qualified faster than Spark TP4)".into(),
        how: "Explicit --spark-ranks selects a Spark layout; launcher EXPERT_BACKEND=spark overrides auto.".into(),
    });
    if options.layout.is_none() { local.memory_layout = None; }
    Ok(local)
}

/// Human-readable report.
pub fn render(report: &PlanReport) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let _ = writeln!(out, "snapshot   {}", report.snapshot);
    let _ = writeln!(out, "arch       {}", report.architectures.join(", "));
    if let Some(quantization) = &report.quantization {
        let _ = writeln!(out, "quant      {quantization}");
    }
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
        (Some(family), None) => {
            let _ = writeln!(out, "family     {family}: configuration refused ({})",
                report.config_error.as_deref().unwrap_or("no spec"));
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
                c.owner.label(report.placement),
                status,
                formats
            );
        }
        if let Some(storage) = report.expert_storage {
            let _ = writeln!(out, "exl3 map   {}", storage.describe());
        }
        let rejected: Vec<&ComponentPlan> =
            report.components.iter().filter(|c| c.status == Status::MissingKernel && c.rejected > 0).collect();
        if !rejected.is_empty() {
            let _ = writeln!(out, "\nrejected weights:");
            for c in rejected {
                for r in c.rejections.iter().take(3) {
                    let _ = writeln!(out, "  {:<16} {}: {}", c.component.label(), r.tensor, r.reason);
                }
                if c.rejected > 3 {
                    let _ = writeln!(out, "  {:<16} ... {} more", c.component.label(), c.rejected - 3);
                }
            }
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
    if let Some(cache) = &report.cache_requirements {
        let context = cache.checkpoint_max_context_tokens.map_or("missing".to_string(), |v| v.to_string());
        let capability = if cache.compiled_index_extent_required {
            "serving manifest must provide the compiled index extent"
        } else { "dynamic context extent" };
        let _ = writeln!(out, "context    checkpoint maximum {context}; {capability}");
        let floor = cache.requested_kv_floor_tokens.map_or("unresolved".to_string(), |v| v.to_string());
        let _ = writeln!(out, "KV target  {floor} tokens (common default pool); C{} with {} state/ring slots",
            cache.concurrency, cache.state_slots);
        for layout in &cache.target_only_layouts {
            let placement = match layout.placement {
                crate::serving_capacity::KvPlacement::SingleDevice => "one owner",
                crate::serving_capacity::KvPlacement::Replicated => "replicated KV",
                crate::serving_capacity::KvPlacement::PartitionedHeads => "partitioned KV heads",
                crate::serving_capacity::KvPlacement::PartitionedLayers => "partitioned KV layers/sources",
            };
            for (rank, cost) in layout.ranks.iter().enumerate() {
                let bytes_per_token = cost.persistent_unit_bytes as f64 / layout.logical_unit_rows as f64;
                let state_gib = cost.active_state_per_sequence_bytes as f64 * f64::from(cache.state_slots) / GIB;
                let pool = cache.requested_kv_floor_tokens.map_or("unresolved".to_string(), |tokens| {
                    let units = tokens.div_ceil(layout.logical_unit_rows);
                    format!("{:.2} GiB", units as f64 * cost.persistent_unit_bytes as f64 / GIB)
                });
                let _ = writeln!(out, "  {} RTX rank {rank}: {placement}, {bytes_per_token:.0} KV B/token, \
                    target {pool}, active state {state_gib:.2} GiB, mark {:.2} MiB/slot",
                    layout.ranks.len(), cost.retained_mark_bytes as f64 / (1u64 << 20) as f64);
            }
        }
        for unavailable in &cache.unavailable_layouts {
            let _ = writeln!(out, "  {} RTX cache layout: {}", unavailable.coordinator_ranks, unavailable.reason);
        }
        let _ = writeln!(out, "  storage costs only: runtime admission must also reserve actual weights, modules, \
            all workspace shapes, prefix marks, transport and optional drafters against each GPU's live budget");
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
    if let Some(experts) = &report.experts {
        let local = match &experts.local {
            Ok(package) => package.clone(),
            Err(_) => "none".into(),
        };
        let _ = writeln!(out, "experts    {}: Spark worlds {:?}; local {local}", experts.package, experts.spark_worlds);
    }
    if report.spec.is_some() && report.bytes_by_owner.keys().any(|o| o != "rtx") {
        let min = report.min_spark_ranks.map_or("no Spark layout fits".into(), |r| format!(">= {r} Spark ranks"));
        let at = match report.placement {
            ExpertPlacement::Local => "local-only".into(),
            ExpertPlacement::Sparks { ranks } => format!("{ranks} ranks"),
        };
        let verdict = match (report.placement_supported, report.fits) {
            (false, _) => "UNSUPPORTED",
            (true, true) => "fits",
            (true, false) => "DOES NOT FIT",
        };
        let _ = writeln!(out, "capacity   routed experts need {min}; at {at}: {verdict}");
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
