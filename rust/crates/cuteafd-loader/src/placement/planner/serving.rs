//! Physical serving owners shared by family recipes. Quotas and selector
//! placement are resolved inputs; this module never chooses admission policy.
use super::*;

fn startup_graph_source(id: &str) -> String { format!("startup graphs: {id}") }

/// Executable residency is opaque driver storage, not a ledger allocation.
/// Estimates have no variance margin and are replaced after startup capture.
pub fn startup_graph_component(id: &str, location: Location, rank: crate::placement::GraphRank,
    maximum: u64) -> Result<Component, Refusal> {
    if id.is_empty() || maximum == 0 || rank.margin != 0 {
        return Err(Refusal::InvalidFact("invalid startup graph owner".into()));
    }
    Ok(Component { activation: Activation::always(), bytes: ByteCurve::fixed(rank.bytes, maximum),
        kind: ComponentKind::Allowance(Allowance { location, bytes: rank.bytes,
            lifetime: Lifetime { first: Phase::Warmup, last: Phase::Growth },
            source: startup_graph_source(id), assumed: rank.basis != Basis::Calibrated }) })
}

/// Reevaluate the selected facts, never select a different pool or placement.
/// A capture may exceed its estimate if physical capacity still admits it.
/// On refusal the original receipt stays intact.
pub fn reconcile_startup_graphs(plan: &mut ResolvedPlan, prefix: &str,
    estimated: &crate::placement::GraphSet, declared: &[String],
    captured: &[crate::placement::inventory::CapturedGraphRank])
    -> Result<crate::placement::GraphSet, Refusal> {
    let measured = estimated.captured(declared, captured).map_err(Refusal::InvalidFact)?;
    let mut facts = plan.facts.clone();
    for (rank, graph) in measured.ranks.iter().enumerate() {
        let source = startup_graph_source(&format!("{prefix}.{rank}"));
        let matches: Vec<_> = facts.allowances.iter_mut().filter(|owner| owner.source == source).collect();
        let [owner] = matches.try_into().map_err(|_: Vec<_>|
            Refusal::InvalidFact("missing or duplicate startup graph owner".into()))?;
        if owner.location != Location::coordinator(rank as u8)
            || owner.lifetime != (Lifetime { first: Phase::Warmup, last: Phase::Growth }) {
            return Err(Refusal::InvalidFact("startup graph owner location or lifetime differs".into()));
        }
        owner.bytes = graph.bytes;
        owner.assumed = false;
    }
    let phases = evaluate(&facts, &plan.budgets)?;
    plan.facts = facts;
    plan.phases = phases;
    Ok(measured)
}

/// A retained native handle group. Caller-owned Lt workspaces stay allocations.
#[derive(Debug, Clone)]
pub struct BlasOwners {
    pub name: String,
    /// Retained libraries are process-local; a remote media worker is not the coordinator.
    pub process: String,
    pub location: Location,
    pub cublas_handles: u64,
    pub lt_handles: u64,
    pub lifetime: Lifetime,
    pub activation: Activation,
}

/// SM120/driver 13020: 64 MiB/cuBLAS handle, shared 2 MiB initialization;
/// first SGEMM and representative Lt plans add zero (explicit 4 MiB scratch).
/// SM121/driver 13020, LAZY: four warmed handles average 67,508,224 B;
/// destroying the warm handle leaves 158,674,944 B. First Lt planning beside
/// BLAS adds 201,461,760 B, later creates/plans zero. These are separate
/// process-local owners, not a padded per-handle price. Probe: 2026-11-07,
/// BF16 M16/K4096/N4096. SM121 Lt-only processes need a separate qualification.
pub fn blas_components(arch: &str, owners: &[BlasOwners], maximum: u64,
    order: PhaseOrder) -> Result<Vec<Component>, Refusal> {
    let per_handle = match arch {
        "sm_120" => 64 << 20,
        "sm_121" => 67_508_224,
        _ => return Err(Refusal::InvalidFact(format!("unmeasured BLAS owner architecture {arch}"))),
    };
    if maximum == 0 { return Err(Refusal::InvalidFact("empty BLAS owner pool domain".into())); }
    let mut names = BTreeSet::new();
    let mut out = Vec::new();
    for owner in owners {
        let device_matches = match arch {
            "sm_120" => matches!(owner.location, Location { device: Device::Coordinator(_), domain: Domain::Device }),
            _ => matches!(owner.location, Location { device: Device::Spark(_), domain: Domain::Managed }),
        };
        if owner.name.is_empty() || owner.process.is_empty()
            || !names.insert((owner.location, owner.process.as_str(), owner.name.as_str()))
            || !device_matches || !order.at_or_before(owner.lifetime.first, owner.lifetime.last)
        {
            return Err(Refusal::InvalidFact("invalid retained BLAS owner".into()));
        }
        if arch == "sm_121" && owner.lt_handles > 0 && owner.cublas_handles == 0 {
            return Err(Refusal::InvalidFact("SM121 Lt-only process lacks a qualified first-use owner (#1463)".into()));
        }
        let bytes = owner.cublas_handles.checked_mul(per_handle)
            .ok_or(Refusal::Overflow("retained cuBLAS handles"))?;
        if bytes > 0 {
            out.push(Component { activation: owner.activation.clone(), bytes: ByteCurve::fixed(bytes, maximum),
                kind: ComponentKind::Allowance(Allowance { location: owner.location, bytes,
                    lifetime: owner.lifetime, assumed: false,
                    source: format!("cuBLAS handles: {} ({} x {per_handle} B, {arch} driver13020, 2026-11-07, process {})",
                        owner.name, owner.cublas_handles, owner.process) }) });
        }
    }
    let initialization = if arch == "sm_120" {
        vec![("BLAS device initialization (cuBLAS/cuBLASLt shared)", 2 << 20, false)]
    } else {
        vec![("cuBLAS first use", 158_674_944, false), ("cuBLASLt first plan", 201_461_760, true)]
    };
    for (location, process) in owners.iter().map(|owner| (owner.location, owner.process.as_str()))
        .collect::<BTreeSet<_>>() {
        for &(source, bytes, lt_only) in &initialization {
            for &phase in order.phases() {
                // Library residency survives handle destruction. Union selected
                // triggers per process/context; independent workers pay separately.
                let clauses = owners.iter().filter(|owner| owner.location == location && owner.process == process
                    && if lt_only { owner.lt_handles > 0 } else { owner.cublas_handles > 0 || owner.lt_handles > 0 })
                    .filter(|owner| order.at_or_before(owner.lifetime.first, phase))
                    .flat_map(|owner| owner.activation.clauses.iter().cloned()).collect::<BTreeSet<_>>();
                if !clauses.is_empty() {
                    out.push(Component { activation: Activation { clauses: clauses.into_iter().collect() },
                        bytes: ByteCurve::fixed(bytes, maximum),
                        kind: ComponentKind::Allowance(Allowance { location, bytes,
                            lifetime: Lifetime { first: phase, last: phase }, assumed: false,
                            source: format!("{source} ({arch} driver13020, LAZY, 2026-11-07, process {process})") }) });
                }
            }
        }
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy)]
pub struct SelectorOwners {
    pub location: Location,
    pub capacity: usize,
    pub vocab: usize,
    pub device_sampling: bool,
}

impl SelectorOwners {
    pub fn allocations(self, prefix: &str, lifetime: Lifetime) -> Result<Vec<Allocation>, Refusal> {
        if prefix.is_empty() || self.location.domain != Domain::Device
            || self.location.device == Device::Host || self.capacity == 0 || self.vocab == 0
        {
            return Err(Refusal::InvalidFact("invalid token selector geometry or identity".into()));
        }
        let bytes = self.capacity.checked_mul(12)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(Refusal::Overflow("token selector output"))?;
        let mut owners = vec![
            Allocation::buffer(format!("{prefix}.output"), self.location, bytes, lifetime,
                Category::Workspace, "token selector"),
            Allocation::buffer(format!("{prefix}.landing"), Location::PINNED, bytes, lifetime,
                Category::Staging, "token selector"),
        ];
        if self.device_sampling {
            // The retained sampler handles at most 128 rows per wave, even
            // when a selector's independent greedy output plane is larger.
            for owner in cuteafd_core::target_sampler_allocations(self.capacity.min(128), self.vocab)
                .map_err(|e| Refusal::InvalidFact(e.into()))?
            {
                owners.push(Allocation::buffer(format!("{prefix}.sampler.{}", owner.id),
                    if owner.pinned { Location::PINNED } else { self.location },
                    u64::try_from(owner.bytes).map_err(|_| Refusal::Overflow("target sampler owner"))?,
                    lifetime, Category::Workspace, "target sampler"));
            }
        }
        Ok(owners)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HostPrefixOwners {
    pub quota_bytes: u64,
    pub chunk_bytes: u64,
}

impl HostPrefixOwners {
    pub fn allocations(self, prefix: &str, lifetime: Lifetime) -> Result<Vec<Allocation>, Refusal> {
        if prefix.is_empty() {
            return Err(Refusal::InvalidFact("empty host prefix owner identity".into()));
        }
        if self.quota_bytes == 0 { return Ok(vec![]); }
        if self.chunk_bytes == 0 || self.chunk_bytes > self.quota_bytes
            || usize::try_from(self.chunk_bytes).is_err()
            || self.quota_bytes / self.chunk_bytes > u32::MAX as u64
        {
            return Err(Refusal::InvalidFact("invalid host prefix chunks".into()));
        }
        // SlabPool retains only whole chunks; the explicit quota remainder
        // is neither an allocation nor a view into another owner.
        Ok((0..self.quota_bytes / self.chunk_bytes).map(|index|
            Allocation::buffer(format!("{prefix}.{index}"), Location::PINNED, self.chunk_bytes,
                lifetime, Category::Prefix, "retained host prefix chunk")).collect())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ServingOwners {
    pub selector: SelectorOwners,
    pub prefix: HostPrefixOwners,
}

impl ServingOwners {
    pub const LIFETIME: Lifetime = Lifetime { first: Phase::Warmup, last: Phase::Growth };

    pub fn allocations(self) -> Result<Vec<Allocation>, Refusal> {
        let mut owners = self.selector.allocations("selector", Self::LIFETIME)?;
        owners.extend(self.prefix.allocations("host.prefix", Self::LIFETIME)?);
        Ok(owners)
    }

    pub fn components(self, maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        if maximum_pages == 0 {
            return Err(Refusal::InvalidFact("serving owners require a positive pool domain".into()));
        }
        Ok(self.allocations()?.into_iter().map(|owner| {
            let bytes = ByteCurve::fixed(owner.bytes, maximum_pages);
            Component::allocation(owner, bytes, false, maximum_pages)
        }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blas_counts_and_phase_activation_share_initialization_without_workspace_overlap() {
        let order = PhaseOrder::TARGET_THEN_DRAFT;
        let target = BlasOwners { name: "target".into(), process: "coordinator".into(), location: Location::coordinator(0),
            cublas_handles: 2, lt_handles: 2, lifetime: Lifetime::TARGET,
            activation: Activation::always() };
        let draft = BlasOwners { name: "draft".into(), cublas_handles: 1, lt_handles: 1,
            lifetime: Lifetime { first: Phase::Warmup, last: Phase::Ready },
            activation: Activation::choice(0, 1), ..target.clone() };
        let lt = BlasOwners { name: "media Lt".into(), process: "coordinator".into(), location: Location::coordinator(1),
            cublas_handles: 0, lt_handles: 4, lifetime: Lifetime { first: Phase::Load, last: Phase::Load },
            activation: Activation::choice(0, 1) };
        let owners = [target.clone(), draft, lt];
        let empty = Facts { phase_order: order, allocations: vec![], deferred: vec![],
            sequential: vec![], reservations: vec![], allowances: vec![],
            resident_weights: vec![], driver_page_tables: vec![] };
        let mut contract = Contract::fixed(64, 1, 2, empty);
        contract.variables = vec![Variable::Encoder(vec![0, 1]),
            Variable::Prefill(vec![PrefillShape { rows: 64, waves: 1 }]), Variable::Lanes(vec![1])];
        contract.components = blas_components("sm_120", &owners, 1, order).unwrap();
        for selected in [0, 1] {
            let facts = contract.facts(&[selected, 0, 0], 1).unwrap();
            for &phase in order.phases() {
                let bytes = |rank| facts.allowances.iter().filter(|owner|
                    owner.location == Location::coordinator(rank) && owner.lifetime.contains(phase, order))
                    .map(|owner| owner.bytes).sum::<u64>();
                let target_live = target.lifetime.contains(phase, order);
                let draft_live = selected == 1 && matches!(phase, Phase::Warmup | Phase::Ready);
                assert_eq!(bytes(0), (if target_live { 128 << 20 } else { 0 })
                    + (if draft_live { 64 << 20 } else { 0 })
                    + (if phase != Phase::Load { 2 << 20 } else { 0 }));
                // Lt has zero per-handle price; initialization remains after its handle drops.
                assert_eq!(bytes(1), if selected == 1 { 2 << 20 } else { 0 });
            }
            assert!(facts.allocations.is_empty(), "explicit Lt scratch must not be invented");
        }
        assert!(blas_components("sm_90", &owners, 1, order).is_err());
        assert!(blas_components("sm_121", &owners, 1, order).is_err(), "coordinator locations cannot use Spark prices");
        assert!(blas_components("sm_120", &[target.clone(), target.clone()], 1, order).is_err());
        assert!(blas_components("sm_120", &[BlasOwners { cublas_handles: u64::MAX, ..target }], 1, order).is_err());
    }

    #[test]
    fn spark_blas_separates_retained_storage_process_first_use_and_lt_first_plan() {
        let order = PhaseOrder::DRAFT_THEN_TARGET;
        let location = Location { device: Device::Spark(0), domain: Domain::Managed };
        let media = BlasOwners { name: "vision".into(), process: "media worker".into(), location,
            cublas_handles: 1, lt_handles: 0,
            lifetime: Lifetime { first: Phase::Load, last: Phase::Load }, activation: Activation::always() };
        let target = BlasOwners { name: "target".into(), process: "expert worker".into(), location,
            cublas_handles: 2, lt_handles: 0, lifetime: Lifetime::TARGET, activation: Activation::always() };
        let draft = BlasOwners { name: "draft head".into(), cublas_handles: 1, lt_handles: 1,
            lifetime: Lifetime { first: Phase::DraftLoad, last: Phase::Ready },
            activation: Activation::choice(0, 1), ..target.clone() };
        let owners = [media.clone(), target.clone(), draft];
        let empty = Facts { phase_order: order, allocations: vec![], deferred: vec![],
            sequential: vec![], reservations: vec![], allowances: vec![],
            resident_weights: vec![], driver_page_tables: vec![] };
        let mut contract = Contract::fixed(64, 1, 1, empty);
        contract.variables = vec![Variable::Encoder(vec![0, 1]),
            Variable::Prefill(vec![PrefillShape { rows: 64, waves: 1 }]), Variable::Lanes(vec![1])];
        contract.components = blas_components("sm_121", &owners, 1, order).unwrap();
        for selected in [0, 1] {
            let facts = contract.facts(&[selected, 0, 0], 1).unwrap();
            for &phase in order.phases() {
                let active: Vec<_> = facts.allowances.iter().filter(|owner| owner.lifetime.contains(phase, order)).collect();
                let handles = active.iter().filter(|owner| owner.source.starts_with("cuBLAS handles:"))
                    .map(|owner| owner.bytes).sum::<u64>();
                let draft_used = selected == 1 && order.at_or_before(Phase::DraftLoad, phase);
                let target_used = order.at_or_before(Phase::PostTargetKv, phase);
                assert_eq!(handles, 67_508_224 * (u64::from(phase == Phase::Load)
                    + 2 * u64::from(target_used) + u64::from(draft_used && phase != Phase::Growth)));
                let first_use = active.iter().filter(|owner| owner.source.starts_with("cuBLAS first use"))
                    .map(|owner| owner.bytes).sum::<u64>();
                assert_eq!(first_use, 158_674_944 * (1 + u64::from(draft_used || target_used)),
                    "one first-use owner per process, persisting after media handle destruction");
                let first_plan = active.iter().filter(|owner| owner.source.starts_with("cuBLASLt first plan"))
                    .map(|owner| owner.bytes).sum::<u64>();
                assert_eq!(first_plan, if draft_used { 201_461_760 } else { 0 });
                assert!(active.iter().all(|owner| !owner.assumed));
            }
            assert!(facts.allocations.is_empty(), "caller-owned Lt scratch stays disjoint");
        }
        let lt_only = BlasOwners { lt_handles: 1, cublas_handles: 0, ..media.clone() };
        assert!(blas_components("sm_121", &[lt_only], 1, order).is_err());
        assert!(blas_components("sm_121", &[BlasOwners { process: String::new(), ..media }], 1, order).is_err());
    }

    #[test]
    fn selector_geometry_preserves_domains_capacity_and_independent_sampler_owners() {
        for capacity in [1, 64, 128, 256] {
            for vocab in [1, 33, 154880] {
                for device_sampling in [false, true] {
                    let spec = SelectorOwners { location: Location::coordinator(1), capacity, vocab, device_sampling };
                    let owners = spec.allocations("target", Lifetime::RESIDENT).unwrap();
                    assert_eq!(owners.len(), if device_sampling {24} else {2});
                    assert_eq!(owners[0].bytes, capacity as u64 * 12);
                    assert_eq!(owners[0].location, Location::coordinator(1));
                    assert_eq!(owners[1].location, Location::PINNED);
                    assert!(owners.iter().all(|a| a.view.is_none() && a.lifetime == Lifetime::RESIDENT));
                    let sampler = cuteafd_core::target_sampler_allocations(capacity.min(128), vocab).unwrap();
                    for expected in sampler.into_iter().filter(|_| device_sampling) {
                        let actual = owners.iter().find(|a| a.id == format!("target.sampler.{}", expected.id)).unwrap();
                        assert_eq!(actual.bytes, expected.bytes as u64);
                        assert_eq!(actual.location, if expected.pinned {Location::PINNED} else {spec.location});
                    }
                }
            }
        }
        let base = SelectorOwners {location: Location::coordinator(0), capacity: 64, vocab: 33, device_sampling: true};
        for bad in [SelectorOwners {capacity: 0, ..base}, SelectorOwners {vocab: 0, ..base},
            SelectorOwners {capacity: usize::MAX, ..base}, SelectorOwners {vocab: usize::MAX, ..base},
            SelectorOwners {location: Location::PINNED, ..base},
            SelectorOwners {location: Location {device: Device::Host, domain: Domain::Device}, ..base}]
        {
            assert!(bad.allocations("selector", Lifetime::RESIDENT).is_err());
        }
        assert!(base.allocations("", Lifetime::RESIDENT).is_err());
    }

    #[test]
    fn prefix_charges_only_whole_retained_chunks_and_composes_without_policy() {
        let prefix = HostPrefixOwners {quota_bytes: 2 * 4096 + 17, chunk_bytes: 4096};
        let owners = prefix.allocations("host.prefix", ServingOwners::LIFETIME).unwrap();
        assert_eq!(owners.len(), 2);
        assert_eq!(owners.iter().map(|a| a.bytes).sum::<u64>(), 8192);
        assert_eq!(owners[1].id, "host.prefix.1");
        assert!(owners.iter().all(|a| a.location == Location::PINNED && a.view.is_none()));
        for chunk_bytes in [0, 4096] {
            assert!(HostPrefixOwners {quota_bytes: 0, chunk_bytes}
                .allocations("host.prefix", ServingOwners::LIFETIME).unwrap().is_empty());
        }
        for bad in [HostPrefixOwners {quota_bytes: 1, chunk_bytes: 0},
            HostPrefixOwners {quota_bytes: 1, chunk_bytes: 2},
            HostPrefixOwners {quota_bytes: u64::MAX, chunk_bytes: 1}]
        {
            assert!(bad.allocations("host.prefix", ServingOwners::LIFETIME).is_err());
        }
        assert!(prefix.allocations("", ServingOwners::LIFETIME).is_err());
        let serving = ServingOwners {selector: SelectorOwners {location: Location::coordinator(0),
            capacity: 64, vocab: 33, device_sampling: true}, prefix};
        let exact = serving.allocations().unwrap();
        assert_eq!(exact.len(), 26);
        let contract = Contract { floor_fallback: None,resident_weight_limit: None, spark_reply_formats: None, phase_order: crate::placement::planner::PhaseOrder::TARGET_THEN_DRAFT,page_rows: 64, pool_page_multiple: 1, maximum_pages: 16, gpus: 1,
            variables: vec![Variable::Prefill(vec![PrefillShape {rows: 64, waves: 1}]), Variable::Lanes(vec![1])],
            requires: vec![], components: serving.components(16).unwrap()};
        for pages in [1, 8, 16] {
            assert_eq!(contract.facts(&[0, 0], pages).unwrap().allocations, exact);
        }
        assert!(serving.components(0).is_err());
    }
}
