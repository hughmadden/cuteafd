//! Native EXLR8 dense records and their exact allocation requests, without
//! allocating or reading tensor bytes. The coordinator loader binds exactly
//! these requests, and admission charges the arena built from them.
use super::{allocation, DenseAxis, DenseComponent, DenseEncoding, DensePlan, DenseRecord,
    DenseSelection, DenseSlice, DenseTransform, KvBLayout, ShardSide};
use anyhow::{ensure, Context, Result};
use std::collections::{BTreeMap, HashMap, HashSet};

/// How one native projection runs: its encoding and this rank's slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeProjection {
    pub encoding: DenseEncoding,
    /// Logical `[output, input]` of this rank's slice.
    pub shape: [usize; 2],
    pub side: ShardSide,
    pub rotation_id: usize,
}

/// One layer's native dense slices on one rank, in record order.
pub struct LayerRecords {
    pub layer: usize,
    pub projections: HashMap<&'static str, (NativeProjection, DenseSlice)>,
    /// Absent for KDA and for attention assigned to the other coordinator.
    pub kv_b: Option<(DenseRecord, KvBLayout)>,
    /// Non-projection operands, with the same conversion recipe admission used.
    pub plain: HashMap<String, (DenseComponent, DenseTransform)>,
}

/// The EXLR8 checkpoint's dense records for one rank of `world` coordinators.
pub struct NativeDenseRecords {
    pub layers: Vec<LayerRecords>,
    pub norm: DenseComponent,
    pub plan: DensePlan,
}

impl NativeDenseRecords {
    pub fn new(snapshot: &std::path::Path, layers: usize, world: usize, rank: usize,
        whole_mla: &BTreeMap<usize, usize>) -> Result<Self> {
        let records = DenseRecord::inventory(snapshot)?;
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
        let selection = DenseSelection::from_hf(&config)?;
        Self::from_records(&records, &selection, layers, world, rank, whole_mla)
    }

    pub fn from_records(records: &[DenseRecord], selection: &DenseSelection, layers: usize, world: usize,
        rank: usize, whole_mla: &BTreeMap<usize, usize>) -> Result<Self> {
        // Selection precedes both admission and loading: external DFlash2 never
        // makes a checkpoint's native MTP operands resident.
        let kept = selection.backbone(records, layers)?;
        let plan = DensePlan::for_rank_owned(&kept, world, rank, whole_mla)?;
        let by_name: HashMap<&str, &DenseSlice> = plan.tensors.iter().map(|s| (s.tensor.as_str(), s)).collect();
        let record = |name: &str| records.iter().find(|r| r.tensor == name);
        let mut out = Vec::with_capacity(layers);
        for layer in 0..layers {
            let prefix = selection.layer_prefix(layer)?;
            let mut projections = HashMap::new();
            let mut projected = HashSet::new();
            for (stem, path) in DenseSelection::PROJECTIONS {
                let tensor = format!("{prefix}.{path}.weight");
                let Some(full) = record(&tensor) else { continue };
                // The shared plan already excludes rank-zero and whole-MLA
                // operands owned by the other rank.
                let Some(slice) = by_name.get(tensor.as_str()) else { continue };
                projected.insert(tensor.clone());
                let channels = slice.channels;
                let shape = match slice.axis {
                    DenseAxis::Input => [full.shape[0], channels],
                    _ => [channels, full.shape[1]],
                };
                projections.insert(stem, (NativeProjection { encoding: full.encoding, shape, side: full.side,
                    rotation_id: full.rotation_id }, (*slice).clone()));
            }
            let kv_b_name = format!("{prefix}.self_attn.kv_b_proj.weight");
            let kv_b = by_name.get(kv_b_name.as_str()).map(|slice| -> Result<_> {
                let full = record(&kv_b_name).context("native kv_b source")?;
                ensure!(full.encoding == DenseEncoding::BlockFp8, "{kv_b_name}: native kv_b is block FP8");
                Ok((full.clone(), slice.absorbed.clone().context("native kv_b has no planned absorption layout")?))
            }).transpose()?;
            let mut plain = HashMap::new();
            for slice in plan.tensors.iter().filter(|s| s.tensor.starts_with(&format!("{prefix}."))
                && s.absorbed.is_none() && !projected.contains(&s.tensor)) {
                let full = record(&slice.tensor).context("native dense source record")?;
                ensure!(full.native_blocks == 0 && slice.components.len() == 1,
                    "{}: no native projection role", slice.tensor);
                plain.insert(slice.tensor.clone(), (slice.components[0].clone(), slice.transform));
            }
            out.push(LayerRecords { layer, projections, kv_b, plain });
        }
        let norm_name = selection.norm_name();
        // Keeping a source descriptor on rank 1 allocates no device memory;
        // only rank 0 loads the final norm and runs the head.
        let norm = record(&norm_name).with_context(|| format!("{norm_name}: missing final norm"))?;
        ensure!(norm.components.len() == 1 && norm.ownership.transform == DenseTransform::Identity,
            "{norm_name}: native final norm recipe");
        let norm = norm.components[0].clone();
        Ok(Self { layers: out, norm, plan })
    }
}

/// Named plain records may land directly into one kernel's packed operand.
/// Total allocator bytes must equal the original separately admitted records;
/// logical bytes are contiguous and no unpacked resident copies are allocated.
#[derive(Debug, PartialEq, Eq)]
pub struct PlainPack {
    pub name: String,
    pub components: Vec<String>,
}

/// The packed allocation bytes and each component's offset inside it.
pub fn packed_plain_layout(records: &LayerRecords, pack: &PlainPack) -> Result<(usize, Vec<usize>)> {
    ensure!(!pack.name.is_empty() && !pack.components.is_empty(), "empty native plain pack");
    let mut unique = HashSet::new();
    let (mut logical, mut admitted) = (0usize, 0usize);
    let mut offsets = Vec::new();
    for name in &pack.components {
        ensure!(unique.insert(name), "duplicate native plain pack component {name}");
        let (component, transform) = records.plain.get(name).context("native plain pack component missing")?;
        offsets.push(logical);
        logical = logical.checked_add(transform.resident_bytes(component)?).context("native plain pack extent")?;
        admitted = admitted.checked_add(transform.allocation_bytes(component)?).context("native plain pack admission")?;
    }
    ensure!(allocation(logical) == admitted, "native plain pack changes admitted allocation bytes");
    Ok((admitted, offsets))
}

/// The four absorbed kv_b operand allocations, in `ABSORBED_NAMES` order.
pub fn absorbed_bytes(layout: &KvBLayout) -> [usize; 4] {
    layout.allocations.map(allocation)
}

/// Owner names of the absorbed kv_b operands.
pub const ABSORBED_NAMES: [&str; 4] = ["w_uk_fp8", "w_uk_scale", "w_uv_fp8", "w_uv_scale"];

/// Operand key of a native projection component.
pub fn key(projection: &str, component: &str) -> String { format!("n.{projection}.{component}") }

/// Validates the packs and returns them with their layouts, rejecting name collisions.
pub fn plain_pack_layouts<'a>(records: &LayerRecords, packs: &'a [PlainPack])
    -> Result<(Vec<(&'a PlainPack, (usize, Vec<usize>))>, HashSet<&'a str>)> {
    let mut consumed = HashSet::new();
    let mut outputs = HashSet::new();
    let layouts = packs.iter().map(|pack| {
        ensure!(outputs.insert(&pack.name) && !records.plain.contains_key(&pack.name)
            && !pack.name.starts_with("n.") && !pack.name.starts_with("w_"), "native plain pack output collision");
        for name in &pack.components { ensure!(consumed.insert(name.as_str()), "native plain record packed twice"); }
        Ok((pack, packed_plain_layout(records, pack)?))
    }).collect::<Result<Vec<_>>>()?;
    Ok((layouts, consumed))
}

/// Every resident allocation one layer makes, in load order, named as the
/// coordinator binds them (`layers.{layer}.{owner}`). Projections land per
/// component, unpacked plain records land alone, packs become one allocation,
/// and an owned kv_b absorbs into its four operands.
pub fn layer_requests(records: &LayerRecords, packs: &[PlainPack]) -> Result<Vec<(String, usize)>> {
    let (layouts, consumed) = plain_pack_layouts(records, packs)?;
    let mut out = Vec::new();
    for (stem, (projection, slice)) in &records.projections {
        ensure!(!matches!(projection.encoding, DenseEncoding::F32 | DenseEncoding::F16 | DenseEncoding::RowFp8),
            "{}: no native dense kernel for {:?}", slice.tensor, projection.encoding);
        for component in &slice.components {
            out.push((key(stem, &component.suffix), slice.transform.allocation_bytes(component)?));
        }
    }
    for (tensor, (component, transform)) in &records.plain {
        if !consumed.contains(tensor.as_str()) {
            out.push((tensor.clone(), transform.allocation_bytes(component)?));
        }
    }
    for (pack, (bytes, _)) in layouts {
        out.push((pack.name.clone(), bytes));
    }
    if let Some((_, layout)) = &records.kv_b {
        out.extend(ABSORBED_NAMES.into_iter().zip(absorbed_bytes(layout)).map(|(name, bytes)| (name.into(), bytes)));
    }
    Ok(out.into_iter().map(|(owner, bytes)| (format!("layers.{}.{owner}", records.layer), bytes)).collect())
}

/// The native vocabulary records (embedding, then head) for `world` coordinators,
/// validated as unrotated output-major row-FP8 `[vocabulary, hidden]` with
/// `fp8` payload and `scale` planes. The coordinator's vocabulary loader
/// plans exactly these records.
pub fn vocabulary_records(records: &[DenseRecord], selection: &DenseSelection,
    hidden: usize, vocabulary: usize) -> Result<Vec<DenseRecord>> {
    ensure!(hidden > 0 && vocabulary > 0, "native vocabulary has empty dimensions");
    let mut selected = Vec::new();
    for name in selection.vocabulary_names() {
        let mut matches = records.iter().filter(|r| r.tensor == name);
        let record = matches.next().with_context(|| format!("missing native vocabulary {name}"))?;
        ensure!(matches.next().is_none(), "duplicate native vocabulary {name}");
        ensure!(record.encoding == DenseEncoding::RowFp8 && record.shape == [vocabulary, hidden]
            && record.side == ShardSide::Output && record.rotation.is_none() && record.rotation_id == 0,
            "native vocabulary {name} requires unrotated output-major row-FP8");
        let payload = vocabulary.checked_mul(hidden).context("native vocabulary extent overflow")?;
        let scales = vocabulary.checked_mul(4).context("native vocabulary scale extent overflow")?;
        ensure!(record.components.len() == 2, "native vocabulary component count");
        for (component, suffix, dtype, shape, bytes) in [
            (&record.components[0], "fp8", "U8", vec![vocabulary, hidden], payload),
            (&record.components[1], "scale", "F32", vec![vocabulary], scales),
        ] {
            ensure!(component.suffix == suffix && component.dtype == dtype && component.shape == shape
                && component.source.bytes == bytes, "native vocabulary {name} component {suffix} layout");
            component.source.offset.checked_add(bytes as u64).context("native vocabulary file extent overflow")?;
        }
        // Validate the complete native partition even on the non-owning rank.
        record.slice(0, vocabulary)?;
        selected.push(record.clone());
    }
    Ok(selected)
}

/// One native row-FP8 vocabulary block's coverage of its two components
/// (payload then scales): each component's blocks must tile it in order.
pub fn row_coverage(slice: &DenseSlice) -> Result<()> {
    ensure!(slice.components.len() == 2 && !slice.units.is_empty(), "native vocabulary partition");
    let mut covered = [0usize; 2];
    for (index, unit) in slice.units.iter().enumerate() {
        ensure!(unit.block == index && unit.source.len() == 2, "native vocabulary block order");
        for (i, source) in unit.source.iter().enumerate() {
            let component = &slice.components[i].source;
            let offset = usize::try_from(source.offset.checked_sub(component.offset)
                .context("native vocabulary component offset")?)?;
            let end = offset.checked_add(source.bytes).context("native vocabulary upload overflow")?;
            ensure!(source.bytes > 0 && source.path == component.path && offset == covered[i]
                && end <= component.bytes, "native vocabulary component coverage");
            covered[i] = end;
        }
    }
    ensure!(covered == [slice.components[0].source.bytes, slice.components[1].source.bytes],
        "native vocabulary incomplete component coverage");
    Ok(())
}

/// The two resident row-FP8 vocabulary planes, keyed by component suffix.
pub fn row_requests(slice: &DenseSlice) -> Result<Vec<(String, usize)>> {
    row_coverage(slice)?;
    slice.components.iter().map(|component| Ok((component.suffix.clone(),
        DenseTransform::Identity.allocation_bytes(component)?))).collect()
}
