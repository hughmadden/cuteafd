//! Routed-expert pieces every family uses: layer identity and native resident
//! weights, per-wave execution and the host exchange, the EXL3 package path,
//! the expertd-native service, the exact FP8 expert packages and the TP2 RTX
//! expert halves (`rtx`).

pub(crate) mod execution;
pub(crate) mod exl3;
pub(crate) mod exlr8;
pub(crate) mod fp8;
pub(crate) mod layer;
pub(crate) mod local;
pub(crate) mod paired_load;
pub(crate) mod rtx;
pub(crate) mod service;
pub(crate) mod staging;
#[cfg(test)]
mod tp_gate;
#[cfg(test)]
mod tp_reference;

#[cfg(test)]
/// Original randomized experimental arm, retained for byte-identity tests.
fn load_in_rank_order<T>(layers: &[usize], rank: usize,
    mut load: impl FnMut(usize, usize) -> anyhow::Result<T>) -> anyhow::Result<Vec<T>> {
    let mut order: Vec<_> = (0..layers.len()).collect();
    let mut state = 0x6a09_e667_f3bc_c909u64.wrapping_add(rank as u64);
    for end in (1..order.len()).rev() {
        // SplitMix64 and rejection sampling make a reproducible Fisher-Yates
        // permutation without a process-global RNG or modulo bias.
        let bound = (end + 1) as u64;
        let threshold = bound.wrapping_neg() % bound;
        let slot = loop {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut value = state;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^= value >> 31;
            if value >= threshold { break (value % bound) as usize; }
        };
        order.swap(end, slot);
    }
    let mut weights: Vec<_> = (0..layers.len()).map(|_| None).collect();
    for index in order { weights[index] = Some(load(index, layers[index])?); }
    Ok(weights.into_iter().map(|weight| weight.expect("every layer loaded once")).collect())
}

type LayerShards = std::collections::BTreeMap<String, u64>;

fn shard_holders(value: &str) -> Option<std::collections::BTreeMap<String, String>> {
    let mut holders = std::collections::BTreeMap::new();
    for entry in value.split(';') {
        let (shard, holder) = entry.split_once('=')?;
        if shard.is_empty() || holder.is_empty() || shard.contains('/')
            || holder.contains('=') || entry.chars().any(char::is_whitespace)
            || holders.insert(shard.into(), holder.into()).is_some() { return None; }
    }
    (!holders.is_empty()).then_some(holders)
}

fn holder_order(layers: &[usize], shards: &[LayerShards],
    holders: &std::collections::BTreeMap<String, String>) -> Option<Vec<usize>> {
    use std::collections::{BTreeMap, VecDeque};
    if layers.len() != shards.len() { return None; }
    let mut files: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (index, sources) in shards.iter().enumerate() {
        // A boundary layer can straddle files. Keep its bank/fence lifetime
        // intact and assign it to the dominant payload file (lexical tie-break).
        if sources.is_empty() || sources.keys().any(|file| !holders.contains_key(file)) { return None; }
        let file = sources.iter().max_by(|(a, x), (b, y)| x.cmp(y).then_with(|| b.cmp(a)))?.0;
        files.entry(file).or_default().push(index);
    }
    let mut hosts: BTreeMap<&str, VecDeque<Vec<usize>>> = BTreeMap::new();
    for (file, indices) in files { hosts.entry(holders.get(file)?.as_str()).or_default().push_back(indices); }
    let mut order = Vec::with_capacity(layers.len());
    while order.len() < layers.len() {
        for groups in hosts.values_mut() {
            if let Some(indices) = groups.pop_front() { order.extend(indices); }
        }
    }
    Some(order)
}

/// Placement is optional; without a complete map retain plain layer order.
/// Admission must finish before invoking this loader. Results retain caller slots.
fn load_in_holder_order<T>(layers: &[usize], mut shards: impl FnMut(usize) -> LayerShards,
    mut load: impl FnMut(usize, usize) -> anyhow::Result<T>) -> anyhow::Result<Vec<T>> {
    let order = std::env::var("CUTEAFD_SHARD_HOLDERS").ok().and_then(|value| shard_holders(&value))
        .and_then(|holders| holder_order(layers, &layers.iter().map(|&layer| shards(layer)).collect::<Vec<_>>(), &holders));
    tracing::info!(holder_interleaved = order.is_some(), layer_count = layers.len(), "expert checkpoint load order");
    load_in_order(layers, order.unwrap_or_else(|| (0..layers.len()).collect()), &mut load)
}

fn load_in_order<T>(layers: &[usize], order: Vec<usize>,
    mut load: impl FnMut(usize, usize) -> anyhow::Result<T>) -> anyhow::Result<Vec<T>> {
    let mut weights: Vec<_> = (0..layers.len()).map(|_| None).collect();
    for index in order { weights[index] = Some(load(index, layers[index])?); }
    Ok(weights.into_iter().map(|weight| weight.expect("every layer loaded once")).collect())
}

fn catalog_layer_shards(catalog: &cuteafd_loader::OfficialV41Catalog, layer: usize) -> LayerShards {
    let marker = format!(".layers.{layer}.mlp.experts.");
    let mut shards = LayerShards::new();
    for tensor in catalog.tensors().iter().filter(|tensor| tensor.metadata.name.contains(&marker)) {
        *shards.entry(tensor.shard.clone()).or_default() += tensor.metadata.byte_length;
    }
    shards
}

#[cfg(test)]
mod load_order_tests {
    use super::load_in_rank_order;
    use anyhow::Result;
    use std::{collections::BTreeSet, fs::OpenOptions, os::unix::fs::FileExt};
    use cuteafd_loader::{formats::fp8_experts::{ExpertFormat, Fp8ExpertTensors, Fp8Projection, Slicing},
        plan::testing as fixture, RoutedExpertShape};

    #[test]
    fn holder_file_groups_round_robin_and_preserve_slots() -> Result<()> {
        use super::{holder_order, shard_holders, load_in_order, LayerShards};
        let holders = shard_holders("a=host1;b=host1;c=host2;d=host2;e=host3") .unwrap();
        let layers = [3, 8, 13, 21, 34, 55, 89];
        let sources: Vec<LayerShards> = ["a", "a", "b", "c", "d", "e", "e"].into_iter()
            .map(|file| [(file.to_string(), 10)].into_iter().collect()).collect();
        let expected = [0, 1, 3, 5, 6, 2, 4];
        assert_eq!(holder_order(&layers, &sources, &holders).unwrap(), expected);
        for _rank in 0..6 {
            let loaded = load_in_order(&layers, holder_order(&layers, &sources, &holders).unwrap(), |_, layer| Ok(layer))?;
            assert_eq!(loaded, layers);
        }
        let mut split = sources.clone();
        split[0].insert("c".into(), 11);
        assert_eq!(holder_order(&layers, &split, &holders).unwrap(), [1, 0, 3, 5, 6, 2, 4]);
        split[0].insert("missing".into(), 1);
        assert!(holder_order(&layers, &split, &holders).is_none());
        assert!(holder_order(&layers, &sources, &shard_holders("a=host1").unwrap()).is_none());
        assert_eq!(holder_order(&[], &[], &holders).unwrap(), Vec::<usize>::new());
        for value in ["", "a=", "=h", "a=h;a=h", "a=h;", "a/b=h", "a=h=h", "a= h"] {
            assert!(shard_holders(value).is_none(), "{value}");
        }
        Ok(())
    }

    #[test]
    fn rank_permutations_are_reproducible_and_preserve_sparse_layer_slots() -> Result<()> {
        let layers = [3, 8, 13, 21, 34, 55, 89, 144];
        let mut orders = BTreeSet::new();
        for rank in 0..6 {
            let mut order = Vec::new();
            let loaded = load_in_rank_order(&layers, rank, |index, layer| {
                assert_eq!(layers[index], layer);
                order.push(layer);
                Ok(layer)
            })?;
            assert_eq!(loaded, layers);
            let mut repeated = Vec::new();
            load_in_rank_order(&layers, rank, |_, layer| { repeated.push(layer); Ok(()) })?;
            assert_eq!(order, repeated);
            assert_ne!(order, layers);
            assert!(orders.insert(order));
        }
        assert!(load_in_rank_order::<()>(&[], 0, |_, _| unreachable!())?.is_empty());
        assert_eq!(load_in_rank_order(&[19], usize::MAX, |_, layer| Ok(layer))?, [19]);
        assert!(load_in_rank_order::<()>(&layers, 0, |_, _| anyhow::bail!("read failure")).is_err());
        Ok(())
    }

    // Distinct finite payloads expose layer/rank/offset mix-ups that zero-filled
    // fixture files would hide. MCG and ModelOpt scalars retain valid values.
    fn fill_payloads(root: &std::path::Path) -> Result<()> {
        let path = root.join("model-00001-of-00001.safetensors");
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let mut size = [0; 8];
        file.read_exact_at(&mut size, 0)?;
        let size = u64::from_le_bytes(size);
        let mut bytes = vec![0; size as usize];
        file.read_exact_at(&mut bytes, 8)?;
        let header: serde_json::Value = serde_json::from_slice(&bytes)?;
        for (name, entry) in header.as_object().unwrap() {
            let start = entry["data_offsets"][0].as_u64().unwrap();
            let end = entry["data_offsets"][1].as_u64().unwrap();
            let tag = name.bytes().fold(0u8, |v, b| v.wrapping_add(b));
            let mut payload: Vec<_> = (0..end-start).map(|i| tag.wrapping_add(i as u8) & 31).collect();
            if name.ends_with(".mcg") { payload.copy_from_slice(&0xcbac1fedu32.to_le_bytes()); }
            if name.ends_with(".input_scale") || name.ends_with(".weight_scale_2") {
                payload.copy_from_slice(&1.0f32.to_le_bytes());
            }
            file.write_all_at(&payload, 8 + size + start)?;
        }
        Ok(())
    }

    #[test]
    fn shuffled_native_checkpoint_weights_and_admission_are_byte_identical() -> Result<()> {
        let shape = RoutedExpertShape { layers: 5, first_layer: 1, experts: 2, topk: 1,
            hidden: 128, intermediate: 768, draft_stages: 0, draft_experts: 0 };
        let layers = [1, 3, 4];
        for format in [ExpertFormat::Fp8Block128, ExpertFormat::Mxfp4, ExpertFormat::Nvfp4, ExpertFormat::Bf16] {
            let root = tempfile::tempdir()?;
            let mut tensors = Vec::new();
            for layer in 1..5 { for expert in 0..shape.experts {
                for (projection, n, k) in [("gate", 768, 128), ("up", 768, 128), ("down", 128, 768)] {
                    let name = format!("model.layers.{layer}.mlp.experts.{expert}.{projection}_proj");
                    tensors.extend(match format {
                        ExpertFormat::Fp8Block128 => fixture::fp8(&name, n, k, None),
                        ExpertFormat::Mxfp4 => fixture::mxfp4(&name, n, k),
                        ExpertFormat::Nvfp4 => fixture::nvfp4(&name, n, k),
                        ExpertFormat::Bf16 => vec![fixture::t(format!("{name}.weight"), "BF16", &[n, k])],
                    });
                }
            }}
            fixture::write_snapshot(root.path(), &serde_json::json!({}), &tensors, None);
            fill_payloads(root.path())?;
            let tensors = Fp8ExpertTensors::read(root.path(), shape)?;
            for rank in 0..6 {
                let load = |layer| -> Result<(Vec<u8>, [usize; 3])> {
                    let budget = tensors.layer_load(layer, 6, rank, Slicing::Blocks(128))?;
                    let mut bytes = Vec::new();
                    for expert in 0..shape.experts { for projection in Fp8Projection::ALL {
                        let (w, s) = tensors.slice_bytes_with(projection, 6, rank, Slicing::Blocks(128))?;
                        let (mut w, mut s) = (vec![0; w], vec![0; s]);
                        tensors.read_slice_with(layer, expert, projection, 6, rank, Slicing::Blocks(128),
                            &mut w, &mut s, &mut Vec::new())?;
                        bytes.extend(w); bytes.extend(s);
                    }}
                    Ok((bytes, [budget.resident, budget.pinned, budget.host]))
                };
                let sequential = layers.iter().map(|&layer| load(layer)).collect::<Result<Vec<_>>>()?;
                let shuffled = load_in_rank_order(&layers, rank, |_, layer| load(layer))?;
                assert_eq!(sequential, shuffled, "{format:?} rank {rank}");
                let holders = super::shard_holders("one=host1;two=host2").unwrap();
                let sources = ["one", "two", "one"].map(|file| [(file.into(), 1)].into_iter().collect());
                let order = super::holder_order(&layers, &sources, &holders).unwrap();
                let interleaved = super::load_in_order(&layers, order, |_, layer| load(layer))?;
                assert_eq!(sequential, interleaved, "holder {format:?} rank {rank}");
                let totals = |weights: &[(Vec<u8>, [usize; 3])]| weights.iter().fold([0; 3], |a, (_, b)|
                    [a[0] + b[0], a[1].max(b[1]), a[2].max(b[2])]);
                assert_eq!(totals(&sequential), totals(&shuffled));
            }
        }
        Ok(())
    }

    #[test]
    fn shuffled_exl3_residency_and_admission_are_byte_identical() -> Result<()> {
        let root = tempfile::tempdir()?;
        let mut config = fixture::glm5_config();
        for (key, value) in [("hidden_size", 128), ("moe_intermediate_size", 768),
            ("n_routed_experts", 2), ("num_experts_per_tok", 1), ("num_hidden_layers", 5)] {
            config[key] = value.into();
        }
        config["indexer_types"] = serde_json::json!(["full", "shared", "shared", "shared", "shared"]);
        config["mlp_layer_types"] = serde_json::json!(["dense", "sparse", "sparse", "sparse", "sparse"]);
        config["quantization_config"] = fixture::exl3_compact(4);
        let mut tensors = Vec::new();
        for layer in 1..5 { for expert in 0..2 {
            for (projection, n, k) in [("gate", 768, 128), ("up", 768, 128), ("down", 128, 768)] {
                tensors.extend(fixture::exl3(&format!("model.layers.{layer}.mlp.experts.{expert}.{projection}_proj"), n, k, 4));
            }
        }}
        fixture::write_snapshot(root.path(), &config, &tensors, None);
        fill_payloads(root.path())?;
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        let layers = [1, 3, 4];
        for rank in 0..6 {
            let load = |layer| -> Result<(Vec<Vec<u8>>, [usize; 4])> {
                let selection = super::layer::ExpertLayer::BackboneExl3Tp { layer, rank, world: 6 };
                let plan = super::exl3::planned_residency(&catalog, selection)?;
                let budget = super::exl3::Exl3Weights::plan(&catalog, selection)?;
                let mut buffers: Vec<_> = plan.buffers.iter().map(|b| {
                    let mut bytes = vec![0; b.bytes.max(16)];
                    for (dst, word) in bytes.chunks_exact_mut(4).zip(&b.initial_words) {
                        dst.copy_from_slice(&word.to_le_bytes());
                    }
                    bytes
                }).collect();
                let mut scratch = vec![0; budget.read_scratch_bytes];
                for (index, job) in plan.loads.iter().enumerate() {
                    let mut bytes = vec![0; job.bytes];
                    plan.read_into(&catalog, index, &mut bytes, &mut scratch)?;
                    for &(buffer, offset) in &job.destinations {
                        buffers[buffer][offset..offset + job.bytes].copy_from_slice(&bytes);
                    }
                }
                Ok((buffers, [budget.resident_bytes, budget.device_staging_bytes,
                    budget.pinned_host_bytes, budget.read_scratch_bytes]))
            };
            let sequential = layers.iter().map(|&layer| load(layer)).collect::<Result<Vec<_>>>()?;
            let shuffled = load_in_rank_order(&layers, rank, |_, layer| load(layer))?;
            assert_eq!(sequential, shuffled, "EXL3 rank {rank}");
            let holders = super::shard_holders("one=host1;two=host2").unwrap();
            let sources = ["one", "two", "one"].map(|file| [(file.into(), 1)].into_iter().collect());
            let order = super::holder_order(&layers, &sources, &holders).unwrap();
            let interleaved = super::load_in_order(&layers, order, |_, layer| load(layer))?;
            assert_eq!(sequential, interleaved, "holder EXL3 rank {rank}");
        }
        Ok(())
    }
}

/// Frozen compatibility set shared by native and EXL3 expert execution.
/// Launch capacity may vary; every decode variant must match canonical m1 sums.
#[derive(Debug)]
pub(crate) struct ExpertReductionPlan {
    decode: Vec<(usize, usize)>,
}

impl ExpertReductionPlan {
    pub(crate) fn new<S: Eq + std::fmt::Debug>(variants: &[(usize, S)]) -> anyhow::Result<Self> {
        use anyhow::{ensure, Context};
        let canonical = &variants
            .iter()
            .find(|(capacity, _)| *capacity == 1)
            .context("expert package needs canonical m1 reduction")?
            .1;
        ensure!(
            variants.iter().all(|(capacity, _)| *capacity > 0),
            "expert export capacity must be positive"
        );
        let maximum = variants
            .iter()
            .map(|(capacity, _)| *capacity)
            .max()
            .context("empty expert reduction package")?;
        let mut decode: Vec<_> = variants
            .iter()
            .enumerate()
            .filter(|(_, (capacity, signature))| *capacity <= 80 && signature == canonical)
            .map(|(index, (capacity, _))| (*capacity, index))
            .collect();
        decode.sort_unstable();
        ensure!(decode.last().is_some_and(|&(capacity, _)| capacity >= maximum.min(80)),
            "expert package has no export covering {} decode rows with canonical m1 reduction {canonical:?}; admitted capacity/recipes: {variants:?}; export a matching capacity",
            maximum.min(80));
        Ok(Self { decode })
    }

    /// One compiled-recipe check for all native families and routed-expert owners.
    pub(crate) fn native(
        library: &cuteafd_ffi::NativeLibrary,
        infos: impl IntoIterator<Item = cuteafd_ffi::V41ExpertInfo>,
    ) -> anyhow::Result<Self> {
        use anyhow::{ensure, Context};
        let variants = infos.into_iter().map(|info| {
            let recipe = library.native_expert_reduction(&info).with_context(|| {
                format!("native MXFP4 role {} m{} reduction export", info.role, info.capacity_rows)
            })?;
            if info.capacity_rows <= 80 {
                ensure!(recipe.plane_fold == 1 && recipe.implementation == 1,
                    "native MXFP4 role {} m{} decode requires ordered FP32 slice planes, got {recipe:?}",
                    info.role, info.capacity_rows);
            }
            Ok((info.capacity_rows as usize, recipe))
        }).collect::<anyhow::Result<Vec<_>>>()?;
        Self::new(&variants)
            .context("native MXFP4 decode differs from canonical m1 slice/fold recipe")
    }

    pub(crate) fn execution_index(
        &self,
        rows: usize,
        required: usize,
        capacities: impl Iterator<Item = usize>,
    ) -> anyhow::Result<usize> {
        use anyhow::{ensure, Context};
        ensure!(rows > 0, "expert live rows must be positive");
        if rows <= 80 {
            return self
                .decode
                .iter()
                .find(|&&(capacity, _)| capacity >= rows)
                .map(|&(_, index)| index)
                .context("decode exceeds fixed expert reduction plan");
        }
        capacities
            .enumerate()
            .find(|&(_, capacity)| capacity >= required)
            .map(|(index, _)| index)
            .context("no expert execution for admitted rows")
    }
}

#[cfg(test)]
mod reduction_tests {
    use super::ExpertReductionPlan;

    #[test]
    fn native_owned_state_order_freezes_serial_and_grouped_decode() {
        // Execution owns main, m1, then m80 (not ascending capacities).
        let variants = [(4096, 192), (1, 64), (80, 64)];
        let plan = ExpertReductionPlan::new(&variants).unwrap();
        for rows in 1..=80 {
            assert_eq!(
                plan.execution_index(rows, rows, variants.iter().map(|v| v.0))
                    .unwrap(),
                if rows == 1 { 1 } else { 2 }
            );
        }
        assert_eq!(
            plan.execution_index(81, 81, variants.iter().map(|v| v.0))
                .unwrap(),
            0
        );
        let small = [(16, 64), (1, 64)];
        let plan = ExpertReductionPlan::new(&small).unwrap();
        for rows in 1..=16 {
            assert_eq!(
                plan.execution_index(rows, rows, small.iter().map(|v| v.0))
                    .unwrap(),
                if rows == 1 { 1 } else { 0 }
            );
        }
        assert!(plan
            .execution_index(17, 17, small.iter().map(|v| v.0))
            .is_err());
        let serial = ExpertReductionPlan::new(&[(1, 64)]).unwrap();
        assert_eq!(serial.execution_index(1, 1, [1].into_iter()).unwrap(), 0);
    }

    #[test]
    fn shared_decode_selector_is_width_invariant_and_ignores_prefill_numerics() {
        let variants = [(1, 64), (16, 192), (80, 64), (4096, 192)];
        let plan = ExpertReductionPlan::new(&variants).unwrap();
        for rows in 1..=80 {
            let index = plan
                .execution_index(rows, rows, variants.iter().map(|v| v.0))
                .unwrap();
            assert_eq!(variants[index].1, 64);
        }
        assert_eq!(
            plan.execution_index(81, 81, variants.iter().map(|v| v.0))
                .unwrap(),
            3
        );
        assert!(plan
            .execution_index(0, 1, variants.iter().map(|v| v.0))
            .is_err());
        assert!(ExpertReductionPlan::new(&[(1, 64), (80, 192), (4096, 192)]).is_err());
        assert!(ExpertReductionPlan::new(&[(80, 64)]).is_err());
        // Metadata changes after admission cannot mutate the owned compatibility set.
        let mut changed = variants;
        changed[2].1 = 192;
        assert_eq!(
            plan.execution_index(80, 80, changed.iter().map(|v| v.0))
                .unwrap(),
            2
        );
        assert!(ExpertReductionPlan::new(&changed).is_err());
    }
}
