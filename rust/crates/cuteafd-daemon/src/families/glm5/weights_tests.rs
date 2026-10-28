//! GLM 5.3 coordinator operands on a patterned synthetic checkpoint, through
//! the fake native DSO with a two-GPU head split. Every operand's bytes are
//! pinned to a golden digest taken from the loader before the landing
//! refactor, so the refactor (and push landings built on it) must reproduce
//! the mount load bit for bit. `w_ik` is excluded: the fake DSO does not
//! implement the FP8 dequantization that fills it.
use super::*;
use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
use cuteafd_loader::plan::testing as fixture;
use sha2::{Digest, Sha256};
use std::os::unix::fs::FileExt;

/// Deterministic, finite payloads: FP32 grids are positive normals, BF16 and
/// E4M3 bytes vary with the tensor and position.
fn fill(root: &std::path::Path) -> Result<()> {
    let catalog = cuteafd_loader::read_expert_catalog(root)?;
    for tensor in catalog.tensors() {
        let meta = &tensor.metadata;
        let seed = Sha256::digest(meta.name.as_bytes());
        let bytes: Vec<u8> = match meta.dtype {
            DType::F32 => (0..meta.byte_length / 4).flat_map(|i| {
                let v = 0.5 + ((i as u32).wrapping_mul(2654435761) ^ u32::from(seed[0])) as f32 / u32::MAX as f32;
                v.to_le_bytes()
            }).collect(),
            DType::Bf16 => (0..meta.byte_length / 2).flat_map(|i| {
                let v = (((i as u32).wrapping_mul(40503) ^ u32::from(seed[1])) % 2000) as f32 / 1000.0 - 1.0;
                ((v.to_bits() >> 16) as u16).to_le_bytes()
            }).collect(),
            _ => (0..meta.byte_length).map(|i| (i as u8).wrapping_mul(31) ^ seed[(i % 32) as usize] & 0x77).collect(),
        };
        std::fs::File::options().write(true).open(root.join(&tensor.shard))?
            .write_all_at(&bytes, meta.byte_offset)?;
    }
    Ok(())
}

fn digests(library: &NativeLibrary, layers: &[Vec<GlmLayer<'_>>], norm: &DeviceAllocation<'_>,
    head: &DeviceAllocation<'_>) -> Result<std::collections::BTreeMap<String, String>> {
    let mut out = std::collections::BTreeMap::new();
    let mut put = |key: String, buffer: CuteafdDeviceBuffer| -> Result<()> {
        let mut bytes = vec![0u8; buffer.bytes];
        library.copy_d2h(&mut bytes, buffer)?;
        out.insert(key, format!("{:x}", Sha256::digest(&bytes)));
        Ok(())
    };
    for (rank, layers) in layers.iter().enumerate() {
        for (index, layer) in layers.iter().enumerate() {
            for (name, allocation) in &layer.operands {
                if *name == "w_ik" { continue; }
                put(format!("rank{rank}.layer{index}.{name}"), allocation.buffer)?;
            }
        }
    }
    put("norm".into(), norm.buffer)?;
    put("head".into(), head.buffer)?;
    Ok(out)
}

/// Set CUTEAFD_WRITE_GLM5_OPERAND_GOLDEN=1 to regenerate (only from a loader
/// already known to be correct).
#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn head_split_operands_match_the_mount_golden() -> Result<()> {
    let fixture_dso = Fixture::build()?;
    let library = fixture_dso.load()?;
    for device in 0..2 { fixture_dso.configure_memory(&library, device, 64 << 30)?; }
    let root = tempfile::tempdir_in(fixture_dso.directory())?;
    let tensors = fixture::glm5_tensors(|name, n, k| (0..256).flat_map(|expert|
        fixture::fp8(&name.replace(".experts.0.", &format!(".experts.{expert}.")), n, k, None)).collect());
    fixture::write_snapshot(root.path(), &fixture::glm5_config(), &tensors, None);
    fill(root.path())?;
    let mut catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::None)?;
    let cfg = GlmDsaConfig::read(root.path())?;
    let stream = std::ptr::null_mut();
    let loader = GlmLoader { library: &library, catalog: &catalog, stream, device: 0,
        peers: vec![crate::shared::peer_split::RankDevice { device: 1, stream }], push: None };
    let (model, shares) = loader.model(&cfg, cfg.layers, None, false)?;
    let layers: Vec<Vec<GlmLayer<'_>>> = std::iter::once(model.layers).chain(shares).collect();
    let actual = digests(&library, &layers, &model.norm, model.head.bf16()?)?;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/glm5/weights_golden.json");
    if std::env::var_os("CUTEAFD_WRITE_GLM5_OPERAND_GOLDEN").is_some() {
        std::fs::write(&path, serde_json::to_vec_pretty(&actual)?)?;
    }
    let golden: std::collections::BTreeMap<String, String> = serde_json::from_slice(&std::fs::read(&path)?)?;
    assert_eq!(actual.keys().collect::<Vec<_>>(), golden.keys().collect::<Vec<_>>(), "operand inventory changed");
    for (key, digest) in &golden {
        assert_eq!(&actual[key], digest, "{key} differs from the mount golden");
    }
    Ok(())
}

/// The same operands pushed: the loader allocates and records class-2
/// operands, each GPU's plan replays through a ring into them, and the
/// deferred transforms run after delivery. Every operand matches the mount golden.
#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn pushed_head_split_operands_match_the_mount_golden() -> Result<()> {
    let fixture_dso = Fixture::build()?;
    let library = fixture_dso.load()?;
    for device in 0..2 { fixture_dso.configure_memory(&library, device, 64 << 30)?; }
    let root = tempfile::tempdir_in(fixture_dso.directory())?;
    let tensors = fixture::glm5_tensors(|name, n, k| (0..256).flat_map(|expert|
        fixture::fp8(&name.replace(".experts.0.", &format!(".experts.{expert}.")), n, k, None)).collect());
    fixture::write_snapshot(root.path(), &fixture::glm5_config(), &tensors, None);
    fill(root.path())?;
    let mut catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::None)?;
    let cfg = GlmDsaConfig::read(root.path())?;
    let stream = std::ptr::null_mut();
    let sink = std::rc::Rc::new(super::landing::PushSink::default());
    let loader = GlmLoader { library: &library, catalog: &catalog, stream, device: 0,
        peers: vec![crate::shared::peer_split::RankDevice { device: 1, stream }], push: Some(sink.clone()) };
    let (model, shares) = loader.model(&cfg, cfg.layers, None, false)?;
    let (records, deferred) = sink.take();
    let (sparks, world) = (6usize, 8usize);
    for gpu in 0..2 {
        use crate::shared::experts::rtx::dense::{DenseArena, RankLayout};
        let device = crate::shared::memory::device::Device { library: &library, id: gpu };
        let layout = RankLayout::new(records.clone(), Vec::new(), gpu)?;
        let checked = layout.checked(root.path(), world, sparks + gpu as usize)?;
        let expected = crate::shared::memory::weight_arena::ExpectedCoverage::from_plan(&checked, (sparks + gpu as usize) as u32)?;
        device.run(|| -> Result<()> {
            let ring = cuteafd_ffi::push_ring::RegisteredRing::new(&library, 8, cuteafd_ffi::push_ring::DEFAULT_SLOT_BYTES)?;
            let fake = crate::shared::experts::rtx::dense::testing::FakeRank::new(&checked, root.path(), (sparks + gpu as usize) as u32, &ring)?;
            let arena = DenseArena::new(device, layout, expected)?;
            crate::shared::memory::push_consumer::consume(device, arena, ring, fake, 8)
        })?;
    }
    for job in &deferred {
        crate::shared::memory::device::Device { library: &library, id: job.device }.run(|| job.run(&library, stream))?;
    }
    let layers: Vec<Vec<GlmLayer<'_>>> = std::iter::once(model.layers).chain(shares).collect();
    let actual = digests(&library, &layers, &model.norm, model.head.bf16()?)?;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/glm5/weights_golden.json");
    let golden: std::collections::BTreeMap<String, String> = serde_json::from_slice(&std::fs::read(&path)?)?;
    for (key, digest) in &golden {
        assert_eq!(&actual[key], digest, "pushed {key} differs from the mount golden");
    }
    Ok(())
}

#[test]
fn native_decode_prefetch_keeps_projection_order_before_budget_and_peer_has_no_head() {
    use crate::shared::l2_prefetch::{prefix, Range};
    let range = |id: usize| -> Range { (std::ptr::without_provenance::<c_void>(id), 1 << 20) };
    let expected = ["q_a", "kv_a", "q_b", "index_q", "index_k", "index_w", "o", "gate", "up", "down"];
    for rank in 0..2 {
        let base = 100 * (rank + 1);
        let ordered = decode_ranges(|_| None, |name| {
            let index = expected.iter().position(|&p| p == name).unwrap();
            vec![range(base + index)]
        });
        assert_eq!(ordered, (0..expected.len()).map(|i| range(base + i)).collect::<Vec<_>>());
        assert_eq!(prefix(&ordered, (2 << 20) + (1 << 19)),
            [range(base), range(base + 1), (range(base + 2).0, 1 << 19)]);
        let head = head_read_order(rank, range(1), || vec![range(2), range(3), range(4)]);
        if rank == 0 { assert_eq!(head, [range(1), range(2), range(3), range(4)]); }
        else { assert!(head.is_empty()); }
    }
    assert!(head_read_order(1, range(1), || panic!("rank 1 touched rank-0 head")).is_empty());
    // Ordinary FP8 ordering remains unchanged, including payload then scales.
    let names = ["input_norm", "w_qkv_a", "q_a_norm", "kv_a_norm", "w_q_b", "w_iq", "w_ik",
        "k_norm_w", "k_norm_b", "w_uk", "w_uv", "w_o", "post_norm", "gate", "gate.bias", "w_gate_up", "w_down"];
    let fp8 = |name: &str| match name {
        "w_qkv_a_fp8" => Some(range(10)), "w_qkv_a_scale" => Some(range(11)),
        _ => names.iter().position(|&n| n == name).map(|i| range(20 + i)),
    };
    assert_eq!(decode_ranges(fp8, |_| Vec::new()), crate::shared::l2_prefetch::operands(&names, fp8));
}

#[test]
fn exlr8_prefetch_enumerates_each_resident_encoding_in_read_order() {
    use crate::shared::l2_prefetch::Range;
    for rank in 0..2 {
        for components in [vec!["suh", "trellis", "svh"], vec!["q4", "s4"],
            vec!["fp8", "e8m0"], vec!["bf16"]] {
            let expected = components.iter().enumerate().map(|(i, _)|
                (std::ptr::without_provenance::<c_void>(100 * (rank + 1) + i), 1 << 20))
                .collect::<Vec<Range>>();
            // Reverse insertion cannot change read order, unlike HashMap iteration.
            let resident = components.iter().zip(&expected).rev().map(|(component, &range)|
                (crate::families::glm5::native_dense::key("q_a", component), range)).collect::<HashMap<_, _>>();
            let actual = native_projection_ranges("q_a", |key| resident.get(key).copied());
            assert_eq!(actual, expected);
            assert_eq!(crate::shared::l2_prefetch::prefix(&actual, 1 << 19),
                [(expected[0].0, 1 << 19)]);
            assert!(native_projection_ranges("kv_a", |key| resident.get(key).copied()).is_empty());
        }
    }
}

#[test]
fn k6_prefetch_enumerates_both_transform_vectors_and_payload() {
    let buffer = |id| CuteafdDeviceBuffer { ptr: std::ptr::without_provenance_mut::<c_void>(id),
        bytes: 1 << 20, device_id: 0, flags: 0 };
    let ranges = trellis_ranges(buffer(2), buffer(1), buffer(3));
    assert_eq!(ranges, [1, 2, 3].map(|id| (buffer(id).ptr.cast_const(), buffer(id).bytes)));
    assert_eq!(crate::shared::l2_prefetch::prefix(&ranges, 3 << 20), ranges);
}
