use super::*;
use crate::plan::testing::{compressed_nvfp4, minimax_m3_config, write_minimax_m3_snapshot, write_snapshot};

fn fixture() -> (tempfile::TempDir, Fp8ExpertTensors) {
    let dir = tempfile::tempdir().unwrap();
    let shape = RoutedExpertShape { layers: 60, first_layer: 3, experts: 2, topk: 1,
        hidden: 128, intermediate: 3072, draft_stages: 0, draft_experts: 0 };
    let mut tensors = Vec::new();
    for expert in 0..shape.experts {
        for (projection, n, k) in [("w1", 3072, 128), ("w3", 3072, 128), ("w2", 128, 3072)] {
            tensors.extend(compressed_nvfp4(&format!("language_model.model.layers.3.block_sparse_moe.experts.{expert}.{projection}"), n, k));
        }
    }
    write_snapshot(dir.path(), &minimax_m3_config(), &tensors, None);
    let tensors = Fp8ExpertTensors::read(dir.path(), shape).unwrap();
    for (name, located) in tensors.tensors.iter() {
        let data = if name.ends_with("_global_scale") { 3.0f32.to_le_bytes().to_vec() }
            else { (0..located.bytes).map(|i| (i.wrapping_mul(37) + i / 128) as u8).collect() };
        std::fs::OpenOptions::new().write(true).open(dir.path().join(&located.shard)).unwrap()
            .write_all_at(&data, located.offset).unwrap();
    }
    (dir, tensors)
}

fn scalar(tensors: &Fp8ExpertTensors, name: &str, value: f32) {
    let located = tensors.located(name).unwrap();
    std::fs::OpenOptions::new().write(true).open(tensors.snapshot.join(&located.shard)).unwrap()
        .write_all_at(&value.to_le_bytes(), located.offset).unwrap();
}

#[test]
fn minimax_catalog_dispatches_before_generic_text_and_excludes_routed_weights() {
    let dir = tempfile::tempdir().unwrap();
    write_minimax_m3_snapshot(dir.path(), false, false);
    let catalog = crate::read_expert_catalog(dir.path()).unwrap();
    let shape = catalog.routed_experts();
    assert_eq!((shape.first_layer, shape.layers, shape.experts, shape.topk, shape.hidden, shape.intermediate),
        (3, 60, 128, 4, 6144, 3072));
    assert_eq!(shape.geometry().unwrap(), cuteafd_core::ExpertGeometry::MINIMAX_M3);
    let tensors = catalog.fp8().unwrap();
    assert_eq!(tensors.format(), ExpertFormat::Nvfp4);
    for layer in 3..60 { tensors.validate_layer(layer).unwrap(); }
    assert!(!tensors.has_layer(2) && !tensors.has_layer(60));
    assert!(catalog.tensors().iter().all(|t| !t.metadata.name.contains(".block_sparse_moe.experts.")));
    assert!(catalog.tensor("language_model.model.layers.3.block_sparse_moe.gate.weight").is_ok());
    assert_eq!(tensors.activation(3), Some(cuteafd_core::FfnActivation::SwigluOai { alpha: 1.702, limit: 7.0 }));
}

#[test]
fn minimax_native_slices_preserve_codes_scales_and_stage_reciprocals() {
    let (_dir, tensors) = fixture();
    tensors.validate_layer(3).unwrap();
    for (tp, width) in [(2, 1536), (3, 1024), (4, 768), (6, 512)] {
        let recipe = crate::plan::experts::spark_recipe("minimax:nvfp4", 3072, tp).unwrap();
        assert_eq!(recipe.slicing, Slicing::Blocks(16));
        assert_eq!(recipe.widths, vec![width; tp]);
        assert_eq!(recipe.required_layouts, [format!("tp{tp}")]);
        assert_eq!(tensors.slicing_for_directory(Path::new(&format!("tp{tp}")), tp), recipe.slicing);
        for rank in 0..tp {
            assert_eq!(tensors.rank_range_with(tp, rank, recipe.slicing).unwrap(), (rank * width, width));
            for projection in Fp8Projection::ALL {
                let sizes = tensors.projection_bytes(projection);
                let (mut full_w, mut full_s) = (vec![0; sizes.0], vec![0; sizes.1]);
                tensors.read_projection_once(3, 0, projection, &mut full_w, &mut full_s).unwrap();
                assert_eq!(&full_s[sizes.1 - 8..], [1.0f32 / 3.0; 2].map(f32::to_le_bytes).concat());
                let sizes = tensors.slice_bytes_with(projection, tp, rank, recipe.slicing).unwrap();
                let (mut w, mut s) = (vec![0; sizes.0], vec![0; sizes.1]);
                tensors.read_slice_with(3, 0, projection, tp, rank, recipe.slicing, &mut w, &mut s, &mut Vec::new()).unwrap();
                for (full, sliced, unit) in [(&full_w, &w, 2), (&full_s, &s, 16)] {
                    let expected = if projection == Fp8Projection::Down {
                        full[..128 * 3072 / unit].chunks_exact(3072 / unit)
                            .flat_map(|row| row[rank * width / unit..(rank + 1) * width / unit].iter().copied()).collect::<Vec<_>>()
                    } else { full[rank * width * 128 / unit..(rank + 1) * width * 128 / unit].to_vec() };
                    assert_eq!(*sliced, expected, "TP{tp} rank {rank} {projection:?} unit {unit}");
                }
            }
        }
    }
    let local = tensors.for_coordinator();
    assert_eq!(local.rank_width(1, 0, local.exact_slicing(1)).unwrap(), 3072);
    for tp in [0, 1, 5, 7] { assert!(crate::plan::experts::spark_recipe("minimax:nvfp4", 3072, tp).is_err()); }
}

#[test]
fn minimax_divisors_fail_closed_before_native_execution_or_push() {
    let (_dir, tensors) = fixture();
    let gate = tensors.name(3, 0, Fp8Projection::Gate);
    let input = input_scale_name(&gate);
    let alpha = alpha_name(&gate);
    for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::from_bits(1)] {
        scalar(&tensors, &alpha, invalid);
        assert!(tensors.read_alpha(3, 0, Fp8Projection::Gate).is_err());
        scalar(&tensors, &input, invalid);
        assert!(tensors.read_input_scale(3, 0, Fp8Projection::Gate).is_err());
    }
    scalar(&tensors, &input, 4.0);
    let error = tensors.read_input_scale(3, 0, Fp8Projection::Up).unwrap_err().to_string();
    assert!(error.contains("layer 3 expert 0") && error.contains("w1.input_global_scale=4")
        && error.contains("w3.input_global_scale=3") && error.contains("bit-identical"), "{error}");
    let error = tensors.validate_push_scales().unwrap_err().to_string();
    assert!(error.contains("divisors") && error.contains("use mount loading"), "{error}");
    let mut builder = crate::push_load::builder::Builder::new(&tensors.snapshot, vec![1024]);
    let destination = crate::push_load::builder::Destination { rank: 0, offset: 0, bytes: 1024 };
    let error = builder.fp8_projection(&tensors, 3, 0, Fp8Projection::Gate, 2, 0, Slicing::Blocks(16),
        destination, destination).unwrap_err().to_string();
    assert!(error.contains("divisors") && error.contains("use mount loading"), "{error}");
    let mut missing = tensors.clone();
    Arc::make_mut(&mut missing.tensors).remove(&alpha);
    assert!(missing.validate_layer(3).is_err());
    let mut bad_shape = tensors.clone();
    Arc::make_mut(&mut bad_shape.tensors).get_mut(&input).unwrap().shape = vec![1];
    assert!(bad_shape.validate_layer(3).is_err());
    let mut config = minimax_m3_config();
    config["quantization_config"]["moe_quant_algo"] = "FP8".into();
    std::fs::write(tensors.snapshot.join("config.json"), config.to_string()).unwrap();
    assert!(Fp8ExpertTensors::read(&tensors.snapshot, tensors.shape).unwrap_err().to_string().contains("declared NVFP4 W4A4"));
}
