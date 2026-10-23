//! Official MiniMax M3 tower, planned from headers before admission.
use super::{tower_catalog, Result, TensorRead, TowerSpec, VisionError};
use cuteafd_core::DType;
use cuteafd_ffi::vision::{VisionBlock, VisionSpec, NO_VISION_OFFSET};
use cuteafd_loader::families::minimax_m3::config::MiniMaxM3VisionConfig;
use std::{fs::File, path::Path};

fn validate(config: &serde_json::Value, max_tokens: usize) -> Result<()> {
    let cfg = MiniMaxM3VisionConfig::from_hf(config)
        .map_err(|e| VisionError::Unsupported(e.to_string()))?
        .ok_or_else(|| VisionError::Unsupported("MiniMax vision_config absent".into()))?;
    if config["model_type"] != "minimax_m3_vl" || config["text_config"]["hidden_size"] != 6144
        || (cfg.hidden, cfg.heads, cfg.layers, cfg.intermediate, cfg.patch, cfg.temporal_patch,
            cfg.spatial_merge, cfg.projection, cfg.projector_hidden)
            != (1280, 16, 32, 5120, 14, 2, 2, 6144, 6144)
        || config["vision_config"]["num_channels"] != 3
        || cfg.rope_theta != 10000.0 || cfg.layer_norm_eps != 1e-5
        || cfg.max_segment_frames != 4 || max_tokens == 0 || max_tokens > 4096
    {
        return Err(VisionError::Unsupported("MiniMax vision geometry/capacity; need the official 32-layer D80 tower".into()));
    }
    Ok(())
}

impl TowerSpec {
    pub fn minimax(snapshot: &Path, max_tokens: usize) -> Result<Self> {
        let config: serde_json::Value = serde_json::from_reader(File::open(snapshot.join("config.json"))?)?;
        validate(&config, max_tokens)?;
        let tensors = tower_catalog(snapshot, &["vision_tower.", "multi_modal_projector.", "patch_merge_mlp."])?;
        let mut reads = Vec::new();
        let mut cursor = 0usize;
        let mut put = |name: &str, shape: &[usize], vector: bool, align: bool, dtype: DType| -> Result<u64> {
            let (path, metadata) = tensors.get(name)
                .ok_or_else(|| VisionError::Unsupported(format!("missing {name}")))?;
            let bytes = shape.iter().product::<usize>() * if dtype == DType::F32 { 4 } else { 2 };
            if metadata.dtype != dtype || metadata.shape != shape || metadata.byte_length != bytes as u64 {
                return Err(VisionError::Unsupported(format!("{name}: {:?} {:?}; need {dtype:?} {shape:?}", metadata.dtype, metadata.shape)));
            }
            if align { cursor = (cursor + 255) & !255; }
            let destination = cursor;
            cursor += bytes * if vector { 2 } else { 1 };
            reads.push(TensorRead { path: path.clone(), metadata: metadata.clone(), destination, vector });
            Ok(destination as u64)
        };
        let mut native = VisionSpec {
            abi_version: 3, reserved: 4, max_tokens: max_tokens as u32,
            hidden: 1280, depth: 32, heads: 16, kv_heads: 16, head_dim: 80,
            intermediate: 5120, patch_size: 14, merger_width: 24576, output_width: 6144, norm_eps: 1e-5,
            patch_bias: NO_VISION_OFFSET, pos_embed: NO_VISION_OFFSET,
            q_norm: [NO_VISION_OFFSET; 28], k_norm: [NO_VISION_OFFSET; 28],
            merger_extra: [NO_VISION_OFFSET; 8], ..Default::default()
        };
        native.patch = put("vision_tower.vision_model.embeddings.patch_embedding.weight", &[1280, 3, 2, 14, 14], false, true, DType::F32)?;
        native.merger_norm = put("vision_tower.vision_model.pre_layrnorm.weight", &[1280], true, true, DType::Bf16)?;
        native.merger_norm_bias = put("vision_tower.vision_model.pre_layrnorm.bias", &[1280], true, true, DType::Bf16)?;
        for i in 0..32 {
            let mut field = |suffix: &str, shape: &[usize], vector, align| {
                put(&format!("vision_tower.vision_model.encoder.layers.{i}.{suffix}"), shape, vector, align, DType::Bf16)
            };
            let qkv = field("self_attn.q_proj.weight", &[1280, 1280], false, true)?;
            field("self_attn.k_proj.weight", &[1280, 1280], false, false)?;
            field("self_attn.v_proj.weight", &[1280, 1280], false, false)?;
            let qkv_bias = field("self_attn.q_proj.bias", &[1280], true, true)?;
            field("self_attn.k_proj.bias", &[1280], true, false)?;
            field("self_attn.v_proj.bias", &[1280], true, false)?;
            let block = VisionBlock {
                qkv, qkv_bias,
                proj: field("self_attn.out_proj.weight", &[1280, 1280], false, true)?,
                proj_bias: field("self_attn.out_proj.bias", &[1280], true, true)?,
                gate_up: field("mlp.fc1.weight", &[5120, 1280], false, true)?,
                gate_up_bias: field("mlp.fc1.bias", &[5120], true, true)?,
                down: field("mlp.fc2.weight", &[1280, 5120], false, true)?,
                down_bias: field("mlp.fc2.bias", &[1280], true, true)?,
                norm1: field("layer_norm1.weight", &[1280], true, true)?,
                norm2: field("layer_norm2.weight", &[1280], true, true)?,
                key0_bias: NO_VISION_OFFSET, ..Default::default()
            };
            let b1 = field("layer_norm1.bias", &[1280], true, true)?;
            let b2 = field("layer_norm2.bias", &[1280], true, true)?;
            if i < 28 {
                native.blocks[i] = block; native.norm1_bias[i] = b1; native.norm2_bias[i] = b2;
            } else {
                native.blocks_extra[i - 28] = block;
                native.norm1_bias_extra[i - 28] = b1; native.norm2_bias_extra[i - 28] = b2;
            }
        }
        for (i, stem, shape) in [
            (0, "multi_modal_projector.linear_1", [6144, 1280]),
            (2, "multi_modal_projector.linear_2", [6144, 6144]),
        ] {
            native.merger_extra[i] = put(&format!("{stem}.weight"), &shape, false, true, DType::Bf16)?;
            native.merger_extra[i+1] = put(&format!("{stem}.bias"), &[6144], true, true, DType::Bf16)?;
        }
        native.merger_fc1 = put("patch_merge_mlp.linear_1.weight", &[6144, 24576], false, true, DType::Bf16)?;
        native.merger_fc1_bias = put("patch_merge_mlp.linear_1.bias", &[6144], true, true, DType::Bf16)?;
        native.merger_fc2 = put("patch_merge_mlp.linear_2.weight", &[6144, 6144], false, true, DType::Bf16)?;
        native.merger_fc2_bias = put("patch_merge_mlp.linear_2.bias", &[6144], true, true, DType::Bf16)?;
        drop(put);
        if reads.len() != tensors.len() {
            return Err(VisionError::Unsupported("unmapped MiniMax vision tensors; update tower contract".into()));
        }
        cursor = (cursor + 255) & !255;
        native.inv_freq = cursor as u64;
        native.weight_bytes = (cursor + 13 * 4) as u64;
        Ok(Self { native, reads })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> serde_json::Value {
        serde_json::from_str(include_str!("../../../../cuteafd-loader/src/families/minimax_m3/fixtures/config.json")).unwrap()
    }
    #[test]
    fn geometry_is_checked_before_headers() {
        assert!(validate(&config(), 4096).is_ok());
        for cap in [0, 4097] { assert!(validate(&config(), cap).is_err()); }
        for (key, value) in [("num_hidden_layers", 28), ("hidden_size", 1024), ("num_attention_heads", 8), ("patch_size", 16)] {
            let mut cfg = config(); cfg["vision_config"][key] = value.into();
            assert!(validate(&cfg, 576).is_err(), "{key}");
        }
    }
    fn header() -> serde_json::Map<String, serde_json::Value> {
        let mut header = serde_json::Map::new();
        let mut add = |name: String, shape: &[usize], dtype: &str| {
            header.insert(name, serde_json::json!({"dtype":dtype,"shape":shape}));
        };
        let prefix = "vision_tower.vision_model";
        add(format!("{prefix}.embeddings.patch_embedding.weight"), &[1280, 3, 2, 14, 14], "F32");
        for suffix in ["weight", "bias"] {
            add(format!("{prefix}.pre_layrnorm.{suffix}"), &[1280], "BF16");
        }
        for i in 0..32 {
            for norm in ["layer_norm1", "layer_norm2"] {
                for suffix in ["weight", "bias"] {
                    add(format!("{prefix}.encoder.layers.{i}.{norm}.{suffix}"), &[1280], "BF16");
                }
            }
            for (name, output, input) in [
                ("self_attn.q_proj", 1280, 1280), ("self_attn.k_proj", 1280, 1280),
                ("self_attn.v_proj", 1280, 1280), ("self_attn.out_proj", 1280, 1280),
                ("mlp.fc1", 5120, 1280), ("mlp.fc2", 1280, 5120),
            ] {
                add(format!("{prefix}.encoder.layers.{i}.{name}.weight"), &[output, input], "BF16");
                add(format!("{prefix}.encoder.layers.{i}.{name}.bias"), &[output], "BF16");
            }
        }
        for (name, input) in [
            ("multi_modal_projector.linear_1", 1280), ("multi_modal_projector.linear_2", 6144),
            ("patch_merge_mlp.linear_1", 24576), ("patch_merge_mlp.linear_2", 6144),
        ] {
            add(format!("{name}.weight"), &[6144, input], "BF16");
            add(format!("{name}.bias"), &[6144], "BF16");
        }
        header
    }
    fn snapshot(mut header: serde_json::Map<String, serde_json::Value>) -> tempfile::TempDir {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), serde_json::to_vec(&config()).unwrap()).unwrap();
        let mut end = 0u64;
        let mut map = serde_json::Map::new();
        for (name, value) in &mut header {
            let start = end;
            let elements = value["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).product::<u64>();
            end += elements * if value["dtype"] == "F32" { 4 } else { 2 };
            value["data_offsets"] = serde_json::json!([start, end]);
            map.insert(name.clone(), serde_json::json!("vision.safetensors"));
        }
        let bytes = serde_json::to_vec(&header).unwrap();
        let mut file = File::create(dir.path().join("vision.safetensors")).unwrap();
        file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&bytes).unwrap();
        file.set_len(8 + bytes.len() as u64 + end).unwrap();
        map.insert("model.language_model.layers.0.unread".into(), serde_json::json!("missing-text.safetensors"));
        std::fs::write(dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map":map})).unwrap()).unwrap();
        dir
    }
    #[test]
    fn header_plan_packs_all_layers_without_reading_text_weights() {
        let dir = snapshot(header());
        let spec = TowerSpec::from_snapshot(dir.path(), 4096).unwrap();
        assert_eq!(spec.reads.len(), 523);
        assert_eq!(spec.native.weight_bytes, 1_735_264_308);
        assert_eq!((spec.native.abi_version, spec.native.reserved), (3, 4));
        assert_eq!(spec.native.head_dim, 80);
        assert_eq!(spec.native.weight_bytes, spec.native.inv_freq + 13 * 4);
        assert_eq!(spec.reads.iter().filter(|r| r.metadata.dtype == DType::F32).count(), 1);
        for i in 0..32 {
            for suffix in ["weight", "bias"] {
                let read = |projection| spec.reads.iter().find(|r| r.metadata.name ==
                    format!("vision_tower.vision_model.encoder.layers.{i}.self_attn.{projection}_proj.{suffix}")).unwrap();
                let span = if suffix == "weight" { 1280 * 1280 * 2 } else { 1280 * 4 };
                assert_eq!(read("k").destination, read("q").destination + span);
                assert_eq!(read("v").destination, read("k").destination + span);
                let block = if i < 28 { &spec.native.blocks[i] } else { &spec.native.blocks_extra[i - 28] };
                assert_eq!(if suffix == "weight" { block.qkv } else { block.qkv_bias }, read("q").destination as u64);
            }
        }
    }
    #[test]
    fn header_plan_rejects_missing_extra_dtype_and_shape() {
        let name = "vision_tower.vision_model.embeddings.patch_embedding.weight";
        for mutation in ["missing", "extra", "dtype", "shape"] {
            let mut value = header();
            match mutation {
                "missing" => { value.remove(name); }
                "extra" => { value.insert("vision_tower.unmapped.weight".into(), serde_json::json!({"dtype":"BF16","shape":[1]})); }
                "dtype" => value.get_mut(name).unwrap()["dtype"] = "BF16".into(),
                "shape" => value.get_mut(name).unwrap()["shape"] = serde_json::json!([1280, 3, 2, 7, 28]),
                _ => unreachable!(),
            }
            let dir = snapshot(value);
            assert!(TowerSpec::from_snapshot(dir.path(), 4096).is_err(), "{mutation}");
        }
    }
    #[test]
    #[ignore = "requires a local header-only MiniMax release snapshot"]
    fn release_snapshot_header_contract() {
        let snapshot = std::env::var("CUTEAFD_MINIMAX_VISION_SNAPSHOT").expect("header snapshot");
        let spec = TowerSpec::from_snapshot(Path::new(&snapshot), 4096).unwrap();
        assert_eq!(spec.reads.len(), 523);
        assert_eq!(spec.native.weight_bytes, 1_735_264_308);
        assert_eq!(spec.native.depth, 32);
        assert_eq!(spec.image_family(), cuteafd_loader::media::ImageFamily::Minimax);
        assert_eq!(spec.reads.iter().filter(|r| r.metadata.dtype == DType::F32).count(), 1);
        assert_eq!(spec.native.blocks_extra[3].key0_bias, NO_VISION_OFFSET);
        for i in 0..32 {
            let read = |p| spec.reads.iter().find(|r| r.metadata.name == format!("vision_tower.vision_model.encoder.layers.{i}.self_attn.{p}_proj.weight")).unwrap();
            assert_eq!(read("k").destination, read("q").destination + 1280*1280*2);
            assert_eq!(read("v").destination, read("k").destination + 1280*1280*2);
        }
    }
}
