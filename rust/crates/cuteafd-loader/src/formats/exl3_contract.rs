//! Optional EXL3 publication metadata cross-checks. Tensor headers own storage;
//! declared core, vocabulary and MTP widths only verify that derived inventory.
use super::exl3_storage::{check_quantization_config, Exl3StorageMap};
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::collections::BTreeMap;

pub(crate) fn check_native_quantization_config(quant: &Value) -> Result<()> {
    let Some(value) = quant.get("native_quantization_config") else { return Ok(()); };
    let object = value.as_object().context("EXL3 native_quantization_config must be an object")?;
    for key in object.keys() {
        ensure!(matches!(key.as_str(), "quant_method" | "activation_scheme" | "weight_block_size"
            | "modules_to_not_convert"), "EXL3 native_quantization_config unexpected key {key}");
    }
    for (key, expected) in [("quant_method", serde_json::json!("fp8")),
        ("activation_scheme", serde_json::json!("dynamic")),
        ("weight_block_size", serde_json::json!([128, 128]))] {
        ensure!(value.get(key) == Some(&expected), "EXL3 native_quantization_config unsupported {key}: {}", value[key]);
    }
    let exclusions = value["modules_to_not_convert"].as_array()
        .context("EXL3 native_quantization_config modules_to_not_convert must be a string list")?;
    let mut seen = std::collections::BTreeSet::new();
    for exclusion in exclusions {
        let name = exclusion.as_str().filter(|s| !s.is_empty())
            .context("EXL3 native_quantization_config modules_to_not_convert must be a string list")?;
        ensure!(seen.insert(name), "EXL3 native_quantization_config duplicate exclusion {name}");
    }
    Ok(())
}

fn head(name: &str) -> bool {
    // Mistral consolidated checkpoints name the vocabulary head `output`.
    name == "lm_head" || name.ends_with(".lm_head") || name == "output"
}

pub(crate) fn routed(name: &str) -> bool {
    name.split_once(".experts.").is_some_and(|(_, tail)|
        tail.split_once('.').is_some_and(|(expert, _)| expert.parse::<usize>().is_ok()))
}

fn mtp(name: &str, backbone_layers: usize) -> bool {
    // `mtp_dspark.`: Mistral Large 4's bundled dSpark drafter.
    name.starts_with("mtp.") || name.starts_with("mtp_dspark.") || name.contains(".mtp.") || name.split_once(".layers.")
        .and_then(|(_, tail)| tail.split('.').next()?.parse::<usize>().ok())
        .is_some_and(|layer| layer >= backbone_layers)
}

fn layer_of(name: &str) -> Option<usize> {
    name.split_once(".layers.").and_then(|(_, tail)| tail.split('.').next()?.parse().ok())
}

fn bits(key: &str, value: &Value, max: usize) -> Result<usize> {
    Ok(value.as_f64().filter(|v| v.fract() == 0.0 && (1.0..=max as f64).contains(v))
        .with_context(|| format!("EXL3 {key} must be an integer in 1..{max}"))? as usize)
}

pub(crate) fn check_header_widths(
    quant: &Value, derived: &Exl3StorageMap, backbone_layers: usize,
) -> Result<()> {
    for key in ["core_bits", "head_bits", "mtp_bits"] {
        let Some(value) = quant.get(key) else { continue };
        let width = bits(key, value, if key == "head_bits" { 16 } else { 8 })?;
        let mut modules = derived.iter().filter(|(name, _)| match key {
            "mtp_bits" => mtp(name, backbone_layers),
            "head_bits" => head(name),
            _ => !routed(name) && !mtp(name, backbone_layers)
                && !(head(name) && quant.get("head_bits").is_some()),
        }).peekable();
        // A native BF16 head has no EXL3 header entry.
        ensure!(modules.peek().is_some() || key == "head_bits" && width == 16,
            "EXL3 {key}={width} but the headers carry no matching EXL3 modules");
        for (name, module) in modules {
            ensure!(module.bits == width,
                "EXL3 {key}={width} disagrees with the headers at {name} (K{})", module.bits);
        }
    }
    // Routed experts: `expert_bits_by_layer` overrides `expert_bits` per target layer.
    if quant.get("expert_bits").is_some() || quant.get("expert_bits_by_layer").is_some() {
        let base = quant.get("expert_bits").map(|v| bits("expert_bits", v, 8)).transpose()?;
        let by_layer = match quant.get("expert_bits_by_layer") {
            None => BTreeMap::new(),
            Some(map) => map.as_object().context("EXL3 expert_bits_by_layer must be an object")?.iter()
                .map(|(layer, v)| Ok((layer.parse::<usize>().with_context(||
                    format!("EXL3 expert_bits_by_layer key {layer} is not a layer"))?, bits("expert_bits_by_layer", v, 8)?)))
                .collect::<Result<BTreeMap<_, _>>>()?,
        };
        for (name, module) in derived.iter().filter(|(name, _)| routed(name) && !mtp(name, backbone_layers)) {
            let layer = layer_of(name);
            let declared = layer.and_then(|l| by_layer.get(&l).copied()).or(base);
            if let Some(width) = declared {
                ensure!(module.bits == width,
                    "EXL3 routed width K{width} disagrees with the headers at {name} (K{})", module.bits);
            }
        }
    }
    if let Some(value) = quant.get("lm_head") {
        let declared = value.as_bool().context("EXL3 lm_head must be boolean")?;
        ensure!(declared == derived.keys().any(|name| head(name)),
            "EXL3 lm_head={declared} disagrees with the vocabulary headers");
    }
    Ok(())
}

fn check_fused_expert_layout(value: &Value, derived: &Exl3StorageMap, layers: usize) -> Result<()> {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Layout {
        down_shape: [usize; 3],
        gate_up_shape: [usize; 3],
        expert_axis: usize,
        input_axis: usize,
        output_axis: usize,
        gate_output_rows: [usize; 2],
        up_output_rows: [usize; 2],
        output_names: String,
    }
    let layout: Layout = serde_json::from_value(value.clone()).context("EXL3 fused_expert_layout schema")?;
    ensure!([layout.expert_axis, layout.output_axis, layout.input_axis] == [0, 1, 2],
        "EXL3 fused_expert_layout unsupported source axes (expected expert/output/input=0/1/2)");
    let [experts, outputs, input] = layout.gate_up_shape;
    let [down_experts, down_output, down_input] = layout.down_shape;
    let [gate_start, gate_end] = layout.gate_output_rows;
    let [up_start, up_end] = layout.up_output_rows;
    ensure!(experts > 0 && input > 0 && gate_start == 0 && gate_end > 0
        && up_start == gate_end && up_end == outputs && up_end > up_start
        && gate_end == up_end - up_start && down_experts == experts
        && down_output == input && down_input == gate_end,
        "EXL3 fused_expert_layout source shapes or gate/up row ranges disagree");
    let template = layout.output_names.strip_suffix(".{gate_proj,up_proj,down_proj}")
        .context("EXL3 fused_expert_layout output_names must name gate_proj,up_proj,down_proj")?;
    let (prefix, tail) = template.split_once(".N.")
        .context("EXL3 fused_expert_layout output_names missing layer placeholder N")?;
    ensure!(tail == "mlp.experts.E" && prefix.ends_with("layers"),
        "EXL3 fused_expert_layout output_names unsupported expert template");
    // These are source-fusion provenance fields: the published payload is already
    // split into independently rotated per-expert EXL3 modules, never a fused view.
    let expected = layers.checked_mul(experts).and_then(|n| n.checked_mul(3))
        .context("EXL3 fused_expert_layout module count overflow")?;
    ensure!(layers > 0 && derived.keys().filter(|name| routed(name)).count() == expected,
        "EXL3 fused_expert_layout expert/module count disagrees with emitted headers");
    for layer in 0..layers {
        for expert in 0..experts {
            for (projection, k, n) in [("gate_proj", input, gate_end),
                ("up_proj", input, up_end - up_start), ("down_proj", down_input, down_output)] {
                let name = format!("{prefix}.{layer}.mlp.experts.{expert}.{projection}");
                let module = derived.get(&name).with_context(||
                    format!("EXL3 fused_expert_layout output_names missing emitted header {name}"))?;
                ensure!(module.input_features == k && module.output_features == n,
                    "EXL3 fused_expert_layout {name}: emitted headers are {}x{}, source declares {k}x{n}",
                    module.input_features, module.output_features);
            }
        }
    }
    Ok(())
}

pub(crate) fn check_retained_native_tensors(
    manifest: &Value, derived: &Exl3StorageMap, headers: &[&crate::SafetensorsTensorMetadata],
) -> Result<()> {
    let Some(value) = manifest.get("retained_native_tensors") else { return Ok(()); };
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct NativeTensor { dtype: String, shape: Vec<usize> }
    let declarations: std::collections::BTreeMap<String, NativeTensor> =
        serde_json::from_value(value.clone()).context("EXL3 retained_native_tensors schema")?;
    let is_exl3 = |name: &str| name.rsplit_once('.').is_some_and(|(stem, suffix)|
        derived.get(stem).is_some_and(|module| module.tensors.contains_key(suffix)));
    let mut native = std::collections::BTreeMap::new();
    for header in headers.iter().filter(|header| !is_exl3(&header.name)) {
        ensure!(native.insert(header.name.as_str(), *header).is_none(),
            "EXL3 retained_native_tensors duplicate header {}", header.name);
    }
    for (name, declaration) in declarations {
        ensure!(!is_exl3(&name), "EXL3 retained_native_tensors {name} is an EXL3 storage member");
        let header = native.remove(name.as_str()).with_context(||
            format!("EXL3 retained_native_tensors {name} missing from native headers"))?;
        let dtype = cuteafd_core::DType::from_safetensors(&declaration.dtype);
        use cuteafd_core::DType;
        let width = match dtype {
            DType::Bf16 | DType::F16 | DType::I16 => 2u64,
            DType::F32 | DType::I32 => 4,
            DType::I64 => 8,
            DType::I8 | DType::U8 | DType::F8E4M3 | DType::F8E5M2 | DType::F8E8M0 => 1,
            _ => anyhow::bail!("EXL3 retained_native_tensors {name} unsupported dtype {}", declaration.dtype),
        };
        let bytes = declaration.shape.iter().try_fold(width, |bytes, &dim| bytes.checked_mul(dim as u64))
            .with_context(|| format!("EXL3 retained_native_tensors {name} byte extent overflow"))?;
        ensure!(dtype == header.dtype && declaration.shape == header.shape && bytes == header.byte_length,
            "EXL3 retained_native_tensors {name} dtype/shape/bytes disagree with the safetensors header");
    }
    ensure!(native.is_empty(), "EXL3 retained_native_tensors omits native header {}",
        native.keys().next().copied().unwrap_or_default());
    Ok(())
}

/// Publication metadata is optional, but a declared attention view is a complete,
/// typed contract. The selected index and tensor headers remain authoritative.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct AttentionBundle { config: String, index: String, quantize: String }

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct AttentionViews {
    hqa: AttentionBundle,
    #[serde(rename = "compact-attention")]
    compact: AttentionBundle,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct AttentionViewDeclaration {
    schema: String,
    mode: String,
    views: AttentionViews,
    native_modules: String,
    #[serde(default)]
    index_aliases: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct AttentionTensor { dtype: String, shape: Vec<usize> }

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
enum NativeAttentionFormat {
    #[serde(rename = "bf16")]
    Bf16,
    #[serde(rename = "fp8_e4m3_block_scaled")]
    Fp8Block,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct AttentionScale {
    tensor: String,
    dtype: String,
    block_size: [usize; 2],
    semantics: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeAttention {
    format: NativeAttentionFormat,
    tensors: BTreeMap<String, AttentionTensor>,
    shard: String,
    compact_attention_bits: usize,
    scale: Option<AttentionScale>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AttentionContract {
    view: AttentionViewDeclaration,
    native: BTreeMap<String, NativeAttention>,
}

impl AttentionContract {
    pub(crate) fn parse(config_quant: &Value, manifest: &Value) -> Result<Option<Self>> {
        let declared = config_quant.get("attention_view");
        let published = manifest.get("attention_view");
        let native = manifest.get("native_attention");
        if declared.is_none() && published.is_none() && native.is_none() { return Ok(None); }
        ensure!(declared.is_some() && declared == published,
            "compact/full EXL3 metadata disagree at attention_view");
        let view: AttentionViewDeclaration = serde_json::from_value(published.unwrap().clone())
            .context("EXL3 attention_view schema")?;
        let native: BTreeMap<String, NativeAttention> = serde_json::from_value(
            native.context("EXL3 attention_view requires native_attention")?.clone())
            .context("EXL3 native_attention schema")?;
        let contract = Self { view, native };
        contract.validate()?;
        Ok(Some(contract))
    }

    fn validate(&self) -> Result<()> {
        use crate::plan::checkpoint::view::{COMPACT_CONFIG, COMPACT_INDEX, COMPACT_QUANT};
        ensure!(self.view.schema == "cuteafd-attention-views-v1" && self.view.mode == "hqa",
            "unsupported EXL3 attention_view schema or mode");
        for (bundle, names) in [(&self.view.views.hqa,
            ["config.json", "model.safetensors.index.json", "quantize_config.json"]),
            (&self.view.views.compact, [COMPACT_CONFIG, COMPACT_INDEX, COMPACT_QUANT])] {
            ensure!([bundle.config.as_str(), bundle.index.as_str(), bundle.quantize.as_str()] == names,
                "EXL3 attention_view references an inconsistent bundle");
        }
        ensure!(self.view.native_modules == "quantize_config.json:native_attention",
            "EXL3 attention_view native_modules must name the selected native_attention map");
        ensure!(!self.native.is_empty(), "EXL3 native_attention is empty");
        for (module, entry) in &self.native {
            ensure!(!module.is_empty() && !module.ends_with('.') && (2..=8).contains(&entry.compact_attention_bits),
                "EXL3 native_attention {module}: invalid module or compact_attention_bits");
            ensure!(entry.shard.ends_with(".safetensors") && !entry.shard.starts_with('.')
                && !entry.shard.contains(['/', '\\']), "EXL3 native_attention {module}: invalid shard");
            let weight_name = format!("{module}.weight");
            let weight = entry.tensors.get(&weight_name)
                .with_context(|| format!("EXL3 native_attention {module}: missing weight"))?;
            ensure!(weight.shape.len() == 2 && weight.shape.iter().all(|&dim| dim > 0),
                "EXL3 native_attention {module}: weight must be a nonempty matrix");
            match entry.format {
                NativeAttentionFormat::Bf16 => ensure!(entry.tensors.len() == 1
                    && weight.dtype == "BF16" && entry.scale.is_none(),
                    "EXL3 native_attention {module}: BF16 requires one BF16 weight and no scales"),
                NativeAttentionFormat::Fp8Block => {
                    let scale = entry.scale.as_ref()
                        .with_context(|| format!("EXL3 native_attention {module}: missing block scale contract"))?;
                    ensure!(weight.dtype == "F8_E4M3" && entry.tensors.len() == 2
                        && scale.tensor.starts_with(&format!("{module}.")) && scale.tensor != weight_name
                        && scale.dtype == "F32" && scale.block_size == [128, 128]
                        && scale.semantics == "checkpoint-native: original FP8 weight with its original per-block FP32 scale",
                        "EXL3 native_attention {module}: unsupported FP8 block scale contract");
                    let tensor = entry.tensors.get(&scale.tensor)
                        .with_context(|| format!("EXL3 native_attention {module}: missing scale tensor {}", scale.tensor))?;
                    ensure!(tensor.dtype == "F32" && tensor.shape == weight.shape.iter()
                        .map(|&dim| dim.div_ceil(128)).collect::<Vec<_>>(),
                        "EXL3 native_attention {module}: scale dtype/shape disagrees with weight geometry");
                }
            }
        }
        Ok(())
    }

    pub(crate) fn check_selection(&self, config: &str, index: &str, quant: &[String],
        weights: Option<&BTreeMap<String, String>>) -> Result<()> {
        let bundle = &self.view.views.hqa;
        ensure!(config == bundle.config && index == bundle.index && quant.contains(&bundle.quantize),
            "EXL3 attention_view does not describe the selected bundle");
        let weights = weights.context("EXL3 attention_view requires a selected tensor index")?;
        for (module, entry) in &self.native {
            for name in entry.tensors.keys() {
                ensure!(weights.get(name) == Some(&entry.shard),
                    "EXL3 native_attention {name}: declared shard disagrees with selected index");
            }
            let prefix = format!("{module}.");
            for (name, _) in weights.range(prefix.clone()..) {
                if !name.starts_with(&prefix) { break; }
                ensure!(entry.tensors.contains_key(name),
                    "EXL3 native_attention {module}: selected index also exposes {name}");
            }
        }
        Ok(())
    }

    pub(crate) fn check_headers<'a>(&self,
        headers: impl IntoIterator<Item = &'a crate::SafetensorsTensorMetadata>) -> Result<()> {
        use cuteafd_core::DType;
        // A role may read only some of the selected native projections. Missing
        // indexed tensors are checked by the shard reader, not invented here.
        for header in headers {
            let Some((module, _)) = header.name.rsplit_once('.') else { continue; };
            let Some(entry) = self.native.get(module) else { continue; };
            let declaration = entry.tensors.get(&header.name).with_context(||
                format!("EXL3 native_attention {}: undeclared selected tensor", header.name))?;
            let dtype = DType::from_safetensors(&declaration.dtype);
            let width = match dtype { DType::Bf16 => 2u64, DType::F32 => 4,
                DType::F8E4M3 => 1, _ => anyhow::bail!("EXL3 native_attention unsupported dtype") };
            let bytes = declaration.shape.iter().try_fold(width, |bytes, &dim| bytes.checked_mul(dim as u64))
                .context("EXL3 native_attention byte extent overflow")?;
            ensure!(dtype == header.dtype && declaration.shape == header.shape && bytes == header.byte_length,
                "EXL3 native_attention {}: dtype/shape/bytes disagree with selected safetensors header", header.name);
        }
        Ok(())
    }
}

pub(crate) fn check_gptqmodel_manifest(
    compact: &Value, manifest: &Value, derived: &Exl3StorageMap, backbone_layers: usize,
) -> Result<()> {
    for field in ["quant_method", "method", "format", "checkpoint_format"] {
        ensure!(manifest.get(field).and_then(Value::as_str) == Some("exl3"),
            "EXL3 manifest requires {field}=exl3");
    }
    // Codebook/packing/scale fields constrain native decoder semantics; core_bits,
    // head_bits and lm_head are checked against headers below. tensor_storage is
    // consumed by parse_storage_metadata before this check; per_module_metadata
    // must point to that map. Headers, not module_include's quantizer-selection
    // regexes, determine inventory. calibration and meta record how weights were
    // produced, not runtime transforms; retain them only as publication provenance.
    // out_scales describes whether the quantizer folded scales into svh.
    ensure!(manifest["codebook"] == "mcg"
        && matches!(manifest["out_scales"].as_str(), Some("never" | "auto"))
        && manifest["group_size"] == -1 && manifest["desc_act"] == false
        && manifest.get("pack_dtype").is_none_or(|v| v == "int32")
        && manifest.get("sym").is_none_or(|v| v == true)
        && manifest.get("calibration").is_none_or(Value::is_object),
        "unsupported EXL3 storage contract");
    ensure!(manifest["bits"].as_f64().is_some_and(|v| v.fract() == 0.0 && (2.0..=5.0).contains(&v)),
        "EXL3 base bits must be an integer in 2..5");
    let object = manifest.as_object().context("EXL3 manifest must be an object")?;
    for key in object.keys() {
        ensure!(matches!(key.as_str(),
            "bits" | "checkpoint_format" | "codebook" | "core_bits" | "head_bits" | "mtp_bits"
            | "desc_act" | "format" | "group_size" | "lm_head" | "meta" | "method" | "module_include"
            | "out_scales" | "pack_dtype" | "quant_method" | "tensor_storage" | "per_module_metadata"
            | "sym" | "calibration" | "native_quantization_config" | "mcg_multiplier" | "expert_bits" | "expert_bits_by_layer"
            | "kept_tensors" | "fused_expert_layout" | "retained_native_tensors"
            | "attention_view" | "native_attention"), "EXL3 manifest has unexpected key {key}");
    }
    if let Some(layout) = manifest.get("fused_expert_layout") {
        check_fused_expert_layout(layout, derived, backbone_layers)?;
    }
    if let Some(multiplier) = manifest.get("mcg_multiplier") {
        ensure!(multiplier.as_u64() == Some(super::exl3_storage::EXL3_MCG_MULTIPLIER),
            "EXL3 mcg_multiplier {multiplier} (the decoders implement 0xCBAC1FED)");
    }
    if let Some(kept) = manifest.get("kept_tensors") {
        ensure!(kept.is_object(), "EXL3 kept_tensors must map tensor names to native dtypes/shapes");
    }
    AttentionContract::parse(compact, manifest)?;
    check_native_quantization_config(manifest)?;
    ensure!(compact.get("native_quantization_config") == manifest.get("native_quantization_config"),
        "compact/full EXL3 metadata disagree at native_quantization_config");
    check_header_widths(manifest, derived, backbone_layers)?;
    let compact = compact.as_object().context("missing compact EXL3 config")?;
    ensure!(compact.get("quant_method") == Some(&Value::from("exl3")), "config must declare EXL3 quantization");
    for (key, value) in compact {
        if key == "per_module_metadata" {
            ensure!(matches!(value.as_str(), Some("quantize_config.json:tensor_storage" | "quantize_config.compact-attention.json:tensor_storage")),
                "EXL3 per_module_metadata must name the selected quantization tensor_storage, not {value}");
            continue;
        }
        ensure!(manifest.get(key) == Some(value), "compact/full EXL3 metadata disagree at {key}");
    }
    if let Some(pointer) = manifest.get("per_module_metadata") {
        ensure!(matches!(pointer.as_str(), Some("quantize_config.json:tensor_storage" | "quantize_config.compact-attention.json:tensor_storage"))
            && compact.get("per_module_metadata").is_none_or(|value| value == pointer),
            "EXL3 per_module_metadata must name the selected quantization tensor_storage, not {pointer}");
    }
    Ok(())
}

pub(crate) fn check_exllamav3_manifest(compact: &Value, manifest: &Value) -> Result<()> {
    let manifest = manifest.as_object().context("EXL3 storage map must be an object")?;
    let compact_object = compact.as_object().context("missing compact EXL3 config")?;
    for key in compact_object.keys().chain(manifest.keys()) {
        ensure!(key == "tensor_storage" || manifest.get(key) == compact_object.get(key),
            "compact/full EXL3 metadata disagree at {key}");
    }
    ensure!(!compact_object.contains_key("tensor_storage"), "config.json carries an inline tensor_storage");
    check_quantization_config(compact)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::exl3_storage::{Exl3Codebook, Exl3Module};
    use std::collections::BTreeMap;

    fn storage() -> Exl3StorageMap {
        let module = |bits| Exl3Module { codebook: Exl3Codebook::Mcg, bits,
            input_features: 128, output_features: 128, tensors: BTreeMap::new() };
        BTreeMap::from([
            ("model.layers.0.self_attn.q_a_proj".into(), module(6)),
            ("model.layers.1.mlp.shared_experts.down_proj".into(), module(6)),
            ("model.layers.1.mlp.experts.0.gate_proj".into(), module(4)),
            ("model.layers.2.self_attn.q_a_proj".into(), module(4)),
            ("model.layers.2.mlp.experts.0.gate_proj".into(), module(4)),
            ("model.layers.2.eh_proj".into(), module(4)),
            ("lm_head".into(), module(6)),
        ])
    }

    fn publication() -> (Value, Value) {
        let compact = serde_json::json!({"quant_method":"exl3", "bits":4.0,
            "core_bits":6.0, "head_bits":6.0, "mtp_bits":4.0, "checkpoint_format":"exl3",
            "format":"exl3", "per_module_metadata":"quantize_config.json:tensor_storage"});
        let mut manifest = compact.clone();
        manifest.as_object_mut().unwrap().extend(serde_json::json!({"codebook":"mcg", "method":"exl3",
            "desc_act":false, "group_size":-1, "lm_head":true, "out_scales":"auto", "sym":true,
            "calibration":{"samples":1426, "tokens":1097737}, "meta":{}, "module_include":[],
            "tensor_storage":true}).as_object().unwrap().clone());
        (compact, manifest)
    }

    fn attention_publication() -> (Value, Value, BTreeMap<String, String>) {
        let (mut config, mut manifest) = publication();
        let view = serde_json::json!({"schema":"cuteafd-attention-views-v1", "mode":"hqa",
            "views":{"hqa":{"config":"config.json", "index":"model.safetensors.index.json",
                "quantize":"quantize_config.json"}, "compact-attention":{
                "config":"config.compact-attention.json", "index":"model.compact-attention.safetensors.index.json",
                "quantize":"quantize_config.compact-attention.json"}},
            "native_modules":"quantize_config.json:native_attention"});
        config["attention_view"] = view.clone();
        manifest["attention_view"] = view;
        manifest["native_attention"] = serde_json::json!({"model.layers.0.self_attn.o_proj":{
            "format":"bf16", "shard":"native.safetensors", "compact_attention_bits":6,
            "tensors":{"model.layers.0.self_attn.o_proj.weight":{"dtype":"BF16", "shape":[128,256]}}}});
        let weights = BTreeMap::from([("model.layers.0.self_attn.o_proj.weight".into(), "native.safetensors".into())]);
        (config, manifest, weights)
    }

    #[test]
    fn hqa_contract_checks_selected_index_headers_and_strict_schema() {
        let (config, manifest, weights) = attention_publication();
        let contract = AttentionContract::parse(&config, &manifest).unwrap().unwrap();
        check_gptqmodel_manifest(&config, &manifest, &storage(), 2).unwrap();
        let check = |contract: &AttentionContract, weights: &BTreeMap<String, String>| contract.check_selection(
            "config.json", "model.safetensors.index.json", &["quantize_config.json".into()], Some(weights));
        check(&contract, &weights).unwrap();
        assert!(contract.check_selection("config.compact-attention.json", "model.safetensors.index.json",
            &["quantize_config.json".into()], Some(&weights)).is_err());
        let mut wrong = weights.clone();
        wrong.values_mut().for_each(|shard| *shard = "wrong.safetensors".into());
        assert!(check(&contract, &wrong).is_err());
        wrong = weights.clone();
        wrong.insert("model.layers.0.self_attn.o_proj.trellis".into(), "parent.safetensors".into());
        assert!(check(&contract, &wrong).is_err());
        assert!(check(&contract, &BTreeMap::new()).is_err());
        let header = crate::SafetensorsTensorMetadata { name: weights.keys().next().unwrap().clone(),
            dtype: cuteafd_core::DType::Bf16, shape: vec![128,256], byte_offset: 0, byte_length: 65536 };
        contract.check_headers([&header]).unwrap();
        for variant in 0..3 {
            let mut wrong = header.clone();
            match variant { 0 => wrong.dtype = cuteafd_core::DType::F16,
                1 => wrong.shape = vec![256,128], _ => wrong.byte_length -= 1 }
            assert!(contract.check_headers([&wrong]).is_err());
        }
        for pointer in ["/attention_view/schema", "/attention_view/native_modules",
            "/attention_view/views/hqa/index", "/native_attention/model.layers.0.self_attn.o_proj/format",
            "/native_attention/model.layers.0.self_attn.o_proj/tensors/model.layers.0.self_attn.o_proj.weight/dtype"] {
            let mut bad = manifest.clone();
            *bad.pointer_mut(pointer).unwrap() = Value::from("wrong");
            let mut bad_config = config.clone(); bad_config["attention_view"] = bad["attention_view"].clone();
            assert!(AttentionContract::parse(&bad_config, &bad).is_err(), "accepted {pointer}");
        }
        for pointer in ["/attention_view", "/attention_view/views/hqa",
            "/native_attention/model.layers.0.self_attn.o_proj",
            "/native_attention/model.layers.0.self_attn.o_proj/tensors/model.layers.0.self_attn.o_proj.weight"] {
            let mut bad = manifest.clone();
            bad.pointer_mut(pointer).unwrap()["unknown"] = Value::Bool(true);
            let mut bad_config = config.clone(); bad_config["attention_view"] = bad["attention_view"].clone();
            assert!(AttentionContract::parse(&bad_config, &bad).is_err(), "accepted unknown field at {pointer}");
        }
        let mut bad = manifest.clone(); bad["future_storage_key"] = Value::Bool(true);
        assert!(check_gptqmodel_manifest(&config, &bad, &storage(), 2).unwrap_err().to_string()
            .contains("unexpected key future_storage_key"));
        bad = manifest.clone(); bad.as_object_mut().unwrap().remove("native_attention");
        assert!(AttentionContract::parse(&config, &bad).is_err());
        bad = manifest.clone(); bad.as_object_mut().unwrap().remove("attention_view");
        assert!(AttentionContract::parse(&config, &bad).is_err());
    }

    #[test]
    fn hqa_native_fp8_scale_contract_is_not_an_untyped_annotation() {
        let (config, mut manifest, mut weights) = attention_publication();
        let module = "model.layers.0.self_attn.o_proj";
        let scale = format!("{module}.qscale_weight");
        let native = &mut manifest["native_attention"][module];
        native["format"] = Value::from("fp8_e4m3_block_scaled");
        native["tensors"][format!("{module}.weight")]["dtype"] = Value::from("F8_E4M3");
        native["tensors"][&scale] = serde_json::json!({"dtype":"F32", "shape":[1,2]});
        native["scale"] = serde_json::json!({"tensor":scale, "dtype":"F32", "block_size":[128,128],
            "semantics":"checkpoint-native: original FP8 weight with its original per-block FP32 scale"});
        weights.insert(scale.clone(), "native.safetensors".into());
        let contract = AttentionContract::parse(&config, &manifest).unwrap().unwrap();
        contract.check_selection("config.json", "model.safetensors.index.json", &["quantize_config.json".into()],
            Some(&weights)).unwrap();
        let native = &mut manifest["native_attention"][module];
        native["tensors"][&scale]["shape"] = serde_json::json!([2,1]);
        assert!(AttentionContract::parse(&config, &manifest).is_err());
    }

    #[test]
    fn published_muse_fused_source_provenance_matches_emitted_expert_headers() {
        use crate::SafetensorsTensorMetadata;
        use cuteafd_core::DType;
        use crate::formats::exl3_storage::derive_storage_map;
        // Muse 89998fb1 layer-0/expert-0 down header, copied from the published shard.
        // Gate/up emit the same geometry with independent rotations and trellises.
        let raw = serde_json::json!({
            "mcg":{"data_offsets":[16384,16388],"dtype":"I32","shape":[]},
            "suh":{"data_offsets":[16388,24580],"dtype":"F16","shape":[4096]},
            "svh":{"data_offsets":[24580,32772],"dtype":"F16","shape":[4096]},
            "trellis":{"data_offsets":[32772,8421380],"dtype":"I16","shape":[256,256,64]}
        });
        let layout = serde_json::json!({"down_shape":[256,4096,4096], "expert_axis":0,
            "gate_output_rows":[0,4096], "gate_up_shape":[256,8192,4096], "input_axis":2,
            "output_axis":1, "output_names":"model.language_model.layers.N.mlp.experts.E.{gate_proj,up_proj,down_proj}",
            "up_output_rows":[4096,8192]});
        let mut headers = Vec::new();
        for expert in 0..256 {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for (suffix, h) in raw.as_object().unwrap() {
                    let start = h["data_offsets"][0].as_u64().unwrap();
                    headers.push(SafetensorsTensorMetadata {
                        name: format!("model.language_model.layers.0.mlp.experts.{expert}.{projection}.{suffix}"),
                        dtype: match h["dtype"].as_str().unwrap() {
                            "I32" => DType::I32, "F16" => DType::F16, "I16" => DType::I16, _ => unreachable!(),
                        },
                        shape: serde_json::from_value(h["shape"].clone()).unwrap(), byte_offset: start,
                        byte_length: h["data_offsets"][1].as_u64().unwrap() - start,
                    });
                }
            }
        }
        let mut derived = derive_storage_map(&headers).unwrap();
        check_fused_expert_layout(&layout, &derived, 1).unwrap();
        let (mut compact, mut manifest) = publication();
        for key in ["core_bits", "head_bits", "mtp_bits", "lm_head"] {
            compact.as_object_mut().unwrap().remove(key);
            manifest.as_object_mut().unwrap().remove(key);
        }
        manifest["fused_expert_layout"] = layout.clone();
        check_gptqmodel_manifest(&compact, &manifest, &derived, 1).unwrap();
        for (key, wrong) in [("expert_axis", serde_json::json!(1)),
            ("gate_output_rows", serde_json::json!([1,4096])),
            ("gate_up_shape", serde_json::json!([255,8192,4096])),
            ("down_shape", serde_json::json!([256,4096,2048])),
            ("output_names", serde_json::json!("other.layers.N.mlp.experts.E.{gate_proj,up_proj,down_proj}"))] {
            let mut bad = layout.clone(); bad[key] = wrong;
            assert!(format!("{:#}", check_fused_expert_layout(&bad, &derived, 1).unwrap_err())
                .contains("fused_expert_layout"), "accepted {key}");
        }
        let name = "model.language_model.layers.0.mlp.experts.17.up_proj";
        derived.get_mut(name).unwrap().output_features = 2048;
        assert!(check_fused_expert_layout(&layout, &derived, 1).unwrap_err().to_string().contains(name));
        derived.remove(name);
        assert!(check_fused_expert_layout(&layout, &derived, 1).unwrap_err().to_string().contains("count"));
    }

    #[test]
    fn published_glm_k6_core_k4_mtp_key_set_checks_header_widths() {
        let (compact, manifest) = publication();
        let storage = storage();
        check_gptqmodel_manifest(&compact, &manifest, &storage, 2).unwrap();
        check_quantization_config(&compact).unwrap();
        check_header_widths(&compact, &storage, 2).unwrap();
        for (stem, key) in [
            ("model.layers.0.self_attn.q_a_proj", "core_bits"),
            ("model.layers.1.mlp.shared_experts.down_proj", "core_bits"),
            ("model.layers.2.self_attn.q_a_proj", "mtp_bits"),
            ("model.layers.2.mlp.experts.0.gate_proj", "mtp_bits"),
            ("model.layers.2.eh_proj", "mtp_bits"),
            ("lm_head", "head_bits"),
        ] {
            let mut wrong = storage.clone();
            wrong.get_mut(stem).unwrap().bits = 3;
            let error = check_gptqmodel_manifest(&compact, &manifest, &wrong, 2).unwrap_err().to_string();
            assert!(error.contains(stem) && error.contains(key), "{error}");
            assert!(check_header_widths(&compact, &wrong, 2).is_err());
        }
    }

    #[test]
    fn native_fp8_compact_and_full_claims_must_agree() {
        let official: Value = serde_json::from_str(include_str!(
            "../families/mistral_large4/official-config.json")).unwrap();
        let (mut compact, mut manifest) = publication();
        compact["native_quantization_config"] = official["quantization_config"].clone();
        manifest["native_quantization_config"] = official["quantization_config"].clone();
        check_gptqmodel_manifest(&compact, &manifest, &storage(), 2).unwrap();
        for key in ["quant_method", "activation_scheme", "weight_block_size", "modules_to_not_convert"] {
            let mut bad = manifest.clone();
            bad["native_quantization_config"].as_object_mut().unwrap().remove(key);
            assert!(check_gptqmodel_manifest(&compact, &bad, &storage(), 2).unwrap_err().to_string().contains(key));
        }
        compact.as_object_mut().unwrap().remove("native_quantization_config");
        assert!(check_gptqmodel_manifest(&compact, &manifest, &storage(), 2).unwrap_err().to_string()
            .contains("compact/full"));
    }

    #[test]
    fn mistral_consolidated_head_and_dspark_drafter_widths() {
        let module = |bits| Exl3Module { codebook: Exl3Codebook::Mcg, bits,
            input_features: 128, output_features: 128, tensors: BTreeMap::new() };
        let storage: Exl3StorageMap = BTreeMap::from([
            ("layers.0.attention.wq".into(), module(6)),
            ("layers.1.shared_experts.w2".into(), module(6)),
            ("layers.1.experts.0.w1".into(), module(4)),
            ("mtp_dspark.backbone.layers.0.attention.wq".into(), module(4)),
            ("output".into(), module(6)),
        ]);
        let (compact, manifest) = publication();
        check_gptqmodel_manifest(&compact, &manifest, &storage, 72).unwrap();
        let mut wrong = storage.clone();
        wrong.get_mut("output").unwrap().bits = 4;
        assert!(check_header_widths(&compact, &wrong, 72).unwrap_err().to_string().contains("head_bits"));
    }

    #[test]
    fn mtp_prefixes_and_backbone_boundary_select_only_mtp() {
        for stem in ["mtp.0.eh_proj", "mtp.layers.0.eh_proj", "model.mtp.layers.0.eh_proj",
            "language_model.model.layers.2.eh_proj", "model.language_model.layers.2.eh_proj"] {
            let mut storage = storage();
            let module = storage.remove("model.layers.2.eh_proj").unwrap();
            storage.insert(stem.into(), module);
            check_header_widths(&publication().0, &storage, 2).unwrap();
        }
    }

    #[test]
    fn mixed_width_contract_rejects_metadata_drift() {
        let (compact, manifest) = publication();
        for key in ["core_bits", "head_bits", "mtp_bits"] {
            for value in [Value::from(0), Value::from(6.5), Value::from("6"), Value::Null] {
                let mut wrong = manifest.clone();
                wrong[key] = value;
                assert!(check_gptqmodel_manifest(&compact, &wrong, &storage(), 2).is_err());
            }
            let mut wrong = compact.clone();
            wrong[key] = Value::from(3.0);
            assert!(check_gptqmodel_manifest(&wrong, &manifest, &storage(), 2).is_err());
        }
        for (key, value) in [("lm_head", Value::from(false)), ("lm_head", Value::from("true")),
            ("sym", Value::from(false)), ("calibration", Value::Null),
            ("per_module_metadata", Value::from("other.json:tensor_storage"))] {
            let mut wrong = manifest.clone();
            wrong[key] = value;
            assert!(check_gptqmodel_manifest(&compact, &wrong, &storage(), 2).is_err(), "accepted {key}");
        }
        let mut missing_head = storage();
        missing_head.remove("lm_head");
        assert!(check_header_widths(&manifest, &missing_head, 2).is_err());
        let mut native_head = manifest;
        native_head["lm_head"] = Value::from(false);
        native_head["head_bits"] = Value::from(16);
        check_header_widths(&native_head, &missing_head, 2).unwrap();
    }
}
