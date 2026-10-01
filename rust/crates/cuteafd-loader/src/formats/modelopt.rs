//! NVIDIA ModelOpt quantization metadata: which modules a ModelOpt export
//! quantized, and how (PLAN.md Phase 5).
//!
//! A ModelOpt checkpoint describes itself in two of its own files:
//! `hf_quant_config.json` (`{"producer", "quantization": {...}}`, or the same
//! keys flattened at the top level) and `config.json`'s
//! `quantization_config` (`quant_method: "modelopt"`, or a release's own
//! method with a `modelopt` producer, as DeepSeek V4.1's NVFP4 release keeps
//! `fp8` for its released FP8 parts). Both carry:
//!
//! - `quant_algo`: `NVFP4` (every quantized Linear is NVFP4 with
//!   `group_size`) or `MIXED_PRECISION` (the per-module table decides);
//! - `quantized_layers`: module name -> `{quant_algo, group_size}` (`NVFP4`,
//!   `FP8` per tensor, `FP8_BLOCK_SCALES` / `FP8_PB_WO`, `MXFP8`); a module may be a
//!   container (`model.layers.3.mlp.experts`) or a packed projection
//!   (`gate_up_proj`, expanded by `packed_modules_mapping`);
//! - `exclude_modules` / `ignore`: fnmatch globs of modules left as released.
//!
//! [`ModelOpt::declared`] resolves a logical weight's stem to its
//! declaration; [`ModelOpt::check`] verifies the tensors' detected operand
//! against it. The tensors are the truth about storage; the metadata says
//! which ones ModelOpt wrote: an NVFP4 operand the metadata excludes, or a
//! listed module stored in another format, is a contradiction the loaders
//! refuse rather than guess about.
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

use crate::plan::format::{Encoding, QuantOperand, RowTiling, ScaleEncoding};

/// A ModelOpt quantization algorithm for one module.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
#[serde(rename_all = "snake_case", tag = "algo")]
pub enum ModelOptAlgo {
    /// Packed E2M1 with E4M3 scales per `group` values along K and an FP32
    /// per-tensor `weight_scale_2`.
    Nvfp4 { group: usize },
    /// E4M3 with one FP32 scale per tensor.
    Fp8,
    /// E4M3 with `block` x `block` scales.
    Fp8BlockScales { block: usize },
    /// E4M3 with UE8M0 scales per `group` values along K.
    Mxfp8 { group: usize },
}

impl ModelOptAlgo {
    fn parse(spec: &Value, default_group: Option<usize>, module: &str) -> Result<Self, ModelOptError> {
        let algo = spec.get("quant_algo").and_then(Value::as_str).ok_or_else(|| ModelOptError::Malformed {
            what: format!("quantized_layers.{module}"),
            reason: "has no quant_algo".into(),
        })?;
        let group = spec.get("group_size").and_then(Value::as_u64).map(|g| g as usize).or(default_group);
        let need = |what: &str| ModelOptError::Malformed {
            what: format!("quantized_layers.{module}"),
            reason: format!("{algo} needs a {what}"),
        };
        Ok(match algo {
            // W4A16_NVFP4: the weight-only export of the same storage (no input_scale).
            "NVFP4" | "W4A16_NVFP4" => Self::Nvfp4 { group: group.ok_or_else(|| need("group_size"))? },
            "FP8" => Self::Fp8,
            // FP8_PB_WO: the per-block weight-only spelling (config.json) of the same storage.
            "FP8_BLOCK_SCALES" | "FP8_PB_WO" => Self::Fp8BlockScales { block: group.ok_or_else(|| need("group_size"))? },
            "MXFP8" => Self::Mxfp8 { group: group.unwrap_or(32) },
            other => return Err(ModelOptError::Unsupported { module: module.into(), algo: other.into() }),
        })
    }

    pub fn label(&self) -> String {
        match self {
            Self::Nvfp4 { group } => format!("NVFP4 g{group}"),
            Self::Fp8 => "FP8 per-tensor".into(),
            Self::Fp8BlockScales { block } => format!("FP8 {block}x{block} blocks"),
            Self::Mxfp8 { group } => format!("MXFP8 g{group}"),
        }
    }

    /// Whether `operand` is stored the way this algorithm writes it.
    pub fn matches(&self, operand: &QuantOperand) -> bool {
        match *self {
            Self::Nvfp4 { group } => operand.is_nvfp4() && operand.scale.as_ref().is_some_and(|s| s.cols == group),
            // Per-tensor FP8; sharded tables keep their one scale in a tensor of
            // its own (a lone scalar `weight_scale` group).
            Self::Fp8 => matches!(operand.encoding, Encoding::E4m3)
                || operand.logical.iter().product::<usize>() == 1
                    && matches!(operand.encoding, Encoding::F32 | Encoding::Bf16),
            Self::Fp8BlockScales { block } => {
                operand.is_fp8_block(block, &[ScaleEncoding::F32, ScaleEncoding::Bf16, ScaleEncoding::Ue8m0])
            }
            Self::Mxfp8 { group } => operand.encoding == Encoding::E4m3
                && operand.scale.as_ref().is_some_and(|s| s.encoding == ScaleEncoding::Ue8m0 && s.cols == group
                    && matches!(s.rows, RowTiling::Uniform { rows: 1 })),
        }
    }
}

/// What the metadata says about one logical weight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Declared<'a> {
    /// Quantized by ModelOpt (`module` is the declaring entry, or the
    /// checkpoint-wide algorithm's `*`).
    Quantized { algo: &'a ModelOptAlgo, module: &'a str },
    /// Left as released by an `exclude_modules` / `ignore` glob.
    Excluded { glob: &'a str },
    /// Neither listed nor excluded (MIXED_PRECISION: as released).
    Unlisted,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ModelOptError {
    #[error("reading {path}: {reason}")]
    Read { path: String, reason: String },
    #[error("ModelOpt metadata {what}: {reason}")]
    Malformed { what: String, reason: String },
    #[error("ModelOpt quantizes {module} as {algo}, which this build does not read")]
    Unsupported { module: String, algo: String },
    #[error("hf_quant_config.json and config.json disagree: {0}")]
    Conflict(String),
}

/// A checkpoint's ModelOpt quantization metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelOpt {
    /// `name version` of the producer.
    pub producer: String,
    /// `NVFP4` or `MIXED_PRECISION`.
    pub quant_algo: String,
    pub kv_cache: Option<String>,
    /// The checkpoint-wide algorithm (`quant_algo` other than MIXED_PRECISION).
    pub default: Option<ModelOptAlgo>,
    /// Declared modules (packed projections expanded), sorted by name.
    pub layers: BTreeMap<String, ModelOptAlgo>,
    pub exclude: Vec<String>,
    /// The file the metadata came from.
    pub source: &'static str,
}

const MAX_JSON_BYTES: u64 = 64 * 1024 * 1024;

fn read_json(path: &Path) -> Result<Option<Value>, ModelOptError> {
    use std::io::Read;
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(ModelOptError::Read { path: path.display().to_string(), reason: error.to_string() }),
    };
    let mut bytes = Vec::new();
    file.take(MAX_JSON_BYTES + 1).read_to_end(&mut bytes)
        .map_err(|error| ModelOptError::Read { path: path.display().to_string(), reason: error.to_string() })?;
    if bytes.len() as u64 > MAX_JSON_BYTES {
        return Err(ModelOptError::Read { path: path.display().to_string(), reason: "too large".into() });
    }
    serde_json::from_slice(&bytes).map(Some)
        .map_err(|error| ModelOptError::Read { path: path.display().to_string(), reason: error.to_string() })
}

fn producer_name(value: &Value) -> Option<&str> {
    value.get("producer").and_then(|p| p.get("name")).and_then(Value::as_str)
}

fn producer(value: &Value) -> String {
    let producer = value.get("producer");
    let field = |key: &str| producer.and_then(|p| p.get(key)).and_then(Value::as_str).unwrap_or("?");
    format!("{} {}", field("name"), field("version"))
}

impl ModelOpt {
    /// Reads `snapshot`'s ModelOpt metadata given its `config.json`; `None`
    /// for checkpoints ModelOpt did not produce.
    pub fn read(snapshot: &Path, config: &Value) -> Result<Option<Self>, ModelOptError> {
        let hf = read_json(&snapshot.join("hf_quant_config.json"))?;
        let quant = config.get("quantization_config")
            .or_else(|| config.get("text_config").and_then(|t| t.get("quantization_config")));
        Self::from_json(hf.as_ref(), quant)
    }

    /// The metadata from `hf_quant_config.json` (when present) checked
    /// against `config.json`'s `quantization_config`.
    pub fn from_json(hf: Option<&Value>, config: Option<&Value>) -> Result<Option<Self>, ModelOptError> {
        let config = config.filter(|c| {
            c.get("quant_method").and_then(Value::as_str) == Some("modelopt") || producer_name(c) == Some("modelopt")
        });
        let hf = hf.filter(|h| producer_name(h) == Some("modelopt") || h.get("quantization").is_some());
        let (body, root, source) = match (hf, config) {
            (Some(hf), _) => (hf.get("quantization").unwrap_or(hf), hf, "hf_quant_config.json"),
            (None, Some(config)) => (config, config, "config.json"),
            (None, None) => return Ok(None),
        };
        let parsed = Self::parse(body, root, source)?;
        if let (Some(_), Some(config)) = (hf, config) {
            let other = Self::parse(config, config, "config.json")?;
            if other.quant_algo != parsed.quant_algo {
                return Err(ModelOptError::Conflict(format!("quant_algo {} vs {}", parsed.quant_algo, other.quant_algo)));
            }
            for (module, algo) in &other.layers {
                if parsed.layers.get(module).is_some_and(|mine| mine != algo) {
                    return Err(ModelOptError::Conflict(format!("{module}: {} vs {}", parsed.layers[module].label(),
                        algo.label())));
                }
            }
        }
        Ok(Some(parsed))
    }

    fn parse(body: &Value, root: &Value, source: &'static str) -> Result<Self, ModelOptError> {
        let quant_algo = body.get("quant_algo").and_then(Value::as_str).ok_or_else(|| ModelOptError::Malformed {
            what: source.into(),
            reason: "has no quant_algo".into(),
        })?.to_owned();
        let group = body.get("group_size").and_then(Value::as_u64).map(|g| g as usize);
        let default = match quant_algo.as_str() {
            "MIXED_PRECISION" => None,
            _ => Some(ModelOptAlgo::parse(&serde_json::json!({"quant_algo": quant_algo,
                "group_size": group.or_else(|| config_group(body))}), group, "*")?),
        };
        let packed: BTreeMap<String, Vec<String>> = body.get("packed_modules_mapping")
            .and_then(|m| serde_json::from_value(m.clone()).ok()).unwrap_or_default();
        let mut layers = BTreeMap::new();
        if let Some(table) = body.get("quantized_layers").and_then(Value::as_object) {
            for (module, spec) in table {
                let algo = ModelOptAlgo::parse(spec, group, module)?;
                match module.rsplit_once('.').and_then(|(parent, leaf)| Some((parent, packed.get(leaf)?))) {
                    Some((parent, parts)) => {
                        for part in parts {
                            layers.insert(format!("{parent}.{part}"), algo.clone());
                        }
                    }
                    None => {
                        layers.insert(module.clone(), algo);
                    }
                }
            }
        }
        if default.is_none() && layers.is_empty() {
            return Err(ModelOptError::Malformed { what: source.into(),
                reason: "MIXED_PRECISION without quantized_layers".into() });
        }
        let exclude = ["exclude_modules", "ignore"].iter()
            .find_map(|key| body.get(*key).and_then(Value::as_array))
            .map(|globs| globs.iter().filter_map(|g| g.as_str().map(str::to_owned)).collect())
            .unwrap_or_default();
        let kv_cache = body.get("kv_cache_quant_algo").and_then(Value::as_str).map(str::to_owned)
            .or_else(|| body.get("kv_cache_scheme").filter(|s| s.is_object())
                .map(|s| format!("{}-bit float", s.get("num_bits").and_then(Value::as_u64).unwrap_or(0))));
        Ok(Self { producer: producer(root), quant_algo, kv_cache, default, layers, exclude, source })
    }

    /// The declaration covering the logical weight `stem` (`a.b.weight`'s `a.b`).
    pub fn declared<'a>(&'a self, stem: &str) -> Declared<'a> {
        // The longest declared module that is the stem or one of its parents.
        let mut parent = Some(stem);
        while let Some(module) = parent {
            if let Some((name, algo)) = self.layers.get_key_value(module) {
                return Declared::Quantized { algo, module: name };
            }
            parent = module.rsplit_once('.').map(|(head, _)| head);
        }
        if let Some(glob) = self.exclude.iter().find(|glob| glob_match(glob, stem)) {
            return Declared::Excluded { glob };
        }
        match &self.default {
            Some(algo) => Declared::Quantized { algo, module: "*" },
            None => Declared::Unlisted,
        }
    }

    /// Checks `operand` (the tensors of `stem`) against the declaration: a
    /// listed module must be stored as its algorithm writes it, and an NVFP4
    /// operand must be declared NVFP4. Unlisted and excluded weights keep the
    /// release's own storage (BF16, its FP8, a checkpoint-wide NVFP4
    /// export's unquantized parameters).
    pub fn check(&self, stem: &str, operand: &QuantOperand) -> Result<(), String> {
        let declared = self.declared(stem);
        match declared {
            Declared::Quantized { algo, module } if module != "*" && !algo.matches(operand) => Err(format!(
                "{} declares {module} {}, but its tensors store {}", self.source, algo.label(), operand.label())),
            // Checkpoint-wide NVFP4: only quantized operands can be checked
            // (BF16 parameters are not Linear weights).
            Declared::Quantized { algo, .. } if operand.encoding == Encoding::E2m1 && !algo.matches(operand) => {
                Err(format!("{} quantizes every Linear as {}, but {stem} stores {}", self.source, algo.label(),
                    operand.label()))
            }
            Declared::Excluded { glob } if operand.is_nvfp4() => Err(format!(
                "{} excludes {stem} ({glob}), but its tensors store NVFP4", self.source)),
            Declared::Unlisted if operand.is_nvfp4() => Err(format!(
                "{} does not list {stem} as NVFP4, but its tensors store it", self.source)),
            _ => Ok(()),
        }
    }

    /// Whether the routed experts are NVFP4: every declared `.experts` module
    /// is (a checkpoint-wide NVFP4 export may list none), and at least one
    /// is declared when the export is MIXED_PRECISION.
    pub fn experts_nvfp4(&self) -> bool {
        let experts: Vec<&ModelOptAlgo> = self.layers.iter()
            .filter(|(module, _)| module.contains(".experts")).map(|(_, algo)| algo).collect();
        match &self.default {
            Some(ModelOptAlgo::Nvfp4 { .. }) => experts.iter().all(|a| matches!(a, ModelOptAlgo::Nvfp4 { .. })),
            _ => experts.iter().any(|a| matches!(a, ModelOptAlgo::Nvfp4 { .. })),
        }
    }

    /// One line: producer, algorithm and the declared modules by algorithm.
    pub fn summary(&self) -> String {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for algo in self.layers.values() {
            *counts.entry(algo.label()).or_default() += 1;
        }
        let mut parts = Vec::new();
        if let Some(algo) = &self.default {
            parts.push(format!("every Linear {}", algo.label()));
        }
        parts.extend(counts.iter().map(|(label, n)| format!("{label} x{n} modules")));
        format!("{} ({}) {}: {}; {} excluded globs{}", self.producer, self.source, self.quant_algo, parts.join(", "),
            self.exclude.len(), self.kv_cache.as_deref().map_or(String::new(), |kv| format!("; KV cache {kv}")))
    }
}

/// `group_size` of the first `config_groups` weights entry (compressed-tensors spelling).
fn config_group(body: &Value) -> Option<usize> {
    body.get("config_groups")?.as_object()?.values()
        .find_map(|group| group.get("weights")?.get("group_size")?.as_u64()).map(|g| g as usize)
}

/// fnmatch-style match: `*` any run (dots included), `?` one character.
pub fn glob_match(glob: &str, name: &str) -> bool {
    let (glob, name) = (glob.as_bytes(), name.as_bytes());
    let (mut g, mut n, mut star, mut mark) = (0usize, 0usize, None, 0usize);
    while n < name.len() {
        if g < glob.len() && (glob[g] == b'?' || glob[g] == name[n]) {
            g += 1;
            n += 1;
        } else if g < glob.len() && glob[g] == b'*' {
            star = Some(g);
            mark = n;
            g += 1;
        } else if let Some(s) = star {
            g = s + 1;
            mark += 1;
            n = mark;
        } else {
            return false;
        }
    }
    glob[g..].iter().all(|&c| c == b'*')
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn globs_match_like_fnmatch() {
        assert!(glob_match("*.attn.*", "layers.0.attn.wq_a"));
        assert!(glob_match("model.layers.78*", "model.layers.78.mlp.experts.3.up_proj"));
        assert!(!glob_match("model.layers.78*", "model.layers.7.mlp.gate"));
        assert!(glob_match("head", "head"));
        assert!(!glob_match("head", "lm_head"));
        assert!(glob_match("model.language_model.layers.1?.self_attn*", "model.language_model.layers.12.self_attn.q"));
    }

    fn nvidia_glm53() -> Value {
        json!({"producer": {"name": "modelopt", "version": "0.0.0"}, "quantization": {
            "quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": "FP8",
            "quantized_layers": {
                "model.layers.0.mlp.gate_proj": {"quant_algo": "FP8"},
                "model.layers.1.mlp.gate_up_proj": {"quant_algo": "FP8"},
                "model.layers.3.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16}},
            "exclude_modules": ["model.layers.78*"],
            "packed_modules_mapping": {"gate_up_proj": ["gate_proj", "up_proj"]}}})
    }

    #[test]
    fn declarations_resolve_containers_packed_projections_and_globs() {
        let mo = ModelOpt::from_json(Some(&nvidia_glm53()), None).unwrap().unwrap();
        let nvfp4 = ModelOptAlgo::Nvfp4 { group: 16 };
        assert_eq!(mo.declared("model.layers.3.mlp.experts.17.gate_proj"),
            Declared::Quantized { algo: &nvfp4, module: "model.layers.3.mlp.experts" });
        assert_eq!(mo.declared("model.layers.1.mlp.up_proj"),
            Declared::Quantized { algo: &ModelOptAlgo::Fp8, module: "model.layers.1.mlp.up_proj" });
        assert_eq!(mo.declared("model.layers.78.mlp.experts.0.down_proj"), Declared::Excluded { glob: "model.layers.78*" });
        assert_eq!(mo.declared("model.layers.5.self_attn.o_proj"), Declared::Unlisted);
        assert!(mo.experts_nvfp4());
        assert_eq!(mo.kv_cache.as_deref(), Some("FP8"));
        assert!(mo.summary().contains("NVFP4 g16 x1 modules"), "{}", mo.summary());
    }

    #[test]
    fn checkpoint_wide_nvfp4_and_flat_files() {
        // nvidia/GLM-5.3-Flash-NVFP4: NVFP4 everywhere but the excluded globs.
        let hf = json!({"producer": {"name": "modelopt", "version": "0.47"}, "quantization": {
            "quant_algo": "NVFP4", "group_size": 16, "kv_cache_quant_algo": "FP8",
            "exclude_modules": ["lm_head", "model.language_model.layers.0.self_attn*"]}});
        let mo = ModelOpt::from_json(Some(&hf), None).unwrap().unwrap();
        assert!(matches!(mo.declared("model.language_model.layers.4.mlp.experts.2.up_proj"),
            Declared::Quantized { module: "*", .. }));
        assert!(matches!(mo.declared("model.language_model.layers.0.self_attn.q_proj"), Declared::Excluded { .. }));
        assert!(mo.experts_nvfp4());
        // local-inference-lab: the keys at the top level, MXFP8 MTP experts.
        let flat = json!({"producer": {"name": "modelopt", "version": "0.39"}, "quant_algo": "MIXED_PRECISION",
            "quant_method": "modelopt", "ignore": [], "quantized_layers": {
                "model.language_model.layers.3.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16},
                "model.language_model.layers.45.mlp.experts": {"quant_algo": "MXFP8", "group_size": 32}}});
        let mo = ModelOpt::from_json(Some(&flat), Some(&flat)).unwrap().unwrap();
        assert_eq!(mo.layers.len(), 2);
        assert!(mo.experts_nvfp4());
        // Not ModelOpt.
        assert!(ModelOpt::from_json(None, Some(&json!({"quant_method": "fp8"}))).unwrap().is_none());
    }

    #[test]
    fn contradictions_are_typed() {
        let mut hf = nvidia_glm53();
        hf["quantization"]["quantized_layers"]["model.layers.3.mlp.experts"]["quant_algo"] = json!("INT4_AWQ");
        assert!(matches!(ModelOpt::from_json(Some(&hf), None), Err(ModelOptError::Unsupported { .. })));
        let config = json!({"quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16});
        assert!(matches!(ModelOpt::from_json(Some(&nvidia_glm53()), Some(&config)), Err(ModelOptError::Conflict(_))));
        let empty = json!({"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "MIXED_PRECISION"}});
        assert!(matches!(ModelOpt::from_json(Some(&empty), None), Err(ModelOptError::Malformed { .. })));
    }
}
