//! EXLR8 v2 inventory and weight-only fit, from FORMAT.md, never a serving claim.
//! Unit payloads are range-readable; a selected projection's block scales are
//! charged once per expert/width/rank, not once per owned unit.
use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use super::checkpoint::read_json;
use super::format::{
    BlockScale, Encoding, RowTiling, ScaleDirection, ScaleEncoding, TrellisCodebook,
};

const FORMAT: &str = "exlr8-v2";
const SCHEME: &str = "exlr8-e4m3-v2";

struct Header {
    value: Value,
    start: u64,
    bytes: u64,
}
impl Header {
    fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mut size = [0; 8];
        file.read_exact(&mut size)?;
        let size = u64::from_le_bytes(size);
        ensure!(size <= 64 << 20, "oversized EXLR8 header");
        let mut data = vec![0; size as usize];
        file.read_exact(&mut data)?;
        Ok(Self {
            value: serde_json::from_slice(&data)?,
            start: size + 8,
            bytes: file.metadata()?.len(),
        })
    }
    fn validate(&self) -> Result<()> {
        ensure!(
            self.value["__metadata__"]["format"] == FORMAT,
            "unsupported EXLR8 format {}",
            self.value["__metadata__"]["format"]
        );
        ensure!(
            self.value["__metadata__"]["scheme"] == SCHEME,
            "unsupported EXLR8 scheme {}: add its decoder",
            self.value["__metadata__"]["scheme"]
        );
        Ok(())
    }
    fn extent(&self, name: &str) -> Result<(u64, u64)> {
        let v = &self.value[name];
        let a = v["data_offsets"][0]
            .as_u64()
            .context("missing data offset")?;
        let b = v["data_offsets"][1].as_u64().context("missing data end")?;
        ensure!(
            b >= a && b <= self.bytes.saturating_sub(self.start),
            "invalid EXLR8 tensor extent {name}"
        );
        Ok((self.start + a, b - a))
    }
    fn read(&self, path: &Path, name: &str) -> Result<Vec<u8>> {
        let (offset, length) = self.extent(name)?;
        ensure!(
            length <= 64 << 20,
            "EXLR8 metadata tensor too large: {name}"
        );
        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0; length as usize];
        file.read_exact(&mut bytes)?;
        Ok(bytes)
    }
}

pub fn recognizes(snapshot: &Path) -> Result<bool> {
    let directory = snapshot.join("experts");
    if !directory.is_dir() {
        return Ok(false);
    }
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        if entry
            .file_name()
            .to_string_lossy()
            .starts_with("exlr8-layer-")
            && entry
                .file_name()
                .to_string_lossy()
                .ends_with(".index.safetensors")
        {
            let header = Header::open(&entry.path())?;
            // A recognized container with a new scheme gets a named refusal in inspect.
            return Ok(header.value["__metadata__"]["format"] == FORMAT);
        }
    }
    Ok(false)
}

#[derive(Debug, Clone, Serialize)]
pub struct DenseOperand {
    pub tensor: String,
    pub shard: String,
    pub encoding: Encoding,
    pub scale: Option<BlockScale>,
    pub shape: Vec<usize>,
    pub bytes: u64,
    pub sharded_side: usize,
    pub channels_per_shard: usize,
    pub rotation: Option<String>,
    pub kernel: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RankBytes {
    pub trellis: u64,
    pub channels: u64,
    pub blocks: u64,
    pub signs: u64,
    pub total: u64,
}
impl RankBytes {
    fn empty() -> Self {
        Self {
            trellis: 0,
            channels: 0,
            blocks: 0,
            signs: 0,
            total: 0,
        }
    }
    fn finish(&mut self) {
        self.total = self.trellis + self.channels + self.blocks + self.signs;
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ManifestPlan {
    pub name: String,
    pub version: String,
    pub set_id: String,
    pub routed_bpw: f64,
    pub widths: BTreeMap<usize, u64>,
    pub ranks: BTreeMap<usize, Vec<RankBytes>>,
    pub shared_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct Fit {
    pub manifest: String,
    pub rtx: usize,
    pub sparks: usize,
    pub rank_bytes: Vec<u64>,
    pub coordinator_bytes: u64,
    pub coordinator_per_gpu_upper_bound: u64,
    pub weights_fit: bool,
}

#[derive(Debug, Serialize)]
pub struct Exlr8Plan {
    pub format: String,
    pub scheme: String,
    pub selected_manifest: String,
    pub manifest_choice: String,
    pub dense_source: String,
    pub dense: Vec<DenseOperand>,
    pub dense_bytes: u64,
    pub manifests: Vec<ManifestPlan>,
    pub fit: Vec<Fit>,
    pub selected_weights_fit: bool,
    pub missing_kernels: Vec<String>,
    pub supported: bool,
    pub notes: Vec<String>,
}
impl Exlr8Plan {
    pub fn render(&self) -> String {
        let mut text = format!("EXLR8 {} / {}: unsupported (planner only)\nManifest: {}\n{}\nCoordinator dense: {} bytes ({})\n",
            self.format, self.scheme, self.selected_manifest, self.manifest_choice, self.dense_bytes, self.dense_source);
        for operand in &self.dense {
            text.push_str(&format!(
                "  {} {:?} {:?} scale={:?}: {} bytes{}\n",
                operand.tensor,
                operand.shape,
                operand.encoding,
                operand.scale,
                operand.bytes,
                operand
                    .kernel
                    .as_ref()
                    .map(|k| format!("; add kernel {k}"))
                    .unwrap_or_default()
            ));
        }
        text.push_str("Weight-only fit (no KV/workspace/drafter admission; 2 RTX conservatively replicates dense):\n");
        for fit in &self.fit {
            text.push_str(&format!("  {} {} RTX + {} Sparks: ranks {:?} bytes; coordinator {} bytes/GPU upper bound; {}\n",
                fit.manifest, fit.rtx, fit.sparks, fit.rank_bytes, fit.coordinator_per_gpu_upper_bound,
                if fit.weights_fit { "weights fit" } else { "does not fit" }));
        }
        for kernel in &self.missing_kernels {
            text.push_str(&format!("unsupported: add kernel {kernel}\n"));
        }
        for note in &self.notes {
            text.push_str(&format!("note: {note}\n"));
        }
        text
    }
}

fn number(config: &Value, key: &str) -> Result<usize> {
    usize::try_from(
        config[key]
            .as_u64()
            .with_context(|| format!("missing geometry {key}"))?,
    )
    .context("geometry overflow")
}

fn files(snapshot: &Path, directory: &str) -> Result<Vec<PathBuf>> {
    let mut files = std::fs::read_dir(snapshot.join(directory))?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.retain(|path| path.extension().is_some_and(|ext| ext == "safetensors"));
    files.sort();
    Ok(files)
}

fn dense_operands(snapshot: &Path, layers: usize) -> Result<Vec<DenseOperand>> {
    let mut operands = Vec::new();
    for path in files(snapshot, "dense")? {
        let header = Header::open(&path)?;
        header.validate()?;
        let metadata = &header.value["__metadata__"];
        let layer = metadata["layer"]
            .as_str()
            .map(str::parse::<usize>)
            .transpose()?;
        if layer.is_some_and(|layer| layer >= layers) {
            continue;
        }
        let entries = header
            .value
            .as_object()
            .context("header must be an object")?;
        let prefix = metadata["tensor_prefix"]
            .as_str()
            .unwrap_or("model.layers.");
        let rotations = metadata["rotations"]
            .as_str()
            .unwrap_or("attn_in,q_resid,mlp_in")
            .split(',')
            .collect::<Vec<_>>();
        let mut claimed = BTreeSet::new();
        for name in entries.keys().filter(|name| name.ends_with(".record")) {
            ensure!(
                entries[name]["dtype"] == "I64" && entries[name]["shape"] == serde_json::json!([8]),
                "invalid dense record {name}"
            );
            let data = header.read(&path, name)?;
            ensure!(data.len() == 64, "invalid dense record length {name}");
            let r = data
                .chunks_exact(8)
                .map(|v| i64::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>();
            ensure!(
                r.iter().all(|v| *v >= 0) && r[1] > 0 && r[2] > 0 && r[4] <= 2,
                "invalid dense record fields {name}: {r:?}"
            );
            let (encoding, suffixes, kernel): (Encoding, &[&str], Option<&str>) = match r[0] {
                1 => {
                    ensure!(
                        (2..=4).contains(&r[3]) && r[6] as usize <= rotations.len(),
                        "invalid dense trellis width/rotation {name}"
                    );
                    (
                        Encoding::Trellis {
                            k: r[3] as usize,
                            codebook: TrellisCodebook::E4M3,
                            rotation: r[6] as usize,
                        },
                        &["trellis", "suh", "svh"],
                        Some("trellis_linear_e4m3_sm120_with_rotation_ids"),
                    )
                }
                2 => (
                    Encoding::Int4 { group: 32 },
                    &["q4", "s4"],
                    Some("int4_g32_linear_sm120"),
                ),
                3 => (
                    Encoding::E4m3,
                    &["fp8", "e8m0"],
                    Some("mxfp8_linear_e8m0_k32_sm120"),
                ),
                4 => (Encoding::E4m3, &["fp8", "scale"], None),
                5 => (
                    Encoding::E4m3,
                    &["fp8", "scale"],
                    Some("fp8_row_embedding_lm_head_sm120"),
                ),
                6 => (Encoding::Bf16, &["bf16"], None),
                7 => (Encoding::F32, &["fp32"], None),
                other => {
                    bail!("unsupported dense encoding {other} in {name}: add its native kernel")
                }
            };
            let stem = name.trim_end_matches(".record");
            let mut bytes = 0;
            for suffix in suffixes {
                let member = format!("{stem}.{suffix}");
                bytes += header.extent(&member)?.1;
                claimed.insert(member);
            }
            let scale = if matches!(r[0], 2..=5) {
                let (suffix, encoding, rows, cols) = match r[0] {
                    2 => ("s4", ScaleEncoding::F16, 1, 32),
                    3 => ("e8m0", ScaleEncoding::Ue8m0, 1, 32),
                    4 => ("scale", ScaleEncoding::F32, 128, 128),
                    _ => ("scale", ScaleEncoding::F32, 1, r[2] as usize),
                };
                Some(BlockScale {
                    tensor: suffix.into(),
                    encoding,
                    direction: ScaleDirection::Multiply,
                    rows: RowTiling::Uniform { rows },
                    cols,
                    grid: serde_json::from_value(
                        entries[&format!("{stem}.{suffix}")]["shape"].clone(),
                    )?,
                })
            } else {
                None
            };
            let tensor = layer.map_or_else(
                || {
                    if stem == "lm_head" {
                        format!("{stem}.weight")
                    } else {
                        format!("model.{stem}.weight")
                    }
                },
                |l| format!("{prefix}{l}.{stem}.weight"),
            );
            operands.push(DenseOperand {
                tensor,
                shard: path.strip_prefix(snapshot)?.display().to_string(),
                encoding,
                scale,
                shape: vec![r[1] as usize, r[2] as usize],
                bytes,
                sharded_side: r[4] as usize,
                channels_per_shard: r[5] as usize,
                rotation: if r[6] == 0 {
                    None
                } else {
                    Some(rotations[r[6] as usize - 1].trim().to_string())
                },
                kernel: kernel.map(str::to_string),
            });
            claimed.insert(name.clone());
            claimed.insert(format!("{stem}.checksum"));
        }
        for (name, value) in entries {
            if name.starts_with("__") || claimed.contains(name) {
                continue;
            }
            let encoding = match value["dtype"].as_str() {
                Some("BF16") => Encoding::Bf16,
                Some("F32") => Encoding::F32,
                Some("F16") => Encoding::F16,
                other => bail!("unsupported unrecorded dense tensor {name}: {other:?}"),
            };
            let tensor = layer.map_or_else(
                || format!("model.{name}"),
                |l| format!("{prefix}{l}.{name}"),
            );
            let shape = serde_json::from_value(value["shape"].clone())?;
            operands.push(DenseOperand {
                tensor,
                shard: path.strip_prefix(snapshot)?.display().to_string(),
                encoding,
                scale: None,
                shape,
                bytes: header.extent(name)?.1,
                sharded_side: 0,
                channels_per_shard: 0,
                rotation: None,
                kernel: None,
            });
        }
    }
    Ok(operands)
}

fn width(value: &Value, stored: &BTreeSet<usize>, context: &str) -> Result<usize> {
    let k = value.as_u64().context("manifest width must be unsigned")? as usize;
    ensure!(
        stored.contains(&k) && (2..=4).contains(&k),
        "{context}: manifest requests absent K{k}"
    );
    Ok(k)
}

struct Layer {
    id: usize,
    widths: BTreeSet<usize>,
    shared_widths: BTreeSet<usize>,
}
fn manifest(
    snapshot: &Path,
    name: &str,
    hidden: usize,
    units: usize,
    experts: usize,
    layers: &[Layer],
) -> Result<ManifestPlan> {
    ensure!(
        !name.is_empty()
            && name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_'),
        "manifest must be a published name, not a path"
    );
    let value = read_json(&snapshot.join(format!("manifests/exlr8-manifest-{name}.json")))?;
    ensure!(
        value["format"] == "exlr8-v2-manifest" && value["name"] == name,
        "invalid EXLR8 manifest identity"
    );
    let set_id = value["set_id"]
        .as_str()
        .context("missing manifest set_id")?
        .to_owned();
    ensure!(
        set_id.len() == 64 && set_id.bytes().all(|c| c.is_ascii_hexdigit()),
        "invalid EXLR8 set_id"
    );
    let mut ranks = (1..=7)
        .map(|n| (n, vec![RankBytes::empty(); n]))
        .collect::<BTreeMap<_, _>>();
    let mut widths = BTreeMap::new();
    let mut shared_bytes = 0;
    for layer in layers {
        let l = &value["layers"][layer.id.to_string()];
        let mut selected = Vec::new();
        for projection in ["gate", "up", "down"] {
            let rows = l[projection]
                .as_array()
                .with_context(|| format!("missing {projection} layer {}", layer.id))?;
            ensure!(rows.len() == experts, "manifest expert count mismatch");
            for (e, row) in rows.iter().enumerate() {
                if projection != "down" {
                    ensure!(
                        row.as_array().is_some_and(|r| r.len() == units),
                        "manifest unit count mismatch"
                    );
                }
                for u in 0..units {
                    let k = width(
                        if projection == "down" { row } else { &row[u] },
                        &layer.widths,
                        projection,
                    )?;
                    *widths.entry(k).or_insert(0) += 1;
                    selected.push((e, u, projection, k));
                }
            }
            let shared = &l["shared"][projection];
            if projection != "down" {
                ensure!(
                    shared.as_array().is_some_and(|r| r.len() == units),
                    "manifest shared unit count mismatch"
                );
            }
            let mut tiers = BTreeSet::new();
            for u in 0..units {
                let k = width(
                    if projection == "down" {
                        shared
                    } else {
                        &shared[u]
                    },
                    &layer.shared_widths,
                    "shared",
                )?;
                shared_bytes += (hidden * 16 * k + 128 * 2) as u64;
                tiers.insert(k);
            }
            shared_bytes += (tiers.len() * (hidden / 128) * 2) as u64;
        }
        shared_bytes += (2 * hidden) as u64;
        for (&n, sizes) in &mut ranks {
            let mut blocks = BTreeSet::new();
            for &(e, u, projection, k) in &selected {
                let rank = owner(units, e, u, layer.id, n);
                sizes[rank].trellis += (hidden * 16 * k) as u64;
                sizes[rank].channels += 128 * 2;
                if blocks.insert((rank, e, projection, k)) {
                    sizes[rank].blocks += (hidden / 128 * 2) as u64;
                }
            }
            for size in sizes {
                size.signs += (2 * hidden) as u64;
                size.finish();
            }
        }
    }
    let count = widths.values().sum::<u64>();
    ensure!(count > 0, "empty EXLR8 manifest");
    let routed_bpw = widths
        .iter()
        .map(|(k, count)| (*k as u64 * count) as f64)
        .sum::<f64>()
        / count as f64;
    Ok(ManifestPlan {
        name: name.into(),
        version: value["version"]
            .as_str()
            .context("missing manifest version")?
            .into(),
        set_id,
        routed_bpw,
        widths,
        ranks,
        shared_bytes,
    })
}

/// EXLR8 FORMAT.md unit ownership, independent of any precomputed index slice.
pub fn owner(units: usize, expert: usize, unit: usize, layer: usize, ranks: usize) -> usize {
    (units * expert + unit + layer) % ranks
}

fn geometry(
    snapshot: &Path,
    config: &Value,
    hidden: usize,
    units: usize,
    experts: usize,
    count: usize,
    first: usize,
) -> Result<Vec<Layer>> {
    let mut layers = Vec::new();
    for layer in first..count {
        let path = snapshot.join(format!("experts/exlr8-layer-{layer:03}.index.safetensors"));
        let header = Header::open(&path)?;
        header.validate()?;
        let metadata = &header.value["__metadata__"];
        for (key, default) in [
            ("hidden_size", 6144),
            ("moe_intermediate_size", 2048),
            ("n_routed_experts", 256),
            ("num_hidden_layers", 78),
            ("first_moe_layer", 3),
        ] {
            let expected = if key == "first_moe_layer" {
                first
            } else {
                number(config, key)?
            };
            let declared = metadata[key]
                .as_str()
                .map(str::parse::<usize>)
                .transpose()?
                .unwrap_or(default);
            ensure!(
                declared == expected,
                "EXLR8 geometry {key}: metadata {declared} != config {expected}"
            );
        }
        ensure!(
            header.value["signs.in"]["shape"] == serde_json::json!([hidden])
                && header.value["signs.out"]["shape"] == serde_json::json!([hidden]),
            "EXLR8 hidden signs mismatch"
        );
        let parse_widths = |name: &str| -> Result<BTreeSet<usize>> {
            metadata[name]
                .as_str()
                .context("missing width list")?
                .split(',')
                .map(|k| k.parse::<usize>().context("invalid width"))
                .collect()
        };
        let widths = parse_widths("widths")?;
        let shared_widths = parse_widths("shared_widths")?;
        for (prefix, es, ks) in [("", experts, &widths), ("shared.", 1, &shared_widths)] {
            for &k in ks {
                ensure!((2..=4).contains(&k), "unsupported EXLR8 K{k}: add decoder");
                let index = format!("{prefix}k{k}.index");
                ensure!(
                    header.value[&index]["shape"] == serde_json::json!([es, 3, units, 2]),
                    "EXLR8 index geometry mismatch {index}"
                );
                for (projection, channel, block) in [
                    ("gate", "svh", "su_block"),
                    ("up", "svh", "su_block"),
                    ("down", "suh", "sv_block"),
                ] {
                    ensure!(
                        header.value[format!("{prefix}k{k}.{projection}.{channel}")]["shape"]
                            == serde_json::json!([es, units * 128]),
                        "EXLR8 channel geometry mismatch"
                    );
                    ensure!(
                        header.value[format!("{prefix}k{k}.{projection}.{block}")]["shape"]
                            == serde_json::json!([es, hidden / 128]),
                        "EXLR8 block geometry mismatch"
                    );
                }
                let raw = header.read(&path, &index)?;
                ensure!(
                    raw.len() == es * 3 * units * 2 * 8,
                    "EXLR8 index length mismatch"
                );
                let weight_path =
                    snapshot.join(format!("experts/exlr8-layer-{layer:03}.k{k}.safetensors"));
                let weight = Header::open(&weight_path)?;
                weight.validate()?;
                let bytes = (hidden * 16 * k) as u64;
                let stride = if weight.value["__metadata__"]["pad_halves"] == "true" {
                    bytes.div_ceil(65536) * 65536
                } else {
                    bytes
                };
                for e in 0..es {
                    for p in 0..3 {
                        for u in 0..units {
                            let start = ((e * 3 + p) * units + u) * 16;
                            let offset =
                                u64::from_le_bytes(raw[start..start + 8].try_into().unwrap());
                            let length =
                                u64::from_le_bytes(raw[start + 8..start + 16].try_into().unwrap());
                            let stem = ["gate", "up", "down"][p];
                            let (base, extent) =
                                weight.extent(&format!("{prefix}trellis.{stem}"))?;
                            ensure!(
                                offset == base + (e * units + u) as u64 * stride
                                    && length == bytes
                                    && offset + length <= base + extent,
                                "EXLR8 index range mismatch layer {layer} K{k} e{e} u{u} {stem}"
                            );
                        }
                    }
                }
            }
        }
        layers.push(Layer {
            id: layer,
            widths,
            shared_widths,
        });
    }
    Ok(layers)
}

fn dense_source_members<'a>(
    group: &BTreeMap<String, &'a super::checkpoint::CheckpointTensor>,
    suffix: &str,
) -> Result<BTreeMap<String, &'a super::checkpoint::CheckpointTensor>> {
    ensure!(
        group.contains_key(suffix),
        "missing source tensor suffix {suffix}"
    );
    // Bias vectors have their own inventory rows; do not count them with weights
    // or accidentally decode a sibling weight when replacing the bias row.
    Ok(group
        .iter()
        .filter(|(key, _)| {
            if suffix == "bias" {
                key.as_str() == "bias"
            } else {
                key.as_str() != "bias"
            }
        })
        .map(|(key, tensor)| (key.clone(), *tensor))
        .collect())
}

pub fn inspect(
    snapshot: &Path,
    selected: &str,
    dense_from: Option<&Path>,
    sparks: usize,
    rtx: usize,
    spark_budget: u64,
    coordinator_budget: u64,
) -> Result<Exlr8Plan> {
    ensure!(
        (1..=2).contains(&rtx) && sparks <= 7,
        "EXLR8 layout needs 1-2 RTX and 0-7 Sparks"
    );
    let config = read_json(&snapshot.join("config.json"))?;
    ensure!(
        config["architectures"] == serde_json::json!(["GlmMoeDsaForCausalLM"]),
        "unsupported EXLR8 architecture {}",
        config["architectures"]
    );
    let hidden = number(&config, "hidden_size")?;
    let intermediate = number(&config, "moe_intermediate_size")?;
    let experts = number(&config, "n_routed_experts")?;
    let count = number(&config, "num_hidden_layers")?;
    let first = number(&config, "first_k_dense_replace")?;
    ensure!(
        hidden > 0
            && hidden % 128 == 0
            && intermediate > 0
            && intermediate % 128 == 0
            && experts > 0
            && count > first,
        "invalid EXLR8 geometry"
    );
    let layers = geometry(
        snapshot,
        &config,
        hidden,
        intermediate / 128,
        experts,
        count,
        first,
    )?;
    let mut dense = dense_operands(snapshot, count)?;
    let dense_source = if let Some(source) = dense_from {
        let original = super::Checkpoint::open(source)?;
        // Compare every architectural config field; only identity/quantization and
        // serialization metadata may differ. Equal architectures alone is insufficient.
        for (key, value) in config.as_object().context("config must be an object")? {
            if key.starts_with('_')
                || [
                    "quantization_config",
                    "torch_dtype",
                    "dtype",
                    "transformers_version",
                    "base_model",
                    "base_model_name_or_path",
                ]
                .contains(&key.as_str())
            {
                continue;
            }
            ensure!(
                original.config.get(key) == Some(value),
                "dense source architecture mismatch at {key}"
            );
        }
        let mut replacements = Vec::new();
        let groups = super::format::group_by_stem(&original.tensors);
        for item in &dense {
            let (stem, suffix) = super::format::split_stem(&item.tensor);
            let group = groups
                .get(stem)
                .with_context(|| format!("dense source missing {stem}"))?;
            let members = dense_source_members(group, suffix)
                .with_context(|| format!("dense source missing {}", item.tensor))?;
            let operand = super::format::detect(&members)?;
            ensure!(
                operand.logical == item.shape,
                "dense source shape mismatch for {stem}"
            );
            replacements.push(DenseOperand {
                tensor: item.tensor.clone(),
                shard: members.values().next().unwrap().shard.clone(),
                encoding: operand.encoding,
                scale: operand.scale,
                shape: operand.logical,
                bytes: members.values().map(|m| m.meta.byte_length).sum(),
                sharded_side: item.sharded_side,
                channels_per_shard: item.channels_per_shard,
                rotation: None,
                kernel: None,
            });
        }
        dense = replacements;
        "official checkpoint config/headers (architecture matched)".to_owned()
    } else {
        "checkpoint native EXLR8 records (no BF16 conversion)".to_owned()
    };
    let mut names = BTreeSet::from([selected.to_string(), "bit325".into(), "up1".into()]);
    let mut manifests = Vec::new();
    for name in std::mem::take(&mut names) {
        manifests.push(manifest(
            snapshot,
            &name,
            hidden,
            intermediate / 128,
            experts,
            &layers,
        )?);
    }
    ensure!(
        manifests.iter().all(|m| m.set_id == manifests[0].set_id),
        "manifest weight set_id mismatch"
    );
    let dense_bytes = dense.iter().map(|d| d.bytes).sum::<u64>();
    let mut fit = Vec::new();
    for manifest in &manifests {
        for rtx in 1..=2 {
            for sparks in 0..=7 {
                let coordinator_bytes = dense_bytes + manifest.shared_bytes;
                let rank_bytes = if sparks == 0 {
                    vec![]
                } else {
                    manifest.ranks[&sparks].iter().map(|r| r.total).collect()
                };
                let local = if sparks == 0 {
                    manifest.ranks[&rtx].iter().map(|r| r.total).max().unwrap()
                } else {
                    0
                };
                let per_gpu = dense_bytes + manifest.shared_bytes.div_ceil(rtx as u64) + local;
                fit.push(Fit {
                    manifest: manifest.name.clone(),
                    rtx,
                    sparks,
                    rank_bytes: rank_bytes.clone(),
                    coordinator_bytes,
                    coordinator_per_gpu_upper_bound: per_gpu,
                    weights_fit: per_gpu <= coordinator_budget
                        && rank_bytes.iter().all(|b| *b <= spark_budget),
                });
            }
        }
    }
    let selected_weights_fit = fit
        .iter()
        .find(|f| f.manifest == selected && f.rtx == rtx && f.sparks == sparks)
        .unwrap()
        .weights_fit;
    let mut missing = dense
        .iter()
        .filter_map(|d| d.kernel.clone())
        .collect::<BTreeSet<_>>();
    for manifest in &manifests {
        for k in manifest.widths.keys() {
            missing.insert(format!("exlr8_unit_experts_k{k}_a16_sm121_and_sm120"));
        }
    }
    missing.extend(["exlr8_rotate_once_sm120", "exlr8_rotated_reduce_sm120"].map(str::to_string));
    Ok(Exlr8Plan { format:FORMAT.into(), scheme:SCHEME.into(), selected_manifest:selected.into(),
        manifest_choice:"bit325 is the 3.25-bpw evaluation choice: checkpoint held-out KL 0.147 vs home 0.149; golden pending. up1 is the 4.0-bpw arm.".into(),
        dense_source, dense, dense_bytes, manifests, fit, selected_weights_fit,
        missing_kernels:missing.into_iter().collect(), supported:false,
        notes:vec!["Layer 78 MTP is unused (GLM DFlash2); shared experts remain on coordinator.".into(),
            "Rank bytes = selected unit payloads + selected channel vectors + deduplicated projection/width block scales + signs. Index/checksum/error-table staging is not resident weight storage.".into(),
            "Fit is weight-only, not serving admission: no kernels, descriptors, KV, workspaces, graphs or drafter charged. Two-RTX coordinator bound replicates dense and splits shared experts.".into(),
            "TP7 is an inventory-only format capability; current expert transport supports at most six Spark ranks.".into()] })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;

    fn write(path: &Path, metadata: Value, tensors: Vec<(&str, &str, Vec<usize>, Vec<u8>)>) {
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".into(), metadata);
        let mut offset = 0;
        for (name, dtype, shape, bytes) in &tensors {
            header.insert(
                (*name).into(),
                json!({"dtype":dtype,"shape":shape,"data_offsets":[offset,offset+bytes.len()]}),
            );
            offset += bytes.len();
        }
        let data = serde_json::to_vec(&header).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(&(data.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&data).unwrap();
        for (_, _, _, bytes) in tensors {
            file.write_all(&bytes).unwrap();
        }
    }
    fn metadata() -> Value {
        json!({"format":FORMAT,"scheme":SCHEME})
    }

    #[test]
    fn dense_record_encodings_and_rotations_are_native() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        std::fs::create_dir(root.join("dense")).unwrap();
        let mut metadata = metadata();
        metadata["layer"] = json!("0");
        let record = [1i64, 128, 128, 4, 1, 128, 2, 1]
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect();
        write(
            &root.join("dense/exlr8-dense-000.safetensors"),
            metadata,
            vec![
                ("self_attn.q_b_proj.record", "I64", vec![8], record),
                (
                    "self_attn.q_b_proj.trellis",
                    "U8",
                    vec![1, 8192],
                    vec![0; 8192],
                ),
                ("self_attn.q_b_proj.suh", "F16", vec![128], vec![0; 256]),
                ("self_attn.q_b_proj.svh", "F16", vec![128], vec![0; 256]),
            ],
        );
        let operands = dense_operands(root, 1).unwrap();
        assert_eq!(operands.len(), 1);
        let operand = &operands[0];
        assert_eq!(operand.tensor, "model.layers.0.self_attn.q_b_proj.weight");
        assert_eq!(
            operand.encoding,
            Encoding::Trellis {
                k: 4,
                codebook: TrellisCodebook::E4M3,
                rotation: 2
            }
        );
        assert_eq!(operand.rotation.as_deref(), Some("q_resid"));
        assert_eq!(operand.bytes, 8192 + 512);
        assert!(operand
            .kernel
            .as_deref()
            .unwrap()
            .contains("trellis_linear"));
    }

    fn fixture_manifest(root: &Path, name: &str, gate: Value) {
        std::fs::create_dir_all(root.join("manifests")).unwrap();
        std::fs::write(
            root.join(format!("manifests/exlr8-manifest-{name}.json")),
            json!({
                "format":"exlr8-v2-manifest","name":name,"version":"test","set_id":"a".repeat(64),
                "layers":{"3":{"gate":gate,"up":[[3,3,3,3]],"down":[4],
                    "shared":{"gate":[3,3,3,3],"up":[3,3,3,3],"down":4}}}
            })
            .to_string(),
        )
        .unwrap();
    }
    fn layer() -> Layer {
        Layer {
            id: 3,
            widths: BTreeSet::from([2, 3, 4]),
            shared_widths: BTreeSet::from([3, 4]),
        }
    }

    #[test]
    fn manifest_mixed_units_count_bytes_and_deduplicate_blocks() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        fixture_manifest(root, "mixed", json!([[2, 3, 4, 2]]));
        let plan = manifest(root, "mixed", 128, 4, 1, &[layer()]).unwrap();
        assert_eq!(plan.routed_bpw, 39.0 / 12.0);
        for n in 1..=7 {
            assert_eq!(
                plan.ranks[&n].iter().map(|r| r.trellis).sum::<u64>(),
                128 * 16 * 39
            );
            assert_eq!(
                plan.ranks[&n].iter().map(|r| r.channels).sum::<u64>(),
                12 * 128 * 2
            );
            assert_eq!(
                plan.ranks[&n].iter().map(|r| r.signs).sum::<u64>(),
                n as u64 * 128 * 2
            );
        }
        // One rank: gate's three tiers + up's K3 + down's K4, each one block scale.
        assert_eq!(plan.ranks[&1][0].blocks, 5 * 2);
        assert_eq!(plan.ranks[&1][0].total, 128 * 16 * 39 + 12 * 256 + 10 + 256);
    }

    #[test]
    fn manifest_rejects_wrong_shapes_absent_widths_and_path_traversal() {
        let tmp = tempfile::tempdir().unwrap();
        fixture_manifest(tmp.path(), "bad", json!([[2, 3, 4]]));
        assert!(manifest(tmp.path(), "bad", 128, 4, 1, &[layer()])
            .unwrap_err()
            .to_string()
            .contains("unit count"));
        fixture_manifest(tmp.path(), "bad", json!([[2, 3, 4, 5]]));
        assert!(manifest(tmp.path(), "bad", 128, 4, 1, &[layer()])
            .unwrap_err()
            .to_string()
            .contains("absent K5"));
        assert!(manifest(tmp.path(), "../bad", 128, 4, 1, &[layer()])
            .unwrap_err()
            .to_string()
            .contains("published name"));
    }

    #[test]
    fn unknown_scheme_has_precise_refusal() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::create_dir(temp.path().join("experts")).unwrap();
        let path = temp
            .path()
            .join("experts/exlr8-layer-003.index.safetensors");
        let mut meta = metadata();
        meta["scheme"] = json!("future-scheme");
        write(&path, meta, vec![]);
        assert!(recognizes(temp.path()).unwrap());
        let error = Header::open(&path)
            .unwrap()
            .validate()
            .unwrap_err()
            .to_string();
        assert!(error.contains("future-scheme") && error.contains("decoder"));
    }

    #[test]
    fn official_dense_weight_and_bias_are_independent_native_operands() {
        use super::super::testing::{t, write_snapshot};
        let temp = tempfile::tempdir().unwrap();
        write_snapshot(
            temp.path(),
            &json!({}),
            &[
                t("norm.weight", "BF16", &[128]),
                t("norm.bias", "F32", &[128]),
                t("proj.weight", "F8_E4M3", &[128, 128]),
                t("proj.weight_scale_inv", "F32", &[1, 1]),
                t("proj.bias", "BF16", &[128]),
            ],
            None,
        );
        let source = super::super::Checkpoint::open(temp.path()).unwrap();
        let groups = super::super::format::group_by_stem(&source.tensors);
        for (name, encoding, bytes) in [
            ("norm.weight", Encoding::Bf16, 256),
            ("norm.bias", Encoding::F32, 512),
            ("proj.weight", Encoding::E4m3, 16388),
            ("proj.bias", Encoding::Bf16, 256),
        ] {
            let (stem, suffix) = super::super::format::split_stem(name);
            let members = dense_source_members(&groups[stem], suffix).unwrap();
            let operand = super::super::format::detect(&members).unwrap();
            assert_eq!(operand.encoding, encoding);
            assert_eq!(
                members.values().map(|m| m.meta.byte_length).sum::<u64>(),
                bytes
            );
            assert!(operand.companions.is_empty() || suffix == "bias");
        }
        assert!(dense_source_members(&groups["norm"], "scale").is_err());
    }

    #[test]
    fn ownership_rotates_every_layer_and_balances_all_units() {
        for n in 3..=7 {
            for layer in [3, 40, 77] {
                let mut counts = vec![0usize; n];
                for e in 0..257 {
                    for u in 0..16 {
                        counts[owner(16, e, u, layer, n)] += 1;
                    }
                }
                assert_eq!(counts.iter().sum::<usize>(), 257 * 16);
                assert!(counts.iter().max().unwrap() - counts.iter().min().unwrap() <= 1);
            }
        }
        assert_eq!(
            (0..16)
                .filter(|u| owner(16, 256, *u, 40, 6) == 1)
                .collect::<Vec<_>>(),
            vec![5, 11]
        );
    }
}

#[cfg(test)]
mod mounted_tests {
    #[test]
    fn inspect_checkpoint_when_explicitly_supplied() {
        let Some(snapshot) = std::env::var_os("CUTEAFD_EXLR8_TEST_SNAPSHOT") else {
            return;
        };
        let root = std::path::Path::new(&snapshot);
        assert!(super::recognizes(root).unwrap());
        let report = super::inspect(root, "bit325", None, 6, 2, 100 << 30, 80 << 30).unwrap();
        assert!(!report.supported);
        assert!(report.selected_weights_fit);
        assert_eq!(report.fit.len(), 32);
        assert!(report.dense.iter().any(|d| d.tensor == "lm_head.weight"));
        assert!(report
            .dense
            .iter()
            .any(|d| matches!(d.encoding, super::Encoding::Int4 { group: 32 })));
        for manifest in &report.manifests {
            let total = manifest.ranks[&1][0].trellis;
            for n in 3..=7 {
                assert_eq!(
                    manifest.ranks[&n].iter().map(|r| r.trellis).sum::<u64>(),
                    total
                );
            }
        }
        if let Some(out) = std::env::var_os("CUTEAFD_EXLR8_TEST_REPORT") {
            std::fs::write(&out, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
            std::fs::write(
                std::path::Path::new(&out).with_extension("txt"),
                report.render(),
            )
            .unwrap();
        }
        if let Some(dense) = std::env::var_os("CUTEAFD_EXLR8_TEST_DENSE") {
            let official = super::inspect(
                root,
                "up1",
                Some(std::path::Path::new(&dense)),
                4,
                1,
                100 << 30,
                80 << 30,
            )
            .unwrap();
            assert!(official.dense_source.contains("official"));
            assert!(official.dense_bytes > report.dense_bytes);
        }
    }
}
