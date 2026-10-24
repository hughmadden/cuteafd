//! Teacher-forced qualification through the serving programs and paged cache.
use super::{cache::{Cache, INDEX_PAGE_BYTES, KV_PAGE_BYTES, PAGE_ROWS}, execution::{Execution, ExpertExecutor},
    native::Native, prefix::{MiniMaxPrefix, Placement}, transaction::Step, MiniMaxError};
use crate::shared::{golden_scores::{scored_kl, scored_nll, HiddenMetrics}, memory::device::Allocation, prefix::view,
    token_io::{EmbedPlacement, TokenEmbedding}};
use anyhow::{ensure, Context, Result};
use cuteafd_core::ExpertGeometry;
use cuteafd_engine::{media::RequestMedia, prefix::{After, PrefixCache, PrefixConfig, PrefixFamily, RefPagePool, SnapshotKind}};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_hostcache::copy::StubCopyEngine;
use cuteafd_loader::{families::{minimax_m3::config::MiniMaxM3Config, qwen4::Qwen4KvCache},
    plan::{checkpoint::{read_json, Checkpoint}, families::minimax::{admission, coordinator::RankWeightLayout}}};
use cuteafd_transport::ExpertV2SourceKind;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{io::{Read, Write}, net::SocketAddr, path::{Component, Path, PathBuf}};

const REVISION: &str = "54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e";

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ScoreShape { Decode, Prefill }

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct GoldenArgs {
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    native_lib: Option<PathBuf>,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")]
    manifest: PathBuf,
    #[arg(long)]
    golden: Option<PathBuf>,
    /// Omit to score every window in the sealed fidelity set.
    #[arg(long)]
    window: Option<String>,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, value_delimiter = ',')]
    peers: Vec<SocketAddr>,
    /// Native RTX-local routed layers; all other executed routed layers need peers.
    #[arg(long, value_delimiter = ',')]
    expert_layers: Vec<usize>,
    /// Partial execution compares layerNN.bin; only layer 59 scores logits.
    #[arg(long, default_value_t = 59)]
    last_layer: usize,
    #[arg(long, default_value_t = 128, value_parser = clap::value_parser!(u32).range(1..=4096))]
    prefill_rows: u32,
    #[arg(long, value_enum, default_value = "decode")]
    score_shape: ScoreShape,
    #[arg(long, alias = "out")]
    save_logits: Option<PathBuf>,
    /// Reuse hash-checked completed windows; replay interrupted windows cold.
    #[arg(long, requires = "save_logits")]
    resume: bool,
    #[arg(long)]
    check_prefix: bool,
    /// Read and validate checkpoint headers without loading a native library.
    #[arg(long)]
    check_headers: bool,
}

fn selection(args: &GoldenArgs) -> Result<Vec<usize>> {
    ensure!(args.last_layer < 60 && args.device >= 0, MiniMaxError::Invalid("golden layer or device"));
    ensure!(matches!(args.peers.len(), 0 | 2 | 3 | 4)
        && args.peers.iter().enumerate().all(|(i, p)| !args.peers[..i].contains(p)),
        MiniMaxError::Invalid("golden peers require distinct TP2/TP3/TP4 endpoints"));
    ensure!(args.expert_layers.iter().all(|&l| (3..=args.last_layer).contains(&l))
        && args.expert_layers.windows(2).all(|w| w[0] < w[1]),
        MiniMaxError::Invalid("local expert layers must be increasing executed routed layers"));
    let remote = (3..=args.last_layer).filter(|l| !args.expert_layers.contains(l)).collect::<Vec<_>>();
    ensure!(!args.peers.is_empty() || remote.is_empty(),
        MiniMaxError::Invalid("without peers --expert-layers must cover every executed routed layer"));
    ensure!(args.peers.is_empty() || !remote.is_empty(), MiniMaxError::Invalid("unused golden peers"));
    Ok(remote)
}

fn hash(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut bytes = [0; 65536];
    loop { let n = file.read(&mut bytes)?; if n == 0 { break; } digest.update(&bytes[..n]); }
    Ok(format!("{:x}", digest.finalize()))
}

fn safe_path(root: &Path, name: &str) -> Result<PathBuf> {
    let relative = Path::new(name);
    ensure!(!name.is_empty() && relative.components().all(|p| matches!(p, Component::Normal(_))),
        "reference path must stay inside its directory");
    let path = root.join(relative);
    ensure!(path.canonicalize()?.starts_with(root.canonicalize()?), "reference symlink escapes directory");
    Ok(path)
}

fn floats(bytes: &[u8], expected: usize) -> Result<Vec<f32>> {
    ensure!(expected.checked_mul(4) == Some(bytes.len()), "reference F32 extent mismatch");
    let values = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
    ensure!(values.iter().all(|v| v.is_finite()), "nonfinite reference or native values");
    Ok(values)
}

struct Window { id: String, directory: PathBuf, tokens: Vec<u32>, positions: Vec<usize>, generated: Vec<bool> }

fn references(args: &GoldenArgs, cfg: &MiniMaxM3Config) -> Result<(Value, Vec<Window>)> {
    let root = args.golden.as_ref().context("MiniMax golden requires --golden")?;
    // A partial-layer run reads the layerNN.bin a full reference exported with --layers.
    let meta = read_json(&root.join("meta.json"))?;
    ensure!(meta["schema"] == "cuteafd.fidelity.golden/2", "MiniMax reference schema differs");
    ensure!(meta["family"] == "minimax_m3"
        && meta["checkpoint"] == "MiniMaxAI/MiniMax-M3.1-Flash" && meta["mtp"] == false
        && meta["snapshot_identity"]["snapshot_revision"] == REVISION,
        "MiniMax golden requires the pinned independent target reference");
    ensure!(matches!(meta["activation"].as_str(), Some("served" | "served-cache" | "bf16")),
        "MiniMax reference activation arm missing");
    ensure!(meta["source_pin"]["transformers"]["revision"] == "6b6792ff3b1a434913dc59659d4461acc9c3a51a"
        && meta["source_pin"]["transformers"]["sha256"]["src/transformers/models/minimax_m3_vl/modular_minimax_m3_vl.py"]
            == "1c755515c8aefdbc6f9145ae8f051e43907e21022d9644e81afe47cd2e37b58d",
        "MiniMax reference source pin differs");
    {
        let proof = &meta["prefix_qualification"];
        ensure!(proof["passed"] == true && proof["finite"] == true
            && proof["set_sha256"] == meta["set_sha256"]
            && proof["snapshot_identity"] == meta["snapshot_identity"], "MiniMax reference lacks matching prefix qualification");
    }
    let panel = read_json(&root.join("windows.json"))?;
    ensure!(panel["set_sha256"] == meta["set_sha256"], "reference panel identity differs");
    for (file, key) in [("config.json", "config_sha256"), ("tokenizer.json", "tokenizer_sha256")] {
        ensure!(meta["snapshot_identity"][key].as_str() == Some(hash(&args.snapshot.join(file))?.as_str()),
            "reference {key} differs from checkpoint");
    }
    let mut windows = Vec::new();
    for entry in meta["windows"].as_array().context("reference windows missing")? {
        let id = entry["id"].as_str().context("reference window id missing")?;
        ensure!(!id.is_empty() && id.bytes().all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b)), "invalid window id");
        if args.window.as_ref().is_some_and(|wanted| wanted != id) { continue; }
        ensure!(entry["vocab"].as_u64() == Some(cfg.vocab_size as u64), "reference vocabulary differs");
        let directory = safe_path(root, entry["path"].as_str().context("reference window path missing")?)?;
        let bytes = std::fs::read(directory.join("tokens.bin"))?;
        ensure!(bytes.len() >= 8 && bytes.len() % 4 == 0, "invalid reference tokens");
        let tokens = bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        ensure!(tokens.len() <= cfg.max_position && tokens.iter().all(|&t| (t as usize) < cfg.vocab_size),
            "reference tokens outside checkpoint extent");
        let source = panel["windows"].as_array().context("panel windows missing")?.iter()
            .find(|w| w["id"] == id).context("reference window absent from panel")?;
        ensure!(source["tokens"] == json!(tokens) && source["media"].as_array().is_none_or(Vec::is_empty),
            "reference token identity differs or window contains media");
        let roles = source["roles"].as_array().context("reference role mask missing")?;
        ensure!(roles.len() == tokens.len() && roles.iter().all(|v| matches!(v.as_str(), Some("gen" | "ctx"))),
            "invalid reference role mask");
        let generated = roles.iter().map(|v| v == "gen").collect();
        let positions = entry["positions"].as_array().context("reference positions missing")?.iter()
            .map(|p| usize::try_from(p.as_u64().context("invalid reference position")?).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;
        ensure!(!positions.is_empty() && positions.iter().all(|&p| p > 0 && p < tokens.len())
            && positions.windows(2).all(|p| p[0] < p[1]), "invalid reference score positions");
        let (file, count) = if args.last_layer == 59 { ("logits.bin".into(), positions.len() * cfg.vocab_size) }
            else { (format!("layer{:02}.bin", args.last_layer), tokens.len() * cfg.hidden) };
        ensure!(std::fs::metadata(directory.join(&file))?.len() == count as u64 * 4, "reference payload extent differs");
        ensure!(!windows.iter().any(|w: &Window| w.id == id), "duplicate reference window");
        windows.push(Window { id: id.into(), directory, tokens, positions, generated });
    }
    ensure!(!windows.is_empty(), "no selected reference window");
    Ok((meta, windows))
}

#[derive(Default)]
struct Scores { rows: usize, agree: usize, nll: f64, reference_nll: f64, kl: f64, max_kl: f64 }
impl Scores {
    fn add(&mut self, ours: &[f32], reference: &[f32], target: u32) -> Result<()> {
        let vocab = ours.len();
        self.nll += scored_nll(ours, &[target], vocab)?;
        self.reference_nll += scored_nll(reference, &[target], vocab)?;
        let (kl, _) = scored_kl(reference, ours, vocab)?;
        self.kl += kl; self.max_kl = self.max_kl.max(kl);
        let argmax = |r: &[f32]| r.iter().enumerate().max_by(|a, b|
            a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0))).unwrap().0;
        self.agree += usize::from(argmax(ours) == argmax(reference)); self.rows += 1;
        Ok(())
    }
    fn result(&self) -> Result<Value> {
        ensure!(self.rows > 0, "no native logits scored");
        let count = self.rows as f64;
        Ok(json!({"scored_tokens":self.rows,"top1":self.agree as f64/count,
            "nll":self.nll/count,"reference_nll":self.reference_nll/count,
            "mean_kl":self.kl/count,"max_kl":self.max_kl}))
    }
}

/// Full-model gates over the generated-role rows, from the first real-weight run (standard set v1,
/// 13,534 rows, TP2; decode- and prefill-shaped scoring: top-1 0.955/0.957, KL 0.0186, NLL 0.2519
/// against the served-cache reference's 0.2536), with margin for kernel route and chunk changes.
const TOP1_MIN: f64 = 0.94;
const KL_MAX: f64 = 0.03;
const NLL_EXCESS_MAX: f64 = 0.01;

fn verdict(result: &Value, full: bool) -> Result<()> {
    let number = |key| result[key].as_f64().filter(|v| v.is_finite()).with_context(|| format!("missing finite {key}"));
    if full { ensure!(number("top1")? >= TOP1_MIN && number("mean_kl")? <= KL_MAX
            && number("nll")? - number("reference_nll")? <= NLL_EXCESS_MAX,
        "MiniMax fidelity failed: top-1 >= {TOP1_MIN}, KL <= {KL_MAX} nat and NLL at most {NLL_EXCESS_MAX} nat \
         above the reference required"); }
    else { ensure!(number("cosine")? >= 0.99 && number("relative_l2")? <= 0.05,
        "MiniMax partial hidden failed: cosine >= 0.99 and relative L2 <= 0.05 required"); }
    Ok(())
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    let temporary = path.with_extension("json.partial");
    let mut file = std::fs::File::create(&temporary)?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?; file.write_all(b"\n")?; file.sync_all()?;
    std::fs::rename(temporary, path)?; Ok(())
}

fn seal_window(directory: &Path, filename: &str, result: &mut Value) -> Result<()> {
    result["sha256"] = json!({(filename):hash(&directory.join(filename))?,
        "tokens.bin":hash(&directory.join("tokens.bin"))?,"positions.bin":hash(&directory.join("positions.bin"))?});
    result["complete"] = true.into();
    write_json(&directory.join("result.json"), result)
}

fn binding(args: &GoldenArgs, reference: &Value, windows: &[Window], checkpoint: &Checkpoint) -> Result<Value> {
    let native = args.native_lib.as_ref().context("MiniMax golden requires --native-lib")?;
    let payload = if args.last_layer == 59 {"logits.bin".into()} else {format!("layer{:02}.bin", args.last_layer)};
    let references = windows.iter().map(|w| Ok((w.id.clone(), json!({
        "tokens":hash(&w.directory.join("tokens.bin"))?,"values":hash(&w.directory.join(&payload))?}))))
        .collect::<Result<serde_json::Map<_, _>>>()?;
    let shards = checkpoint.tensors.iter().map(|t| t.shard.clone()).collect::<std::collections::BTreeSet<_>>()
        .into_iter().map(|name| {
            let metadata = std::fs::metadata(args.snapshot.join(&name))?;
            Ok((name, json!({"bytes":metadata.len(),
                "modified_ns":metadata.modified()?.duration_since(std::time::UNIX_EPOCH)?.as_nanos().to_string()})))
        }).collect::<Result<serde_json::Map<_, _>>>()?;
    Ok(json!({"schema":"cuteafd.minimax.golden.binding/1", "reference":reference,
        "reference_files":references,"shards":shards,
        "native_sha256":hash(native)?, "programs_sha256":hash(&args.manifest)?,
        "binary_sha256":hash(&std::env::current_exe()?)?, "config_sha256":hash(&args.snapshot.join("config.json"))?,
        "index_sha256":hash(&args.snapshot.join("model.safetensors.index.json"))?,
        "last_layer":args.last_layer,"local_layers":args.expert_layers,"peer_count":args.peers.len(),
        "prefill_rows":args.prefill_rows,"score_shape":args.score_shape,"check_prefix":args.check_prefix}))
}

fn completed(directory: &Path, binding: &Value, full: bool) -> Result<Option<Value>> {
    let path = directory.join("result.json");
    if !path.is_file() { return Ok(None); }
    let result = read_json(&path)?;
    ensure!(result["binding"] == *binding && result["complete"] == true, "resume window identity differs");
    let hashes = result["sha256"].as_object().context("resume output hashes missing")?;
    let payload = if full {"logits.bin".into()} else {format!("layer{:02}.bin", binding["last_layer"].as_u64().context("resume layer missing")?)};
    ensure!(hashes.len() == 3 && hashes.contains_key(&payload) && hashes.contains_key("tokens.bin")
        && hashes.contains_key("positions.bin") && result["prefix_byte_exact"] == binding["check_prefix"],
        "resume receipt has incomplete output or prefix evidence");
    for (name, expected) in hashes {
        ensure!(expected.as_str() == Some(hash(&safe_path(directory, name)?)?.as_str()), "resume output hash differs: {name}");
    }
    if !full {verdict(&result, false)?;}
    Ok(Some(result))
}

struct DenseOnly;
impl<'a> ExpertExecutor<'a> for DenseOnly {
    fn dispatch(&mut self, _: usize, _: usize, _: usize, _: CuteafdDeviceBuffer, _: CuteafdDeviceBuffer, _: CuteafdDeviceBuffer,
        _: &Native<'a>) -> Result<()> {
        Err(MiniMaxError::Invalid("unexpected routed layer in dense-only golden").into())
    }
    fn collect(&mut self, _: usize, _: usize, _: usize, _: CuteafdDeviceBuffer, _: &Native<'a>) -> Result<()> {
        Err(MiniMaxError::Invalid("unexpected routed collect in dense-only golden").into())
    }
}

struct Runner<'e, 'a> {
    execution: &'e Execution<'a>, cache: &'e Cache<'a>, embedding: &'e TokenEmbedding<'a>,
    ids: Allocation<'a>, embedded: Allocation<'a>, experts: Box<dyn ExpertExecutor<'a> + 'a>,
    last_layer: usize, rows: usize,
}
impl Runner<'_, '_> {
    /// One step of `tokens` from `placement`'s frontier, through the serving
    /// path's wave checks against `pool`.
    fn evaluate(&mut self, pool: &RefPagePool, placement: &mut Placement, tokens: &[u32], prefill: bool) -> Result<Vec<f32>> {
        let batch = Step::begin(&[(&mut *placement, tokens)], pool, self.cache.sequences, self.rows,
            self.cache.context, self.cache.pages)?.batch(self.cache.pages);
        let embedded = self.embedding.embed(tokens, self.ids.buffer, 1, self.embedded.buffer, self.execution.stream());
        let drained = self.execution.drain();
        embedded?; drained?;
        let output = self.execution.execute_through(self.embedded.buffer, &batch, &self.cache.layers,
            &mut *self.experts, if prefill {ExpertV2SourceKind::Prefill} else {ExpertV2SourceKind::Decode}, self.last_layer,
            0..tokens.len())?;
        let values = if let Some(logits) = output { logits.to_host(self.cache.library)? } else {
            let hidden = self.execution.hidden(tokens.len())?;
            let mut bytes = vec![0; tokens.len() * 6144 * 4];
            self.cache.library.copy_d2h(&mut bytes, hidden)?;
            floats(&bytes, tokens.len() * 6144)?
        };
        ensure!(values.iter().all(|v| v.is_finite()), "nonfinite native golden output");
        placement.committed += tokens.len();
        Ok(values)
    }
    fn prefill(&mut self, pool: &RefPagePool, placement: &mut Placement, tokens: &[u32]) -> Result<()> {
        for chunk in tokens.chunks(self.rows) { self.evaluate(pool, placement, chunk, true)?; }
        Ok(())
    }
}

fn placement(pages: Vec<u32>) -> Placement {
    Placement {pages, sequence:0, committed:0, media:RequestMedia::new(Vec::new(), 6144, 0).expect("fixed text geometry")}
}

fn state(cache: &Cache<'_>, p: &Placement, layers: usize) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    for layer in cache.layers.iter().take(layers) {
        for (logical, &page) in p.pages.iter().take(p.committed.div_ceil(64)).enumerate() {
            let rows = (p.committed - logical * 64).min(64);
            for (plane, stride, offset, bytes) in [(&layer.key, KV_PAGE_BYTES, 0, rows * 512),
                (&layer.value, KV_PAGE_BYTES, 0, rows * 512), (&layer.index, INDEX_PAGE_BYTES, 0, rows * 128),
                (&layer.index, INDEX_PAGE_BYTES, 8192, rows * 4)] {
                let mut values = vec![0; bytes];
                cache.library.copy_d2h(&mut values, view(plane.buffer, page as usize * stride + offset, bytes)?)?;
                output.extend(values);
            }
        }
    }
    Ok(output)
}

fn check_prefix(runner: &mut Runner<'_, '_>, tokens: &[u32]) -> Result<()> {
    ensure!(tokens.len() >= 66, "prefix qualification requires at least 66 input tokens");
    let family = MiniMaxPrefix { cache:runner.cache };
    let mut pool = PrefixCache::<StubCopyEngine>::new(family.layout(), PrefixConfig {
        entries:4, mark_slots:0, keep_logits:false, min_tokens:1 }, None)?;
    for frontier in [63, 64, 65] {
        pool.clear(&family)?;
        let mut source = pool.admit_cold(&family, frontier, frontier + 1, placement)?.placement;
        runner.prefill(pool.pool(), &mut source, &tokens[..frontier])?;
        let expected = state(runner.cache, &source, runner.last_layer + 1)?;
        ensure!(pool.capture(&family, SnapshotKind::Prompt, &tokens[..frontier], &source, After::default())?,
            "prefix prompt capture failed");
        let admitted = pool.admit(&family, &tokens[..frontier + 1], frontier + 1, false, placement)?;
        ensure!(admitted.resume == frontier, "prefix restore missed deepest snapshot");
        let mut restored = admitted.placement;
        ensure!(state(runner.cache, &restored, runner.last_layer + 1)? == expected, "restored cache bytes differ");
        let mut cold = pool.admit_cold(&family, frontier, frontier + 1, placement)?.placement;
        runner.prefill(pool.pool(), &mut cold, &tokens[..frontier])?;
        ensure!(state(runner.cache, &cold, runner.last_layer + 1)? == expected, "cold prefix cache bytes differ");
        let a = runner.evaluate(pool.pool(), &mut restored, &tokens[frontier..frontier + 1], false)?;
        let b = runner.evaluate(pool.pool(), &mut cold, &tokens[frontier..frontier + 1], false)?;
        ensure!(a.iter().map(|v| v.to_bits()).eq(b.iter().map(|v| v.to_bits())), "prefix continuation output differs");
        ensure!(state(runner.cache, &restored, runner.last_layer + 1)? == state(runner.cache, &cold, runner.last_layer + 1)?,
            "prefix continuation cache bytes differ");
        ensure!(state(runner.cache, &source, runner.last_layer + 1)? == expected, "prefix append mutated captured pages");
        ensure!(pool.capture(&family, SnapshotKind::Turn, &tokens[..frontier + 1], &restored, After::default())?,
            "prefix turn capture failed");
        let mut next_turn = tokens[..frontier + 1].to_vec(); next_turn.push(tokens[0]);
        let turn = pool.admit(&family, &next_turn, frontier + 2, false, placement)?;
        ensure!(turn.resume == frontier + 1
            && state(runner.cache, &turn.placement, runner.last_layer + 1)? == state(runner.cache, &cold, runner.last_layer + 1)?,
            "turn restore missed the deepest snapshot or changed cache bytes");
        pool.release(&family, &turn.placement.pages)?;
        for p in [&source, &restored, &cold] { pool.release(&family, &p.pages)?; }
    }
    pool.clear(&family)?;
    println!("MiniMax prefix byte-exact restore and continuation: passed (63,64,65)");
    Ok(())
}

fn score_window(runner: &mut Runner<'_, '_>, args: &GoldenArgs, window: &Window,
    directory: &Path, binding: &Value, vocab: usize) -> Result<Value> {
    std::fs::create_dir_all(directory)?;
    let full = args.last_layer == 59;
    let filename = if full {"logits.bin".into()} else {format!("layer{:02}.bin", args.last_layer)};
    let expected_rows = if full {window.positions.len()} else {window.tokens.len()};
    let width = if full {vocab} else {6144};
    let expected = floats(&std::fs::read(window.directory.join(&filename))?, expected_rows * width)?;
    let mut file = std::fs::File::create(directory.join(format!("{filename}.partial")))?;
    // A fresh pool hands out pages 0, 1, ... in order.
    let mut pages = RefPagePool::new(runner.cache.pages, PAGE_ROWS);
    let mut current = placement(pages.alloc(window.tokens.len().div_ceil(PAGE_ROWS))?);
    let mut scores = Scores::default();
    let mut generated = Scores::default();
    let mut hidden = HiddenMetrics::default();
    let mut first = 0;
    while first < window.tokens.len() {
        let unscored = first < window.positions[0] - 1;
        let count = if args.score_shape == ScoreShape::Decode && !unscored {1}
            else {runner.rows.min(if unscored {window.positions[0] - 1 - first} else {window.tokens.len() - first})};
        let count = count.min(window.tokens.len() - first);
        let values = runner.evaluate(&pages, &mut current, &window.tokens[first..first + count], unscored || args.score_shape == ScoreShape::Prefill)?;
        if full {
            for (row, ours) in values.chunks_exact(vocab).enumerate() {
                if let Ok(index) = window.positions.binary_search(&(first + row + 1)) {
                    // Only scored rows are kept, as the reference stores them: row i predicts positions[i].
                    file.write_all(&ours.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
                    scores.add(ours, &expected[index * vocab..(index + 1) * vocab], window.tokens[first + row + 1])?;
                    if window.generated[first + row + 1] {
                        generated.add(ours, &expected[index * vocab..(index + 1) * vocab], window.tokens[first + row + 1])?;
                    }
                }
            }
        } else {
            file.write_all(&values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
            hidden.add(&values, &expected[first * width..(first + count) * width])?;
        }
        first += count;
    }
    file.sync_all()?;
    std::fs::rename(directory.join(format!("{filename}.partial")), directory.join(&filename))?;
    std::fs::write(directory.join("tokens.bin"), window.tokens.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    std::fs::write(directory.join("positions.bin"), window.positions.iter().flat_map(|&p| (p as u32).to_le_bytes()).collect::<Vec<_>>())?;
    let mut result = if full {scores.result()?} else {
        let (cosine, relative_l2) = hidden.result()?;
        json!({"cosine":cosine, "relative_l2":relative_l2})
    };
    result["schema"] = "cuteafd.minimax.golden.window/2".into();
    result["window"] = window.id.clone().into(); result["binding"] = binding.clone();
    result["positions"] = json!(window.positions); result["rows"] = expected_rows.into();
    result["width"] = width.into(); result["dtype"] = "F32".into(); result["byte_order"] = "little".into();
    // Full: row i predicts positions[i]. Partial: row i is token i's hidden state.
    result["row_targets"] = (if full {"positions"} else {"tokens"}).into(); result["partial_target"] = (!full).into();
    result["prefix_byte_exact"] = args.check_prefix.into();
    result["generated"] = if generated.rows > 0 {generated.result()?} else {Value::Null};
    seal_window(directory, &filename, &mut result)?;
    if full {
        println!("MiniMax golden window {}: {} scored rows; top-1 {}; NLL {}; KL(reference||engine) {}",
            window.id, result["scored_tokens"], result["top1"], result["nll"], result["mean_kl"]);
    } else {
        println!("MiniMax partial layer {} window {}: cosine {}; relative L2 {}",
            args.last_layer, window.id, result["cosine"], result["relative_l2"]);
    }
    if !full {verdict(&result, false)?;}
    Ok(result)
}

pub(crate) async fn run_golden(args: GoldenArgs) -> Result<()> {
    let checkpoint = Checkpoint::coordinator(&args.snapshot, false, false)?;
    let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
    cfg.release_geometry()?;
    let weights = RankWeightLayout::from_checkpoint(&checkpoint, 0, 1)?;
    let mut catalog = cuteafd_loader::read_expert_catalog(&args.snapshot)?;
    catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::None)?;
    if args.check_headers {
        println!("MiniMax golden checkpoint headers passed: {} layers, native MXFP8/NVFP4", cfg.layers);
        return Ok(());
    }
    let remote = selection(&args)?;
    if std::env::var_os("CUTEAFD_MINIMAX_DUMP_DIR").is_some() {
        ensure!(args.window.is_some() && !args.check_prefix,
            "layer dumps require one --window and a separate --check-prefix invocation");
    }
    let (reference, windows) = references(&args, &cfg)?;
    let binding = binding(&args, &reference, &windows, &checkpoint)?;
    let output = args.save_logits.as_ref().context("MiniMax golden requires --save-logits output directory")?;
    std::fs::create_dir_all(output)?;
    let identity_path = output.join("binding.json");
    if identity_path.exists() {
        ensure!(args.resume && read_json(&identity_path)? == binding,
            "output already exists or resume identity differs; use a fresh output directory");
    } else {
        ensure!(std::fs::read_dir(output)?.next().is_none(), "output directory must be empty before first run");
        write_json(&identity_path, &binding)?;
    }
    let native = args.native_lib.as_ref().context("MiniMax golden requires --native-lib")?;
    cuteafd_core::set_expert_geometry(ExpertGeometry::MINIMAX_M3)
        .map_err(|_| MiniMaxError::Invalid("expert geometry already installed"))?;
    let library = std::sync::Arc::new(cuteafd_ffi::minimax::load_library(native)?);
    library.cuda_set_device(args.device)?;
    let programs = library.programs()?.with_manifest(&args.manifest)?;
    programs.load_matching(|name| name.starts_with("minimax_"))?;
    let context = windows.iter().map(|w| w.tokens.len()).max().unwrap().div_ceil(64) * 64;
    ensure!(!args.check_prefix || windows[0].tokens.len() >= 66, "prefix qualification needs 66 tokens in first window");
    let sizing = admission::RuntimeSizing {rows:args.prefill_rows as usize, decode_rows:(args.prefill_rows as usize).min(64),
        sequences:1, context:context as u64, spark_ranks:args.peers.len(), logit_rows:args.prefill_rows as usize, lanes:1,
        host_landing:!args.peers.is_empty() && super::routed::host_landing(&library)?};
    let program_sizes = admission::ProgramSizes::from_manifest(&read_json(&args.manifest)?, sizing)?;
    let ledger = admission::RuntimeLedger::new(&cfg, sizing, &program_sizes)?;
    let package = crate::shared::experts::fp8::package_directory(native, 1, cuteafd_loader::formats::fp8_experts::ExpertFormat::Nvfp4);
    let workspace = if args.expert_layers.is_empty() {0} else {
        admission::expert_workspace(&read_json(&package.join("manifest.json"))?, 1, sizing.rows)?
    };
    let mut reservations = admission::Reservations {weights:weights.resident_bytes,
        local_experts:admission::expert_resident_bytes(&catalog, &args.expert_layers, 1, 0)?,
        expert_workspace:workspace, headroom:2 << 30, ..Default::default()};
    reservations.startup = reservations.fixed_bytes(&ledger)?.checked_add(admission::scale_prep_staging_bytes(&weights))
        .context("MiniMax golden startup admission overflow")?;
    let (free, total) = library.cuda_memory_info()?;
    let available = cuteafd_ffi::coordinator_gpu_budget().map_or(free as u64,
        |b| (free as u64).min(b.0.saturating_sub((total - free) as u64)));
    let pool_tokens = if args.check_prefix {(context as u64).max(128) * 3 + 256} else {context as u64};
    let admitted = admission::admit(&cfg, sizing, &ledger, reservations, available, Some(pool_tokens), Qwen4KvCache::Fp8)?;
    if !remote.is_empty() {
        if let Ok(selected) = std::env::var("CUTEAFD_GOLDEN_EXPERT_LAYERS") {
            let selected = selected.split(',').map(str::parse::<usize>).collect::<std::result::Result<Vec<_>, _>>()?;
            ensure!(selected == remote, "golden runner worker layers differ from executed remote layers");
        }
    }
    // Serve's start-up, without its startup decode recordings: every golden step runs eagerly, so
    // the gate scores the kernels themselves. A served replay is the same launch sequence
    // (CUTEAFD_MINIMAX_GRAPH_VERIFY=1 checks replay against eager bit for bit).
    let super::serve::Runtime { execution, cache, embedding, token_ids, embedded, routed } =
        super::serve::start(&library, args.device, &programs, super::serve::Start {
            snapshot: &args.snapshot, checkpoint: &checkpoint, cfg: &cfg, sizing, ledger: &ledger, cache: &admitted,
            local_layers: &args.expert_layers, spark_layers: &remote, peers: &args.peers, catalog: &catalog,
            package: &package, embed: EmbedPlacement::Gpu, available, draft: None })?;
    let experts: Box<dyn ExpertExecutor<'_>> = match routed { Some(routed) => Box::new(routed), None => Box::new(DenseOnly) };
    let mut runner = Runner {execution:&execution, cache:&cache, embedding:&embedding, ids:token_ids, embedded, experts,
        last_layer:args.last_layer, rows:sizing.rows};
    if args.check_prefix {check_prefix(&mut runner, &windows[0].tokens)?;}
    let mut results = Vec::new();
    for window in &windows {
        let directory = output.join("windows").join(&window.id);
        let result = if args.resume {completed(&directory, &binding, args.last_layer == 59)?} else {None};
        results.push(match result {Some(value) => value,
            None => score_window(&mut runner, &args, window, &directory, &binding, cfg.vocab_size)?});
    }
    execution.drain()?;
    let mut summary = json!({"schema":"cuteafd.minimax.golden/1","binding":binding,"windows":results,
        "passed":true,"partial_target":args.last_layer != 59,"prefix_byte_exact":args.check_prefix});
    if args.last_layer == 59 {
        let generated = results.iter().filter_map(|r| r["generated"].as_object()).collect::<Vec<_>>();
        let rows = generated.iter().map(|r| r["scored_tokens"].as_u64().unwrap()).sum::<u64>();
        ensure!(rows > 0, "full fidelity qualification requires generated-role scored positions");
        summary["scored_tokens"] = rows.into();
        summary["gate_scope"] = "generated".into();
        for field in ["top1", "mean_kl", "nll", "reference_nll"] {
            summary[field] = (generated.iter().map(|r| r[field].as_f64().unwrap() * r["scored_tokens"].as_u64().unwrap() as f64)
                .sum::<f64>() / rows as f64).into();
        }
        verdict(&summary, true)?;
    }
    write_json(&output.join("summary.json"), &summary)?;
    println!("MiniMax golden passed: {} windows; partial_target={}", windows.len(), args.last_layer != 59);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn args(extra: &[&str]) -> GoldenArgs {
        let cli = crate::cli::Cli::try_parse_from(["cuteafd", "minimax-golden", "--snapshot", "/snapshot"]
            .into_iter().chain(extra.iter().copied())).unwrap();
        let crate::cli::Commands::MinimaxGolden(args) = cli.command else {panic!("MiniMax golden registration")};
        args
    }

    #[test]
    fn minimax_golden_parses_runner_arguments_and_local_subset() -> Result<()> {
        let args = args(&["--golden", "/reference", "--window", "code-1", "--native-lib", "/native.so",
            "--manifest", "/PROGRAMS.json", "--last-layer", "5", "--expert-layers", "3,4,5",
            "--save-logits", "/output", "--resume", "--check-prefix", "--prefill-rows", "64", "--score-shape", "prefill"]);
        assert_eq!(args.last_layer, 5); assert_eq!(args.expert_layers, [3, 4, 5]);
        assert_eq!(args.prefill_rows, 64); assert_eq!(args.score_shape, ScoreShape::Prefill);
        assert!(args.resume && args.check_prefix); assert!(selection(&args)?.is_empty());
        Ok(())
    }

    #[test]
    fn minimax_golden_rejects_missing_duplicate_and_unexecuted_experts() {
        for extra in [&["--last-layer", "5"][..], &["--last-layer", "5", "--expert-layers", "3,4"],
            &["--last-layer", "5", "--expert-layers", "3,4,4,5"],
            &["--last-layer", "3", "--expert-layers", "3,4"], &["--last-layer", "60"]] {
            assert!(selection(&args(extra)).is_err());
        }
        assert!(selection(&args(&["--last-layer", "0"])).is_ok());
        for extra in [&["--prefill-rows", "0"][..], &["--prefill-rows", "4097"], &["--resume"], &["--score-shape", "unknown"]] {
            assert!(crate::cli::Cli::try_parse_from(["cuteafd", "minimax-golden", "--snapshot", "/snapshot"]
                .into_iter().chain(extra.iter().copied())).is_err());
        }
    }

    #[test]
    fn minimax_golden_remote_selection_covers_only_nonlocal_layers() -> Result<()> {
        let mut args = args(&["--expert-layers", "3,4"]);
        args.peers = (1..=4).map(|port| SocketAddr::new(std::net::Ipv4Addr::LOCALHOST.into(), port)).collect();
        assert_eq!(selection(&args)?, (5..60).collect::<Vec<_>>());
        args.peers.pop(); assert!(selection(&args).is_ok());
        args.peers.pop(); assert!(selection(&args).is_ok());
        args.peers.pop(); assert!(selection(&args).is_err());
        args.peers.push(args.peers[0]); assert!(selection(&args).is_err());
        Ok(())
    }

    #[test]
    fn minimax_golden_scores_shifted_targets_finite_rows_and_inclusive_bars() -> Result<()> {
        let mut scores = Scores::default();
        scores.add(&[4., 0.], &[4., 0.], 0)?;
        scores.add(&[0., 4.], &[0., 4.], 1)?;
        let result = scores.result()?;
        assert_eq!(result["scored_tokens"], 2); assert_eq!(result["top1"], 1.0);
        assert_eq!(result["mean_kl"], 0.0); assert!(result["nll"].as_f64().unwrap() < 0.02);
        verdict(&result, true)?;
        assert!(Scores::default().result().is_err());
        assert!(scores.add(&[f32::NAN, 0.], &[0., 1.], 0).is_err());
        assert!(scores.add(&[0., 1.], &[0., 1.], 2).is_err());
        verdict(&json!({"top1":0.94,"mean_kl":0.03,"nll":0.255,"reference_nll":0.25}), true)?;
        for invalid in [json!({"top1":0.939,"mean_kl":0.0,"nll":0.25,"reference_nll":0.25}),
            json!({"top1":1.,"mean_kl":0.031,"nll":0.25,"reference_nll":0.25}),
            json!({"top1":1.,"mean_kl":0.0,"nll":0.27,"reference_nll":0.25}),
            json!({"top1":1.,"mean_kl":0.0,"nll":0.25}), json!({"top1":null,"mean_kl":0.0})] {
            assert!(verdict(&invalid, true).is_err());
        }
        verdict(&json!({"cosine":0.99,"relative_l2":0.05}), false)?;
        assert!(verdict(&json!({"cosine":0.98,"relative_l2":0.0}), false).is_err());
        Ok(())
    }

    #[test]
    fn minimax_golden_output_preserves_f32_bits_and_resume_checks_every_payload() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let binding = json!({"last_layer":59,"check_prefix":true});
        let values = [-0.0f32, 4., 2., -1.];
        let raw = values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        std::fs::write(directory.path().join("logits.bin"), &raw)?;
        std::fs::write(directory.path().join("tokens.bin"), [1u32, 0].into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>())?;
        std::fs::write(directory.path().join("positions.bin"), 1u32.to_le_bytes())?;
        let mut result = json!({"schema":"cuteafd.minimax.golden.window/2", "binding":binding,
            "top1":1.0,"mean_kl":0.0,"prefix_byte_exact":true,"rows":2,"width":2,
            "dtype":"F32","byte_order":"little","positions":[1],"row_targets":"positions"});
        seal_window(directory.path(), "logits.bin", &mut result)?;
        let read = completed(directory.path(), &binding, true)?.unwrap();
        assert_eq!(read, result);
        assert_eq!(read["sha256"]["logits.bin"], hash(&directory.path().join("logits.bin"))?);
        assert_eq!(floats(&raw, 4)?.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        for filename in ["tokens.bin", "positions.bin", "logits.bin"] {
            let path = directory.path().join(filename);
            let previous = std::fs::read(&path)?; std::fs::write(&path, b"corrupt")?;
            assert!(completed(directory.path(), &binding, true).is_err()); std::fs::write(path, previous)?;
        }
        let changed = json!({"last_layer":59,"check_prefix":false});
        assert!(completed(directory.path(), &changed, true).is_err());
        result["sha256"] = json!({}); write_json(&directory.path().join("result.json"), &result)?;
        assert!(completed(directory.path(), &binding, true).is_err());
        Ok(())
    }

    #[test]
    fn minimax_golden_refuses_bad_extents_nonfinite_and_escaped_paths() -> Result<()> {
        let root = tempfile::tempdir()?;
        std::fs::create_dir(root.path().join("windows"))?;
        assert!(safe_path(root.path(), "windows").is_ok());
        for path in ["", "../outside", "/outside", "windows/../outside"] {assert!(safe_path(root.path(), path).is_err());}
        assert!(floats(&[0; 3], 1).is_err());
        assert!(floats(&f32::INFINITY.to_le_bytes(), 1).is_err());
        assert!(floats(&[0; 4], usize::MAX).is_err());
        Ok(())
    }

    #[test]
    fn minimax_golden_reads_layer_exports_of_a_qualified_full_reference() -> Result<()> {
        use cuteafd_loader::families::minimax_m3::config::MsaConfig;
        let root = tempfile::tempdir()?;
        let snapshot = root.path().join("snapshot"); let golden = root.path().join("golden");
        let folder = golden.join("windows/code-1");
        std::fs::create_dir_all(&snapshot)?; std::fs::create_dir_all(&folder)?;
        std::fs::write(snapshot.join("config.json"), b"config")?;
        std::fs::write(snapshot.join("tokenizer.json"), b"tokenizer")?;
        std::fs::write(folder.join("tokens.bin"), [0u32, 1, 0].into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>())?;
        std::fs::write(folder.join("layer00.bin"), [0f32; 6].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>())?;
        write_json(&golden.join("windows.json"), &json!({"set_sha256":"set", "windows":[{
            "id":"code-1","tokens":[0,1,0],"roles":["ctx","ctx","gen"]}]}))?;
        let identity = json!({"snapshot_revision":REVISION,
            "config_sha256":hash(&snapshot.join("config.json"))?,"tokenizer_sha256":hash(&snapshot.join("tokenizer.json"))?});
        let mut meta = json!({"schema":"cuteafd.fidelity.golden/2", "family":"minimax_m3",
            "checkpoint":"MiniMaxAI/MiniMax-M3.1-Flash","mtp":false,
            "activation":"served-cache","set_sha256":"set", "snapshot_identity":identity,
            "prefix_qualification":{"passed":true,"finite":true,"set_sha256":"set","snapshot_identity":identity},
            "source_pin":{"transformers":{"revision":"6b6792ff3b1a434913dc59659d4461acc9c3a51a",
                "sha256":{"src/transformers/models/minimax_m3_vl/modular_minimax_m3_vl.py":
                    "1c755515c8aefdbc6f9145ae8f051e43907e21022d9644e81afe47cd2e37b58d"}}},
            "windows":[{"id":"code-1","path":"windows/code-1","vocab":2,"positions":[1,2]}]});
        write_json(&golden.join("meta.json"), &meta)?;
        let mut args = args(&["--last-layer", "0"]); args.snapshot = snapshot; args.golden = Some(golden.clone());
        let cfg = MiniMaxM3Config {vocab_size:2,hidden:2,layers:60,heads:1,kv_heads:1,head_dim:2,rotary_dim:2,
            rope_theta:10000.,rms_norm_eps:1e-6,moe:vec![false;60],dense_intermediate:2,moe_intermediate:2,
            shared_intermediate:2,experts:2,topk:1,routed_scale:2.,swiglu_alpha:1.702,swiglu_limit:7.,
            sparse:vec![true;60],msa:MsaConfig{index_heads:1,index_dim:2,block:64,topk_blocks:16,local_blocks:1},
            max_position:128,tie_embeddings:false,bos:0,eos:vec![1]};
        let (_, windows) = references(&args, &cfg)?;
        assert_eq!(windows[0].tokens, [0,1,0]); assert_eq!(windows[0].generated, [false,false,true]);
        // A full-model run needs logits.bin; an unqualified reference is refused.
        args.last_layer = 59; assert!(references(&args, &cfg).is_err()); args.last_layer = 0;
        meta["prefix_qualification"]["passed"] = false.into();
        write_json(&golden.join("meta.json"), &meta)?;
        assert!(references(&args, &cfg).is_err());
        meta["prefix_qualification"]["passed"] = true.into();
        write_json(&golden.join("meta.json"), &meta)?;
        std::fs::write(folder.join("layer00.bin"), [1f32; 5].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>())?;
        assert!(references(&args, &cfg).is_err(), "a short layer export is refused");
        Ok(())
    }
}
