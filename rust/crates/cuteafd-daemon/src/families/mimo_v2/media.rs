//! MiMo's image request state and resident local encoder readiness.
use anyhow::{Context, Result};
use cuteafd_api::openai::{media::MediaPreparer, NativeRequest};
use cuteafd_engine::media::{EncodeJob, MediaKeys, RequestMedia};
use cuteafd_loader::{media::{ImageFamily, ProcessorConfig, SpanExpander}, plan::MediaMode};
use std::sync::Arc;

pub(super) struct ReadyVision {
    pub encoder: crate::shared::vision::local::LocalEncoder,
    pub preparer: Arc<MediaPreparer>,
    pub cache_bytes: usize,
}
impl ReadyVision {
    pub fn load(args: &super::EngineArgs, library: &cuteafd_ffi::NativeLibrary, mode: MediaMode, prefix: &super::serve::PrefixArgs,
        cache_bytes: Option<u64>) -> Result<(Option<Self>, super::serve::PrefixArgs)> {
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(args.snapshot.join("config.json"))?)?;
        if mode == MediaMode::Off || config.get("vision_config").is_none() {
            return Ok((None, prefix.clone()));
        }
        let gpu = match mode {
            MediaMode::Rtx(gpu) => gpu.map(|gpu| i32::try_from(gpu)).transpose()?.unwrap_or(args.device),
            MediaMode::Auto | MediaMode::Spark(_) => {
                tracing::warn!(?mode, "MiMo encoder placement pending: use --vision rtx[:gpu] to enable images");
                return Ok((None, prefix.clone()));
            }
            MediaMode::Off => unreachable!(),
        };
        let processor = ProcessorConfig::from_snapshot(&args.snapshot, ImageFamily::Mimo)?;
        let spec = crate::shared::vision::TowerSpec::mimo(&args.snapshot, 4096)?;
        let info = library.cuda_device_info(gpu)?;
        let sm = u32::try_from(info.compute_capability_major * 10 + info.compute_capability_minor)?;
        let revision = args.snapshot.file_name().and_then(|v| v.to_str()).context("snapshot revision")?;
        let preparer = Arc::new(MediaPreparer::for_loaded_encoder(processor.clone(), spec.encoder_id(revision, sm), 4)?);
        anyhow::ensure!(preparer.config().max_image_tokens <= 4096, "MiMo tower capacity is 4096 tokens per image");
        let (prefix, cache_bytes) = prefix.with_media_headroom(cache_bytes)?;
        let ledger = cuteafd_ffi::vision::NativeVision::required(&args.native_lib, &spec.native)?;
        library.cuda_set_device(gpu)?;
        let admitted = ledger.total_bytes();
        let loaded = (|| -> Result<_> {
        let (free, total) = library.cuda_memory_info()?;
        cuteafd_core::serving_capacity::admit_device_reservations(95,
            cuteafd_core::serving_capacity::DeviceMemory { device: gpu as u32,
                total_bytes: total as u64, baseline_free_bytes: free as u64 },
            &[cuteafd_core::serving_capacity::MemoryReservation { name: "vision.resident_weights_scratch".into(), bytes: admitted }])?;
        // Start before LM preflight: its live baseline already includes the admitted tower.
        Ok(crate::shared::vision::EncoderService::start(spec, args.native_lib.clone(), gpu, admitted)?)
        })();
        let restored = library.cuda_set_device(args.device);
        let service = loaded?;
        restored?;
        let width = config["hidden_size"].as_u64().context("MiMo hidden_size")? as usize;
        tracing::info!(gpu, admitted_bytes = admitted, cache_bytes, sm, "MiMo resident vision encoder ready");
        Ok((Some(Self { encoder: crate::shared::vision::local::LocalEncoder::new(service, &processor, width, 4096),
            preparer, cache_bytes }), prefix))
    }
}

pub(super) enum Encoder {
    Local(crate::shared::vision::local::LocalEncoder),
    Off,
}
impl cuteafd_engine::media::EncoderClient for Encoder {
    fn submit(&mut self, job: EncodeJob) -> std::result::Result<cuteafd_engine::media::EncoderTicket, cuteafd_engine::media::MediaError> {
        use cuteafd_engine::media::MediaError;
        match self { Self::Local(client) => client.submit(job), Self::Off => Err(MediaError::Encoder("encoder not loaded".into())) }
    }
    fn poll(&mut self, ticket: cuteafd_engine::media::EncoderTicket) -> Option<std::result::Result<cuteafd_engine::media::EncodeOutput, cuteafd_engine::media::MediaError>> {
        match self { Self::Local(client) => client.poll(ticket), Self::Off => None }
    }
    fn cancel(&mut self, ticket: cuteafd_engine::media::EncoderTicket) {
        if let Self::Local(client) = self { client.cancel(ticket); }
    }
}

pub(super) struct Prompt {
    pub job: NativeRequest,
    pub tokens: Vec<u32>,
    pub keys: MediaKeys,
}
pub(super) fn prepare(job: NativeRequest, tokens: Vec<u32>, config: &serde_json::Value,
    vocabulary: usize, hidden: usize, max_context: usize) -> Result<(Prompt, RequestMedia, Vec<EncodeJob>)> {
    let (tokens, spans) = if config.get("vision_config").is_some() {
        let expander = SpanExpander::from_config(config, vocabulary as u32)?;
        let images = job.media.iter().map(|image| image.as_ref().clone()).collect::<Vec<_>>();
        if job.probe.as_ref().is_some_and(|p| p.spec.prompt_ids.is_some() && !images.is_empty()) {
            let probe = job.probe.as_ref().unwrap();
            probe.spec.validate_media()?;
            anyhow::ensure!(probe.spec.media.len() == images.len() && tokens.len() <= max_context,
                "expanded probe media count/context differs");
            anyhow::ensure!(tokens.iter().all(|&id| id < expander.vocabulary), "probe token outside vocabulary");
            let mut spans = Vec::with_capacity(images.len());
            for (span, image) in probe.spec.media.iter().zip(&images) {
                let end = span.start.checked_add(span.len).context("probe media extent")?;
                anyhow::ensure!(span.len == image.tokens && span.grid == [image.grid.t, image.grid.h, image.grid.w]
                    && span.key == key_hex(image.key), "probe prepared image identity differs");
                anyhow::ensure!(span.start > 0 && end < tokens.len() && tokens[span.start - 1] == expander.start
                    && tokens[end] == expander.end && tokens[span.start..end].iter().all(|&id| id == expander.placeholder),
                    "probe image rows/marker boundaries differ");
                spans.push(cuteafd_loader::media::MediaSpan { start: span.start, len: span.len, key: image.key });
            }
            anyhow::ensure!(tokens.iter().filter(|&&id| id == expander.placeholder).count()
                == spans.iter().map(|span| span.len).sum::<usize>(), "unbound probe image placeholders");
            (tokens, spans)
        } else {
            let expanded = expander.expand(&tokens, &images, max_context)?;
            (expanded.tokens, expanded.media)
        }
    } else {
        anyhow::ensure!(job.media.is_empty(), "checkpoint has no vision tower");
        (tokens, Vec::new())
    };
    if let Some(probe) = &job.probe {
        anyhow::ensure!(spans.len() == job.media.len(), "probe image count differs");
        let echo = spans.iter().zip(&job.media).enumerate().map(|(i, (span, image))| {
            cuteafd_api::openai::probe::ProbeMedia { start: span.start, len: span.len, kind: "image".into(),
                key: key_hex(span.key), grid: [image.grid.t, image.grid.h, image.grid.w],
                fixture: probe.spec.media.get(i).and_then(|s| s.fixture.clone()), image_url: None }
        }).collect();
        probe.media(echo);
    }
    let media = RequestMedia::new(spans.clone(), hidden, tokens.len())?;
    let keys = MediaKeys::new(&tokens, vocabulary as u32, &spans)?;
    let jobs = job.media.iter().map(|image| EncodeJob { key: image.key,
        grid: [image.grid.t, image.grid.h, image.grid.w], rgb8: image.rgb8.clone(),
        tokens: image.tokens, hidden_width: hidden }).collect();
    Ok((Prompt { job, tokens, keys }, media, jobs))
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct FeatureMetadata {
    schema: String,
    key: String,
    grid: [u32; 3],
    shape: [usize; 2],
    dtype: String,
    sha256: String,
    tower_dtype: String,
    fixture_sha256: String,
    snapshot_identity: serde_json::Value,
}

fn snapshot_identity(snapshot: &std::path::Path) -> Result<serde_json::Value> {
    use sha2::{Digest, Sha256};
    let mut identity = serde_json::json!({"snapshot_revision": snapshot.file_name().and_then(|s| s.to_str())
        .context("snapshot revision")?});
    for (key, file) in [("config", "config.json"), ("tokenizer", "tokenizer.json"),
        ("modeling", "modeling_mimo_v2.py"), ("preprocessor", "preprocessor_config.json")] {
        identity[format!("{key}_sha256")] = format!("{:x}", Sha256::digest(std::fs::read(snapshot.join(file))?)).into();
    }
    Ok(identity)
}

fn read_bounded(path: &std::path::Path, limit: usize) -> Result<Vec<u8>> {
    use std::io::Read;
    let file = std::fs::File::open(path)?;
    anyhow::ensure!(file.metadata()?.len() <= limit as u64, "feature file exceeds bound");
    let mut bytes = Vec::new(); file.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    anyhow::ensure!(bytes.len() <= limit, "feature file exceeds bound");
    Ok(bytes)
}

/// Strict paired-G4 hook. Only scoring probes bypass the encoder; ordinary
/// generation remains native. Separate cache identities cannot warm native images.
pub(super) fn probe_features(prompt: &Prompt, media: &mut RequestMedia,
    cache: &mut cuteafd_engine::media::EmbeddingCache, snapshot: &std::path::Path) -> Result<()> {
    let Some(probe) = prompt.job.probe.as_ref().filter(|p| p.spec.score_from.is_some() && !p.spec.media.is_empty()) else {
        return Ok(());
    };
    let Some(root) = std::env::var_os("CUTEAFD_MEDIA_FEATURES_DIR") else { return Ok(()); };
    anyhow::ensure!(probe.spec.cold && probe.spec.no_speculation, "feature probes require explicit cold/no_speculation");
    probe.spec.validate_media()?;
    apply_probe_features(prompt, media, cache, &std::path::PathBuf::from(root), &snapshot_identity(snapshot)?)
}

fn apply_probe_features(prompt: &Prompt, media: &mut RequestMedia,
    cache: &mut cuteafd_engine::media::EmbeddingCache, root: &std::path::Path, identity: &serde_json::Value) -> Result<()> {
    use sha2::{Digest, Sha256};
    let probe = prompt.job.probe.as_ref().context("features require a probe")?;
    anyhow::ensure!(probe.spec.cold && probe.spec.no_speculation && probe.spec.score_from.is_some()
        && !probe.spec.media.is_empty(), "features require a cold, speculation-free media scoring probe");
    let root = root.canonicalize()?;
    let mut provenance = Vec::new();
    for span in &probe.spec.media {
        let fixture = span.fixture.as_ref().context("feature probes require fixture identity")?;
        anyhow::ensure!(cuteafd_api::openai::probe::sha256_hex(&span.key), "invalid feature key");
        let metadata_path = root.join(format!("{}.json", span.key)).canonicalize()?;
        let payload_path = root.join(format!("{}.bf16", span.key)).canonicalize()?;
        anyhow::ensure!(metadata_path.starts_with(&root) && payload_path.starts_with(&root), "feature path escapes root");
        let raw = read_bounded(&metadata_path, 64 << 10)?;
        let meta: FeatureMetadata = serde_json::from_slice(&raw)?;
        anyhow::ensure!(meta.schema == "cuteafd.media.features/1" && meta.key == span.key && meta.grid == span.grid
            && meta.shape == [span.len, media.row_bytes() / 2] && meta.dtype == "bf16-le"
            && matches!(meta.tower_dtype.as_str(), "bf16" | "fp32")
            && cuteafd_api::openai::probe::sha256_hex(&meta.sha256)
            && meta.fixture_sha256 == fixture.sha256 && meta.snapshot_identity == *identity,
            "reference feature metadata differs from prepared request/snapshot");
        let bytes = span.len.checked_mul(media.row_bytes()).context("feature byte extent")?;
        let mut hash = Sha256::new();
        hash.update(b"cuteafd.probe.feature_override/1\0"); hash.update(span.key.as_bytes()); hash.update(&raw);
        let override_key = cuteafd_loader::media::ImageKey(hash.finalize().into());
        let image = prompt.job.media.iter().find(|i| key_hex(i.key) == span.key).context("feature span has no prepared image")?;
        // Admission precedes payload allocation, including on a warm override-cache hit.
        let pin = cache.reserve(override_key, bytes)?;
        let payload = read_bounded(&payload_path, bytes)?;
        anyhow::ensure!(payload.len() == bytes && format!("{:x}", Sha256::digest(&payload)) == meta.sha256,
            "reference feature payload length/hash differs");
        anyhow::ensure!(payload.chunks_exact(2).all(|b| {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            (bits & 0x7f80) != 0x7f80
        }), "reference features contain nonfinite BF16 values");
        let lease = cache.complete(override_key, Arc::from(payload))?;
        media.attach_probe_override(image.key, lease)?; drop(pin);
        provenance.push(serde_json::from_slice::<serde_json::Value>(&raw)?);
    }
    probe.provenance(serde_json::json!({"mode": "reference_features", "probe_only": true,
        "encoder_bypassed": true, "features": provenance}));
    Ok(())
}

fn key_hex(key: cuteafd_loader::media::ImageKey) -> String {
    key.0.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::TargetSamplingParams;
    use cuteafd_loader::media::{ImageGrid, ImageKey, PreparedImage};

    fn request(images: Vec<Arc<PreparedImage>>) -> NativeRequest {
        NativeRequest { prompt: String::new(), constraint: None, images: Vec::new(), media: images,
            max_tokens: 8, sampling: TargetSamplingParams::default(), stop_token_ids: Vec::new(),
            events: tokio::sync::mpsc::unbounded_channel().0, probe: None }
    }
    fn config() -> serde_json::Value {
        serde_json::json!({"vision_config": {}, "image_token_id": 5,
            "vision_start_token_id": 4, "vision_end_token_id": 6})
    }
    fn image() -> Arc<PreparedImage> {
        Arc::new(PreparedImage { key: ImageKey([7; 32]), grid: ImageGrid { t: 1, h: 4, w: 4 },
            rgb8: Arc::from(vec![0; 4 * 4 * 768]), tokens: 4 })
    }
    #[test]
    fn expanded_image_ids_and_prefix_hints_are_separate() {
        let (prompt, media, jobs) = prepare(request(vec![image()]), vec![1, 4, 5, 6, 2],
            &config(), 32, 2, 16).unwrap();
        assert_eq!(prompt.tokens, [1, 4, 5, 5, 5, 5, 6, 2]);
        assert_eq!(&prompt.keys.tokens()[..2], &[1, 4]);
        assert!(prompt.keys.tokens()[2..6].iter().all(|t| t & (1 << 31) != 0));
        assert_eq!(media.spans()[0].start, 2);
        assert_eq!(media.spans()[0].len, 4);
        assert!(!media.ready(0, 8));
        assert!(media.ready(6, 8));
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].grid, [1, 4, 4]);
        assert_eq!(jobs[0].feature_bytes().unwrap(), 16);
    }
    #[test]
    fn generation_probe_echoes_real_expanded_spans_without_sources() {
        let mut job = request(vec![image()]);
        let probe = cuteafd_api::openai::probe::Probe::new(Default::default());
        job.probe = Some(probe.clone());
        prepare(job, vec![1, 4, 5, 6, 2], &config(), 32, 2, 16).unwrap();
        let record = probe.record();
        assert_eq!((record.media[0].start, record.media[0].len, record.media[0].grid), (2, 4, [1, 4, 4]));
        assert_eq!(record.media[0].key, "07".repeat(32));
        assert!(record.media[0].image_url.is_none());
    }
    #[test]
    fn expanded_probe_ids_are_verified_not_expanded_twice() {
        use cuteafd_api::openai::probe::{Probe, ProbeSpec, ProbeMedia, ProbeImageUrl, ProbeFixture};
        let tokens = vec![1, 4, 5, 5, 5, 5, 6, 2];
        let span = ProbeMedia { start: 2, len: 4, kind: "image".into(), key: "07".repeat(32), grid: [1, 4, 4],
            fixture: Some(ProbeFixture { path: "chart.png".into(), sha256: "ab".repeat(32) }),
            image_url: Some(ProbeImageUrl { url: "data:image/png;base64,fixture".into(), detail: None }) };
        let run = |span: ProbeMedia, tokens: Vec<u32>| {
            let probe = Probe::new(ProbeSpec { prompt_ids: Some(tokens.clone()), media: vec![span], ..Default::default() });
            let mut job = request(vec![image()]); job.probe = Some(probe.clone());
            prepare(job, tokens, &config(), 32, 2, 16).map(|prepared| (prepared, probe.record()))
        };
        let ((prompt, _, _), record) = run(span.clone(), tokens.clone()).unwrap();
        assert_eq!(prompt.tokens, tokens);
        assert_eq!(record.media[0].fixture, span.fixture);
        assert!(record.media[0].image_url.is_none());
        let mut wrong = span.clone(); wrong.key = "08".repeat(32); assert!(run(wrong, tokens.clone()).is_err());
        let mut wrong = span.clone(); wrong.grid = [1, 2, 8]; assert!(run(wrong, tokens.clone()).is_err());
        let mut wrong = tokens.clone(); wrong[3] = 1; assert!(run(span.clone(), wrong).is_err());
        let mut wrong = tokens; wrong[1] = 1; assert!(run(span, wrong).is_err());
        let mut job = request(vec![image()]);
        job.probe = Some(Probe::new(ProbeSpec { prompt_ids: Some(vec![4, 5, 6]), ..Default::default() }));
        assert!(prepare(job, vec![4, 5, 6], &config(), 32, 2, 16).is_err());
    }
    #[test]
    fn text_path_keeps_native_ids_and_rejects_missing_image_rows() {
        let tokens = vec![1, 2, 3];
        let (prompt, media, jobs) = prepare(request(Vec::new()), tokens.clone(), &config(), 32, 2, 16).unwrap();
        assert_eq!(prompt.tokens, tokens);
        assert_eq!(prompt.keys.tokens(), tokens);
        assert!(media.spans().is_empty() && jobs.is_empty());
        assert!(prepare(request(vec![image()]), vec![1, 2], &config(), 32, 2, 16).is_err());
        assert!(prepare(request(vec![image()]), vec![4, 5, 6], &config(), 32, 2, 4).is_err());
        assert!(prepare(request(vec![image()]), vec![4, 5, 6], &serde_json::json!({}), 32, 2, 16).is_err());
    }
}
