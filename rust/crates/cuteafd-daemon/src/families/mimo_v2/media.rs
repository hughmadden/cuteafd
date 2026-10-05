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
        let expanded = expander.expand(&tokens, &images, max_context)?;
        (expanded.tokens, expanded.media)
    } else {
        anyhow::ensure!(job.media.is_empty(), "checkpoint has no vision tower");
        (tokens, Vec::new())
    };
    let media = RequestMedia::new(spans.clone(), hidden, tokens.len())?;
    let keys = MediaKeys::new(&tokens, vocabulary as u32, &spans)?;
    let jobs = job.media.iter().map(|image| EncodeJob { key: image.key,
        grid: [image.grid.t, image.grid.h, image.grid.w], rgb8: image.rgb8.clone(),
        tokens: image.tokens, hidden_width: hidden }).collect();
    Ok((Prompt { job, tokens, keys }, media, jobs))
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
