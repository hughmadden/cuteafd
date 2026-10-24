//! Bundled image encoder and exact language-model feature spans.
use super::MiniMaxError;
use anyhow::{Context, Result};
use cuteafd_api::openai::{media::MediaPreparer, NativeRequest};
use cuteafd_engine::media::{EncodeJob, MediaKeys, RequestMedia};
use cuteafd_loader::{media::{ImageFamily, ProcessorConfig, SpanExpander, minimax::{ENCODER_TOKENS, MAX_SEGMENT_FRAMES}}, plan::MediaMode};
use crate::shared::{prefix::PrefixArgs, qwen::{Encoder, scheduler::Prompt}, vision::{TowerSpec, EncoderService}};
use std::{path::Path, sync::Arc};

pub(super) struct ReadyVision {
    pub encoder: Encoder,
    pub preparer: Arc<MediaPreparer>,
    pub cache_bytes: usize,
    pub spark_reserve_bytes: u64,
}
pub(super) struct RemoteVision {
    peers: Vec<std::net::SocketAddr>,
    hash: [u8; 32],
    revision: String,
}
impl RemoteVision {
    pub fn parse(peers: Option<&str>, hash: Option<&str>, revision: Option<&str>) -> Result<Option<Self>> {
        let Some(peers) = peers else {
            if hash.is_some() || revision.is_some() { return Err(MiniMaxError::Invalid("encoder identity without peers").into()); }
            return Ok(None);
        };
        let peers = peers.split(',').map(str::parse).collect::<std::result::Result<Vec<_>, _>>()?;
        if peers.is_empty() || peers.len() > 4 { return Err(MiniMaxError::Invalid("encoder peer count").into()); }
        let hash = crate::shared::vision::worker::parse_plan_hash(hash.context("encoder plan hash required")?)?;
        let revision = revision.filter(|s| !s.is_empty()).context("encoder revision required")?.to_owned();
        Ok(Some(Self { peers, hash, revision }))
    }
}
impl ReadyVision {
    pub fn local_bytes(snapshot: &Path, native: &Path) -> Result<u64> {
        let spec = TowerSpec::minimax(snapshot, ENCODER_TOKENS)?;
        Ok(cuteafd_ffi::vision::NativeVision::required(native, &spec.native)?.total_bytes())
    }
    pub fn load(snapshot: &Path, native: &Path, library: &cuteafd_ffi::NativeLibrary, device: i32,
        mode: MediaMode, prefix: &PrefixArgs, requested_cache: Option<u64>, remote: Option<RemoteVision>) -> Result<(Option<Self>, PrefixArgs)> {
        if mode == MediaMode::Off {
            if remote.is_some() { return Err(MiniMaxError::Invalid("vision peers with vision off").into()); }
            return Ok((None, prefix.clone()));
        }
        if let Some(remote) = remote {
            if !matches!(mode, MediaMode::Auto | MediaMode::Spark(_)) {
                return Err(MiniMaxError::Invalid("remote encoder placement").into());
            }
            let spec = TowerSpec::minimax(snapshot, ENCODER_TOKENS)?;
            let spark_reserve_bytes = spec.native.weight_bytes.checked_add(762_383_616 + (1 << 30))
                .context("MiniMax remote vision reservation overflow")?;
            let processor = ProcessorConfig::from_snapshot(snapshot, ImageFamily::Minimax)?;
            let (prefix, cache_bytes) = prefix.with_media_headroom(requested_cache)?;
            let id = spec.encoder_id(&remote.revision, 121);
            let preparer = Arc::new(MediaPreparer::for_loaded_encoder(processor.clone(), id, 4)?);
            let expected = crate::shared::vision::remote::EncoderHandshake { encoder_id: id,
                max_patches: ENCODER_TOKENS as u32 * processor.merge.pow(2), output_width: 6144,
                patch_size: processor.patch, merge_size: processor.merge, plan_hash: remote.hash };
            let encoder = Encoder::Remote(crate::shared::vision::remote::RemoteEncoder::connect_minimax(remote.peers,
                expected, std::time::Duration::from_secs(60))?);
            return Ok((Some(Self { encoder, preparer, cache_bytes, spark_reserve_bytes }), prefix));
        }
        if !matches!(mode, MediaMode::Auto | MediaMode::Rtx(None))
            && mode != MediaMode::Rtx(Some(device as usize)) {
            return Err(MiniMaxError::Unsupported("vision must use the admitted coordinator GPU").into());
        }
        let spec = TowerSpec::minimax(snapshot, ENCODER_TOKENS)?;
        let processor = ProcessorConfig::from_snapshot(snapshot, ImageFamily::Minimax)?;
        let (prefix, cache_bytes) = prefix.with_media_headroom(requested_cache)?;
        let revision = snapshot.file_name().and_then(|n| n.to_str()).context("MiniMax snapshot revision")?;
        let info = library.cuda_device_info(device)?;
        let sm = (info.compute_capability_major * 10 + info.compute_capability_minor) as u32;
        let preparer = Arc::new(MediaPreparer::for_loaded_encoder(processor.clone(), spec.encoder_id(revision, sm), 4)?);
        let ledger = cuteafd_ffi::vision::NativeVision::required(native, &spec.native)?;
        library.cuda_set_device(device)?;
        let (free, total) = library.cuda_memory_info()?;
        cuteafd_core::serving_capacity::admit_device_reservations(95,
            cuteafd_core::serving_capacity::DeviceMemory { device: device as u32,
                total_bytes: total as u64, baseline_free_bytes: free as u64 },
            &[cuteafd_core::serving_capacity::MemoryReservation {
                name: "minimax.vision".into(), bytes: ledger.total_bytes() }])?;
        let service = EncoderService::start(spec, native.to_path_buf(), device, ledger.total_bytes())?;
        let encoder = Encoder::Local(crate::shared::vision::local::LocalEncoder::new(service, &processor, 6144, ENCODER_TOKENS));
        Ok((Some(Self { encoder, preparer, cache_bytes, spark_reserve_bytes: 0 }), prefix))
    }
}

pub(super) fn local_auto_mode(available: u64, text_fixed: u64, vision: u64, cache_target: u64) -> Result<MediaMode> {
    let needed = text_fixed.checked_add(vision).and_then(|v| v.checked_add(cache_target))
        .context("MiniMax automatic vision admission overflow")?;
    Ok(if needed <= available { MediaMode::Rtx(None) } else { MediaMode::Off })
}

pub(super) fn prepare(job: NativeRequest, tokens: Vec<u32>, expander: Option<&SpanExpander>,
    vocabulary: usize, max_context: usize) -> Result<(Prompt, RequestMedia, Vec<EncodeJob>)> {
    if tokens.is_empty() || tokens.len() > max_context || tokens.iter().any(|&t| t as usize >= vocabulary) {
        return Err(MiniMaxError::Invalid("prompt token extent").into());
    }
    if !job.audio.is_empty() {
        return Err(MiniMaxError::Unsupported("audio input is not admitted").into());
    }
    let (tokens, mut spans) = if let Some(expander) = expander {
        if job.probe.as_ref().is_some_and(|p| p.spec.prompt_ids.is_some()) && (!job.media.is_empty() || !job.videos.is_empty()) {
            return Err(MiniMaxError::Unsupported("expanded media probe overrides").into());
        }
        let images = job.media.iter().map(|image| image.as_ref().clone()).collect::<Vec<_>>();
        let expanded = expander.expand(&tokens, &images, max_context)?;
        (expanded.tokens, expanded.media)
    } else {
        if !job.media.is_empty() || !job.videos.is_empty() || tokens.contains(&cuteafd_loader::media::minimax::IMAGE_TOKEN) || tokens.contains(&cuteafd_loader::media::minimax::VIDEO_TOKEN) {
            return Err(MiniMaxError::Unsupported("image input requires the bundled encoder").into());
        }
        (tokens, Vec::new())
    };
    let mut layout = spans.drain(..).map(|span| (span, 0, span.len)).collect::<Vec<_>>();
    let mut cursor = 0;
    for segment in &job.videos {
        let markers = expander.context("video requires the bundled encoder")?;
        if segment.grid.t == 0 || segment.grid.t as usize > MAX_SEGMENT_FRAMES / 2
            || segment.grid.h == 0 || segment.grid.w == 0
            || segment.grid.h % 2 != 0 || segment.grid.w % 2 != 0 {
            return Err(MiniMaxError::Invalid("video segment grid").into());
        }
        let rows = (segment.grid.h as usize).checked_mul(segment.grid.w as usize)
            .context("video row extent overflow")? / 4;
        if rows.checked_mul(segment.grid.t as usize) != Some(segment.tokens) {
            return Err(MiniMaxError::Invalid("video segment token count").into());
        }
        for pair in 0..segment.grid.t as usize {
            let start = tokens[cursor..].iter().position(|&t| t == cuteafd_loader::media::minimax::VIDEO_TOKEN)
                .map(|n| n + cursor).context("missing video placeholder span")?;
            let end = start.checked_add(rows).context("video span overflow")?;
            if start == 0 || tokens.get(start - 1) != Some(&markers.start)
                || tokens.get(end) != Some(&markers.end)
                || tokens.get(start..end).is_none_or(|span| span.iter().any(|&t| t != cuteafd_loader::media::minimax::VIDEO_TOKEN)) {
                return Err(MiniMaxError::Invalid("video placeholder span differs from prepared segment").into());
            }
            layout.push((cuteafd_core::MediaSpan { start, len: rows, key: segment.key.into() }, pair * rows, segment.tokens));
            cursor = end;
        }
    }
    if tokens[cursor..].contains(&cuteafd_loader::media::minimax::VIDEO_TOKEN) {
        return Err(MiniMaxError::Invalid("unowned video placeholder").into());
    }
    layout.sort_by_key(|(span, _, _)| span.start);
    let spans = layout.iter().map(|&(span, _, _)| span).collect::<Vec<_>>();
    let offsets = layout.iter().map(|&(_, offset, _)| offset).collect::<Vec<_>>();
    let rows = layout.iter().map(|&(_, _, rows)| rows).collect();
    let media = RequestMedia::new_with_layout(spans.clone(), 6144, tokens.len(), offsets.clone(), rows)?;
    let keys = MediaKeys::new_with_offsets(&tokens, vocabulary as u32, &spans, &offsets)?;
    let jobs = job.media.iter().map(|image| EncodeJob::image(image.key,
        [image.grid.t, image.grid.h, image.grid.w], image.rgb8.clone(), image.tokens, 6144))
        .chain(job.videos.iter().map(|segment| EncodeJob::video(segment.key,
            [segment.grid.t, segment.grid.h, segment.grid.w], segment.rgb8.clone(), segment.tokens, 6144))).collect();
    Ok((Prompt { job, tokens, keys }, media, jobs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::media::{ImageGrid, ImageKey, PreparedImage, minimax::{IMAGE_TOKEN, VIDEO_TOKEN}};
    fn request(images: Vec<Arc<PreparedImage>>) -> NativeRequest {
        NativeRequest { prefix_cache_source: Default::default(), literal_spans: Vec::new(),
            prompt: String::new(), constraint: None, images: Vec::new(), media: images, videos: Vec::new(), audio: Vec::new(),
            max_tokens: 8, sampling: Default::default(), stop_token_ids: Vec::new(),
            events: tokio::sync::mpsc::unbounded_channel().0, usage: None, probe: None }
    }
    #[test]
    fn minimax_auto_vision_preserves_text_and_requested_cache() -> Result<()> {
        assert_eq!(local_auto_mode(1000, 600, 100, 300)?, MediaMode::Rtx(None));
        assert_eq!(local_auto_mode(999, 600, 100, 300)?, MediaMode::Off);
        assert_eq!(local_auto_mode(1000, 600, 100, 301)?, MediaMode::Off);
        assert!(local_auto_mode(u64::MAX, u64::MAX, 1, 1).is_err());
        Ok(())
    }
    #[test]
    fn minimax_media_refuses_unadmitted_tokens_and_encoder_identity() {
        for tokens in [vec![], vec![200960], vec![IMAGE_TOKEN], vec![VIDEO_TOKEN]] {
            assert!(prepare(request(Vec::new()), tokens, None, 200960, 64).is_err());
        }
        assert!(prepare(request(Vec::new()), vec![1,2], None, 200960, 1).is_err());
        assert!(RemoteVision::parse(None, Some("aa"), None).is_err());
        assert!(RemoteVision::parse(Some("127.0.0.1:1"), None, None).is_err());
    }
    #[test]
    fn minimax_image_features_follow_expanded_rows_across_chunks() -> Result<()> {
        let image = Arc::new(PreparedImage { key: ImageKey([7;32]), grid: ImageGrid {t:1,h:4,w:4},
            rgb8: Arc::from(vec![0; 56*56*3]), tokens: 4 });
        let expander = SpanExpander::from_config(&serde_json::json!({"image_token_id":IMAGE_TOKEN,
            "vision_start_token_id":200027,"vision_end_token_id":200028}), 200960)?;
        let (prompt, mut media, jobs) = prepare(request(vec![image.clone()]),
            vec![1,200027,IMAGE_TOKEN,200028,2], Some(&expander), 200960, 64)?;
        assert_eq!(prompt.tokens, vec![1,200027,IMAGE_TOKEN,IMAGE_TOKEN,IMAGE_TOKEN,IMAGE_TOKEN,200028,2]);
        assert_eq!(jobs.len(),1);
        assert!(!media.ready(2,6));
        let mut cache = cuteafd_engine::media::EmbeddingCache::new(4*6144*2);
        let reserved = cache.reserve(image.key,4*6144*2)?;
        let data: Arc<[u8]> = (0..4).flat_map(|row| vec![row as u8+1;6144*2]).collect::<Vec<_>>().into();
        media.attach(cache.complete(image.key,data.clone())?)?;
        drop(reserved);
        for (start,end,indices,bytes) in [(0,3,vec![2],0..6144*2),
            (3,5,vec![0,1],6144*2..6144*6), (5,8,vec![0],6144*6..6144*8)] {
            let mut chunk = cuteafd_engine::media::MediaChunk::default();
            media.write_chunk(start,end,&mut chunk)?;
            assert_eq!(chunk.indices,indices);
            assert_eq!(chunk.features,&data[bytes]);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "media/tests.rs"]
mod integration_tests;
