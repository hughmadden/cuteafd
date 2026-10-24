//! Bundled MiniMax processors at release revision 54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e.
//! The executable processors use smart resize, not the legacy grid-pinpoint fields.
use super::{invalid, resize, EncoderId, ImageFamily, ImageGrid, ImageKey, MediaError, ProcessorConfig, Result, RgbImage, SpanExpander};
use sha2::{Digest, Sha256};
use std::{path::Path, sync::Arc};

pub const IMAGE_TOKEN: u32 = 200_025;
pub const VIDEO_TOKEN: u32 = 200_026;
/// sglang `_apply_max_frames_limit` splits `grid_t` into runs of `vision_segment_max_frames`
/// (4) temporal patches, each its own attention segment with positions 0..t: 8 frames.
pub const MAX_SEGMENT_FRAMES: usize = 8;
/// Token capacity of the loaded tower (16,384 patches of scratch); one video segment
/// must fit it whatever the image detail cap is.
pub const ENCODER_TOKENS: usize = 4096;
/// Compatible framework source for inherited sampling; the checkpoint does not pin it.
pub const SAMPLING_REFERENCE: (&str, &str) = (
    "transformers/v4.57.1/src/transformers/video_processing_utils.py",
    "067dc959fe800d5a5a5ab39c54b53817a4b59424f31436767d7361c20b3ba409",
);
/// MiniMax's serving processor, the source of the default video sampling and sizing.
pub const SERVING_REFERENCE: (&str, &str) = (
    "sglang/python/sglang/srt/multimodal/processors/minimax_m3_vl.py",
    "8e1f9b30d5cc845ec8c016719ebd68921b50dfe4b30e7d514d97ff4967079499",
);
/// The serving default rate; its frame cap is `isqrt(max_pixels)` floored to 28.
pub const VIDEO_FPS: f64 = 1.0;
const VIDEO_SIDE: u32 = 756;
pub const PROCESSOR_SOURCES: [(&str, &str); 3] = [
    ("image_processor.py", "8218f98634c365608a98ea887bfebac37131ec2acdb8445188350490de7cf994"),
    ("video_processor.py", "ced0f8d85d73f8722fecb26033d2d91ecf4a4e5112c91ce1b1522ca7edd0943d"),
    ("processing_minimax.py", "0706ae8b93b3489a3b3bc7f48a5f29fb34b1449ebebb37586250e4d5fa8f5e35"),
];

pub(super) fn validate_config(config: &ProcessorConfig, value: &serde_json::Value) -> Result<()> {
    if (config.patch, config.merge, config.temporal) != (14, 2, 2) {
        return Err(MediaError::Config("MiniMax processor requires patch/merge/temporal 14/2/2".into()));
    }
    for (key, expected) in [("spatial_merge_size", 2), ("temporal_patch_size", 2)] {
        if value["img_token_compression_config"].get(key).is_some_and(|v| v.as_u64() != Some(expected)) {
            return Err(MediaError::Config(format!("MiniMax {key} differs from bundled processor")));
        }
    }
    Ok(())
}

pub(super) fn resize_shape(height: u32, width: u32, factor: u32, min: u64, max: u64) -> Result<(u32, u32)> {
    let (h, w, f) = (height as f64, width as f64, factor as f64);
    if h == 0.0 || w == 0.0 || h.max(w) / h.min(w) > 200.0 {
        return Err(invalid("dimensions must be positive with aspect ratio at most 200"));
    }
    let (mut rh, mut rw) = ((h / f).round_ties_even().max(1.0) * f, (w / f).round_ties_even().max(1.0) * f);
    if rh * rw > max as f64 {
        let beta = (h * w / max as f64).sqrt();
        rh = (h / beta / f).floor() * f;
        rw = (w / beta / f).floor() * f;
    } else if rh * rw < min as f64 {
        let beta = (min as f64 / (h * w)).sqrt();
        rh = (h * beta / f).ceil() * f;
        rw = (w * beta / f).ceil() * f;
    }
    if rh < f || rw < f { return Err(invalid("resize budget cannot hold a merged patch")); }
    Ok((rh as u32, rw as u32))
}

/// Video frame sizing: the serving default, the bundled processor's smart resize
/// (`do_resize`) or none (`do_resize: false`; dimensions must already align to 28).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoResize { Vendor, Smart, Off }
impl VideoResize {
    pub fn shape(self, height: u32, width: u32) -> Result<(u32, u32)> {
        match self {
            // sglang `get_hw_multiple_of` with an integer cap: scale the longer side down
            // to 756, round each side half to even, then up to 28; smaller frames only align.
            Self::Vendor => {
                if height == 0 || width == 0 { return Err(invalid("video dimensions must be positive")); }
                let longer = height.max(width);
                let ratio = if longer > VIDEO_SIDE { f64::from(VIDEO_SIDE) / f64::from(longer) } else { 1.0 };
                let side = |n: u32| ((f64::from(n) * ratio).round_ties_even() as u32).div_ceil(28) * 28;
                let (h, w) = (side(height), side(width));
                if h == 0 || w == 0 { return Err(invalid("video aspect ratio leaves no merged patch")); }
                Ok((h, w))
            }
            Self::Smart => resize_shape(height, width, 28, 3136, 602_112),
            Self::Off => Ok((height, width)),
        }
    }
}

/// Resolve image delimiters from the bundled tokenizer, as MiniMaxVLProcessor does.
pub fn span_expander(snapshot: &Path, vocabulary: u32, video: bool) -> Result<SpanExpander> {
    let bytes = std::fs::read(snapshot.join("tokenizer_config.json")).map_err(|e| MediaError::Config(e.to_string()))?;
    let config: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| MediaError::Config(e.to_string()))?;
    let entries = config["added_tokens_decoder"].as_object().ok_or_else(|| MediaError::Config("tokenizer lacks added_tokens_decoder".into()))?;
    let id = |token: &str| -> Result<u32> {
        entries.iter().find(|(_, value)| value["content"] == token)
            .and_then(|(key, _)| key.parse().ok()).ok_or_else(|| MediaError::Config(format!("tokenizer lacks {token}")))
    };
    let placeholder = id(if video { "]<]video[>[" } else { "]<]image[>[" })?;
    if placeholder != if video { VIDEO_TOKEN } else { IMAGE_TOKEN } { return Err(MediaError::Config("MiniMax media token differs from release".into())); }
    SpanExpander::from_config(&serde_json::json!({
        "image_token_id": placeholder,
        "vision_start_token_id": id("]<]start of image[>[")?,
        "vision_end_token_id": id("]<]end of image[>[")?,
    }), vocabulary)
}

/// A bounded tower invocation. RGB frames are contiguous HWC, with the final
/// frame repeated for an odd frame count; `grid.t` counts temporal patch pairs.
#[derive(Debug, Clone)]
pub struct PreparedVideoSegment {
    pub key: ImageKey,
    pub grid: ImageGrid,
    pub rgb8: Arc<[u8]>,
    pub tokens: usize,
    pub source_frames: usize,
}

/// Explicit `do_sample_frames` sampling inherited from Transformers 4.57.1
/// BaseVideoProcessor. With sampling disabled every frame is retained.
/// The bundled `min_frames`/`max_frames` fields are not consulted by this method.
pub fn sample_frame_indices(total: usize, source_fps: Option<f64>, sample: bool,
    num_frames: Option<usize>, fps: Option<f64>) -> Result<Vec<usize>> {
    if total == 0 || total > i32::MAX as usize { return Err(invalid("invalid video frame count")); }
    if !sample { return Ok((0..total).collect()); }
    if num_frames.is_some() && fps.is_some() { return Err(invalid("num_frames and fps are mutually exclusive")); }
    let count = if let Some(count) = num_frames { count } else {
        let source = source_fps.filter(|n| n.is_finite() && *n > 0.0).ok_or_else(|| invalid("fps sampling requires source fps"))?;
        let target = fps.unwrap_or(1.0);
        if !target.is_finite() || target <= 0.0 { return Err(invalid("sampling fps must be positive")); }
        (total as f64 / source * target) as usize
    };
    if count == 0 || count > total { return Err(invalid("sample count must be positive and at most the source frame count")); }
    let step = total as f64 / count as f64;
    // torch.arange's floating endpoint can produce one extra sample when
    // ceil(total / step) exceeds count because of division rounding.
    let len = (total as f64 / step).ceil() as usize;
    let indices: Vec<_> = (0..len).map(|index| (index as f64 * step) as f32 as usize).collect();
    if indices.iter().any(|index| *index >= total) {
        return Err(invalid("uniform sampling rounded an index outside the video"));
    }
    Ok(indices)
}

/// The serving sampler (sglang `_compute_sampled_frame_indices`, the training
/// extractor's constant mode): frames at least `1/fps` apart from the first, then
/// always the last. The serving path applies no frame cap by default.
pub fn constant_interval_indices(total: usize, source_fps: Option<f64>, fps: f64) -> Result<Vec<usize>> {
    if total == 0 || total > i32::MAX as usize { return Err(invalid("invalid video frame count")); }
    let source = source_fps.filter(|n| n.is_finite() && *n > 0.0).ok_or_else(|| invalid("fps sampling requires source fps"))?;
    if !fps.is_finite() || fps <= 0.0 { return Err(invalid("sampling fps must be positive")); }
    const EPS: f64 = 1e-4;
    let (mut indices, mut kept) = (vec![0usize], 0.0f64);
    loop {
        let next = (((kept + 1.0 / fps - EPS) * source).ceil() as usize).max(indices[indices.len() - 1] + 1);
        if next >= total { break; }
        indices.push(next);
        kept = next as f64 / source;
    }
    if indices[indices.len() - 1] != total - 1 && (total - 1) as f64 / source - kept > EPS { indices.push(total - 1); }
    Ok(indices)
}

/// Input frames are already decoded and sampled; `sizing` picks their size (see
/// [`VideoResize`]). Odd segments repeat their last frame, as the processor pads.
pub fn prepare_video_frames(frames: &[RgbImage], encoder: EncoderId, sizing: VideoResize) -> Result<Vec<PreparedVideoSegment>> {
    let first = frames.first().ok_or_else(|| invalid("video contains no frames"))?;
    if frames.len() > 768 { return Err(invalid("video exceeds 768 frames")); }
    if first.width == 0 || first.height == 0 || u64::from(first.width) * u64::from(first.height) > 64 << 20 {
        return Err(invalid("video dimensions must be positive and at most 64 megapixels"));
    }
    if frames.iter().any(|frame| frame.width != first.width || frame.height != first.height
        || frame.data.len() != frame.width as usize * frame.height as usize * 3) {
        return Err(invalid("video frames must share dimensions and complete RGB extents"));
    }
    let (height, width) = sizing.shape(first.height, first.width)?;
    if height % 28 != 0 || width % 28 != 0 { return Err(invalid("unresized video dimensions must be multiples of 28")); }
    let frame_tokens = height as usize / 28 * (width as usize / 28);
    if frame_tokens.checked_mul(frames.len().div_ceil(2)).is_none_or(|n| n > 57_600) {
        return Err(invalid("video exceeds total token budget"));
    }
    let config = ProcessorConfig::for_family(ImageFamily::Minimax);
    let mut segments = Vec::new();
    for frames in frames.chunks(MAX_SEGMENT_FRAMES) {
        let grid = ImageGrid { t: frames.len().div_ceil(2) as u32, h: height / 14, w: width / 14 };
        let mut rgb = Vec::with_capacity(grid.t as usize * 2 * height as usize * width as usize * 3);
        for index in 0..grid.t as usize * 2 {
            let frame = &frames[index.min(frames.len() - 1)];
            if (frame.height, frame.width) != (height, width) { rgb.extend(resize::resize_bicubic_torch(frame, width, height).data); }
            else { rgb.extend_from_slice(&frame.data); }
        }
        let mut hash = Sha256::new();
        hash.update(b"cuteafd-minimax-video-v1");
        hash.update(encoder.0); hash.update(config.id().0);
        for dim in [grid.t, grid.h, grid.w] { hash.update(dim.to_le_bytes()); }
        hash.update(&rgb);
        segments.push(PreparedVideoSegment { key: ImageKey(hash.finalize().into()), grid,
            rgb8: rgb.into(), tokens: grid.t as usize * frame_tokens, source_frames: frames.len() });
    }
    Ok(segments)
}

impl PreparedVideoSegment {
    /// Expand one video's marker before tokenization, matching MiniMaxVLProcessor.
    /// Each temporal pair has its own image delimiters. `frame_indices` contains
    /// source indices for this segment, so timestamps retain the video's timeline.
    pub fn placeholder(&self, fps: Option<f64>, frame_indices: Option<&[usize]>) -> Result<String> {
        let (t,h,w) = (self.grid.t as usize,self.grid.h as usize,self.grid.w as usize);
        let frame_tokens = h.checked_mul(w).ok_or_else(|| invalid("video grid overflows"))? / 4;
        if t == 0 || t > MAX_SEGMENT_FRAMES / 2 || h == 0 || w == 0 || h % 2 != 0 || w % 2 != 0
            || frame_tokens.checked_mul(t) != Some(self.tokens) || self.tokens > 57_600 {
            return Err(invalid("video placeholder grid and token count differ"));
        }
        let timestamps = if let (Some(fps),Some(indices)) = (fps,frame_indices) {
            if !fps.is_finite() || fps <= 0.0 || indices.is_empty() { return Err(invalid("invalid video timestamp metadata")); }
            Some((fps,indices))
        } else { None };
        let mut text = String::new();
        for frame in 0..t {
            if let Some((fps,indices)) = timestamps {
                let timestamp = indices[(frame * 2).min(indices.len()-1)] as f64 / fps;
                text.push_str(&format!("]<]{timestamp:.1} seconds[>["));
            }
            text.push_str("]<]start of image[>[");
            for _ in 0..frame_tokens { text.push_str("]<]video[>["); }
            text.push_str("]<]end of image[>[");
        }
        Ok(text)
    }

    /// `[t, block_h, block_w, inner_h, inner_w, channel, temporal, py, px]`.
    pub fn patches(&self) -> Result<Vec<f32>> {
        let (t, h, w) = (self.grid.t as usize, self.grid.h as usize, self.grid.w as usize);
        if t == 0 || t > MAX_SEGMENT_FRAMES / 2 || h == 0 || w == 0 || h % 2 != 0 || w % 2 != 0
            || self.rgb8.len() != t * 2 * h * w * 14 * 14 * 3 {
            return Err(invalid("video patch grid and RGB byte extent differ"));
        }
        let lut = ProcessorConfig::for_family(ImageFamily::Minimax).normalization_lut();
        let mut patches = Vec::with_capacity(self.rgb8.len());
        for time in 0..t { for bh in 0..h/2 { for bw in 0..w/2 { for mh in 0..2 { for mw in 0..2 {
            for (channel, table) in lut.iter().enumerate() { for frame in 0..2 { for py in 0..14 { for px in 0..14 {
                let y = (bh * 2 + mh) * 14 + py;
                let x = (bw * 2 + mw) * 14 + px;
                let offset = (((time * 2 + frame) * h * 14 + y) * w * 14 + x) * 3 + channel;
                patches.push(table[self.rgb8[offset] as usize]);
            }}}}
        }}}}}
        Ok(patches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::ImageProcessor;

    #[test]
    fn release_pinpoint_shapes_follow_smart_resize_without_tiling() {
        let processor = ProcessorConfig::for_family(ImageFamily::Minimax);
        let expected = [
            [(336,336),(336,672),(336,1008),(336,1344),(280,1484),(252,1624)],
            [(672,336),(672,672),(532,812),(448,924),(420,1036),(364,1148)],
            [(1008,336),(812,532),(672,672),(560,756),(504,840),(448,924)],
            [(1344,336),(924,448),(756,560),(672,672),(588,728),(532,812)],
            [(1484,280),(1036,420),(840,504),(728,588),(672,672),(588,728)],
            [(1624,252),(1148,364),(924,448),(812,532),(728,588),(672,672)],
        ];
        for h in 1..=6 { for w in 1..=6 {
            let (height, width) = processor.resize_shape(h * 336, w * 336).unwrap();
            assert_eq!((height,width),expected[h as usize-1][w as usize-1]);
            assert_eq!((height % 28, width % 28), (0, 0));
            assert!(height * width <= 451_584);
        }}
        for (h, w, expected) in [(672,672,(672,672)), (336,672,(336,672)),
            (1008,1008,(672,672)), (336,2016,(252,1624)), (1,1,(56,56)), (1,200,(28,196))] {
            assert_eq!(processor.resize_shape(h,w).unwrap(), expected);
        }
        assert!(processor.resize_shape(1,201).is_err());
    }

    #[test]
    fn merge_rows_repeat_temporal_samples_and_use_fused_normalization() {
        let config = ProcessorConfig::for_family(ImageFamily::Minimax);
        let mut rgb = RgbImage { width: 56, height: 56, data: vec![0; 56 * 56 * 3] };
        for y in 0..56 { for x in 0..56 { for c in 0..3 {
            rgb.data[(y*56+x)*3+c] = (y/14*4+x/14+c*30) as u8;
        }}}
        let prepared = config.prepare_rgb(rgb, EncoderId([0;32])).unwrap();
        assert_eq!(prepared.tokens, 4);
        let patches = prepared.patches(&config).unwrap();
        let lut = config.normalization_lut();
        for (row, patch) in [0usize,1,4,5,2,3,6,7,8,9,12,13,10,11,14,15].into_iter().enumerate() {
            for c in 0..3 { for t in 0..2 {
                let start = row*1176+c*392+t*196;
                assert!(patches[start..start+196].iter().all(|v| *v == lut[c][patch+c*30]));
            }}
        }
        assert_eq!(lut[0][17], (17.0f32 - 0.48145466f32*255.0)/(0.26862954f32*255.0));
    }

    #[test]
    fn tokenizer_delimiters_and_index_alias_expand_image_rows() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tokenizer_config.json"), serde_json::to_vec(&serde_json::json!({
            "added_tokens_decoder": {
                "200025":{"content":"]<]image[>["}, "200026":{"content":"]<]video[>["},
                "200029":{"content":"]<]start of image[>["}, "200030":{"content":"]<]end of image[>["}
            }
        })).unwrap()).unwrap();
        let image = ProcessorConfig::for_family(ImageFamily::Minimax).prepare_rgb(
            RgbImage {width:56,height:56,data:vec![0;56*56*3]}, EncoderId([0;32])).unwrap();
        let expander = span_expander(dir.path(),200064,false).unwrap();
        let expanded = expander.expand(&[200029,IMAGE_TOKEN,200030], &[image], 6).unwrap();
        assert_eq!(expanded.tokens,[200029,IMAGE_TOKEN,IMAGE_TOKEN,IMAGE_TOKEN,IMAGE_TOKEN,200030]);
        assert_eq!((expanded.media[0].start,expanded.media[0].len),(1,4));
        let video = span_expander(dir.path(),200064,true).unwrap();
        assert_eq!((video.placeholder,video.start,video.end),(VIDEO_TOKEN,200029,200030));
        assert_eq!(SpanExpander::from_config(&serde_json::json!({"image_token_index":IMAGE_TOKEN,
            "vision_start_token_id":200029,"vision_end_token_id":200030}),200064).unwrap().placeholder, IMAGE_TOKEN);
    }

    #[test]
    fn legacy_tiling_metadata_does_not_override_fast_resize() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("preprocessor_config.json"), r#"{
            "process_image_mode":"dynamic_res","size":[672,672],"patch_size":14,
            "img_token_compression_config":{"spatial_merge_size":2,"temporal_patch_size":2,
                "max_image_resolution":1008,"image_token_compression_threshold":1.1}}
        "#).unwrap();
        let processor = ProcessorConfig::from_snapshot(dir.path(),ImageFamily::Minimax).unwrap();
        assert_eq!(processor.max_pixels,451_584);
        assert_eq!(processor.resize_shape(1008,1008).unwrap(),(672,672));
    }

    #[test]
    fn optional_video_sampling_uses_uniform_floor_indices() {
        assert_eq!(sample_frame_indices(10,None,false,None,None).unwrap(), (0..10).collect::<Vec<_>>());
        assert_eq!(sample_frame_indices(10,None,true,Some(4),None).unwrap(), [0,2,5,7]);
        assert_eq!(sample_frame_indices(90,Some(30.0),true,None,None).unwrap(), [0,30,60]);
        assert_eq!(sample_frame_indices(11,Some(2.5),true,None,Some(1.0)).unwrap(), [0,2,5,8]);
        assert!(sample_frame_indices(10,None,true,None,None).is_err());
        assert!(sample_frame_indices(10,None,true,Some(11),None).is_err());
        assert!(sample_frame_indices(15,None,true,Some(13),None).is_err());
        assert!(sample_frame_indices(10,Some(30.0),true,None,None).is_err());
        assert!(sample_frame_indices(10,Some(30.0),true,Some(4),Some(1.0)).is_err());
    }

    #[test]
    fn serving_sampler_keeps_constant_intervals_and_the_last_frame() {
        // Expectations evaluated with sglang `_compute_sampled_frame_indices` (SERVING_REFERENCE).
        let thirty = [0,30,60,90,120,150,180,210,240,270,299];
        assert_eq!(constant_interval_indices(300,Some(30.0),VIDEO_FPS).unwrap(), thirty);
        assert_eq!(constant_interval_indices(300,Some(30000.0/1001.0),VIDEO_FPS).unwrap(), thirty);
        assert_eq!(constant_interval_indices(18,Some(5.0),1.0).unwrap(), [0,5,10,15,17]);
        assert_eq!(constant_interval_indices(16,Some(5.0),1.0).unwrap(), [0,5,10,15]);
        assert_eq!(constant_interval_indices(90,Some(30.0),2.0).unwrap(), [0,15,30,45,60,75,89]);
        assert_eq!(constant_interval_indices(10,Some(4.0),8.0).unwrap(), (0..10).collect::<Vec<_>>());
        assert_eq!(constant_interval_indices(1,Some(30.0),1.0).unwrap(), [0]);
        assert!(constant_interval_indices(0,Some(30.0),1.0).is_err());
        assert!(constant_interval_indices(10,None,1.0).is_err());
        assert!(constant_interval_indices(10,Some(30.0),0.0).is_err());
    }

    #[test]
    fn serving_sizing_caps_the_longer_side_at_756_and_aligns_to_28() {
        // sglang `get_hw_multiple_of((w, h), 28, 756)`, written here as (height, width).
        for ((h, w), expected) in [((720,1280),(448,756)), ((1080,1920),(448,756)), ((480,640),(504,644)),
            ((1280,720),(756,448)), ((2160,3840),(448,756)), ((760,1000),(588,756)), ((28,28),(28,28)), ((1,1),(28,28))] {
            assert_eq!(VideoResize::Vendor.shape(h,w).unwrap(), expected, "{h}x{w}");
        }
        assert!(VideoResize::Vendor.shape(1,2000).is_err());
        assert_eq!(VideoResize::Smart.shape(720,1280).unwrap(), (560,1008));
        assert_eq!(VideoResize::Off.shape(720,1280).unwrap(), (720,1280));
        // 720p is 16 x 27 merged tokens per temporal patch; unresized it does not align.
        let frame = RgbImage { width:1280, height:720, data:vec![9;1280*720*3] };
        let segment = prepare_video_frames(&[frame.clone()],EncoderId([0;32]),VideoResize::Vendor).unwrap().remove(0);
        assert_eq!((segment.grid.h, segment.grid.w, segment.tokens), (32, 54, 432));
        assert!(segment.rgb8.iter().all(|&v| v == 9));
        assert!(prepare_video_frames(&[frame],EncoderId([0;32]),VideoResize::Off).is_err());
    }

    #[test]
    fn video_placeholder_wraps_each_temporal_pair_and_keeps_sampled_timestamps() {
        let frames = vec![RgbImage {width:56,height:28,data:vec![0;56*28*3]};3];
        let segment = prepare_video_frames(&frames,EncoderId([0;32]),VideoResize::Off).unwrap().remove(0);
        let pair = "]<]start of image[>[".to_owned()
            + "]<]video[>[" + "]<]video[>[" + "]<]end of image[>[";
        assert_eq!(segment.placeholder(None,None).unwrap(),pair.repeat(2));
        assert_eq!(segment.placeholder(Some(4.0),Some(&[1,8,15])).unwrap(),
            format!("]<]0.2 seconds[>[{pair}]<]3.8 seconds[>[{pair}"));
        assert_eq!(segment.placeholder(Some(2.0),Some(&[7])).unwrap(),
            format!("]<]3.5 seconds[>[{pair}]<]3.5 seconds[>[{pair}"));
        assert_eq!(segment.placeholder(None,Some(&[1,8,15])).unwrap(),pair.repeat(2));
        assert_eq!(segment.placeholder(Some(4.0),None).unwrap(),pair.repeat(2));
        assert!(segment.placeholder(Some(4.0),Some(&[])).is_err());
        assert!(segment.placeholder(Some(0.0),Some(&[0])).is_err());
        let mut invalid = segment.clone(); invalid.tokens += 1;
        assert!(invalid.placeholder(None,None).is_err());
    }

    #[test]
    fn video_keeps_all_frames_segments_eight_and_pads_last() {
        let frames: Vec<_> = (0..11).map(|v| RgbImage { width:28, height:28, data:vec![v;28*28*3] }).collect();
        let segments = prepare_video_frames(&frames, EncoderId([0;32]), VideoResize::Off).unwrap();
        assert_eq!(segments.len(),2);
        assert_eq!((segments[0].grid.t,segments[0].tokens,segments[0].source_frames),(4,4,8));
        assert_eq!((segments[1].grid.t,segments[1].tokens,segments[1].source_frames),(2,2,3));
        let frame = 28*28*3;
        assert_eq!(&segments[1].rgb8[2*frame..3*frame], &segments[1].rgb8[3*frame..]);
        let lut = ProcessorConfig::for_family(ImageFamily::Minimax).normalization_lut();
        let patches = segments[0].patches().unwrap();
        for time in 0..4 {
            assert_eq!(patches[time*4*1176],lut[0][time*2]); assert_eq!(patches[time*4*1176+196],lut[0][time*2+1]);
        }
        let mut wide = segments[0].clone();
        (wide.grid.t, wide.tokens, wide.rgb8) = (5, 5, vec![0; 10*frame].into());
        assert!(wide.patches().is_err() && wide.placeholder(None,None).is_err());
        assert!(prepare_video_frames(&[],EncoderId([0;32]),VideoResize::Off).is_err());
        assert!(prepare_video_frames(&[RgbImage {width:29,height:28,data:vec![0;29*28*3]}],EncoderId([0;32]),VideoResize::Off).is_err());
    }
}
