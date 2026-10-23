//! MiniMax video inputs reuse the image fetch policy and bundled processor.
use super::MediaPreparer;
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{LiteralSpans, media::{decode_bounded, minimax::{self, PreparedVideoSegment}}};
use serde_json::{json, Value};
use std::{sync::Arc, time::{Duration, Instant}};

#[derive(Debug, Clone)]
pub struct VideoSource {
    url: Option<String>,
    frames: Vec<String>,
    source_fps: Option<f64>,
    sample: bool,
    count: Option<usize>,
    fps: Option<f64>,
    resize: bool,
}
#[derive(Debug)]
pub struct PreparedVideo {
    pub segments: Vec<Arc<PreparedVideoSegment>>,
    pub placeholder: String,
}

/// Replace admitted video parts after literal shielding; the recipe adapter only
/// sees text while the checkpoint template retains its structural video marker.
pub fn take_sources(body: &mut Value) -> Result<Vec<VideoSource>> {
    let mut sources = Vec::new();
    for message in body.get_mut("messages").and_then(Value::as_array_mut).into_iter().flatten() {
        for part in message.get_mut("content").and_then(Value::as_array_mut).into_iter().flatten() {
            if !matches!(part["type"].as_str(), Some("video" | "video_url" | "input_video")) { continue; }
            ensure!(sources.len() < 16, "at most 16 videos are supported including history");
            let value = crate::gateway::video::openai(part)?;
            let url = value["url"].as_str().map(str::to_owned);
            let frames = value["frames"].as_array().into_iter().flatten().map(|frame|
                frame.as_str().or_else(|| frame["url"].as_str()).or_else(|| frame["image_url"]["url"].as_str())
                    .expect("canonical frame URL").to_owned()).collect();
            sources.push(VideoSource { url, frames, source_fps: value["source_fps"].as_f64(),
                sample: value["do_sample_frames"].as_bool().unwrap_or(false),
                count: value["num_frames"].as_u64().map(|n| n as usize), fps: value["fps"].as_f64(),
                resize: value["do_resize"].as_bool().unwrap_or(false) });
            *part = json!({"type":"text","text":"]<]video[>["});
        }
    }
    Ok(sources)
}

impl MediaPreparer {
    pub fn prepare_videos(&self, sources: &[VideoSource]) -> Result<Vec<PreparedVideo>> {
        ensure!(self.config.family == cuteafd_loader::media::ImageFamily::Minimax, "video requires the MiniMax processor");
        ensure!(sources.len() <= 16, "too many videos");
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut remaining = self.limits.decoded_bytes;
        let mut encoded_remaining = self.limits.encoded_bytes;
        let mut total_tokens = 0usize;
        let mut prepared_bytes = 0usize;
        let mut prepared = Vec::new();
        for source in sources {
            let (frames, source_fps) = if let Some(url) = &source.url {
                let bytes = if url.starts_with("data:") {
                    super::super::images::data_url_bytes_for(url, encoded_remaining, "video/")?
                } else { self.source_bytes(url, deadline)? };
                ensure!(bytes.len() <= encoded_remaining, "encoded videos exceed request byte limit");
                encoded_remaining -= bytes.len();
                let video = cuteafd_loader::media::video::decode_video(&bytes, remaining)?;
                (video.frames, source.source_fps.or(video.fps))
            } else {
                let mut frames = Vec::new();
                let mut bytes_left = remaining;
                for url in &source.frames {
                    ensure!(Instant::now() < deadline, "video preparation deadline exceeded");
                    let bytes = self.source_bytes(url, deadline)?;
                    ensure!(bytes.len() <= encoded_remaining, "encoded videos exceed request byte limit");
                    encoded_remaining -= bytes.len();
                    let frame = decode_bounded(&bytes, self.config.decode, bytes_left)?;
                    bytes_left = bytes_left.checked_sub(frame.data.len()).context("decoded video exceeds request byte limit")?;
                    frames.push(frame);
                }
                (frames, source.source_fps)
            };
            let decoded = frames.iter().try_fold(0usize, |n, f| n.checked_add(f.data.len())).context("decoded video extent overflow")?;
            remaining = remaining.checked_sub(decoded).context("decoded videos exceed request byte limit")?;
            let indices = minimax::sample_frame_indices(frames.len(), source_fps, source.sample, source.count, source.fps)?;
            let frames = indices.iter().map(|&i| frames[i].clone()).collect::<Vec<_>>();
            ensure!(frames.windows(2).all(|pair| pair[0].width == pair[1].width && pair[0].height == pair[1].height),
                "video frames must share dimensions");
            let mut segments = Vec::new();
            let mut placeholder = String::new();
            let mut offset = 0;
            for frames in frames.chunks(minimax::MAX_SEGMENT_FRAMES) {
                let segment = minimax::prepare_video_frames(frames, self.encoder, source.resize)?
                    .pop().context("video segment is empty")?;
                prepared_bytes = prepared_bytes.checked_add(segment.rgb8.len()).context("prepared video extent overflow")?;
                ensure!(prepared_bytes <= self.limits.decoded_bytes, "prepared videos exceed request byte limit");
                ensure!(segment.tokens <= self.config.max_image_tokens, "video segment exceeds encoder token capacity");
                total_tokens = total_tokens.checked_add(segment.tokens).context("video token count overflow")?;
                ensure!(total_tokens <= 57_600, "videos exceed total token budget");
                placeholder.push_str(&segment.placeholder(source_fps, Some(&indices[offset..offset + segment.source_frames]))?);
                offset += segment.source_frames;
                segments.push(segment);
            }
            prepared.push(PreparedVideo { segments: segments.into_iter().map(Arc::new).collect(), placeholder });
            ensure!(Instant::now() < deadline, "video preparation deadline exceeded");
        }
        Ok(prepared)
    }
}

/// Only structural markers are expanded; literal byte ranges retain their identity.
pub fn expand_prompt(prompt: &str, literals: &[std::ops::Range<usize>], images: usize,
    videos: &[PreparedVideo]) -> Result<(String, LiteralSpans)> {
    let mut out = String::new();
    let mut spans = Vec::new();
    let (mut image, mut video, mut cursor) = (0, 0, 0);
    let mut expand = |text: &str, out: &mut String| -> Result<()> {
        let mut rest = text;
        while let Some(at) = rest.find("]<]") {
            out.push_str(&rest[..at]); rest = &rest[at..];
            if let Some(tail) = rest.strip_prefix("]<]image[>[") {
                ensure!(image < images, "image placeholder count differs from inputs");
                image += 1; out.push_str("]<]start of image[>[]<]image[>[]<]end of image[>["); rest = tail;
            } else if let Some(tail) = rest.strip_prefix("]<]video[>[") {
                let prepared = videos.get(video).context("video placeholder count differs from inputs")?;
                video += 1; out.push_str(&prepared.placeholder); rest = tail;
            } else { out.push_str("]<]"); rest = &rest[3..]; }
        }
        out.push_str(rest); Ok(())
    };
    for span in literals {
        ensure!(span.start >= cursor && span.end <= prompt.len() && span.start <= span.end
            && prompt.is_char_boundary(span.start) && prompt.is_char_boundary(span.end), "invalid literal span");
        expand(&prompt[cursor..span.start], &mut out)?;
        let start = out.len(); out.push_str(&prompt[span.clone()]); spans.push(start..out.len()); cursor = span.end;
    }
    expand(&prompt[cursor..], &mut out)?;
    ensure!(image == images && video == videos.len(), "media input count differs from rendered placeholders");
    Ok((out, spans))
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod url_tests;
