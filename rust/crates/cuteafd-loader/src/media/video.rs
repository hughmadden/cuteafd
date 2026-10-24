//! Bounded CPU video decoding through the optional ffmpeg/ffprobe executables.
use super::{invalid, MediaError, Result, RgbImage};
use serde::Deserialize;
use std::{
    io::{Read, Write},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

const MAX_FRAMES: usize = 768;
const MAX_PIXELS: u64 = 64 << 20;
const MAX_ENCODED_BYTES: usize = 64 << 20;
const MAX_DECODED_BYTES: usize = 512 << 20;
const MAX_PACKETS: usize = 1 << 18;
const FORMATS: &str = "mov,matroska,webm,avi,mpegts,ogg";

/// Stream facts read before decoding. `frames` counts the packets an edit list does
/// not discard: what the decoder emits and what torchcodec's exact scan indexes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoProbe {
    pub width: u32,
    pub height: u32,
    pub fps: Option<f64>,
    pub frames: usize,
}
/// Ascending source frame indices to keep, and the size they are scaled to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodePlan {
    pub indices: Vec<usize>,
    pub width: u32,
    pub height: u32,
}
#[derive(Debug)]
pub struct DecodedVideo {
    pub frames: Vec<RgbImage>,
    pub fps: Option<f64>,
    /// Source index of each decoded frame.
    pub indices: Vec<usize>,
}

/// Decode only the planned frames: ffmpeg selects them in decode order, converts
/// them to RGB and scales them, so the byte budget covers the frames the processor
/// keeps rather than the source. No rotation or frame-rate conversion.
pub fn decode_video(encoded: &[u8], max_decoded_bytes: usize,
    plan: impl FnOnce(&VideoProbe) -> Result<DecodePlan>) -> Result<DecodedVideo> {
    if encoded.is_empty() || encoded.len() > MAX_ENCODED_BYTES {
        return Err(invalid("encoded video must be 1 byte through 64 MiB"));
    }
    let budget = max_decoded_bytes.min(MAX_DECODED_BYTES);
    if budget == 0 { return Err(invalid("decoded video exceeds request byte limit")); }
    let mut input = tempfile::NamedTempFile::new().map_err(io_error)?;
    input.write_all(encoded).map_err(io_error)?;
    let mut probe = command("ffprobe");
    probe.args(["-protocol_whitelist", "file", "-format_whitelist", FORMATS,
        "-threads", "1", "-max_pixels", "67108864", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate", "-of", "json"])
        .arg(input.path());
    let stream = probe_stream(&run(&mut probe, 16 << 10, Duration::from_secs(10))?)?;
    let mut packets = command("ffprobe");
    packets.args(["-protocol_whitelist", "file", "-format_whitelist", FORMATS,
        "-threads", "1", "-max_pixels", "67108864", "-select_streams", "v:0",
        "-show_entries", "packet=flags", "-of", "csv=p=0"])
        .arg(input.path());
    let flags = run(&mut packets, MAX_PACKETS * 4, Duration::from_secs(10))?;
    let frames = flags.split(|&b| b == b'\n').filter(|line| !line.is_empty() && !line.contains(&b'D')).count();
    if frames == 0 { return Err(invalid("video stream has no frames")); }
    let probe = VideoProbe { width: stream.width, height: stream.height, fps: frame_rate(&stream.avg_frame_rate), frames };
    let DecodePlan { indices, width, height } = plan(&probe)?;
    let pixels = u64::from(width) * u64::from(height);
    if indices.is_empty() || indices.len() > MAX_FRAMES || indices.windows(2).any(|pair| pair[0] >= pair[1])
        || indices[indices.len() - 1] >= frames || pixels == 0 || pixels > MAX_PIXELS {
        return Err(invalid("video plan must keep 1 through 768 ordered stream frames at a valid size"));
    }
    let frame_bytes = pixels as usize * 3;
    let limit = frame_bytes.checked_mul(indices.len()).filter(|&bytes| bytes <= budget)
        .ok_or_else(|| invalid("sampled video exceeds decoded byte limit"))?;
    let select = indices.iter().map(|index| format!("eq(n,{index})")).collect::<Vec<_>>().join("+");
    let mut decoder = command("ffmpeg");
    decoder.args(["-nostdin", "-xerror", "-protocol_whitelist", "file",
        "-format_whitelist", FORMATS, "-hwaccel", "none", "-threads", "1",
        "-max_pixels", "67108864", "-noautorotate", "-i"])
        .arg(input.path())
        .args(["-map", "0:v:0", "-an", "-sn", "-dn", "-threads", "1",
            "-filter_threads", "1", "-filter_complex_threads", "1", "-fps_mode", "passthrough", "-vf"])
        // Convert before scaling, as processors resize decoded RGB; B=0, C=0.5 is the
        // Keys a=-0.5 cubic of torch's antialiased bicubic.
        .arg(format!("select='{select}',format=rgb24,scale={width}:{height}:flags=bicubic:param0=0:param1=0.5"))
        .arg("-frames:v").arg(indices.len().to_string())
        .args(["-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]);
    let bytes = run(&mut decoder, limit, Duration::from_secs(30))?;
    if bytes.len() != limit {
        return Err(invalid("video decoded fewer frames than its packet index"));
    }
    let frames = bytes.chunks_exact(frame_bytes).map(|data| RgbImage { width, height, data: data.to_vec() }).collect();
    Ok(DecodedVideo { frames, fps: probe.fps, indices })
}

#[derive(Deserialize)]
struct Probe {
    streams: Vec<Stream>,
}
#[derive(Deserialize)]
struct Stream {
    width: u32,
    height: u32,
    #[serde(default)]
    avg_frame_rate: String,
}
fn probe_stream(metadata: &[u8]) -> Result<Stream> {
    let probe: Probe = serde_json::from_slice(metadata)
        .map_err(|_| invalid("video probe returned invalid metadata"))?;
    let stream = probe.streams.into_iter().next().ok_or_else(|| invalid("video has no video stream"))?;
    let pixels = u64::from(stream.width) * u64::from(stream.height);
    if pixels == 0 || pixels > MAX_PIXELS {
        return Err(invalid("video dimensions exceed the pixel limit"));
    }
    Ok(stream)
}
fn frame_rate(value: &str) -> Option<f64> {
    let (numerator, denominator) = value.split_once('/')?;
    let fps = numerator.parse::<f64>().ok()? / denominator.parse::<f64>().ok()?;
    (fps.is_finite() && fps > 0.0).then_some(fps)
}
fn io_error(error: std::io::Error) -> MediaError {
    MediaError::Decode(format!("video decoder I/O: {error}"))
}
fn command(program: &str) -> Command {
    let mut command = Command::new(program);
    command.args(["-hide_banner", "-loglevel", "error", "-max_alloc", "268435456"])
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
    command
}

struct DecoderChild(Child);
impl Drop for DecoderChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn run(command: &mut Command, limit: usize, timeout: Duration) -> Result<Vec<u8>> {
    let mut child = DecoderChild(command.spawn().map_err(|error| MediaError::Decode(format!(
        "CPU video decoding requires ffmpeg and ffprobe: {error}")))?);
    let stdout = child.0.stdout.take().expect("piped decoder output");
    thread::scope(|scope| {
        let reader = scope.spawn(move || {
            let mut bytes = Vec::new();
            stdout.take(limit as u64 + 1).read_to_end(&mut bytes).map_err(io_error)?;
            if bytes.len() > limit { return Err(invalid("video decoder output exceeds byte or frame limit")); }
            Ok(bytes)
        });
        let start = Instant::now();
        let status = loop {
            match child.0.try_wait() {
                Ok(Some(status)) => break Ok(status),
                Ok(None) if start.elapsed() < timeout => thread::sleep(Duration::from_millis(10)),
                result => {
                    let _ = child.0.kill();
                    let _ = child.0.wait();
                    break Err(match result {
                        Err(error) => io_error(error),
                        _ => invalid("video decoder exceeded its time limit"),
                    });
                }
            }
        };
        let bytes = reader.join().map_err(|_| invalid("video decoder output reader failed"))??;
        if !status?.success() {
            return Err(MediaError::Decode("invalid or unsupported video stream".into()));
        }
        Ok(bytes)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{minimax::{constant_interval_indices, VideoResize}, resize::resize_bicubic_torch};

    fn lossless_video(frames: usize) -> (Vec<u8>, Vec<u8>) {
        let rgb: Vec<_> = (0..frames * 28 * 28 * 3).map(|i| (i % 251) as u8).collect();
        let mut input = tempfile::NamedTempFile::new().unwrap();
        input.write_all(&rgb).unwrap();
        let mut encoder = command("ffmpeg");
        encoder.args(["-nostdin", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "28x28",
            "-r", "5", "-threads", "1", "-i"])
            .arg(input.path()).args(["-threads", "1", "-filter_threads", "1", "-c:v", "ffv1",
                "-level", "3", "-pix_fmt", "bgr0", "-f", "matroska", "pipe:1"]);
        (run(&mut encoder, 8 << 20, Duration::from_secs(10)).unwrap(), rgb)
    }

    fn every(probe: &VideoProbe) -> Result<DecodePlan> {
        Ok(DecodePlan { indices: (0..probe.frames).collect(), width: probe.width, height: probe.height })
    }
    fn source_frames(rgb: &[u8], decoded: &DecodedVideo) {
        for (frame, index) in decoded.frames.iter().zip(&decoded.indices) {
            assert_eq!(frame.data, rgb[index * 28 * 28 * 3..(index + 1) * 28 * 28 * 3], "frame {index}");
        }
    }

    #[test]
    fn lossless_video_preserves_frames_rgb_and_rate() {
        let (encoded, rgb) = lossless_video(5);
        let decoded = decode_video(&encoded, rgb.len(), every).unwrap();
        assert_eq!(decoded.fps, Some(5.0));
        assert_eq!(decoded.indices, [0, 1, 2, 3, 4]);
        assert!(decoded.frames.iter().all(|frame| (frame.width, frame.height) == (28, 28)));
        assert_eq!(decoded.frames.into_iter().flat_map(|frame| frame.data).collect::<Vec<_>>(), rgb);
    }
    #[test]
    fn rejects_decoded_byte_overflow_without_truncation() {
        let (encoded, rgb) = lossless_video(5);
        assert!(decode_video(&encoded, rgb.len() - 1, every).is_err());
    }
    #[test]
    fn plans_keep_at_most_768_frames_of_longer_sources() {
        let (encoded, rgb) = lossless_video(MAX_FRAMES + 1);
        assert!(decode_video(&encoded, 8 << 20, every).is_err());
        let decoded = decode_video(&encoded, 8 << 20, |probe| {
            assert_eq!(probe.frames, MAX_FRAMES + 1);
            Ok(DecodePlan { indices: (0..probe.frames).step_by(2).collect(), width: probe.width, height: probe.height })
        }).unwrap();
        assert_eq!(decoded.frames.len(), MAX_FRAMES / 2 + 1);
        source_frames(&rgb, &decoded);
    }
    #[test]
    fn serving_plan_selects_exact_source_frames() {
        // 18 frames at 5 fps: the serving sampler keeps 0, 5, 10, 15 and the last.
        let (encoded, rgb) = lossless_video(18);
        let decoded = decode_video(&encoded, 1 << 20, |probe| Ok(DecodePlan {
            indices: constant_interval_indices(probe.frames, probe.fps, 1.0)?, width: probe.width, height: probe.height,
        })).unwrap();
        assert_eq!(decoded.indices, [0, 5, 10, 15, 17]);
        source_frames(&rgb, &decoded);
    }
    #[test]
    fn rejects_invalid_input_plans_and_missing_streams() {
        assert!(decode_video(&[], 1024, every).is_err());
        assert!(decode_video(b"invalid", 0, every).is_err());
        assert!(decode_video(b"invalid", 1024, every).is_err());
        assert!(probe_stream(br#"{"streams":[]}"#).is_err());
        let (encoded, _) = lossless_video(5);
        for (indices, width) in [(vec![], 28), (vec![3, 1], 28), (vec![1, 1], 28), (vec![5], 28), (vec![0], 0)] {
            let plan = DecodePlan { indices, width, height: 28 };
            assert!(decode_video(&encoded, 1 << 20, |_| Ok(plan.clone())).is_err(), "{plan:?}");
        }
    }
    #[test]
    fn rejects_oversized_headers_before_decode() {
        for header in [
            br#"{"streams":[{"width":0,"height":1}]}"#.as_slice(),
            br#"{"streams":[{"width":10000,"height":10000}]}"#,
        ] {
            assert!(probe_stream(header).is_err());
        }
        assert!(probe_stream(br#"{"streams":[{"width":28,"height":28}]}"#).is_ok());
    }
    #[test]
    fn samples_and_scales_a_720p_clip_inside_the_decoder() {
        // Decoded at source size, this 10 s 30 fps clip passes 64 MiB after 24 frames.
        let mut encoder = command("ffmpeg");
        encoder.args(["-nostdin", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30:duration=10",
            "-threads", "1", "-c:v", "mpeg4", "-q:v", "3", "-f", "matroska", "pipe:1"]);
        let encoded = run(&mut encoder, 32 << 20, Duration::from_secs(60)).unwrap();
        let decoded = decode_video(&encoded, 64 << 20, |probe| {
            assert_eq!((probe.width, probe.height, probe.fps, probe.frames), (1280, 720, Some(30.0), 300));
            let (height, width) = VideoResize::Vendor.shape(probe.height, probe.width)?;
            Ok(DecodePlan { indices: constant_interval_indices(probe.frames, probe.fps, 1.0)?, width, height })
        }).unwrap();
        assert_eq!(decoded.indices, [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 299]);
        assert!(decoded.frames.iter().all(|frame| (frame.width, frame.height) == (756, 448)));
        // ffmpeg's RGB bicubic tracks the processor's torch resize of the full-size frame.
        let full = decode_video(&encoded, 64 << 20, |probe| {
            Ok(DecodePlan { indices: vec![0], width: probe.width, height: probe.height })
        }).unwrap();
        let torch = resize_bicubic_torch(&full.frames[0], 756, 448);
        let difference = torch.data.iter().zip(&decoded.frames[0].data)
            .map(|(a, b)| u64::from(a.abs_diff(*b))).sum::<u64>() as f64 / torch.data.len() as f64;
        assert!(difference < 0.25, "mean absolute RGB difference {difference}");
    }
    #[test]
    fn validates_frame_rates() {
        assert_eq!(frame_rate("30000/1001"), Some(30000.0 / 1001.0));
        for value in ["0/0", "5/0", "-1/2", "nan/1", "invalid"] {
            assert_eq!(frame_rate(value), None);
        }
    }
    #[test]
    fn terminates_decoder_on_deadline() {
        let mut sleeper = Command::new("sleep");
        sleeper.arg("5").stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
        let start = Instant::now();
        assert!(run(&mut sleeper, 1024, Duration::from_millis(20)).is_err());
        assert!(start.elapsed() < Duration::from_secs(2));
    }
}
