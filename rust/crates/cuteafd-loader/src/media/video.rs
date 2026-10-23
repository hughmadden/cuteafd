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
const FORMATS: &str = "mov,matroska,webm,avi,mpegts,ogg";

#[derive(Debug)]
pub struct DecodedVideo {
    pub frames: Vec<RgbImage>,
    pub fps: Option<f64>,
}

/// Decode every source frame without sampling, resizing, rotation or frame-rate conversion.
/// Sampling and the checkpoint's resizing/segment rules apply after this step.
pub fn decode_video(encoded: &[u8], max_decoded_bytes: usize) -> Result<DecodedVideo> {
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
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames", "-of", "json"])
        .arg(input.path());
    let metadata = run(&mut probe, 16 << 10, Duration::from_secs(10))?;
    let stream = probe_stream(&metadata, budget)?;
    let frame_bytes = stream.width as usize * stream.height as usize * 3;
    let limit = budget.min(frame_bytes.saturating_mul(MAX_FRAMES));
    let mut decoder = command("ffmpeg");
    decoder.args(["-nostdin", "-xerror", "-protocol_whitelist", "file",
        "-format_whitelist", FORMATS, "-hwaccel", "none", "-threads", "1",
        "-max_pixels", "67108864", "-noautorotate", "-i"])
        .arg(input.path())
        .args(["-map", "0:v:0", "-an", "-sn", "-dn", "-threads", "1",
            "-filter_threads", "1", "-filter_complex_threads", "1", "-fps_mode", "passthrough",
            "-frames:v", "769", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]);
    let bytes = run(&mut decoder, limit, Duration::from_secs(30))?;
    if bytes.is_empty() || bytes.len() % frame_bytes != 0 {
        return Err(invalid("video contains no complete RGB frames"));
    }
    let frames = bytes.chunks_exact(frame_bytes).map(|data| RgbImage {
        width: stream.width, height: stream.height, data: data.to_vec(),
    }).collect();
    Ok(DecodedVideo { frames, fps: frame_rate(&stream.avg_frame_rate) })
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
    nb_frames: Option<String>,
}
fn probe_stream(metadata: &[u8], budget: usize) -> Result<Stream> {
    let probe: Probe = serde_json::from_slice(metadata)
        .map_err(|_| invalid("video probe returned invalid metadata"))?;
    let stream = probe.streams.into_iter().next().ok_or_else(|| invalid("video has no video stream"))?;
    let pixels = u64::from(stream.width) * u64::from(stream.height);
    if pixels == 0 || pixels > MAX_PIXELS || pixels * 3 > budget as u64 {
        return Err(invalid("video dimensions exceed decoded byte or pixel limit"));
    }
    if let Some(frames) = stream.nb_frames.as_deref().and_then(|v| v.parse::<u64>().ok()) {
        if frames > MAX_FRAMES as u64 || pixels * 3 > (budget as u64).checked_div(frames).unwrap_or(0) {
            return Err(invalid("video exceeds source frame or decoded byte limit"));
        }
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

    #[test]
    fn lossless_video_preserves_frames_rgb_and_rate() {
        let (encoded, rgb) = lossless_video(5);
        let decoded = decode_video(&encoded, rgb.len()).unwrap();
        assert_eq!(decoded.fps, Some(5.0));
        assert_eq!(decoded.frames.len(), 5);
        assert!(decoded.frames.iter().all(|frame| (frame.width, frame.height) == (28, 28)));
        assert_eq!(decoded.frames.into_iter().flat_map(|frame| frame.data).collect::<Vec<_>>(), rgb);
    }
    #[test]
    fn rejects_decoded_byte_overflow_without_truncation() {
        let (encoded, rgb) = lossless_video(5);
        assert!(decode_video(&encoded, rgb.len() - 1).is_err());
    }
    #[test]
    fn rejects_source_frame_overflow_without_truncation() {
        let (encoded, _) = lossless_video(MAX_FRAMES + 1);
        assert!(decode_video(&encoded, 8 << 20).is_err());
    }
    #[test]
    fn rejects_invalid_input_and_missing_streams() {
        assert!(decode_video(&[], 1024).is_err());
        assert!(decode_video(b"invalid", 0).is_err());
        assert!(decode_video(b"invalid", 1024).is_err());
        assert!(probe_stream(br#"{"streams":[]}"#, 1024).is_err());
    }
    #[test]
    fn rejects_oversized_headers_before_decode() {
        for header in [
            br#"{"streams":[{"width":0,"height":1}]}"#.as_slice(),
            br#"{"streams":[{"width":10000,"height":10000}]}"#,
            br#"{"streams":[{"width":28,"height":28,"nb_frames":"769"}]}"#,
            br#"{"streams":[{"width":28,"height":28,"nb_frames":"0"}]}"#,
        ] {
            assert!(probe_stream(header, MAX_DECODED_BYTES).is_err());
        }
        assert!(probe_stream(br#"{"streams":[{"width":28,"height":28}]}"#, 1024).is_err());
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
