use super::{take_sources, MediaPreparer, PreparedVideo, VideoSource};
use crate::openai::media::ImageUrlFetch;
use base64::{engine::general_purpose::STANDARD, Engine};
use cuteafd_loader::media::{EncoderId, ImageFamily, ProcessorConfig};
use serde_json::json;
use std::{io::{Read, Write}, net::TcpListener, thread, time::{Duration, Instant}};

// FFV1 RGB, eight 28x28 constant frames (values 0..7), four frames per second.
const VIDEO: &[u8] = include_bytes!("../../fixtures/video-eight-frames.mkv");

fn preparer(fetch: ImageUrlFetch) -> MediaPreparer {
    MediaPreparer::new(ProcessorConfig::for_family(ImageFamily::Minimax), EncoderId([1; 32]), fetch, 1).unwrap()
}
fn source(url: &str) -> Vec<VideoSource> {
    let mut body = json!({"messages":[{"role":"user","content":[{"type":"video_url",
        "video_url":{"url":url,"do_sample_frames":true,"num_frames":5}}]}]});
    let sources = take_sources(&mut body).unwrap();
    assert_eq!(body["messages"][0]["content"][0], json!({"type":"text","text":"]<]video[>["}));
    sources
}
fn assert_sampled(prepared: &[PreparedVideo]) {
    assert_eq!(prepared.len(), 1);
    let segments = &prepared[0].segments;
    assert_eq!(segments.len(), 1);
    assert_eq!((segments[0].source_frames, segments[0].grid.t, segments[0].tokens), (5, 3, 3));
    let frames = segments.iter().flat_map(|segment| segment.rgb8.chunks_exact(28 * 28 * 3)).collect::<Vec<_>>();
    assert_eq!(frames.len(), 6);
    for (frame, value) in frames.into_iter().zip([0, 1, 3, 4, 6, 6]) {
        assert!(frame.iter().all(|&pixel| pixel == value), "sampling or RGB decoding changed source pixels");
    }
    let expected = ["0.0", "0.8", "1.5"].map(|time| format!(
        "]<]{time} seconds[>[]<]start of image[>[]<]video[>[]<]end of image[>[")).concat();
    assert_eq!(prepared[0].placeholder, expected);
}

#[test]
fn encoded_data_video_decodes_samples_segments_and_preserves_source_timestamps() {
    let url = format!("data:video/x-matroska;base64,{}", STANDARD.encode(VIDEO));
    let prepared = preparer(ImageUrlFetch::Off).prepare_videos(&source(&url)).unwrap();
    assert_sampled(&prepared);
}

#[test]
fn encoded_http_video_uses_fetch_policy_and_the_same_processor() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("http://{}/video", listener.local_addr().unwrap());
    let server = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < deadline =>
                    thread::sleep(Duration::from_millis(10)),
                Err(error) => panic!("video fixture connection failed or timed out: {error}"),
            }
        };
        stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        stream.set_write_timeout(Some(Duration::from_secs(2))).unwrap();
        let mut request = [0; 4096];
        assert!(stream.read(&mut request).unwrap() > 0);
        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: video/x-matroska\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", VIDEO.len()).unwrap();
        stream.write_all(VIDEO).unwrap();
    });
    assert!(preparer(ImageUrlFetch::Off).prepare_videos(&source(&url)).is_err());
    let result = preparer(ImageUrlFetch::Any).prepare_videos(&source(&url));
    server.join().unwrap();
    assert_sampled(&result.unwrap());
}

fn generated(source: &str) -> String {
    let output = std::process::Command::new("ffmpeg").args(["-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "lavfi", "-i", source, "-threads", "1", "-c:v", "mpeg4", "-q:v", "4", "-f", "matroska", "pipe:1"])
        .stdin(std::process::Stdio::null()).output().unwrap();
    assert!(output.status.success(), "ffmpeg lavfi fixture");
    format!("data:video/x-matroska;base64,{}", STANDARD.encode(output.stdout))
}
fn unset(url: String) -> Vec<VideoSource> {
    take_sources(&mut json!({"messages":[{"role":"user","content":[{"type":"video_url","video_url":url}]}]})).unwrap()
}

#[test]
fn defaults_follow_the_serving_sampler_frame_size_and_timestamps() {
    // sglang at 1 fps keeps frames 0, 4, 8 and the last (9) of a 2.5 s, 4 fps clip and
    // scales 1280x720, 1920x1080 and 640x480 to 756x448, 756x448 and 644x504.
    for (size, (height, width)) in [("1280x720", (448, 756)), ("1920x1080", (448, 756)), ("640x480", (504, 644))] {
        let source = unset(generated(&format!("color=c=gray:s={size}:r=4:d=2.5")));
        let prepared = preparer(ImageUrlFetch::Off).prepare_videos(&source).unwrap();
        let segment = &prepared[0].segments[0];
        assert_eq!((segment.grid.t, segment.grid.h * 14, segment.grid.w * 14, segment.source_frames), (2, height, width, 4));
        let pair = format!("]<]start of image[>[{}]<]end of image[>[", "]<]video[>[".repeat(height as usize / 28 * (width as usize / 28)));
        assert_eq!(prepared[0].placeholder, format!("]<]0.0 seconds[>[{pair}]<]2.0 seconds[>[{pair}"));
    }
}

#[test]
fn ten_second_720p_clip_fits_the_decoded_budget_by_default() {
    // At source size the 64 MiB budget held 24 frames (0.8 s); sampled and scaled
    // inside ffmpeg, 1 fps keeps 11 frames of 756x448.
    let source = unset(generated("testsrc=size=1280x720:rate=30:duration=10"));
    let prepared = preparer(ImageUrlFetch::Off).prepare_videos(&source).unwrap();
    assert_eq!(prepared[0].segments.iter().map(|s| (s.grid.t, s.source_frames, s.tokens)).collect::<Vec<_>>(),
        [(4, 8, 1728), (2, 3, 864)]);
    for time in ["0.0", "2.0", "4.0", "6.0", "8.0", "10.0"] {
        assert_eq!(prepared[0].placeholder.matches(&format!("]<]{time} seconds[>[")).count(), 1, "{time}");
    }
    assert_eq!(prepared[0].placeholder.matches("seconds").count(), 6);
}

#[test]
fn explicit_options_keep_every_frame_and_smart_resize_in_the_decoder() {
    let url = format!("data:video/x-matroska;base64,{}", STANDARD.encode(VIDEO));
    let mut body = json!({"messages":[{"role":"user","content":[{"type":"video_url","video_url":{
        "url":url,"do_sample_frames":false,"do_resize":true}}]}]});
    let prepared = preparer(ImageUrlFetch::Off).prepare_videos(&take_sources(&mut body).unwrap()).unwrap();
    let segment = &prepared[0].segments[0];
    // Smart resize lifts 28x28 to its 3,136-pixel floor, 56x56.
    assert_eq!((segment.grid.t, segment.grid.h, segment.grid.w, segment.tokens), (4, 4, 4, 16));
    for (value, frame) in segment.rgb8.chunks_exact(56 * 56 * 3).enumerate() {
        assert!(frame.iter().all(|&pixel| pixel == value as u8), "frame {value}");
    }
}

#[test]
fn segments_are_bounded_by_tower_capacity_not_the_image_cap() {
    let mut low = preparer(ImageUrlFetch::Off);
    low.config.max_image_tokens = 1;
    let url = format!("data:video/x-matroska;base64,{}", STANDARD.encode(VIDEO));
    assert!(low.prepare_videos(&source(&url)).is_ok());
    // Eight unresized 924x924 frames: one segment of 4 x 33 x 33 = 4,356 tokens.
    let mut body = json!({"messages":[{"role":"user","content":[{"type":"video_url","video_url":{
        "url":generated("color=c=gray:s=924x924:r=8:d=1"),"do_sample_frames":false,"do_resize":false}}]}]});
    let error = preparer(ImageUrlFetch::Off).prepare_videos(&take_sources(&mut body).unwrap()).unwrap_err();
    assert!(error.to_string().contains("token capacity"), "{error:#}");
}

#[test]
fn encoded_video_rejects_bad_payload_mime_and_request_byte_limits() {
    let encoded = STANDARD.encode(VIDEO);
    for url in [format!("data:image/png;base64,{encoded}"), "data:video/x-matroska;base64,aW52YWxpZA==".into()] {
        assert!(preparer(ImageUrlFetch::Off).prepare_videos(&source(&url)).is_err());
    }
    let url = format!("data:video/x-matroska;base64,{encoded}");
    let mut small = preparer(ImageUrlFetch::Off);
    small.limits.encoded_bytes = VIDEO.len() - 1;
    assert!(small.prepare_videos(&source(&url)).is_err());
    let mut small = preparer(ImageUrlFetch::Off);
    small.limits.decoded_bytes = 28 * 28 * 3 - 1;
    assert!(small.prepare_videos(&source(&url)).is_err());
}
