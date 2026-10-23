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
    assert_eq!(segments.len(), 2);
    assert_eq!((segments[0].source_frames, segments[0].grid.t, segments[0].tokens), (4, 2, 2));
    assert_eq!((segments[1].source_frames, segments[1].grid.t, segments[1].tokens), (1, 1, 1));
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
