use super::*;
use crate::openai::media::ImageUrlFetch;
use cuteafd_loader::media::{EncoderId, ImageFamily, ProcessorConfig};

const FRAMES: [&str; 8] = [
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAAGUlEQVR4nO3BMQEAAADCoPVPbQdvoAAAeA0JTAAB9ZfCmAAAAABJRU5ErkJggg==",
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAAIklEQVR4nGNgpAFgGDV01NBRQ0cNHTV01NBRQ0cNHRmGAgDA4gkxOWScjgAAAABJRU5ErkJggg==",
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAAIklEQVR4nGNgogFgGDV01NBRQ0cNHTV01NBRQ0cNHRmGAgB4hxJh5KEjbgAAAABJRU5ErkJggg==",
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAAIklEQVR4nGNgpgFgGDV01NBRQ0cNHTV01NBRQ0cNHRmGAgAwLBuR8tAkSAAAAABJRU5ErkJggg==",
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAAIklEQVR4nGNgoQFgGDV01NBRQ0cNHTV01NBRQ0cNHRmGAgDnwiTBbLiKMgAAAABJRU5ErkJggg==",
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAAIklEQVR4nGNgpQFgGDV01NBRQ0cNHTV01NBRQ0cNHRmGAgCfZy3xGxiaDwAAAABJRU5ErkJggg==",
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAAIklEQVR4nGNgowFgGDV01NBRQ0cNHTV01NBRQ0cNHRmGAgBXDDchU98aGAAAAABJRU5ErkJggg==",
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAAIklEQVR4nGNgpwFgGDV01NBRQ0cNHTV01NBRQ0cNHRmGAgAOsUBR6aQlXgAAAABJRU5ErkJggg==",
 ];
fn body(part: Value) -> Value { json!({"messages":[{"role":"user", "content":[part]}]}) }
fn sources(part: Value) -> Vec<VideoSource> { take_sources(&mut body(part)).unwrap() }
fn preparer() -> MediaPreparer {
    MediaPreparer::new(ProcessorConfig::for_family(ImageFamily::Minimax), EncoderId([1; 32]), ImageUrlFetch::Off, 1).unwrap()
}

#[test]
fn extraction_normalizes_shapes_defaults_and_top_level_options() {
    for part in [json!({"type":"video_url", "video_url":{"frames":[FRAMES[0]]}}),
        json!({"type":"video", "frames":[{"url":FRAMES[0]}]}),
        json!({"type":"video", "video":{"frames":[{"image_url":{"url":FRAMES[0]}}]}}),
        json!({"type":"input_video", "frames":[FRAMES[0]]})] {
        let source = sources(part).remove(0);
        assert_eq!(source.frames, [FRAMES[0]]);
        assert!(!source.sample && !source.resize);
        assert!(source.count.is_none() && source.source_fps.is_none() && source.fps.is_none());
    }
    for value in [json!("https://example.com/a.mp4"), json!({"url":"https://example.com/a.mp4"})] {
        let mut request = body(json!({"type":"video_url", "video_url":value, "do_sample_frames":true,
            "source_fps":24, "num_frames":3, "do_resize":true}));
        let source = take_sources(&mut request).unwrap().remove(0);
        assert_eq!(source.url.as_deref(), Some("https://example.com/a.mp4"));
        assert_eq!((source.sample, source.resize, source.count, source.source_fps), (true, true, Some(3), Some(24.0)));
        assert_eq!(request["messages"][0]["content"][0], json!({"type":"text", "text":"]<]video[>["}));
    }
}

#[test]
fn extraction_rejects_malformed_sources_options_and_request_counts() {
    for part in [json!({"type":"video_url", "video_url":""}),
        json!({"type":"video_url", "video_url":{"url":3, "frames":[FRAMES[0]]}}),
        json!({"type":"video", "frames":[]}), json!({"type":"video", "frames":[""]}),
        json!({"type":"video_url", "video_url":{"url":"x", "fps":2}, "fps":3}),
        json!({"type":"video_url", "video_url":{"url":"x"}, "url":"y"}),
        json!({"type":"video_url", "video_url":"x", "video":"y"}),
        json!({"type":"video", "frames":[FRAMES[0]], "num_frames":769}),
        json!({"type":"video", "frames":[FRAMES[0]], "num_frames":1, "fps":1}),
        json!({"type":"video", "frames":[FRAMES[0]], "do_sample_frames":1}),
        json!({"type":"video", "frames":vec![FRAMES[0];769]})] {
        assert!(take_sources(&mut body(part.clone())).is_err(), "{part}");
    }
    let part = json!({"type":"video", "frames":[FRAMES[0]]});
    let mut request = json!({"messages":[{"content":vec![part;17]}]});
    assert!(take_sources(&mut request).is_err());
}

#[test]
fn defaults_keep_all_frames_segment_four_and_repeat_the_odd_tail() {
    let source = sources(json!({"type":"video", "frames":&FRAMES[..5], "num_frames":1}));
    let prepared = preparer().prepare_videos(&source).unwrap();
    let segments = &prepared[0].segments;
    assert_eq!(segments.len(), 2);
    assert_eq!((segments[0].source_frames, segments[0].grid.t, segments[0].tokens), (4, 2, 2));
    assert_eq!((segments[1].source_frames, segments[1].grid.t, segments[1].tokens), (1, 1, 1));
    for (index, frame) in segments[0].rgb8.chunks_exact(28*28*3).enumerate() {
        assert!(frame.iter().all(|&p| p == index as u8));
    }
    assert!(segments[1].rgb8.iter().all(|&p| p == 4));
    assert_eq!(segments[1].rgb8.len(), 2*28*28*3);
    assert_eq!(prepared[0].placeholder.matches("]<]video[>[").count(), 3);
    assert!(!prepared[0].placeholder.contains("seconds"));
}

#[test]
fn uniform_sampling_and_timestamps_follow_original_frame_indices() {
    let source = sources(json!({"type":"video_url", "video_url":{"frames":FRAMES},
        "source_fps":2, "do_sample_frames":true, "num_frames":4}));
    let prepared = preparer().prepare_videos(&source).unwrap();
    let segment = &prepared[0].segments[0];
    for (index, frame) in segment.rgb8.chunks_exact(28*28*3).enumerate() {
        assert!(frame.iter().all(|&p| p == (index * 2) as u8));
    }
    assert_eq!(prepared[0].placeholder,
        "]<]0.0 seconds[>[]<]start of image[>[]<]video[>[]<]end of image[>[]<]2.0 seconds[>[]<]start of image[>[]<]video[>[]<]end of image[>[");
    let source = sources(json!({"type":"video", "frames":FRAMES, "source_fps":2, "do_sample_frames":true, "fps":1}));
    let by_fps = preparer().prepare_videos(&source).unwrap();
    assert_eq!(by_fps[0].segments[0].key, segment.key);
    assert_eq!(by_fps[0].placeholder, prepared[0].placeholder);
    for part in [json!({"type":"video", "frames":FRAMES, "do_sample_frames":true}),
        json!({"type":"video", "frames":[FRAMES[0]], "do_sample_frames":true, "num_frames":2})] {
        assert!(preparer().prepare_videos(&sources(part)).is_err());
    }
}

#[test]
fn decoded_prepared_encoded_and_encoder_limits_apply_to_video() {
    let source = sources(json!({"type":"video", "frames":&FRAMES[..5]}));
    let mut p = preparer(); p.limits.decoded_bytes = 5*28*28*3;
    assert!(p.prepare_videos(&source).unwrap_err().to_string().contains("prepared videos"));
    p.limits.decoded_bytes = 28*28*3 - 1;
    assert!(p.prepare_videos(&source).is_err());
    let mut p = preparer(); p.limits.encoded_bytes = 1;
    assert!(p.prepare_videos(&source).is_err());
    let mut p = preparer(); p.config.max_image_tokens = 1;
    assert!(p.prepare_videos(&source).unwrap_err().to_string().contains("token capacity"));
    let mut p = preparer(); p.limits.decoded_bytes = 3*28*28*3;
    let source = sources(json!({"type":"video", "frames":[FRAMES[0]]}));
    assert!(p.prepare_videos(&[source[0].clone(), source[0].clone()]).is_err());
    let resized = sources(json!({"type":"video", "frames":[FRAMES[0]], "do_resize":true}));
    assert!(p.prepare_videos(&resized).unwrap_err().to_string().contains("prepared videos"));
    let mut p = preparer(); p.config.family = ImageFamily::Mimo;
    assert!(p.prepare_videos(&source).is_err());
    assert!(preparer().prepare_videos(&vec![source[0].clone();17]).is_err());
}

#[test]
fn url_policy_refuses_remote_fetches_and_local_files_without_io() {
    let p = preparer();
    for part in [json!({"type":"video_url", "video_url":"https://example.com/a.mp4"}),
        json!({"type":"video", "frames":["https://example.com/a.png"]})] {
        assert!(p.prepare_videos(&sources(part)).unwrap_err().to_string().contains("disabled"));
    }
    let mut p = preparer(); p.fetch = ImageUrlFetch::Any;
    for part in [json!({"type":"video_url", "video_url":"file:///video.mp4"}),
        json!({"type":"video", "frames":["file:///frame.png"]})] {
        assert!(p.prepare_videos(&sources(part)).unwrap_err().to_string().contains("HTTP or HTTPS"));
    }
}

#[test]
fn expansion_preserves_literal_marker_bytes_and_shifts_their_offsets() {
    let prepared = preparer().prepare_videos(&sources(json!({"type":"video", "frames":&FRAMES[..5]}))).unwrap();
    let literal = "literal ]<]video[>[ and ]<]image[>[";
    let prompt = format!("é ]<]video[>[ {literal} ]<]image[>[");
    let start = prompt.find(literal).unwrap();
    let (expanded, spans) = expand_prompt(&prompt, &[start..start+literal.len()], 1, &prepared).unwrap();
    assert_eq!(spans.len(), 1);
    assert_eq!(&expanded[spans[0].clone()], literal);
    assert!(spans[0].start > start);
    assert_eq!(expanded, format!("é {} {literal} ]<]start of image[>[]<]image[>[]<]end of image[>[", prepared[0].placeholder));
    assert!(expand_prompt(&prompt, &[], 1, &prepared).is_err());
    assert!(expand_prompt(&prompt, &[1..2], 1, &prepared).is_err());
    assert!(expand_prompt("]<]video[>[", &[], 0, &[]).is_err());
    assert!(expand_prompt("", &[], 0, &prepared).is_err());
}

#[test]
fn malformed_message_containers_do_not_panic_before_request_validation() {
    for mut request in [json!(null), json!([]), json!({"messages":7}),
        json!({"messages":[null, [], {"content":null}, {"content":[null, []]}]})] {
        assert!(take_sources(&mut request).unwrap().is_empty());
    }
}
