//! Canonical video sources shared by the Messages and Responses front ends.
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde_json::{json, Value};
use super::{turn::Part, GatewayError};

pub(super) fn responses(part: &Value) -> Result<Part, GatewayError> {
    Ok(Part::Video { source: openai(part)? })
}

pub(crate) fn openai(part: &Value) -> Result<Value, GatewayError> {
    if part.get("file_id").is_some_and(|v| !v.is_null()) {
        return Err(GatewayError::unsupported("input_video file_id is not supported; supply video_url or frames"));
    }
    if part.get("video_url").is_some() && part.get("video").is_some() {
        return Err(GatewayError::invalid("video_url and video cannot both be supplied"));
    }
    let source = part.get("video_url").or_else(|| part.get("video")).unwrap_or(part);
    let Part::Video { source } = canonical(source.clone(), part)? else { unreachable!() };
    Ok(source)
}

pub(super) fn anthropic(part: &Value) -> Result<Part, GatewayError> {
    let Some(source) = part.get("source") else { return canonical(part.clone(), part) };
    let mut source = source.as_object().cloned().ok_or_else(|| GatewayError::invalid("video source must be an object"))?;
    match source.remove("type").and_then(|v| v.as_str().map(str::to_owned)).as_deref() {
        Some("url") if source.contains_key("url") => (),
        Some("frames") if source.contains_key("frames") => (),
        Some("base64") => {
            if source.contains_key("url") || source.contains_key("frames") {
                return Err(GatewayError::invalid("base64 video source cannot also contain URL or frames"));
            }
            let media_type = source.remove("media_type").and_then(|v| v.as_str().map(str::to_owned))
                .filter(|v| v.starts_with("video/") && !v.contains([',', ';', '\r', '\n']))
                .ok_or_else(|| GatewayError::invalid("video source media_type must be a video MIME type"))?;
            let data = source.remove("data").and_then(|v| v.as_str().map(str::to_owned))
                .filter(|v| !v.is_empty()).ok_or_else(|| GatewayError::invalid("video source data is required"))?;
            STANDARD.decode(&data).map_err(|_| GatewayError::invalid("invalid video base64"))?;
            source.insert("url".into(), json!(format!("data:{media_type};base64,{data}")));
        }
        _ => return Err(GatewayError::unsupported("video source type must be url, base64 or frames")),
    }
    canonical(Value::Object(source), part)
}

fn canonical(source: Value, part: &Value) -> Result<Part, GatewayError> {
    let mut source = match source {
        Value::String(url) => json!({"url":url}),
        Value::Object(source) => Value::Object(source),
        _ => return Err(GatewayError::invalid("video source must be a URL or object")),
    };
    let source = source.as_object_mut().unwrap();
    for name in ["url", "frames", "source_fps", "do_sample_frames", "num_frames", "fps", "do_resize"] {
        if let Some(value) = part.get(name) {
            if source.get(name).is_some_and(|old| old != value) {
                return Err(GatewayError::invalid(format!("conflicting video {name}")));
            }
            source.insert(name.into(), value.clone());
        }
    }
    let url = source.get("url");
    let frames = source.get("frames");
    if url.is_some() == frames.is_some() {
        return Err(GatewayError::invalid("video requires exactly one URL or frames array"));
    }
    if url.is_some_and(|v| v.as_str().is_none_or(str::is_empty)) {
        return Err(GatewayError::invalid("video URL must be a nonempty string"));
    }
    if let Some(frames) = frames {
        let frames = frames.as_array().filter(|v| !v.is_empty() && v.len() <= 768)
            .ok_or_else(|| GatewayError::invalid("video frames must contain 1 through 768 entries"))?;
        for frame in frames {
            if frame.as_str().or_else(|| frame["url"].as_str())
                .or_else(|| frame["image_url"]["url"].as_str()).is_none_or(str::is_empty) {
                return Err(GatewayError::invalid("video frames must contain image URLs"));
            }
        }
    }
    for name in ["do_sample_frames", "do_resize"] {
        if source.get(name).is_some_and(|v| !v.is_boolean()) {
            return Err(GatewayError::invalid(format!("video {name} must be boolean")));
        }
    }
    for name in ["source_fps", "fps"] {
        if source.get(name).is_some_and(|v| v.as_f64().is_none_or(|n| !n.is_finite() || n <= 0.0)) {
            return Err(GatewayError::invalid(format!("video {name} must be positive")));
        }
    }
    if source.get("num_frames").is_some_and(|v| v.as_u64().is_none_or(|n| n == 0 || n > 768)) {
        return Err(GatewayError::invalid("video num_frames must be 1 through 768"));
    }
    if source.contains_key("num_frames") && source.contains_key("fps") {
        return Err(GatewayError::invalid("video num_frames and fps are mutually exclusive"));
    }
    source.retain(|key, _| matches!(key.as_str(), "url" | "frames" | "source_fps" | "do_sample_frames" | "num_frames" | "fps" | "do_resize"));
    Ok(Part::Video { source: Value::Object(source.clone()) })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn protocol_sources_preserve_frames_and_processor_options() {
        let frames = json!(["https://example.com/a.png", {"url":"https://example.com/b.png"},
            {"image_url":{"url":"data:image/png;base64,aQ=="}}]);
        let expected = json!({"frames":frames, "source_fps":24, "do_sample_frames":true,
            "num_frames":3, "do_resize":false});
        let a = anthropic(&json!({"type":"video", "source":{"type":"frames", "frames":frames},
            "source_fps":24, "do_sample_frames":true, "num_frames":3, "do_resize":false})).unwrap();
        let b = responses(&json!({"type":"input_video", "video_url":expected})).unwrap();
        let c = responses(&json!({"type":"input_video", "frames":frames,
            "source_fps":24, "do_sample_frames":true, "num_frames":3, "do_resize":false})).unwrap();
        assert_eq!(a, Part::Video { source: expected }); assert_eq!(a, b); assert_eq!(a, c);
        assert_eq!(anthropic(&json!({"type":"video", "source":{"type":"base64", "media_type":"video/mp4", "data":"aQ=="}})).unwrap(),
            Part::Video { source: json!({"url":"data:video/mp4;base64,aQ=="}) });
        assert_eq!(responses(&json!({"type":"input_video", "video_url":"https://example.com/a.mp4", "fps":2})).unwrap(),
            Part::Video { source: json!({"url":"https://example.com/a.mp4", "fps":2}) });
        assert_eq!(anthropic(&json!({"type":"video", "source":{"type":"url", "url":"https://example.com/a.mp4", "fps":2}})).unwrap(),
            Part::Video { source: json!({"url":"https://example.com/a.mp4", "fps":2}) });
    }
    #[test]
    fn malformed_video_blocks_never_disappear_or_drop_requested_options() {
        for value in [json!(null), json!({}), json!({"url":""}), json!({"frames":[]}),
            json!({"frames":[42]}), json!({"frames":[{"url":null}]}),
            json!({"url":"x", "frames":["y"]}), json!({"url":"x", "fps":0}),
            json!({"url":"x", "source_fps":"24"}), json!({"url":"x", "num_frames":1.5}),
            json!({"url":"x", "do_sample_frames":"yes"}), json!({"url":"x", "do_resize":1}),
            json!({"url":"x", "fps":2, "num_frames":3})] {
            assert!(responses(&json!({"type":"input_video", "video_url":value})).is_err(), "{value}");
        }
        for source in [json!({"type":"file", "file_id":"a"}), json!({"type":"url", "frames":["x"]}),
            json!({"type":"base64", "media_type":"image/png", "data":"aQ=="}),
            json!({"type":"base64", "media_type":"video/mp4", "data":"!"})] {
            assert!(anthropic(&json!({"type":"video", "source":source})).is_err());
        }
        assert!(responses(&json!({"type":"input_video", "file_id":"a"})).is_err());
        assert!(responses(&json!({"type":"input_video", "video_url":{"url":"x", "fps":2}, "fps":3})).is_err());
    }
}
