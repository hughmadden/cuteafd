use super::*;
use base64::{engine::general_purpose::STANDARD, Engine};
use cuteafd_loader::media::ImageFamily;
use serde_json::json;
const PNG: &[u8] = include_bytes!("../fixtures/black.png");
fn source() -> MediaSource {
    MediaSource {
        url: format!("data:image/png;base64,{}", STANDARD.encode(PNG)),
        low: false,
    }
}
fn preparer() -> MediaPreparer {
    MediaPreparer::new(
        ProcessorConfig::for_family(ImageFamily::Mimo),
        EncoderId([1; 32]),
        ImageUrlFetch::Public,
        1,
    )
    .unwrap()
}
#[test]
fn extraction_retains_history_and_content_order() {
    let body = json!({"messages":[{"content":[{"type":"text","text":"x"},{"type":"image_url","image_url":{"url":"A","detail":"low"}}]},
        {"content":[{"type":"image_url","image_url":{"url":"B"}},{"type":"image_url","image_url":"C"}]}]});
    let sources = extract_image_sources(&body, 128).unwrap();
    assert_eq!(
        sources.iter().map(|s| s.url.as_str()).collect::<Vec<_>>(),
        ["A", "B", "C"]
    );
    assert!(sources[0].low);
    assert!(extract_image_sources(&body, 2).is_err());
    for detail in [json!("bad"), json!(4), json!(null), json!({})] {
        assert!(extract_image_sources(&json!({"messages":[{"content":[{"type":"image_url","image_url":{"url":"A","detail":detail}}]}]}),128).is_err());
    }
}
#[test]
fn memo_limits_detail_identity_eviction_and_failure_recovery() {
    let mut preparer = preparer();
    let first = preparer.prepare(&[source()]).unwrap();
    assert_eq!((first.decode_misses, first.memo_hits), (1, 0));
    let history = preparer.prepare(&vec![source(); 128]).unwrap();
    assert_eq!((history.decode_misses, history.memo_hits), (0, 128));
    assert!(Arc::ptr_eq(&first.images[0], &history.images[0]));
    let mut low = source();
    low.low = true;
    let changed = preparer.prepare(&[low.clone()]).unwrap();
    assert_ne!(first.images[0].key, changed.images[0].key);
    assert!(preparer.prepare(&vec![source(); 129]).is_err());
    preparer.limits.memo_entries = 1;
    preparer.memo.lock().unwrap().clear();
    preparer.prepare(&[source(), low]).unwrap();
    assert_eq!(preparer.memo.lock().unwrap().len(), 1);
    assert_eq!(preparer.prepare(&[source()]).unwrap().decode_misses, 1);
    preparer.limits.decode_misses = 0;
    preparer.memo.lock().unwrap().clear();
    assert!(preparer.prepare(&[source()]).is_err());
    preparer.limits.decode_misses = 16;
    preparer.limits.decoded_bytes = 1;
    assert!(preparer.prepare(&[source()]).is_err());
    preparer.limits.decoded_bytes = 64 << 20;
    assert!(preparer
        .prepare(&[MediaSource {
            url: "data:image/png;base64,!".into(),
            low: false
        }])
        .is_err());
    assert_eq!(preparer.prepare(&[source()]).unwrap().images.len(), 1);
}
#[test]
fn public_fetch_rejects_local_special_and_mapped_addresses() {
    for address in [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "100.64.0.1",
        "192.168.1.1",
        "198.18.0.1",
        "0.0.0.0",
        "::1",
        "fe80::1",
        "fc00::1",
        "::ffff:127.0.0.1",
        "2001:db8::1",
        "2002:7f00:1::",
    ] {
        assert!(!public_ip(address.parse().unwrap()), "{address}");
    }
    for address in ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"] {
        assert!(public_ip(address.parse().unwrap()));
    }
    for url in [
        "http://127.0.0.1/x",
        "http://localhost/x",
        "http://[::1]/x",
        "file:///etc/passwd",
        "http://u:p@8.8.8.8/x",
    ] {
        assert!(
            preparer()
                .prepare(&[MediaSource {
                    url: url.into(),
                    low: false
                }])
                .is_err(),
            "{url}"
        );
    }
    let off = MediaPreparer::new(
        ProcessorConfig::for_family(ImageFamily::Mimo),
        EncoderId([1; 32]),
        ImageUrlFetch::Off,
        1,
    )
    .unwrap();
    assert!(off
        .prepare(&[MediaSource {
            url: "http://8.8.8.8/x".into(),
            low: false
        }])
        .is_err());
    assert!(off.prepare(&[source()]).is_ok());
}
#[test]
fn stalled_dns_returns_before_the_resolver_finishes() {
    let (release, wait) = std::sync::mpsc::channel();
    let (finished, done) = std::sync::mpsc::channel();
    let error = resolve_with_timeout(move || {
        wait.recv().unwrap();
        finished.send(()).unwrap();
        Ok(vec![])
    }, Duration::from_millis(10)).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
    release.send(()).unwrap();
    done.recv_timeout(Duration::from_secs(1)).unwrap();
}

#[test]
fn any_fetch_checks_redirects_and_response_bounds() {
    use std::{
        io::{Read, Write},
        net::TcpListener,
    };
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/first", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        for redirect in [true, false] {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut request = [0; 4096];
            stream.read(&mut request).unwrap();
            if redirect {
                write!(stream,"HTTP/1.1 302 Found\r\nLocation: /image\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
            } else {
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    PNG.len()
                )
                .unwrap();
                stream.write_all(PNG).unwrap();
            }
        }
    });
    let any = MediaPreparer::new(
        ProcessorConfig::for_family(ImageFamily::Mimo),
        EncoderId([1; 32]),
        ImageUrlFetch::Any,
        1,
    )
    .unwrap();
    assert_eq!(
        any.prepare(&[MediaSource { url, low: false }])
            .unwrap()
            .images
            .len(),
        1
    );
    server.join().unwrap();
}

#[tokio::test]
async fn generic_route_prepares_history_and_reports_image_usage() {
    use crate::openai::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;
    for streaming in [false, true] {
        let profile = ModelProfile::new(
            "test-qwen",
            ModelEncoding::Qwen(Arc::new(chat::qwen4::fixtures::encoding())),
        )
        .with_loaded_vision(Arc::new(preparer()));
        let (tx, mut rx) = tokio::sync::mpsc::channel::<NativeRequest>(1);
        let worker = tokio::spawn(async move {
            let job = rx.recv().await.unwrap();
            assert!(job.images.is_empty());
            assert_eq!(job.media.len(), 2);
            assert!(Arc::ptr_eq(&job.media[0], &job.media[1]));
            assert_eq!(job.prompt.matches("<|image_pad|>").count(), 2);
            for event in [
                InferenceChunk::Ready {
                    system_fingerprint: None,
                    prompt_usage: PromptUsage {
                        prompt_tokens: 32,
                        prompt_cache_hit_tokens: 0,
                    },
                },
                InferenceChunk::Text {
                    content: "ok".into(),
                    content_tokens: 1,
                },
                InferenceChunk::Finish {
                    finish_reason: InferenceFinishReason::Stop,
                },
            ] {
                job.events.send(Ok(event)).unwrap();
            }
        });
        let image = json!({"type":"image_url","image_url":{"url":source().url}});
        let mut body = json!({"model":"test-qwen", "messages":[
            {"role":"user","content":[image.clone(),{"type":"text","text":"first"}]},
            {"role":"assistant","content":"seen"},
            {"role":"user","content":[{"type":"text","text":"again"},image]}],
            "max_tokens":4,"stream":streaming});
        if streaming { body["stream_options"] = json!({"include_usage":true}); }
        let app = router_for_model(
            tx,
            NativeLimits::default(),
            Arc::new(Mutex::new(Value::Null)),
            Duration::from_secs(5),
            ConsoleHub::disabled(),
            profile,
        );
        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&bytes));
        let value: Value = if streaming {
            std::str::from_utf8(&bytes)
                .unwrap()
                .split("data: ")
                .filter_map(|frame| serde_json::from_str::<Value>(frame.trim()).ok())
                .find(|value| value["usage"].is_object())
                .unwrap()
        } else {
            serde_json::from_slice(&bytes).unwrap()
        };
        assert_eq!(value["usage"]["prompt_tokens_details"]["image_tokens"], 8);
        worker.await.unwrap();
    }
}
