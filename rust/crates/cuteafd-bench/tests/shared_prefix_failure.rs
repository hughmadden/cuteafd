//! Shared-prefix failures survive the real runner, JSON export and SQLite store.
use axum::{http::StatusCode, response::IntoResponse, routing::{get, post}, Json, Router};
use cuteafd_bench::{context, report::Setting, store::Store, Bench};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

async fn chat(Json(body): Json<Value>) -> axum::response::Response {
    if body["messages"].as_array().unwrap().iter().any(|m|
        m["role"] == "assistant" && m["content"] == "" && m["reasoning_content"] != "Let me check") {
        return (StatusCode::BAD_REQUEST, "empty assistant turn: reasoning was lost").into_response();
    }
    if body["messages"].as_array().unwrap().iter().any(|m|
        m["content"].as_str().is_some_and(|s| s.contains("-warm-C4-"))) {
        return (StatusCode::BAD_REQUEST, "mock retained C4 rejection: assistant content is empty").into_response();
    }
    let prompt = body["messages"].to_string().len() / 4;
    let delta = if body["max_tokens"] == 1 {
        json!({"choices": [{"delta": {}, "finish_reason": "length"}]})
    } else {
        json!({"choices": [{"delta": {"content": "OK", "reasoning_content": "Let me check"}, "finish_reason": "stop"}]})
    };
    let usage = json!({"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": 2,
        "prompt_tokens_details": {"cached_tokens": prompt}}});
    ([("content-type", "text/event-stream")],
        format!("data: {delta}\n\ndata: {usage}\n\ndata: [DONE]\n\n")).into_response()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn retained_http_rejection_is_exported_with_point_and_body() {
    context::set(context::ServerContext { settings: vec![
        Setting { name: "pool-tokens".into(), value: Some("2097152".into()), ..Default::default() },
        Setting { name: "concurrency".into(), value: Some("16".into()), ..Default::default() },
    ], ..Default::default() });
    let router = Router::new()
        .route("/v1/models", get(|| async { Json(json!({"data": [{"id": "mock",
            "max_context_tokens": 131072, "max_output_tokens": 512}]})) }))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/stats", get(|| async { Json(json!({})) }))
        .route("/v1/console/snapshot", get(|| async { Json(json!({})) }));
    let dir = tempfile::tempdir().unwrap();
    let bench = Bench::new(Store::open(dir.path()).unwrap());
    bench.set_api_key(cuteafd_api::openai::auth::ApiKey::new("integration-test-key").unwrap());
    let app = cuteafd_bench::http::mount(router, bench.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    cuteafd_bench::ready(&listener);
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
    let (id, report) = tokio::task::spawn_blocking(move || {
        let agent = ureq::agent();
        let started: Value = agent.post(&format!("{base}/v1/bench/runs")).set("Authorization", "Bearer integration-test-key")
            .send_json(json!({"panels": ["shared_prefix"], "baseline": false, "fabric": false}))
            .unwrap().into_json().unwrap();
        let id = started["id"].as_str().unwrap().to_owned();
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let report: Value = agent.get(&format!("{base}/v1/bench/runs/{id}/report.json")).set("Authorization", "Bearer integration-test-key")
                .call().unwrap().into_json().unwrap();
            if report["status"] != "running" { return (id, report); }
            assert!(Instant::now() < deadline, "benchmark did not finish");
            std::thread::sleep(Duration::from_millis(20));
        }
    }).await.unwrap();
    let panel = report["panels"].as_array().unwrap().iter().find(|p| p["id"] == "shared_prefix").unwrap();
    assert_eq!(panel["status"], "failed", "{report:#}");
    let error = panel["error"].as_str().unwrap();
    assert!(error.contains("tokens C4") && error.contains("HTTP 400") && error.contains("assistant content is empty"), "{error}");
    let base = &panel["partial"]["retained"]["bases"][0];
    assert_eq!(base["cold_response"]["raw_choices"][0], json!({"delta": {}, "finish_reason": "length"}));
    assert_eq!(base["attempted_assistant"], json!({"role": "assistant", "content": ""}));
    assert_eq!(base["assistant_replay"], json!({"role": "assistant", "content": "OK", "reasoning_content": "Let me check"}));
    assert_eq!(base["completed_response"]["timing"]["finish_reason"], "stop");
    let points = panel["partial"]["retained"]["points"].as_array().unwrap();
    assert_eq!(points[0]["status"], "ok");
    let failed = &points[1];
    assert_eq!(failed["requested_context"], 32768);
    assert_eq!(failed["c"], 4);
    assert_eq!(failed["context"], base["retained_prefix_tokens"]);
    assert_eq!(failed["base_tokens"], base["retained_prefix_tokens"]);
    assert_eq!(failed["status"], "failed");
    assert_eq!(failed["request_errors"][0]["phase"], "warm");
    assert_eq!(failed["request_errors"][0]["http_status"], 400);
    assert_eq!(failed["request_errors"][0]["response_body"], "mock retained C4 rejection: assistant content is empty");
    let stored = bench.store(|s| s.load(&id)).unwrap().unwrap();
    let stored = serde_json::to_value(stored).unwrap();
    let stored_panel = stored["panels"].as_array().unwrap().iter().find(|p| p["id"] == "shared_prefix").unwrap();
    assert_eq!(stored_panel["partial"], panel["partial"]);
    server.abort();
}
