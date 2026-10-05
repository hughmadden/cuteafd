//! The benchmark console-text override: a run allows token text on the
//! in-process console for as long as it holds the server (the lockout makes
//! the run's own synthetic prompts the only requests served) and clears it
//! when the run finishes or is cancelled.
mod common;
use common::fake_engine;
use cuteafd_api::openai::{router_with_console, ConsoleHub, NativeLimits, NativeRequest};
use cuteafd_bench::store::Store;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// True within `timeout` once `done` holds.
fn wait_until(timeout: Duration, mut done: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if done() { return true; }
        std::thread::sleep(Duration::from_millis(25));
    }
    done()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn runs_allow_console_text_and_clear_it_on_finish_and_cancel() {
    let (queue, receive) = mpsc::channel::<NativeRequest>(16);
    fake_engine(receive);
    // The server as launched without --console-text: text off by default.
    let hub = ConsoleHub::new(false);
    let router = router_with_console(queue, NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
        Duration::from_secs(5), hub.clone());
    let dir = tempfile::tempdir().unwrap();
    let bench = cuteafd_bench::Bench::new(Store::open(dir.path()).unwrap());
    bench.set_console(hub.clone());
    let app = cuteafd_bench::http::mount(router, bench.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    cuteafd_bench::ready(&listener);
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(listener, app.into_make_service_with_connect_info::<std::net::SocketAddr>()).await.unwrap();
    });
    tokio::task::spawn_blocking(move || {
        let agent = ureq::agent();
        let status = |id: &str| -> Value {
            agent.get(&format!("{base}/v1/bench/runs/{id}")).call().unwrap().into_json().unwrap()
        };
        let start = || -> String {
            let started: Value = agent.post(&format!("{base}/v1/bench/runs")).send_json(json!({"profile": "share"}))
                .unwrap().into_json().unwrap();
            started["id"].as_str().unwrap().to_string()
        };
        // A run that is cancelled: text is allowed while it holds the server
        // and cleared once the cancel lands.
        let id = start();
        assert!(hub.text_enabled(), "a run must allow console text");
        agent.post(&format!("{base}/v1/bench/runs/{id}/cancel")).call().unwrap();
        assert!(wait_until(Duration::from_secs(30), || !hub.text_enabled()), "text stayed on after cancel");
        let report = status(&id);
        assert_eq!(report["status"], "cancelled", "{report:#}");
        // A run that finishes clears it the same way.
        let id = start();
        assert!(hub.text_enabled(), "a run must allow console text");
        let finished = wait_until(Duration::from_secs(120), || {
            !matches!(status(&id)["status"].as_str(), Some("running"))
        });
        assert!(finished, "the run did not finish");
        let report = status(&id);
        assert_eq!(report["status"], "done", "{report:#}");
        assert!(wait_until(Duration::from_secs(30), || !hub.text_enabled()),
            "text stayed on after the run finished");
    }).await.unwrap();
}
