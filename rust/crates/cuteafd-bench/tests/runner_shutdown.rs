//! Serving lifecycle around a benchmark: discovery after ready, and a shutdown
//! during a run (its own binary: the loopback address is process-wide).
mod common;
use common::fake_engine;
use cuteafd_api::openai::{auth::ApiKey, router_with_console, ConsoleHub, NativeLimits, NativeRequest};
use cuteafd_bench::report::RunStatus;
use cuteafd_bench::store::Store;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;

/// Discovery names the fingerprint before any run; a shutdown mid-run saves the run as interrupted with the reason,
/// never as a panel failure from the connection errors that follow.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn discovery_then_shutdown_mid_run_is_recorded_as_interrupted() {
    let (queue, receive) = mpsc::channel::<NativeRequest>(16);
    fake_engine(receive);
    let router = router_with_console(queue, NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
        Duration::from_secs(5), ConsoleHub::disabled());
    let dir = tempfile::tempdir().unwrap();
    let bench = cuteafd_bench::Bench::new(Store::open(dir.path()).unwrap());
    bench.set_api_key(ApiKey::new("integration-test-key").unwrap());
    let app = cuteafd_bench::http::mount(router, bench.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    cuteafd_bench::context::mark_ready(listener.local_addr().unwrap());
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(listener, app.into_make_service_with_connect_info::<std::net::SocketAddr>()).await.unwrap();
    });
    // Discovery after ready: the status names this server's fingerprint before any run.
    let discovering = bench.clone();
    tokio::task::spawn_blocking(move || discovering.discover()).await.unwrap();
    assert!(bench.status()["fingerprint"].is_string(), "fingerprint known before the first run");
    let stopper = bench.clone();
    let report = tokio::task::spawn_blocking(move || {
        let agent = ureq::agent();
        let started: Value = agent.post(&format!("{base}/v1/bench/runs")).set("Authorization", "Bearer integration-test-key")
            .send_json(json!({"profile": "share"})).unwrap().into_json().unwrap();
        let id = started["id"].as_str().unwrap().to_string();
        std::thread::sleep(Duration::from_millis(800));
        stopper.stopping("SIGTERM");
        let refused = agent.post(&format!("{base}/v1/bench/runs")).set("Authorization", "Bearer integration-test-key")
            .send_json(json!({"profile": "share"}));
        assert!(refused.is_err(), "no new run while stopping");
        for _ in 0..300 {
            if stopper.active().is_none() { return stopper.report(&id).unwrap(); }
            std::thread::sleep(Duration::from_millis(100));
        }
        panic!("the run did not stop");
    }).await.unwrap();
    assert_eq!(report.status, RunStatus::Cancelled);
    assert_eq!(report.error.as_deref(), Some("interrupted: the server stopped (SIGTERM) during this run"));
}
