//! A model-free server for working on the /bench page: the real OpenAI router
//! and console assets in front of a fake engine (2 ms tokens, probes honoured
//! with fake rows), with three sample reports (one failed, one with every
//! panel) in a scratch store.
//!
//!     cargo run -p cuteafd-bench --example fake_server -- 8399 [STORE_DIR]
//!     open http://127.0.0.1:8399/bench
//!
//! STORE_DIR (a copy of a real `bench.sqlite` directory) shows real reports
//! instead of the samples; `CUTEAFD_BENCH_PAGE` / `CUTEAFD_CONSOLE_ASSETS`
//! serve the pages from a checkout as on a WIP launch.
use cuteafd_api::openai::{router_with_console, ConsoleHub, InferenceChunk, InferenceFinishReason, NativeLimits,
    NativeRequest, PromptUsage};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;

fn fake_engine(mut receive: mpsc::Receiver<NativeRequest>) {
    std::thread::spawn(move || {
        while let Some(job) = receive.blocking_recv() {
            std::thread::spawn(move || {
                let ids: Vec<u32> = job.probe.as_ref().and_then(|p| p.spec.prompt_ids.clone())
                    .unwrap_or_else(|| job.prompt.bytes().map(u32::from).collect());
                if let Some(probe) = &job.probe {
                    probe.admitted("fake", &ids, 0);
                }
                let _ = job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                    prompt_usage: PromptUsage { prompt_tokens: ids.len(), prompt_cache_hit_tokens: 0 } }));
                std::thread::sleep(Duration::from_millis(ids.len() as u64 / 20));
                if let Some(probe) = &job.probe {
                    let logits: Vec<f32> = (0..64).map(|v| ((v * 31 + ids.len()) % 17) as f32).collect();
                    if probe.spec.record_first {
                        probe.row(ids.len(), &logits);
                    }
                }
                for i in 0..job.max_tokens.min(200) {
                    std::thread::sleep(Duration::from_millis(12));
                    if let Some(probe) = &job.probe {
                        probe.token(7 + i as u32);
                    }
                    if job.events.send(Ok(InferenceChunk::Text { content: "a ".into(), content_tokens: 1 })).is_err() {
                        return;
                    }
                }
                let _ = job.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Length }));
            });
        }
    });
}

#[tokio::main]
async fn main() {
    let port: u16 = std::env::args().nth(1).and_then(|p| p.parse().ok()).unwrap_or(8399);
    let (queue, receive) = mpsc::channel::<NativeRequest>(16);
    fake_engine(receive);
    let router = router_with_console(queue, NativeLimits::default(), Arc::new(Mutex::new(serde_json::Value::Null)),
        Duration::from_secs(5), ConsoleHub::disabled());
    let given = std::env::args().nth(2).map(std::path::PathBuf::from);
    let dir = given.clone().unwrap_or_else(|| std::env::temp_dir().join(format!("cuteafd-bench-fake-{port}")));
    let store = cuteafd_bench::store::Store::open(&dir).expect("store");
    for (failed, created) in [(true, "2026-10-01T09:00:00Z"), (false, "2026-10-01T10:00:00Z")].into_iter().filter(|_| given.is_none()) {
        let mut report = cuteafd_bench::sample::report(failed);
        report.id = format!("sample-{}", if failed { "failed" } else { "ok" });
        report.created = created.into();
        store.save(&report).expect("sample");
    }
    // Every measurement panel with synthetic passes, newest, for chart work.
    if given.is_none() {
        let mut report = cuteafd_bench::sample::full_report();
        report.id = "sample-full".into();
        report.created = "2026-10-01T11:00:00Z".into();
        store.save(&report).expect("full sample");
    }
    // As the serving bench does after start (here inline: the store is a small copy).
    if store.stale().unwrap_or(false) { store.backfill().expect("backfill"); }
    let bench = cuteafd_bench::Bench::new(store);
    // CUTEAFD_CONSOLE_SECRET_FILE: the admin cookie flow as on a real server
    // (`/console/unlock?token=<file contents>`).
    let router = match std::env::var_os("CUTEAFD_CONSOLE_SECRET_FILE") {
        Some(path) => {
            let gate = cuteafd_api::console_gate::ConsoleGate::from_file(std::path::Path::new(&path), false).expect("console secret");
            bench.set_console_gate(gate.clone());
            gate.mount(router)
        }
        None => router,
    };
    let app = cuteafd_bench::http::mount(router, bench);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await.expect("bind");
    cuteafd_bench::ready(&listener);
    eprintln!("http://127.0.0.1:{port}/bench (store {})", dir.display());
    axum::serve(listener, app.into_make_service_with_connect_info::<std::net::SocketAddr>()).await.expect("serve");
}
