//! The in-server benchmark: `/bench`, `/v1/bench/*` and `cuteafd bench`.
//!
//! A normally launched server runs benchmarks on request. The runner drives
//! the server through its own OpenAI API over loopback (so SSE and the
//! scheduler are measured), holds every other inference request off with
//! 503 + Retry-After while it runs, and stores each report in SQLite. The
//! mandatory baseline (basic card + quick quality) runs once per server
//! lifetime and configuration fingerprint and is shared by later runs.
pub mod baseline;
pub mod cli;
pub mod client;
pub mod context;
pub mod http;
pub mod panels;
pub mod profiles;
pub mod publish;
pub mod reference;
pub mod render;
pub mod report;
pub mod runner;
pub mod sample;
pub mod server;
pub mod smoke;
pub mod store;
pub mod text;

pub use runner::Bench;

/// `router` with the benchmark mounted (routes and lockout) on the process-wide bench.
pub fn app(router: axum::Router) -> axum::Router {
    http::mount(router, Bench::global())
}

/// Records that the API now accepts requests on `listener` (readiness time
/// and the runner's loopback address).
pub fn ready(listener: &tokio::net::TcpListener) {
    if let Ok(addr) = listener.local_addr() {
        context::mark_ready(addr);
    }
}
