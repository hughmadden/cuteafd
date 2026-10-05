//! `/bench` and `/v1/bench/*`, the lockout of other inference while a run is
//! active, and the access rule for controls: local network, or the server's
//! API key when one is set.
use crate::client::BENCH_HEADER;
use crate::profiles::Profile;
use crate::render;
use crate::report::Report;
use crate::runner::{Bench, RunRequest, StartError};
use axum::body::Body;
use axum::extract::{ConnectInfo, Path, Request, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use serde_json::json;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

/// The benchmark page, compiled in.
pub const PAGE: &str = include_str!("../assets/bench.html");
/// The BENCHMARKING banner any page can include (`/bench/banner.js`).
pub const BANNER: &str = include_str!("../assets/banner.js");

/// Paths that run inference (and so are refused while a benchmark runs).
fn inference(method: &Method, path: &str) -> bool {
    method == Method::POST && matches!(path, "/v1/chat/completions" | "/v1/completions" | "/v1/responses"
        | "/v1/messages" | "/v1/embeddings")
}

/// Refuses other clients' inference with 503 + Retry-After while a run is active.
pub async fn lockout(State(bench): State<Arc<Bench>>, request: Request, next: Next) -> Response {
    if inference(request.method(), request.uri().path()) {
        // The run's own requests carry its token; tools it runs as subprocesses
        // (tool-eval-bench) pass it as their API key.
        let token = request.headers().get(BENCH_HEADER).and_then(|v| v.to_str().ok()).or_else(|| {
            request.headers().get(header::AUTHORIZATION).and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer ")).map(str::trim)
        });
        if let Some(retry) = bench.locked(token) {
            let mut response = (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": {
                "message": format!("a benchmark is running on this server; retry in about {retry} s"),
                "type": "benchmark_running"}}))).into_response();
            response.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from(retry));
            return response;
        }
    }
    next.run(request).await
}

/// Loopback, RFC 1918, link-local, CGNAT and IPv6 unique-local addresses.
pub fn local_network(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_loopback() || v4.is_private() || v4.is_link_local()
            || (v4.octets()[0] == 100 && (64..128).contains(&v4.octets()[1])),
        IpAddr::V6(v6) => v6.is_loopback() || (v6.segments()[0] & 0xfe00) == 0xfc00
            || (v6.segments()[0] & 0xffc0) == 0xfe80
            || v6.to_ipv4_mapped().is_some_and(|v4| local_network(IpAddr::V4(v4))),
    }
}

fn authorized(bench: &Bench, peer: Option<SocketAddr>, headers: &HeaderMap) -> bool {
    if peer.is_some_and(|p| local_network(p.ip())) {
        return true;
    }
    let Some(key) = &bench.api_key else { return false };
    let bearer = headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer ")).map(str::trim);
    bearer == Some(key.as_str())
}

fn forbidden() -> Response {
    (StatusCode::FORBIDDEN, Json(json!({"error": {"message":
        "benchmark controls are accepted from the local network or with the server's API key",
        "type": "forbidden"}}))).into_response()
}

fn not_found(what: &str) -> Response {
    (StatusCode::NOT_FOUND, Json(json!({"error": {"message": format!("{what} not found"), "type": "not_found"}})))
        .into_response()
}

fn peer(connect: Option<ConnectInfo<SocketAddr>>) -> Option<SocketAddr> {
    connect.map(|ConnectInfo(addr)| addr)
}

async fn page() -> Response {
    let page = match std::env::var_os("CUTEAFD_BENCH_PAGE") {
        Some(path) => tokio::fs::read_to_string(&path).await.unwrap_or_else(|_| PAGE.to_string()),
        None => PAGE.to_string(),
    };
    ([(header::CACHE_CONTROL, "no-cache")], axum::response::Html(page)).into_response()
}

async fn banner() -> Response {
    ([(header::CONTENT_TYPE, "text/javascript; charset=utf-8"), (header::CACHE_CONTROL, "no-cache")], BANNER)
        .into_response()
}

async fn status(State(bench): State<Arc<Bench>>) -> Json<serde_json::Value> {
    Json(bench.status())
}

async fn panels(State(bench): State<Arc<Bench>>) -> Json<serde_json::Value> {
    Json(bench.catalog())
}

async fn profiles(State(bench): State<Arc<Bench>>) -> Json<serde_json::Value> {
    Json(json!({"profiles": bench.profiles()}))
}

async fn save_profile(State(bench): State<Arc<Bench>>, connect: Option<ConnectInfo<SocketAddr>>, headers: HeaderMap,
    Path(name): Path<String>, Json(mut profile): Json<Profile>) -> Response {
    if !authorized(&bench, peer(connect), &headers) {
        return forbidden();
    }
    if crate::profiles::builtin().iter().any(|p| p.name == name) {
        return (StatusCode::CONFLICT, Json(json!({"error": {"message": "built-in profiles cannot be replaced"}})))
            .into_response();
    }
    profile.name = name;
    profile.builtin = false;
    match bench.store(|s| s.save_profile(&profile)) {
        Ok(()) => Json(json!({"saved": profile.name})).into_response(),
        Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}")).into_response(),
    }
}

async fn delete_profile(State(bench): State<Arc<Bench>>, connect: Option<ConnectInfo<SocketAddr>>, headers: HeaderMap,
    Path(name): Path<String>) -> Response {
    if !authorized(&bench, peer(connect), &headers) {
        return forbidden();
    }
    match bench.store(|s| s.delete_profile(&name)) {
        Ok(true) => Json(json!({"deleted": name})).into_response(),
        Ok(false) => not_found("profile"),
        Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}")).into_response(),
    }
}

async fn runs(State(bench): State<Arc<Bench>>) -> Response {
    match bench.store(|s| s.list(200)) {
        Ok(rows) => Json(json!({"runs": rows})).into_response(),
        Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}")).into_response(),
    }
}

async fn start(State(bench): State<Arc<Bench>>, connect: Option<ConnectInfo<SocketAddr>>, headers: HeaderMap,
    Json(request): Json<RunRequest>) -> Response {
    if !authorized(&bench, peer(connect), &headers) {
        return forbidden();
    }
    match bench.start(request) {
        Ok(id) => (StatusCode::ACCEPTED, Json(json!({"id": id}))).into_response(),
        Err(error @ StartError::Busy(_)) => (StatusCode::CONFLICT, Json(json!({"error": {"message": error.to_string(),
            "type": "busy"}}))).into_response(),
        Err(error) => (StatusCode::BAD_REQUEST, Json(json!({"error": {"message": error.to_string()}}))).into_response(),
    }
}

#[derive(serde::Deserialize)]
struct ProbeRequest {
    body: serde_json::Value,
    spec: cuteafd_api::openai::probe::ProbeSpec,
}

async fn probe_request(State(bench): State<Arc<Bench>>, connect: Option<ConnectInfo<SocketAddr>>, headers: HeaderMap,
    Json(request): Json<ProbeRequest>) -> Response {
    if !authorized(&bench, peer(connect), &headers) { return forbidden(); }
    match tokio::task::spawn_blocking(move || bench.probe(request.body, request.spec)).await {
        Ok(Ok(chat)) => Json(chat).into_response(),
        Ok(Err(error)) => {
            let status = if error.downcast_ref::<StartError>().is_some_and(|e| matches!(e, StartError::Busy(_))) {
                StatusCode::CONFLICT
            } else { StatusCode::BAD_REQUEST };
            (status, Json(json!({"error": {"message": format!("{error:#}")}}))).into_response()
        }
        Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response(),
    }
}

async fn cancel(State(bench): State<Arc<Bench>>, connect: Option<ConnectInfo<SocketAddr>>, headers: HeaderMap,
    Path(id): Path<String>) -> Response {
    if !authorized(&bench, peer(connect), &headers) {
        return forbidden();
    }
    if bench.cancel(&id) { Json(json!({"cancelled": id})).into_response() } else { not_found("active run") }
}

async fn import(State(bench): State<Arc<Bench>>, connect: Option<ConnectInfo<SocketAddr>>, headers: HeaderMap,
    Json(report): Json<Report>) -> Response {
    if !authorized(&bench, peer(connect), &headers) {
        return forbidden();
    }
    match bench.import(report) {
        Ok(id) => Json(json!({"id": id})).into_response(),
        Err(error) => (StatusCode::BAD_REQUEST, Json(json!({"error": {"message": format!("{error:#}")}}))).into_response(),
    }
}

fn svg(body: String) -> Response {
    ([(header::CONTENT_TYPE, "image/svg+xml"), (header::CACHE_CONTROL, "no-cache")], body).into_response()
}

fn png(svg: &str, file: &str) -> Response {
    match render::png::png(svg, 1.0) {
        Ok(bytes) => ([(header::CONTENT_TYPE, "image/png".to_string()),
            (header::CONTENT_DISPOSITION, format!("inline; filename=\"{file}\""))], bytes).into_response(),
        Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}")).into_response(),
    }
}

/// Every export of one run, by file name.
pub fn export(report: &Report, file: &str) -> Option<(&'static str, Vec<u8>)> {
    let (stem, extension) = file.rsplit_once('.')?;
    let svg = match stem {
        "report" => render::report::report_svg(report),
        "card" => render::card::card_svg(report),
        _ if stem.starts_with("panel-") => render::report::panel_svg(report, &stem["panel-".len()..]),
        _ => return None,
    };
    match extension {
        "svg" => Some(("image/svg+xml", svg.into_bytes())),
        "png" => render::png::png(&svg, 1.0).ok().map(|bytes| ("image/png", bytes)),
        "json" if stem == "report" => serde_json::to_vec_pretty(report).ok().map(|b| ("application/json", b)),
        _ => None,
    }
}

#[derive(serde::Deserialize, Default)]
struct FileQuery {
    /// Panel exports: the body alone at this width (the dashboard's chart view).
    #[serde(default)]
    bare: Option<f64>,
}

async fn run_file(State(bench): State<Arc<Bench>>, Path((id, file)): Path<(String, String)>,
    axum::extract::Query(query): axum::extract::Query<FileQuery>) -> Response {
    let report = if id == "latest" { bench.latest() } else { bench.report(&id) };
    let Some(report) = report else { return not_found("run") };
    if let (Some(width), Some(panel)) = (query.bare, file.strip_prefix("panel-").and_then(|f| f.strip_suffix(".svg"))) {
        return svg(render::report::panel_body_svg(&report, panel, width.clamp(320.0, 2400.0)));
    }
    if file == "report.json" {
        return ([(header::CONTENT_TYPE, "application/json")], serde_json::to_string_pretty(&report)
            .unwrap_or_default()).into_response();
    }
    if file.ends_with(".png") {
        let svg_name = file.replace(".png", ".svg");
        return match export(&report, &svg_name) {
            Some((_, bytes)) => png(&String::from_utf8_lossy(&bytes), &file),
            None => not_found("export"),
        };
    }
    match export(&report, &file) {
        Some((_, bytes)) if file.ends_with(".svg") => svg(String::from_utf8_lossy(&bytes).into_owned()),
        Some((kind, bytes)) => ([(header::CONTENT_TYPE, kind)], bytes).into_response(),
        None => not_found("export"),
    }
}

async fn run_report(State(bench): State<Arc<Bench>>, Path(id): Path<String>) -> Response {
    let report = if id == "latest" { bench.latest() } else { bench.report(&id) };
    match report {
        Some(report) => Json(report).into_response(),
        None => not_found("run"),
    }
}

async fn events(State(bench): State<Arc<Bench>>) -> Response {
    let mut receiver = bench.subscribe();
    let first = json!({"type": "status", "status": bench.status()}).to_string();
    let stream = async_stream::stream! {
        yield Ok::<_, std::convert::Infallible>(format!("data: {first}\n\n"));
        loop {
            match tokio::time::timeout(std::time::Duration::from_secs(15), receiver.recv()).await {
                Ok(Ok(event)) => yield Ok(format!("data: {event}\n\n")),
                Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => continue,
                Ok(Err(_)) => break,
                Err(_) => yield Ok(": keepalive\n\n".to_string()),
            }
        }
    };
    ([(header::CONTENT_TYPE, "text/event-stream"), (header::CACHE_CONTROL, "no-cache")], Body::from_stream(stream))
        .into_response()
}

/// The benchmark routes.
pub fn routes(bench: Arc<Bench>) -> Router {
    Router::new()
        .route("/bench", get(page))
        .route("/bench/banner.js", get(banner))
        .route("/v1/bench/status", get(status))
        .route("/v1/bench/probe", post(probe_request))
        .route("/v1/bench/panels", get(panels))
        .route("/v1/bench/profiles", get(profiles))
        .route("/v1/bench/profiles/:name", put(save_profile).delete(delete_profile))
        .route("/v1/bench/runs", get(runs).post(start))
        .route("/v1/bench/runs/:id", get(run_report))
        .route("/v1/bench/runs/:id/cancel", post(cancel))
        .route("/v1/bench/runs/:id/:file", get(run_file))
        .route("/v1/bench/import", post(import))
        .route("/v1/bench/events", get(events))
        .layer(axum::extract::DefaultBodyLimit::max(64 << 20))
        .with_state(bench)
}

/// `router` with the benchmark mounted and the lockout in front of it.
pub fn mount(router: Router, bench: Arc<Bench>) -> Router {
    router.merge(routes(bench.clone())).layer(axum::middleware::from_fn_with_state(bench, lockout))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_networks() {
        for ip in ["127.0.0.1", "10.55.0.3", "192.168.1.9", "172.20.0.1", "::1", "fd00::1", "fe80::1",
            "::ffff:10.0.0.1", "100.100.1.1"] {
            assert!(local_network(ip.parse().unwrap()), "{ip}");
        }
        for ip in ["8.8.8.8", "2001:4860::8888", "172.32.0.1"] {
            assert!(!local_network(ip.parse().unwrap()), "{ip}");
        }
    }
}
