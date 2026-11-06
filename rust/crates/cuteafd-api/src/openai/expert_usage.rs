//! Shared expert dashboard; data access uses the existing console cookie.
use super::ConsoleHub;
use axum::{
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::get,
    Json, Router,
};
use cuteafd_core::expert_usage::Reader;
use std::sync::Arc;

#[derive(Clone)]
struct Dashboard {
    console: Arc<ConsoleHub>,
    reader: Reader,
}

pub(super) fn routes(console: Arc<ConsoleHub>, reader: Reader) -> Router {
    Router::new()
        .route("/experts", get(page))
        .route("/console/expert-usage", get(snapshot))
        .with_state(Dashboard { console, reader })
}

async fn page() -> Response {
    (
        [(header::CACHE_CONTROL, "no-store")],
        Html(include_str!("../../assets/experts.html")),
    )
        .into_response()
}

async fn snapshot(State(state): State<Dashboard>, headers: HeaderMap) -> Response {
    let cache = [(header::CACHE_CONTROL, "no-store")];
    if !state.console.admin_unlocked(&headers) {
        return (
            StatusCode::UNAUTHORIZED,
            cache,
            Json(serde_json::json!({"error": {"type": "console_locked"}})),
        )
            .into_response();
    }
    (cache, Json(state.reader.snapshot())).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        http::Request,
    };
    use cuteafd_core::expert_usage::{Geometry, Snapshot};
    use tower::ServiceExt;

    #[tokio::test]
    async fn dashboard_data_needs_console_cookie_even_during_bench() {
        let secret = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(secret.path(), "b".repeat(64)).unwrap();
        let gate = crate::console_gate::ConsoleGate::from_file(secret.path(), false).unwrap();
        let console = ConsoleHub::disabled();
        console.set_gate(gate.clone());
        console.set_bench_active(true);
        let reader = Reader::default();
        reader
            .initialize(Geometry {
                layers: 1,
                experts: 2,
                topk: 1,
                routed_layers: vec![0],
            })
            .unwrap();
        reader
            .publish(Snapshot {
                elapsed_s: 1.0,
                prefill: vec![1, 0],
                decode: vec![0, 2],
            })
            .unwrap();
        let app = gate.mount(routes(console, reader));
        let locked = app
            .clone()
            .oneshot(
                Request::get("/console/expert-usage")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(locked.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(locked.headers()[header::CACHE_CONTROL], "no-store");
        let unlock = app
            .clone()
            .oneshot(
                Request::get(format!("/console/unlock?token={}", "b".repeat(64)))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let cookie = unlock.headers()[header::SET_COOKIE]
            .to_str()
            .unwrap()
            .split(';')
            .next()
            .unwrap();
        let response = app
            .oneshot(
                Request::get("/console/expert-usage")
                    .header(header::COOKIE, cookie)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let value: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 1 << 20).await.unwrap())
                .unwrap();
        assert_eq!(
            value["windows"]["total"]["decode_routed_rows"],
            serde_json::json!([0, 2])
        );
    }

    #[tokio::test]
    async fn dashboard_without_configured_gate_stays_locked() {
        let app = routes(ConsoleHub::disabled(), Reader::default());
        let response = app
            .oneshot(
                Request::get("/console/expert-usage")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
}
