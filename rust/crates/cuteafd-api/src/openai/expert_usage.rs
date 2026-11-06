//! Shared expert dashboard; data access uses the existing console cookie.
use super::ConsoleHub;
use axum::{
    extract::{rejection::QueryRejection, Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::get,
    Json, Router,
};
use cuteafd_core::expert_usage::{Reader, Selected, Selection, UsagePhase, UsageWindow};
use serde::Deserialize;
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
        .route("/console/expert-usage.bin", get(binary_snapshot))
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

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct SelectionQuery {
    window: UsageWindow,
    phase: UsagePhase,
}

fn error(status: StatusCode, kind: &str) -> Response {
    (
        status,
        [(header::CACHE_CONTROL, "no-store")],
        Json(serde_json::json!({"error": {"type": kind}})),
    )
        .into_response()
}

fn etag(id: u64, query: &SelectionQuery) -> String {
    format!(
        "\"eu1-{id:016x}-{}-{}\"",
        window_id(query.window),
        phase_id(query.phase)
    )
}
fn window_id(window: UsageWindow) -> u8 {
    match window {
        UsageWindow::Short => 0,
        UsageWindow::Long => 1,
        UsageWindow::Total => 2,
    }
}
fn phase_id(phase: UsagePhase) -> u8 {
    match phase {
        UsagePhase::Combined => 0,
        UsagePhase::Prefill => 1,
        UsagePhase::Decode => 2,
    }
}

// EU01 v1, all little-endian. Fixed header (128 bytes): magic/version/header size;
// total layers, experts, routed rows, top-k, first layer; window/phase/encoding/stale;
// sampled/window seconds; exact maximum and sample id; peak share; touched/possible;
// coverage/HHI; exact routed rows; count offset and reserved. Then routed u32 IDs,
// padding to 8 bytes, exact u64 row totals, and row-major counts. Encoding 1 = u32,
// 2 = u64 (uploaded directly as low/high RG32UI words). No count is saturated.
fn encode(selection: &Selection, query: &SelectionQuery) -> Vec<u8> {
    let g = &selection.geometry;
    let rows = g.routed_layers.len();
    let totals_offset = (128 + rows * 4 + 7) & !7;
    let counts_offset = totals_offset + rows * 8;
    let wide = query.window == UsageWindow::Total;
    let mut bytes =
        Vec::with_capacity(counts_offset + selection.counts.len() * if wide { 8 } else { 4 });
    bytes.extend_from_slice(b"EU01");
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&128u16.to_le_bytes());
    for value in [g.layers, g.experts, rows, g.topk, g.routed_layers[0]] {
        bytes.extend_from_slice(&(value as u32).to_le_bytes());
    }
    bytes.extend_from_slice(&[
        window_id(query.window),
        phase_id(query.phase),
        if wide { 2 } else { 1 },
        u8::from(selection.stale),
    ]);
    bytes.extend_from_slice(&selection.sampled_s.to_le_bytes());
    bytes.extend_from_slice(&selection.elapsed_s.to_le_bytes());
    bytes.extend_from_slice(&selection.maximum.to_le_bytes());
    bytes.extend_from_slice(&selection.sample_id.to_le_bytes());
    bytes.extend_from_slice(&selection.peak_share.to_le_bytes());
    bytes.extend_from_slice(&(selection.summary.touched as u64).to_le_bytes());
    bytes.extend_from_slice(&(selection.summary.possible as u64).to_le_bytes());
    bytes.extend_from_slice(&selection.summary.coverage.to_le_bytes());
    bytes.extend_from_slice(&selection.summary.concentration.to_le_bytes());
    bytes.extend_from_slice(&selection.summary.routed_rows.to_le_bytes());
    bytes.extend_from_slice(&(counts_offset as u32).to_le_bytes());
    bytes.resize(128, 0);
    for &layer in &g.routed_layers {
        bytes.extend_from_slice(&(layer as u32).to_le_bytes());
    }
    bytes.resize(totals_offset, 0);
    for &total in &selection.totals {
        bytes.extend_from_slice(&total.to_le_bytes());
    }
    for &count in &selection.counts {
        if wide {
            bytes.extend_from_slice(&count.to_le_bytes());
        } else {
            bytes.extend_from_slice(
                &u32::try_from(count)
                    .expect("selected checked u32 window")
                    .to_le_bytes(),
            );
        }
    }
    bytes
}

async fn binary_snapshot(
    State(state): State<Dashboard>,
    headers: HeaderMap,
    query: Result<Query<SelectionQuery>, QueryRejection>,
) -> Response {
    if !state.console.admin_unlocked(&headers) {
        return error(StatusCode::UNAUTHORIZED, "console_locked");
    }
    let Ok(Query(query)) = query else {
        return error(StatusCode::BAD_REQUEST, "invalid_expert_selection");
    };
    let mut selected_etag = String::new();
    let selected = state.reader.selected(query.window, query.phase, |id| {
        selected_etag = etag(id, &query);
        headers
            .get(header::IF_NONE_MATCH)
            .and_then(|h| h.to_str().ok())
            .is_some_and(|tags| {
                tags.split(',')
                    .any(|tag| tag.trim() == selected_etag || tag.trim() == "*")
            })
    });
    let selection = match selected {
        Ok(Selected::Disabled) => {
            return (
                StatusCode::NO_CONTENT,
                [(header::CACHE_CONTROL, "no-store")],
            )
                .into_response()
        }
        Ok(Selected::Unchanged) => {
            return (
                StatusCode::NOT_MODIFIED,
                [
                    (header::CACHE_CONTROL, "no-store"),
                    (header::ETAG, selected_etag.as_str()),
                ],
            )
                .into_response()
        }
        Ok(Selected::Snapshot(selection)) => selection,
        Err(cuteafd_core::expert_usage::SelectionError::WindowCountOverflow) => {
            return error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "expert_window_count_overflow",
            )
        }
        Err(cuteafd_core::expert_usage::SelectionError::TotalOverflow) => {
            return error(StatusCode::INTERNAL_SERVER_ERROR, "expert_total_overflow")
        }
    };
    let bytes = encode(&selection, &query);
    (
        [
            (header::CACHE_CONTROL, "no-store"),
            (header::CONTENT_TYPE, "application/octet-stream"),
            (header::ETAG, selected_etag.as_str()),
        ],
        bytes,
    )
        .into_response()
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

    #[test]
    fn binary_header_row_order_length_and_wide_counts() {
        let reader = Reader::default();
        reader
            .initialize(Geometry {
                layers: 4,
                experts: 2,
                topk: 1,
                routed_layers: vec![1, 3],
            })
            .unwrap();
        reader
            .publish(Snapshot {
                elapsed_s: 2.5,
                prefill: vec![0, 0, 1, 2, 0, 0, 3, 4],
                decode: vec![0; 8],
            })
            .unwrap();
        for window in [UsageWindow::Short, UsageWindow::Total] {
            let query = SelectionQuery {
                window,
                phase: UsagePhase::Prefill,
            };
            let Selected::Snapshot(selected) =
                reader.selected(window, query.phase, |_| false).unwrap()
            else {
                panic!("missing snapshot")
            };
            let bytes = encode(&selected, &query);
            assert_eq!(&bytes[..8], b"EU01\x01\x00\x80\x00");
            let u32_at = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            let u64_at = |offset| u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            assert_eq!(
                (u32_at(8), u32_at(12), u32_at(16), u32_at(20), u32_at(24)),
                (4, 2, 2, 1, 1)
            );
            assert_eq!(
                (bytes[28], bytes[29], bytes[30]),
                (
                    window_id(window),
                    1,
                    if window == UsageWindow::Total { 2 } else { 1 }
                )
            );
            assert_eq!(f64::from_le_bytes(bytes[32..40].try_into().unwrap()), 2.5);
            assert_eq!(u64_at(48), 4);
            assert_eq!((u32_at(128), u32_at(132)), (1, 3));
            assert_eq!((u64_at(136), u64_at(144)), (3, 7));
            assert_eq!(u32_at(112), 152);
            if window == UsageWindow::Total {
                assert_eq!(bytes.len(), 184);
                assert_eq!(
                    (0..4).map(|i| u64_at(152 + i * 8)).collect::<Vec<_>>(),
                    vec![1, 2, 3, 4]
                );
            } else {
                assert_eq!(bytes.len(), 168);
                assert_eq!(
                    (0..4).map(|i| u32_at(152 + i * 4)).collect::<Vec<_>>(),
                    vec![1, 2, 3, 4]
                );
            }
        }
    }

    #[tokio::test]
    async fn binary_route_cookie_query_conditional_and_overflow() {
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
                prefill: vec![1, 2],
                decode: vec![3, 4],
            })
            .unwrap();
        let app = gate.mount(routes(console, reader.clone()));
        let response = app
            .clone()
            .oneshot(
                Request::get("/console/expert-usage.bin?window=bad")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
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
        let request = |uri| {
            Request::get(uri)
                .header(header::COOKIE, cookie)
                .body(Body::empty())
                .unwrap()
        };
        let response = app
            .clone()
            .oneshot(request(
                "/console/expert-usage.bin?window=short&phase=decode",
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "application/octet-stream"
        );
        let tag = response.headers()[header::ETAG].clone();
        assert_eq!(
            to_bytes(response.into_body(), 1024).await.unwrap().len(),
            152
        );
        let mut req = request("/console/expert-usage.bin?window=short&phase=decode");
        req.headers_mut().insert(header::IF_NONE_MATCH, tag.clone());
        let unchanged = app.clone().oneshot(req).await.unwrap();
        assert_eq!(unchanged.status(), StatusCode::NOT_MODIFIED);
        assert_eq!(unchanged.headers()[header::CACHE_CONTROL], "no-store");
        assert!(to_bytes(unchanged.into_body(), 1024)
            .await
            .unwrap()
            .is_empty());
        let mut req = request("/console/expert-usage.bin?window=short&phase=prefill");
        req.headers_mut().insert(header::IF_NONE_MATCH, tag);
        assert_eq!(
            app.clone().oneshot(req).await.unwrap().status(),
            StatusCode::OK
        );
        assert_eq!(
            app.clone()
                .oneshot(request("/console/expert-usage.bin?window=bad"))
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
        reader
            .publish(Snapshot {
                elapsed_s: 2.0,
                prefill: vec![u32::MAX as u64 + 1, 2],
                decode: vec![3, 4],
            })
            .unwrap();
        let overflow = app
            .clone()
            .oneshot(request("/console/expert-usage.bin"))
            .await
            .unwrap();
        assert_eq!(overflow.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let error: serde_json::Value =
            serde_json::from_slice(&to_bytes(overflow.into_body(), 1024).await.unwrap()).unwrap();
        assert_eq!(error["error"]["type"], "expert_window_count_overflow");
        assert_eq!(
            app.oneshot(request("/console/expert-usage.bin?window=total"))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn explicit_off_keeps_authenticated_routes_unavailable() {
        let secret = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(secret.path(), "b".repeat(64)).unwrap();
        let gate = crate::console_gate::ConsoleGate::from_file(secret.path(), false).unwrap();
        let console = ConsoleHub::disabled();
        console.set_gate(gate.clone());
        let reader = Reader::default();
        assert!(!reader.configure(false, false).unwrap());
        assert!(reader.require_enabled().is_err());
        let app = gate.mount(routes(console, reader));
        let unlock = app.clone().oneshot(
            Request::get(format!("/console/unlock?token={}", "b".repeat(64)))
                .body(Body::empty()).unwrap(),
        ).await.unwrap();
        let cookie = unlock.headers()[header::SET_COOKIE].to_str().unwrap()
            .split(';').next().unwrap();
        let request = |uri| Request::get(uri).header(header::COOKIE, cookie)
            .body(Body::empty()).unwrap();
        let json = app.clone().oneshot(request("/console/expert-usage")).await.unwrap();
        assert_eq!(json.status(), StatusCode::OK);
        let body: serde_json::Value = serde_json::from_slice(
            &to_bytes(json.into_body(), 1024).await.unwrap(),
        ).unwrap();
        assert_eq!(body, serde_json::json!({"enabled": false}));
        let binary = app.oneshot(request("/console/expert-usage.bin")).await.unwrap();
        assert_eq!(binary.status(), StatusCode::NO_CONTENT);
        assert_eq!(binary.headers()[header::CACHE_CONTROL], "no-store");
        assert!(!binary.headers().contains_key(header::ETAG));
        assert!(to_bytes(binary.into_body(), 1024).await.unwrap().is_empty());
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
