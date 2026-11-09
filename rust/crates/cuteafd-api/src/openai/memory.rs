//! Console-authenticated memory view. HTTP only reads published snapshots.
use super::{ConsoleHub, SharedStats};
use axum::{extract::State, http::{header, HeaderMap, StatusCode}, response::{Html, IntoResponse, Response}, routing::get, Json, Router};
use serde_json::{json, Value};
use std::sync::{Arc, RwLock};

static INVENTORY: RwLock<Value> = RwLock::new(Value::Null);

pub fn publish_memory_inventory(value: Value) {
    *INVENTORY.write().unwrap_or_else(|e| e.into_inner()) = value;
}

#[derive(Clone)]
struct Dashboard { console: Arc<ConsoleHub>, stats: SharedStats }

pub(super) fn routes(console: Arc<ConsoleHub>, stats: SharedStats) -> Router {
    Router::new().route("/memory", get(page))
        .route("/assets/memory.js", get(script))
        .route("/console/memory", get(snapshot))
        .with_state(Dashboard { console, stats })
}
async fn page() -> impl IntoResponse {
    ([(header::CACHE_CONTROL, "no-store")], Html(include_str!("../../assets/memory.html")))
}
async fn script() -> impl IntoResponse {
    ([(header::CACHE_CONTROL, "no-store"), (header::CONTENT_TYPE, "text/javascript; charset=utf-8")], include_str!("../../assets/memory.js"))
}
// Keep the physical-device mapping and numeric worker ownership required by the
// UI, not checkpoint paths, slices, environment arguments or process samples.
fn reconciliation_view(mut value: Value) -> Value {
    for stage in ["ready", "first_request"] {
        if let Some(rows) = value[stage]["devices"].as_array_mut() {
            for row in rows {
                let provenance = &row["provenance"];
                let ledger = &provenance["owner_ledger"];
                let allocations: Vec<_> = ledger["runtime"]["detail"]["allocations"].as_array()
                    .into_iter().flatten().map(|owner| json!({"space": owner["space"],
                        "scope": owner["scope"], "bytes": owner["bytes"]})).collect();
                let worker = if ledger.is_object() { json!({
                    "rank": ledger["rank"], "resident_bytes": ledger["resident_bytes"],
                    "mapped_ring_bytes": ledger["mapped_ring_bytes"],
                    "host_exchange_bytes": ledger["host_exchange_bytes"],
                    "row_indices_bytes": ledger["row_indices_bytes"],
                    "runtime": {"workspace_bytes": ledger["runtime"]["workspace_bytes"],
                        "pinned_workspace_bytes": ledger["runtime"]["pinned_workspace_bytes"],
                        "detail": {"allocations": allocations}}
                }) } else { Value::Null };
                let mapping = json!({"device_sample": {"device": provenance["device_sample"]["device"]},
                    "owner_ledger": worker});
                row["provenance"] = mapping;
            }
        }
    }
    value
}

async fn snapshot(State(state): State<Dashboard>, headers: HeaderMap) -> Response {
    let cache = [(header::CACHE_CONTROL, "no-store")];
    if !state.console.admin_unlocked(&headers) {
        return (StatusCode::UNAUTHORIZED, cache, Json(json!({"error": {"type": "console_locked"}}))).into_response();
    }
    let stats = state.stats.lock().unwrap_or_else(|e| e.into_inner());
    // No arbitrary stats, checkpoint paths, prompts or provenance arguments.
    let mut selected = serde_json::Map::new();
    for key in ["prefix_cache", "kv_reserve", "graph_cache", "memory_reconciliation"] {
        let value = stats.get(key).cloned().unwrap_or(Value::Null);
        selected.insert(key.into(), if key == "memory_reconciliation" { reconciliation_view(value) } else { value });
    }
    drop(stats);
    selected.insert("host_heap".into(), super::HOST_HEAP.read().unwrap_or_else(|e| e.into_inner()).clone());
    (cache, Json(json!({"schema_version": 1, "stats": selected,
        "inventory": INVENTORY.read().unwrap_or_else(|e| e.into_inner()).clone()}))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::{to_bytes, Body}, http::Request};
    use std::sync::Mutex;
    use tower::ServiceExt;

    #[tokio::test]
    async fn memory_data_requires_console_cookie_even_during_bench() {
        let secret = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(secret.path(), "b".repeat(64)).unwrap();
        let gate = crate::console_gate::ConsoleGate::from_file(secret.path(), false).unwrap();
        let console = ConsoleHub::disabled();
        console.set_gate(gate.clone());
        console.set_bench_active(true);
        let stats = Arc::new(Mutex::new(json!({"prefix_cache": {"pages": 20}, "secret": "excluded"})));
        let app = gate.mount(routes(console, stats));
        let locked = app.clone().oneshot(Request::get("/console/memory").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(locked.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(locked.headers()[header::CACHE_CONTROL], "no-store");
        let page = app.clone().oneshot(Request::get("/memory").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(page.status(), StatusCode::OK);
        let unlock = app.clone().oneshot(Request::get(format!("/console/unlock?token={}", "b".repeat(64))).body(Body::empty()).unwrap()).await.unwrap();
        let cookie = unlock.headers()[header::SET_COOKIE].to_str().unwrap().split(';').next().unwrap();
        let response = app.oneshot(Request::get("/console/memory").header(header::COOKIE, cookie).body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
        let value: Value = serde_json::from_slice(&to_bytes(response.into_body(), 1 << 20).await.unwrap()).unwrap();
        assert_eq!(value["stats"]["prefix_cache"]["pages"], 20);
        assert!(value["stats"].get("secret").is_none());
        assert!(value["stats"]["graph_cache"].is_null());
    }

    #[test]
    fn slim_reconciliation_retains_physical_mapping_and_worker_owners_without_paths() {
        let value = reconciliation_view(json!({"ready": {"devices": [{"device": "rtx0",
            "measured_bytes": null, "provenance": {"device_sample": {"device": 1},
                "layout": "/private/checkpoint", "owner_ledger": {"rank": 0,
                    "resident_bytes": 123, "mapped_ring_bytes": 44,
                    "runtime": {"workspace_bytes": 55, "pinned_workspace_bytes": 66,
                        "detail": {"process_samples": [{"path": "/private/process"}],
                            "allocations": [{"space": "host", "scope": "exchange", "bytes": 77,
                                "method": "/private/method"}]}}}}}]}}));
        let row = &value["ready"]["devices"][0];
        assert_eq!(row["provenance"]["device_sample"]["device"], 1);
        assert_eq!(row["provenance"]["owner_ledger"]["runtime"]["workspace_bytes"], 55);
        assert_eq!(row["provenance"]["owner_ledger"]["runtime"]["detail"]["allocations"][0]["bytes"], 77);
        assert!(row["measured_bytes"].is_null());
        assert!(!value.to_string().contains("/private"));
    }

    #[tokio::test]
    async fn memory_without_configured_gate_stays_locked() {
        let app = routes(ConsoleHub::disabled(), Arc::new(Mutex::new(Value::Null)));
        assert_eq!(app.oneshot(Request::get("/console/memory").body(Body::empty()).unwrap()).await.unwrap().status(), StatusCode::UNAUTHORIZED);
    }
}
