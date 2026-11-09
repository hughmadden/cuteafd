//! Per-server cancellation: monitoring stops immediately; completions get a grace period.
use axum::{body::Body, extract::{Request, State}, middleware::Next, response::{IntoResponse, Response}, Extension, Router};
use futures::StreamExt;
use tokio::sync::watch;

#[derive(Clone)]
pub struct Drain {
    monitors: watch::Sender<bool>,
    requests: watch::Sender<bool>,
    sessions: watch::Sender<usize>,
}
impl Default for Drain {
    fn default() -> Self {
        Self { monitors: watch::channel(false).0, requests: watch::channel(false).0,
            sessions: watch::channel(0).0 }
    }
}
impl Drain {
    pub fn stop_monitors(&self) { self.monitors.send_replace(true); }
    pub fn stop_requests(&self) { self.requests.send_replace(true); }
    pub fn is_stopping(&self) -> bool { *self.monitors.borrow() }
    pub async fn sessions_finished(&self) {
        let _ = self.sessions.subscribe().wait_for(|count| *count == 0).await;
    }
    pub fn session(&self) -> Session {
        self.sessions.send_modify(|count| *count += 1);
        Session(self.clone())
    }
    pub async fn monitor_stopped(&self) { stopped(self.monitors.subscribe()).await; }
    async fn requests_stopped(&self) { stopped(self.requests.subscribe()).await; }
    pub fn layer(&self, router: Router) -> Router {
        router.layer(axum::middleware::from_fn_with_state(self.clone(), cancel_requests))
            .layer(Extension(self.clone()))
    }
}
/// Registered before HTTP upgrade, so Axum cannot finish ahead of the session.
pub struct Session(Drain);
impl Session {
    pub async fn run(&self, work: impl std::future::Future<Output = ()>) {
        tokio::select! {
            biased;
            _ = self.0.requests_stopped() => (),
            _ = work => (),
        }
    }
}
impl Drop for Session {
    fn drop(&mut self) { self.0.sessions.send_modify(|count| *count -= 1); }
}

async fn stopped(mut receiver: watch::Receiver<bool>) {
    let _ = receiver.wait_for(|stopped| *stopped).await;
}
async fn cancel_requests(State(drain): State<Drain>, request: Request, next: Next) -> Response {
    let response = tokio::select! {
        biased;
        _ = drain.requests_stopped() => return axum::http::StatusCode::SERVICE_UNAVAILABLE.into_response(),
        response = next.run(request) => response,
    };
    let (parts, body) = response.into_parts();
    let stream = async_stream::stream! {
        let mut stream = body.into_data_stream();
        loop {
            tokio::select! {
                biased;
                _ = drain.requests_stopped() => break,
                chunk = stream.next() => match chunk { Some(chunk) => yield chunk, None => break },
            }
        }
    };
    Response::from_parts(parts, Body::from_stream(stream))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::{Backend, BackendCapabilities, Gateway, GatewayError, ModelInfo, ModelMap,
        TurnRequest, TurnStream, turn::{StopReason, TurnEvent}};
    use futures::{future::BoxFuture, SinkExt};
    use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};
    use serde_json::json;
    struct Held {
        wait: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
        started: tokio::sync::Notify,
        starts: AtomicUsize,
    }
    impl Backend for Held {
        fn name(&self) -> &str { "shutdown fixture" }
        fn capabilities(&self) -> BackendCapabilities { BackendCapabilities::default() }
        fn models(&self) -> Vec<ModelInfo> { vec![] }
        fn start(&self, _: TurnRequest) -> BoxFuture<'static, Result<TurnStream, GatewayError>> {
            self.starts.fetch_add(1, Ordering::SeqCst);
            let wait = self.wait.lock().unwrap().take().unwrap();
            self.started.notify_one();
            Box::pin(async move {
                Ok(Box::pin(futures::stream::once(async move {
                    let _ = wait.await;
                    Ok(TurnEvent::Done { stop: StopReason::EndTurn })
                })) as TurnStream)
            })
        }
        fn count_tokens(&self, _: TurnRequest) -> BoxFuture<'static, Result<u32, GatewayError>> { Box::pin(async { Ok(0) }) }
    }

    #[tokio::test]
    async fn upgraded_inference_stops_new_turns_and_drains_current_turns() {
        for endpoint in ["responses", "realtime"] {
            let (finish, wait) = tokio::sync::oneshot::channel();
            let backend = Arc::new(Held { wait: Mutex::new(Some(wait)), started: tokio::sync::Notify::new(), starts: AtomicUsize::new(0) });
            let gateway = Arc::new(Gateway::new(backend.clone(), ModelMap::single("test")));
            let drain = Drain::default();
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let router = drain.layer(crate::gateway::router(gateway));
            let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
            let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{address}/v1/{endpoint}?model=test")).await.unwrap();
            let request = if endpoint == "responses" { json!({"type":"response.create","model":"test","input":"hi"}) }
                else { json!({"type":"response.create"}) };
            socket.send(tokio_tungstenite::tungstenite::Message::Text(request.to_string())).await.unwrap();
            tokio::time::timeout(std::time::Duration::from_secs(3), backend.started.notified()).await.unwrap();
            drain.stop_monitors();
            assert!(tokio::time::timeout(std::time::Duration::from_millis(20), drain.sessions_finished()).await.is_err(), "{endpoint} lost its active turn");
            socket.send(tokio_tungstenite::tungstenite::Message::Text(request.to_string())).await.unwrap();
            finish.send(()).unwrap();
            let completed = async {
                while let Some(Ok(message)) = socket.next().await {
                    if let tokio_tungstenite::tungstenite::Message::Text(text) = message {
                        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
                        if value["type"] == "response.completed" || value["type"] == "response.done" { return; }
                    }
                }
                panic!("{endpoint} did not complete its active turn");
            };
            tokio::time::timeout(std::time::Duration::from_secs(3), completed).await.unwrap();
            tokio::time::timeout(std::time::Duration::from_secs(3), drain.sessions_finished()).await.unwrap();
            assert_eq!(backend.starts.load(Ordering::SeqCst), 1);
            server.abort();
        }
    }

    #[tokio::test]
    async fn request_deadline_cancels_registered_upgrade_work() {
        let drain = Drain::default();
        let session = drain.session();
        let task = tokio::spawn(async move { session.run(std::future::pending()).await });
        drain.stop_monitors();
        assert!(tokio::time::timeout(std::time::Duration::from_millis(10), drain.sessions_finished()).await.is_err());
        drain.stop_requests();
        task.await.unwrap();
        drain.sessions_finished().await;
    }
}

/// Routers embedded without a serving drain retain their normal stream lifetime.
pub async fn monitor_stopped(drain: Option<Extension<Drain>>) {
    match drain { Some(Extension(drain)) => drain.monitor_stopped().await, None => std::future::pending().await }
}
