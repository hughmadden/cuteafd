//! Per-server cancellation: monitoring stops immediately; completions get a grace period.
use axum::{body::Body, extract::{Request, State}, middleware::Next, response::{IntoResponse, Response}, Extension, Router};
use futures::StreamExt;
use tokio::sync::watch;

#[derive(Clone)]
pub struct Drain {
    monitors: watch::Sender<bool>,
    requests: watch::Sender<bool>,
}
impl Default for Drain {
    fn default() -> Self {
        Self { monitors: watch::channel(false).0, requests: watch::channel(false).0 }
    }
}
impl Drain {
    pub fn stop_monitors(&self) { self.monitors.send_replace(true); }
    pub fn stop_requests(&self) { self.requests.send_replace(true); }
    pub async fn monitor_stopped(&self) { stopped(self.monitors.subscribe()).await; }
    async fn requests_stopped(&self) { stopped(self.requests.subscribe()).await; }
    pub fn layer(&self, router: Router) -> Router {
        router.layer(axum::middleware::from_fn_with_state(self.clone(), cancel_requests))
            .layer(Extension(self.clone()))
    }
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

/// Routers embedded without a serving drain retain their normal stream lifetime.
pub async fn monitor_stopped(drain: Option<Extension<Drain>>) {
    match drain { Some(Extension(drain)) => drain.monitor_stopped().await, None => std::future::pending().await }
}
