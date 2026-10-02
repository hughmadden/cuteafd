//! Live engine console: the `/` page, its WebSocket feed and a JSON snapshot.
//!
//! The CUDA worker never touches this module's sockets. A producer on the worker
//! side checks [`ConsoleHub::viewers`] (one relaxed atomic load) before building
//! any per-round telemetry, hands owned events to a separate console thread, and
//! that thread publishes serialized frames here. Frames fan out to every viewer
//! through a bounded broadcast channel; a viewer that falls behind receives a
//! fresh snapshot instead of the frames it missed.
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::header,
    response::{Html, IntoResponse, Response},
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use tokio::sync::broadcast;

/// The console page, compiled into the binary so it needs no asset path or CDN.
pub const PAGE: &str = include_str!("../../assets/console.html");
/// Shared page shell, palette and chart styles for every built-in page
/// (`/assets/cuteafd-ui.css`).
pub const UI_CSS: &str = include_str!("../../assets/cuteafd-ui.css");
/// Shared page shell, formatting and SVG chart primitives (`/assets/cuteafd-ui.js`, `window.CuteUI`).
pub const UI_JS: &str = include_str!("../../assets/cuteafd-ui.js");
/// The header logo for the dark pages (`/assets/cuteafd-logo.svg`).
pub const LOGO: &str = include_str!("../../../../../assets/brand/cuteafd-logo-color-dark.svg");
/// The square swift mark, the pages' favicon (`/assets/cuteafd-mark.svg`).
pub const MARK: &str = include_str!("../../../../../assets/brand/cuteafd-mark-color-dark.svg");

const DISABLED: &str = r#"{"type":"snapshot","disabled":true}"#;
const STARTING: &str = r#"{"type":"snapshot","starting":true}"#;

pub struct ConsoleHub {
    viewers: AtomicUsize,
    text: bool,
    enabled: bool,
    frames: broadcast::Sender<Arc<str>>,
    snapshot: Mutex<Arc<str>>,
}

impl ConsoleHub {
    /// A hub the serving worker publishes to. `text` allows token text on the wire.
    pub fn new(text: bool) -> Arc<Self> {
        Self::build(true, text, STARTING)
    }
    /// A hub with no producer: the page loads and reports that the feed is off.
    pub fn disabled() -> Arc<Self> {
        Self::build(false, false, DISABLED)
    }
    fn build(enabled: bool, text: bool, snapshot: &str) -> Arc<Self> {
        let (frames, _) = broadcast::channel(256);
        Arc::new(Self {
            viewers: AtomicUsize::new(0),
            text,
            enabled,
            frames,
            snapshot: Mutex::new(Arc::from(snapshot)),
        })
    }
    /// Connected console sockets. Producers skip all per-round work at zero.
    #[inline]
    pub fn viewers(&self) -> usize {
        self.viewers.load(Ordering::Relaxed)
    }
    pub fn text_enabled(&self) -> bool {
        self.text
    }
    pub fn publish(&self, frame: String) {
        // No receivers is the normal idle case, not an error.
        let _ = self.frames.send(Arc::from(frame));
    }
    pub fn set_snapshot(&self, snapshot: String) {
        if let Ok(mut slot) = self.snapshot.lock() {
            *slot = Arc::from(snapshot);
        }
    }
    pub fn snapshot(&self) -> Arc<str> {
        self.snapshot
            .lock()
            .map(|slot| slot.clone())
            .unwrap_or_else(|_| Arc::from("{}"))
    }
}

struct Viewer(Arc<ConsoleHub>);
impl Drop for Viewer {
    fn drop(&mut self) {
        self.0.viewers.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Serve the compiled-in page, or, for page development, the file named by
/// `CUTEAFD_CONSOLE_PAGE`, re-read on every load.
pub(super) async fn page() -> Response {
    let page = match std::env::var_os("CUTEAFD_CONSOLE_PAGE") {
        Some(path) => match tokio::fs::read_to_string(&path).await {
            Ok(page) => page,
            Err(error) => {
                tracing::warn!(%error, path = %path.to_string_lossy(), "console page override unreadable");
                PAGE.to_string()
            }
        },
        None => PAGE.to_string(),
    };
    ([(header::CACHE_CONTROL, "no-cache")], Html(page)).into_response()
}

/// A shared UI asset, or for page development the same-named file in the
/// directory `CUTEAFD_CONSOLE_ASSETS`, re-read on every load.
async fn asset(name: &str, builtin: &'static str, content_type: &'static str) -> Response {
    let body = match std::env::var_os("CUTEAFD_CONSOLE_ASSETS") {
        Some(dir) => tokio::fs::read_to_string(std::path::Path::new(&dir).join(name)).await
            .unwrap_or_else(|_| builtin.to_string()),
        None => builtin.to_string(),
    };
    ([(header::CONTENT_TYPE, content_type), (header::CACHE_CONTROL, "no-cache")], body).into_response()
}

pub(super) async fn ui_css() -> Response {
    asset("cuteafd-ui.css", UI_CSS, "text/css; charset=utf-8").await
}

pub(super) async fn ui_js() -> Response {
    asset("cuteafd-ui.js", UI_JS, "text/javascript; charset=utf-8").await
}

pub(super) async fn logo() -> Response {
    asset("cuteafd-logo.svg", LOGO, "image/svg+xml").await
}

pub(super) async fn mark() -> Response {
    asset("cuteafd-mark.svg", MARK, "image/svg+xml").await
}

pub(super) async fn snapshot(State(hub): State<Arc<ConsoleHub>>) -> Response {
    (
        [
            (header::CONTENT_TYPE, "application/json"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        hub.snapshot().to_string(),
    )
        .into_response()
}

pub(super) async fn socket(State(hub): State<Arc<ConsoleHub>>, upgrade: WebSocketUpgrade) -> Response {
    upgrade.on_upgrade(move |socket| serve(hub, socket))
}

async fn serve(hub: Arc<ConsoleHub>, mut socket: WebSocket) {
    // Subscribe before counting the viewer so no frame published after the
    // producer sees this viewer can be missed.
    let mut frames = hub.frames.subscribe();
    let _viewer = hub.enabled.then(|| {
        hub.viewers.fetch_add(1, Ordering::Relaxed);
        Viewer(hub.clone())
    });
    if socket.send(Message::Text(hub.snapshot().to_string())).await.is_err() {
        return;
    }
    loop {
        tokio::select! {
            frame = frames.recv() => {
                let text = match frame {
                    Ok(frame) => frame.to_string(),
                    Err(broadcast::error::RecvError::Lagged(_)) => hub.snapshot().to_string(),
                    Err(broadcast::error::RecvError::Closed) => break,
                };
                if socket.send(Message::Text(text)).await.is_err() { break; }
            }
            incoming = socket.recv() => match incoming {
                Some(Ok(Message::Close(_))) | Some(Err(_)) | None => break,
                Some(Ok(_)) => {}
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_is_self_contained() {
        assert!(PAGE.contains("/v1/console"));
        assert!(PAGE.contains("/assets/cuteafd-ui.css") && PAGE.contains("/assets/cuteafd-ui.js"));
        assert!(UI_JS.contains("window.CuteUI") && UI_JS.contains("/bench"));
        // The console must work on hosts without internet access.
        assert!(LOGO.starts_with("<svg") && MARK.starts_with("<svg") && !LOGO.contains("href="));
        for text in [PAGE, UI_CSS, UI_JS] {
            for external in ["http://", "https://"] {
                assert!(!text.contains(&format!("src=\"{external}")), "page loads an external script");
                assert!(!text.contains(&format!("href=\"{external}")), "page loads an external stylesheet");
                assert!(!text.contains(&format!("url({external}")), "page loads an external resource");
            }
        }
    }

    #[tokio::test]
    async fn viewers_are_counted_only_while_connected() {
        let hub = ConsoleHub::new(false);
        assert_eq!(hub.viewers(), 0);
        let viewer = {
            hub.viewers.fetch_add(1, Ordering::Relaxed);
            Viewer(hub.clone())
        };
        assert_eq!(hub.viewers(), 1);
        drop(viewer);
        assert_eq!(hub.viewers(), 0);
        let mut frames = hub.frames.subscribe();
        hub.publish("{\"type\":\"frame\"}".into());
        assert_eq!(&*frames.recv().await.unwrap(), "{\"type\":\"frame\"}");
    }
}
