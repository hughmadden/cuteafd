//! Time-resolved routing census: the daemon's trace beside its
//! `CUTEAFD_V41_ROUTE_CENSUS` file. Each window says how many rows chose
//! each expert of each routed layer since the previous window, so hotness
//! can be followed over time and replayed through tier policies, where the
//! census file holds only totals.
//!
//! JSON lines, appended: a `session` line when serving starts counting,
//! then one `window` line per window. A window closes at a drained serving
//! boundary once it spans at least the session's `window` decode and verify
//! steps, or where a request finishes, so with one request at a time every
//! window belongs to one request.
use super::expert_tier::{ExpertRouteProfile, ExpertTierError};
use serde::{Deserialize, Serialize};

/// What a serving session counts: the trace's shape and provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteTraceSession {
    /// The served model id.
    pub model: String,
    /// When counting began, RFC 3339 UTC.
    pub started: String,
    /// Routed layers, experts per layer and routes per row.
    pub layers: usize,
    pub experts: usize,
    pub topk: usize,
    /// Decode and verify steps a window spans before a boundary may close it.
    pub window: u64,
}

/// The routes counted between two drained boundaries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteWindow {
    /// Windows of the session, from 0.
    pub index: u64,
    /// The session's decode and verify step count when the window opened and closed.
    pub steps: [u64; 2],
    /// Decode and verify rows, and prefill rows, routed in the window.
    pub tokens: u64,
    pub prefill_tokens: u64,
    /// When the window opened and closed, RFC 3339 UTC.
    pub started: String,
    pub time: String,
    /// Requests active at any boundary of the window, and those that
    /// finished in it, ascending.
    pub requests: Vec<u64>,
    pub finished: Vec<u64>,
    /// Per routed layer and expert, the decode and verify rows that chose it.
    pub layers: Vec<Vec<u64>>,
    /// The same for prefill rows.
    pub prefill_layers: Vec<Vec<u64>>,
}

impl RouteWindow {
    /// Decode and verify steps the window spans.
    pub fn step_count(&self) -> u64 {
        self.steps[1].saturating_sub(self.steps[0])
    }

    /// The window's decode and verify counts as a planner profile.
    pub fn profile(&self) -> ExpertRouteProfile {
        ExpertRouteProfile { layers: self.layers.clone() }
    }
}

/// One trace line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RouteTraceLine {
    Session(RouteTraceSession),
    Window(RouteWindow),
}

/// A trace read back: each session with its windows in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteTrace {
    pub sessions: Vec<(RouteTraceSession, Vec<RouteWindow>)>,
}

impl RouteTrace {
    /// Reads a trace file. A torn last line (a server stopped mid-write) is
    /// dropped; any other malformed line, a window outside a session or a
    /// window of the wrong shape is an error.
    pub fn read(path: &std::path::Path) -> Result<Self, ExpertTierError> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| ExpertTierError::Profile(format!("reading {}: {error}", path.display())))?;
        Self::parse(&text).map_err(|error| match error {
            ExpertTierError::Profile(message) => ExpertTierError::Profile(format!("{}: {message}", path.display())),
        })
    }

    pub fn parse(text: &str) -> Result<Self, ExpertTierError> {
        let lines: Vec<_> = text.lines().enumerate().filter(|(_, line)| !line.trim().is_empty()).collect();
        let mut sessions: Vec<(RouteTraceSession, Vec<RouteWindow>)> = Vec::new();
        for (position, &(number, line)) in lines.iter().enumerate() {
            let parsed = match serde_json::from_str::<RouteTraceLine>(line) {
                Ok(parsed) => parsed,
                // Only the file's last line may be torn.
                Err(_) if position + 1 == lines.len() && !text.ends_with('\n') => break,
                Err(error) => return Err(ExpertTierError::Profile(format!("line {}: {error}", number + 1))),
            };
            match parsed {
                RouteTraceLine::Session(session) => sessions.push((session, Vec::new())),
                RouteTraceLine::Window(window) => {
                    let (session, windows) = sessions.last_mut().ok_or_else(||
                        ExpertTierError::Profile(format!("line {}: a window before any session", number + 1)))?;
                    let shaped = |planes: &[Vec<u64>]| planes.len() == session.layers
                        && planes.iter().all(|row| row.len() == session.experts);
                    if !shaped(&window.layers) || !shaped(&window.prefill_layers) {
                        return Err(ExpertTierError::Profile(format!(
                            "line {}: window {} is not {} layers of {} experts", number + 1, window.index,
                            session.layers, session.experts)));
                    }
                    windows.push(window);
                }
            }
        }
        Ok(Self { sessions })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> RouteTraceSession {
        RouteTraceSession { model: "org/model".into(), started: "2026-11-05T00:00:00Z".into(), layers: 2,
            experts: 3, topk: 1, window: 64 }
    }

    fn window(index: u64, counts: [u64; 2]) -> RouteWindow {
        RouteWindow { index, steps: [index * 64, index * 64 + 64], tokens: counts[0], prefill_tokens: 0,
            started: "a".into(), time: "b".into(), requests: vec![7], finished: vec![],
            layers: vec![vec![counts[0], 0, 0], vec![0, counts[1], 0]], prefill_layers: vec![vec![0; 3]; 2] }
    }

    fn lines(lines: &[RouteTraceLine]) -> String {
        lines.iter().map(|line| serde_json::to_string(line).unwrap() + "\n").collect()
    }

    #[test]
    fn a_trace_reads_back_by_session_and_drops_only_a_torn_last_line() {
        let text = lines(&[RouteTraceLine::Session(session()), RouteTraceLine::Window(window(0, [5, 6])),
            RouteTraceLine::Window(window(1, [7, 8])), RouteTraceLine::Session(session()),
            RouteTraceLine::Window(window(0, [1, 2]))]);
        assert!(text.lines().next().unwrap().starts_with(r#"{"kind":"session""#));
        let trace = RouteTrace::parse(&text).unwrap();
        assert_eq!(trace.sessions.len(), 2);
        assert_eq!(trace.sessions[0].1, vec![window(0, [5, 6]), window(1, [7, 8])]);
        assert_eq!((trace.sessions[1].1.len(), trace.sessions[0].1[1].step_count()), (1, 64));
        assert_eq!(trace.sessions[0].1[0].profile().layers, vec![vec![5, 0, 0], vec![0, 6, 0]]);
        // A server stopped mid-line leaves a torn last line.
        let torn = format!("{text}{{\"kind\":\"window\",\"ind");
        assert_eq!(RouteTrace::parse(&torn).unwrap(), trace);
        // Anywhere else it is an error, as are orphan and misshapen windows.
        assert!(RouteTrace::parse(&format!("{torn}\n")).is_err());
        assert!(RouteTrace::parse(&lines(&[RouteTraceLine::Window(window(0, [1, 1]))])).is_err());
        let mut wide = window(0, [1, 1]);
        wide.layers[1].push(0);
        assert!(RouteTrace::parse(&lines(&[RouteTraceLine::Session(session()), RouteTraceLine::Window(wide)])).is_err());
    }
}
