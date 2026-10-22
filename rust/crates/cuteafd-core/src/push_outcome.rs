//! How the last startup weight push ended, for `/v1/stats` and cards. A
//! fallback to the mount is never silent: its ranks and causes are kept here
//! and logged at WARN where they happen.
use serde_json::{json, Value};
use std::sync::{Mutex, OnceLock};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Committed { group: u64, ranks: usize },
    Fallback { group: Option<u64>, failures: Vec<(usize, String)> },
}

fn state() -> &'static Mutex<Option<Outcome>> {
    static STATE: OnceLock<Mutex<Option<Outcome>>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(None))
}

pub fn record(outcome: Outcome) {
    if let Ok(mut state) = state().lock() { *state = Some(outcome); }
}

pub fn last() -> Option<Outcome> { state().lock().ok().and_then(|state| state.clone()) }

/// `null` until a push group was offered.
pub fn current() -> Value {
    match last() {
        None => Value::Null,
        Some(Outcome::Committed { group, ranks }) => json!({"status": "committed", "group": group, "ranks": ranks}),
        Some(Outcome::Fallback { group, failures }) => json!({"status": "fallback", "group": group,
            "failures": failures.iter().map(|(rank, reason)| json!({"rank": rank, "reason": reason})).collect::<Vec<_>>()}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fallback_reports_ranks_and_causes() {
        record(Outcome::Fallback { group: Some(7), failures: vec![(2, "coverage".into())] });
        assert_eq!(current(), json!({"status": "fallback", "group": 7, "failures": [{"rank": 2, "reason": "coverage"}]}));
        record(Outcome::Committed { group: 8, ranks: 8 });
        assert_eq!(current()["status"], "committed");
    }
}
