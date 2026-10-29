//! Mapped-table read backends: a switch to the mmap fallback mid-run halves
//! PLE/Engram gather speed, so reports flag the affected phases like fabric
//! contamination instead of silently comparing unequal runs.
use crate::report::Report;
use serde_json::Value;
use std::collections::BTreeMap;

/// What a table reports when io_uring is unavailable or failed.
const FALLBACK: &str = "mmap-fallback";

/// Caveat lines and whether any of them is contamination (the A/B must rerun).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Findings {
    pub caveats: Vec<String>,
    pub contaminated: bool,
}

/// One measured phase's display name and its mapped-table stats.
pub type Phase = (String, Vec<Value>);

/// Every measured phase's mapped tables in run order: each panel pass and the
/// sub-phases it recorded (decode contents, concurrency levels).
pub fn phases(report: &Report) -> Vec<Phase> {
    let mut out = Vec::new();
    for panel in &report.panels {
        for (index, pass) in panel.passes.iter().enumerate() {
            collect(pass, &format!("{}/pass{}", panel.id, index + 1), &mut out);
        }
    }
    out
}

/// Nested measured phases first (recording order), the pass aggregate last.
/// `mapped_tables` is added to a pass after its panel ran, so a pass's own
/// backend is its read-after state.
fn collect(value: &Value, path: &str, out: &mut Vec<Phase>) {
    match value {
        Value::Object(fields) => {
            for (key, child) in fields {
                if key != "mapped_tables" && (child.is_object() || child.is_array()) {
                    collect(child, &format!("{path}/{key}"), out);
                }
            }
            if let Some(tables) = fields.get("mapped_tables").and_then(Value::as_array) {
                let name = fields.get("fabric").and_then(|fabric| fabric.get("name")).and_then(Value::as_str)
                    .unwrap_or(path);
                out.push((name.to_string(), tables.clone()));
            }
        }
        Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                collect(item, &format!("{path}/{index}"), out);
            }
        }
        _ => {}
    }
}

/// Compare each table's backend with its first observation. A later change is
/// contamination (the affected A/B mixed gather speeds); mmap from the first
/// phase is one configuration note, because io_uring was never available.
pub fn findings(phases: &[Phase]) -> Findings {
    let mut first: BTreeMap<&str, &str> = BTreeMap::new();
    let mut findings = Findings::default();
    for (phase, tables) in phases {
        for table in tables {
            let (Some(name), Some(backend)) = (table["name"].as_str(), table["backend"].as_str()) else { continue };
            match first.get(name).copied() {
                None => {
                    first.insert(name, backend);
                    if backend == FALLBACK {
                        findings.caveats.push(format!(
                            "{name}: read backend {FALLBACK} from startup (io_uring unavailable); mapped gathers run slower — configuration, not contamination"));
                    }
                }
                Some(start) if start == backend => {}
                Some(start) => {
                    findings.contaminated = true;
                    findings.caveats.push(format!(
                        "{phase}: {name} read backend changed from {start} to {backend}; mapped gathers are slower, compare in a quiet window"));
                }
            }
        }
    }
    findings
}

/// [`findings`] over a whole report.
pub fn report_findings(report: &Report) -> Findings {
    findings(&phases(report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn phase(name: &str, backend: &str) -> Phase {
        (name.into(), vec![json!({"name": "engram", "backend": backend})])
    }

    #[test]
    fn a_backend_change_is_flagged_on_the_phase_where_it_appears() {
        let found = findings(&[phase("decode_content/0", "mincore-routed"),
            phase("decode_content/1", "mincore-routed"), phase("decode_content/2", "mmap-fallback")]);
        assert!(found.contaminated);
        assert_eq!(found.caveats.len(), 1);
        assert!(found.caveats[0].starts_with("decode_content/2: engram read backend changed from mincore-routed to mmap-fallback"),
            "{:?}", found.caveats);
    }

    #[test]
    fn a_stable_backend_produces_no_flag() {
        let found = findings(&[phase("a", "uring"), phase("b", "uring"), phase("c", "uring")]);
        assert_eq!(found, Findings::default());
    }

    #[test]
    fn mmap_from_startup_is_one_configuration_note_not_contamination() {
        let found = findings(&[phase("a", "mmap-fallback"), phase("b", "mmap-fallback")]);
        assert!(!found.contaminated);
        assert_eq!(found.caveats.len(), 1);
        assert!(found.caveats[0].contains("configuration, not contamination"), "{:?}", found.caveats);
    }
}
