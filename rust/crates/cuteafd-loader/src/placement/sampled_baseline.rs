//! Already-held physical occupancy moved from a sampled budget into a named owner.
//! Reserve overrides must preserve these bytes rather than treating them as headroom.
const SOURCE: &str = "sampled baseline: ";

pub fn source(family_detail: &str) -> String { format!("{SOURCE}{family_detail}") }
pub fn is_source(source: &str) -> bool { source.starts_with(SOURCE) }

#[cfg(test)]
mod tests {
    #[test]
    fn source_identity_does_not_match_anonymous_allowances() {
        assert!(super::is_source(&super::source("Dense Qwen context and selected modules")));
        assert!(!super::is_source("sampled baseline estimate"));
        assert!(!super::is_source("estimated runtime growth"));
    }
}
