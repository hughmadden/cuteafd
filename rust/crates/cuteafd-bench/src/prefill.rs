//! Exact, cold-prefill prompt sizing through the served chat tokenizer.
use anyhow::{bail, ensure, Context, Result};
use crate::{client::Client, report::{PrefillShape, Setting}};

pub const CARD_TOKENS: u64 = 8192;

/// The resolved execution unit, not the multi-wave span or reserve.
pub fn unit_rows(settings: &[Setting]) -> Result<u64> {
    for name in ["prefill-units", "prefill-batch-tokens", "prefill-rows"] {
        if let Some(rows) = settings.iter().find(|s| s.name == name && s.source == "resolved")
            .and_then(|s| s.value.as_deref()?.parse::<u64>().ok()).filter(|n| *n > 0) {
            return Ok(rows);
        }
    }
    bail!("exact prefill requires the server's resolved prefill unit rows")
}

pub fn tail_tokens(unit: u64) -> Result<u64> {
    unit.checked_mul(3).and_then(|n| n.checked_add(40)).context("prefill tail length overflow")
}

pub fn shape(tokens: u64, unit: u64) -> PrefillShape {
    PrefillShape { unit_rows: unit, full_units: tokens / unit, tail_rows: tokens % unit }
}

pub fn validate(requested: u64, actual: u64) -> Result<()> {
    ensure!(actual == requested, "exact prefill requested {requested} prompt tokens but server reported {actual}; count endpoint and timed chat template must agree");
    Ok(())
}

/// Keep one nonce throughout sizing; only truncate the seeded corpus, at UTF-8
/// character boundaries. No inference, cache retention or timed work happens here.
pub fn prompt(client: &Client, seed: u64, estimate: usize, target: u64, instruction: &str) -> Result<String> {
    let prefix = format!("[{}] {instruction}\n\n", crate::text::nonce());
    let count = |text: &str| client.count_chat_tokens(&crate::baseline::plain(client, text, 1));
    let mut reference_tokens = estimate.max(64);
    for _ in 0..20 {
        let corpus = crate::text::corpus(seed, reference_tokens);
        let whole = format!("{prefix}{corpus}");
        let n = count(&whole)?;
        if n == target { return Ok(whole); }
        if n > target {
            return exact_prefix(&corpus, target, |body| count(&format!("{prefix}{body}")))
                .map(|body| format!("{prefix}{body}"));
        }
        reference_tokens = reference_tokens.checked_mul(2).context("prefill corpus size overflow")?;
    }
    bail!("could not bracket exact {target}-token prefill prompt")
}

/// Token counts are not strictly monotone at character boundaries (a new
/// character can merge earlier tokens). Bisect for proximity, then inspect a
/// bounded local neighborhood rather than assuming a monotone tokenizer.
fn exact_prefix(text: &str, target: u64, mut count: impl FnMut(&str) -> Result<u64>) -> Result<String> {
    let offsets: Vec<usize> = text.char_indices().map(|(i, _)| i).chain(std::iter::once(text.len())).collect();
    let mut seen = std::collections::BTreeMap::new();
    let (mut lo, mut hi) = (0, offsets.len() - 1);
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        let n = count(&text[..offsets[mid]])?;
        seen.insert(mid, n);
        if n == target { return Ok(text[..offsets[mid]].to_string()); }
        if n < target { lo = mid + 1; }
        else if mid == 0 { break; }
        else { hi = mid - 1; }
    }
    let center = lo.min(offsets.len() - 1);
    for index in center.saturating_sub(128)..=(center + 128).min(offsets.len() - 1) {
        let n = match seen.get(&index) { Some(n) => *n, None => count(&text[..offsets[index]])? };
        if n == target { return Ok(text[..offsets[index]].to_string()); }
    }
    bail!("no exact {target}-token corpus prefix found near the tokenizer boundary; refusing an approximate prefill")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_search_uses_nonlinear_tokenizer_and_character_boundaries() {
        let text = "aébc tokenization with repeated content. ".repeat(200);
        let count = |s: &str| -> Result<u64> {
            // Variable-length words and UTF-8, with a non-monotone merge.
            Ok(11 + s.split_whitespace().map(|w| w.chars().count().div_ceil(3) as u64).sum::<u64>()
                - u64::from(s.ends_with("tokenization")))
        };
        for target in [20, 127, 511, 819] {
            let prompt = exact_prefix(&text, target, count).unwrap();
            assert_eq!(count(&prompt).unwrap(), target);
            assert!(text.starts_with(&prompt));
        }
    }

    #[test]
    fn unreachable_length_and_endpoint_errors_fail_loudly() {
        assert!(exact_prefix(&"x".repeat(1000), 101, |s| Ok(s.len() as u64 * 2)).is_err());
        assert!(exact_prefix("abc", 1, |_| bail!("endpoint unavailable")).unwrap_err().to_string().contains("endpoint unavailable"));
    }

    #[test]
    fn card_is_exactly_8192_and_rejects_mismatch() {
        assert_eq!(CARD_TOKENS, 8192);
        validate(CARD_TOKENS, 8192).unwrap();
        assert!(validate(CARD_TOKENS, 8310).is_err());
        assert_eq!(tail_tokens(2752).unwrap(), 8296);
        assert_eq!(shape(8192, 2752), PrefillShape { unit_rows: 2752, full_units: 2, tail_rows: 2688 });
        assert_eq!(shape(8296, 2752).tail_rows, 40);
    }

    #[test]
    fn unit_uses_resolved_unit_not_span_or_launch_request() {
        let s = |name: &str, value: &str, source: &str| Setting { name: name.into(), value: Some(value.into()), source: source.into(), ..Default::default() };
        let settings = [s("prefill-rows", "8256", "resolved"), s("prefill-units", "2752", "resolved")];
        assert_eq!(unit_rows(&settings).unwrap(), 2752);
        assert_eq!(unit_rows(&[s("prefill-batch-tokens", "1792", "resolved")]).unwrap(), 1792);
        assert!(unit_rows(&[s("prefill-rows", "2752", "cli")]).is_err());
    }
}
