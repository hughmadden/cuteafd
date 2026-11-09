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

const CORPUS_VARIANTS: u64 = 16;

/// Keep one nonce throughout sizing. If a tokenizer skips the target at every
/// nearby character boundary, try another deterministic offset in the same bank.
/// No inference, cache retention or timed work happens here.
pub fn prompt(client: &Client, seed: u64, estimate: usize, target: u64, instruction: &str) -> Result<String> {
    let prefix = format!("[{}] {instruction}\n\n", crate::text::nonce());
    sized_prompt(&prefix, seed, estimate, target, 0,
        |text| client.count_chat_tokens(&crate::baseline::plain(client, text, 1)))
}

fn sized_prompt(prefix: &str, seed: u64, estimate: usize, target: u64, tolerance: u64,
    mut count: impl FnMut(&str) -> Result<u64>) -> Result<String> {
    let variants = if tolerance == 0 { CORPUS_VARIANTS } else { 1 };
    for variant in 0..variants {
        let mut reference_tokens = estimate.max(64);
        // Three seeds rotate back to the same corpus kind, with a new offset.
        let variant_seed = seed.wrapping_add(variant * 3);
        for _ in 0..20 {
            let corpus = crate::text::corpus(variant_seed, reference_tokens);
            let whole = format!("{prefix}{corpus}");
            let n = count(&whole)?;
            if n.abs_diff(target) <= tolerance { return Ok(whole); }
            if n > target {
                if let Some(body) = prefix_within(&corpus, target, tolerance, |body| count(&format!("{prefix}{body}")))? {
                    return Ok(format!("{prefix}{body}"));
                }
                break;
            }
            reference_tokens = reference_tokens.checked_mul(2).context("prefill corpus size overflow")?;
        }
    }
    if tolerance != 0 {
        bail!("no prefill within {tolerance} token of {target} found in the diagnostic corpus");
    }
    bail!("no exact {target}-token prefill found after {CORPUS_VARIANTS} deterministic corpus variants; refusing an approximate prefill")
}

/// Size both requests before any inference, so an optional diagnostic can be
/// skipped without partially warming or timing it.
pub struct SizedPrompt {
    pub text: String,
    pub tokens: u64,
}

pub fn prompts(client: &Client, seeds: (u64, u64), estimate: usize, target: u64,
    max_context: u64, instruction: &str, diagnostic: bool) -> Result<(SizedPrompt, SizedPrompt)> {
    let tolerance = u64::from(diagnostic);
    ensure!(target.saturating_add(tolerance) <= max_context.saturating_sub(64),
        "{target}-token prefill (tolerance {tolerance}) exceeds max context {max_context} with output headroom");
    let size = |seed| -> Result<SizedPrompt> {
        let text = if diagnostic {
            let prefix = format!("[{}] {instruction}\n\n", crate::text::nonce());
            sized_prompt(&prefix, seed, estimate, target, tolerance,
                |text| client.count_chat_tokens(&crate::baseline::plain(client, text, 1)))?
        } else {
            prompt(client, seed, estimate, target, instruction)?
        };
        let tokens = client.count_chat_tokens(&crate::baseline::plain(client, &text, 1))?;
        ensure!(tokens.abs_diff(target) <= tolerance, "prefill count changed during sizing");
        Ok(SizedPrompt { text, tokens })
    };
    Ok((size(seeds.0)?, size(seeds.1)?))
}

/// Token counts are not strictly monotone at character boundaries (a new
/// character can merge earlier tokens). Bisect for proximity, then inspect a
/// bounded local neighborhood rather than assuming a monotone tokenizer.
fn prefix_within(text: &str, target: u64, tolerance: u64,
    mut count: impl FnMut(&str) -> Result<u64>) -> Result<Option<String>> {
    let offsets: Vec<usize> = text.char_indices().map(|(i, _)| i).chain(std::iter::once(text.len())).collect();
    let mut seen = std::collections::BTreeMap::new();
    let (mut lo, mut hi) = (0, offsets.len() - 1);
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        let n = count(&text[..offsets[mid]])?;
        seen.insert(mid, n);
        if n.abs_diff(target) <= tolerance { return Ok(Some(text[..offsets[mid]].to_string())); }
        if n < target { lo = mid + 1; }
        else if mid == 0 { break; }
        else { hi = mid - 1; }
    }
    let center = lo.min(offsets.len() - 1);
    for index in center.saturating_sub(128)..=(center + 128).min(offsets.len() - 1) {
        let n = match seen.get(&index) { Some(n) => *n, None => count(&text[..offsets[index]])? };
        if n.abs_diff(target) <= tolerance { return Ok(Some(text[..offsets[index]].to_string())); }
    }
    Ok(None)
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
            let prompt = prefix_within(&text, target, 0, count).unwrap().unwrap();
            assert_eq!(count(&prompt).unwrap(), target);
            assert!(text.starts_with(&prompt));
        }
    }

    #[test]
    fn diagnostic_accepts_first_neighbor_without_switching_corpus() {
        let prefix = "[fixed] instruction\n\n";
        let original = crate::text::corpus(98, 65536);
        let count = |text: &str| (text.strip_prefix(prefix).unwrap().chars().count() as u64 / 3) * 2;
        let mut queries = 0;
        let mut last_count = 0;
        let text = sized_prompt(prefix, 98, 8192, 8297, 1, |text| {
            queries += 1;
            assert!(original.starts_with(text.strip_prefix(prefix).unwrap()));
            last_count = count(text);
            Ok(last_count)
        }).unwrap();
        assert!(matches!(count(&text), 8296 | 8298));
        assert_eq!(count(&text), last_count);
        assert!(queries < 30, "stop at the first acceptable prefix: {queries} queries");
    }

    #[test]
    fn diagnostic_accepts_either_neighbor_but_not_two_tokens_away() {
        for actual in [8295, 8297] {
            let text = prefix_within("abcdef", 8296, 1, |_| Ok(actual)).unwrap().unwrap();
            assert_eq!(text, "abc"); // first bisection candidate, no exact-first scan
        }
        assert!(prefix_within("abcdef", 8296, 1, |_| Ok(8298)).unwrap().is_none());
    }

    #[test]
    fn exact_prompt_queries_fake_nonlinear_count_endpoint() {
        fake_count_endpoint(false, false);
    }

    #[test]
    fn exact_prompt_switches_corpus_after_a_tokenizer_gap() {
        fake_count_endpoint(true, false);
    }

    #[test]
    fn exact_prompt_fails_after_all_variants_miss() {
        fake_count_endpoint(true, true);
    }

    fn fake_count_endpoint(gap: bool, all_miss: bool) {
        use std::io::{BufRead, BufReader, Read, Write};
        use serde_json::{json, Value};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop = done.clone();
        let original = crate::text::corpus(99, 64);
        let original_start = original.chars().take(32).collect::<String>();
        let count = move |text: &str| {
            let n = 17 + text.split_whitespace().map(|w| w.chars().count().div_ceil(3) as u64).sum::<u64>();
            let body = text.split_once("\n\n").unwrap().1;
            if gap && (all_miss || body.is_empty() || original_start.starts_with(body) || body.starts_with(&original_start)) {
                // Every original-corpus prefix jumps from target - 1 to target + 1.
                if n >= CARD_TOKENS { n + 1 } else { n }
            } else { n }
        };
        let expected = count.clone();
        let server = std::thread::spawn(move || {
            let mut requests = 0;
            let mut prefix = None;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let (mut socket, _) = match listener.accept() {
                    Ok(connection) => connection,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(std::time::Duration::from_millis(1)); continue;
                    }
                    Err(e) => panic!("{e}"),
                };
                socket.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
                let mut reader = BufReader::new(&mut socket);
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                assert!(line.starts_with("POST /v1/messages/count_tokens "));
                let mut length = 0;
                loop {
                    line.clear(); reader.read_line(&mut line).unwrap();
                    if line == "\r\n" { break; }
                    if let Some(value) = line.to_lowercase().strip_prefix("content-length:") {
                        length = value.trim().parse::<usize>().unwrap();
                    }
                }
                let mut bytes = vec![0; length]; reader.read_exact(&mut bytes).unwrap();
                let body: Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(body["thinking"]["type"], "disabled");
                let text = body["messages"][0]["content"].as_str().unwrap();
                let nonce_prefix = text.split_once("\n\n").unwrap().0;
                assert_eq!(prefix.get_or_insert_with(|| nonce_prefix.to_string()), nonce_prefix);
                let reply = json!({"input_tokens":count(text)}).to_string();
                write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}", reply.len()).unwrap();
                requests += 1;
            }
            requests
        });
        let client = Client::new(&format!("http://{address}"), None, Default::default());
        let result = prompt(&client, 99, 300, CARD_TOKENS, "Reply with the single word OK.");
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(server.join().unwrap() > 2);
        if all_miss {
            let error = result.unwrap_err().to_string();
            assert!(error.contains("8192-token") && error.contains("16 deterministic corpus variants"), "{error}");
        } else {
            let text = result.unwrap();
            assert_eq!(expected(&text), CARD_TOKENS);
            if gap {
                assert!(!original.starts_with(text.split_once("\n\n").unwrap().1));
            }
        }
    }

    #[test]
    fn unreachable_length_and_endpoint_errors_fail_loudly() {
        assert!(prefix_within(&"x".repeat(1000), 101, 0, |s| Ok(s.len() as u64 * 2)).unwrap().is_none());
        assert!(prefix_within("abc", 1, 0, |_| bail!("endpoint unavailable")).unwrap_err().to_string().contains("endpoint unavailable"));
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
