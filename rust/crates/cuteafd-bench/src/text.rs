//! Deterministic real prompt material and unique tags for cold prefixes.
use std::sync::OnceLock;

pub const PROMPT_CONTENT: &str = "realistic-v1";
const CHUNK_TOKENS: usize = 32;

/// A short unique tag for one request.
pub fn nonce() -> String {
    // A fixed seed lets independent fresh servers run the same benchmark
    // prompts. The sequence still keeps unrelated probes from sharing a cache
    // prefix. Ordinary runs retain random nonces.
    if let Ok(seed) = std::env::var("CUTEAFD_BENCH_NONCE_SEED") {
        if !seed.is_empty() {
            static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let index = SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return format!("{seed}-{index:010}");
        }
    }
    uuid::Uuid::new_v4().simple().to_string()[..10].to_string()
}

struct Bank {
    text: &'static str,
    offsets: Vec<usize>,
}

fn banks() -> &'static [Bank; 3] {
    static BANKS: OnceLock<[Bank; 3]> = OnceLock::new();
    BANKS.get_or_init(|| {
        [
            (include_str!("../assets/corpus/code.txt"), include_str!("../assets/corpus/code.offsets.json")),
            (include_str!("../assets/corpus/agentic.txt"), include_str!("../assets/corpus/agentic.offsets.json")),
            (include_str!("../assets/corpus/prose.txt"), include_str!("../assets/corpus/prose.offsets.json")),
        ].map(|(text, offsets)| Bank {
            text, offsets: serde_json::from_str(offsets).expect("embedded corpus offsets"),
        })
    })
}

/// About `tokens` reference tokens of literal source code, tool transcripts or
/// technical prose. Consecutive seeds rotate the initial kind; the remaining
/// seed bits choose a starting chunk. Each bank is exhausted once before moving
/// to the next kind. Very long contexts repeat only after all three banks.
///
/// Fixed reference-token byte boundaries normalize density across kinds while
/// retaining the server's existing two-point length fit for its own tokenizer.
pub fn corpus(seed: u64, tokens: usize) -> String {
    let mut remaining = tokens.div_ceil(CHUNK_TOKENS);
    let mut kind = (seed % 3) as usize;
    let offset = (seed / 3).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut out = String::with_capacity(tokens.saturating_mul(4));
    while remaining > 0 {
        let bank = &banks()[kind];
        let chunks = bank.offsets.len() - 1;
        let start = (offset % chunks as u64) as usize;
        let take = remaining.min(chunks);
        let tail = take.min(chunks - start);
        out.push_str(&bank.text[bank.offsets[start]..bank.offsets[start + tail]]);
        if tail < take {
            out.push_str(&bank.text[..bank.offsets[take - tail]]);
        }
        remaining -= take;
        kind = (kind + 1) % 3;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn corpus_is_deterministic_and_nonce_is_unique() {
        assert_eq!(corpus(7, 8192), corpus(7, 8192));
        assert_ne!(corpus(7, 8192), corpus(10, 8192));
        assert_ne!(nonce(), nonce());
        assert!(corpus(0, 0).is_empty());
    }

    #[test]
    fn seeds_rotate_kinds_and_offsets() {
        for seed in 0..9 {
            let bank = &banks()[(seed % 3) as usize];
            let text = corpus(seed, 256);
            assert!(bank.text.contains(&text), "seed {seed}");
            assert_eq!(text, corpus(seed, 256));
        }
        assert_ne!(corpus(0, 256), corpus(3, 256));
    }

    #[test]
    fn length_targeting_uses_reference_token_chunks() {
        for seed in 0..9 {
            let bank = &banks()[(seed % 3) as usize];
            let chunks = bank.offsets.len() - 1;
            let offset = (seed / 3).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let start = (offset % chunks as u64) as usize;
            for tokens in [1usize, 256, 1024, 8192] {
                let take = tokens.div_ceil(CHUNK_TOKENS);
                let tail = take.min(chunks - start);
                let mut expected = bank.text[bank.offsets[start]..bank.offsets[start + tail]].to_string();
                if tail < take { expected.push_str(&bank.text[..bank.offsets[take - tail]]); }
                assert_eq!(corpus(seed, tokens), expected);
            }
        }
    }

    #[test]
    fn exhausts_distinct_banks_before_repeating_and_supports_long_context() {
        let total_chunks: usize = banks().iter().map(|b| b.offsets.len() - 1).sum();
        for seed in 0..3 {
            let first = corpus(seed, total_chunks * CHUNK_TOKENS);
            for bank in banks() {
                // Every byte appears once per bank cycle, including a rotated bank's two pieces.
                assert!(first.len() >= bank.text.len());
            }
            assert_eq!(first.len(), banks().iter().map(|b| b.text.len()).sum::<usize>());
            assert_eq!(corpus(seed, 2 * total_chunks * CHUNK_TOKENS), first.repeat(2));
            assert!(corpus(seed, 1 << 20).len() > first.len());
        }
    }

    #[test]
    fn manifest_attests_embedded_banks_and_minimum_distinct_size() {
        let manifest: serde_json::Value = serde_json::from_str(include_str!("../assets/corpus/manifest.json")).unwrap();
        assert_eq!(manifest["version"], PROMPT_CONTENT);
        for (i, bank) in banks().iter().enumerate() {
            let record = &manifest["banks"][i];
            assert!(record["reference_tokens"].as_u64().unwrap() >= 32768);
            assert_eq!(record["sha256"], format!("{:x}", Sha256::digest(bank.text.as_bytes())));
            assert_eq!(bank.offsets[0], 0);
            assert_eq!(bank.offsets.last(), Some(&bank.text.len()));
            assert!(bank.offsets.windows(2).all(|w| w[0] < w[1] && bank.text.is_char_boundary(w[1])));
        }
    }
}
