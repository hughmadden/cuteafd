//! PLE n-gram hashing (`Qwen4ExpTextNGramEmbedding`): each token's bigram and
//! trigram contexts hash into `heads_per_ngram` table rows per n-gram order.
//!
//! Context tokens never cross an EOS: at shift `s`, a position reads the token
//! `s` back unless an EOS lies strictly between them, in which case it reads
//! EOS (the reference's `_shift_right_ignore_eos` over the sequence history,
//! which starts as `ngram_size - 1` EOS tokens). Hash of order `n`:
//! `(t_0 m_0 ^ t_1 m_1 ^ ... ^ t_{n-1} m_{n-1}) mod size_h + offset_h` (i64,
//! the products stay below 2^63 by construction of the multipliers).
use anyhow::{ensure, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NgramHasher {
    pub multipliers: Vec<i64>,
    pub sizes: Vec<i64>,
    pub offsets: Vec<i64>,
    pub heads_per_ngram: usize,
    pub eos: u32,
}

/// The last `ngram_size - 1` tokens of a sequence (EOS before its start).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NgramHistory(pub Vec<u32>);

const MASK64: u128 = (1u128 << 64) - 1;
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const PRIME_1: u64 = 10007;

fn splitmix64(value: u64) -> u64 {
    let mut v = value.wrapping_add(SPLITMIX_GAMMA);
    v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^ (v >> 31)
}

fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value % 2 == 0 {
        return value == 2;
    }
    let mut d = 3;
    while d * d <= value {
        if value % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

impl NgramHasher {
    /// The reference's construction from the config (`seed` 1234 by default):
    /// per-layer odd multipliers from splitmix64, head sizes the successive
    /// primes above `vocab_base - 1`.
    pub fn from_config(vocab: usize, vocab_base: u64, ngram_size: usize, heads_per_ngram: usize, ple_index: usize,
        seed: u64, eos: u32) -> Self {
        let max_long = (1u64 << 63) - 1;
        let half = (max_long / (vocab.max(1) as u64) / 2).max(1);
        let base = (u128::from(seed) + u128::from(PRIME_1) * ple_index as u128) & MASK64;
        let multipliers = (0..ngram_size).map(|i| {
            let value = ((base + u128::from(SPLITMIX_GAMMA) * (i as u128 + 1)) & MASK64) as u64;
            (2 * (splitmix64(value) % half) + 1) as i64
        }).collect();
        let heads = (ngram_size - 1) * heads_per_ngram;
        let (mut sizes, mut offsets, mut total, mut prime) = (Vec::new(), Vec::new(), 0i64, vocab_base - 1);
        // Head `h` of PLE layer `ple_index` takes the (ple_index * heads + h + 1)-th prime after base - 1.
        for _ in 0..ple_index * heads {
            prime += 1;
            while !is_prime(prime) {
                prime += 1;
            }
        }
        for _ in 0..heads {
            prime += 1;
            while !is_prime(prime) {
                prime += 1;
            }
            sizes.push(prime as i64);
            offsets.push(total);
            total += prime as i64;
        }
        Self { multipliers, sizes, offsets, heads_per_ngram, eos }
    }

    pub fn heads(&self) -> usize {
        self.sizes.len()
    }

    pub fn ngram_size(&self) -> usize {
        self.multipliers.len()
    }

    /// Total table rows the ids address.
    pub fn rows(&self) -> i64 {
        self.sizes.iter().sum()
    }

    pub fn start(&self) -> NgramHistory {
        NgramHistory(vec![self.eos; self.ngram_size() - 1])
    }

    /// Appends `tokens` to `history`, writing `heads()` table rows per token into `out`.
    pub fn hash(&self, history: &mut NgramHistory, tokens: &[u32], out: &mut Vec<i64>) -> Result<()> {
        let context = self.ngram_size() - 1;
        ensure!(history.0.len() == context, "n-gram history must hold {context} tokens");
        let mut h: Vec<u32> = history.0.clone();
        h.extend_from_slice(tokens);
        for q in context..h.len() {
            // shifted[s]: the token s back, or EOS when an EOS lies strictly between.
            let mut shifted = vec![h[q]; self.ngram_size()];
            let mut blocked = false;
            for s in 1..self.ngram_size() {
                shifted[s] = if blocked { self.eos } else { h[q - s] };
                blocked |= h[q - s] == self.eos;
            }
            for n in 2..=self.ngram_size() {
                let mut mixed = i64::from(shifted[0]) * self.multipliers[0];
                for p in 1..n {
                    mixed ^= i64::from(shifted[p]) * self.multipliers[p];
                }
                let first = (n - 2) * self.heads_per_ngram;
                for head in first..first + self.heads_per_ngram {
                    out.push(mixed.rem_euclid(self.sizes[head]) + self.offsets[head]);
                }
            }
        }
        history.0 = h[h.len() - context..].to_vec();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eos_blocks_context() -> Result<()> {
        let hasher = NgramHasher { multipliers: vec![3, 5, 7], sizes: vec![1_000_003; 4], offsets: vec![0, 1, 2, 3],
            heads_per_ngram: 2, eos: 9 };
        let mut history = hasher.start();
        let mut ids = Vec::new();
        hasher.hash(&mut history, &[1, 9, 2, 4], &mut ids)?;
        // Token 2 follows an EOS: its bigram and trigram contexts are all EOS.
        let bigram = |a: i64, b: i64| (a * 3 ^ b * 5).rem_euclid(1_000_003);
        let trigram = |a: i64, b: i64, c: i64| (a * 3 ^ b * 5 ^ c * 7).rem_euclid(1_000_003);
        assert_eq!(ids[2 * 4], bigram(2, 9));
        assert_eq!(ids[2 * 4 + 2], trigram(2, 9, 9) + 2);
        assert_eq!(ids[3 * 4 + 2], trigram(4, 2, 9) + 2);
        assert_eq!(history.0, vec![2, 4]);
        // Streaming in pieces equals one pass.
        let (mut a, mut b) = (hasher.start(), Vec::new());
        hasher.hash(&mut a, &[1, 9], &mut b)?;
        hasher.hash(&mut a, &[2, 4], &mut b)?;
        assert_eq!(b, ids);
        Ok(())
    }

    #[test]
    fn qwen38_construction() {
        // Values stored in Qwen3.8-Flash-Next (layer 1 PLE: layer_multipliers, head sizes).
        let hasher = NgramHasher::from_config(248_320, 20_000_000, 3, 8, 0, 1234, 248_044);
        assert_eq!(hasher.heads(), 16);
        assert!(hasher.multipliers.iter().all(|m| m % 2 == 1));
        assert!(hasher.sizes.windows(2).all(|w| w[0] < w[1]));
        assert!(hasher.sizes[0] >= 20_000_000);
    }
}
