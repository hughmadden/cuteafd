//! Long texts by pieces. A Hugging Face tokenizer extracts its added tokens from the raw text
//! first, by content and leftmost-longest, and encodes the text between them independently: no
//! normalizer, pre-tokenizer or model state crosses an added token. A long text's ids are its
//! added tokens' ids and its pieces' ids in order, so a long piece seen before keeps its ids.
//! Chat templates put added tokens between messages: a repeated prompt, or a conversation that
//! grows, encodes only what is new.
//!
//! That holds for tokenizers whose added tokens match by content alone (no `single_word`,
//! `lstrip` or `rstrip`), that extract special tokens (`encode_special_tokens` off) and that
//! neither truncate nor pad; any other tokenizer encodes whole. The matcher is the tokenizer's
//! own (daachorse, leftmost-longest over the added tokens it does not normalize); a normalized
//! added token is left to the tokenizer within its piece, as it does itself.
use anyhow::Result;
use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder, MatchKind};
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, MutexGuard};

/// Texts shorter than this encode whole.
pub(super) const LONG_TEXT: usize = 64 << 10;
/// Pieces at least this long keep their ids.
const KEPT_PIECE: usize = 4 << 10;
/// Kept pieces' bytes (text and ids) before the least recently used goes.
const KEPT_BYTES: usize = 256 << 20;

pub(super) struct Pieces {
    added: DoubleArrayAhoCorasick<u32>,
    kept: Mutex<Kept>,
}

impl Pieces {
    /// `None` when `tokenizer` does not extract its added tokens by content alone.
    pub(super) fn new(tokenizer: &tokenizers::Tokenizer) -> Option<Self> {
        if tokenizer.get_truncation().is_some() || tokenizer.get_padding().is_some()
            || tokenizer.get_encode_special_tokens() {
            return None;
        }
        let tokens = tokenizer.get_added_tokens_decoder();
        if tokens.values().any(|token| token.single_word || token.lstrip || token.rstrip) {
            return None;
        }
        let patterns: Vec<(String, u32)> = tokens.into_iter().filter(|(_, token)| !token.normalized)
            .map(|(id, token)| (token.content, id)).collect();
        let added = DoubleArrayAhoCorasickBuilder::new().match_kind(MatchKind::LeftmostLongest)
            .build_with_values(patterns).ok()?;
        Some(Self { added, kept: Mutex::new(Kept::default()) })
    }

    /// `text`'s ids: each added token's id, and each piece's ids, kept or from `whole`.
    pub(super) fn encode(&self, text: &str, whole: impl Fn(&str) -> Result<Vec<u32>>) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        let mut start = 0;
        for found in self.added.leftmost_find_iter(text) {
            self.piece(&text[start..found.start()], &whole, &mut ids)?;
            ids.push(found.value());
            start = found.end();
        }
        self.piece(&text[start..], &whole, &mut ids)?;
        Ok(ids)
    }

    fn piece(&self, piece: &str, whole: &impl Fn(&str) -> Result<Vec<u32>>, ids: &mut Vec<u32>) -> Result<()> {
        if piece.len() < KEPT_PIECE {
            if !piece.is_empty() {
                ids.extend(whole(piece)?);
            }
            return Ok(());
        }
        let key = key(piece);
        if let Some(kept) = self.kept().get(key, piece) {
            ids.extend_from_slice(&kept);
            return Ok(());
        }
        let encoded: Arc<[u32]> = whole(piece)?.into();
        ids.extend_from_slice(&encoded);
        self.kept().insert(key, piece, encoded);
        Ok(())
    }

    fn kept(&self) -> MutexGuard<'_, Kept> {
        self.kept.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[cfg(test)]
    pub(super) fn kept_pieces(&self) -> usize {
        self.kept().pieces.len()
    }
}

fn key(piece: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    piece.hash(&mut hasher);
    hasher.finish()
}

/// Kept pieces by key: the text (a hit compares it whole), the ids and the last use, with the
/// uses in order for least-recently-used eviction.
#[derive(Default)]
struct Kept {
    pieces: HashMap<u64, (Box<str>, Arc<[u32]>, u64)>,
    uses: BTreeMap<u64, u64>,
    bytes: usize,
    clock: u64,
}

impl Kept {
    fn get(&mut self, key: u64, piece: &str) -> Option<Arc<[u32]>> {
        let (text, ids, used) = self.pieces.get_mut(&key)?;
        if **text != *piece {
            return None;
        }
        self.uses.remove(used);
        self.clock += 1;
        *used = self.clock;
        self.uses.insert(self.clock, key);
        Some(Arc::clone(ids))
    }

    fn insert(&mut self, key: u64, piece: &str, ids: Arc<[u32]>) {
        let size = size(piece, &ids);
        if size > KEPT_BYTES {
            return;
        }
        // A colliding key, or the same piece kept meanwhile: the newest stays.
        self.remove(key);
        self.clock += 1;
        self.pieces.insert(key, (piece.into(), ids, self.clock));
        self.uses.insert(self.clock, key);
        self.bytes += size;
        while self.bytes > KEPT_BYTES {
            let Some((_, &oldest)) = self.uses.first_key_value() else { break };
            self.remove(oldest);
        }
    }

    fn remove(&mut self, key: u64) {
        if let Some((text, ids, used)) = self.pieces.remove(&key) {
            self.uses.remove(&used);
            self.bytes -= size(&text, &ids);
        }
    }
}

fn size(text: &str, ids: &[u32]) -> usize {
    text.len() + std::mem::size_of_val(ids)
}
