//! Target-side contract for block drafters. Attention stays causal; only the
//! drafter may attend bidirectionally to its context/noise block.
use crate::shared::token_io::DeviceLogits;
use anyhow::{ensure, Result};
use std::collections::BTreeMap;
use std::ffi::c_void;

/// One physical verify row. Padding has no token and must not access a slot,
/// recurrent state, KV, positions or prefix metadata (even for sentinel ids).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VerifyRow {
    pub slot: usize,
    pub position: usize,
    pub token: Option<u32>,
}

/// Requested taps are zero-based decoder block ids, after residual+FFN and
/// before the final norm. Their order is significant, not sorted by the engine.
pub(crate) struct VerifyRequest<'a> {
    pub rows: &'a [VerifyRow],
    pub tap_layers: &'a [usize],
}

impl VerifyRequest<'_> {
    /// Validate active rows against exact per-slot checkpoint positions.
    /// Interleaved slots are allowed; each slot advances in causal order.
    pub fn validate(&self, layers: usize, vocab: usize, starts: &[(usize, usize)]) -> Result<()> {
        ensure!(!self.rows.is_empty(), "target verify requires physical rows");
        let mut next = BTreeMap::new();
        for &(slot, position) in starts {
            ensure!(next.insert(slot, position).is_none(), "duplicate target slot {slot}");
        }
        for (i, &layer) in self.tap_layers.iter().enumerate() {
            ensure!(layer < layers, "target tap layer {layer} outside {layers} blocks");
            ensure!(!self.tap_layers[..i].contains(&layer), "duplicate target tap layer {layer}");
        }
        for row in self.rows {
            let Some(token) = row.token else { continue };
            ensure!((token as usize) < vocab, "target verify token {token} outside vocabulary {vocab}");
            let expected = next.get_mut(&row.slot)
                .ok_or_else(|| anyhow::anyhow!("target verify slot {} has no checkpoint", row.slot))?;
            ensure!(row.position == *expected, "target verify slot {} position {} expected {}",
                row.slot, row.position, *expected);
            *expected = expected.checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("target verify position overflow"))?;
        }
        Ok(())
    }
}

/// BF16 [physical rows, hidden] on the target's stream; padding rows are zero.
/// The producer owns the buffer until its next mutable operation. Consumers
/// must enqueue their reads on `stream` or establish an explicit device event.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HiddenTap {
    pub layer: usize,
    pub ptr: *const c_void,
    pub rows: usize,
    pub hidden: usize,
    /// BF16 elements between physical rows.
    pub stride: usize,
    pub stream: *mut c_void,
}

pub(crate) struct VerifyOutput<'a> {
    /// One FP32 next-token distribution per physical input row. Masked rows
    /// must not be selected; active row i predicts the token after input i.
    pub logits: DeviceLogits,
    /// Same order as VerifyRequest::tap_layers, including all physical rows.
    pub taps: &'a [HiddenTap],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AcceptedPrefix {
    pub slot: usize,
    /// Number of active INPUT rows retained from this verify. A sampled bonus
    /// token has not yet been evaluated and is not included in this count.
    pub rows: usize,
}

/// The exact mark owns every state component needed to resume: FP32 recurrence,
/// convolution tail, KV lengths/pages, token/RoPE/media metadata and slot epoch.
/// It cannot refer to mutable scratch that verify overwrites. Stale/cross-engine
/// marks and out-of-range commits must fail without changing live state.
///
/// Implementations drain queued users before recycling storage. Successful
/// restore and commit are byte-identical to prefill of the retained prefix;
/// truncating KV alone or approximately reconstructing recurrence is forbidden.
/// This is an interface, not a declaration of dense runtime capability.
pub(crate) trait BlockVerifyTarget {
    type Mark;

    fn checkpoint(&mut self, slots: &[usize]) -> Result<Self::Mark>;

    /// Validate the entire request before launch. On failure, restore `mark`
    /// before returning the error. The borrowed output prevents the next
    /// mutable target operation until the consumer has finished enqueueing reads.
    fn verify<'a>(&'a mut self, mark: &Self::Mark, request: VerifyRequest<'_>)
        -> Result<VerifyOutput<'a>>;

    /// Restore all marked slots, including exact prefix-cache state.
    fn restore(&mut self, mark: &Self::Mark) -> Result<()>;

    /// Restore the mark then retain exactly each slot's accepted active rows.
    /// Every marked slot must appear exactly once (zero means reject all).
    /// Intermediate recurrence must come from exact per-row marks or replay.
    fn commit(&mut self, mark: &Self::Mark, accepted: &[AcceptedPrefix]) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn post_block_taps_keep_requested_order() {
        let rows = [VerifyRow { slot: 0, position: 17, token: Some(3) }];
        let taps = [52, 4, 16, 28, 40];
        let request = VerifyRequest { rows: &rows, tap_layers: &taps };
        request.validate(64, 248320, &[(0, 17)]).unwrap();
        assert_eq!(request.tap_layers, &taps);
    }

    #[test]
    fn masked_padding_never_looks_up_slot_or_position() {
        let rows = [
            VerifyRow { slot: 3, position: 9, token: Some(4) },
            VerifyRow { slot: usize::MAX, position: usize::MAX, token: None },
            VerifyRow { slot: 7, position: 2, token: Some(5) },
            VerifyRow { slot: 3, position: 10, token: Some(6) },
        ];
        VerifyRequest { rows: &rows, tap_layers: &[] }.validate(32, 10, &[(3, 9), (7, 2)]).unwrap();
    }

    #[test]
    fn invalid_requests_fail_closed() {
        let valid = VerifyRow { slot: 3, position: 9, token: Some(4) };
        for row in [VerifyRow { slot: 2, ..valid }, VerifyRow { position: 10, ..valid },
            VerifyRow { token: Some(10), ..valid }] {
            assert!(VerifyRequest { rows: &[row], tap_layers: &[] }.validate(32, 10, &[(3, 9)]).is_err());
        }
        for taps in [&[32][..], &[4, 4][..]] {
            assert!(VerifyRequest { rows: &[valid], tap_layers: taps }.validate(32, 10, &[(3, 9)]).is_err());
        }
        assert!(VerifyRequest { rows: &[valid], tap_layers: &[] }.validate(32, 10, &[(3, 9), (3, 9)]).is_err());
        assert!(VerifyRequest { rows: &[], tap_layers: &[] }.validate(32, 10, &[]).is_err());
        let overflow = VerifyRow { position: usize::MAX, ..valid };
        assert!(VerifyRequest { rows: &[overflow], tap_layers: &[] }.validate(32, 10, &[(3, usize::MAX)]).is_err());
    }
}
