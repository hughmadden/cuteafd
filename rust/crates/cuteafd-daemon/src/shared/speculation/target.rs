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

/// Consume borrowed logits/taps before committing target state. Consumers must
/// finish enqueueing device reads on the producer stream before returning, as
/// required by `HiddenTap`; a rejected consumer restores the exact checkpoint.
/// The accepted counts retain input rows only, never a sampled bonus token.
pub(crate) fn verify_commit<T: BlockVerifyTarget, R>(target: &mut T, slots: &[usize],
    request: VerifyRequest<'_>, consume: impl FnOnce(VerifyOutput<'_>) -> Result<(R, Vec<AcceptedPrefix>)>) -> Result<R> {
    let mut counts = BTreeMap::new();
    for &slot in slots {
        ensure!(counts.insert(slot, 0usize).is_none(), "duplicate transaction slot {slot}");
    }
    for row in request.rows.iter().filter(|row| row.token.is_some()) {
        let count = counts.get_mut(&row.slot).ok_or_else(|| anyhow::anyhow!("unmarked transaction slot {}", row.slot))?;
        *count += 1;
    }
    let mark = target.checkpoint(slots)?;
    let result = target.verify(&mark, request).and_then(consume).and_then(|(value, accepted)| {
        ensure!(accepted.len() == counts.len(), "transaction must retain every marked slot exactly once");
        for prefix in &accepted {
            let verified = counts.remove(&prefix.slot).ok_or_else(|| anyhow::anyhow!(
                "duplicate or unmarked accepted slot {}", prefix.slot))?;
            ensure!(prefix.rows <= verified, "accepted input count exceeds verified inputs");
        }
        target.commit(&mark, &accepted)?;
        Ok(value)
    });
    match result {
        Ok(value) => Ok(value),
        Err(error) => {
            if let Err(restore) = target.restore(&mark) {
                return Err(error.context(format!("target transaction rollback also failed: {restore:#}")));
            }
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct RecurrentState {
        accumulator: u32,
        convolution: [u32; 3],
        tokens: Vec<u32>,
    }

    impl RecurrentState {
        fn new(tokens: &[u32]) -> Self {
            let mut state = Self { accumulator: 0.25f32.to_bits(), convolution: [0; 3], tokens: vec![] };
            for &token in tokens { state.append(token); }
            state
        }

        fn append(&mut self, token: u32) {
            let previous = f32::from_bits(self.accumulator);
            self.accumulator = (previous * 0.9375 + token as f32 / 17.).to_bits();
            self.convolution.rotate_left(1);
            self.convolution[2] = token;
            self.tokens.push(token);
        }
    }

    struct RecurrentTarget {
        states: BTreeMap<usize, RecurrentState>,
        verified: Vec<VerifyRow>,
        taps: Vec<HiddenTap>,
        restores: usize,
        fail_verify: bool,
        fail_commit: bool,
    }

    impl RecurrentTarget {
        fn new() -> Self {
            Self { states: BTreeMap::from([(3, RecurrentState::new(&[2, 4])),
                (7, RecurrentState::new(&[9]))]), verified: vec![], taps: vec![],
                restores: 0, fail_verify: false, fail_commit: false }
        }
    }

    impl BlockVerifyTarget for RecurrentTarget {
        type Mark = BTreeMap<usize, RecurrentState>;

        fn checkpoint(&mut self, slots: &[usize]) -> Result<Self::Mark> {
            slots.iter().map(|slot| Ok((*slot, self.states.get(slot)
                .ok_or_else(|| anyhow::anyhow!("unknown test slot"))?.clone()))).collect()
        }

        fn verify<'a>(&'a mut self, mark: &Self::Mark, request: VerifyRequest<'_>) -> Result<VerifyOutput<'a>> {
            let starts: Vec<_> = mark.iter().map(|(&slot, state)| (slot, state.tokens.len())).collect();
            request.validate(64, 248320, &starts)?;
            self.verified = request.rows.iter().filter(|row| row.token.is_some()).copied().collect();
            for row in &self.verified {
                self.states.get_mut(&row.slot).unwrap().append(row.token.unwrap());
            }
            // Deliberately fail after mutation to exercise transaction rollback.
            ensure!(!self.fail_verify, "injected verify failure");
            self.taps = request.tap_layers.iter().map(|&layer| HiddenTap {
                layer, ptr: std::ptr::null(), rows: request.rows.len(), hidden: 5120,
                stride: 5128, stream: std::ptr::null_mut(),
            }).collect();
            Ok(VerifyOutput { logits: DeviceLogits { ptr: std::ptr::null(), rows: request.rows.len(),
                vocab: 248320, stride: 248320, stream: std::ptr::null_mut(), greedy: None }, taps: &self.taps })
        }

        fn restore(&mut self, mark: &Self::Mark) -> Result<()> {
            self.restores += 1;
            for (&slot, state) in mark { self.states.insert(slot, state.clone()); }
            Ok(())
        }

        fn commit(&mut self, mark: &Self::Mark, accepted: &[AcceptedPrefix]) -> Result<()> {
            self.restore(mark)?;
            for prefix in accepted {
                for row in self.verified.iter().filter(|row| row.slot == prefix.slot).take(prefix.rows) {
                    self.states.get_mut(&prefix.slot).unwrap().append(row.token.unwrap());
                }
            }
            ensure!(!self.fail_commit, "injected commit failure");
            Ok(())
        }
    }

    fn transaction_rows() -> [VerifyRow; 5] {
        [VerifyRow { slot: 3, position: 2, token: Some(11) },
            VerifyRow { slot: usize::MAX, position: usize::MAX, token: None },
            VerifyRow { slot: 7, position: 1, token: Some(12) },
            VerifyRow { slot: 3, position: 3, token: Some(13) },
            VerifyRow { slot: 7, position: 2, token: Some(14) }]
    }

    #[test]
    fn draft_transaction_retains_exact_recurrent_inputs_and_tap_order() {
        let mut target = RecurrentTarget::new();
        let rows = transaction_rows();
        let layers = [52, 4, 16, 28, 40];
        let result = verify_commit(&mut target, &[3, 7], VerifyRequest { rows: &rows, tap_layers: &layers }, |output| {
            assert_eq!(output.logits.rows, rows.len());
            assert_eq!(output.taps.iter().map(|tap| tap.layer).collect::<Vec<_>>(), layers);
            assert!(output.taps.iter().all(|tap| tap.rows == rows.len() && tap.stride == 5128));
            Ok((99, vec![AcceptedPrefix { slot: 7, rows: 0 }, AcceptedPrefix { slot: 3, rows: 1 }]))
        }).unwrap();
        assert_eq!(result, 99);
        assert_eq!(target.states[&3], RecurrentState::new(&[2, 4, 11]));
        assert_eq!(target.states[&7], RecurrentState::new(&[9]));
    }

    #[test]
    fn draft_transaction_failure_restores_exact_state() {
        let rows = transaction_rows();
        for failure in 0..3 {
            let mut target = RecurrentTarget::new();
            let original = target.states.clone();
            target.fail_verify = failure == 0;
            target.fail_commit = failure == 2;
            let result = verify_commit(&mut target, &[3, 7], VerifyRequest { rows: &rows, tap_layers: &[] }, |_| {
                ensure!(failure != 1, "injected consumer failure");
                Ok(((), vec![AcceptedPrefix { slot: 3, rows: 1 }, AcceptedPrefix { slot: 7, rows: 2 }]))
            });
            assert!(result.is_err());
            assert_eq!(target.states, original);
            assert!(target.restores > 0);
        }
    }

    #[test]
    fn draft_transaction_rejects_invalid_counts_and_bonus_rows() {
        let rows = transaction_rows();
        for accepted in [vec![AcceptedPrefix { slot: 3, rows: 3 }, AcceptedPrefix { slot: 7, rows: 0 }],
            vec![AcceptedPrefix { slot: 3, rows: 1 }],
            vec![AcceptedPrefix { slot: 3, rows: 1 }, AcceptedPrefix { slot: 3, rows: 1 }],
            vec![AcceptedPrefix { slot: 3, rows: 1 }, AcceptedPrefix { slot: 8, rows: 1 }]] {
            let mut target = RecurrentTarget::new();
            let original = target.states.clone();
            assert!(verify_commit(&mut target, &[3, 7], VerifyRequest { rows: &rows, tap_layers: &[] },
                |_| Ok(((), accepted))).is_err());
            assert_eq!(target.states, original);
        }
    }

    #[test]
    fn draft_transaction_rejects_unmarked_or_duplicate_slots_before_mutation() {
        let rows = transaction_rows();
        for slots in [&[3][..], &[3, 3, 7][..]] {
            let mut target = RecurrentTarget::new();
            let original = target.states.clone();
            assert!(verify_commit(&mut target, slots, VerifyRequest { rows: &rows, tap_layers: &[] },
                |_| -> Result<((), Vec<AcceptedPrefix>)> { panic!("invalid request reached consumer") }).is_err());
            assert_eq!(target.states, original);
            assert_eq!(target.restores, 0);
        }
    }

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
