//! MiniMax's binding of the shared DFlash2 drafter ([`dflash::DflashDrafter`]) for
//! RadixArk/Minimax-M3.1-Flash-Dflash2: five Gemma-norm layers over a 4096-position ring, fed by
//! the target's FP32 residual after layers 3, 17, 31, 45 and 59 (the bundled DSpark's taps).
//!
//! A step taps the last [`dflash::DflashConfig::tap_rows`] rows of its batch on the target's
//! stream: a prefill call's tail, across its units, fills the ring (GLM 5.3 Flash's unit tail
//! window); a verify step taps every row. The drafter runs on that stream too, borrows the target's
//! BF16 head and gathers its block inputs from the target's embedding table. Its block attends
//! as SGLang serves the release ([`dflash::BlockMask::upstream`]).
use super::{native::Native, MiniMaxError};
use crate::shared::{fp8_linear::{Fp8Rows, Fp8Scales},
    speculation::dflash::{self, BlockMask, ContextRow, DflashConfig, DflashDrafter, Draft, DraftSeq, TargetHead},
    token_io::TokenEmbedding};
use anyhow::{ensure, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::families::{glm5::draft_representation::GlmDraftRepresentation, minimax_m3::dflash::DflashMemory,
    qwen4::Qwen4KvCache};
use std::{ffi::c_void, path::{Path, PathBuf}};

const HIDDEN: usize = 6144;
const VOCAB: usize = 200064;
const LAYERS: usize = 60;

/// The tap rows of a `rows`-row step: its last `tap_rows` rows (`first`, `count`).
fn tail(rows: usize, tap_rows: usize) -> (usize, usize) {
    let count = rows.min(tap_rows);
    (rows - count, count)
}

/// Batch rows `[unit_first, unit_first + unit_rows)` (a prefill unit's, or a whole step's) of a
/// `rows`-row step that lie in its tapped tail: `(row within the unit, tap row, count)`, if any.
/// Units tap disjoint tap rows, so the order in which units in flight tap does not matter.
fn unit_tail(unit_first: usize, unit_rows: usize, rows: usize, tap_rows: usize) -> Option<(usize, usize, usize)> {
    let (first, _) = tail(rows, tap_rows);
    let from = unit_first.max(first);
    (from < unit_first + unit_rows).then(|| (from - unit_first, from - first, unit_first + unit_rows - from))
}

/// The context rows of a `rows`-row single-request step from `start`: its tapped tail, at its
/// tap rows, into ring `slot`. Earlier rows of a long call would only be overwritten in the ring.
fn tail_rows(start: usize, rows: usize, slot: usize, tap_rows: usize) -> Vec<ContextRow> {
    let (first, count) = tail(rows, tap_rows);
    (0..count).map(|tap_row| ContextRow { tap_row, slot, position: start + first + tap_row }).collect()
}

/// What serve and golden admit before loading the drafter.
#[derive(Debug, Clone)]
pub(super) struct Contract {
    pub snapshot: PathBuf,
    pub cfg: DflashConfig,
    pub mask: BlockMask,
    pub representation: GlmDraftRepresentation,
    pub kv_cache: Qwen4KvCache,
    /// Context rings, one per drafting request.
    pub slots: usize,
    /// Sequences one draft step takes.
    pub sequences: usize,
    /// The drafter's resident bytes and scratch (the shared admission's).
    pub bytes: u64,
    /// The largest BF16 source matrix held while packing one FP8 weight (0 for BF16).
    pub staging: u64,
    /// The selector scores candidates in FP32, as SGLang's DFlash2 selector does (the release's
    /// serving reference); off, it rounds each score to BF16 as the GLM ports do.
    pub selector_fp32: bool,
}

impl Contract {
    /// The header-checked drafter at `snapshot` for `slots` rings (draft steps of as many
    /// sequences, at most 32), sized for a GPU of `sms` SMs. `head` is the target's
    /// `language_model.lm_head.weight` header, which the drafter borrows: anything but a BF16
    /// [200064, 6144] head is refused here, before startup. `causal` overrides the upstream
    /// mask's block causality (a diagnostic A/B; acceptance only, never output).
    #[allow(clippy::too_many_arguments)]
    pub fn open(snapshot: &Path, head: &cuteafd_loader::SafetensorsTensorMetadata, fp8: Option<bool>,
        kv_cache: Qwen4KvCache, slots: usize, sms: usize, causal: Option<bool>, selector_fp32: bool) -> Result<Self> {
        dflash::check_target_head_source(head, HIDDEN, VOCAB)?;
        let config = cuteafd_loader::plan::checkpoint::read_json(&snapshot.join("config.json"))?;
        let representation = GlmDraftRepresentation::from_fp8_option(fp8);
        // The planner's recipe: one memory for both ledgers.
        let memory = DflashMemory::new(&config, slots, sms as u64, representation, kv_cache)?;
        let cfg = dflash::check_headers(snapshot, fp8, slots, memory.sequences)?;
        ensure!(cfg.hidden == HIDDEN && cfg.vocab == VOCAB && cfg.taps.iter().all(|&l| l < LAYERS)
            && (cfg.mask_token as usize) < VOCAB && cfg.drafts() == 7,
            MiniMaxError::Unsupported("DFlash2 drafter geometry outside MiniMax M3 (hidden, vocabulary, taps, block of 8)"));
        let upstream = BlockMask::upstream(&config)?;
        let mask = BlockMask { causal: causal.unwrap_or(upstream.causal), ..upstream };
        Ok(Self { snapshot: snapshot.into(), mask, cfg, representation, kv_cache,
            slots, sequences: memory.sequences, bytes: memory.bytes, staging: memory.staging, selector_fp32 })
    }
}

pub(super) struct Dflash<'a> { drafter: DflashDrafter<'a> }

impl<'a> Dflash<'a> {
    /// Loads the drafter from its checkpoint bytes `file` onto `stream` (the target's, on the
    /// current device); `mask_row` is the target embedding of the mask token. Its workspace is
    /// allocated before serving.
    pub fn load(library: &'a NativeLibrary, stream: *mut c_void, contract: &Contract, file: Vec<u8>,
        mask_row: Vec<u8>) -> Result<Self> {
        let drafter = DflashDrafter::load(library, &contract.snapshot, file, stream, contract.slots,
            contract.sequences, mask_row, contract.mask, contract.representation, Fp8Scales::Amax, Fp8Rows::W8a16,
            contract.kv_cache)?;
        ensure!(drafter.cfg.window == contract.cfg.window && drafter.cfg.causal == contract.mask.causal,
            MiniMaxError::Invalid("DFlash2 drafter differs from its admission"));
        drafter.set_selector_fp32(contract.selector_fp32);
        drafter.prepare_workspace()?;
        tracing::info!(representation = contract.representation.name(), kv_cache = %contract.kv_cache,
            rings = contract.slots, sequences = contract.sequences, bytes = contract.bytes, ring = contract.cfg.window,
            causal = contract.mask.causal, row_window = contract.mask.row_window,
            selector_score = if contract.selector_fp32 { "fp32" } else { "bf16" }, "MiniMax DFlash2 drafter resident");
        Ok(Self { drafter })
    }

    /// Sequences one draft step takes.
    pub fn sequences(&self) -> usize { self.drafter.max_batch_sequences() }

    /// Rows of the block a draft runs: the anchor and its drafts.
    pub fn block(&self) -> usize { self.drafter.cfg.block }

    /// Taps target layer `layer`'s FP32 residual of unit rows `[unit_first, unit_first + unit_rows)`
    /// (`residual`: the unit's view) of a `rows`-row step, where they lie in its tapped tail.
    pub fn tap(&self, native: &Native<'a>, layer: usize, residual: CuteafdDeviceBuffer, unit_first: usize,
        unit_rows: usize, rows: usize) -> Result<()> {
        if self.drafter.tap_index(layer).is_none() { return Ok(()) }
        let Some((from, to, count)) = unit_tail(unit_first, unit_rows, rows, self.drafter.cfg.tap_rows()) else {
            return Ok(())
        };
        ensure!(residual.bytes >= (from + count) * HIDDEN * 4, MiniMaxError::Invalid("DFlash2 tap residual extent"));
        native.check_device(&[residual])?;
        self.drafter.tap_f32_at(layer, residual.ptr.cast_const(), from, count, to)
    }

    /// The context rows a `rows`-row prefill step from `start` leaves for ring `slot`.
    pub fn prefill_rows(&self, start: usize, rows: usize, slot: usize) -> Vec<ContextRow> {
        tail_rows(start, rows, slot, self.drafter.cfg.tap_rows())
    }

    /// Writes the context K/V of committed tapped rows.
    pub fn update(&self, rows: &[ContextRow]) -> Result<()> { self.drafter.update(rows) }

    /// Seven drafts after each sequence's anchor, through the target's BF16 `head` [vocab, hidden].
    pub fn propose(&self, seqs: &[DraftSeq], embedding: &TokenEmbedding<'a>, head: CuteafdDeviceBuffer)
        -> Result<Vec<Draft>> {
        self.drafter.draft_device(seqs, embedding, TargetHead::Bf16Buffer(&head))
    }

    /// The shared drafter (diagnostic replay and oracle).
    pub fn drafter(&self) -> &DflashDrafter<'a> { &self.drafter }

    /// The keyed selector prior of these numerics (`minimax_m3/dflash2/...`, filed beside the
    /// drafter's config by `scripts/bench/calibrate-draft-confidence.py`): the generic GLM-5.3
    /// fit until one exists. The head is the target's BF16 one.
    pub fn selector_prior(&self) -> Result<crate::shared::qwen::scheduler::SelectorPrior> {
        let key = self.drafter.confidence_key("minimax_m3", false);
        let fit = self.drafter.selector_fit("minimax_m3", false)?;
        tracing::info!(key, fit = fit.key, "MiniMax DFlash2 selector prior");
        Ok(crate::shared::qwen::scheduler::SelectorPrior { key, fit })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The borrowed head is checked before the drafter is read: a non-BF16 or misshapen target
    /// head refuses at startup, where it would otherwise fail the first draft.
    #[test]
    fn minimax_dflash_refuses_a_head_it_cannot_borrow_at_contract_open() {
        let head = |dtype, shape: Vec<usize>| cuteafd_loader::SafetensorsTensorMetadata {
            name: "language_model.lm_head.weight".into(), dtype, byte_offset: 0,
            byte_length: shape.iter().product::<usize>() as u64 * 2, shape };
        let open = |head: &cuteafd_loader::SafetensorsTensorMetadata| Contract::open(Path::new("/nonexistent-draft"),
            head, None, Qwen4KvCache::Fp8, 8, 170, None, true).unwrap_err().to_string();
        let fp8 = open(&head(cuteafd_core::DType::F8E4M3, vec![VOCAB, HIDDEN]));
        assert!(fp8.contains("requires checkpoint BF16"), "{fp8}");
        let narrow = open(&head(cuteafd_core::DType::Bf16, vec![VOCAB, 4096]));
        assert!(narrow.contains("expected [200064, 6144]"), "{narrow}");
        // A borrowable head passes on to the drafter's own files.
        let fine = open(&head(cuteafd_core::DType::Bf16, vec![VOCAB, HIDDEN]));
        assert!(!fine.contains("BF16") && !fine.contains("expected ["), "{fine}");
    }

    /// A long call taps its last 4096 rows across its units, each at its offset in the tail; a short
    /// call or a verify step taps every row in place.
    #[test]
    fn minimax_dflash_taps_the_call_tail_across_units() {
        // Two units of 4096 rows: only the second unit's rows are tapped, from tap row 0.
        assert_eq!(unit_tail(0, 4096, 8192, 4096), None);
        assert_eq!(unit_tail(4096, 4096, 8192, 4096), Some((0, 0, 4096)));
        // Page-aligned uneven units: the tail starts inside the first unit.
        assert_eq!(unit_tail(0, 2240, 4412, 4096), Some((316, 0, 1924)));
        assert_eq!(unit_tail(2240, 2172, 4412, 4096), Some((0, 1924, 2172)));
        // Three units of 2752 (a pinned 3 x 2752, an 8256-row span): the tail starts inside the second.
        assert_eq!(unit_tail(0, 2752, 8256, 4096), None);
        assert_eq!(unit_tail(2752, 2752, 8256, 4096), Some((1408, 0, 1344)));
        assert_eq!(unit_tail(5504, 2752, 8256, 4096), Some((0, 1344, 2752)));
        // Verify steps and short calls keep every row at its own index.
        assert_eq!(unit_tail(0, 24, 24, 4096), Some((0, 0, 24)));
        assert_eq!(unit_tail(0, 4096, 4096, 4096), Some((0, 0, 4096)));
        // The prefill context rows are the same tail, at its positions.
        let rows = tail_rows(100, 4412, 3, 4096);
        assert_eq!(rows.len(), 4096);
        assert_eq!((rows[0].tap_row, rows[0].slot, rows[0].position), (0, 3, 100 + 316));
        assert_eq!((rows[4095].tap_row, rows[4095].position), (4095, 100 + 4411));
        let short = tail_rows(7, 5, 1, 4096);
        assert_eq!(short.iter().map(|r| (r.tap_row, r.position)).collect::<Vec<_>>(),
            (0..5).map(|r| (r, 7 + r)).collect::<Vec<_>>());
    }

    /// Every tapped row of a unit call lands once: on the shared executor's cuts for every admitted
    /// shape, the units' tail pieces tile the tap rows exactly.
    #[test]
    fn minimax_dflash_unit_tails_tile_the_tap_rows_exactly() -> Result<()> {
        use crate::shared::prefill_pipeline::prefill_units;
        for (unit, waves) in [(4096, 1), (4096, 2), (2752, 3), (2048, 4), (1024, 2)] {
            let span = unit * waves;
            for rows in (256..=span).step_by(131).chain([span, 4097, 4412]).filter(|&rows| rows <= span) {
                let mut covered = Vec::new();
                for piece in prefill_units(rows, unit, waves, 256)? {
                    if let Some((from, to, count)) = unit_tail(piece.start, piece.len(), rows, 4096) {
                        assert!(from + count <= piece.len());
                        covered.extend(to..to + count);
                    }
                }
                assert_eq!(covered, (0..rows.min(4096)).collect::<Vec<_>>(), "{waves} x {unit}, {rows} rows");
            }
        }
        Ok(())
    }
}
