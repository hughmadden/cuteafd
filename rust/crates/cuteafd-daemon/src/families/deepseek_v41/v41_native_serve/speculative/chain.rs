//! Static draft-chain dispatch for the shared request lifecycle.
use super::*;
use crate::families::deepseek_v41::v41_experts::dspark::DistributedDsparkChain;

pub(crate) trait DraftChain<'a> {
    fn visit_graph_banks(&mut self, f: &mut crate::shared::decode_graph::device::GraphVisitor<'_>) -> Result<()>;
    fn execution_device(&self) -> Option<crate::shared::memory::device::Device<'a>>;
    fn stage_tokens(&mut self, tokens: &[i32]) -> Result<()>;
    fn stage_sampling(&mut self, rngs: &mut [&mut cuteafd_core::DsparkRng], temperatures: &[f32]) -> Result<()>;
    /// # Safety
    /// Windows, bindings and staged inputs obey the concrete chain's contract;
    /// backing storage remains live until completion or chain destruction.
    unsafe fn begin_replay(&mut self, windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3]) -> Result<()>;
    fn poll_replay(&mut self) -> Result<Option<(Vec<u32>, Vec<f32>)>>;
    /// Drop a pending replay so its requests can be released.
    fn cancel_replay(&mut self);
    /// Record a CUDA/replay failure as terminal for this chain.
    fn note_failure(&mut self, error: &anyhow::Error);
    /// Draft width of the next (or pending) proposal.
    fn width(&self) -> usize;
    /// Select the width (5 or 7, within the loaded maximum) before staging.
    fn set_width(&mut self, width: usize) -> Result<()>;
    /// A terminal draft (dead expert wire or a sticky CUDA/replay failure) is
    /// not a reusable target-only fallback: the lane must fail the round
    /// instead of verifying anchors forever. Mirrors GLM Flash's `require_live`
    /// and MiMo's `is_terminal`.
    fn require_live(&self) -> Result<()>;
}
/// The synchronous draft operations the single-lane `propose` path needs. Only
/// the local chain implements them; the distributed chain drafts through
/// `poll_propose` instead.
pub(crate) trait ProposeChain<'a>: DraftChain<'a> {
    /// Drain queued replay, including a cold capture followed by replay.
    fn wait_replay(&mut self) -> Result<(Vec<u32>, Vec<f32>)>;
    fn has_graph(&self, count: usize) -> bool;
    /// Stage the seed tokens and upload them to the device.
    fn set_tokens(&mut self, tokens: &[i32]) -> Result<()>;
    /// Reserve a fresh sampling attempt (synchronizes and uploads).
    fn prepare_sampling(&mut self, rngs: &mut [&mut cuteafd_core::DsparkRng], temperatures: &[f32]) -> Result<()>;
    /// # Safety
    /// Same contract as [`DraftChain::begin_replay`].
    unsafe fn capture(&mut self, windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3]) -> Result<()>;
    /// # Safety
    /// Same contract as [`DraftChain::begin_replay`].
    unsafe fn replay(&mut self, windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3]) -> Result<()>;
    fn copy_d2h(&self, dst: &mut [u8], buffer: cuteafd_ffi::CuteafdDeviceBuffer) -> Result<()>;
    fn draft_output(&self) -> Result<[cuteafd_ffi::CuteafdDeviceBuffer; 3]>;
}
impl<'w, 'a> ProposeChain<'a> for DsparkChain<'w, 'a> {
    fn wait_replay(&mut self) -> Result<(Vec<u32>, Vec<f32>)> { DsparkChain::wait_replay(self) }
    fn has_graph(&self, count: usize) -> bool { self.has_graph(count) }
    fn set_tokens(&mut self, tokens: &[i32]) -> Result<()> { self.set_tokens(tokens) }
    fn prepare_sampling(&mut self, rngs: &mut [&mut cuteafd_core::DsparkRng], temperatures: &[f32]) -> Result<()> {
        self.prepare_sampling(rngs, temperatures)
    }
    unsafe fn capture(&mut self, windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3]) -> Result<()> {
        // SAFETY: forwards the caller's stable window/binding contract.
        unsafe { self.capture(windows, bindings) }
    }
    unsafe fn replay(&mut self, windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3]) -> Result<()> {
        // SAFETY: forwards the caller's stable window/binding contract.
        unsafe { self.replay(windows, bindings).map(drop) }
    }
    fn copy_d2h(&self, dst: &mut [u8], buffer: cuteafd_ffi::CuteafdDeviceBuffer) -> Result<()> { self.copy_d2h(dst, buffer) }
    fn draft_output(&self) -> Result<[cuteafd_ffi::CuteafdDeviceBuffer; 3]> { self.draft_output() }
}

macro_rules! chain {
    ($ty:ident, $device:expr, $live:expr) => {
        impl<'w, 'a> DraftChain<'a> for $ty<'w, 'a> {
            fn visit_graph_banks(&mut self, f: &mut crate::shared::decode_graph::device::GraphVisitor<'_>) -> Result<()> {
                self.visit_graph_banks(f)
            }
            fn execution_device(&self) -> Option<crate::shared::memory::device::Device<'a>> { ($device)(self) }
            fn stage_tokens(&mut self, tokens: &[i32]) -> Result<()> {
                self.stage_tokens(tokens)
            }
            fn stage_sampling(&mut self, rngs: &mut [&mut cuteafd_core::DsparkRng], temperatures: &[f32]) -> Result<()> {
                self.stage_sampling(rngs, temperatures)
            }
            unsafe fn begin_replay(&mut self, windows: [&DsparkWindow<'_>; 3],
                bindings: [&[(WindowLease, u64)]; 3]) -> Result<()> {
                // SAFETY: forwards the caller's window, binding and lifetime contract.
                unsafe { self.begin_replay(windows, bindings) }
            }
            fn poll_replay(&mut self) -> Result<Option<(Vec<u32>, Vec<f32>)>> {
                self.poll_replay()
            }
            fn cancel_replay(&mut self) { self.cancel_replay() }
            fn note_failure(&mut self, error: &anyhow::Error) { self.note_failure(error) }
            fn width(&self) -> usize { self.width() }
            fn set_width(&mut self, width: usize) -> Result<()> {
                self.set_width(width)
            }
            fn require_live(&self) -> Result<()> { ($live)(self) }
        }
    };
}
chain!(DsparkChain, |_: &DsparkChain| None, |chain: &DsparkChain| chain.require_live());
chain!(DistributedDsparkChain, |chain: &DistributedDsparkChain<'w, 'a>| Some(chain.device()),
    |chain: &DistributedDsparkChain<'w, 'a>| { chain.require_live()?; super::require_peer_live() });

#[cfg(test)]
mod anchor_fallback_tests {
    use super::*;
    use crate::families::deepseek_v41::v41_experts::dspark::DraftFailure;
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use std::cell::Cell;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum At { Width, Tokens, Sampling, Capture, Replay, Output, TokensCopy, ConfidenceCopy, Begin, Poll, Never }
    struct FailingChain {
        failure: DraftFailure,
        at: At,
        device_error: bool,
        tokens: Vec<u32>,
        polls: usize,
        cancelled: bool,
        downloads: Cell<usize>,
        graph_bank: Option<crate::shared::decode_graph::GraphBank<usize, ()>>,
    }
    impl FailingChain {
        fn new(at: At, device_error: bool) -> Self {
            Self { failure: Default::default(), at, device_error, tokens: Vec::new(), polls: 0,
                cancelled: false, downloads: Cell::new(0), graph_bank: None }
        }
        // Deliberately does not record: only the real runtime can latch this mock.
        fn step(&self, at: At) -> Result<()> {
            if self.at != at { return Ok(()); }
            if self.device_error {
                Err(anyhow::Error::new(cuteafd_ffi::NativeStatusError {
                    context: format!("injected {at:?}"), status: 98, detail: "invalid device function".into(),
                }).context("draft operation"))
            } else { anyhow::bail!("injected host guard at {at:?}") }
        }
        fn packed(&self) -> Vec<u32> {
            (0..=5).flat_map(|step| self.tokens.iter().map(move |&token| token + step)).collect()
        }
    }
    impl<'a> DraftChain<'a> for FailingChain {
        fn visit_graph_banks(&mut self, f: &mut crate::shared::decode_graph::device::GraphVisitor<'_>) -> Result<()> {
            if let Some(bank) = &mut self.graph_bank { f(0, "dspark_chain", false, bank)?; }
            Ok(())
        }
        fn execution_device(&self) -> Option<crate::shared::memory::device::Device<'a>> { None }
        fn stage_tokens(&mut self, tokens: &[i32]) -> Result<()> {
            self.tokens = tokens.iter().map(|&token| token as u32).collect();
            self.step(At::Tokens)
        }
        fn stage_sampling(&mut self, _: &mut [&mut cuteafd_core::DsparkRng], _: &[f32]) -> Result<()> { self.step(At::Sampling) }
        unsafe fn begin_replay(&mut self, _: [&DsparkWindow<'_>; 3], _: [&[(WindowLease, u64)]; 3]) -> Result<()> { self.step(At::Begin) }
        fn poll_replay(&mut self) -> Result<Option<(Vec<u32>, Vec<f32>)>> {
            self.step(At::Poll)?;
            self.polls += 1;
            Ok((self.polls > 1).then(|| (self.packed(), vec![0.5; self.tokens.len() * 5])))
        }
        fn cancel_replay(&mut self) { self.cancelled = true; }
        fn note_failure(&mut self, error: &anyhow::Error) { self.failure.record(error); }
        fn width(&self) -> usize { 5 }
        fn set_width(&mut self, _: usize) -> Result<()> { self.step(At::Width) }
        fn require_live(&self) -> Result<()> { self.failure.check("test draft chain") }
    }
    impl<'a> ProposeChain<'a> for FailingChain {
        fn wait_replay(&mut self) -> Result<(Vec<u32>, Vec<f32>)> {
            for _ in 0..2 {
                if let Some(output) = self.poll_replay()? { return Ok(output); }
            }
            anyhow::bail!("mock replay incomplete after draining")
        }
        fn has_graph(&self, _: usize) -> bool { self.at != At::Capture }
        fn set_tokens(&mut self, tokens: &[i32]) -> Result<()> { self.stage_tokens(tokens) }
        fn prepare_sampling(&mut self, _: &mut [&mut cuteafd_core::DsparkRng], _: &[f32]) -> Result<()> { self.step(At::Sampling) }
        unsafe fn capture(&mut self, _: [&DsparkWindow<'_>; 3], _: [&[(WindowLease, u64)]; 3]) -> Result<()> { self.step(At::Capture) }
        unsafe fn replay(&mut self, _: [&DsparkWindow<'_>; 3], _: [&[(WindowLease, u64)]; 3]) -> Result<()> { self.step(At::Replay) }
        fn draft_output(&self) -> Result<[CuteafdDeviceBuffer; 3]> {
            self.step(At::Output)?;
            Ok([CuteafdDeviceBuffer { ptr: 1usize as *mut _, bytes: self.tokens.len() * 6 * 4, device_id: 0, flags: 0 },
                CuteafdDeviceBuffer::default(),
                CuteafdDeviceBuffer { ptr: 2usize as *mut _, bytes: self.tokens.len() * 5 * 4, device_id: 0, flags: 0 }])
        }
        fn copy_d2h(&self, dst: &mut [u8], buffer: CuteafdDeviceBuffer) -> Result<()> {
            self.downloads.set(self.downloads.get() + 1);
            let confidence = buffer.ptr as usize == 2;
            self.step(if confidence { At::ConfidenceCopy } else { At::TokensCopy })?;
            let bytes: Vec<_> = if confidence { vec![0.5f32; self.tokens.len() * 5].iter().flat_map(|v| v.to_ne_bytes()).collect() }
                else { self.packed().iter().flat_map(|v| v.to_ne_bytes()).collect() };
            dst.copy_from_slice(&bytes);
            Ok(())
        }
    }
    fn fixture() -> Result<Fixture> {
        Fixture::build_for_tests()
    }
    fn runtime<'a>(lib: &'a NativeLibrary, at: At, device_error: bool) -> Result<DraftRuntime<'a, 'a, FailingChain>> {
        let window = || DsparkWindow::new(lib, 4, 8, DsparkWindow::device_bytes(4, 8)?);
        let mut runtime = DraftRuntime {
            mains: Vec::new(), chains: (0..2).map(|_| FailingChain::new(at, device_error)).collect(),
            windows: [window()?, window()?, window()?], pending_commit_ids: vec![Vec::new(); 2],
            pending_prefix_ids: [None, None], requests: Default::default(), pending: vec![None; 2],
            request_limit: 4, draft_limit: 5, draft_width: 5, confidence_trace: Default::default(),
            neural_lengths: Default::default(), length: Default::default(), policy: None, published: None,
            lanes: (0..2).map(|_| LaneRound::new(policy::LAYERS, policy::TOPK)).collect(),
        };
        for id in 1..=4 { runtime.admit(id)?; }
        runtime.bind_policy(policy::remote_geometry(5))?;
        Ok(runtime)
    }
    fn seeds(ids: &[u64]) -> Vec<DraftSeed> {
        ids.iter().enumerate().map(|(i, &id)| (id, 11 + i as u32, 5, 9)).collect()
    }
    #[test]
    fn synchronous_proposal_records_each_failure_on_its_own_lane() -> Result<()> {
        let fixture = fixture()?;
        let lib = fixture.load()?;
        fixture.immediate_copies(&lib)?;
        fixture.configure_memory(&lib, 0, 1 << 30)?;
        for lane in 0..2 {
            for at in [At::Width, At::Tokens, At::Sampling, At::Begin, At::Poll] {
                for device in [false, true] {
                    let mut runtime = runtime(&lib, at, device)?;
                    let round = seeds(&[1, 2]);
                    let error = runtime.propose(lane, &round).unwrap_err();
                    let fallback = runtime.anchor_fallback(lane, &round, &error);
                    assert_eq!(fallback.is_err(), device, "lane {lane}, {at:?}");
                    assert!(runtime.require_live(1 - lane).is_ok(), "other lane must stay live");
                    if !device { assert_eq!(fallback?, vec![vec![11], vec![12]]); }
                }
            }
        }
        Ok(())
    }
    #[test]
    fn independent_proposal_records_submission_and_poll_failures() -> Result<()> {
        let fixture = fixture()?;
        let lib = fixture.load()?;
        fixture.immediate_copies(&lib)?;
        fixture.configure_memory(&lib, 0, 1 << 30)?;
        for lane in 0..2 {
            for at in [At::Width, At::Tokens, At::Sampling, At::Begin, At::Poll] {
                for device in [false, true] {
                    let mut runtime = runtime(&lib, at, device)?;
                    let round = seeds(&[1, 2]);
                    let result = match runtime.poll_propose(lane, &round) {
                        Ok(None) => runtime.poll_propose(lane, &round),
                        result => result,
                    };
                    let error = result.unwrap_err();
                    assert!(runtime.pending[lane].is_none());
                    assert_eq!(runtime.anchor_fallback(lane, &round, &error).is_err(), device, "lane {lane}, {at:?}");
                    assert!(runtime.require_live(1 - lane).is_ok());
                }
            }
        }
        Ok(())
    }
    #[test]
    fn independent_pending_blocks_release_and_cancellation_resets_old_ids() -> Result<()> {
        let fixture = fixture()?;
        let lib = fixture.load()?;
        fixture.immediate_copies(&lib)?;
        fixture.configure_memory(&lib, 0, 1 << 30)?;
        let mut runtime = runtime(&lib, At::Never, false)?;
        let round = seeds(&[1, 2]);
        assert!(runtime.poll_propose(1, &round)?.is_none());
        assert!(runtime.release(1).is_err());
        assert!(runtime.poll_propose(1, &round)?.is_none());
        assert!(runtime.release(2).is_err());
        runtime.confidence_trace.insert(1, vec![0.5; 5]);
        runtime.neural_lengths.insert(1, 3);
        assert!(runtime.poll_propose(1, &seeds(&[3, 4])).is_err());
        assert!(runtime.chains[1].cancelled);
        assert_eq!((runtime.lanes[1].width, runtime.lanes[1].drafted_requests), (0, 0));
        assert!(runtime.confidence_trace(1).is_none() && !runtime.neural_lengths.contains_key(&1));
        runtime.release(1)?;
        runtime.release(2)?;
        Ok(())
    }
    #[test]
    fn both_proposal_completion_paths_measure_pending_captures() -> Result<()> {
        use crate::shared::decode_graph::{device::{Basis, Cycle, Preference, Site}, GraphBank};
        let fixture = fixture()?;
        let lib = fixture.load()?;
        fixture.immediate_copies(&lib)?;
        fixture.configure_memory(&lib, 0, 1 << 30)?;
        for independent in [false, true] {
            let mut runtime = runtime(&lib, At::Never, false)?;
            let mut bank = GraphBank::new(None);
            bank.attach_device(&lib, 0, 4096, Site { name: "draft_completion_fixture", cycle: Cycle::Other,
                preference: Preference::Adaptive, bootstrap: 16, basis: Basis::Bytes, spans_devices: false })?;
            let ticket = bank.cache_ticket(&1, false, None).context("fixture admission")?;
            bank.cache_commit(1, (), ticket, 1, 1, false);
            assert!(bank.cache_needs_boundary());
            runtime.chains[1].graph_bank = Some(bank);
            let round = seeds(&[1, 2]);
            if independent {
                assert!(runtime.poll_propose(1, &round)?.is_none());
                assert!(runtime.chains[1].graph_bank.as_ref().unwrap().cache_needs_boundary());
                assert!(runtime.poll_propose(1, &round)?.is_none());
                assert!(runtime.poll_propose(1, &round)?.is_some());
            } else {
                runtime.propose(1, &round)?;
            }
            assert!(!runtime.chains[1].graph_bank.as_ref().unwrap().cache_needs_boundary(),
                "completed proposal must feed measured_capture, independent={independent}");
            runtime.chains[1].graph_bank.as_mut().unwrap().cache_clear_drained();
        }
        Ok(())
    }
    #[test]
    fn successful_poll_preserves_order_and_invalid_lane_is_an_error() -> Result<()> {
        let fixture = fixture()?;
        let lib = fixture.load()?;
        fixture.immediate_copies(&lib)?;
        fixture.configure_memory(&lib, 0, 1 << 30)?;
        let mut runtime = runtime(&lib, At::Never, false)?;
        assert!(runtime.poll_propose(2, &seeds(&[1])).is_err());
        let round = seeds(&[2, 1]);
        assert!(runtime.poll_propose(1, &round)?.is_none());
        assert!(runtime.poll_propose(1, &round)?.is_none());
        let (tokens, _) = runtime.poll_propose(1, &round)?.context("proposal should finish")?;
        assert_eq!(tokens, vec![vec![11, 12, 13, 14, 15, 16], vec![12, 13, 14, 15, 16, 17]]);
        runtime.release(1)?;
        runtime.release(2)?;
        Ok(())
    }
    #[test]
    fn four_then_two_failed_proposal_is_an_undrafted_policy_round() -> Result<()> {
        let fixture = fixture()?;
        let lib = fixture.load()?;
        fixture.immediate_copies(&lib)?;
        fixture.configure_memory(&lib, 0, 1 << 30)?;
        for independent in [false, true] {
            let mut runtime = runtime(&lib, At::Never, false)?;
            let previous = seeds(&[1, 2, 3, 4]);
            runtime.propose(0, &previous)?;
            assert_eq!(runtime.lanes[0].drafted_requests, 4);
            runtime.neural_lengths.insert(1, 3);
            let mut round = seeds(&[1, 2]);
            round[0].1 = 129280; // Host validation fails before drafting starts.
            let error = if independent { runtime.poll_propose(0, &round).unwrap_err() }
                else { runtime.propose(0, &round).unwrap_err() };
            assert_eq!((runtime.lanes[0].width, runtime.lanes[0].drafted_requests), (0, 0));
            assert!(runtime.confidence_trace(1).is_none() && !runtime.neural_lengths.contains_key(&1));
            round[0].1 = 11;
            assert_eq!(runtime.anchor_fallback(0, &round, &error)?, vec![vec![11], vec![12]]);
            runtime.select_lengths(0, &[(1, 0), (2, 0)], false)?;
        }
        Ok(())
    }
}
