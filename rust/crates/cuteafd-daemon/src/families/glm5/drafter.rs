//! GLM 5.3 / Mistral Large 4's single drafter copy on its admitted GPU home.
//! GPU0 keeps the embedding and target head; a GPU1 home runs the same common
//! executor through the shared device-driven bridge (`shared::draft::remote_dflash`).
//! Every GPU-touching call runs on the drafter's own device; metadata reads
//! deref to the common executor.
use crate::shared::draft::remote_dflash::RemoteDflash;
use crate::shared::memory::device::Device;
use crate::shared::speculation::dflash::{ContextRow, DflashDrafter, Draft, DraftSeq, TargetHead};
use crate::shared::token_io::TokenEmbedding;
use anyhow::{ensure, Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use std::ffi::c_void;

pub(crate) const IDENTITY: &str = "glm-remote-dflash:taps,input,hidden,logits";

pub(crate) enum GlmDrafter<'a> {
    Local(DflashDrafter<'a>),
    Remote(RemoteDflash<'a>),
}

impl<'a> std::ops::Deref for GlmDrafter<'a> {
    type Target = DflashDrafter<'a>;
    fn deref(&self) -> &DflashDrafter<'a> {
        match self { Self::Local(d) => d, Self::Remote(d) => &d.draft }
    }
}

impl<'a> GlmDrafter<'a> {
    fn on<T>(&self, work: impl FnOnce(&DflashDrafter<'a>) -> Result<T>) -> Result<T> {
        match self {
            Self::Local(d) => work(d),
            Self::Remote(d) => d.draft.device.run(|| work(&d.draft)),
        }
    }

    pub fn device(&self) -> Option<Device<'a>> {
        match self { Self::Local(_) => None, Self::Remote(d) => Some(d.draft.device) }
    }

    pub fn is_remote(&self) -> bool { matches!(self, Self::Remote(_)) }

    /// A GPU1 drafter's own stream (the bridge's draft side).
    pub fn stream(&self) -> Option<*mut c_void> {
        match self { Self::Local(_) => None, Self::Remote(d) => Some(d.draft_stream()) }
    }

    pub fn prepare_workspace(&self) -> Result<()> { self.on(|d| d.prepare_workspace()) }

    /// Bind a GPU1 drafter to the target: its context waits must follow the
    /// cooperative GPU1 target producer, and every target-side head width is
    /// primed before any peer wait exists. Local drafters primed at load.
    pub fn bind_target(&self, producer: crate::shared::peer_split::RankDevice,
        embedding: &TokenEmbedding<'_>, head: TargetHead<'_>) -> Result<()> {
        match self {
            Self::Local(_) => Ok(()),
            Self::Remote(d) => {
                d.validate_target_peer(producer)?;
                d.prime(embedding, head)
            }
        }
    }

    /// Taps write only GPU0 staging; the bridge publishes them at `update`.
    pub fn tap(&self, layer: usize, hidden: *const c_void, first: usize, n: usize) -> Result<()> {
        self.tap_at(layer, hidden, first, n, 0)
    }

    pub fn tap_at(&self, layer: usize, hidden: *const c_void, first: usize, n: usize, to: usize) -> Result<()> {
        match self {
            Self::Local(d) => d.tap_at(layer, hidden, first, n, to),
            Self::Remote(d) => d.tap_at(layer, hidden, first, n, to),
        }
    }

    pub fn update(&self, rows: &[ContextRow]) -> Result<()> {
        match self { Self::Local(d) => d.update(rows), Self::Remote(d) => d.update(rows) }
    }

    pub fn put_taps(&self, rows: &[u8]) -> Result<()> {
        match self { Self::Local(d) => d.put_taps(rows), Self::Remote(d) => d.put_taps(rows) }
    }

    pub fn draft_device(&self, sequences: &[DraftSeq], embedding: &TokenEmbedding<'_>, head: TargetHead<'_>)
        -> Result<Vec<Draft>> {
        match self {
            Self::Local(d) => d.draft_device(sequences, embedding, head),
            Self::Remote(d) => d.draft_device(sequences, embedding, head),
        }
    }

    pub fn draft_device_diagnostic(&self, sequences: &[DraftSeq], embedding: &TokenEmbedding<'_>,
        head: TargetHead<'_>) -> Result<Vec<Draft>> {
        match self {
            Self::Local(d) => d.draft_device_diagnostic(sequences, embedding, head),
            Self::Remote(d) => d.draft_device_diagnostic(sequences, embedding, head),
        }
    }

    pub fn last_hidden(&self, sequences: usize) -> Result<Vec<u8>> { self.on(|d| d.last_hidden(sequences)) }

    /// A slot rebinding first retires the bridge's last context acknowledgement.
    pub fn reset_context(&self, slot: usize, origin: usize) -> Result<()> {
        self.finish_request()?;
        self.on(|d| d.reset_context(slot, origin))
    }

    pub fn discard_context(&self, slot: usize) -> Result<()> {
        self.finish_request()?;
        self.on(|d| d.discard_context(slot))
    }

    pub fn restore_mla_windows(&self, slot: usize, frontier: usize) -> Result<()> {
        self.finish_request()?;
        self.on(|d| d.restore_mla_windows(slot, frontier))
    }

    /// Window marks live on the drafter's device (they follow its home).
    pub fn copy_mla_windows(&self, slot: usize, marks: &[CuteafdDeviceBuffer], capture: bool) -> Result<()> {
        ensure!(marks.iter().all(|mark| self.device().is_none_or(|device| mark.device_id == device.id)),
            "GLM draft window marks are not on the drafter's device");
        self.on(|d| d.copy_mla_windows(slot, marks, capture))
    }

    pub fn require_live(&self) -> Result<()> {
        match self { Self::Local(_) => Ok(()), Self::Remote(d) => d.require_live() }
    }

    pub fn publish_abort(&self) -> Result<()> {
        match self { Self::Local(_) => Ok(()), Self::Remote(d) => d.publish_abort() }
    }

    pub fn finish_request(&self) -> Result<()> {
        match self { Self::Local(_) => Ok(()), Self::Remote(d) => d.finish_request() }
    }

    pub fn finish_terminal(&self, drained: bool) {
        if let Self::Remote(d) = self { d.finish_terminal(drained); }
    }

    pub fn synchronize(&self) -> Result<()> {
        match self { Self::Local(_) => Ok(()), Self::Remote(d) => d.synchronize() }
    }

    pub fn allocation_receipts(&self, devices: cuteafd_loader::placement::planner::CoordinatorDevices<'_>,
        prefix: &str) -> Result<Vec<cuteafd_loader::placement::planner::Receipt>> {
        use cuteafd_core::memory_layout::Category;
        use cuteafd_loader::placement::planner::{Lifetime, Phase};
        let mut receipts = (**self).allocation_receipts(devices, prefix)?;
        // The shared contract covers push: its draft survives later Load work.
        for receipt in &mut receipts {
            let allocation = &mut receipt.allocation;
            if allocation.id.starts_with("glm/draft/") || allocation.view.as_ref()
                .is_some_and(|view| view.backing.starts_with("glm/draft/")) {
                allocation.lifetime = Lifetime::RESIDENT;
            }
        }
        if let Self::Remote(d) = self {
            let mut inventory = crate::shared::memory::receipts::Inventory::new(devices, &format!("{prefix}.bridge"),
                Lifetime { first: Phase::DraftLoad, last: Phase::Growth }, Category::Drafter, "remote draft bridge");
            for (name, buffer) in d.bridge_buffers() { inventory.device(name, buffer)?; }
            receipts.extend(inventory.receipts);
        }
        Ok(receipts)
    }
}

impl crate::shared::speculation::dflash::ReplayDrafter for GlmDrafter<'_> {
    fn block(&self) -> usize { self.cfg.block }
    fn sequences(&self) -> usize { self.max_batch_sequences() }
    fn resident_modes(&self) -> Vec<crate::shared::speculation::dflash::ReplayMode> { (**self).resident_modes() }
    fn context(&self, taps: &[u8], first: usize) -> Result<()> {
        match self { Self::Local(d) => d.context(taps, first), Self::Remote(d) => d.context(taps, first) }
    }
    fn draft_tokens(&self, sequences: &[(usize, u32, usize)], rows: &[u8], head: *const c_void) -> Result<Vec<Vec<u32>>> {
        match self {
            Self::Local(d) => d.draft_tokens(sequences, rows, head),
            Self::Remote(d) => d.draft_tokens(sequences, rows, head),
        }
    }
    fn tap_rows(&self) -> usize { crate::shared::speculation::dflash::TAP_ROWS }
    fn last_hidden(&self, sequences: usize) -> Result<Vec<u8>> { GlmDrafter::last_hidden(self, sequences) }
}
