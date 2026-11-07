//! Physical coordinator scopes; every final view retains its backing.
use super::*;
use crate::shared::memory::weight_arena::DeviceWeightArena;
use std::{collections::BTreeMap, rc::Rc};

pub(crate) struct Arenas<'a> {
    target: BTreeMap<i32, DeviceWeightArena<'a>>,
    draft: Option<DeviceWeightArena<'a>>,
}

impl<'a> Qwen4Loader<'a> {
    pub(super) fn allocate(&self, name: &str, bytes: usize) -> Result<Allocation<'a>> {
        let device = self.device()?;
        let arenas = self.arenas.as_ref().context("Qwen weights require planned arenas")?;
        let arena = if name.starts_with("mtp.") { arenas.draft.as_ref() }
            else { arenas.target.get(&device.id) }.context("Qwen weight arena device/scope")?;
        crate::shared::memory::weight_arena::weight_allocation(device, Some(arena), name, bytes)
    }

    pub(crate) fn embedding_allocation(&self, name: &str, bytes: usize) -> Result<Allocation<'a>> {
        self.allocate(name, bytes)
    }

    pub(crate) fn with_arenas(&self, cfg: &Qwen4Config, owners: &[usize],
        ranks: &[(Device<'a>, *mut c_void)], mtp: bool, fp8_head: bool,
        embedding: bool) -> Result<Self> {
        ensure!(self.arenas.is_none(), "Qwen arenas already planned");
        ensure!(!ranks.is_empty() && ranks.iter().all(|(device, _)| std::ptr::eq(device.library, self.library))
            && ranks.iter().map(|(d, _)| d.id).collect::<std::collections::BTreeSet<_>>().len() == ranks.len(),
            "Qwen arena needs distinct devices in one native library");
        let embedding_bytes = if embedding {
            usize::try_from(self.tensor(&format!("{PREFIX}embed_tokens.weight"))?.meta.byte_length)?
        } else { 0 };
        let layouts = cuteafd_loader::families::qwen4::weight_layout::coordinator_arenas(
            self.checkpoint, cfg, owners, ranks.len(), mtp,
            resident::Qwen4Representation { fp8_projections: self.fp8_decode, fp8_head }, embedding_bytes)?;
        let mut target = BTreeMap::new();
        let mut draft = None;
        for (rank, layout) in layouts {
            let is_draft = layout.id().contains("/mtp/");
            let arena = DeviceWeightArena::new(ranks[rank].0, layout)?;
            if is_draft { draft = Some(arena); }
            else { target.insert(ranks[rank].0.id, arena); }
        }
        Ok(Self { arenas: Some(Rc::new(Arenas { target, draft })), ..self.clone() })
    }
}

impl Qwen4Weights<'_> {
    pub(crate) fn allocation_receipts(&self,
        devices: cuteafd_loader::placement::planner::CoordinatorDevices<'_>)
        -> Result<Vec<cuteafd_loader::placement::planner::Receipt>> {
        use cuteafd_core::memory_layout::Category;
        use cuteafd_loader::placement::planner::Lifetime;
        let mut inventory = crate::shared::memory::receipts::Inventory::new(devices, "qwen4.weights",
            Lifetime::TARGET, Category::Weights, "coordinator weights");
        for (layer, part) in self.layers.iter().enumerate() {
            for (name, owner) in &part.operands {
                owner.inventory(&mut inventory, &format!("layers.{layer}.{name}"))?;
            }
        }
        for (index, owner) in self.mixer.iter().enumerate() { owner.inventory(&mut inventory, &format!("mixer.{index}"))?; }
        for (index, owner) in self.head.allocations().iter().enumerate() {
            owner.inventory(&mut inventory, &format!("head.{index}"))?;
        }
        if let Some(mtp) = &self.mtp {
            for (name, owner) in &mtp.layer.operands { owner.inventory(&mut inventory, &format!("mtp.layer.{name}"))?; }
            for (name, owner) in [("norm_hidden", &mtp.norm_hidden), ("norm_embed", &mtp.norm_embed),
                ("fc_hidden", &mtp.fc_hidden), ("fc_embed", &mtp.fc_embed)] {
                owner.inventory(&mut inventory, &format!("mtp.{name}"))?;
            }
            for (index, owner) in mtp.mixer.iter().enumerate() { owner.inventory(&mut inventory, &format!("mtp.mixer.{index}"))?; }
        }
        Ok(inventory.receipts)
    }
}
