//! Retained draft owners only; program/graph runtime memory remains opaque.
use super::*;
use crate::shared::memory::receipts::Inventory;
use cuteafd_core::memory_layout::Category;
use cuteafd_loader::placement::planner::{CoordinatorDevices, Lifetime, Phase, Receipt};

impl DflashDrafter<'_> {
    pub(crate) fn allocation_receipts(&self, devices: CoordinatorDevices<'_>, prefix: &str) -> Result<Vec<Receipt>> {
        self.allocation_receipts_with_lifetime(devices, prefix,
            Lifetime { first: Phase::DraftLoad, last: Phase::Growth })
    }

    /// The caller supplies its actual construction phase; Kimi co-loads before
    /// target weights, while ordinary post-target drafters begin in DraftLoad.
    pub(crate) fn allocation_receipts_with_lifetime(&self, devices: CoordinatorDevices<'_>, prefix: &str,
        lifetime: Lifetime) -> Result<Vec<Receipt>> {
        let mut inventory = Inventory::new(devices, prefix, lifetime, Category::Drafter, "draft owners");
        self.fc.allocation_receipts(&mut inventory, "weight.fc")?;
        for (name, owner) in [("hidden_norm", &self.hidden_norm), ("norm", &self.norm),
            ("taps", &self.taps), ("fused", &self.fused), ("fused_norm", &self.fused_norm),
            ("context_positions", &self.context_positions)] {
            owner.inventory(&mut inventory, name)?;
        }
        for (name, owner) in [("mask_rows", &self.mask_rows), ("markov_workspace", &self.markov_workspace),
            ("fp8_workspace", &self.fp8_workspace)] {
            if let Some(owner) = owner { owner.inventory(&mut inventory, name)?; }
        }
        inventory.host("mask_row", &self.mask_row)?;
        match &self.head_ops {
            DraftHeadOps::Peer => {}
            DraftHeadOps::Selector { projection, predecessor, successor } => {
                projection.allocation_receipts(&mut inventory, "head.projection")?;
                predecessor.inventory(&mut inventory, "head.predecessor")?;
                successor.inventory(&mut inventory, "head.successor")?;
            }
            DraftHeadOps::RnnMarkov { embedding, joint_w, joint_b, output, norms, .. } => {
                for (name, owner) in [("embedding", embedding), ("joint_w", joint_w), ("joint_b", joint_b),
                    ("output", output), ("norms", norms)] {
                    owner.inventory(&mut inventory, &format!("head.{name}"))?;
                }
            }
        }
        for (index, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layer.{index}");
            for (name, owner) in [("input_norm", &layer.input_norm), ("post_norm", &layer.post_norm)] {
                owner.inventory(&mut inventory, &format!("{prefix}.{name}"))?;
            }
            layer.gate_up.allocation_receipts(&mut inventory, &format!("{prefix}.gate_up"))?;
            layer.down.allocation_receipts(&mut inventory, &format!("{prefix}.down"))?;
            if let LayerBody::Dflash { attn_conv, attn_base, mlp_conv, mlp_base } = &layer.body {
                attn_conv.allocation_receipts(&mut inventory, &format!("{prefix}.attn_conv"))?;
                mlp_conv.allocation_receipts(&mut inventory, &format!("{prefix}.mlp_conv"))?;
                attn_base.inventory(&mut inventory, &format!("{prefix}.attn_base"))?;
                mlp_base.inventory(&mut inventory, &format!("{prefix}.mlp_base"))?;
            }
            match &layer.attention {
                DraftAttention::Gqa(attention) => {
                    attention.qkv.allocation_receipts(&mut inventory, &format!("{prefix}.qkv"))?;
                    attention.o.allocation_receipts(&mut inventory, &format!("{prefix}.o"))?;
                    for (name, owner) in [("q_norm", &attention.q_norm), ("k_norm", &attention.k_norm),
                        ("k_ring", &attention.k_ring), ("v_ring", &attention.v_ring)] {
                        owner.inventory(&mut inventory, &format!("{prefix}.{name}"))?;
                    }
                }
                DraftAttention::Mla(attention) => {
                    for (name, owner) in [("qkv_a", &attention.qkv_a), ("q_norm", &attention.q_norm),
                        ("kv_norm", &attention.kv_norm), ("q_b", &attention.q_b), ("uk", &attention.uk),
                        ("uv", &attention.uv), ("output", &attention.output), ("cache", &attention.cache),
                        ("page_table", &attention.page_table)] {
                        owner.inventory(&mut inventory, &format!("{prefix}.{name}"))?;
                    }
                }
            }
        }
        match &self.context {
            DraftContext::Gqa(context) => {
                inventory.device("context.kv", context.kv.buffer)?;
                inventory.device("context.slots", context.slots.buffer)?;
            }
            DraftContext::Mla(context) => {
                inventory.device("context.slots", context.slots.buffer)?;
                if let Some(split) = &context.split {
                    for (rank, name, buffer) in split.buffers() {
                        inventory.device(&format!("context.{rank}.{name}"), buffer)?;
                    }
                }
            }
        }
        if let Some(workspace) = self.workspace.try_borrow()?.as_ref() {
            for (name, owner) in [("h", &workspace.h), ("n", &workspace.n), ("delta", &workspace.delta),
                ("gate_up", &workspace.gate_up), ("act", &workspace.act), ("positions", &workspace.positions)] {
                owner.inventory(&mut inventory, &format!("workspace.{name}"))?;
            }
            if let Some(head) = &workspace.head {
                for (name, owner) in [("logits", &head.logits), ("anchors", &head.anchors), ("ids", &head.ids),
                    ("tokens", &head.tokens), ("head", &head._head_workspace)] {
                    owner.inventory(&mut inventory, &format!("workspace.{name}"))?;
                }
                inventory.pinned("workspace.landing", head.landing.buffer)?;
                for (name, owner) in [("unary", &head.unary), ("candidates", &head.candidates),
                    ("projected", &head.projected), ("features", &head.features), ("topk", &head.topk_workspace)] {
                    if let Some(owner) = owner { owner.inventory(&mut inventory, &format!("workspace.dflash.{name}"))?; }
                }
            }
            if let Some(workspace) = &workspace.dflash {
                for (name, owner) in [("conv", &workspace.conv), ("dynamic", &workspace.dynamic)] {
                    owner.inventory(&mut inventory, &format!("workspace.dflash.{name}"))?;
                }
            }
            match &workspace.attention {
                AttentionWorkspace::Gqa(workspace) => {
                    for (name, owner) in [("qkv", &workspace.qkv), ("q", &workspace.q), ("k", &workspace.k),
                        ("v", &workspace.v), ("attn", &workspace.attn), ("tables", &workspace.tables),
                        ("scratch", &workspace.scratch)] {
                        owner.inventory(&mut inventory, &format!("workspace.attention.{name}"))?;
                    }
                }
                AttentionWorkspace::Mla(workspace) => {
                    for (name, owner) in [("query", &workspace.query), ("attention", &workspace.attention),
                        ("lse", &workspace.lse), ("scratch", &workspace.scratch), ("lengths", &workspace.lengths),
                        ("query_offsets", &workspace.query_offsets), ("slots", &workspace.slots),
                        ("positions", &workspace.positions), ("inverse_frequencies", &workspace.inverse_frequencies)] {
                        owner.inventory(&mut inventory, &format!("workspace.attention.{name}"))?;
                    }
                    if let Some(owner) = &workspace.q_residual {
                        owner.inventory(&mut inventory, "workspace.attention.q_residual")?;
                    }
                }
            }
        }
        if let DraftContext::Mla(context) = &self.context {
            if let Some(split) = &context.split {
                for (rank, stream) in split.exchange_streams().enumerate() {
                    let first = inventory.receipts.len();
                    inventory.ordered_uploads(self.library, stream.raw)?;
                    // Staging is keyed by destination AND stream, not destination alone.
                    for receipt in &mut inventory.receipts[first..] {
                        receipt.allocation.id.push_str(&format!(".exchange.{rank}"));
                    }
                }
            }
        }
        inventory.ordered_uploads(self.library, self.stream)?;
        if let Some(split) = &self.head_split {
            inventory.device("workspace.reduced", split.reduced.buffer)?;
            inventory.device("peer.workspace.reduced", split.peer_reduced.buffer)?;
            inventory.receipts.extend(split.peer.device.run(||
                split.peer.allocation_receipts_with_lifetime(devices, &format!("{prefix}.peer"), lifetime))?);
        }
        Ok(inventory.receipts)
    }
}
