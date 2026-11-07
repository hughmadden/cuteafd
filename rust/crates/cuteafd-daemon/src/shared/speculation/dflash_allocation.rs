//! Retained draft owners only; program/graph runtime memory remains opaque.
use super::*;
use crate::shared::memory::receipts::Inventory;
use cuteafd_core::memory_layout::Category;
use cuteafd_loader::placement::planner::{CoordinatorDevices, Lifetime, Phase, Receipt};

impl DflashDrafter<'_> {
    pub(crate) fn allocation_receipts(&self, devices: CoordinatorDevices<'_>, prefix: &str) -> Result<Vec<Receipt>> {
        let mut inventory = Inventory::new(devices, prefix,
            Lifetime { first: Phase::DraftLoad, last: Phase::Growth }, Category::Drafter, "draft owners");
        self.fc.allocation_receipts(&mut inventory, "weight.fc")?;
        for (name, owner) in [("hidden_norm", &self.hidden_norm), ("norm", &self.norm),
            ("taps", &self.taps), ("fused", &self.fused), ("fused_norm", &self.fused_norm),
            ("context_positions", &self.context_positions)] {
            inventory.device(name, owner.buffer)?;
        }
        for (name, owner) in [("mask_rows", &self.mask_rows), ("markov_workspace", &self.markov_workspace),
            ("fp8_workspace", &self.fp8_workspace)] {
            if let Some(owner) = owner { inventory.device(name, owner.buffer)?; }
        }
        inventory.host("mask_row", &self.mask_row)?;
        match &self.head_ops {
            DraftHeadOps::Selector { projection, predecessor, successor } => {
                projection.allocation_receipts(&mut inventory, "head.projection")?;
                inventory.device("head.predecessor", predecessor.buffer)?;
                inventory.device("head.successor", successor.buffer)?;
            }
            DraftHeadOps::RnnMarkov { embedding, joint_w, joint_b, output, norms, .. } => {
                for (name, owner) in [("embedding", embedding), ("joint_w", joint_w), ("joint_b", joint_b),
                    ("output", output), ("norms", norms)] {
                    inventory.device(&format!("head.{name}"), owner.buffer)?;
                }
            }
        }
        for (index, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layer.{index}");
            for (name, owner) in [("input_norm", &layer.input_norm), ("post_norm", &layer.post_norm)] {
                inventory.device(&format!("{prefix}.{name}"), owner.buffer)?;
            }
            layer.gate_up.allocation_receipts(&mut inventory, &format!("{prefix}.gate_up"))?;
            layer.down.allocation_receipts(&mut inventory, &format!("{prefix}.down"))?;
            if let LayerBody::Dflash { attn_conv, attn_base, mlp_conv, mlp_base } = &layer.body {
                attn_conv.allocation_receipts(&mut inventory, &format!("{prefix}.attn_conv"))?;
                mlp_conv.allocation_receipts(&mut inventory, &format!("{prefix}.mlp_conv"))?;
                inventory.device(&format!("{prefix}.attn_base"), attn_base.buffer)?;
                inventory.device(&format!("{prefix}.mlp_base"), mlp_base.buffer)?;
            }
            match &layer.attention {
                DraftAttention::Gqa(attention) => {
                    attention.qkv.allocation_receipts(&mut inventory, &format!("{prefix}.qkv"))?;
                    attention.o.allocation_receipts(&mut inventory, &format!("{prefix}.o"))?;
                    for (name, owner) in [("q_norm", &attention.q_norm), ("k_norm", &attention.k_norm),
                        ("k_ring", &attention.k_ring), ("v_ring", &attention.v_ring)] {
                        inventory.device(&format!("{prefix}.{name}"), owner.buffer)?;
                    }
                }
                DraftAttention::Mla(attention) => {
                    for (name, owner) in [("qkv_a", &attention.qkv_a), ("q_norm", &attention.q_norm),
                        ("kv_norm", &attention.kv_norm), ("q_b", &attention.q_b), ("uk", &attention.uk),
                        ("uv", &attention.uv), ("output", &attention.output), ("cache", &attention.cache),
                        ("page_table", &attention.page_table)] {
                        inventory.device(&format!("{prefix}.{name}"), owner.buffer)?;
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
                ("gate_up", &workspace.gate_up), ("act", &workspace.act), ("logits", &workspace.logits),
                ("anchors", &workspace.anchors), ("ids", &workspace.ids), ("tokens", &workspace.tokens),
                ("positions", &workspace.positions), ("head", &workspace._head_workspace)] {
                inventory.device(&format!("workspace.{name}"), owner.buffer)?;
            }
            inventory.pinned("workspace.landing", workspace.landing.buffer)?;
            if let Some(workspace) = &workspace.dflash {
                for (name, owner) in [("conv", &workspace.conv), ("dynamic", &workspace.dynamic),
                    ("unary", &workspace.unary), ("candidates", &workspace.candidates),
                    ("projected", &workspace.projected), ("features", &workspace.features),
                    ("topk", &workspace.topk_workspace)] {
                    inventory.device(&format!("workspace.dflash.{name}"), owner.buffer)?;
                }
            }
            match &workspace.attention {
                AttentionWorkspace::Gqa(workspace) => {
                    for (name, owner) in [("qkv", &workspace.qkv), ("q", &workspace.q), ("k", &workspace.k),
                        ("v", &workspace.v), ("attn", &workspace.attn), ("tables", &workspace.tables),
                        ("scratch", &workspace.scratch)] {
                        inventory.device(&format!("workspace.attention.{name}"), owner.buffer)?;
                    }
                }
                AttentionWorkspace::Mla(workspace) => {
                    for (name, owner) in [("query", &workspace.query), ("attention", &workspace.attention),
                        ("lse", &workspace.lse), ("scratch", &workspace.scratch), ("lengths", &workspace.lengths),
                        ("query_offsets", &workspace.query_offsets), ("slots", &workspace.slots),
                        ("positions", &workspace.positions), ("inverse_frequencies", &workspace.inverse_frequencies)] {
                        inventory.device(&format!("workspace.attention.{name}"), owner.buffer)?;
                    }
                    if let Some(owner) = &workspace.q_residual {
                        inventory.device("workspace.attention.q_residual", owner.buffer)?;
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
                split.peer.allocation_receipts(devices, &format!("{prefix}.peer")))?);
        }
        Ok(inventory.receipts)
    }
}
