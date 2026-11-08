//! Final coordinator allocation scopes, derived from the native operand contract.
use super::resident::ResidentOperand;
use crate::serving_capacity::WeightArenaLayout;
use anyhow::Result;

pub fn operand_name(item: &ResidentOperand) -> String {
    if let Some(layer) = item.layer {
        format!("layers.{layer}.{}", item.operand)
    } else if let Some(stage) = item.draft_stage {
        format!("mtp.{stage}.{}", item.operand)
    } else if item.group == "embedding" {
        "embedding.table".into()
    } else {
        item.operand.clone()
    }
}

/// The optional drafter is independently admitted and never shares target backing.
pub fn coordinator_arena(operands: &[ResidentOperand], rank: usize, draft: bool,
    device_embedding: bool) -> Result<Option<WeightArenaLayout>> {
    let requests = operands.iter().filter(|item| item.rank == rank
        && (item.group == "speculator") == draft
        && (item.group != "embedding" || device_embedding))
        .map(|item| Ok((operand_name(item), usize::try_from(item.bytes)?)))
        .collect::<Result<Vec<_>>>()?;
    if requests.is_empty() { return Ok(None); }
    Ok(Some(WeightArenaLayout::new(format!("v4/{}/{rank}", if draft { "draft" } else { "target" }), requests)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_rank_and_lifetime_and_keeps_final_operand_extents() -> Result<()> {
        let make = |rank, layer, draft_stage, operand: &str, group: &str, bytes| ResidentOperand {
            rank, layer, draft_stage, operand: operand.into(), group: group.into(),
            format: "fixture".into(), bytes, attention: false,
        };
        let items = vec![make(0, Some(0), None, "w_q_scale", "target", 513),
            make(1, Some(0), None, "w_q_scale", "target", 257),
            make(0, None, None, "embed.weight", "embedding", 1024),
            make(0, None, Some(0), "w_q_scale", "speculator", 512),
            make(0, None, None, "mtp.0.main_proj.scale", "speculator", 3)];
        for embedding in [false, true] {
            let target = coordinator_arena(&items, 0, false, embedding)?.unwrap();
            assert_eq!(target.views().len(), 1 + usize::from(embedding));
            assert_eq!(target.view("layers.0.w_q_scale").unwrap().bytes(), 513);
            assert_eq!(target.bytes(), if embedding { 1792 } else { 513 });
            assert_eq!(target.padding_bytes(), if embedding { 255 } else { 0 });
            assert!(!target.views().keys().any(|name| name.starts_with("mtp.")));
        }
        let peer = coordinator_arena(&items, 1, false, true)?.unwrap();
        assert_eq!(peer.bytes(), 257);
        let draft = coordinator_arena(&items, 0, true, true)?.unwrap();
        assert_eq!(draft.bytes(), 515);
        assert_eq!(draft.view("mtp.0.main_proj.scale").unwrap().bytes(), 3);
        assert!(coordinator_arena(&items, 1, true, true)?.is_none());
        for layout in [peer, draft] {
            let mut ranges = layout.views().values().map(|v| {
                assert_eq!(v.offset() % 256, 0);
                v.offset()..v.offset() + v.bytes()
            }).collect::<Vec<_>>();
            ranges.sort_by_key(|r| r.start);
            assert!(ranges.windows(2).all(|w| w[0].end <= w[1].start));
            assert_eq!(ranges.last().unwrap().end, layout.bytes());
        }
        let mut duplicate = items.clone();
        duplicate.push(items[0].clone());
        assert!(coordinator_arena(&duplicate, 0, false, true).is_err());
        Ok(())
    }
}
