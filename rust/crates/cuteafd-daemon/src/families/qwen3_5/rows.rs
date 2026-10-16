//! Validate physical rows, then group each sequence for the native GDN ABI.
use crate::shared::{qwen::REPLAY_ROWS, speculation::target::{AcceptedPrefix, VerifyRequest}};
use anyhow::{ensure, Context, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Slot {
    pub position: usize,
    pub pages: Vec<u32>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Group {
    pub slot: usize,
    pub first: usize,
    pub rows: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Rows {
    pub tokens: Vec<u32>,
    pub physical: Vec<usize>,
    pub groups: Vec<Group>,
    pub slots: Vec<i32>,
    pub first: Vec<i32>,
    pub positions: Vec<i64>,
    pub kv: Vec<i64>,
    pub pages: Vec<i32>,
    pub stride: usize,
}

impl Rows {
    pub fn new(request: &VerifyRequest<'_>, layers: usize, vocab: usize,
        marked: &[(usize, Slot)], context: usize, pool: usize) -> Result<Self> {
        ensure!((1..=REPLAY_ROWS).contains(&request.rows.len()), "dense verify needs 1..64 physical rows");
        request.validate(layers, vocab, &marked.iter().map(|(slot, state)| (*slot, state.position)).collect::<Vec<_>>())?;
        let mut result = Self { tokens: Vec::new(), physical: Vec::new(), groups: Vec::new(),
            slots: Vec::new(), first: Vec::new(), positions: Vec::new(), kv: Vec::new(), pages: Vec::new(), stride: 0 };
        // Padding's sentinel slot and position never participate in grouping.
        for (slot, state) in marked {
            let active = request.rows.iter().enumerate().filter(|(_, row)| row.token.is_some() && row.slot == *slot)
                .collect::<Vec<_>>();
            let first = result.tokens.len();
            result.groups.push(Group { slot: *slot, first, rows: active.len() });
            if active.is_empty() { continue; }
            let end = state.position.checked_add(active.len()).context("dense verify position overflow")?;
            ensure!(end <= context, "dense verify exceeds context");
            if !state.pages.is_empty() {
                ensure!(end.div_ceil(64) <= state.pages.len(), "dense verify exceeds assigned pages");
                ensure!(state.pages.iter().all(|&page| (page as usize) < pool / 64), "dense page outside admitted pool");
            } else {
                ensure!(end <= pool, "dense verify exceeds admitted pool");
            }
            result.stride = result.stride.max(end.div_ceil(64));
            for (physical, row) in active {
                let position = row.position;
                let page = if state.pages.is_empty() { position / 64 } else { state.pages[position / 64] as usize };
                result.tokens.push(row.token.unwrap());
                result.physical.push(physical);
                result.slots.push(i32::try_from(*slot)?);
                result.first.push(i32::try_from(first)?);
                result.positions.push(i64::try_from(position)?);
                result.kv.push(i64::try_from(page.checked_mul(64).and_then(|v| v.checked_add(position % 64))
                    .context("dense physical KV position overflow")?)?);
            }
        }
        for group in &result.groups {
            if group.rows == 0 { continue; }
            let state = &marked.iter().find(|(slot, _)| *slot == group.slot).unwrap().1;
            let end = state.position + group.rows;
            let mut pages = vec![-1; result.stride];
            for (index, value) in pages.iter_mut().take(end.div_ceil(64)).enumerate() {
                *value = i32::try_from(if state.pages.is_empty() { index } else { state.pages[index] as usize })?;
            }
            for _ in 0..group.rows { result.pages.extend_from_slice(&pages); }
        }
        Ok(result)
    }

    pub fn accepted(&self, accepted: &[AcceptedPrefix]) -> Result<Vec<i32>> {
        ensure!(accepted.len() == self.groups.len(), "dense commit requires every marked slot exactly once");
        let mut table = Vec::with_capacity(3 * accepted.len());
        for group in &self.groups {
            let matches = accepted.iter().filter(|entry| entry.slot == group.slot).collect::<Vec<_>>();
            ensure!(matches.len() == 1 && matches[0].rows <= group.rows, "dense accepted prefix outside verified slot/input rows");
            table.push(i32::try_from(group.slot)?);
        }
        for group in &self.groups { table.push(i32::try_from(group.first)?); }
        for group in &self.groups {
            table.push(i32::try_from(accepted.iter().find(|entry| entry.slot == group.slot).unwrap().rows)?);
        }
        Ok(table)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::speculation::target::VerifyRow;

    #[test]
    fn interleaved_physical_rows_group_native_sequences_and_preserve_scatter() -> Result<()> {
        let pad = VerifyRow { slot: usize::MAX, position: usize::MAX, token: None };
        let rows = [pad, VerifyRow { slot: 7, position: 63, token: Some(8) },
            VerifyRow { slot: 2, position: 9, token: Some(3) }, pad,
            VerifyRow { slot: 7, position: 64, token: Some(9) }, VerifyRow { slot: 2, position: 10, token: Some(4) }];
        let marked = [(2, Slot { position: 9, pages: vec![5] }), (7, Slot { position: 63, pages: vec![9, 3] })];
        let plan = Rows::new(&VerifyRequest { rows: &rows, tap_layers: &[3, 0] }, 4, 10, &marked, 256, 1024)?;
        assert_eq!(plan.tokens, [3, 4, 8, 9]);
        assert_eq!(plan.physical, [2, 5, 1, 4]);
        assert_eq!(plan.slots, [2, 2, 7, 7]);
        assert_eq!(plan.first, [0, 0, 2, 2]);
        assert_eq!(plan.kv, [329, 330, 639, 192]);
        assert_eq!(plan.pages, [5, -1, 5, -1, 9, 3, 9, 3]);
        assert_eq!(plan.accepted(&[AcceptedPrefix { slot: 7, rows: 1 }, AcceptedPrefix { slot: 2, rows: 2 }])?,
            [2, 7, 0, 2, 2, 1]);
        for bad in [vec![], vec![AcceptedPrefix { slot: 2, rows: 1 }; 2],
            vec![AcceptedPrefix { slot: 2, rows: 3 }, AcceptedPrefix { slot: 7, rows: 1 }]] {
            assert!(plan.accepted(&bad).is_err());
        }
        Ok(())
    }

    #[test]
    fn padding_and_inactive_marked_slots_do_not_access_pages() -> Result<()> {
        let pad = VerifyRow { slot: usize::MAX, position: usize::MAX, token: None };
        let marked = [(3, Slot { position: usize::MAX, pages: vec![u32::MAX] }),
            (9, Slot { position: usize::MAX, pages: vec![] })];
        let plan = Rows::new(&VerifyRequest { rows: &[pad; 64], tap_layers: &[] }, 4, 10, &marked, 256, 256)?;
        assert!(plan.tokens.is_empty() && plan.pages.is_empty());
        assert_eq!(plan.accepted(&[AcceptedPrefix { slot: 9, rows: 0 }, AcceptedPrefix { slot: 3, rows: 0 }])?,
            [3, 9, 0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn invalid_pages_positions_and_duplicate_slots_fail_before_native_work() {
        let row = VerifyRow { slot: 3, position: 64, token: Some(2) };
        for state in [Slot { position: 63, pages: vec![0, 1] }, Slot { position: 64, pages: vec![0] },
            Slot { position: 64, pages: vec![0, 4] }] {
            assert!(Rows::new(&VerifyRequest { rows: &[row], tap_layers: &[] }, 4, 10, &[(3, state)], 256, 256).is_err());
        }
        let state = Slot { position: 64, pages: vec![0, 1] };
        assert!(Rows::new(&VerifyRequest { rows: &[row], tap_layers: &[] }, 4, 10,
            &[(3, state.clone()), (3, state)], 256, 256).is_err());
    }
}
