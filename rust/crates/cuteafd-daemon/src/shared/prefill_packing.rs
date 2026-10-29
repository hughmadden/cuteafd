//! Row and completion contracts shared by packed prefill families.
use anyhow::{ensure, Result};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum PackingMode {
    /// Fill the admitted unit, accepting the same route switches as long prompts.
    #[default]
    Unrestricted,
    /// Keep packs on the first coordinator and expert routes; unknown routes run alone.
    Exact,
}

impl PackingMode {
    pub(crate) fn resolve(explicit: Option<Self>, packed_check: bool) -> Self {
        explicit.unwrap_or(if packed_check { Self::Exact } else { Self::Unrestricted })
    }
}

/// How a coordinator program receives rows in a packed pass. No implicit role:
/// even a projection may be sequence-local when fused into an attention producer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProgramRows { PackedRows, SequenceLocal }

/// Intersect the first arithmetic routes published by all packed-row programs.
/// Missing/invalid metadata is not evidence of row invariance: run alone.
pub(crate) fn coordinator_row_cap<'a>(programs: &cuteafd_ffi::programs::Programs<'_>,
    inventory: impl IntoIterator<Item = (&'a str, ProgramRows)>, capacity: usize) -> Result<usize> {
    let mut cap = capacity;
    for (name, role) in inventory {
        let spec = programs.spec(name)?;
        if role == ProgramRows::PackedRows { cap = cap.min(manifest_row_cap(&spec.geometry)); }
    }
    Ok(cap)
}

fn manifest_row_cap(geometry: &serde_json::Value) -> usize {
    geometry.get("exact_row_cap").and_then(serde_json::Value::as_u64)
        .and_then(|v| usize::try_from(v).ok()).unwrap_or(0)
}

/// Compare the owner's packed-path catalogue against its explicitly classified
/// inventory. A newly added program cannot silently inherit a role.
pub(crate) fn validate_program_inventory<'a>(required: impl IntoIterator<Item = &'a str>,
    inventory: &[(&str, ProgramRows)]) -> Result<()> {
    let required: std::collections::BTreeSet<_> = required.into_iter().collect();
    let classified: std::collections::BTreeSet<_> = inventory.iter().map(|(name, _)| *name).collect();
    ensure!(classified.len() == inventory.len() && required == classified,
        "packed program inventory differs: required {required:?}, classified {classified:?}");
    Ok(())
}

/// A scoped packed pass checks every coordinator launch against its owner's
/// explicit inventory, including sequence-local helpers. Future programs have
/// no default role and fail before launch until the inventory is updated.
#[derive(Default)]
pub(crate) struct ProgramInventory {
    #[cfg(debug_assertions)]
    active: std::cell::RefCell<Option<std::collections::HashMap<String, ProgramRows>>>,
}

impl ProgramInventory {
    pub(crate) fn enter(&self, entries: impl FnOnce() -> Vec<(String, ProgramRows)>) -> Result<ProgramInventoryGuard<'_>> {
        #[cfg(debug_assertions)] {
            let entries = entries();
            let borrowed: Vec<_> = entries.iter().map(|(name, role)| (name.as_str(), *role)).collect();
            validate_program_inventory(entries.iter().map(|(name, _)| name.as_str()), &borrowed)?;
            let mut active = self.active.borrow_mut();
            ensure!(active.is_none(), "nested packed program inventory");
            *active = Some(entries.into_iter().collect());
        }
        #[cfg(not(debug_assertions))]
        let _ = entries;
        Ok(ProgramInventoryGuard(self))
    }

    #[inline(always)]
    pub(crate) fn check(&self, name: &str) -> Result<()> {
        #[cfg(debug_assertions)]
        if let Some(active) = self.active.borrow().as_ref() {
            ensure!(active.contains_key(name), "unclassified packed-path program {name}");
        }
        #[cfg(not(debug_assertions))]
        let _ = name;
        Ok(())
    }
}

pub(crate) struct ProgramInventoryGuard<'a>(&'a ProgramInventory);
impl Drop for ProgramInventoryGuard<'_> {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        { *self.0.active.borrow_mut() = None; }
        #[cfg(not(debug_assertions))]
        let _ = self.0;
    }
}

/// Greedy adjacent groups, retaining oversized requests as ordinary singletons.
/// Splitting a request itself would change the alone reference's arithmetic.
pub(crate) fn groups(members: usize, fits: impl Fn(std::ops::Range<usize>) -> bool)
    -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let mut first = 0;
    while first < members {
        let mut end = first + 1;
        while end < members && fits(first..end + 1) { end += 1; }
        out.push(first..end);
        first = end;
    }
    out
}

pub(crate) fn log_mode(mode: PackingMode, rows: usize) {
    if mode == PackingMode::Exact {
        if rows == 0 {
            tracing::info!("exact packed prefill disabled: loaded coordinator/expert route metadata cannot bound arithmetic transitions (including occupancy-dependent NVFP4 sparse/wide); requests run alone");
        } else { tracing::info!(rows, "exact packed prefill uses the minimum loaded coordinator/expert route cap"); }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Limits {
    pub rows: usize,
    pub chunk_rows: usize,
    pub max_context: usize,
    pub taps: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Segment {
    pub first: usize,
    pub rows: usize,
    pub start: usize,
    pub tap_offset: usize,
    pub tap_rows: usize,
}

impl Segment {
    pub fn end_row(&self) -> usize { self.first + self.rows }
}

/// Lay out already-queued chunks, never collecting or waiting for more work.
/// Family-specific table/state capacities are checked in addition to this plan.
pub(crate) fn plan(chunks: &[(usize, usize)], limits: &Limits) -> Result<Vec<Segment>> {
    ensure!(!chunks.is_empty(), "a packed prefill needs a sequence");
    let (mut first, mut taps) = (0usize, 0usize);
    let mut out = Vec::with_capacity(chunks.len());
    for &(start, rows) in chunks {
        ensure!(rows > 0 && rows <= limits.chunk_rows, "packed chunk exceeds its row capacity");
        let end = start.checked_add(rows).ok_or_else(|| anyhow::anyhow!("packed position overflow"))?;
        ensure!(end <= limits.max_context, "packed chunk exceeds its context");
        let tap_rows = limits.taps.map_or(0, |cap| rows.min(cap));
        out.push(Segment { first, rows, start, tap_offset: taps, tap_rows });
        first = first.checked_add(rows).ok_or_else(|| anyhow::anyhow!("packed row overflow"))?;
        taps = taps.checked_add(tap_rows).ok_or_else(|| anyhow::anyhow!("packed tap overflow"))?;
        ensure!(first <= limits.rows, "packed rows exceed the admitted unit");
        ensure!(limits.taps.is_none_or(|cap| taps <= cap), "packed taps exceed admitted storage");
    }
    Ok(out)
}

/// Outer failure invalidates shared device work; inner failure belongs to one client.
pub(crate) type MemberOutcome = Result<Result<()>>;

/// Heads use one scratch buffer. Consume each head before generating the next.
pub(crate) fn finish_members<L>(members: usize, head: impl FnMut(usize) -> Result<L>,
    on_logits: &mut dyn FnMut(usize, L) -> MemberOutcome) -> Result<Vec<Result<()>>> {
    finish_members_selected(members, |_| true, head, on_logits)
}

/// Intermediate chunks and cancelled members have no head consumer.
/// Keep their outcome slots without launching an unused vocabulary projection.
pub(crate) fn finish_members_selected<L>(members: usize, mut needs_head: impl FnMut(usize) -> bool,
    mut head: impl FnMut(usize) -> Result<L>,
    on_logits: &mut dyn FnMut(usize, L) -> MemberOutcome) -> Result<Vec<Result<()>>> {
    (0..members).map(|i| if needs_head(i) { on_logits(i, head(i)?) } else { Ok(Ok(())) }).collect()
}

pub(crate) fn member_outcomes(members: usize, pass: Result<Vec<Result<()>>>) -> Vec<Result<()>> {
    match pass {
        Ok(outcomes) => {
            if outcomes.len() != members {
                return (0..members).map(|_| Err(anyhow::anyhow!("packed prefill outcome count differs"))).collect();
            }
            outcomes
        }
        Err(error) => {
            let message = format!("packed prefill of {members} prompts: {error:#}");
            (0..members).map(|_| Err(anyhow::anyhow!("{message}"))).collect()
        }
    }
}

pub(crate) fn row_shares(rows: &[usize]) -> Vec<f64> {
    let total: usize = rows.iter().sum();
    rows.iter().map(|&r| if total == 0 { 1.0 / rows.len() as f64 } else { r as f64 / total as f64 }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn limits() -> Limits { Limits { rows: 256, chunk_rows: 256, max_context: 4096, taps: Some(128) } }
    #[test]
    fn coordinator_routes_fail_closed_and_inventory_is_exhaustive() {
        use ProgramRows::*;
        assert_eq!(manifest_row_cap(&serde_json::json!({"exact_row_cap": 64})), 64);
        for value in [serde_json::json!({}), serde_json::json!({"exact_row_cap": -1}),
            serde_json::json!({"exact_row_cap": "64"}), serde_json::json!({"exact_row_cap": 0})] {
            assert_eq!(manifest_row_cap(&value), 0);
        }
        let inventory = [("router", PackedRows), ("attention", SequenceLocal)];
        validate_program_inventory(["router", "attention"], &inventory).unwrap();
        assert!(validate_program_inventory(["router", "attention", "new_projection"], &inventory).is_err());
        assert!(validate_program_inventory(["router"], &inventory).is_err());
        assert!(validate_program_inventory(["router"], &[("router", PackedRows), ("router", SequenceLocal)]).is_err());
    }

    #[test]
    #[cfg(debug_assertions)]
    fn every_packed_launch_requires_an_explicit_role_and_guard_resets_on_error() {
        use ProgramRows::*;
        let inventory = ProgramInventory::default();
        inventory.check("outside_packed_pass").unwrap();
        {
            let _pass = inventory.enter(|| vec![("router".into(), PackedRows), ("attention".into(), SequenceLocal)]).unwrap();
            inventory.check("router").unwrap();
            inventory.check("attention").unwrap();
            assert!(inventory.check("new_projection").is_err());
            assert!(inventory.enter(|| vec![]).is_err());
        }
        inventory.check("outside_packed_pass").unwrap();
        assert!(inventory.enter(|| vec![("router".into(), PackedRows), ("router".into(), SequenceLocal)]).is_err());
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn release_inventory_has_no_storage_or_launch_validation() {
        assert_eq!(std::mem::size_of::<ProgramInventory>(), 0);
        let inventory = ProgramInventory::default();
        let _pass = inventory.enter(|| panic!("release must not build the inventory")).unwrap();
        inventory.check("unclassified_release_launch").unwrap();
    }

    #[test]
    fn exact_coordinator_router_boundary_retains_large_singletons() {
        let chunks = [(0, 32), (0, 80), (0, 16), (0, 48)];
        let cap = 256usize.min(manifest_row_cap(&serde_json::json!({"exact_row_cap": 64})));
        let split = groups(chunks.len(), |range| plan(&chunks[range], &Limits {
            rows: cap, chunk_rows: 256, max_context: 4096, taps: None }).is_ok());
        assert_eq!(split, vec![0..1, 1..2, 2..4]);
        assert!(plan(&chunks[..2], &Limits { rows: 256, taps: None, ..limits() }).is_ok());
        assert_eq!(chunks[1].1, 80, "a singleton must not be rechunked at the pack cap");
    }

    #[test]
    fn exact_groups_split_at_cap_without_rechunking_singletons() {
        assert_eq!(PackingMode::default(), PackingMode::Unrestricted);
        assert_eq!(PackingMode::resolve(None, false), PackingMode::Unrestricted);
        assert_eq!(PackingMode::resolve(None, true), PackingMode::Exact);
        assert_eq!(PackingMode::resolve(Some(PackingMode::Unrestricted), true), PackingMode::Unrestricted);
        assert_eq!(PackingMode::resolve(Some(PackingMode::Exact), false), PackingMode::Exact);
        let chunks = [(0, 300), (100, 340), (0, 1), (0, 700), (0, 29), (0, 40)];
        let split = |cap| groups(chunks.len(), |range|
            plan(&chunks[range], &Limits { rows: cap, chunk_rows: 4096, max_context: 4096, taps: None }).is_ok());
        assert_eq!(split(640), vec![0..2, 2..3, 3..4, 4..6]);
        assert_eq!(split(4096), vec![0..6]);
        assert_eq!(split(0), (0..6).map(|i| i..i + 1).collect::<Vec<_>>());
        assert!(groups(0, |_| true).is_empty());
    }

    #[test]
    fn layout_has_private_rows_and_taps_without_extra_storage() {
        let p = plan(&[(200, 40), (0, 80), (3000, 8)], &limits()).unwrap();
        assert_eq!(p.iter().map(|s| (s.first, s.end_row(), s.tap_offset, s.tap_rows)).collect::<Vec<_>>(),
            [(0, 40, 0, 40), (40, 120, 40, 80), (120, 128, 120, 8)]);
        assert!(plan(&[(200, 40), (0, 89)], &limits()).is_err());
        assert!(plan(&[(usize::MAX, 2)], &limits()).is_err());
        assert!(plan(&[(4096, 1)], &limits()).is_err());
        assert!(plan(&[(0, 0)], &limits()).is_err());
        assert!(plan(&[], &limits()).is_err());
        assert!(plan(&[(0, 200), (0, 57)], &Limits { taps: None, ..limits() }).is_err());
    }
    #[test]
    fn intermediate_and_cancelled_members_launch_no_head() -> Result<()> {
        let mut heads = Vec::new();
        let mut consumed = Vec::new();
        let outcomes = finish_members_selected(4, |i| i == 1 || i == 3,
            |i| { heads.push(i); Ok(i) }, &mut |i, row| {
                consumed.push(row);
                Ok(if i == 1 { Err(anyhow::anyhow!("grammar")) } else { Ok(()) })
            })?;
        assert_eq!(heads, [1, 3]);
        assert_eq!(consumed, [1, 3]);
        assert!(outcomes[0].is_ok() && outcomes[1].is_err() && outcomes[2].is_ok() && outcomes[3].is_ok());
        Ok(())
    }

    #[test]
    fn local_failure_does_not_stop_other_heads_but_device_failure_does() {
        let mut seen = Vec::new();
        let outcomes = finish_members(3, Ok, &mut |i, row| {
            seen.push(row);
            Ok(if i == 1 { Err(anyhow::anyhow!("grammar")) } else { Ok(()) })
        }).unwrap();
        assert_eq!(seen, [0, 1, 2]);
        assert!(outcomes[0].is_ok() && outcomes[1].is_err() && outcomes[2].is_ok());
        seen.clear();
        let result = finish_members(3, |i| { ensure!(i != 1, "device"); Ok(i) }, &mut |_, row| {
            seen.push(row); Ok(Ok(()))
        });
        assert_eq!(seen, [0]);
        assert!(member_outcomes(3, result).iter().all(Result::is_err));
        assert!(member_outcomes(2, Ok(vec![Ok(())])).iter().all(Result::is_err));
        assert!(member_outcomes(1, Ok(vec![Ok(()), Ok(())])).iter().all(Result::is_err));
    }
}
