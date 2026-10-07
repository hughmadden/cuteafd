//! `plan_appends` on host-side tables: what a retained prefix sharing an appended tail costs,
//! and what counting its references as dropped changes. No device is needed.
use super::ownership::PagePool;
use super::*;

const ROWS: usize = PAGE_ROWS;

fn fits(pool: &PagePool, tables: &[Vec<u32>], rows: &[usize], appends: &[(usize, usize, usize)],
    released: &dyn Fn(u32) -> usize) -> Result<bool> {
    match plan_appends(pool, tables, rows, 0, appends, released) {
        Ok(_) => Ok(true),
        Err(error) if error.downcast_ref::<SourcePoolExhausted>().is_some() => Ok(false),
        Err(error) => Err(error),
    }
}

/// A seven-page source snapshot with a partial tail, restored into table 0 of a ten-page pool.
fn restored() -> (PagePool, Vec<u32>, usize) {
    let mut pool = PagePool::new(10);
    let source = pool.allocate(7).unwrap();
    pool.retain(&source);
    (pool, source, 6 * ROWS + 100)
}

#[test]
fn a_released_copy_sharer_lets_a_restored_table_append_like_a_cold_one() -> Result<()> {
    let (pool, source, end) = restored();
    let tail = source[6];
    let tables = [source.clone()];
    let none = |_: u32| 0;
    let snapshot = |page: u32| usize::from(page == tail);
    // Kept, the snapshot's reference makes the first append copy the tail: one page less.
    assert!(fits(&pool, &tables, &[end], &[(0, end, 9 * ROWS)], &none)?);
    assert!(!fits(&pool, &tables, &[end], &[(0, end, 9 * ROWS + 1)], &none)?);
    // Released, the table owns its tail alone and appends in place.
    assert!(fits(&pool, &tables, &[end], &[(0, end, 10 * ROWS)], &snapshot)?);
    assert!(!fits(&pool, &tables, &[end], &[(0, end, 10 * ROWS + 1)], &snapshot)?);
    let kept = plan_appends(&pool, &tables, &[end], 0, &[(0, end, end + 1)], &none)?;
    let dropped = plan_appends(&pool, &tables, &[end], 0, &[(0, end, end + 1)], &snapshot)?;
    assert_eq!((kept.replacements.len(), dropped.replacements.len()), (1, 0));
    // The same request cold, with nothing retained: the same ten pages.
    let cold = PagePool::new(10);
    assert!(fits(&cold, &[vec![]], &[0], &[(0, 0, 10 * ROWS)], &none)?);
    assert!(!fits(&cold, &[vec![]], &[0], &[(0, 0, 10 * ROWS + 1)], &none)?);
    Ok(())
}

#[test]
fn owners_that_all_append_keep_one_original_once_the_snapshot_is_released() -> Result<()> {
    let (mut pool, source, end) = restored();
    pool.retain(&source); // A second request restored from the same snapshot.
    let tail = source[6];
    let tables = [source.clone(), source.clone()];
    let appends = [(0, end, end + 8), (1, end, end + 8)];
    let kept = plan_appends(&pool, &tables, &[end, end], 0, &appends, &|_| 0)?;
    assert_eq!(kept.replacements.len(), 2, "the snapshot keeps the original, both writers copy");
    let released = plan_appends(&pool, &tables, &[end, end], 0, &appends, &|page| usize::from(page == tail))?;
    assert_eq!(released.replacements.len(), 1, "the first writer copies, the last keeps the original");
    assert_eq!(released.replacements[0].0, 0);
    Ok(())
}

#[test]
fn releasing_more_references_than_the_tables_leave_is_refused() -> Result<()> {
    let (pool, source, end) = restored();
    let tail = source[6];
    let error = plan_appends(&pool, &[source], &[end], 0, &[(0, end, end + 1)], &|page| 2 * usize::from(page == tail))
        .err().expect("over-release is refused");
    assert!(error.downcast_ref::<SourcePoolExhausted>().is_none());
    Ok(())
}

#[test]
fn planning_keeps_reserves_participant_checks() -> Result<()> {
    let (pool, source, end) = restored();
    let tables = [source.clone(), vec![]];
    let rows = [end, 0];
    let plan = |appends: &[(usize, usize, usize)], writing| plan_appends(&pool, &tables, &rows, writing, appends, &|_| 0);
    for (appends, writing, message) in [
        (vec![(2, 0, 1)], 0, "duplicate or invalid index slot"),
        (vec![(1, 0, 1), (1, 0, 2)], 0, "duplicate or invalid index slot"),
        (vec![(1, 0, 1)], 0b10, "compressed cache slot has a pending append"),
        (vec![(0, end - 1, end)], 0, "index history binding differs"),
        (vec![(0, end, end - 1)], 0, "index history binding differs"),
    ] {
        assert_eq!(plan(&appends, writing).err().expect("refused").to_string(), message, "{appends:?}");
    }
    let plan = plan(&[(1, 0, 2 * ROWS), (0, end, end + 1)], 0)?;
    assert_eq!(plan.used, 3, "two new pages, then one copy of the shared tail");
    assert_eq!(plan.lengths, [(1, 2 * ROWS as u64), (0, end as u64 + 1)]);
    assert_eq!(plan.replacements, [(0, 6, source[6], pool.free[0])]);
    Ok(())
}

#[test]
fn append_tails_are_the_partial_pages_appends_write() -> Result<()> {
    let tables = [vec![10, 11], vec![20, 21], vec![30]];
    let appends = [(0, ROWS + 5, ROWS + 6), (1, 2 * ROWS, 2 * ROWS + 1), (2, 7, 7)];
    // Slot 1's tail is full (the append starts a new page); slot 2 does not write.
    assert_eq!(append_tails(&tables, &appends)?, std::collections::HashSet::from([11]));
    assert!(append_tails(&tables, &[(0, 2 * ROWS + 5, 2 * ROWS + 6)]).is_err());
    Ok(())
}
