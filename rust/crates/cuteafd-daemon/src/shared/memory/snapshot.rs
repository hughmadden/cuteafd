//! Bounded snapshot storage allocated once, before serving becomes ready.
use super::*;
use anyhow::{ensure, Context};
use std::{cell::RefCell, rc::Rc};
use crate::shared::memory::Tag;

/// Reserve before enqueue: a failed partial enqueue must still own its source.
pub(crate) fn reserve_snapshot<T>(pending: &mut Option<T>, owner: T) -> Result<()> {
    ensure!(pending.is_none(), "snapshot lane is occupied");
    *pending = Some(owner);
    Ok(())
}

/// Failed synchronization leaves ownership intact; only proven completion frees it.
pub(crate) fn drain_snapshot<T>(pending: &mut Option<T>, drain: impl FnOnce() -> Result<()>) -> Result<()> {
    if pending.is_none() { return Ok(()); }
    drain()?;
    *pending = None;
    Ok(())
}

/// Stream destruction drains before the queued source/storage owners are dropped.
pub(crate) struct SnapshotCopies<'a, T> {
    pub stream: LoadStream<'a>,
    pub pending: Option<T>,
}
impl<'a, T> SnapshotCopies<'a, T> {
    pub fn new(library: &'a NativeLibrary) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("prefix/snapshot");
        Ok(Self { stream: LoadStream { library, raw: library.cuda_stream_create()? }, pending: None })
    }
    pub fn ready(&self) -> Result<bool> {
        ensure!(self.pending.is_some(), "snapshot copy is not pending");
        unsafe { self.stream.library.cuda_stream_query(self.stream.raw) }
    }
    pub fn abort(&mut self) -> Result<()> {
        drain_snapshot(&mut self.pending, || unsafe {
            self.stream.library.cuda_stream_synchronize(self.stream.raw)
        })
    }
}

struct Arena<'a> {
    allocation: DeviceAllocation<'a>,
    stride: usize,
    free: RefCell<Vec<usize>>,
}

#[derive(Clone)]
pub(crate) struct SnapshotPool<'a>(Rc<Arena<'a>>);
impl<'a> SnapshotPool<'a> {
    pub fn new(library: &'a NativeLibrary, bytes: usize, slots: usize) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("prefix/snapshot");
        ensure!(bytes > 0 && slots > 0, "empty snapshot arena");
        let stride = bytes.checked_add(255).context("snapshot stride overflow")? / 256 * 256;
        let allocation = DeviceAllocation::new(library,
            stride.checked_mul(slots).context("snapshot arena overflow")?, Tag::prefix("shared/prefix-snapshots"))?;
        Ok(Self(Rc::new(Arena { allocation, stride, free: RefCell::new((0..slots).rev().collect()) })))
    }
    pub fn device_bytes(&self) -> usize { self.0.allocation.buffer.bytes }
    pub(crate) fn allocation_receipts(&self, inventory: &mut super::receipts::Inventory<'_>,
        name: &str) -> Result<()> {
        inventory.backed_device(name, self.0.allocation.buffer, self.0.allocation.buffer)?;
        let free = self.0.free.try_borrow()?;
        inventory.host(&format!("{name}.free"), &free)?;
        Ok(())
    }
    pub fn take(&self, bytes: usize) -> Result<SnapshotStorage<'a>> {
        ensure!(bytes > 0 && bytes <= self.0.stride, "snapshot exceeds arena slot");
        let slot = self.0.free.borrow_mut().pop().context("snapshot arena exhausted")?;
        let mut buffer = self.0.allocation.buffer;
        buffer.ptr = unsafe { buffer.ptr.cast::<u8>().add(slot * self.0.stride).cast() };
        buffer.bytes = bytes;
        Ok(SnapshotStorage { buffer, _owned: None, pooled: Some((self.clone(), slot)) })
    }
}

/// The owner must drain every reader/writer before dropping this storage.
/// A pooled slot returns to the free list without calling the CUDA allocator.
pub(crate) struct SnapshotStorage<'a> {
    pub buffer: CuteafdDeviceBuffer,
    _owned: Option<DeviceAllocation<'a>>,
    pooled: Option<(SnapshotPool<'a>, usize)>,
}
impl<'a> SnapshotStorage<'a> {
    pub(crate) fn allocation_receipts(&self, inventory: &mut super::receipts::Inventory<'_>,
        name: &str) -> Result<()> {
        let backing = self.pooled.as_ref().map(|(pool, _)| pool.0.allocation.buffer)
            .or_else(|| self._owned.as_ref().map(|owner| owner.buffer))
            .context("snapshot storage has no actual owner")?;
        inventory.backed_device(name, self.buffer, backing)
    }
    pub fn new(library: &'a NativeLibrary, bytes: usize, pool: Option<&SnapshotPool<'a>>) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("prefix/snapshot");
        if let Some(pool) = pool { return pool.take(bytes); }
        let owned = DeviceAllocation::new(library, bytes, Tag::prefix("shared/prefix-snapshot"))?;
        Ok(Self { buffer: owned.buffer, _owned: Some(owned), pooled: None })
    }
}
impl Drop for SnapshotStorage<'_> {
    fn drop(&mut self) {
        if let Some((pool, slot)) = self.pooled.take() { pool.0.free.borrow_mut().push(slot); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_enqueue_keeps_reservations_until_its_own_lane_drains() {
        use std::cell::Cell;
        struct Owner(Rc<Cell<usize>>);
        impl Drop for Owner { fn drop(&mut self) { self.0.set(self.0.get() + 1); } }
        for (target_ok, draft_ok) in [(true, true), (false, true), (true, false), (false, false)] {
            let drops = Rc::new(Cell::new(0));
            let mut target = None;
            let mut draft = None;
            reserve_snapshot(&mut target, Owner(drops.clone())).unwrap();
            reserve_snapshot(&mut draft, Owner(drops.clone())).unwrap();
            // Inject a partial enqueue error before any successful enqueue result.
            // The ordinary stream is independently healthy and owns neither lane.
            let ordinary: Result<()> = Ok(());
            assert!(ordinary.is_ok() && target.is_some() && draft.is_some());
            assert!(reserve_snapshot(&mut target, Owner(Rc::new(Cell::new(0)))).is_err());
            let target_result = drain_snapshot(&mut target, || {
                ensure!(target_ok, "target lane pending"); Ok(())
            });
            let draft_result = drain_snapshot(&mut draft, || {
                ensure!(draft_ok, "draft lane pending"); Ok(())
            });
            assert_eq!(target_result.is_ok(), target_ok);
            assert_eq!(draft_result.is_ok(), draft_ok);
            assert_eq!(target.is_some(), !target_ok);
            assert_eq!(draft.is_some(), !draft_ok);
            assert_eq!(drops.get(), usize::from(target_ok) + usize::from(draft_ok));
            // A shutdown owner settles each reservation once; repeated safe aborts
            // never run the drain callback again or release an owner twice.
            drain_snapshot(&mut target, || Ok(())).unwrap();
            drain_snapshot(&mut draft, || Ok(())).unwrap();
            drain_snapshot(&mut target, || panic!("already settled")).unwrap();
            drain_snapshot(&mut draft, || panic!("already settled")).unwrap();
            assert_eq!(drops.get(), 2);
        }
    }

    #[test]
    fn snapshot_census_charges_shared_backing_once_without_native_work() -> Result<()> {
        use cuteafd_core::memory_layout::Category;
        use cuteafd_loader::placement::planner::{CoordinatorDevices, Lifetime, Location};
        let (_directory, fixture) = super::super::cpu_stream_fixture()?;
        let library = fixture.load()?;
        let pool = SnapshotPool::new(&library, 513, 3)?;
        let first = pool.take(513)?;
        let second = pool.take(129)?;
        let before = fixture.events()?;
        let physical = [0];
        let mut inventory = super::super::receipts::Inventory::new(CoordinatorDevices::new(&physical)?,
            "fixture", Lifetime::TARGET, Category::Prefix, "snapshot");
        pool.allocation_receipts(&mut inventory, "pool")?;
        first.allocation_receipts(&mut inventory, "first")?;
        second.allocation_receipts(&mut inventory, "second")?;
        let owners: Vec<_> = inventory.receipts.iter().filter(|r|
            r.allocation.location == Location::coordinator(0) && r.allocation.view.is_none()).collect();
        assert_eq!(owners.len(), 1);
        assert_eq!(owners[0].allocation.bytes, pool.device_bytes() as u64);
        for (name, actual) in [("first", first.buffer), ("second", second.buffer)] {
            let receipt = inventory.receipts.iter().find(|r| r.allocation.id == format!("fixture.{name}")).unwrap();
            assert_eq!((receipt.address, receipt.allocation.bytes), (actual.ptr as u64, actual.bytes as u64));
            assert_eq!(receipt.allocation.view.as_ref().unwrap().offset, actual.ptr as u64 - owners[0].address);
        }
        assert_eq!(fixture.events()?, before);
        Ok(())
    }
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and CUDA"]
    fn native_snapshot_arena_reuses_only_released_slots() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let pool = SnapshotPool::new(&lib, 513, 2)?;
        assert!(pool.take(769).is_err());
        let first = pool.take(513)?;
        let second = pool.take(129)?;
        assert_ne!(first.buffer.ptr, second.buffer.ptr);
        assert!(pool.take(1).is_err());
        lib.copy_h2d(first.buffer, &[17; 513])?;
        lib.copy_h2d(second.buffer, &[29; 129])?;
        let address = first.buffer.ptr;
        drop(first);
        let reused = pool.take(513)?;
        assert_eq!(reused.buffer.ptr, address);
        lib.copy_h2d(reused.buffer, &[41; 513])?;
        // Outstanding storage owns the arena even when its allocator is dropped.
        drop(pool);
        let mut actual = [0; 129];
        lib.copy_d2h(&mut actual, second.buffer)?;
        assert_eq!(actual, [29; 129]);
        Ok(())
    }
}
