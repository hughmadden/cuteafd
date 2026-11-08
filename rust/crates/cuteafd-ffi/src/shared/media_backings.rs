//! The ledger side of a native media runtime's two device backings (weights,
//! scratch). Recorded under the tower's tags right after create; each retired
//! only when native reports that its free succeeded.
use crate::memory_ledger::{self, MediaTags, Space};
use libloading::Library;
use std::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RawBackings {
    pub weights: *mut c_void,
    pub weight_bytes: u64,
    pub workspace: *mut c_void,
    pub workspace_bytes: u64,
}
impl Default for RawBackings {
    fn default() -> Self {
        Self { weights: std::ptr::null_mut(), weight_bytes: 0, workspace: std::ptr::null_mut(), workspace_bytes: 0 }
    }
}

pub(crate) const RELEASED_WEIGHTS: u32 = 1;
pub(crate) const RELEASED_WORKSPACE: u32 = 2;

type GetBackings = unsafe extern "C" fn(*mut c_void, *mut RawBackings) -> i32;
type DestroyBackings = unsafe extern "C" fn(*mut c_void, *mut u32) -> i32;

/// What the ledger holds for one native media owner.
#[derive(Debug, Default)]
pub(crate) struct MediaBackings {
    weights: usize,
    workspace: usize,
}

impl MediaBackings {
    /// Record a freshly created owner's backings. A query failure records
    /// nothing; the caller destroys the owner and returns the error.
    ///
    /// # Safety
    /// `owner` is a live owner created by `library`'s `<prefix>_create`.
    pub(crate) unsafe fn record(library: &Library, prefix: &str, owner: *mut c_void, device: i32, tags: MediaTags)
        -> Result<Self, i32> {
        let symbol = format!("{prefix}_get_backings");
        // SAFETY: versioned native ABI; the out struct has its exact C layout.
        let get = unsafe { library.get::<GetBackings>(symbol.as_bytes()) }.map_err(|_| -1)?;
        let mut raw = RawBackings::default();
        // SAFETY: passive query of a live owner into a local output.
        let status = unsafe { get(owner, &mut raw) };
        if status != 0 { return Err(status); }
        let weights = raw.weights as usize;
        let workspace = raw.workspace as usize;
        memory_ledger::record_tagged(Space::Device, device, weights, raw.weight_bytes as usize, tags.weights);
        memory_ledger::record_tagged(Space::Device, device, workspace, raw.workspace_bytes as usize, tags.workspace);
        Ok(Self { weights, workspace })
    }

    /// Destroy through `<prefix>_destroy_backings` and retire each backing
    /// whose free native reports as successful; a failed free stays live.
    ///
    /// # Safety
    /// `owner` is live, exclusively held, and never used again.
    pub(crate) unsafe fn destroy(&mut self, library: &Library, prefix: &str, owner: *mut c_void) -> i32 {
        let symbol = format!("{prefix}_destroy_backings");
        // SAFETY: versioned native ABI.
        let Ok(destroy) = (unsafe { library.get::<DestroyBackings>(symbol.as_bytes()) }) else { return -1 };
        let mut released = 0u32;
        // SAFETY: forwarded from this function's contract.
        let status = unsafe { destroy(owner, &mut released) };
        self.retire(released);
        status
    }

    fn retire(&mut self, released: u32) {
        if released & RELEASED_WEIGHTS != 0 && self.weights != 0 {
            memory_ledger::record_released(std::mem::take(&mut self.weights));
        }
        if released & RELEASED_WORKSPACE != 0 && self.workspace != 0 {
            memory_ledger::record_released(std::mem::take(&mut self.workspace));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_ledger::{snapshot, test_guard, MemoryKind};

    #[test]
    fn media_backings_record_their_tags_and_retire_only_released_frees() {
        let _ledger = test_guard();
        let tags = MediaTags::new("test/tower");
        let (weights, workspace) = (0x7a00_0000_0000usize, 0x7a00_0001_0000usize);
        memory_ledger::record_tagged(Space::Device, 51, weights, 4096, tags.weights);
        memory_ledger::record_tagged(Space::Device, 51, workspace, 1024, tags.workspace);
        let mut backings = MediaBackings { weights, workspace };
        let (kinds, untagged) = snapshot().by_kind(Space::Device, 51);
        assert_eq!((kinds[&MemoryKind::Weights], kinds[&MemoryKind::Workspace], untagged), (4096, 1024, 0));
        // A failed scratch free keeps it live; only the weights retire.
        backings.retire(RELEASED_WEIGHTS);
        let (kinds, _) = snapshot().by_kind(Space::Device, 51);
        assert_eq!((kinds.get(&MemoryKind::Weights), kinds[&MemoryKind::Workspace]), (None, 1024));
        // A repeated report never retires twice.
        backings.retire(RELEASED_WEIGHTS | RELEASED_WORKSPACE);
        backings.retire(RELEASED_WEIGHTS | RELEASED_WORKSPACE);
        assert_eq!(snapshot().total(Space::Device, 51), 0);
    }
}
