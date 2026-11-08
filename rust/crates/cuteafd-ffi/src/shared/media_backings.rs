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

/// `cuteafd_media_tags`: the kinds and NUL-terminated owners create allocates under.
#[repr(C)]
pub(crate) struct RawTags {
    pub weight_kind: u32,
    pub weight_owner: *const std::ffi::c_char,
    pub workspace_kind: u32,
    pub workspace_owner: *const std::ffi::c_char,
}

/// The C strings behind one `RawTags`, kept alive through create.
pub(crate) struct NativeTags {
    weight_owner: std::ffi::CString,
    workspace_owner: std::ffi::CString,
    tags: MediaTags,
}
impl NativeTags {
    pub(crate) fn new(tags: MediaTags) -> Result<Self, i32> {
        let owner = |tag: crate::memory_ledger::Tag| std::ffi::CString::new(tag.owner).map_err(|_| -1);
        Ok(Self { weight_owner: owner(tags.weights)?, workspace_owner: owner(tags.workspace)?, tags })
    }
    pub(crate) fn raw(&self) -> RawTags {
        RawTags { weight_kind: self.tags.weights.kind.as_raw(), weight_owner: self.weight_owner.as_ptr(),
            workspace_kind: self.tags.workspace.kind.as_raw(), workspace_owner: self.workspace_owner.as_ptr() }
    }
}

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
        // A null pointer is a backing already freed (or never allocated): nothing to record.
        memory_ledger::record_tagged(Space::Device, device, weights, raw.weight_bytes as usize, tags.weights);
        memory_ledger::record_tagged(Space::Device, device, workspace, raw.workspace_bytes as usize, tags.workspace);
        Ok(Self { weights, workspace })
    }

    /// The ledger side of a create. `status` and `owner` are what create
    /// returned: on success both backings are recorded; on failure a non-null
    /// owner holds exactly the backings native could not free, which are
    /// recorded too (the caller keeps the owner and destroys it later); a null
    /// owner means nothing is live.
    ///
    /// # Safety
    /// `owner` is null or a live owner created by `library`'s `<prefix>_create`.
    pub(crate) unsafe fn created(library: &Library, prefix: &str, status: i32, owner: *mut c_void,
        device: i32, tags: MediaTags) -> Result<Self, (i32, Self)> {
        if owner.is_null() { return Err((if status == 0 { -1 } else { status }, Self::default())); }
        // SAFETY: forwarded from this function's contract.
        let recorded = match unsafe { Self::record(library, prefix, owner, device, tags) } {
            Ok(recorded) => recorded,
            Err(query) => return Err((if status == 0 { query } else { status }, Self::default())),
        };
        if status == 0 { Ok(recorded) } else { Err((status, recorded)) }
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

    type FixtureCreate = unsafe extern "C" fn(u64, u64, *const RawTags, *mut *mut c_void) -> i32;

    /// The create contract end to end against the CPU fixture's tower: a
    /// failed create whose cleanup succeeds records nothing; one whose cleanup
    /// fails leaves exactly the live backing recorded and its owner held;
    /// and the failure reports the tag it was attempting.
    #[test]
    fn failed_create_records_only_backings_native_could_not_free() -> anyhow::Result<()> {
        let _ledger = test_guard();
        let fixture = crate::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        let device = 52;
        let device_bytes = || snapshot().total(Space::Device, device);
        let create = |tags: MediaTags| -> (i32, *mut c_void) {
            let native = NativeTags::new(tags).unwrap();
            let mut owner = std::ptr::null_mut();
            // SAFETY: fixture export with the tower create ABI; tags outlive the call.
            let status = unsafe { library.lib.get::<FixtureCreate>(b"cuteafd_fixture_media_create").unwrap()(
                4096, 1024, &native.raw(), &mut owner) };
            (status, owner)
        };
        let tags = MediaTags::new("test/tower");
        // Success: both backings, under their tags.
        fixture.media_configure(&library, false, false)?;
        let (status, owner) = create(tags);
        // SAFETY: a live fixture owner.
        let mut backings = unsafe { MediaBackings::created(&library.lib, "cuteafd_fixture_media", status, owner, device, tags) }
            .map_err(|(status, _)| anyhow::anyhow!("create failed {status}"))?;
        let (kinds, untagged) = snapshot().by_kind(Space::Device, device);
        assert_eq!((kinds[&MemoryKind::Weights], kinds[&MemoryKind::Workspace], untagged), (4096, 1024, 0));
        // SAFETY: the owner is destroyed exactly once here.
        assert_eq!(unsafe { backings.destroy(&library.lib, "cuteafd_fixture_media", owner) }, 0);
        assert_eq!(device_bytes(), 0);
        // Scratch fails, cleanup frees the weights: nothing live, nothing recorded.
        fixture.media_configure(&library, false, true)?;
        let (status, owner) = create(tags);
        assert!(status != 0 && owner.is_null());
        // SAFETY: null owner.
        assert!(unsafe { MediaBackings::created(&library.lib, "cuteafd_fixture_media", status, owner, device, tags) }.is_err());
        assert_eq!(device_bytes(), 0);
        assert_eq!(crate::NativeLibrary::last_allocation_tag(&library)?, Some(tags.workspace));
        // Scratch fails and cleanup cannot free the weights: the weights stay recorded.
        fixture.media_configure(&library, true, true)?;
        let (status, owner) = create(tags);
        assert!(status != 0 && !owner.is_null());
        // SAFETY: a live partial owner holding only its weights.
        let Err((_, mut partial)) = (unsafe { MediaBackings::created(&library.lib, "cuteafd_fixture_media", status, owner, device, tags) })
            else { panic!("a failed create must report its error") };
        let (kinds, _) = snapshot().by_kind(Space::Device, device);
        assert_eq!((kinds.get(&MemoryKind::Weights), kinds.get(&MemoryKind::Workspace)), (Some(&4096), None));
        // A retried destroy that still cannot free keeps it live; once frees work it retires.
        // SAFETY: the owner stays live while native holds its backing.
        assert_ne!(unsafe { partial.destroy(&library.lib, "cuteafd_fixture_media", owner) }, 0);
        assert_eq!(device_bytes(), 4096);
        fixture.media_configure(&library, false, false)?;
        // SAFETY: still live; this destroy frees and deletes it.
        assert_eq!(unsafe { partial.destroy(&library.lib, "cuteafd_fixture_media", owner) }, 0);
        assert_eq!(device_bytes(), 0);
        Ok(())
    }

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
