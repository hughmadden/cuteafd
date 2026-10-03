// Included as a unit module so the fixture can exercise the real private
// staging owner without adding production test hooks. These directed tests
// build a tiny CPU-only DSO and require an explicitly allocated build slot:
// CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR=$TASK/fixtures cargo test -p cuteafd-ffi \
//   --lib native_library_lifetime -- --ignored --nocapture
use super::*;
use crate::native_library_lifetime_fixture::Fixture;

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn normal_drop_frees_pinned_staging_before_unloading_module() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    library
        .sync_h2d_staging
        .lock()
        .unwrap()
        .ensure(&library, 64)?;
    assert!(fixture.resident()?);
    assert_eq!(fixture.events()?, "A");
    drop(library);
    assert_eq!(fixture.events()?, "AFU");
    assert!(
        !fixture.resident()?,
        "ordinary Drop must unload the actual module"
    );
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn failed_drain_marker_retains_module_and_pinned_staging() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    library
        .sync_h2d_staging
        .lock()
        .unwrap()
        .ensure(&library, 64)?;
    library.quarantine_module_after_failed_drain();
    library.quarantine_module_after_failed_drain(); // Repeated caller drains are harmless.
    assert!(library.is_quarantined_after_failed_drain());
    drop(library);
    assert_eq!(
        fixture.events()?,
        "A",
        "quarantine must neither free pinned staging nor unload the module"
    );
    assert!(
        fixture.resident()?,
        "the actual DSO must remain mapped after owner Drop"
    );

    // The failure is local to one NativeLibrary, not a global cleanup switch.
    let healthy = Fixture::build()?;
    let library = healthy.load()?;
    assert!(!library.is_quarantined_after_failed_drain());
    library
        .sync_h2d_staging
        .lock()
        .unwrap()
        .ensure(&library, 64)?;
    drop(library);
    assert_eq!(healthy.events()?, "AFU");
    assert!(!healthy.resident()?);
    assert!(fixture.resident()?);
    Ok(())
}
