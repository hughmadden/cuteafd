// Included as a unit module so the fixture can exercise the real private
// staging owner without adding production test hooks. These directed tests
// build a tiny CPU-only DSO and require an explicitly allocated build slot:
// CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR=$TASK/fixtures cargo test -p cuteafd-ffi \
//   --lib native_library_lifetime -- --ignored --nocapture
use super::*;
use crate::native_library_lifetime_fixture::Fixture;

#[test]
#[ignore = "CPU-only budget fixture; requires an explicit NVMe fixture directory"]
fn coordinator_budget_refuses_before_allocating_and_checks_untracked_graphs() -> Result<()> {
    // The process-wide ceiling is immutable; isolate this test from the other
    // unit tests rather than changing production configuration back and forth.
    if std::env::var_os("CUTEAFD_BUDGET_FIXTURE_CHILD").is_none() {
        let status = std::process::Command::new(std::env::current_exe()?)
            .args(["--exact", "native_library_lifetime::coordinator_budget_refuses_before_allocating_and_checks_untracked_graphs",
                "--ignored", "--nocapture"])
            .env("CUTEAFD_BUDGET_FIXTURE_CHILD", "1").status()?;
        anyhow::ensure!(status.success(), "CPU budget fixture child failed");
        return Ok(());
    }
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    std::fs::write(fixture.directory().join("events"), "")?;
    // SAFETY: the fixture's control export accepts only synthetic byte counts;
    // no GPU or CUDA runtime is present in this DSO.
    unsafe {
        let configure: Symbol<unsafe extern "C" fn(i32, usize, usize, usize)> =
            library.lib.get(b"fixture_configure_memory")?;
        configure(0, 4096, 100, 400);
        configure(1, 512, 100, 400);
    }
    assert_eq!(library.cuda_memory_info()?, (3996, 4096), "no budget preserves the physical sample");
    set_coordinator_gpu_budget(1024)?;
    assert!(fixture.events()?.is_empty(), "installing a ceiling allocates no physical guard");
    assert_eq!(library.cuda_physical_memory_info()?, (3996, 4096));
    assert_eq!(library.cuda_memory_info()?, (924, 1024));
    let mut weights = library.alloc_device_buffer(300)?;
    let mut draft = library.alloc_managed_device_buffer(200)?;
    assert_eq!(library.cuda_physical_memory_info()?, (3696, 4096), "managed fixture pages stay on host");
    assert_eq!(library.cuda_memory_info()?, (424, 1024));
    let events = fixture.events()?;
    let error = library.alloc_device_buffer(425).unwrap_err().to_string();
    assert!(error.contains("shortfall 1 bytes"), "{error}");
    assert_eq!(fixture.events()?, events, "refused before the native allocator");
    // SAFETY: the CPU graph fixture accepts a null stream and returns an
    // unlaunched executable owned by this test.
    let graph = unsafe { library.cuda_graph_end_capture(std::ptr::null_mut())? };
    assert_eq!(library.cuda_memory_info()?, (24, 1024));
    // SAFETY: graph belongs to this fixture and has never launched.
    let error = unsafe { library.cuda_graph_end_capture(std::ptr::null_mut()) }.unwrap_err();
    assert!(format!("{error:#}").contains("shortfall 376 bytes"), "{error:#}");
    assert_eq!(library.cuda_memory_info()?, (24, 1024), "over-budget graph destroyed before publishing");
    // SAFETY: the retained graph has never launched.
    unsafe { library.cuda_graph_exec_destroy(graph)? };
    library.free_device_buffer(&mut draft)?;
    library.free_device_buffer(&mut weights)?;
    library.cuda_set_device(1)?;
    assert_eq!(library.cuda_memory_info()?, (412, 512), "budget never enlarges a physical device");
    let error = library.alloc_device_buffer(413).unwrap_err().to_string();
    assert!(error.contains("GPU 1") && error.contains("shortfall 1 bytes"), "{error}");
    Ok(())
}

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
