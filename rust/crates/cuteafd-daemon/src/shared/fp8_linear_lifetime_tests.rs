//! Directed CPU fixture checks of the real common pack/quarantine entry points.
//! The caller source retention below models the loader contract; it does not
//! exercise the family loaders or prove their complete error unwind behavior.
use super::*;
use cuteafd_ffi::native_library_lifetime_fixture::Fixture;

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn failed_pack_and_failed_drain_retain_module_and_destinations() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    fixture.stage(&library)?;
    fixture.configure_pack(&library, 1, 1)?;
    let source = DeviceAllocation::new(&library, 16 * 128 * 2)?;
    let result = Fp8Weight::pack(
        &library,
        source.buffer.ptr,
        16,
        128,
        Fp8Scales::Amax,
        std::ptr::null_mut(),
    );
    assert!(result.is_err());
    drop(result);
    // Required caller contract after the unprovable source drain.
    std::mem::forget(source);
    drop(library);
    assert_eq!(fixture.events()?, "ADDDPS");
    assert!(fixture.resident()?, "pack must retain the actual module");
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn successful_pack_then_quarantine_retains_module_and_destinations() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    fixture.stage(&library)?;
    fixture.configure_pack(&library, 0, 1)?;
    let source = DeviceAllocation::new(&library, 16 * 128 * 2)?;
    let packed = Fp8Weight::pack(
        &library,
        source.buffer.ptr,
        16,
        128,
        Fp8Scales::Amax,
        std::ptr::null_mut(),
    )?;
    assert!(fixture.drain(&library).is_err());
    std::mem::forget(source);
    packed.quarantine();
    drop(library);
    assert_eq!(fixture.events()?, "ADDDPS");
    assert!(
        fixture.resident()?,
        "weight quarantine must retain its module"
    );
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn successful_drain_preserves_cleanup_on_pack_success_and_error() -> Result<()> {
    for pack_status in [0, 1] {
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.stage(&library)?;
        fixture.configure_pack(&library, pack_status, 0)?;
        let source = DeviceAllocation::new(&library, 16 * 128 * 2)?;
        let result = Fp8Weight::pack(
            &library,
            source.buffer.ptr,
            16,
            128,
            Fp8Scales::Amax,
            std::ptr::null_mut(),
        );
        assert_eq!(result.is_err(), pack_status != 0);
        // On launch error pack itself drained; on success the caller drains.
        if pack_status == 0 {
            fixture.drain(&library)?;
        }
        drop(result);
        drop(source);
        drop(library);
        assert_eq!(fixture.events()?, "ADDDPSdddFU");
        assert!(
            !fixture.resident()?,
            "successful drain must preserve ordinary unload"
        );
    }
    Ok(())
}
