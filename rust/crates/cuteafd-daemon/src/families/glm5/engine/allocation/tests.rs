use super::*;

#[test]
fn actual_owner_uses_launch_order_not_cuda_ordinal() -> Result<()> {
    for physical in [[0, 1], [1, 0]] {
        let devices = CoordinatorDevices::new(&physical)?;
        for rank in 0..2 {
            let mut census = Census { prefix: "actual".into(), receipts: Vec::new(),
                devices, rank: rank as u8 };
            let buffer = cuteafd_ffi::CuteafdDeviceBuffer {
                ptr: std::ptr::without_provenance_mut(0x1000 + rank * 0x1000),
                bytes: 768 + rank * 256, device_id: physical[rank], flags: 0,
            };
            census.record("owner", buffer, Lifetime::TARGET, Category::Workspace, "fixture")?;
            assert_eq!(census.receipts[0].allocation.location, devices.location(physical[rank])?);
            assert_eq!(census.receipts[0].allocation.bytes, buffer.bytes as u64);
            assert_eq!(census.receipts[0].address, buffer.ptr as u64);
            assert!(census.record("wrong-device", cuteafd_ffi::CuteafdDeviceBuffer {
                device_id: physical[1 - rank], ..buffer
            }, Lifetime::TARGET, Category::Workspace, "fixture").is_err());
            assert_eq!(census.receipts.len(), 1);
        }
    }
    Ok(())
}

#[test]
fn null_and_empty_owners_are_not_manufactured_receipts() -> Result<()> {
    let physical = [1];
    let mut census = Census { prefix: "actual".into(), receipts: Vec::new(),
        devices: CoordinatorDevices::new(&physical)?, rank: 0 };
    for (ptr, bytes) in [(std::ptr::null_mut(), 256),
        (std::ptr::without_provenance_mut(0x1000), 0)] {
        assert!(census.record("absent", cuteafd_ffi::CuteafdDeviceBuffer {
            ptr, bytes, device_id: 1, flags: 0,
        }, Lifetime::TARGET, Category::Workspace, "fixture").is_err());
    }
    assert!(census.receipts.is_empty());
    Ok(())
}
