//! Run in the matching coordinator dev image through golden-run's native-test mode.
#![cfg(feature = "test-support")]
use anyhow::{ensure, Result};
use cuteafd_core::expert_usage::{Geometry, Reader};
use cuteafd_ffi::{
    expert_usage::{ExpertUsage, UsageDevice, UsagePhase},
    CuteafdDeviceBuffer, NativeLibrary,
};
use std::{ffi::c_void, time::Duration};

/// Scratch allocated by this integration test.
const TEST: cuteafd_ffi::memory_ledger::Tag = cuteafd_ffi::memory_ledger::Tag::workspace("test/scratch");

struct Fixture<'a> {
    library: &'a NativeLibrary,
    device: i32,
    stream: *mut c_void,
    graph: *mut c_void,
    buffers: Vec<CuteafdDeviceBuffer>,
}
impl Fixture<'_> {
    fn upload(&mut self, data: &[u32]) -> Result<CuteafdDeviceBuffer> {
        self.library.cuda_set_device(self.device)?;
        let bytes: Vec<u8> = data.iter().flat_map(|n| n.to_ne_bytes()).collect();
        let buffer = self.library.alloc_device_buffer(bytes.len(), TEST)?;
        self.buffers.push(buffer);
        self.library.copy_h2d(buffer, &bytes)?;
        Ok(buffer)
    }
    fn drain(&self) -> Result<()> {
        self.library.cuda_set_device(self.device)?;
        // SAFETY: fixture owns this stream and retains all graph operands.
        unsafe { self.library.cuda_stream_synchronize(self.stream) }
    }
}
impl Drop for Fixture<'_> {
    fn drop(&mut self) {
        if self.drain().is_err() {
            std::process::abort();
        }
        // SAFETY: queued work drained before destroying graphs, operands and stream.
        unsafe {
            if !self.graph.is_null() {
                let _ = self.library.cuda_graph_exec_destroy(self.graph);
            }
            for buffer in &mut self.buffers {
                let _ = self.library.free_device_buffer(buffer);
            }
            let _ = self.library.cuda_stream_destroy(self.stream);
        }
    }
}

#[test]
#[ignore = "requires CUDA and the matching CUTEAFD_NATIVE_LIB"]
fn real_rows_phases_canonical_rank_and_graph_replays() -> Result<()> {
    // SAFETY: fixture retains the matching native ABI library through CUDA teardown.
    let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    library.cuda_set_device(0)?;
    let mut fixture = Fixture {
        library: &library,
        device: 0,
        stream: library.cuda_stream_create()?,
        graph: std::ptr::null_mut(),
        buffers: Vec::new(),
    };
    let reader = Reader::default();
    let usage = ExpertUsage::new(
        &library,
        Geometry {
            layers: 3,
            experts: 4,
            topk: 2,
            routed_layers: vec![1, 2],
        },
        &[UsageDevice {
            logical: 0,
            cuda: 0,
        }],
        reader.clone(),
    )?;
    let table_bytes = 3 * 4 * 16;
    for name in ["counters", "snapshot", "pinned"] {
        let owner = format!("expert-usage.gpu0.{name}");
        let records: Vec<_> = cuteafd_ffi::memory_ledger::allocations()
            .into_iter()
            .filter(|record| record.key.owner == Some(owner.as_str()))
            .collect();
        ensure!(
            records.len() == 1 && records[0].bytes == table_bytes,
            "missing/mismatched owner {owner}"
        );
    }
    ensure!(
        usage.routes(1, 1, UsagePhase::Decode).is_err(),
        "unconfigured device must refuse"
    );
    let result = (|| -> Result<()> {
        let ids = fixture.upload(&[0x8000_0000, 1, 1, 2, 3, 3, 3, 3])?;
        let real_rows = fixture.upload(&[2])?;
        // SAFETY: live device operands have four rows; only two are real.
        unsafe {
            usage.routes(0, 1, UsagePhase::Prefill)?.record(
                ids.ptr,
                4,
                Some(real_rows.ptr.cast()),
                0x7fff_ffff,
                true,
                fixture.stream,
            )?;
            // A replica has no operands on this device and must enqueue nothing.
            usage.routes(0, 1, UsagePhase::Prefill)?.record(
                std::ptr::null(),
                4,
                None,
                u32::MAX,
                false,
                fixture.stream,
            )?;
            library.cuda_graph_begin_capture(fixture.stream)?;
            usage.routes(0, 2, UsagePhase::Decode)?.record(
                ids.ptr,
                4,
                Some(real_rows.ptr.cast()),
                0x7fff_ffff,
                true,
                fixture.stream,
            )?;
            fixture.graph = library.cuda_graph_end_capture_for_test(fixture.stream)?;
            for _ in 0..3 {
                library.cuda_graph_launch(fixture.graph, fixture.stream)?;
            }
        }
        fixture.drain()?;
        // One worker interval, outside the serving stream; no host-side route counting.
        std::thread::sleep(Duration::from_millis(1500));
        let value = reader.snapshot();
        ensure!(value["error"].is_null(), "snapshot failed: {value}");
        let total = &value["windows"]["total"];
        ensure!(
            total["prefill_routed_rows"] == serde_json::json!([0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0]),
            "prefill/padding/replica counts: {total}"
        );
        ensure!(
            total["decode_routed_rows"] == serde_json::json!([0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 3, 0]),
            "decode replay counts: {total}"
        );
        Ok(())
    })();
    if fixture.drain().is_err() {
        std::process::abort();
    }
    drop(usage);
    result
}

#[test]
#[ignore = "requires two CUDA devices and the matching CUTEAFD_NATIVE_LIB"]
fn two_devices_share_one_history_and_preserve_logical_owner_names() -> Result<()> {
    // SAFETY: matching native module remains loaded through fixture teardown.
    let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    let mut fixtures = Vec::new();
    for device in [0, 1] {
        library.cuda_set_device(device)?;
        fixtures.push(Fixture {
            library: &library,
            device,
            stream: library.cuda_stream_create()?,
            graph: std::ptr::null_mut(),
            buffers: Vec::new(),
        });
    }
    library.cuda_set_device(1)?;
    let reader = Reader::default();
    let usage = ExpertUsage::new(
        &library,
        Geometry {
            layers: 2,
            experts: 2,
            topk: 1,
            routed_layers: vec![0, 1],
        },
        &[
            UsageDevice {
                logical: 0,
                cuda: 1,
            },
            UsageDevice {
                logical: 1,
                cuda: 0,
            },
        ],
        reader.clone(),
    )?;
    ensure!(
        library.cuda_get_device()? == 1,
        "constructor changed caller device"
    );
    let result = (|| -> Result<()> {
        for (cuda, logical) in [(0, 1), (1, 0)] {
            let fixture = &mut fixtures[cuda];
            let ids = fixture.upload(&[cuda as u32])?;
            for name in ["counters", "snapshot", "pinned"] {
                let owner = format!("expert-usage.gpu{logical}.{name}");
                let records: Vec<_> = cuteafd_ffi::memory_ledger::allocations()
                    .into_iter()
                    .filter(|record| record.key.owner == Some(owner.as_str()))
                    .collect();
                ensure!(
                    records.len() == 1 && records[0].bytes == 64,
                    "owner mismatch {owner}"
                );
                if name != "pinned" {
                    ensure!(records[0].key.device == cuda as i32, "wrong CUDA owner");
                }
            }
            // SAFETY: canonical producer uses a descriptor/buffer/stream on the selected device.
            unsafe {
                for phase in [UsagePhase::Prefill, UsagePhase::Decode] {
                    usage.routes(cuda as i32, cuda, phase)?.record(
                        ids.ptr,
                        1,
                        None,
                        u32::MAX,
                        true,
                        fixture.stream,
                    )?;
                }
            }
            fixture.drain()?;
        }
        std::thread::sleep(Duration::from_millis(1500));
        let value = reader.snapshot();
        ensure!(value["error"].is_null(), "snapshot failed: {value}");
        for phase in ["prefill_routed_rows", "decode_routed_rows"] {
            ensure!(
                value["windows"]["total"][phase] == serde_json::json!([1, 0, 0, 1]),
                "merged {phase}: {value}"
            );
        }
        Ok(())
    })();
    for fixture in &fixtures {
        if fixture.drain().is_err() {
            std::process::abort();
        }
    }
    library.cuda_set_device(1)?;
    drop(usage);
    ensure!(
        library.cuda_get_device()? == 1,
        "teardown changed caller device"
    );
    result
}
