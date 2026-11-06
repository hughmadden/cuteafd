//! Run in the matching coordinator dev image through golden-run's native-test mode.
use anyhow::{ensure, Result};
use cuteafd_core::expert_usage::{Geometry, Reader};
use cuteafd_ffi::{
    expert_usage::{ExpertUsage, UsagePhase},
    CuteafdDeviceBuffer, NativeLibrary,
};
use std::{ffi::c_void, time::Duration};

struct Fixture<'a> {
    library: &'a NativeLibrary,
    stream: *mut c_void,
    graph: *mut c_void,
    buffers: Vec<CuteafdDeviceBuffer>,
}
impl Fixture<'_> {
    fn upload(&mut self, data: &[u32]) -> Result<CuteafdDeviceBuffer> {
        let bytes: Vec<u8> = data.iter().flat_map(|n| n.to_ne_bytes()).collect();
        let buffer = self.library.alloc_device_buffer(bytes.len())?;
        self.buffers.push(buffer);
        self.library.copy_h2d(buffer, &bytes)?;
        Ok(buffer)
    }
    fn drain(&self) -> Result<()> {
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
        reader.clone(),
    )?;
    let result = (|| -> Result<()> {
        let ids = fixture.upload(&[0x8000_0000, 1, 1, 2, 3, 3, 3, 3])?;
        let real_rows = fixture.upload(&[2])?;
        // SAFETY: live device operands have four rows; only two are real.
        unsafe {
            usage.routes(1, UsagePhase::Prefill)?.record(
                ids.ptr,
                4,
                Some(real_rows.ptr.cast()),
                0x7fff_ffff,
                true,
                fixture.stream,
            )?;
            // A replica has no operands on this device and must enqueue nothing.
            usage.routes(1, UsagePhase::Prefill)?.record(
                std::ptr::null(),
                4,
                None,
                u32::MAX,
                false,
                fixture.stream,
            )?;
            library.cuda_graph_begin_capture(fixture.stream)?;
            usage.routes(2, UsagePhase::Decode)?.record(
                ids.ptr,
                4,
                Some(real_rows.ptr.cast()),
                0x7fff_ffff,
                true,
                fixture.stream,
            )?;
            fixture.graph = library.cuda_graph_end_capture(fixture.stream)?;
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
