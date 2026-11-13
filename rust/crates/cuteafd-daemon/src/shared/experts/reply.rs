//! Combined-reply producers: reduction and conversion into registered wire
//! output. Transport retains the packed destination through its unit fence.
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{programs::{BoundProgram, Scalar}, CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_transport::{expert::BackboneRequest, ExpertV2Dtype};
use std::ffi::c_void;

/// The full-rotation EXL3 sum writes the compressed reply directly to the
/// registered slot, without materializing a BF16 token plane.
pub(crate) struct Exl3Reply<'a> {
    fp8: Option<BoundProgram<'a>>,
    nvfp4: Option<BoundProgram<'a>>,
}
impl<'a> Exl3Reply<'a> {
    pub(crate) fn load(library: &'a NativeLibrary) -> Result<Self> {
        let programs = library.programs()?;
        let geometry = cuteafd_core::expert_geometry();
        let program = |format| programs.reply_bound(
            &format!("expert_reply_exl3_h{}_k{}_{format}", geometry.hidden, geometry.topk),
            &["fc2_ptr", "output_ptr", "topk_weights_ptr", "route_expert_ids_ptr", "expert_map_ptr", "svh_ptr"], reply_format(format));
        Ok(Self { fp8: program(1)?, nvfp4: program(2)? })
    }

    /// # Safety
    /// The core's FC2 workspace, layer weights and staged routes remain live on
    /// the current device until the worker's unit fence; output is exclusive
    /// registered reply storage retained by the transport through that fence.
    pub(crate) unsafe fn launch(&self, request: &BackboneRequest<'_>,
        sources: super::exl3::execution::PartialSources, inputs: [CuteafdDeviceBuffer; 3],
        output: CuteafdDeviceBuffer, rows: usize, stream: *mut c_void) -> Result<()> {
        let program = match request.output_dtype() {
            ExpertV2Dtype::Fp8E4m3Ue8m0K32 => &self.fp8,
            ExpertV2Dtype::Nvfp4E2m1Fp8E4m3 => &self.nvfp4,
            _ => anyhow::bail!("fused EXL3 reply requires a compressed combined reply"),
        };
        let program = program.as_ref().context("compressed EXL3 reply producer was not admitted")?;
        let geometry = cuteafd_core::expert_geometry();
        let hidden = geometry.hidden as usize;
        let routes = rows.checked_mul(geometry.topk as usize).context("EXL3 reply routes overflow")?;
        ensure!(rows > 0 && rows <= request.rows() as usize && request.partial_plan()?.is_none(),
            "fused EXL3 reply requires whole combined rows");
        ensure!(sources.fc2.bytes >= routes.checked_mul(hidden).and_then(|n| n.checked_mul(2))
                .context("EXL3 reply FC2 extent overflow")?
            && inputs[1].bytes >= routes * 4 && inputs[2].bytes >= routes * 4
            && output.bytes >= rows.checked_mul(request.output_row_bytes()? as usize)
                .context("EXL3 reply output extent overflow")?, "fused EXL3 reply extent mismatch");
        let local_experts = sources.svh.bytes / (hidden * 2);
        ensure!(local_experts > 0 && sources.svh.bytes % (hidden * 2) == 0
            && sources.expert_map.bytes > 0 && sources.expert_map.bytes % 4 == 0,
            "fused EXL3 reply weight extent mismatch");
        for buffer in [sources.fc2, sources.svh, sources.expert_map, inputs[1], inputs[2], output] {
            ensure!(!buffer.ptr.is_null() && buffer.ptr as usize % 16 == 0
                && buffer.device_id == sources.fc2.device_id,
                "fused EXL3 reply buffer contract mismatch");
        }
        // SAFETY: caller retains every operand through the unit fence; runtime
        // counts bound the exact resident svh and expert-map allocations.
        unsafe { program.launch(&[sources.fc2.ptr, output.ptr, inputs[2].ptr,
                inputs[1].ptr, sources.expert_map.ptr, sources.svh.ptr],
            &[Scalar::I32(i32::try_from(local_experts)?),
                Scalar::I32(i32::try_from(sources.expert_map.bytes / 4)?),
                Scalar::I32(i32::try_from(rows)?)], stream) }
    }
}

/// Native source reduction and wire conversion share one launch. Programs are
/// bound before any request or graph capture and retained by the worker.
pub(crate) struct NativeReply<'a> {
    fp8: Option<BoundProgram<'a>>,
    nvfp4: Option<BoundProgram<'a>>,
}
impl<'a> NativeReply<'a> {
    pub(crate) fn load(library: &'a NativeLibrary) -> Result<Self> {
        let programs = library.programs()?;
        let hidden = cuteafd_core::expert_geometry().hidden;
        let program = |format| programs.reply_bound(
            &format!("expert_reply_native_h{hidden}_{format}"), &["source", "out"], reply_format(format));
        Ok(Self { fp8: program(1)?, nvfp4: program(2)? })
    }

    /// # Safety
    /// Source is the native kernel's token or token-major route plane; source,
    /// destination and this program remain live through the owner unit fence.
    pub(crate) unsafe fn launch(&self, request: &BackboneRequest<'_>,
        kind: cuteafd_ffi::V41ExpertOutputKind, source: CuteafdDeviceBuffer,
        output: CuteafdDeviceBuffer, stream: *mut c_void) -> Result<()> {
        let program = match request.output_dtype() {
            ExpertV2Dtype::Fp8E4m3Ue8m0K32 => &self.fp8,
            ExpertV2Dtype::Nvfp4E2m1Fp8E4m3 => &self.nvfp4,
            _ => anyhow::bail!("native fused reply requires a compressed combined reply"),
        };
        let program = program.as_ref().context("compressed native reply producer was not admitted")?;
        let geometry = cuteafd_core::expert_geometry();
        let rows = request.rows() as usize;
        let (source_kind, routes, element_bytes) = match kind {
            cuteafd_ffi::V41ExpertOutputKind::Fp32Tokens => (0, 1, 4),
            cuteafd_ffi::V41ExpertOutputKind::Fp32Routes => (1, geometry.topk as usize, 4),
            cuteafd_ffi::V41ExpertOutputKind::Bf16Routes => (2, geometry.topk as usize, 2),
        };
        ensure!(rows > 0 && request.partial_plan()?.is_none(),
            "native fused reply requires whole combined rows");
        let source_bytes = rows.checked_mul(geometry.hidden as usize)
            .and_then(|n| n.checked_mul(routes)).and_then(|n| n.checked_mul(element_bytes))
            .context("native reply source extent overflow")?;
        let output_bytes = rows.checked_mul(request.output_row_bytes()? as usize)
            .context("native reply output extent overflow")?;
        ensure!(source.bytes >= source_bytes && output.bytes >= output_bytes,
            "native fused reply extent mismatch");
        for buffer in [source, output] {
            ensure!(!buffer.ptr.is_null() && buffer.ptr as usize % 16 == 0
                && buffer.device_id == source.device_id,
                "native fused reply buffer contract mismatch");
        }
        // SAFETY: caller retains the native plane and exclusive reply slot until
        // completion; checked dimensions bound all reads and writes.
        unsafe { program.launch(&[source.ptr, output.ptr],
            &[Scalar::I32(i32::try_from(rows)?), Scalar::I32(i32::try_from(geometry.topk)?),
                Scalar::I32(source_kind)], stream) }
    }
}

/// Shared FP8/MXFP4/NVFP4 producer epilogue over the original BF16 route plane.
pub(crate) struct Fp8Reply<'a> { programs: [Option<BoundProgram<'a>>; 4] }
impl<'a> Fp8Reply<'a> {
    pub(crate) fn load(library: &'a NativeLibrary) -> Result<Self> {
        let programs = library.programs()?;
        let geometry = cuteafd_core::expert_geometry();
        let bind = |format, segmented| programs.reply_bound(&format!("expert_reply_fp8_h{}_k{}_{format}{}",
            geometry.hidden, geometry.topk, if segmented { "_segmented" } else { "" }),
            &["y", "pair_pos", "weights", "out"], reply_format(format));
        Ok(Self { programs: [bind(1, false)?, bind(2, false)?, bind(1, true)?, bind(2, true)?] })
    }

    /// # Safety
    /// The owner retains sources/routes and every addressed output row until its
    /// unit fences. Descriptor copies precede this launch on the same stream.
    pub(crate) unsafe fn launch(&self, dtype: ExpertV2Dtype, sources: (CuteafdDeviceBuffer, CuteafdDeviceBuffer),
        weights: CuteafdDeviceBuffer, output: CuteafdDeviceBuffer, rows: usize, segmented: bool,
        stream: *mut c_void) -> Result<()> {
        let (index, stride) = match dtype {
            ExpertV2Dtype::Fp8E4m3Ue8m0K32 => (0, cuteafd_core::expert_geometry().hidden as usize * 33 / 32),
            ExpertV2Dtype::Nvfp4E2m1Fp8E4m3 => (1, cuteafd_core::expert_geometry().hidden as usize * 9 / 16),
            _ => anyhow::bail!("FP8 fused reply requires compressed output"),
        };
        let routes = rows.checked_mul(cuteafd_core::expert_geometry().topk as usize)
            .and_then(|n| n.checked_mul(4)).context("FP8 reply routes overflow")?;
        ensure!(rows > 0 && sources.1.bytes >= routes && weights.bytes >= routes
            && output.bytes >= rows.checked_mul(if segmented { 8 } else { stride }).context("FP8 reply extent overflow")?,
            "FP8 fused reply extent mismatch");
        for buffer in [sources.0, sources.1, weights, output] {
            ensure!(!buffer.ptr.is_null() && buffer.ptr as usize % 4 == 0
                && buffer.device_id == sources.0.device_id, "FP8 fused reply buffer mismatch");
        }
        // SAFETY: checked extents plus caller's retained row addresses and route-plane contract.
        unsafe { self.programs[index + if segmented { 2 } else { 0 }].as_ref()
            .context("compressed FP8 reply producer was not admitted")?.launch(
            &[sources.0.ptr, sources.1.ptr, weights.ptr, output.ptr], &[Scalar::I32(rows.try_into()?)], stream) }
    }
}

fn reply_format(format: usize) -> cuteafd_core::SparkReplyFormat {
    if format == 1 { cuteafd_core::SparkReplyFormat::Fp8 } else { cuteafd_core::SparkReplyFormat::Nvfp4 }
}

pub(crate) fn validate_exports(programs: &cuteafd_ffi::programs::Programs<'_>, kind: &str) -> Result<()> {
    let geometry = cuteafd_core::expert_geometry();
    admit_producers(cuteafd_core::spark_reply::requested(), kind, geometry.hidden, geometry.topk,
        |name| programs.contains(name))
}

fn admit_producers(requested: cuteafd_core::spark_reply::SparkReplyFormats, kind: &str,
    hidden: u32, topk: u32, present: impl Fn(&str) -> bool) -> Result<()> {
    // Muse's historical per-expert regions are not combined replies. Their
    // BF16/FP8/NVFP4 epilogues remain required in development as well as release.
    if kind == "partial" {
        for name in ["expert_partial_bf16", "expert_partial_nvfp4", "expert_partial_fp8"] {
            ensure!(present(name), "missing partial Spark reply export {name}");
        }
        return Ok(());
    }
    for format in [1, 2] {
        let codec = reply_format(format);
        if requested.prefill != Some(codec) && requested.decode != Some(codec) { continue; }
        let names = match kind {
            "exlr8" | "exlr8_glmf" => vec![format!("{kind}_a16_serving_core"),
                format!("{kind}_reply_{format}"), format!("{kind}_a16_grouped_reply{format}"),
                format!("{kind}_a8_grouped_reply{format}"), format!("{kind}_a16_grouped64_reply{format}")],
            "fp8" => ["", "_segmented"].map(|suffix| format!("expert_reply_fp8_h{hidden}_k{topk}_{format}{suffix}")).to_vec(),
            "exl3" => vec![format!("expert_reply_exl3_h{hidden}_k{topk}_{format}")],
            _ => vec![format!("expert_reply_native_h{hidden}_{format}")],
        };
        for name in names { cuteafd_ffi::programs::require_reply_export(&name, present(&name))?; }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::{SparkReplyFormat as F, spark_reply::SparkReplyFormats};

    #[test]
    fn development_preserves_muse_partial_default_but_refuses_explicit_combined_compression() {
        use cuteafd_loader::families::muse_spark::replies::{Cutoff, PartialFormat};
        let development = ["expert_partial_bf16", "expert_partial_nvfp4", "expert_partial_fp8",
            "expert_reply_quantize", "expert_reply_dequantize", "expert_reply_stage", "expert_reply_reduce"];
        let present = |name: &str| development.contains(&name);
        let omitted = SparkReplyFormats::default();
        let cutoff = Cutoff::with_formats(64, 64, omitted).unwrap();
        assert_eq!(cutoff.phase_format(64, true), PartialFormat::Bf16);
        assert_eq!(cutoff.phase_format(65, true), PartialFormat::Nvfp4);
        assert!(admit_producers(omitted, "partial", 4096, 8, present).is_ok());
        assert!(admit_producers(omitted, "exl3", 4096, 8, present).is_ok());
        for codec in [F::Fp8, F::Nvfp4] {
            for formats in [SparkReplyFormats { prefill: Some(codec), decode: None },
                SparkReplyFormats { prefill: None, decode: Some(codec) }] {
                assert!(admit_producers(formats, "partial", 4096, 8, present).is_ok());
                let error = admit_producers(formats, "exl3", 4096, 8, present).unwrap_err();
                assert!(error.downcast_ref::<cuteafd_ffi::programs::MissingReplyExport>().is_some());
                assert!(error.to_string().contains("expert_reply_exl3_h4096_k8_"));
                assert!(error.to_string().contains("release builds include"));
            }
        }
        assert!(admit_producers(omitted, "partial", 4096, 8,
            |name| present(name) && name != "expert_partial_bf16").is_err());
    }
}
