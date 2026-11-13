use crate::{CuteafdDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;
type Mixes = unsafe extern "C" fn(
    *const u16,
    *const f32,
    *const f32,
    *const f32,
    *mut f32,
    *mut f32,
    *mut f32,
    i32,
    *mut c_void,
) -> i32;
type MixesWorkspace = unsafe extern "C" fn(
    *const u16,
    *const f32,
    *const f32,
    *const f32,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut c_void,
    u64,
    i32,
    *mut c_void,
) -> i32;
type Begin = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    u64,
    i32,
    *mut c_void,
) -> i32;
type Pre = unsafe extern "C" fn(*const u16, *const f32, *mut u16, i32, *mut c_void) -> i32;
type PostFinish = unsafe extern "C" fn(
    *const u16,
    *const u16,
    *const c_void,
    u64,
    *const f32,
    *const f32,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut u16,
    i32,
    *mut c_void,
) -> i32;
type Post = unsafe extern "C" fn(
    *const u16,
    *const u16,
    *const f32,
    *const f32,
    *mut u16,
    i32,
    *mut c_void,
) -> i32;
/// How a V4.1 mHC boundary runs on this device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HcBoundary {
    /// Mixes, pre and RMSNorm at begin, post at finish.
    Chain,
    /// SparkInfer's lagged begin (SM120 builds that export it), then post.
    Lagged,
    /// SM89: one begin launch, its finish folded into post (v41_hc_sm89.cu),
    /// in the chain's decode arithmetic.
    Sm89,
}

/// The A/B switch: `chain` keeps today's mixes, pre and RMSNorm, then post on
/// any GPU; `fused` (the SM89 default) asks for the SM89 boundary. Unset takes
/// the architecture's default.
pub const HC_BOUNDARY_SWITCH: &str = "CUTEAFD_V41_HC_BOUNDARY";

impl HcBoundary {
    /// Rows at or below which this boundary's begin entry point runs; wider
    /// units (prefill) keep the chain.
    pub const fn max_rows(self) -> usize {
        match self {
            HcBoundary::Chain => 0,
            HcBoundary::Lagged => 80,
            HcBoundary::Sm89 => 128,
        }
    }
    /// Scratch bytes per row the begin entry point needs.
    const fn scratch_row_bytes(self) -> usize {
        match self {
            HcBoundary::Sm89 => 1536,
            HcBoundary::Chain | HcBoundary::Lagged => 8000,
        }
    }
    /// The device architecture decides; the switch can only select the chain,
    /// or confirm the SM89 boundary on an SM89 device.
    fn select(arch: cuteafd_core::GpuArch, requested: Option<&str>, lagged: bool) -> Result<Self> {
        match (requested, arch) {
            (Some("chain"), _) => Ok(HcBoundary::Chain),
            (None | Some("fused"), cuteafd_core::GpuArch::Sm89) => Ok(HcBoundary::Sm89),
            (None, _) => Ok(if lagged { HcBoundary::Lagged } else { HcBoundary::Chain }),
            (Some("fused"), arch) => anyhow::bail!("{HC_BOUNDARY_SWITCH}=fused is the SM89 boundary; this device is {arch}"),
            (Some(other), _) => anyhow::bail!("{HC_BOUNDARY_SWITCH} must be chain or fused, got {other:?}"),
        }
    }
}

pub struct V41Hc<'a> {
    _library: &'a NativeLibrary,
    pre: Pre,
    post: Post,
    mixes: Mixes,
    mixes_workspace: MixesWorkspace,
    begin: Option<Begin>,
    post_finish: Option<PostFinish>,
    boundary: HcBoundary,
}
impl NativeLibrary {
    pub fn v41_hc(&self) -> Result<V41Hc<'_>> {
        let initialize = unsafe {
            self.lib
                .get::<unsafe extern "C" fn() -> i32>(b"cuteafd_v41_hc_project_initialize")?
        };
        let status = unsafe { initialize() };
        self.status_to_result("mHC AOT initialization CUDA status", status)?;
        let begin = unsafe {
            self.lib
                .get::<Begin>(b"cuteafd_v41_hc_begin")
                .ok()
                .map(|symbol| *symbol)
        };
        let info = self.cuda_device_info(self.cuda_get_device()?)?;
        let arch = cuteafd_core::GpuArch::from_compute_capability(
            info.compute_capability_major,
            info.compute_capability_minor,
        )?;
        let requested = std::env::var(HC_BOUNDARY_SWITCH).ok().filter(|value| !value.is_empty());
        let boundary = HcBoundary::select(arch, requested.as_deref(), begin.is_some())?;
        let mut post_finish = None;
        if boundary == HcBoundary::Sm89 {
            // SM89 libraries always carry the boundary; anything else is the
            // wrong build for this device, not a reason to fall back.
            ensure!(begin.is_some(), "SM89 mHC boundary: the library has no cuteafd_v41_hc_begin");
            let sm89_initialize = unsafe {
                self.lib
                    .get::<unsafe extern "C" fn() -> i32>(b"cuteafd_v41_hc_sm89_initialize")
                    .context("SM89 mHC boundary: the library is not an sm_89 build")?
            };
            let status = unsafe { sm89_initialize() };
            self.status_to_result("SM89 mHC initialization CUDA status", status)?;
            post_finish = Some(unsafe { *self.lib.get::<PostFinish>(b"cuteafd_v41_hc_post_finish")? });
        }
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| tracing::info!(?boundary, %arch, switch = requested.as_deref().unwrap_or("default"),
            "V4.1 mHC boundary"));
        Ok(V41Hc {
            _library: self,
            begin: if boundary == HcBoundary::Chain { None } else { begin },
            post_finish,
            boundary,
            mixes_workspace: unsafe {
                *self
                    .lib
                    .get::<MixesWorkspace>(b"cuteafd_v41_hc_mixes_workspace")?
            },
            mixes: unsafe { *self.lib.get::<Mixes>(b"cuteafd_v41_hc_mixes")? },
            pre: unsafe { *self.lib.get::<Pre>(b"cuteafd_v41_hc_pre")? },
            post: unsafe { *self.lib.get::<Post>(b"cuteafd_v41_hc_post")? },
        })
    }
}
fn buffers(rows: usize, views: &[(CuteafdDeviceBuffer, usize)]) -> Result<()> {
    ensure!((1..=4096).contains(&rows), "invalid mHC rows");
    for (view, stride) in views {
        ensure!(
            !view.ptr.is_null() && view.bytes >= rows * stride,
            "invalid mHC buffer"
        );
    }
    Ok(())
}
impl V41Hc<'_> {
    pub fn boundary(&self) -> HcBoundary {
        self.boundary
    }
    /// Whether a begin of `rows` leaves next pre, post and comb to
    /// [`Self::post_finish`] (the SM89 boundary) instead of producing them.
    pub fn defers_finish(&self, rows: usize) -> bool {
        self.post_finish.is_some() && rows <= self.boundary.max_rows()
    }
    /// # Safety
    /// Inputs are initialized on this device. Outputs and scratch are disjoint
    /// from every input and each other, and remain live through stream completion;
    /// on the SM89 boundary the scratch holds the projection partials until the
    /// boundary's [`Self::post_finish`] completes.
    /// Returns false only when this boundary or row count uses the chain.
    pub unsafe fn try_begin(
        &self,
        residual: CuteafdDeviceBuffer,
        projection: CuteafdDeviceBuffer,
        scale: CuteafdDeviceBuffer,
        base: CuteafdDeviceBuffer,
        incoming: CuteafdDeviceBuffer,
        norm: CuteafdDeviceBuffer,
        predicted: CuteafdDeviceBuffer,
        post: CuteafdDeviceBuffer,
        comb: CuteafdDeviceBuffer,
        normalized: CuteafdDeviceBuffer,
        scratch: CuteafdDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<bool> {
        let Some(begin) = self.begin else {
            return Ok(false);
        };
        if rows > self.boundary.max_rows() {
            return Ok(false);
        }
        buffers(
            rows,
            &[
                (residual, 40960),
                (incoming, 16),
                (predicted, 16),
                (post, 16),
                (comb, 64),
                (normalized, 10240),
                (scratch, self.boundary.scratch_row_bytes()),
            ],
        )?;
        buffers(
            1,
            &[
                (projection, 24 * 20480 * 4),
                (scale, 12),
                (base, 96),
                (norm, 10240),
            ],
        )?;
        let status = unsafe {
            begin(
                residual.ptr,
                projection.ptr,
                scale.ptr,
                base.ptr,
                incoming.ptr,
                norm.ptr,
                predicted.ptr,
                post.ptr,
                comb.ptr,
                normalized.ptr,
                scratch.ptr,
                scratch.bytes as u64,
                rows as i32,
                stream,
            )
        };
        self._library.status_to_result("mHC begin CUDA status", status)?;
        Ok(true)
    }

    /// # Safety
    /// Inputs must be initialized on the stream device and all buffers remain
    /// live through completion; outputs must be mutually disjoint and disjoint
    /// from inputs. Coefficients produced here belong to the following sublayer.
    pub unsafe fn mixes(
        &self,
        residual: CuteafdDeviceBuffer,
        projection: CuteafdDeviceBuffer,
        scale: CuteafdDeviceBuffer,
        base: CuteafdDeviceBuffer,
        pre: CuteafdDeviceBuffer,
        post: CuteafdDeviceBuffer,
        comb: CuteafdDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        buffers(
            rows,
            &[(residual, 40960), (pre, 16), (post, 16), (comb, 64)],
        )?;
        buffers(1, &[(projection, 24 * 20480 * 4), (scale, 12), (base, 96)])?;
        let status = unsafe {
            (self.mixes)(
                residual.ptr.cast(),
                projection.ptr.cast(),
                scale.ptr.cast(),
                base.ptr.cast(),
                pre.ptr.cast(),
                post.ptr.cast(),
                comb.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        self._library.status_to_result("mHC mixes CUDA status", status)?;
        Ok(())
    }

    /// # Safety
    /// Inputs must be initialized on the stream device and all buffers remain
    /// live through completion; outputs must be mutually disjoint and disjoint
    /// from inputs. Scratch must also be disjoint from all inputs and outputs,
    /// and remain live through completion. Coefficients produced here belong to the following sublayer.
    pub unsafe fn mixes_workspace(
        &self,
        residual: CuteafdDeviceBuffer,
        projection: CuteafdDeviceBuffer,
        scale: CuteafdDeviceBuffer,
        base: CuteafdDeviceBuffer,
        pre: CuteafdDeviceBuffer,
        post: CuteafdDeviceBuffer,
        comb: CuteafdDeviceBuffer,
        scratch: CuteafdDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        buffers(
            rows,
            &[(residual, 40960), (pre, 16), (post, 16), (comb, 64)],
        )?;
        buffers(1, &[(projection, 24 * 20480 * 4), (scale, 12), (base, 96)])?;
        if rows <= 16 {
            buffers(rows, &[(scratch, 1536)])?;
        }
        let status = unsafe {
            (self.mixes_workspace)(
                residual.ptr.cast(),
                projection.ptr.cast(),
                scale.ptr.cast(),
                base.ptr.cast(),
                pre.ptr.cast(),
                post.ptr.cast(),
                comb.ptr.cast(),
                scratch.ptr,
                scratch.bytes as u64,
                rows as i32,
                stream,
            )
        };
        self._library.status_to_result("mHC mixes CUDA status", status)?;
        Ok(())
    }

    /// # Safety
    /// Inputs must be initialized on the stream's device and all buffers remain
    /// live through completion; output must be disjoint from both inputs.
    pub unsafe fn pre(
        &self,
        residual: CuteafdDeviceBuffer,
        pre: CuteafdDeviceBuffer,
        output: CuteafdDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        buffers(rows, &[(residual, 40960), (pre, 16), (output, 10240)])?;
        let status = unsafe {
            (self.pre)(
                residual.ptr.cast(),
                pre.ptr.cast(),
                output.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        self._library.status_to_result("mHC pre CUDA status", status)?;
        Ok(())
    }
    /// # Safety
    /// Same device/lifetime/output-disjointness contract as pre. Comb is FP32
    /// [rows,source,destination], not destination-major.
    pub unsafe fn post(
        &self,
        sublayer: CuteafdDeviceBuffer,
        residual: CuteafdDeviceBuffer,
        post: CuteafdDeviceBuffer,
        comb: CuteafdDeviceBuffer,
        output: CuteafdDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        buffers(
            rows,
            &[
                (sublayer, 10240),
                (residual, 40960),
                (post, 16),
                (comb, 64),
                (output, 40960),
            ],
        )?;
        let status = unsafe {
            (self.post)(
                sublayer.ptr.cast(),
                residual.ptr.cast(),
                post.ptr.cast(),
                comb.ptr.cast(),
                output.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        self._library.status_to_result("mHC post CUDA status", status)?;
        Ok(())
    }
    /// The SM89 boundary's finish and post: next pre, post and comb from the
    /// projection partials the begin left in `scratch`, then the post output.
    ///
    /// # Safety
    /// The begin of these rows and the sublayer result have completed on (or
    /// are ordered before) this stream, and scratch is unchanged since that
    /// begin. Outputs are disjoint from every input and each other; sublayer,
    /// residual, scratch and output are 16-byte aligned; all stay live through
    /// completion.
    pub unsafe fn post_finish(
        &self,
        sublayer: CuteafdDeviceBuffer,
        residual: CuteafdDeviceBuffer,
        scratch: CuteafdDeviceBuffer,
        scale: CuteafdDeviceBuffer,
        base: CuteafdDeviceBuffer,
        predicted: CuteafdDeviceBuffer,
        post: CuteafdDeviceBuffer,
        comb: CuteafdDeviceBuffer,
        output: CuteafdDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        let post_finish = self.post_finish.context("this mHC boundary has no post finish")?;
        ensure!(rows <= self.boundary.max_rows(), "mHC post finish rows {rows} exceed the SM89 boundary");
        buffers(
            rows,
            &[
                (sublayer, 10240),
                (residual, 40960),
                (scratch, 1536),
                (predicted, 16),
                (post, 16),
                (comb, 64),
                (output, 40960),
            ],
        )?;
        buffers(1, &[(scale, 12), (base, 96)])?;
        let status = unsafe {
            post_finish(
                sublayer.ptr.cast(),
                residual.ptr.cast(),
                scratch.ptr,
                scratch.bytes as u64,
                scale.ptr.cast(),
                base.ptr.cast(),
                predicted.ptr.cast(),
                post.ptr.cast(),
                comb.ptr.cast(),
                output.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        self._library.status_to_result("mHC post finish CUDA status", status)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::GpuArch;

    #[test]
    fn hc_boundary_follows_the_device_and_the_switch_only_selects_the_chain() {
        for lagged in [false, true] {
            assert_eq!(HcBoundary::select(GpuArch::Sm89, None, lagged).unwrap(), HcBoundary::Sm89);
            assert_eq!(HcBoundary::select(GpuArch::Sm89, Some("fused"), lagged).unwrap(), HcBoundary::Sm89);
            assert_eq!(HcBoundary::select(GpuArch::Sm89, Some("chain"), lagged).unwrap(), HcBoundary::Chain);
            assert_eq!(HcBoundary::select(GpuArch::Sm120, Some("chain"), lagged).unwrap(), HcBoundary::Chain);
            assert!(HcBoundary::select(GpuArch::Sm120, Some("fused"), lagged).is_err());
            assert!(HcBoundary::select(GpuArch::Sm89, Some("lagged"), lagged).is_err());
        }
        // SM120 keeps today's choice: the lagged begin when its build exports it.
        assert_eq!(HcBoundary::select(GpuArch::Sm120, None, true).unwrap(), HcBoundary::Lagged);
        assert_eq!(HcBoundary::select(GpuArch::Sm120, None, false).unwrap(), HcBoundary::Chain);
        assert_eq!((HcBoundary::Chain.max_rows(), HcBoundary::Lagged.max_rows(), HcBoundary::Sm89.max_rows()),
                   (0, 80, 128));
    }
}
