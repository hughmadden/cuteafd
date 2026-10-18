//! Coordinator H128 transforms: F16 egress and one stable-order F32 reduction.
use super::Exlr8Weights;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary, programs::{Program, Programs, Scalar}};
use std::ffi::c_void;

pub(crate) struct Exlr8Transforms<'a> {
    rotate: Program<'a>,
    reduce: Program<'a>,
    hidden: usize,
    rows: usize,
    ranks: usize,
    ctas: [i32; 2],
    device: i32,
}

fn geometry(program: &Program<'_>, key: &str) -> Result<usize> {
    usize::try_from(program.spec().geometry[key].as_u64().context("EXLR8 transform geometry")?)
        .context("EXLR8 transform geometry exceeds address space")
}

fn require(buffer: CuteafdDeviceBuffer, bytes: usize, device: i32) -> Result<()> {
    ensure!(!buffer.ptr.is_null() && buffer.bytes >= bytes && buffer.device_id == device,
        "EXLR8 transform view is null, too small or on a different device");
    Ok(())
}

impl<'a> Exlr8Transforms<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, programs: &'a Programs<'a>) -> Result<Self> {
        let rotate = programs.program("exlr8_rotate", &["x", "signs", "output"])?;
        let reduce = programs.program("exlr8_reduce", &["planes", "shared", "signs", "residual", "output"])?;
        let hidden = geometry(&rotate, "hidden")?;
        ensure!(hidden == geometry(&reduce, "hidden")?, "EXLR8 transform hidden widths differ");
        let rows = geometry(&rotate, "max_rows")?.min(geometry(&reduce, "max_rows")?);
        let ranks = geometry(&reduce, "max_ranks")?;
        let sms = usize::try_from(library.sm_count()?)?;
        let cta = |program: &Program<'_>| -> Result<i32> {
            let limit = sms.checked_mul(geometry(program, "ctas_per_sm")?).context("EXLR8 transform CTA overflow")?;
            ensure!(limit > 0, "EXLR8 transform needs positive runtime SM count");
            Ok(i32::try_from(limit)?)
        };
        let ctas = [cta(&rotate)?, cta(&reduce)?];
        Ok(Self { rotate, reduce, hidden, rows, ranks, ctas, device: library.cuda_get_device()? })
    }
    fn rows(&self, weights: &Exlr8Weights<'_>, rows: usize) -> Result<usize> {
        ensure!(rows > 0 && rows <= self.rows && weights.layout.hidden == self.hidden,
            "EXLR8 transform rows or weight geometry differ");
        rows.checked_mul(self.hidden).context("EXLR8 transform extent overflow")
    }
    /// # Safety
    /// Views and weights remain live on this device until `stream` drains. The
    /// caller orders preceding producers and retains the F16 wire for its send.
    pub(crate) unsafe fn rotate(&self, weights: &Exlr8Weights<'_>, rows: usize,
        input: CuteafdDeviceBuffer, output: CuteafdDeviceBuffer, stream: *mut c_void) -> Result<()> {
        let bytes = self.rows(weights, rows)?.checked_mul(2).context("EXLR8 rotation overflow")?;
        require(input, bytes, self.device)?;
        require(output, bytes, self.device)?;
        let signs = weights.signs(false)?;
        require(signs, self.hidden, self.device)?;
        // SAFETY: caller retains all validated views through stream completion.
        unsafe { self.rotate.launch(&[input.ptr, signs.ptr, output.ptr],
            &[Scalar::I32(rows as i32), Scalar::I32(self.ctas[0])], stream) }
    }
    /// # Safety
    /// `planes` is a device I64 table of `ranks` live F32 planes in rank order,
    /// each [rows, hidden]. It and every input/output owner outlive stream drain;
    /// transport completion precedes reads. Shared output is rotated F32 too.
    pub(crate) unsafe fn reduce(&self, weights: &Exlr8Weights<'_>, rows: usize, ranks: usize,
        planes: CuteafdDeviceBuffer, shared: CuteafdDeviceBuffer, residual: CuteafdDeviceBuffer,
        output: CuteafdDeviceBuffer, stream: *mut c_void) -> Result<()> {
        let elements = self.rows(weights, rows)?;
        ensure!(ranks <= self.ranks, "EXLR8 rank count exceeds transform capacity");
        require(planes, ranks.max(1) * 8, self.device)?;
        require(shared, elements.checked_mul(4).context("EXLR8 shared extent overflow")?, self.device)?;
        let bytes = elements.checked_mul(2).context("EXLR8 residual extent overflow")?;
        require(residual, bytes, self.device)?;
        require(output, bytes, self.device)?;
        let signs = weights.signs(true)?;
        require(signs, self.hidden, self.device)?;
        // SAFETY: caller guarantees pointee ownership and stable rank ordering.
        unsafe { self.reduce.launch(&[planes.ptr, shared.ptr, signs.ptr, residual.ptr, output.ptr],
            &[Scalar::I32(rows as i32), Scalar::I32(ranks as i32), Scalar::I32(self.ctas[1])], stream) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exlr8_transform_views_refuse_short_null_or_wrong_device() {
        let buffer = CuteafdDeviceBuffer { ptr: 256usize as *mut c_void, bytes: 512, device_id: 1, flags: 0 };
        require(buffer, 512, 1).unwrap();
        assert!(require(buffer, 513, 1).is_err());
        assert!(require(buffer, 512, 0).is_err());
        assert!(require(CuteafdDeviceBuffer { ptr: std::ptr::null_mut(), ..buffer }, 512, 1).is_err());
    }
}
