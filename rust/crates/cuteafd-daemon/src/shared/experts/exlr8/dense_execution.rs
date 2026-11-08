//! Native dense launches over validated checkpoint geometry and borrowed lane storage.
use cuteafd_loader::plan::exlr8::dense::requests::NativeProjection;
use anyhow::{Context, Result, ensure};
use cuteafd_ffi::native_dense::{DenseLinear, Encoding};
use cuteafd_ffi::native_dense_glue::{Companion, Op};
use cuteafd_ffi::programs::Programs;
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_loader::plan::exlr8::dense::{DenseEncoding, ShardSide};
use std::{collections::BTreeSet, ffi::c_void};

pub(crate) const DECODE_ROWS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Input { F16, Bf16 }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Key { encoding: Encoding, k: usize, n: usize, side: usize, input: Input }

impl Key {
    fn validated(spec: &cuteafd_ffi::programs::ProgramSpec) -> Result<Self> {
        let g = cuteafd_ffi::native_dense::validate_spec(spec)?;
        let input = if matches!(g.encoding, Encoding::Trellis | Encoding::TrellisA8)
            && spec.geometry["input_dtype"].as_str().unwrap_or("f16") == "f16" { Input::F16 } else { Input::Bf16 };
        Ok(Self { encoding: g.encoding, k: g.input_features as usize, n: g.output_features as usize,
            side: g.side as usize, input })
    }

    fn projection(projection: &NativeProjection, rows: usize, a8: bool) -> Result<Self> {
        let (encoding, input) = match projection.encoding {
            DenseEncoding::Trellis { bits: 4 } => if rows > DECODE_ROWS && a8 {
                (Encoding::TrellisA8, Input::F16)
            } else { (Encoding::Trellis, if rows > DECODE_ROWS { Input::Bf16 } else { Input::F16 }) },
            DenseEncoding::Mxfp8 => (Encoding::Mxfp8, Input::Bf16),
            DenseEncoding::Int4G32 => (Encoding::Int4G32, Input::Bf16),
            DenseEncoding::RowFp8 => (Encoding::RowFp8, Input::Bf16),
            other => anyhow::bail!("no native dense route for {other:?}"),
        };
        Ok(Self { encoding, input, k: projection.shape[1], n: projection.shape[0],
            side: if projection.side == ShardSide::Input { 2 } else { 1 } })
    }
}

/// One GPU's selected programs. Export names remain only inventory identities;
/// execution resolves cores and transforms by their validated ABI geometry.
pub(crate) struct NativePrograms<'a> {
    linears: Vec<(Key, String, DenseLinear<'a>)>,
    companions: Vec<(Op, String, Companion<'a>)>,
}

impl<'a> NativePrograms<'a> {
    pub(crate) fn bind(programs: &'a Programs<'a>, manifest: &serde_json::Value,
        selected: &BTreeSet<String>) -> Result<Self> {
        let entries = manifest["programs"].as_array().context("manifest programs")?;
        let (mut linears, mut companions) = (Vec::new(), Vec::new());
        for name in selected {
            let entry = entries.iter().find(|e| e["name"].as_str() == Some(name)).context("selected manifest entry")?;
            if let Some(op) = cuteafd_ffi::native_dense_glue::op_of(entry)? {
                ensure!(!companions.iter().any(|(key, _, _)| *key == op), "duplicate native companion geometry {op:?}");
                companions.push((op, name.clone(), Companion::bind(programs, name, op)?));
            } else {
                let key = Key::validated(programs.spec(name)?)?;
                ensure!(!linears.iter().any(|(existing, _, _)| *existing == key), "duplicate native linear geometry {key:?}");
                linears.push((key, name.clone(), DenseLinear::bind(programs, name)?));
            }
        }
        Ok(Self { linears, companions })
    }

    pub(crate) fn packed_programs(&self) -> impl Iterator<Item = &str> {
        self.linears.iter().map(|(_, name, _)| name.as_str())
            .chain(self.companions.iter().map(|(_, name, _)| name.as_str()))
    }

    pub(crate) fn companion(&self, name: &str) -> Result<&Companion<'a>> {
        self.companions.iter().find(|(_, n, _)| n == name).map(|(_, _, program)| program)
            .with_context(|| format!("native companion {name} is not selected"))
    }

    fn op(&self, op: Op) -> Result<&Companion<'a>> {
        self.companions.iter().find(|(key, _, _)| *key == op).map(|(_, _, program)| program)
            .with_context(|| format!("native companion {op:?} is not selected"))
    }

    /// Caller enters the bound device and keeps every borrowed buffer alive
    /// until this stream has drained. No allocation or synchronization occurs.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear(&self, projection: &NativeProjection, operand: impl Fn(&str) -> Result<Buffer>,
        scratch: Scratch, x: Buffer, rows: usize, output: Buffer, a8: bool, stream: *mut c_void) -> Result<NativeOut> {
        let key = Key::projection(projection, rows, a8)?;
        let core = self.linears.iter().find(|(k, _, _)| *k == key).map(|(_, _, p)| p)
            .with_context(|| format!("native linear {key:?} is not selected"))?;
        let (k, n) = (key.k, key.n);
        let rows_u32 = u32::try_from(rows)?;
        let x = matrix(x, rows, k, 2)?;
        let acc = matrix(scratch.acc, rows, n, 4)?;
        match key.encoding {
            Encoding::Mxfp8 | Encoding::Int4G32 | Encoding::RowFp8 => {
                let (payload, scales) = match key.encoding {
                    Encoding::Mxfp8 => (operand("fp8")?, operand("e8m0")?),
                    Encoding::Int4G32 => (operand("q4")?, operand("s4")?),
                    _ => (operand("fp8")?, operand("scale")?),
                };
                // SAFETY: checkpoint operands match the validated rank geometry;
                // checked input/accumulator and caller-owned scratch live on stream.
                unsafe { core.launch(x.ptr, payload.ptr, scales.ptr, scratch.partial.ptr,
                    scratch.partial.bytes as u64, acc.ptr, rows_u32, 0, stream)?; }
                Ok(NativeOut::F32 { n })
            }
            Encoding::Trellis | Encoding::TrellisA8 => {
                let (payload, suh, svh) = (operand("trellis")?, operand("suh")?, operand("svh")?);
                let rotated = matrix(scratch.rotated, rows, k, 2)?;
                let output = matrix(output, rows, n, 2)?;
                let input = self.op(Op::TrellisInput { features: u32::try_from(k)?, rotated_bf16: key.input == Input::Bf16 })?;
                let transform = self.op(Op::TrellisOutput { features: u32::try_from(n)? })?;
                ensure!(scratch.table.bytes >= 8 && scratch.block_scales.bytes >= k / 128 * 2,
                    "native trellis table/scales scratch extent");
                // SAFETY: operands and lane views satisfy the validated contracts;
                // the caller initialized table[0] to rotated and stream orders use.
                unsafe {
                    input.launch(&[x.ptr, suh.ptr, rotated.ptr, scratch.block_scales.ptr], rows_u32, 0, stream)?;
                    let partial = if key.encoding == Encoding::TrellisA8 {
                        let records = scratch.records.context("A8 prefill lane records")?;
                        self.op(Op::Mx8Quant { features: u32::try_from(k)? })?
                            .launch(&[scratch.table.ptr, records.ptr], rows_u32, 0, stream)?;
                        records
                    } else { scratch.partial };
                    core.launch(scratch.table.ptr, payload.ptr, scratch.block_scales.ptr, partial.ptr,
                        partial.bytes as u64, acc.ptr, rows_u32, 0, stream)?;
                    transform.launch(&[acc.ptr, svh.ptr, output.ptr], rows_u32, 0, stream)?;
                }
                Ok(NativeOut::Bf16 { n })
            }
            other => anyhow::bail!("native linear cannot launch {other:?}"),
        }
    }
}

/// The allocations are exactly the format-level admission layout, shared by every family.
pub(crate) struct ScratchOwner<'a> {
    rotated: crate::shared::memory::device::Allocation<'a>,
    table: crate::shared::memory::device::Allocation<'a>,
    block_scales: crate::shared::memory::device::Allocation<'a>,
    partial: crate::shared::memory::device::Allocation<'a>,
    records: Option<crate::shared::memory::device::Allocation<'a>>,
    acc: crate::shared::memory::device::Allocation<'a>,
}

impl<'a> ScratchOwner<'a> {
    pub(crate) fn new(device: crate::shared::memory::device::Device<'a>,
        layout: &cuteafd_loader::plan::exlr8::dense::scratch::LinearScratchLayout) -> Result<Self> {
        use crate::shared::memory::device::Allocation;
        layout.bytes()?;
        let a = |bytes| Allocation::new(device, bytes);
        let owner = Self { rotated: a(layout.rotated)?, table: a(layout.table)?,
            block_scales: a(layout.block_scales)?, partial: a(layout.partial)?,
            records: layout.records.map(a).transpose()?, acc: a(layout.acc)? };
        device.run(|| device.library.copy_h2d(owner.table.buffer,
            &(owner.rotated.buffer.ptr as u64).to_le_bytes()))?;
        Ok(owner)
    }
    pub(crate) fn view(&self) -> Scratch {
        Scratch { rotated: self.rotated.buffer, table: self.table.buffer,
            block_scales: self.block_scales.buffer, partial: self.partial.buffer,
            records: self.records.as_ref().map(|a| a.buffer), acc: self.acc.buffer }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Scratch {
    pub rotated: Buffer,
    pub table: Buffer,
    pub block_scales: Buffer,
    pub partial: Buffer,
    pub records: Option<Buffer>,
    pub acc: Buffer,
}

pub(crate) enum NativeOut {
    F32 { n: usize },
    Bf16 { #[allow(dead_code)] n: usize },
}

fn matrix(buffer: Buffer, rows: usize, columns: usize, bytes: usize) -> Result<Buffer> {
    let extent = rows.checked_mul(columns).and_then(|v| v.checked_mul(bytes)).context("native matrix extent overflow")?;
    ensure!(rows > 0 && !buffer.ptr.is_null() && extent <= buffer.bytes, "native matrix view outside live storage");
    Ok(Buffer { bytes: extent, ..buffer })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_geometry_not_projection_alias_selects_routes() {
        let projection = |encoding, shape, side| NativeProjection { encoding, shape, side, rotation_id: 0 };
        for (shape, side) in [([4096, 4096], ShardSide::Output), ([4096, 2048], ShardSide::Input),
            ([32, 4096], ShardSide::Replicated)] {
            let p = projection(DenseEncoding::Mxfp8, shape, side);
            let key = Key::projection(&p, 1, true).unwrap();
            assert_eq!(key, Key::projection(&p, 4096, true).unwrap());
            assert_eq!((key.k, key.n, key.input), (shape[1], shape[0], Input::Bf16));
            assert_eq!(key.encoding, Encoding::Mxfp8);
        }
        let p = projection(DenseEncoding::Trellis { bits: 4 }, [8192, 4096], ShardSide::Output);
        assert_eq!(Key::projection(&p, 64, true).unwrap().input, Input::F16);
        assert_eq!(Key::projection(&p, 65, false).unwrap().input, Input::Bf16);
        assert_eq!(Key::projection(&p, 65, true).unwrap().encoding, Encoding::TrellisA8);
        // Checkpoint rotation IDs identify shared suh records, not runtime
        // table slots: this launch materializes its own rotation in slot zero.
        let shared_rotation = NativeProjection { rotation_id: 3, ..p.clone() };
        assert_eq!(Key::projection(&shared_rotation, 1, false).unwrap(), Key::projection(&p, 1, false).unwrap());
    }

    #[test]
    fn equal_shapes_keep_physical_side_and_input_dtype_distinct() {
        let p = NativeProjection { encoding: DenseEncoding::Trellis { bits: 4 }, shape: [8192, 4096],
            side: ShardSide::Output, rotation_id: 0 };
        assert_ne!(Key::projection(&p, 1, false).unwrap(), Key::projection(&p, 4096, false).unwrap());
        let input = NativeProjection { side: ShardSide::Input, ..p.clone() };
        assert_ne!(Key::projection(&p, 1, false).unwrap(), Key::projection(&input, 1, false).unwrap());
    }
}
