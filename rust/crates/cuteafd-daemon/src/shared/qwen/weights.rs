//! Checkpoint-native tensor reads and packing shared by hybrid Qwen families.
use crate::shared::memory::device::{Allocation, Device};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::plan::checkpoint::{Checkpoint, CheckpointTensor};
use std::os::unix::fs::FileExt;

pub(crate) struct WeightReader<'a> {
    pub(crate) library: &'a NativeLibrary,
    pub(crate) checkpoint: &'a Checkpoint,
}

pub(crate) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn widen(bytes: &[u8], dtype: DType) -> Result<Vec<u8>> {
    match dtype {
        DType::Bf16 => {
            ensure!(bytes.len() % 2 == 0, "incomplete BF16 tensor");
            Ok(bytes.chunks_exact(2).flat_map(|b| {
                (u32::from(u16::from_le_bytes([b[0], b[1]])) << 16).to_le_bytes()
            }).collect())
        }
        DType::F32 => {
            ensure!(bytes.len() % 4 == 0, "incomplete FP32 tensor");
            Ok(bytes.to_vec())
        }
        other => anyhow::bail!("expected BF16 or FP32, found {other:?}"),
    }
}

fn row_layout(tensors: &[(Vec<u8>, DType, Vec<usize>)], pad_rows: usize) -> Result<(usize, usize)> {
    let first = tensors.first().context("empty Qwen row concatenation")?;
    ensure!(first.2.len() == 2 && first.2[0] > 0 && first.2[1] > 0, "expected nonempty BF16 matrix");
    let cols = first.2[1];
    let mut rows = pad_rows;
    for (bytes, dtype, shape) in tensors {
        ensure!(*dtype == DType::Bf16 && shape.len() == 2 && shape[0] > 0 && shape[1] == cols,
            "Qwen packed rows require BF16 matrices sharing columns");
        let size = shape[0].checked_mul(cols).and_then(|n| n.checked_mul(2)).context("Qwen packed tensor size overflow")?;
        ensure!(bytes.len() == size, "Qwen packed tensor byte count disagrees with shape");
        rows = rows.checked_add(shape[0]).context("Qwen packed row count overflow")?;
    }
    let bytes = rows.checked_mul(cols).and_then(|n| n.checked_mul(2)).context("Qwen packed tensor size overflow")?;
    Ok((cols, bytes))
}

fn tensor<'a>(checkpoint: &'a Checkpoint, name: &str) -> Result<&'a CheckpointTensor> {
    cuteafd_ffi::memory_ledger::tensor(name);
    let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
        .map_err(|_| anyhow::anyhow!("checkpoint has no tensor {name}"))?;
    Ok(&checkpoint.tensors[at])
}

fn raw(checkpoint: &Checkpoint, name: &str) -> Result<(Vec<u8>, DType, Vec<usize>)> {
    let tensor = tensor(checkpoint, name)?;
    let size = usize::try_from(tensor.meta.byte_length).context("Qwen tensor byte count exceeds usize")?;
    let mut bytes = vec![0u8; size];
    std::fs::File::open(checkpoint.snapshot.join(&tensor.shard))?
        .read_exact_at(&mut bytes, tensor.meta.byte_offset)
        .with_context(|| format!("reading {name}"))?;
    Ok((bytes, tensor.meta.dtype.clone(), tensor.meta.shape.clone()))
}

impl<'a> WeightReader<'a> {
    pub(crate) fn tensor(&self, name: &str) -> Result<&'a CheckpointTensor> {
        tensor(self.checkpoint, name)
    }

    pub(crate) fn raw(&self, name: &str) -> Result<(Vec<u8>, DType, Vec<usize>)> {
        raw(self.checkpoint, name)
    }

    pub(crate) fn upload(&self, bytes: &[u8]) -> Result<Allocation<'a>> {
        let device = Device { library: self.library, id: self.library.cuda_get_device()? };
        let allocation = Allocation::new(device, bytes.len().max(256))?;
        self.library.copy_h2d(allocation.buffer, bytes)?;
        Ok(allocation)
    }

    pub(crate) fn bf16(&self, name: &str) -> Result<Allocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        let (bytes, dtype, _) = self.raw(name)?;
        ensure!(dtype == DType::Bf16, "{name}: expected BF16, found {dtype:?}");
        self.upload(&bytes)
    }

    /// Concatenate BF16 matrices in source order, retaining zero padded rows.
    pub(crate) fn rows(&self, names: &[String], pad_rows: usize) -> Result<Allocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        let tensors = names.iter().map(|n| self.raw(n)).collect::<Result<Vec<_>>>()?;
        let (cols, bytes) = row_layout(&tensors, pad_rows).with_context(|| format!("packing {names:?}"))?;
        let device = Device { library: self.library, id: self.library.cuda_get_device()? };
        let out = Allocation::new(device, bytes)?;
        self.library.cuda_zero_bytes(out.buffer, out.buffer.bytes)?;
        let mut row = 0;
        for (bytes, _, shape) in &tensors {
            let dest = CuteafdDeviceBuffer {
                ptr: out.buffer.ptr.cast::<u8>().wrapping_add(row * cols * 2).cast(),
                bytes: bytes.len(), ..out.buffer
            };
            self.library.copy_h2d(dest, bytes)?;
            row += shape[0];
        }
        Ok(out)
    }

    /// Native FP32 values stay byte-identical; BF16 values widen exactly.
    pub(crate) fn f32(&self, name: &str) -> Result<Allocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("f32");
        let (bytes, dtype, _) = self.raw(name)?;
        self.upload(&widen(&bytes, dtype).with_context(|| name.to_owned())?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_reads_respect_offsets_and_reject_missing_or_truncated_payloads() -> Result<()> {
        use cuteafd_loader::SafetensorsTensorMetadata;
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join("weights.bin"), [9, 9, 0, 63, 128, 63, 7])?;
        let checkpoint = Checkpoint { snapshot: dir.path().into(), config: serde_json::json!({}),
            quantize_config: None, weight_map: Default::default(), missing_shards: vec![], shard_bytes: 7,
            tensors: vec![CheckpointTensor { shard: "weights.bin".into(), meta: SafetensorsTensorMetadata {
                name: "weight".into(), dtype: DType::Bf16, shape: vec![1, 2], byte_offset: 2, byte_length: 4,
            } }] };
        assert_eq!(raw(&checkpoint, "weight")?, (vec![0, 63, 128, 63], DType::Bf16, vec![1, 2]));
        assert!(raw(&checkpoint, "missing").is_err());
        std::fs::write(dir.path().join("weights.bin"), [9, 9, 0])?;
        assert!(raw(&checkpoint, "weight").is_err());
        Ok(())
    }

    #[test]
    fn widening_preserves_all_bf16_bits_and_native_fp32_bytes() -> Result<()> {
        let bits: Vec<u16> = (0..=u16::MAX).collect();
        let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let widened = widen(&bytes, DType::Bf16)?;
        for (value, bytes) in bits.iter().zip(widened.chunks_exact(4)) {
            assert_eq!(u32::from_le_bytes(bytes.try_into().unwrap()), u32::from(*value) << 16);
        }
        assert_eq!(widen(&widened, DType::F32)?, widened);
        assert!(widen(&[0], DType::Bf16).is_err());
        assert!(widen(&[0; 3], DType::F32).is_err());
        Ok(())
    }

    #[test]
    fn packed_rows_validate_shape_before_index_or_allocation() -> Result<()> {
        let matrix = (vec![0; 12], DType::Bf16, vec![2, 3]);
        assert_eq!(row_layout(&[matrix.clone(), matrix.clone()], 1)?, (3, 30));
        assert!(row_layout(&[], 0).is_err());
        for invalid in [(vec![], DType::Bf16, vec![]), (vec![0; 12], DType::F32, vec![2, 3]),
            (vec![0; 12], DType::Bf16, vec![2, 4]), (vec![0; 11], DType::Bf16, vec![2, 3]),
            (vec![], DType::Bf16, vec![usize::MAX, 3])] {
            assert!(row_layout(&[matrix.clone(), invalid], 0).is_err());
        }
        assert!(row_layout(&[matrix], usize::MAX).is_err());
        Ok(())
    }
}
