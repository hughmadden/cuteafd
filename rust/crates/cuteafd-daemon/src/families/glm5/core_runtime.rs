//! GLM's native K6 projection seams over shared compressed-weight owners.
use crate::shared::{dense_trellis::{CoreWeights, Projection, ProjectionLayout, ProjectionScratch},
    memory::device::Device, speculation::dflash::TargetHead};
use anyhow::{ensure, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer, dense_trellis::TrellisDType, programs::Programs};
use std::ffi::c_void;

/// Target and drafter use independent scratch even though they share the head weights.
pub(crate) struct CoreHead<'a> {
    projection: Projection<'a>, scratch: ProjectionScratch<'a>, device: Device<'a>,
    stream: *mut c_void, rows: usize,
}
impl<'a> CoreHead<'a> {
    pub(crate) fn new(programs: &'a Programs<'a>, weights: &CoreWeights<'a>, rows: usize,
        device: Device<'a>, stream: *mut c_void) -> Result<Self> {
        ensure!(weights.side == cuteafd_loader::formats::dense_trellis::TrellisSide::Output,
            "GLM K6 head must be whole on rank 0");
        let projection = device.run(|| Projection::load_geometry(programs, weights.shape[1], weights.shape[0],
            1, book(weights), rows, TrellisDType::F32))?;
        let scratch = ProjectionLayout::new(&projection.program, rows)?.allocate(device)?;
        Ok(Self { projection, scratch, device, stream, rows })
    }

    pub(crate) fn launch(&self, weights: &CoreWeights<'_>, input: Buffer, output: Buffer,
        rows: usize, stream: *mut c_void) -> Result<()> {
        ensure!((1..=self.rows).contains(&rows) && stream == self.stream,
            "GLM K6 head rows/stream differ from admitted scratch");
        self.device.run(|| self.projection.local(rows, input, weights, &self.scratch, output, stream))
    }

    pub(crate) fn target<'s>(&'s self, weights: &'s CoreWeights<'_>) -> TargetHead<'s> {
        TargetHead::Launch(Box::new(move |input, logits, rows, stream| {
            let buffer = |ptr, bytes| Buffer { ptr, bytes, device_id: self.device.id, flags: 0 };
            self.launch(weights, buffer(input.cast_mut(), rows * weights.shape[1] * 2),
                buffer(logits.cast(), rows * weights.shape[0] * 4), rows, stream)
        }))
    }
}

/// Copy only logical columns after the complete stored-N transform has run.
pub(crate) fn pack_layout(rows: usize, stored: usize, logical: usize, pitch: usize, column: usize)
    -> Result<(usize, usize, usize, usize)> {
    ensure!(rows > 0 && stored >= logical && logical > 0
        && column.checked_add(logical).is_some_and(|end| end <= pitch),
        "GLM K6 logical crop/pack geometry");
    let bytes = |n: usize| n.checked_mul(2).ok_or_else(|| anyhow::anyhow!("GLM K6 pack byte overflow"));
    let source = rows.checked_mul(stored).ok_or_else(|| anyhow::anyhow!("GLM K6 source extent overflow"))?;
    let target = rows.checked_mul(pitch).ok_or_else(|| anyhow::anyhow!("GLM K6 target extent overflow"))?;
    bytes(source)?; bytes(target)?;
    Ok((bytes(stored)?, bytes(pitch)?, bytes(logical)?, bytes(column)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_padded_transform_is_cropped_only_when_packing_qkv_rows() -> Result<()> {
        for rows in [1, 8, 64, 4096] {
            let (src_pitch, dst_pitch, width, offset) = pack_layout(rows, 640, 576, 2624, 2048)?;
            assert_eq!((src_pitch, dst_pitch, width, offset), (1280, 5248, 1152, 4096));
            let full = (0..rows * 640).map(|i| (i ^ (i >> 8)) as u16).collect::<Vec<_>>();
            let mut packed = vec![0xDEADu16; rows * 2624];
            for r in 0..rows {
                packed[r * 2624 + 2048..][..576].copy_from_slice(&full[r * 640..][..576]);
            }
            for r in 0..rows {
                assert_eq!(&packed[r * 2624 + 2048..][..576], &full[r * 640..][..576]);
                assert!(packed[r * 2624..][..2048].iter().all(|&v| v == 0xDEAD));
            }
        }
        assert_eq!(pack_layout(1, 2048, 2048, 2624, 0)?, (4096, 5248, 4096, 0));
        for args in [(0, 640, 576, 2624, 2048), (1, 576, 640, 2624, 0),
            (1, 640, 576, 2624, 2049), (usize::MAX, 640, 576, 2624, 2048)] {
            assert!(pack_layout(args.0, args.1, args.2, args.3, args.4).is_err());
        }
        Ok(())
    }
}

pub(crate) fn book(weights: &CoreWeights<'_>) -> &'static str {
    match weights.codebook {
        cuteafd_loader::formats::exl3_storage::Exl3Codebook::Mcg => "mcg",
        cuteafd_loader::formats::exl3_storage::Exl3Codebook::Mul1 => "mul1",
    }
}
