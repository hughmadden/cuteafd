//! Lossless EXL3 checkpoint tile reorder for the shared native dense ABI.
//! No codebook decode or floating-point conversion occurs during packing.
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DenseTrellisError {
    #[error("dense trellis geometry: {0}")]
    Geometry(&'static str),
    #[error("dense trellis {0} byte extent differs from checkpoint geometry")]
    Extent(&'static str),
}

/// Payload storage order (not necessarily the ownership axis for whole tensors).
/// Both layouts contain identical 16x16 tiles without changing packed bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrellisSide { Output, Input }

/// Standard EXL3 payload [K/16,N/16,16*bits] in little-endian I16 storage,
/// with unpacked F16 input/output sign-and-scale vectors.
pub struct DenseTrellisCheckpoint<'a> {
    /// Stored [N,K], including any explicitly declared output padding.
    pub shape: [usize; 2],
    pub bits: usize,
    pub payload: &'a [u8],
    pub suh: &'a [u8],
    pub svh: &'a [u8],
}

#[derive(Debug, PartialEq, Eq)]
pub struct NativeTrellisShard {
    pub shape: [usize; 2],
    pub side: TrellisSide,
    pub payload: Vec<u8>,
    pub suh: Vec<u8>,
    pub svh: Vec<u8>,
}

impl DenseTrellisCheckpoint<'_> {
    /// N-shards duplicate suh and slice svh; K-shards do the reverse. Shard
    /// boundaries must preserve H128 blocks, not merely whole 16x16 tiles.
    pub fn native_shard(&self, side: TrellisSide, rank: usize, ranks: usize)
        -> Result<NativeTrellisShard, DenseTrellisError> {
        use DenseTrellisError::{Extent, Geometry};
        let [n, k] = self.shape;
        if !matches!(self.bits, 2..=6) || !matches!(ranks, 1 | 2) || rank >= ranks
            || n == 0 || k == 0 || n % 128 != 0 || k % 128 != 0 {
            return Err(Geometry("K2-K6, TP1/TP2 and nonempty H128-aligned dimensions required"));
        }
        let bytes = n.checked_mul(k).and_then(|v| v.checked_mul(self.bits))
            .map(|v| v / 8).ok_or(Geometry("payload size overflow"))?;
        let input_bytes = k.checked_mul(2).ok_or(Geometry("input vector size overflow"))?;
        let output_bytes = n.checked_mul(2).ok_or(Geometry("output vector size overflow"))?;
        if self.payload.len() != bytes { return Err(Extent("payload")); }
        if self.suh.len() != input_bytes { return Err(Extent("suh")); }
        if self.svh.len() != output_bytes { return Err(Extent("svh")); }
        let (local_n, local_k) = match side {
            TrellisSide::Output => (n / ranks, k),
            TrellisSide::Input => (n, k / ranks),
        };
        if local_n % 128 != 0 || local_k % 128 != 0 {
            return Err(Geometry("TP shard splits an H128 block"));
        }
        let (n0, k0) = match side {
            TrellisSide::Output => (rank * local_n / 16, 0),
            TrellisSide::Input => (0, rank * local_k / 16),
        };
        let tile_bytes = 32 * self.bits;
        let mut payload = Vec::with_capacity(bytes / ranks);
        let mut append = |ki: usize, ni: usize| {
            let at = ((k0 + ki) * (n / 16) + n0 + ni) * tile_bytes;
            payload.extend_from_slice(&self.payload[at..at + tile_bytes]);
        };
        match side {
            TrellisSide::Output => {
                // [N128, K16, N16-within-N128]
                for nb in 0..local_n / 128 {
                    for ki in 0..local_k / 16 {
                        for ni in 0..8 { append(ki, nb * 8 + ni); }
                    }
                }
            }
            TrellisSide::Input => {
                // [K128, N128, K16-within-K128, N16-within-N128]
                for kb in 0..local_k / 128 {
                    for nb in 0..local_n / 128 {
                        for ki in 0..8 {
                            for ni in 0..8 { append(kb * 8 + ki, nb * 8 + ni); }
                        }
                    }
                }
            }
        }
        let suh = self.suh[k0 * 32..(k0 * 16 + local_k) * 2].to_vec();
        let svh = self.svh[n0 * 32..(n0 * 16 + local_n) * 2].to_vec();
        Ok(NativeTrellisShard { shape: [local_n, local_k], side, payload, suh, svh })
    }
}

/// Owned native bytes, read through the checkpoint's ordinary tensor reader.
/// Codebook markers are validated on the host and never uploaded to the GPU.
#[derive(Debug)]
pub struct LoadedDenseTrellis {
    pub codebook: super::exl3_storage::Exl3Codebook,
    pub shard: NativeTrellisShard,
}

pub fn load_native_shard(stem: &str, module: &super::exl3_storage::Exl3Module,
    side: TrellisSide, rank: usize, ranks: usize,
    mut read: impl FnMut(&str) -> anyhow::Result<crate::LoadedTensor>) -> anyhow::Result<LoadedDenseTrellis> {
    use anyhow::{ensure, Context};
    let mut load = |suffix: &str| -> anyhow::Result<Vec<u8>> {
        let name = format!("{stem}.{suffix}");
        let expected = module.tensors.get(suffix).with_context(||format!("{name}: missing validated header"))?;
        let tensor = read(&name)?;
        ensure!(tensor.info.name == name && tensor.info.dtype == expected.dtype && tensor.info.shape == expected.shape,
            "{name}: loaded tensor differs from validated checkpoint headers");
        ensure!(tensor.info.byte_length == tensor.bytes.len() as u64,
            "{name}: loaded byte extent differs from checkpoint header");
        Ok(tensor.bytes)
    };
    let marker = load(module.codebook.marker())?;
    ensure!(marker.as_slice() == module.codebook.multiplier().to_le_bytes(),
        "{stem}: unsupported {} marker payload", module.codebook.marker());
    let payload = load("trellis")?;
    let suh = load("suh")?;
    let svh = load("svh")?;
    let checkpoint = DenseTrellisCheckpoint { shape: [module.output_features,module.input_features],
        bits: module.bits, payload: &payload, suh: &suh, svh: &svh };
    let shard = checkpoint.native_shard(side,rank,ranks)?;
    Ok(LoadedDenseTrellis { codebook: module.codebook, shard })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_tile_addresses_match_both_abi_layouts_without_changing_bits() {
        let (n, k) = (512, 768);
        let suh = (0..k as u16).flat_map(u16::to_le_bytes).collect::<Vec<_>>();
        let svh = (0..n as u16).map(|v| v | 0x8000).flat_map(u16::to_le_bytes).collect::<Vec<_>>();
        for bits in 2..=6 {
            let tile_bytes = 32 * bits;
            let payload = (0..n * k * bits / 8).map(|i| (i ^ (i >> 8)) as u8).collect::<Vec<_>>();
            let source = DenseTrellisCheckpoint { shape: [n, k], bits, payload: &payload, suh: &suh, svh: &svh };
            for side in [TrellisSide::Output, TrellisSide::Input] {
                for ranks in [1, 2] {
                    for rank in 0..ranks {
                        let shard = source.native_shard(side, rank, ranks).unwrap();
                        let [ln, lk] = shard.shape;
                        assert_eq!(shard.payload.len(), payload.len() / ranks);
                        let (k0, n0) = match side {
                            TrellisSide::Output => (0, rank * ln / 16),
                            TrellisSide::Input => (rank * lk / 16, 0),
                        };
                        assert_eq!(shard.suh, suh[k0 * 32..(k0 * 16 + lk) * 2]);
                        assert_eq!(shard.svh, svh[n0 * 32..(n0 * 16 + ln) * 2]);
                        for nb in 0..ln / 128 {
                            for kb in 0..lk / 128 {
                                for k16 in 0..8 {
                                    for warp in 0..4 {
                                        for n16 in 0..2 {
                                            // These are the native kernel's multiply() tile addresses.
                                            let dest = match side {
                                                TrellisSide::Output => nb * (lk / 16 * 8) + (kb * 8 + k16) * 8 + warp * 2 + n16,
                                                TrellisSide::Input => kb * (ln / 16 * 8) + nb * 64 + k16 * 8 + warp * 2 + n16,
                                            } * tile_bytes;
                                            let src = ((k0 + kb * 8 + k16) * (n / 16) + n0 + nb * 8 + warp * 2 + n16) * tile_bytes;
                                            assert_eq!(&shard.payload[dest..dest + tile_bytes], &payload[src..src + tile_bytes]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn loads_both_codebooks_and_checks_payload_markers_before_packing() {
        use super::super::exl3_storage::{Exl3Codebook, Exl3StoredTensor, Exl3Module};
        use cuteafd_core::{DType, TensorInfo, TensorRole};
        use std::collections::BTreeMap;
        for codebook in [Exl3Codebook::Mcg, Exl3Codebook::Mul1] {
            let module = Exl3Module { codebook, bits:6, input_features:128, output_features:256,
                tensors:BTreeMap::from([
                    ("trellis".into(),Exl3StoredTensor{dtype:DType::I16,shape:vec![8,16,96]}),
                    ("suh".into(),Exl3StoredTensor{dtype:DType::F16,shape:vec![128]}),
                    ("svh".into(),Exl3StoredTensor{dtype:DType::F16,shape:vec![256]}),
                    (codebook.marker().into(),Exl3StoredTensor{dtype:DType::I32,shape:vec![]})]) };
            let read = |name: &str, corrupt: bool| -> anyhow::Result<crate::LoadedTensor> {
                let suffix=name.strip_prefix("fixture.").unwrap();
                let meta=&module.tensors[suffix];
                let mut bytes=if suffix==codebook.marker() {codebook.multiplier().to_le_bytes().to_vec()}
                    else {vec![0;meta.shape.iter().product::<usize>()*2]};
                if corrupt && suffix==codebook.marker() {bytes[0]^=1;}
                Ok(crate::LoadedTensor {info:TensorInfo{name:name.into(),file:"fixture".into(),
                    dtype:meta.dtype.clone(),shape:meta.shape.clone(),byte_offset:0,byte_length:bytes.len()as u64,
                    role:TensorRole::Other,layer_id:None,expert_id:None,is_quantization_metadata:false},
                    source_path:Default::default(),bytes,elapsed_micros:0,sha256:String::new()})
            };
            let loaded=load_native_shard("fixture",&module,TrellisSide::Output,1,2,|n|read(n,false)).unwrap();
            assert_eq!(loaded.codebook,codebook);
            assert_eq!(loaded.shard.shape,[128,128]);
            let error=load_native_shard("fixture",&module,TrellisSide::Output,0,1,|n|read(n,true)).unwrap_err();
            assert!(error.to_string().contains("marker payload"));
        }
    }

    #[test]
    fn rejects_bad_extents_overflow_and_non_h128_shards() {
        let payload = vec![0; 128 * 128 * 6 / 8];
        let rotation = vec![0; 256];
        let mut s = DenseTrellisCheckpoint { shape: [128, 128], bits: 6, payload: &payload, suh: &rotation, svh: &rotation };
        assert!(s.native_shard(TrellisSide::Input, 0, 1).is_ok());
        for (rank, ranks) in [(1, 1), (0, 0), (0, 3), (0, 2)] {
            assert!(s.native_shard(TrellisSide::Input, rank, ranks).is_err());
        }
        s.payload = &payload[..payload.len() - 1];
        assert!(s.native_shard(TrellisSide::Output, 0, 1).is_err());
        s.payload = &payload;
        s.suh = &rotation[..254];
        assert!(s.native_shard(TrellisSide::Output, 0, 1).is_err());
        s.suh = &rotation;
        s.svh = &rotation[..254];
        assert!(s.native_shard(TrellisSide::Output, 0, 1).is_err());
        s.svh = &rotation;
        s.shape = [576, 128];
        assert!(s.native_shard(TrellisSide::Output, 0, 1).is_err());
        s.shape = [usize::MAX - 127, 128];
        assert!(s.native_shard(TrellisSide::Output, 0, 1).is_err());
        s.shape = [128, 128];
        s.bits = 7;
        assert!(s.native_shard(TrellisSide::Output, 0, 1).is_err());
    }
}
