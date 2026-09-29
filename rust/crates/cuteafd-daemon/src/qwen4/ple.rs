//! The PLE n-gram table (`ple.ple_embedding.ngram_embedding.shard_*`): 128
//! shards of 2,500,012 rows x 160, BF16 (~95 GiB) or E4M3 with one BF16
//! scale (~48 GiB), held as one array the `qwen4_ple_*` program gathers from.
//!
//! Placement: `host` keeps it in mapped pinned host memory (the program reads
//! each token's 16 rows over PCIe, 5 KiB per token for BF16); `device` copies
//! it to the GPU. Token ids hash to rows on the host (`NgramHasher`).
use crate::v41_memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{CuteafdHostBuffer, NativeLibrary};
use cuteafd_loader::plan::checkpoint::Checkpoint;
use cuteafd_loader::qwen4_exp::{NgramHasher, Qwen4Config};
use std::ffi::c_void;
use std::os::unix::fs::FileExt;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum PlePlacement {
    /// Mapped pinned host memory, gathered over PCIe.
    Host,
    /// GPU memory.
    Device,
}

enum Storage<'a> {
    Host(&'a NativeLibrary, CuteafdHostBuffer),
    Device(DeviceAllocation<'a>),
}

pub(crate) struct PleTable<'a> {
    storage: Storage<'a>,
    /// Device-visible address of row 0.
    pub table: *mut c_void,
    /// FP32 [1] table scale on the device (1.0 for BF16 tables).
    pub scale: DeviceAllocation<'a>,
    pub fp8: bool,
    pub rows: usize,
    pub hasher: NgramHasher,
}

impl Drop for PleTable<'_> {
    fn drop(&mut self) {
        if let Storage::Host(library, buffer) = &mut self.storage {
            if let Err(error) = library.free_host_buffer(buffer) {
                tracing::error!(%error, "freeing the PLE table");
            }
        }
    }
}

fn i64s(bytes: &[u8]) -> Vec<i64> {
    bytes.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect()
}

impl<'a> PleTable<'a> {
    pub fn load(library: &'a NativeLibrary, checkpoint: &Checkpoint, cfg: &Qwen4Config, layer: usize,
        placement: PlePlacement, threads: usize) -> Result<Self> {
        let loader = super::weights::Qwen4Loader { library, checkpoint, fp8_decode: false, stream: std::ptr::null_mut() };
        let prefix = format!("{}layers.{layer}.ple.ple_embedding.", super::weights::PREFIX);
        let multipliers = i64s(&loader.raw(&format!("{prefix}layer_multipliers"))?.0);
        let sizes = i64s(&loader.raw(&format!("{prefix}ngram_heads_vocab_sizes"))?.0);
        let offsets = i64s(&loader.raw(&format!("{prefix}ngram_heads_offsets"))?.0);
        let hasher = NgramHasher { multipliers, sizes, offsets, heads_per_ngram: cfg.heads_per_ngram, eos: cfg.eos };
        ensure!(hasher.heads() == cfg.ple_rows() && hasher.ngram_size() == cfg.ngram_size,
            "PLE hash tables do not match the config");
        let ple_index = cfg.ple_layers.iter().position(|&l| l == layer).context("not a PLE layer")?;
        let derived = NgramHasher::from_config(cfg.vocab_size, 20_000_000, cfg.ngram_size, cfg.heads_per_ngram,
            ple_index, 1234, cfg.eos);
        if derived != hasher {
            tracing::warn!("PLE hash tables differ from the config-derived ones; using the checkpoint's");
        }
        // The shards, in row order.
        let mut shards = Vec::new();
        for shard in 0.. {
            let name = format!("{prefix}ngram_embedding.shard_{shard}.weight");
            let Ok(tensor) = loader.tensor(&name) else { break };
            shards.push(tensor);
        }
        ensure!(!shards.is_empty(), "no PLE table shards under {prefix}");
        let dtype = shards[0].meta.dtype.clone();
        let fp8 = match dtype {
            DType::Bf16 => false,
            DType::F8E4M3 => true,
            other => anyhow::bail!("PLE table dtype {other:?} is not BF16 or E4M3"),
        };
        let width = shards[0].meta.shape[1];
        ensure!(width * cfg.ple_rows() == cfg.ple_dim, "PLE rows of {width} do not tile {}", cfg.ple_dim);
        ensure!(shards.iter().all(|s| s.meta.dtype == dtype && s.meta.shape.len() == 2 && s.meta.shape[1] == width),
            "PLE shards disagree on dtype or width");
        let rows: usize = shards.iter().map(|s| s.meta.shape[0]).sum();
        ensure!(rows as i64 >= hasher.rows(), "PLE table has {rows} rows, the hash addresses {}", hasher.rows());
        let row_bytes = width * if fp8 { 1 } else { 2 };
        let total = rows * row_bytes;
        let scale = if fp8 {
            let (bytes, dtype, _) = loader.raw(&format!("{prefix}ngram_embedding.weight_scale"))?;
            match dtype {
                DType::Bf16 => f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16),
                DType::F32 => f32::from_le_bytes(bytes[..4].try_into().unwrap()),
                other => anyhow::bail!("PLE table scale dtype {other:?}"),
            }
        } else {
            1.0
        };
        let scale_dev = DeviceAllocation::new(library, 256)?;
        library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: 4, ..scale_dev.buffer }, &scale.to_le_bytes())?;
        let started = Instant::now();
        // Byte offset of each shard in the array.
        let mut starts = Vec::with_capacity(shards.len());
        let mut at = 0usize;
        for shard in &shards {
            starts.push(at);
            at += shard.meta.shape[0] * row_bytes;
        }
        let read_into = |base: *mut u8| -> Result<()> {
            let next = std::sync::atomic::AtomicUsize::new(0);
            let base = base as usize;
            std::thread::scope(|scope| -> Result<()> {
                let workers: Vec<_> = (0..threads.max(1)).map(|_| scope.spawn(|| -> Result<()> {
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(shard) = shards.get(i) else { return Ok(()) };
                        let file = std::fs::File::open(checkpoint.snapshot.join(&shard.shard))?;
                        let len = shard.meta.shape[0] * row_bytes;
                        ensure!(len as u64 == shard.meta.byte_length, "PLE shard {i} byte length");
                        // SAFETY: shard i owns bytes starts[i]..starts[i]+len of the table.
                        let dest = unsafe { std::slice::from_raw_parts_mut((base + starts[i]) as *mut u8, len) };
                        for (k, chunk) in dest.chunks_mut(64 << 20).enumerate() {
                            file.read_exact_at(chunk, shard.meta.byte_offset + (k * (64 << 20)) as u64)
                                .with_context(|| format!("reading PLE shard {i}"))?;
                        }
                    }
                })).collect();
                for worker in workers {
                    worker.join().map_err(|_| anyhow::anyhow!("PLE reader panicked"))??;
                }
                Ok(())
            })
        };
        let (storage, table) = match placement {
            PlePlacement::Host => {
                let buffer = library.alloc_host_buffer(total)?;
                read_into(buffer.ptr.cast())?;
                let alias = library.cuda_host_buffer_device_alias(buffer)?;
                (Storage::Host(library, buffer), alias.ptr)
            }
            PlePlacement::Device => {
                let device = DeviceAllocation::new(library, total)?;
                // Stage shard by shard through pageable memory (a one-time load).
                let mut staging = vec![0u8; shards.iter().map(|s| s.meta.byte_length as usize).max().unwrap_or(0)];
                for (i, shard) in shards.iter().enumerate() {
                    let len = shard.meta.byte_length as usize;
                    let file = std::fs::File::open(checkpoint.snapshot.join(&shard.shard))?;
                    file.read_exact_at(&mut staging[..len], shard.meta.byte_offset)?;
                    let dest = cuteafd_ffi::CuteafdDeviceBuffer {
                        // SAFETY: the shard's bytes lie inside the device table.
                        ptr: unsafe { device.buffer.ptr.cast::<u8>().add(starts[i]) }.cast(),
                        bytes: len,
                        ..device.buffer
                    };
                    library.copy_h2d(dest, &staging[..len])?;
                }
                let ptr = device.buffer.ptr;
                (Storage::Device(device), ptr)
            }
        };
        tracing::info!(gib = total as f64 / (1u64 << 30) as f64, fp8, ?placement,
            elapsed_ms = started.elapsed().as_millis() as u64, "PLE n-gram table resident");
        Ok(Self { storage, table, scale: scale_dev, fp8, rows, hasher })
    }
}
