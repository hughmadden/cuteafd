//! Opt-in program boundaries, copied only after the native stream drains.
use super::{execution::BatchView, MiniMaxError};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::minimax::{Buffer, Commands};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::{Path, PathBuf}};

pub(super) struct Diagnostics {
    root: PathBuf,
    pub layer: usize,
    steps: Vec<Value>,
    next_position: usize,
}

pub(super) struct StepDump {
    root: PathBuf,
    directory: String,
    positions: Vec<i64>,
    tensors: Map<String, Value>,
    q2k: Vec<u8>,
}

fn extent(dtype: &str, shape: &[usize]) -> Result<usize> {
    let width = match dtype { "bf16" => 2, "f32" | "i32" => 4, "u8" => 1,
        _ => return Err(MiniMaxError::Invalid("diagnostic dtype").into()) };
    ensure!(!shape.is_empty() && shape.iter().all(|&n| n > 0), MiniMaxError::Invalid("diagnostic shape"));
    shape.iter().try_fold(width, |n: usize, &d| n.checked_mul(d).context("diagnostic extent overflow"))
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = fs::OpenOptions::new().create_new(true).write(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

impl Diagnostics {
    pub fn from_env(layers: usize) -> Result<Option<Self>> {
        let root = std::env::var_os("CUTEAFD_MINIMAX_DUMP_DIR");
        let layer = std::env::var_os("CUTEAFD_MINIMAX_DUMP_LAYER");
        match (root, layer) {
            (None, None) => Ok(None),
            (Some(root), Some(layer)) => {
                let layer: usize = layer.to_str().context("diagnostic layer encoding")?.parse()?;
                ensure!(layer < layers, MiniMaxError::Invalid("diagnostic layer"));
                Self::new(PathBuf::from(root), layer).map(Some)
            }
            _ => Err(MiniMaxError::Invalid("set both diagnostic directory and layer").into()),
        }
    }
    fn new(root: PathBuf, layer: usize) -> Result<Self> {
        fs::create_dir_all(&root)?;
        ensure!(fs::read_dir(&root)?.next().is_none(), MiniMaxError::Invalid("diagnostic directory must be empty"));
        Ok(Self { root, layer, steps: Vec::new(), next_position: 0 })
    }
    pub fn begin(&self, batch: &BatchView) -> Result<StepDump> {
        ensure!(batch.requests.len() == 1 && batch.requests[0].start_position == self.next_position
            && batch.positions.iter().enumerate().all(|(i, &p)| p == (self.next_position + i) as i64),
            MiniMaxError::Invalid("diagnostic requires one contiguous request starting at zero"));
        let directory = format!("step{:06}", self.steps.len());
        fs::create_dir(self.root.join(&directory))?;
        Ok(StepDump { root: self.root.clone(), directory, positions: batch.positions.clone(),
            tensors: Map::new(), q2k: Vec::new() })
    }
    pub fn finish(&mut self, step: StepDump) -> Result<()> {
        let rows = step.positions.len();
        self.steps.push(json!({"positions":step.positions,"tensors":step.tensors}));
        let manifest = json!({"schema":"cuteafd.minimax.layer-diagnostic/1", "layer":self.layer,
            "activation":"served-cache", "steps":self.steps});
        let temporary = self.root.join("manifest.pending.json");
        write_new(&temporary, &serde_json::to_vec_pretty(&manifest)?)?;
        fs::rename(temporary, self.root.join("manifest.json"))?;
        self.next_position += rows;
        Ok(())
    }
}

impl StepDump {
    pub fn bytes(&mut self, name: &str, dtype: &str, shape: &[usize], bytes: &[u8]) -> Result<()> {
        ensure!(name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !name.is_empty()
            && !self.tensors.contains_key(name), MiniMaxError::Invalid("diagnostic tensor name"));
        ensure!(bytes.len() == extent(dtype, shape)?, MiniMaxError::Invalid("diagnostic byte extent"));
        let path = format!("{}/{name}.bin", self.directory);
        write_new(&self.root.join(&path), bytes)?;
        self.tensors.insert(name.into(), json!({"path":path,"dtype":dtype,"shape":shape,
            "sha256":format!("{:x}", Sha256::digest(bytes))}));
        Ok(())
    }
    pub fn tensor(&mut self, commands: &Commands<'_>, name: &str, dtype: &str,
        shape: &[usize], source: &Buffer<'_>) -> Result<()> {
        commands.drain()?;
        let mut bytes = vec![0; extent(dtype, shape)?];
        commands.library().cuda_set_device(commands.device())?;
        let source=source.view(0, bytes.len())?;
        commands.library().copy_d2h(&mut bytes, source.raw())?;
        self.bytes(name, dtype, shape, &bytes)
    }
    pub fn selection(&mut self, commands: &Commands<'_>, source: &Buffer<'_>, rows: usize,
        capacity: usize) -> Result<()> {
        commands.drain()?;
        let mut bytes = vec![0; 4 * capacity * 16 * 4];
        commands.library().cuda_set_device(commands.device())?;
        commands.library().copy_d2h(&mut bytes, source.raw())?;
        self.q2k.extend(selection_rows(&bytes, rows, capacity)?);
        Ok(())
    }
    pub fn finish_selection(&mut self) -> Result<()> {
        let bytes = std::mem::take(&mut self.q2k);
        self.bytes("q2k", "i32", &[self.positions.len(), 4, 16], &bytes)
    }
    pub fn cache_rows(&mut self, commands: &Commands<'_>, name: &str, dtype: &str,
        shape: &[usize], source: &Buffer<'_>, slots: &[i64], page_bytes: usize,
        plane_offset: usize, row_bytes: usize) -> Result<()> {
        commands.drain()?;
        commands.library().cuda_set_device(commands.device())?;
        let mut bytes = vec![0; slots.len() * row_bytes];
        for (&slot, out) in slots.iter().zip(bytes.chunks_exact_mut(row_bytes)) {
            let slot = usize::try_from(slot)?;
            let offset = slot / 64 * page_bytes + plane_offset + slot % 64 * row_bytes;
            commands.library().copy_d2h(out, source.view(offset, row_bytes)?.raw())?;
        }
        self.bytes(name, dtype, shape, &bytes)
    }
}

fn selection_rows(bytes: &[u8], rows: usize, capacity: usize) -> Result<Vec<u8>> {
    ensure!(rows > 0 && rows <= capacity && bytes.len() == 4 * capacity * 16 * 4,
        MiniMaxError::Invalid("diagnostic selection stride"));
    let mut out = Vec::with_capacity(rows * 4 * 16 * 4);
    for row in 0..rows {
        for head in 0..4 {
            let start = (head * capacity + row) * 16 * 4;
            out.extend_from_slice(&bytes[start..start + 16 * 4]);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::execution::RequestView;

    #[test]
    fn minimax_diagnostic_selection_uses_exported_head_stride() -> Result<()> {
        let bytes: Vec<_> = (0..4 * 64 * 16i32).flat_map(i32::to_le_bytes).collect();
        let out = selection_rows(&bytes, 3, 64)?;
        let ids: Vec<_> = out.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
        assert_eq!(ids.len(), 3 * 4 * 16);
        assert_eq!(&ids[16..32], &(1024..1040).collect::<Vec<_>>());
        assert_eq!(&ids[64..80], &(16..32).collect::<Vec<_>>());
        assert!(selection_rows(&bytes, 65, 64).is_err());
        Ok(())
    }

    #[test]
    fn minimax_diagnostic_manifest_is_complete_and_refuses_overwrite() -> Result<()> {
        let temporary = tempfile::tempdir()?;
        let root = temporary.path().join("dump");
        let mut diagnostic = Diagnostics::new(root.clone(), 3)?;
        let mut batch = BatchView { positions:vec![0,1], slots:vec![0,1], pool_pages:1,
            requests:vec![RequestView { row_start:0, rows:2, start_position:0, pages:vec![0] }] };
        let mut step = diagnostic.begin(&batch)?;
        assert!(!root.join("manifest.json").exists());
        step.bytes("input", "f32", &[2,1], &[0;8])?;
        assert!(step.bytes("input", "f32", &[2,1], &[0;8]).is_err());
        assert!(step.bytes("bad", "f32", &[2,1], &[0;7]).is_err());
        diagnostic.finish(step)?;
        let manifest: Value = serde_json::from_slice(&fs::read(root.join("manifest.json"))?)?;
        assert_eq!(manifest["schema"], "cuteafd.minimax.layer-diagnostic/1");
        assert_eq!(manifest["steps"][0]["tensors"]["input"]["shape"], json!([2,1]));
        assert!(Diagnostics::new(root, 3).is_err());
        assert!(diagnostic.begin(&batch).is_err());
        batch.positions = vec![2,3]; batch.requests[0].start_position = 2;
        let step = diagnostic.begin(&batch)?;
        diagnostic.finish(step)?;
        assert_eq!(diagnostic.next_position, 4);
        assert!(extent("f32", &[usize::MAX, 2]).is_err());
        Ok(())
    }
}
