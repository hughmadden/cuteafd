//! Exact stream-serialized linear scratch from the owned native geometry.
use super::{programs, DenseAxis, DenseEncoding, DensePlan, DenseRecord, DenseSelection};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::native_dense::{DenseGeometry, Encoding};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearScratchLayout {
    pub rotated: usize,
    pub table: usize,
    pub block_scales: usize,
    pub partial: usize,
    pub records: Option<usize>,
    pub acc: usize,
}

impl LinearScratchLayout {
    fn empty(rows: usize) -> Result<Self> {
        ensure!((1..=4096).contains(&rows), "native linear scratch rows outside compiled capacity");
        Ok(Self { rotated: 256, table: 256, block_scales: 256, partial: 256,
            records: None, acc: 256 })
    }

    fn include(&mut self, geometry: DenseGeometry, rows: usize, a8: bool) -> Result<()> {
        let (k, n) = (usize::try_from(geometry.input_features)?, usize::try_from(geometry.output_features)?);
        let bytes = |r: usize, n: usize, width: usize| r.checked_mul(n)
            .and_then(|v| v.checked_mul(width)).context("native linear scratch overflow");
        self.acc = self.acc.max(bytes(rows, n, 4)?);
        if matches!(geometry.encoding, Encoding::Trellis | Encoding::TrellisA8) {
            self.rotated = self.rotated.max(bytes(rows, k, 2)?);
            self.block_scales = self.block_scales.max(bytes(1, k / 128, 2)?);
            if geometry.encoding == Encoding::TrellisA8 {
                if a8 && rows > 64 {
                    let records = usize::try_from(geometry.scratch_bytes(rows.try_into()?)?)?.max(256);
                    self.records = Some(self.records.unwrap_or(0).max(records));
                }
            } else {
                // Wide A16 prefill reduces on chip; the same lane also serves narrow units.
                self.partial = self.partial.max(usize::try_from(
                    geometry.scratch_bytes(rows.min(64).try_into()?)?)?);
            }
        }
        Ok(())
    }

    pub fn from_manifest(manifest: &Value, selection: &DenseSelection, records: &[DenseRecord],
        layers: usize, plan: &DensePlan, rows: usize, a8: bool) -> Result<Self> {
        let selected = programs::select_manifest(manifest, selection, records, layers, plan, a8)?;
        let layout = Self::from_plan(records, plan, rows, a8)?;
        ensure!(layout == Self::from_selected(manifest, &selected, rows, a8)?,
            "native scratch owned geometry differs from selected programs");
        Ok(layout)
    }

    /// Admission without a package still sizes only the actual owned backbone slices.
    pub fn from_plan(records: &[DenseRecord], plan: &DensePlan, rows: usize, a8: bool) -> Result<Self> {
        let mut result = Self::empty(rows)?;
        for slice in &plan.tensors {
            if slice.absorbed.is_some() { continue; }
            let record = records.iter().find(|r| r.tensor == slice.tensor).context("native scratch slice record")?;
            if record.shape.len() != 2 { continue; }
            let encoding = match record.encoding {
                DenseEncoding::Trellis { bits: 4 } => Encoding::Trellis,
                DenseEncoding::Mxfp8 => Encoding::Mxfp8,
                DenseEncoding::Int4G32 => Encoding::Int4G32,
                DenseEncoding::RowFp8 => Encoding::RowFp8,
                DenseEncoding::Bf16 | DenseEncoding::F16 | DenseEncoding::F32 => continue,
                other => anyhow::bail!("unsupported native scratch encoding {other:?}"),
            };
            let (mut n, mut k) = (record.shape[0], record.shape[1]);
            match slice.axis { DenseAxis::Output => n = slice.channels, DenseAxis::Input => k = slice.channels }
            let geometry = DenseGeometry { encoding, input_features: k.try_into()?,
                output_features: n.try_into()?, side: 1, max_rows: 4096 };
            result.include(geometry, rows, a8)?;
            if encoding == Encoding::Trellis && a8 {
                result.include(DenseGeometry { encoding: Encoding::TrellisA8, ..geometry }, rows, a8)?;
            }
        }
        result.bytes()?;
        Ok(result)
    }

    pub fn from_selected(manifest: &Value, selected: &std::collections::BTreeSet<String>,
        rows: usize, a8: bool) -> Result<Self> {
        let mut result = Self::empty(rows)?;
        for name in selected {
            let spec = super::super::workspace::program_spec(manifest, name)?;
            let entry = manifest["programs"].as_array().context("native manifest programs")?.iter()
                .find(|entry| entry["name"] == *name).context("selected native program")?;
            if cuteafd_ffi::native_dense_glue::op_of(entry)?.is_some() { continue; }
            result.include(cuteafd_ffi::native_dense::validate_spec(&spec)?, rows, a8)?;
        }
        result.bytes()?;
        Ok(result)
    }

    pub fn bytes(&self) -> Result<usize> {
        [self.rotated, self.table, self.block_scales, self.partial, self.acc]
            .into_iter().chain(self.records).try_fold(0usize, |sum, bytes| {
                ensure!(bytes >= 256, "native linear scratch below allocator minimum");
                sum.checked_add(bytes).context("native linear scratch total overflow")
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partials_use_actual_pairs_and_remain_decode_sized_on_wide_lanes() {
        // Published GLM q_b and FFN down dimensions, not their unrelated maxima.
        let geometry = |k, n| DenseGeometry { encoding: Encoding::Trellis,
            input_features: k, output_features: n, side: 1, max_rows: 4096 };
        let q = geometry(2048, 16384);
        let down = geometry(12288, 6144);
        for rows in [1, 64, 65, 4096] {
            let mut layout = LinearScratchLayout::empty(rows).unwrap();
            layout.include(q, rows, false).unwrap();
            layout.include(down, rows, false).unwrap();
            let expected = q.scratch_bytes(rows.min(64) as u32).unwrap()
                .max(down.scratch_bytes(rows.min(64) as u32).unwrap()) as usize;
            assert_eq!(layout.partial, expected);
            assert_eq!(layout.rotated, rows * 12288 * 2);
            assert_eq!(layout.acc, rows * 16384 * 4);
            assert!(layout.records.is_none());
            assert!(layout.partial < rows.min(64) * 12 * 16384 * 4);
        }
    }

    #[test]
    fn a8_records_are_prefill_only_and_scaled_a16_uses_no_partial() {
        for rows in [64, 65, 4096] {
            let geometry = DenseGeometry { encoding: Encoding::TrellisA8,
                input_features: 6144, output_features: 12288, side: 1, max_rows: 4096 };
            let mut layout = LinearScratchLayout::empty(rows).unwrap();
            layout.include(geometry, rows, true).unwrap();
            assert_eq!(layout.records, (rows > 64).then(|| geometry.scratch_bytes(rows as u32).unwrap() as usize));
            let mut scaled = LinearScratchLayout::empty(rows).unwrap();
            scaled.include(DenseGeometry { encoding: Encoding::Mxfp8, ..geometry }, rows, false).unwrap();
            assert_eq!(scaled.partial, 256);
            assert_eq!(scaled.rotated, 256);
            assert_eq!(scaled.acc, rows * 12288 * 4);
        }
    }

    #[test]
    fn invalid_extents_are_refused_before_allocation() {
        assert!(LinearScratchLayout::empty(0).is_err());
        assert!(LinearScratchLayout::empty(4097).is_err());
        let mut layout = LinearScratchLayout::empty(1).unwrap();
        layout.table = 0;
        assert!(layout.bytes().is_err());
        layout.table = usize::MAX;
        assert!(layout.bytes().is_err());
        let mut layout = LinearScratchLayout::empty(4096).unwrap();
        assert!(layout.include(DenseGeometry { encoding: Encoding::Mxfp8,
            input_features: 128, output_features: usize::MAX as u64,
            side: 1, max_rows: 4096 }, 4096, false).is_err());
    }

    #[test]
    #[ignore = "requires published native checkpoint headers and real exported union manifest; CPU only"]
    fn published_owned_geometry_equals_selected_program_scratch() {
        let manifest: Value = serde_json::from_slice(&std::fs::read(
            std::env::var_os("CUTEAFD_EXLR8_DENSE_MANIFEST").expect("union manifest")).unwrap()).unwrap();
        for variable in ["CUTEAFD_EXLR8_GLM_SNAPSHOT", "CUTEAFD_EXLR8_FLASH_SNAPSHOT"] {
            let root = std::path::PathBuf::from(std::env::var_os(variable).expect(variable));
            let config: Value = serde_json::from_slice(&std::fs::read(root.join("config.json")).unwrap()).unwrap();
            let text = config.get("text_config").unwrap_or(&config);
            let layers = text["num_hidden_layers"].as_u64().unwrap() as usize;
            let selection = DenseSelection::from_hf(&config).unwrap();
            let records = DenseRecord::inventory(&root).unwrap();
            let backbone = selection.backbone(&records, layers).unwrap();
            for world in [1, 2] { for rank in 0..world {
                let plan = DensePlan::for_rank_owned(&backbone, world, rank, &Default::default()).unwrap();
                for (rows, a8) in [(64, false), (4096, false), (4096, true)] {
                    // Flash exports A16 only; optional A8 is a pre-existing GLM route.
                    if a8 && variable == "CUTEAFD_EXLR8_FLASH_SNAPSHOT" { continue; }
                    let layout = LinearScratchLayout::from_manifest(&manifest, &selection, &records,
                        layers, &plan, rows, a8).unwrap();
                    eprintln!("{variable} TP{world} rank{rank} rows{rows} a8{a8}: {} B", layout.bytes().unwrap());
                }
            } }
        }
    }
}
