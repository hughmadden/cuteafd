//! Exact MSA program and backing inventory shared by admission and execution.
//! This describes the one-GPU coordinator; it is not a placement policy.
use super::MiniMaxM3Config;
use crate::placement::{
    inventory::ProgramEntry,
    planner::{fixed_request_tails, Allocation, ByteCurve, Component, Lifetime, Location, Refusal},
};
use cuteafd_core::memory_layout::Category;
use cuteafd_ffi::{
    minimax_m3::{Layout, Mode},
    programs::ProgramSpec,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MsaPrograms {
    pub layouts: [Layout; 4],
}
impl MsaPrograms {
    pub fn resolve(
        cfg: &MiniMaxM3Config,
        manifest: &serde_json::Value,
        decode_rows: u32,
        prefill_rows: u32,
        max_context: u32,
    ) -> Result<(Self, Vec<ProgramSpec>), Refusal> {
        let bad = |message: &str| Refusal::InvalidFact(message.into());
        if (
            cfg.heads,
            cfg.kv_heads,
            cfg.head_dim,
            cfg.msa.index_heads,
            cfg.msa.index_dim,
            cfg.msa.block,
            cfg.msa.topk_blocks,
            cfg.msa.local_blocks,
        ) != (64, 4, 128, 4, 128, 128, 16, 1)
            || cfg.sparse.iter().any(|enabled| !enabled)
        {
            return Err(bad("MiniMax native MSA requires 64Q/4KV/D128, independent 4x128 index heads, top16/block128/local1 on every layer"));
        }
        if max_context == 0
            || max_context as usize > cfg.max_position
            || decode_rows == 0
            || decode_rows > prefill_rows
            || prefill_rows > 4096
        {
            return Err(bad("MiniMax MSA requested row or context capacity"));
        }
        let expected = [
            (Mode::IndexDecode, decode_rows, max_context.div_ceil(64)),
            (Mode::IndexPrefill, prefill_rows, max_context.div_ceil(64)),
            (Mode::AttentionDecode, decode_rows, decode_rows),
            (Mode::AttentionExtend, prefill_rows, decode_rows),
        ];
        let entries = manifest["programs"]
            .as_array()
            .ok_or_else(|| bad("MiniMax PROGRAMS.json has no program list"))?;
        let mut specs = Vec::new();
        let mut layouts = Vec::new();
        for (mode, rows, extent) in expected {
            let refusal = |detail| Refusal::MissingExport {
                export: mode.name().into(),
                geometry: "MiniMax native MSA 64Q/4KV/D128, top16/block128".into(),
                context: u64::from(max_context),
                detail,
            };
            let mut found = entries.iter().filter(|entry| entry["name"] == mode.name());
            let entry = found
                .next()
                .ok_or_else(|| refusal("absent from PROGRAMS.json".into()))?;
            if found.next().is_some() {
                return Err(refusal("duplicate program".into()));
            }
            let parsed = ProgramEntry::from_manifest(entry).map_err(|e| refusal(e.to_string()))?;
            let layout = cuteafd_ffi::minimax_m3::validate_entry(&parsed.spec, entry)
                .map_err(|e| refusal(e.to_string()))?;
            if layout.rows < rows || layout.extent < extent {
                return Err(refusal(format!(
                    "compiled capacities rows={} extent={}, requested rows={rows} extent={extent}",
                    layout.rows, layout.extent
                )));
            }
            // The compiled selection head stride is layout.rows, even for a
            // smaller live row count. Owners use that stride, not the request.
            layouts.push(layout);
            specs.push(parsed.spec);
        }
        if layouts[0].rows != layouts[2].rows || layouts[1].rows != layouts[3].rows {
            return Err(bad(
                "MiniMax index/attention compiled selection strides differ",
            ));
        }
        Ok((
            Self {
                layouts: layouts.try_into().expect("four MSA modes"),
            },
            specs,
        ))
    }

    /// One serial execution owner. Independent lanes/prefill slots call this
    /// separately; scratch is never shared between simultaneously queued owners.
    pub fn workspace(&self, prefill: bool) -> Vec<(&'static str, u64)> {
        let [index, attention] = if prefill {
            [self.layouts[1], self.layouts[3]]
        } else {
            [self.layouts[0], self.layouts[2]]
        };
        let rows = u64::from(index.rows);
        let batch = u64::from(attention.extent);
        let mut buffers = vec![
            ("index.query", rows * 512),
            ("index.scales", rows * 16),
            ("selections", rows * 256),
            ("index.scratch", index.scratch),
            ("attention.query", rows * 16384),
            ("attention.output", rows * 16384),
            ("attention.lse", rows * 256),
            ("attention.scratch", attention.scratch),
            ("attention.k_descale", batch * 16),
            ("attention.v_descale", batch * 16),
            ("attention.query_offsets", (batch + 1) * 4),
        ];
        if prefill {
            // Contiguous scorer reads a 256-row padded K extent, including
            // request-alignment gaps. This is separate from the paged KV pool.
            let tokens = (u64::from(index.extent) * 64).div_ceil(256) * 256;
            buffers.extend([
                ("index.keys", tokens * 128),
                ("index.key_scales", tokens * 4),
                ("index.starts", rows * 4),
                ("index.ends", rows * 4),
            ]);
        }
        buffers
    }

    pub fn workspace_components(
        &self,
        prefix: &str,
        rank: u8,
        prefill: bool,
        lifetime: Lifetime,
        maximum_pages: u64,
    ) -> Result<Vec<Component>, Refusal> {
        components(
            self.workspace(prefill),
            prefix,
            rank,
            lifetime,
            maximum_pages,
            Category::Workspace,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MsaCache {
    pub layers: u32,
    pub pool_pages: u32,
    pub requests: u32,
    pub table_width: u32,
}
impl MsaCache {
    pub fn new(
        layers: u32,
        pool_pages: u32,
        requests: u32,
        max_context: u32,
    ) -> Result<Self, Refusal> {
        if [layers, pool_pages, requests, max_context].contains(&0)
            || [pool_pages, requests, max_context]
                .iter()
                .any(|&n| n > i32::MAX as u32)
        {
            return Err(Refusal::InvalidFact(
                "MiniMax MSA cache needs positive signed-int32 extents".into(),
            ));
        }
        Ok(Self {
            layers,
            pool_pages,
            requests,
            table_width: max_context.div_ceil(64),
        })
    }
    pub fn buffers(self) -> Result<Vec<(String, u64)>, Refusal> {
        let mut out = Vec::new();
        for layer in 0..self.layers {
            for (name, stride) in [("keys", 32768u64), ("values", 32768), ("index", 8448)] {
                let bytes = u64::from(self.pool_pages)
                    .checked_mul(stride)
                    .filter(|&n| n <= isize::MAX as u64)
                    .ok_or(Refusal::Overflow("MiniMax MSA cache"))?;
                out.push((format!("layer.{layer}.{name}"), bytes));
            }
        }
        out.push((
            "page_table".into(),
            u64::from(self.requests) * u64::from(self.table_width) * 4,
        ));
        out.push(("lengths".into(), u64::from(self.requests) * 4));
        Ok(out)
    }
    pub fn components(
        self,
        prefix: &str,
        rank: u8,
        maximum_pages: u64,
    ) -> Result<Vec<Component>, Refusal> {
        components(
            self.buffers()?,
            prefix,
            rank,
            Lifetime::TARGET,
            maximum_pages,
            Category::Kv,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    fn manifest() -> Value {
        serde_json::from_str(include_str!(
            "../../../../cuteafd-ffi/tests/fixtures/minimax_msa_programs.json"
        ))
        .unwrap()
    }
    fn cfg() -> MiniMaxM3Config {
        MiniMaxM3Config::from_hf(
            &serde_json::from_str(include_str!("fixtures/config.json")).unwrap(),
        )
        .unwrap()
    }
    #[test]
    fn executable_inventory_requires_every_mode_and_exact_selection_strides() {
        let manifest = manifest();
        let (programs, specs) = MsaPrograms::resolve(&cfg(), &manifest, 16, 2048, 2049).unwrap();
        assert_eq!(specs.len(), 4);
        assert_eq!(programs.layouts[0].rows, 64);
        assert_eq!(programs.layouts[1].rows, 4096);
        let mut broken = manifest.clone();
        broken["programs"].as_array_mut().unwrap().pop();
        assert!(matches!(
            MsaPrograms::resolve(&cfg(), &broken, 16, 2048, 2049),
            Err(Refusal::MissingExport { .. })
        ));
        let mut broken = manifest.clone();
        broken["programs"]
            .as_array_mut()
            .unwrap()
            .push(manifest["programs"][0].clone());
        assert!(MsaPrograms::resolve(&cfg(), &broken, 16, 2048, 2049).is_err());
        for field in ["shape", "dtype", "role"] {
            let mut broken = manifest.clone();
            broken["programs"][0]["pointers"][0][field] = json!("wrong");
            assert!(MsaPrograms::resolve(&cfg(), &broken, 16, 2048, 2049).is_err());
        }
        assert!(MsaPrograms::resolve(&cfg(), &manifest, 65, 2048, 2049).is_err());
        assert!(MsaPrograms::resolve(&cfg(), &manifest, 16, 4097, 2049).is_err());
        assert!(MsaPrograms::resolve(&cfg(), &manifest, 16, 2048, 1048577).is_err());
        let mut config = cfg();
        config.heads = 32;
        assert!(MsaPrograms::resolve(&config, &manifest, 16, 2048, 2049).is_err());
    }
    #[test]
    fn owners_keep_compiled_strides_padding_and_independent_backing_tails() {
        let (programs, _) = MsaPrograms::resolve(&cfg(), &manifest(), 16, 2048, 2049).unwrap();
        let decode = programs
            .workspace(false)
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();
        assert_eq!(decode["selections"], 64 * 256);
        assert_eq!(decode["index.scratch"], 25202688);
        let prefill = programs
            .workspace(true)
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();
        assert_eq!(prefill["selections"], 4096 * 256);
        assert_eq!(prefill["index.keys"], 1048576 * 128);
        assert_eq!(prefill["index.key_scales"], 1048576 * 4);
        assert_eq!(prefill["index.scratch"], 539197952);
        let components = programs
            .workspace_components("minimax.prefill.0", 0, true, Lifetime::TARGET, 16)
            .unwrap();
        assert!(components.len() > prefill.len());
        let serialized = serde_json::to_value(&components).unwrap();
        assert!(serialized
            .to_string()
            .contains("minimax.prefill.0.index.scratch.tail"));
        assert!(programs
            .workspace_components("", 0, true, Lifetime::TARGET, 16)
            .is_err());
    }
    #[test]
    fn cache_charges_planar_kv_and_packed_index_once_per_layer() {
        for (context, width) in [
            (127, 2),
            (128, 2),
            (129, 3),
            (2048, 32),
            (2049, 33),
            (1048576, 16384),
        ] {
            let cache = MsaCache::new(60, 17, 16, context).unwrap();
            assert_eq!(cache.table_width, width);
            let buffers = cache
                .buffers()
                .unwrap()
                .into_iter()
                .collect::<std::collections::BTreeMap<_, _>>();
            assert_eq!(buffers.len(), 60 * 3 + 2);
            assert_eq!(buffers["layer.59.keys"], 17 * 32768);
            assert_eq!(buffers["layer.0.values"], 17 * 32768);
            assert_eq!(buffers["layer.17.index"], 17 * 8448);
            assert_eq!(buffers["page_table"], 16 * u64::from(width) * 4);
            assert_eq!(buffers["lengths"], 16 * 4);
            assert!(!cache.components("minimax.cache", 0, 17).unwrap().is_empty());
        }
        assert!(MsaCache::new(0, 1, 1, 128).is_err());
        assert!(MsaCache::new(1, u32::MAX, 1, 128).is_err());
    }
}

fn components<S: AsRef<str>>(
    buffers: Vec<(S, u64)>,
    prefix: &str,
    rank: u8,
    lifetime: Lifetime,
    maximum_pages: u64,
    category: Category,
) -> Result<Vec<Component>, Refusal> {
    if prefix.is_empty() || rank > 1 || maximum_pages == 0 {
        return Err(Refusal::InvalidFact(
            "invalid MiniMax MSA owner identity".into(),
        ));
    }
    let mut out = Vec::new();
    for (name, bytes) in buffers {
        if bytes == 0 || bytes > isize::MAX as u64 {
            return Err(Refusal::Overflow("MiniMax MSA owner"));
        }
        let allocation = Allocation::buffer(
            format!("{prefix}.{}", name.as_ref()),
            Location::coordinator(rank),
            bytes,
            lifetime,
            category,
            "MiniMax native MSA",
        );
        out.push(Component::allocation(
            allocation,
            ByteCurve::fixed(bytes, maximum_pages),
            false,
            maximum_pages,
        ));
    }
    fixed_request_tails(&mut out, maximum_pages, &[])?;
    Ok(out)
}
