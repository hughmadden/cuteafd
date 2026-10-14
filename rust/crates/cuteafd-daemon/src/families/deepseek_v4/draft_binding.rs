//! V4 token-only dSpark evidence on the shared resource-priced policy.
use super::{engine::Engine, Loaded};
use crate::shared::{draft::{resource, runtime::SharedPolicy}, spark_intake::SparkLink};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{LayerResource, PolicyGeometry, PolicySeed, ResourceClass};

/// Native V4 kernels read gate/up/down FP4 values and one K32 scale per
/// projection. A 128-aligned shard has no packer padding; the exporter rejects
/// any slice grid extending past that extent, so each projection is read once.
fn native_slice_bytes(hidden: usize, intermediate: usize, tp: usize) -> Result<f64> {
    ensure!(tp > 0 && intermediate % tp == 0, "native V4 intermediate {intermediate} cannot split across TP{tp}");
    let shard = intermediate / tp;
    ensure!(hidden % 128 == 0 && shard > 0 && shard % 128 == 0,
        "native V4 draft pricing needs an aligned installed slice, got hidden {hidden}, intermediate {shard}");
    let elements = hidden.checked_mul(shard).context("native V4 slice extent overflow")?;
    Ok(3. * (elements / 2 + elements / 32) as f64)
}

fn geometry(homes: &[(u8, f64)], experts: usize, topk: usize, rows: usize, width: usize,
    group: impl Fn(usize, usize) -> Result<usize>) -> Result<PolicyGeometry> {
    let groups = (0..=rows).map(|rows| (0..homes.len()).map(|layer|
        u8::try_from(group(layer, rows)?).context("installed V4 expert group exceeds 255"))
        .collect::<Result<Vec<_>>>()).collect::<Result<Vec<_>>>()?;
    Ok(PolicyGeometry {
        layers: homes.iter().enumerate().map(|(layer, &(class, slice_bytes))| LayerResource {
            class: Some(class), slice_bytes, group_rows: groups[0][layer], timed: layer > 0,
        }).collect(),
        group_rows_by_rows: groups, experts: u16::try_from(experts)?, topk: u8::try_from(topk)?,
        max_requests: rows, max_positions: width + resource::COPY_EXTENSION, widths: vec![width],
        classes: vec![ResourceClass::rtx(), ResourceClass::spark()], regimes: 1,
    })
}

pub(super) fn new(engine: &Engine<'_>, loaded: &Loaded, transports: &[SparkLink<'_>]) -> Result<Option<SharedPolicy>> {
    let width = engine.draft_block();
    if width == 0 || engine.skip_routed { return Ok(None); }
    let shape = loaded.catalog.routed_experts();
    ensure!(shape.layers == engine.cfg.n_layers && shape.experts == engine.cfg.n_routed_experts
        && shape.topk == engine.cfg.n_activated_experts, "V4 checkpoint/router geometry differs");
    let local = engine.local_layers();
    let remote = if local < shape.layers {
        transports.first().context("no installed V4 Spark transport")?.transport.installed_grouping()?
    } else { Vec::new() };
    let bytes = |layer: usize, tp: usize| match loaded.catalog.exl3() {
        Some(manifest) => resource::exl3_slice_bytes(manifest, layer, tp),
        None => {
            ensure!(loaded.catalog.fp8().is_none() && loaded.catalog.nvfp4().is_none(),
                "V4 draft pricing needs checkpoint-native FP4 K32 or EXL3 experts");
            native_slice_bytes(shape.hidden, shape.intermediate, tp)
        }
    };
    let homes = (0..shape.layers).map(|layer| {
        let here = layer < local;
        Ok((if here { 0 } else { 1 }, bytes(layer, if here { engine.local_expert_tp() } else { remote.len() })?))
    }).collect::<Result<Vec<_>>>()?;
    let geometry = geometry(&homes, shape.experts, shape.topk, engine.decode_rows, width, |layer, rows| {
        if layer < local { return engine.local_group_rows(rows); }
        let mut groups = remote.iter().filter(|report| report.layers().iter()
            .any(|range| (range.first as usize..range.end as usize).contains(&layer)))
            .map(|report| report.group_rows(rows));
        let group = groups.next().context("V4 installed worker report missing layer")??;
        for other in groups { ensure!(other? == group, "V4 installed workers disagree on layer {layer} grouping"); }
        Ok(group)
    })?;
    let shared = SharedPolicy::with_geometry(geometry, resource::Prior::History, PolicySeed::weak(width))?;
    resource::publish(&shared.policy, 0, 0, None);
    Ok(Some(shared))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_traffic_prices_logical_checkpoint_shards_not_allocations() {
        for (hidden, intermediate) in [(4096, 2048), (7168, 3072)] {
            let full = native_slice_bytes(hidden, intermediate, 1).unwrap();
            for tp in [1, 2, 4] {
                assert_eq!(native_slice_bytes(hidden, intermediate, tp).unwrap(), full / tp as f64);
            }
        }
        assert!(native_slice_bytes(4096, 2048, 3).is_err());
        assert!(native_slice_bytes(4096, 576, 1).is_err());
    }

    #[test]
    fn installed_geometry_keeps_tp_bytes_and_live_grouping() {
        let homes = [(0, 100.), (0, 200.), (1, 300.)];
        let geometry = geometry(&homes, 256, 6, 16, 5, |layer, rows| {
            Ok(if rows > 0 && rows <= 4 { 1 } else if layer < 2 { 8 } else { 16 })
        }).unwrap();
        let shared = SharedPolicy::with_geometry(geometry, resource::Prior::History, PolicySeed::weak(5)).unwrap();
        let geometry = shared.policy.geometry();
        assert!(!geometry.layers[0].timed);
        assert_eq!(geometry.layers[1].slice_bytes, 200.);
        assert_eq!(geometry.layers[2].class, Some(1));
        assert_eq!(geometry.group_rows(2, 4), 1);
        assert_eq!(geometry.group_rows(2, 5), 16);
        assert_eq!(geometry.group_rows(1, 16), 8);
        assert_eq!(geometry.classes, vec![ResourceClass::rtx(), ResourceClass::spark()]);
    }
}
