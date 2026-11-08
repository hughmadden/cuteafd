//! GLM launch-count model, deliberately not wired to admission until the memory
//! campaign lands. This first calibration covers EXL3 K4/Core K6 HQA, remote
//! experts, single RTX. Do not silently apply it to local/TP2/head variants.
use crate::placement::graph_nodes::{GraphNodeCounts, GraphNodeError, GraphNodeRate,
    GraphRateProvenance, ORDINARY, DSA_PAGED};
use super::glm_graphs::{GlmGraphKey, GlmGraphPrevious};

const PROVENANCE: GraphRateProvenance = GraphRateProvenance {
    engine_sha: "796b6f7ee18f28af547c4ad88f0e1b2982153477",
    sparkinfer_sha: "851c42833da26711c68868014cad2fc043916416",
    cuda_driver_version: None,
    nvml_driver_version: None,
    evidence: "#1605; mem-campaign-glm5-v4/glm53-hqa-min-glm5-r1/profile/graph-census-captures-raptor.jsonl",
    limitation: "launcher SHA; capture binary attribution unproved; inferred warm-order join; single-rank remote HQA; driver unrecorded",
};

pub const GLM_SM120_NODE_RATES: [GraphNodeRate; 2] = [
    GraphNodeRate { architecture: "sm120", class: ORDINARY, bytes_per_node: 6680, provenance: PROVENANCE },
    GraphNodeRate { architecture: "sm120", class: DSA_PAGED, bytes_per_node: 2792, provenance: PROVENANCE },
];

/// Launch metadata, not a per-layer capture table. The caller must establish
/// this measured route (including projection/head format) before using it.
#[derive(Debug, Clone, Copy)]
pub struct GlmRemoteHqaNodeClass {
    pub dense: bool,
    pub paged_dsa: bool,
    pub speculator_tap: bool,
}

pub fn glm_remote_hqa_nodes(key: &GlmGraphKey, class: GlmRemoteHqaNodeClass)
    -> Result<GraphNodeCounts, GraphNodeError> {
    if key.rows == 0 || key.rows > 64 || key.table_width == 0 || key.table_stride != key.table_width {
        return Err(GraphNodeError::Nodes("GLM remote HQA rows/width/stride outside measured launch route".into()));
    }
    let mut nodes = GraphNodeCounts::default();
    if key.head {
        nodes.add(ORDINARY, 7)?;
        return Ok(nodes);
    }
    let previous = match (class.dense, key.previous) {
        (true, GlmGraphPrevious::First) => 1i64,
        (true, GlmGraphPrevious::Delta) => 0,
        (false, GlmGraphPrevious::Delta) => -1,
        (false, GlmGraphPrevious::Planes(4)) => 0,
        _ => return Err(GraphNodeError::Nodes("GLM residual route needs a node-count derivation (local/TP2 is not the archived remote route)".into())),
    };
    if class.dense && !class.paged_dsa {
        return Err(GraphNodeError::Nodes("GLM dense non-DSA route is not calibrated".into()));
    }
    let base = if class.dense { 35 } else if class.paged_dsa { 28 } else { 24 };
    let ordinary = base + previous + i64::from(class.speculator_tap) + 2 * i64::from(key.rows > 16);
    nodes.add(ORDINARY, ordinary as u64)?;
    if class.paged_dsa {
        let repeats = u64::try_from(key.table_width.div_ceil(512))
            .ok().and_then(|v| v.checked_mul(2)).ok_or(GraphNodeError::Overflow("DSA launches"))?;
        nodes.add(DSA_PAGED, repeats)?;
    }
    Ok(nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::{attention::AttentionPlacement, graph_nodes::charge};
    use crate::serving_capacity::glm_graphs::{glm_declared_graphs, glm_graph_shapes, GlmGraphLayer};
    use cuteafd_core::memory_policy::GraphNodePadding;

    #[test]
    fn declared_keys_reproduce_archived_node_totals_and_context_subsets() {
        let layers: Vec<_> = (0..78).map(|layer| GlmGraphLayer { dense: layer < 3,
            local: false, attention_owner: None }).collect();
        for (context, executables, ordinary, dsa, bytes) in [
            (131072, 15168, 392064, 12096, 2652897280),
            (262144, 15721, 406306, 14448, 2755657728),
            (1048576, 16827, 434790, 28560, 2984247296),
        ] {
            let shapes = glm_graph_shapes(context, 1048576 / 64);
            let declared = glm_declared_graphs(&shapes, AttentionPlacement::Heads, 1, &layers, 4).unwrap();
            assert_eq!(declared[0].len(), executables);
            let counts = GraphNodeCounts::declared(&declared[0], |key| glm_remote_hqa_nodes(key,
                GlmRemoteHqaNodeClass { dense: key.layer < 3,
                    paged_dsa: key.layer < 3 || (key.layer >= 6 && (key.layer - 6) % 4 == 0),
                    speculator_tap: [6, 20, 34, 48, 62, 76].contains(&key.layer) })).unwrap();
            assert_eq!(counts.0[ORDINARY], ordinary);
            assert_eq!(counts.0[DSA_PAGED], dsa);
            assert_eq!(charge("glm5", "sm120", &counts, &GLM_SM120_NODE_RATES,
                GraphNodePadding { rtx: 0, spark: 0 }).unwrap().packed_bytes, bytes);
        }
    }

    #[test]
    fn unknown_residual_route_is_not_silently_treated_as_remote() {
        let key = GlmGraphKey { layer: 4, rows: 1, table_width: 16, table_stride: 16,
            previous: GlmGraphPrevious::Planes(0), head: false };
        assert!(glm_remote_hqa_nodes(&key, GlmRemoteHqaNodeClass { dense: false,
            paged_dsa: false, speculator_tap: false }).is_err());
    }
}
