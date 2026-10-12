//! V4.1 CED ownership stays fixed; pool and local experts use shared admission.
use super::super::*;

/// All weights, lanes, graphs and prefix state are charged in the inventory's
/// baseline. Pool units include private tails supplied by the caller as fixed.
pub fn request(
    inventory: Inventory,
    pool: PoolPolicy,
    unit_bytes: Vec<u64>,
    experts: Vec<ExpertCost>,
    workspace: u64,
    onboard: Onboard,
) -> PlacementRequest {
    let gpus = inventory.gpus.len();
    let count = experts.len();
    PlacementRequest {
        inventory,
        attention_placement: Some(AttentionPlacement::Heads),
        context_buffers: ContextBuffers::default(),
        layers_first_gpu: 0,
        pool,
        pool_overhead: unit_bytes,
        fixed: vec![],
        movables: vec![],
        layers: experts
            .into_iter()
            .enumerate()
            .map(|(layer, cost)| LayerDemand {
                kind: AttentionClass::Csa,
                weights: ModeBytes::default(),
                kv_unit: ModeBytes::default().into(),
                fixed_bytes: ModeBytes::default(),
                extra: vec![],
                context_indexer: false,
                colocate: None,
                experts: Some(cost),
                modes: vec![LayerMode::Whole {
                    gpu: u8::from(gpus == 2 && layer >= count / 2),
                    ffn: FfnMode::Owner,
                }],
            })
            .collect(),
        expert_workspace: workspace,
        tp2_workspace: [workspace; 2],
        onboard,
        expert_gpus: gpus,
        policy: LayerPolicy::default(),
        hops: HopSpec::default(),
        executor: super::DEEPSEEK_V41,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ced_ownership_and_tp2_loading_peak_are_preserved() {
        let inventory = Inventory {
            gpus: vec![
                GpuBudget {
                    capacity_bytes: 8 << 30,
                    headroom_bytes: 0,
                    baseline: Baseline::Measured {
                        free_bytes: 8 << 30
                    }
                };
                2
            ],
            spark_ranks: 4,
            peer_access: true,
        };
        let pool = PoolPolicy::resolve(&[8 << 30; 2], 512, Some(512), 512, false);
        let cost = ExpertCost {
            whole: Bytes2 {
                resident: 200 << 20,
                staging: 32 << 20,
            },
            half: [Bytes2 {
                resident: 100 << 20,
                staging: 16 << 20,
            }; 2],
            tp2: true,
            spark_ok: true,
        };
        let placement = solve(&request(
            inventory,
            pool,
            vec![1024; 2],
            vec![cost; 40],
            64 << 20,
            Onboard::Layers(3),
        ))
        .unwrap();
        assert_eq!(placement.onboard_layers, 3);
        for (layer, assignment) in placement.layers.iter().enumerate() {
            assert_eq!(
                assignment.mode,
                LayerMode::Whole {
                    gpu: u8::from(layer >= 20),
                    ffn: FfnMode::Owner
                }
            );
            assert_eq!(
                assignment.experts,
                if layer < 3 {
                    ExpertHome::RtxTp2
                } else {
                    ExpertHome::Spark
                }
            );
        }
        assert_eq!(placement.tp2.unwrap().peak_bytes, [380 << 20; 2]);
    }
}
