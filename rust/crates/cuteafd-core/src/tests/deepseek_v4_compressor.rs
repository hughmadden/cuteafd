use super::*;

fn flash_compressors() -> Vec<DeepseekV4CompressorLayerExecutionPlan> {
    deepseek_v4_compressor_execution_plans(&ModelFacts::default()).unwrap()
}

#[test]
fn flash_layer_plans_match_joint_projection_and_state_geometry() {
    let plans = flash_compressors();
    let sliding = &plans[0];
    assert!(sliding.main.is_none());
    assert!(sliding.indexer.is_none());
    assert_eq!(sliding.joint_projection_width, 0);

    let c4 = plans
        .iter()
        .find(|plan| {
            plan.main
                .as_ref()
                .is_some_and(|spec| spec.compress_ratio == 4)
        })
        .unwrap();
    let main = c4.main.as_ref().unwrap();
    assert_eq!(main.kind, DeepseekV4CompressorKind::Main);
    assert!(main.overlap);
    assert_eq!(main.coefficient, 2);
    assert_eq!(main.projected_width, 1_024);
    assert_eq!(main.state_rows, 16);
    assert_eq!(main.state_width, 1_024);
    assert_eq!(main.paired_state_bytes(), Some(131_072));
    let indexer = c4.indexer.as_ref().unwrap();
    assert_eq!(indexer.projected_width, 256);
    assert_eq!(indexer.paired_state_bytes(), Some(32_768));
    assert_eq!(c4.joint_projection_width, 2_560);
    assert_eq!(c4.paired_state_bytes(), Some(163_840));

    let c128 = plans
        .iter()
        .find(|plan| {
            plan.main
                .as_ref()
                .is_some_and(|spec| spec.compress_ratio == 128)
        })
        .unwrap();
    let main = c128.main.as_ref().unwrap();
    assert!(!main.overlap);
    assert_eq!(main.coefficient, 1);
    assert_eq!(main.projected_width, 512);
    assert_eq!(main.state_rows, 256);
    assert_eq!(main.state_width, 512);
    assert_eq!(c128.joint_projection_width, 1_024);
    assert_eq!(c128.paired_state_bytes(), Some(1_048_576));
}



#[test]
fn c4_prefill_retains_last_complete_group_and_current_remainder() {
    let plans = flash_compressors();
    let c4 = plans.iter().find(|plan| plan.indexer.is_some()).unwrap();
    let prefill = c4.prefill(10).unwrap().unwrap();

    assert_eq!(prefill.complete_groups, 2);
    assert_eq!(prefill.remainder, 2);
    assert!(prefill.reset_state_before_projection);
    assert!(prefill.first_overlap_half_is_inactive);
    assert_eq!(prefill.output_rope_position(0), Some(0));
    assert_eq!(prefill.output_rope_position(1), Some(4));
    assert_eq!(prefill.output_rope_position(2), None);
    assert_eq!(
        prefill.previous_window,
        Some(DeepseekV4CompressorStateFill {
            source_start: 4,
            rows: 4,
            state_row_start: 0,
            ape_row_start: 0,
        })
    );
    assert_eq!(
        prefill.current_window,
        Some(DeepseekV4CompressorStateFill {
            source_start: 8,
            rows: 2,
            state_row_start: 4,
            ape_row_start: 0,
        })
    );
}

#[test]
fn c128_prefill_keeps_only_incomplete_current_group_in_state() {
    let plans = flash_compressors();
    let c128 = plans
        .iter()
        .find(|plan| {
            plan.main
                .as_ref()
                .is_some_and(|spec| spec.compress_ratio == 128)
        })
        .unwrap();
    let prefill = c128.prefill(260).unwrap().unwrap();

    assert_eq!(prefill.complete_groups, 2);
    assert_eq!(prefill.remainder, 4);
    assert_eq!(prefill.previous_window, None);
    assert_eq!(
        prefill.current_window,
        Some(DeepseekV4CompressorStateFill {
            source_start: 256,
            rows: 4,
            state_row_start: 0,
            ape_row_start: 0,
        })
    );
    assert_eq!(prefill.output_rope_position(1), Some(128));
}





