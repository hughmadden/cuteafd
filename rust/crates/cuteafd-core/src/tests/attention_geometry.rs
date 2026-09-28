use super::*;



#[test]
fn attention_plan_rejects_truncated_schedule_and_kernel_incompatible_heads() {
    let mut facts = ModelFacts::default();
    facts.compress_ratios.pop();
    let error = DeepseekV4AttentionPlan::from_model_facts(&facts)
        .unwrap_err()
        .to_string();
    assert!(error.contains("compression schedule has 45 entries"));

    let mut facts = ModelFacts::default();
    facts.kv_heads = 2;
    let plan = DeepseekV4AttentionPlan::from_model_facts(&facts).unwrap();
    let error = plan
        .validate_sparkinfer_sm120_contract()
        .unwrap_err()
        .to_string();
    assert!(error.contains("requires one KV head"));
}

