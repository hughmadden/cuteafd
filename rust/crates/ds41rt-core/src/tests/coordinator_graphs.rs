use super::*;



#[test]
fn coordinator_graph_bucket_selection_uses_decode_and_prefill_ceilings() {
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(1).unwrap(),
        GraphBucket::decode()
    );
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(2).unwrap(),
        GraphBucket::new(16)
    );
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(17).unwrap(),
        GraphBucket::new(32)
    );
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(512).unwrap(),
        GraphBucket::new(512)
    );
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(513).unwrap(),
        GraphBucket::new(1024)
    );
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(1024).unwrap(),
        GraphBucket::new(1024)
    );
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(1025).unwrap(),
        GraphBucket::new(2048)
    );
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(2048).unwrap(),
        GraphBucket::new(2048)
    );
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(2049).unwrap(),
        GraphBucket::new(4096)
    );
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(16384).unwrap(),
        GraphBucket::new(16384)
    );
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(16385).unwrap(),
        GraphBucket::new(32768)
    );
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(32769).unwrap(),
        GraphBucket::new(65536)
    );
    assert!(matches!(
        coordinator_graph_bucket_for_active_rows(0),
        Err(Ds41rtError::GraphBufferContractInvalid { .. })
    ));
    assert!(matches!(
        coordinator_graph_bucket_for_active_rows(65537),
        Err(Ds41rtError::GraphBufferContractInvalid { .. })
    ));
}

