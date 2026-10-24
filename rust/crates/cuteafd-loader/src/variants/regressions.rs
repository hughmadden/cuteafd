use super::*;

#[test]
fn fp8_partial_blocks_use_each_sources_own_metadata() {
    let make = |block: usize, values: &[f32]| {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), serde_json::json!({
            "quantization_config":{"quant_method":"fp8", "weight_block_size":[block,2]}}).to_string()).unwrap();
        write_safetensors(&dir.path().join("model.safetensors"), &[
            ("w.weight", "F8_E4M3", vec![129,2], vec![0x38;129*2]),
            ("w.weight_scale_inv", "F32", vec![values.len(),1], values.iter().flat_map(|v| v.to_le_bytes()).collect()),
        ]);
        dir
    };
    let base = make(128, &[1.0,7.0]);
    let variant = make(64, &[2.0,3.0,4.0]);
    for checkpoint in [false,true] {
        let open = |path: &Path| if checkpoint {
            TensorSource::from_checkpoint(&crate::plan::checkpoint::Checkpoint::open_for_role(path,"test", |_| true).unwrap())
        } else { TensorSource::from_files([path.join("model.safetensors")]).unwrap() };
        let b = open(base.path()).matrix("w.weight").unwrap();
        let v = open(variant.path()).matrix("w.weight").unwrap();
        assert_eq!((b.data[64*2],b.data[127*2],b.data[128*2]), (1.0,1.0,7.0));
        assert_eq!((v.data[63*2],v.data[64*2],v.data[127*2],v.data[128*2]), (2.0,3.0,3.0,4.0));
    }
}

#[test]
fn fp8_exl3_wrapper_uses_original_quantization_config_for_dense_blocks() {
    // Metadata shape in the recovered alternate weights revision: routed experts are EXL3,
    // while retained dense FP8 operands keep the original HF block declaration.
    let dir = tempfile::tempdir().unwrap();
    let quantization = serde_json::json!({"quant_method":"exl3", "bits":4,
        "weight_block_size":[7,7],
        "original_quantization_config":{"quant_method":"fp8", "fmt":"e4m3", "weight_block_size":[128,128]}});
    std::fs::write(dir.path().join("config.json"),serde_json::json!({"quantization_config":quantization}).to_string()).unwrap();
    write_safetensors(&dir.path().join("model.safetensors"),&[
        ("w.weight","F8_E4M3",vec![129,128],vec![0x38;129*128]),
        ("w.weight_scale_inv","F32",vec![2,1],[2f32,5.0].into_iter().flat_map(f32::to_le_bytes).collect()),
    ]);
    for checkpoint in [false,true] {
        let source = if checkpoint {
            TensorSource::from_checkpoint(&crate::plan::checkpoint::Checkpoint::open_for_role(dir.path(),"test", |_| true).unwrap())
        } else { TensorSource::from_files([dir.path().join("model.safetensors")]).unwrap() };
        let matrix = source.matrix("w.weight").unwrap();
        assert_eq!((matrix.data[127*128],matrix.data[128*128]),(2.0,5.0));
    }
}

#[test]
fn fp8_external_declaration_is_not_masked_by_exl3_config_and_conflicts_reject() {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("config.json");
    std::fs::write(&config,r#"{"quantization_config":{"quant_method":"exl3","bits":4}}"#).unwrap();
    std::fs::write(dir.path().join("quantize_config.json"),r#"{"quant_method":"exl3","original_quantization_config":{"weight_block_size":[128,2]}}"#).unwrap();
    write_safetensors(&dir.path().join("model.safetensors"),&[
        ("w.weight","F8_E4M3",vec![129,2],vec![0x38;129*2]),
        ("w.weight_scale_inv","F32",vec![2,1],[2f32,5.0].into_iter().flat_map(f32::to_le_bytes).collect()),
    ]);
    let open = || TensorSource::from_checkpoint(&crate::plan::checkpoint::Checkpoint::open_for_role(dir.path(),"test", |_| true).unwrap());
    let matrix = open().matrix("w.weight").unwrap();
    assert_eq!((matrix.data[127*2],matrix.data[128*2]),(2.0,5.0));
    std::fs::write(&config,r#"{"quantization_config":{"weight_block_size":[64,2]}}"#).unwrap();
    assert!(open().matrix("w.weight").unwrap_err().to_string().contains("disagree"));
}

#[test]
fn metadata_free_fully_tiled_fp8_uses_exact_source_header_ratios() {
    let dir = tempfile::tempdir().unwrap();
    // The served Romeo K4 config has this EXL3-only declaration. Its FP32
    // dense scale headers describe full tiles; no engine sidecar is needed.
    std::fs::write(dir.path().join("config.json"),r#"{"quantization_config":{"bits":4,"checkpoint_format":"exl3","format":"exl3","quant_method":"exl3"}}"#).unwrap();
    let file = dir.path().join("model.safetensors");
    for (rows,cols,srows,scols) in [(256,384,2,3),(96,128,3,4)] {
        let scales: Vec<f32> = (1..=srows*scols).map(|n| n as f32).collect();
        write_safetensors(&file,&[("w.weight","F8_E4M3",vec![rows,cols],vec![0x38;rows*cols]),
            ("w.weight_scale_inv","F32",vec![srows,scols],scales.iter().flat_map(|v| v.to_le_bytes()).collect())]);
        let source = TensorSource::from_files([file.clone()]).unwrap();
        let matrix = source.matrix("w.weight").unwrap();
        for r in 0..rows { for c in 0..cols {
            assert_eq!(matrix.data[r*cols+c],scales[(r/(rows/srows))*scols+c/(cols/scols)]);
        }}
    }
}

#[test]
fn fp8_bad_layout_and_short_scales_return_typed_errors() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("model.safetensors");
    let config = dir.path().join("config.json");
    let write = |shape: Vec<usize>, scale_shape: Vec<usize>, scale_type: &str, scales: Vec<u8>| {
        write_safetensors(&file, &[("w.weight","F8_E4M3",shape,vec![0x38;129*2]),
            ("w.weight_scale_inv",scale_type,scale_shape,scales)]);
    };
    std::fs::write(&config, r#"{"quantization_config":{"weight_block_size":[128,2]}}"#).unwrap();
    for (shape,grid,dtype,bytes,expected) in [
        (vec![129,2],vec![2,1],"F32",1f32.to_le_bytes().to_vec(),"requires 8 bytes"),
        (vec![129,2],vec![1,1],"F32",1f32.to_le_bytes().to_vec(),"scale grid"),
        (vec![usize::MAX,2],vec![1,1],"F32",1f32.to_le_bytes().to_vec(),"overflows"),
        (vec![129,2],vec![2,1],"F16",vec![0;4],"scale dtype"),
        (vec![129,2],vec![0,1],"F32",vec![],"nonempty"),
    ] {
        write(shape,grid,dtype,bytes);
        let error = TensorSource::from_files([file.clone()]).unwrap().matrix("w.weight").unwrap_err();
        assert!(error.to_string().contains(expected),"{error:#}");
        assert!(error.downcast_ref::<TensorSourceError>().is_some(),"{error:#}");
    }
    write(vec![129,2],vec![2,1],"F32",[1f32,2.0].into_iter().flat_map(f32::to_le_bytes).collect());
    for metadata in [r#"{}"#,r#"{"quantization_config":{"weight_block_size":[0,128]}}"#,
        r#"{"quantization_config":{"weight_block_size":[128]}}"#] {
        std::fs::write(&config,metadata).unwrap();
        let error = TensorSource::from_files([file.clone()]).unwrap().matrix("w.weight").unwrap_err();
        assert!(matches!(error.downcast_ref::<TensorSourceError>(),Some(TensorSourceError::Fp8Layout { .. })),"{error:#}");
    }
}

fn variant_fixture() -> (tempfile::TempDir, PathBuf, Matrix, Matrix) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.json"),serde_json::json!({"model_type":"glm_moe_dsa",
        "num_hidden_layers":2,"first_k_dense_replace":1,"hidden_size":32}).to_string()).unwrap();
    let mut rng = Rng(41);
    let w = rng.matrix(32,48);
    let v = project(&w,&[rng.unit(32)],&[2.0]);
    let file = dir.path().join("edited.safetensors");
    let bytes = |m: &Matrix| m.data.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_safetensors(&dir.path().join("model.safetensors"),&[
        ("model.layers.1.mlp.shared_experts.down_proj.weight","F32",vec![32,48],bytes(&w))]);
    write_safetensors(&file,&[("model.layers.1.mlp.shared_experts.down_proj.weight","F32",vec![32,48],bytes(&v))]);
    (dir,file,w,v)
}
fn fixture_spec(file: &Path) -> cuteafd_core::variants::VariantSpec {
    cuteafd_core::variants::VariantSpec {name:"alternate".into(),source:file.display().to_string(),base:None,id:None,mode:None}
}

#[test]
fn variant_defaults_to_shared_only_and_combined_policy_has_distinct_namespace() {
    let (dir,file,_,_) = variant_fixture();
    let mut spec = fixture_spec(&file);
    let shared = derive_variant(dir.path(),"m",1,&spec,1).unwrap();
    assert_eq!(shared.mode,"shared-only");
    let p = shared.plan.as_ref().unwrap();
    assert_eq!(p.sites[0].0.kind,SiteKind::SharedExpert);
    assert_eq!(shared.verify_summary()["routed_experts_verified"],false);
    assert_eq!(shared.verify_summary()["full_checkpoint_equivalence"],false);
    spec.mode = Some("ffn".into());
    let combined = derive_variant(dir.path(),"m",1,&spec,1).unwrap();
    assert_eq!(combined.mode,"ffn");
    let cp = combined.plan.unwrap();
    assert_eq!(cp.sites[0].0.kind,SiteKind::Ffn);
    assert_ne!(p.namespace,cp.namespace);
    spec.mode = Some("routed".into());
    assert!(derive_variant(dir.path(),"m",1,&spec,1).unwrap_err().to_string().contains("mode"));
}

#[test]
fn incomplete_variant_is_rejected_atomically_before_deriving_valid_sites() {
    let (dir,file,_,v) = variant_fixture();
    let bytes = || v.data.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_safetensors(&file,&[
        ("model.layers.1.mlp.shared_experts.down_proj.weight","F32",vec![32,48],bytes()),
        ("model.layers.0.self_attn.o_proj.weight","F32",vec![32,48],bytes()),
    ]);
    let o = derive_variant(dir.path(),"m",1,&fixture_spec(&file),1).unwrap();
    assert!(o.plan.is_none());
    assert!(matches!(o.rejection_kind,Some(VariantRejection::Incomplete { .. })));
    assert_eq!(o.missing_in_base,["model.layers.0.self_attn.o_proj.weight"]);
    assert!(o.verdicts.is_empty(),"partial operators must not be accepted");
}

#[test]
fn unsupported_unchanged_and_singular_are_unavailable_but_launch_errors_propagate() {
    let (dir,file,w,_) = variant_fixture();
    let spec = fixture_spec(&file);
    let bytes = |m: &Matrix| m.data.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_safetensors(&file,&[("model.layers.1.mlp.shared_experts.down_proj.weight","F32",vec![32,48],bytes(&w))]);
    assert!(matches!(derive_variant(dir.path(),"m",1,&spec,1).unwrap().rejection_kind,Some(VariantRejection::Unchanged)));
    write_safetensors(&file,&[("model.layers.1.mlp.experts.0.down_proj.weight","F32",vec![32,48],bytes(&w))]);
    assert!(matches!(derive_variant(dir.path(),"m",1,&spec,1).unwrap().rejection_kind,Some(VariantRejection::NoSupportedSites)));
    let mut nonzero = w.clone(); nonzero.data.fill(1.0);
    let mut zero = w.clone(); zero.data.fill(0.0);
    write_safetensors(&dir.path().join("model.safetensors"),&[("model.layers.1.mlp.shared_experts.down_proj.weight","F32",vec![32,48],bytes(&zero))]);
    write_safetensors(&file,&[("model.layers.1.mlp.shared_experts.down_proj.weight","F32",vec![32,48],bytes(&nonzero))]);
    let o = derive_variant(dir.path(),"m",1,&spec,1).unwrap();
    assert!(o.plan.is_none());
    assert!(o.rejection.unwrap().contains("zero base"));
    // Nonzero base, but delta lives in a row outside its column span: U^T W
    // is exactly singular, so an output-side operator cannot reproduce it.
    let mut rank_deficient = zero.clone(); rank_deficient.data[..48].fill(1.0);
    let mut outside_span = rank_deficient.clone(); outside_span.data[48..96].fill(1.0);
    write_safetensors(&dir.path().join("model.safetensors"),&[("model.layers.1.mlp.shared_experts.down_proj.weight","F32",vec![32,48],bytes(&rank_deficient))]);
    write_safetensors(&file,&[("model.layers.1.mlp.shared_experts.down_proj.weight","F32",vec![32,48],bytes(&outside_span))]);
    let o = derive_variant(dir.path(),"m",1,&spec,1).unwrap();
    assert!(o.plan.is_none());
    assert!(o.rejection.unwrap().contains("singular"));
    let mut bad = spec.clone(); bad.source = dir.path().join("missing").display().to_string();
    assert!(derive_variant(dir.path(),"m",1,&bad,1).is_err());
    std::fs::write(dir.path().join("config.json"),"{").unwrap();
    assert!(derive_variant(dir.path(),"m",1,&spec,1).is_err());
}

#[test]
fn alternate_base_needs_compatibility_with_served_site_tensors() {
    let (dir,file,_,_) = variant_fixture();
    let (other,_,_,_) = variant_fixture();
    let mut spec = fixture_spec(&file); spec.base = Some(other.path().display().to_string());
    let o = derive_variant(dir.path(),"m",1,&spec,1).unwrap();
    assert!(o.plan.is_none());
    assert!(matches!(o.rejection_kind,Some(VariantRejection::UnverifiedBase { .. })));
    spec.base = Some(dir.path().display().to_string());
    assert!(derive_variant(dir.path(),"m",1,&spec,1).unwrap().plan.is_some());
}

#[test]
fn bare_bf16_variant_extract_needs_no_engine_sidecar_or_source_config() {
    let (dir,_,_,_) = variant_fixture();
    let extract = tempfile::tempdir().unwrap();
    let name = "model.layers.1.mlp.shared_experts.down_proj.weight";
    let base: Vec<u16> = (0..32*48).map(|i| cuteafd_core::variants::bf16_rne(if i/48 == i%48 {1.0} else {0.0})).collect();
    let mut edited = base.clone();
    edited[0] = cuteafd_core::variants::bf16_rne(-1.0);
    let bytes = |v: &[u16]| v.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_safetensors(&dir.path().join("model.safetensors"), &[(name,"BF16",vec![32,48],bytes(&base))]);
    let file = extract.path().join("model.safetensors");
    write_safetensors(&file, &[(name,"BF16",vec![32,48],bytes(&edited))]);
    for path in [file.as_path(), extract.path()] {
        let result = derive_variant(dir.path(),"m",1,&fixture_spec(path),1).unwrap();
        assert!(result.plan.is_some(), "{:?}", result.rejection);
        assert_eq!(result.mode,"shared-only");
    }
}
