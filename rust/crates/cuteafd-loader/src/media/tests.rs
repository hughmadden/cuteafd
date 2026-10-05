use super::*;
fn rgb(width: u32, height: u32) -> RgbImage {
    RgbImage {
        width,
        height,
        data: (0..width as usize * height as usize * 3)
            .map(|i| (i % 251) as u8)
            .collect(),
    }
}
#[test]
fn keys_cover_pixels_geometry_policy_revision_numerics_and_arch() {
    let config = ProcessorConfig::for_family(ImageFamily::Mimo);
    let headers = std::collections::BTreeMap::from([(
        "visual.w".into(),
        serde_json::json!({"dtype":"BF16","shape":[4,8]}),
    )]);
    let encoder = EncoderId::derive("mimo", "rev", &headers, 1, 120);
    let a = config.prepare_rgb(rgb(64, 64), encoder).unwrap();
    assert_eq!(a.key, config.prepare_rgb(rgb(64, 64), encoder).unwrap().key);
    for other in [
        EncoderId::derive("mimo", "other", &headers, 1, 120),
        EncoderId::derive("mimo", "rev", &headers, 2, 120),
        EncoderId::derive("mimo", "rev", &headers, 1, 121),
    ] {
        assert_ne!(a.key, config.prepare_rgb(rgb(64, 64), other).unwrap().key);
    }
    let mut changed = rgb(64, 64);
    changed.data[0] ^= 1;
    assert_ne!(a.key, config.prepare_rgb(changed, encoder).unwrap().key);
    assert_ne!(
        a.key,
        config
            .with_detail(true)
            .prepare_rgb(rgb(64, 64), encoder)
            .unwrap()
            .key
    );
    assert_ne!(
        config.id(),
        ProcessorConfig {
            decode: DecodePolicy {
                exif_transpose: false,
                ..Default::default()
            },
            ..config.clone()
        }
        .id()
    );
    assert_ne!(
        a.key,
        config.prepare_rgb(rgb(32, 128), encoder).unwrap().key
    );
}
#[test]
fn official_budgets_caps_and_patch_layout() {
    for family in [ImageFamily::Mimo, ImageFamily::Qwen, ImageFamily::GlmFlash] {
        let config = ProcessorConfig::for_family(family);
        let image = config
            .prepare_rgb(rgb(6000, 4000), EncoderId([0; 32]))
            .unwrap();
        assert!(image.tokens <= 4096);
        let low = config
            .with_detail(true)
            .prepare_rgb(rgb(1000, 600), EncoderId([0; 32]))
            .unwrap();
        assert!(low.tokens <= 256);
        let patches = low.patches(&config).unwrap();
        assert_eq!(
            patches.len(),
            low.grid.h as usize * low.grid.w as usize * 3 * 2 * config.patch.pow(2) as usize
        );
        let p2 = config.patch.pow(2) as usize;
        assert_eq!(&patches[..p2], &patches[p2..2 * p2]);
    }
    assert!(ProcessorConfig::for_family(ImageFamily::Qwen)
        .resize_shape(1, 201)
        .is_err());
    assert!(ProcessorConfig::for_family(ImageFamily::Mimo)
        .resize_shape(0, 1)
        .is_err());
}
#[test]
fn snapshot_processor_overrides_defaults_and_rejects_bad_config() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"processor_config":{"min_pixels":8192}}"#,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("preprocessor_config.json"),
        r#"{"min_pixels":3136,"max_pixels":12845056}"#,
    )
    .unwrap();
    let config = ProcessorConfig::from_snapshot(dir.path(), ImageFamily::Mimo).unwrap();
    assert_eq!(config.min_pixels, 3136);
    std::fs::write(
        dir.path().join("preprocessor_config.json"),
        r#"{"patch_size":0}"#,
    )
    .unwrap();
    assert!(ProcessorConfig::from_snapshot(dir.path(), ImageFamily::Mimo).is_err());
}
#[test]
fn span_expansion_rejects_literal_markers_and_preserves_native_ids() {
    let config = serde_json::json!({"image_token_id":12,"vision_start_token_id":11,"vision_end_token_id":13});
    let expander = SpanExpander::from_config(&config, 100).unwrap();
    let image = ProcessorConfig::for_family(ImageFamily::Mimo)
        .prepare_rgb(rgb(64, 64), EncoderId([0; 32]))
        .unwrap();
    let expanded = expander
        .expand(&[1, 11, 12, 13, 2], &[image.clone()], 128)
        .unwrap();
    assert_eq!(expanded.tokens, [1, 11, 12, 12, 12, 12, 13, 2]);
    assert_eq!(
        expanded.media,
        [MediaSpan {
            start: 2,
            len: 4,
            key: image.key
        }]
    );
    assert!(expander.expand(&[1, 12, 2], &[image.clone()], 128).is_err());
    assert!(expander.expand(&[11, 12, 13], &[], 128).is_err());
    assert!(expander.expand(&[11, 12, 13], &[image.clone()], 5).is_err());
    assert!(expander.expand(&[101, 11, 12, 13], &[image], 128).is_err());
    assert!(SpanExpander {
        placeholder: 101,
        ..expander
    }
    .expand(&[], &[], 128)
    .is_err());
    assert!(SpanExpander {
        start: 12,
        ..expander
    }
    .expand(&[], &[], 128)
    .is_err());
}

#[test]
fn decode_policy_and_header_admission() {
    let rgba = image::RgbaImage::from_raw(2, 1, vec![10, 20, 30, 0, 100, 150, 200, 128]).unwrap();
    let mut encoded = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(rgba)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .unwrap();
    let encoded = encoded.into_inner();
    assert_eq!(
        decode(&encoded, DecodePolicy::default()).unwrap().data,
        [10, 20, 30, 100, 150, 200]
    );
    assert_eq!(
        decode(
            &encoded,
            DecodePolicy {
                alpha: AlphaPolicy::White,
                ..Default::default()
            }
        )
        .unwrap()
        .data,
        [255, 255, 255, 177, 202, 227]
    );
    assert!(decode_bounded(&encoded, DecodePolicy::default(), 5).is_err());
    assert!(decode_bounded(&encoded, DecodePolicy::default(), 6).is_ok());
}
