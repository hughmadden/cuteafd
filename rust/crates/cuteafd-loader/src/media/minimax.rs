//! Bundled MiniMax processors at release revision 54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e.
//! The executable processors use smart resize, not the legacy grid-pinpoint fields.
use super::{MediaError, ProcessorConfig, Result, SpanExpander};
use std::path::Path;

pub const IMAGE_TOKEN: u32 = 200_025;
pub const VIDEO_TOKEN: u32 = 200_026;
/// Token capacity of the loaded tower (16,384 patches of scratch).
pub const ENCODER_TOKENS: usize = 4096;
pub const PROCESSOR_SOURCES: [(&str, &str); 3] = [
    ("image_processor.py", "8218f98634c365608a98ea887bfebac37131ec2acdb8445188350490de7cf994"),
    ("video_processor.py", "ced0f8d85d73f8722fecb26033d2d91ecf4a4e5112c91ce1b1522ca7edd0943d"),
    ("processing_minimax.py", "0706ae8b93b3489a3b3bc7f48a5f29fb34b1449ebebb37586250e4d5fa8f5e35"),
];

pub(super) fn validate_config(config: &ProcessorConfig, value: &serde_json::Value) -> Result<()> {
    if (config.patch, config.merge, config.temporal) != (14, 2, 2) {
        return Err(MediaError::Config("MiniMax processor requires patch/merge/temporal 14/2/2".into()));
    }
    for (key, expected) in [("spatial_merge_size", 2), ("temporal_patch_size", 2)] {
        if value["img_token_compression_config"].get(key).is_some_and(|v| v.as_u64() != Some(expected)) {
            return Err(MediaError::Config(format!("MiniMax {key} differs from bundled processor")));
        }
    }
    Ok(())
}

/// Resolve image delimiters from the bundled tokenizer, as MiniMaxVLProcessor does.
pub fn span_expander(snapshot: &Path, vocabulary: u32) -> Result<SpanExpander> {
    let bytes = std::fs::read(snapshot.join("tokenizer_config.json")).map_err(|e| MediaError::Config(e.to_string()))?;
    let config: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| MediaError::Config(e.to_string()))?;
    let entries = config["added_tokens_decoder"].as_object().ok_or_else(|| MediaError::Config("tokenizer lacks added_tokens_decoder".into()))?;
    let id = |token: &str| -> Result<u32> {
        entries.iter().find(|(_, value)| value["content"] == token)
            .and_then(|(key, _)| key.parse().ok()).ok_or_else(|| MediaError::Config(format!("tokenizer lacks {token}")))
    };
    let placeholder = id("]<]image[>[")?;
    if placeholder != IMAGE_TOKEN { return Err(MediaError::Config("MiniMax image token differs from release".into())); }
    SpanExpander::from_config(&serde_json::json!({
        "image_token_id": placeholder,
        "vision_start_token_id": id("]<]start of image[>[")?,
        "vision_end_token_id": id("]<]end of image[>[")?,
    }), vocabulary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::{EncoderId, ImageFamily, ImageProcessor, RgbImage};
    use sha2::{Digest, Sha256};

    #[test]
    fn release_pinpoint_shapes_follow_smart_resize_without_tiling() {
        let processor = ProcessorConfig::for_family(ImageFamily::Minimax);
        let expected = [
            [(336,336),(336,672),(336,1008),(336,1344),(280,1484),(252,1624)],
            [(672,336),(672,672),(532,812),(448,924),(420,1036),(364,1148)],
            [(1008,336),(812,532),(672,672),(560,756),(504,840),(448,924)],
            [(1344,336),(924,448),(756,560),(672,672),(588,728),(532,812)],
            [(1484,280),(1036,420),(840,504),(728,588),(672,672),(588,728)],
            [(1624,252),(1148,364),(924,448),(812,532),(728,588),(672,672)],
        ];
        for h in 1..=6 { for w in 1..=6 {
            let (height, width) = processor.resize_shape(h * 336, w * 336).unwrap();
            assert_eq!((height,width),expected[h as usize-1][w as usize-1]);
            assert_eq!((height % 28, width % 28), (0, 0));
            assert!(height * width <= 451_584);
        }}
        for (h, w, expected) in [(672,672,(672,672)), (336,672,(336,672)),
            (1008,1008,(672,672)), (336,2016,(252,1624)), (1,1,(56,56)), (1,200,(28,196))] {
            assert_eq!(processor.resize_shape(h,w).unwrap(), expected);
        }
        assert!(processor.resize_shape(1,201).is_err());
    }

    #[test]
    fn merge_rows_repeat_temporal_samples_and_use_fused_normalization() {
        let config = ProcessorConfig::for_family(ImageFamily::Minimax);
        let mut rgb = RgbImage { width: 56, height: 56, data: vec![0; 56 * 56 * 3] };
        for y in 0..56 { for x in 0..56 { for c in 0..3 {
            rgb.data[(y*56+x)*3+c] = (y/14*4+x/14+c*30) as u8;
        }}}
        let prepared = config.prepare_rgb(rgb, EncoderId([0;32])).unwrap();
        assert_eq!(prepared.tokens, 4);
        let patches = prepared.patches(&config).unwrap();
        let lut = config.normalization_lut();
        for (row, patch) in [0usize,1,4,5,2,3,6,7,8,9,12,13,10,11,14,15].into_iter().enumerate() {
            for c in 0..3 { for t in 0..2 {
                let start = row*1176+c*392+t*196;
                assert!(patches[start..start+196].iter().all(|v| *v == lut[c][patch+c*30]));
            }}
        }
        assert_eq!(lut[0][17], (17.0f32 - 0.48145466f32*255.0)/(0.26862954f32*255.0));
    }

    #[test]
    fn tokenizer_delimiters_and_index_alias_expand_image_rows() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tokenizer_config.json"), serde_json::to_vec(&serde_json::json!({
            "added_tokens_decoder": {
                "200025":{"content":"]<]image[>["}, "200026":{"content":"]<]video[>["},
                "200029":{"content":"]<]start of image[>["}, "200030":{"content":"]<]end of image[>["}
            }
        })).unwrap()).unwrap();
        let image = ProcessorConfig::for_family(ImageFamily::Minimax).prepare_rgb(
            RgbImage {width:56,height:56,data:vec![0;56*56*3]}, EncoderId([0;32])).unwrap();
        let expander = span_expander(dir.path(),200064).unwrap();
        let expanded = expander.expand(&[200029,IMAGE_TOKEN,200030], &[image], 6).unwrap();
        assert_eq!(expanded.tokens,[200029,IMAGE_TOKEN,IMAGE_TOKEN,IMAGE_TOKEN,IMAGE_TOKEN,200030]);
        assert_eq!((expanded.media[0].start,expanded.media[0].len),(1,4));
    }

    #[test]
    fn legacy_tiling_metadata_does_not_override_fast_resize() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("preprocessor_config.json"), r#"{
            "process_image_mode":"dynamic_res","size":[672,672],"patch_size":14,
            "img_token_compression_config":{"spatial_merge_size":2,"temporal_patch_size":2,
                "max_image_resolution":1008,"image_token_compression_threshold":1.1}}
        "#).unwrap();
        let processor = ProcessorConfig::from_snapshot(dir.path(),ImageFamily::Minimax).unwrap();
        assert_eq!(processor.max_pixels,451_584);
        assert_eq!(processor.resize_shape(1008,1008).unwrap(),(672,672));
    }

    /// `processor-parity.py`'s integer pattern, so pinned vendor hashes need no fixture files.
    fn formula(height: u32, width: u32, salt: u64) -> RgbImage {
        let data = (0..u64::from(height)).flat_map(|y| (0..u64::from(width)).flat_map(move |x|
            (0..3u64).map(move |c| ((x * 37 + y * 101 + c * 59 + x * y % 97 + salt * 13) % 256) as u8)))
            .collect();
        RgbImage { width, height, data }
    }
    fn f32_sha(values: &[f32]) -> String {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        format!("{:x}", Sha256::digest(&bytes))
    }

    #[test]
    fn release_processor_patch_hashes_match_the_bundled_python_processor() {
        // python/tools/qualify/minimax_m3/processor-parity.py: the release's own
        // image_processor.py on Transformers 5.14.0, torch 2.14.1 CPU.
        let config = ProcessorConfig::for_family(ImageFamily::Minimax);
        for (h, w, grid, sha) in [
            (1080, 1920, (36, 64), "e5831b6ddc674249ecb85e88ee9aeb30d38a70f2ffde33e66b9349aa9c64e95a"),
            (333, 777, (24, 56), "28c600049fa12945370205c806f7a168ffc381b6ef13e304b680b34e247fdb10"),
            (100, 37, (8, 2), "0142e2010f5c0de5367be52218d740e442397f58bf74b7e9802958f77eab9a90"),
        ] {
            let image = config.prepare_rgb(formula(h, w, 0), EncoderId([0; 32])).unwrap();
            assert_eq!((image.grid.t, image.grid.h, image.grid.w), (1, grid.0, grid.1), "{h}x{w}");
            assert_eq!(f32_sha(&image.patches(&config).unwrap()), sha, "{h}x{w}");
        }
    }

    /// Full comparison against `processor-parity.py --out DIR` (its images), bit for bit.
    #[test]
    #[ignore = "set CUTEAFD_MINIMAX_PROCESSOR_PARITY to a processor-parity.py output directory"]
    fn minimax_vendor_parity() {
        let dir = std::path::PathBuf::from(std::env::var("CUTEAFD_MINIMAX_PROCESSOR_PARITY").unwrap());
        let cases: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("cases.json")).unwrap()).unwrap();
        let config = ProcessorConfig::from_snapshot(&dir, ImageFamily::Minimax).unwrap();
        let read_png = |name: &str| super::super::decode(&std::fs::read(dir.join(name)).unwrap(), config.decode).unwrap();
        let expect = |name: &str, actual: &[f32]| {
            let bytes = std::fs::read(dir.join(format!("{name}.f32"))).unwrap();
            let vendor: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            assert_eq!(vendor.len(), actual.len(), "{name}: patch count");
            let differing = vendor.iter().zip(actual).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            let worst = vendor.iter().zip(actual).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert_eq!(differing, 0, "{name}: {differing} of {} values differ, max |diff| {worst}", vendor.len());
        };
        let grid = |value: &serde_json::Value| -> Vec<u32> {
            value.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect()
        };
        for case in cases["images"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let image = config.prepare_rgb(read_png(&format!("{name}.png")), EncoderId([0; 32])).unwrap();
            assert_eq!(vec![image.grid.t, image.grid.h, image.grid.w], grid(&case["grid"]), "{name}");
            assert_eq!(image.tokens as u64, case["tokens"].as_u64().unwrap(), "{name}");
            expect(name, &image.patches(&config).unwrap());
        }
    }
}
