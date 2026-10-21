use super::*;
use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
use cuteafd_loader::push_load::{builder::{Builder, Destination, Source}, native::NativePlanes};
use crate::shared::memory::weight_arena::{ArenaLayout, ExpectedCoverage, LoadingArena};

#[test]
fn calibrated_vectors_match_mount_operations_and_reject_bad_scalars() -> Result<()> {
    let image = [0.5f32, 0.5, 0.25, 2., 2., 4.].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>();
    let v = nvfp4_vectors(&image, 1)?;
    assert_eq!(v, [vec![1.], vec![1.], vec![0.5], vec![0.25]]);
    for (index, value) in [(0, f32::NAN), (1, 0.75), (2, 0.), (3, f32::INFINITY), (4, 3.), (5, -1.)] {
        let mut bad = image.clone();
        bad[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
        assert!(nvfp4_vectors(&bad, 1).is_err());
    }
    assert!(nvfp4_vectors(&image[..20], 1).is_err());
    Ok(())
}

fn delivered<'a>(device: Device<'a>, image: &[u8], destination_bytes: usize)
    -> Result<DeliveredArena<'a>> {
    let directory = tempfile::tempdir_in(std::env::var("CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR")?)?;
    let path = directory.path().join("source");
    std::fs::write(&path, image)?;
    let mut builder = Builder::new(directory.path(), vec![destination_bytes as u64]);
    builder.contiguous(&Source { path, offset: 0, bytes: image.len() as u64 }, 0, image.len() as u64,
        Destination { rank: 0, offset: 0, bytes: image.len() as u64 })?;
    let plan = builder.finish()?;
    let mut loading = LoadingArena::new(device, ArenaLayout::new([destination_bytes], destination_bytes)?,
        ExpectedCoverage::from_plan(&plan, 0)?)?;
    device.run(|| device.library.copy_h2d(loading.destination(0, image.len())?, image))?;
    loading.copied(0, image.len() as u64)?;
    loading.done()
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn fp8_scale_widening_and_generated_padding_are_exact() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let device = Device { library: &library, id: 0 };
    let halves = [0u16, 0x8000, 0x3f80, 0x7fc1, 0xffff, 1, 0x7f80, 0x0080, 0xbf80];
    let image = halves.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    let arena = delivered(device, &image, halves.len() * 4 + 16)?;
    arena.zero_padding()?;
    widen_scales(&arena, 0..halves.len() * 4, 12)?;
    let ready = arena.initialize(|_| Ok(()))?;
    let mut output = vec![0xa5; halves.len() * 4 + 16];
    library.copy_d2h(&mut output, ready.region(0)?.buffer())?;
    let expected = halves.iter().flat_map(|v| (u32::from(*v) << 16).to_le_bytes())
        .chain([0; 16]).collect::<Vec<_>>();
    assert_eq!(output, expected);
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn exl3_semantics_reject_nonfinite_and_upload_exact_descriptors() -> Result<()> {
    use cuteafd_loader::{V41Exl3Layer, V41Exl3Load, V41Exl3ResidentBuffer};
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let device = Device { library: &library, id: 0 };
    let residency = V41Exl3Residency {
        layer: V41Exl3Layer::Backbone(0), world: 1, rank: 0,
        intermediate_start: 0, experts: 1, intermediate: 128, tiers: vec![4],
        projection_counts: vec![[1; 3]], unit_map: None,
        buffers: vec![
            V41Exl3ResidentBuffer { name: "rotation".into(), bytes: 8, initial_words: vec![] },
            V41Exl3ResidentBuffer { name: "descriptor".into(), bytes: 16, initial_words: vec![1, -1, 128, 0] },
        ],
        loads: vec![V41Exl3Load { tensor: "projection.su".into(), expert: Some(0), bytes: 8,
            destinations: vec![(0, 0)] }],
    };
    let (_, bytes) = residency.device_arena_layout()?;
    let source = [0x3c00u16, 0x0000, 0x8000, 0x7bff].into_iter().flat_map(u16::to_le_bytes).collect::<Vec<_>>();
    let arena = delivered(device, &source, bytes)?;
    let ready = initialize(arena, &[Layer::Exl3 { region: 0, residency: &residency }], None, 0)?;
    let mut actual = vec![0xa5; 272];
    library.copy_d2h(&mut actual, ready.region(0)?.buffer())?;
    assert_eq!(&actual[..8], source);
    assert!(actual[8..256].iter().all(|&v| v == 0));
    assert_eq!(&actual[256..], [1i32, -1, 128, 0].into_iter().flat_map(i32::to_le_bytes).collect::<Vec<_>>());
    for invalid in [0x7c00u16, 0xfc00, 0x7e01] {
        let mut source = source.clone();
        source[..2].copy_from_slice(&invalid.to_le_bytes());
        let arena = delivered(device, &source, bytes)?;
        assert!(initialize(arena, &[Layer::Exl3 { region: 0, residency: &residency }], None, 0).is_err());
    }
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn native_out_of_place_planes_match_mount_reference() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let device = Device { library: &library, id: 0 };
    for nv in [false, true] {
        for width in [384usize, 576] {
            let layout = NativePlanes::new(5120, width, width.div_ceil(128) * 128, 1, nv)?;
            let mut raw = vec![0u8; layout.bytes];
            for plane in 0..4 {
                let (offset, bytes) = layout.expert_plane(plane, 0)?;
                let divisor = if plane % 2 == 0 { 2 } else if nv { 16 } else { 32 };
                let cols = if plane < 2 { 5120 / divisor } else if nv {
                    layout.kernel_intermediate / divisor
                } else { width / divisor };
                let rows = if plane < 2 { 2 * layout.kernel_intermediate } else { 5120 };
                for row in 0..rows {
                    for col in 0..cols {
                        let valid = if plane < 2 { row % layout.kernel_intermediate < width }
                            else { col < width / divisor };
                        if valid { raw[offset + row * cols + col] = ((row * 37 + col * 13 + plane * 71) % 251 + 1) as u8; }
                    }
                }
                assert!(rows * cols <= bytes);
            }
            let arena = delivered(device, &raw, raw.len())?;
            let scratch = Transforms::new(device, layout.scratch_bytes(), layout.scratch_bytes())?;
            let scalars = if nv { [0.5f32, 0.5, 0.25, 2., 2., 4.].into_iter()
                .flat_map(f32::to_le_bytes).collect::<Vec<_>>() } else { vec![] };
            scratch.native(&arena, 0, &layout, &scalars)?;
            let mut actual = vec![0; raw.len()];
            library.copy_d2h(&mut actual, arena.region(0)?)?;
            for plane in 0..4 {
                let (offset, bytes) = layout.expert_plane(plane, 0)?;
                let expected = if nv && plane % 2 == 0 { raw[offset..offset + bytes].to_vec() }
                    else if nv {
                        let (rows, cols) = if plane == 1 { (2 * layout.kernel_intermediate, 320) }
                            else { (5120, layout.kernel_intermediate / 16) };
                        let mut out = vec![0; bytes];
                        for row in 0..rows { for col in 0..cols {
                            let at = (row / 128) * cols * 128 + (col / 4) * 512
                                + (row % 32) * 16 + ((row / 32) % 4) * 4 + col % 4;
                            out[at] = raw[offset + row * cols + col];
                        }}
                        out
                    } else { reference_mxfp4(&raw[offset..offset + bytes], plane, width,
                        layout.kernel_intermediate, bytes) };
                assert_eq!(&actual[offset..offset + bytes], expected, "nv={nv} width={width} plane={plane}");
            }
            drop(scratch);
        }
    }
    Ok(())
}

// Independent forward (row,col)->tile/lane oracle, unlike the native inverse
// mapping from a packed lane index. Every valid raw word writes one final word.
fn reference_mxfp4(raw: &[u8], plane: usize, width: usize, padded: usize, bytes: usize) -> Vec<u8> {
    let scales = plane % 2 == 1;
    let element = if scales { 1 } else { 4 };
    let divisor = if scales { 32 } else { 8 };
    let rows = if plane < 2 { 2 * padded } else { 5120 };
    let k = if plane < 2 { 5120 } else { width };
    let cols = k / divisor;
    let kt = k.div_ceil(128);
    let mut output = vec![0; bytes];
    for row in 0..rows { for col in 0..cols {
        let tile = row / 256 * kt + col / (128 / divisor);
        let local_row = row % 256;
        let local_col = col % (128 / divisor);
        let lane = if scales { local_row * 4 + local_col }
            else { ((local_row % 32) / 8) + ((local_row % 8 * 4 + local_col % 4) << 2)
                + ((local_row / 32) << 7) + ((local_col / 4) << 10) };
        let target = (tile * if scales { 1024 } else { 4096 } + lane) * element;
        let source = (row * cols + col) * element;
        output[target..target + element].copy_from_slice(&raw[source..source + element]);
    }}
    output
}
