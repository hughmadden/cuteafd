//! Same-input norm qualification, independent of the reduction implementation.
use anyhow::{ensure, Result};

pub(super) fn check(actual: &[u8], official: &[u8], fp64: &[u8], width: usize) -> Result<()> {
    ensure!(
        width > 0
            && actual.len() == official.len()
            && actual.len() % (width * 2) == 0
            && fp64.len() == actual.len() * 4
            && !actual.is_empty(),
        "norm oracle extent"
    );
    let decode = |bytes: &[u8]| {
        bytes
            .chunks_exact(2)
            .map(|v| f32::from_bits(u32::from(u16::from_le_bytes([v[0], v[1]])) << 16) as f64)
            .collect::<Vec<_>>()
    };
    let a = decode(actual);
    let b = decode(official);
    let r = fp64
        .chunks_exact(8)
        .map(|v| f64::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    ensure!(
        a.iter().chain(&b).chain(&r).all(|v| v.is_finite()),
        "nonfinite norm oracle"
    );
    let error = |a: &[f64], r: &[f64]| {
        a.iter()
            .zip(r)
            .map(|(a, r)| (a - r).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let native = error(&a, &r);
    let reference = error(&b, &r);
    let flips = actual
        .chunks_exact(2)
        .zip(official.chunks_exact(2))
        .filter_map(|(a, b)| {
            let a = u16::from_le_bytes([a[0], a[1]]);
            let b = u16::from_le_bytes([b[0], b[1]]);
            (a != b).then_some(a.abs_diff(b))
        })
        .collect::<Vec<_>>();
    eprintln!(
        "norm native/official FP64 error ratio={} flips={}/{} max_ulp={}",
        native / reference,
        flips.len(),
        a.len(),
        flips.iter().max().copied().unwrap_or(0)
    );
    ensure!(native <= reference * 1.01, "global norm FP64 accuracy bar");
    let mut worst = 0f64;
    for (row, ((a, b), r)) in a
        .chunks_exact(width)
        .zip(b.chunks_exact(width))
        .zip(r.chunks_exact(width))
        .enumerate()
    {
        let maximum = r.iter().map(|v| v.abs()).fold(0f64, f64::max);
        let bits = (maximum as f32).to_bits() & 0x7f80_0000;
        let ulp = f32::from_bits(bits.max(0x0080_0000)) as f64 / 128.;
        let native = error(a, r);
        let reference = error(b, r);
        let allowed = reference * 1.01 + ulp;
        worst = worst.max(native / allowed);
        ensure!(native <= allowed, "norm row {row} FP64 accuracy bar");
    }
    eprintln!("norm worst row error/allowed={worst}");
    Ok(())
}

#[test]
fn accuracy_checks_global_and_each_row() -> Result<()> {
    let bf16 = |v: &[f32]| {
        v.iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>()
    };
    let fp64 = |v: &[f64]| v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    let reference = fp64(&[1.003, 2.003]);
    assert!(check(&bf16(&[1., 2.]), &bf16(&[1., 2.]), &reference, 1).is_ok());
    assert!(check(&bf16(&[1.0625, 2.]), &bf16(&[1., 2.]), &reference, 1).is_err());
    // A large accurate row must not hide a bad small row in the global norm.
    assert!(check(
        &bf16(&[1.03125, 1000.]),
        &bf16(&[1., 1004.]),
        &fp64(&[1., 1000.]),
        1
    )
    .is_err());
    assert!(check(&[], &[], &[], 1).is_err());
    assert!(check(&bf16(&[1.]), &bf16(&[1.]), &fp64(&[f64::NAN]), 1).is_err());
    Ok(())
}
