//! Physical layouts for independently retired V4.1 workspace sets.
use super::WeightArenaLayout;
use anyhow::{ensure, Result};

fn layout(component: &str, planes: impl IntoIterator<Item = (String, usize)>) -> Result<WeightArenaLayout> {
    WeightArenaLayout::new(format!("v41/{component}"), planes.into_iter().map(|(name, bytes)| (name, bytes.max(1))))
}

pub fn hc(rows: usize) -> Result<WeightArenaLayout> {
    ensure!((1..=8256).contains(&rows), "invalid mHC workspace rows");
    layout("hc", [("residual", 40960), ("incoming_pre", 16), ("next_pre", 16),
        ("post", 16), ("comb", 64), ("collapsed", 10240), ("normalized", 10240),
        ("sublayer", 10240), ("output", 40960)].map(|(name, stride)| (name.into(), rows * stride)))
}

pub fn window(rows: usize, unit: usize, layers: &[usize], scratch: usize) -> Result<WeightArenaLayout> {
    ensure!(unit > 0 && unit <= rows && rows <= 8256 && !layers.is_empty(), "invalid window workspace geometry");
    let mut planes = vec![("scratch".into(), scratch), ("alpha".into(), 4),
        ("input".into(), unit * 10240), ("projected".into(), unit * 1024),
        ("normalized".into(), unit * 1024), ("positions".into(), unit * 8),
        ("frequencies".into(), unit * 256)];
    for layer in layers {
        planes.extend([("values", 512), ("scales", 16), ("destinations", 8)]
            .map(|(name, stride)| (format!("layer{layer}.{name}"), rows * stride)));
    }
    layout("window-proposal", planes)
}

pub fn engram(unit: usize, scratch: usize) -> Result<WeightArenaLayout> {
    ensure!((1..=4096).contains(&unit), "invalid Engram workspace unit");
    layout("engram-layer", [("output", unit * 40960), ("residual", unit * 40960),
        ("embeddings", unit * 24 * 512), ("text_mask", unit), ("projected", unit * 51200),
        ("scratch", scratch), ("alpha", 4)].map(|(name, bytes)| (name.into(), bytes)))
}

pub fn projection(rows: usize, k: usize, n: usize, scratch: usize, owned_input: bool) -> Result<WeightArenaLayout> {
    ensure!((1..=4096).contains(&rows) && k > 0 && n > 0, "invalid draft projection geometry");
    let mut planes = vec![("scratch".into(), scratch), ("output".into(), rows * n * 2)];
    if owned_input { planes.push(("input".into(), rows * k * 2)); }
    planes.push(("alpha".into(), 4));
    layout("draft-projection", planes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_components_preserve_planes_and_one_physical_extent() -> Result<()> {
        for rows in [1, 16, 64, 256, 4096] {
            for layout in [hc(rows)?, window(rows, rows, &(0..40).collect::<Vec<_>>(), 12345)?,
                engram(rows, 12345)?, projection(rows, 5120, 512, 12345, true)?,
                projection(rows, 5120, 512, 12345, false)?] {
                let mut planes = layout.views().values().collect::<Vec<_>>();
                planes.sort_by_key(|view| view.offset());
                assert!(planes.iter().all(|view| view.offset() % 256 == 0 && view.bytes() > 0));
                assert!(planes.windows(2).all(|pair| pair[0].offset() + pair[0].bytes() <= pair[1].offset()));
                assert_eq!(planes.last().unwrap().offset() + planes.last().unwrap().bytes(), layout.bytes());
            }
        }
        Ok(())
    }
}
