//! Qwen interleaved M-RoPE positions. Logical row ids remain authoritative for
//! paging and causality; these coordinates are read only at rotary sites.
use thiserror::Error;

/// An expanded image's first placeholder row and unmerged patch grid (T,H,W).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSpan {
    pub start: usize,
    pub grid: [usize; 3],
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RopeError {
    #[error("Qwen M-RoPE requires a positive spatial merge size")]
    MergeSize,
    #[error("image {0} needs a nonempty, merge-aligned still-image grid (T=1)")]
    Grid(usize),
    #[error("image {0} spans overlap, are out of order, or exceed the native prompt")]
    Span(usize),
    #[error("image placeholders and span grids disagree at native row {0}")]
    Placeholder(usize),
    #[error("Qwen rotary position exceeds the i32 table range")]
    PositionOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Span {
    start: usize,
    end: usize,
    base: i64,
    width: usize,
    delta_after: i64,
}

/// Immutable request metadata, recomputed from native ids and image grids on
/// restore, not stored in a prefix mark. Empty metadata is the text-only path.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RopePositions {
    spans: Vec<Span>,
}

impl RopePositions {
    pub fn new(native_ids: &[u32], images: &[ImageSpan], image_token: u32, merge: usize)
        -> Result<Self, RopeError> {
        if merge == 0 {
            return Err(RopeError::MergeSize);
        }
        let mut spans = Vec::with_capacity(images.len());
        let (mut end, mut delta) = (0, 0i64);
        for (i, image) in images.iter().enumerate() {
            let [t, h, w] = image.grid;
            if t != 1 || h == 0 || w == 0 || h % merge != 0 || w % merge != 0 {
                return Err(RopeError::Grid(i));
            }
            let (height, width) = (h / merge, w / merge);
            let len = height.checked_mul(width).ok_or(RopeError::PositionOverflow)?;
            let next = image.start.checked_add(len).ok_or(RopeError::PositionOverflow)?;
            if image.start < end || next > native_ids.len() {
                return Err(RopeError::Span(i));
            }
            if let Some(row) = native_ids[end..image.start].iter().position(|&id| id == image_token) {
                return Err(RopeError::Placeholder(end + row));
            }
            if let Some(row) = native_ids[image.start..next].iter().position(|&id| id != image_token) {
                return Err(RopeError::Placeholder(image.start + row));
            }
            let base = i64::try_from(image.start).map_err(|_| RopeError::PositionOverflow)? + delta;
            let extent = i64::try_from(height.max(width)).map_err(|_| RopeError::PositionOverflow)?;
            i32::try_from(base + extent - 1).map_err(|_| RopeError::PositionOverflow)?;
            delta += extent - i64::try_from(len).map_err(|_| RopeError::PositionOverflow)?;
            spans.push(Span { start: image.start, end: next, base, width, delta_after: delta });
            end = next;
        }
        if let Some(row) = native_ids[end..].iter().position(|&id| id == image_token) {
            return Err(RopeError::Placeholder(end + row));
        }
        let positions = Self { spans };
        if !native_ids.is_empty() {
            positions.at(native_ids.len() - 1)?;
        }
        Ok(positions)
    }

    /// Coordinates of a prompt, decode, verify or MTP row. A block key must
    /// call this at the logical block-start row, even across a chunk boundary.
    pub fn at(&self, row: usize) -> Result<[i32; 3], RopeError> {
        let n = self.spans.partition_point(|span| span.start <= row);
        if let Some(span) = n.checked_sub(1).map(|i| &self.spans[i]) {
            if row < span.end {
                let offset = row - span.start;
                let base = span.base;
                let r = i64::try_from(offset / span.width).map_err(|_| RopeError::PositionOverflow)?;
                let c = i64::try_from(offset % span.width).map_err(|_| RopeError::PositionOverflow)?;
                return Ok([to_i32(base)?, to_i32(base + r)?, to_i32(base + c)?]);
            }
        }
        let delta = n.checked_sub(1).map_or(0, |i| self.spans[i].delta_after);
        let position = i64::try_from(row).map_err(|_| RopeError::PositionOverflow)?
            .checked_add(delta).ok_or(RopeError::PositionOverflow)?;
        Ok([to_i32(position)?; 3])
    }

    pub fn delta(&self) -> i64 {
        self.spans.last().map_or(0, |s| s.delta_after)
    }
}

fn to_i32(position: i64) -> Result<i32, RopeError> {
    i32::try_from(position).map_err(|_| RopeError::PositionOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    const IMAGE: u32 = 248056;

    #[test]
    fn reference_get_rope_index_fifty_layouts() {
        let fixtures: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../../scripts/fixtures/qwen4-rope-positions.json")).unwrap();
        let cases = fixtures["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 50);
        for case in cases {
            let ids: Vec<u32> = case["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            let images: Vec<ImageSpan> = case["images"].as_array().unwrap().iter().map(|v| ImageSpan {
                start: v["start"].as_u64().unwrap() as usize,
                grid: std::array::from_fn(|i| v["grid"][i].as_u64().unwrap() as usize),
            }).collect();
            let positions = RopePositions::new(&ids, &images, IMAGE, 2).unwrap();
            assert_eq!(positions.delta(), case["delta"].as_i64().unwrap());
            for (row, expected) in case["positions"].as_array().unwrap().iter().enumerate() {
                let expected: [i32; 3] = std::array::from_fn(|i| expected[i].as_i64().unwrap() as i32);
                assert_eq!(positions.at(row).unwrap(), expected, "case {} row {row}", case["name"]);
                // Same lookup after arbitrary chunk/prefix boundaries; pooled keys
                // rotate at the group start, not end coordinates minus three.
                assert_eq!(positions.clone().at(row).unwrap(), expected);
            }
            let next = (ids.len() as i64 + positions.delta()) as i32;
            assert_eq!(positions.at(ids.len()).unwrap(), [next; 3]);
            assert_eq!(positions.at(ids.len() + 63).unwrap(), [next + 63; 3]);
        }
    }

    #[test]
    fn text_rows_and_i32_boundary() {
        let positions = RopePositions::default();
        for row in [0, 1, 3, 64, 4096, 1_048_575, i32::MAX as usize] {
            assert_eq!(positions.at(row).unwrap(), [row as i32; 3]);
        }
        assert_eq!(positions.at(i32::MAX as usize + 1), Err(RopeError::PositionOverflow));
        assert_eq!(positions.at(usize::MAX), Err(RopeError::PositionOverflow));
    }

    #[test]
    fn block_start_may_cross_text_image_or_chunk_boundary() {
        let ids = [10, 248053, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, 248054];
        let positions = RopePositions::new(&ids, &[ImageSpan { start: 2, grid: [1, 4, 6] }], IMAGE, 2).unwrap();
        assert_eq!(positions.at(3).unwrap(), [2, 2, 3]);
        assert_eq!(positions.at(0).unwrap(), [0; 3]);
        assert_eq!(positions.at(7).unwrap(), [2, 3, 4]);
        assert_eq!(positions.at(4).unwrap(), [2, 2, 4]);
        assert_eq!(positions.at(8).unwrap(), [5; 3]);
    }

    #[test]
    fn reject_bad_grids_spans_and_non_native_ids() {
        let span = ImageSpan { start: 1, grid: [1, 2, 2] };
        assert_eq!(RopePositions::new(&[10, IMAGE], &[span], IMAGE, 0), Err(RopeError::MergeSize));
        for grid in [[0, 2, 2], [2, 2, 2], [1, 0, 2], [1, 3, 2]] {
            assert_eq!(RopePositions::new(&[10, IMAGE], &[ImageSpan { grid, ..span }], IMAGE, 2), Err(RopeError::Grid(0)));
        }
        assert_eq!(RopePositions::new(&[10], &[span], IMAGE, 2), Err(RopeError::Span(0)));
        assert_eq!(RopePositions::new(&[10, IMAGE], &[span, span], IMAGE, 2), Err(RopeError::Span(1)));
        assert_eq!(RopePositions::new(&[10, 0x8000_1234], &[span], IMAGE, 2), Err(RopeError::Placeholder(1)));
        assert_eq!(RopePositions::new(&[IMAGE], &[], IMAGE, 2), Err(RopeError::Placeholder(0)));
    }
}
