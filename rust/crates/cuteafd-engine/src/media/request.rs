use super::{keys::validate_spans, EmbeddingLease, MediaKey, MediaError, MediaSpan};

/// Lazy per-span BF16 features. Prefix restores need only the span descriptors.
#[derive(Clone, Debug)]
pub struct RequestMedia {
    spans: Vec<MediaSpan>,
    features: Vec<Option<EmbeddingLease>>,
    offsets: Vec<usize>,
    feature_rows: Vec<usize>,
    row_bytes: usize,
    prompt_len: usize,
}
#[derive(Debug, Default, PartialEq, Eq)]
pub struct MediaChunk {
    /// Chunk-relative rows overwritten by the injection kernel, in packed-feature order.
    pub indices: Vec<u32>,
    pub features: Vec<u8>,
    pub mask: Vec<u8>,
}
impl RequestMedia {
    pub fn new(
        spans: Vec<MediaSpan>,
        hidden_width: usize,
        prompt_len: usize,
    ) -> Result<Self, MediaError> {
        let offsets = vec![0; spans.len()];
        let feature_rows = spans.iter().map(|span| span.len).collect();
        Self::new_with_layout(spans, hidden_width, prompt_len, offsets, feature_rows)
    }
    pub fn new_with_layout(spans: Vec<MediaSpan>, hidden_width: usize, prompt_len: usize,
        offsets: Vec<usize>, feature_rows: Vec<usize>) -> Result<Self, MediaError> {
        validate_spans(&spans)?;
        if offsets.len() != spans.len() || feature_rows.len() != spans.len() {
            return Err(MediaError::Spans);
        }
        let row_bytes = hidden_width
            .checked_mul(2)
            .filter(|&n| n > 0)
            .ok_or(MediaError::Features)?;
        for (index, span) in spans.iter().enumerate() {
            if span.checked_end().unwrap() > prompt_len
                || feature_rows[index].checked_mul(row_bytes).is_none()
                || offsets[index].checked_add(span.len).is_none_or(|end| end > feature_rows[index])
                || spans.iter().enumerate().any(|(i, s)| s.key == span.key && feature_rows[i] != feature_rows[index])
            {
                return Err(MediaError::Spans);
            }
        }
        let features = vec![None; spans.len()];
        Ok(Self {
            spans,
            features,
            offsets,
            feature_rows,
            row_bytes,
            prompt_len,
        })
    }
    pub fn spans(&self) -> &[MediaSpan] {
        &self.spans
    }
    pub fn feature_rows(&self, index: usize) -> usize { self.feature_rows[index] }
    pub fn span_features(&self, index: usize) -> Option<&std::sync::Arc<[u8]>> {
        self.features.get(index)?.as_ref()?.features()
    }
    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }
    pub fn needed(&self, resume: usize, end: usize) -> impl Iterator<Item = &MediaSpan> {
        self.spans
            .iter()
            .filter(move |s| s.start < end && s.checked_end().unwrap() > resume)
    }
    pub fn ready(&self, resume: usize, end: usize) -> bool {
        self.spans.iter().zip(&self.features).all(|(s, f)| {
            s.start >= end
                || s.checked_end().unwrap() <= resume
                || f.as_ref().is_some_and(|f| f.features().is_some())
        })
    }
    pub fn attach(&mut self, lease: EmbeddingLease) -> Result<(), MediaError> {
        let rows = lease.features().ok_or(MediaError::NotReady(lease.key()))?;
        let mut found = false;
        for ((span, slot), &rows_count) in self.spans.iter().zip(&mut self.features).zip(&self.feature_rows) {
            if span.key == lease.key() {
                if rows.len() != rows_count * self.row_bytes {
                    return Err(MediaError::Features);
                }
                *slot = Some(lease.clone());
                found = true;
            }
        }
        if !found {
            return Err(MediaError::Spans);
        }
        Ok(())
    }
    /// Probe-only override: bypasses the encoder with an admitted, domain-separated
    /// feature lease. The real span identity stays unchanged; callers must cold-admit
    /// the probe and disable every prefix snapshot. Never cache overrides by image key.
    pub fn attach_probe_override(&mut self, image_key: impl Into<MediaKey>, lease: EmbeddingLease) -> Result<(), MediaError> {
        let image_key = image_key.into();
        if lease.key() == image_key { return Err(MediaError::Features); }
        let rows = lease.features().ok_or(MediaError::NotReady(lease.key()))?;
        let mut found = false;
        for ((span, slot), &rows_count) in self.spans.iter().zip(&mut self.features).zip(&self.feature_rows) {
            if span.key == image_key {
                if rows.len() != rows_count * self.row_bytes { return Err(MediaError::Features); }
                *slot = Some(lease.clone());
                found = true;
            }
        }
        if !found { return Err(MediaError::Spans); }
        Ok(())
    }
    pub fn has_features(&self, key: impl Into<MediaKey>) -> bool {
        let key = key.into();
        self.spans
            .iter()
            .zip(&self.features)
            .any(|(s, f)| s.key == key && f.is_some())
    }
    /// Reuse a caller-owned staging vector. Decode chunks outside the prompt have no media.
    pub fn write_chunk(
        &self,
        start: usize,
        end: usize,
        out: &mut MediaChunk,
    ) -> Result<(), MediaError> {
        let count = end
            .checked_sub(start)
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or(MediaError::Features)?;
        if !self.ready(start, end) {
            let key = self
                .spans
                .iter()
                .zip(&self.features)
                .find(|(s, f)| s.start < end && s.checked_end().unwrap() > start && f.is_none())
                .unwrap()
                .0
                .key;
            return Err(MediaError::NotReady(key));
        }
        out.indices.clear();
        out.features.clear();
        out.mask.clear();
        out.mask.resize(count, 0);
        for ((span, lease), &offset) in self.spans.iter().zip(&self.features).zip(&self.offsets) {
            let lo = start.max(span.start);
            let hi = end.min(span.checked_end().unwrap());
            if lo >= hi {
                continue;
            }
            let data = lease.as_ref().unwrap().features().unwrap();
            out.features.extend_from_slice(
                &data[(offset + (lo - span.start)) * self.row_bytes..(offset + (hi - span.start)) * self.row_bytes],
            );
            for row in lo..hi {
                out.indices.push((row - start) as u32);
                out.mask[row - start] = 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ImageKey;
    use crate::media::EmbeddingCache;
    use std::sync::Arc;
    #[test]
    fn separated_spans_slice_one_complete_feature_buffer() {
        let key = ImageKey([7; 32]);
        let spans = vec![MediaSpan { start: 1, len: 2, key: key.into() },
            MediaSpan { start: 5, len: 2, key: key.into() }];
        let mut request = RequestMedia::new_with_layout(spans.clone(), 2, 8, vec![0, 2], vec![4, 4]).unwrap();
        assert_eq!(request.feature_rows(1), 4);
        let mut cache = EmbeddingCache::new(16);
        let pin = cache.reserve(key, 16).unwrap();
        let lease = cache.complete(key, (0u8..16).collect::<Vec<_>>().into()).unwrap();
        request.attach(lease).unwrap(); drop(pin);
        let mut chunk = MediaChunk::default();
        request.write_chunk(2, 6, &mut chunk).unwrap();
        assert_eq!(chunk.indices, [0, 3]);
        assert_eq!(chunk.features, (4u8..12).collect::<Vec<_>>());
        assert_eq!(chunk.mask, [1, 0, 0, 1]);
        request.write_chunk(5, 8, &mut chunk).unwrap();
        assert_eq!(chunk.features, (8u8..16).collect::<Vec<_>>());
        assert_eq!(chunk.indices, [0, 1]);
        let mut cache = EmbeddingCache::new(8);
        let pin = cache.reserve(key, 8).unwrap();
        let short = cache.complete(key, vec![0; 8].into()).unwrap();
        assert!(request.attach(short).is_err()); drop(pin);
        let probe_key = ImageKey([8; 32]);
        let mut cache = EmbeddingCache::new(16);
        let pin = cache.reserve(probe_key, 16).unwrap();
        let probe = cache.complete(probe_key, (16u8..32).collect::<Vec<_>>().into()).unwrap();
        request.attach_probe_override(key, probe).unwrap(); drop(pin);
        request.write_chunk(5, 7, &mut chunk).unwrap();
        assert_eq!(chunk.features, (24u8..32).collect::<Vec<_>>());
        for (offsets, rows) in [(vec![0], vec![4, 4]), (vec![0, 2], vec![4]),
            (vec![0, 3], vec![4, 4]), (vec![0, usize::MAX], vec![4, 4]),
            (vec![0, 2], vec![4, 5]), (vec![0, 2], vec![usize::MAX; 2])] {
            assert!(RequestMedia::new_with_layout(spans.clone(), 2, 8, offsets, rows).is_err());
        }
        let unequal = vec![spans[0], MediaSpan { len: 1, ..spans[1] }];
        assert!(RequestMedia::new(unequal.clone(), 2, 8).is_err());
        assert!(RequestMedia::new_with_layout(unequal, 2, 8, vec![0, 2], vec![4, 4]).is_ok());
    }

    #[test]
    fn span_features_borrows_attached_features_by_index() {
        let key = ImageKey([7; 32]);
        let other = ImageKey([8; 32]);
        let mut request = RequestMedia::new(vec![
            MediaSpan { start: 0, len: 1, key: key.into() },
            MediaSpan { start: 1, len: 1, key: other.into() },
            MediaSpan { start: 2, len: 1, key: key.into() },
        ], 2, 3).unwrap();
        assert!(request.span_features(0).is_none());
        assert!(request.span_features(3).is_none());
        let features: Arc<[u8]> = Arc::from([0, 0, 0x80, 0x3f]);
        let mut cache = EmbeddingCache::new(4);
        let pin = cache.reserve(key, 4).unwrap();
        let lease = cache.complete(key, Arc::clone(&features)).unwrap();
        request.attach(lease).unwrap();
        drop(pin);
        assert!(Arc::ptr_eq(request.span_features(0).unwrap(), &features));
        assert!(request.span_features(1).is_none());
        assert!(Arc::ptr_eq(request.span_features(2).unwrap(), &features));
        assert!(request.span_features(usize::MAX).is_none());
    }
    #[test]
    fn probe_override_uses_a_separate_budgeted_cache_identity() {
        let image = ImageKey([7; 32]); let override_key = ImageKey([8; 32]);
        let mut request = RequestMedia::new(vec![MediaSpan { start: 1, len: 1, key: image.into() }], 2, 3).unwrap();
        let mut cache = EmbeddingCache::new(4);
        let pin = cache.reserve(override_key, 4).unwrap();
        let lease = cache.complete(override_key, Arc::from([0, 0, 0x80, 0x3f])).unwrap();
        request.attach_probe_override(image, lease).unwrap(); drop(pin);
        assert_eq!(request.spans()[0].key, image.into());
        assert!(!cache.contains(image)); assert!(cache.contains(override_key));
        assert!(cache.reserve(image, 4).is_err(), "override remains admitted and pinned");
        let mut chunk = MediaChunk::default(); request.write_chunk(0, 3, &mut chunk).unwrap();
        assert_eq!(chunk.indices, [1]); assert_eq!(chunk.features, [0, 0, 0x80, 0x3f]);
        let mut native = EmbeddingCache::new(4); let pin = native.reserve(image, 4).unwrap();
        let lease = native.complete(image, Arc::from([0; 4])).unwrap();
        assert!(request.attach_probe_override(image, lease).is_err()); drop(pin);
    }
    #[test]
    fn lazy_spans_stage_only_overlapping_rows_with_native_row_indices() {
        let key = ImageKey([7; 32]);
        let span = MediaSpan {
            start: 2,
            len: 3,
            key: key.into(),
        };
        let mut request = RequestMedia::new(vec![span], 2, 6).unwrap();
        assert!(request.ready(5, 6));
        assert!(!request.ready(4, 6));
        assert_eq!(request.needed(5, 6).count(), 0);
        let mut out = MediaChunk::default();
        assert!(request.write_chunk(1, 4, &mut out).is_err());
        let mut cache = EmbeddingCache::new(12);
        let reserved = cache.reserve(key, 12).unwrap();
        let lease = cache
            .complete(key, Arc::from((0u8..12).collect::<Vec<_>>()))
            .unwrap();
        request.attach(lease).unwrap();
        drop(reserved);
        request.write_chunk(3, 6, &mut out).unwrap();
        assert_eq!(out.indices, [0, 1]);
        assert_eq!(out.features, (4u8..12).collect::<Vec<_>>());
        assert_eq!(out.mask, [1, 1, 0]);
        request.write_chunk(6, 8, &mut out).unwrap();
        assert!(out.features.is_empty() && out.indices.is_empty());
        assert_eq!(out.mask, [0, 0]);
    }
}
