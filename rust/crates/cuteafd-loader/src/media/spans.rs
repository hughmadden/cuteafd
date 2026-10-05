use super::{invalid, MediaSpan, PreparedImage, Result};

#[derive(Debug, Clone, Copy)]
pub struct SpanExpander {
    pub placeholder: u32,
    pub start: u32,
    pub end: u32,
    pub vocabulary: u32,
}
#[derive(Debug)]
pub struct ExpandedMediaPrompt {
    /// Only these native ids reach the LM, PLE and drafters.
    pub tokens: Vec<u32>,
    pub media: Vec<MediaSpan>,
}
impl SpanExpander {
    /// Marker ids come from the checkpoint config, never hard-coded by the API.
    pub fn from_config(config: &serde_json::Value, vocabulary: u32) -> Result<Self> {
        let id = |names: &[&str]| {
            names
                .iter()
                .find_map(|n| config.get(*n))
                .and_then(|v| v.as_u64())
                .and_then(|v| u32::try_from(v).ok())
                .ok_or_else(|| invalid(format!("missing image marker: {}", names[0])))
        };
        let expander = Self {
            placeholder: id(&["image_token_id"])?,
            start: id(&["vision_start_token_id", "image_start_token_id", "boi_token_id"])?,
            end: id(&["vision_end_token_id", "image_end_token_id", "eoi_token_id"])?,
            vocabulary,
        };
        if vocabulary >= 1 << 31
            || [expander.placeholder, expander.start, expander.end]
                .iter()
                .any(|v| *v >= vocabulary)
            || expander.placeholder == expander.start
            || expander.placeholder == expander.end
            || expander.start == expander.end
        {
            return Err(invalid("invalid image marker ids or vocabulary"));
        }
        Ok(expander)
    }
    pub fn expand(
        &self,
        tokens: &[u32],
        images: &[PreparedImage],
        context_limit: usize,
    ) -> Result<ExpandedMediaPrompt> {
        if self.vocabulary == 0
            || self.vocabulary >= 1 << 31
            || [self.placeholder, self.start, self.end]
                .iter()
                .any(|&v| v >= self.vocabulary)
            || self.placeholder == self.start
            || self.placeholder == self.end
            || self.start == self.end
        {
            return Err(invalid("invalid image marker ids or vocabulary"));
        }
        if tokens.iter().any(|&v| v >= self.vocabulary) {
            return Err(invalid("token outside checkpoint vocabulary"));
        }
        let count = tokens.iter().filter(|&&v| v == self.placeholder).count();
        if count != images.len() {
            return Err(invalid(
                "image placeholder count differs from supplied images",
            ));
        }
        let mut expanded_len = tokens.len();
        for image in images {
            if image.tokens == 0 {
                return Err(invalid("image has no LM rows"));
            }
            expanded_len = expanded_len
                .checked_add(image.tokens - 1)
                .ok_or_else(|| invalid("image prompt length overflow"))?;
        }
        if expanded_len > context_limit {
            return Err(invalid("expanded image prompt exceeds context limit"));
        }
        let mut expanded = Vec::with_capacity(expanded_len);
        let mut media = Vec::with_capacity(images.len());
        for (i, &token) in tokens.iter().enumerate() {
            if token != self.placeholder {
                expanded.push(token);
                continue;
            }
            if i == 0
                || tokens.get(i - 1) != Some(&self.start)
                || tokens.get(i + 1) != Some(&self.end)
            {
                return Err(invalid(
                    "image placeholder must be enclosed by image start/end tokens",
                ));
            }
            let image = &images[media.len()];
            media.push(MediaSpan {
                start: expanded.len(),
                len: image.tokens,
                key: image.key,
            });
            expanded.extend(std::iter::repeat_n(self.placeholder, image.tokens));
        }
        Ok(ExpandedMediaPrompt {
            tokens: expanded,
            media,
        })
    }
}
