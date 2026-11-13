//! Complete-frontier stages; each end is an ordinary whole-prompt prefill.
use cuteafd_engine::prefix::{plan_media_points, PointPolicy};
use cuteafd_core::MediaSpan;
use std::collections::VecDeque;

/// Native message starts emitted by the V4.1 template (not ordinary vocabulary).
pub(crate) const MESSAGE_STARTS: [&str; 3] = ["<｜System｜>", "<｜User｜>", "<｜Assistant｜>"];

#[derive(Debug, Default)]
pub(crate) struct Stages {
    points: VecDeque<usize>,
}
impl Stages {
    pub fn plan(tokens: &[u32], markers: &[u32], media: &[MediaSpan], resume: usize,
        chunk: usize, minimum: usize, policy: PointPolicy) -> Self {
        if policy.per_request == 0 || (policy.boundaries == 0 && policy.gap == 0) { return Self::default(); }
        let boundaries = crate::shared::prefix::boundaries(tokens, markers);
        // Reach zero: a complete mark cannot capture behind the committed frontier.
        let plan = plan_media_points(resume, tokens.len(), chunk, &boundaries, 0, minimum, policy, media);
        Self { points: plan.points.into_iter().map(|(_, p)| p).collect() }
    }
    /// A failed mark or restore may need a cold restart. Materialize every
    /// image beforehand, including images covered by the initial cache hit.
    pub fn image_prepare_from(&self, resume: usize) -> usize {
        if self.has_points() { 0 } else { resume }
    }
    pub fn len(&self) -> usize { self.points.len() }
    pub fn has_points(&self) -> bool { !self.points.is_empty() }
    pub fn end(&self, prompt_end: usize) -> usize { self.points.front().copied().unwrap_or(prompt_end) }
    /// A failed capture restarts cold for the entire prompt, with no more staging.
    pub fn completed(&mut self, captured: bool) {
        if captured { self.points.pop_front(); } else { self.points.clear(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn policy(boundaries: usize) -> PointPolicy { PointPolicy { gap: 0, boundaries, per_request: 4 } }
    #[test]
    fn last_message_start_is_an_exclusive_frontier() {
        let mut tokens = vec![1; 1024];
        for p in [0, 256, 700, 1019] { tokens[p] = 7; }
        let mut stages = Stages::plan(&tokens, &[7], &[], 0, 256, 64, policy(1));
        assert_eq!(stages.end(tokens.len()), 1019);
        assert_eq!(tokens[..stages.end(tokens.len())].last(), Some(&1));
        stages.completed(true);
        assert_eq!(stages.end(tokens.len()), tokens.len());
        assert!(!Stages::plan(&tokens, &[7], &[], 1019, 256, 64, policy(1)).has_points());
        assert!(!Stages::plan(&tokens, &[7], &[], 0, 256, 64, policy(0)).has_points());
        assert!(!Stages::plan(&tokens, &[7], &[], 0, 256, 64,
            PointPolicy { per_request: 0, ..policy(1) }).has_points());
    }
    #[test]
    fn points_round_to_media_start_and_honor_minimum_resume_and_cap() {
        let mut tokens = vec![1; 1024];
        for p in [50, 400, 600, 900] { tokens[p] = 7; }
        let media = [MediaSpan { start: 512, len: 128, key: cuteafd_core::ImageKey([1; 32]).into() }];
        let stages = Stages::plan(&tokens, &[7], &media, 400, 256, 64,
            PointPolicy { per_request: 1, ..policy(4) });
        assert_eq!(stages.points.iter().copied().collect::<Vec<_>>(), [900]);
        let mut stages = Stages::plan(&tokens, &[7], &media, 400, 256, 64, policy(4));
        assert_eq!(stages.points.iter().copied().collect::<Vec<_>>(), [512, 900]);
        stages.completed(false);
        assert_eq!(stages.end(tokens.len()), tokens.len());
        assert!(stages.points.is_empty());
    }
    #[test]
    fn periodic_points_use_the_existing_gap() {
        let stages = Stages::plan(&vec![1; 2049], &[], &[], 0, 256, 64,
            PointPolicy { gap: 512, boundaries: 0, per_request: 2 });
        assert_eq!(stages.points.iter().copied().collect::<Vec<_>>(), [1536, 2048]);
    }
    #[test]
    fn staged_images_prepare_cold_fallback_even_after_a_prefix_hit() {
        let mut tokens = vec![1; 1024]; tokens[900] = 7;
        let stages = Stages::plan(&tokens, &[7], &[], 600, 256, 64, policy(1));
        assert_eq!(stages.image_prepare_from(600), 0);
        assert_eq!(Stages::default().image_prepare_from(600), 600);
    }


}
