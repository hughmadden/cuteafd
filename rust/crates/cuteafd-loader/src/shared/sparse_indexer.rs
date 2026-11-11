//! CPU contract for sparse selection. Families own projections, positional
//! transforms and device hooks; no scheduler or CUDA storage policy lives here.
//! V4 compressed-entry producers do not fit token/block-max granularity: only
//! their post-transform scores/top-k can reuse this contract through a family hook.
use anyhow::{ensure, Context, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Norm { None, LayerNorm, RmsNorm, GemmaRmsNorm }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeStyle { NeoX, Interleaved }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rope { pub dim: usize, pub style: RopeStyle }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation { Identity, Relu }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadReduction { None, WeightedSum { learned: bool } }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pool { Max }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForcedWindows { WithinTopK, Union }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Granularity { Token, Block { tokens: usize, pool: Pool } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexerGeometry {
    pub query_heads: usize,
    pub key_heads: usize,
    pub head_dim: usize,
    pub rope: Option<Rope>,
    pub query_norm: Norm,
    pub key_norm: Norm,
    pub activation: Activation,
    pub head_reduction: HeadReduction,
    pub selection_heads: usize,
    pub granularity: Granularity,
    pub topk: usize,
    /// Count of forced prefix blocks: 0 = none, 1 = block 0.
    pub initial_blocks: usize,
    /// Count including the query's current block: 1 = current block only.
    pub local_blocks: usize,
    /// Forced windows consume top-k slots or extend the learned selection.
    pub forced_windows: ForcedWindows,
    /// A separate index-value stream, not the model's attention value cache.
    pub index_values: bool,
}

impl IndexerGeometry {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.query_heads > 0 && self.key_heads > 0 && self.head_dim > 0
            && self.topk > 0 && self.query_heads % self.key_heads == 0,
            "invalid sparse indexer head/top-k geometry");
        ensure!(self.rope.is_none_or(|rope| rope.dim > 0 && rope.dim <= self.head_dim && rope.dim % 2 == 0),
            "invalid sparse indexer RoPE width");
        let expected = match self.head_reduction { HeadReduction::None => self.query_heads, _ => 1 };
        ensure!(self.selection_heads == expected, "sparse indexer selection heads disagree with head reduction");
        match self.granularity {
            Granularity::Token => ensure!(self.initial_blocks == 0 && self.local_blocks == 0,
                "forced block windows require block granularity"),
            Granularity::Block { tokens, .. } => ensure!(tokens > 0, "zero sparse indexer block width"),
        }
        self.selection_width()?;
        Ok(())
    }

    /// Payload only; the physical allocator adds its one backing/tail charge.
    /// Quantized cache scales are a separate, format-specific owner.
    pub fn key_cache_bytes(&self, tokens: usize, element_bytes: usize) -> Result<usize> {
        self.validate()?;
        ensure!(element_bytes > 0, "zero sparse indexer element width");
        tokens.checked_mul(self.key_heads).and_then(|n| n.checked_mul(self.head_dim))
            .and_then(|n| n.checked_mul(element_bytes)).context("sparse index cache extent overflow")
    }

    pub fn selection_width(&self) -> Result<usize> {
        let width = match self.forced_windows {
            ForcedWindows::WithinTopK => Some(self.topk),
            ForcedWindows::Union => self.topk.checked_add(self.initial_blocks)
                .and_then(|n| n.checked_add(self.local_blocks)),
        }.context("sparse selection width overflow")?;
        ensure!(width <= i32::MAX as usize, "sparse selection exceeds int32 ABI");
        Ok(width)
    }

    /// Fixed-width int32 logical ids. Invalid slots are -1; Union reserves
    /// worst-case disjoint learned/forced sets, WithinTopK reserves only topk.
    pub fn selection_bytes(&self, rows: usize) -> Result<usize> {
        self.validate()?;
        let width = self.selection_width()?;
        rows.checked_mul(self.selection_heads).and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_mul(4)).context("sparse selection extent overflow")
    }

    /// Mathematical scores for one already normalized/rotated query. Q is
    /// [query_heads, dim], K [tokens, key_heads, dim]. Learned weights already
    /// include the family's head/dimension scaling (as in the shared DSA ABI).
    pub fn scores(&self, query: &[f32], keys: &[f32], weights: Option<&[f32]>) -> Result<Vec<Vec<f32>>> {
        self.validate()?;
        let qw = self.query_heads.checked_mul(self.head_dim).context("index query extent overflow")?;
        let kw = self.key_heads.checked_mul(self.head_dim).context("index key extent overflow")?;
        ensure!(query.len() == qw && keys.len() % kw == 0, "sparse score tensor shape mismatch");
        ensure!(query.iter().chain(keys).all(|v| v.is_finite()), "nonfinite sparse index input");
        match self.head_reduction {
            HeadReduction::WeightedSum { learned: true } => ensure!(weights.is_some_and(|w|
                w.len() == self.query_heads && w.iter().all(|v| v.is_finite())), "missing learned index head weights"),
            _ => ensure!(weights.is_none(), "unexpected index head weights"),
        }
        let mut result = vec![vec![0.0; keys.len() / kw]; self.selection_heads];
        for (token, key) in keys.chunks_exact(kw).enumerate() {
            for head in 0..self.query_heads {
                let kh = head / (self.query_heads / self.key_heads);
                let dot: f32 = query[head*self.head_dim..(head+1)*self.head_dim].iter()
                    .zip(&key[kh*self.head_dim..(kh+1)*self.head_dim]).map(|(q,k)| q*k).sum();
                let score = match self.activation { Activation::Identity => dot, Activation::Relu => dot.max(0.0) };
                match self.head_reduction {
                    HeadReduction::None => result[head][token] = score,
                    HeadReduction::WeightedSum { learned } => result[0][token] += score * if learned { weights.unwrap()[head] } else { 1.0 },
                }
            }
        }
        ensure!(result.iter().flatten().all(|v| v.is_finite()), "sparse score overflow");
        Ok(result)
    }

    /// Causal logical selection. Equal scores prefer smaller ids; output ids
    /// are ascending. No future token participates in a block's max pool.
    pub fn select(&self, scores: &[Vec<f32>], position: usize) -> Result<Vec<Vec<i32>>> {
        self.validate()?;
        ensure!(scores.len() == self.selection_heads && scores.iter().all(|s| s.len() > position),
            "sparse selection score shape mismatch");
        let block = match self.granularity { Granularity::Token => 1, Granularity::Block { tokens, .. } => tokens };
        let count = position / block + 1;
        ensure!(count <= i32::MAX as usize, "sparse logical ids exceed int32 ABI");
        let width = self.selection_width()?;
        let mut result = Vec::with_capacity(self.selection_heads);
        for head in scores {
            ensure!(head[..=position].iter().all(|s| s.is_finite()), "nonfinite causal index score");
            let pooled: Vec<f32> = head[..=position].chunks(block)
                .map(|values| values.iter().copied().fold(f32::NEG_INFINITY, f32::max)).collect();
            let forced = |id: usize| id < self.initial_blocks.min(count)
                || id >= count.saturating_sub(self.local_blocks);
            let mut ids: Vec<usize> = (0..count).collect();
            ids.sort_by(|&a,&b| {
                let priority = if self.forced_windows == ForcedWindows::WithinTopK {
                    forced(b).cmp(&forced(a))
                } else { std::cmp::Ordering::Equal };
                priority.then_with(|| {
                    if self.forced_windows == ForcedWindows::WithinTopK && forced(a) && forced(b) {
                        a.cmp(&b)
                    } else { pooled[b].partial_cmp(&pooled[a]).unwrap().then(a.cmp(&b)) }
                })
            });
            ids.truncate(self.topk);
            if self.forced_windows == ForcedWindows::Union {
                ids.extend(0..self.initial_blocks.min(count));
                ids.extend(count.saturating_sub(self.local_blocks)..count);
            }
            ids.sort_unstable();
            ids.dedup();
            let mut ids: Vec<i32> = ids.into_iter().map(|id| id as i32).collect();
            ids.resize(width, -1);
            result.push(ids);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token() -> IndexerGeometry {
        IndexerGeometry { query_heads: 2, key_heads: 1, head_dim: 2,
            rope: None, query_norm: Norm::None, key_norm: Norm::LayerNorm,
            activation: Activation::Relu, head_reduction: HeadReduction::WeightedSum { learned: true },
            selection_heads: 1, granularity: Granularity::Token, topk: 2,
            initial_blocks: 0, local_blocks: 0, forced_windows: ForcedWindows::WithinTopK, index_values: false }
    }

    #[test]
    fn weighted_relu_token_selection_is_causal_and_stable() -> Result<()> {
        let g = token();
        let scores = g.scores(&[1.,-1.,-1.,1.], &[1.,-1.,-1.,1.,0.,0.], Some(&[0.5,0.25]))?;
        assert_eq!(scores, vec![vec![1.,0.5,0.]]);
        assert_eq!(g.select(&scores, 0)?, vec![vec![0,-1]]);
        assert_eq!(g.select(&[vec![2.,2.,100.]], 1)?, vec![vec![0,1]]);
        assert_eq!(g.select(&[vec![2.,2.,2.]], 2)?, vec![vec![0,1]]);
        assert_eq!(g.key_cache_bytes(64, 2)?, 256);
        assert_eq!(g.selection_bytes(3)?, 24);
        assert!(g.key_cache_bytes(usize::MAX, 2).is_err());
        assert!(g.selection_bytes(usize::MAX).is_err());
        assert!(g.scores(&[1.,2.], &[], None).is_err());
        Ok(())
    }

    #[test]
    fn signed_per_head_blocks_pool_only_causal_tokens_and_union_windows() -> Result<()> {
        let g = IndexerGeometry { activation: Activation::Identity, head_reduction: HeadReduction::None,
            selection_heads: 2, granularity: Granularity::Block { tokens: 2, pool: Pool::Max },
            topk: 1, local_blocks: 1, initial_blocks: 1, forced_windows: ForcedWindows::Union, ..token() };
        assert_eq!(g.select(&[vec![-1.,-2.,-3.,100.,9.], vec![-9.,-8.,-1.,100.,9.]], 2)?,
            vec![vec![0,1,-1],vec![0,1,-1]]);
        let within = IndexerGeometry { forced_windows: ForcedWindows::WithinTopK, initial_blocks: 0, ..g };
        assert_eq!(within.selection_width()?, 1);
        assert_eq!(within.selection_bytes(3)?, 24);
        assert_eq!(within.select(&[vec![-1.,-2.,-3.,100.],vec![-9.,-8.,-1.,100.]], 2)?,
            vec![vec![1],vec![1]]);
        let overlapping = IndexerGeometry { topk: 3, initial_blocks: 1, ..within };
        assert_eq!(overlapping.select(&[vec![0.;4],vec![0.;4]], 2)?, vec![vec![0,1,-1],vec![0,1,-1]]);
        let no_windows = IndexerGeometry { initial_blocks: 0, local_blocks: 0, ..g };
        assert_eq!(no_windows.select(&[vec![-1.,-2.,-3.,100.],vec![-9.,-8.,-1.,100.]], 2)?,
            vec![vec![0],vec![1]]);
        Ok(())
    }
}
