//! The prefix cache's serving knobs and host tier setup, shared by every generic family
//! (`cuteafd_engine::prefix` does the work; each family implements `PrefixFamily`).
use crate::families::deepseek_v41::v41_native_serve::prefix::CudaCopyEngine;
use anyhow::Result;
use cuteafd_engine::prefix::PointPolicy;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};

/// The prefix cache's knobs.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct PrefixArgs {
    /// Retained snapshots per bank (prompts, completed turns); 0 turns the prefix cache off.
    #[arg(long, env = "CUTEAFD_PREFIX_CACHE_ENTRIES", default_value_t = 20)]
    pub prefix_cache_entries: usize,
    /// Device memory for retained positional marks (SWA rows, MTP hidden rows, recurrent
    /// state), MiB; the arena holds two marks per entry pair while they fit, and never fewer
    /// than two per decoding sequence plus two.
    #[arg(long, env = "CUTEAFD_PREFIX_CACHE_MARK_MIB", default_value_t = 2048)]
    pub prefix_cache_mark_mib: usize,
    /// Shortest prompt or turn worth a snapshot.
    #[arg(long, default_value_t = 64)]
    pub prefix_cache_min_tokens: usize,
    /// Pinned host memory for snapshots the device evicts (e.g. 64GiB; 0 = off).
    #[arg(long, env = "CUTEAFD_HOST_CACHE_BYTES", default_value = "0", value_parser = parse_bytes)]
    pub host_cache_bytes: u64,
    /// Shortest snapshot the host tier keeps.
    #[arg(long, default_value_t = 512)]
    pub host_cache_min_tokens: u32,
    /// Intermediate snapshot points (off by default; agentic sessions hit prompt-end and
    /// turn-end snapshots): one every N prefilled tokens at a chunk end (0 = none, e.g. 8192).
    #[arg(long, env = "CUTEAFD_PREFIX_POINT_GAP", default_value_t = 0)]
    pub prefix_point_gap: usize,
    /// Intermediate snapshot points at the last N message boundaries of the rendered prompt
    /// (before the generation prompt, before the last message, ...; 0 = none, e.g. 2).
    #[arg(long, env = "CUTEAFD_PREFIX_POINT_BOUNDARIES", default_value_t = 0)]
    pub prefix_point_boundaries: usize,
    /// Most intermediate points per prompt (the deepest are kept).
    #[arg(long, env = "CUTEAFD_PREFIX_POINTS_PER_REQUEST", default_value_t = 4)]
    pub prefix_points_per_request: usize,
    /// Partial reuse: MiMo replays the SWA window before the aligned common prefix
    /// (approximate); GLM 5.3 resumes at the last common page (exact: its pages are its whole
    /// state). Off: exact snapshot frontiers only. GLM 5.3 Flash has none (recurrent state).
    #[arg(long, env = "CUTEAFD_PREFIX_PARTIAL", value_enum, default_value_t = Toggle::Off)]
    pub prefix_partial: Toggle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum Toggle {
    On,
    Off,
}

impl PrefixArgs {
    pub fn points(&self) -> PointPolicy {
        PointPolicy { gap: self.prefix_point_gap, boundaries: self.prefix_point_boundaries,
            per_request: self.prefix_points_per_request }
    }

    /// The pinned host tier's config and copy engine (`template`: any device buffer of the
    /// engine), or None when the cache or the tier is off.
    pub fn host_tier<'a>(&self, library: &'a NativeLibrary, template: CuteafdDeviceBuffer, mark_bytes: usize)
        -> Result<Option<(cuteafd_hostcache::config::Config, CudaCopyEngine<'a>)>> {
        if self.prefix_cache_entries == 0 || self.host_cache_bytes == 0 {
            return Ok(None);
        }
        let config = cuteafd_hostcache::config::Config {
            bytes: self.host_cache_bytes,
            chunk_bytes: (256u64 << 20).max(mark_bytes as u64).min(self.host_cache_bytes),
            min_tokens: self.host_cache_min_tokens,
            ..Default::default()
        };
        Ok(Some((config, CudaCopyEngine::new(library, template)?)))
    }
}

/// Ids of the `markers` (message-start tokens such as `<|user|>`) that `tokenizer.json`'s added
/// tokens define, for intermediate snapshot points at message boundaries.
pub(crate) fn marker_ids(snapshot: &std::path::Path, markers: &[&str]) -> Result<Vec<u32>> {
    let text = std::fs::read_to_string(snapshot.join("tokenizer.json"))?;
    let tokenizer: serde_json::Value = serde_json::from_str(&text)?;
    let added = tokenizer["added_tokens"].as_array().map_or(&[][..], Vec::as_slice);
    Ok(markers.iter().filter_map(|marker| added.iter().find(|t| t["content"] == *marker)
        .and_then(|t| t["id"].as_u64()).map(|id| id as u32)).collect())
}

/// Positions of any of `markers` in `tokens` (message boundaries, in order).
pub(crate) fn boundaries(tokens: &[u32], markers: &[u32]) -> Vec<usize> {
    tokens.iter().enumerate().filter(|(_, t)| markers.contains(t)).map(|(i, _)| i).collect()
}

/// `123`, `512MiB`, `64GiB`, `1.5GB`.
pub(crate) fn parse_bytes(text: &str) -> std::result::Result<u64, String> {
    let text = text.trim();
    let split = text.find(|c: char| c.is_ascii_alphabetic()).unwrap_or(text.len());
    let (number, unit) = text.split_at(split);
    let scale: f64 = match unit {
        "" | "B" => 1.0,
        "KB" => 1e3,
        "MB" => 1e6,
        "GB" => 1e9,
        "KiB" => 1024.0,
        "MiB" => 1024.0 * 1024.0,
        "GiB" => 1024.0 * 1024.0 * 1024.0,
        other => return Err(format!("unknown byte unit {other:?}")),
    };
    let value: f64 = number.parse().map_err(|e| format!("{text:?}: {e}"))?;
    if !(value >= 0.0) {
        return Err(format!("{text:?} is negative"));
    }
    Ok((value * scale) as u64)
}

/// A byte range inside `buffer` (bounds-checked; pointer arithmetic only).
pub(crate) fn view(buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    anyhow::ensure!(offset.checked_add(bytes).is_some_and(|end| end <= buffer.bytes),
        "view {offset}+{bytes} past a {}-byte buffer", buffer.bytes);
    Ok(CuteafdDeviceBuffer { ptr: buffer.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..buffer })
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        prefix: PrefixArgs,
    }

    #[test]
    fn prefix_knobs_default_on_with_the_host_tier_off() {
        let cli = Cli::parse_from(["serve"]);
        assert_eq!((cli.prefix.prefix_cache_entries, cli.prefix.host_cache_bytes), (20, 0));
        assert_eq!(cli.prefix.prefix_partial, Toggle::Off);
        assert_eq!(cli.prefix.points(), PointPolicy { gap: 0, boundaries: 0, per_request: 4 });
        let cli = Cli::parse_from(["serve", "--prefix-partial", "on", "--prefix-point-gap", "8192",
            "--prefix-point-boundaries", "2"]);
        assert_eq!(cli.prefix.points(), PointPolicy { gap: 8192, boundaries: 2, per_request: 4 });
        assert_eq!(cli.prefix.prefix_partial, Toggle::On);
        let cli = Cli::parse_from(["serve", "--prefix-cache-entries", "0", "--host-cache-bytes", "64GiB"]);
        assert_eq!((cli.prefix.prefix_cache_entries, cli.prefix.host_cache_bytes), (0, 64 << 30));
        assert_eq!(parse_bytes("512MiB"), Ok(512 << 20));
        assert_eq!(parse_bytes("1.5GB"), Ok(1_500_000_000));
        assert_eq!(parse_bytes("123"), Ok(123));
        assert!(parse_bytes("12 parsecs").is_err() && parse_bytes("-1").is_err());
    }
}
