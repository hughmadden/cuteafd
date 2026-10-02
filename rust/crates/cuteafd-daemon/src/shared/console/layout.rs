//! What a family's console shows: the static half of the telemetry schema.
//!
//! The page renders whatever a layout declares. Stage keys name values of a
//! round's `stages`, except for three derived namespaces the page computes:
//! `round.cycle` / `round.cycleN` (a round's wall time, any lane / lane N),
//! `layers.sum` / `layers.max` / `layers.mean.K` (the round's layer profile:
//! total, slowest, mean of class K) and `prefill.chunk` / `prefill.short` /
//! `prefill.replay` / `prefill.restore` (admission steps, host clock).
use serde_json::{json, Value};
use std::path::PathBuf;

/// Palette tokens of the page (`--accepted`, `--target`, ...).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Color { Accepted, Target, Rejected, Grammar, Prefill, Restore, Replay, Rtx, Spark, Warn, Ink }

impl Color {
    pub fn token(self) -> &'static str {
        match self {
            Self::Accepted => "accepted", Self::Target => "target", Self::Rejected => "rejected",
            Self::Grammar => "grammar", Self::Prefill => "prefill", Self::Restore => "restore",
            Self::Replay => "replay", Self::Rtx => "rtx", Self::Spark => "spark", Self::Warn => "warn",
            Self::Ink => "ink-2",
        }
    }
}

/// One group of "pipeline micro-steps": named stages with a palette color.
pub(crate) struct StepGroup {
    pub title: &'static str,
    pub note: &'static str,
    pub rows: Vec<(String, String, Color)>,
}

impl StepGroup {
    pub fn new(title: &'static str, note: &'static str, rows: &[(&str, &str, Color)]) -> Self {
        Self { title, note, rows: rows.iter().map(|&(k, l, c)| (k.to_string(), l.to_string(), c)).collect() }
    }

    /// The admission steps every family shows (prefix restores and prefill chunks).
    pub fn admission(replay: bool) -> Self {
        let mut rows = vec![("prefill.chunk", "prefill chunk ≥ 1024 rows", Color::Prefill),
            ("prefill.short", "short prompt prefill", Color::Prefill)];
        if replay { rows.push(("prefill.replay", "decoder replay", Color::Replay)); }
        rows.push(("prefill.restore", "prefix restore", Color::Restore));
        Self::new("Admission", "host clock", &rows)
    }
}

/// The per-layer profile: one bar per layer from `first`, colored by class.
pub(crate) struct Layers {
    pub title: String,
    /// The first layer drawn (V4.1's layer 0 has no FFN finish to measure from).
    pub first: usize,
    pub classes: Vec<(String, Color)>,
    /// Class of every layer (index into `classes`).
    pub class: Vec<u8>,
}

impl Layers {
    /// The per-layer host-clock profile of a generic engine's decode step
    /// (`console::layer_mark` at each layer's end): class 0 runs its FFN on
    /// the coordinator (dense or local experts), class 1 on the Sparks.
    pub fn host_clock() -> Self {
        Self { title: "Layer profile · last step · host clock".into(), first: 0,
            classes: vec![("dense".into(), Color::Ink), ("Spark-routed experts".into(), Color::Spark),
                ("RTX-resident experts".into(), Color::Rtx)],
            class: Vec::new() }
    }
}

/// A generic engine layer's class in [`Layers::host_clock`]: dense, MoE with
/// its experts on the Sparks, or MoE with its experts on the coordinator.
pub(crate) fn layer_class(dense: bool, spark: bool) -> u8 {
    match (dense, spark) {
        (true, _) => 0,
        (false, true) => 1,
        (false, false) => 2,
    }
}

impl StepGroup {
    /// The layer-profile rows of a generic engine's decode step.
    pub fn layers() -> Self {
        Self::new("Layers", "host clock, layer end to layer end", &[("layers.sum", "all layers", Color::Target),
            ("layers.mean.0", "dense layer (mean)", Color::Ink), ("layers.mean.1", "Spark-expert layer (mean)", Color::Spark),
            ("layers.mean.2", "RTX-expert layer (mean)", Color::Rtx), ("layers.max", "slowest layer", Color::Warn)])
    }
}

/// The speculator a family runs; families without one omit it and the page
/// hides the acceptance panel.
pub(crate) struct Speculator {
    pub name: String,
    /// Draft positions the acceptance panel shows.
    pub positions: usize,
    pub policy: String,
}

/// A labeled value in the speculation panel: label, value, note.
pub(crate) type Fact = (String, String, String);

pub(crate) struct Layout {
    pub family: &'static str,
    /// Public model id.
    pub model: String,
    /// Snapshot directory: the header's checkpoint, and the tokenizer for the text view.
    pub snapshot: PathBuf,
    /// Hardware, e.g. `2×RTX + 4 Spark`.
    pub hardware: String,
    /// How the coordinator splits work across its GPUs (None: one GPU).
    pub split: Option<String>,
    pub speculator: Option<Speculator>,
    pub lanes: usize,
    pub concurrency: usize,
    pub steps: Vec<StepGroup>,
    pub layers: Option<Layers>,
    /// Tokens the text view shows as an end marker.
    pub eos: Vec<u32>,
    /// Family facts for the header tooltip and `/v1/console/snapshot`.
    pub extra: Value,
    /// Polled about once a second on the console thread: `{"facts": [[label,
    /// value, note]], "layer_class": [..]}`, both optional.
    pub dynamic: Option<Box<dyn Fn() -> Value + Send>>,
}

impl Layout {
    pub fn new(family: &'static str, model: String, snapshot: PathBuf) -> Self {
        Self { family, model, snapshot, hardware: String::new(), split: None, speculator: None, lanes: 1,
            concurrency: 1, steps: Vec::new(), layers: None, eos: Vec::new(), extra: Value::Null, dynamic: None }
    }

    pub fn json(&self, text: bool) -> Value {
        let steps: Vec<Value> = self.steps.iter().map(|group| json!({"title": group.title, "note": group.note,
            "rows": group.rows.iter().map(|(k, l, c)| json!([k, l, c.token()])).collect::<Vec<_>>()})).collect();
        let layers = self.layers.as_ref().map(|layers| json!({"title": layers.title, "first": layers.first,
            "classes": layers.classes.iter().map(|(l, c)| json!([l, c.token()])).collect::<Vec<_>>(),
            "class": layers.class}));
        let speculator = self.speculator.as_ref().map(|s| json!({"name": s.name, "positions": s.positions,
            "policy": s.policy}));
        json!({
            "family": self.family,
            "model": self.model,
            "checkpoint": super::checkpoint(&self.snapshot),
            "layout": self.hardware,
            "split": self.split,
            "speculator": speculator,
            "lanes": self.lanes,
            "concurrency": self.concurrency,
            "steps": steps,
            "layers": layers,
            "build": build(),
            "text": text,
            "extra": self.extra,
            "started_unix_ms": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_millis() as u64),
        })
    }
}

/// The running build: a release tag when the image carries one, the engine
/// commit (the image's, else the one this binary was compiled from) and
/// whether that source had uncommitted changes.
pub(crate) fn build() -> Value {
    let env = |name: &str| std::env::var(name).ok().filter(|value| !value.is_empty() && value != "unknown");
    let release = env("CUTEAFD_RELEASE_VERSION");
    let compiled = option_env!("CUTEAFD_BUILD_COMMIT").filter(|commit| !commit.is_empty());
    let commit = env("CUTEAFD_CONSOLE_REVISION").or_else(|| env("CUTEAFD_ENGINE_COMMIT"))
        .or_else(|| compiled.map(str::to_string));
    let dirty = option_env!("CUTEAFD_BUILD_DIRTY") == Some("1");
    json!({"release": release, "commit": commit, "dirty": dirty})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_serializes_steps_layers_and_speculator() {
        let mut layout = Layout::new("glm5", "zai-org/GLM-5.3".into(), PathBuf::from("/x/models--a--b/snapshots/cafe1234"));
        layout.speculator = Some(Speculator { name: "DFlash2".into(), positions: 7, policy: "adaptive".into() });
        layout.steps = vec![StepGroup::new("Decode step", "host clock", &[("verify", "verify", Color::Target)]),
            StepGroup::admission(false)];
        layout.layers = Some(Layers { title: "x".into(), first: 0, classes: vec![("Spark".into(), Color::Spark)],
            class: vec![0, 0] });
        let value = layout.json(false);
        assert_eq!(value["checkpoint"], "a/b@cafe1234");
        assert_eq!(value["speculator"]["positions"], 7);
        assert_eq!(value["steps"][0]["rows"][0], json!(["verify", "verify", "target"]));
        assert_eq!(value["steps"][1]["rows"].as_array().unwrap().len(), 3);
        assert_eq!(value["layers"]["classes"][0], json!(["Spark", "spark"]));
        assert!(value["build"].get("dirty").is_some());
    }
}
