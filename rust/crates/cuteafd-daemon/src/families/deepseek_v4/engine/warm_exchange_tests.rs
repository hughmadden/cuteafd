//! Structural coverage of the warm passes' peer generations on a head split.
//!
//! A capture-only pass records graph bodies and runs none of them, so no eager
//! rank-coupled operation may run there: its peer push or wait would be matched
//! only by a body the pass never executes. The eager pass and serving run every
//! body, so the same eager work is matched on both ranks.

fn body<'a>(source: &'a str, first: &str, next: &str) -> &'a str {
    source.split_once(first).expect(first).1.split_once(next).expect(next).0
}

#[test]
fn the_capture_pass_runs_no_eager_peer_operation() {
    let engine = include_str!("../engine.rs");
    // The eager tail is the step's only rank-coupled work outside a graph
    // body: its post closes the layer over the exchange (`tp2_post` waits for
    // the peer's payload). It runs only where segment bodies run.
    let step = body(engine, "fn step_inner(", "/// The step's token embeddings");
    assert!(step.contains("let mode = StepMode::of(self.graph_warming.get(), self.capture_only.get());"));
    assert!(step.contains(
        "if warming {\n                if mode.tail_runs() { tail()?; }\n            } else {\n                self.replay(key, tail)?;"));
    let tail = body(step, "let tail = || -> Result<()> {", "};\n            // The tail is never");
    assert!(tail.contains("self.post_layer("), "the tail is the coupled close");
    // Everything cross-rank else in the step is scheduled through the peer
    // rank's graph bodies, which the capture pass records and never runs.
    assert!(step.contains("self.peer_segment(0, tables, w1, t, cap)?;"));
    assert!(step.contains("self.peer_experts(layer, 0, w1, t, true)?;"));
    let peer_segment = body(engine, "fn peer_segment(", "fn post(");
    assert!(peer_segment.contains("self.peer_post(layer - 1, lane, 0, w1, rows)?;"));
    assert!(peer_segment.contains("self.peer_front(layer, tables, lane, 0, w1, rows, cap)"));
    assert!(peer_segment.contains("self.replay_on(1, key, segment)"));
    let replay = body(engine, "fn replay_on(", "fn peer_segment(");
    let peer_experts = body(engine, "fn peer_experts(", "fn expert_key(");
    assert!(peer_experts.contains("self.replay_on(1, self.tp2_key(layer, t, index), run)"));
    // A capture pass never runs a body it recorded.
    assert_eq!(replay.matches("if self.capture_only.get() { return Ok(()); }").count(), 5);
    // The warm expert dispatch posts no Spark request and lands no wave.
    let experts = body(engine, "fn decode_experts(", "/// Logits of the last `n` of `t` rows");
    let warm = body(experts, "if self.graph_warming.get() || self.capture_only.get() {", "if self.skip_routed {");
    assert!(!warm.contains(".send(") && !warm.contains(".land("));
}

#[test]
fn each_tail_close_has_its_matching_peer_publication() {
    let engine = include_str!("../engine.rs");
    let tp2 = include_str!("../engine/tp2.rs");
    // The tail's TP2 close waits on the layer's ffn slot; the peer's half is
    // published by its own tp2_experts, inside the peer body `peer_experts`
    // replays on rank 1.
    let post = body(tp2, "pub(super) fn tp2_post(", "pub(super) fn prepare_route_checks(");
    assert_eq!(post.matches("exchange.wait(rank, at)?;").count(), 1);
    assert_eq!(post.matches("exchange.recv(rank, at)?").count(), 1);
    let experts = body(tp2, "pub(super) fn tp2_experts(", "pub(super) fn tp2_post(");
    assert!(experts.contains("self.exchange()?.push(rank, slot(layer, true, index), payload.buffer.ptr,"));
    // Under a head split the non-TP2 close publishes and waits the same slot.
    let split = body(engine, "fn post_split(", "fn peer_front(");
    assert_eq!(split.matches("exchange.push(0, ffn, base, count as usize * self.cfg.dim * 2)?;").count(), 1);
    assert_eq!(split.matches("exchange.wait(0, ffn)?;").count(), 1);
    let front = body(engine, "fn peer_front(", "fn peer_experts(");
    assert!(front.contains("exchange.wait(1, entry)?;"));
    assert!(front.contains("exchange.push(1, slot(layer, true, index), lane.shared.buffer.ptr,"));
}
