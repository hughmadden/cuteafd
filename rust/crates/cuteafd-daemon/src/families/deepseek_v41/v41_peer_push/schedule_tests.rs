//! One decode lane's device-ordered schedule on the dual layout, queued in the
//! order the code queues it, checked by `shared::peer_split::order::happens`.
//!
//! The push sites need no parity slots, acknowledgements or scratch guards
//! because each layer's FFN output gates the next layer's input: every next
//! write of a reused destination follows its last read through the flags and
//! the stage chain. These tests keep that true. A future "GPU1 a layer
//! ahead" pipeline (or anything else that lets a layer's remote work run
//! before the previous layer's FFN finished) fails `remote_work_must_follow
//! _the_previous_layer` until such sites get parity slots.
use crate::shared::peer_split::order::happens::{Name, Schedule, Violation};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layer {
    /// Routed and shared experts TP2 on both RTX, attention on GPU0.
    Tp2,
    /// Remote experts; attention on GPU0, reduce on GPU1, result returned.
    SparkLow,
    /// Remote experts; attention and reduce on GPU1.
    SparkHigh,
}

#[derive(Clone, Copy, Debug, Default)]
struct Options {
    /// The converted sites push (`CUTEAFD_V41_PEER_EXCHANGE=push`).
    push: bool,
    /// Negative control: leave out the edge with this name.
    drop: Option<&'static str>,
    /// Negative control: the remote shared half of every layer after the
    /// first reads an input of its own, not the layer's (layers pipelined).
    independent_remote: bool,
}

struct Lane {
    s: Schedule,
    o: Options,
    layer: usize,
}

const LAYERS: [Layer; 6] = [Layer::Tp2, Layer::Tp2, Layer::SparkLow, Layer::SparkLow, Layer::SparkHigh, Layer::SparkHigh];
/// The first layer on GPU1 (the hop lands before it).
const BOUNDARY: usize = 4;

fn leak(text: String) -> Name { Box::leak(text.into_boxed_str()) }

impl Lane {
    fn new(o: Options) -> Self { Self { s: Schedule::default(), o, layer: 0 } }
    fn keep(&self, edge: &str) -> bool { self.o.drop != Some(edge) }
    fn label(&self, what: &str) -> String { format!("layer {} {what}", self.layer) }
    // The stage chain: one head event, re-recorded by every finish.
    fn join(&mut self, stream: Name) { self.s.wait_event(stream, "head"); }
    fn finish(&mut self, stream: Name) { self.s.wait_event(stream, "head").record(stream, "head"); }
    /// `chain::finish_after_flag`: the receiver adopts the head, no merge.
    fn adopt(&mut self, stream: Name) { self.s.record(stream, "head"); }
    /// `ScratchGuard::release`: the head's fence stream records the guard.
    fn release(&mut self, guard: Name) { self.s.wait_event("guard fence", "head").record("guard fence", guard); }
    fn work(&mut self, stream: Name, what: &str, reads: &[Name], writes: &[Name]) {
        let label = self.label(what);
        self.s.work(stream, label, reads, writes);
    }
    fn push(&mut self, stream: Name, flag: Name, reads: &[Name], writes: &[Name]) {
        let label = self.label(&format!("{flag} push"));
        self.s.push(stream, flag, label, reads, writes);
    }
    fn wait(&mut self, stream: Name, flag: Name) {
        if self.keep(leak(format!("{flag} wait"))) { self.s.wait_flag(stream, flag); }
    }

    /// One layer: attention, the FFN by `kind`, then the block finish.
    fn layer(&mut self, kind: Layer) {
        let gpu1 = self.layer >= BOUNDARY;
        let (att, x, values, router) = if gpu1 { ("att1", "x1", "values1", ["wire1", "ids1", "routing1"]) }
            else { ("att0", "x0", "values0", ["wire0", "ids0", "routing0"]) };
        if self.layer > 0 {
            // `layer_schedule`: the host waits for the previous layer's fence.
            self.s.host_sync(if (self.layer - 1) % 2 == 0 { "fence0" } else { "fence1" });
        }
        if self.layer == BOUNDARY { self.hop(); }
        self.join(att);
        self.work(att, "attention and router", &[x], &[values, router[0], router[1], router[2]]);
        self.finish(att);
        // `execute_tp4`: the staging fence once attention joined the chain.
        self.s.wait_event("fence", "head").record("fence", if self.layer % 2 == 0 { "fence0" } else { "fence1" });
        match kind {
            Layer::Tp2 => self.tp2(),
            Layer::SparkLow | Layer::SparkHigh => self.spark(gpu1),
        }
        let output = match kind { Layer::Tp2 | Layer::SparkLow => "tp2 output0", Layer::SparkHigh => "reduce output1" };
        self.join(att);
        self.work(att, "block finish", &[output, x], &[x]);
        self.finish(att);
        self.layer += 1;
    }

    /// `tp2_ffn::Execution::execute` (A2 pulled; A3 and A4 by mode).
    fn tp2(&mut self) {
        // The TP2 FFN's own scratch guard (A2 still pulls in Stage 1).
        self.s.wait_event("tp2 stream0", "tp2 released").wait_event("tp2 stream1", "tp2 released");
        self.join("tp2 stream0");
        if self.keep("tp2 stream1 join") { self.join("tp2 stream1"); }
        self.work("tp2 stream1", "A2 pull", &["values0", "wire0", "ids0", "routing0"],
            &["peer1 values", "peer1 wire", "peer1 ids", "peer1 routing"]);
        // Routed: OwnerPair::execute, owner = GPU0.
        self.s.record("tp2 stream0", "rank0 input").wait_event("rank0", "rank0 input");
        self.work("rank0", "routed rank0", &["wire0", "ids0", "routing0"], &["partial0"]);
        self.s.record("tp2 stream1", "rank1 input").wait_event("rank1", "rank1 input");
        self.work("rank1", "routed rank1", &["peer1 wire", "peer1 ids", "peer1 routing"], &["partial1"]);
        self.s.record("rank0", "local ready0");
        if self.o.push {
            self.push("rank1", "A3", &["partial1"], &["owner staging0"]);
            self.wait("owner transfer0", "A3");
        } else {
            self.s.record("rank1", "remote ready1").wait_event("owner transfer0", "remote ready1");
            self.join("owner transfer0");
            self.work("owner transfer0", "A3 pull", &["partial1"], &["owner staging0"]);
        }
        self.s.wait_event("owner transfer0", "local ready0");
        self.work("owner transfer0", "routed reduce", &["owner staging0", "partial0"], &["routed output0"]);
        self.finish("owner transfer0");
        // Shared: SharedWave::execute, destination = GPU0.
        self.shared(0, ["values0", "peer1 values"]);
        self.join("tp2 stream0");
        self.work("tp2 stream0", "TP2 add", &["routed output0", "shared output0"], &["tp2 output0"]);
        self.finish("tp2 stream0");
        if !self.o.push { self.release("shared released0"); }
        self.release("tp2 released");
    }

    /// The shared expert's two halves reduced on `destination`
    /// (`v41_backbone_shared::tp2::Execution::execute`); `inputs` per GPU.
    fn shared(&mut self, destination: usize, inputs: [Name; 2]) {
        let guards = !self.o.push;
        if guards { self.s.wait_event("shared rank0", "rank released").wait_event("shared rank1", "rank released"); }
        let remote = 1 - destination;
        let rank = |gpu: usize| if gpu == 0 { "shared rank0" } else { "shared rank1" };
        let tp2 = |gpu: usize| if gpu == 0 { "tp2 stream0" } else { "tp2 stream1" };
        let out = |gpu: usize| if gpu == 0 { "shared rank output0" } else { "shared rank output1" };
        for gpu in [0, 1] {
            let independent = self.o.independent_remote && gpu == remote && self.layer > 0;
            let event = if gpu == 0 { "shared input0" } else { "shared input1" };
            self.s.record(tp2(gpu), event);
            if !independent { self.s.wait_event(rank(gpu), event); }
            let input = if independent { "pipelined input" } else { inputs[gpu] };
            self.work(rank(gpu), "shared rank", &[input], &[out(gpu)]);
        }
        let (transfer, staging, output, flag) = if destination == 0 {
            ("shared transfer0", "shared staging0", "shared output0", "A4")
        } else { ("shared transfer1", "shared staging1", "shared output1", "A6") };
        let ready = if destination == 0 { "shared ready0" } else { "shared ready1" };
        self.s.record(rank(destination), ready);
        if self.o.push {
            self.push(rank(remote), flag, &[out(remote)], &[staging]);
            self.wait(transfer, flag);
        } else {
            let produced = if remote == 0 { "shared produced0" } else { "shared produced1" };
            self.s.record(rank(remote), produced).wait_event(transfer, produced);
            self.s.wait_event(transfer, if destination == 0 { "shared released0" } else { "shared released1" });
            self.work(transfer, "shared pull", &[out(remote)], &[staging]);
        }
        self.s.wait_event(transfer, ready);
        self.work(transfer, "shared add", &[out(destination), staging], &[output]);
        self.finish(transfer);
        if guards { self.release("rank released"); }
    }

    /// A remote layer: shared TP2 reduced on GPU1, the Spark reduce there,
    /// and below the boundary the result returned to GPU0 (A7).
    fn spark(&mut self, gpu1: bool) {
        let (local, values) = if gpu1 { (1, "values1") } else { (0, "values0") };
        let remote = 1 - local;
        let (tp2_local, tp2_remote) = if gpu1 { ("tp2 stream1", "tp2 stream0") } else { ("tp2 stream0", "tp2 stream1") };
        let peer_values = if gpu1 { "peer0 values" } else { "peer1 values" };
        // `execute_shared_on` (A5 still pulls in Stage 1).
        self.join(tp2_local);
        if self.keep("tp2 remote join") { self.join(tp2_remote); }
        self.work(tp2_remote, "A5 pull", &[values], &[peer_values]);
        let mut inputs = [values; 2];
        inputs[remote] = peer_values;
        self.shared(1, inputs);
        // `finish_values`: land the wave and reduce on the transport GPU.
        self.join("reduce1");
        self.work("reduce1", "land and reduce", &["shared output1"], &["reduce output1"]);
        self.finish("reduce1");
        if !self.o.push { self.release("shared released1"); }
        if !gpu1 {
            // `return_result` (A7).
            if self.o.push {
                self.join("tp2 stream1");
                self.push("tp2 stream1", "A7", &["reduce output1"], &["tp2 output0"]);
                self.wait("tp2 stream0", "A7");
                self.adopt("tp2 stream0");
            } else {
                self.join("tp2 stream0");
                self.work("tp2 stream0", "A7 pull", &["reduce output1"], &["tp2 output0"]);
                self.finish("tp2 stream0");
            }
        }
    }

    /// `advance` across the boundary (A1): layer 19's block output into
    /// GPU1's block input.
    fn hop(&mut self) {
        if self.o.push {
            self.join("hop sender0");
            self.push("hop sender0", "A1", &["x0"], &["x1"]);
            self.wait("hop landing1", "A1");
            self.adopt("hop landing1");
        } else {
            self.join("hop landing1");
            self.work("hop landing1", "A1 pull", &["x0"], &["x1"]);
            self.finish("hop landing1");
        }
    }

    /// The chained greedy head (A8, A9), the download and the pass drain.
    fn head(&mut self) {
        self.join("normalize1");
        self.work("normalize1", "final norm", &["x1"], &["normalized1"]);
        self.finish("normalize1");
        if self.o.push {
            self.join("head rank1");
            self.push("head rank1", "A8", &["normalized1"], &["head input0"]);
            self.work("head rank1", "rank1 copy and projection", &["normalized1"], &["head input1", "winners1"]);
            self.finish("head rank1");
            self.wait("head rank0", "A8");
            self.work("head rank0", "rank0 projection", &["head input0"], &["winners0"]);
            self.push("head rank0", "A9", &["winners0"], &["merge remote1"]);
            self.join("merge1");
            self.wait("merge1", "A9");
            self.work("merge1", "merge", &["merge remote1", "winners1"], &["merged1"]);
            self.finish("merge1");
        } else {
            self.join("head rank1");
            self.work("head rank1", "rank1 copy and projection", &["normalized1"], &["head input1", "winners1"]);
            self.join("head rank0");
            self.work("head rank0", "A8 pull and rank0 projection", &["normalized1"], &["head input0", "winners0"]);
            self.finish("head rank0");
            self.finish("head rank1");
            self.join("merge1");
            self.work("merge1", "A9 pulls and merge", &["winners0", "winners1"], &["merge remote1", "merged1"]);
            self.finish("merge1");
        }
        self.join("download1");
        self.work("download1", "greedy download", &["merged1"], &[]);
        self.s.record("download1", "downloaded").host_sync("downloaded");
        // `drain_chain` before the next pass; the next pass embeds into x0.
        self.s.host_sync("head");
        self.layer = 0;
        self.embed();
    }

    /// The next pass's first stage: the embedding into GPU0's block input.
    fn embed(&mut self) {
        self.join("embed0");
        self.work("embed0", "embedding", &[], &["x0"]);
        self.finish("embed0");
    }

    fn pass(&mut self) {
        for kind in LAYERS { self.layer(kind); }
        self.head();
    }
}

fn schedule(o: Options) -> Schedule {
    let mut lane = Lane::new(o);
    lane.embed();
    lane.pass();
    lane.pass();
    lane.s
}

fn race(result: Result<(), Violation>) -> (Name, String, String) {
    match result {
        Err(Violation::Race { buffer, first, second }) => (buffer, first, second),
        other => panic!("expected a race, got {other:?}"),
    }
}

#[test]
fn pulled_decode_lane_is_ordered() {
    assert_eq!(schedule(Options::default()).check(), Ok(()));
}

#[test]
fn pushed_decode_lane_is_ordered_without_the_shared_guards() {
    assert_eq!(schedule(Options { push: true, ..Options::default() }).check(), Ok(()));
}

/// Every flag wait carries a reuse or a read; the checker finds each one
/// missing, on the buffer the scratch guards or the pull used to protect.
#[test]
fn pushed_lane_without_a_flag_wait_races() {
    for (edge, buffer) in [("A3 wait", "owner staging0"), ("A4 wait", "shared staging0"),
        ("A6 wait", "shared staging1"), ("A7 wait", "tp2 output0"), ("A1 wait", "x1"),
        ("A8 wait", "head input0"), ("A9 wait", "merge remote1")] {
        let (found, first, second) = race(schedule(Options { push: true, drop: Some(edge), ..Options::default() }).check());
        assert_eq!(found, buffer, "{edge}: {first} -> {second}");
    }
}

/// The reuse argument: each layer's remote half follows the previous layer's
/// readers only because it consumes the layer's own input. Remote work that
/// does not (a pipelined layer) overwrites the shared staging the previous
/// layer still reads, in pull mode behind its guard but not in push mode.
#[test]
fn remote_work_must_follow_the_previous_layer() {
    let pushed = Options { push: true, independent_remote: true, ..Options::default() };
    let (buffer, first, second) = race(schedule(pushed).check());
    assert!(buffer.starts_with("shared staging") || buffer.starts_with("shared rank output"),
        "{buffer}: {first} -> {second}");
    assert!(first.contains("layer 0") || first.contains("layer 2") || first.contains("layer 4"), "{first}");
}

/// The chain edge the pulls relied on is still checked where pulls remain.
#[test]
fn pulled_lane_without_its_remote_join_races() {
    let (buffer, ..) = race(schedule(Options { drop: Some("tp2 stream1 join"), ..Options::default() }).check());
    assert_eq!(buffer, "values0");
}
