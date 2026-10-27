"""The shared prefill panel primes each shape outside the measured interval."""
from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[2] / "rust/crates/cuteafd-bench/src/panels/speed.rs").read_text()
PREFILL = SOURCE.split("impl Panel for Prefill {", 1)[1].split("impl Retained {", 1)[0]


def test_each_prefill_shape_has_one_fresh_cold_warmup_before_measurement() -> None:
    loop = PREFILL.split("for (i, &target) in lengths.iter().enumerate() {", 1)[1]
    warm = loop.index("let warm = sized_prompt(ctx, 2000 + i as u64, target,")
    timed = loop.index("let text = sized_prompt(ctx, 1000 + i as u64, target,")
    mark = loop.index("let fabric_mark = ctx.client.fabric_mark();")
    cold = loop.index("let cold = ctx.client.chat(")
    assert warm < timed < mark < cold
    request = loop[warm:timed]
    assert request.count("ctx.client.chat(") == 1
    assert "plain(ctx.client, &warm, 1)" in request
    assert "Some(ProbeSpec { cold: true, ..ProbeSpec::default() })" in request
    assert '.with_context(|| format!("{target}-token prefill warm-up"))?;' in request


def test_prefill_estimate_includes_the_extra_full_length_request() -> None:
    estimate = PREFILL.split("fn estimate_s", 1)[1].split("fn run", 1)[0]
    assert "rates.seconds(l as f64 * 2.15, 3.0)" in estimate


def test_sized_prompts_always_start_with_a_fresh_nonce() -> None:
    common = (Path(__file__).resolve().parents[2] / "rust/crates/cuteafd-bench/src/panels/common.rs").read_text()
    sized = common.split("pub fn sized_prompt(", 1)[1].split("/// One request", 1)[0]
    assert 'format!("[{}] Read the notes below.' in sized
    assert "nonce(), corpus(seed, words)" in sized
