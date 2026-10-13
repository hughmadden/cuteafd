"""Control law and failsafes of scripts/host/gpu-power-governor against a fake sensor."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "host" / "gpu-power-governor"


def load():
    loader = importlib.machinery.SourceFileLoader("gpu_power_governor", str(SCRIPT))
    spec = importlib.util.spec_from_loader("gpu_power_governor", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


gov = load()
UUIDS = ["GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0", "GPU-95f8f212-9131-df99-fd53-7535965197d7"]


def config(**overrides):
    value = {**gov.DEFAULTS, "hosts": ["raptor"], "gpu_uuids": UUIDS, "ceiling_w": 600, "state_file": "", **overrides}
    gov.validate(value)
    return value


class Card:
    """A first-order thermal model: headroom falls with power above what the cooler sheds."""

    def __init__(self, headroom=50.0, demand_w=600.0, c_per_w=0.09, tau_s=40.0):
        self.ambient_headroom = headroom
        self.headroom = headroom
        self.demand_w = demand_w
        self.c_per_w = c_per_w
        self.tau_s = tau_s

    def step(self, limit_w, dt=1.0):
        power = min(self.demand_w, limit_w)
        steady = self.ambient_headroom - self.c_per_w * power
        self.headroom += (steady - self.headroom) * (1 - math.exp(-dt / self.tau_s))
        return power


def simulate(controller, card, seconds, dt=1.0):
    trace = []
    for _ in range(int(seconds / dt)):
        limit = controller.update(round(card.headroom), dt)
        power = card.step(controller.quantized(), dt)
        trace.append((card.headroom, limit, power))
    return trace


def test_cool_card_stays_at_the_ceiling():
    controller = gov.Controller(config())
    trace = simulate(controller, Card(demand_w=200), 300)
    assert all(limit == 600 for _, limit, _ in trace)
    assert controller.reason == "ceiling"


def test_sustained_full_load_settles_near_the_target_without_crossing_the_hard_margin():
    cfg = config()
    controller = gov.Controller(cfg)
    # Unthrottled, this card would settle at 50 - 0.09 * 600 = -4 C headroom: slowdown and worse.
    trace = simulate(controller, Card(), 1200)
    minimum = min(h for h, _, _ in trace)
    assert minimum > cfg["hard_headroom_c"], minimum
    late = trace[-300:]
    assert all(abs(h - cfg["target_headroom_c"]) < 3 for h, _, _ in late), late[-5:]
    assert cfg["floor_w"] < late[-1][1] < cfg["ceiling_w"]


def test_limit_falls_fast_and_rises_slowly():
    cfg = config()
    controller = gov.Controller(cfg)
    controller.update(40, 1.0)
    for _ in range(10):
        controller.update(12, 1.0)
    falling = controller.limit
    assert cfg["ceiling_w"] - falling > 5 * cfg["up_w_per_s"]
    before = controller.limit
    controller.update(45, 1.0)
    assert controller.limit - before <= cfg["up_w_per_s"] + 1e-9


def test_rising_temperature_starts_pacing_before_the_target():
    cfg = config()
    calm, heating = gov.Controller(cfg), gov.Controller(cfg)
    # 1 C/s of heating toward the target: pacing starts while the level is still 5+ C above it.
    for headroom in [22] * 15:
        calm.update(headroom, 1.0)
    for headroom in range(36, 21, -1):
        heating.update(headroom, 1.0)
    assert calm.limit == cfg["ceiling_w"]
    assert heating.ema > cfg["target_headroom_c"] + 5
    assert heating.limit < cfg["ceiling_w"], "the slope must act before the level reaches the target"


def test_hard_margin_goes_straight_to_the_floor():
    cfg = config()
    controller = gov.Controller(cfg)
    controller.update(30, 1.0)
    assert controller.update(cfg["hard_headroom_c"], 1.0) == cfg["floor_w"]
    assert controller.reason.startswith("hard")


@pytest.mark.parametrize("bad", [None, float("nan")])
def test_a_failed_reading_drops_to_the_floor_and_recovers_slowly(bad):
    cfg = config()
    controller = gov.Controller(cfg)
    controller.update(45, 1.0)
    assert controller.update(bad, 1.0) == cfg["floor_w"]
    assert "sensor" in controller.reason
    controller.update(45, 1.0)
    assert controller.limit <= cfg["floor_w"] + cfg["up_w_per_s"] + 1e-9


def test_quantized_limit_steps_down_and_stays_in_range():
    cfg = config(step_w=5)
    controller = gov.Controller(cfg)
    controller.limit = 327.9
    assert controller.quantized() == 325
    controller.limit = 10
    assert controller.quantized() == cfg["floor_w"]


@pytest.mark.parametrize("override,message", [
    ({"up_w_per_s": 50, "down_w_per_s": 10}, "raise slowly"),
    ({"floor_w": 400, "safe_w": 325}, "floor_w"),
    ({"hard_headroom_c": 20, "target_headroom_c": 15}, "hard_headroom_c"),
    ({"gpu_uuids": ["not-a-uuid"]}, "gpu_uuids"),
    ({"hosts": []}, "hosts"),
])
def test_config_refuses_unsafe_values(override, message):
    with pytest.raises(ValueError, match=message):
        config(**override)


def test_config_file_defaults_keep_todays_cap(tmp_path):
    path = tmp_path / "governor.json"
    path.write_text((ROOT / "scripts/host/gpu-power-governor.example.json").read_text())
    value = gov.read_config(path)
    assert value["ceiling_w"] == value["safe_w"] == 325 and value["floor_w"] == 150
    path.write_text(json.dumps({**json.loads(path.read_text()), "typo": 1}))
    with pytest.raises(ValueError, match="Unknown config keys"):
        gov.read_config(path)


class FakeNvml:
    def __init__(self, headrooms, fail_reads=(), fail_writes=()):
        self.headrooms = headrooms
        self.fail_reads = set(fail_reads)
        self.fail_writes = set(fail_writes)
        self.limits = {}
        self.writes = []

    def device(self, uuid):
        return uuid

    def constraints_w(self, handle):
        return 150.0, 600.0

    def headroom(self, handle):
        if handle in self.fail_reads:
            raise RuntimeError("GPU is lost")
        return self.headrooms[handle]

    def temperature(self, handle):
        return 93 - self.headroom(handle)

    def power_w(self, handle):
        return 300.0

    def set_limit_w(self, handle, watts):
        if handle in self.fail_writes:
            raise RuntimeError("write failed")
        self.limits[handle] = watts
        self.writes.append((handle, watts))


def run_ticks(cfg, nvml, ticks, monkeypatch):
    calls = {"n": 0}

    def wait(_):
        calls["n"] += 1
        if calls["n"] >= ticks:
            gov.STOP.set()
        return gov.STOP.is_set()

    gov.STOP.clear()
    monkeypatch.setattr(gov.STOP, "wait", wait)
    try:
        gov.control(nvml, cfg)
    finally:
        gov.STOP.clear()


def test_daemon_drives_every_gpu_writes_state_and_restores_safe_on_exit(tmp_path, monkeypatch):
    state = tmp_path / "run" / "state.json"
    cfg = config(state_file=str(state), period_s=1.0)
    nvml = FakeNvml({UUIDS[0]: 45, UUIDS[1]: 9})
    run_ticks(cfg, nvml, 3, monkeypatch)
    hot = [w for u, w in nvml.writes if u == UUIDS[1]]
    assert hot[0] < cfg["ceiling_w"], "the hot card is lowered on the first tick"
    assert nvml.limits == {UUIDS[0]: cfg["safe_w"], UUIDS[1]: cfg["safe_w"]}, "exit restores the safe limit"
    final = json.loads(state.read_text())
    assert final["stopped"] and "safe" in final["reason"]


def test_a_lost_gpu_goes_to_the_floor_while_the_other_keeps_running(tmp_path, monkeypatch):
    cfg = config(state_file=str(tmp_path / "state.json"))
    nvml = FakeNvml({UUIDS[0]: 45, UUIDS[1]: 45}, fail_reads={UUIDS[1]})
    seen = []
    original = gov.write_state

    def capture(path, value):
        seen.append(value)
        original(path, value)

    monkeypatch.setattr(gov, "write_state", capture)
    run_ticks(cfg, nvml, 2, monkeypatch)
    running = [s for s in seen if "gpus" in s][0]
    by_uuid = {g["uuid"]: g for g in running["gpus"]}
    assert by_uuid[UUIDS[1]]["limit_w"] == cfg["floor_w"]
    assert by_uuid[UUIDS[0]]["limit_w"] == cfg["ceiling_w"]


def test_a_failed_write_stops_the_loop_and_still_restores(monkeypatch, tmp_path):
    cfg = config(state_file=str(tmp_path / "state.json"))
    nvml = FakeNvml({UUIDS[0]: 45, UUIDS[1]: 45}, fail_writes={UUIDS[0]})
    with pytest.raises(RuntimeError, match="write failed"):
        run_ticks(cfg, nvml, 3, monkeypatch)
    assert nvml.limits.get(UUIDS[1]) == cfg["safe_w"]


def test_out_of_range_config_is_refused_before_any_write(monkeypatch, tmp_path):
    cfg = config(state_file=str(tmp_path / "state.json"), ceiling_w=700, safe_w=325)
    nvml = FakeNvml({UUIDS[0]: 45, UUIDS[1]: 45})
    with pytest.raises(RuntimeError, match="board range"):
        run_ticks(cfg, nvml, 1, monkeypatch)
    assert all(watts == cfg["safe_w"] for _, watts in nvml.writes)
