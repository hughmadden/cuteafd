#!/usr/bin/env python3
"""Render MiniMax M3.1 chat-template goldens with Hugging Face's Jinja environment.

The environment mirrors Transformers' `_compile_jinja_template` (sandboxed
Jinja2, trim_blocks, lstrip_blocks, loopcontrols, json.dumps-backed `tojson`,
`raise_exception`); the Rust renderer (cuteafd-api `chat::minimax_m3`) must
reproduce these strings byte for byte. Each case is a template context, so the
goldens stay independent of the request conversion under test.

CPU only; needs jinja2>=3.1:

  .venv/bin/python scripts/qualify/minimax_m3/template-goldens.py \\
    [--template PATH]    # default: the committed fixture
"""
import argparse
import hashlib
import json
import pathlib

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = pathlib.Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "rust/crates/cuteafd-api/src/openai/chat/minimax_m3/fixtures"

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city (°C).",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name, e.g. 台北"},
        "days": {"type": "integer", "minimum": 1}, "units": {"type": "string", "enum": ["c", "f"]}},
        "required": ["city"]}}}
WRITE = {"type": "function", "function": {
    "name": "write_file", "description": "Write \"text\" to a path.\nOverwrites.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "text": {"type": "string"},
        "mode": {"type": "object", "properties": {"append": {"type": "boolean"}}},
        "tags": {"type": "array", "items": {"type": "string"}}},
        "required": ["path", "text"], "additionalProperties": False}}}
REFRESH = {"type": "function", "function": {"name": "refresh"}}


def gen(**context):
    return dict(add_generation_prompt=True, **context)


def call(cid, name, arguments):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": arguments}}


USER = [{"role": "user", "content": "What is 2 + 2?"}]
CASES = {
    "single_user": gen(messages=USER),
    "system_user": gen(messages=[{"role": "system", "content": "You are terse."}, *USER]),
    "developer_user": gen(messages=[{"role": "developer", "content": "Answer in French."}, *USER]),
    "root_and_system": gen(messages=[
        {"role": "root", "content": "Platform rules.", "current_date": "2026-10-23", "current_location": "Paris"},
        {"role": "system", "content": "You are terse."}, *USER]),
    "root_only": gen(messages=[{"role": "root", "content": "Platform rules."}, *USER]),
    "model_identity_kwarg": gen(messages=USER, model_identity="You are Sol."),
    "effort_kwarg": gen(messages=USER, effort="high"),
    "multi_turn_earlier_reasoning": gen(messages=[
        {"role": "user", "content": "Pick a number."},
        {"role": "assistant", "content": " 7 ", "reasoning_content": "Any number works."},
        {"role": "user", "content": "Another?"},
        {"role": "assistant", "content": "Nine.", "reasoning_content": ""},
        {"role": "user", "content": "And one more?"},
        {"role": "assistant", "content": "Eleven."},
        {"role": "user", "content": "Thanks."}]),
    "inline_think_in_content": gen(messages=[
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<mm:think>\nhidden\n</mm:think>\n\nvisible answer\n"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "only closer</mm:think>\nanswer"},
        {"role": "user", "content": "q3"}]),
    "mid_conversation_system": gen(messages=[
        *USER, {"role": "assistant", "content": "4"},
        {"role": "system", "content": "Now be verbose."}, {"role": "user", "content": "And 3 + 3?"}]),
    "images_and_videos": gen(messages=[
        {"role": "system", "content": [{"type": "text", "text": "sys "}, {"type": "text", "text": "parts"}]},
        {"role": "user", "content": [
            {"type": "text", "text": "Look: "}, {"type": "image", "image": "a.png"},
            {"type": "text", "text": " and "}, {"type": "video", "video": "b.mp4"},
            {"type": "image", "image": "c.png"}, {"type": "text", "text": " describe."}]}]),
    "tools_declared": gen(messages=[{"role": "user", "content": "Weather in Taipei?"}], tools=[WEATHER, REFRESH, WRITE]),
    "tool_round_trip": gen(
        messages=[
            {"role": "system", "content": "Use tools when helpful."},
            {"role": "user", "content": "Weather in Taipei and Oslo, then save it."},
            {"role": "assistant", "content": "", "reasoning_content": "Two lookups.", "tool_calls": [
                call("call_a", "get_weather", {"city": "Taipei", "days": 2, "units": "c"}),
                call("call_b", "get_weather", {"city": "Oslo", "days": 1})]},
            {"role": "tool", "tool_call_id": "call_a", "content": "{\"temp\": 31}"},
            {"role": "tool", "tool_call_id": "call_b", "content": "{\"temp\": -3}"},
            {"role": "assistant", "content": "Saving.", "tool_calls": [
                call("call_c", "write_file", {
                    "path": "/tmp/w.txt", "text": "Taipei 31 \"hot\"\nOslo -3 ❄ <b>&</b>",
                    "mode": {"append": False}, "tags": ["天气", "weather"]})]},
            {"role": "tool", "tool_call_id": "call_c", "content": [{"type": "text", "text": "ok"}]},
            {"role": "assistant", "content": "Done: Taipei 31°C, Oslo -3°C.", "reasoning_content": "Report."},
            {"role": "user", "content": "Thanks, and now Paris?"}],
        tools=[WEATHER, WRITE], reasoning_effort="high"),
    "tool_results_with_media": gen(
        messages=[
            {"role": "user", "content": "Refresh twice."},
            {"role": "assistant", "content": "", "tool_calls": [call("r1", "refresh", {}), call("r2", "refresh", {})]},
            {"role": "tool", "tool_call_id": "r1", "content": [
                {"type": "text", "text": "shot "}, {"type": "image", "image": "x.png"}, {"type": "video", "video": "y.mp4"}]},
            {"role": "tool", "tool_call_id": "r2", "content": ""},
            {"role": "user", "content": "Again."}],
        tools=[REFRESH]),
    "tool_without_prior_call_raises": gen(messages=[{"role": "user", "content": "x"},
                                                      {"role": "tool", "tool_call_id": "z", "content": "orphan"}]),
    "arguments_nesting_and_scalars": gen(messages=[
        {"role": "user", "content": "Numbers"},
        {"role": "assistant", "content": "", "tool_calls": [call("n1", "calc", {
            "a": 1e16, "b": 1.5e-05, "c": 0.0001, "d": -0.0, "e": 123456789.125, "f": 12345678901234567890,
            "g": None, "h": [1, 2.5, "x\ty\u0001 "], "i": {"k": "💡", "z": None, "t": True, "f": False},
            "j": [[1, 2], {"deep": [None, "v"]}], "k": [], "l": {}, "m": "plain & <tag>"})]},
        {"role": "tool", "tool_call_id": "n1", "content": "ok"}]),
    "string_content_items": gen(messages=[{"role": "user", "content": ["raw ", {"type": "text", "text": "mapped"}]}]),
    "no_generation_prompt": {"messages": USER + [{"role": "assistant", "content": "4"}], "add_generation_prompt": False},
}
for level in ("low", "medium", "high", "xhigh", "max"):
    CASES[f"effort_{level}"] = gen(messages=[{"role": "system", "content": "Be brief."}, *USER], reasoning_effort=level)
CASES["effort_and_kwarg_precedence"] = gen(messages=USER, reasoning_effort="low", effort="max")


def environment():
    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default=str(FIXTURES / "chat_template.jinja"))
    parser.add_argument("--out", default=str(FIXTURES / "template_goldens.json"))
    args = parser.parse_args()
    source = pathlib.Path(args.template).read_text()
    template = environment().from_string(source)
    goldens = []
    for name, context in CASES.items():
        entry = {"name": name, "context": context}
        try:
            entry["expected"] = template.render(**context)
        except jinja2.exceptions.TemplateError as error:
            entry["error"] = str(error)
        goldens.append(entry)
    pathlib.Path(args.out).write_text(json.dumps(goldens, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {len(goldens)} cases to {args.out} (jinja2 {jinja2.__version__}, "
          f"template sha256 {hashlib.sha256(source.encode()).hexdigest()[:16]})")


if __name__ == "__main__":
    main()
