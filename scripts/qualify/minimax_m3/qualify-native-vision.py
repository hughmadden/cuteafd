#!/usr/bin/env python3
"""Live MiniMax M3.1 image and video semantics, identity, replay and exact-hit qualification.

--fixtures-dir holds scripts/bench/generate-media-fixtures.py output for both sets (the
fidelity chart and terminal windows, and --set semantics: counted shapes, a 1344x1344
four-quadrant image, a 2016x672 banner and numbered clips). Expected answers are ordered
terms taken from the fixture manifests. --timing N adds streamed TTFT rows: one fresh
pixel per image request, a distinct clip per video request, and text-only prompts of
matching lengths, plus the `/v1/thermal` window they ran in. The API key arrives on
stdin; the report records fixture hashes.
"""
import argparse
import base64
import copy
import io
import json
from pathlib import Path
import random
import sys
import time
import urllib.error
import urllib.request

MODEL = "minimax-m3.1-flash"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--fixtures-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session", default="minimax-vision-live-v1")
    parser.add_argument("--timing", type=int, default=0, help="TTFT repeats per media shape")
    args = parser.parse_args()
    assert not args.output.exists(), "choose a new output path to preserve evidence"
    key = sys.stdin.read().strip()
    root = args.fixtures_dir
    semantics = {f["id"]: f for f in json.loads((root / "semantics.json").read_text())["fixtures"]}
    fidelity = {f["path"]: f for f in json.loads((root / "fixtures.json").read_text())["fixtures"]}
    report = dict(base_url=args.base_url, fixtures=[dict(path=f["path"], sha256=f["sha256"])
                  for f in list(semantics.values()) + list(fidelity.values())], cases=[], timing=[], passed=False)

    def save():
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    def url(path):
        mime = {"png": "image/png", "mp4": "video/mp4"}[path.rsplit(".", 1)[1]]
        return f"data:{mime};base64," + base64.b64encode((root / path).read_bytes()).decode()

    def image(path):
        return dict(type="image_url", image_url=dict(url=url(path)))

    def post(body, stream=False):
        request = urllib.request.Request(args.base_url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        return urllib.request.urlopen(request, timeout=600)

    def call(name, body, contains):
        """Ordered terms (a tuple is alternatives) must appear in the answer, in order."""
        record = dict(name=name, expected_ordered_terms=contains)
        report["cases"].append(record)
        started = time.monotonic()
        try:
            with post(body) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            record.update(passed=False, error=f"HTTP {error.code}: {error.read().decode()[:500]}")
            save()
            print(json.dumps(dict(name=name, error=record["error"])), flush=True)
            return None
        record.update(response=result, elapsed_seconds=round(time.monotonic() - started, 3))
        text = (result["choices"][0]["message"].get("content") or "").lower()
        offset, ok = 0, result["choices"][0]["finish_reason"] == "stop"
        for word in contains:
            alternatives = (word,) if isinstance(word, str) else word
            matches = [(text.find(term.lower(), offset), term) for term in alternatives
                       if text.find(term.lower(), offset) >= offset]
            if not matches:
                ok = False
                break
            found, term = min(matches)
            offset = found + len(term)
        record["passed"] = ok
        save()
        print(json.dumps(dict(name=name, passed=ok, text=text[:240], usage=result["usage"]), ensure_ascii=False), flush=True)
        return result

    def cached(result):
        usage = result["usage"]
        return (usage.get("prompt_tokens_details") or {}).get("cached_tokens", usage.get("prompt_cache_hit_tokens", 0))

    def check(name, condition, detail):
        report["cases"].append(dict(name=name, passed=bool(condition), detail=detail))
        save()
        print(json.dumps(dict(name=name, passed=bool(condition), detail=detail)), flush=True)

    def exact(name, first, repeated):
        if first and repeated:
            same = first["choices"][0]["message"] == repeated["choices"][0]["message"]
            hit, prompt = cached(repeated), repeated["usage"]["prompt_tokens"]
            check(name, same and hit == prompt, dict(same_message=same, cached_tokens=hit, prompt_tokens=prompt))

    def body(parts, question, session=None, max_tokens=160):
        return dict(model=MODEL, messages=[dict(role="user", content=[
            dict(type="text", text=(session or args.session) + ". " + question)] + parts)],
            thinking=dict(type="disabled"), temperature=0, max_tokens=max_tokens)

    shapes = lambda f: [str(f["count"]), f["color"]]
    count = "How many shapes are in this image, and what color are they? Answer like: N COLOR SHAPES."
    circles = semantics["circles7"]
    seed = call("circles", body([image("circles7.png")], count), shapes(circles))
    exact("circles-exact", seed, call("circles-repeat", body([image("circles7.png")], count), shapes(circles)))
    call("changed-image-squares", body([image("squares3.png")], count), shapes(semantics["squares3"]))
    pair = ("Describe the first image and then the second image, in that order: for each, the number of "
            "shapes and their color.")
    call("squares-triangles", body([image("squares3.png"), image("triangles5.png")], pair),
         shapes(semantics["squares3"]) + shapes(semantics["triangles5"]))
    # MiniMax restores prompt-end and turn-end snapshots only: a shared first image is not a hit.
    call("squares-circles", body([image("squares3.png"), image("circles7.png")], pair),
         shapes(semantics["squares3"]) + shapes(circles))
    call("triangles-squares-reordered", body([image("triangles5.png"), image("squares3.png")], pair),
         shapes(semantics["triangles5"]) + shapes(semantics["squares3"]))
    follow = copy.deepcopy(body([image("circles7.png")], count))
    if seed:
        follow["messages"] += [dict(role="assistant", content=seed["choices"][0]["message"]["content"]),
                               dict(role="user", content="What shape are they? Answer in one word.")]
        resumed = call("completed-turn-resume", follow, [("circle",)])
        if resumed:
            check("completed-turn-resume-hit", cached(resumed) >= seed["usage"]["total_tokens"] - 1,
                  dict(cached_tokens=cached(resumed), first_total_tokens=seed["usage"]["total_tokens"]))
    call("quadrants-1344", body([image("quadrants.png")], "Read the word in each quadrant: top-left, top-right, "
         "bottom-left, bottom-right. Answer with the four words in that order."), semantics["quadrants"]["words"])
    call("banner-2016x672", body([image("banner.png")], "Read the three words from left to right."),
         semantics["banner"]["words"])
    call("chart", body([image("chart0.png")], "Read the chart. Give the count for each batch."),
         ["12", "28", "19", "35"])
    call("terminal", body([image("terminal0.png")], "What command was run, how many tests passed, and what was "
         "the exit status?"), ["cargo test media", "17"])
    clip = semantics["clip"]
    question = ("The video shows numbers one after another and a red ball. List the numbers in the order they "
                "appear, then say whether the ball moves left-to-right or right-to-left.")
    terms = [str(n) for n in clip["numbers"]] + [("left to right", "left-to-right")]
    video = dict(type="video_url", video_url=dict(url=url("clip.mp4")))
    first = call("clip", body([video], question), terms)
    exact("clip-exact", first, call("clip-repeat", body([video], question), terms))
    frames = dict(type="video_url", video_url=dict(frames=[url(name) for name in clip["frames"]], source_fps=1))
    call("clip-frames", body([frames], question), terms)
    if args.timing:
        start = thermal(args.base_url)["now_s"]
        timing(args, post, report, save, root, url)
        # Whether, why and how much the GPUs throttled while the timing rows ran.
        report["thermal"] = thermal(args.base_url, start).get("window")
        for gpu in (report["thermal"] or {}).get("gpus", []):
            print(json.dumps(dict(gpu=gpu.get("index"), throttle_s=gpu.get("throttle_s"),
                                  max_c=(gpu.get("temperature_c") or {}).get("max"))), flush=True)
    report["passed"] = all(case.get("passed") for case in report["cases"])
    save()
    print(json.dumps(dict(passed=report["passed"], failed=[c["name"] for c in report["cases"] if not c.get("passed")])),
          flush=True)
    return 0 if report["passed"] else 1


def thermal(base_url, start=None):
    """The server's NVML view (`/v1/thermal`, no key); with `start`, the window since then."""
    query = f"?from={start}" if start is not None else ""
    with urllib.request.urlopen(base_url + "/v1/thermal" + query, timeout=30) as response:
        return json.load(response)


def timing(args, post, report, save, root, url):
    from PIL import Image
    rnd = random.Random(time.time_ns())

    def fresh(path):
        """One corner pixel changes, so the media key and the encode are new."""
        image = Image.open(root / path).convert("RGB")
        image.putpixel((0, 0), tuple(rnd.randrange(256) for _ in range(3)))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return dict(type="image_url", image_url=dict(url="data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()))

    def text(words):
        vocabulary = "the quick brown fox jumps over a lazy dog while seven grey wolves watch the misty ridge".split()
        return dict(type="text", text=" ".join(rnd.choice(vocabulary) for _ in range(words)) + " Describe the text.")

    def stream(name, parts):
        body = dict(model=MODEL, messages=[dict(role="user", content=parts)], max_tokens=1, temperature=0,
                    stream=True, stream_options=dict(include_usage=True), thinking=dict(type="disabled"))
        started, first, usage = time.monotonic(), None, None
        with post(body) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if first is None and ((delta.get("content") or "") + (delta.get("reasoning_content") or "")):
                        first = time.monotonic()
        end = time.monotonic()
        row = dict(case=name, ttft_s=round((first or end) - started, 3), prompt_tokens=(usage or {}).get("prompt_tokens"),
                   cached_tokens=((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"))
        report["timing"].append(row)
        save()
        print(json.dumps(row), flush=True)

    names = next(f for f in json.loads((root / "semantics.json").read_text())["fixtures"] if f["id"] == "clip")["frames"]
    for k in range(args.timing):
        stream("tile-672", [fresh("circles7.png"), dict(type="text", text="Describe the image.")])
        stream("quadrants-1344", [fresh("quadrants.png"), dict(type="text", text="Describe the image.")])
        stream("text-600", [text(570)])
        clip = dict(type="video_url", video_url=dict(url=url(f"clip-timing{k % 3}.mp4")))
        stream("clip-480p" if k < 3 else "clip-480p-cached", [clip, dict(type="text", text="Describe the video.")])
        # The same sampled frames as PNGs: no ffmpeg, the same tokens.
        frames = [fresh(names[0])["image_url"]["url"]] + [url(name) for name in names[1:]]
        stream("clip-frames", [dict(type="video_url", video_url=dict(frames=frames, source_fps=1)),
                               dict(type="text", text="Describe the video.")])
        stream("text-1700", [text(1630)])


if __name__ == "__main__":
    sys.exit(main())
