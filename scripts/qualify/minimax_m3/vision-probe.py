#!/usr/bin/env python3
"""End-to-end MiniMax M3.1 image and video checks against a running server.

Builds deterministic media (Pillow; ffmpeg encodes the clip), asks questions with known
answers, checks the answers, and records TTFT, prompt and cached tokens. TTFT cases put a
fresh nonce in the pixels so every request encodes. The API key arrives on stdin.

  vision-probe.py URL OUT.json [--fixtures DIR] [--cases a,b] [--timing N]

--fixtures is a generate-media-fixtures.py output (the chart and terminal reads).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FONT = ROOT / "rust/crates/cuteafd-bench/assets/fonts/DejaVuSansMono-Bold.ttf"
MODEL = "minimax-m3.1-flash"


def font(size):
    from PIL import ImageFont
    return ImageFont.truetype(str(FONT), size)


def png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def data_url(data, mime):
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def nonce_pixel(image, rnd):
    """Change one corner pixel so the media key, and so the encode, is fresh."""
    image = image.copy()
    image.putpixel((0, 0), tuple(rnd.randrange(256) for _ in range(3)))
    return image


def circles(count, color, size=672, seed=1):
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (size, size), "white")
    draw, rnd, placed = ImageDraw.Draw(image), random.Random(seed), []
    while len(placed) < count:
        x, y = rnd.randrange(70, size - 70), rnd.randrange(70, size - 70)
        if all((x - a) ** 2 + (y - b) ** 2 > 130 ** 2 for a, b in placed):
            placed.append((x, y))
            draw.ellipse((x - 45, y - 45, x + 45, y + 45), fill=color)
    return image


def squares(count, color, triangles=False, size=672, seed=2):
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (size, size), "white")
    draw, rnd, placed = ImageDraw.Draw(image), random.Random(seed), []
    while len(placed) < count:
        x, y = rnd.randrange(80, size - 80), rnd.randrange(80, size - 80)
        if all(abs(x - a) > 150 or abs(y - b) > 150 for a, b in placed):
            placed.append((x, y))
            if triangles:
                draw.polygon([(x, y - 55), (x - 55, y + 45), (x + 55, y + 45)], fill=color)
            else:
                draw.rectangle((x - 50, y - 50, x + 50, y + 50), fill=color)
    return image


def quadrants():
    """1344x1344, four 672x672 tiles: the release resizes it to one 672x672 grid."""
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (1344, 1344))
    draw = ImageDraw.Draw(image)
    for (x, y), color, word in (((0, 0), "#d62728", "ALPHA"), ((672, 0), "#2ca02c", "BRAVO"),
                                ((0, 672), "#1f77b4", "CHARLIE"), ((672, 672), "#ffd700", "DELTA")):
        draw.rectangle((x, y, x + 671, y + 671), fill=color)
        draw.text((x + 336, y + 336), word, font=font(120), fill="black", anchor="mm")
    return image


def banner():
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (2016, 672), "white")
    draw = ImageDraw.Draw(image)
    for x, word in ((336, "APPLE"), (1008, "RIVER"), (1680, "CASTLE")):
        draw.text((x, 336), word, font=font(150), fill="black", anchor="mm")
    return image


def clip_frames(seconds=6, fps=10, size=(640, 480), seed=0):
    """A number per second (1..N) and a red ball moving left to right."""
    from PIL import Image, ImageDraw
    frames = []
    for k in range(seconds * fps):
        image = Image.new("RGB", size, (40, 40, 40))
        draw = ImageDraw.Draw(image)
        draw.text((size[0] // 2, size[1] // 2 - 40), str(1 + k // fps), font=font(200), fill="white", anchor="mm")
        x = 40 + (size[0] - 80) * k // (seconds * fps - 1)
        draw.ellipse((x - 25, size[1] - 90, x + 25, size[1] - 40), fill=(220, 30, 30))
        if seed:
            image.putpixel((0, 0), (seed % 256, seed // 256 % 256, 7))
        frames.append(image)
    return frames


def encode_clip(frames, fps):
    with tempfile.TemporaryDirectory() as tmp:
        for k, frame in enumerate(frames):
            frame.save(f"{tmp}/f{k:04d}.png")
        out = f"{tmp}/clip.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", f"{tmp}/f%04d.png",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out], check=True)
        return Path(out).read_bytes()


class Client:
    def __init__(self, url, key):
        self.url, self.key = url.rstrip("/"), key

    def get(self, path):
        request = urllib.request.Request(self.url + path, headers={"Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    def chat(self, content, max_tokens=200):
        body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
                "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                "thinking": {"type": "disabled"}}
        request = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {self.key}"})
        started, first, usage, text = time.monotonic(), None, None, []
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                for raw in response:
                    line = raw.decode().strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[5:])
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                        if piece and first is None:
                            first = time.monotonic()
                        text.append(delta.get("content") or "")
        except urllib.error.HTTPError as error:
            return {"error": f"HTTP {error.code}: {error.read().decode()[:400]}"}
        end = time.monotonic()
        details = (usage or {}).get("prompt_tokens_details") or {}
        return {"text": "".join(text), "ttft_s": round((first or end) - started, 3), "total_s": round(end - started, 3),
                "prompt_tokens": (usage or {}).get("prompt_tokens"), "completion_tokens": (usage or {}).get("completion_tokens"),
                "cached_tokens": details.get("cached_tokens")}


def image_part(image):
    return {"type": "image_url", "image_url": {"url": data_url(png(image), "image/png")}}


def in_order(text, words):
    positions, at = [], 0
    for word in words:
        found = re.search(re.escape(word), text[at:], re.IGNORECASE)
        if not found:
            return False
        at += found.end()
    return True


def cases(fixtures):
    from PIL import Image
    count7 = circles(7, (220, 30, 30))
    clip = clip_frames()
    yield "chart", [image_part(Image.open(fixtures / "chart0.png")),
                    {"type": "text", "text": "Read the chart. Give the count for each batch."}], \
        lambda t: all(n in t for n in ("12", "28", "19", "35"))
    yield "ocr", [image_part(Image.open(fixtures / "terminal0.png")),
                  {"type": "text", "text": "What command was run, how many tests passed, and what was the exit status?"}], \
        lambda t: "cargo test media" in t and "17" in t
    yield "count", [image_part(count7), {"type": "text", "text": "How many red circles are in this image? Answer with the number only."}], \
        lambda t: re.search(r"\b7\b|seven", t, re.IGNORECASE) is not None
    yield "multi-image", [image_part(squares(3, (30, 80, 220))), image_part(squares(5, (30, 160, 60), triangles=True)),
                          {"type": "text", "text": "How many shapes are in the first image and in the second image, and what "
                           "color are they? Answer as: first: N COLOR SHAPE; second: N COLOR SHAPE."}], \
        lambda t: in_order(t.lower(), ["3", "blue", "5", "green"])
    yield "quadrants", [image_part(quadrants()), {"type": "text", "text": "Read the word in each quadrant: top-left, "
                        "top-right, bottom-left, bottom-right. Answer with the four words in that order."}], \
        lambda t: in_order(t.upper(), ["ALPHA", "BRAVO", "CHARLIE", "DELTA"])
    yield "wide", [image_part(banner()), {"type": "text", "text": "Read the three words from left to right."}], \
        lambda t: in_order(t.upper(), ["APPLE", "RIVER", "CASTLE"])
    question = ("The video shows numbers one after another and a red ball. List the numbers in the order they "
                "appear, then say whether the ball moves left-to-right or right-to-left.")
    yield "video", [{"type": "video_url", "video_url": {"url": data_url(encode_clip(clip, 10), "video/mp4")}},
                    {"type": "text", "text": question}], \
        lambda t: in_order(t, ["1", "2", "3", "4", "5", "6"]) and re.search(r"left[- ]to[- ]right", t, re.I) is not None
    frames = [data_url(png(clip[10 * s]), "image/png") for s in range(6)] + [data_url(png(clip[-1]), "image/png")]
    yield "video-frames", [{"type": "video_url", "video_url": {"frames": frames}},
                           {"type": "text", "text": question}], \
        lambda t: in_order(t, ["1", "2", "3", "4", "5", "6"]) and re.search(r"left[- ]to[- ]right", t, re.I) is not None
    # The same image request again: its prompt should restore from the prefix cache.
    yield "count-repeat", [image_part(count7), {"type": "text", "text": "How many red circles are in this image? Answer with the number only."}], \
        lambda t: re.search(r"\b7\b|seven", t, re.IGNORECASE) is not None


def timing(client, repeats):
    rnd = random.Random(time.time_ns())
    tile, wide = circles(5, (30, 30, 200), seed=5), quadrants()
    rows = []
    for _ in range(repeats):
        for name, content in (
                ("tile-672", [image_part(nonce_pixel(tile, rnd)), {"type": "text", "text": "Describe the image."}]),
                ("quadrants-1344", [image_part(nonce_pixel(wide, rnd)), {"type": "text", "text": "Describe the image."}]),
                ("video-6s-480p", [{"type": "video_url", "video_url": {"url": data_url(
                    encode_clip(clip_frames(seed=rnd.randrange(1, 60000)), 10), "video/mp4")}},
                    {"type": "text", "text": "Describe the video."}])):
            result = client.chat(content, max_tokens=1)
            rows.append(dict(case=name, **{k: result.get(k) for k in ("ttft_s", "prompt_tokens", "cached_tokens", "error")}))
            print(json.dumps(rows[-1]), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("out", type=Path)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--cases", default="")
    parser.add_argument("--timing", type=int, default=0, help="TTFT repeats per media shape (fresh pixels each)")
    args = parser.parse_args()
    client = Client(args.url, sys.stdin.read().strip())
    wanted = {c for c in args.cases.split(",") if c}
    report = {"url": args.url, "started": time.time(), "cases": {}}
    for name, content, check in cases(args.fixtures):
        if wanted and name not in wanted:
            continue
        result = client.chat(content)
        result["pass"] = "error" not in result and bool(check(result["text"]))
        report["cases"][name] = result
        print(json.dumps({"case": name, **{k: result.get(k) for k in ("pass", "ttft_s", "prompt_tokens", "cached_tokens", "error")},
                          "text": result.get("text", "")[:160]}), flush=True)
    if args.timing:
        report["timing"] = timing(client, args.timing)
    report["stats"] = client.get("/v1/stats")
    try:
        report["thermal"] = client.get("/v1/thermal")
    except Exception as error:  # thermal is optional on some builds
        report["thermal"] = {"error": str(error)}
    report["passed"] = sum(c["pass"] for c in report["cases"].values())
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"passed": report["passed"], "of": len(report["cases"]),
                      "media": (report["stats"] or {}).get("media")}), flush=True)
    return 0 if report["passed"] == len(report["cases"]) else 1


if __name__ == "__main__":
    sys.exit(main())
