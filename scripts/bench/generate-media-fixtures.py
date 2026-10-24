#!/usr/bin/env python3
"""Generate first-party vision fixtures (the eight fidelity windows, or the semantics set), without external assets."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/reference"))
from fidelity_windows import canonical, validate_public_text

FONT = ROOT / "rust/crates/cuteafd-bench/assets/fonts/DejaVuSansMono.ttf"
CODE = ["native/shared/cuda/norm.cu", "scripts/build/assert-build-filesystem.py"]


def generate(out: Path, root: Path = ROOT):
    import PIL
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.font_manager import FontProperties

    out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(FONT), 16)
    title = ImageFont.truetype(str(FONT), 21)
    records = []
    for i in range(8):
        image = Image.new("RGB", (512, 512), "#fcfcfb")
        draw = ImageDraw.Draw(image)
        kind = ("code", "terminal", "chart", "diagram")[i // 2]
        variant = i % 2
        sources = []
        if kind == "code":
            path = CODE[variant]
            content = (root / path).read_bytes()
            lines = content.decode().splitlines()[:18]
            text = "\n".join(line[:46] for line in lines)
            validate_public_text(text)
            sources.append({"path": path, "sha256": hashlib.sha256(content).hexdigest()})
            draw.text((16, 16), "REPOSITORY CODE", font=title, fill="#0b0b0b")
            draw.multiline_text((16, 60), text, font=font, fill="#0b0b0b", spacing=6)
            question = "Read the code screenshot. Describe the function, inputs and one invariant visible in it."
        elif kind == "terminal":
            command = "cargo test media" if variant == 0 else "python -m pytest fixtures"
            count = 17 if variant == 0 else 24
            text = f"$ {command}\n\ncollecting tests ...\n\ntest resize_exact ... ok\ntest hash_binding ... ok\ntest cold_restore ... ok\n\nresult: {count} passed; 0 failed\n\nexit status: 0"
            draw.text((16, 16), "SYNTHETIC TERMINAL LOG", font=title, fill="#0b0b0b")
            draw.multiline_text((16, 70), text, font=font, fill="#0b0b0b", spacing=10)
            question = "Read the terminal screenshot. State the command, passed-test count, failed-test count and exit status. Explain the evidence."
        elif kind == "chart":
            values = [12, 28, 19, 35] if variant == 0 else [21, 14, 32, 26]
            names = ["Alpha", "Beta", "Gamma", "Delta"]
            fig = Figure(figsize=(5.12, 5.12), dpi=100, facecolor="#fcfcfb")
            ax = fig.subplots()
            prop = FontProperties(fname=str(FONT), size=11)
            ax.bar(names, values, color="#2a78d6", width=0.55, hatch="/", zorder=3)
            ax.set_ylim(0, 40)
            ax.set_title("Synthetic batch counts", fontproperties=prop, pad=16, color="#0b0b0b")
            ax.set_ylabel("Count", fontproperties=prop, color="#0b0b0b")
            ax.set_facecolor("#fcfcfb")
            ax.yaxis.grid(True, color="#deded8", linewidth=0.6, zorder=0)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(prop)
                label.set_color("#52514e")
            for edge in ("top", "right"):
                ax.spines[edge].set_visible(False)
            # Visible values plus a numeric table provide a color-independent read.
            for x, value in enumerate(values):
                ax.text(x, value + 0.7, str(value), ha="center", fontproperties=prop, color="#0b0b0b")
            fig.subplots_adjust(bottom=0.28, left=0.16, right=0.96, top=0.86)
            fig.text(0.12, 0.09, "Counts: " + ", ".join(f"{n}={v}" for n, v in zip(names, values)),
                     fontproperties=FontProperties(fname=str(FONT), size=9), color="#0b0b0b")
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            image = Image.frombytes("RGBA", canvas.get_width_height(), bytes(canvas.buffer_rgba())).convert("RGB")
            question = "Read the chart. Give the count for each batch, identify the largest and smallest, and compute the total."
        else:
            nodes = ["INPUT", "DECODE", "ENCODE", "CACHE"] if variant == 0 else ["REQUEST", "VERIFY", "RESTORE", "SCORE"]
            draw.text((16, 16), "SYNTHETIC PIPELINE", font=title, fill="#0b0b0b")
            for j, node in enumerate(nodes):
                y = 65 + j * 108
                draw.rounded_rectangle((120, y, 392, y + 60), radius=8, outline="#0b0b0b", width=2)
                draw.text((140, y + 20), f"{j + 1}. {node}", font=title, fill="#0b0b0b")
                if j < 3:
                    draw.line((256, y + 60, 256, y + 99), fill="#52514e", width=2)
                    draw.polygon([(250, y + 93), (262, y + 93), (256, y + 103)], fill="#52514e")
            question = "Read the pipeline diagram. List the four numbered stages in order and explain how the arrows connect them."
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=False, compress_level=9)
        data = buffer.getvalue()
        path = f"{kind}{variant}.png"
        dest = out / path
        if dest.exists() and dest.read_bytes() != data:
            raise ValueError("fixture versions are immutable; choose a new output directory")
        dest.write_bytes(data)
        records.append({"id": f"vision{i:02d}", "kind": kind, "path": path,
                        "sha256": hashlib.sha256(data).hexdigest(), "width": 512, "height": 512,
                        "question": question, "sources": sources})
    manifest = {"schema": "cuteafd.media.fixtures/1", "font_sha256": hashlib.sha256(FONT.read_bytes()).hexdigest(),
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "libraries": {"pillow": PIL.__version__, "matplotlib": matplotlib.__version__},
                "fixtures": records, "quick_windows": ["vision00", "vision04"]}
    path = out / "fixtures.json"
    content = canonical(manifest) + b"\n"
    if path.exists() and path.read_bytes() != content:
        raise ValueError("fixture manifest changed; choose a new output directory")
    path.write_bytes(content)
    return manifest


BOLD = ROOT / "rust/crates/cuteafd-bench/assets/fonts/DejaVuSansMono-Bold.ttf"


def generate_semantics(out: Path):
    """Live-API semantics media with known content, recorded in semantics.json: counted
    shapes, a 1344x1344 four-quadrant image (four 672^2 tiles' worth; MiniMax shrinks it to
    one 576-token grid), a 2016x672 banner, and numbered clips with a moving ball (H.264,
    plus the clip's 1 fps samples as frames). Clip encoding needs ffmpeg with libx264."""
    import random
    import subprocess
    import tempfile
    import PIL
    from PIL import Image, ImageDraw, ImageFont

    out.mkdir(parents=True, exist_ok=True)
    records = []

    def keep(path, data, **facts):
        dest = out / path
        if dest.exists() and dest.read_bytes() != data:
            raise ValueError("fixture versions are immutable; choose a new output directory")
        dest.write_bytes(data)
        records.append({"path": path, "sha256": hashlib.sha256(data).hexdigest(), **facts})

    def png(image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=False, compress_level=9)
        return buffer.getvalue()

    def shapes(count, color, shape, seed):
        image = Image.new("RGB", (672, 672), "white")
        draw, rnd, placed = ImageDraw.Draw(image), random.Random(seed), []
        while len(placed) < count:
            x, y = rnd.randrange(80, 592), rnd.randrange(80, 592)
            if all((x - a) ** 2 + (y - b) ** 2 > 140 ** 2 for a, b in placed):
                placed.append((x, y))
                if shape == "circle":
                    draw.ellipse((x - 45, y - 45, x + 45, y + 45), fill=color)
                elif shape == "square":
                    draw.rectangle((x - 50, y - 50, x + 50, y + 50), fill=color)
                else:
                    draw.polygon([(x, y - 55), (x - 55, y + 45), (x + 55, y + 45)], fill=color)
        return image

    for name, count, color, rgb, shape, seed in (("circles7", 7, "red", (220, 30, 30), "circle", 1),
                                                ("squares3", 3, "blue", (30, 80, 220), "square", 2),
                                                ("triangles5", 5, "green", (30, 160, 60), "triangle", 3)):
        keep(name + ".png", png(shapes(count, rgb, shape, seed)), id=name, kind="shapes",
             width=672, height=672, count=count, color=color, shape=shape)
    quadrants = Image.new("RGB", (1344, 1344))
    draw = ImageDraw.Draw(quadrants)
    words = ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]
    for (x, y), color, word in zip(((0, 0), (672, 0), (0, 672), (672, 672)),
                                   ("#d62728", "#2ca02c", "#1f77b4", "#ffd700"), words):
        draw.rectangle((x, y, x + 671, y + 671), fill=color)
        draw.text((x + 336, y + 336), word, font=ImageFont.truetype(str(BOLD), 120), fill="black", anchor="mm")
    keep("quadrants.png", png(quadrants), id="quadrants", kind="tiles", width=1344, height=1344,
         words=words, colors=["red", "green", "blue", "yellow"])
    banner = Image.new("RGB", (2016, 672), "white")
    draw = ImageDraw.Draw(banner)
    words = ["APPLE", "RIVER", "CASTLE"]
    for x, word in zip((336, 1008, 1680), words):
        draw.text((x, 336), word, font=ImageFont.truetype(str(BOLD), 150), fill="black", anchor="mm")
    keep("banner.png", png(banner), id="banner", kind="wide", width=2016, height=672, words=words)

    def clip(background, seconds=6, fps=10, size=(640, 480)):
        """A number per second (1..N) and a red ball moving left to right."""
        frames = []
        for k in range(seconds * fps):
            image = Image.new("RGB", size, background)
            draw = ImageDraw.Draw(image)
            draw.text((size[0] // 2, size[1] // 2 - 40), str(1 + k // fps), font=ImageFont.truetype(str(BOLD), 200),
                      fill="white", anchor="mm")
            x = 40 + (size[0] - 80) * k // (seconds * fps - 1)
            draw.ellipse((x - 25, size[1] - 90, x + 25, size[1] - 40), fill=(220, 30, 30))
            frames.append(image)
        with tempfile.TemporaryDirectory() as tmp:
            for k, frame in enumerate(frames):
                frame.save(f"{tmp}/f{k:04d}.png")
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-framerate", str(fps),
                            "-i", f"{tmp}/f%04d.png", "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p",
                            "-crf", "18", f"{tmp}/clip.mp4"], check=True)
            return frames, Path(f"{tmp}/clip.mp4").read_bytes()

    frames, data = clip((40, 40, 40))
    samples = list(range(0, 60, 10)) + [59]  # the serving sampler's 1 fps indices, last frame kept
    names = []
    for index in samples:
        names.append(f"clip-{index:02d}.png")
        keep(names[-1], png(frames[index]), id=f"clip-{index:02d}", kind="clip-frame", width=640, height=480)
    keep("clip.mp4", data, id="clip", kind="clip", width=640, height=480, fps=10, seconds=6,
         numbers=list(range(1, 7)), motion="left to right", frames=names, frame_indices=samples)
    for k, background in enumerate([(40, 40, 70), (70, 40, 40), (40, 70, 40)]):
        keep(f"clip-timing{k}.mp4", clip(background)[1], id=f"clip-timing{k}", kind="clip",
             width=640, height=480, fps=10, seconds=6)
    version = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True).stdout.split("\n")[0]
    manifest = {"schema": "cuteafd.media.semantics/1", "font_sha256": hashlib.sha256(BOLD.read_bytes()).hexdigest(),
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "libraries": {"pillow": PIL.__version__, "ffmpeg": version}, "fixtures": records}
    path = out / "semantics.json"
    content = canonical(manifest) + b"\n"
    if path.exists() and path.read_bytes() != content:
        raise ValueError("fixture manifest changed; choose a new output directory")
    path.write_bytes(content)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--set", choices=["fidelity", "semantics"], default="fidelity",
                   help="fidelity: the eight first-party windows; semantics: live-API image/video checks")
    args = p.parse_args()
    result = generate(args.out) if args.set == "fidelity" else generate_semantics(args.out)
    print(f"{args.out}: {len(result['fixtures'])} deterministic first-party {args.set} fixtures")


if __name__ == "__main__":
    main()
