#!/usr/bin/env python3
"""Make an additive owned UI/OCR/chart G6 set; never modify fidelity fixtures."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/reference"))
from fidelity_windows import canonical

FONT = ROOT / "rust/crates/cuteafd-bench/assets/fonts/DejaVuSansMono.ttf"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ui_html(kind):
    font = base64.b64encode(FONT.read_bytes()).decode()
    style = f"""@font-face{{font-family:Fixture;src:url(data:font/ttf;base64,{font})}}
*{{box-sizing:border-box}}html,body{{margin:0;width:512px;height:512px;overflow:hidden}}
body{{font:16px Fixture;background:#e8edf1;color:#172b38;padding:20px}}
main{{background:#fff;border:2px solid #172b38;height:472px}}
header{{background:#172b38;color:#fff;padding:16px;font-size:20px}}
section{{padding:18px}}h2{{font-size:18px;margin:0 0 18px}}
.row{{display:flex;justify-content:space-between;margin:20px 0}}
.value{{border:1px solid #526776;background:#f3f5f7;padding:6px 10px}}
button{{font:16px Fixture;border:1px solid #172b38;background:#fff;padding:10px 16px}}
.primary{{background:#2168a5;color:#fff}}footer{{padding:16px;background:#e8edf1}}
.tabs{{padding:12px;background:#e8edf1}}.tab{{padding:8px;border:1px solid #526776}}
.active{{background:#fff;border-bottom:3px solid #2168a5}}pre{{font:15px Fixture;line-height:1.8;margin:0}}
.status{{background:#2168a5;color:#fff;padding:12px;font-size:14px}}
"""
    if kind == "settings":
        body = """<main><header>Editor Settings</header><section><h2>Display</h2>
<div class="row"><span>Theme</span><span class="value">Light</span></div>
<div class="row"><span>Font size</span><span class="value">16 px</span></div>
<div class="row"><span>Word wrap</span><span class="value">On</span></div>
<div class="row"><span>Auto save</span><span class="value">Off</span></div>
</section><footer><button>Cancel</button> <button class="primary">Apply</button></footer></main>"""
    else:
        body = """<main><header>Local Code Workspace</header><div class="tabs">
<span class="tab active">parser.py</span> <span class="tab">tests.py</span></div>
<section><h2>Files</h2><div>parser.py</div><div>tests.py</div><hr>
<pre>def parse_count(text):
    return int(text)

assert parse_count("17") == 17</pre></section>
<footer><button class="primary">Run Tests</button> <button>Format</button></footer>
<div class="status">Ready | Python 3.12 | Ln 4, Col 1</div></main>"""
    return '<!doctype html><html><head><meta charset="utf-8"><title>Owned UI Fixture</title><style>' + style + '</style></head><body>' + body + '</body></html>'


def render(html, png, browser):
    with tempfile.TemporaryDirectory(prefix="browser-", dir=png.parent) as profile:
        subprocess.run([browser, "--headless", "--disable-gpu", "--hide-scrollbars",
            "--no-first-run", "--no-default-browser-check", "--disable-background-networking",
            "--force-device-scale-factor=1", "--window-size=512,512", "--virtual-time-budget=5000",
            "--user-data-dir=" + profile, "--screenshot=" + str(png.resolve()), html.resolve().as_uri()],
            check=True, timeout=45, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    from PIL import Image
    with Image.open(png) as image:
        if image.size != (512, 512):
            raise ValueError("UI screenshot dimensions differ")


def generate(source, out, browser):
    if out.exists():
        raise ValueError("G6 output must be new")
    manifest = json.loads((source / "fixtures.json").read_text())
    fixtures, questions = [], []
    definitions = {
        "vision00": [("Which header filename is in the include directive?", "common.h"),
                     ("What is the CUDA kernel function name?", "rmsnorm_f32_kernel"),
                     ("What synchronization function is visible?", "__syncthreads")],
        "vision01": [("What interpreter is named in the first line?", "python3"),
                     ("What module is imported immediately after json?", "subprocess"),
                     ("What uppercase variable names the unsafe filesystem types?", "UNSAFE_TYPES")],
        "vision02": [("What exact command follows the dollar sign?", "cargo test media"),
                     ("How many tests passed? Answer with one integer.", "17"),
                     ("What is the exit status? Answer with one integer.", "0")],
        "vision03": [("What exact command follows the dollar sign?", "python -m pytest fixtures"),
                     ("How many tests passed? Answer with one integer.", "24"),
                     ("How many tests failed? Answer with one integer.", "0")],
        "vision04": [("What is the count for Beta? Answer with one integer.", "28"),
                     ("Which batch has the largest count?", "Delta"),
                     ("What is the total of all four counts? Answer with one integer.", "94")],
        "vision05": [("What is the count for Gamma? Answer with one integer.", "32"),
                     ("Which batch has the smallest count?", "Beta"),
                     ("What is the total of all four counts? Answer with one integer.", "93")],
        "vision06": [("How many rounded rectangles enclose numbered stages? Answer with one integer.", "4"),
                     ("What color is the stage text?", "black")],
        "ui-settings": [("What is the current Theme value?", "Light"),
                        ("What is the label of the blue action button?", "Apply")],
        "ui-ide": [("What filename is shown on the active tab?", "parser.py"),
                   ("What word begins the blue status bar?", "Ready")],
    }
    out.mkdir(parents=True)
    for entry in manifest["fixtures"]:
        if entry["id"] not in definitions:
            continue
        data = (source / entry["path"]).read_bytes()
        if sha(data) != entry["sha256"]:
            raise ValueError("source fixture seal differs")
        (out / entry["path"]).write_bytes(data)
        fixtures.append({k: entry[k] for k in ("id", "kind", "path", "sha256", "width", "height")})
    for kind in ("settings", "ide"):
        html = out / ("ui-" + kind + ".html")
        html.write_text(ui_html(kind), encoding="ascii")
        png = html.with_suffix(".png")
        render(html, png, browser)
        fixtures.append({"id": "ui-" + kind, "kind": "ui", "path": png.name,
                         "sha256": sha(png.read_bytes()), "width": 512, "height": 512,
                         "html_sha256": sha(html.read_bytes())})
    for fixture in fixtures:
        for i, (prompt, answer) in enumerate(definitions[fixture["id"]]):
            questions.append({"id": fixture["id"] + "-q" + str(i + 1), "fixture_id": fixture["id"],
                "category": {"code": "ocr", "terminal": "ocr", "chart": "chart_values",
                             "diagram": "shapes_colors", "ui": "ui_labels"}[fixture["kind"]],
                "prompt": prompt + " Answer only the requested value; no explanation.",
                "answer": answer, "match": "casefold_whitespace_exact"})
    result = {"schema": "cuteafd.media.g6/1", "version": "g6-v1", "owned_content": True,
              "source_manifest_sha256": sha((source / "fixtures.json").read_bytes()),
              "generator_sha256": sha(Path(__file__).read_bytes()), "font_sha256": sha(FONT.read_bytes()),
              "browser": subprocess.check_output([browser, "--version"], text=True).strip(),
              "fixtures": fixtures, "questions": questions}
    if len(fixtures) != 9 or len(questions) != 24:
        raise ValueError("G6 must contain nine images and 24 exact questions")
    (out / "g6.json").write_bytes(canonical(result) + b"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--browser", default=shutil.which("google-chrome") or shutil.which("chromium"))
    args = parser.parse_args()
    if not args.browser:
        parser.error("--browser must name an installed Chrome/Chromium renderer")
    result = generate(args.source_fixtures, args.out, args.browser)
    print(f"{args.out}: {len(result['fixtures'])} owned images, {len(result['questions'])} exact questions")


if __name__ == "__main__":
    main()
