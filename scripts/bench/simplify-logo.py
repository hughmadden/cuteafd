#!/usr/bin/env python3
"""Compact copies of the cuteafd logo for the benchmark's SVG exports.

assets/brand/*.svg trace pixels (one vertex per pixel step, ~57 KB). Exports
inline the logo, so this keeps every subpath's outline with
Ramer-Douglas-Peucker at EPSILON px, prefixes ids (several logos may share one
document) and writes a nestable <svg> with the source viewBox:

    simplify-logo.py assets/brand/cuteafd-logo-color-dark.svg rust/crates/cuteafd-bench/assets/logo-dark.svg
"""
import re
import sys

EPSILON = 0.9


def rdp(points, epsilon):
    if len(points) < 3:
        return points
    (x0, y0), (x1, y1) = points[0], points[-1]
    dx, dy = x1 - x0, y1 - y0
    norm = (dx * dx + dy * dy) ** 0.5
    best, index = -1.0, 0
    for i in range(1, len(points) - 1):
        px, py = points[i]
        d = abs(dy * px - dx * py + x1 * y0 - y1 * x0) / norm if norm else ((px - x0) ** 2 + (py - y0) ** 2) ** 0.5
        if d > best:
            best, index = d, i
    if best <= epsilon:
        return [points[0], points[-1]]
    return rdp(points[: index + 1], epsilon)[:-1] + rdp(points[index:], epsilon)


def simplify(d):
    out = []
    for sub in re.findall(r"M[^M]+", d):
        numbers = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", sub)]
        points = list(zip(numbers[0::2], numbers[1::2]))
        closed = points + [points[0]]
        # Split the ring at its farthest point so RDP keeps both halves.
        far = max(range(len(points)), key=lambda i: (points[i][0] - points[0][0]) ** 2 + (points[i][1] - points[0][1]) ** 2)
        kept = rdp(closed[: far + 1], EPSILON)[:-1] + rdp(closed[far:], EPSILON)[:-1]
        out.append("M" + " ".join(f"{x:g} {y:g}" for x, y in kept[:1]) + "L" + " ".join(f"{x:g} {y:g}" for x, y in kept[1:]) + "Z")
    return "".join(out)


def main(source, target):
    text = open(source).read()
    view = re.search(r'viewBox="([^"]+)"', text).group(1)
    body = text[text.index(">", text.index("<svg")) + 1 : text.rindex("</svg>")]
    body = re.sub(r"<title[^<]*</title>|<desc[^<]*</desc>", "", body)
    body = re.sub(r'\sd="([^"]+)"', lambda m: f' d="{simplify(m.group(1))}"', body)
    body = body.replace('id="', 'id="cuteafd-').replace("url(#", "url(#cuteafd-")
    body = "".join(line.strip() for line in body.splitlines())
    out = f'<svg viewBox="{view}" preserveAspectRatio="xMinYMid meet">{body}</svg>\n'
    open(target, "w").write(out)
    print(f"{target}: {len(text)} -> {len(out)} bytes")


if __name__ == "__main__":
    main(*sys.argv[1:])
