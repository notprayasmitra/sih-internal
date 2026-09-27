#!/usr/bin/env python3
"""Generate lightweight, dependency-free SVG figures for the experiment report."""
# The SVG strings are intentionally kept inline; line wrapping them obscures the graphics.
# ruff: noqa: E501, E702
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"


def font(size: int):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FAMILIES = ["DDoS", "Infiltration", "Neris", "Rbot"]
F1 = {"LSTM": [0.6319, 0.5685, 0.6982, 0.5274], "JEPA": [0.6856, 0.5970, 0.8315, 0.2428], "Transformer": [0.6358, 0.5730, 0.5983, 0.5122], "TCN": [0.6358, 0.5520, 0.7419, 0.5210]}
FPR = {"LSTM": [0.0000, 0.1076, 0.1870, 0.1693], "JEPA": [0.6120, 0.2687, None, 1.0000], "Transformer": [0.0000, 0.1663, 0.8532, 0.1114], "TCN": [0.0000, 0.0914, 0.2062, 0.0356]}


def esc(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bar_chart(path: Path, title: str, values: dict[str, list[float | None]]) -> None:
    width, height, left, bottom = 900, 500, 90, 85
    plot_w, plot_h = width - left - 25, height - bottom - 55
    colors = ["#2563eb", "#f59e0b", "#7c3aed", "#059669"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#fff"/><text x="{width/2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="700">{esc(title)}</text>']
    for tick in range(6):
        y = height - bottom - tick * plot_h / 5
        parts.append(f'<line x1="{left}" x2="{width-25}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/><text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick/5:.1f}</text>')
    names = list(values)
    group_w = plot_w / len(FAMILIES)
    bar_w = group_w / (len(names) + 1)
    for gi, family in enumerate(FAMILIES):
        gx = left + gi * group_w
        parts.append(f'<text x="{gx+group_w/2:.1f}" y="{height-42}" text-anchor="middle" font-family="sans-serif" font-size="13">{family}</text>')
        for mi, name in enumerate(names):
            value = values[name][gi]
            if value is None:
                continue
            h = value * plot_h
            x = gx + (mi + 0.35) * bar_w
            y = height - bottom - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*.75:.1f}" height="{h:.1f}" fill="{colors[mi]}"/><text x="{x+bar_w*.375:.1f}" y="{y-4:.1f}" text-anchor="middle" font-family="sans-serif" font-size="10">{value:.2f}</text>')
    for i, name in enumerate(names):
        x = left + i * 145
        parts.append(f'<rect x="{x}" y="{height-18}" width="12" height="12" fill="{colors[i]}"/><text x="{x+17}" y="{height-8}" font-family="sans-serif" font-size="12">{name}</text>')
    parts.append("</svg>")
    path.write_text("".join(parts))


def line_chart(path: Path) -> None:
    vals = [1.6709, 1.6611, 0.4211, 0.2094, 0.3293, 0.4352, 0.8948, 1.3763, 4.2932, 2.8201, 6.1516, 5.4542]
    width, height, left, bottom = 760, 420, 65, 60
    pw, ph, ymax = width-left-30, height-bottom-45, 6.5
    points = []
    for i, v in enumerate(vals):
        x = left + i * pw / (len(vals)-1); y = height-bottom-v/ymax*ph; points.append((x,y))
    poly = " ".join(f"{x:.1f},{y:.1f}" for x,y in points)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#fff"/><text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="21" font-weight="700">TCN validation-loss curve</text>']
    for t in range(0, 7):
        y=height-bottom-t/ymax*ph; out.append(f'<line x1="{left}" x2="{width-30}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/><text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{t}</text>')
    out.append(f'<polyline points="{poly}" fill="none" stroke="#2563eb" stroke-width="3"/>')
    for i,(x,y) in enumerate(points):
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2563eb"/><text x="{x:.1f}" y="{height-35}" text-anchor="middle" font-family="sans-serif" font-size="11">{i+1}</text>')
    out.append(f'<text x="{width/2}" y="{height-8}" text-anchor="middle" font-family="sans-serif" font-size="12">epoch</text></svg>')
    path.write_text("".join(out))


def png_bar(path: Path, title: str, values: dict[str, list[float | None]]) -> None:
    image = Image.new("RGB", (2800, 1520), "white")
    draw = ImageDraw.Draw(image)
    draw.text((1400, 50), title, fill="#111827", anchor="ma", font=font(48))
    colors = ["#2563eb", "#f59e0b", "#7c3aed", "#059669"]
    left, top, right, bottom = 240, 200, 2720, 1300
    for tick in range(6):
        y = bottom - tick * (bottom-top) / 5
        draw.line((left, y, right, y), fill="#e5e7eb")
        draw.text((left-30, y), f"{tick/5:.1f}", fill="#374151", anchor="rm", font=font(30))
    group = (right-left) / len(FAMILIES)
    names = list(values)
    bar = group / (len(names)+1)
    for gi, family in enumerate(FAMILIES):
        gx = left + gi * group
        draw.text((gx+group/2, bottom+50), family, fill="#374151", anchor="ma", font=font(32))
        for mi, name in enumerate(names):
            value = values[name][gi]
            if value is None: continue
            x = gx + (mi+.25)*bar; y = bottom - value*(bottom-top)
            draw.rectangle((x, y, x+bar*.7, bottom), fill=colors[mi])
            draw.text((x+bar*.35, y-16), f"{value:.2f}", fill="#111827", anchor="ms", font=font(26))
    for i, name in enumerate(names):
        x = 140 + i*220
        draw.rectangle((x, 1400, x+36, 1436), fill=colors[i])
        draw.text((x+50, 1418), name, fill="#374151", anchor="lm", font=font(30))
    image.save(path)


def png_line(path: Path) -> None:
    vals = [1.6709, 1.6611, .4211, .2094, .3293, .4352, .8948, 1.3763, 4.2932, 2.8201, 6.1516, 5.4542]
    image = Image.new("RGB", (2800, 1400), "white")
    draw = ImageDraw.Draw(image); left, top, right, bottom = 200, 170, 2720, 1180; ymax = 6.5
    draw.text((1400, 50), "TCN validation-loss curve", fill="#111827", anchor="ma", font=font(48))
    for tick in range(7):
        y = bottom - tick/ymax*(bottom-top); draw.line((left,y,right,y), fill="#e5e7eb"); draw.text((left-24,y), str(tick), fill="#374151", anchor="rm", font=font(30))
    points = [(left+i*(right-left)/11, bottom-v/ymax*(bottom-top)) for i,v in enumerate(vals)]
    draw.line(points, fill="#2563eb", width=5)
    for i,(x,y) in enumerate(points): draw.ellipse((x-12,y-12,x+12,y+12), fill="#2563eb"); draw.text((x,bottom+50), str(i+1), fill="#374151", anchor="ma", font=font(28))
    image.save(path)


bar_chart(OUT / "architecture_f1.svg", "Best-observed F1 by attack family", F1)
bar_chart(OUT / "architecture_fpr.svg", "Best-observed false-positive rate by attack family", FPR)
line_chart(OUT / "tcn_validation_loss.svg")
png_bar(OUT / "architecture_f1.png", "Best-observed F1 by attack family", F1)
png_bar(OUT / "architecture_fpr.png", "Best-observed false-positive rate by attack family", FPR)
png_line(OUT / "tcn_validation_loss.png")
print(f"Wrote figures to {OUT}")
