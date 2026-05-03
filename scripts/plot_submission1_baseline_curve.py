#!/usr/bin/env python3
import csv
import math
import os
import statistics
import subprocess
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
BASELINE_CSV = REPO_DIR / "baseline_results" / "submission1_reward_timeseries.csv"
OUT_SVG = REPO_DIR / "baseline_results" / "reward_curve_submission1_baseline.svg"
OUT_PNG = REPO_DIR / "baseline_results" / "reward_curve_submission1_baseline.png"


def load_series(csv_path: Path):
    points = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = int(float(row["timesteps_total"]))
                y = float(row["episode_reward_mean"])
            except Exception:
                continue
            if not math.isfinite(y):
                continue
            points.append((x, y))
    return points


def rolling_mean(points, window: int):
    out = []
    if len(points) < window:
        return out
    for i in range(window - 1, len(points)):
        out.append(
            (
                points[i][0],
                statistics.mean(v for _, v in points[i - window + 1 : i + 1]),
            )
        )
    return out


def scale_points(points, left, top, width, height, x_min, x_max, y_min, y_max):
    scaled = []
    x_span = (x_max - x_min) if x_max != x_min else 1.0
    y_span = (y_max - y_min) if y_max != y_min else 1.0
    for x, y in points:
        sx = left + ((x - x_min) / x_span) * width
        sy = top + height - ((y - y_min) / y_span) * height
        scaled.append((sx, sy))
    return scaled


def polyline(points, color, width=2.0, dash=None, opacity=1.0):
    if not points:
        return ""
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="{width}"'
        f'{dash_attr} opacity="{opacity}" points="{pts}" />'
    )


def shutil_which(cmd: str):
    from shutil import which

    return which(cmd)


def main():
    raw = load_series(BASELINE_CSV)
    smooth = rolling_mean(raw, 100)

    if not raw:
        raise RuntimeError(f"No data loaded from: {BASELINE_CSV}")

    os.makedirs(OUT_SVG.parent, exist_ok=True)

    width, height = 1200, 760
    left, top = 110, 120
    plot_w, plot_h = 980, 470
    bottom = top + plot_h

    x_min, x_max = 0, raw[-1][0]
    y_values = [y for _, y in raw] + [y for _, y in smooth]
    y_min = min(y_values) - 0.1
    y_max = max(y_values) + 0.1

    raw_scaled = scale_points(raw, left, top, plot_w, plot_h, x_min, x_max, y_min, y_max)
    smooth_scaled = scale_points(
        smooth, left, top, plot_w, plot_h, x_min, x_max, y_min, y_max
    )

    x_ticks_m = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y_ticks = [round(y_min + i * (y_max - y_min) / 6, 2) for i in range(7)]

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f7f8fb" />')
    parts.append(
        '<text x="110" y="60" font-family="Arial" font-size="28" font-weight="bold">'
        "Reward Curve for Baseline PPO (Submission 1)"
        "</text>"
    )
    parts.append(
        '<text x="110" y="92" font-family="Arial" font-size="15" fill="#444">'
        "Metric: episode_reward_mean"
        "</text>"
    )

    parts.append(
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="white" stroke="#cfcfcf" />'
    )

    for x_m in x_ticks_m:
        x = left + (x_m / (x_max / 1_000_000.0)) * plot_w
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#efefef" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{bottom + 24}" text-anchor="middle" font-family="Arial" font-size="12">{x_m:g}M</text>'
        )

    for yv in y_ticks:
        y = top + plot_h - ((yv - y_min) / (y_max - y_min)) * plot_h
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#d6d6d6" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{yv:.2f}</text>'
        )

    parts.append(polyline(raw_scaled, "#9bbbe6", width=1.5, opacity=0.9))
    parts.append(polyline(smooth_scaled, "#2158a8", width=3.0))

    parts.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{bottom + 52}" text-anchor="middle" font-family="Arial" font-size="14">'
        "Timesteps"
        "</text>"
    )
    parts.append(
        f'<g transform="translate({left - 68},{top + plot_h / 2}) rotate(-90)">'
        '<text text-anchor="middle" font-family="Arial" font-size="14">episode_reward_mean</text>'
        "</g>"
    )

    legend_x, legend_y = left + plot_w - 320, top + 18
    parts.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="295" height="70" fill="#fafafa" stroke="#dddddd" />'
    )
    parts.append(
        f'<line x1="{legend_x + 12}" y1="{legend_y + 22}" x2="{legend_x + 52}" y2="{legend_y + 22}" stroke="#9bbbe6" stroke-width="1.5" />'
    )
    parts.append(
        f'<text x="{legend_x + 60}" y="{legend_y + 26}" font-family="Arial" font-size="12">Raw episode_reward_mean</text>'
    )
    parts.append(
        f'<line x1="{legend_x + 12}" y1="{legend_y + 48}" x2="{legend_x + 52}" y2="{legend_y + 48}" stroke="#2158a8" stroke-width="3" />'
    )
    parts.append(
        f'<text x="{legend_x + 60}" y="{legend_y + 52}" font-family="Arial" font-size="12">Rolling mean (window=100)</text>'
    )

    parts.append("</svg>")

    OUT_SVG.write_text("\n".join(parts), encoding="utf-8")

    converter = shutil_which("rsvg-convert")
    if converter:
        subprocess.run(
            [converter, "-w", "1600", "-h", "1000", str(OUT_SVG), "-o", str(OUT_PNG)],
            check=True,
        )

    print(f"Wrote SVG: {OUT_SVG}")
    if OUT_PNG.exists():
        print(f"Wrote PNG: {OUT_PNG}")


if __name__ == "__main__":
    main()
