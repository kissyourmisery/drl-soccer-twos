#!/usr/bin/env python3
import csv
import math
import os
import statistics
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
RUN1_CSV = REPO_DIR / "ray_results" / "PPO_reward_cpu_slurm" / "PPO_Soccer_876b4_00000_0_2026-04-23_00-11-41" / "progress.csv"
RUN2_CSV = REPO_DIR / "ray_results" / "PPO_option2_selfplay_dense_cpu_slurm" / "PPO_Soccer_ba9af_00000_0_2026-04-23_19-39-54" / "progress.csv"
OUT_SVG = REPO_DIR / "report_artifacts" / "reward_curves_submission1_vs_submission2.svg"


def load_series(csv_path: Path, y_col: str):
    points = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = int(float(row["timesteps_total"]))
                y = float(row[y_col])
            except Exception:
                continue
            if not math.isfinite(y):
                continue
            points.append((x, y))
    return points


def rolling_mean(points, window: int):
    if len(points) < window:
        return []
    out = []
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


def polyline(points, color, width=2, dash=None, opacity=1.0):
    if not points:
        return ""
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="{width}"'
        f'{dash_attr} opacity="{opacity}" points="{pts}" />'
    )


def hline(y, left, right, color="#d6d6d6"):
    return (
        f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{right:.1f}" y2="{y:.1f}" '
        f'stroke="{color}" stroke-width="1" />'
    )


def vline(x, top, bottom, color="#efefef"):
    return (
        f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{bottom:.1f}" '
        f'stroke="{color}" stroke-width="1" />'
    )


def build_panel(
    title,
    left,
    top,
    width,
    height,
    x_max,
    y_min,
    y_max,
    series,
    x_ticks_millions,
    y_ticks,
):
    right = left + width
    bottom = top + height
    parts = []
    parts.append(
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" '
        f'fill="white" stroke="#cfcfcf" />'
    )
    parts.append(
        f'<text x="{left}" y="{top - 10}" font-family="Arial" font-size="16" '
        f'font-weight="bold">{title}</text>'
    )

    for x_m in x_ticks_millions:
        x = left + (x_m / (x_max / 1_000_000.0)) * width
        parts.append(vline(x, top, bottom))
        parts.append(
            f'<text x="{x:.1f}" y="{bottom + 20}" text-anchor="middle" '
            f'font-family="Arial" font-size="11">{x_m:g}M</text>'
        )

    for yv in y_ticks:
        y = top + height - ((yv - y_min) / (y_max - y_min)) * height
        parts.append(hline(y, left, right))
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="Arial" font-size="11">{yv:.2f}</text>'
        )

    for s in series:
        scaled = scale_points(
            s["points"], left, top, width, height, 0, x_max, y_min, y_max
        )
        parts.append(
            polyline(
                scaled,
                s["color"],
                width=s.get("width", 2),
                dash=s.get("dash"),
                opacity=s.get("opacity", 1.0),
            )
        )

    parts.append(
        f'<text x="{left + width/2:.1f}" y="{bottom + 38}" text-anchor="middle" '
        f'font-family="Arial" font-size="12">Timesteps (millions)</text>'
    )
    parts.append(
        f'<g transform="translate({left - 52},{top + height/2}) rotate(-90)">'
        f'<text text-anchor="middle" font-family="Arial" font-size="12">Reward</text>'
        f"</g>"
    )

    # Legend
    legend_x = right - 280
    legend_y = top + 14
    parts.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="265" height="{22*len(series)+10}" '
        f'fill="#fafafa" stroke="#dddddd" />'
    )
    for i, s in enumerate(series):
        y = legend_y + 20 + i * 22
        dash_attr = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        parts.append(
            f'<line x1="{legend_x+10}" y1="{y}" x2="{legend_x+48}" y2="{y}" '
            f'stroke="{s["color"]}" stroke-width="{s.get("width",2)}"{dash_attr} />'
        )
        parts.append(
            f'<text x="{legend_x+56}" y="{y+4}" font-family="Arial" '
            f'font-size="11">{s["label"]}</text>'
        )

    return "\n".join(parts)


def main():
    run1_raw = load_series(RUN1_CSV, "episode_reward_mean")
    run1_roll = rolling_mean(run1_raw, 100)

    run2_default_raw = load_series(RUN2_CSV, "policy_reward_mean/default")
    run2_default_roll = rolling_mean(run2_default_raw, 50)
    run2_episode_raw = load_series(RUN2_CSV, "episode_reward_mean")

    os.makedirs(OUT_SVG.parent, exist_ok=True)

    width = 1400
    height = 900
    panel_w = 620
    panel_h = 320
    p1_left = 90
    p1_top = 90
    p2_left = 710
    p2_top = 90

    # Panel 1 scales
    p1_x_max = run1_raw[-1][0]
    p1_y_vals = [y for _, y in run1_raw] + [y for _, y in run1_roll]
    p1_y_min = min(p1_y_vals) - 0.1
    p1_y_max = max(p1_y_vals) + 0.1

    # Panel 2 scales
    p2_x_max = run2_default_raw[-1][0]
    p2_y_vals = (
        [y for _, y in run2_default_raw]
        + [y for _, y in run2_default_roll]
        + [y for _, y in run2_episode_raw]
    )
    p2_y_min = min(p2_y_vals) - 0.1
    p2_y_max = max(p2_y_vals) + 0.1

    p1 = build_panel(
        title="Submission 1 (Simple PPO + Reward Scale): episode_reward_mean",
        left=p1_left,
        top=p1_top,
        width=panel_w,
        height=panel_h,
        x_max=p1_x_max,
        y_min=p1_y_min,
        y_max=p1_y_max,
        series=[
            {
                "label": "Raw episode_reward_mean",
                "points": run1_raw,
                "color": "#9bbbe6",
                "width": 1.5,
                "opacity": 0.8,
            },
            {
                "label": "Rolling mean (window=100)",
                "points": run1_roll,
                "color": "#2158a8",
                "width": 2.5,
            },
        ],
        x_ticks_millions=[0, 2, 4, 6, 8, 9],
        y_ticks=[
            round(p1_y_min + i * (p1_y_max - p1_y_min) / 6, 2) for i in range(7)
        ],
    )

    p2 = build_panel(
        title="Submission 2 (Self-Play + Dense Shaping): reward metrics",
        left=p2_left,
        top=p2_top,
        width=panel_w,
        height=panel_h,
        x_max=p2_x_max,
        y_min=p2_y_min,
        y_max=p2_y_max,
        series=[
            {
                "label": "Raw policy_reward_mean/default",
                "points": run2_default_raw,
                "color": "#9fcdb5",
                "width": 1.5,
                "opacity": 0.85,
            },
            {
                "label": "Rolling mean default (window=50)",
                "points": run2_default_roll,
                "color": "#1d7f49",
                "width": 2.5,
            },
            {
                "label": "Raw episode_reward_mean (all-agent aggregate)",
                "points": run2_episode_raw,
                "color": "#7f7f7f",
                "width": 1.3,
                "dash": "5 4",
                "opacity": 0.85,
            },
        ],
        x_ticks_millions=[0, 1, 2, 3, 4],
        y_ticks=[
            round(p2_y_min + i * (p2_y_max - p2_y_min) / 6, 2) for i in range(7)
        ],
    )

    notes_y = 500
    notes = [
        "Notes for report:",
        "- Submission 1: reward climbs steadily after early negative phase and reaches high late-stage values.",
        "- Submission 2: all-agent episode reward spikes early then stabilizes; default-policy reward is the better training metric.",
        "- Use win-rate results (baseline/random/TA) together with reward curves for performance claims.",
    ]
    note_svg = []
    for i, line in enumerate(notes):
        note_svg.append(
            f'<text x="90" y="{notes_y + i*26}" font-family="Arial" font-size="18">{line}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="#f7f8fb" />
<text x="90" y="48" font-family="Arial" font-size="26" font-weight="bold">
Reward-vs-Steps Curves for Final Project Submissions
</text>
{p1}
{p2}
{''.join(note_svg)}
</svg>
"""
    OUT_SVG.write_text(svg, encoding="utf-8")
    print(f"Wrote: {OUT_SVG}")


if __name__ == "__main__":
    main()
