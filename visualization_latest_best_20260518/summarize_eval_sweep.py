#!/usr/bin/env python3
"""Summarize D4RT eval sweep outputs and select best runs per metric."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METRICS = [
    ("depth_S_abs_rel", "min"),
    ("depth_SS_abs_rel", "min"),
    ("pointcloud_l1", "min"),
    ("pose_ate", "min"),
    ("pose_rpe_trans", "min"),
    ("pose_rpe_rot", "min"),
    ("pose_auc_30", "max"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize eval sweep summary.json files.")
    parser.add_argument("--root", required=True, help="Sweep root containing one summary.json per run.")
    parser.add_argument("--output-md", default=None, help="Optional markdown output path.")
    parser.add_argument("--output-json", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_runs(root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        protocol = summary.get("protocol", {})
        mean = summary.get("mean_metrics", {})
        run = {
            "name": summary_path.parent.name,
            "summary_path": str(summary_path),
            "checkpoint": summary.get("checkpoint", ""),
            "num_scenes": summary.get("num_scenes"),
            "num_failed_scenes": summary.get("num_failed_scenes"),
            "paper_aligned": protocol.get("paper_aligned"),
            "paper_metric_settings": protocol.get("paper_metric_settings"),
            "paper_density_settings": protocol.get("paper_density_settings"),
            "depth_confidence_quantile": protocol.get("depth_confidence_quantile"),
            "pointcloud_confidence_quantile": protocol.get("pointcloud_confidence_quantile"),
            "pose_confidence_threshold": protocol.get("pose_confidence_threshold"),
            "pose_confidence_quantile": protocol.get("pose_confidence_quantile"),
            "pose_solver": protocol.get("pose_solver"),
            "pose_mode": protocol.get("pose_mode"),
            "pose_grid_h": protocol.get("pose_grid_h"),
            "pose_grid_w": protocol.get("pose_grid_w"),
            "pose_weight_mode": protocol.get("pose_weight_mode"),
            "depth_stride": protocol.get("depth_stride"),
            "pointcloud_stride": protocol.get("pointcloud_stride"),
            "max_pointcloud_points": protocol.get("max_pointcloud_points"),
            "mean_metrics": mean,
        }
        runs.append(run)
    return runs


def metric_value(run: dict[str, Any], metric: str) -> float | None:
    return finite_float(run.get("mean_metrics", {}).get(metric))


def select_best(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_rows: list[dict[str, Any]] = []
    for metric, direction in METRICS:
        candidates = [(metric_value(run, metric), run) for run in runs]
        candidates = [(value, run) for value, run in candidates if value is not None]
        if not candidates:
            continue
        def sort_key(item: tuple[float, dict[str, Any]]) -> tuple[float, float, str]:
            value, run = item
            score = value if direction == "min" else -value
            # If two runs are numerically tied, prefer the less filtered and
            # less diagnostic configuration because it keeps more coverage.
            penalty = 0.0
            penalty += finite_float(run.get("depth_confidence_quantile")) or 0.0
            penalty += finite_float(run.get("pointcloud_confidence_quantile")) or 0.0
            penalty += finite_float(run.get("pose_confidence_threshold")) or 0.0
            if run.get("pose_solver") not in {None, "umeyama"}:
                penalty += 0.1
            if run.get("paper_metric_settings") is False:
                penalty += 1.0
            return score, penalty, str(run.get("name", ""))

        value, run = min(candidates, key=sort_key)
        best_rows.append({"metric": metric, "direction": direction, "value": value, "run": run})
    return best_rows


def fmt(value: Any) -> str:
    number = finite_float(value)
    if number is None:
        return "-" if value is None else str(value)
    if abs(number) >= 1000:
        return f"{number:.1f}"
    return f"{number:.6g}"


def config_string(run: dict[str, Any]) -> str:
    grid = "-"
    if run.get("pose_grid_h") is not None and run.get("pose_grid_w") is not None:
        grid = f"{run.get('pose_grid_h')}x{run.get('pose_grid_w')}"
    return (
        f"dq={run.get('depth_confidence_quantile')} "
        f"pq={run.get('pointcloud_confidence_quantile')} "
        f"pose={run.get('pose_mode') or '-'}:{run.get('pose_solver')}@{run.get('pose_confidence_threshold')}/q{run.get('pose_confidence_quantile')} "
        f"grid={grid} "
        f"w={run.get('pose_weight_mode') or '-'} "
        f"stride={run.get('depth_stride')}/{run.get('pointcloud_stride')} "
        f"cap={run.get('max_pointcloud_points')}"
    )


def build_markdown(root: Path, runs: list[dict[str, Any]], best_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Sintel Tuned Eval Sweep",
        "",
        f"- root: `{root}`",
        f"- runs: {len(runs)}",
        "",
        "## Best By Metric",
        "",
        "| metric | direction | value | run | config | summary |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in best_rows:
        run = row["run"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["metric"],
                    row["direction"],
                    fmt(row["value"]),
                    run["name"],
                    config_string(run),
                    run["summary_path"],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## All Runs",
            "",
            "| run | depth_S | depth_SS | pc_l1 | ate | rpe_t | rpe_r | auc30 | config |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for run in runs:
        mean = run.get("mean_metrics", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    run["name"],
                    fmt(mean.get("depth_S_abs_rel")),
                    fmt(mean.get("depth_SS_abs_rel")),
                    fmt(mean.get("pointcloud_l1")),
                    fmt(mean.get("pose_ate")),
                    fmt(mean.get("pose_rpe_trans")),
                    fmt(mean.get("pose_rpe_rot")),
                    fmt(mean.get("pose_auc_30")),
                    config_string(run),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"No */summary.json files found under {root}")
    best_rows = select_best(runs)
    payload = {"root": str(root), "num_runs": len(runs), "best": best_rows, "runs": runs}
    markdown = build_markdown(root, runs, best_rows)

    if args.output_json:
        output_json = Path(args.output_json).resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[sweep] json -> {output_json}")
    if args.output_md:
        output_md = Path(args.output_md).resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")
        print(f"[sweep] markdown -> {output_md}")

    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
