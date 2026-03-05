#!/usr/bin/env python3
"""Aggregate token sampling ablation results across experiment directories.

Each experiment directory can contain multiple split files:
- step_stats_split*.jsonl
- speed_summary_split*.json
- metrics_summary_split*.json

Outputs:
- ablation_summary.csv
- ablation_summary.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional


OUTPUT_COLUMNS = [
    "experiment",
    "experiment_dir",
    "vis_tokens_step_mean",
    "vis_tokens_step_p50",
    "vis_tokens_step_p90",
    "vis_structure_step_total_visual_tokens",
    "step_total_ms",
    "approx_hz_mean",
    "sr",
    "tr",
    "cr",
    "token_reduction_%",
    "hz_gain_%",
    "sr_delta",
    "tr_delta",
    "cr_delta",
]


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _percentile(sorted_vals: List[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return sorted_vals[low]
    frac = pos - low
    return sorted_vals[low] * (1 - frac) + sorted_vals[high] * frac


def _extract_step_values(step_paths: Iterable[str], exp_dir: Path) -> Dict[str, Optional[float]]:
    vis_tokens: List[float] = []

    step_paths = list(step_paths)
    if not step_paths:
        _warn(f"{exp_dir}: missing step_stats_split*.jsonl")

    for path in step_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        _warn(f"{path}:{line_no} invalid JSONL row, skipped")
                        continue
                    v = _safe_float(row.get("vis_tokens_step"))
                    if v is not None:
                        vis_tokens.append(v)
        except OSError as err:
            _warn(f"{path}: read failed ({err})")

    vis_tokens.sort()
    return {
        "vis_tokens_step_mean": mean(vis_tokens) if vis_tokens else None,
        "vis_tokens_step_p50": _percentile(vis_tokens, 0.5),
        "vis_tokens_step_p90": _percentile(vis_tokens, 0.9),
    }


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as err:
        _warn(f"{path}: read failed ({err})")
        return None
    except json.JSONDecodeError as err:
        _warn(f"{path}: invalid JSON ({err})")
        return None
    if not isinstance(data, dict):
        _warn(f"{path}: top-level JSON must be object, got {type(data).__name__}")
        return None
    return data


def _extract_speed_values(speed_paths: Iterable[str], exp_dir: Path) -> Dict[str, Optional[float]]:
    step_total_ms_vals: List[float] = []
    approx_hz_vals: List[float] = []
    total_visual_tokens_vals: List[float] = []

    speed_paths = list(speed_paths)
    if not speed_paths:
        _warn(f"{exp_dir}: missing speed_summary_split*.json")

    for path in speed_paths:
        payload = _load_json(path)
        if payload is None:
            continue

        step_total_ms = payload.get("step_total_ms")
        if isinstance(step_total_ms, dict):
            v = _safe_float(step_total_ms.get("mean"))
            if v is not None:
                step_total_ms_vals.append(v)

        v = _safe_float(payload.get("approx_hz_mean"))
        if v is not None:
            approx_hz_vals.append(v)

        total_visual_tokens = payload.get("vis_structure_step_total_visual_tokens")
        if isinstance(total_visual_tokens, dict):
            v = _safe_float(total_visual_tokens.get("mean"))
            if v is not None:
                total_visual_tokens_vals.append(v)

    return {
        "step_total_ms": mean(step_total_ms_vals) if step_total_ms_vals else None,
        "approx_hz_mean": mean(approx_hz_vals) if approx_hz_vals else None,
        "vis_structure_step_total_visual_tokens": mean(total_visual_tokens_vals)
        if total_visual_tokens_vals
        else None,
    }


def _extract_metrics_values(metrics_paths: Iterable[str], exp_dir: Path) -> Dict[str, Optional[float]]:
    sr_vals: List[float] = []
    tr_vals: List[float] = []
    cr_vals: List[float] = []

    metrics_paths = list(metrics_paths)
    if not metrics_paths:
        _warn(f"{exp_dir}: missing metrics_summary_split*.json")

    for path in metrics_paths:
        payload = _load_json(path)
        if payload is None:
            continue

        sr = _safe_float(payload.get("success_rate"))
        if sr is not None:
            sr_vals.append(sr)

        tr = _safe_float(payload.get("following_rate"))
        if tr is not None:
            tr_vals.append(tr)

        cr = _safe_float(payload.get("collision_rate"))
        if cr is not None:
            cr_vals.append(cr)

    return {
        "sr": mean(sr_vals) if sr_vals else None,
        "tr": mean(tr_vals) if tr_vals else None,
        "cr": mean(cr_vals) if cr_vals else None,
    }


def _pct_change(value: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if value is None or baseline is None or baseline == 0:
        return None
    return (value - baseline) / baseline * 100.0


def _delta(value: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if value is None or baseline is None:
        return None
    return value - baseline


def _fmt(value: Optional[float], precision: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{value:.{precision}f}"


def analyze_experiment(exp_dir: Path, label: str) -> Dict[str, Optional[float]]:
    step_paths = sorted(glob.glob(str(exp_dir / "step_stats_split*.jsonl")))
    speed_paths = sorted(glob.glob(str(exp_dir / "speed_summary_split*.json")))
    metrics_paths = sorted(glob.glob(str(exp_dir / "metrics_summary_split*.json")))

    result: Dict[str, Optional[float]] = {
        "experiment": label,
        "experiment_dir": str(exp_dir),
    }
    result.update(_extract_step_values(step_paths, exp_dir))
    result.update(_extract_speed_values(speed_paths, exp_dir))
    result.update(_extract_metrics_values(metrics_paths, exp_dir))
    return result


def write_csv(rows: List[Dict[str, Optional[float]]], out_path: Path) -> None:
    if not rows:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: List[Dict[str, Optional[float]]], out_path: Path) -> None:
    headers = [c for c in OUTPUT_COLUMNS if c != "experiment_dir"]

    lines = []
    lines.append("# Token Sampling Ablation Summary")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        vals = []
        for h in headers:
            v = row.get(h)
            if h == "experiment":
                vals.append(str(v))
            else:
                vals.append(_fmt(_safe_float(v), precision=4))
        lines.append("| " + " | ".join(vals) + " |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze token sampling ablation experiments")
    parser.add_argument(
        "exp_dirs",
        nargs="+",
        help="Experiment directories. Format: /path/to/exp or label=/path/to/exp",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline label or directory path. Default: first experiment.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for ablation_summary.csv and ablation_summary.md",
    )
    return parser.parse_args()


def _parse_exp_arg(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
    else:
        path = raw
        label = Path(raw).name
    return label, Path(path).expanduser().resolve()


def main() -> None:
    args = parse_args()
    specs = [_parse_exp_arg(s) for s in args.exp_dirs]

    rows: List[Dict[str, Optional[float]]] = []
    for label, exp_dir in specs:
        if not exp_dir.exists() or not exp_dir.is_dir():
            _warn(f"{exp_dir}: directory not found, keeping row with NA values")
            rows.append({"experiment": label, "experiment_dir": str(exp_dir)})
            continue
        rows.append(analyze_experiment(exp_dir, label))

    baseline_key = args.baseline
    baseline_row = None
    if baseline_key is None and rows:
        baseline_row = rows[0]
    else:
        for row in rows:
            if row.get("experiment") == baseline_key or row.get("experiment_dir") == baseline_key:
                baseline_row = row
                break

    if baseline_row is None and rows:
        _warn("baseline not found; relative fields will be NA")

    for row in rows:
        for key in OUTPUT_COLUMNS:
            row.setdefault(key, None)
        row["token_reduction_%"] = None
        row["hz_gain_%"] = None
        row["sr_delta"] = None
        row["tr_delta"] = None
        row["cr_delta"] = None
        if baseline_row is None:
            continue
        token_pct = _pct_change(
            _safe_float(row.get("vis_tokens_step_mean")),
            _safe_float(baseline_row.get("vis_tokens_step_mean")),
        )
        if token_pct is None:
            row["token_reduction_%"] = None
        else:
            row["token_reduction_%"] = 0.0 if abs(token_pct) < 1e-12 else -token_pct
        row["hz_gain_%"] = _pct_change(
            _safe_float(row.get("approx_hz_mean")),
            _safe_float(baseline_row.get("approx_hz_mean")),
        )
        row["sr_delta"] = _delta(_safe_float(row.get("sr")), _safe_float(baseline_row.get("sr")))
        row["tr_delta"] = _delta(_safe_float(row.get("tr")), _safe_float(baseline_row.get("tr")))
        row["cr_delta"] = _delta(_safe_float(row.get("cr")), _safe_float(baseline_row.get("cr")))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "ablation_summary.csv"
    md_path = output_dir / "ablation_summary.md"

    write_csv(rows, csv_path)
    write_md(rows, md_path)

    print(f"[INFO] Wrote: {csv_path}")
    print(f"[INFO] Wrote: {md_path}")


if __name__ == "__main__":
    main()
