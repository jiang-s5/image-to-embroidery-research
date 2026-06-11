from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pyembroidery import COLOR_CHANGE, COMMAND_MASK, END, JUMP, STITCH, TRIM, read_dst, read_pes


COMMAND_NAMES = {
    STITCH: "stitch",
    JUMP: "jump",
    TRIM: "trim",
    COLOR_CHANGE: "color_change",
    END: "end",
}


def read_pattern(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pes":
        return read_pes(str(path))
    if suffix == ".dst":
        return read_dst(str(path))
    raise ValueError(f"Unsupported embroidery file: {path}")


def command_name(command: int) -> str:
    return COMMAND_NAMES.get(command & COMMAND_MASK, f"unsupported_{command & COMMAND_MASK}")


def analyze_file(path: Path, max_stitch_mm: float, high_risk_jump_mm: float) -> dict[str, object]:
    metrics: dict[str, object] = {
        "path": str(path),
        "round_trip_parse_success": False,
        "error": "",
    }
    try:
        pattern = read_pattern(path)
    except Exception as exc:
        metrics["error"] = str(exc)
        return metrics

    command_counts: dict[str, int] = {}
    last = None
    stitch_count = 0
    jump_count = 0
    trim_count = 0
    color_change_count = 0
    unsupported_count = 0
    illegal_long_stitches = 0
    high_risk_jumps = 0
    total_stitch_mm = 0.0
    total_jump_mm = 0.0
    max_stitch_observed = 0.0
    max_jump_observed = 0.0
    lock_tack_like_count = 0
    short_stitch_run = 0

    for x_raw, y_raw, command_raw in pattern.stitches:
        command = command_raw & COMMAND_MASK
        name = command_name(command)
        command_counts[name] = command_counts.get(name, 0) + 1
        x = float(x_raw) / 10.0
        y = float(y_raw) / 10.0
        distance = math.dist(last, (x, y)) if last is not None else 0.0

        if command == STITCH:
            stitch_count += 1
            total_stitch_mm += distance
            max_stitch_observed = max(max_stitch_observed, distance)
            if distance > max_stitch_mm:
                illegal_long_stitches += 1
            if 0.05 <= distance <= 1.2:
                short_stitch_run += 1
            else:
                if short_stitch_run >= 3:
                    lock_tack_like_count += 1
                short_stitch_run = 0
        elif command == JUMP:
            jump_count += 1
            total_jump_mm += distance
            max_jump_observed = max(max_jump_observed, distance)
            if distance >= high_risk_jump_mm:
                high_risk_jumps += 1
            if short_stitch_run >= 3:
                lock_tack_like_count += 1
            short_stitch_run = 0
        elif command == TRIM:
            trim_count += 1
            if short_stitch_run >= 3:
                lock_tack_like_count += 1
            short_stitch_run = 0
        elif command == COLOR_CHANGE:
            color_change_count += 1
            short_stitch_run = 0
        elif command != END:
            unsupported_count += 1
            short_stitch_run = 0
        last = (x, y)

    if short_stitch_run >= 3:
        lock_tack_like_count += 1

    metrics.update(
        {
            "round_trip_parse_success": True,
            "commands": len(pattern.stitches),
            "command_counts": command_counts,
            "stitch_count": stitch_count,
            "jump_count": jump_count,
            "trim_count": trim_count,
            "color_change_count": color_change_count,
            "unsupported_command_count": unsupported_count,
            "illegal_long_stitch_count": illegal_long_stitches,
            "high_risk_jump_count": high_risk_jumps,
            "max_stitch_mm": round(max_stitch_observed, 4),
            "max_jump_mm": round(max_jump_observed, 4),
            "stitch_path_mm": round(total_stitch_mm, 4),
            "jump_path_mm": round(total_jump_mm, 4),
            "lock_tack_like_count": lock_tack_like_count,
            "max_stitch_threshold_mm": max_stitch_mm,
            "high_risk_jump_threshold_mm": high_risk_jump_mm,
        }
    )
    return metrics


def compare_metrics(pred: dict[str, object], gt: dict[str, object]) -> dict[str, object]:
    keys = [
        "stitch_count",
        "jump_count",
        "trim_count",
        "color_change_count",
        "illegal_long_stitch_count",
        "high_risk_jump_count",
        "max_jump_mm",
        "stitch_path_mm",
        "jump_path_mm",
    ]
    comparison: dict[str, object] = {}
    for key in keys:
        p = pred.get(key)
        g = gt.get(key)
        if isinstance(p, (int, float)) and isinstance(g, (int, float)):
            comparison[f"{key}_delta"] = round(float(p) - float(g), 4)
            comparison[f"{key}_ratio"] = round(float(p) / max(1e-6, float(g)), 4)
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate DST/PES executability and path-risk metrics.")
    parser.add_argument("--pred", required=True, help="Predicted DST/PES file.")
    parser.add_argument("--gt", default="", help="Optional ground-truth DST/PES file.")
    parser.add_argument("--report", default="", help="Optional JSON report path.")
    parser.add_argument("--max-stitch-mm", type=float, default=4.0)
    parser.add_argument("--high-risk-jump-mm", type=float, default=8.0)
    args = parser.parse_args()

    pred = analyze_file(Path(args.pred), args.max_stitch_mm, args.high_risk_jump_mm)
    report: dict[str, object] = {"pred": pred}
    if args.gt:
        gt = analyze_file(Path(args.gt), args.max_stitch_mm, args.high_risk_jump_mm)
        report["gt"] = gt
        if pred.get("round_trip_parse_success") and gt.get("round_trip_parse_success"):
            report["comparison"] = compare_metrics(pred, gt)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
