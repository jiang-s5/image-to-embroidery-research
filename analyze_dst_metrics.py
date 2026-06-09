from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyembroidery import COLOR_CHANGE, COMMAND_MASK, JUMP, STITCH, read_dst


def analyze(path: Path, trim_threshold_mm: float, high_risk_threshold_mm: float) -> dict[str, object]:
    pattern = read_dst(str(path))
    last: tuple[float, float] | None = None
    stitch_count = 0
    jump_count = 0
    color_changes = 0
    stitch_distance = 0.0
    jump_distance = 0.0
    max_jump = 0.0
    estimated_trims = 0
    high_risk_jumps = 0
    for x_raw, y_raw, cmd_raw in pattern.stitches:
        x = float(x_raw) / 10.0
        y = float(y_raw) / 10.0
        cmd = cmd_raw & COMMAND_MASK
        if cmd == COLOR_CHANGE:
            color_changes += 1
        if last is not None:
            dx = x - last[0]
            dy = y - last[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if cmd == STITCH:
                stitch_count += 1
                stitch_distance += dist
            elif cmd == JUMP:
                jump_count += 1
                jump_distance += dist
                max_jump = max(max_jump, dist)
                if dist >= trim_threshold_mm:
                    estimated_trims += 1
                if dist >= high_risk_threshold_mm:
                    high_risk_jumps += 1
        last = (x, y)
    return {
        "file": str(path),
        "stitch_segments": stitch_count,
        "jump_segments": jump_count,
        "color_changes": color_changes,
        "stitch_distance_mm": round(stitch_distance, 3),
        "jump_distance_mm": round(jump_distance, 3),
        "max_jump_mm": round(max_jump, 3),
        "estimated_trim_jumps": estimated_trims,
        "high_risk_jumps": high_risk_jumps,
        "trim_threshold_mm": trim_threshold_mm,
        "high_risk_threshold_mm": high_risk_threshold_mm,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze DST stitch and jump metrics.")
    parser.add_argument("dst", nargs="+")
    parser.add_argument("--trim-threshold-mm", type=float, default=6.0)
    parser.add_argument("--high-risk-threshold-mm", type=float, default=12.0)
    args = parser.parse_args()
    print(
        json.dumps(
            [analyze(Path(item), args.trim_threshold_mm, args.high_risk_threshold_mm) for item in args.dst],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
