from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pyembroidery import COLOR_CHANGE, COMMAND_MASK, END, JUMP, STITCH, TRIM, read_dst, read_pes


def read_pattern(path: Path):
    if path.suffix.lower() == ".pes":
        return read_pes(str(path))
    return read_dst(str(path))


def distance_mm(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def command_name(cmd: int) -> str:
    if cmd == STITCH:
        return "STITCH"
    if cmd == JUMP:
        return "JUMP"
    if cmd == TRIM:
        return "TRIM"
    if cmd == COLOR_CHANGE:
        return "COLOR_CHANGE"
    if cmd == END:
        return "END"
    return str(cmd)


def analyze(path: Path, max_stitch_mm: float, high_risk_jump_mm: float, near_connection_mm: float) -> dict[str, object]:
    pattern = read_pattern(path)
    points: list[dict[str, object]] = []
    vectors: list[dict[str, object]] = []
    color_index = 0
    previous: tuple[float, float] | None = None
    previous_cmd: int | None = None
    stitch_vectors = 0
    jump_vectors = 0
    trim_count = 0
    color_changes = 0
    illegal_long_stitches: list[dict[str, object]] = []
    high_risk_jumps: list[dict[str, object]] = []
    near_jump_candidates: list[dict[str, object]] = []

    for index, (x_raw, y_raw, raw_cmd) in enumerate(pattern.stitches):
        cmd = raw_cmd & COMMAND_MASK
        x = float(x_raw) / 10.0
        y = float(y_raw) / 10.0
        if cmd == COLOR_CHANGE:
            color_index += 1
            color_changes += 1
        elif cmd == TRIM:
            trim_count += 1
        point = {"i": index, "x_mm": x, "y_mm": y, "cmd": command_name(cmd), "color": color_index}
        points.append(point)

        if previous is not None and previous_cmd is not None:
            dist = distance_mm(previous, (x, y))
            vector = {
                "i0": index - 1,
                "i1": index,
                "dx_mm": x - previous[0],
                "dy_mm": y - previous[1],
                "distance_mm": dist,
                "cmd": command_name(cmd),
                "color": color_index,
            }
            vectors.append(vector)
            if cmd == STITCH:
                stitch_vectors += 1
                if dist > max_stitch_mm:
                    illegal_long_stitches.append(vector)
            elif cmd == JUMP:
                jump_vectors += 1
                if dist > high_risk_jump_mm:
                    high_risk_jumps.append(vector)
                elif dist <= near_connection_mm:
                    near_jump_candidates.append(vector)
        previous = (x, y)
        previous_cmd = cmd

    stitch_lengths = [float(item["distance_mm"]) for item in vectors if item["cmd"] == "STITCH"]
    jump_lengths = [float(item["distance_mm"]) for item in vectors if item["cmd"] == "JUMP"]
    max_stitch = max(stitch_lengths) if stitch_lengths else 0.0
    max_jump = max(jump_lengths) if jump_lengths else 0.0
    continuity_score = 1.0
    if stitch_vectors:
        continuity_score -= min(1.0, len(illegal_long_stitches) / stitch_vectors)
    if jump_vectors:
        continuity_score -= 0.35 * min(1.0, len(high_risk_jumps) / jump_vectors)
        continuity_score -= 0.15 * min(1.0, len(near_jump_candidates) / jump_vectors)
    continuity_score = max(0.0, min(1.0, continuity_score))

    return {
        "input": str(path),
        "total_points": len(points),
        "stitch_vectors": stitch_vectors,
        "jump_vectors": jump_vectors,
        "trim_count": trim_count,
        "color_changes": color_changes,
        "max_stitch_mm": round(max_stitch, 3),
        "max_jump_mm": round(max_jump, 3),
        "illegal_long_stitch_count": len(illegal_long_stitches),
        "high_risk_jump_count": len(high_risk_jumps),
        "near_jump_candidate_count": len(near_jump_candidates),
        "continuity_score": round(continuity_score, 4),
        "thresholds": {
            "max_stitch_mm": max_stitch_mm,
            "high_risk_jump_mm": high_risk_jump_mm,
            "near_connection_mm": near_connection_mm,
        },
        "examples": {
            "illegal_long_stitches": illegal_long_stitches[:20],
            "high_risk_jumps": high_risk_jumps[:20],
            "near_jump_candidates": near_jump_candidates[:20],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze vector continuity of DST/PES stitch paths.")
    parser.add_argument("input")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--max-stitch-mm", type=float, default=4.0)
    parser.add_argument("--high-risk-jump-mm", type=float, default=8.0)
    parser.add_argument("--near-connection-mm", type=float, default=2.5)
    args = parser.parse_args()
    summary = analyze(Path(args.input), args.max_stitch_mm, args.high_risk_jump_mm, args.near_connection_mm)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
