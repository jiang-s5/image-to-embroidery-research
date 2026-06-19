from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def read_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a JSON object.")
    return payload


def append_arg(command: list[str], flag: str, value: Any) -> None:
    if isinstance(value, bool):
        if value:
            command.append(flag)
    elif value is not None and str(value) != "":
        command.extend([flag, str(value)])


def run(command: list[str], cwd: Path) -> None:
    print(" ".join(command), flush=True)
    completed = subprocess.run(command, cwd=cwd, text=True)
    if completed.returncode != 0:
        completed.check_returncode()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run M2 edge-policy dataset build and training loop.")
    parser.add_argument("--config", default="configs/m2_edge_policy.yaml")
    args = parser.parse_args()

    root = Path.cwd()
    config = read_config(Path(args.config))
    builder = config.get("builder", {})
    trainer = config.get("trainer", {})
    if not isinstance(builder, dict) or not isinstance(trainer, dict):
        raise SystemExit("Config must contain builder and trainer objects.")

    build_cmd = [sys.executable, "tools/build_m2_edge_dataset.py"]
    for pattern in builder.get("trace_globs", []):
        append_arg(build_cmd, "--trace-glob", pattern)
    append_arg(build_cmd, "--output-dir", builder.get("output_dir"))
    append_arg(build_cmd, "--max-negatives-per-decision", builder.get("max_negatives_per_decision"))
    append_arg(build_cmd, "--include-start-step", bool(builder.get("include_start_step", False)))

    train_cmd = [sys.executable, "tools/train_m2_edge_policy.py"]
    append_arg(train_cmd, "--dataset-jsonl", trainer.get("dataset_jsonl"))
    append_arg(train_cmd, "--output-dir", trainer.get("output_dir"))
    append_arg(train_cmd, "--model-output", trainer.get("model_output"))
    append_arg(train_cmd, "--lrs", trainer.get("lrs"))
    append_arg(train_cmd, "--weight-decays", trainer.get("weight_decays"))
    append_arg(train_cmd, "--epochs-list", trainer.get("epochs_list"))
    append_arg(train_cmd, "--seeds", trainer.get("seeds"))

    run(build_cmd, root)
    run(train_cmd, root)
    print(
        json.dumps(
            {
                "status": "ok",
                "dataset": trainer.get("dataset_jsonl"),
                "model": trainer.get("model_output"),
                "report": str(Path(str(trainer.get("output_dir", ""))) / "m2_edge_policy_report.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
