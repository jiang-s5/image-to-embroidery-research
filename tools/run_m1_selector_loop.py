from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def read_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} must be JSON-compatible YAML: {exc}") from exc
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
    parser = argparse.ArgumentParser(description="Run the full M1 planner selector loop.")
    parser.add_argument("--config", default="configs/m1_selector.yaml")
    args = parser.parse_args()

    root = Path.cwd()
    config = read_config(Path(args.config))
    builder = config.get("dataset_builder", {})
    trainer = config.get("trainer", {})
    if not isinstance(builder, dict) or not isinstance(trainer, dict):
        raise SystemExit("Config must contain dataset_builder and trainer objects.")

    build_cmd = [
        sys.executable,
        "tools/build_m1_selector_dataset.py",
    ]
    append_arg(build_cmd, "--sweep-results", builder.get("sweep_results"))
    append_arg(build_cmd, "--sweep-config", builder.get("sweep_config"))
    append_arg(build_cmd, "--outputs-root", builder.get("outputs_root"))
    append_arg(build_cmd, "--pair-root", builder.get("pair_root"))
    append_arg(build_cmd, "--output-dir", builder.get("output_dir"))
    append_arg(build_cmd, "--include-post-export-features", bool(builder.get("include_post_export_features", False)))

    train_cmd = [
        sys.executable,
        "tools/train_m1_planner_selector.py",
    ]
    append_arg(train_cmd, "--dataset-jsonl", trainer.get("dataset_jsonl"))
    append_arg(train_cmd, "--output-dir", trainer.get("output_dir"))
    append_arg(train_cmd, "--model-output", trainer.get("model_output"))
    append_arg(train_cmd, "--selection-target", trainer.get("selection_target"))
    append_arg(train_cmd, "--targets", trainer.get("targets"))
    append_arg(train_cmd, "--alphas", trainer.get("alphas"))
    append_arg(train_cmd, "--shrinkages", trainer.get("shrinkages"))

    run(build_cmd, root)
    run(train_cmd, root)
    print(
        json.dumps(
            {
                "status": "ok",
                "dataset": trainer.get("dataset_jsonl"),
                "model": trainer.get("model_output"),
                "report": str(Path(str(trainer.get("output_dir", ""))) / "m1_selector_report.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
