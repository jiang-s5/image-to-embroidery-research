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
    parser = argparse.ArgumentParser(description="Run M1 ranking selector training loop.")
    parser.add_argument("--config", default="configs/m1_ranking_selector.yaml")
    args = parser.parse_args()

    root = Path.cwd()
    config = read_config(Path(args.config))
    trainer = config.get("trainer", {})
    if not isinstance(trainer, dict):
        raise SystemExit("Config must contain a trainer object.")

    command = [sys.executable, "tools/train_m1_ranking_selector.py"]
    append_arg(command, "--candidates-jsonl", trainer.get("candidates_jsonl"))
    append_arg(command, "--sweep-config", trainer.get("sweep_config"))
    append_arg(command, "--feature-method", trainer.get("feature_method"))
    append_arg(command, "--label-target", trainer.get("label_target"))
    append_arg(command, "--min-delta", trainer.get("min_delta"))
    append_arg(command, "--output-dir", trainer.get("output_dir"))
    append_arg(command, "--model-output", trainer.get("model_output"))
    append_arg(command, "--lrs", trainer.get("lrs"))
    append_arg(command, "--weight-decays", trainer.get("weight_decays"))
    append_arg(command, "--epochs-list", trainer.get("epochs_list"))
    append_arg(command, "--seeds", trainer.get("seeds"))
    run(command, root)
    print(
        json.dumps(
            {
                "status": "ok",
                "model": trainer.get("model_output"),
                "report": str(Path(str(trainer.get("output_dir", ""))) / "m1_ranking_report.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
