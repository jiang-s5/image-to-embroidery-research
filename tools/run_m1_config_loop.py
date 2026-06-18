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
    parser = argparse.ArgumentParser(description="Run M1 continuous config regressor training and DST evaluation loop.")
    parser.add_argument("--config", default="configs/m1_config_regressor.yaml")
    args = parser.parse_args()

    root = Path.cwd()
    config = read_config(Path(args.config))
    trainer = config.get("trainer", {})
    evaluator = config.get("evaluator", {})
    if not isinstance(trainer, dict) or not isinstance(evaluator, dict):
        raise SystemExit("Config must contain trainer and evaluator objects.")

    train_cmd = [sys.executable, "tools/train_m1_config_regressor.py"]
    append_arg(train_cmd, "--candidates-jsonl", trainer.get("candidates_jsonl"))
    append_arg(train_cmd, "--sweep-config", trainer.get("sweep_config"))
    append_arg(train_cmd, "--feature-method", trainer.get("feature_method"))
    append_arg(train_cmd, "--label-target", trainer.get("label_target"))
    append_arg(train_cmd, "--output-dir", trainer.get("output_dir"))
    append_arg(train_cmd, "--model-output", trainer.get("model_output"))
    append_arg(train_cmd, "--hidden-dims", trainer.get("hidden_dims"))
    append_arg(train_cmd, "--lrs", trainer.get("lrs"))
    append_arg(train_cmd, "--weight-decays", trainer.get("weight_decays"))
    append_arg(train_cmd, "--epochs", trainer.get("epochs"))
    append_arg(train_cmd, "--seeds", trainer.get("seeds"))

    eval_cmd = [sys.executable, "tools/run_m1_config_eval.py"]
    append_arg(eval_cmd, "--predictions-json", evaluator.get("predictions_json"))
    append_arg(eval_cmd, "--output-dir", evaluator.get("output_dir"))
    append_arg(eval_cmd, "--checkpoint", evaluator.get("checkpoint"))
    append_arg(eval_cmd, "--planner-config", evaluator.get("planner_config"))
    append_arg(eval_cmd, "--score-config", evaluator.get("score_config"))
    append_arg(eval_cmd, "--reuse", bool(evaluator.get("reuse", False)))
    append_arg(eval_cmd, "--cpu", bool(evaluator.get("cpu", False)))

    run(train_cmd, root)
    run(eval_cmd, root)
    print(
        json.dumps(
            {
                "status": "ok",
                "model": trainer.get("model_output"),
                "predictions": evaluator.get("predictions_json"),
                "eval_report": str(Path(str(evaluator.get("output_dir", ""))) / "m1_config_eval_report.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
