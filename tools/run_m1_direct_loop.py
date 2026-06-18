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
    parser = argparse.ArgumentParser(description="Run strict M1 direct selector training and apply loop.")
    parser.add_argument("--config", default="configs/m1_direct_selector.yaml")
    args = parser.parse_args()

    root = Path.cwd()
    config = read_config(Path(args.config))
    trainer = config.get("trainer", {})
    apply_cfg = config.get("apply", {})
    if not isinstance(trainer, dict) or not isinstance(apply_cfg, dict):
        raise SystemExit("Config must contain trainer and apply objects.")

    train_cmd = [
        sys.executable,
        "tools/train_m1_direct_selector.py",
    ]
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

    apply_cmd = [
        sys.executable,
        "tools/apply_m1_direct_selector.py",
    ]
    append_arg(apply_cmd, "--model", apply_cfg.get("model"))
    append_arg(apply_cmd, "--samples-jsonl", apply_cfg.get("samples_jsonl"))
    append_arg(apply_cmd, "--output-csv", apply_cfg.get("output_csv"))
    append_arg(apply_cmd, "--output-json", apply_cfg.get("output_json"))

    run(train_cmd, root)
    run(apply_cmd, root)
    print(
        json.dumps(
            {
                "status": "ok",
                "model": apply_cfg.get("model"),
                "selected_configs": apply_cfg.get("output_json"),
                "report": str(Path(str(trainer.get("output_dir", ""))) / "m1_direct_report.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
