from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from pyembroidery import COLOR_CHANGE, COMMAND_MASK, END, JUMP, STITCH, TRIM, read_dst
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


COMMANDS = [STITCH, JUMP, TRIM, COLOR_CHANGE, END]
OTHER_COMMAND = len(COMMANDS)
COMMAND_TO_ID = {cmd: i for i, cmd in enumerate(COMMANDS)}
COMMAND_DIM = len(COMMANDS) + 1


class StitchSequenceMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 + COMMAND_DIM),
        )

    def forward(self, x):
        out = self.net(x)
        return out[:, :2], out[:, 2:]


def command_id(raw_command: int) -> int:
    return COMMAND_TO_ID.get(raw_command & COMMAND_MASK, OTHER_COMMAND)


def load_sequence(dst_path: Path) -> tuple[np.ndarray, np.ndarray]:
    pattern = read_dst(str(dst_path))
    if len(pattern.stitches) < 2:
        raise ValueError(f"not enough stitches in {dst_path}")
    deltas = []
    commands = []
    last_x, last_y = pattern.stitches[0][0] / 10.0, pattern.stitches[0][1] / 10.0
    for x_raw, y_raw, cmd_raw in pattern.stitches[1:]:
        x, y = x_raw / 10.0, y_raw / 10.0
        deltas.append([x - last_x, y - last_y])
        commands.append(command_id(cmd_raw))
        last_x, last_y = x, y
    return np.asarray(deltas, dtype=np.float32), np.asarray(commands, dtype=np.int64)


def make_windows(
    dst_paths: list[Path],
    seq_len: int,
    delta_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = []
    y_delta = []
    y_cmd = []
    for path in dst_paths:
        deltas, commands = load_sequence(path)
        if len(deltas) <= seq_len:
            continue
        command_one_hot = np.eye(COMMAND_DIM, dtype=np.float32)[commands]
        features = np.concatenate([deltas / delta_scale, command_one_hot], axis=1)
        for i in range(seq_len, len(features)):
            xs.append(features[i - seq_len : i].reshape(-1))
            y_delta.append(deltas[i] / delta_scale)
            y_cmd.append(commands[i])
    if not xs:
        raise ValueError("no training windows were created")
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(y_delta, dtype=np.float32),
        np.asarray(y_cmd, dtype=np.int64),
    )


def train(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dst_paths = sorted(Path(args.input_dir).rglob("*.dst"))
    if not dst_paths:
        raise FileNotFoundError(f"no DST files found under {args.input_dir}")
    design_count = len(dst_paths)
    if design_count < args.min_designs and not args.allow_small_data:
        raise SystemExit(
            f"found only {design_count} DST design(s). "
            f"Use --allow-small-data for a smoke/overfit run, or provide at least {args.min_designs} designs."
        )

    x, yd, yc = make_windows(dst_paths, seq_len=args.seq_len, delta_scale=args.delta_scale)
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    split = max(1, int(len(indices) * args.train_ratio))
    train_idx = indices[:split]
    val_idx = indices[split:] if split < len(indices) else indices[:1]

    train_ds = TensorDataset(
        torch.from_numpy(x[train_idx]),
        torch.from_numpy(yd[train_idx]),
        torch.from_numpy(yc[train_idx]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x[val_idx]),
        torch.from_numpy(yd[val_idx]),
        torch.from_numpy(yc[val_idx]),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = choose_device(force_cpu=args.cpu)
    model = StitchSequenceMLP(input_dim=x.shape[1], hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for xb, ydb, ycb in train_loader:
            xb, ydb, ycb = xb.to(device), ydb.to(device), ycb.to(device)
            pred_delta, pred_cmd = model(xb)
            loss = mse(pred_delta, ydb) + args.command_loss_weight * ce(pred_cmd, ycb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
            train_correct += (pred_cmd.argmax(dim=1) == ycb).sum().item()
            train_total += len(xb)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_delta_abs = 0.0
        with torch.no_grad():
            for xb, ydb, ycb in val_loader:
                xb, ydb, ycb = xb.to(device), ydb.to(device), ycb.to(device)
                pred_delta, pred_cmd = model(xb)
                loss = mse(pred_delta, ydb) + args.command_loss_weight * ce(pred_cmd, ycb)
                val_loss += loss.item() * len(xb)
                val_correct += (pred_cmd.argmax(dim=1) == ycb).sum().item()
                val_delta_abs += (pred_delta - ydb).abs().sum().item()
                val_total += len(xb)

        record = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, train_total),
            "train_command_acc": train_correct / max(1, train_total),
            "val_loss": val_loss / max(1, val_total),
            "val_command_acc": val_correct / max(1, val_total),
            "val_mean_abs_delta_mm": (val_delta_abs / max(1, val_total * 2)) * args.delta_scale,
        }
        history.append(record)
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 5) == 0:
            print(json.dumps(record, ensure_ascii=False))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "dst_sequence_smoke_model.pt"
    metrics_path = output_dir / "dst_sequence_smoke_metrics.json"
    torch.save(
        {
            "model_state": model.state_dict(),
            "seq_len": args.seq_len,
            "delta_scale": args.delta_scale,
            "command_dim": COMMAND_DIM,
            "input_dim": x.shape[1],
            "hidden_dim": args.hidden_dim,
            "dst_paths": [str(path) for path in dst_paths],
        },
        model_path,
    )
    metrics = {
        "design_count": design_count,
        "window_count": int(len(x)),
        "train_windows": int(len(train_idx)),
        "val_windows": int(len(val_idx)),
        "device": str(device),
        "warning": (
            "Smoke/overfit run on too few designs; not a generalizable image-to-stitch model."
            if design_count < args.min_designs
            else ""
        ),
        "history": history,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved model: {model_path}")
    print(f"saved metrics: {metrics_path}")
    return metrics


def choose_device(force_cpu: bool = False) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        device = torch.device("cuda")
        _ = torch.ones(1, device=device) + 1
        torch.cuda.synchronize()
        return device
    except Exception as exc:
        print(f"warning: CUDA is visible but unusable for this PyTorch build; falling back to CPU. {exc}")
        return torch.device("cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small next-stitch sequence smoke model from DST files.")
    parser.add_argument("--input-dir", required=True, help="Folder containing DST files.")
    parser.add_argument("--output-dir", default="models/dst_sequence_smoke")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--delta-scale", type=float, default=10.0)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--command-loss-weight", type=float, default=0.25)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--min-designs", type=int, default=20)
    parser.add_argument("--allow-small-data", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=3830)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
