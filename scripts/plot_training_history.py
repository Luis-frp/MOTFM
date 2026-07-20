import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train/val loss and learning-rate curves from a MOTFM metrics_history.json file."
    )
    parser.add_argument(
        "--history_path",
        type=str,
        default=None,
        help="Path to metrics_history.json. If omitted, --checkpoint_dir/metrics_history.json is used.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Run checkpoint directory (e.g. checkpoints/<run_name>) containing metrics_history.json.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output PNG path. Defaults to training_curves.png next to the history file.",
    )
    return parser.parse_args()


def resolve_history_path(args: argparse.Namespace) -> Path:
    if args.history_path:
        return Path(args.history_path)
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir) / "metrics_history.json"
    raise ValueError("Provide either --history_path or --checkpoint_dir.")


def main() -> None:
    args = parse_args()
    history_path = resolve_history_path(args)

    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    if not history:
        raise ValueError(f"No metrics found in {history_path}.")

    epochs = [row["epoch"] for row in history]
    train_loss = [row.get("train_loss") for row in history]
    val_loss = [row.get("val_loss") for row in history]
    lr = [row.get("lr") for row in history]

    fig, (ax_loss, ax_lr) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax_loss.plot(epochs, train_loss, label="train/loss", marker="o", markersize=3)
    val_epochs = [e for e, v in zip(epochs, val_loss) if v is not None]
    val_values = [v for v in val_loss if v is not None]
    if val_values:
        ax_loss.plot(val_epochs, val_values, label="val/loss", marker="o", markersize=3)
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("MOTFM training curves")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    lr_epochs = [e for e, v in zip(epochs, lr) if v is not None]
    lr_values = [v for v in lr if v is not None]
    if lr_values:
        ax_lr.plot(lr_epochs, lr_values, color="tab:green", marker="o", markersize=3)
    ax_lr.set_ylabel("learning rate")
    ax_lr.set_xlabel("epoch")
    ax_lr.grid(alpha=0.3)

    fig.tight_layout()
    output_path = Path(args.output_path) if args.output_path else history_path.with_name("training_curves.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
