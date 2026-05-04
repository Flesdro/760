import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path(__file__).with_name("data.md")
OUTPUT_PATH = Path(__file__).with_name("training_metrics_comparison.png")


EPOCH_PATTERN = re.compile(
    r"Epoch \[(\d+)/(\d+)\]\s+"
    r"Loss:\s+([\d.]+)\s+\|\s+"
    r"Val mAP:\s+([\d.]+)\s+\|\s+"
    r"Mi-F1:\s+([\d.]+)\s+\|\s+"
    r"Ma-F1:\s+([\d.]+)"
)


def parse_training_log(path: Path) -> pd.DataFrame:
    rows = []
    current_model = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if line.startswith("# "):
            current_model = line.removeprefix("# ").strip()
            continue

        match = EPOCH_PATTERN.search(line)
        if not match or current_model is None:
            continue

        rows.append(
            {
                "model": current_model,
                "epoch": int(match.group(1)),
                "loss": float(match.group(3)),
                "val_map": float(match.group(4)),
                "micro_f1": float(match.group(5)),
                "macro_f1": float(match.group(6)),
            }
        )

    if not rows:
        raise ValueError(f"No epoch rows found in {path}")

    return pd.DataFrame(rows)


def plot_metrics(df: pd.DataFrame) -> None:
    metrics = [
        ("loss", "Loss"),
        ("val_map", "Validation mAP"),
        ("micro_f1", "Micro-F1"),
        ("macro_f1", "Macro-F1"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)

    for ax, (column, title) in zip(axes.ravel(), metrics):
        for model_name, model_df in df.groupby("model", sort=False):
            ax.plot(
                model_df["epoch"],
                model_df[column],
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                label=model_name,
            )

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(True, linestyle="--", alpha=0.35)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Early Fusion + MLP Training Metrics", fontsize=16)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.show()

    print(f"Saved plot to: {OUTPUT_PATH}")


def main() -> None:
    df = parse_training_log(DATA_PATH)
    print(df.head())
    print(f"Parsed {len(df)} rows from {df['model'].nunique()} models.")
    plot_metrics(df)


if __name__ == "__main__":
    main()
