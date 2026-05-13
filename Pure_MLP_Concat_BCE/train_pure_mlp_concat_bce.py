from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


TRAIN_FEATURE_PATHS = [
    Path("Extracted_Features/BoW_int.npy"),
    Path("Extracted_Features/Normalized_CH.npy"),
    Path("Extracted_Features/Normalized_CM55.npy"),
    Path("Extracted_Features/Normalized_CORR.npy"),
    Path("Extracted_Features/Normalized_EDH.npy"),
    Path("Extracted_Features/Normalized_WT.npy"),
]
TEST_FEATURE_PATHS = [
    Path("Extracted_Features_Test/BoW_int.npy"),
    Path("Extracted_Features_Test/Normalized_CH.npy"),
    Path("Extracted_Features_Test/Normalized_CM55.npy"),
    Path("Extracted_Features_Test/Normalized_CORR.npy"),
    Path("Extracted_Features_Test/Normalized_EDH.npy"),
    Path("Extracted_Features_Test/Normalized_WT.npy"),
]


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 35
    lr: float = 1e-4
    weight_decay: float = 1e-4
    hidden_dims: tuple[int, ...] = (1024, 512, 256)
    dropout: float = 0.3
    activation: str = "gelu"
    val_ratio: float = 0.2
    threshold: float = 0.5
    random_seed: int = 42


class FeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class PureMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")
    return np.load(path).astype(np.float32)


def load_features(data_root: Path, paths: list[Path]) -> np.ndarray:
    arrays = [load_array(data_root / path) for path in paths]
    n_rows = {array.shape[0] for array in arrays}
    if len(n_rows) != 1:
        raise ValueError(f"Feature row counts do not match: {sorted(n_rows)}")
    return np.concatenate(arrays, axis=1)


def make_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]()


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims:
        raise argparse.ArgumentTypeError("hidden dims must contain at least one integer")
    return dims


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float) -> dict[str, float]:
    model.eval()
    all_targets, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.sigmoid(model(x))
            all_probs.append(probs.cpu().numpy())
            all_preds.append((probs > threshold).float().cpu().numpy())
            all_targets.append(y.numpy())

    targets = np.vstack(all_targets)
    probs = np.vstack(all_probs)
    preds = np.vstack(all_preds)
    try:
        map_score = average_precision_score(targets, probs, average="macro")
    except ValueError:
        map_score = 0.0
    return {
        "mAP": float(map_score),
        "micro_f1": float(f1_score(targets, preds, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_seen = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        total_seen += len(y)

    return total_loss / max(total_seen, 1)


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pure MLP concat-only BCE baseline.")
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("Pure_MLP_Concat_BCE/runs"))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=(1024, 512, 256))
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--activation", choices=("relu", "gelu", "silu"), default="gelu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        activation=args.activation,
        val_ratio=args.val_ratio,
        threshold=args.threshold,
        random_seed=args.seed,
    )
    set_seed(config.random_seed)
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print("Loading concatenated feature arrays...")
    x_all = load_features(args.data_root, TRAIN_FEATURE_PATHS)
    y_all = load_array(args.data_root / "database_labels_81_big.npy")
    x_test = load_features(args.data_root, TEST_FEATURE_PATHS)
    y_test = load_array(args.data_root / "database_labels_81_test.npy")

    if x_all.shape[0] != y_all.shape[0]:
        raise ValueError(f"Train features/labels mismatch: {x_all.shape[0]} vs {y_all.shape[0]}")
    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"Test features/labels mismatch: {x_test.shape[0]} vs {y_test.shape[0]}")

    idx_train, idx_val = train_test_split(
        np.arange(len(y_all)),
        test_size=config.val_ratio,
        random_state=config.random_seed,
        shuffle=True,
    )

    train_loader = DataLoader(
        FeatureDataset(x_all[idx_train], y_all[idx_train]),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        FeatureDataset(x_all[idx_val], y_all[idx_val]),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        FeatureDataset(x_test, y_test),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = PureMLP(
        input_dim=x_all.shape[1],
        hidden_dims=config.hidden_dims,
        output_dim=y_all.shape[1],
        dropout=config.dropout,
        activation=config.activation,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_map = -1.0
    history: list[dict[str, float]] = []
    checkpoint_path = args.output_dir / "pure_mlp_concat_bce_best.pt"

    print("\n" + "=" * 55)
    print("Training: Pure MLP + Concat Only + BCE")
    print("=" * 55)
    print(f"Train samples: {len(idx_train)} | Val samples: {len(idx_val)} | Test samples: {len(y_test)}")
    print(f"Input dim: {x_all.shape[1]} | Classes: {y_all.shape[1]}")
    print(f"MLP hidden_dims={config.hidden_dims}, activation={config.activation}, dropout={config.dropout}")

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device, config.threshold)
        row = {"epoch": epoch, "train_loss": float(train_loss), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)

        if val_metrics["mAP"] > best_map:
            best_map = val_metrics["mAP"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "input_dim": x_all.shape[1],
                    "num_classes": y_all.shape[1],
                    "best_val_metrics": val_metrics,
                },
                checkpoint_path,
            )

        print(
            f"Epoch {epoch:03d}/{config.epochs} "
            f"loss={train_loss:.4f} "
            f"val_mAP={val_metrics['mAP']:.4f} "
            f"val_micro_f1={val_metrics['micro_f1']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, config.threshold)

    save_json(args.output_dir / "training_history.json", history)
    save_json(args.output_dir / "test_metrics.json", test_metrics)
    save_json(args.output_dir / "config.json", asdict(config))
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
