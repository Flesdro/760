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
GROUP_NAMES = ["bow", "ch", "cm55", "corr", "edh", "wt"]


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    threshold: float = 0.5
    random_seed: int = 42
    channels: int = 64
    dropout: float = 0.3
    max_train_samples: int | None = None


class FeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class BasicBlock1D(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + x)


class ResNet6Features(nn.Module):
    """ResNet-6 for concatenated tabular feature vectors.

    Counted as: stem conv + 2 residual blocks * 2 convs + linear head = 6
    trainable layers. The feature vector is treated as a 1D signal.
    """

    def __init__(self, input_dim: int, num_classes: int, channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.layer1 = BasicBlock1D(channels, dropout)
        self.layer2 = BasicBlock1D(channels, dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


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


def load_reference_config(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing reference config: {path.resolve()}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_text_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")
    return path.read_text(encoding="utf-8").splitlines()


def normalize_name(path: str) -> str:
    path = path.replace("\\", "/")
    return Path(path).name.lower()


def read_image_names(path: Path) -> list[str]:
    return [normalize_name(line.strip()) for line in load_text_lines(path) if line.strip()]


def resolve_input_path(data_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    data_path = data_root / path
    if data_path.exists():
        return data_path
    return path


def build_image_alignment(local_names: list[str], official_names: list[str]) -> tuple[np.ndarray, dict[str, object]]:
    official_index: dict[str, int] = {}
    for idx, name in enumerate(official_names):
        official_index.setdefault(name, idx)

    matched_local_indices: list[int] = []
    missing = 0
    for local_idx, name in enumerate(local_names):
        if name not in official_index:
            missing += 1
            continue
        matched_local_indices.append(local_idx)

    metadata = {
        "local_images": len(local_names),
        "official_train_images": len(official_names),
        "matched": int(len(matched_local_indices)),
        "missing": int(missing),
        "match_ratio": float(len(matched_local_indices) / max(len(local_names), 1)),
    }
    return np.asarray(matched_local_indices, dtype=np.int64), metadata


def load_features(data_root: Path, paths: list[Path]) -> tuple[np.ndarray, list[int]]:
    arrays = [load_array(data_root / path) for path in paths]
    n_rows = {array.shape[0] for array in arrays}
    if len(n_rows) != 1:
        min_rows = min(n_rows)
        print(
            "Warning: visual feature row counts do not match "
            f"{sorted(n_rows)}; trimming all groups to {min_rows} rows by prefix."
        )
        arrays = [array[:min_rows] for array in arrays]
    group_dims = [array.shape[1] for array in arrays]
    return np.concatenate(arrays, axis=1), group_dims


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
    valid_classes = targets.sum(axis=0) > 0
    if valid_classes.any():
        map_score = average_precision_score(targets[:, valid_classes], probs[:, valid_classes], average="macro")
    else:
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


def load_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def make_parser(reference_config: dict[str, object]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a ResNet-6 baseline on extracted feature vectors.")
    parser.add_argument("--config-path", type=Path, default=Path("20_train_gated_fusion_residual_mlp_asl_tag_no_overlap/config.json"))
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--local-image-list", type=Path, default=Path(str(reference_config.get("local_image_list", "database_img.txt"))))
    parser.add_argument("--official-image-list", type=Path, default=Path(str(reference_config.get("official_image_list", "TrainImagelist.txt"))))
    parser.add_argument("--output-dir", type=Path, default=Path("02_ResNet/resnet6_feature_clean_runs"))
    parser.add_argument("--epochs", type=int, default=int(reference_config.get("epochs", 50)))
    parser.add_argument("--batch-size", type=int, default=int(reference_config.get("batch_size", 128)))
    parser.add_argument("--lr", type=float, default=float(reference_config.get("lr", 1e-3)))
    parser.add_argument("--weight-decay", type=float, default=float(reference_config.get("weight_decay", 1e-4)))
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=float(reference_config.get("dropout", 0.3)))
    parser.add_argument("--threshold", type=float, default=float(reference_config.get("threshold", 0.5)))
    parser.add_argument("--val-ratio", type=float, default=float(reference_config.get("val_ratio", 0.1)))
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=int(reference_config.get("random_seed", 42)))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=reference_config.get("max_train_samples"))
    return parser


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-path", type=Path, default=Path("20_train_gated_fusion_residual_mlp_asl_tag_no_overlap/config.json"))
    pre_args, _ = pre_parser.parse_known_args()
    reference_config = load_reference_config(pre_args.config_path)

    parser = make_parser(reference_config)
    args = parser.parse_args()
    reference_config = load_reference_config(args.config_path)

    config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        threshold=args.threshold,
        random_seed=args.seed,
        channels=args.channels,
        dropout=args.dropout,
        max_train_samples=args.max_train_samples,
    )
    if config.epochs < 1:
        raise ValueError("--epochs must be at least 1 so a checkpoint can be selected.")
    if config.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if not 0.0 < config.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")
    if not 0.0 < config.test_ratio < 1.0:
        raise ValueError("--test-ratio must be between 0 and 1.")
    if config.val_ratio + config.test_ratio >= 1.0:
        raise ValueError("--val-ratio + --test-ratio must be less than 1.")
    if config.max_train_samples is not None and config.max_train_samples < 2:
        raise ValueError("--max-train-samples must be at least 2 when provided.")

    set_seed(config.random_seed)
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Reference config: {args.config_path}")
    print("Loading feature arrays using experiment-07 baseline inputs...")
    x_all, group_dims = load_features(args.data_root, TRAIN_FEATURE_PATHS)
    y_all = load_array(args.data_root / "database_labels_81_big.npy")

    if x_all.shape[0] != y_all.shape[0]:
        raise ValueError(f"Train features/labels mismatch: {x_all.shape[0]} vs {y_all.shape[0]}")

    local_names = read_image_names(resolve_input_path(args.data_root, args.local_image_list))
    official_names = read_image_names(resolve_input_path(args.data_root, args.official_image_list))
    matched_indices, alignment_metadata = build_image_alignment(local_names, official_names)
    x_all = x_all[matched_indices]
    y_all = y_all[matched_indices]

    if config.max_train_samples is not None:
        keep = min(config.max_train_samples, len(y_all))
        x_all = x_all[:keep]
        y_all = y_all[:keep]

    idx_train_val, idx_test = train_test_split(
        np.arange(len(y_all)),
        test_size=config.test_ratio,
        random_state=config.random_seed,
        shuffle=True,
    )
    val_ratio_within_train_val = config.val_ratio / (1.0 - config.test_ratio)
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=val_ratio_within_train_val,
        random_state=config.random_seed,
        shuffle=True,
    )
    split_stats = {
        "train_ratio": float(1.0 - config.val_ratio - config.test_ratio),
        "test_ratio": config.test_ratio,
        "val_ratio": config.val_ratio,
        "train_samples": int(len(idx_train)),
        "test_samples": int(len(idx_test)),
        "val_samples": int(len(idx_val)),
    }


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
        FeatureDataset(x_all[idx_test], y_all[idx_test]),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = ResNet6Features(
        input_dim=x_all.shape[1],
        num_classes=y_all.shape[1],
        channels=config.channels,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_map = -1.0
    history: list[dict[str, float]] = []
    checkpoint_path = args.output_dir / "resnet6_features_clean_best.pt"

    print("Alignment metadata:")
    print(json.dumps(alignment_metadata, indent=2))
    print(f"Samples: {len(y_all)}")
    print(f"Test ratio: {split_stats['test_ratio']:.2f}")
    print(f"Train samples: {len(idx_train)} | Val samples: {len(idx_val)} | Test samples: {len(idx_test)}")
    print(f"Group names: {GROUP_NAMES}")
    print(f"Group dims: {group_dims}")
    print(f"Input dim: {x_all.shape[1]} | Classes: {y_all.shape[1]}")

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
                    "reference_config_path": str(args.config_path),
                    "input_dim": x_all.shape[1],
                    "num_classes": y_all.shape[1],
                    "best_val_metrics": val_metrics,
                    "group_dims": group_dims,
                    "group_names": GROUP_NAMES,
                    "split": split_stats,
                    "alignment_metadata": alignment_metadata,
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

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, config.threshold)

    save_json(args.output_dir / "training_history.json", history)
    save_json(args.output_dir / "test_metrics.json", test_metrics)
    save_json(
        args.output_dir / "config.json",
        {
            **asdict(config),
            "reference_config_path": str(args.config_path),
            "group_names": GROUP_NAMES,
            "group_dims": group_dims,
            "train_feature_paths": [str(path) for path in TRAIN_FEATURE_PATHS],
            "local_image_list": str(args.local_image_list),
            "official_image_list": str(args.official_image_list),
            "alignment_metadata": alignment_metadata,
            "split": split_stats,
            "loss": "BCEWithLogitsLoss",
            "model": "ResNet6Features",
            "reported_metrics": "test_metrics.json",
        },
    )
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
