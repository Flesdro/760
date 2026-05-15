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


VISUAL_TRAIN_FEATURE_PATHS = [
    Path("Extracted_Features/Normalized_CH.npy"),
    Path("Extracted_Features/Normalized_CM55.npy"),
    Path("Extracted_Features/Normalized_CORR.npy"),
    Path("Extracted_Features/Normalized_EDH.npy"),
    Path("Extracted_Features/Normalized_WT.npy"),
]
VISUAL_TEST_FEATURE_PATHS = [
    Path("Extracted_Features_Test/Normalized_CH.npy"),
    Path("Extracted_Features_Test/Normalized_CM55.npy"),
    Path("Extracted_Features_Test/Normalized_CORR.npy"),
    Path("Extracted_Features_Test/Normalized_EDH.npy"),
    Path("Extracted_Features_Test/Normalized_WT.npy"),
]
VISUAL_GROUP_NAMES = ["ch", "cm55", "corr", "edh", "wt"]


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 512
    num_blocks: int = 4
    dropout: float = 0.3
    val_ratio: float = 0.1
    threshold: float = 0.5
    random_seed: int = 42
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 1.0
    asl_clip: float = 0.05
    max_train_samples: int | None = None
    max_test_samples: int | None = None


class FeatureGroupDataset(Dataset):
    def __init__(self, feature_groups: list[np.ndarray], labels: np.ndarray):
        self.feature_groups = [torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)) for features in feature_groups]
        self.labels = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[list[torch.Tensor], torch.Tensor]:
        return [features[idx] for features in self.feature_groups], self.labels[idx]


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SemanticTagGatedFusionResidualMLP(nn.Module):
    def __init__(
        self,
        group_dims: list[int],
        group_names: list[str],
        num_classes: int,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        if len(group_dims) != len(group_names):
            raise ValueError("group_dims and group_names must have the same length")

        self.group_names = group_names
        self.group_embed_dims = [
            infer_group_embed_dim(group_dim)
            for group_dim in group_dims
        ]
        self.group_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(group_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for group_dim, embed_dim in zip(group_dims, self.group_embed_dims)
            ]
        )

        fused_dim = sum(self.group_embed_dims)
        self.gate = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(group_dims)),
        )
        self.input_proj = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, feature_groups: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_groups = [encoder(features) for encoder, features in zip(self.group_encoders, feature_groups)]
        concat = torch.cat(encoded_groups, dim=1)
        gate_weights = torch.softmax(self.gate(concat), dim=1)
        gated_groups = [
            encoded * gate_weights[:, idx : idx + 1]
            for idx, encoded in enumerate(encoded_groups)
        ]
        fused = torch.cat(gated_groups, dim=1)
        hidden = self.input_proj(fused)
        hidden = self.blocks(hidden)
        logits = self.classifier(hidden)
        return logits, gate_weights


class AsymmetricLossWithLogits(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pos_probs = probs
        neg_probs = 1.0 - probs

        if self.clip is not None and self.clip > 0:
            neg_probs = (neg_probs + self.clip).clamp(max=1.0)

        pos_loss = targets * torch.log(pos_probs.clamp(min=self.eps))
        neg_loss = (1.0 - targets) * torch.log(neg_probs.clamp(min=self.eps))
        pos_weight = torch.pow(1.0 - pos_probs, self.gamma_pos)
        neg_weight = torch.pow(1.0 - neg_probs, self.gamma_neg)
        return -(pos_weight * pos_loss + neg_weight * neg_loss).mean()


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


def infer_group_embed_dim(group_dim: int) -> int:
    if group_dim >= 1000:
        return 512
    if group_dim >= 200:
        return 256
    return 128


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    return np.loadtxt(path).astype(np.float32)


def trim_to_match(features: np.ndarray, labels: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    if features.shape[0] == labels.shape[0]:
        return features, labels
    n_rows = min(features.shape[0], labels.shape[0])
    print(f"Warning: trimming {name} rows to {n_rows} ({features.shape[0]} features, {labels.shape[0]} labels).")
    return features[:n_rows], labels[:n_rows]


def validate_args(config: TrainConfig) -> None:
    if config.epochs < 1:
        raise ValueError("--epochs must be at least 1 so a checkpoint can be selected.")
    if config.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if not 0.0 < config.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")
    if config.max_train_samples is not None and config.max_train_samples < 2:
        raise ValueError("--max-train-samples must be at least 2 when provided.")
    if config.max_test_samples is not None and config.max_test_samples < 1:
        raise ValueError("--max-test-samples must be at least 1 when provided.")


def load_feature_groups(
    data_root: Path,
    paths: list[Path],
    tag_path: Path,
) -> tuple[list[np.ndarray], list[str]]:
    arrays = [load_array(data_root / path) for path in paths]
    tag_features = load_array(tag_path if tag_path.is_absolute() else data_root / tag_path)
    arrays.append(tag_features)
    group_names = [*VISUAL_GROUP_NAMES, "semantic_tag"]

    n_rows = {array.shape[0] for array in arrays}
    if len(n_rows) != 1:
        min_rows = min(n_rows)
        print(
            "Warning: feature row counts do not match "
            f"{sorted(n_rows)}; trimming all groups to {min_rows} rows by prefix."
        )
        arrays = [array[:min_rows] for array in arrays]
    return arrays, group_names


def select_rows(feature_groups: list[np.ndarray], indices: np.ndarray) -> list[np.ndarray]:
    return [features[indices] for features in feature_groups]


def evaluate(
    model: SemanticTagGatedFusionResidualMLP,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    group_names: list[str],
) -> dict[str, float]:
    model.eval()
    all_targets, all_probs, all_preds, all_gates = [], [], [], []

    with torch.no_grad():
        for feature_groups, y in loader:
            feature_groups = [features.to(device) for features in feature_groups]
            logits, gate_weights = model(feature_groups)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_preds.append((probs > threshold).float().cpu().numpy())
            all_targets.append(y.numpy())
            all_gates.append(gate_weights.cpu().numpy())

    targets = np.vstack(all_targets)
    probs = np.vstack(all_probs)
    preds = np.vstack(all_preds)
    gates = np.vstack(all_gates)
    valid_classes = targets.sum(axis=0) > 0
    if valid_classes.any():
        map_score = average_precision_score(targets[:, valid_classes], probs[:, valid_classes], average="macro")
    else:
        map_score = 0.0
    metrics = {
        "mAP": float(map_score),
        "micro_f1": float(f1_score(targets, preds, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
    }
    for name, weight in zip(group_names, gates.mean(axis=0)):
        metrics[f"gate_{name}_mean"] = float(weight)
    return metrics


def train_one_epoch(
    model: SemanticTagGatedFusionResidualMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_seen = 0

    for feature_groups, y in loader:
        feature_groups = [features.to(device) for features in feature_groups]
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(feature_groups)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        total_seen += len(y)

    return total_loss / max(total_seen, 1)


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Residual MLP + ASL + Semantic Tag Modality + Gated Fusion."
    )
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("Semantic_Tag_Gated_Fusion_Residual_MLP_ASL/runs"))
    parser.add_argument("--train-tag-path", type=Path, default=Path("NUS_WID_Tags/Train_Tags1k.dat"))
    parser.add_argument("--test-tag-path", type=Path, default=Path("NUS_WID_Tags/Test_Tags1k.dat"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--asl-gamma-neg", type=float, default=4.0)
    parser.add_argument("--asl-gamma-pos", type=float, default=1.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap for quick smoke tests.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional cap for quick smoke tests.")
    args = parser.parse_args()

    config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        val_ratio=args.val_ratio,
        threshold=args.threshold,
        random_seed=args.seed,
        asl_gamma_neg=args.asl_gamma_neg,
        asl_gamma_pos=args.asl_gamma_pos,
        asl_clip=args.asl_clip,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )
    validate_args(config)
    set_seed(config.random_seed)
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print("Loading visual feature groups plus semantic tag modality...")
    feature_groups, group_names = load_feature_groups(
        args.data_root,
        VISUAL_TRAIN_FEATURE_PATHS,
        args.train_tag_path,
    )
    test_feature_groups, test_group_names = load_feature_groups(
        args.data_root,
        VISUAL_TEST_FEATURE_PATHS,
        args.test_tag_path,
    )
    if group_names != test_group_names:
        raise ValueError(f"Train/test group names differ: {group_names} vs {test_group_names}")

    y_all = load_array(args.data_root / "database_labels_81_big.npy")
    y_test = load_array(args.data_root / "database_labels_81_test.npy")

    feature_groups = [features for features in feature_groups]
    test_feature_groups = [features for features in test_feature_groups]
    feature_groups[0], y_all = trim_to_match(feature_groups[0], y_all, "train labels")
    feature_groups = [features[: len(y_all)] for features in feature_groups]
    test_feature_groups[0], y_test = trim_to_match(test_feature_groups[0], y_test, "test labels")
    test_feature_groups = [features[: len(y_test)] for features in test_feature_groups]

    if config.max_train_samples is not None:
        y_all = y_all[: config.max_train_samples]
        feature_groups = [features[: config.max_train_samples] for features in feature_groups]
    if config.max_test_samples is not None:
        y_test = y_test[: config.max_test_samples]
        test_feature_groups = [features[: config.max_test_samples] for features in test_feature_groups]

    idx_train, idx_val = train_test_split(
        np.arange(len(y_all)),
        test_size=config.val_ratio,
        random_state=config.random_seed,
        shuffle=True,
    )

    train_loader = DataLoader(
        FeatureGroupDataset(select_rows(feature_groups, idx_train), y_all[idx_train]),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        FeatureGroupDataset(select_rows(feature_groups, idx_val), y_all[idx_val]),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        FeatureGroupDataset(test_feature_groups, y_test),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    group_dims = [features.shape[1] for features in feature_groups]
    model = SemanticTagGatedFusionResidualMLP(
        group_dims=group_dims,
        group_names=group_names,
        num_classes=y_all.shape[1],
        hidden_dim=config.hidden_dim,
        num_blocks=config.num_blocks,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = AsymmetricLossWithLogits(
        gamma_neg=config.asl_gamma_neg,
        gamma_pos=config.asl_gamma_pos,
        clip=config.asl_clip,
    )

    best_map = -1.0
    history: list[dict[str, float]] = []
    checkpoint_path = args.output_dir / "semantic_tag_gated_fusion_residual_mlp_asl_best.pt"

    print(f"Train samples: {len(idx_train)} | Val samples: {len(idx_val)} | Test samples: {len(y_test)}")
    print(f"Groups: {list(zip(group_names, group_dims))} | Classes: {y_all.shape[1]}")
    print(
        f"Group embed dims: {model.group_embed_dims} | "
        f"Hidden dim: {config.hidden_dim} | Residual blocks: {config.num_blocks}"
    )
    print(
        f"ASL gamma_neg: {config.asl_gamma_neg} | "
        f"gamma_pos: {config.asl_gamma_pos} | clip: {config.asl_clip}"
    )

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device, config.threshold, group_names)
        row = {"epoch": epoch, "train_loss": float(train_loss), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)

        if val_metrics["mAP"] > best_map:
            best_map = val_metrics["mAP"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "group_dims": group_dims,
                    "group_names": group_names,
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
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"tag_gate={val_metrics.get('gate_semantic_tag_mean', 0.0):.4f}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, config.threshold, group_names)

    save_json(args.output_dir / "training_history.json", history)
    save_json(args.output_dir / "test_metrics.json", test_metrics)
    save_json(
        args.output_dir / "config.json",
        {
            **asdict(config),
            "train_tag_path": str(args.train_tag_path),
            "test_tag_path": str(args.test_tag_path),
            "group_names": group_names,
            "group_dims": group_dims,
        },
    )
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
