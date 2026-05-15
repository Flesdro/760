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
    batch_size: int = 128
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    group_embed_dim: int = 128
    hidden_dim: int = 512
    num_blocks: int = 4
    dropout: float = 0.3
    val_ratio: float = 0.2
    threshold: float = 0.5
    random_seed: int = 42
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 1.0
    asl_clip: float = 0.05


class FeatureGroupDataset(Dataset):
    def __init__(self, feature_groups: list[np.ndarray], labels: np.ndarray):
        self.feature_groups = [torch.tensor(features, dtype=torch.float32) for features in feature_groups]
        self.labels = torch.tensor(labels, dtype=torch.float32)

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


class GatedFusionResidualMLP(nn.Module):
    def __init__(
        self,
        group_dims: list[int],
        num_classes: int,
        group_embed_dim: int = 128,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.group_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(group_dim, group_embed_dim),
                    nn.LayerNorm(group_embed_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for group_dim in group_dims
            ]
        )
        self.gate = nn.Sequential(
            nn.Linear(group_embed_dim * len(group_dims), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(group_dims)),
        )
        fused_dim = group_embed_dim * len(group_dims)
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
        stacked = torch.stack(encoded_groups, dim=1)
        concat = torch.cat(encoded_groups, dim=1)
        gate_weights = torch.softmax(self.gate(concat), dim=1)
        gated = stacked * gate_weights.unsqueeze(-1)
        fused = gated.flatten(start_dim=1)
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


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")
    return np.load(path).astype(np.float32)


def load_feature_groups(data_root: Path, paths: list[Path]) -> list[np.ndarray]:
    arrays = [load_array(data_root / path) for path in paths]
    n_rows = {array.shape[0] for array in arrays}
    if len(n_rows) != 1:
        raise ValueError(f"Feature row counts do not match: {sorted(n_rows)}")
    return arrays


def select_rows(feature_groups: list[np.ndarray], indices: np.ndarray) -> list[np.ndarray]:
    return [features[indices] for features in feature_groups]


def evaluate(
    model: GatedFusionResidualMLP,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
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
    for idx, weight in enumerate(gates.mean(axis=0)):
        metrics[f"gate_{idx}_mean"] = float(weight)
    return metrics


def train_one_epoch(
    model: GatedFusionResidualMLP,
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


def print_run_settings(args: argparse.Namespace, config: TrainConfig) -> None:
    print("\nRun settings:")
    print(json.dumps(asdict(config), indent=2))
    print(f"data_root: {args.data_root}")
    print(f"output_dir: {args.output_dir}")
    print(f"num_workers: {args.num_workers}")
    print()


def main() -> None:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train a Gated Fusion Residual MLP with ASL on feature groups.")
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("Gated_Fusion_Residual_MLP_ASL/runs"))
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--group-embed-dim", type=int, default=defaults.group_embed_dim)
    parser.add_argument("--hidden-dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--num-blocks", type=int, default=defaults.num_blocks)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--threshold", type=float, default=defaults.threshold)
    parser.add_argument("--val-ratio", type=float, default=defaults.val_ratio)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--asl-gamma-neg", type=float, default=defaults.asl_gamma_neg)
    parser.add_argument("--asl-gamma-pos", type=float, default=defaults.asl_gamma_pos)
    parser.add_argument("--asl-clip", type=float, default=defaults.asl_clip)
    args = parser.parse_args()

    config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        group_embed_dim=args.group_embed_dim,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        val_ratio=args.val_ratio,
        threshold=args.threshold,
        random_seed=args.seed,
        asl_gamma_neg=args.asl_gamma_neg,
        asl_gamma_pos=args.asl_gamma_pos,
        asl_clip=args.asl_clip,
    )
    set_seed(config.random_seed)
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print_run_settings(args, config)
    print(f"Using device: {device}")
    print("Loading feature groups...")
    feature_groups = load_feature_groups(args.data_root, TRAIN_FEATURE_PATHS)
    y_all = load_array(args.data_root / "database_labels_81_big.npy")
    test_feature_groups = load_feature_groups(args.data_root, TEST_FEATURE_PATHS)
    y_test = load_array(args.data_root / "database_labels_81_test.npy")

    if feature_groups[0].shape[0] != y_all.shape[0]:
        raise ValueError(f"Train features/labels mismatch: {feature_groups[0].shape[0]} vs {y_all.shape[0]}")
    if test_feature_groups[0].shape[0] != y_test.shape[0]:
        raise ValueError(f"Test features/labels mismatch: {test_feature_groups[0].shape[0]} vs {y_test.shape[0]}")

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
    model = GatedFusionResidualMLP(
        group_dims=group_dims,
        num_classes=y_all.shape[1],
        group_embed_dim=config.group_embed_dim,
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
    checkpoint_path = args.output_dir / "gated_fusion_residual_mlp_asl_best.pt"

    print(f"Train samples: {len(idx_train)} | Val samples: {len(idx_val)} | Test samples: {len(y_test)}")
    print(f"Group dims: {group_dims} | Classes: {y_all.shape[1]}")
    print(
        f"Group embed dim: {config.group_embed_dim} | "
        f"Hidden dim: {config.hidden_dim} | Residual blocks: {config.num_blocks}"
    )
    print(
        f"ASL gamma_neg: {config.asl_gamma_neg} | "
        f"gamma_pos: {config.asl_gamma_pos} | clip: {config.asl_clip}"
    )

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
                    "group_dims": group_dims,
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
