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


HANDCRAFTED_FEATURE_PATHS = [
    Path("Extracted_Features/Normalized_CH.npy"),
    Path("Extracted_Features/Normalized_CM55.npy"),
    Path("Extracted_Features/Normalized_CORR.npy"),
    Path("Extracted_Features/Normalized_EDH.npy"),
    Path("Extracted_Features/Normalized_WT.npy"),
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    batch_size: int = 256
    epochs: int = 50
    lr: float = 1e-4
    patience: int = 10
    test_ratio: float = 0.1
    threshold: float = 0.5
    random_seed: int = 42
    dropout: float = 0.3
    refiner_alpha: float = 0.2
    use_refiner: bool = True
    use_handcrafted: bool = False
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 1.0
    asl_clip: float = 0.05
    limit_samples: int | None = None


class MultiViewDataset(Dataset):
    def __init__(self, views: list[np.ndarray], labels: np.ndarray):
        self.views = [torch.tensor(view, dtype=torch.float32) for view in views]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[list[torch.Tensor], torch.Tensor]:
        return [view[idx] for view in self.views], self.labels[idx]


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MultiViewBackbone(nn.Module):
    def __init__(self, view_dims: list[int], num_classes: int = 81, dropout: float = 0.3):
        super().__init__()
        self.encoders = nn.ModuleList()
        fusion_dim = 0

        for dim in view_dims:
            if dim >= 512:
                hidden_dim = 512
            elif dim >= 200:
                hidden_dim = 256
            else:
                hidden_dim = 128

            self.encoders.append(FeatureEncoder(dim, hidden_dim, dropout))
            fusion_dim += hidden_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.shortcut = nn.Linear(fusion_dim, 512)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, views: list[torch.Tensor]) -> torch.Tensor:
        encoded_views = [encoder(view) for encoder, view in zip(self.encoders, views)]
        fused = torch.cat(encoded_views, dim=1)
        fused_feat = self.fusion(fused) + self.shortcut(fused)
        return self.classifier(fused_feat)


class LabelCorrelationRefiner(nn.Module):
    def __init__(self, correlation_matrix: np.ndarray, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha
        self.register_buffer("M", torch.tensor(correlation_matrix, dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + self.alpha * torch.matmul(logits, self.M)


class MultiViewModel(nn.Module):
    def __init__(
        self,
        backbone: MultiViewBackbone,
        refiner: LabelCorrelationRefiner | None = None,
        use_refiner: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.refiner = refiner
        self.use_refiner = use_refiner

    def forward(self, views: list[torch.Tensor]) -> torch.Tensor:
        logits = self.backbone(views)
        if self.use_refiner and self.refiner is not None:
            logits = self.refiner(logits)
        return logits


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


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def select_rows(views: list[np.ndarray], indices: np.ndarray) -> list[np.ndarray]:
    return [view[indices] for view in views]


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float) -> dict[str, float]:
    model.eval()
    all_targets, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for views, labels in loader:
            views = [view.to(device) for view in views]
            logits = model(views)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_preds.append((probs > threshold).float().cpu().numpy())
            all_targets.append(labels.numpy())

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

    for views, labels in loader:
        views = [view.to(device) for view in views]
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(views), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        total_seen += len(labels)

    return total_loss / max(total_seen, 1)


def print_run_settings(args: argparse.Namespace, config: TrainConfig, n_samples: int, view_dims: list[int]) -> None:
    print("\nRun settings:")
    print(json.dumps(asdict(config), indent=2))
    print(f"data_root:       {args.data_root}")
    print(f"output_dir:      {args.output_dir}")
    print(f"clip_feats_path: {args.clip_feats_path}")
    print(f"num_workers:     {args.num_workers}")
    print(f"total_samples:   {n_samples}")
    print(f"view_dims:       {view_dims}")
    print()


def main() -> None:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Train multi-label classifier using CLIP features with Asymmetric Loss."
    )
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("14_CLIP_ASL/runs"))
    parser.add_argument(
        "--clip-feats-path", type=Path,
        default=Path("13_CLIP_Visual_Features/clip_visual_features.npy"),
        help="Pre-extracted CLIP feature array (N, 512) from experiment 13.",
    )
    parser.add_argument("--graph-path", type=Path, default=Path("label_graph_fused.npy"))
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--threshold", type=float, default=defaults.threshold)
    parser.add_argument("--test-ratio", type=float, default=defaults.test_ratio)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--refiner-alpha", type=float, default=defaults.refiner_alpha)
    parser.add_argument("--disable-refiner", action="store_true")
    parser.add_argument(
        "--use-handcrafted", action="store_true",
        help="Also include the 5 handcrafted visual features (CH/CM55/CORR/EDH/WT) as additional views.",
    )
    parser.add_argument("--asl-gamma-neg", type=float, default=defaults.asl_gamma_neg)
    parser.add_argument("--asl-gamma-pos", type=float, default=defaults.asl_gamma_pos)
    parser.add_argument("--asl-clip", type=float, default=defaults.asl_clip)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    args.data_root = resolve_project_path(args.data_root)
    args.output_dir = resolve_project_path(args.output_dir)
    args.clip_feats_path = resolve_project_path(args.clip_feats_path)
    args.graph_path = resolve_project_path(args.graph_path)

    config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        test_ratio=args.test_ratio,
        threshold=args.threshold,
        random_seed=args.seed,
        dropout=args.dropout,
        refiner_alpha=args.refiner_alpha,
        use_refiner=not args.disable_refiner,
        use_handcrafted=args.use_handcrafted,
        asl_gamma_neg=args.asl_gamma_neg,
        asl_gamma_pos=args.asl_gamma_pos,
        asl_clip=args.asl_clip,
        limit_samples=args.limit_samples,
    )
    set_seed(config.random_seed)
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CLIP features...")
    clip_feats = load_array(args.clip_feats_path)
    labels = load_array(args.data_root / "database_labels_81_big.npy")

    if clip_feats.shape[0] != labels.shape[0]:
        raise ValueError(
            f"CLIP feature rows ({clip_feats.shape[0]}) != label rows ({labels.shape[0]}). "
            "Re-run 13_CLIP_Visual_Features/extract_clip_features.py."
        )

    valid_mask = clip_feats.any(axis=1)
    valid_indices = np.where(valid_mask)[0]
    print(f"Valid (non-zero) CLIP feature rows: {len(valid_indices)} / {len(clip_feats)}")

    views: list[np.ndarray] = [clip_feats[valid_indices]]

    if config.use_handcrafted:
        print("Loading handcrafted visual features...")
        for relative_path in HANDCRAFTED_FEATURE_PATHS:
            feat = load_array(args.data_root / relative_path)
            views.append(feat[valid_indices])

    labels = labels[valid_indices]

    if config.limit_samples is not None:
        keep = min(config.limit_samples, len(labels))
        views = [v[:keep] for v in views]
        labels = labels[:keep]

    view_dims = [v.shape[1] for v in views]
    print_run_settings(args, config, len(labels), view_dims)
    print(f"Using device: {device}")

    idx_train, idx_test = train_test_split(
        np.arange(len(labels)),
        test_size=config.test_ratio,
        random_state=config.random_seed,
        shuffle=True,
    )

    train_loader = DataLoader(
        MultiViewDataset(select_rows(views, idx_train), labels[idx_train]),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        MultiViewDataset(select_rows(views, idx_test), labels[idx_test]),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    graph = load_array(args.graph_path)
    if graph.shape != (labels.shape[1], labels.shape[1]):
        raise ValueError(f"Graph shape {graph.shape} does not match num classes {labels.shape[1]}")

    backbone = MultiViewBackbone(view_dims=view_dims, num_classes=labels.shape[1], dropout=config.dropout)
    refiner = LabelCorrelationRefiner(graph, alpha=config.refiner_alpha)
    model = MultiViewModel(backbone=backbone, refiner=refiner, use_refiner=config.use_refiner).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = AsymmetricLossWithLogits(
        gamma_neg=config.asl_gamma_neg,
        gamma_pos=config.asl_gamma_pos,
        clip=config.asl_clip,
    )

    best_map = -1.0
    best_test_metrics: dict[str, float] = {}
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, float | int]] = []
    checkpoint_path = args.output_dir / "clip_asl_best.pt"

    print(f"Train samples: {len(idx_train)} | Test samples: {len(idx_test)}")
    print(f"View dims: {view_dims} | Classes: {labels.shape[1]}")
    print(f"ASL gamma_neg={config.asl_gamma_neg} gamma_pos={config.asl_gamma_pos} clip={config.asl_clip}")

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics = evaluate(model, test_loader, device, config.threshold)
        row = {"epoch": epoch, "train_loss": float(train_loss), **{f"test_{k}": v for k, v in test_metrics.items()}}
        history.append(row)

        if test_metrics["mAP"] > best_map:
            best_map = test_metrics["mAP"]
            best_test_metrics = test_metrics
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(config),
                    "view_dims": view_dims,
                    "num_classes": labels.shape[1],
                    "best_test_metrics": test_metrics,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:03d}/{config.epochs} "
            f"loss={train_loss:.4f} "
            f"test_mAP={test_metrics['mAP']:.4f} "
            f"test_micro_f1={test_metrics['micro_f1']:.4f} "
            f"test_macro_f1={test_metrics['macro_f1']:.4f}"
        )

        if patience_counter >= config.patience:
            print("Early stopping triggered.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_test_metrics = evaluate(model, test_loader, device, config.threshold)
    reported_metrics = {
        **final_test_metrics,
        "evaluation_split": "test",
        "best_epoch": best_epoch,
        "best_test_mAP": best_test_metrics.get("mAP", 0.0),
    }

    save_json(args.output_dir / "training_history.json", history)
    save_json(args.output_dir / "test_metrics.json", reported_metrics)
    save_json(args.output_dir / "config.json", asdict(config))
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Reported metrics: {reported_metrics}")


if __name__ == "__main__":
    main()
