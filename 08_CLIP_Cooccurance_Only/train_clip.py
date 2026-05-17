from __future__ import annotations

import argparse
import json
import os
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
    Path("Extracted_Features/Normalized_CH.npy"),
    Path("Extracted_Features/Normalized_CM55.npy"),
    Path("Extracted_Features/Normalized_CORR.npy"),
    Path("Extracted_Features/Normalized_EDH.npy"),
    Path("Extracted_Features/Normalized_WT.npy"),
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-4
    patience: int = 5
    val_ratio: float = 0.1
    threshold: float = 0.5
    random_seed: int = 42
    dropout: float = 0.3
    refiner_alpha: float = 0.2
    use_refiner: bool = True
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
            if dim >= 1000:
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


def normalize_name(path: str) -> str:
    return os.path.basename(path.replace("\\", "/")).lower()


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


def build_aligned_tag_cache(
    local_image_list: Path,
    official_image_list: Path,
    official_tag_path: Path,
    output_dir: Path,
    force: bool = False,
) -> tuple[Path, Path, dict[str, float | int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_tag_path = output_dir / "aligned_tag_feature.npy"
    matched_indices_path = output_dir / "matched_indices.npy"
    meta_path = output_dir / "alignment_metadata.json"

    if not force and aligned_tag_path.exists() and matched_indices_path.exists():
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            aligned_tags = np.load(aligned_tag_path, mmap_mode="r")
            matched_indices = np.load(matched_indices_path, mmap_mode="r")
            metadata = {
                "local_images": int(aligned_tags.shape[0]),
                "official_train_images": -1,
                "tag_dim": int(aligned_tags.shape[1]),
                "matched": int(matched_indices.shape[0]),
                "missing": int(aligned_tags.shape[0] - matched_indices.shape[0]),
                "match_ratio": float(matched_indices.shape[0] / max(aligned_tags.shape[0], 1)),
            }
        return aligned_tag_path, matched_indices_path, metadata

    local_imgs = [line.strip() for line in local_image_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    official_imgs = [
        line.strip() for line in official_image_list.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    local_imgs_norm = [normalize_name(path) for path in local_imgs]
    official_index = {normalize_name(path): idx for idx, path in enumerate(official_imgs)}

    official_tags = np.loadtxt(official_tag_path).astype(np.float32)
    if official_tags.shape[0] != len(official_imgs):
        raise ValueError(
            f"Official tags/list mismatch: {official_tags.shape[0]} tag rows vs {len(official_imgs)} image names"
        )

    aligned_tags = []
    matched_indices = []
    matched = 0
    missing = 0
    tag_dim = official_tags.shape[1]

    for idx, name in enumerate(local_imgs_norm):
        official_row = official_index.get(name)
        if official_row is None:
            aligned_tags.append(np.zeros(tag_dim, dtype=np.float32))
            missing += 1
        else:
            aligned_tags.append(official_tags[official_row])
            matched_indices.append(idx)
            matched += 1

    aligned_tags_array = np.asarray(aligned_tags, dtype=np.float32)
    matched_indices_array = np.asarray(matched_indices, dtype=np.int64)
    np.save(aligned_tag_path, aligned_tags_array)
    np.save(matched_indices_path, matched_indices_array)

    metadata: dict[str, float | int] = {
        "local_images": len(local_imgs_norm),
        "official_train_images": len(official_imgs),
        "tag_dim": tag_dim,
        "matched": matched,
        "missing": missing,
        "match_ratio": matched / max(len(local_imgs_norm), 1),
    }
    save_json(meta_path, metadata)
    return aligned_tag_path, matched_indices_path, metadata


def load_training_views(
    data_root: Path,
    aligned_tag_path: Path,
    matched_indices_path: Path,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    matched_indices = np.load(matched_indices_path)
    views = []

    for relative_path in TRAIN_FEATURE_PATHS:
        feature = load_array(data_root / relative_path)
        views.append(feature[matched_indices])

    tags = load_array(aligned_tag_path)
    views.append(tags[matched_indices])

    labels = load_array(data_root / "database_labels_81_big.npy")[matched_indices]
    n_rows = {view.shape[0] for view in views}
    n_rows.add(labels.shape[0])
    if len(n_rows) != 1:
        raise ValueError(f"View/label row counts do not match: {sorted(n_rows)}")
    return views, labels, matched_indices


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


def print_run_settings(args: argparse.Namespace, config: TrainConfig, alignment_metadata: dict[str, float | int]) -> None:
    print("\nRun settings:")
    print(json.dumps(asdict(config), indent=2))
    print(f"data_root: {args.data_root}")
    print(f"output_dir: {args.output_dir}")
    print(f"tag_cache_dir: {args.tag_cache_dir}")
    print(f"num_workers: {args.num_workers}")
    print("\nAlignment metadata:")
    print(json.dumps(alignment_metadata, indent=2))
    print()


def main() -> None:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train the notebook CLIP-style multi-view model.")
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("08_CLIP/runs"))
    parser.add_argument("--tag-cache-dir", type=Path, default=Path("."))
    parser.add_argument("--local-image-list", type=Path, default=Path("database_img.txt"))
    parser.add_argument("--official-image-list", type=Path, default=Path("TrainImagelist.txt"))
    parser.add_argument("--official-tag-path", type=Path, default=Path("dataset/NUS_WID_Tags/Train_Tags1k.dat"))
    parser.add_argument("--graph-path", type=Path, default=Path("label_graph_fused.npy"))
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--threshold", type=float, default=defaults.threshold)
    parser.add_argument("--val-ratio", type=float, default=defaults.val_ratio)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--refiner-alpha", type=float, default=defaults.refiner_alpha)
    parser.add_argument("--disable-refiner", action="store_true")
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    args.data_root = resolve_project_path(args.data_root)
    args.output_dir = resolve_project_path(args.output_dir)
    args.tag_cache_dir = resolve_project_path(args.tag_cache_dir)
    args.local_image_list = resolve_project_path(args.local_image_list)
    args.official_image_list = resolve_project_path(args.official_image_list)
    args.official_tag_path = resolve_project_path(args.official_tag_path)
    args.graph_path = resolve_project_path(args.graph_path)

    config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        val_ratio=args.val_ratio,
        threshold=args.threshold,
        random_seed=args.seed,
        dropout=args.dropout,
        refiner_alpha=args.refiner_alpha,
        use_refiner=not args.disable_refiner,
        limit_samples=args.limit_samples,
    )
    set_seed(config.random_seed)
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aligned_tag_path, matched_indices_path, alignment_metadata = build_aligned_tag_cache(
        local_image_list=args.local_image_list,
        official_image_list=args.official_image_list,
        official_tag_path=args.official_tag_path,
        output_dir=args.tag_cache_dir,
        force=args.force_rebuild_cache,
    )

    print_run_settings(args, config, alignment_metadata)
    print(f"Using device: {device}")
    print("Loading aligned views...")
    views, labels, matched_indices = load_training_views(args.data_root, aligned_tag_path, matched_indices_path)
    if config.limit_samples is not None:
        if config.limit_samples < 2:
            raise ValueError("--limit-samples must be at least 2")
        keep = min(config.limit_samples, len(labels))
        views = [view[:keep] for view in views]
        labels = labels[:keep]
        matched_indices = matched_indices[:keep]
    view_dims = [view.shape[1] for view in views]

    idx_train, idx_val = train_test_split(
        np.arange(len(labels)),
        test_size=config.val_ratio,
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
    val_loader = DataLoader(
        MultiViewDataset(select_rows(views, idx_val), labels[idx_val]),
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
    criterion = nn.BCEWithLogitsLoss()

    best_map = -1.0
    best_val_metrics: dict[str, float] = {}
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, float | int]] = []
    checkpoint_path = args.output_dir / "clip_multiview_best.pt"

    print(f"Train samples: {len(idx_train)} | Val samples: {len(idx_val)}")
    print(f"Matched source rows: {len(matched_indices)}")
    print(f"View dims: {view_dims} | Classes: {labels.shape[1]}")

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device, config.threshold)
        row = {"epoch": epoch, "train_loss": float(train_loss), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)

        if val_metrics["mAP"] > best_map:
            best_map = val_metrics["mAP"]
            best_val_metrics = val_metrics
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
                    "best_val_metrics": val_metrics,
                    "alignment_metadata": alignment_metadata,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:03d}/{config.epochs} "
            f"loss={train_loss:.4f} "
            f"val_mAP={val_metrics['mAP']:.4f} "
            f"val_micro_f1={val_metrics['micro_f1']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if patience_counter >= config.patience:
            print("Early stopping triggered.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_val_metrics = evaluate(model, val_loader, device, config.threshold)
    reported_metrics = {
        **final_val_metrics,
        "evaluation_split": "validation",
        "best_epoch": best_epoch,
        "best_val_mAP": best_val_metrics.get("mAP", 0.0),
    }

    save_json(args.output_dir / "training_history.json", history)
    save_json(args.output_dir / "test_metrics.json", reported_metrics)
    save_json(args.output_dir / "config.json", asdict(config))
    save_json(args.output_dir / "alignment_metadata.json", alignment_metadata)
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Reported metrics: {reported_metrics}")


if __name__ == "__main__":
    main()
