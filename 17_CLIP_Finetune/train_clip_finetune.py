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
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 30
    lr_backbone: float = 1e-5
    lr_head: float = 1e-4
    patience: int = 8
    test_ratio: float = 0.1
    threshold: float = 0.5
    random_seed: int = 42
    dropout: float = 0.3
    unfreeze_blocks: int = 2
    clip_model: str = "ViT-B/32"
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 1.0
    asl_clip: float = 0.05
    limit_samples: int | None = None


class NUSWIDEImageDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        labels: np.ndarray,
        preprocess,
    ):
        self.image_paths = image_paths
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.preprocess(img), self.labels[idx]


class CLIPFineTuner(nn.Module):
    def __init__(
        self,
        clip_model,
        num_classes: int,
        dropout: float = 0.3,
        unfreeze_blocks: int = 2,
    ):
        super().__init__()
        self.visual = clip_model.visual
        feat_dim = clip_model.visual.output_dim

        # Freeze all CLIP visual parameters first
        for param in self.visual.parameters():
            param.requires_grad = False

        # Unfreeze the last N transformer blocks
        resblocks = self.visual.transformer.resblocks
        num_blocks = len(resblocks)
        for i in range(num_blocks - unfreeze_blocks, num_blocks):
            for param in resblocks[i].parameters():
                param.requires_grad = True

        # Always unfreeze the final layer norm and projection
        for param in self.visual.ln_post.parameters():
            param.requires_grad = True
        if self.visual.proj is not None:
            self.visual.proj.requires_grad = True

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.visual(images.half() if images.dtype == torch.float16 else images)
        features = features.float()
        return self.classifier(features)


class AsymmetricLossWithLogits(nn.Module):
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pos_probs = probs
        neg_probs = (1.0 - probs + self.clip).clamp(max=1.0) if self.clip > 0 else 1.0 - probs
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


def build_valid_samples(
    image_list_path: Path,
    images_dir: Path,
    labels: np.ndarray,
) -> tuple[list[Path], np.ndarray]:
    """Return (image_paths, filtered_labels) for images that exist on disk."""
    raw_paths = [
        line.strip() for line in image_list_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    valid_paths: list[Path] = []
    valid_label_indices: list[int] = []

    for idx, raw in enumerate(raw_paths):
        filename = os.path.basename(raw.replace("\\", "/"))
        full = images_dir / filename
        if full.exists():
            valid_paths.append(full)
            valid_label_indices.append(idx)

    valid_labels = labels[np.array(valid_label_indices)]
    return valid_paths, valid_labels


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    all_targets, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
            probs = torch.sigmoid(logits.float())
            all_probs.append(probs.cpu().numpy())
            all_preds.append((probs > threshold).float().cpu().numpy())
            all_targets.append(labels.numpy())

    targets = np.vstack(all_targets)
    probs = np.vstack(all_probs)
    preds = np.vstack(all_preds)
    valid_classes = targets.sum(axis=0) > 0
    map_score = (
        average_precision_score(targets[:, valid_classes], probs[:, valid_classes], average="macro")
        if valid_classes.any() else 0.0
    )
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
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_seen = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = criterion(model(images), labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(labels)
        total_seen += len(labels)

    return total_loss / max(total_seen, 1)


def main() -> None:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Fine-tune CLIP image encoder end-to-end for multi-label tagging."
    )
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("17_CLIP_Finetune/runs"))
    parser.add_argument("--image-list", type=Path, default=Path("database_img.txt"))
    parser.add_argument("--images-dir", type=Path, default=Path("NUS-WIDE-images/images"))
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr-backbone", type=float, default=defaults.lr_backbone)
    parser.add_argument("--lr-head", type=float, default=defaults.lr_head)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--threshold", type=float, default=defaults.threshold)
    parser.add_argument("--test-ratio", type=float, default=defaults.test_ratio)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--unfreeze-blocks", type=int, default=defaults.unfreeze_blocks,
                        help="Number of final CLIP transformer blocks to unfreeze (default 2).")
    parser.add_argument("--clip-model", type=str, default=defaults.clip_model,
                        help="CLIP backbone: ViT-B/32 (default) or ViT-B/16. ViT-L/14 may OOM on 6GB VRAM.")
    parser.add_argument("--asl-gamma-neg", type=float, default=defaults.asl_gamma_neg)
    parser.add_argument("--asl-gamma-pos", type=float, default=defaults.asl_gamma_pos)
    parser.add_argument("--asl-clip", type=float, default=defaults.asl_clip)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed-precision training.")
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    args.data_root = resolve_project_path(args.data_root)
    args.output_dir = resolve_project_path(args.output_dir)
    args.image_list = resolve_project_path(args.image_list)
    args.images_dir = resolve_project_path(args.images_dir)
    use_amp = not args.no_amp and torch.cuda.is_available()

    config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        patience=args.patience,
        test_ratio=args.test_ratio,
        threshold=args.threshold,
        random_seed=args.seed,
        dropout=args.dropout,
        unfreeze_blocks=args.unfreeze_blocks,
        clip_model=args.clip_model,
        asl_gamma_neg=args.asl_gamma_neg,
        asl_gamma_pos=args.asl_gamma_pos,
        asl_clip=args.asl_clip,
        limit_samples=args.limit_samples,
    )
    set_seed(config.random_seed)
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import clip
    except ImportError as e:
        raise ImportError(
            "openai-clip is not installed. Run:\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        ) from e

    print(f"Loading CLIP {config.clip_model}...")
    clip_model, preprocess = clip.load(config.clip_model, device="cpu")
    clip_model = clip_model.float()

    print("Building valid sample list (scanning image files)...")
    labels_all = load_array(args.data_root / "database_labels_81_big.npy")
    image_paths, labels = build_valid_samples(args.image_list, args.images_dir, labels_all)
    print(f"Valid images found: {len(image_paths)} / {len(labels_all)}")

    if config.limit_samples is not None:
        keep = min(config.limit_samples, len(image_paths))
        image_paths = image_paths[:keep]
        labels = labels[:keep]

    num_classes = labels.shape[1]
    idx_train, idx_test = train_test_split(
        np.arange(len(labels)),
        test_size=config.test_ratio,
        random_state=config.random_seed,
        shuffle=True,
    )

    train_paths = [image_paths[i] for i in idx_train]
    test_paths = [image_paths[i] for i in idx_test]

    train_loader = DataLoader(
        NUSWIDEImageDataset(train_paths, labels[idx_train], preprocess),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        NUSWIDEImageDataset(test_paths, labels[idx_test], preprocess),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = CLIPFineTuner(
        clip_model=clip_model,
        num_classes=num_classes,
        dropout=config.dropout,
        unfreeze_blocks=config.unfreeze_blocks,
    ).to(device)

    # Differential learning rates: lower LR for backbone, normal for classifier head
    backbone_params = [p for p in model.visual.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": config.lr_backbone},
        {"params": head_params, "lr": config.lr_head},
    ])

    criterion = AsymmetricLossWithLogits(
        gamma_neg=config.asl_gamma_neg,
        gamma_pos=config.asl_gamma_pos,
        clip=config.asl_clip,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print("\nRun settings:")
    print(json.dumps(asdict(config), indent=2))
    print(f"Train samples: {len(idx_train)} | Test samples: {len(idx_test)}")
    print(f"Mixed precision: {use_amp}")
    print(f"Using device: {device}\n")

    best_map = -1.0
    best_test_metrics: dict[str, float] = {}
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, float | int]] = []
    checkpoint_path = args.output_dir / "clip_finetune_best.pt"

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp)
        test_metrics = evaluate(model, test_loader, device, config.threshold, use_amp)
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
                    "num_classes": num_classes,
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
    final_test_metrics = evaluate(model, test_loader, device, config.threshold, use_amp)
    reported_metrics = {
        **final_test_metrics,
        "evaluation_split": "test",
        "best_epoch": best_epoch,
        "best_test_mAP": best_test_metrics.get("mAP", 0.0),
    }

    save_json(args.output_dir / "training_history.json", history)
    save_json(args.output_dir / "test_metrics.json", reported_metrics)
    save_json(args.output_dir / "config.json", asdict(config))
    print(f"\nBest checkpoint: {checkpoint_path}")
    print(f"Reported metrics: {reported_metrics}")


if __name__ == "__main__":
    main()
