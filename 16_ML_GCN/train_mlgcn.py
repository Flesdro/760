from __future__ import annotations

import argparse
import json
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
    batch_size: int = 256
    epochs: int = 50
    lr: float = 1e-4
    patience: int = 10
    test_ratio: float = 0.1
    threshold: float = 0.5
    random_seed: int = 42
    dropout: float = 0.3
    gcn_hidden_dim: int = 1024
    graph_threshold: float = 0.01
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 1.0
    asl_clip: float = 0.05
    label_embed_model: str = "ViT-B/32"
    limit_samples: int | None = None


class MultiLabelDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x:   (num_labels, in_features)
        # adj: (num_labels, num_labels)
        return adj @ (x @ self.weight)


class MLGCN(nn.Module):
    """
    Multi-Label GCN classifier.

    Two GCN layers propagate label embeddings through the co-occurrence graph
    to produce a label-specific classifier weight matrix W (num_classes x feat_dim).
    Classification: logits = img_features @ W.T
    """

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        label_embeddings: torch.Tensor,
        adj: torch.Tensor,
        gcn_hidden_dim: int = 1024,
        dropout: float = 0.3,
    ):
        super().__init__()
        embed_dim = label_embeddings.shape[1]
        self.gc1 = GraphConvolution(embed_dim, gcn_hidden_dim)
        self.gc2 = GraphConvolution(gcn_hidden_dim, feat_dim)
        self.dropout = nn.Dropout(dropout)
        # Learnable label embeddings — initialised from CLIP text, fine-tuned during training
        self.label_embeddings = nn.Parameter(label_embeddings.clone())
        self.register_buffer("adj", adj)

    def forward(self, img_features: torch.Tensor) -> torch.Tensor:
        # Generate label-specific classifier weights via GCN
        x = F.leaky_relu(self.gc1(self.label_embeddings, self.adj), negative_slope=0.2)
        x = self.dropout(x)
        W = self.gc2(x, self.adj)  # (num_classes, feat_dim)

        # Raw dot product — no normalisation so logits are not compressed
        return img_features @ W.T  # (batch, num_classes)


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


def build_adjacency(graph: np.ndarray, threshold: float) -> torch.Tensor:
    """Threshold, add self-loops, row-normalise."""
    adj = graph.copy()
    adj[adj < threshold] = 0.0
    np.fill_diagonal(adj, 1.0)
    row_sum = adj.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    adj = adj / row_sum
    return torch.tensor(adj, dtype=torch.float32)


def build_label_embeddings(
    label_names: list[str],
    clip_model: str,
    device: torch.device,
) -> torch.Tensor:
    try:
        import clip
    except ImportError as e:
        raise ImportError(
            "openai-clip is not installed. Run:\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        ) from e

    print(f"Generating label embeddings with CLIP {clip_model}...")
    model, _ = clip.load(clip_model, device=device)
    model.eval()
    prompts = [f"a photo of {name}" for name in label_names]
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        embeddings = model.encode_text(tokens).float()
    embeddings = F.normalize(embeddings, p=2, dim=1)
    print(f"Label embeddings: {embeddings.shape}")
    return embeddings.cpu()


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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float) -> dict[str, float]:
    model.eval()
    all_targets, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            logits = model(features)
            probs = torch.sigmoid(logits)
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
) -> float:
    model.train()
    total_loss = 0.0
    total_seen = 0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(features), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        total_seen += len(labels)

    return total_loss / max(total_seen, 1)


def main() -> None:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="ML-GCN: graph-convolutional label-specific classifiers on CLIP features."
    )
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("16_ML_GCN/runs"))
    parser.add_argument(
        "--clip-feats-path", type=Path,
        default=Path("13_CLIP_Visual_Features/clip_visual_features.npy"),
        help="Pre-extracted CLIP image feature array (N, feat_dim).",
    )
    parser.add_argument(
        "--concepts-path", type=Path,
        default=Path("Concepts81.txt"),
        help="Text file with one label name per line.",
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
    parser.add_argument("--gcn-hidden-dim", type=int, default=defaults.gcn_hidden_dim)
    parser.add_argument("--graph-threshold", type=float, default=defaults.graph_threshold)
    parser.add_argument("--asl-gamma-neg", type=float, default=defaults.asl_gamma_neg)
    parser.add_argument("--asl-gamma-pos", type=float, default=defaults.asl_gamma_pos)
    parser.add_argument("--asl-clip", type=float, default=defaults.asl_clip)
    parser.add_argument(
        "--label-embed-model", type=str, default=defaults.label_embed_model,
        help="CLIP model used to encode label text prompts (ViT-B/32 or ViT-L/14).",
    )
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    args.data_root = resolve_project_path(args.data_root)
    args.output_dir = resolve_project_path(args.output_dir)
    args.clip_feats_path = resolve_project_path(args.clip_feats_path)
    args.concepts_path = resolve_project_path(args.concepts_path)
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
        gcn_hidden_dim=args.gcn_hidden_dim,
        graph_threshold=args.graph_threshold,
        asl_gamma_neg=args.asl_gamma_neg,
        asl_gamma_pos=args.asl_gamma_pos,
        asl_clip=args.asl_clip,
        label_embed_model=args.label_embed_model,
        limit_samples=args.limit_samples,
    )
    set_seed(config.random_seed)
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load image features ---
    print("Loading CLIP image features...")
    clip_feats = load_array(args.clip_feats_path)
    labels = load_array(args.data_root / "database_labels_81_big.npy")

    if clip_feats.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Feature rows ({clip_feats.shape[0]}) != label rows ({labels.shape[0]}). "
            "Re-run the extraction step."
        )

    valid_mask = clip_feats.any(axis=1)
    valid_indices = np.where(valid_mask)[0]
    clip_feats = clip_feats[valid_indices]
    labels = labels[valid_indices]
    feat_dim = clip_feats.shape[1]
    num_classes = labels.shape[1]
    print(f"Valid samples: {len(valid_indices)}  |  feat_dim: {feat_dim}  |  classes: {num_classes}")

    if config.limit_samples is not None:
        keep = min(config.limit_samples, len(labels))
        clip_feats = clip_feats[:keep]
        labels = labels[:keep]

    # --- Label embeddings via CLIP text encoder ---
    label_names = [
        line.strip() for line in args.concepts_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(label_names) != num_classes:
        raise ValueError(f"Concepts file has {len(label_names)} names but labels have {num_classes} classes.")

    label_embeddings = build_label_embeddings(label_names, config.label_embed_model, device)

    # --- Adjacency matrix ---
    graph = load_array(args.graph_path)
    if graph.shape != (num_classes, num_classes):
        raise ValueError(f"Graph shape {graph.shape} does not match num_classes={num_classes}.")
    adj = build_adjacency(graph, threshold=config.graph_threshold)
    print(f"Adjacency matrix: {adj.shape}, non-zero edges: {(adj > 0).sum().item()}")

    # --- Data splits ---
    idx_train, idx_test = train_test_split(
        np.arange(len(labels)),
        test_size=config.test_ratio,
        random_state=config.random_seed,
        shuffle=True,
    )
    train_loader = DataLoader(
        MultiLabelDataset(clip_feats[idx_train], labels[idx_train]),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        MultiLabelDataset(clip_feats[idx_test], labels[idx_test]),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # --- Model ---
    model = MLGCN(
        feat_dim=feat_dim,
        num_classes=num_classes,
        label_embeddings=label_embeddings,
        adj=adj,
        gcn_hidden_dim=config.gcn_hidden_dim,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = AsymmetricLossWithLogits(
        gamma_neg=config.asl_gamma_neg,
        gamma_pos=config.asl_gamma_pos,
        clip=config.asl_clip,
    )

    print("\nRun settings:")
    print(json.dumps(asdict(config), indent=2))
    print(f"Train samples: {len(idx_train)} | Test samples: {len(idx_test)}")
    print(f"GCN: {label_embeddings.shape[1]} → {config.gcn_hidden_dim} → {feat_dim}")
    print(f"Using device: {device}\n")

    best_map = -1.0
    best_test_metrics: dict[str, float] = {}
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, float | int]] = []
    checkpoint_path = args.output_dir / "mlgcn_best.pt"

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
                    "feat_dim": feat_dim,
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
    print(f"\nBest checkpoint: {checkpoint_path}")
    print(f"Reported metrics: {reported_metrics}")


if __name__ == "__main__":
    main()
