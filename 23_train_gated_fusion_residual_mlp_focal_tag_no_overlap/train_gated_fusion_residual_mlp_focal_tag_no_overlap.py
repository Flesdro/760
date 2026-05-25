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


TRAIN_VISUAL_FEATURE_PATHS = [
    Path("Extracted_Features/BoW_int.npy"),
    Path("Extracted_Features/Normalized_CH.npy"),
    Path("Extracted_Features/Normalized_CM55.npy"),
    Path("Extracted_Features/Normalized_CORR.npy"),
    Path("Extracted_Features/Normalized_EDH.npy"),
    Path("Extracted_Features/Normalized_WT.npy"),
]
GROUP_NAMES = ["bow", "ch", "cm55", "corr", "edh", "wt", "semantic_tag_no_overlap"]


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    group_embed_dim: int = 128
    hidden_dim: int = 512
    num_blocks: int = 4
    dropout: float = 0.3
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    threshold: float = 0.5
    random_seed: int = 42
    focal_gamma: float = 2.0
    focal_class_weight: bool = True
    cooc_threshold: float = 0.2
    cooc_alpha: float = 0.2
    max_train_samples: int | None = None


class FeatureGroupDataset(Dataset):
    def __init__(self, feature_groups: list[np.ndarray], labels: np.ndarray):
        self.feature_groups = [
            torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
            for features in feature_groups
        ]
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


class GatedFusionResidualMLP(nn.Module):
    def __init__(
        self,
        group_dims: list[int],
        num_classes: int,
        group_embed_dim: int = 128,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        dropout: float = 0.3,
        refiner: nn.Module | None = None,
        use_refiner: bool = True,
    ):
        super().__init__()
        self.refiner = refiner
        self.use_refiner = use_refiner
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
        if self.use_refiner and self.refiner is not None:
            logits = self.refiner(logits)
        return logits, gate_weights


class MultiLabelFocalLossWithLogits(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        positive_rates: torch.Tensor | None = None,
        use_class_weight: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.use_class_weight = use_class_weight
        self.reduction = reduction
        if positive_rates is None:
            positive_rates = torch.empty(0, dtype=torch.float32)
        self.register_buffer("positive_rates", positive_rates.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        focal_weight = (1.0 - pt).clamp(min=0.0, max=1.0).pow(self.gamma)
        loss = focal_weight * bce

        if self.use_class_weight and self.positive_rates.numel() > 0:
            rates = self.positive_rates.to(logits.device).view(1, -1).clamp(0.0, 1.0)
            class_weight = torch.exp(targets * (1.0 - rates) + (1.0 - targets) * rates)
            loss = class_weight * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction: {self.reduction}")


def build_cooccurrence_matrix(label_matrix: np.ndarray, threshold: float = 0.2) -> torch.Tensor:
    """
    Build a row-normalized label co-occurrence adjacency from binary multi-label targets.
    """
    cooc = label_matrix.T @ label_matrix
    diag = cooc.diagonal().clip(min=1)
    cooc_prob = cooc / diag[:, None]

    adjacency = (cooc_prob > threshold).astype(np.float32)
    np.fill_diagonal(adjacency, 1.0)
    adjacency = np.maximum(adjacency, adjacency.T)

    row_sum = adjacency.sum(axis=1, keepdims=True).clip(min=1.0)
    adjacency = adjacency / row_sum
    return torch.tensor(adjacency, dtype=torch.float32)


class LabelCorrelationRefiner(nn.Module):
    def __init__(self, correlation_matrix: torch.Tensor | np.ndarray, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha
        if isinstance(correlation_matrix, torch.Tensor):
            matrix = correlation_matrix.detach().float()
        else:
            matrix = torch.tensor(correlation_matrix, dtype=torch.float32)
        self.register_buffer("M", matrix)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + self.alpha * torch.matmul(logits, self.M)


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
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    return np.loadtxt(path).astype(np.float32)


def load_text_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")
    return path.read_text(encoding="utf-8").splitlines()


def normalize_name(path: str) -> str:
    path = path.replace("\\", "/")
    return Path(path).name.lower()


def read_image_names(path: Path) -> list[str]:
    return [normalize_name(line.strip()) for line in load_text_lines(path) if line.strip()]


def load_label_names(path: Path) -> list[str]:
    return [line.strip().lower() for line in load_text_lines(path) if line.strip()]


def resolve_input_path(data_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    data_path = data_root / path
    if data_path.exists():
        return data_path
    return path


def build_image_alignment(
    local_names: list[str],
    official_names: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    official_index: dict[str, int] = {}
    for idx, name in enumerate(official_names):
        official_index.setdefault(name, idx)

    matched_local_indices: list[int] = []
    matched_official_indices: list[int] = []
    missing_local_names: list[str] = []
    for local_idx, name in enumerate(local_names):
        official_idx = official_index.get(name)
        if official_idx is None:
            missing_local_names.append(name)
            continue
        matched_local_indices.append(local_idx)
        matched_official_indices.append(official_idx)

    metadata = {
        "local_total": len(local_names),
        "official_total": len(official_names),
        "matched": len(matched_local_indices),
        "missing": len(missing_local_names),
        "match_ratio": float(len(matched_local_indices) / max(len(local_names), 1)),
        "missing_examples": missing_local_names[:10],
    }
    return (
        np.asarray(matched_local_indices, dtype=np.int64),
        np.asarray(matched_official_indices, dtype=np.int64),
        metadata,
    )


def load_visual_groups(data_root: Path, visual_paths: list[Path]) -> list[np.ndarray]:
    arrays = [load_array(data_root / path) for path in visual_paths]
    n_rows = {array.shape[0] for array in arrays}
    if len(n_rows) != 1:
        min_rows = min(n_rows)
        print(
            "Warning: visual feature row counts do not match "
            f"{sorted(n_rows)}; trimming all groups to {min_rows} rows by prefix."
        )
        arrays = [array[:min_rows] for array in arrays]
    return arrays


def build_overlap_indices(tag_names: list[str], label_names: list[str]) -> tuple[list[str], list[int]]:
    overlap_labels: list[str] = []
    overlap_indices: list[int] = []
    label_set = set(label_names)
    for idx, tag in enumerate(tag_names):
        if tag in label_set:
            overlap_labels.append(tag)
            overlap_indices.append(idx)
    return overlap_labels, overlap_indices


def clean_tag_matrix(
    tag_matrix: np.ndarray,
    overlap_indices: list[int],
) -> np.ndarray:
    if not overlap_indices:
        return tag_matrix.astype(np.float32, copy=False)
    return np.delete(tag_matrix, overlap_indices, axis=1).astype(np.float32, copy=False)


def load_clean_tag_matrix(
    data_root: Path,
    tag_path: Path,
    tag_names: list[str],
    label_names: list[str],
    cache_path: Path | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        if isinstance(cached, np.ndarray) and cached.dtype == object and cached.shape == ():
            payload = cached.item()
            if isinstance(payload, dict) and "matrix" in payload and "metadata" in payload:
                return payload["matrix"].astype(np.float32), payload["metadata"]

    tag_matrix = load_array(data_root / tag_path)
    overlap_labels, overlap_indices = build_overlap_indices(tag_names, label_names)
    clean_matrix = clean_tag_matrix(tag_matrix, overlap_indices)
    metadata = {
        "tag_path": str(tag_path),
        "original_shape": list(tag_matrix.shape),
        "clean_shape": list(clean_matrix.shape),
        "num_overlap": len(overlap_indices),
        "overlap_labels": overlap_labels,
        "overlap_indices": overlap_indices,
    }

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            cache_path,
            {
                "matrix": clean_matrix.astype(np.float32),
                "metadata": metadata,
            },
            allow_pickle=True,
        )
    return clean_matrix.astype(np.float32), metadata


def load_feature_groups(
    data_root: Path,
    visual_paths: list[Path],
    tag_path: Path,
    tag_names: list[str],
    label_names: list[str],
    cache_dir: Path | None = None,
) -> tuple[list[np.ndarray], list[str], dict[str, object]]:
    arrays = [load_array(data_root / path) for path in visual_paths]
    cache_path = None if cache_dir is None else cache_dir / f"{tag_path.stem}_clean_no_overlap.npy"
    tag_features, metadata = load_clean_tag_matrix(
        data_root=data_root,
        tag_path=tag_path,
        tag_names=tag_names,
        label_names=label_names,
        cache_path=cache_path,
    )
    arrays.append(tag_features)

    n_rows = {array.shape[0] for array in arrays}
    if len(n_rows) != 1:
        min_rows = min(n_rows)
        print(
            "Warning: feature row counts do not match "
            f"{sorted(n_rows)}; trimming all groups to {min_rows} rows by prefix."
        )
        arrays = [array[:min_rows] for array in arrays]
    return arrays, GROUP_NAMES, metadata


def select_rows(feature_groups: list[np.ndarray], indices: np.ndarray) -> list[np.ndarray]:
    return [features[indices] for features in feature_groups]


def build_label_positive_rates(label_matrix: np.ndarray) -> torch.Tensor:
    rates = label_matrix.mean(axis=0).astype(np.float32)
    return torch.from_numpy(rates)


def trim_to_match(features: np.ndarray, labels: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    if features.shape[0] == labels.shape[0]:
        return features, labels
    n_rows = min(features.shape[0], labels.shape[0])
    print(f"Warning: trimming {name} rows to {n_rows} ({features.shape[0]} features, {labels.shape[0]} labels).")
    return features[:n_rows], labels[:n_rows]


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


def load_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def print_run_settings(args: argparse.Namespace, config: TrainConfig) -> None:
    print("\nRun settings:")
    print(json.dumps(asdict(config), indent=2))
    print(f"data_root: {args.data_root}")
    print(f"output_dir: {args.output_dir}")
    print(f"num_workers: {args.num_workers}")
    print(f"train_tag_path: {args.train_tag_path}")
    print(f"local_image_list: {args.local_image_list}")
    print(f"official_image_list: {args.official_image_list}")
    print()


def main() -> None:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Train Gated Fusion Residual MLP + multi-label focal loss + "
            "cleaned no-overlap tag modality."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("23_train_gated_fusion_residual_mlp_focal_tag_no_overlap/runs"),
    )
    parser.add_argument("--local-image-list", type=Path, default=Path("database_img.txt"))
    parser.add_argument("--official-image-list", type=Path, default=Path("TrainImagelist.txt"))
    parser.add_argument("--train-tag-path", type=Path, default=Path("NUS_WID_Tags/Train_Tags1k.dat"))
    parser.add_argument("--tag-list-path", type=Path, default=Path("NUS_WID_Tags/TagList1k.txt"))
    parser.add_argument(
        "--label-names-path",
        type=Path,
        default=Path("Concepts81.txt"),
    )
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
    parser.add_argument("--test-ratio", type=float, default=defaults.test_ratio)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--focal-gamma", type=float, default=defaults.focal_gamma)
    parser.add_argument(
        "--no-focal-class-weight",
        action="store_false",
        dest="focal_class_weight",
        help="Disable the paper-style positive-rate class weighting term.",
    )
    parser.set_defaults(focal_class_weight=defaults.focal_class_weight)
    parser.add_argument("--cooc-threshold", type=float, default=defaults.cooc_threshold)
    parser.add_argument("--cooc-alpha", type=float, default=defaults.cooc_alpha)
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap for quick smoke tests.")
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
        test_ratio=args.test_ratio,
        threshold=args.threshold,
        random_seed=args.seed,
        focal_gamma=args.focal_gamma,
        focal_class_weight=args.focal_class_weight,
        cooc_threshold=args.cooc_threshold,
        cooc_alpha=args.cooc_alpha,
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
    if config.max_train_samples is not None and config.max_train_samples < 3:
        raise ValueError("--max-train-samples must be at least 3 when provided.")

    set_seed(config.random_seed)
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print_run_settings(args, config)
    print(f"Using device: {device}")
    print("Loading notebook-style aligned views...")
    tag_names = load_label_names(resolve_input_path(args.data_root, args.tag_list_path))
    label_names = load_label_names(resolve_input_path(args.data_root, args.label_names_path))
    local_names = read_image_names(resolve_input_path(args.data_root, args.local_image_list))
    official_names = read_image_names(resolve_input_path(args.data_root, args.official_image_list))
    official_index = {name: idx for idx, name in enumerate(official_names)}

    official_tags = load_array(resolve_input_path(args.data_root, args.train_tag_path))
    overlap_labels, overlap_indices = build_overlap_indices(tag_names, label_names)
    clean_tags = clean_tag_matrix(official_tags, overlap_indices)
    tag_metadata = {
        "tag_path": str(args.train_tag_path),
        "original_shape": list(official_tags.shape),
        "clean_shape": list(clean_tags.shape),
        "num_overlap": len(overlap_indices),
        "overlap_labels": overlap_labels,
        "overlap_indices": overlap_indices,
    }

    aligned_tags = np.zeros((len(local_names), clean_tags.shape[1]), dtype=np.float32)
    matched_indices: list[int] = []
    missing = 0
    for local_idx, name in enumerate(local_names):
        official_idx = official_index.get(name)
        if official_idx is None:
            missing += 1
            continue
        aligned_tags[local_idx] = clean_tags[official_idx]
        matched_indices.append(local_idx)

    matched_indices_array = np.asarray(matched_indices, dtype=np.int64)
    alignment_metadata = {
        "local_images": len(local_names),
        "official_train_images": len(official_names),
        "matched": int(len(matched_indices_array)),
        "missing": int(missing),
        "match_ratio": float(len(matched_indices_array) / max(len(local_names), 1)),
    }

    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "Train_Tags_NoOverlap.npy", clean_tags)
    np.save(cache_dir / "aligned_tag_feature_no_overlap.npy", aligned_tags)
    np.save(cache_dir / "matched_indices_no_overlap.npy", matched_indices_array)
    save_json(cache_dir / "alignment_metadata.json", alignment_metadata)
    save_json(cache_dir / "tag_metadata.json", tag_metadata)

    print("Alignment metadata:")
    print(json.dumps(alignment_metadata, indent=2))

    feature_groups = load_visual_groups(args.data_root, TRAIN_VISUAL_FEATURE_PATHS)
    labels = load_array(args.data_root / "database_labels_81_big.npy")

    feature_groups = [features[matched_indices_array] for features in feature_groups]
    labels = labels[matched_indices_array]
    tags = aligned_tags[matched_indices_array]

    if config.max_train_samples is not None:
        keep = min(config.max_train_samples, len(labels))
        feature_groups = [features[:keep] for features in feature_groups]
        labels = labels[:keep]
        tags = tags[:keep]
        matched_indices_array = matched_indices_array[:keep]

    feature_groups.append(tags)
    view_dims = [view.shape[1] for view in feature_groups]

    idx_trainval, idx_test = train_test_split(
        np.arange(len(labels)),
        test_size=config.test_ratio,
        random_state=config.random_seed,
        shuffle=True,
    )
    effective_val_ratio = config.val_ratio / (1.0 - config.test_ratio)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=effective_val_ratio,
        random_state=config.random_seed,
        shuffle=True,
    )
    cooc_matrix = build_cooccurrence_matrix(np.asarray(labels[idx_train], dtype=np.float32), config.cooc_threshold)
    cooc_refiner = LabelCorrelationRefiner(cooc_matrix, alpha=config.cooc_alpha)
    positive_rates = build_label_positive_rates(labels[idx_train])
    focal_stats = {
        "gamma": config.focal_gamma,
        "class_weight": config.focal_class_weight,
        "positive_rate_min": float(positive_rates.min().item()),
        "positive_rate_max": float(positive_rates.max().item()),
        "positive_rate_mean": float(positive_rates.mean().item()),
    }

    train_loader = DataLoader(
        FeatureGroupDataset(select_rows(feature_groups, idx_train), labels[idx_train]),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        FeatureGroupDataset(select_rows(feature_groups, idx_val), labels[idx_val]),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        FeatureGroupDataset(select_rows(feature_groups, idx_test), labels[idx_test]),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = GatedFusionResidualMLP(
        group_dims=view_dims,
        num_classes=labels.shape[1],
        group_embed_dim=config.group_embed_dim,
        hidden_dim=config.hidden_dim,
        num_blocks=config.num_blocks,
        dropout=config.dropout,
        refiner=cooc_refiner,
        use_refiner=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = MultiLabelFocalLossWithLogits(
        gamma=config.focal_gamma,
        positive_rates=positive_rates,
        use_class_weight=config.focal_class_weight,
    )

    best_map = -1.0
    history: list[dict[str, float]] = []
    checkpoint_path = args.output_dir / "gated_fusion_residual_mlp_focal_tag_no_overlap_best.pt"

    print(f"Matched samples: {len(matched_indices_array)}")
    print(f"Train samples: {len(idx_train)} | Val samples: {len(idx_val)} | Test samples: {len(idx_test)}")
    print(f"Group names: {GROUP_NAMES}")
    print(f"Group dims: {view_dims} | Classes: {labels.shape[1]}")
    print(
        f"Group embed dim: {config.group_embed_dim} | "
        f"Hidden dim: {config.hidden_dim} | Residual blocks: {config.num_blocks}"
    )
    print(
        "Loss: MultiLabelFocalLossWithLogits "
        f"(gamma={config.focal_gamma}, class_weight={config.focal_class_weight})"
    )
    print(f"Co-occurrence threshold: {config.cooc_threshold} | alpha: {config.cooc_alpha}")
    print(
        "Tag metadata clean shape: "
        f"{tag_metadata['clean_shape']} (overlap removed: {tag_metadata['num_overlap']})"
    )
    print(
        f"Co-occurrence matrix shape: {list(cooc_matrix.shape)} | "
        f"non-zero ratio: {float((cooc_matrix > 0).float().mean().item()):.6f}"
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
                    "group_dims": view_dims,
                    "group_names": GROUP_NAMES,
                    "num_classes": labels.shape[1],
                    "best_val_metrics": val_metrics,
                    "tag_metadata": tag_metadata,
                    "alignment_metadata": alignment_metadata,
                    "focal_loss": focal_stats,
                    "cooc_metadata": {
                        "threshold": config.cooc_threshold,
                        "alpha": config.cooc_alpha,
                        "shape": list(cooc_matrix.shape),
                        "non_zero_ratio": float((cooc_matrix > 0).float().mean().item()),
                    },
                },
                checkpoint_path,
            )

        print(
            f"Epoch {epoch:03d}/{config.epochs} "
            f"loss={train_loss:.4f} "
            f"val_mAP={val_metrics['mAP']:.4f} "
            f"val_micro_f1={val_metrics['micro_f1']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"gate_tag={val_metrics.get('gate_6_mean', 0.0):.4f}"
        )

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_val_metrics = evaluate(model, val_loader, device, config.threshold)
    test_metrics = evaluate(model, test_loader, device, config.threshold)

    save_json(args.output_dir / "training_history.json", history)
    save_json(args.output_dir / "validation_metrics.json", final_val_metrics)
    save_json(args.output_dir / "test_metrics.json", test_metrics)
    save_json(
        args.output_dir / "config.json",
        {
            **asdict(config),
            "local_image_list": str(args.local_image_list),
            "official_image_list": str(args.official_image_list),
            "train_tag_path": str(args.train_tag_path),
            "tag_list_path": str(args.tag_list_path),
            "label_names_path": str(args.label_names_path),
            "group_names": GROUP_NAMES,
            "group_dims": view_dims,
            "tag_metadata": tag_metadata,
            "alignment_metadata": alignment_metadata,
            "focal_loss": focal_stats,
            "loss": "MultiLabelFocalLossWithLogits",
            "model": "GatedFusionResidualMLP",
            "reported_metrics": "test_metrics.json",
            "cooc_metadata": {
                "threshold": config.cooc_threshold,
                "alpha": config.cooc_alpha,
                "shape": list(cooc_matrix.shape),
                "non_zero_ratio": float((cooc_matrix > 0).float().mean().item()),
            },
        },
    )
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Validation metrics: {final_val_metrics}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
