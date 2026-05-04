from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


FEATURE_A_PATH = Path("Extracted_Features/BoW_int.npy")
FEATURE_B_PATHS = [
    Path("Extracted_Features/Normalized_CH.npy"),
    Path("Extracted_Features/Normalized_CM55.npy"),
    Path("Extracted_Features/Normalized_CORR.npy"),
    Path("Extracted_Features/Normalized_EDH.npy"),
    Path("Extracted_Features/Normalized_WT.npy"),
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
    threshold: float = 0.5
    val_ratio: float = 0.2
    random_seed: int = 42
    embed_dim: int = 128
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 1.0
    asl_clip: float = 0.05


class MultiModalDataset(Dataset):
    def __init__(self, xa: np.ndarray, xb: np.ndarray, y: np.ndarray):
        self.xa = torch.tensor(xa, dtype=torch.float32)
        self.xb = torch.tensor(xb, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.xa[idx], self.xb[idx], self.y[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")
    return np.load(path).astype(np.float32)


def load_feature_group(data_root: Path, paths: list[Path]) -> np.ndarray:
    arrays = [load_array(data_root / path) for path in paths]
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


def make_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, dropout: float, activation: str):
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(make_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class EarlyFusion(nn.Module):
    def __init__(self, dim_a: int, dim_b: int, num_classes: int, config: TrainConfig):
        super().__init__()
        self.proj_A = nn.Sequential(nn.Linear(dim_a, config.embed_dim), make_activation(config.activation))
        self.proj_B = nn.Sequential(nn.Linear(dim_b, config.embed_dim), make_activation(config.activation))
        self.classifier = make_mlp(
            input_dim=config.embed_dim * 2,
            hidden_dims=config.hidden_dims,
            output_dim=num_classes,
            dropout=config.dropout,
            activation=config.activation,
        )

    def forward(self, xa: torch.Tensor, xb: torch.Tensor):
        h1 = self.proj_A(xa)
        h2 = self.proj_B(xb)
        fused = torch.cat([h1, h2], dim=1)
        return self.classifier(fused), h1, h2


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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float):
    model.eval()
    all_targets, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for xa, xb, y in loader:
            xa = xa.to(device)
            xb = xb.to(device)
            logits, _, _ = model(xa, xb)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_preds.append((probs > threshold).float().cpu().numpy())
            all_targets.append(y.numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    mi_f1 = f1_score(all_targets, all_preds, average="micro", zero_division=0)
    ma_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    try:
        map_score = average_precision_score(all_targets, all_probs, average="macro")
    except ValueError:
        map_score = 0.0
    return map_score, mi_f1, ma_f1


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ASL Early Fusion and save a checkpoint.")
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("81/E5/asl_early_fusion_concat.pt"))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size)
    set_seed(config.random_seed)
    device = get_device()
    print(f"Using device: {device}")

    xa_all = load_array(args.data_root / FEATURE_A_PATH)
    xb_all = load_feature_group(args.data_root, FEATURE_B_PATHS)
    y_all = load_array(args.data_root / "database_labels_81_big.npy")

    idx_train, idx_val = train_test_split(
        np.arange(len(y_all)),
        test_size=config.val_ratio,
        random_state=config.random_seed,
        shuffle=True,
    )

    train_loader = DataLoader(
        MultiModalDataset(xa_all[idx_train], xb_all[idx_train], y_all[idx_train]),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        MultiModalDataset(xa_all[idx_val], xb_all[idx_val], y_all[idx_val]),
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = EarlyFusion(
        dim_a=xa_all.shape[1],
        dim_b=xb_all.shape[1],
        num_classes=y_all.shape[1],
        config=config,
    ).to(device)

    criterion = AsymmetricLossWithLogits(
        gamma_neg=config.asl_gamma_neg,
        gamma_pos=config.asl_gamma_pos,
        clip=config.asl_clip,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_map = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        for xa, xb, y in train_loader:
            xa = xa.to(device)
            xb = xb.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits, _, _ = model(xa, xb)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_map, val_mi, val_ma = evaluate(model, val_loader, device, config.threshold)
        print(
            f"Epoch [{epoch + 1:02d}/{config.epochs}] "
            f"Loss: {epoch_loss / max(len(train_loader), 1):.4f} | "
            f"Val mAP: {val_map:.4f} | Mi-F1: {val_mi:.4f} | Ma-F1: {val_ma:.4f}"
        )

        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), args.output)
            print(f"Saved best checkpoint: {args.output}")

    print(f"Training complete. Best validation mAP: {best_map:.4f}")
    print(f"Checkpoint: {args.output}")


if __name__ == "__main__":
    main()
