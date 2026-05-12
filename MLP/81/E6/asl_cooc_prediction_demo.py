from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split


FEATURE_A_PATH = Path("Extracted_Features/BoW_int.npy")
FEATURE_B_PATHS = [
    Path("Extracted_Features/Normalized_CH.npy"),
    Path("Extracted_Features/Normalized_CM55.npy"),
    Path("Extracted_Features/Normalized_CORR.npy"),
    Path("Extracted_Features/Normalized_EDH.npy"),
    Path("Extracted_Features/Normalized_WT.npy"),
]


@dataclass
class ModelConfig:
    hidden_dims: tuple[int, ...] = (1024, 512, 256)
    dropout: float = 0.3
    activation: str = "gelu"
    embed_dim: int = 128
    cooc_threshold: float = 0.2
    lambda_cooc: float = 0.5
    val_ratio: float = 0.2
    random_seed: int = 42


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_activation(name: str) -> nn.Module:
    activations = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
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
    return nn.Sequential(*layers), nn.Linear(prev_dim, output_dim)


def build_cooccurrence_matrix(label_matrix: np.ndarray, threshold: float = 0.2) -> torch.Tensor:
    cooc = label_matrix.T @ label_matrix
    diag = cooc.diagonal().clip(min=1)
    cooc_prob = cooc / diag[:, None]
    adjacency = (cooc_prob > threshold).astype(np.float32)
    np.fill_diagonal(adjacency, 1.0)
    row_sum = adjacency.sum(axis=1, keepdims=True).clip(min=1.0)
    adjacency = adjacency / row_sum
    return torch.tensor(adjacency, dtype=torch.float32)


class SimpleLabelCorrelation(nn.Module):
    def __init__(self, dim_a: int, dim_b: int, num_classes: int, config: ModelConfig, adjacency: torch.Tensor):
        super().__init__()
        self.proj_A = nn.Sequential(nn.Linear(dim_a, config.embed_dim), make_activation(config.activation))
        self.proj_B = nn.Sequential(nn.Linear(dim_b, config.embed_dim), make_activation(config.activation))
        self.backbone, self.classifier_head = make_mlp(
            input_dim=config.embed_dim * 2,
            hidden_dims=config.hidden_dims,
            output_dim=num_classes,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.register_buffer("A", adjacency)
        self.lambda_cooc = nn.Parameter(torch.tensor(config.lambda_cooc))

    def forward(self, xa: torch.Tensor, xb: torch.Tensor):
        h1 = self.proj_A(xa)
        h2 = self.proj_B(xb)
        feat = self.backbone(torch.cat([h1, h2], dim=1))
        logits_raw = self.classifier_head(feat)
        logits_correlated = torch.matmul(logits_raw, self.A)
        final_logits = logits_raw + self.lambda_cooc * logits_correlated
        return final_logits, h1, h2


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8").splitlines()


def build_image_index(image_paths: list[str]) -> dict[str, int]:
    lookup = {}
    for idx, image_path in enumerate(image_paths):
        normalized = image_path.replace("\\", "/")
        lookup[normalized] = idx
        lookup[Path(normalized).name] = idx
    return lookup


def load_feature_row(path: Path, idx: int) -> np.ndarray:
    array = np.load(path, mmap_mode="r")
    return np.asarray(array[idx], dtype=np.float32)


def load_feature_group_row(paths: list[Path], idx: int) -> np.ndarray:
    return np.concatenate([load_feature_row(path, idx) for path in paths], axis=0).astype(np.float32)


def load_state_dict(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def get_original_index(num_samples: int, source: str, local_idx: int, config: ModelConfig) -> int:
    idx_train, idx_val = train_test_split(
        np.arange(num_samples),
        test_size=config.val_ratio,
        random_state=config.random_seed,
        shuffle=True,
    )
    selected = idx_val if source == "val" else idx_train
    if local_idx < 0 or local_idx >= len(selected):
        raise IndexError(f"{source} index {local_idx} is out of range. Available: 0 to {len(selected) - 1}")
    return int(selected[local_idx])


def resolve_image_path(image_root: Path, image_rel_path: str) -> Path:
    image_path = image_root / image_rel_path
    if image_path.exists():
        return image_path
    raise FileNotFoundError(
        f"Image not found: {image_path}\n"
        "Set --image-root to the folder that contains the images/ directory."
    )


def predict_probs(
    model: nn.Module,
    device: torch.device,
    feature_a_path: Path,
    feature_b_paths: list[Path],
    original_idx: int,
) -> np.ndarray:
    xa_sample = load_feature_row(feature_a_path, original_idx)
    xb_sample = load_feature_group_row(feature_b_paths, original_idx)
    with torch.no_grad():
        logits, _, _ = model(
            torch.tensor(xa_sample, dtype=torch.float32).unsqueeze(0).to(device),
            torch.tensor(xb_sample, dtype=torch.float32).unsqueeze(0).to(device),
        )
        return torch.sigmoid(logits).cpu().numpy()[0]


def plot_prediction(
    image_path: Path,
    probs: np.ndarray,
    true_label_text: str,
    class_names: list[str],
    threshold: float,
    top_k: int,
    output_path: Path,
) -> None:
    top_indices = np.argsort(probs)[::-1][:top_k]
    pred_indices = np.where(probs >= threshold)[0]

    print("\nPredicted labels above threshold:")
    if len(pred_indices) == 0:
        print("  none")
    else:
        for idx in pred_indices:
            print(f"  {class_names[idx]}: {probs[idx]:.3f}")

    print("\nTop predictions:")
    for idx in top_indices:
        print(f"  {class_names[idx]}: {probs[idx]:.3f}")

    print(f"\nGround truth from database_label.txt: {true_label_text}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    image = Image.open(image_path).convert("RGB")
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Input Image")

    names = [class_names[idx] for idx in top_indices][::-1]
    values = probs[top_indices][::-1]
    colors = ["#2f7d6d" if probs[idx] >= threshold else "#9ca3af" for idx in top_indices][::-1]
    axes[1].barh(names, values, color=colors)
    axes[1].axvline(threshold, color="#dc2626", linestyle="--", label=f"threshold={threshold:g}")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_title(f"Top-{top_k} Predicted Labels")
    axes[1].legend()

    fig.suptitle("ASL Early Fusion + Co-occurrence Prediction Demo", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"\nSaved figure to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create image + predicted labels figures for ASL Co-occurrence model.")
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--meta-root", type=Path, default=Path("."))
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, default=Path("81/E6/asl_cooc_final.pt"))
    parser.add_argument("--source", choices=["train", "val"], default="val")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--original-index", type=int, default=None)
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument("--local-images-dir", type=Path, default=None)
    parser.add_argument("--batch-local-images", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("81/E6/cooc_prediction_demo.png"))
    parser.add_argument("--class-names", type=Path, default=Path("Concepts81.txt"))
    args = parser.parse_args()

    no_explicit_sample = (
        args.original_index is None
        and args.image_path is None
        and args.local_images_dir is None
        and not args.batch_local_images
    )
    default_images_dir = Path("images")
    if no_explicit_sample and default_images_dir.exists():
        args.local_images_dir = default_images_dir
        args.batch_local_images = True
        if args.output == Path("81/E6/cooc_prediction_demo.png"):
            args.output = Path("81/E6/demo_predictions")

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {args.checkpoint}\n\n"
            "Train first:\n"
            "python 81/E6/train_asl_cooc.py --data-root dataset --output 81/E6/asl_cooc_final.pt"
        )

    config = ModelConfig()
    feature_a_path = args.data_root / FEATURE_A_PATH
    feature_b_paths = [args.data_root / rel_path for rel_path in FEATURE_B_PATHS]
    label_path = args.data_root / "database_labels_81_big.npy"
    image_list_path = args.meta_root / "database_img.txt"
    label_text_path = args.meta_root / "database_label.txt"

    y_all = np.load(label_path, mmap_mode="r")
    num_samples, num_classes = y_all.shape
    idx_train, _ = train_test_split(
        np.arange(num_samples),
        test_size=config.val_ratio,
        random_state=config.random_seed,
        shuffle=True,
    )
    adjacency = build_cooccurrence_matrix(np.asarray(y_all[idx_train], dtype=np.float32), config.cooc_threshold)

    image_paths = read_lines(image_list_path)
    label_texts = read_lines(label_text_path)
    if len(image_paths) != num_samples or len(label_texts) != num_samples:
        raise ValueError(
            f"Metadata row count mismatch. labels={num_samples}, "
            f"images={len(image_paths)}, label_texts={len(label_texts)}"
        )

    if args.class_names is not None and args.class_names.exists():
        class_names = read_lines(args.class_names)
        if len(class_names) != num_classes:
            raise ValueError(f"--class-names must contain {num_classes} lines, got {len(class_names)}")
    else:
        class_names = [f"Class {idx:02d}" for idx in range(num_classes)]

    dim_a = np.load(feature_a_path, mmap_mode="r").shape[1]
    dim_b = sum(np.load(path, mmap_mode="r").shape[1] for path in feature_b_paths)
    device = get_device()
    print(f"Using device: {device}")

    model = SimpleLabelCorrelation(
        dim_a=dim_a,
        dim_b=dim_b,
        num_classes=num_classes,
        config=config,
        adjacency=adjacency,
    ).to(device)
    model.load_state_dict(load_state_dict(args.checkpoint, device))
    model.eval()

    image_lookup = build_image_index(image_paths)

    if args.batch_local_images:
        if args.local_images_dir is None:
            raise ValueError("--batch-local-images requires --local-images-dir")

        local_images = sorted(
            path for path in args.local_images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if not local_images:
            raise ValueError(f"No image files found in {args.local_images_dir}")

        output_dir = args.output
        output_dir.mkdir(parents=True, exist_ok=True)
        for local_image_path in local_images:
            original_idx = image_lookup.get(local_image_path.name)
            if original_idx is None:
                print(f"Skipping {local_image_path}: filename not found in database_img.txt")
                continue
            probs = predict_probs(model, device, feature_a_path, feature_b_paths, original_idx)
            output_path = output_dir / f"{local_image_path.stem}_cooc_prediction.png"
            print("\n" + "=" * 80)
            print(f"Original database index: {original_idx}")
            print(f"Image: {local_image_path}")
            plot_prediction(
                image_path=local_image_path,
                probs=probs,
                true_label_text=label_texts[original_idx],
                class_names=class_names,
                threshold=args.threshold,
                top_k=args.top_k,
                output_path=output_path,
            )
        return

    if args.image_path is not None:
        original_idx = image_lookup.get(args.image_path.as_posix(), image_lookup.get(args.image_path.name))
        if original_idx is None:
            raise ValueError(f"{args.image_path} was not found in database_img.txt")
        image_path = args.image_path
    elif args.original_index is not None:
        original_idx = args.original_index
        image_path = resolve_image_path(args.image_root, image_paths[original_idx])
    else:
        original_idx = get_original_index(num_samples, args.source, args.idx, config)
        image_path = resolve_image_path(args.image_root, image_paths[original_idx])

    probs = predict_probs(model, device, feature_a_path, feature_b_paths, original_idx)
    print(f"Original database index: {original_idx}")
    print(f"Image: {image_path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_prediction(
        image_path=image_path,
        probs=probs,
        true_label_text=label_texts[original_idx],
        class_names=class_names,
        threshold=args.threshold,
        top_k=args.top_k,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
