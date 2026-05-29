from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont


TRAIN_VISUAL_FEATURE_PATHS = [
    Path("Extracted_Features/BoW_int.npy"),
    Path("Extracted_Features/Normalized_CH.npy"),
    Path("Extracted_Features/Normalized_CM55.npy"),
    Path("Extracted_Features/Normalized_CORR.npy"),
    Path("Extracted_Features/Normalized_EDH.npy"),
    Path("Extracted_Features/Normalized_WT.npy"),
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FONT_REGULAR_CANDIDATES = [
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
]
FONT_BOLD_CANDIDATES = [
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
]


@dataclass
class ModelConfig:
    hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    dropout: float = 0.3
    activation: str = "gelu"
    threshold: float = 0.5


class PureMLP(nn.Module):
    def __init__(
        self,
        group_dims: list[int],
        num_classes: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
        activation: str,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = sum(group_dims)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, feature_groups: list[torch.Tensor]) -> torch.Tensor:
        return self.mlp(torch.cat(feature_groups, dim=1))


def make_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]()


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")
    return path.read_text(encoding="utf-8").splitlines()


def normalize_name(path: str) -> str:
    path = path.replace("\\", "/")
    return Path(path).name.lower()


def build_image_index(image_paths: list[str]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for idx, image_path in enumerate(image_paths):
        normalized = image_path.replace("\\", "/")
        lookup.setdefault(normalized.lower(), idx)
        lookup.setdefault(Path(normalized).name.lower(), idx)
    return lookup


def load_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def model_config_from_checkpoint(checkpoint: dict[str, object]) -> ModelConfig:
    raw_config = checkpoint.get("config", {})
    if not isinstance(raw_config, dict):
        raw_config = {}
    hidden_dims = raw_config.get("hidden_dims", (512, 512, 512, 512))
    return ModelConfig(
        hidden_dims=tuple(int(dim) for dim in hidden_dims),
        dropout=float(raw_config.get("dropout", 0.3)),
        activation=str(raw_config.get("activation", "gelu")),
        threshold=float(raw_config.get("threshold", 0.5)),
    )


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[PureMLP, dict[str, object]]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unexpected checkpoint format: {checkpoint_path}")

    group_dims = checkpoint.get("group_dims")
    num_classes = checkpoint.get("num_classes")
    if not isinstance(group_dims, list) or not isinstance(num_classes, int):
        raise ValueError("Checkpoint must contain group_dims and num_classes.")

    config = model_config_from_checkpoint(checkpoint)
    model = PureMLP(
        group_dims=[int(dim) for dim in group_dims],
        num_classes=num_classes,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        activation=config.activation,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_feature_row(path: Path, idx: int) -> np.ndarray:
    array = np.load(path, mmap_mode="r")
    return np.asarray(array[idx], dtype=np.float32)


def load_feature_groups(data_root: Path, local_idx: int) -> list[np.ndarray]:
    return [load_feature_row(data_root / path, local_idx) for path in TRAIN_VISUAL_FEATURE_PATHS]


def predict(
    model: PureMLP,
    device: torch.device,
    feature_groups: list[np.ndarray],
) -> np.ndarray:
    tensors = [
        torch.tensor(group, dtype=torch.float32).unsqueeze(0).to(device)
        for group in feature_groups
    ]
    with torch.no_grad():
        logits = model(tensors)
        return torch.sigmoid(logits).cpu().numpy()[0]


def label_names_from_onehot(labels: np.ndarray, class_names: list[str]) -> list[str]:
    indices = np.flatnonzero(labels > 0.5)
    return [class_names[int(idx)] for idx in indices]


def resize_to_fit(image: Image.Image, box_size: tuple[int, int]) -> Image.Image:
    max_w, max_h = box_size
    image = image.convert("RGB")
    scale = min(max_w / image.width, max_h / image.height)
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    try:
        resampling = Image.Resampling.LANCZOS
    except AttributeError:
        resampling = Image.LANCZOS
    return image.resize(new_size, resampling)


def text_width(text: str, font: ImageFont.ImageFont) -> int:
    left, _, right, _ = font.getbbox(text)
    return right - left


def wrap_text_to_width(text: str, max_width: int, font: ImageFont.ImageFont) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = current + " " + word
        if text_width(candidate, font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def load_ui_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = FONT_BOLD_CANDIDATES if bold else FONT_REGULAR_CANDIDATES
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_probability_figure(
    image_path: Path,
    output_path: Path,
    probs: np.ndarray,
    class_names: list[str],
    true_labels: list[str],
    threshold: float,
    top_indices: np.ndarray,
    font_scale: float = 1.0,
) -> None:
    font_scale = max(0.8, font_scale)
    margin = 40
    image_box = (620, 560)
    chart_x = 740
    chart_y = 124
    label_w = max(170, int(170 * font_scale))
    bar_h = max(28, int(24 * font_scale))
    gap = max(14, int(12 * font_scale))
    row_step = bar_h + gap
    line_h = max(18, int(17 * font_scale))
    canvas_w = max(1400, int(1400 + (font_scale - 1.0) * 260))
    canvas_h = max(760, chart_y + len(top_indices) * row_step + 36)
    bar_w = canvas_w - chart_x - label_w - 118

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_ui_font(max(15, int(16 * font_scale)), bold=True)
    font = load_ui_font(max(13, int(14 * font_scale)))
    small_font = load_ui_font(max(12, int(13 * font_scale)))
    gt_font = load_ui_font(max(14, int(16 * font_scale)))
    gt_line_h = max(22, int(20 * font_scale))

    draw.text((margin, 24), "Pure MLP + BCE Baseline", fill=(17, 24, 39), font=title_font)
    draw.text((margin, 58), image_path.name, fill=(75, 85, 99), font=font)

    image = resize_to_fit(Image.open(image_path), image_box)
    image_x = margin + (image_box[0] - image.width) // 2
    image_y = 100 + (image_box[1] - image.height) // 2
    canvas.paste(image, (image_x, image_y))
    draw.rectangle((margin, 100, margin + image_box[0], 100 + image_box[1]), outline=(229, 231, 235), width=2)

    gt_text = "Ground truth: " + (", ".join(true_labels) if true_labels else "none")
    for line_no, line in enumerate(wrap_text_to_width(gt_text, image_box[0], gt_font)[:3]):
        draw.text((margin, 680 + line_no * gt_line_h), line, fill=(31, 41, 55), font=gt_font)

    draw.text((chart_x, 60), f"Top-{len(top_indices)} predictions", fill=(17, 24, 39), font=title_font)
    draw.text((chart_x, 90), f"Threshold: {threshold:g}", fill=(220, 38, 38), font=font)

    for rank, idx in enumerate(top_indices):
        y = chart_y + rank * row_step
        label = class_names[int(idx)]
        value = float(probs[int(idx)])
        draw.text((chart_x, y + 6), label[:24], fill=(31, 41, 55), font=font)

        x0 = chart_x + label_w
        y0 = y
        x1 = x0 + bar_w
        y1 = y + bar_h
        draw.rectangle((x0, y0, x1, y1), fill=(243, 244, 246), outline=(229, 231, 235))
        fill_w = int(bar_w * max(0.0, min(1.0, value)))
        color = (47, 125, 109) if value >= threshold else (156, 163, 175)
        draw.rectangle((x0, y0, x0 + fill_w, y1), fill=color)
        draw.text((x1 + 12, y + 6), f"{value:.3f}", fill=(31, 41, 55), font=font)

    threshold_x = chart_x + label_w + int(bar_w * max(0.0, min(1.0, threshold)))
    draw.line(
        (threshold_x, chart_y - 8, threshold_x, chart_y + len(top_indices) * row_step - gap + 8),
        fill=(220, 38, 38),
        width=2,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def plot_prediction(
    image_path: Path,
    output_path: Path,
    probs: np.ndarray,
    class_names: list[str],
    true_labels: list[str],
    threshold: float,
    top_k: int,
    font_scale: float,
) -> dict[str, object]:
    top_indices = np.argsort(probs)[::-1][:top_k]
    pred_indices = np.where(probs >= threshold)[0]

    print("\nPredicted labels above threshold:")
    if len(pred_indices) == 0:
        print("  none")
    else:
        for idx in pred_indices:
            print(f"  {class_names[int(idx)]}: {probs[int(idx)]:.3f}")

    print("\nTop predictions:")
    for idx in top_indices:
        print(f"  {class_names[int(idx)]}: {probs[int(idx)]:.3f}")

    print("\nGround truth labels:")
    print("  " + (", ".join(true_labels) if true_labels else "none"))

    draw_probability_figure(
        image_path=image_path,
        output_path=output_path,
        probs=probs,
        class_names=class_names,
        true_labels=true_labels,
        threshold=threshold,
        top_indices=top_indices,
        font_scale=font_scale,
    )
    print(f"\nSaved figure to: {output_path}")

    return {
        "image": str(image_path),
        "output": str(output_path),
        "top_predictions": [
            {"label": class_names[int(idx)], "probability": float(probs[int(idx)])}
            for idx in top_indices
        ],
        "threshold_predictions": [
            {"label": class_names[int(idx)], "probability": float(probs[int(idx)])}
            for idx in pred_indices
        ],
        "ground_truth": true_labels,
        "font_scale": float(font_scale),
    }


def iter_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {images_dir.resolve()}")
    return sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def resolve_selected_images(images_dir: Path, selected_images_path: Path) -> list[Path]:
    selected: list[Path] = []
    for raw_line in read_lines(selected_images_path):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        candidate = Path(line)
        if candidate.is_absolute() and candidate.exists():
            selected.append(candidate)
            continue

        image_path = images_dir / candidate.name
        if not image_path.exists():
            print(f"Skipping selected image that was not found: {image_path}")
            continue
        selected.append(image_path)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run image prediction figures for experiment 07 baseline.")
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--meta-root", type=Path, default=Path("."))
    parser.add_argument("--images-dir", type=Path, default=Path("NUS-WIDE-images/images"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("07_Pure_MLP_Concat_BCE/pure_mlp_bce_clean_runs/pure_mlp_bce_visual_clean_best.pt"),
    )
    parser.add_argument("--class-names", type=Path, default=Path("Concepts81.txt"))
    parser.add_argument("--label-onehot", type=Path, default=Path("dataset/database_labels_81_big.npy"))
    parser.add_argument("--image-list", type=Path, default=Path("database_img.txt"))
    parser.add_argument(
        "--selected-images",
        type=Path,
        default=Path("07_Pure_MLP_Concat_BCE/pure_mlp_bce_clean_runs/demo_selected_images.txt"),
        help="Optional text file of image filenames/paths to process from --images-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("07_Pure_MLP_Concat_BCE/pure_mlp_bce_clean_runs/demo_predictions_nuswide_rich_large_font"),
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--font-scale", type=float, default=1.45)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    model, checkpoint = load_model(args.checkpoint, device)
    config = model_config_from_checkpoint(checkpoint)
    threshold = config.threshold if args.threshold is None else args.threshold

    image_paths = read_lines(args.meta_root / args.image_list)
    image_lookup = build_image_index(image_paths)
    class_names = [line.strip() for line in read_lines(args.class_names) if line.strip()]
    if len(class_names) != int(checkpoint["num_classes"]):
        raise ValueError(f"--class-names must contain {checkpoint['num_classes']} lines, got {len(class_names)}")

    labels = np.load(args.label_onehot, mmap_mode="r")
    images = resolve_selected_images(args.images_dir, args.selected_images) if args.selected_images else iter_images(args.images_dir)
    if not images:
        raise ValueError(f"No image files found in {args.images_dir}")

    summaries: list[dict[str, object]] = []
    for image_path in images:
        print("\n" + "=" * 80)
        print(f"Image: {image_path}")
        local_idx = image_lookup.get(normalize_name(image_path.name))
        if local_idx is None:
            print("Skipping: filename was not found in database_img.txt")
            continue

        feature_groups = load_feature_groups(args.data_root, local_idx)
        probs = predict(model, device, feature_groups)
        true_labels = label_names_from_onehot(np.asarray(labels[local_idx], dtype=np.float32), class_names)
        output_path = args.output_dir / f"{image_path.stem}_exp07_prediction.png"

        print(f"Database index: {local_idx}")
        summary = plot_prediction(
            image_path=image_path,
            output_path=output_path,
            probs=probs,
            class_names=class_names,
            true_labels=true_labels,
            threshold=threshold,
            top_k=args.top_k,
            font_scale=args.font_scale,
        )
        summary["database_index"] = int(local_idx)
        summaries.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "prediction_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("\n" + "=" * 80)
    print(f"Processed {len(summaries)} image(s).")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
