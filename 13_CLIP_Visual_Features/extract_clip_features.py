from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_image_list(image_list_path: Path) -> list[str]:
    lines = image_list_path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def extract_clip_features(
    image_list_path: Path,
    images_dir: Path,
    output_path: Path,
    batch_size: int,
    clip_model: str,
    device: torch.device,
) -> dict[str, float | int | str]:
    try:
        import clip
    except ImportError as e:
        raise ImportError(
            "openai-clip is not installed. Run:\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        ) from e

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    print(f"Loading CLIP model: {clip_model}")
    model, preprocess = clip.load(clip_model, device=device)
    model.eval()

    image_paths = load_image_list(image_list_path)
    n_total = len(image_paths)
    print(f"Image list entries: {n_total}")

    feat_dim = model.visual.output_dim
    features = np.zeros((n_total, feat_dim), dtype=np.float32)

    found = 0
    missing = 0
    batch_imgs: list[torch.Tensor] = []
    batch_indices: list[int] = []

    def flush_batch() -> None:
        nonlocal found
        if not batch_imgs:
            return
        tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            feats = model.encode_image(tensor).float().cpu().numpy()
        for local_idx, feat in zip(batch_indices, feats):
            features[local_idx] = feat
        found += len(batch_imgs)
        batch_imgs.clear()
        batch_indices.clear()

    iterator = enumerate(image_paths)
    if use_tqdm:
        iterator = tqdm(iterator, total=n_total, desc="Extracting CLIP features", unit="img")

    for i, raw_path in iterator:
        filename = os.path.basename(raw_path.replace("\\", "/"))
        full_path = images_dir / filename

        if not full_path.exists():
            missing += 1
            continue

        try:
            img = preprocess(Image.open(full_path).convert("RGB"))
        except Exception:
            missing += 1
            continue

        batch_imgs.append(img)
        batch_indices.append(i)

        if len(batch_imgs) >= batch_size:
            flush_batch()

    flush_batch()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, features)

    metadata: dict[str, float | int | str] = {
        "clip_model": clip_model,
        "total_entries": n_total,
        "found": found,
        "missing": missing,
        "match_ratio": found / max(n_total, 1),
        "feature_dim": feat_dim,
        "output_path": str(output_path),
    }
    print(f"\nExtraction complete: {found}/{n_total} images found ({metadata['match_ratio']:.4f})")
    print(f"Feature shape: {features.shape}")
    print(f"Saved to: {output_path}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract CLIP ViT visual features for all NUS-WIDE images."
    )
    parser.add_argument(
        "--image-list", type=Path, default=Path("database_img.txt"),
        help="Text file listing image paths (one per line).",
    )
    parser.add_argument(
        "--images-dir", type=Path, default=Path("NUS-WIDE-images/images"),
        help="Directory containing the actual image files.",
    )
    parser.add_argument(
        "--output-path", type=Path,
        default=Path("13_CLIP_Visual_Features/clip_visual_features.npy"),
        help="Where to save the (N, 512) feature array.",
    )
    parser.add_argument(
        "--meta-path", type=Path,
        default=Path("13_CLIP_Visual_Features/clip_extraction_metadata.json"),
    )
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    args.image_list = resolve_project_path(args.image_list)
    args.images_dir = resolve_project_path(args.images_dir)
    args.output_path = resolve_project_path(args.output_path)
    args.meta_path = resolve_project_path(args.meta_path)

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Images dir: {args.images_dir}")
    print(f"Output: {args.output_path}")

    metadata = extract_clip_features(
        image_list_path=args.image_list,
        images_dir=args.images_dir,
        output_path=args.output_path,
        batch_size=args.batch_size,
        clip_model=args.clip_model,
        device=device,
    )
    save_json(args.meta_path, metadata)
    print(f"Metadata saved to: {args.meta_path}")


if __name__ == "__main__":
    main()
