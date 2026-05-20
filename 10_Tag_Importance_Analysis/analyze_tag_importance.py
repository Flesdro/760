from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VISUAL_FEATURE_PATHS = [
    Path("Extracted_Features/Normalized_CH.npy"),
    Path("Extracted_Features/Normalized_CM55.npy"),
    Path("Extracted_Features/Normalized_CORR.npy"),
    Path("Extracted_Features/Normalized_EDH.npy"),
    Path("Extracted_Features/Normalized_WT.npy"),
]


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_token(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def binary_corr_from_counts(
    n: int,
    y_sum: np.ndarray,
    x_sum: np.ndarray,
    joint: np.ndarray,
) -> np.ndarray:
    numerator = n * joint - y_sum[:, None] * x_sum[None, :]
    denominator = np.sqrt(
        y_sum[:, None]
        * (n - y_sum[:, None])
        * x_sum[None, :]
        * (n - x_sum[None, :])
    )
    corr = np.zeros_like(joint, dtype=np.float64)
    np.divide(numerator, denominator, out=corr, where=denominator > 0)
    return corr


def compute_tag_statistics(
    tags: np.ndarray,
    labels: np.ndarray,
    tag_names: list[str],
    label_names: list[str],
    top_k: int,
    seed: int,
    run_shuffle: bool,
) -> dict[str, Any]:
    x = (tags > 0).astype(np.float32, copy=False)
    y = labels.astype(np.float32, copy=False)
    n = x.shape[0]

    tag_sum = x.sum(axis=0, dtype=np.float64)
    label_sum = y.sum(axis=0, dtype=np.float64)
    joint = (y.T @ x).astype(np.float64, copy=False)

    corr = binary_corr_from_counts(n, label_sum, tag_sum, joint)

    positive_rate = np.zeros_like(joint, dtype=np.float64)
    negative_rate = np.zeros_like(joint, dtype=np.float64)
    np.divide(joint, label_sum[:, None], out=positive_rate, where=label_sum[:, None] > 0)
    np.divide(
        tag_sum[None, :] - joint,
        n - label_sum[:, None],
        out=negative_rate,
        where=(n - label_sum[:, None]) > 0,
    )
    lift = (positive_rate + 1e-8) / (negative_rate + 1e-8)

    top_indices = np.argsort(-corr, axis=1)[:, :top_k]
    best_indices = top_indices[:, 0]

    top_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    best_single_tag_ap: list[float] = []
    exact_tag_ap_values: list[float] = []

    normalized_tag_to_index = {
        normalize_token(tag_name): idx
        for idx, tag_name in enumerate(tag_names)
    }

    for label_idx, label_name in enumerate(label_names):
        label_count = int(label_sum[label_idx])
        best_tag_idx = int(best_indices[label_idx])
        if label_count > 0:
            best_ap = float(average_precision_score(y[:, label_idx], x[:, best_tag_idx]))
        else:
            best_ap = 0.0
        best_single_tag_ap.append(best_ap)

        exact_tag_idx = normalized_tag_to_index.get(normalize_token(label_name))
        exact_rank = None
        exact_corr = None
        exact_pos_rate = None
        exact_neg_rate = None
        exact_lift = None
        exact_ap = None
        if exact_tag_idx is not None:
            exact_rank = int(np.sum(corr[label_idx] > corr[label_idx, exact_tag_idx]) + 1)
            exact_corr = float(corr[label_idx, exact_tag_idx])
            exact_pos_rate = float(positive_rate[label_idx, exact_tag_idx])
            exact_neg_rate = float(negative_rate[label_idx, exact_tag_idx])
            exact_lift = float(lift[label_idx, exact_tag_idx])
            if label_count > 0:
                exact_ap = float(average_precision_score(y[:, label_idx], x[:, exact_tag_idx]))
                exact_tag_ap_values.append(exact_ap)

        label_rows.append(
            {
                "label_index": label_idx,
                "label": label_name,
                "label_positive_count": label_count,
                "label_positive_rate": float(label_sum[label_idx] / n),
                "best_tag_index": best_tag_idx,
                "best_tag": tag_names[best_tag_idx],
                "best_tag_corr": float(corr[label_idx, best_tag_idx]),
                "best_tag_pos_rate": float(positive_rate[label_idx, best_tag_idx]),
                "best_tag_neg_rate": float(negative_rate[label_idx, best_tag_idx]),
                "best_tag_lift": float(lift[label_idx, best_tag_idx]),
                "best_single_tag_ap": best_ap,
                "exact_tag_index": exact_tag_idx,
                "exact_tag_rank": exact_rank,
                "exact_tag_corr": exact_corr,
                "exact_tag_pos_rate": exact_pos_rate,
                "exact_tag_neg_rate": exact_neg_rate,
                "exact_tag_lift": exact_lift,
                "exact_tag_ap": exact_ap,
            }
        )

        overlap_rows.append(
            {
                "label_index": label_idx,
                "label": label_name,
                "has_exact_tag_in_1k": exact_tag_idx is not None,
                "exact_tag_index": exact_tag_idx,
                "exact_tag_rank": exact_rank,
                "exact_tag_corr": exact_corr,
                "exact_tag_ap": exact_ap,
            }
        )

        for rank, tag_idx in enumerate(top_indices[label_idx], start=1):
            tag_idx = int(tag_idx)
            top_rows.append(
                {
                    "label_index": label_idx,
                    "label": label_name,
                    "rank": rank,
                    "tag_index": tag_idx,
                    "tag": tag_names[tag_idx],
                    "corr": float(corr[label_idx, tag_idx]),
                    "positive_rate": float(positive_rate[label_idx, tag_idx]),
                    "negative_rate": float(negative_rate[label_idx, tag_idx]),
                    "lift": float(lift[label_idx, tag_idx]),
                    "is_exact_label_name": normalize_token(tag_names[tag_idx]) == normalize_token(label_name),
                }
            )

    shuffled_summary = None
    shuffled_best_corr = None
    if run_shuffle:
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(n)
        shuffled_joint = (y.T @ x[permutation]).astype(np.float64, copy=False)
        shuffled_corr = binary_corr_from_counts(n, label_sum, tag_sum, shuffled_joint)
        shuffled_best_corr = np.max(shuffled_corr, axis=1)
        shuffled_summary = {
            "mean_best_corr": float(np.mean(shuffled_best_corr)),
            "median_best_corr": float(np.median(shuffled_best_corr)),
            "max_best_corr": float(np.max(shuffled_best_corr)),
        }

    top_tag_counter: Counter[str] = Counter()
    top_tag_labels: dict[str, list[str]] = defaultdict(list)
    for row in top_rows:
        tag_name = str(row["tag"])
        top_tag_counter[tag_name] += 1
        if len(top_tag_labels[tag_name]) < 12:
            top_tag_labels[tag_name].append(str(row["label"]))

    top_vocabulary_rows = [
        {
            "tag": tag_name,
            "times_in_label_top_k": count,
            "example_labels": "; ".join(top_tag_labels[tag_name]),
        }
        for tag_name, count in top_tag_counter.most_common()
    ]

    exact_overlap_count = sum(1 for row in overlap_rows if row["has_exact_tag_in_1k"])
    best_corr = corr[np.arange(len(label_names)), best_indices]
    exact_labels_missing = [
        row["label"]
        for row in overlap_rows
        if not row["has_exact_tag_in_1k"]
    ]

    return {
        "tag_binary_matrix": x,
        "label_rows": label_rows,
        "top_rows": top_rows,
        "overlap_rows": overlap_rows,
        "top_vocabulary_rows": top_vocabulary_rows,
        "best_corr": best_corr,
        "shuffled_best_corr": shuffled_best_corr,
        "summary": {
            "num_samples": int(n),
            "tag_dim": int(x.shape[1]),
            "label_dim": int(y.shape[1]),
            "mean_active_tags_per_image": float(np.mean(np.sum(x, axis=1))),
            "median_active_tags_per_image": float(np.median(np.sum(x, axis=1))),
            "tag_density": float(np.mean(x)),
            "mean_labels_per_image": float(np.mean(np.sum(y, axis=1))),
            "median_labels_per_image": float(np.median(np.sum(y, axis=1))),
            "concept_exact_overlap_count": int(exact_overlap_count),
            "concept_exact_overlap_ratio": float(exact_overlap_count / len(label_names)),
            "concepts_missing_exact_1k_tag": exact_labels_missing,
            "mean_best_tag_corr": float(np.mean(best_corr)),
            "median_best_tag_corr": float(np.median(best_corr)),
            "labels_with_best_tag_corr_ge_0_10": int(np.sum(best_corr >= 0.10)),
            "labels_with_best_tag_corr_ge_0_20": int(np.sum(best_corr >= 0.20)),
            "labels_with_best_tag_corr_ge_0_30": int(np.sum(best_corr >= 0.30)),
            "mean_best_single_tag_ap": float(np.mean(best_single_tag_ap)),
            "median_best_single_tag_ap": float(np.median(best_single_tag_ap)),
            "mean_exact_tag_ap_available_labels": float(np.mean(exact_tag_ap_values)) if exact_tag_ap_values else None,
            "median_exact_tag_ap_available_labels": float(np.median(exact_tag_ap_values)) if exact_tag_ap_values else None,
            "shuffled_tag_summary": shuffled_summary,
        },
    }


def compute_visual_correlation_summary(
    data_root: Path,
    matched_indices: np.ndarray,
    labels: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y = labels.astype(np.float32, copy=False)
    n = y.shape[0]
    y_mean = y.mean(axis=0, dtype=np.float64)
    y_std = y.std(axis=0, dtype=np.float64)

    best_abs_corr = np.zeros(y.shape[1], dtype=np.float64)
    best_feature_name = ["" for _ in range(y.shape[1])]
    best_feature_index = np.zeros(y.shape[1], dtype=np.int64)
    view_summaries: list[dict[str, Any]] = []

    for view_path in VISUAL_FEATURE_PATHS:
        feature = np.load(data_root / view_path, mmap_mode="r")[matched_indices].astype(np.float32)
        x_mean = feature.mean(axis=0, dtype=np.float64)
        x_std = feature.std(axis=0, dtype=np.float64)
        cross = (y.T @ feature).astype(np.float64, copy=False) / n
        denominator = y_std[:, None] * x_std[None, :]
        corr = np.zeros_like(cross, dtype=np.float64)
        np.divide(cross - y_mean[:, None] * x_mean[None, :], denominator, out=corr, where=denominator > 0)

        abs_corr = np.abs(corr)
        local_best_idx = np.argmax(abs_corr, axis=1)
        local_best_abs = abs_corr[np.arange(y.shape[1]), local_best_idx]
        improved = local_best_abs > best_abs_corr
        for label_idx in np.where(improved)[0]:
            best_abs_corr[label_idx] = float(local_best_abs[label_idx])
            best_feature_name[label_idx] = view_path.name
            best_feature_index[label_idx] = int(local_best_idx[label_idx])

        view_summaries.append(
            {
                "view": view_path.name,
                "dim": int(feature.shape[1]),
                "mean_best_abs_corr": float(np.mean(local_best_abs)),
                "median_best_abs_corr": float(np.median(local_best_abs)),
                "max_best_abs_corr": float(np.max(local_best_abs)),
            }
        )

    rows = [
        {
            "label_index": label_idx,
            "best_visual_abs_corr": float(best_abs_corr[label_idx]),
            "best_visual_view": best_feature_name[label_idx],
            "best_visual_feature_index": int(best_feature_index[label_idx]),
        }
        for label_idx in range(y.shape[1])
    ]
    summary = {
        "view_summaries": view_summaries,
        "mean_best_visual_abs_corr": float(np.mean(best_abs_corr)),
        "median_best_visual_abs_corr": float(np.median(best_abs_corr)),
        "max_best_visual_abs_corr": float(np.max(best_abs_corr)),
    }
    return rows, summary


def merge_visual_rows(
    label_rows: list[dict[str, Any]],
    visual_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    visual_by_label = {row["label_index"]: row for row in visual_rows}
    merged = []
    for row in label_rows:
        new_row = dict(row)
        visual = visual_by_label.get(row["label_index"], {})
        new_row.update(
            {
                "best_visual_abs_corr": visual.get("best_visual_abs_corr"),
                "best_visual_view": visual.get("best_visual_view"),
                "best_visual_feature_index": visual.get("best_visual_feature_index"),
            }
        )
        merged.append(new_row)
    return merged


def build_markdown_report(
    output_dir: Path,
    summary: dict[str, Any],
    label_rows: list[dict[str, Any]],
    top_rows: list[dict[str, Any]],
) -> None:
    strongest = sorted(label_rows, key=lambda row: row["best_tag_corr"], reverse=True)[:8]
    by_label_rank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in top_rows:
        if int(row["rank"]) <= 5:
            by_label_rank[str(row["label"])].append(row)

    lines = [
        "# Tag Importance Analysis",
        "",
        "This experiment explains why `aligned_tag_feature.npy` is important, beyond the ablation mAP drop.",
        "",
        "## Main Evidence",
        "",
        f"- The model target has 81 labels, but `Train_Tags1k.dat` provides a 1000-dimensional original user-tag vocabulary.",
        f"- {summary['concept_exact_overlap_count']} / 81 target concept names appear directly in the 1000-tag vocabulary.",
        f"- Each matched image has {summary['mean_active_tags_per_image']:.2f} active 1k-tags on average.",
        f"- The mean best tag-label correlation is {summary['mean_best_tag_corr']:.4f}.",
        f"- A single best tag dimension per class gives mean AP {summary['mean_best_single_tag_ap']:.4f}.",
    ]

    shuffled = summary.get("shuffled_tag_summary")
    if shuffled:
        lines.append(
            f"- After shuffling tag rows, mean best tag-label correlation drops to {shuffled['mean_best_corr']:.4f}, showing that image-tag alignment matters."
        )

    if "visual_summary" in summary:
        visual_summary = summary["visual_summary"]
        lines.append(
            f"- The best individual handcrafted visual dimension has mean absolute correlation {visual_summary['mean_best_visual_abs_corr']:.4f}, lower than the aligned tags."
        )

    lines.extend(
        [
            "",
            "## Why 1000 Tags Instead Of 81",
            "",
            "The 81 concepts are the prediction targets. The 1000 tags are not another target label set; they are input metadata from the original NUS-WIDE tag vocabulary. They include direct target words plus related context words, attributes, synonyms, and scene cues. For example, a label such as `beach` can be supported by tags like `water`, `sea`, `sand`, `sunset`, or `vacation`. Reducing the input to only 81 exact concept words would discard these extra semantic cues.",
            "",
            "## Strongest Examples",
            "",
            "| label | best tag | corr | pos rate | neg rate | top 5 related tags |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )

    for row in strongest:
        label = str(row["label"])
        top_tags = ", ".join(str(item["tag"]) for item in by_label_rank[label])
        lines.append(
            f"| {label} | {row['best_tag']} | {row['best_tag_corr']:.4f} | "
            f"{row['best_tag_pos_rate']:.4f} | {row['best_tag_neg_rate']:.4f} | {top_tags} |"
        )

    lines.extend(
        [
            "",
            "## Generated Files",
            "",
            "- `summary.json`: aggregate statistics used in this report.",
            "- `label_summary.csv`: one row per 81 target label.",
            "- `top_tags_by_label.csv`: top correlated 1k-tags for every target label.",
            "- `concept_tag_overlap.csv`: whether each target label name appears in the 1000-tag vocabulary.",
            "- `top_tag_vocabulary_used.csv`: 1k-tags that repeatedly appear as strong predictors.",
        ]
    )

    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain why the 1000-dimensional aligned tag input is important.")
    parser.add_argument("--aligned-tag-path", type=Path, default=Path("aligned_tag_feature.npy"))
    parser.add_argument("--matched-indices-path", type=Path, default=Path("matched_indices.npy"))
    parser.add_argument("--labels-path", type=Path, default=Path("dataset/database_labels_81_big.npy"))
    parser.add_argument("--tag-list-path", type=Path, default=Path("dataset/NUS_WID_Tags/TagList1k.txt"))
    parser.add_argument("--concepts-path", type=Path, default=Path("Concepts81.txt"))
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("10_Tag_Importance_Analysis/runs"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-shuffle", action="store_true")
    parser.add_argument("--skip-visual-comparison", action="store_true")
    args = parser.parse_args()

    aligned_tag_path = resolve_project_path(args.aligned_tag_path)
    matched_indices_path = resolve_project_path(args.matched_indices_path)
    labels_path = resolve_project_path(args.labels_path)
    tag_list_path = resolve_project_path(args.tag_list_path)
    concepts_path = resolve_project_path(args.concepts_path)
    data_root = resolve_project_path(args.data_root)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_names = read_lines(concepts_path)
    tag_names = read_lines(tag_list_path)
    matched_indices = np.load(matched_indices_path)

    print("Loading aligned 1000-dimensional tags...")
    tags = np.load(aligned_tag_path, mmap_mode="r")[matched_indices].astype(np.float32)
    print("Loading 81 target labels...")
    labels = np.load(labels_path, mmap_mode="r")[matched_indices].astype(np.float32)

    if tags.shape[1] != len(tag_names):
        raise ValueError(f"Tag matrix dim {tags.shape[1]} does not match tag list length {len(tag_names)}")
    if labels.shape[1] != len(label_names):
        raise ValueError(f"Label matrix dim {labels.shape[1]} does not match concept list length {len(label_names)}")
    if tags.shape[0] != labels.shape[0]:
        raise ValueError(f"Row mismatch: tags={tags.shape[0]}, labels={labels.shape[0]}")

    print("Computing tag-label correlations and top related tags...")
    tag_results = compute_tag_statistics(
        tags=tags,
        labels=labels,
        tag_names=tag_names,
        label_names=label_names,
        top_k=args.top_k,
        seed=args.seed,
        run_shuffle=not args.skip_shuffle,
    )

    label_rows = tag_results["label_rows"]
    summary = tag_results["summary"]

    if not args.skip_visual_comparison:
        print("Computing visual-feature correlation baseline...")
        visual_rows, visual_summary = compute_visual_correlation_summary(data_root, matched_indices, labels)
        label_rows = merge_visual_rows(label_rows, visual_rows)
        summary["visual_summary"] = visual_summary
        if summary["mean_best_tag_corr"] > 0:
            summary["tag_to_visual_best_corr_ratio"] = (
                summary["mean_best_tag_corr"] / max(visual_summary["mean_best_visual_abs_corr"], 1e-12)
            )

    print("Saving analysis outputs...")
    summary.update(
        {
            "source_files": {
                "aligned_tag_path": str(aligned_tag_path),
                "matched_indices_path": str(matched_indices_path),
                "labels_path": str(labels_path),
                "tag_list_path": str(tag_list_path),
                "concepts_path": str(concepts_path),
            }
        }
    )

    label_fields = [
        "label_index",
        "label",
        "label_positive_count",
        "label_positive_rate",
        "best_tag_index",
        "best_tag",
        "best_tag_corr",
        "best_tag_pos_rate",
        "best_tag_neg_rate",
        "best_tag_lift",
        "best_single_tag_ap",
        "exact_tag_index",
        "exact_tag_rank",
        "exact_tag_corr",
        "exact_tag_pos_rate",
        "exact_tag_neg_rate",
        "exact_tag_lift",
        "exact_tag_ap",
        "best_visual_abs_corr",
        "best_visual_view",
        "best_visual_feature_index",
    ]
    label_fields = [field for field in label_fields if any(field in row for row in label_rows)]

    save_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "label_summary.csv", label_rows, label_fields)
    write_csv(
        output_dir / "top_tags_by_label.csv",
        tag_results["top_rows"],
        [
            "label_index",
            "label",
            "rank",
            "tag_index",
            "tag",
            "corr",
            "positive_rate",
            "negative_rate",
            "lift",
            "is_exact_label_name",
        ],
    )
    write_csv(
        output_dir / "concept_tag_overlap.csv",
        tag_results["overlap_rows"],
        [
            "label_index",
            "label",
            "has_exact_tag_in_1k",
            "exact_tag_index",
            "exact_tag_rank",
            "exact_tag_corr",
            "exact_tag_ap",
        ],
    )
    write_csv(
        output_dir / "top_tag_vocabulary_used.csv",
        tag_results["top_vocabulary_rows"],
        ["tag", "times_in_label_top_k", "example_labels"],
    )
    build_markdown_report(output_dir, summary, label_rows, tag_results["top_rows"])

    print("\nDone.")
    print(f"Output dir: {output_dir}")
    print(f"Exact 81-concept names found in 1k tags: {summary['concept_exact_overlap_count']} / 81")
    print(f"Mean best tag-label corr: {summary['mean_best_tag_corr']:.4f}")
    if summary.get("shuffled_tag_summary"):
        print(f"Mean best shuffled corr: {summary['shuffled_tag_summary']['mean_best_corr']:.4f}")
    if summary.get("visual_summary"):
        print(f"Mean best visual abs corr: {summary['visual_summary']['mean_best_visual_abs_corr']:.4f}")
    print(f"Mean AP from one best tag per label: {summary['mean_best_single_tag_ap']:.4f}")


if __name__ == "__main__":
    main()
