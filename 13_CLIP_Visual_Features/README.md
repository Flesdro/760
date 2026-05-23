# CLIP Visual Features Experiment

This experiment replaces the handcrafted visual features (CH/CM55/CORR/EDH/WT) with frozen
CLIP ViT-B/32 image embeddings (512-dim). No tag features are used, so there is no label
leakage.

The pipeline has two steps:

1. **Extract** CLIP features from the original images (run once, ~20 min on RTX 3060).
2. **Train** the multi-label classifier on the extracted features.

## Setup

Install openai-clip if not already present:

```bash
pip install git+https://github.com/openai/CLIP.git
pip install tqdm  # optional but shows a progress bar
```

## Step 1 — Extract CLIP features

Run from the repository root:

```bash
/home/jue/miniconda3/envs/ml/bin/python 13_CLIP_Visual_Features/extract_clip_features.py
```

This reads `database_img.txt` (193 734 entries), finds each image in
`NUS-WIDE-images/images/`, and writes:

- `13_CLIP_Visual_Features/clip_visual_features.npy` — shape (193 734, 512), float32
- `13_CLIP_Visual_Features/clip_extraction_metadata.json`

Images not found on disk are stored as zero vectors and excluded during training.

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--images-dir` | `NUS-WIDE-images/images` | Directory containing jpg files |
| `--clip-model` | `ViT-B/32` | CLIP backbone (`ViT-B/16` gives 512-dim too, slightly stronger) |
| `--batch-size` | `256` | Inference batch size (reduce if OOM) |

Quick smoke test (first 512 images):

```bash
/home/jue/miniconda3/envs/ml/bin/python 13_CLIP_Visual_Features/extract_clip_features.py \
    --batch-size 64
```

## Step 2 — Train

Run from the repository root after extraction completes:

```bash
/home/jue/miniconda3/envs/ml/bin/python 13_CLIP_Visual_Features/train_clip_visual.py
```

To also include the 5 handcrafted visual features as additional views (CLIP + CH/CM55/CORR/EDH/WT):

```bash
/home/jue/miniconda3/envs/ml/bin/python 13_CLIP_Visual_Features/train_clip_visual.py \
    --use-handcrafted
```

Outputs are saved to `13_CLIP_Visual_Features/runs/`:

- `clip_visual_best.pt`
- `training_history.json`
- `test_metrics.json`
- `config.json`

## Design notes

- Features are L2-normalised by CLIP internally; no additional normalisation is applied.
- Rows where the image was not found (zero vector) are filtered out before training.
- The same residual MLP backbone and label correlation refiner as experiments 08/09 are used
  so results are directly comparable.
- Default epochs/patience are slightly higher than exp09 because CLIP features converge more
  slowly than tag-heavy inputs.
