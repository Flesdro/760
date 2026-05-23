# CLIP ViT-L/14 + ASL

This experiment upgrades the CLIP backbone from ViT-B/32 (512-dim) to ViT-L/14 (768-dim).
ViT-L/14 uses a larger Vision Transformer with smaller 14×14 patches, producing richer
semantic embeddings. ASL is kept from experiment 14 as it handles class imbalance better
than BCE.

## Backbone comparison

| Model | Dim | Patch size | Params |
|---|---|---|---|
| ViT-B/32 (Exp13/14) | 512 | 32×32 | 88M |
| ViT-L/14 (Exp15) | 768 | 14×14 | 307M |

## Step 1 — Extract ViT-L/14 features

Reuse the extractor from experiment 13 with `--clip-model ViT-L/14`:

```bash
/home/jue/miniconda3/envs/ml/bin/python 13_CLIP_Visual_Features/extract_clip_features.py \
    --clip-model ViT-L/14 \
    --output-path 15_CLIP_ViT_L14/clip_vitl14_features.npy \
    --meta-path 15_CLIP_ViT_L14/clip_extraction_metadata.json
```

Output: `15_CLIP_ViT_L14/clip_vitl14_features.npy` — shape (193734, 768), ~560MB.

Note: ViT-L/14 model weights (~1.7GB) are downloaded automatically on first run.

## Step 2 — Train

```bash
/home/jue/miniconda3/envs/ml/bin/python 15_CLIP_ViT_L14/train_clip_vitl14.py
```

Outputs are saved to `15_CLIP_ViT_L14/runs/`:

- `clip_vitl14_best.pt`
- `training_history.json`
- `test_metrics.json`
- `config.json`

## Architecture change vs Exp13/14

The only backbone change is an extra hidden_dim tier for the FeatureEncoder:

```
dim >= 768  →  hidden_dim = 768   (new, preserves full ViT-L/14 dimensionality)
dim >= 512  →  hidden_dim = 512
dim >= 200  →  hidden_dim = 256
else        →  hidden_dim = 128
```

Everything else (fusion MLP, residual shortcut, LabelCorrelationRefiner, ASL) is identical
to experiment 14.
