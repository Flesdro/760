# ML-GCN: Multi-Label Graph Convolutional Network

Instead of a shared linear classifier, ML-GCN uses two GCN layers to generate a
**label-specific classifier weight matrix** from label semantic embeddings and the
co-occurrence graph. Each label's classifier is informed by its graph neighbours.

```
Label names  →  CLIP text encoder  →  label embeddings (81 × 512)
                                              ↓
                                    GCN layer 1  (512 → 1024)
                                    LeakyReLU + Dropout
                                    GCN layer 2  (1024 → feat_dim)
                                              ↓
                              W  =  classifier weights  (81 × feat_dim)

Image features (batch × feat_dim)  →  L2-normalise  →  @ W.T  →  logits (batch × 81)
```

The adjacency matrix comes from `label_graph_fused.npy` (thresholded + row-normalised).
Label embeddings are generated on-the-fly via the CLIP text encoder at startup (81 prompts,
takes < 1 s).

## Prerequisites

- CLIP installed (see experiment 13)
- CLIP image features extracted (experiment 13 by default)

## Run

Default (uses ViT-B/32 features from exp13, ASL loss):

```bash
/home/jue/miniconda3/envs/ml/bin/python 16_ML_GCN/train_mlgcn.py
```

Using ViT-L/14 features from exp15:

```bash
/home/jue/miniconda3/envs/ml/bin/python 16_ML_GCN/train_mlgcn.py \
    --clip-feats-path 15_CLIP_ViT_L14/clip_vitl14_features.npy \
    --output-dir 16_ML_GCN/runs_vitl14
```

Outputs are saved to `16_ML_GCN/runs/`:

- `mlgcn_best.pt`
- `training_history.json`
- `test_metrics.json`
- `config.json`

## Key differences from experiments 13–15

| | Exp13/14/15 | Exp16 (ML-GCN) |
|---|---|---|
| Classifier | Shared linear layer (512→81) | Label-specific weights via GCN |
| Label correlations | Post-hoc linear refiner | Baked into classifier weights |
| Label semantics | Ignored | CLIP text embeddings as GCN input |
| Parameters | ~2M | ~2.1M (GCN adds 2 small weight matrices) |

## Tunable arguments

| Argument | Default | Description |
|---|---|---|
| `--gcn-hidden-dim` | 1024 | Hidden dimension of GCN layer 1 |
| `--graph-threshold` | 0.01 | Min edge weight to keep (sparsifies the graph) |
| `--label-embed-model` | ViT-B/32 | CLIP model for text label embeddings |
