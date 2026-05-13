# Pure MLP Concat BCE Baseline

This baseline trains on the extracted feature vectors in `dataset/`.
It directly concatenates all feature arrays and feeds the result into a plain MLP.

It intentionally does not use projection branches, gated fusion, residual blocks, InfoNCE, Focal Loss, or ASL.

Default run:

```bash
python Pure_MLP_Concat_BCE/train_pure_mlp_concat_bce.py
```

The default setting mirrors the 81-label BCE early-fusion experiment:

```text
epochs = 35
hidden_dims = 1024,512,256
dropout = 0.3
batch_size = 32
lr = 0.0001
loss = BCEWithLogitsLoss
```

Outputs are saved to `Pure_MLP_Concat_BCE/runs/`:

- `pure_mlp_concat_bce_best.pt`
- `training_history.json`
- `test_metrics.json`
- `config.json`
