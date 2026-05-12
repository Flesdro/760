# Gated Fusion Residual MLP with ASL

This version uses the same feature-group fusion model as `Gated_Fusion_Residual_MLP`, but replaces `BCEWithLogitsLoss` with Asymmetric Loss.

It is intended for the 81-class multi-label setting, where negative labels dominate and some labels are rare.

Default run:

```bash
python Gated_Fusion_Residual_MLP_ASL/train_gated_fusion_residual_mlp_asl.py
```

Default settings:

```text
epochs = 50
group_embed_dim = 128
hidden_dim = 512
num_blocks = 4
dropout = 0.3
batch_size = 256
lr = 0.001
asl_gamma_neg = 4.0
asl_gamma_pos = 1.0
asl_clip = 0.05
```

Outputs are saved to `Gated_Fusion_Residual_MLP_ASL/runs/`:

- `gated_fusion_residual_mlp_asl_best.pt`
- `training_history.json`
- `test_metrics.json`
- `config.json`

The mAP calculation skips classes with no positive samples in the evaluated split, which avoids misleading warnings from `sklearn`.
