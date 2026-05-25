# ResNet-6 Feature Baseline

This baseline trains a small ResNet-style model on the extracted feature files in `dataset/`.
It does not read images.
By default it uses the same clean matched feature input and shared training defaults as experiment 07,
with `20_train_gated_fusion_residual_mlp_asl_tag_no_overlap/config.json` as the reference config.

The model treats the concatenated feature vector as a 1D signal:

- `Conv1d` stem
- 2 residual `BasicBlock1D` blocks
- global average pooling
- linear multi-label classifier

Layer count: 1 stem convolution + 2 blocks x 2 convolutions + 1 linear head = 6 trainable layers.

Run from the repository root:

```bash
python 02_ResNet/train_resnet6_features.py
```

Useful quick run:

```bash
python 02_ResNet/train_resnet6_features.py --epochs 3 --batch-size 512
```

Outputs are written to `02_ResNet/resnet6_feature_clean_runs/` by default:

- `resnet6_features_clean_best.pt`
- `training_history.json`
- `test_metrics.json`
- `config.json`
