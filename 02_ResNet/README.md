# ResNet-6 Feature Baseline

This baseline trains a small ResNet-style model on the extracted feature files in `dataset/`.
It does not read images.

The model treats the concatenated feature vector as a 1D signal:

- `Conv1d` stem
- 2 residual `BasicBlock1D` blocks
- global average pooling
- linear multi-label classifier

Layer count: 1 stem convolution + 2 blocks x 2 convolutions + 1 linear head = 6 trainable layers.

Run from the repository root:

```bash
python ResNet/train_resnet6_features.py
```

Useful quick run:

```bash
python ResNet/train_resnet6_features.py --epochs 3 --batch-size 512
```

Outputs are written to `ResNet/resnet6_feature_runs/`:

- `resnet6_features_best.pt`
- `training_history.json`
- `test_metrics.json`
- `config.json`
