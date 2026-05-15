# Residual MLP Baseline

This baseline trains on the extracted feature vectors in `dataset/`.
It does not use raw images.

Default run:

```bash
python Residual_MLP/train_residual_mlp.py
```

The default setting trains for 50 epochs:

```text
epochs = 50
hidden_dim = 512
num_blocks = 4
dropout = 0.3
batch_size = 256
lr = 0.001
```

Outputs are saved to `Residual_MLP/residual_mlp_runs/`:

- `residual_mlp_best.pt`
- `training_history.json`
- `test_metrics.json`
- `config.json`
