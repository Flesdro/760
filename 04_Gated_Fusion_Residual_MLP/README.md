# Gated Fusion Residual MLP

This model trains on the six extracted feature groups in `dataset/`.
It does not use raw images.

Each feature group is encoded separately:

- BoW
- CH
- CM55
- CORR
- EDH
- WT

The model learns a softmax gate over the six encoded feature groups, scales each group by its gate weight, concatenates the gated representations, and sends the result through a Residual MLP classifier.

Default run:

```bash
python Gated_Fusion_Residual_MLP/train_gated_fusion_residual_mlp.py
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
```

Outputs are saved to `Gated_Fusion_Residual_MLP/runs/`:

- `gated_fusion_residual_mlp_best.pt`
- `training_history.json`
- `test_metrics.json`
- `config.json`

The metric JSON files also include average gate weights, which can help show which feature groups the model uses most.
