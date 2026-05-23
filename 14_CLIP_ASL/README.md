# CLIP + Asymmetric Loss (ASL)

This experiment combines CLIP ViT-B/32 visual features (experiment 13) with Asymmetric Loss
instead of BCE. ASL is designed for heavily imbalanced multi-label settings: it applies a
stronger down-weighting to easy negatives (gamma_neg) than to positives (gamma_pos), which
helps rare classes that are dominated by negatives in standard BCE training.

This experiment reuses `13_CLIP_Visual_Features/clip_visual_features.npy` — run experiment 13
extraction step first.

## Run

```bash
/home/jue/miniconda3/envs/ml/bin/python 14_CLIP_ASL/train_clip_asl.py
```

With handcrafted features added (ablation):

```bash
/home/jue/miniconda3/envs/ml/bin/python 14_CLIP_ASL/train_clip_asl.py \
    --use-handcrafted \
    --output-dir 14_CLIP_ASL/runs_clip_handcrafted_asl
```

Outputs are saved to `14_CLIP_ASL/runs/`:

- `clip_asl_best.pt`
- `training_history.json`
- `test_metrics.json`
- `config.json`

## ASL parameters

| Parameter | Default | Effect |
|---|---|---|
| `--asl-gamma-neg` | 4.0 | Down-weights easy negatives more aggressively |
| `--asl-gamma-pos` | 1.0 | Mild focusing on hard positives |
| `--asl-clip` | 0.05 | Probability margin that prevents hard negatives from dominating |

## Comparison with other experiments

| Experiment | Features | Loss | Expected trend |
|---|---|---|---|
| Exp09 | Handcrafted only | BCE | baseline |
| Exp05 | Handcrafted only | ASL | loss improvement |
| Exp13 | CLIP only | BCE | feature improvement |
| **Exp14** | **CLIP only** | **ASL** | **both improvements** |
