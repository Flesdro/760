# Combined Final Test Results

All values are test-set scores reported as percentages. Bold values indicate the best result for that metric across all losses and model variants.

## Full Comparison

| Model | BCE mAP | BCE Micro-F1 | BCE Macro-F1 | Focal mAP | Focal Micro-F1 | Focal Macro-F1 | ASL mAP | ASL Micro-F1 | ASL Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline A (BoW) | 17.46 | 40.92 | 9.40 | 17.37 | 28.73 | 4.07 | 16.66 | 47.88 | 15.78 |
| Baseline B (Color+Texture) | 34.90 | 60.18 | 22.88 | 34.44 | 45.71 | 14.07 | 35.02 | **63.47** | 32.75 |
| Early Fusion (Concat) | 35.16 | 60.72 | 24.28 | **36.67** | 49.07 | 15.78 | 35.77 | 62.44 | 33.22 |
| Gated Fusion + InfoNCE | 33.15 | 58.65 | 17.59 | 31.72 | 38.98 | 7.35 | 32.97 | 61.79 | 29.52 |
| Gated Fusion (No InfoNCE) | 34.74 | 60.47 | 23.65 | 35.66 | 46.23 | 16.53 | 34.86 | 63.32 | **33.98** |

## Best Result by Metric

| Metric | Best Loss | Best Model | Score (%) |
|---|---|---|---:|
| mAP | Focal | Early Fusion (Concat) | 36.67 |
| Micro-F1 | ASL | Baseline B (Color+Texture) | 63.47 |
| Macro-F1 | ASL | Gated Fusion (No InfoNCE) | 33.98 |

## Best Loss for Each Model

| Model | Best mAP | Best Micro-F1 | Best Macro-F1 |
|---|---|---|---|
| Baseline A (BoW) | BCE, 17.46 | ASL, 47.88 | ASL, 15.78 |
| Baseline B (Color+Texture) | ASL, 35.02 | ASL, 63.47 | ASL, 32.75 |
| Early Fusion (Concat) | Focal, 36.67 | ASL, 62.44 | ASL, 33.22 |
| Gated Fusion + InfoNCE | BCE, 33.15 | ASL, 61.79 | ASL, 29.52 |
| Gated Fusion (No InfoNCE) | Focal, 35.66 | ASL, 63.32 | ASL, 33.98 |

## Suggested Reporting Focus

The full comparison table is the main table for the report because it preserves both design axes: model architecture and loss function. The second table can be used in the text to summarize headline findings: Focal gives the highest mAP through Early Fusion, while ASL gives the strongest F1 scores, especially Macro-F1. This suggests ASL is more effective for label imbalance and rare-label performance, even when its mAP is not the absolute highest.
