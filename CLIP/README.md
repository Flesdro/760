# CLIP Multi-View Experiment

This folder contains the Python version of `new_CLIP.ipynb`.

Run the full experiment from the repository root:

```bash
python3 CLIP/train_clip.py
```

Default outputs are saved to `CLIP/runs/`:

- `clip_multiview_best.pt`
- `training_history.json`
- `test_metrics.json`
- `config.json`
- `alignment_metadata.json`

The script reuses the notebook cache files in the repository root by default:

- `aligned_tag_feature.npy`
- `matched_indices.npy`

For a quick smoke test:

```bash
python3 CLIP/train_clip.py --output-dir CLIP/smoke_test --epochs 1 --batch-size 64 --limit-samples 256 --patience 1
```
