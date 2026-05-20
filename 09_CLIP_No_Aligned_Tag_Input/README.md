# CLIP Without Aligned Tag Input

This experiment keeps the same matched image subset as `08_CLIP_Cooccurance_Only`, but removes the `aligned_tag_feature.npy` / `ALIGNED_TAG_PATH` input view. The model only receives the five visual feature views:

- `Normalized_CH.npy`
- `Normalized_CM55.npy`
- `Normalized_CORR.npy`
- `Normalized_EDH.npy`
- `Normalized_WT.npy`

Run from the repository root with the `ml` conda environment:

```bash
/home/jue/miniconda3/envs/ml/bin/python 09_CLIP_No_Aligned_Tag_Input/train_clip_no_aligned_tag.py
```

Default outputs are saved to `09_CLIP_No_Aligned_Tag_Input/runs/`:

- `clip_multiview_best.pt`
- `training_history.json`
- `test_metrics.json`
- `config.json`
- `alignment_metadata.json`
- `matched_indices.npy`
