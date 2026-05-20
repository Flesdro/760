# Tag Importance Analysis

This experiment explains why `aligned_tag_feature.npy` is useful rather than only showing that removing it lowers mAP.

It analyzes the relationship between:

- `aligned_tag_feature.npy`: 1000-dimensional image tag input aligned from `Train_Tags1k.dat`
- `TagList1k.txt`: names of the 1000 original NUS-WIDE user tags
- `Concepts81.txt`: the 81 prediction targets
- `database_labels_81_big.npy`: the 81-dimensional target labels

Run from the repository root with the `ml` conda environment:

```bash
/home/jue/miniconda3/envs/ml/bin/python 10_Tag_Importance_Analysis/analyze_tag_importance.py
```

Default outputs are saved to `10_Tag_Importance_Analysis/runs/`:

- `summary.json`
- `report.md`
- `label_summary.csv`
- `top_tags_by_label.csv`
- `concept_tag_overlap.csv`
- `top_tag_vocabulary_used.csv`

The most important evidence is:

- how many 81 target concept names appear directly in the 1000-tag vocabulary
- which tags are most correlated with each target label
- how much correlation disappears when tag rows are shuffled
- whether tag dimensions have stronger direct semantic correlation than handcrafted visual features
