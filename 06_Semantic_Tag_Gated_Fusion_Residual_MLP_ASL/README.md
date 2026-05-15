# Semantic Tag Gated Fusion Residual MLP with ASL

This experiment combines four pieces:

- Residual MLP backbone
- Asymmetric Loss for imbalanced multi-label classification
- Semantic tag modality from the NUS-WIDE 1k tag features
- Gated fusion over each feature group, including a separate gate for `semantic_tag`

The model follows `new_CLIP.ipynb`: it uses five visual descriptors (`CH`, `CM55`, `CORR`, `EDH`, `WT`) and adds the 1k tag feature file as a sixth modality. Branch dimensions are inferred from the feature size: `>=1000 -> 512`, `>=200 -> 256`, otherwise `128`.

Default run:

```bash
python Semantic_Tag_Gated_Fusion_Residual_MLP_ASL/train_semantic_tag_gated_fusion_residual_mlp_asl.py \
  --data-root dataset
```

If the tag files are placed under `dataset/NUS_WID_Tags/`, the defaults are:

```text
dataset/NUS_WID_Tags/Train_Tags1k.dat
dataset/NUS_WID_Tags/Test_Tags1k.dat
```

Useful options:

```bash
python Semantic_Tag_Gated_Fusion_Residual_MLP_ASL/train_semantic_tag_gated_fusion_residual_mlp_asl.py \
  --data-root dataset \
  --train-tag-path NUS_WID_Tags/Train_Tags1k.dat \
  --test-tag-path NUS_WID_Tags/Test_Tags1k.dat \
  --epochs 50 \
  --batch-size 128 \
  --hidden-dim 512 \
  --num-blocks 4
```

Quick smoke test:

```bash
python Semantic_Tag_Gated_Fusion_Residual_MLP_ASL/train_semantic_tag_gated_fusion_residual_mlp_asl.py \
  --epochs 1 \
  --batch-size 128 \
  --max-train-samples 4096 \
  --max-test-samples 512 \
  --output-dir Semantic_Tag_Gated_Fusion_Residual_MLP_ASL/smoke_test
```

Outputs are saved to `Semantic_Tag_Gated_Fusion_Residual_MLP_ASL/runs/`:

- `semantic_tag_gated_fusion_residual_mlp_asl_best.pt`
- `training_history.json`
- `test_metrics.json`
- `config.json`

The metrics JSON includes per-modality average gate weights such as `gate_semantic_tag_mean`, which is useful for checking how much the model relied on the semantic tag modality.
