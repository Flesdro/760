# CLIP End-to-End Fine-tuning

Instead of using pre-extracted frozen CLIP features, this experiment fine-tunes the CLIP
image encoder directly on the NUS-WIDE task. The last N transformer blocks are unfrozen
and updated with a lower learning rate than the classification head (differential LR).

```
Raw image  →  CLIP preprocessing  →  CLIP ViT (partially unfrozen)
                                              ↓
                                   [CLS] feature  (512-dim)
                                              ↓
                                    Dropout  →  Linear(512, 81)
                                              ↓
                                         ASL loss
```

No pre-extraction step needed — images are read from disk during training.

## Run

```bash
/home/jue/miniconda3/envs/ml/bin/python 17_CLIP_Finetune/train_clip_finetune.py
```

Outputs are saved to `17_CLIP_Finetune/runs/`:

- `clip_finetune_best.pt`
- `training_history.json`
- `test_metrics.json`
- `config.json`

## Key design choices

| Setting | Value | Reason |
|---|---|---|
| `--unfreeze-blocks` | 2 | Last 2 transformer blocks + ln_post + proj unfrozen |
| `--lr-backbone` | 1e-5 | 10× lower than head to avoid destroying pretrained features |
| `--lr-head` | 1e-4 | Normal learning rate for the new classifier |
| `--batch-size` | 32 | Conservative for 6 GB VRAM; increase to 64 if no OOM |
| Mixed precision | on | `torch.cuda.amp` halves activation memory |
| `--clip-model` | ViT-B/32 | Default; ViT-B/16 works too; ViT-L/14 may OOM on 6 GB |

## Adjusting for OOM

If you get CUDA out-of-memory, try in this order:

1. Reduce batch size: `--batch-size 16`
2. Unfreeze fewer blocks: `--unfreeze-blocks 1`
3. Disable mixed precision (rarely needed): `--no-amp`

## Unfreezing more blocks

To unfreeze more of the backbone (slower but potentially stronger):

```bash
python 17_CLIP_Finetune/train_clip_finetune.py --unfreeze-blocks 4
```

ViT-B/32 has 12 transformer blocks total.
