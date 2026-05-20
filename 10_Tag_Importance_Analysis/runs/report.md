# Tag Importance Analysis

This experiment explains why `aligned_tag_feature.npy` is important, beyond the ablation mAP drop.

## Main Evidence

- The model target has 81 labels, but `Train_Tags1k.dat` provides a 1000-dimensional original user-tag vocabulary.
- 75 / 81 target concept names appear directly in the 1000-tag vocabulary.
- Each matched image has 6.13 active 1k-tags on average.
- The mean best tag-label correlation is 0.4876.
- A single best tag dimension per class gives mean AP 0.3081.
- After shuffling tag rows, mean best tag-label correlation drops to 0.0124, showing that image-tag alignment matters.
- The best individual handcrafted visual dimension has mean absolute correlation 0.1148, lower than the aligned tags.

## Why 1000 Tags Instead Of 81

The 81 concepts are the prediction targets. The 1000 tags are not another target label set; they are input metadata from the original NUS-WIDE tag vocabulary. They include direct target words plus related context words, attributes, synonyms, and scene cues. For example, a label such as `beach` can be supported by tags like `water`, `sea`, `sand`, `sunset`, or `vacation`. Reducing the input to only 81 exact concept words would discard these extra semantic cues.

## Strongest Examples

| label | best tag | corr | pos rate | neg rate | top 5 related tags |
|---|---:|---:|---:|---:|---|
| dog | dog | 0.8543 | 0.9008 | 0.0025 | dog, puppy, dogs, pups, pet |
| railroad | railroad | 0.8502 | 0.8815 | 0.0008 | railroad, train, locomotive, rail, railway |
| cat | cat | 0.8254 | 0.9414 | 0.0041 | cat, kitten, kitty, cats, feline |
| horses | horses | 0.8123 | 0.7243 | 0.0006 | horses, horse, foals, argentina, brasil |
| bear | bear | 0.8010 | 0.8563 | 0.0017 | bear, bears, polar, zoo, giant |
| cow | cow | 0.7751 | 0.8244 | 0.0012 | cow, cows, horns, farm, moose |
| snow | snow | 0.7712 | 0.7745 | 0.0061 | snow, winter, ice, skiing, cold |
| statue | statue | 0.7686 | 0.9486 | 0.0014 | statue, sculpture, buddha, monument, cemetery |

## Generated Files

- `summary.json`: aggregate statistics used in this report.
- `label_summary.csv`: one row per 81 target label.
- `top_tags_by_label.csv`: top correlated 1k-tags for every target label.
- `concept_tag_overlap.csv`: whether each target label name appears in the 1000-tag vocabulary.
- `top_tag_vocabulary_used.csv`: 1k-tags that repeatedly appear as strong predictors.
