replace:
```
COMMON_CONFIG = dict(
    batch_size=32,               # Number of samples sent to the model per training step.
    epochs=35,                    # Number of full passes over the training set.
    lr=1e-4,                      # Learning rate, controls the optimizer update size.
    weight_decay=1e-4,            # AdamW regularization strength to reduce overfitting.
    hidden_dims=[1024, 512, 256], # Shared MLP hidden layers used by all classifiers.
    dropout=0.3,                  # Dropout ratio applied after each hidden activation.
    activation="gelu",           # Activation function: relu / gelu / silu.
    loss="focal",
    focal_alpha=0.25,
    focal_gamma=2.0,
    threshold=0.5,                # Probability threshold for multi-label prediction.
    top_k=5,                      # Reserved for top-k evaluation/reporting consistency.
    val_ratio=0.2,                # Fraction of training data used for validation.
    random_seed=42,               # Random seed for reproducible splitting/training.
    standardize=False,            # Whether to fit StandardScaler on train features.
)
```

```
@dataclass
class FusionConfig:
    train_label_path: Path
    test_label_path: Path
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    hidden_dims: list[int]
    dropout: float
    activation: str
    loss: str
    focal_alpha: float
    focal_gamma: float
    threshold: float
    top_k: int
    val_ratio: float
    random_seed: int
    standardize: bool
    embed_dim: int = 128
    lambda_nce: float = 0.1
    temperature: float = 0.1
    device: str = "cuda"



config = FusionConfig(
    **COMMON_CONFIG,
    train_label_path=Path("database_labels_81_big.npy"),
    test_label_path=Path("database_labels_81_test.npy"),
)
```

add:
```
class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt).pow(self.gamma)

        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
```
replace:
```
def build_criterion(config: FusionConfig) -> nn.Module:
    if config.loss == "bce":
        return nn.BCEWithLogitsLoss()
    if config.loss == "focal":
        return FocalLossWithLogits(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma,
        )
    raise ValueError(f"Unsupported loss for this fusion baseline: {config.loss}")

```