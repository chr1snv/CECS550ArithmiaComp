"""Explicit, serializable configuration for training and inference."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    # Data and validation
    id_column: str = "patient_ID"
    target_column: str = "Rhythm"
    n_splits: int = 5
    random_state: int = 42
    batch_size: int = 256
    num_workers: int = 0

    # Model choice is explicit so a registry can add model families later.
    model_name: str = "loop_resnet"
    hidden_dim: int = 40
    loop_layers: int = 4
    max_loops: int = 6
    exit_confidence: float = 0.72
    residual_scale: float = 0.30
    dropout: float = 0.18
    input_noise_std: float = 0.05
    protected_indicator_count: int = 3
    protected_noise_scale: float = 0.15

    # Optimization and imbalance handling
    epochs: int = 280
    patience: int = 40
    min_epochs: int = 100
    learning_rate: float = 0.0025
    weight_decay: float = 0.05
    sampler_power: float = 0.65
    rarity_power: float = 0.68
    rarity_weight_cap: float = 14.0
    l1_penalty: float = 0.0
    val_loss_guard: float = 1.20
    max_overfit_ratio: float = 1.45
    device: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly settings for artifacts and trial history."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "PipelineConfig":
        """Use known settings only; this also supports forward-compatible configs."""
        allowed = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
