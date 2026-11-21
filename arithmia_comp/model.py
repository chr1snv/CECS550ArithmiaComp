"""Configurable iterative residual network and model-family registry."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


class DeepFeedbackLoopResNet(nn.Module):
    """A compact classifier that refines class probabilities over feedback steps."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        hidden_dim: int = 40,
        loop_layers: int = 4,
        max_loops: int = 6,
        exit_confidence: float = 0.72,
        dropout: float = 0.18,
        input_noise_std: float = 0.05,
        residual_scale: float = 0.30,
        protected_indicator_count: int = 3,
        protected_noise_scale: float = 0.15,
    ) -> None:
        super().__init__()
        if input_dim < 1 or num_classes < 2 or hidden_dim < 1 or loop_layers < 1 or max_loops < 1:
            raise ValueError("Model dimensions, layers, and loop count must be positive")
        self.num_classes = num_classes
        self.max_loops = max_loops
        self.exit_confidence = exit_confidence
        self.input_noise_std = input_noise_std
        self.residual_scale = residual_scale
        self.protected_indicator_count = protected_indicator_count
        self.protected_noise_scale = protected_noise_scale
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Dropout(dropout * 0.5) if dropout else nn.Identity(),
        )
        layers: list[nn.Module] = []
        for index in range(loop_layers):
            layers.extend((
                nn.Linear(hidden_dim * 2 + num_classes if index == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            ))
            if index < loop_layers - 1:
                layers.append(nn.SiLU())
                if dropout:
                    layers.append(nn.Dropout(dropout))
        self.loop_cell = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.training and self.input_noise_std > 0:
            noise = torch.randn_like(features) * self.input_noise_std
            protected = min(self.protected_indicator_count, features.shape[1])
            if protected:
                noise[:, -protected:] *= self.protected_noise_scale
            features = features + noise

        projected = self.input_projection(features)
        hidden = projected
        previous = torch.full(
            (features.shape[0], self.num_classes), 1 / self.num_classes,
            device=features.device, dtype=features.dtype,
        )
        active = torch.ones(features.shape[0], dtype=torch.bool, device=features.device)
        final_logits = torch.zeros_like(previous)
        step_logits: list[torch.Tensor] = []
        for _ in range(self.max_loops):
            loop_input = torch.cat((projected, hidden, previous), dim=-1)
            hidden = F.silu(hidden + self.residual_scale * self.loop_cell(loop_input))
            logits = self.head(hidden)
            step_logits.append(logits)
            # Keep each row's last logits after it reaches the confidence threshold.
            final_logits = torch.where(active[:, None], logits, final_logits)
            previous = F.softmax(logits, dim=-1)
            active = active & (previous.max(dim=-1).values < self.exit_confidence)
            if not self.training and not active.any():
                break
        return final_logits, step_logits


ModelFactory = Callable[..., nn.Module]
MODEL_REGISTRY: dict[str, ModelFactory] = {"loop_resnet": DeepFeedbackLoopResNet}


def build_model(name: str, input_dim: int, num_classes: int, **parameters: object) -> nn.Module:
    """Create a registered model; add future model families in one place."""
    try:
        factory = MODEL_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"Unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}") from error
    return factory(input_dim, num_classes, **parameters)
