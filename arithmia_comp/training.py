"""Cross-validation training with fold-local preprocessing and saved artifacts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .config import PipelineConfig
from .features import RARE_INDICATORS
from .model import build_model

SyntheticGenerator = Callable[[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]


def _preprocessor(features: pd.DataFrame, indicator_count: int) -> ColumnTransformer:
    indicators = [name for name in RARE_INDICATORS if name in features.columns]
    passthrough = [
        name for name in features.columns
        if name in indicators or features[name].dropna().isin([0, 1]).all()
    ]
    scale = [name for name in features.columns if name not in passthrough]
    return ColumnTransformer(
        [("scale", RobustScaler(), scale), ("pass", "passthrough", passthrough)],
        remainder="drop", verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def _loss(step_logits: list[torch.Tensor], targets: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
    """Combine class-weighted cross entropy, expected confusion cost, and soft F1."""
    one_hot = F.one_hot(targets, num_classes=costs.shape[0]).float()
    losses = []
    for logits in step_logits:
        probabilities = logits.softmax(dim=-1)
        ce = F.cross_entropy(logits, targets, weight=costs.diag().clamp_min(1e-3))
        confusion_cost = (probabilities * costs[targets]).sum(dim=-1).mean()
        tp = (probabilities * one_hot).sum(dim=0)
        fp = (probabilities * (1 - one_hot)).sum(dim=0)
        fn = ((1 - probabilities) * one_hot).sum(dim=0)
        soft_f1 = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
        losses.append(0.5 * ce + 0.25 * confusion_cost + 0.25 * (1 - soft_f1.mean()))
    return torch.stack(losses).mean()


def _predict(model: nn.Module, values: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.as_tensor(values, dtype=torch.float32, device=device))
        return logits.softmax(dim=-1).cpu().numpy()


def _macro_ap(y_true: np.ndarray, probabilities: np.ndarray, n_classes: int) -> float:
    truth = np.eye(n_classes, dtype=np.float32)[y_true]
    scores = [
        average_precision_score(truth[:, index], probabilities[:, index])
        for index in range(n_classes) if truth[:, index].sum() > 0
    ]
    return float(np.mean(scores)) if scores else 0.0


def _fit_fold(
    features: pd.DataFrame,
    labels: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    config: PipelineConfig,
    *,
    synthetic_generator: SyntheticGenerator | None = None,
) -> tuple[dict[str, Any], np.ndarray, dict[str, float]]:
    device = config.resolved_device()
    train_x, train_y = features.iloc[train_indices].copy(), labels[train_indices].copy()
    valid_x, valid_y = features.iloc[valid_indices], labels[valid_indices]

    # Synthetic generation is deliberately train-fold-only to avoid validation leakage.
    if synthetic_generator is not None:
        train_x, train_y_series = synthetic_generator(train_x, pd.Series(train_y, index=train_x.index))
        train_y = train_y_series.to_numpy(dtype=np.int64)

    preprocessor = _preprocessor(train_x, config.protected_indicator_count)
    train_values = preprocessor.fit_transform(train_x).to_numpy(dtype=np.float32)
    valid_values = preprocessor.transform(valid_x).to_numpy(dtype=np.float32)
    train_tensor = torch.as_tensor(train_values, dtype=torch.float32)
    label_tensor = torch.as_tensor(train_y, dtype=torch.long)
    counts = np.bincount(train_y, minlength=len(np.unique(labels)))
    row_weights = np.power(1 / np.maximum(counts[train_y], 1), config.sampler_power)
    sampler = WeightedRandomSampler(torch.as_tensor(row_weights, dtype=torch.double), len(row_weights), replacement=True)
    loader = DataLoader(
        TensorDataset(train_tensor, label_tensor), batch_size=config.batch_size,
        sampler=sampler, drop_last=False, num_workers=config.num_workers,
    )

    class_weights = np.clip(
        (counts.max() / np.maximum(counts, 1)) ** config.rarity_power,
        1.0, config.rarity_weight_cap,
    )
    costs_array = np.tile(class_weights, (len(class_weights), 1)).astype(np.float32)
    np.fill_diagonal(costs_array, 0.0)
    costs = torch.as_tensor(costs_array, dtype=torch.float32, device=device)
    model = build_model(
        config.model_name, train_values.shape[1], len(np.unique(labels)),
        hidden_dim=config.hidden_dim, loop_layers=config.loop_layers,
        max_loops=config.max_loops, exit_confidence=config.exit_confidence,
        dropout=config.dropout, input_noise_std=config.input_noise_std,
        residual_scale=config.residual_scale,
        protected_indicator_count=config.protected_indicator_count,
        protected_noise_scale=config.protected_noise_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_state: dict[str, torch.Tensor] | None = None
    best_f1, best_loss, stale = -1.0, float("inf"), 0
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, step_logits = model(batch_x)
            loss = _loss(step_logits, batch_y, costs)
            if config.l1_penalty:
                loss = loss + config.l1_penalty * sum(p.abs().sum() for p in model.parameters())
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())

        valid_probs = _predict(model, valid_values, device)
        valid_predictions = valid_probs.argmax(axis=1)
        fold_f1 = float(f1_score(valid_y, valid_predictions, average="macro", zero_division=0))
        fold_loss = total_loss / max(len(loader), 1)
        if fold_f1 > best_f1 and fold_loss <= best_loss * config.val_loss_guard:
            best_f1, best_loss, stale = fold_f1, fold_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch + 1 >= config.min_epochs and stale >= config.patience:
            break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    probabilities = _predict(model, valid_values, device)
    metrics = {
        "macro_f1": float(f1_score(valid_y, probabilities.argmax(axis=1), average="macro", zero_division=0)),
        "macro_pr_auc": _macro_ap(valid_y, probabilities, len(np.unique(labels))),
        "best_epoch_f1": float(best_f1),
    }
    fold = {
        "state_dict": best_state,
        "preprocessor": preprocessor,
        "feature_names": list(features.columns),
        "medians": features.median(numeric_only=True).fillna(0).to_dict(),
    }
    return fold, probabilities, metrics


def fit_ensemble(
    frame: pd.DataFrame,
    config: PipelineConfig,
    *,
    synthetic_generator: SyntheticGenerator | None = None,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """Fit configurable stratified folds; return an inference bundle and fold metrics."""
    from .features import prepare_training_features

    features, raw_labels, _ = prepare_training_features(
        frame, id_column=config.id_column, target_column=config.target_column
    )
    encoder = LabelEncoder()
    labels = encoder.fit_transform(raw_labels)
    counts = np.bincount(labels)
    if config.n_splits < 2 or counts.min() < config.n_splits:
        raise ValueError(f"Each class needs at least {config.n_splits} rows for stratified CV")

    splitter = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_state)
    folds, metrics = [], []
    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(features, labels), start=1):
        fold, _, fold_metrics = _fit_fold(
            features, labels, train_idx, valid_idx, config,
            synthetic_generator=synthetic_generator,
        )
        fold_metrics["fold"] = fold_number
        folds.append(fold)
        metrics.append(fold_metrics)

    bundle = {
        "config": config.to_dict(), "classes": encoder.classes_.tolist(),
        "feature_names": list(features.columns), "folds": folds,
        "metrics": metrics, "multipliers": np.ones(len(encoder.classes_), dtype=np.float32),
    }
    return bundle, metrics


def save_bundle(bundle: dict[str, Any], path: str | Path) -> None:
    """Persist weights and fold preprocessors for later prediction."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, target)


def load_bundle(path: str | Path, *, device: str = "cpu") -> dict[str, Any]:
    """Load a trusted local training artifact."""
    return torch.load(path, map_location=device, weights_only=False)


def predict_probabilities(bundle: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    """Average fold probabilities and apply saved class multipliers."""
    config = PipelineConfig.from_dict(bundle["config"])
    device = config.resolved_device()
    probabilities = []
    for fold in bundle["folds"]:
        ordered = features.reindex(columns=fold["feature_names"]).fillna(fold["medians"]).fillna(0.0)
        values = fold["preprocessor"].transform(ordered).to_numpy(dtype=np.float32)
        model = build_model(
            config.model_name, values.shape[1], len(bundle["classes"]),
            hidden_dim=config.hidden_dim, loop_layers=config.loop_layers,
            max_loops=config.max_loops, exit_confidence=config.exit_confidence,
            dropout=config.dropout, input_noise_std=config.input_noise_std,
            residual_scale=config.residual_scale,
            protected_indicator_count=config.protected_indicator_count,
            protected_noise_scale=config.protected_noise_scale,
        ).to(device)
        model.load_state_dict(fold["state_dict"])
        probabilities.append(_predict(model, values, device))
    mean_probs = np.mean(probabilities, axis=0)
    return mean_probs * np.asarray(bundle.get("multipliers", 1.0))
