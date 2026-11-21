"""Persist compact Groq tuning history and resume from its best trial.

This module is a small companion to the notebook's LLM-controlled search.
It deliberately contains no model-training code: the notebook passes trial
metrics and fold summaries here, then supplies the returned context to Groq.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

_HISTORY_VERSION = 1


def load_search_history(path: str | Path) -> dict[str, Any]:
    """Load persisted trials, or return an empty history if none exists."""
    history_path = Path(path)
    if not history_path.exists():
        return {"version": _HISTORY_VERSION, "trials": []}

    with history_path.open(encoding="utf-8") as stream:
        history = json.load(stream)
    if not isinstance(history, dict) or not isinstance(history.get("trials"), list):
        raise ValueError(f"Invalid search history file: {history_path}")
    return history


def _atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    """Replace a JSON file atomically so an interrupted run keeps prior history."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def best_trial(history: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the highest-scoring recorded trial with usable parameters."""
    valid = [
        trial for trial in history.get("trials", [])
        if isinstance(trial, dict)
        and isinstance(trial.get("params"), dict)
        and isinstance(trial.get("score"), (int, float))
    ]
    return max(valid, key=lambda trial: float(trial["score"]), default=None)


def resume_parameters(
    history: Mapping[str, Any], initial_params: Mapping[str, Any]
) -> tuple[dict[str, Any], float | None]:
    """Resume at the best recorded parameters, falling back to the seed config."""
    best = best_trial(history)
    if best is None:
        return dict(initial_params), None
    # Keep newly introduced defaults while restoring known values from the best run.
    return {**initial_params, **best["params"]}, float(best["score"])


def record_trial(
    path: str | Path,
    *,
    params: Mapping[str, Any],
    score: float,
    fold_metrics: Sequence[Mapping[str, Any]] | None = None,
    per_class_metrics: Mapping[str, Any] | None = None,
    hypothesis: str = "",
    reason: str = "",
) -> dict[str, Any]:
    """Append one concise trial record and persist it immediately.

    ``fold_metrics`` should contain one small dictionary per held-out fold,
    for example ``{"fold": 1, "macro_f1": 0.41, "pr_auc": 0.40}``.
    """
    if not isinstance(score, (int, float)) or not float("-inf") < float(score) < float("inf"):
        raise ValueError("score must be a finite number")

    history = load_search_history(path)
    trial = {
        "trial": len(history["trials"]) + 1,
        "params": dict(params),
        "score": float(score),
        "hypothesis": hypothesis.strip(),
        "reason": reason.strip(),
        "fold_metrics": [dict(metric) for metric in (fold_metrics or [])],
        "per_class_metrics": dict(per_class_metrics or {}),
    }
    history["version"] = _HISTORY_VERSION
    history["trials"].append(trial)
    history["best_trial"] = best_trial(history)["trial"]
    _atomic_write_json(path, history)
    return trial


def compact_history_for_llm(
    history: Mapping[str, Any], *, recent: int = 4, max_chars: int = 6000
) -> str:
    """Format the best, recent, and fold-level results for a bounded LLM prompt."""
    trials = history.get("trials", [])
    best = best_trial(history)
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for trial in ([best] if best else []) + list(trials[-recent:]):
        if trial is None:
            continue
        number = int(trial.get("trial", 0))
        if number not in seen:
            selected.append(trial)
            seen.add(number)

    lines = ["Previous hyperparameter trials (OOF score; fold metrics are validation results):"]
    for trial in selected:
        lines.append(
            f"Trial {trial.get('trial', '?')}: score={float(trial['score']):.4f}; "
            f"params={json.dumps(trial.get('params', {}), sort_keys=True, separators=(',', ':'))}"
        )
        if trial.get("hypothesis"):
            lines.append(f"  hypothesis: {trial['hypothesis']}")
        if trial.get("reason"):
            lines.append(f"  result: {trial['reason']}")
        folds = trial.get("fold_metrics", [])
        if folds:
            short_folds = [
                {key: metric[key] for key in ("fold", "macro_f1", "pr_auc", "val_loss") if key in metric}
                for metric in folds
            ]
            lines.append(f"  folds: {json.dumps(short_folds, separators=(',', ':'))}")
        if trial.get("per_class_metrics"):
            lines.append(
                "  per-class: "
                + json.dumps(trial["per_class_metrics"], sort_keys=True, separators=(",", ":"))
            )

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 20] + "\n...[history trimmed]"
    return text


def bounded_parameters(
    proposed: Mapping[str, Any],
    fallback: Mapping[str, Any],
    bounds: Mapping[str, tuple[float, float, type]],
) -> dict[str, Any]:
    """Clamp known numeric proposals to bounds; ignore malformed/unknown values."""
    clean = dict(fallback)
    for name, (low, high, value_type) in bounds.items():
        if name not in proposed:
            continue
        try:
            value = float(proposed[name])
            if value_type is int:
                value = int(round(value))
            clean[name] = value_type(min(max(value, low), high))
        except (TypeError, ValueError, OverflowError):
            continue
    return clean
