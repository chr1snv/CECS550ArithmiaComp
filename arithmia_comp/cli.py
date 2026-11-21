"""Train, save, and predict from the functional pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .features import prepare_inference_features
from .training import fit_ensemble, load_bundle, predict_probabilities, save_bundle


def train_from_csv(
    train_csv: str | Path,
    artifact_path: str | Path,
    config: PipelineConfig,
) -> tuple[dict, list[dict[str, float]]]:
    """Train folds from a CSV, persist the bundle, and return fold metrics."""
    bundle, metrics = fit_ensemble(pd.read_csv(train_csv), config)
    save_bundle(bundle, artifact_path)
    return bundle, metrics


def predict_csv(
    test_csv: str | Path,
    artifact_path: str | Path,
    submission_path: str | Path,
) -> pd.DataFrame:
    """Write a two-column identifier/prediction CSV using a saved bundle."""
    bundle = load_bundle(artifact_path)
    config = PipelineConfig.from_dict(bundle["config"])
    test = pd.read_csv(test_csv)
    ids, features = prepare_inference_features(
        test, id_column=config.id_column,
        feature_names=bundle["feature_names"],
        medians=bundle["folds"][0]["medians"],
    )
    probabilities = predict_probabilities(bundle, features)
    class_indices = probabilities.argmax(axis=1)
    predictions = [bundle["classes"][index] for index in class_indices]
    output = pd.DataFrame({config.id_column: ids, config.target_column: predictions})
    target = Path(submission_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(target, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run the ECG rhythm classifier")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="fit stratified folds and save an artifact")
    train.add_argument("train_csv")
    train.add_argument("artifact")
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--epochs", type=int, default=280)
    train.add_argument("--hidden-dim", type=int, default=40)
    train.add_argument("--model", default="loop_resnet")
    infer = commands.add_parser("predict", help="create predictions from a saved artifact")
    infer.add_argument("test_csv")
    infer.add_argument("artifact")
    infer.add_argument("submission_csv")
    args = parser.parse_args()

    if args.command == "train":
        config = PipelineConfig(
            n_splits=args.folds, epochs=args.epochs,
            hidden_dim=args.hidden_dim, model_name=args.model,
        )
        _, metrics = train_from_csv(args.train_csv, args.artifact, config)
        print(pd.DataFrame(metrics).to_string(index=False))
    else:
        result = predict_csv(args.test_csv, args.artifact, args.submission_csv)
        print(f"Wrote {len(result)} predictions to {args.submission_csv}")


if __name__ == "__main__":
    main()
