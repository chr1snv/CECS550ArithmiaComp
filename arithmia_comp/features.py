"""Deterministic tabular feature preparation for the ECG rhythm dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

RARE_INDICATORS = ("ind_class3_av nrt", "ind_class4_avrt", "ind_class6_sa awr")
# Keep source-compatible feature names used by saved artifacts and dashboards.
RARE_INDICATORS = ("ind_class3_avnrt", "ind_class4_avrt", "ind_class6_sa awr")
RARE_INDICATORS = ("ind_class3_avnrt", "ind_class4_avrt", "ind_class6_saawr")
_DROPPED_DERIVED = ("QRSCount", "QTCorrected", "QOnset", "QOffset", "TOffset", "TAxis", "RAxis")


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add compact rhythm indicators; leave labels and identifiers to the caller."""
    df = frame.copy()
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    required = ("VentricularRate", "AtrialRate", "QRSDuration", "PatientAge")
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"Missing required ECG columns: {missing}")

    vr = df["VentricularRate"].clip(20, 300)
    ar = df["AtrialRate"].clip(20, 450)
    qrs = df["QRSDuration"].clip(40, 250)
    df["av_log_ratio"] = np.log(ar / vr).clip(-0.8, 1.8)
    df["qrs_span"] = (df["QOffset"] - df["QOnset"]).clip(lower=1) if {"QOffset", "QOnset"} <= set(df) else qrs
    if {"TOffset", "QOnset"} <= set(df):
        df["qt_span"] = (df["TOffset"] - df["QOnset"]).clip(lower=1)
    if "QTInterval" in df:
        df["qrs_qt_ratio"] = qrs / df["QTInterval"].clip(200, 700)
    df["age_rate_prod"] = (df["PatientAge"].clip(1, 110) / 100) * (vr / 100)

    # Circular axes are represented by sine/cosine rather than raw degrees.
    for name in ("TAxis", "RAxis"):
        if name in df:
            radians = np.deg2rad(df[name])
            prefix = name[0].lower()
            df[f"{prefix}_sin"] = np.sin(radians)
            df[f"{prefix}_cos"] = np.cos(radians)
    if "TAxis" in df and "RAxis" in df:
        df["qrs_t_align"] = np.cos(np.deg2rad(df["RAxis"] - df["TAxis"]))

    if {"TOffset", "QOffset"} <= set(df):
        st_span = (df["TOffset"] - df["QOffset"]).clip(lower=1)
        df["st_t_span"] = st_span
    else:
        st_span = pd.Series(1.0, index=df.index)

    # Smooth bounded indicators highlight candidate rare rhythms; these are
    # engineered model inputs, not clinical diagnoses.
    qrs_narrow = 1 / (1 + np.exp((qrs - 110) / 8))
    regular = np.exp(-0.5 * ((df["QRSCount"] - (vr / 6) * 10) / 1.2) ** 2) if "QRSCount" in df else 1.0
    av_lock = np.exp(-0.5 * (np.log(ar / vr) / 0.15) ** 2)
    class3 = av_lock * qrs_narrow * regular * np.exp(-0.5 * ((vr - 165) / 16) ** 2)
    class4 = av_lock * qrs_narrow * regular / (1 + np.exp(-(vr - 192) / 10))
    class4 *= 1 / (1 + np.exp((st_span - 88) / 8))
    age_factor = 1 / (1 + np.exp((df["PatientAge"] - 34) / 8))
    rate_factor = np.exp(-0.5 * ((vr - 70) / 14) ** 2)
    upright_t = (1 / (1 + np.exp(-(df["t_sin"] - 0.5) / 0.2))) if "t_sin" in df else 0.5
    class6 = av_lock * qrs_narrow * age_factor * rate_factor * upright_t
    df["ind_class3_avnrt"] = np.sqrt(np.clip(class3, 0, 1))
    df["ind_class4_avrt"] = np.sqrt(np.clip(class4, 0, 1))
    df["ind_class6_saawr"] = np.sqrt(np.clip(class6, 0, 1))
    return df.drop(columns=[column for column in _DROPPED_DERIVED if column in df])


def prepare_training_features(
    frame: pd.DataFrame, *, id_column: str, target_column: str
) -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    """Separate target/ID, engineer features, and learn train-only imputations."""
    if target_column not in frame or id_column not in frame:
        raise ValueError(f"Expected columns {id_column!r} and {target_column!r}")
    labels = frame[target_column].astype(str)
    raw = frame.drop(columns=[id_column, target_column])
    features = engineer_features(raw)
    medians = features.median(numeric_only=True).fillna(0.0).to_dict()
    return features.fillna(medians), labels, medians


def prepare_inference_features(
    frame: pd.DataFrame,
    *,
    id_column: str,
    feature_names: list[str],
    medians: dict[str, float],
) -> tuple[pd.Series, pd.DataFrame]:
    """Apply saved feature order and train-fold imputation to an unlabeled table."""
    if id_column not in frame:
        raise ValueError(f"Missing identifier column {id_column!r}")
    ids = frame[id_column].copy()
    features = engineer_features(frame.drop(columns=[id_column], errors="ignore"))
    features = features.reindex(columns=feature_names)
    return ids, features.fillna(medians).fillna(0.0)
