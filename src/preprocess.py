# src/preprocess_xgb.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

@dataclass
class PreprocessArtifacts:
    feature_cols: List[str]
    dropped_lowvar_cols: List[str]
    scaler: StandardScaler
    max_rul: int

def add_rul_labels(train_df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
    """
    Train file runs to failure, so failure cycle Tf is max cycle per unit.
    RUL = Tf - t, optionally clipped (often 125). [web:168][web:92]
    """
    df = train_df.copy()
    tf = df.groupby("unit")["cycle"].transform("max")
    df["RUL_raw"] = tf - df["cycle"]
    df["RUL"] = df["RUL_raw"].clip(upper=max_rul)
    return df

def split_by_unit(df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    units = df["unit"].unique()
    tr_u, va_u = train_test_split(units, test_size=val_ratio, random_state=seed)
    tr_df = df[df["unit"].isin(tr_u)].copy()
    va_df = df[df["unit"].isin(va_u)].copy()
    return tr_df, va_df

def variance_filter(
    tr_df: pd.DataFrame,
    candidate_cols: List[str],
    var_threshold: float = 1e-8,
) -> Tuple[List[str], List[str]]:
    """
    Drop near-constant channels using variance threshold. [web:149]
    Compute on training split only.
    """
    v = tr_df[candidate_cols].var(axis=0, ddof=0)
    dropped = v[v <= var_threshold].index.tolist()
    kept = [c for c in candidate_cols if c not in set(dropped)]
    return kept, dropped

def fit_scaler(tr_df: pd.DataFrame, feature_cols: List[str]) -> StandardScaler:
    """
    Fit StandardScaler on TRAIN split only (mean/std stored), then transform others. [web:197]
    """
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(tr_df[feature_cols].values)
    return scaler

def apply_scaler(df: pd.DataFrame, scaler: StandardScaler, feature_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    out[feature_cols] = scaler.transform(out[feature_cols].values)
    return out


def preprocess_fd001_for_xgb(
    train_df_raw: pd.DataFrame,
    test_df_raw: pd.DataFrame,
    *,
    max_rul: int = 336,
    val_ratio: float = 0.2,
    seed: int = 42,
    keep_settings: bool = True,
    var_threshold: float = 1e-8,
):
    # 1) Add RUL to TRAIN immediately after loading.
    train_df = add_rul_labels(train_df_raw, max_rul=max_rul)

    # 2) Split by unit (engine id).
    tr_df, va_df = split_by_unit(train_df, val_ratio=val_ratio, seed=seed)

    # 3) Decide candidate X columns (exclude ids + labels)
    sensor_cols = [c for c in train_df.columns if c.startswith("s")]
    setting_cols = [c for c in train_df.columns if c.startswith("setting")]
    candidate_cols = (setting_cols if keep_settings else []) + sensor_cols

    # 4) Drop near-zero variance columns based on TRAIN split only.
    feature_cols, dropped = variance_filter(tr_df, candidate_cols, var_threshold=var_threshold)

    # 5) Fit scaler on TRAIN split only, transform train/val/test.
    scaler = fit_scaler(tr_df, feature_cols)
    tr_df = apply_scaler(tr_df, scaler, feature_cols)
    va_df = apply_scaler(va_df, scaler, feature_cols)
    test_df = apply_scaler(test_df_raw.copy(), scaler, feature_cols)

    artifacts = PreprocessArtifacts(
        feature_cols=feature_cols,
        dropped_lowvar_cols=dropped,
        scaler=scaler,
        max_rul=max_rul,
    )

    return tr_df, va_df, test_df, artifacts
