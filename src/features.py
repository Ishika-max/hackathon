# src/features_xgb.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List

def make_window_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    window: int = 30,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each unit:
      - take sliding window of length W ending at cycle t
      - features = concat(last, mean, std, last-first) over the window
      - label y = RUL at cycle t (last row of the window)
    """
    X_list, y_list = [], []

    for _, g in df.groupby("unit", sort=False):
        g = g.sort_values("cycle")
        Xv = g[feature_cols].to_numpy(dtype=np.float32)
        yv = g["RUL"].to_numpy(dtype=np.float32)

        n = len(g)
        for end in range(window - 1, n, step):
            start = end - window + 1
            w = Xv[start:end + 1]  # (W,F)

            last  = w[-1]
            mean  = w.mean(axis=0)
            std   = w.std(axis=0)
            trend = w[-1] - w[0]

            feats = np.concatenate([last, mean, std, trend], axis=0)  # (4F,)
            X_list.append(feats)
            y_list.append(yv[end])

    return np.stack(X_list), np.array(y_list, dtype=np.float32)

# src/features.py
import numpy as np
import pandas as pd

def make_last_window_features(test_df: pd.DataFrame, feature_cols, window: int = 50):
    """
    Returns:
      X: (num_engines, 4*F) features using ONLY last window per engine
      units: (num_engines,) engine ids
    Uses same feature recipe as training: [last, mean, std, trend].
    """
    X_list, units = [], []

    for unit, g in test_df.groupby("unit", sort=False):
        g = g.sort_values("cycle")

        if len(g) < window:
            # simplest: skip short units (rare); alternatively pad with first row
            continue

        w = g[feature_cols].to_numpy(dtype=np.float32)[-window:]  # last W rows

        last = w[-1]
        mean = w.mean(axis=0)
        std  = w.std(axis=0)
        trend = w[-1] - w[0]

        feats = np.concatenate([last, mean, std, trend], axis=0)
        X_list.append(feats)
        units.append(int(unit))

    return np.stack(X_list), np.array(units, dtype=int)

import numpy as np
import pandas as pd

def make_features_from_window(w: np.ndarray) -> np.ndarray:
    # w: (W, F) already scaled
    last = w[-1]
    mean = w.mean(axis=0)
    std  = w.std(axis=0)
    trend = w[-1] - w[0]
    return np.concatenate([last, mean, std, trend], axis=0)

def predict_rul_weighted_last_k(
    model,
    test_df: pd.DataFrame,
    feature_cols,
    W: int = 50,
    K: int = 5,
    alpha: float = 0.6,   # higher = more weight to most recent [web:349]
):
    """
    For each engine:
      - compute predictions for the last K window positions
      - apply EMA smoothing to get one final RUL per engine [web:349]
    """
    out_units, out_rul = [], []

    for unit, g in test_df.groupby("unit", sort=False):
        g = g.sort_values("cycle")
        Xv = g[feature_cols].to_numpy(dtype=np.float32)

        if len(Xv) < W:
            continue

        # last K window end indices
        last_end = len(Xv) - 1
        ends = list(range(last_end, max(W - 1, last_end - K + 1) - 1, -1))
        ends = ends[::-1]  # oldest -> newest

        preds = []
        for end in ends:
            w = Xv[end - W + 1 : end + 1]
            preds.append(model.predict(make_features_from_window(w).reshape(1, -1))[0])

        # EMA smoothing over preds
        sm = preds[0]
        for p in preds[1:]:
            sm = alpha * p + (1 - alpha) * sm

        out_units.append(int(unit))
        out_rul.append(float(sm))

    return np.array(out_units), np.array(out_rul)
