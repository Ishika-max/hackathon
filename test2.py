# test.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from src.load_data import load_fd001
from src.preprocess import preprocess_fd001_for_xgb
from src.features import (
    predict_rul_weighted_last_k,
    make_features_from_window,   # uses [last, mean, std, trend]
)

# ----- must match training -----
W = 20
MAX_RUL = 250        # must match train.py max_rul
MODEL_PATH = Path("models/xgb_fd001_2.json")
OUT_PATH = Path("results_fd001.csv")


def health_percent(rul: float, max_rul: float = MAX_RUL) -> float:
    return float(np.clip(rul / max_rul, 0.0, 1.0) * 100.0)


def predict_engine_curve(model, engine_df: pd.DataFrame, feature_cols, *, W: int = 50, step: int = 1):
    """
    Returns arrays: cycles, rul_preds, health%
    Predict at each window end (cycle t) by sliding window inside one engine.
    """
    g = engine_df.sort_values("cycle")
    Xv = g[list(feature_cols)].to_numpy(dtype=np.float32)
    cycles = g["cycle"].to_numpy(dtype=int)

    if len(g) < W:
        raise ValueError(f"Engine has only {len(g)} rows, need at least W={W}")

    out_cycle, out_rul, out_health = [], [], []
    for end in range(W - 1, len(g), step):
        w = Xv[end - W + 1 : end + 1]  # (W,F)
        feats = make_features_from_window(w).reshape(1, -1)
        pred = float(model.predict(feats)[0])

        out_cycle.append(int(cycles[end]))
        out_rul.append(pred)
        out_health.append(health_percent(pred, MAX_RUL))

    return np.array(out_cycle), np.array(out_rul), np.array(out_health)


def plot_health_graph(cycles, health, *, engine_id: int, green=75, yellow=40):
    fig, ax = plt.subplots(figsize=(10, 4))

    # Background bands (RAG-style) using axhspan [web:379]
    ax.axhspan(green, 100, facecolor="green", alpha=0.12)
    ax.axhspan(yellow, green, facecolor="gold", alpha=0.12)
    ax.axhspan(0, yellow, facecolor="red", alpha=0.12)

    ax.plot(cycles, health, linewidth=2)

    ax.axhline(green, color="green", linestyle="--", linewidth=1)
    ax.axhline(yellow, color="red", linestyle="--", linewidth=1)

    ax.set_title(f"Engine {engine_id} Health% vs Cycle (W={W})")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Health (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    # 1) Load raw FD001
    train_df_raw, test_df_raw = load_fd001("data/raw")

    # 2) Preprocess (must match train.py)
    tr_df, va_df, test_df, art = preprocess_fd001_for_xgb(
        train_df_raw, test_df_raw,
        max_rul=MAX_RUL,
        var_threshold=1e-8,
        keep_settings=True
    )

    # 3) Load model [web:335]
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model: {MODEL_PATH.resolve()}\n"
            f"Run train.py and save with model.save_model('models/xgb_fd001.json')."
        )
    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))  # [web:335]

    # 4) One RUL per engine (EMA smoothing over last K windows)
    units, rul_pred = predict_rul_weighted_last_k(
        model=model,
        test_df=test_df,
        feature_cols=art.feature_cols,
        W=W,
        K=10,
        alpha=0.6
    )

    res = pd.DataFrame({"unit": units, "RUL_pred": rul_pred}).sort_values("unit")
    res["Health_percent"] = np.clip(res["RUL_pred"] / MAX_RUL, 0.0, 1.0) * 100.0
    res.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH} with {len(res)} engines.")
    print(res.head())

    # 5) Health graph for a chosen engine (change this id anytime)
    # ENGINE_ID = int(res["unit"].iloc[0])  # or set manually, e.g., 5
    ENGINE_ID = 100
    engine_df = test_df[test_df["unit"] == ENGINE_ID].copy()

    cycles, rul_curve, health_curve = predict_engine_curve(
        model, engine_df, art.feature_cols, W=W, step=1
    )
    plot_health_graph(cycles, health_curve, engine_id=ENGINE_ID)


if __name__ == "__main__":
    main()
