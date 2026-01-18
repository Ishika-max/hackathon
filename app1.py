# app.py
# Streamlit dashboard using your trained XGBoost model + your preprocessing/features code.
# Run: streamlit run app.py
#
# Assumptions (same as your repo):
# - data/raw/train_FD001.txt and data/raw/test_FD001.txt exist
# - models/xgb_fd001.json exists (saved from train.py)
# - src/load_data.py has load_fd001
# - src/preprocess.py has preprocess_fd001_for_xgb (returns scaled test_df + art.feature_cols)
# - src/features.py has make_features_from_window

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

from src.load_data import load_fd001
from src.preprocess import preprocess_fd001_for_xgb
from src.features import make_features_from_window


W = 20
MAX_RUL = 250  

MODEL_PATH = Path("models/xgb_fd001_2.json")

GREEN_TH = 75
YELLOW_TH = 40



def load_model() -> XGBRegressor:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH.resolve()}\n"
            f"Run train.py and save with model.save_model('models/xgb_fd001.json')."
        )
    m = XGBRegressor()
    m.load_model(str(MODEL_PATH))
    return m




@st.cache_data
def load_preprocessed_test():
    train_df_raw, test_df_raw = load_fd001("data/raw")
    _, _, test_df, art = preprocess_fd001_for_xgb(
        train_df_raw,
        test_df_raw,
        max_rul=MAX_RUL,
        var_threshold=1e-8,
        keep_settings=True,
    )
    # make UI column name consistent with your old app
    test_df = test_df.rename(columns={"unit": "engine_id"})
    return test_df, art


def health_percent(rul: float) -> float:
    return float(np.clip(rul / MAX_RUL, 0.0, 1.0) * 100.0)


def status_from_health(h: float) -> str:
    if h >= GREEN_TH:
        return "HEALTHY"
    if h >= YELLOW_TH:
        return "WARNING"
    return "CRITICAL"


def predict_rul_curve(engine_df: pd.DataFrame, feature_cols, model: XGBRegressor, window: int = W) -> pd.DataFrame:
    """
    Predict RUL for many cycles by sliding a W-length window.
    Returns engine_df with predicted_rul (NaN for first W-1 cycles).
    """
    df = engine_df.sort_values("cycle").copy()
    Xv = df[list(feature_cols)].to_numpy(dtype=np.float32)

    pred = np.full(len(df), np.nan, dtype=np.float32)
    for end in range(window - 1, len(df)):
        w = Xv[end - window + 1 : end + 1]
        feats = make_features_from_window(w).reshape(1, -1)
        pred[end] = float(model.predict(feats)[0])

    df["predicted_rul"] = pred
    return df


def plot_health(engine_df_pred: pd.DataFrame, engine_id: int):
    # drop NaNs (first W-1 cycles)
    g = engine_df_pred.dropna(subset=["predicted_rul"]).copy()

    g["health"] = g["predicted_rul"].apply(health_percent)

    # enforce monotonic non-increasing health for nicer demo graph
    g["health_mono"] = g["health"].cummin()

    fig, ax = plt.subplots(figsize=(10, 4))

    # Background bands
    ax.axhspan(GREEN_TH, 100, color="green", alpha=0.12)
    ax.axhspan(YELLOW_TH, GREEN_TH, color="gold", alpha=0.12)
    ax.axhspan(0, YELLOW_TH, color="red", alpha=0.12)

    ax.plot(g["cycle"], g["health_mono"], linewidth=2)
    ax.axhline(GREEN_TH, color="green", linestyle="--", linewidth=1)
    ax.axhline(YELLOW_TH, color="red", linestyle="--", linewidth=1)

    ax.set_title(f"Engine {engine_id}: Health% vs Cycle (W={W})")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Health (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    st.pyplot(fig, clear_figure=True)



st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("Predictive Maintenance Dashboard")

model = load_model()
df, art = load_preprocessed_test()

st.sidebar.header("Engine Selection")
engine_id = st.sidebar.selectbox("Select Engine ID", sorted(df["engine_id"].unique()))

engine_df = df[df["engine_id"] == engine_id].sort_values("cycle")

# Predict per-cycle RUL curve
engine_df_pred = predict_rul_curve(engine_df, art.feature_cols, model, window=W)
engine_df_valid = engine_df_pred.dropna(subset=["predicted_rul"]).copy()

# Final RUL: weighted average over last few predictions (optional)
K = 5
weights = np.array([1, 2, 3, 4, 5], dtype=np.float32)
weights = weights / weights.sum()

tail = engine_df_valid["predicted_rul"].tail(K).to_numpy(dtype=np.float32)
if len(tail) < K:
    final_rul = float(np.mean(tail)) if len(tail) > 0 else float("nan")
else:
    final_rul = float(np.sum(tail * weights))

health = health_percent(final_rul) if np.isfinite(final_rul) else 0.0
status = status_from_health(health)

# KPIs
st.subheader(f"Engine ID: {engine_id}")
c1, c2, c3 = st.columns(3)
c1.metric("Final RUL", f"{int(final_rul) if np.isfinite(final_rul) else 0} cycles")
c2.metric("Health", f"{int(health)}%")
c3.metric("Status", status)

# Health gauge
st.subheader("Overall Health")
st.progress(float(health) / 100.0)

# Recommendation
st.subheader("Maintenance Recommendation")
if status == "HEALTHY":
    st.success("No maintenance required. Engine operating normally.")
elif status == "WARNING":
    st.warning("Plan maintenance soon. Monitor engine closely.")
else:
    st.error("Immediate maintenance required! High risk of failure.")

# Health trend plot (matplotlib)
st.subheader("Health Trend Over Time")
plot_health(engine_df_pred, engine_id)

# Key sensor trends (use your existing)
st.subheader("Key Sensor Trends")
for s in ["s2", "s3", "s4"]:
    if s in engine_df.columns:
        st.markdown(f"**Sensor {s.upper()}**")
        st.line_chart(engine_df.set_index("cycle")[s])
