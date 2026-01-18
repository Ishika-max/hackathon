from src.load_data import load_fd001
from src.preprocess import preprocess_fd001_for_xgb
from torch.utils.data import DataLoader



train_df_raw, test_df_raw = load_fd001("data/raw")

tr_df, va_df, test_df, art = preprocess_fd001_for_xgb(
    train_df_raw, test_df_raw,
    max_rul=250,
    var_threshold=1e-8,
    keep_settings=True
)

print("Train columns contain RUL?", "RUL" in tr_df.columns)
print("Example labels:\n", tr_df[["unit","cycle","RUL_raw","RUL"]].head())
print("Dropped low-var:", art.dropped_lowvar_cols)
print("Kept feature count:", len(art.feature_cols))

# train_xgb.py (core training part)
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from src.features import make_window_features

W = 50
X_train, y_train = make_window_features(tr_df, art.feature_cols, window=W)
X_val, y_val     = make_window_features(va_df, art.feature_cols, window=W)


model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    eval_metric="rmse"
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    
    
    verbose=50,
)

val_pred = model.predict(X_val)
print("y_val shape:", y_val.shape, "dtype:", y_val.dtype)
print("pred shape:", val_pred.shape, "dtype:", val_pred.dtype)

rmse = mean_squared_error(y_val, val_pred)
print("Val RMSE:", rmse**0.5)
import os
os.makedirs("models", exist_ok=True)
model.save_model("models/xgb_fd001_2.json")  # XGBoost model IO [web:335]
print("Saved model to models/xgb_fd001_2.json")


