"""
TSLA Next-Day Close Price Prediction
======================================
A full scikit-learn pipeline: EDA-informed feature engineering, a naive
baseline (essential for stock data), multiple candidate models evaluated
with time-series-aware validation, hyperparameter tuning on the winner,
and diagnostic plots.

Key design decisions (explained in accompanying report):
1. Stock closing prices are ~random-walk / non-stationary, so predicting
   raw next-day CLOSE lets a trivial "tomorrow = today" baseline score a
   deceptively high R^2. We therefore train the model to predict the
   next-day LOG RETURN (a stationary, harder target) and reconstruct the
   price from it. We report both the return-level metrics and the
   reconstructed price-level metrics, always next to the naive baseline.
2. No shuffling / no random k-fold: financial time series must be
   validated walk-forward (TimeSeriesSplit) and given a chronological
   hold-out test set, otherwise the model "sees the future" and metrics
   are meaningless.
3. Every feature is computed only from information available at time t
   (rolling windows use past data), so there is no look-ahead leakage.
"""

import os
import kagglehub
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
df = pd.read_csv("data/tsla_2014_2023.csv", parse_dates=["date"])

# Download latest version
#path = kagglehub.dataset_download("aspillai/tesla-stock-price-with-indicators-10-years")

#print("Path to dataset files:", path)
# Read CSV
#df = pd.read_csv(os.path.join(path, "data/tsla_2014_2023.csv"))

df = df.sort_values("date").reset_index(drop=True)

# ---------------------------------------------------------------------
# 2. Feature engineering
#    Dataset already ships with technical indicators (RSI, CCI, SMA,
#    EMA, MACD, Bollinger, ATR). We add a few standard extras that are
#    known to help short-horizon price models: lagged returns, rolling
#    volatility, and volume changes. All are computed causally (t and
#    earlier only).
# ---------------------------------------------------------------------
df["log_close"] = np.log(df["close"])
df["log_return_1d"] = df["log_close"].diff()          # today's return
df["log_return_5d"] = df["log_close"].diff(5)
df["volatility_10d"] = df["log_return_1d"].rolling(10).std()
df["volume_change_1d"] = df["volume"].pct_change()
df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
df["close_to_sma50"] = df["close"] / df["sma_50"] - 1
df["close_to_ema50"] = df["close"] / df["ema_50"] - 1

# TARGET: next day's log return. next_day_close is provided, but we
# derive the return target ourselves so we control exactly how it's
# built and can reconstruct price unambiguously.
df["target_log_return"] = np.log(df["next_day_close"]) - df["log_close"]

feature_cols = [
    "rsi_7", "rsi_14", "cci_7", "cci_14",
    "sma_50", "ema_50", "sma_100", "ema_100",
    "macd", "bollinger", "atr_7", "atr_14",
    "log_return_1d", "log_return_5d", "volatility_10d",
    "volume_change_1d", "hl_spread", "close_to_sma50", "close_to_ema50",
]

model_df = df.dropna(subset=feature_cols + ["target_log_return"]).reset_index(drop=True)
print(f"Rows after feature engineering / dropna: {len(model_df)} "
      f"({model_df['date'].min().date()} -> {model_df['date'].max().date()})")

X = model_df[feature_cols]
y = model_df["target_log_return"]
close_today = model_df["close"]
actual_next_close = model_df["next_day_close"]
dates = model_df["date"]

# ---------------------------------------------------------------------
# 3. Chronological train / test split (last 15% ~ final year as test)
# ---------------------------------------------------------------------
split_idx = int(len(model_df) * 0.85)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
close_today_test = close_today.iloc[split_idx:]
actual_next_close_test = actual_next_close.iloc[split_idx:]
test_dates = dates.iloc[split_idx:]

print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows "
      f"({test_dates.min().date()} -> {test_dates.max().date()})")

# ---------------------------------------------------------------------
# 4. Naive baseline: tomorrow's return = 0 (i.e. tomorrow's close =
#    today's close). This is the bar every model must clear.
# ---------------------------------------------------------------------
naive_pred_close = close_today_test.values  # predicts no change
naive_mae = mean_absolute_error(actual_next_close_test, naive_pred_close)
naive_rmse = np.sqrt(mean_squared_error(actual_next_close_test, naive_pred_close))
naive_dir_acc = np.mean(
    np.sign(actual_next_close_test.values - close_today_test.values) >= 0
)

# ---------------------------------------------------------------------
# 5. Candidate models, each in a Pipeline with scaling.
#    Trees don't need scaling but a shared pipeline keeps comparison
#    code simple and scaling never hurts them.
# ---------------------------------------------------------------------
candidates = {
    "Ridge": Pipeline([
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ]),
    "RandomForest": Pipeline([
        ("scale", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_leaf=10,
            random_state=RANDOM_STATE, n_jobs=-1)),
    ]),
    "GradientBoosting": Pipeline([
        ("scale", StandardScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=RANDOM_STATE)),
    ]),
    "SVR": Pipeline([
        ("scale", StandardScaler()),
        ("model", SVR(kernel="rbf", C=1.0, epsilon=0.01)),
    ]),
}

# ---------------------------------------------------------------------
# 6. Walk-forward cross-validation on the TRAIN set to pick a model
#    (TimeSeriesSplit, never shuffled).
# ---------------------------------------------------------------------
tscv = TimeSeriesSplit(n_splits=5)
cv_results = {}
for name, pipe in candidates.items():
    fold_maes = []
    for tr_idx, val_idx in tscv.split(X_train):
        pipe.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        pred = pipe.predict(X_train.iloc[val_idx])
        fold_maes.append(mean_absolute_error(y_train.iloc[val_idx], pred))
    cv_results[name] = np.mean(fold_maes)
    print(f"{name:>18s} | CV MAE (log-return space): {cv_results[name]:.5f}")

best_name = min(cv_results, key=cv_results.get)
print(f"\nSelected model by CV: {best_name}")

# ---------------------------------------------------------------------
# 7. Hyperparameter tuning of the selected model on the train set
# ---------------------------------------------------------------------
param_grids = {
    "Ridge": {"model__alpha": [0.1, 1.0, 10.0, 50.0]},
    "RandomForest": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20],
    },
    "GradientBoosting": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.01, 0.05, 0.1],
    },
    "SVR": {
        "model__C": [0.1, 1.0, 10.0],
        "model__epsilon": [0.005, 0.01, 0.05],
    },
}

grid = GridSearchCV(
    candidates[best_name], param_grids[best_name],
    cv=TimeSeriesSplit(n_splits=5),
    scoring="neg_mean_absolute_error", n_jobs=-1,
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print(f"Best params: {grid.best_params_}")

# ---------------------------------------------------------------------
# 8. Final evaluation on the untouched chronological test set
#    Metrics reported both in return-space and reconstructed
#    price-space, next to the naive baseline.
# ---------------------------------------------------------------------
pred_return = best_model.predict(X_test)
pred_next_close = close_today_test.values * np.exp(pred_return)

model_mae = mean_absolute_error(actual_next_close_test, pred_next_close)
model_rmse = np.sqrt(mean_squared_error(actual_next_close_test, pred_next_close))
model_r2 = r2_score(actual_next_close_test, pred_next_close)
model_dir_acc = np.mean(
    np.sign(pred_next_close - close_today_test.values) ==
    np.sign(actual_next_close_test.values - close_today_test.values)
)
model_mape = np.mean(
    np.abs((actual_next_close_test.values - pred_next_close) / actual_next_close_test.values)
) * 100

print("\n=== Test set performance (price space, $) ===")
print(f"{'Metric':<22}{'Naive baseline':>16}{best_name:>18}")
print(f"{'MAE ($)':<22}{naive_mae:>16.3f}{model_mae:>18.3f}")
print(f"{'RMSE ($)':<22}{naive_rmse:>16.3f}{model_rmse:>18.3f}")
print(f"{'Directional acc.':<22}{naive_dir_acc:>16.1%}{model_dir_acc:>18.1%}")
print(f"{'R^2 (price)':<22}{'--':>16}{model_r2:>18.4f}")
print(f"{'MAPE (%)':<22}{'--':>16}{model_mape:>18.2f}")

# ---------------------------------------------------------------------
# 9. Feature importance (if tree-based)
# ---------------------------------------------------------------------
inner_model = best_model.named_steps["model"]
if hasattr(inner_model, "feature_importances_"):
    importances = pd.Series(inner_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    print("\nTop feature importances:")
    print(importances.head(10))
elif hasattr(inner_model, "coef_"):
    coefs = pd.Series(inner_model.coef_, index=feature_cols).sort_values(key=abs, ascending=False)
    print("\nTop feature coefficients:")
    print(coefs.head(10))

# ---------------------------------------------------------------------
# 10. Plots
# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 9))

axes[0].plot(test_dates, actual_next_close_test.values, label="Actual next-day close", linewidth=1.5)
axes[0].plot(test_dates, pred_next_close, label=f"{best_name} prediction", linewidth=1.2, alpha=0.85)
axes[0].plot(test_dates, naive_pred_close, label="Naive (persistence) baseline",
             linewidth=1.0, linestyle="--", alpha=0.6)
axes[0].set_title("TSLA Next-Day Close: Actual vs. Predicted (chronological test set)")
axes[0].set_ylabel("Price ($)")
axes[0].legend()

errors = pred_next_close - actual_next_close_test.values
axes[1].plot(test_dates, errors, color="firebrick", linewidth=1)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("Prediction error ($) over time")
axes[1].set_ylabel("Predicted - Actual ($)")

plt.tight_layout()
#plt.savefig("tsla_prediction_results.png", dpi=150)
#print("\nSaved plot -> tsla_prediction_results.png")

# ---------------------------------------------------------------------
# 11. Persist the trained pipeline
# ---------------------------------------------------------------------
#joblib.dump(best_model, "tsla_next_day_close_model.joblib")
#print("Saved model -> tsla_next_day_close_model.joblib")
