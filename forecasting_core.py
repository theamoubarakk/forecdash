from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---------- Data helpers ----------
def load_sales_from_excel(path_or_buffer, sheet=0, date_col="Date", value_col="Sales",
                          agg_to_month=True, agg_func="sum") -> pd.DataFrame:
    df = pd.read_excel(path_or_buffer, sheet_name=sheet)
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Expected columns '{date_col}' and '{value_col}'.")
    df = df[[date_col, value_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna().sort_values(date_col).rename(columns={date_col: "ds", value_col: "y"})
    if agg_to_month:
        df = (df.set_index("ds")["y"].resample("M").agg(agg_func).to_frame().reset_index())
    return df.dropna(subset=["y"]).reset_index(drop=True)

def split_train_test(df: pd.DataFrame, cutoff_date=None, test_periods=None):
    if cutoff_date is not None:
        cutoff_date = pd.to_datetime(cutoff_date)
        train = df[df["ds"] <= cutoff_date].copy()
        test  = df[df["ds"] >  cutoff_date].copy()
    elif test_periods is not None and int(test_periods) > 0:
        n = int(test_periods)
        if n >= len(df): raise ValueError("test_periods must be < number of rows.")
        train = df.iloc[:-n].copy(); test = df.iloc[-n:].copy()
    else:
        raise ValueError("Provide either cutoff_date or test_periods.")
    if train.empty or test.empty: raise ValueError("Empty train/test after split.")
    return train.reset_index(drop=True), test.reset_index(drop=True)

# ---------- SARIMA: 3 figures ----------
def sarima_three_figs(train_df, test_df,
                      order=(1,1,1), seasonal_order=(0,1,1,12)):
    """
    Returns:
      - figs: (fig_forecast, fig_resid_hist, fig_resid_qq)
      - forecast_df: [ds, yhat, yhat_lower, yhat_upper] aligned to test horizon
    """
    endog = train_df.set_index("ds")["y"]
    model = SARIMAX(endog, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)

    steps = len(test_df)
    pred = res.get_forecast(steps=steps)
    pred_mean = pred.predicted_mean if steps > 0 else pd.Series(dtype=float)
    pred_ci = pred.conf_int() if steps > 0 else pd.DataFrame()
    idx = test_df.set_index("ds").index if steps > 0 else pd.DatetimeIndex([])

    # 1) Forecast vs Actuals
    fig1, ax1 = plt.subplots(figsize=(11, 4))
    ax1.plot(train_df["ds"], train_df["y"], label="Train", linewidth=1.8)
    if steps > 0:
        ax1.plot(test_df["ds"], test_df["y"], label="Test", linewidth=1.8)
        ax1.plot(idx, pred_mean.reindex(idx), linestyle="--", label="Forecast")
        try:
            lower = pred_ci["lower y"].reindex(idx)
            upper = pred_ci["upper y"].reindex(idx)
        except KeyError:
            lower = pred_ci.iloc[:, 0].reindex(idx)
            upper = pred_ci.iloc[:, 1].reindex(idx)
        ax1.fill_between(idx, lower, upper, alpha=0.2, label="95% PI")
    ax1.set_title(f"SARIMA{order}×{seasonal_order}: Forecast vs Actuals")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Value"); ax1.legend(loc="best")

    # 2) Residuals Histogram
    resid = res.resid.dropna()
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    ax2.hist(resid, bins=30, alpha=0.7)
    ax2.set_title("SARIMA Residuals: Histogram")
    ax2.set_xlabel("Residual"); ax2.set_ylabel("Frequency")

    # 3) Residuals Q–Q
    fig3 = sm.qqplot(resid, line="s")
    fig3.suptitle("SARIMA Residuals: Q–Q Plot")

    # Forecast table aligned to test horizon
    out = pd.DataFrame({"ds": test_df["ds"].values}) if steps > 0 else pd.DataFrame(columns=["ds","yhat","yhat_lower","yhat_upper"])
    if steps > 0:
        out["yhat"] = pred_mean.reindex(idx).values
        try:
            out["yhat_lower"] = lower.values
            out["yhat_upper"] = upper.values
        except Exception:
            out["yhat_lower"] = np.nan
            out["yhat_upper"] = np.nan

    return (fig1, fig2, fig3), out
