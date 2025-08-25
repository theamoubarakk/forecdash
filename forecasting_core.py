# Interactive forecasting core used by app.py
# - Data loading and monthly aggregation
# - Train/Test split helpers
# - Prophet (PyStan backend) -> 3 figures: Forecast, Components, Residual Q–Q
# - SARIMA (statsmodels)     -> 3 figures: Forecast, Residual Histogram, Residual Q–Q

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# =========================
# Data loading / utilities
# =========================
def load_sales_from_excel(
    path_or_buffer,
    sheet: int | str = 0,
    date_col: str = "Date",
    value_col: str = "Sales",
    agg_to_month: bool = True,
    agg_func: str = "sum",
) -> pd.DataFrame:
    df = pd.read_excel(path_or_buffer, sheet_name=sheet)

    if date_col not in df.columns or value_col not in df.columns:
        cols = ", ".join(df.columns.astype(str).tolist())
        raise ValueError(
            f"Expected columns '{date_col}' and '{value_col}' in the Excel sheet. Found: {cols}"
        )

    df = df[[date_col, value_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col]).sort_values(date_col)
    df = df.rename(columns={date_col: "ds", value_col: "y"})

    if agg_to_month:
        df = (
            df.set_index("ds")["y"]
            .resample("M")
            .agg(agg_func)
            .to_frame()
            .reset_index()
        )

    return df.dropna(subset=["y"]).reset_index(drop=True)


def split_train_test(
    df: pd.DataFrame, cutoff_date: str | pd.Timestamp | None = None, test_periods: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cutoff_date is not None:
        cutoff_date = pd.to_datetime(cutoff_date)
        train = df[df["ds"] <= cutoff_date].copy()
        test = df[df["ds"] > cutoff_date].copy()
    elif test_periods is not None and int(test_periods) > 0:
        test_periods = int(test_periods)
        if test_periods >= len(df):
            raise ValueError("test_periods must be smaller than the number of rows.")
        train = df.iloc[:-test_periods].copy()
        test = df.iloc[-test_periods:].copy()
    else:
        raise ValueError("Provide either cutoff_date or test_periods.")

    if train.empty or test.empty:
        raise ValueError("Train/Test split produced an empty set. Adjust your parameters.")

    return train.reset_index(drop=True), test.reset_index(drop=True)

# =====================
# Prophet (PyStan) path
# =====================
def prophet_three_figs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    yearly: bool = True,
    weekly: bool = False,
    daily: bool = False,
    multiplicative: bool = False,
    horizon: int | None = None,
    add_oct_dec: bool = False,
):
    """
    Returns:
      figs: (fig_forecast, fig_components, fig_resid_qq)
      forecast: full Prophet forecast (train + future)
    """
    # Lazy import + force PyStan backend
    from prophet import Prophet

    m = Prophet(
        yearly_seasonality=yearly,
        weekly_seasonality=weekly,
        daily_seasonality=daily,
        seasonality_mode="multiplicative" if multiplicative else "additive",
        stan_backend="pystan",
    )

    train = train_df.copy()
    if add_oct_dec:
        train["oct_bump"] = (train["ds"].dt.month == 10).astype(int)
        train["dec_peak"] = (train["ds"].dt.month == 12).astype(int)
        m.add_regressor("oct_bump")
        m.add_regressor("dec_peak")

    m.fit(train)

    if horizon is None:
        horizon = len(test_df)

    future = m.make_future_dataframe(periods=horizon, freq="M")
    if add_oct_dec:
        future["oct_bump"] = (future["ds"].dt.month == 10).astype(int)
        future["dec_peak"] = (future["ds"].dt.month == 12).astype(int)

    forecast = m.predict(future)

    # 1) Forecast vs Actuals
    fig1, ax1 = plt.subplots(figsize=(11, 4))
    ax1.plot(train_df["ds"], train_df["y"], label="Train", linewidth=1.8)
    if len(test_df) > 0:
        ax1.plot(test_df["ds"], test_df["y"], label="Test", linewidth=1.8)
    ax1.plot(forecast["ds"], forecast["yhat"], linestyle="--", label="Forecast")
    ax1.fill_between(
        forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.2, label="PI"
    )
    ax1.set_title("Prophet: Forecast vs Actuals")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Value")
    ax1.legend(loc="best")

    # 2) Components
    fig2 = m.plot_components(forecast)

    # 3) Residuals Q–Q on train
    train_pred = m.predict(train_df[["ds"]])
    resid = train_df["y"].values - train_pred["yhat"].values
    fig3 = sm.qqplot(resid, line="s")
    fig3.suptitle("Prophet Residuals: Q–Q Plot")

    return (fig1, fig2, fig3), forecast

# ==================
# SARIMA (statsmodels)
# ==================
def sarima_three_figs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (0, 1, 1, 12),
):
    """
    Returns:
      figs: (fig_forecast, fig_resid_hist, fig_resid_qq)
      forecast_df: [ds, yhat, yhat_lower, yhat_upper] on the test horizon
    """
    endog = train_df.set_index("ds")["y"]
    model = SARIMAX(
        endog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    steps = len(test_df)
    pred = res.get_forecast(steps=steps)
    pred_mean = pred.predicted_mean
    pred_ci = pred.conf_int()

    idx = test_df.set_index("ds").index

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
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Value")
    ax1.legend(loc="best")

    # 2) Residuals Histogram
    resid = res.resid.dropna()
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    ax2.hist(resid, bins=30, alpha=0.7)
    ax2.set_title("SARIMA Residuals: Histogram")
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Frequency")

    # 3) Residuals Q–Q
    fig3 = sm.qqplot(resid, line="s")
    fig3.suptitle("SARIMA Residuals: Q–Q Plot")

    out = pd.DataFrame({"ds": test_df["ds"].values})
    if steps > 0:
        out["yhat"] = pred_mean.reindex(idx).values
        try:
            out["yhat_lower"] = lower.values
            out["yhat_upper"] = upper.values
        except Exception:
            out["yhat_lower"] = np.nan
            out["yhat_upper"] = np.nan
    else:
        out["yhat"] = np.nan
        out["yhat_lower"] = np.nan
        out["yhat_upper"] = np.nan

    return (fig1, fig2, fig3), out
