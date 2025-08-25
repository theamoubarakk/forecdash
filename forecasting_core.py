# forecasting_core.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Prophet is optional; import only if available
try:
    from prophet import Prophet
except Exception:
    Prophet = None


# -------- Data --------
def load_sales_from_excel(path_or_buffer, sheet=0, date_col="Date", value_col="Sales",
                          agg_to_month=True, agg_func="sum"):
    df = pd.read_excel(path_or_buffer, sheet_name=sheet)
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Expected columns '{date_col}' and '{value_col}'.")
    df = df[[date_col, value_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna().sort_values(date_col).rename(columns={date_col: "ds", value_col: "y"})
    if agg_to_month:
        df = (df.set_index("ds")["y"].resample("M").agg(agg_func).to_frame().reset_index())
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    return df.dropna()


def split_train_test(df, cutoff_date=None, test_periods=None):
    if cutoff_date is not None:
        cutoff_date = pd.to_datetime(cutoff_date)
        train = df[df["ds"] <= cutoff_date].copy()
        test  = df[df["ds"] >  cutoff_date].copy()
    elif test_periods:
        train = df.iloc[:-int(test_periods)].copy()
        test  = df.iloc[-int(test_periods):].copy()
    else:
        raise ValueError("Provide either cutoff_date or test_periods.")
    if train.empty or test.empty:
        raise ValueError("Empty train/test after split. Adjust the split.")
    return train, test


# -------- Prophet (3 figures: Forecast, Components, Residual QQ) --------
def _require_prophet():
    if Prophet is None:
        raise ImportError("Prophet is not installed. Add `prophet` to requirements.txt.")


def prophet_three_figs(train_df, test_df, yearly=True, weekly=False, daily=False,
                       multiplicative=False, horizon=None, add_oct_dec=False):
    _require_prophet()

    m = Prophet(
        yearly_seasonality=yearly,
        weekly_seasonality=weekly,
        daily_seasonality=daily,
        seasonality_mode="multiplicative" if multiplicative else "additive",
    )

    # Optional seasonal regressors (if your notebook used them)
    if add_oct_dec:
        for d in (train_df,):
            d["oct_bump"] = (d["ds"].dt.month == 10).astype(int)
            d["dec_peak"] = (d["ds"].dt.month == 12).astype(int)
        m.add_regressor("oct_bump")
        m.add_regressor("dec_peak")

    m.fit(train_df)

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
    ax1.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.2, label="PI")
    ax1.set_title("Prophet: Forecast vs Actuals")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Value"); ax1.legend(loc="best")

    # 2) Components (Prophet's own fig)
    fig2 = m.plot_components(forecast)

    # 3) Residual Q–Q (use in-sample)
    train_pred = m.predict(train_df[["ds"]])
    resid = train_df["y"].values - train_pred["yhat"].values
    fig3 = sm.qqplot(resid, line="s")
    fig3.suptitle("Prophet Residuals: Q–Q Plot")

    return (fig1, fig2, fig3), forecast


# -------- SARIMA (3 figures: Forecast, Residual Histogram, Residual Q–Q) --------
def sarima_three_figs(train_df, test_df, order=(1,1,1), seasonal_order=(0,1,1,12)):
    endog = train_df.set_index("ds")["y"]
    model = SARIMAX(endog, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)

    # Forecast on test horizon
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
    ax1.set_xlabel("Date"); ax1.set_ylabel("Value"); ax1.legend(loc="best")

    # 2) Residual histogram
    resid = res.resid.dropna()
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    ax2.hist(resid, bins=30, alpha=0.7)
    ax2.set_title("SARIMA Residuals: Histogram")
    ax2.set_xlabel("Residual"); ax2.set_ylabel("Frequency")

    # 3) Residual Q–Q
    fig3 = sm.qqplot(resid, line="s")
    fig3.suptitle("SARIMA Residuals: Q–Q Plot")

    # For download convenience
    out = pd.DataFrame({"ds": test_df["ds"].values,
                        "yhat": pred_mean.reindex(idx).values,
                        "yhat_lower": lower.values if steps > 0 else np.nan,
                        "yhat_upper": upper.values if steps > 0 else np.nan})
    return (fig1, fig2, fig3), out
