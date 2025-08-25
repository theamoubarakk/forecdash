import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import date

from forecasting_core import (
    load_sales_from_excel,
    split_train_test,
    sarima_one_fig,
    prophet_two_figs,   # returns (forecast_fig, components_fig)
)

st.set_page_config(page_title="Forecasting Dashboard — SARIMA + Prophet", layout="wide")
st.title("Forecasting Dashboard — 3 Graphs (SARIMA + Prophet)")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Data")
    source = st.radio(
        "Source",
        ["Use repo file (data/BABA JINA SALES DATA.xlsx)", "Upload Excel"],
        index=0
    )
    sheet_name = st.text_input("Sheet (name or index)", value="0")
    date_col = st.text_input("Date column", value="Date")
    value_col = st.text_input("Value column", value="Sales")
    agg_to_month = st.checkbox("Aggregate monthly", value=True)

    st.divider()
    st.header("Train/Test split")
    mode = st.radio("Mode", ["By last N periods", "By cutoff date"], index=0)
    test_periods = st.number_input("Test periods (months)", 1, 120, 12) if mode == "By last N periods" else None
    cutoff_date = st.date_input("Cutoff date", value=date(2024, 12, 31)) if mode == "By cutoff date" else None

    st.divider()
    st.header("SARIMA params")
    p = st.number_input("p", 0, 5, 1)
    d = st.number_input("d", 0, 2, 1)
    q = st.number_input("q", 0, 5, 1)
    P = st.number_input("P", 0, 5, 0)
    D = st.number_input("D", 0, 2, 1)
    Q = st.number_input("Q", 0, 5, 1)
    s = st.number_input("Seasonal period", 0, 52, 12)

    st.divider()
    st.header("Prophet options")
    yearly = st.checkbox("Yearly seasonality", True)
    weekly = st.checkbox("Weekly seasonality", False)
    daily  = st.checkbox("Daily seasonality", False)
    multiplicative = st.checkbox("Multiplicative seasonality", False)
    add_oct_dec = st.checkbox("Add Oct/Dec regressors", False)

# ---------------- Data load ----------------
def _make_demo_df():
    """Fallback monthly demo series so graphs render if no file is present."""
    idx = pd.date_range("2019-01-01", periods=72, freq="M")
    trend = pd.Series(range(len(idx)), index=idx) * 1.2
    season = 10 * pd.Series(pd.sin(2 * 3.14159 * (idx.month / 12)), index=idx)
    noise = pd.Series(pd.Series(pd.np.random.randn(len(idx))).rolling(2, min_periods=1).mean().values, index=idx)
    y = (100 + trend + season + noise).rename("y")
    return pd.DataFrame({"ds": idx, "y": y.values})

df = None
if source.startswith("Use repo file"):
    default_path = Path("data/BABA JINA SALES DATA.xlsx")
    if default_path.exists():
        sheet = sheet_name if not sheet_name.isdigit() else int(sheet_name)
        df = load_sales_from_excel(default_path, sheet=sheet, date_col=date_col, value_col=value_col, agg_to_month=agg_to_month)
    else:
        st.info("Repo file not found — showing a demo dataset so you can see the graphs.")
        df = _make_demo_df()
else:
    upl = st.file_uploader("Upload .xlsx", type=["xlsx"])
    if upl:
        sheet = sheet_name if not sheet_name.isdigit() else int(sheet_name)
        df = load_sales_from_excel(upl, sheet=sheet, date_col=date_col, value_col=value_col, agg_to_month=agg_to_month)

if df is None:
    st.info("Upload a file or add data/BABA JINA SALES DATA.xlsx to the repo.")
    st.stop()

if len(df) < 6:
    st.error("Not enough data to fit models (need at least ~6 points).")
    st.stop()

st.success(f"Loaded {len(df):,} rows.")
with st.expander("Preview data", expanded=False):
    st.dataframe(df.head(20))

# ---------------- Split ----------------
try:
    train_df, test_df = split_train_test(
        df,
        cutoff_date=pd.to_datetime(cutoff_date) if cutoff_date else None,
        test_periods=int(test_periods) if test_periods else None
    )
    st.write(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
except Exception as e:
    st.error(f"Split error: {e}")
    st.stop()

# Guard: if s==0 ensure no seasonal terms
if s == 0 and any(v > 0 for v in (P, D, Q)):
    st.warning("Seasonal period is 0 → forcing P=D=Q=0 to avoid errors.")
    P = D = Q = 0

# ---------------- Caches ----------------
@st.cache_resource(show_spinner=False)
def _sarima_cached(train_df, test_df, order, seasonal_order):
    return sarima_one_fig(train_df, test_df, order=order, seasonal_order=seasonal_order)

@st.cache_resource(show_spinner=False)
def _prophet_cached(train_df, test_df, yearly, weekly, daily, multiplicative, add_oct_dec):
    return prophet_two_figs(
        train_df, test_df,
        yearly=yearly, weekly=weekly, daily=daily,
        multiplicative=multiplicative, add_oct_dec=add_oct_dec
    )

# ---------------- Render 3 graphs ----------------
# 1) SARIMA
try:
    sarima_fig, sarima_forecast = _sarima_cached(
        train_df, test_df,
        (int(p), int(d), int(q)), (int(P), int(D), int(Q), int(s))
    )
    st.subheader("1) SARIMA — Forecast vs Actuals")
    st.pyplot(sarima_fig, clear_figure=False)
except Exception as e:
    st.error(f"SARIMA error: {e}")

# 2 & 3) Prophet (handle ImportError gracefully)
try:
    (p_forecast_fig, p_components_fig), p_forecast_df = _prophet_cached(
        train_df, test_df, yearly, weekly, daily, multiplicative, add_oct_dec
    )
    st.subheader("2) Prophet — Forecast vs Actuals")
    st.pyplot(p_forecast_fig, clear_figure=False)

    st.subheader("3) Prophet — Components")
    st.pyplot(p_components_fig, clear_figure=False)
except ImportError:
    st.warning(
        "Prophet isn’t installed, so only the SARIMA graph is shown. "
        "Add `prophet==1.1.5` and `pystan==2.19.1.1` to requirements.txt to enable Prophet graphs."
    )
except Exception as e:
    st.error(f"Prophet error: {e}")

# Downloads (optional)
with st.expander("Download outputs"):
    if 'sarima_forecast' in locals():
        st.download_button(
            "SARIMA forecast (CSV)",
            sarima_forecast.to_csv(index=False).encode("utf-8"),
            "sarima_forecast.csv",
            "text/csv"
        )
    if 'p_forecast_df' in locals():
        st.download_button(
            "Prophet forecast (CSV)",
            p_forecast_df.to_csv(index=False).encode("utf-8"),
            "prophet_forecast.csv",
            "text/csv"
        )
