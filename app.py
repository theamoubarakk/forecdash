import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

from forecasting_core import (
    load_sales_from_excel,
    split_train_test,
    sarima_three_figs,
)

st.set_page_config(page_title="Forecasting Dashboard — SARIMA (3 Graphs)", layout="wide")
st.title("Forecasting Dashboard — SARIMA (3 Graphs)")

with st.sidebar:
    st.header("Data")
    source = st.radio(
        "Source",
        ["Use repo file (data/(3)BABA JINA SALES DATA.xlsx)", "Upload Excel"],
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

def _demo_df():
    """Fallback monthly demo so page always shows graphs."""
    idx = pd.date_range("2019-01-01", periods=72, freq="M")
    y = 100 + np.arange(len(idx))*1.2 + 10*np.sin(2*np.pi*idx.month/12) + np.random.normal(0,2,len(idx))
    return pd.DataFrame({"ds": idx, "y": y})

# ---- Load data
df = None
if source.startswith("Use repo file"):
    path = Path("data/BABA JINA SALES DATA.xlsx")
    if path.exists():
        sheet = sheet_name if not sheet_name.isdigit() else int(sheet_name)
        df = load_sales_from_excel(path, sheet=sheet, date_col=date_col, value_col=value_col, agg_to_month=agg_to_month)
    else:
        st.info("Repo file not found — showing a demo dataset so you can see the graphs.")
        df = _demo_df()
else:
    upl = st.file_uploader("Upload .xlsx", type=["xlsx"])
    if upl:
        sheet = sheet_name if not sheet_name.isdigit() else int(sheet_name)
        df = load_sales_from_excel(upl, sheet=sheet, date_col=date_col, value_col=value_col, agg_to_month=agg_to_month)

if df is None:
    st.info("Upload a file or add data/BABA JINA SALES DATA.xlsx to the repo.")
    st.stop()
if len(df) < 6:
    st.error("Not enough data to fit SARIMA (need at least ~6 points).")
    st.stop()

st.success(f"Loaded {len(df):,} rows.")
with st.expander("Preview data", expanded=False):
    st.dataframe(df.head(20))

# ---- Split
try:
    train_df, test_df = split_train_test(
        df,
        cutoff_date=pd.to_datetime(cutoff_date) if cutoff_date else None,
        test_periods=int(test_periods) if test_periods else None,
    )
except Exception as e:
    st.error(f"Split error: {e}")
    st.stop()
st.write(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")

# Guard: if s==0, remove seasonal terms
if s == 0 and any(v > 0 for v in (P, D, Q)):
    P = D = Q = 0
    st.warning("Seasonal period is 0 → forcing P=D=Q=0 to avoid errors.")

@st.cache_resource(show_spinner=False)
def _sarima_cached(train_df, test_df, order, seasonal_order):
    return sarima_three_figs(train_df, test_df, order=order, seasonal_order=seasonal_order)

# ---- Fit & plot 3 graphs
try:
    figs, forecast_df = _sarima_cached(
        train_df,
        test_df,
        (int(p), int(d), int(q)),
        (int(P), int(D), int(Q), int(s)),
    )
    f1, f2, f3 = figs

    st.subheader("1) Forecast vs Actuals")
    st.pyplot(f1, clear_figure=False)

    st.subheader("2) Residuals Histogram")
    st.pyplot(f2, clear_figure=False)

    st.subheader("3) Residuals Q–Q")
    st.pyplot(f3, clear_figure=False)

    with st.expander("Download forecast (CSV)"):
        st.download_button(
            "Download",
            forecast_df.to_csv(index=False).encode("utf-8"),
            "sarima_forecast.csv",
            "text/csv",
        )
except Exception as e:
    st.error(f"SARIMA error: {e}")
