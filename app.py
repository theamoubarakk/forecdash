import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import date

from forecasting_core import (
    load_sales_from_excel,
    split_train_test,
    prophet_three_figs,
    sarima_three_figs,
)

st.set_page_config(page_title="Forecasting Dashboard (3 Graphs)", layout="wide")
st.title("Forecasting Dashboard (3 Graphs Only)")

# --------- Sidebar ---------
with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Upload Excel", "Use repo file (data/BABA JINA SALES DATA.xlsx)"])
    sheet_name = st.text_input("Sheet (name or index)", value="0")
    date_col = st.text_input("Date column", value="Date")
    value_col = st.text_input("Value column", value="Sales")
    agg_to_month = st.checkbox("Aggregate monthly", value=True)

    st.divider()
    st.header("Train/Test split")
    mode = st.radio("Mode", ["By cutoff date", "By last N periods"])
    cutoff_date = st.date_input("Cutoff date", value=date(2024, 12, 31)) if mode == "By cutoff date" else None
    test_periods = st.number_input("Test periods (months)", 1, 120, 12) if mode != "By cutoff date" else None

    st.divider()
    st.header("Model")
    model = st.selectbox("Choose", ["Prophet", "SARIMA"])

    if model == "Prophet":
        yearly = st.checkbox("Yearly seasonality", True)
        weekly = st.checkbox("Weekly seasonality", False)
        daily = st.checkbox("Daily seasonality", False)
        multiplicative = st.checkbox("Multiplicative seasonality", False)
        add_oct_dec = st.checkbox("Add Oct/Dec regressors (optional)", False)
    else:
        p = st.number_input("p", 0, 5, 1)
        d = st.number_input("d", 0, 2, 1)
        q = st.number_input("q", 0, 5, 1)
        P = st.number_input("P", 0, 5, 0)
        D = st.number_input("D", 0, 2, 1)
        Q = st.number_input("Q", 0, 5, 1)
        s = st.number_input("Seasonal period", 0, 52, 12)

# --------- Data load ---------
df = None
if source == "Upload Excel":
    upl = st.file_uploader("Upload .xlsx", type=["xlsx"])
    if upl:
        sheet = sheet_name if not sheet_name.isdigit() else int(sheet_name)
        df = load_sales_from_excel(
            upl, sheet=sheet, date_col=date_col, value_col=value_col, agg_to_month=agg_to_month
        )
else:
    default_path = Path("data/BABA JINA SALES DATA.xlsx")
    if default_path.exists():
        sheet = sheet_name if not sheet_name.isdigit() else int(sheet_name)
        df = load_sales_from_excel(
            default_path, sheet=sheet, date_col=date_col, value_col=value_col, agg_to_month=agg_to_month
        )
    else:
        st.warning("Repo file not found at data/BABA JINA SALES DATA.xlsx. Upload instead.")

if df is None:
    st.info("Load data from the sidebar to begin.")
    st.stop()

st.success(f"Loaded {len(df):,} rows.")
with st.expander("Preview data"):
    st.dataframe(df.head(20))

# --------- Split ---------
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

# --------- Optional caching to speed up repeated runs ---------
@st.cache_resource(show_spinner=False)
def _prophet_cached(train_df, test_df, yearly, weekly, daily, multiplicative, add_oct_dec):
    return prophet_three_figs(
        train_df, test_df,
        yearly=yearly, weekly=weekly, daily=daily,
        multiplicative=multiplicative, horizon=len(test_df),
        add_oct_dec=add_oct_dec
    )

@st.cache_resource(show_spinner=False)
def _sarima_cached(train_df, test_df, order, seasonal_order):
    return sarima_three_figs(train_df, test_df, order=order, seasonal_order=seasonal_order)

# --------- Fit and plot exactly 3 graphs ---------
if model == "Prophet":
    try:
        figs, forecast_full = _prophet_cached(
            train_df, test_df, yearly, weekly, daily, multiplicative, add_oct_dec
        )
        f1, f2, f3 = figs

        st.subheader("1) Forecast vs Actuals")
        st.pyplot(f1, clear_figure=False)

        st.subheader("2) Components")
        st.pyplot(f2, clear_figure=False)

        st.subheader("3) Residuals Q–Q")
        st.pyplot(f3, clear_figure=False)

        with st.expander("Download forecast (CSV)"):
            out = forecast_full[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            merged = pd.merge(out, df, on="ds", how="left")
            st.download_button(
                "Download", merged.to_csv(index=False).encode("utf-8"),
                "prophet_forecast.csv", "text/csv"
            )
    except Exception as e:
        st.error(f"Prophet error: {e}")

else:
    try:
        figs, forecast_df = _sarima_cached(
            train_df, test_df,
            (int(p), int(d), int(q)), (int(P), int(D), int(Q), int(s))
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
                "Download", forecast_df.to_csv(index=False).encode("utf-8"),
                "sarima_forecast.csv", "text/csv"
            )
    except Exception as e:
        st.error(f"SARIMA error: {e}")
